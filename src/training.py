import os
import pandas as pd
from catboost import Pool, CatBoostClassifier
from model import gender_model, handed_model, play_years_model, level_model
import config
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import shutil
from utils import print_and_log_overall_mean, compute_class_weights, plot_feature_importance
import copy
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import pickle
import json
from feature_selection import select_features
import math
import random
from data_processing import extract_features_from_array, jitter_signal

# 隨機種子
np.random.seed(config.RANDOM_SEED)

# ========== 參數設定 ==========
TARGETS = {
    'gender': gender_model,
    'hold racket handed': handed_model,
    'play years': play_years_model,
    'level': level_model
}

use_tabpfn = config.MODEL_TYPE.lower() == 'tabpfn'

# ========== 主訓練流程 ==========
def main():
    """以StratifiedGroupKFold訓練四個CatBoost模型，並記錄cv結果與平均值"""
    # 建立儲存資料夾
    save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('models', save_time)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.txt')

    # 紀錄log
    with open(log_path, 'w', encoding='utf-8-sig') as logf:
        logf.write(f'StratifiedGroupKFold: {config.K_FOLD}\n')
        logf.write(f'Features: {config.FEATURES}\n')
        
        # 載入原始訓練資訊
        info_path = os.path.join('data', 'raw', 'train_info.csv')
        info_df = pd.read_csv(info_path)
        uids_all = info_df['unique_id'].tolist()
        
        # per-fold 特徵選擇
        selected_features_folds_by_target = {}
        best_selected_features = {}

        # CatBoost 類別特徵
        cat_features = ['mode']

        # CV 分割
        cv_scores_dict = {}

        # 不同任務的交叉驗證
        for target, model in TARGETS.items():
            print(f'\n====== {target} 任務交叉驗證 ======')

            # 使用 raw 訓練資訊進行 CV 分割
            groups = info_df[config.PLAYER_ID_COL]
            y = info_df[target]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(np.unique(y_encoded))
            sgkf = StratifiedGroupKFold(n_splits=config.K_FOLD, shuffle=True, random_state=42)
            cv_scores_dict[target] = []
            selected_features_folds_by_target[target] = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(info_df, y_encoded, groups=groups)):
                # 讀取 fold 的 raw 訓練集並執行初始 jitter 增強
                uids_train = [uids_all[i] for i in train_idx]
                labs_train = [y_encoded[i] for i in train_idx]
                feats_list, labs_list = [], []
                for uid, lab in zip(uids_train, labs_train):
                    # 結合 meta 和原始/jitter 特徵
                    row_meta = info_df[info_df['unique_id'] == uid].iloc[0].to_dict()
                    row_meta.pop('cut_point', None)
                    txt_file = os.path.join('data', 'raw', 'train_data', f'{uid}.txt')
                    data = np.loadtxt(txt_file)
                    # 原始特徵
                    feat = extract_features_from_array(data)
                    row = row_meta.copy(); row.update(feat)
                    feats_list.append(row); labs_list.append(lab)
                    # 初始 jitter 增強
                    for _ in range(config.AUGMENT_JITTER_COUNT):
                        jittered = np.stack(
                            [jitter_signal(arr, config.AUGMENT_JITTER_STD_RATIO) for arr in data.T],
                            axis=1)
                        feat_jit = extract_features_from_array(jittered)
                        row_jit = row_meta.copy(); row_jit.update(feat_jit)
                        feats_list.append(row_jit); labs_list.append(lab)
                X_train_init = pd.DataFrame(feats_list)[config.FEATURES]
                y_train_init = np.array(labs_list)
                # 平衡資料：必要時額外 jitter 增強
                from collections import Counter
                cnts = Counter(y_train_init)
                max_n = max(cnts.values())
                jitter_extras, jitter_labels = [], []
                for label, count in cnts.items():
                    if count < max_n:
                        need = max_n - count
                        samples = math.ceil(need / config.AUGMENT_JITTER_COUNT)
                        uids_of_label = [u for u, lab in zip(uids_train, labs_train) if lab == label]
                        chosen = random.choices(uids_of_label, k=samples)
                        for uid_sel in chosen:
                            row_meta = info_df[info_df['unique_id'] == uid_sel].iloc[0].to_dict()
                            row_meta.pop('cut_point', None)
                            txtp = os.path.join('data', 'raw', 'train_data', f'{uid_sel}.txt')
                            dat = np.loadtxt(txtp)
                            for _ in range(config.AUGMENT_JITTER_COUNT):
                                jit = np.stack(
                                    [jitter_signal(arr, config.AUGMENT_JITTER_STD_RATIO) for arr in dat.T],
                                    axis=1)
                                feat_jit = extract_features_from_array(jit)
                                row_jit = row_meta.copy(); row_jit.update(feat_jit)
                                jitter_extras.append(row_jit); jitter_labels.append(label)
                # 限制至所需數量
                extra_need = max_n - len(y_train_init)
                jitter_extras = jitter_extras[:extra_need]
                jitter_labels = jitter_labels[:extra_need]
                if jitter_extras:
                    X_train_aug = pd.concat(
                        [X_train_init, pd.DataFrame(jitter_extras)[config.FEATURES]],
                        ignore_index=True)
                    y_train_aug = np.concatenate([y_train_init, np.array(jitter_labels)])
                else:
                    X_train_aug, y_train_aug = X_train_init, y_train_init
                # 隨機打散
                perm = np.random.permutation(len(y_train_aug))
                X_train_aug = X_train_aug.iloc[perm].reset_index(drop=True)
                y_train_aug = y_train_aug[perm]
                # 準備驗證集：只做原始特徵萃取
                uids_val = [uids_all[i] for i in val_idx]
                labs_val = [y_encoded[i] for i in val_idx]
                feats_val = []
                for uid, lab in zip(uids_val, labs_val):
                    row_meta = info_df[info_df['unique_id'] == uid].iloc[0].to_dict()
                    row_meta.pop('cut_point', None)
                    txtv = os.path.join('data', 'raw', 'train_data', f'{uid}.txt')
                    datv = np.loadtxt(txtv)
                    feat_val = extract_features_from_array(datv)
                    row = row_meta.copy(); row.update(feat_val)
                    feats_val.append(row)
                X_val_df = pd.DataFrame(feats_val)[config.FEATURES]
                y_val = np.array(labs_val)
                # per-fold 特徵選擇
                selector = select_features(
                    X_train_aug.values,
                    y_train_aug,
                    config.FEATURE_SELECTION_N_FEATURES,
                    config.FEATURES,
                    target
                )
                mask = selector.get_support()
                selected_features_fold = [config.FEATURES[i] for i, m in enumerate(mask) if m]
                selected_features_folds_by_target[target].append(selected_features_fold)
                logf.write(f"Fold {fold+1} selected_features: {selected_features_fold}\n")
                X_train = X_train_aug[selected_features_fold]
                X_val = X_val_df[selected_features_fold]
                # TabPFN: NOTE: 可以先用一般版快速推論看效果
                if use_tabpfn:
                    # model = AutoTabPFNClassifier(max_time=config.PHE_TIME, preset='avoid_overfitting', device='cuda', categorical_feature_indices=[0], random_state=config.RANDOM_SEED)
                    model = TabPFNClassifier(categorical_features_indices=[0], random_state=config.RANDOM_SEED)
                    model.fit(X_train.values, y_train_aug)
                    y_pred = model.predict_proba(X_val.values)
                # CatBoost
                else:
                    class_weights = compute_class_weights(y_train_aug)
                    class_weights_rounded = [round(w, 4) for w in class_weights]
                    print(f"{target} Fold {fold+1} 類別權重: {class_weights_rounded}")
                    model = copy.deepcopy(TARGETS[target])
                    model.set_params(class_weights=class_weights)
                    cat_features_loop = [f for f in cat_features if f in selected_features_fold]
                    train_pool = Pool(X_train, y_train_aug, cat_features=cat_features_loop)
                    val_pool = Pool(X_val, y_val, cat_features=cat_features_loop)
                    model.fit(train_pool, eval_set=val_pool)
                    y_pred = model.predict_proba(X_val)
                # ======== 評分前標籤數檢查 ========
                unique_labels = np.unique(y_val)
                if len(unique_labels) < n_classes:
                    print(f'{target} Fold {fold+1}: 標籤數不足（僅有{len(unique_labels)}類，需{n_classes}類），跳過評分')
                    cv_scores_dict[target].append(np.nan)  # 跳過時填入nan，保持fold對應關係
                    continue
                # ======== 計算AUC ========
                if target in config.BINARY_TARGETS:
                    auc = roc_auc_score(y_val, y_pred[:,1])
                    print(f'{target} Fold {fold+1}: AUC={auc:.4f}')
                    cv_scores_dict[target].append(auc)
                else:
                    try:
                        auc = roc_auc_score(y_val, y_pred, multi_class='ovr', average='micro')
                        print(f'{target} Fold {fold+1}: micro OvR AUC={auc:.4f}')
                    except ValueError as e:
                        print(f'{target} Fold {fold+1}: AUC 計算失敗，跳過，原因: {e}')
                        auc = np.nan
                    cv_scores_dict[target].append(auc)
                # ======== 儲存模型與參數到分層資料夾 ========
                target_dir = os.path.join(save_dir, target)
                os.makedirs(target_dir, exist_ok=True)
                fold_dir = os.path.join(target_dir, f'FOLD_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)
                if use_tabpfn:
                    model_path = os.path.join(fold_dir, 'model.tabpfn')
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                else:
                    model_path = os.path.join(fold_dir, 'model.cbm')
                    model.save_model(model_path)
                param_path = os.path.join(fold_dir, 'params.txt')
                if not use_tabpfn:
                    with open(param_path, 'w', encoding='utf-8') as pf:
                        pf.write(str(model.get_params()))
                if not use_tabpfn:
                    try:
                        plot_feature_importance(model, selected_features_fold, os.path.join(fold_dir, 'importance.png'), title=f'Feature Importance - {target} FOLD_{fold+1}', top_n=15)
                    except Exception as e:
                        print(f"[特徵重要度繪圖失敗] {target} fold {fold+1}: {e}")
        # 計算平均分數
        print(f"======結果======")
        for target in TARGETS:
            scores = cv_scores_dict[target]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.nanmean(scores)
                # 尋找有效fold的索引
                valid_indices = [i for i, s in enumerate(scores) if not np.isnan(s)]
                valid_scores_arr = np.array([scores[i] for i in valid_indices])
                best_valid_idx_in_valid = np.nanargmax(valid_scores_arr)
                best_idx = valid_indices[best_valid_idx_in_valid]
                best_score = scores[best_idx]
                # 儲存最佳 fold 的選中特徵
                best_selected_features[target] = selected_features_folds_by_target[target][best_idx]
                # 判斷最佳模型副檔名，log紀錄要正確
                best_model_ext = 'tabpfn' if use_tabpfn else 'cbm'
                # 最佳模型存於target資料夾下
                best_model_src = os.path.join(save_dir, target, f'FOLD_{best_idx+1}', f'model.{best_model_ext}')
                best_model_dst = os.path.join(save_dir, target, f'best_{target}.{best_model_ext}')
                # 若最佳模型存在則複製，否則跳過
                if os.path.exists(best_model_src):
                    shutil.copyfile(best_model_src, best_model_dst)
                else:
                    print(f'{target} 最佳模型檔案 {best_model_src} 不存在，跳過複製')
                    logf.write(f'{target} 最佳模型檔案 {best_model_src} 不存在，跳過複製\n')
                if target in config.BINARY_TARGETS:
                    print(f'{target} 平均AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})')
                    logf.write(f'{target} 平均AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})\n')
                    logf.write(f'{target} 最佳fold: {best_idx+1}, AUC={best_score:.4f}, 檔案: {target}/best_{target}.{best_model_ext}\n')
                else:
                    print(f'{target} 平均micro OvR AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})')
                    logf.write(f'{target} 平均micro OvR AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})\n')
                    logf.write(f'{target} 最佳fold: {best_idx+1}, micro OvR AUC={best_score:.4f}, 檔案: {target}/best_{target}.{best_model_ext}\n')
            else:
                print(f'{target} 無有效AUC fold，無最佳模型')
                logf.write(f'{target} 無有效AUC fold，無最佳模型\n')
            logf.write('\n')
        # ======== 四任務平均分數（本地評估用） ========
        print_and_log_overall_mean(cv_scores_dict, list(TARGETS.keys()), logf)
        logf.write('\n')
        # 儲存 per-task 最佳 fold 特徵映射
        os.makedirs('data', exist_ok=True)
        with open(os.path.join('data', 'selected_features.json'), 'w', encoding='utf-8-sig') as jf:
            json.dump(best_selected_features, jf, ensure_ascii=False, indent=2)
        logf.write('儲存 per-task 最佳 fold 特徵至 data/selected_features.json\n')

if __name__ == '__main__':
    main()