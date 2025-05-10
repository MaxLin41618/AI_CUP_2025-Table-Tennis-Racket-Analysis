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
from utils import print_and_log_overall_mean, compute_class_weights, plot_feature_importance, export_feature_importance_csv, compute_feature_fingerprint, save_feature_cache, load_feature_cache
import copy
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
import pickle
import json
from feature_selection import select_features, select_global_features
import random
from data_processing import extract_features_from_array
from imblearn.over_sampling import BorderlineSMOTE
import time
from sklearn.inspection import permutation_importance

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
    # 設定訓練資料集    
    try:
        df = pd.read_csv(config.TEST_CSV, nrows=0)
        config.FEATURES = ['mode'] + [c for c in df.columns if c != 'unique_id']
        # 保持順序並移除重複
        config.FEATURES = list(dict.fromkeys(config.FEATURES))
        print(f"更新後的特徵數量: {len(config.FEATURES)}")
    except Exception as e:
        print(f"無法更新特徵列表: {e}")

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
        
        # ===== 直接從 training.csv 讀取特徵 =====
        print("直接從 training.csv 讀取特徵...")
        # 讀取 training.csv
        train_df = pd.read_csv(config.TRAIN_CSV)
        # 設定 'unique_id' 為索引
        train_df = train_df.set_index('unique_id')
        # 排除不需要的欄位
        meta_cols = ['player_id', 'gender', 'hold racket handed', 'play years', 'level']
        feature_cols = [col for col in train_df.columns if col not in meta_cols]
        # 選取需要的特徵
        all_feats_df = train_df[feature_cols]
        print(f"訓練篩選前使用的特徵數量: {len(all_feats_df.columns)}")
        
        # per-fold 特徵選擇
        selected_features_folds_by_target = {}
        best_selected_features = {}
        all_importances_by_target = {}
        all_perm_importances_by_target = {}
        
        # CatBoost 類別特徵
        cat_features = ['mode'] # NOTE: CatBoost 類別特徵

        # CV 分數
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
            sgkf = StratifiedGroupKFold(n_splits=config.K_FOLD, shuffle=True, random_state=config.RANDOM_SEED)
            cv_scores_dict[target] = []
            selected_features_folds_by_target[target] = []
            all_importances_by_target[target] = []
            all_perm_importances_by_target[target] = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(info_df, y_encoded, groups=groups)):
                # 設定 python random 的 seed，確保取樣一致
                random.seed(config.RANDOM_SEED + fold)
                # 檢查此 fold 是否包含所有訓練與驗證集標籤
                train_labels = y_encoded[train_idx]
                val_labels = y_encoded[val_idx]
                if len(np.unique(train_labels)) < n_classes or len(np.unique(val_labels)) < n_classes:
                    print(f"[Fold {fold+1}] 標籤不足(訓練 {len(np.unique(train_labels))}/{n_classes}, 驗證 {len(np.unique(val_labels))}/{n_classes})，跳過此 fold")
                    logf.write(f"[Fold {fold+1}] 標籤不足(訓練 {len(np.unique(train_labels))}/{n_classes}, 驗證 {len(np.unique(val_labels))}/{n_classes})，跳過此 fold\n")
                    cv_scores_dict[target].append(np.nan)
                    # 跳過 fold 時，補全零向量以保持與全量特徵一致的維度
                    selected_features_folds_by_target[target].append([])
                    all_importances_by_target[target].append(np.zeros(len(config.FEATURES)))
                    all_perm_importances_by_target[target].append(np.zeros(len(config.FEATURES)))
                    continue
                
                # 從全量特徵 DataFrame 中選出訓練集
                uids_train = [uids_all[i] for i in train_idx]
                labs_train = [y_encoded[i] for i in train_idx]
                X_train_init = all_feats_df.loc[uids_train]
                y_train_init = np.array(labs_train)
                X_train_aug, y_train_aug = X_train_init, y_train_init

                # 從全量特徵 DataFrame 中選出驗證集
                uids_val = [uids_all[i] for i in val_idx]
                labs_val = [y_encoded[i] for i in val_idx]
                X_val_df = all_feats_df.loc[uids_val]
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
                # 將單折重要度映射回全量特徵向量
                full_imp = np.zeros(len(config.FEATURES))
                for i, feat in enumerate(selected_features_fold):
                    idx_full = config.FEATURES.index(feat)
                    full_imp[idx_full] = selector.importances_[i]
                all_importances_by_target[target].append(full_imp)
                logf.write(f"Fold {fold+1} selected_features: {selected_features_fold}\n")
                X_train = X_train_aug[selected_features_fold]
                X_val = X_val_df[selected_features_fold]
                # print(f"Fold {fold+1} selected_features: {selected_features_fold}")
                print(f"特徵選擇後總共使用{len(selected_features_fold)}個特徵")

                # permutation importance 篩選
                X_train = X_train.fillna(X_train.mean())
                model_perm = copy.deepcopy(TARGETS[target])
                model_perm.fit(X_train, y_train_aug)
                perm_res = permutation_importance(model_perm, X_train, y_train_aug, n_repeats=5, random_state=config.RANDOM_SEED)
                perm_imp = perm_res.importances_mean
                perm_idxs = np.argsort(perm_imp)[::-1][:config.PERM_FEATURE_SELECTION_K]
                perm_selected = [selected_features_fold[i] for i in perm_idxs]
                perm_full_imp = np.zeros(len(config.FEATURES))
                for idx in perm_idxs:
                    feat = selected_features_fold[idx]
                    perm_full_imp[config.FEATURES.index(feat)] = perm_imp[idx]
                all_perm_importances_by_target[target].append(perm_full_imp)
                logf.write(f"Fold {fold+1} perm_selected_features: {perm_selected}\n")
                print(f"Permutation 篩後共使用 {len(perm_selected)} 個特徵")
                # 更新訓練與驗證集特徵
                X_train = X_train[perm_selected]
                X_val = X_val_df[perm_selected]
                
                # 處理缺失值 (NaN)，因為 BorderlineSMOTE 不接受 NaN 值
                # 檢查是否有 NaN 值
                has_nan = X_train.isna().any().any()
                if has_nan:
                    print(f"檢測到 NaN 值，進行填補處理")
                    # 使用均值填補缺失值
                    from sklearn.impute import SimpleImputer
                    imputer = SimpleImputer(strategy='mean')
                    X_train_imputed = pd.DataFrame(
                        imputer.fit_transform(X_train),
                        columns=X_train.columns,
                        index=X_train.index
                    )
                else:
                    X_train_imputed = X_train
                
                # Borderline-SMOTE 過採樣（僅訓練集）
                smote = BorderlineSMOTE(random_state=config.RANDOM_SEED)
                X_train, y_train_aug = smote.fit_resample(X_train_imputed, y_train_aug)
        
                # TabPFN: TODO: 可以先用一般版快速推論看效果
                if use_tabpfn:
                    # 動態指定 'mode' 欄位索引為類別特徵 index
                    if 'mode' in perm_selected:
                        cat_idx_list = [perm_selected.index('mode')]
                    else:
                        cat_idx_list = []
                    # model = AutoTabPFNClassifier(max_time=config.PHE_TIME, preset='avoid_overfitting', device='cuda', categorical_feature_indices=cat_idx_list, random_state=config.RANDOM_SEED)
                    model = TabPFNClassifier(categorical_features_indices=cat_idx_list, random_state=config.RANDOM_SEED)
                    model.fit(X_train.values, y_train_aug)
                    y_pred = model.predict_proba(X_val.values)
                # CatBoost
                else:
                    cat_features_loop = [f for f in cat_features if f in perm_selected]
                    # 將分類特徵轉為字串，以供 CatBoost 處理
                    if cat_features_loop:
                        X_train = X_train.copy()
                        X_val = X_val.copy()
                        for f in cat_features_loop:
                            X_train[f] = X_train[f].astype(str)
                            X_val[f] = X_val[f].astype(str)
                    class_weights = compute_class_weights(y_train_aug)
                    class_weights_rounded = [round(w, 4) for w in class_weights]
                    print(f"{target} Fold {fold+1} 類別權重: {class_weights_rounded}")
                    model = copy.deepcopy(TARGETS[target])
                    model.set_params(class_weights=class_weights)
                    train_pool = Pool(X_train, y_train_aug, cat_features=cat_features_loop)
                    val_pool = Pool(X_val, y_val, cat_features=cat_features_loop)
                    model.fit(train_pool, eval_set=val_pool)
                    y_pred = model.predict_proba(X_val)
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
                        plot_feature_importance(model, selected_features_fold, os.path.join(fold_dir, 'importance.png'), title=f'Feature Importance - {target} FOLD_{fold+1}', top_n=20)
                        export_feature_importance_csv(model, selected_features_fold, os.path.join(fold_dir, 'importance.csv'), top_n=len(selected_features_fold))
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

        # 全局特徵選擇：聚合多折重要性
        global_selected_features = {}
        for target in TARGETS:
            global_selector = select_global_features(
                all_importances_by_target[target],
                config.FEATURES,
                config.GLOBAL_FEATURE_SELECTION_METHOD,
                config.GLOBAL_TOP_K_FEATURES
            )
            mask = global_selector.get_support()
            global_selected_features[target] = [config.FEATURES[i] for i, m in enumerate(mask) if m]
        with open(os.path.join('data', 'global_selected_features.json'), 'w', encoding='utf-8-sig') as jf2:
            json.dump(global_selected_features, jf2, ensure_ascii=False, indent=2)
        logf.write('儲存全局特徵選擇至 data/global_selected_features.json\n')

        # 全局 permutation 特徵選擇：聚合多折 permutation 重要度
        global_perm = {}
        for t in TARGETS:
            agg = np.mean(all_perm_importances_by_target[t], axis=0)
            idxs = np.argsort(agg)[::-1][:config.PERM_FEATURE_SELECTION_K]
            global_perm[t] = [config.FEATURES[i] for i in idxs]
        with open(os.path.join('data', 'global_permutation_selected_features.json'), 'w', encoding='utf-8-sig') as jf3:
            json.dump(global_perm, jf3, ensure_ascii=False, indent=2)
        logf.write('儲存全局 permutation 特徵至 data/global_permutation_selected_features.json\n')

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time/60:.2f} minutes")