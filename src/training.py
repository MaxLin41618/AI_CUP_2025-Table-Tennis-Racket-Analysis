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
        
        # 載入已選特徵映射
        with open(os.path.join('data', 'selected_features.json'), 'r', encoding='utf-8-sig') as jf:
            selected_features_dict = json.load(jf)
        logf.write(f"Loaded selected_features: {selected_features_dict}\n")

        cat_features = ['mode']  # 類別特徵
        cv_scores_dict = {}

        for target, model in TARGETS.items():
            print(f'\n====== {target} 任務交叉驗證 ======')

            # 讀取 per-task 訓練集
            train_file = config.TRAIN_CSVS[target]
            logf.write(f'Training file for {target}: {train_file}\n')
            df = pd.read_csv(train_file)
            groups = df[config.PLAYER_ID_COL]
            y = df[target]
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            n_classes = len(np.unique(y_encoded))  # 該任務總標籤數

            # 使用 data_processing 選好的特徵
            selected_features = selected_features_dict[target]
            logf.write(f"Using selected_features for {target}: {selected_features}\n")
            X = df[selected_features]
            sgkf = StratifiedGroupKFold(n_splits=config.K_FOLD, shuffle=True, random_state=42)
            cv_scores_dict[target] = []
            
            for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y_encoded, groups=groups)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
                # TabPFN: NOTE: 可以先用一般版快速推論看效果
                if use_tabpfn:
                    # model = AutoTabPFNClassifier(max_time=config.PHE_TIME, preset='avoid_overfitting', device='cuda', categorical_feature_indices=[0], random_state=config.RANDOM_SEED)
                    model = TabPFNClassifier(categorical_features_indices=[0], random_state=config.RANDOM_SEED)
                    model.fit(X_train.values, y_train)
                    y_pred = model.predict_proba(X_val.values)
                # CatBoost
                else:
                    class_weights = compute_class_weights(y_train)
                    class_weights_rounded = [round(w, 4) for w in class_weights]
                    print(f"{target} Fold {fold+1} 類別權重: {class_weights_rounded}")
                    model = copy.deepcopy(TARGETS[target])
                    model.set_params(class_weights=class_weights)
                    # 僅保留在 selected_features 中的類別特徵
                    cat_features_loop = [f for f in cat_features if f in selected_features]
                    train_pool = Pool(X_train, y_train, cat_features=cat_features_loop)
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
                        plot_feature_importance(model, selected_features, os.path.join(fold_dir, 'importance.png'), title=f'Feature Importance - {target} FOLD_{fold+1}', top_n=15)
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

if __name__ == '__main__':
    main()