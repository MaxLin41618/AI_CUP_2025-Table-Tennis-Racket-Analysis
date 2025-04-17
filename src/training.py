import os
import pandas as pd
from catboost import Pool, CatBoostClassifier
from model import gender_model, handed_model, play_years_model, level_model
import config
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import shutil
from utils import print_and_log_overall_mean, compute_class_weights
import copy

# 隨機種子
np.random.seed(config.RANDOM_SEED)

# ========== 參數設定 ==========
TARGETS = {
    'gender': gender_model,
    'hold racket handed': handed_model,
    'play years': play_years_model,
    'level': level_model
}

# ========== 主訓練流程 ==========
def main():
    """以GroupKFold訓練四個CatBoost模型，並記錄cv結果與平均值"""
    # 載入資料
    df = pd.read_csv(config.TRAIN_CSV)
    X = df[config.FEATURES]
    groups = df[config.PLAYER_ID_COL]

    # 建立儲存資料夾
    save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('models', save_time)
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, 'log.txt')

    # 紀錄log
    with open(log_path, 'w', encoding='utf-8-sig') as logf:
        logf.write(f'GroupKFold: {config.K_FOLD}\n')
        logf.write(f'Features: {config.FEATURES}\n')
        logf.write(f'Training file: {config.TRAIN_CSV}\n')

        # GroupKFold分群交叉驗證
        gkf = GroupKFold(n_splits=config.K_FOLD)
        cv_scores_dict = {target: [] for target in TARGETS}
        label_encoders = {}  # 標籤編碼器
        cat_features = ['mode']  # 類別特徵
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, df[config.PLAYER_ID_COL], groups)):
            print(f"\n========== 第{fold+1}折 ==========")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            for target, model in TARGETS.items():
                y = df[target]
                # ====== 標籤編碼，確保從0開始且連續 ======
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                label_encoders[target] = le
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
                
                # ======== 動態計算類別權重並設定到模型 ========
                class_weights = compute_class_weights(y_encoded)
                class_weights_rounded = [round(w, 4) for w in class_weights]
                print(f"{target} Fold {fold+1} 類別權重: {class_weights_rounded}")
                model = copy.deepcopy(TARGETS[target])
                model.set_params(class_weights=class_weights)
                # ======== 訓練模型 ========
                train_pool = Pool(X_train, y_train, cat_features=cat_features)
                val_pool = Pool(X_val, y_val, cat_features=cat_features)
                model.fit(train_pool, eval_set=val_pool)
                # 預測機率
                y_pred = model.predict_proba(X_val)
                if target in config.BINARY_TARGETS:
                    val_classes = np.unique(y_val)
                    if len(val_classes) < 2:
                        print(f'{target} Fold {fold+1}: 驗證集類別不足，AUC=NaN')
                        cv_scores_dict[target].append(np.nan)
                    else:
                        auc = roc_auc_score(y_val, y_pred[:,1])
                        print(f'{target} Fold {fold+1}: AUC={auc:.4f}')
                        cv_scores_dict[target].append(auc)
                else:
                    n_class = len(le.classes_)
                    val_classes = np.unique(y_val)
                    if len(val_classes) < n_class:
                        print(f'{target} Fold {fold+1}: 驗證集類別不足，AUC=NaN')                        
                        cv_scores_dict[target].append(np.nan)
                    else:
                        auc = roc_auc_score(y_val, y_pred, multi_class='ovr', average='micro')
                        print(f'{target} Fold {fold+1}: micro OvR AUC={auc:.4f}')                        
                        cv_scores_dict[target].append(auc)
                # ======== 儲存模型與參數到分層資料夾 ========
                target_dir = os.path.join(save_dir, target)
                os.makedirs(target_dir, exist_ok=True)
                fold_dir = os.path.join(target_dir, f'FOLD_{fold+1}')
                os.makedirs(fold_dir, exist_ok=True)
                model_path = os.path.join(fold_dir, 'model.cbm')
                param_path = os.path.join(fold_dir, 'params.txt')
                model.save_model(model_path)
                with open(param_path, 'w', encoding='utf-8') as pf:
                    pf.write(str(model.get_params()))

                # ======== 特徵重要度繪圖 ========
                from utils import plot_feature_importance
                try:
                    plot_feature_importance(model, config.FEATURES, os.path.join(fold_dir, 'importance.png'), title=f'Feature Importance - {target} FOLD_{fold+1}', top_n=15)
                except Exception as e:
                    print(f"[特徵重要度繪圖失敗] {target} fold {fold+1}: {e}")
        # 計算平均分數
        print(f"======結果======")
        for target in TARGETS:
            scores = cv_scores_dict[target]
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                mean_score = np.nanmean(scores)
                # 尋找最佳fold
                best_idx = np.nanargmax(scores)
                best_score = scores[best_idx]
                # best模型存於target資料夾下
                best_model_src = os.path.join(save_dir, target, f'FOLD_{best_idx+1}', 'model.cbm')
                best_model_dst = os.path.join(save_dir, target, f'best_{target}.cbm')
                shutil.copyfile(best_model_src, best_model_dst)
                if target in config.BINARY_TARGETS:
                    print(f'{target} 平均AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})')
                    logf.write(f'{target} 平均AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})\n')
                    logf.write(f'{target} 最佳fold: {best_idx+1}, AUC={best_score:.4f}, 檔案: {target}/best_{target}.cbm\n')
                else:
                    print(f'{target} 平均micro OvR AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})')
                    logf.write(f'{target} 平均micro OvR AUC: {mean_score:.4f} (有效fold數: {len(valid_scores)})\n')
                    logf.write(f'{target} 最佳fold: {best_idx+1}, micro OvR AUC={best_score:.4f}, 檔案: {target}/best_{target}.cbm\n')
            else:
                print(f'{target} 無有效AUC fold，無最佳模型')
                logf.write(f'{target} 無有效AUC fold，無最佳模型\n')
            logf.write('\n')
        # ======== 四任務平均分數（本地評估用） ========
        print_and_log_overall_mean(cv_scores_dict, list(TARGETS.keys()), logf)
        logf.write('\n')

if __name__ == '__main__':
    main()