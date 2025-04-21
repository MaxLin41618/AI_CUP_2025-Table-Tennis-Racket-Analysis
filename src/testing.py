import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
import config
from data_processing import extract_features_from_array
import pickle
import json

def get_latest_model_dir(base_dir='models'):
    """自動取得最新訓練的模型資料夾"""
    subdirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        raise FileNotFoundError('找不到任何 models 子資料夾')
    latest = max(subdirs, key=os.path.getmtime)
    return latest


def load_best_models(model_dir):
    """載入每個 target 的最佳模型"""
    models = {}
    for target in config.BINARY_TARGETS | config.MULTI_TARGETS:
        target_dir = os.path.join(model_dir, target)
        # 優先載入 TabPFN pickle 檔，其次載入 CatBoost cbm 檔
        tab_path = os.path.join(target_dir, f'best_{target}.tabpfn')
        cbm_path = os.path.join(target_dir, f'best_{target}.cbm')
        if os.path.exists(tab_path):
            with open(tab_path, 'rb') as f:
                models[target] = pickle.load(f)
        elif os.path.exists(cbm_path):
            m = CatBoostClassifier()
            m.load_model(cbm_path)
            models[target] = m
        else:
            raise FileNotFoundError(f'找不到 {tab_path} 或 {cbm_path}')
    return models


def predict_and_save(models, test_df, selected_features_dict, output_path):
    """以最佳模型做預測，產生 submission.csv"""
    result = pd.DataFrame()
    result['unique_id'] = test_df['unique_id']
    # 決定預測時使用 TabPFN 還是 CatBoost
    use_tabpfn = config.MODEL_TYPE.lower() == 'tabpfn'

    # 二分類任務
    for target in config.BINARY_TARGETS_ORDER:
        selected_feats = selected_features_dict[target]
        if use_tabpfn:
            # TabPFN 接受 numpy array
            X_np = test_df[selected_feats].values
            proba = models[target].predict_proba(X_np)[:, 0]
        else:
            # CatBoost 使用 Pool 處理類別特徵
            X_test_sel = test_df[selected_feats]
            cat_features_loop = [f for f in ['mode'] if f in selected_feats]
            test_pool = Pool(X_test_sel, cat_features=cat_features_loop)
            proba = models[target].predict_proba(test_pool)[:, 0]
        result[target] = np.round(proba, 4)

    # 三分類任務: play years
    selected_feats = selected_features_dict['play years']
    if use_tabpfn:
        X_np = test_df[selected_feats].values
        proba = models['play years'].predict_proba(X_np)
    else:
        X_test_sel = test_df[selected_feats]
        cat_features_loop = [f for f in ['mode'] if f in selected_feats]
        test_pool = Pool(X_test_sel, cat_features=cat_features_loop)
        proba = models['play years'].predict_proba(test_pool)
    for i, col in enumerate(config.PLAY_YEARS_COLS):
        result[col] = np.round(proba[:, i], 4)

    # 四分類任務: level
    selected_feats = selected_features_dict['level']
    if use_tabpfn:
        X_np = test_df[selected_feats].values
        proba = models['level'].predict_proba(X_np)
    else:
        X_test_sel = test_df[selected_feats]
        cat_features_loop = [f for f in ['mode'] if f in selected_feats]
        test_pool = Pool(X_test_sel, cat_features=cat_features_loop)
        proba = models['level'].predict_proba(test_pool)
    for i, col in enumerate(config.LEVEL_COLS):
        result[col] = np.round(proba[:, i], 4)

    # 明確指定輸出欄位順序
    columns = ['unique_id'] + config.BINARY_TARGETS_ORDER + config.PLAY_YEARS_COLS + config.LEVEL_COLS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 強制浮點數以小數點格式輸出，避免科學記號（如1.00E-04）
    result.to_csv(output_path, index=False, columns=columns, float_format='%.4f')
    print(f'已輸出 submission 至 {output_path}')


# ========== 主程式 ===========
def main():
    """主流程：載入數據、模型、預測、輸出 submission"""
    # 讀取測試集 meta 並產生測試特徵
    meta_df = pd.read_csv(config.TEST_CSV)
    feature_list = []
    for _, row in meta_df.iterrows():
        # 將 unique_id 轉為整數以對應檔案名稱
        uid = int(row['unique_id'])
        row_meta = row.to_dict()
        # 更新 unique_id 為整數，避免 CSV 顯示浮點
        row_meta['unique_id'] = uid
        txt_path = os.path.join('data', 'raw', 'test_data', f'{uid}.txt')
        data = np.loadtxt(txt_path)
        feat = extract_features_from_array(data)
        row_meta.update(feat)
        feature_list.append(row_meta)
    test_df = pd.DataFrame(feature_list)
    # 取得最新模型資料夾
    latest_model_dir = get_latest_model_dir()
    # 載入最佳模型
    models = load_best_models(latest_model_dir)
    # 從 data 資料夾讀取已選特徵映射
    selected_json_path = os.path.join('data', 'selected_features.json')
    with open(selected_json_path, 'r', encoding='utf-8-sig') as jf:
        selected_features_dict = json.load(jf)
    # 預測並輸出
    predict_and_save(models, test_df, selected_features_dict, output_path='outputs/submission.csv')


if __name__ == '__main__':
    main()