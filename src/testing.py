import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
import config
import pickle

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


def predict_and_save(models, test_df, output_path):
    """以最佳模型做預測，產生 submission.csv"""
    result = pd.DataFrame()
    result['unique_id'] = test_df['unique_id']
    X = test_df[config.FEATURES]
    # gender, hold racket handed: 二分類，取 proba 的 0，並依照固定順序
    for target in config.BINARY_TARGETS_ORDER:
        proba = models[target].predict_proba(X)[:,0]
        result[target] = np.round(proba, 4)
    # play years: 三分類，明確指定欄位順序
    proba = models['play years'].predict_proba(X)
    for i, col in enumerate(config.PLAY_YEARS_COLS):
        result[col] = np.round(proba[:,i], 4)
    # level: 四分類，明確指定欄位順序
    proba = models['level'].predict_proba(X)
    for i, col in enumerate(config.LEVEL_COLS):
        result[col] = np.round(proba[:,i], 4)
    # 明確指定輸出欄位順序
    columns = ['unique_id'] + config.BINARY_TARGETS_ORDER + config.PLAY_YEARS_COLS + config.LEVEL_COLS
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # 強制浮點數以小數點格式輸出，避免科學記號（如1.00E-04）
    result.to_csv(output_path, index=False, columns=columns, float_format='%.4f')
    print(f'已輸出 submission 至 {output_path}')

# ========== 主程式 ===========
def main():
    """主流程：載入數據、模型、預測、輸出 submission"""
    # 讀取測試集
    test_df = pd.read_csv(config.TEST_CSV)
    # 取得最新模型資料夾
    latest_model_dir = get_latest_model_dir()
    # 載入最佳模型
    models = load_best_models(latest_model_dir)
    # 預測並輸出
    predict_and_save(models, test_df, output_path='outputs/submission.csv')

if __name__ == '__main__':
    main()