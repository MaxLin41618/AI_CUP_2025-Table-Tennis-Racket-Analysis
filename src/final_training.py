"""
final_training.py

使用整個訓練集，套用全局特徵選擇與 BorderlineSMOTE，訓練最終模型並儲存。
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import pickle

import config
from data_processing import extract_features_from_array
from model import gender_model, handed_model, play_years_model, level_model
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import BorderlineSMOTE
from catboost import Pool
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier
from tabpfn import TabPFNClassifier
from utils import compute_class_weights
import time

def main():
    """主流程：載入數據、特徵、增強、訓練最終模型並儲存。"""
    # 設定訓練資料集    
    try:
        df = pd.read_csv(config.TEST_CSV, nrows=0)
        config.FEATURES = ['mode'] + [c for c in df.columns if c != 'unique_id']
        # 保持順序並移除重複
        config.FEATURES = list(dict.fromkeys(config.FEATURES))
        print(f"更新後的特徵數量: {len(config.FEATURES)}")
    except Exception as e:
        print(f"無法更新特徵列表: {e}")
        
    # 設定儲存資料夾
    save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('models', f'{save_time}_final')
    os.makedirs(save_dir, exist_ok=True)

    # 載入全局選中特徵
    with open(os.path.join('data', 'global_permutation_selected_features.json'), 'r', encoding='utf-8-sig') as f:
        global_selected = json.load(f)

    # 讀取原始訓練訊息
    info_path = os.path.join('data', 'raw', 'train_info.csv')
    info_df = pd.read_csv(info_path)
    uids_all = info_df['unique_id'].tolist()

    # 直接從 training.csv 讀取特徵
    print("直接從 training.csv 讀取特徵...")
    # 讀取 training.csv
    train_df = pd.read_csv(config.TRAIN_CSV)
    # 設定 'unique_id' 為索引
    train_df = train_df.set_index('unique_id')
    # 排除不需要的欄位
    exclude_cols = ['player_id', 'gender', 'hold racket handed', 'play years', 'level']
    feature_cols = [col for col in train_df.columns if col not in exclude_cols]
    # 選取需要的特徵
    train_df = train_df[feature_cols]
    print(f"最終訓練使用的特徵數量: {len(train_df.columns)}")
    
    # 確保 config.FEATURES 存在，如果不存在則從 train_df 的列名創建
    if not hasattr(config, 'FEATURES'):
        config.FEATURES = list(train_df.columns)

    # 目標模型對應
    TARGETS = {
        'gender': gender_model,
        'hold racket handed': handed_model,
        'play years': play_years_model,
        'level': level_model
    }
    use_tabpfn = config.MODEL_TYPE.lower() == 'tabpfn'

    # 分別訓練最終模型
    for target, base_model in TARGETS.items():
        print(f'訓練最終模型：{target}')

        # 選取有效特徵
        selected_feats = global_selected[target]
        valid_feats = [feat for feat in selected_feats if feat in train_df.columns]
        print(f'原始特徵數量: {len(selected_feats)}, 有效特徵數量: {len(valid_feats)}')
        if len(valid_feats) == 0:
            print(f'警告: {target} 沒有有效特徵，使用所有特徵')
            X = train_df
        else:
            X = train_df[valid_feats]
        y = info_df[target]
        # 標籤編碼
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # 處理缺失值 (NaN)，因為 BorderlineSMOTE 不接受 NaN 值
        has_nan = X.isna().any().any()
        if has_nan:
            print(f"檢測到 NaN 值，進行填補處理")
            # 使用均值填補缺失值
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_imputed = pd.DataFrame(
                imputer.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_imputed = X
            
        # Borderline-SMOTE 過採樣
        smote = BorderlineSMOTE(random_state=config.RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X_imputed, y_encoded)

        # 模型訓練與儲存
        if use_tabpfn:
            # TabPFNClassifier
            cat_idx = [selected_feats.index('mode')] if 'mode' in selected_feats else []
            # NOTE: 可註解改用 AutoTabPFNClassifier
            model = copy.deepcopy(TabPFNClassifier(categorical_features_indices=cat_idx, random_state=config.RANDOM_SEED))
            # model = copy.deepcopy(AutoTabPFNClassifier(max_time=config.PHE_TIME, preset='avoid_overfitting', device='cuda', categorical_feature_indices=cat_idx, random_state=config.RANDOM_SEED))
            model.fit(X_res.values, y_res)
            model_path = os.path.join(save_dir, f'{target}.tabpfn')
            with open(model_path, 'wb') as mf:
                pickle.dump(model, mf)
        else:
            # CatBoostClassifier
            cat_features_loop = [f for f in ['mode'] if f in selected_feats]
            X_train = X_res.copy()
            if cat_features_loop:
                for f_name in cat_features_loop:
                    X_train[f_name] = X_train[f_name].astype(str)
            model = copy.deepcopy(base_model)
            train_pool = Pool(X_train, y_res, cat_features=cat_features_loop)
            model.fit(train_pool)
            model_path = os.path.join(save_dir, f'{target}.cbm')
            model.save_model(model_path)
        print(f'已儲存模型：{model_path}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time: {elapsed_time/60:.2f} minutes")
