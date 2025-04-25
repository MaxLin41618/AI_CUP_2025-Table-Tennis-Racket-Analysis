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
from tabpfn import TabPFNClassifier
from utils import compute_feature_fingerprint, save_feature_cache, load_feature_cache


def main():
    """主流程：載入數據、特徵、增強、訓練最終模型並儲存。"""
    # 設定儲存資料夾
    save_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join('models', f'{save_time}_final')
    os.makedirs(save_dir, exist_ok=True)

    # 載入全局選中特徵
    with open(os.path.join('data', 'global_selected_features.json'), 'r', encoding='utf-8-sig') as f:
        global_selected = json.load(f)

    # 讀取原始訓練訊息
    info_path = os.path.join('data', 'raw', 'train_info.csv')
    info_df = pd.read_csv(info_path)
    uids_all = info_df['unique_id'].tolist()

    # 讀取或計算並快取訓練樣本特徵
    fp = compute_feature_fingerprint(config.FEATURES)
    cache_fname = f"final_train_{fp}.pkl"
    cached = load_feature_cache(config.FEATURE_CACHE_DIR, cache_fname) if config.ENABLE_FEATURE_CACHE else None
    if cached is not None:
        print("使用最終訓練特徵快取...")
        train_df = cached['train_df']
    else:
        print("計算最終訓練特徵並快取...")
        feats_list = []
        for uid in uids_all:
            row_meta = info_df[info_df['unique_id'] == uid].iloc[0].to_dict()
            row_meta.pop('cut_point', None)
            txt_file = os.path.join('data', 'raw', 'train_data', f'{uid}.txt')
            data = np.loadtxt(txt_file)
            feat = extract_features_from_array(data)
            row = row_meta.copy()
            row.update(feat)
            feats_list.append(row)
        train_df = pd.DataFrame(feats_list)[config.FEATURES]
        if config.ENABLE_FEATURE_CACHE:
            save_feature_cache(config.FEATURE_CACHE_DIR, cache_fname, {'train_df': train_df})

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
        selected_feats = global_selected[target]
        X = train_df[selected_feats]
        y = info_df[target]
        # 標籤編碼
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Borderline-SMOTE 過採樣
        smote = BorderlineSMOTE(random_state=config.RANDOM_SEED)
        X_res, y_res = smote.fit_resample(X, y_encoded)

        # 模型訓練與儲存
        if use_tabpfn:
            # TabPFNClassifier
            cat_idx = [selected_feats.index('mode')] if 'mode' in selected_feats else []
            model = copy.deepcopy(TabPFNClassifier(categorical_features_indices=cat_idx, random_state=config.RANDOM_SEED))
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
