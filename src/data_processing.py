"""
資料處理與特徵工程
====================
本模組負責將原始時序感測資料 (Ax, Ay, Az, Gx, Gy, Gz) 及 meta 資訊 (unique_id, player_id, mode, gender, hold racket handed, play years, level)
結合並萃取豐富的時域與頻域特徵，包含：
- 均值、標準差、方差、最大/最小值、中位數、四分位數、峰度、偏度、過零率、均方根、能量
- 主頻、頻譜質心、頻譜熵、頻譜能量
- 合加速度、合角速度及其所有統計特徵
- 小波轉換近似係數的均值、標準差、能量
- jitter 增強

為每個任務產生各自的training.csv，供後續模型訓練使用。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import rfft, rfftfreq
import pywt  # 小波轉換
import config
import math
import random
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# ========== 特徵計算工具 ==========
def calc_time_features(x: np.ndarray, prefix: str) -> dict:
    """計算時域特徵"""
    features = {}
    features[f'{prefix}_mean'] = np.mean(x)
    features[f'{prefix}_std'] = np.std(x)
    features[f'{prefix}_var'] = np.var(x)
    features[f'{prefix}_min'] = np.min(x)
    features[f'{prefix}_max'] = np.max(x)
    features[f'{prefix}_median'] = np.median(x)
    features[f'{prefix}_q25'] = np.percentile(x, 25)
    features[f'{prefix}_q75'] = np.percentile(x, 75)
    features[f'{prefix}_kurtosis'] = kurtosis(x)
    features[f'{prefix}_skew'] = skew(x)
    features[f'{prefix}_rms'] = np.sqrt(np.mean(np.square(x)))
    features[f'{prefix}_energy'] = np.sum(np.square(x))
    return features


def calc_freq_features(x: np.ndarray, prefix: str, fs: float = 85.0) -> dict:
    """計算頻域特徵 (需指定取樣頻率)"""
    features = {}
    N = len(x)
    X = rfft(x)
    freqs = rfftfreq(N, d=1/fs)
    mag = np.abs(X)
    # 主頻
    dom_idx = np.argmax(mag[1:]) + 1  # 跳過直流分量
    features[f'{prefix}_dominant_freq'] = freqs[dom_idx]
    # 頻譜質心
    spectral_centroid = np.sum(freqs * mag) / (np.sum(mag) + 1e-8)
    features[f'{prefix}_spectral_centroid'] = spectral_centroid
    # 頻譜熵
    mag_norm = mag / (np.sum(mag) + 1e-8)
    features[f'{prefix}_spectral_entropy'] = entropy(mag_norm)
    # 頻譜能量
    features[f'{prefix}_spectral_energy'] = np.sum(np.square(mag))
    return features


def calc_wavelet_features(x: np.ndarray, prefix: str, wavelet: str = 'db4') -> dict:
    """
    計算小波轉換特徵
    Args:
        x: 時序資料
        prefix: 特徵前綴
        wavelet: 小波基底（預設'db4'）
    Returns:
        dict: 小波特徵
    """
    features = {}
    # 進行一階離散小波轉換，分解出 approximation (cA) 與 detail (cD) 係數
    cA, cD = pywt.dwt(x, wavelet)  # cA: 近似(低頻)係數, cD: 細節(高頻)係數
    # 只取cA(近似)的統計特徵，避免特徵爆炸
    features[f'{prefix}_wavelet_mean'] = np.mean(cA)     # 近似係數均值
    features[f'{prefix}_wavelet_std'] = np.std(cA)       # 近似係數標準差
    features[f'{prefix}_wavelet_energy'] = np.sum(np.square(cA))  # 近似係數能量
    return features


def jitter_signal(x: np.ndarray, std_ratio: float) -> np.ndarray:
    """對單軸訊號加入高斯雜訊"""
    noise = np.random.normal(0, np.std(x) * std_ratio, size=x.shape)
    return x + noise


def extract_features_from_array(data: np.ndarray) -> dict:
    """根據六軸數據陣列萃取特徵"""
    Ax, Ay, Az, Gx, Gy, Gz = data.T
    features = {}
    for axis, arr in zip(['Ax','Ay','Az','Gx','Gy','Gz'], [Ax, Ay, Az, Gx, Gy, Gz]):
        features.update(calc_time_features(arr, axis))
        features.update(calc_freq_features(arr, axis))
        features.update(calc_wavelet_features(arr, axis))
    acc = np.sqrt(Ax**2 + Ay**2 + Az**2)
    gyro = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    features.update(calc_time_features(acc, 'AccVec'))
    features.update(calc_freq_features(acc, 'AccVec'))
    features.update(calc_wavelet_features(acc, 'AccVec'))
    features.update(calc_time_features(gyro, 'GyroVec'))
    features.update(calc_freq_features(gyro, 'GyroVec'))
    features.update(calc_wavelet_features(gyro, 'GyroVec'))
    return features


def extract_features_from_txt(txt_path: str, augment: bool = True) -> list:
    """讀取單一檔案並萃取所有特徵（含小波特徵）"""
    data = np.loadtxt(txt_path)
    feature_dicts = []
    # 原始特徵
    feature_dicts.append(extract_features_from_array(data))
    # 數據增強
    if augment and config.AUGMENT_JITTER_COUNT > 0:
        for _ in range(config.AUGMENT_JITTER_COUNT):
            jittered = np.stack(
                [jitter_signal(arr, config.AUGMENT_JITTER_STD_RATIO) for arr in data.T], axis=1
            )
            feature_dicts.append(extract_features_from_array(jittered))
    return feature_dicts

# ========== 主特徵萃取流程 ==========
if __name__ == '__main__':
    # 設定路徑
    info_path = os.path.join('data', 'raw', 'train_info.csv')
    txt_dir = os.path.join('data', 'raw', 'train_data')
    output_path = os.path.join('data', 'training.csv')
    # 讀取 meta 資訊
    info_df = pd.read_csv(info_path)
    # 建立各任務共用的 base_features
    base_features = []
    for idx, row in info_df.iterrows():
        uid = row['unique_id']
        txt_path = os.path.join(txt_dir, f'{uid}.txt')
        if not os.path.exists(txt_path):
            print(f"找不到 {txt_path}")
            continue
        data = np.loadtxt(txt_path)
        base_feat = extract_features_from_array(data)
        meta_base = row.to_dict()
        meta_base.pop('cut_point', None)
        meta_base.update(base_feat)
        base_features.append(meta_base)

    # 全量 jitter 增強：對每筆原始樣本生成 AUGMENT_JITTER_COUNT 筆 jitter
    jitter_list = []
    if config.AUGMENT_JITTER_COUNT > 0:
        for meta_base in base_features:
            uid = meta_base['unique_id']
            row0 = info_df[info_df['unique_id'] == uid].iloc[0].to_dict()
            row0.pop('cut_point', None)
            txt_file = os.path.join(txt_dir, f'{uid}.txt')
            data = np.loadtxt(txt_file)
            for _ in range(config.AUGMENT_JITTER_COUNT):
                jittered = np.stack([
                    jitter_signal(arr, config.AUGMENT_JITTER_STD_RATIO) for arr in data.T
                ], axis=1)
                feat = extract_features_from_array(jittered)
                meta_new = row0.copy()
                meta_new.update(feat)
                jitter_list.append(meta_new)
    # 合併原始樣本與全量 jitter
    base_df = pd.DataFrame(base_features + jitter_list)

    # per-task 訓練集平衡並產生 jitter 增強樣本
    for task, path in config.TRAIN_CSVS.items():
        grouped = base_df.groupby(task)
        counts = grouped.size().to_dict()
        max_n = max(counts.values())
        augmented = []
        total_need = sum(max_n - c for c in counts.values())
        for lbl, group in grouped:
            need = max_n - len(group)
            if need <= 0:
                continue
            samples = math.ceil(need / config.AUGMENT_JITTER_COUNT)
            uids = info_df.loc[info_df[task] == lbl, 'unique_id'].tolist()
            chosen = random.choices(uids, k=samples)
            for uid_sel in chosen:
                row_info = info_df[info_df['unique_id'] == uid_sel].iloc[0].to_dict()
                row_info.pop('cut_point', None)
                txt_p = os.path.join(txt_dir, f'{uid_sel}.txt')
                data_sel = np.loadtxt(txt_p)
                for _ in range(config.AUGMENT_JITTER_COUNT):
                    jittered = np.stack([
                        jitter_signal(arr, config.AUGMENT_JITTER_STD_RATIO) for arr in data_sel.T
                    ], axis=1)
                    feat = extract_features_from_array(jittered)
                    meta = row_info.copy()
                    meta.update(feat)
                    augmented.append(meta)
        augmented = augmented[:total_need]
        df_aug = pd.DataFrame(augmented) if augmented else pd.DataFrame(columns=base_df.columns)
        df_task = pd.concat([base_df, df_aug], ignore_index=True).sample(frac=1, random_state=config.RANDOM_SEED).reset_index(drop=True)
        df_task.to_csv(path, index=False)
        print(f"已輸出平衡後的 {path}")

    # ========== 測試集處理 ==========
    test_info_path = os.path.join('data', 'raw', 'test_info.csv')
    test_txt_dir = os.path.join('data', 'raw', 'test_data')
    test_output_path = os.path.join('data', 'testing.csv')
    if os.path.exists(test_info_path):
        test_info_df = pd.read_csv(test_info_path)
        test_features_list = []
        for idx, row in test_info_df.iterrows():
            uid = row['unique_id']
            txt_path = os.path.join(test_txt_dir, f'{uid}.txt')
            if not os.path.exists(txt_path):
                print(f"找不到 {txt_path}")
                continue
            # 原始測試資料（不做增強）
            feats = extract_features_from_txt(txt_path, augment=False)
            for feat in feats:
                meta = row.to_dict()
                if 'cut_point' in meta:
                    meta.pop('cut_point')
                if 'player_id' in meta:
                    meta.pop('player_id')
                meta.update(feat)
                test_features_list.append(meta)
        test_out_df = pd.DataFrame(test_features_list)
        test_out_df.to_csv(test_output_path, index=False)
        print(f"已輸出 {test_output_path}")
