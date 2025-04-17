"""
資料處理與特徵工程
====================
本模組負責將原始時序感測資料 (Ax, Ay, Az, Gx, Gy, Gz) 及 meta 資訊 (unique_id, player_id, mode, gender, hold racket handed, play years, level)
結合並萃取豐富的時域與頻域特徵，包含：
- 均值、標準差、方差、最大/最小值、中位數、四分位數、峰度、偏度、過零率、均方根、能量
- 主頻、頻譜質心、頻譜熵、頻譜能量
- 合加速度、合角速度及其所有統計特徵
最終產生 training.csv，供後續模型訓練使用。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import rfft, rfftfreq

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
    # 均方根
    features[f'{prefix}_rms'] = np.sqrt(np.mean(np.square(x)))
    # 能量
    features[f'{prefix}_energy'] = np.sum(np.square(x))
    return features

# 頻域特徵
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

# 組合訊號
def calc_composite_signal(arrays: list) -> np.ndarray:
    """計算合加速度或合角速度"""
    return np.sqrt(np.sum([a**2 for a in arrays], axis=0))

# ========== 主處理流程 ==========
def extract_features_from_txt(txt_path: str) -> dict:
    """讀取單一 txt，計算所有特徵"""
    data = np.loadtxt(txt_path)
    feature_dict = {}
    channel_names = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
    for i, ch in enumerate(channel_names):
        x = data[:, i]
        feature_dict.update(calc_time_features(x, ch))
        feature_dict.update(calc_freq_features(x, ch))
    # 合加速度
    acc = calc_composite_signal([data[:, 0], data[:, 1], data[:, 2]])
    feature_dict.update(calc_time_features(acc, 'AccVec'))
    feature_dict.update(calc_freq_features(acc, 'AccVec'))
    # 合角速度
    gyro = calc_composite_signal([data[:, 3], data[:, 4], data[:, 5]])
    feature_dict.update(calc_time_features(gyro, 'GyroVec'))
    feature_dict.update(calc_freq_features(gyro, 'GyroVec'))
    return feature_dict

if __name__ == '__main__':
    # 設定路徑
    info_path = os.path.join('data', 'raw', 'train_info.csv')
    txt_dir = os.path.join('data', 'raw', 'train_data')
    output_path = os.path.join('data', 'training.csv')
    # 讀取 meta 資訊
    info_df = pd.read_csv(info_path)
    features_list = []
    for idx, row in info_df.iterrows():
        uid = row['unique_id']
        txt_path = os.path.join(txt_dir, f'{uid}.txt')
        if not os.path.exists(txt_path):
            print(f"找不到 {txt_path}")
            continue
        feat = extract_features_from_txt(txt_path)
        meta = row.to_dict()
        # 移除 cut point 欄位
        if 'cut_point' in meta:
            meta.pop('cut_point')
        meta.update(feat)
        features_list.append(meta)
    out_df = pd.DataFrame(features_list)
    out_df.to_csv(output_path, index=False)
    print(f"已輸出 {output_path}")

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
            feat = extract_features_from_txt(txt_path)
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