"""data_processing.py
資料處理與特徵工程模組。

此模組負責從六軸感測資料 (Ax, Ay, Az, Gx, Gy, Gz)
與 meta 資訊 (unique_id, player_id, mode, gender, hold racket handed, play years, level)
結合並萃取豐富的特徵，包括時域、頻域、小波、窗口摘要、
跨軸相關、分形維度及 Hjorth 參數。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, entropy
from scipy.fft import rfft, rfftfreq
import pywt  # 小波轉換
import config
import math
import json
from scipy.fftpack import dct  # 新增 DCT
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


def calc_advanced_freq_features(x: np.ndarray, prefix: str, fs: float = 85.0) -> dict:
    """計算頻域進階特徵：Dominant Frequency Amplitude、Band Power、Spectral Peak Count"""
    features = {}
    N = len(x)
    X = rfft(x)
    freqs = rfftfreq(N, d=1/fs)
    mag = np.abs(X)
    # 主頻幅值
    dom_idx = np.argmax(mag[1:]) + 1
    features[f'{prefix}_dominant_freq_amp'] = mag[dom_idx]
    # 頻帶能量
    for low, high in [(0,1), (1,3), (3,5)]:
        idx = np.where((freqs >= low) & (freqs < high))[0]
        features[f'{prefix}_band_{low}_{high}_power'] = np.sum(np.square(mag[idx]))
    # 頻譜峰值數量
    peaks = np.where((mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:]))[0]
    features[f'{prefix}_spectral_peak_count'] = len(peaks)
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


def calc_advanced_time_features(x: np.ndarray, prefix: str) -> dict:
    """計算時域進階特徵：平均絕對變化量、平均變化率、零交叉率、斜率均值/變異"""
    features = {}
    diffs = np.diff(x)
    features[f'{prefix}_mean_abs_change'] = np.mean(np.abs(diffs))
    features[f'{prefix}_mean_change_rate'] = np.mean(diffs / (x[:-1] + 1e-8))
    zero_crossings = np.where(np.diff(np.sign(x)) != 0)[0]
    features[f'{prefix}_zero_crossing_rate'] = len(zero_crossings) / len(x)
    features[f'{prefix}_slope_mean'] = np.mean(diffs)
    features[f'{prefix}_slope_std'] = np.std(diffs)
    return features


def calc_window_features(x: np.ndarray, prefix: str, window_size: int = 85, step: int = None) -> dict:
    """計算滑動視窗特徵：每窗平均、方差、最小、最大，並對窗級特徵取摘要"""
    if step is None:
        step = window_size // 2
    N = len(x)
    windows = [x[i:i+window_size] for i in range(0, N - window_size + 1, step)]
    features = {}
    if not windows:
        # 無足夠長度時填0
        for stat in ['mean', 'std', 'min', 'max']:
            features[f'{prefix}_win_{stat}_mean'] = 0.0
            features[f'{prefix}_win_{stat}_std'] = 0.0
        return features
    means = np.array([np.mean(w) for w in windows])
    vars_ = np.array([np.var(w) for w in windows])
    mins = np.array([np.min(w) for w in windows])
    maxs = np.array([np.max(w) for w in windows])
    # 摘要
    features[f'{prefix}_win_mean_mean'] = np.mean(means)
    features[f'{prefix}_win_mean_std'] = np.std(means)
    features[f'{prefix}_win_var_mean'] = np.mean(vars_)
    features[f'{prefix}_win_var_std'] = np.std(vars_)
    features[f'{prefix}_win_min_mean'] = np.mean(mins)
    features[f'{prefix}_win_min_std'] = np.std(mins)
    features[f'{prefix}_win_max_mean'] = np.mean(maxs)
    features[f'{prefix}_win_max_std'] = np.std(maxs)
    return features


def calc_cross_axis_features(data: np.ndarray) -> dict:
    """計算跨軸相關及 AccVec/GyroVec 比率"""
    Ax, Ay, Az, Gx, Gy, Gz = data.T
    features = {}
    def corr(x, y):
        if np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return np.corrcoef(x, y)[0,1]
    # 加速度各軸相關
    features['Ax_Ay_corr'] = corr(Ax, Ay)
    features['Ax_Az_corr'] = corr(Ax, Az)
    features['Ay_Az_corr'] = corr(Ay, Az)
    # 角速度各軸相關
    features['Gx_Gy_corr'] = corr(Gx, Gy)
    features['Gx_Gz_corr'] = corr(Gx, Gz)
    features['Gy_Gz_corr'] = corr(Gy, Gz)
    # AccVec/GyroVec mean ratio
    acc = np.sqrt(Ax**2 + Ay**2 + Az**2)
    gyro = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    features['AccGyro_mean_ratio'] = np.mean(acc) / (np.mean(gyro) + 1e-8)
    return features


def calc_fractal_dimension(x: np.ndarray, prefix: str) -> dict:
    """計算 Katz 分形維度 (Fractal Dimension)"""
    N = len(x)
    if N < 2:
        return {f'{prefix}_fractal_dimension': 0.0}
    L = np.sum(np.sqrt(1 + np.diff(x) ** 2))
    d = np.max(np.sqrt((np.arange(N)) ** 2 + (x - x[0]) ** 2))
    fd = np.log10(N) / (np.log10(N) + np.log10(L / d)) if d > 0 and L > 0 else 0.0
    return {f'{prefix}_fractal_dimension': fd}


def calc_hjorth_parameters(x: np.ndarray, prefix: str) -> dict:
    """計算 Hjorth 參數：Activity, Mobility, Complexity"""
    var_x = np.var(x)
    dx = np.diff(x)
    var_dx = np.var(dx)
    ddx = np.diff(dx) if len(dx) > 1 else np.array([0.0])
    var_ddx = np.var(ddx)
    activity = var_x
    mobility = np.sqrt(var_dx / (var_x + 1e-8)) if var_x > 0 else 0.0
    complexity = np.sqrt(var_ddx / (var_dx + 1e-8)) / (mobility + 1e-8) if var_dx > 0 else 0.0
    return {
        f'{prefix}_hjorth_activity': activity,
        f'{prefix}_hjorth_mobility': mobility,
        f'{prefix}_hjorth_complexity': complexity
    }


def calc_dct_features(x: np.ndarray, prefix: str) -> dict:
    """計算離散餘弦變換 (DCT) 特徵"""
    X = dct(x, norm='ortho')
    mag = np.abs(X)
    return {
        f'{prefix}_dct_mean': np.mean(mag),
        f'{prefix}_dct_std': np.std(mag),
        f'{prefix}_dct_energy': np.sum(np.square(mag))
    }


def extract_features_from_array(data: np.ndarray) -> dict:
    """根據六軸數據陣列萃取特徵"""
    Ax, Ay, Az, Gx, Gy, Gz = data.T
    features = {}
    for axis, arr in zip(['Ax','Ay','Az','Gx','Gy','Gz'], [Ax, Ay, Az, Gx, Gy, Gz]):
        features.update(calc_time_features(arr, axis))
        features.update(calc_advanced_time_features(arr, axis))
        features.update(calc_freq_features(arr, axis))
        features.update(calc_advanced_freq_features(arr, axis))
        features.update(calc_wavelet_features(arr, axis))
        features.update(calc_window_features(arr, axis))
        features.update(calc_fractal_dimension(arr, axis))
        features.update(calc_hjorth_parameters(arr, axis))
        features.update(calc_dct_features(arr, axis))  # 新增 DCT 特徵
    acc = np.sqrt(Ax**2 + Ay**2 + Az**2)
    gyro = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    features.update(calc_time_features(acc, 'AccVec'))
    features.update(calc_advanced_time_features(acc, 'AccVec'))
    features.update(calc_freq_features(acc, 'AccVec'))
    features.update(calc_advanced_freq_features(acc, 'AccVec'))
    features.update(calc_wavelet_features(acc, 'AccVec'))
    features.update(calc_window_features(acc, 'AccVec'))
    features.update(calc_fractal_dimension(acc, 'AccVec'))
    features.update(calc_hjorth_parameters(acc, 'AccVec'))
    features.update(calc_dct_features(acc, 'AccVec'))  # 新增 DCT 特徵
    features.update(calc_time_features(gyro, 'GyroVec'))
    features.update(calc_advanced_time_features(gyro, 'GyroVec'))
    features.update(calc_freq_features(gyro, 'GyroVec'))
    features.update(calc_advanced_freq_features(gyro, 'GyroVec'))
    features.update(calc_wavelet_features(gyro, 'GyroVec'))
    features.update(calc_window_features(gyro, 'GyroVec'))
    features.update(calc_fractal_dimension(gyro, 'GyroVec'))
    features.update(calc_hjorth_parameters(gyro, 'GyroVec'))
    features.update(calc_dct_features(gyro, 'GyroVec'))  # 新增 DCT 特徵
    # 跨軸相關特徵
    features.update(calc_cross_axis_features(data))
    return features


def extract_features_from_txt(txt_path: str, augment: bool = True) -> list:
    """讀取單一檔案並萃取所有特徵（含小波特徵）"""
    data = np.loadtxt(txt_path)
    feature_dicts = []
    # 原始特徵
    feature_dicts.append(extract_features_from_array(data))
    return feature_dicts

def process_test_data():
    """處理測試集"""
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
        test_out_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')
        print(f"已輸出 {test_output_path}")

if __name__ == '__main__':
    process_test_data()