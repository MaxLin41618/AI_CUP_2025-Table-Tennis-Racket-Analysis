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
from scipy.signal import butter, filtfilt  # 新增濾波器
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


def calc_wavelet_features(data: np.ndarray, prefix: str, wavelet: str = 'db4', level: int = 3) -> dict:
    """計算多級小波特徵 (cA 和 cD 係數)

    Args:
        data (np.ndarray): 輸入的一維時間序列數據。
        prefix (str): 特徵名稱的前綴 (例如 'Ax')。
        wavelet (str): 使用的小波基函數，預設為 'db4'。
        level (int): 小波分解的級數，預設為 3。

    Returns:
        dict: 包含小波係數統計特徵的字典。
            鍵的格式為 f'{prefix}_wavelet_L{level}_{coeff_type}_{stat}'
            例如: 'Ax_wavelet_L1_cD_mean', 'Ax_wavelet_L3_cA_std'
    """
    features = {}
    coeffs = None
    try:
        # 執行多級小波分解
        # wavedec 返回 [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        coeffs = pywt.wavedec(data, wavelet, level=level)

        # 提取最終的近似係數 cA_n (n=level)
        cA = coeffs[0]
        features[f'{prefix}_wavelet_L{level}_cA_mean'] = np.mean(cA) if len(cA) > 0 else 0
        features[f'{prefix}_wavelet_L{level}_cA_std'] = np.std(cA) if len(cA) > 1 else 0
        features[f'{prefix}_wavelet_L{level}_cA_energy'] = np.sum(cA**2) if len(cA) > 0 else 0

        # 提取各級的細節係數 cD_n, cD_{n-1}, ..., cD_1
        # coeffs 的索引從 1 開始對應 cD_n, cD_{n-1}, ... cD_1
        for i in range(1, level + 1):
            cD = coeffs[i]
            current_level = level - i + 1 # 計算當前係數對應的層級 (cD_3, cD_2, cD_1)
            features[f'{prefix}_wavelet_L{current_level}_cD_mean'] = np.mean(cD) if len(cD) > 0 else 0
            features[f'{prefix}_wavelet_L{current_level}_cD_std'] = np.std(cD) if len(cD) > 1 else 0
            features[f'{prefix}_wavelet_L{current_level}_cD_energy'] = np.sum(cD**2) if len(cD) > 0 else 0

    except ValueError as e:
        # print(f"警告：計算 {prefix} 的小波特徵時出錯 ({e})。數據長度可能太短。填充 0。")
        # 如果分解失敗 (例如數據太短)，填充 0
        # 需要為所有預期的特徵填充 0
        # 填充 cA 特徵
        for stat in ['mean', 'std', 'energy']:
            features[f'{prefix}_wavelet_L{level}_cA_{stat}'] = 0
        # 填充 cD 特徵
        for i in range(1, level + 1):
            current_level = level - i + 1
            for stat in ['mean', 'std', 'energy']:
                features[f'{prefix}_wavelet_L{current_level}_cD_{stat}'] = 0

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
    """計算跨軸特徵，包含靜態相關性和動態滑窗相關性"""
    features = {}
    Ax, Ay, Az = data[:, 0], data[:, 1], data[:, 2]
    Gx, Gy, Gz = data[:, 3], data[:, 4], data[:, 5]

    def corr(x, y):
        # 確保 x 和 y 有相同的長度，以防萬一
        min_len = min(len(x), len(y))
        if min_len == 0:
            return 0.0
        return np.corrcoef(x[:min_len], y[:min_len])[0,1]

    # 加速度各軸相關 - 靜態
    features['Ax_Ay_corr'] = corr(Ax, Ay)
    features['Ax_Az_corr'] = corr(Ax, Az)
    features['Ay_Az_corr'] = corr(Ay, Az)
    
    # 角速度各軸相關 - 靜態
    features['Gx_Gy_corr'] = corr(Gx, Gy)
    features['Gx_Gz_corr'] = corr(Gx, Gz)
    features['Gy_Gz_corr'] = corr(Gy, Gz)
    
    # 跨感測器軸相關 - 靜態
    features['Ax_Gx_corr'] = corr(Ax, Gx)
    features['Ay_Gy_corr'] = corr(Ay, Gy)
    features['Az_Gz_corr'] = corr(Az, Gz)

    # AccVec/GyroVec mean ratio and correlation
    acc = np.sqrt(Ax**2 + Ay**2 + Az**2)
    gyro = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    features['AccGyro_mean_ratio'] = np.mean(acc) / (np.mean(gyro) + 1e-8)
    features['AccVec_GyroVec_corr'] = corr(acc, gyro)

    # 動態滑窗相關性特徵
    # 1. 加速度各軸
    acc_pairs = [('Ax_Ay', Ax, Ay), ('Ax_Az', Ax, Az), ('Ay_Az', Ay, Az)]
    for name, x, y in acc_pairs:
        wc = calc_windowed_corr(x, y)
        for stat, val in wc.items():
            features[f'{name}_windowed_corr_{stat}'] = val

    # 2. 角速度各軸
    gyro_pairs = [('Gx_Gy', Gx, Gy), ('Gx_Gz', Gx, Gz), ('Gy_Gz', Gy, Gz)]
    for name, x, y in gyro_pairs:
        wc = calc_windowed_corr(x, y)
        for stat, val in wc.items():
            features[f'{name}_windowed_corr_{stat}'] = val

    # 3. 跨感測器軸
    cross_pairs = [('Ax_Gx', Ax, Gx), ('Ay_Gy', Ay, Gy), ('Az_Gz', Az, Gz)]
    for name, x, y in cross_pairs:
        wc = calc_windowed_corr(x, y)
        for stat, val in wc.items():
            features[f'{name}_windowed_corr_{stat}'] = val

    return features


def calc_windowed_corr(x: np.ndarray, y: np.ndarray,
                      window_size: int = 85, step: int = 42) -> dict:
    """計算滑動視窗的相關性統計特徵
    
    Args:
        x: 第一個時間序列
        y: 第二個時間序列
        window_size: 視窗大小，預設 85 (約 1 秒)
        step: 視窗滑動步長，預設 42 (50% 重疊)
    
    Returns:
        dict: 包含 mean, std, min, max, median, 25%, 75%, skewness, kurtosis 等統計量
    """
    if step is None:
        step = window_size // 2
    
    N = len(x)
    corrs = []
    
    for i in range(0, N - window_size + 1, step):
        xi, yi = x[i:i+window_size], y[i:i+window_size]
        # 避免常數序列
        if xi.std() < 1e-8 or yi.std() < 1e-8:
            corrs.append(0.0)
        else:
            corrs.append(np.corrcoef(xi, yi)[0,1])
    
    if not corrs:
        return {
            'mean': 0.0, 'std': 0.0,
            'min': 0.0, 'max': 0.0,
            'median': 0.0,
            'q25': 0.0, 'q75': 0.0,
            'skew': 0.0, 'kurtosis': 0.0
        }
    
    arr = np.array(corrs)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'min': np.min(arr),
        'max': np.max(arr),
        'median': np.median(arr),
        'q25': np.percentile(arr, 25),
        'q75': np.percentile(arr, 75),
        'skew': skew(arr),
        'kurtosis': kurtosis(arr)
    }


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


def calc_autocorr_features(x: np.ndarray, prefix: str) -> dict:
    """計算 Lag-1 自相關係數"""
    features = {}
    if len(x) < 2:
        features[f'{prefix}_autocorr_lag1'] = 0.0
        return features
    # 計算自相關，處理標準差為零的情況
    x_std = np.std(x)
    if x_std == 0:
        features[f'{prefix}_autocorr_lag1'] = 0.0 # 或 1.0，取決於定義，常數序列與自身完全相關但標準差為0
    else:
        # np.corrcoef 會自動處理均值中心化
        autocorr = np.corrcoef(x[:-1], x[1:])[0, 1]
        # 處理計算結果為 NaN 的情況 (可能由極端值或非常短的序列引起)
        features[f'{prefix}_autocorr_lag1'] = autocorr if not np.isnan(autocorr) else 0.0
    return features


def extract_features_from_array(data: np.ndarray) -> dict:
    """根據六軸數據陣列萃取特徵"""
    features = {}
    if data.shape[0] < 2: # 需要至少 2 個點來計算 diff
        # print("警告：數據點少於 2，無法計算 Jerk 特徵。將跳過此 Trial 的所有特徵提取。")
        # 返回空字典或根據需求填充 NaN/0
        # 為了與 config 中的 FEATURES 列表匹配，最好填充 0
        # TODO: 更健壯的處理方式 - 填充所有預期特徵的 0 值
        return {}

    Ax, Ay, Az, Gx, Gy, Gz = data.T

    # 0. 分離重力與身體加速度
    fs = 85.0
    order = 4
    b_lp, a_lp = butter(order, 0.3/(fs/2), btype='low')
    b_hp, a_hp = butter(order, 0.3/(fs/2), btype='high')
    gravityAx = filtfilt(b_lp, a_lp, Ax)
    gravityAy = filtfilt(b_lp, a_lp, Ay)
    gravityAz = filtfilt(b_lp, a_lp, Az)
    bodyAx = filtfilt(b_hp, a_hp, Ax)
    bodyAy = filtfilt(b_hp, a_hp, Ay)
    bodyAz = filtfilt(b_hp, a_hp, Az)
    separated_data = {
        'bodyAx': bodyAx, 'bodyAy': bodyAy, 'bodyAz': bodyAz,
        'gravAx': gravityAx, 'gravAy': gravityAy, 'gravAz': gravityAz
    }
    for axis, arr in separated_data.items():
        if len(arr) == 0: continue
        features.update(calc_time_features(arr, axis))
        features.update(calc_advanced_time_features(arr, axis))
        features.update(calc_freq_features(arr, axis))
        features.update(calc_advanced_freq_features(arr, axis))
        features.update(calc_wavelet_features(arr, axis))
        features.update(calc_window_features(arr, axis))
        features.update(calc_fractal_dimension(arr, axis))
        features.update(calc_hjorth_parameters(arr, axis))
        features.update(calc_dct_features(arr, axis))
        features.update(calc_autocorr_features(arr, axis))

    # 0.5 計算分離向量大小特徵
    bodyAccVec = np.sqrt(bodyAx**2 + bodyAy**2 + bodyAz**2)
    gravAccVec = np.sqrt(gravityAx**2 + gravityAy**2 + gravityAz**2)
    vector_sep_data = {'bodyAccVec': bodyAccVec, 'gravAccVec': gravAccVec}
    for vec_name, vec_arr in vector_sep_data.items():
        if len(vec_arr) == 0: continue
        features.update(calc_time_features(vec_arr, vec_name))
        features.update(calc_advanced_time_features(vec_arr, vec_name))
        features.update(calc_freq_features(vec_arr, vec_name))
        features.update(calc_advanced_freq_features(vec_arr, vec_name))
        features.update(calc_wavelet_features(vec_arr, vec_name))
        features.update(calc_window_features(vec_arr, vec_name))
        features.update(calc_fractal_dimension(vec_arr, vec_name))
        features.update(calc_hjorth_parameters(vec_arr, vec_name))
        features.update(calc_dct_features(vec_arr, vec_name))
        features.update(calc_autocorr_features(vec_arr, vec_name))

    # 1. 計算原始信號特徵
    axes_data = {'Ax': Ax, 'Ay': Ay, 'Az': Az, 'Gx': Gx, 'Gy': Gy, 'Gz': Gz}
    for axis, arr in axes_data.items():
        if len(arr) == 0: continue # 跳過空陣列
        features.update(calc_time_features(arr, axis))
        features.update(calc_advanced_time_features(arr, axis))
        features.update(calc_freq_features(arr, axis))
        features.update(calc_advanced_freq_features(arr, axis))
        features.update(calc_wavelet_features(arr, axis))
        features.update(calc_window_features(arr, axis))
        features.update(calc_fractal_dimension(arr, axis))
        features.update(calc_hjorth_parameters(arr, axis))
        features.update(calc_dct_features(arr, axis))
        features.update(calc_autocorr_features(arr, axis))

    # 2. 計算向量大小特徵 (AccVec, GyroVec)
    AccVec = np.sqrt(Ax**2 + Ay**2 + Az**2)
    GyroVec = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    vector_data = {'AccVec': AccVec, 'GyroVec': GyroVec}
    for vec_name, vec_arr in vector_data.items():
        if len(vec_arr) == 0: continue
        features.update(calc_time_features(vec_arr, vec_name))
        features.update(calc_advanced_time_features(vec_arr, vec_name))
        features.update(calc_freq_features(vec_arr, vec_name))
        features.update(calc_advanced_freq_features(vec_arr, vec_name))
        features.update(calc_wavelet_features(vec_arr, vec_name))
        features.update(calc_window_features(vec_arr, vec_name))
        features.update(calc_fractal_dimension(vec_arr, vec_name))
        features.update(calc_hjorth_parameters(vec_arr, vec_name))
        features.update(calc_dct_features(vec_arr, vec_name))
        features.update(calc_autocorr_features(vec_arr, vec_name))

    # 3. 計算 Jerk 信號特徵
    # 計算單軸 Jerk (加速度和陀螺儀)
    JerkAx = np.diff(Ax)
    JerkAy = np.diff(Ay)
    JerkAz = np.diff(Az)
    JerkGx = np.diff(Gx)
    JerkGy = np.diff(Gy)
    JerkGz = np.diff(Gz)

    jerk_axes_data = {
        'Ax_jerk': JerkAx, 'Ay_jerk': JerkAy, 'Az_jerk': JerkAz,
        'Gx_jerk': JerkGx, 'Gy_jerk': JerkGy, 'Gz_jerk': JerkGz
    }

    for axis, arr in jerk_axes_data.items():
        if len(arr) == 0: continue
        # 對 Jerk 信號只提取基礎時域特徵
        features.update(calc_time_features(arr, axis))
        # 可以考慮添加其他特徵，例如進階時域特徵
        # features.update(calc_advanced_time_features(arr, axis))

    # 計算 Jerk 向量大小 (Acc Jerk Mag, Gyro Jerk Mag)
    if len(JerkAx) > 0: # 確保 diff 產生了結果
        JerkAccVec = np.sqrt(JerkAx**2 + JerkAy**2 + JerkAz**2)
        JerkGyroVec = np.sqrt(JerkGx**2 + JerkGy**2 + JerkGz**2)

        jerk_vector_data = {'AccVec_jerk': JerkAccVec, 'GyroVec_jerk': JerkGyroVec}
        for vec_name, vec_arr in jerk_vector_data.items():
            if len(vec_arr) == 0: continue
            # 對 Jerk 向量大小也只提取基礎時域特徵
            features.update(calc_time_features(vec_arr, vec_name))
            # features.update(calc_advanced_time_features(vec_arr, vec_name))

    # 4. 計算跨軸特徵
    features.update(calc_cross_axis_features(data))

    # 確保返回的特徵字典包含 config 中定義的所有特徵，即使某些計算失敗也用 0 填充
    # (這一步驟最好在調用此函數的外部完成，基於 config.FEATURES 列表)

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