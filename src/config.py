# 檔案路徑
import os  # 用於原始資料目錄定義
TRAIN_CSV = 'data/training.csv'
TEST_CSV = 'data/testing.csv'

# per-task training CSV
TRAIN_CSVS = {
    'gender': 'data/training_gender.csv',
    'hold racket handed': 'data/training_hold_racket_handed.csv',
    'play years': 'data/training_play_years.csv',
    'level': 'data/training_level.csv'
}

# 固定目標
PLAYER_ID_COL = 'player_id'  # GroupKFold用
BINARY_TARGETS = {'gender', 'hold racket handed'}
BINARY_TARGETS_ORDER = ['gender', 'hold racket handed']
MULTI_TARGETS = {'play years', 'level'}
PLAY_YEARS_COLS = ['play years_0', 'play years_1', 'play years_2']
LEVEL_COLS = ['level_2', 'level_3', 'level_4', 'level_5']

# ============= TODO =============
# 訓練相關參數
K_FOLD = 3
RANDOM_SEED = 42
VERBOSE = 200

# 模型設定：'catboost' 或 'tabpfn'
MODEL_TYPE = 'catboost'
PHE_TIME = 60 * 10
FEATURE_SELECTION_N_FEATURES = 3000  # 一次性特徵選擇保留特徵數量， 300是個選擇、如果用tabpfn則選擇100

# 全局特徵選擇設定
GLOBAL_FEATURE_SELECTION_METHOD = 'mean_importance'  # 聚合多折重要性方法
GLOBAL_TOP_K_FEATURES = FEATURE_SELECTION_N_FEATURES  # 全局特徵選擇保留特徵數量


# ================================
# 特徵工程快取配置
FEATURE_CACHE_DIR = 'feature_cache'  # 快取特徵工程結果的資料夾
ENABLE_FEATURE_CACHE = True            # 是否啟用特徵工程快取

# 原始訓練資料目錄。
RAW_TRAIN_DATA_DIR = os.path.join('data', 'raw', 'train_data')

# 使用的特徵（含小波特徵）
def _gen_wavelet_feats(prefix, level=3):
    """生成多級小波分解的統計特徵名稱

    Args:
        prefix (str): 特徵名稱的前綴 (例如 'Ax')。
        level (int): 小波分解的級數，預設為 3。

    Returns:
        list: 包含多級小波特徵名稱的列表。
    """
    features = []
    stats = ['mean', 'std', 'energy']

    # 添加最終近似係數 (cA_level) 的特徵
    for stat in stats:
        features.append(f'{prefix}_wavelet_L{level}_cA_{stat}')

    # 添加各級細節係數 (cD_level, ..., cD_1) 的特徵
    for i in range(level, 0, -1):
        for stat in stats:
            features.append(f'{prefix}_wavelet_L{i}_cD_{stat}')

    return features

def _gen_advanced_time_feats(prefix):
    """生成進階時域特徵名稱"""
    return [
        f'{prefix}_mean_abs_change',
        f'{prefix}_mean_change_rate',
        f'{prefix}_zero_crossing_rate',
        f'{prefix}_slope_mean',
        f'{prefix}_slope_std',
    ]

def _gen_advanced_freq_feats(prefix):
    """生成進階頻域特徵名稱"""
    feats = [f'{prefix}_dominant_freq_amp']
    for low, high in [(0,1), (1,3), (3,5)]:
        feats.append(f'{prefix}_band_{low}_{high}_power')
    feats.append(f'{prefix}_spectral_peak_count')
    return feats

def _gen_window_feats(prefix):
    """生成滑動視窗摘要特徵名稱"""
    stats = ['mean', 'var', 'min', 'max']
    feats = []
    for s in stats:
        feats.append(f'{prefix}_win_{s}_mean')
        feats.append(f'{prefix}_win_{s}_std')
    return feats

def _gen_autocorr_feats(prefix):
    """生成 Lag-1 自相關特徵名稱"""
    return [f'{prefix}_autocorr_lag1']

def _gen_jerk_feats(prefix):
    """生成 Jerk 信號的基礎時域特徵名稱"""
    # 與 calc_time_features 保持一致
    # 包含：mean, std, var, min, max, median, q25, q75, kurtosis, skew, rms, energy
    # 不包含：range, iqr
    basic_stats = ['mean', 'std', 'var', 'min', 'max', 'median', 'q25', 'q75', 'kurtosis', 'skew', 'rms', 'energy']
    return [f'{prefix}_{stat}' for stat in basic_stats]

AXES = ['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']
VECTORS = ['AccVec', 'GyroVec']
JERK_AXES = [f'{ax}_jerk' for ax in AXES]
JERK_VECTORS = [f'{vec}_jerk' for vec in VECTORS]

FEATURES = ['mode']
# 1. 基礎時域特徵
for prefix in AXES + VECTORS:
    FEATURES.extend([
        f'{prefix}_mean', f'{prefix}_std', f'{prefix}_var', f'{prefix}_min', f'{prefix}_max', f'{prefix}_median', f'{prefix}_q25', f'{prefix}_q75', f'{prefix}_kurtosis', f'{prefix}_skew', f'{prefix}_rms', f'{prefix}_energy', f'{prefix}_dominant_freq', f'{prefix}_spectral_centroid', f'{prefix}_spectral_entropy', f'{prefix}_spectral_energy',
    ])

# 2. 進階時域特徵
for prefix in AXES + VECTORS:
    FEATURES.extend(_gen_advanced_time_feats(prefix))

# 3. 基礎頻域特徵
for prefix in AXES + VECTORS:
    FEATURES.extend(_gen_advanced_freq_feats(prefix))

# 4. 小波特徵 (3級)
for prefix in AXES + VECTORS:
    FEATURES.extend(_gen_wavelet_feats(prefix, level=3))

# 5. 滑動視窗摘要特徵
for prefix in AXES + VECTORS:
    FEATURES.extend(_gen_window_feats(prefix))

# 6. 碎形維度特徵
for prefix in AXES + VECTORS:
    FEATURES.append(f'{prefix}_fractal_dimension')
    FEATURES.append(f'{prefix}_hjorth_activity')
    FEATURES.append(f'{prefix}_hjorth_mobility')
    FEATURES.append(f'{prefix}_hjorth_complexity')

# 7. DCT 特徵
for prefix in AXES + VECTORS:
    FEATURES.append(f'{prefix}_dct_mean')
    FEATURES.append(f'{prefix}_dct_std')
    FEATURES.append(f'{prefix}_dct_energy')

# 8. 自相關特徵
for prefix in AXES + VECTORS:
    FEATURES.extend(_gen_autocorr_feats(prefix))

# 9. Jerk 特徵 (基礎時域)
for prefix in JERK_AXES + JERK_VECTORS:
    FEATURES.extend(_gen_jerk_feats(prefix))

# 10. 跨軸相關特徵
_CROSS_AXES = [('Ax','Ay'), ('Ax','Az'), ('Ay','Az'), ('Gx','Gy'), ('Gx','Gz'), ('Gy','Gz')]
for x, y in _CROSS_AXES:
    FEATURES.append(f'{x}_{y}_corr')
FEATURES.append('AccGyro_mean_ratio')

# 11. Gravity/Body Acc 分離特徵
SEPARATED_AXES = ['bodyAx','bodyAy','bodyAz','gravAx','gravAy','gravAz','bodyAccVec','gravAccVec']
for prefix in SEPARATED_AXES:
    FEATURES.extend([
        f'{prefix}_mean', f'{prefix}_std', f'{prefix}_var', f'{prefix}_min', f'{prefix}_max', f'{prefix}_median', f'{prefix}_q25', f'{prefix}_q75', f'{prefix}_kurtosis', f'{prefix}_skew', f'{prefix}_rms', f'{prefix}_energy',
        f'{prefix}_dominant_freq', f'{prefix}_spectral_centroid', f'{prefix}_spectral_entropy', f'{prefix}_spectral_energy'
    ])
    FEATURES.extend(_gen_advanced_time_feats(prefix))
    FEATURES.extend(_gen_advanced_freq_feats(prefix))
    FEATURES.extend(_gen_wavelet_feats(prefix))
    FEATURES.extend(_gen_window_feats(prefix))
    FEATURES.append(f'{prefix}_fractal_dimension')
    FEATURES.append(f'{prefix}_hjorth_activity')
    FEATURES.append(f'{prefix}_hjorth_mobility')
    FEATURES.append(f'{prefix}_hjorth_complexity')
    FEATURES.extend(_gen_autocorr_feats(prefix))
    FEATURES.extend([f'{prefix}_dct_mean', f'{prefix}_dct_std', f'{prefix}_dct_energy'])

# 12. 跨軸和向量相關性特徵
_CORRESPONDING_AXES = [('Ax', 'Gx'), ('Ay', 'Gy'), ('Az', 'Gz')]
for x, y in _CORRESPONDING_AXES:
    FEATURES.append(f'{x}_{y}_corr')
FEATURES.append('AccVec_GyroVec_corr')

# 移除重複特徵（以防萬一）
FEATURES = sorted(list(set(FEATURES)))

print(f"Total features: {len(FEATURES)}")
