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
FEATURE_SELECTION_N_FEATURES = 1000  # 一次性特徵選擇保留特徵數量

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
def _gen_wavelet_feats(prefix):
    return [f'{prefix}_wavelet_mean', f'{prefix}_wavelet_std', f'{prefix}_wavelet_energy']

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

FEATURES = [
    'mode',
    # Ax
    'Ax_mean', 'Ax_std', 'Ax_var', 'Ax_min', 'Ax_max', 'Ax_median', 'Ax_q25', 'Ax_q75', 'Ax_kurtosis', 'Ax_skew', 'Ax_rms', 'Ax_energy', 'Ax_dominant_freq', 'Ax_spectral_centroid', 'Ax_spectral_entropy', 'Ax_spectral_energy',
    *_gen_wavelet_feats('Ax'),
    # Ay
    'Ay_mean', 'Ay_std', 'Ay_var', 'Ay_min', 'Ay_max', 'Ay_median', 'Ay_q25', 'Ay_q75', 'Ay_kurtosis', 'Ay_skew', 'Ay_rms', 'Ay_energy', 'Ay_dominant_freq', 'Ay_spectral_centroid', 'Ay_spectral_entropy', 'Ay_spectral_energy',
    *_gen_wavelet_feats('Ay'),
    # Az
    'Az_mean', 'Az_std', 'Az_var', 'Az_min', 'Az_max', 'Az_median', 'Az_q25', 'Az_q75', 'Az_kurtosis', 'Az_skew', 'Az_rms', 'Az_energy', 'Az_dominant_freq', 'Az_spectral_centroid', 'Az_spectral_entropy', 'Az_spectral_energy',
    *_gen_wavelet_feats('Az'),
    # Gx
    'Gx_mean', 'Gx_std', 'Gx_var', 'Gx_min', 'Gx_max', 'Gx_median', 'Gx_q25', 'Gx_q75', 'Gx_kurtosis', 'Gx_skew', 'Gx_rms', 'Gx_energy', 'Gx_dominant_freq', 'Gx_spectral_centroid', 'Gx_spectral_entropy', 'Gx_spectral_energy',
    *_gen_wavelet_feats('Gx'),
    # Gy
    'Gy_mean', 'Gy_std', 'Gy_var', 'Gy_min', 'Gy_max', 'Gy_median', 'Gy_q25', 'Gy_q75', 'Gy_kurtosis', 'Gy_skew', 'Gy_rms', 'Gy_energy', 'Gy_dominant_freq', 'Gy_spectral_centroid', 'Gy_spectral_entropy', 'Gy_spectral_energy',
    *_gen_wavelet_feats('Gy'),
    # Gz
    'Gz_mean', 'Gz_std', 'Gz_var', 'Gz_min', 'Gz_max', 'Gz_median', 'Gz_q25', 'Gz_q75', 'Gz_kurtosis', 'Gz_skew', 'Gz_rms', 'Gz_energy', 'Gz_dominant_freq', 'Gz_spectral_centroid', 'Gz_spectral_entropy', 'Gz_spectral_energy',
    *_gen_wavelet_feats('Gz'),
    # AccVec
    'AccVec_mean', 'AccVec_std', 'AccVec_var', 'AccVec_min', 'AccVec_max', 'AccVec_median', 'AccVec_q25', 'AccVec_q75', 'AccVec_kurtosis', 'AccVec_skew', 'AccVec_rms', 'AccVec_energy', 'AccVec_dominant_freq', 'AccVec_spectral_centroid', 'AccVec_spectral_entropy', 'AccVec_spectral_energy',
    *_gen_wavelet_feats('AccVec'),
    # GyroVec
    'GyroVec_mean', 'GyroVec_std', 'GyroVec_var', 'GyroVec_min', 'GyroVec_max', 'GyroVec_median', 'GyroVec_q25', 'GyroVec_q75', 'GyroVec_kurtosis', 'GyroVec_skew', 'GyroVec_rms', 'GyroVec_energy', 'GyroVec_dominant_freq', 'GyroVec_spectral_centroid', 'GyroVec_spectral_entropy', 'GyroVec_spectral_energy',
    *_gen_wavelet_feats('GyroVec'),
]

# 將進階時域特徵加入 FEATURES
for _prefix in ['Ax','Ay','Az','Gx','Gy','Gz','AccVec','GyroVec']:
    FEATURES.extend(_gen_advanced_time_feats(_prefix))

# 將進階頻域特徵加入 FEATURES
for _prefix in ['Ax','Ay','Az','Gx','Gy','Gz','AccVec','GyroVec']:
    FEATURES.extend(_gen_advanced_freq_feats(_prefix))

# 將滑動視窗特徵加入 FEATURES
for _prefix in ['Ax','Ay','Az','Gx','Gy','Gz','AccVec','GyroVec']:
    FEATURES.extend(_gen_window_feats(_prefix))

# 新增高階特徵：分形維度、Hjorth 參數
for _prefix in ['Ax','Ay','Az','Gx','Gy','Gz','AccVec','GyroVec']:
    FEATURES.append(f'{_prefix}_fractal_dimension')
    FEATURES.append(f'{_prefix}_hjorth_activity')
    FEATURES.append(f'{_prefix}_hjorth_mobility')
    FEATURES.append(f'{_prefix}_hjorth_complexity')

# 新增 DCT 特徵
for _prefix in ['Ax','Ay','Az','Gx','Gy','Gz','AccVec','GyroVec']:
    FEATURES.append(f'{_prefix}_dct_mean')
    FEATURES.append(f'{_prefix}_dct_std')
    FEATURES.append(f'{_prefix}_dct_energy')

# 將跨軸相關特徵加入 FEATURES
_CROSS_AXES = [('Ax','Ay'), ('Ax','Az'), ('Ay','Az'), ('Gx','Gy'), ('Gx','Gz'), ('Gy','Gz')]
for x, y in _CROSS_AXES:
    FEATURES.append(f'{x}_{y}_corr')
FEATURES.append('AccGyro_mean_ratio')
