# 檔案路徑
TRAIN_CSV = 'data/training.csv'
TEST_CSV = 'data/testing.csv'

# 固定目標
PLAYER_ID_COL = 'player_id'  # GroupKFold用
BINARY_TARGETS = {'gender', 'hold racket handed'}
BINARY_TARGETS_ORDER = ['gender', 'hold racket handed']
MULTI_TARGETS = {'play years', 'level'}
PLAY_YEARS_COLS = ['play years_0', 'play years_1', 'play years_2']
LEVEL_COLS = ['level_2', 'level_3', 'level_4', 'level_5']

# 訓練相關參數
K_FOLD = 3
RANDOM_SEED = 42
VERBOSE = 0

# 使用的特徵
FEATURES = [
    'mode',
    'Ax_mean', 'Ax_std', 'Ax_var', 'Ax_min', 'Ax_max', 'Ax_median', 'Ax_q25', 'Ax_q75', 'Ax_kurtosis', 'Ax_skew', 'Ax_rms', 'Ax_energy', 'Ax_dominant_freq', 'Ax_spectral_centroid', 'Ax_spectral_entropy', 'Ax_spectral_energy',
    'Ay_mean', 'Ay_std', 'Ay_var', 'Ay_min', 'Ay_max', 'Ay_median', 'Ay_q25', 'Ay_q75', 'Ay_kurtosis', 'Ay_skew', 'Ay_rms', 'Ay_energy', 'Ay_dominant_freq', 'Ay_spectral_centroid', 'Ay_spectral_entropy', 'Ay_spectral_energy',
    'Az_mean', 'Az_std', 'Az_var', 'Az_min', 'Az_max', 'Az_median', 'Az_q25', 'Az_q75', 'Az_kurtosis', 'Az_skew', 'Az_rms', 'Az_energy', 'Az_dominant_freq', 'Az_spectral_centroid', 'Az_spectral_entropy', 'Az_spectral_energy',
    'Gx_mean', 'Gx_std', 'Gx_var', 'Gx_min', 'Gx_max', 'Gx_median', 'Gx_q25', 'Gx_q75', 'Gx_kurtosis', 'Gx_skew', 'Gx_rms', 'Gx_energy', 'Gx_dominant_freq', 'Gx_spectral_centroid', 'Gx_spectral_entropy', 'Gx_spectral_energy',
    'Gy_mean', 'Gy_std', 'Gy_var', 'Gy_min', 'Gy_max', 'Gy_median', 'Gy_q25', 'Gy_q75', 'Gy_kurtosis', 'Gy_skew', 'Gy_rms', 'Gy_energy', 'Gy_dominant_freq', 'Gy_spectral_centroid', 'Gy_spectral_entropy', 'Gy_spectral_energy',
    'Gz_mean', 'Gz_std', 'Gz_var', 'Gz_min', 'Gz_max', 'Gz_median', 'Gz_q25', 'Gz_q75', 'Gz_kurtosis', 'Gz_skew', 'Gz_rms', 'Gz_energy', 'Gz_dominant_freq', 'Gz_spectral_centroid', 'Gz_spectral_entropy', 'Gz_spectral_energy',
    'AccVec_mean', 'AccVec_std', 'AccVec_var', 'AccVec_min', 'AccVec_max', 'AccVec_median', 'AccVec_q25', 'AccVec_q75', 'AccVec_kurtosis', 'AccVec_skew', 'AccVec_rms', 'AccVec_energy', 'AccVec_dominant_freq', 'AccVec_spectral_centroid', 'AccVec_spectral_entropy', 'AccVec_spectral_energy',
    'GyroVec_mean', 'GyroVec_std', 'GyroVec_var', 'GyroVec_min', 'GyroVec_max', 'GyroVec_median', 'GyroVec_q25', 'GyroVec_q75', 'GyroVec_kurtosis', 'GyroVec_skew', 'GyroVec_rms', 'GyroVec_energy', 'GyroVec_dominant_freq', 'GyroVec_spectral_centroid', 'GyroVec_spectral_entropy', 'GyroVec_spectral_energy'
]