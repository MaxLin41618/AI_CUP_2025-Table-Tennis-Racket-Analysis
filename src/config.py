# 檔案路徑
import os  # 用於原始資料目錄定義
import pandas as pd  # 用於動態讀取 testing.csv 欄位
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
K_FOLD = 5
RANDOM_SEED = 48
VERBOSE = 500

# 模型設定：'catboost' 或 'tabpfn'
MODEL_TYPE = 'catboost'
PHE_TIME = 60 * 10
FEATURE_SELECTION_N_FEATURES = 1000  # 一次性特徵選擇保留特徵數量
PERM_FEATURE_SELECTION_K = 500      # permutation 特徵選擇保留特徵數量

# 全局特徵選擇設定
GLOBAL_FEATURE_SELECTION_METHOD = 'mean_importance'  # 聚合多折重要性方法
GLOBAL_TOP_K_FEATURES = FEATURE_SELECTION_N_FEATURES  # 全局特徵選擇保留特徵數量


# ================================
# 特徵工程快取配置
FEATURE_CACHE_DIR = 'feature_cache'  # 快取特徵工程結果的資料夾
ENABLE_FEATURE_CACHE = True            # 是否啟用特徵工程快取

# 原始訓練資料目錄。
RAW_TRAIN_DATA_DIR = os.path.join('data', 'raw', 'train_data')

if __name__ == '__main__':
    # 如果 testing.csv 已生成，使用其欄位更新 FEATURES 列表
    try:
        df = pd.read_csv(TEST_CSV, nrows=0)
        FEATURES = ['mode'] + [c for c in df.columns if c != 'unique_id']
        # 保持順序並移除重複
        FEATURES = list(dict.fromkeys(FEATURES))
        print(f"Total features: {len(FEATURES)}")
    except Exception:
        print("Error: testing.csv not found")
