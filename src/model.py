from catboost import CatBoostClassifier
import config

# gender_model 用於預測性別（二分類）
gender_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=9,
    loss_function='Logloss',
    verbose=config.VERBOSE,
    early_stopping_rounds=30,
    random_seed=config.RANDOM_SEED
)

# handed_model 用於預測持拍手（二分類）
handed_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=9,
    loss_function='Logloss',
    verbose=config.VERBOSE,
    early_stopping_rounds=30,
    random_seed=config.RANDOM_SEED
)

# play_years_model 用於預測打球年資（三分類）
play_years_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=9,
    loss_function='MultiClass',
    classes_count=3,
    verbose=config.VERBOSE,
    early_stopping_rounds=30,
    random_seed=config.RANDOM_SEED
)

# level_model 用於預測球技等級（四分類）
level_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=9,
    loss_function='MultiClass',
    classes_count=4,
    verbose=config.VERBOSE,
    early_stopping_rounds=30,
    random_seed=config.RANDOM_SEED
)