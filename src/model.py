from catboost import CatBoostClassifier
import config

# gender_model 用於預測性別（二分類）
gender_model = CatBoostClassifier(
    iterations=6000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=config.VERBOSE,
    metric_period=1,
    early_stopping_rounds=10000,
    random_seed=config.RANDOM_SEED,
    thread_count=-1,
    task_type='GPU',
    use_best_model=False,
)

# handed_model 用於預測持拍手（二分類）
handed_model = CatBoostClassifier(
    iterations=6000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=config.VERBOSE,
    metric_period=1,
    early_stopping_rounds=10000,
    random_seed=config.RANDOM_SEED,
    thread_count=-1,
    task_type='GPU',
    use_best_model=False,
)

# play_years_model 用於預測打球年資（三分類）
play_years_model = CatBoostClassifier(
    iterations=6000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    classes_count=3,
    verbose=config.VERBOSE,
    metric_period=1,
    early_stopping_rounds=10000,
    random_seed=config.RANDOM_SEED,
    thread_count=-1,
    task_type='GPU',
    use_best_model=False,
)

# level_model 用於預測球技等級（四分類）
level_model = CatBoostClassifier(
    iterations=6000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    classes_count=4,
    verbose=config.VERBOSE,
    metric_period=1,
    early_stopping_rounds=10000,
    random_seed=config.RANDOM_SEED,
    thread_count=-1,
    task_type='GPU',
    use_best_model=False,
)