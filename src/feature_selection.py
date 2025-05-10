"""
feature_selection.py

透過 CatBoostClassifier 預訓練計算特徵重要性並回傳 sklearn-like selector。
"""

import numpy as np
import config
import copy
from catboost import CatBoostClassifier
from model import gender_model, handed_model, play_years_model, level_model
from sklearn.model_selection import StratifiedShuffleSplit


def select_features(X: np.ndarray, y: np.ndarray, n_features_to_select: int, feature_names: list[str], task: str):
    """一次性特徵選擇：使用 CatBoostClassifier 計算特徵重要性並選擇 top k 特徵。

    Args:
        X (np.ndarray): 特徵矩陣，維度為 (樣本數, 特徵數)。
        y (np.ndarray): 樣本標籤。
        n_features_to_select (int): 欲保留的特徵數量 k。
        feature_names (list[str]): 特徵名稱列表。
        task (str): 任務類型，決定 objective、metric 與 importance_type。

    Returns:
        selector: 具有 `.get_support()` 方法的特徵選擇器，返回 bool 型 mask。
    """
    # 使用 CatBoostClassifier 作為特徵選擇模型
    model_map = {
        'gender': gender_model,
        'hold racket handed': handed_model,
        'play years': play_years_model,
        'level': level_model,
    }
    base_model = copy.deepcopy(model_map[task])
    model = base_model
    
    # 使用 StratifiedShuffleSplit 分割驗證集以進行早停
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=45)
    train_idx, val_idx = next(sss.split(X, y))
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=100,
        verbose=False
    )
    # 擷取特徵重要性
    importances = model.get_feature_importance()
    # 由大到小排序，取前 k
    indices = np.argsort(importances)[::-1][:n_features_to_select]
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[indices] = True

    class _Selector:
        """透過 get_support 回傳選中特徵 mask。"""
        def __init__(self, support: np.ndarray):
            self._support = support

        def get_support(self) -> np.ndarray:
            """回傳 bool 型 mask，表示被選中特徵位置。"""
            return self._support

    sel = _Selector(mask)
    # 保存特徵重要性以供全局選擇使用
    sel.importances_ = importances
    return sel


def select_global_features(all_importances: list[np.ndarray], feature_names: list[str], method: str, k: int):
    """
    全局特徵選擇：聚合多折特徵重要性，選擇前 k 個特徵

    Args:
        all_importances (list[np.ndarray]): 每折特徵重要性陣列清單
        feature_names (list[str]): 特徵名稱列表
        method (str): 聚合方式，目前支援 'mean_importance'
        k (int): 保留特徵數量

    Returns:
        selector: sklearn-like selector with get_support()
    """
    # 聚合前，標準化各折重要度向量長度
    norm_imps = []
    for imp in all_importances:
        arr = np.asarray(imp)
        # 若不是一維或長度不符則補零
        if arr.ndim != 1 or arr.shape[0] != len(feature_names):
            arr = np.zeros(len(feature_names))
        norm_imps.append(arr)
    imps = np.stack(norm_imps, axis=0)  # shape: (n_folds, n_features)
    if method == 'mean_importance':
        agg = imps.mean(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
    # 取 top k
    idxs = np.argsort(agg)[::-1][:k]
    mask = np.zeros(len(feature_names), dtype=bool)
    mask[idxs] = True
    class _SelectorGlobal:
        """返回全局選中特徵 mask"""
        def __init__(self, support: np.ndarray):
            self._support = support
        def get_support(self) -> np.ndarray:
            return self._support
    return _SelectorGlobal(mask)
