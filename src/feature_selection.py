"""
feature_selection.py

透過 LightGBMClassifier 預訓練計算特徵重要性並回傳 sklearn-like selector。
"""

import numpy as np
import config
from lightgbm import LGBMClassifier


def select_features(X: np.ndarray, y: np.ndarray, n_features_to_select: int, feature_names: list[str]):
    """一次性特徵選擇：使用 LightGBMClassifier 計算特徵重要性並選擇 top k 特徵。

    Args:
        X (np.ndarray): 特徵矩陣，維度為 (樣本數, 特徵數)。
        y (np.ndarray): 樣本標籤。
        n_features_to_select (int): 欲保留的特徵數量 k。
        feature_names (list[str]): 特徵名稱列表。

    Returns:
        selector: 具有 `.get_support()` 方法的特徵選擇器，返回 bool 型 mask。
    """
    # 使用 LightGBM 計算特徵重要性並選取 top k 特徵
    model = LGBMClassifier(random_state=config.RANDOM_SEED)
    model.fit(X, y)
    importances = model.feature_importances_
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

    return _Selector(mask)
