"""
工具函式集
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importance(model, feature_names, save_path, title=None, top_n=15):
    """
    使用CatBoost內建方法繪製特徵重要度圖（僅列出前top_n個），並儲存為圖片。

    Args:
        model: CatBoostClassifier 已訓練模型
        feature_names: list 特徵名稱
        save_path: str 圖片儲存路徑
        title: str, optional 圖片標題
        top_n: int, optional 只顯示前N大特徵
    """
    importance = model.get_feature_importance()
    indices = importance.argsort()[::-1][:top_n]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    plt.figure(figsize=(8, max(5, top_n//2)))
    plt.barh(range(len(sorted_names)), sorted_importance[::-1], align='center')
    plt.yticks(range(len(sorted_names)), sorted_names[::-1], fontsize=9)
    plt.xlabel('Importance')
    plt.title(title or 'Feature Importance')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_class_weights(y):
    """
    根據y計算每個類別的class_weight，回傳list。
    Args:
        y: array-like，已經label encode過的目標
    Returns:
        list，每個類別的權重
    """
    from collections import Counter
    import numpy as np
    counter = Counter(y)
    n_classes = len(set(y))
    total = len(y)
    weights = []
    for i in range(n_classes):
        if counter.get(i, 0) == 0:
            weights.append(1.0)
        else:
            weights.append(total / (n_classes * counter[i]))
    return weights


def print_and_log_overall_mean(cv_scores_dict, target_names, logf):
    """
    計算所有任務的有效fold平均分數，再計算四任務平均，並print與log。

    Args:
        cv_scores_dict: dict，每個target對應list of scores
        target_names: list，任務名稱（如TARGETS.keys()）
        logf: 檔案物件，log檔案
    """
    means = []
    for target in target_names:
        scores = cv_scores_dict[target]
        valid_scores = [s for s in scores if not np.isnan(s)]
        if valid_scores:
            mean_score = np.nanmean(scores)
            means.append(mean_score)
    if means:
        avg_score = np.mean(means)
        print(f"四任務平均分數: {avg_score:.4f}")
        logf.write(f"四任務平均分數: {avg_score:.4f}\n")
    else:
        print("四任務皆無有效分數")
        logf.write("四任務皆無有效分數\n")