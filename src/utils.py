"""
工具函式集
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os, json, hashlib, pickle, csv
import config  # 用於讀取 RAW_TRAIN_DATA_DIR


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
    plt.figure(figsize=(100, max(5, top_n//2)))
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


def compute_feature_fingerprint(features):
    """
    計算當前特徵工程設定的 MD5 fingerprint

    Args:
        features (list): 特徵名稱清單
    Returns:
        str: fingerprint 字串
    """
    cfg = {
        "features": features
    }
    try:
        dp_path = os.path.join(os.path.dirname(__file__), 'data_processing.py')
        with open(dp_path, 'rb') as f:
            cfg['data_processing_hash'] = hashlib.md5(f.read()).hexdigest()
    except Exception:
        cfg['data_processing_hash'] = ''
    try:
        raw_dir = config.RAW_TRAIN_DATA_DIR
        files = sorted(os.listdir(raw_dir))
        raw_meta = [(f, os.stat(os.path.join(raw_dir, f)).st_mtime, os.stat(os.path.join(raw_dir, f)).st_size) for f in files]
        cfg['raw_data_meta'] = hashlib.md5(json.dumps(raw_meta, sort_keys=True).encode()).hexdigest()
    except Exception:
        cfg['raw_data_meta'] = ''
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode('utf-8')).hexdigest()


def save_feature_cache(cache_dir, fname, data):
    """
    將 data 存成 pickle 快取

    Args:
        cache_dir (str): 快取資料夾
        fname (str): 檔名
        data (dict): 要存的資料
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, fname)
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def load_feature_cache(cache_dir, fname):
    """
    從 pickle 快取載入 data，若不存在回傳 None

    Args:
        cache_dir (str): 快取資料夾
        fname (str): 檔名
    Returns:
        dict or None
    """
    path = os.path.join(cache_dir, fname)
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


def export_feature_importance_csv(model, feature_names, csv_path, top_n=None):
    """
    匯出特徵重要度到 CSV 檔

    Args:
        model: CatBoostClassifier 已訓練模型
        feature_names: list 特徵名稱
        csv_path: str CSV 儲存路徑
        top_n: int, optional 只輸出前 N 大特徵，若 None 則輸出全部
    """
    importance = model.get_feature_importance()
    if top_n is not None:
        indices = importance.argsort()[::-1][:top_n]
    else:
        indices = importance.argsort()[::-1]
    sorted_names = [feature_names[i] for i in indices]
    sorted_importance = importance[indices]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['feature', 'importance'])
        for name, imp in zip(sorted_names, sorted_importance):
            writer.writerow([name, imp])