# Feature Selection Example

使用代理模型進行一次性選擇 (在所有訓練數據上，於 CV 外部)：

在進行 CV 之前，將你的所有數據（或者至少是訓練集和驗證集合起來的部分，如果留了獨立測試集的話）用於特徵選擇。

在全部訓練數據上訓練一次，獲取特徵重要性。

根據這個重要性排序，選擇你最終想要保留的 K 個特徵。

然後，在後續的 5-fold CV 過程中，以及最終在整個訓練集上訓練你的主要比賽模型時，始終使用這同一個固定的 K 個特徵子集。

這種方法只執行了一次特徵選擇，計算成本較低，且只要選擇模型與主要模型差異夠大，或是在足夠大的訓練數據上進行，數據洩漏風險相對可控。

```python
"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
Feature selection involves training multiple TabPFN models, which is computationally intensive.
"""

from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, interpretability

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Initialize model
clf = TabPFNClassifier(n_estimators=3)

# Feature selection
sfs = interpretability.feature_selection.feature_selection(
    estimator=clf,
    X=X,
    y=y,
    n_features_to_select=5,  # How many features to select
    feature_names=feature_names,
)

# Print selected features
selected_features = [
    feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]
]
print("\nSelected features:")
for feature in selected_features:
    print(f"- {feature}")
```
