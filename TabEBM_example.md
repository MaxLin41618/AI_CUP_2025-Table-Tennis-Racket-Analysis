```

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import openml
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from tabebm.TabEBM import TabEBM, seed_everything


seed_everything(42)


# ===== Load the dataset form OpenML =====
dataset = openml.datasets.get_dataset(14)
print(f"Dataset {dataset.name} with id {dataset.dataset_id}")
print(f"Target feature: {dataset.default_target_attribute}")

# ===== Subsample a small subset of the data to simulate low-sample-size scenario =====
data_df = dataset.get_data()[0]
X_df = data_df.drop(columns=[dataset.default_target_attribute])
y_df = data_df[dataset.default_target_attribute]
X_df = X_df.sample(n=200)
y_df = y_df[X_df.index]

print(f"Subsampled dataset size: {len(X_df)}")
print(f"Number of features: {len(X_df.columns)}")


# ===== Split the data into training and testing sets =====
X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.5)
print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ===== Normalise the data =====
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===== Encode the target variable =====
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# ===== Fit TabEBM and fenerate synthetic samples =====
tabebm = TabEBM()
# === Generate 50 synthetic samples per class ===
data_syn = tabebm.generate(X_train, y_train, num_samples=50)

# ===== Combine the synthetic samples with the real samples =====
X_syn = np.concatenate(list(data_syn.values()))
y_syn = np.concatenate([np.full(len(data_syn[f"class_{i}"]), i) for i in range(len(data_syn.keys()))])

X_train_augmented = np.concatenate([X_train, X_syn])
y_train_augmented = np.concatenate([y_train, y_syn])

model_vanilla = KNeighborsClassifier()
model_vanilla.fit(X_train, y_train)

model_augmented = KNeighborsClassifier()
model_augmented.fit(X_train_augmented, y_train_augmented)

from sklearn.metrics import balanced_accuracy_score

y_pred_vanilla = model_vanilla.predict(X_test)
y_pred_augmented = model_augmented.predict(X_test)
acc_vanilla = balanced_accuracy_score(y_test, y_pred_vanilla) * 100
acc_augmented = balanced_accuracy_score(y_test, y_pred_augmented) * 100

print(f"Vanilla model's balanced accuracy: {acc_vanilla:.2f}")
print(f"Augmented model's balanced accuracy: {acc_augmented:.2f}")
```

```
Tutorial 1: Generating data with TabEBM
%load_ext autoreload
%autoreload 2
%matplotlib inline
!pip install tabebm
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from tabebm.TabEBM import TabEBM
def create_two_blobs_at_distance(num_samples=200, blob1_num_samples=None, blob2_num_samples=None, distance=1.0, random_state=42):
	"""
	Create two Gaussian blobs at distance D from the center
	"""
	if num_samples!=None:
		np.random.seed(random_state)
		X1 = np.random.randn(num_samples//2, 2) # class 1
		X2 = np.random.randn(num_samples//2, 2) # class 2
	else:
		X1 = np.random.randn(blob1_num_samples, 2) # class 1
		X2 = np.random.randn(blob2_num_samples, 2) # class 2

	X1[:,0] -= distance/2
	X2[:,0] += distance/2

	X = np.concatenate([X1, X2], axis=0)
	y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))], axis=0)

	return X, y
Data Generation using TabEBM
X, y = create_two_blobs_at_distance(num_samples=None, blob1_num_samples=150, blob2_num_samples=150, distance=2, random_state=40)
# ==== scatter the points ====
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Real data class 0', alpha=0.2, s=10)
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Real data class 1', alpha=0.2, s=10)
ax.set_title('Real data')
ax.legend()
plt.show()

tabebm = TabEBM()
augmented_data = tabebm.generate(
    X, y, num_samples = 50,
    sgld_steps = 200
)
# ==== Scatter the real points ====
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='red', label='Real data class 0', alpha=0.2, s=10)
ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='blue', label='Real data class 1', alpha=0.2, s=10)

# ==== Scatter the generated points ====
ax.scatter(augmented_data['class_0'][:, 0], augmented_data['class_0'][:, 1], c='red', s=15, marker='x', label='Synthetic data class 0')
ax.scatter(augmented_data['class_1'][:, 0], augmented_data['class_1'][:, 1], c='blue', s=15, marker='x', label='Synthetic data class 1')

ax.set_title('Real and synthetic data')
ax.legend()
plt.show()
```
