# AI CUP 2025春季賽－桌球智慧球拍資料的精準分析競賽

[AI CUP 2025春季賽－桌球智慧球拍資料的精準分析競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/39)
排名: 8/633

## 競賽目標

本次 AI CUP 競賽目標為：

- 針對桌球選手智慧球拍所蒐集的85Hz取樣感測資料Ax, Ay, Az, Gx, Gy, Gz(加速度、角速度)，預測四個目標變數：
    1. gender（性別）二分類
    2. hold racket handed（持拍手別）二分類
    3. play years（打球年資）三分類 OvR
    4. level（球技等級）四分類 OvR

- 評分方式為四任務平均 AUC 分數。

## 專案流程與程式功能

### 執行流程

請依序執行以下程式：

1. `data_processing.py`
2. `training.py`
3. `final_training.py`
4. `testing.py`

### 程式功能

**程式**
- `data_processing.py`: 處理原始資料，包括特徵工程等。
- `model.py`: 使用 CatBoost 或 TabPFN 做為主要模型，並針對每個任務分別建模。
- `feature_selection.py`: 
    - 單折特徵選擇，使用 CatBoost 計算特徵重要性並選擇前 k 特徵
    - 聚合多折特徵重要性，選擇前 k 特徵
- `training.py`: 
  - 使用 **StratifiedGroupKFold** 交叉驗證，確保同一選手（player_id）不跨 fold，避免資料洩漏。
  - 針對每個任務分別使用 BorderlineSMOTE 訓練模型、每折使用`feature_selection.py` 的 **CatBoost重要度**後再用 **Permutation Importance** 進行兩階段篩選特徵並紀錄每折特徵重要度。
  - 紀錄全局特徵選擇重要度。
- `final_training.py`: 使用整個訓練集，套用**全局特徵選擇**與 BorderlineSMOTE，訓練最終模型並儲存。
- `testing.py`: 使用最新模型資料夾，對測試集進行預測並輸出 submission.csv。
- `utils.py`: 包含一些工具函數，如計算特徵重要度、打印平均分數等。
- `config.py`: 包含一些配置參數，如訓練參數、模型參數等。

**檔案**
- `selected_features.json`: 每個 target 最佳AUC fold 用CatBoost特徵選擇的特徵
- `global_selected_features.json`: 每個 target 聚合全部 fold 用 CatBoost 特徵選擇重要度的前 i 特徵
- `global_permutation_selected_features.json`: 每個 target 聚合全部 fold 用排列重要性的前 j 特徵

**補充:** 每fold的CatBoost特徵選擇跟Permutation Importance之間有兩階段關係 e.g. 先選1000個特徵，再選500個特徵

## 補充

- 因為每fold都執行特徵選擇，所以訓練流程時間約230分鐘

### CV實驗結果 
| target | 平均AUC | 有效fold數 |
| --- | --- | --- |
| gender | 0.9180 | 5 |
| hold racket handed | 0.9973 | 5 |
| play years | 0.7039 | 5 |
| level | 0.8112 | 2 |

四任務平均分數: 0.8576

## 軟體環境

- python 3.10.16
- catboost 1.2.7
- tabpfn 2.0.8 
- tabpfn-extensions 0.0.4 
- pandas 2.2.3
- numpy 1.26.4
- imbalanced-learn 0.13.0 
- scikit-learn 1.6.1 
- scipy 1.15.2

## 硬體環境與作業系統

- Windows11家用版10.0.22631
- CPU: 11th Gen Intel(R) Core(TM) i7-11700 @ 2.50GHz
- RAM: 32GB
- GPU: NVIDIA GeForce RTX 3060 12GB
