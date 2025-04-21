# 詳細工作流程 (在 GroupKFold 框架內)

整個數據處理和模型訓練過程嚴格在 GroupKFold 的每個 Fold 內部執行，以確保驗證的公正性。

**GroupKFold 分割 (在整個原始感測器數據上執行一次)**

*   使用 GroupKFold (n_splits=5) 將**原始的感測器數據**按 `player_id` 分割成 5 個 Fold。這一步只產生數據索引，不修改或處理數據。

**對於 GroupKFold 的每一個 Fold (共 5 次迭代)：**

1.  **獲取當前 Fold 的原始訓練數據：** 讀取屬於該 Fold 訓練集索引 (`train_index`) 的所有原始感測器數據。
2.  **數據增強 (Jittering) + 任務專用平衡：**
    *   針對**當前 Fold 的原始訓練數據**，應用 Jittering (`std_ratio=0.01`)。
    *   根據**當前 Fold 訓練數據中**每個任務 (`gender`, `handed`, `play years`, `level`) 的類別分佈，計算需要增強的樣本數，以達到設定的平衡目標（例如，讓少數類別樣本數與多數類別相同）。
    *   生成相應數量的增強感測器數據。
    *   將該 Fold 的**原始訓練感測器數據**與生成的**增強感測器數據**合併，形成該 Fold 專用的、針對各任務平衡後的**增強感測器訓練數據集**。
3.  **特徵工程 (訓練數據)：**
    *   對該 Fold 專用的**增強感測器訓練數據集**進行特徵工程（統計量、頻域特徵等）。
    *   生成該 Fold 專用的**表格訓練數據集** (`X_train_fold_augmented`, `y_train_fold_augmented`)。
4.  **特徵選擇 (在增強訓練數據上，為每個任務獨立執行)：**
    *   對於每個任務 (`gender`, `handed`, `play years`, `level`)：
        *   從 `X_train_fold_augmented` 和 `y_train_fold_augmented` 中提取對應任務的數據。
        *   在該數據上訓練一個 LightGBM 模型，設置 `objective` 和 `eval_metric` (AUC 相關)，並使用早停。
        *   **早停驗證集：** LightGBM 早停所需的驗證集從 `X_train_fold_augmented` 內部使用 `StratifiedShuffleSplit` 切分約 10-20% 的數據獲得。
        *   獲取 LightGBM 模型的特徵重要性 (`importance_type='gain'`)。
        *   根據重要性選擇一個特徵子集 (`selected_features_task_fold`)。
5.  **訓練最終模型 (為每個任務獨立訓練)：**
    *   對於每個任務 (`gender`, `handed`, `play years`, `level`)：
        *   使用該任務選定的特徵子集 (`selected_features_task_fold`) 篩選 `X_train_fold_augmented`。
        *   使用篩選後的數據和 `y_train_fold_augmented` 中對應任務的標籤，訓練該任務的 CatBoostClassifier 模型。
        *   CatBoost 參數設置：`loss_function` (Logloss 或 MultiClass)，`eval_metric` (AUC 或 MultiAUC)，並使用早停。
6.  **獲取當前 Fold 的原始驗證數據：** 讀取屬於該 Fold 驗證集索引 (`val_index`) 的所有原始感測器數據。
7.  **特徵工程 (驗證數據)：**
    *   對該 Fold 的**原始驗證感測器數據**進行**相同的**特徵工程。
    *   生成該 Fold 專用的**表格驗證數據集** (`X_val_fold`, `y_val_fold`)。**注意：驗證數據不進行增強。**
8.  **評估最終模型 (為每個任務獨立評估)：**
    *   對於每個任務 (`gender`, `handed`, `play years`, `level`)：
        *   使用該任務選定的特徵子集 (`selected_features_task_fold`) 篩選 `X_val_fold`。
        *   使用該任務訓練好的 CatBoost 模型，在篩選後的 `X_val_fold` 上進行預測。
        *   計算該 Fold 在該任務上的 AUC 分數。