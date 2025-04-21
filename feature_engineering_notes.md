# Feature Engineering Notes

## 建議新增特徵

### 一、時域進階特徵

- Mean Absolute Change（平均絕對變化量）
- Mean Change Rate（平均變化率）
- Zero Crossing Rate（零交叉率）
- Peak 特徵（最大/最小值出現位置、peak 數量、peak 間距均值/方差）
- Slope（斜率均值/方差）
- Autocorrelation（自相關最大值/延遲）

### 二、頻域進階特徵

- Dominant Frequency Amplitude（主頻率幅值）
- Band Power（頻帶能量分布，如 0-1Hz, 1-3Hz, 3-5Hz）
- Number of Peaks in Spectrum（頻譜峰值數量）

### 三、統合特徵

- 各軸之間的相關係數（Corr(Ax, Ay), Corr(Ax, Az), ...）
- 角度特徵（Pitch, Roll, Yaw，若有原始資料可計算）
- AccVec, GyroVec 的比值（如 AccVec_mean / GyroVec_mean）

### 四、其他進階特徵

- 滑動視窗特徵（如每個視窗的均值/方差/最大最小值等）
- FFT 前幾個主成分（PCA on FFT）

### 五、非線性特徵

- Sample Entropy
- Approximate Entropy
- Fractal Dimension
- Hjorth Parameters（Activity, Mobility, Complexity）

---

## 注意事項

- 所有特徵選擇（Feature Selection）必須在訓練集（training set）上進行，嚴禁在測試集或驗證集上做特徵選擇，避免資料洩漏（Data Leakage）。
- 特徵選擇建議流程：
  1. 先用所有特徵訓練模型，取得基準分數。
  2. 根據模型特徵重要性（如 LGBM、RF、SHAP）或統計方法，初步篩選掉貢獻極低的特徵。
  3. 用交叉驗證測試不同特徵組合對模型表現的影響。
- 若有領域知識，請優先考慮與任務高度相關的特徵。

---

> 本文件用於追蹤特徵工程進度與團隊溝通，若有新想法請隨時補充。
