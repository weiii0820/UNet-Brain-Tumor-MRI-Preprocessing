# 🧠 不同影像預處理方式在U-Net卷積神經網路的重疊程度分析-以腦腫瘤影像為例
*以不同影像預處理方式比較 U-Net 在腦腫瘤分割任務的效能*

## 📌 專案簡介
本研究探討 **不同影像預處理方式** 對 **U-Net 卷積神經網路** 在腦腫瘤 MRI 分割中的影響。  
研究比較了 **Gaussian 濾波**、**直方圖均衡化 (HE)** 與 **CLAHE** 三種常見的影像處理方法，並透過腦腫瘤影像資料進行實驗。  

研究重點：  
- 🔹 比較三種影像預處理方式的優缺點  
- 🔹 評估不同方法對模型訓練與腫瘤邊界偵測的影響  
- 🔹 分析 **Accuracy、Dice、Mean IoU** 三項分割效能指標  

結果顯示，**HE 在 Dice 與 Mean IoU 上表現最佳**，能有效提升腫瘤邊界分割的精準度。  

---

## 📊 資料集
本研究使用多來源腦腫瘤 MRI 影像，約 **4000 張影像**，腫瘤與非腫瘤各半。  
- **Kaggle Brain Tumor MRI Dataset**  
- **Figshare Brain Tumor Dataset**  
- **台北榮總臨床 MRI 影像**  

標記方式：使用 **LabelMe** 軟體人工標註腫瘤遮罩（mask）。  
資料劃分：**85% 訓練集，15% 驗證集**。  

---

## 🏗️ 研究方法
- **模型**：U-Net (encoder–decoder 架構 + skip connections)  
- **輸入**：256×256 單通道灰階影像  
- **預處理**：  
  - Gaussian 濾波（降噪，邊界平滑）  
  - Histogram Equalization（提升整體對比度）  
  - CLAHE（局部對比增強）  
- **數據增強**：隨機翻轉、旋轉、亮度與對比度偏移  
- **訓練**：  
  - Optimizer：Adam (lr=1e-4)  
  - Loss：Binary Cross Entropy + Tversky Loss  
  - Batch size：4，Epochs：50  
  - Early Stopping + ReduceLROnPlateau  
  - AMP 自動混合精度訓練  

---

## 📈 實驗結果
### 1. 準確率 (Accuracy)  
三種方法皆達到 ~0.986，差異不大。  

### 2. Dice coefficient  
- HE：**0.1975**  
- CLAHE：0.1844  
- Gaussian：0.1835  

### 3. Mean IoU  
- HE：**0.5606**  
- CLAHE：0.5572  
- Gaussian：0.5551  

👉 整體而言，**HE 在重疊度相關指標中表現最佳**，支持研究假設。  

---

## 📂 專案結構
```
├── Report/ # 專題報告 PDF
├── code/ # 模型程式碼
│ ├── build.py
│ ├── data_process.py
│ ├── metrics.py
│ └── train.py
└── README.md
```

---

## 📥 資料及來源
- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- [Figshare Brain Tumor Dataset](https://doi.org/10.6084/m9.figshare.1512427.v5)  
- 臺北榮總臨床 MRI 影像
