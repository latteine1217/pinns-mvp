# Google Colab A100 訓練指南

## 📋 目標
在 Google Colab A100 GPU 上執行 2000 epoch 標準化訓練（預計 2-4 小時）

---

## 🚀 快速開始

### 1. **上傳專案到 Colab**

```bash
# 方法 A：使用 Google Drive
# 1. 在本地打包專案（排除大文件）
tar -czf pinns-mvp.tar.gz \
  --exclude='checkpoints' \
  --exclude='results' \
  --exclude='log' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  .

# 2. 上傳 pinns-mvp.tar.gz 到 Google Drive
# 3. 在 Colab 中掛載 Drive 並解壓
```

```python
# Colab Notebook 中執行
from google.colab import drive
drive.mount('/content/drive')

# 解壓專案
!cd /content && tar -xzf /content/drive/MyDrive/pinns-mvp.tar.gz
%cd /content/pinns-mvp
```

---

### 2. **環境檢查**

```python
# Colab Notebook 執行
!python scripts/colab_setup_check.py
```

**預期輸出**：
```
✅ CUDA 可用
   CUDA 版本: 12.2
   GPU 數量: 1
   GPU 0: NVIDIA A100-SXM4-40GB
      記憶體: 40.00 GB
      計算能力: 8.0

✅ 所有檢查通過！可以開始訓練。
```

---

### 3. **安裝依賴（如需要）**

```python
# Colab 通常已預裝 PyTorch，若版本不符：
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝其他依賴
!pip install pyyaml h5py scipy matplotlib
```

---

### 4. **啟動訓練**

#### 方法 A：前台運行（適合短期訓練/調試）
```python
!python scripts/train.py --cfg configs/test_normalization_main_2000epochs.yml
```

#### 方法 B：背景運行（推薦，適合長期訓練）
```python
# 啟動背景訓練
!nohup python scripts/train.py \
  --cfg configs/test_normalization_main_2000epochs.yml \
  > log/normalization_main_2000epochs_stdout.log 2>&1 &

# 查看進程
!ps aux | grep train.py

# 實時監控日誌
!tail -f log/normalization_main_2000epochs/training.log
```

---

### 5. **監控訓練進度**

#### 檢查訓練日誌
```python
# 查看最近 50 行日誌
!tail -n 50 log/normalization_main_2000epochs/training.log
```

#### 檢查檢查點
```python
!ls -lh checkpoints/normalization_main_2000epochs/
```

#### TensorBoard 可視化
```python
%load_ext tensorboard
%tensorboard --logdir log/normalization_main_2000epochs/tensorboard
```

---

### 6. **下載結果**

```python
# 壓縮檢查點
!tar -czf checkpoints_main_2000epochs.tar.gz \
  checkpoints/normalization_main_2000epochs

# 壓縮日誌
!tar -czf logs_main_2000epochs.tar.gz \
  log/normalization_main_2000epochs

# 下載到本地
from google.colab import files
files.download('checkpoints_main_2000epochs.tar.gz')
files.download('logs_main_2000epochs.tar.gz')
```

---

## 📊 訓練配置詳情

### **關鍵參數**
- **配置文件**：`configs/test_normalization_main_2000epochs.yml`
- **Epochs**：2000
- **感測點**：K=50（QR-Pivot）
- **優化器**：Adam（lr=1e-3）
- **學習率調度**：CosineAnnealingWarmRestarts（T_0=200）
- **檢查點頻率**：每 200 epochs
- **驗證頻率**：每 100 epochs
- **早停**：啟用（patience=500）

### **Fourier 退火階段**
| 階段 | Epochs | 頻率 | 描述 |
|------|--------|------|------|
| 1 | 0-600 | [1, 2] | 低頻預熱 |
| 2 | 600-1200 | [1, 2, 4] | 中頻解鎖 |
| 3 | 1200-2000 | [1, 2, 4, 8] | 全頻段 |

### **預期輸出**
- **檢查點**：`checkpoints/normalization_main_2000epochs/epoch_{200,400,...,2000}.pth`
- **最佳模型**：`checkpoints/normalization_main_2000epochs/best_model.pth`
- **訓練日誌**：`log/normalization_main_2000epochs/training.log`

---

## ⚠️ 常見問題

### **Q1: 訓練中斷怎麼辦？**
```python
# 從最新檢查點恢復
!python scripts/train.py \
  --cfg configs/test_normalization_main_2000epochs.yml \
  --resume checkpoints/normalization_main_2000epochs/latest.pth
```

### **Q2: Colab 連接超時？**
- 使用背景運行（nohup）可以減少依賴活躍連接
- 設置定期保存檢查點（已配置為每 200 epochs）
- 考慮使用 Colab Pro（更長運行時間）

### **Q3: 記憶體不足（OOM）？**
1. 降低批次大小：修改 `training.batch_size`（目前 10000）
2. 減少 PDE 採樣點：修改 `training.sampling.pde_points`（目前 20000）
3. 啟用梯度檢查點：設置 `model.use_gradient_checkpointing: true`

### **Q4: 如何確認感測點文件存在？**
```python
!ls -lh data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz

# 若缺失，需從本地上傳
from google.colab import files
uploaded = files.upload()  # 上傳 sensors_K50_qr_pivot.npz
!mkdir -p data/jhtdb/channel_flow_re1000
!mv sensors_K50_qr_pivot.npz data/jhtdb/channel_flow_re1000/
```

---

## 📈 訓練後評估

### **載入檢查點並評估**
```python
# 評估最佳模型
!python scripts/evaluate.py \
  --cfg configs/test_normalization_main_2000epochs.yml \
  --checkpoint checkpoints/normalization_main_2000epochs/best_model.pth \
  --output results/normalization_main_2000epochs

# 生成可視化
!python scripts/visualize_results.py \
  --results results/normalization_main_2000epochs \
  --output results/normalization_main_2000epochs/visualizations
```

---

## 🔑 Colab Pro 優勢

| 功能 | Colab 免費版 | Colab Pro |
|------|--------------|-----------|
| GPU | T4/V100 | A100/V100 (優先) |
| 運行時間 | ~12 小時 | ~24 小時 |
| 記憶體 | 標準 | 高記憶體可選 |
| 價格 | 免費 | ~$10/月 |

**建議**：本訓練（2-4 小時）在免費版即可完成，但 Pro 版可提供更穩定的 A100 訪問。

---

## 📝 檢查清單

上傳前準備：
- [ ] 確認 `configs/test_normalization_main_2000epochs.yml` 存在
- [ ] 確認 `data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz` 存在
- [ ] 打包專案（排除大文件）
- [ ] 上傳到 Google Drive

Colab 執行：
- [ ] 掛載 Google Drive
- [ ] 解壓專案
- [ ] 運行 `colab_setup_check.py`（通過所有檢查）
- [ ] 啟動訓練（背景運行）
- [ ] 定期監控日誌
- [ ] 下載檢查點和日誌

---

## 🔗 相關文檔

- [配置文件說明](../configs/README.md)
- [訓練腳本使用](../scripts/README.md)
- [QR-Pivot 感測點視覺化指南](QR_SENSOR_VISUALIZATION_GUIDE.md)
- [PirateNet 訓練失敗診斷](PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md)

---

**最後更新**：2025-10-17  
**狀態**：✅ 已驗證（本地初始化成功，等待 Colab 完整訓練）
