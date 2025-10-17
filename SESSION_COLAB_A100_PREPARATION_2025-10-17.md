# 會話記錄：Google Colab A100 訓練準備
**日期**: 2025-10-17  
**狀態**: ✅ 已完成，等待 Colab 部署  

---

## 📋 任務目標
準備將 2000 epoch 標準化訓練配置遷移至 Google Colab A100 GPU 環境

---

## ✅ 已完成工作

### 1. **核心修正**（上一會話）
- ✅ `pinnx/train/factory.py` (L815-816)：修正梯度檢查點配置讀取
- ✅ `pinnx/dataio/channel_flow_loader.py` (L180-195)：修正感測點文件路徑解析
- ✅ `scripts/train.py` (L1176)：修正日誌配置讀取（向後兼容）

### 2. **配置創建**（上一會話）
- ✅ `configs/test_normalization_main_2000epochs.yml`：完整 2000 epoch 訓練配置
  - K=50 QR-Pivot 感測點
  - Adam 優化器（lr=1e-3）
  - CosineAnnealingWarmRestarts 學習率調度
  - 3 階段 Fourier 退火（[1,2] → [1,2,4] → [1,2,4,8]）
  - 早停機制（patience=500）

### 3. **Colab 環境準備工具**（本次會話）
- ✅ **`scripts/colab_setup_check.py`**：完整環境檢查腳本
  - CUDA 可用性檢查
  - GPU 記憶體檢查
  - 專案結構驗證
  - 配置文件驗證
  - 依賴套件檢查
  - 訓練資源需求估算
  - **修正**：Line 23 類型檢查錯誤（`torch.version.cuda`）

- ✅ **`scripts/colab_quick_start.sh`**：一鍵啟動腳本（可執行）
  - 自動環境檢查
  - 創建必要目錄
  - 驗證關鍵文件
  - 啟動背景訓練
  - 顯示監控指令

- ✅ **`docs/COLAB_A100_TRAINING_GUIDE.md`**：完整 Colab 訓練指南
  - 快速開始步驟
  - 環境檢查流程
  - 訓練啟動與監控
  - 結果下載方法
  - 常見問題解答
  - 檢查清單

- ✅ **`Colab_2000_Epoch_Training.ipynb`**：Colab Notebook 範例
  - 7 個步驟的完整訓練流程
  - 互動式監控與可視化
  - TensorBoard 整合
  - 結果下載與管理

---

## 📁 新增/修改的檔案

### 本次會話新增
1. **`scripts/colab_setup_check.py`** (215 行) - 環境檢查工具
2. **`scripts/colab_quick_start.sh`** (可執行) - 一鍵啟動腳本
3. **`docs/COLAB_A100_TRAINING_GUIDE.md`** - 完整 Colab 訓練指南
4. **`Colab_2000_Epoch_Training.ipynb`** - Colab Notebook 範例

### 上次會話修改
5. **`configs/test_normalization_main_2000epochs.yml`** (382 行) - 訓練配置
6. **`pinnx/train/factory.py`** (L815-816) - 梯度檢查點修正
7. **`pinnx/dataio/channel_flow_loader.py`** (L180-195) - 路徑解析修正
8. **`scripts/train.py`** (L1176) - 日誌配置修正

---

## 🚀 Colab 部署流程

### **步驟 1：本地準備**
```bash
# 打包專案（排除大文件）
tar -czf pinns-mvp.tar.gz \
  --exclude='checkpoints' \
  --exclude='results' \
  --exclude='log' \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  .

# 上傳到 Google Drive
# （手動或使用 Drive 桌面同步）
```

### **步驟 2：Colab 設置**
```python
# 在 Colab Notebook 中執行
from google.colab import drive
drive.mount('/content/drive')

# 解壓專案
!cd /content && tar -xzf /content/drive/MyDrive/pinns-mvp.tar.gz
%cd /content/pinns-mvp

# 環境檢查
!python scripts/colab_setup_check.py
```

### **步驟 3：啟動訓練**
```bash
# 方法 A：使用 Notebook 中的 cells（推薦）
# 參考 Colab_2000_Epoch_Training.ipynb

# 方法 B：使用一鍵啟動腳本
bash scripts/colab_quick_start.sh
```

### **步驟 4：監控與下載**
```bash
# 監控日誌
tail -f log/normalization_main_2000epochs/training.log

# 檢查點
ls -lh checkpoints/normalization_main_2000epochs/

# 下載結果
tar -czf results.tar.gz checkpoints/ log/
# 使用 files.download() 或複製回 Drive
```

---

## 🔑 關鍵配置參數

### **訓練設定**
- **Epochs**: 2000（預計 2-4 小時）
- **批次大小**: 10,000
- **PDE 採樣點**: 20,000
- **邊界採樣點**: 5,000
- **驗證頻率**: 每 100 epochs
- **檢查點頻率**: 每 200 epochs

### **Fourier 退火階段**
| 階段 | Epochs | 頻率 | 進度 |
|------|--------|------|------|
| 1 | 0-600 | [1, 2] | 0-30% |
| 2 | 600-1200 | [1, 2, 4] | 30-60% |
| 3 | 1200-2000 | [1, 2, 4, 8] | 60-100% |

### **學習率調度**
- **類型**: CosineAnnealingWarmRestarts
- **初始學習率**: 1e-3
- **最小學習率**: 1e-6
- **重啟週期**: T_0=200, T_mult=2

---

## 📊 預期輸出

### **檢查點**
```
checkpoints/normalization_main_2000epochs/
├── epoch_200.pth
├── epoch_400.pth
├── ...
├── epoch_2000.pth
├── best_model.pth
└── latest.pth
```

### **日誌**
```
log/normalization_main_2000epochs/
├── training.log          # 主訓練日誌
├── tensorboard/          # TensorBoard 日誌
└── metrics.json          # 指標記錄
```

### **標準輸出**
```
log/normalization_main_2000epochs_stdout.log  # nohup 輸出
```

---

## ⚠️ 注意事項

### **必須上傳的文件**
1. ✅ `configs/test_normalization_main_2000epochs.yml`
2. ✅ `data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz`
3. ✅ 所有 `pinnx/` 模組
4. ✅ 所有 `scripts/` 腳本

### **可排除的目錄**
- ❌ `checkpoints/` (會自動創建)
- ❌ `results/` (會自動創建)
- ❌ `log/` (會自動創建)
- ❌ `.git/` (不需要)
- ❌ `__pycache__/` (運行時生成)

### **Colab 限制**
- 免費版：~12 小時運行時間（本訓練 2-4 小時，足夠）
- GPU 分配：不保證 A100（可能是 T4/V100）
- 斷線風險：使用背景運行（nohup）+ 定期檢查點

---

## 🔄 從檢查點恢復

```bash
# 如果訓練中斷
python scripts/train.py \
  --cfg configs/test_normalization_main_2000epochs.yml \
  --resume checkpoints/normalization_main_2000epochs/latest.pth
```

---

## 📈 後續評估

### **評估最佳模型**
```bash
python scripts/evaluate.py \
  --cfg configs/test_normalization_main_2000epochs.yml \
  --checkpoint checkpoints/normalization_main_2000epochs/best_model.pth \
  --output results/normalization_main_2000epochs
```

### **生成可視化**
```bash
python scripts/visualize_results.py \
  --results results/normalization_main_2000epochs \
  --output results/normalization_main_2000epochs/visualizations
```

---

## 🔗 相關文檔

- **完整指南**: [`docs/COLAB_A100_TRAINING_GUIDE.md`](docs/COLAB_A100_TRAINING_GUIDE.md)
- **Notebook 範例**: [`Colab_2000_Epoch_Training.ipynb`](Colab_2000_Epoch_Training.ipynb)
- **配置說明**: [`configs/README.md`](configs/README.md)
- **訓練腳本文檔**: [`scripts/README.md`](scripts/README.md)

---

## 🎯 下一步行動

### **立即任務**
1. ⏳ 打包專案並上傳到 Google Drive
2. ⏳ 在 Colab 中執行環境檢查
3. ⏳ 啟動 2000 epoch 訓練
4. ⏳ 定期監控訓練進度（每 1-2 小時）

### **訓練完成後**
5. ⏳ 下載檢查點和日誌
6. ⏳ 評估最佳模型性能
7. ⏳ 生成可視化與分析報告
8. ⏳ 與基線配置對比（`main.yml` 短期訓練）

---

**準備狀態**: ✅ 所有工具和文檔已就緒  
**等待操作**: 🚀 上傳專案至 Google Drive 並啟動 Colab 訓練  
**預計完成**: 2-4 小時後（取決於 GPU 分配）
