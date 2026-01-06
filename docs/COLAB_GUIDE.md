# Google Colab 執行指南

## 📚 快速開始

### 方法 1: 使用 Jupyter Notebook（推薦）

1. **上傳 Notebook 到 Colab**
   - 打開 [Google Colab](https://colab.research.google.com/)
   - 選擇 `File` → `Upload notebook`
   - 上傳 `notebooks/S2_K_Scan_Colab.ipynb`

2. **設置 GPU Runtime**
   - `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`
   - 推薦選擇 `T4` 或更高

3. **依序執行每個 Cell**
   - 按 `Shift + Enter` 執行當前 cell
   - 或 `Runtime` → `Run all` 執行全部

### 方法 2: 使用 Bash 腳本

1. **在 Colab Notebook 中執行**

```python
# Cell 1: Clone 專案
!git clone https://github.com/YOUR_USERNAME/pinns-sparse-flow.git /content/pinns-sparse-flow
%cd /content/pinns-sparse-flow

# Cell 2: 安裝依賴
!pip install -q -e .

# Cell 3: 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 4: 執行 K-scan
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

2. **查看結果**

```python
# Cell 5: 顯示結果
from IPython.display import Image, display
display(Image(filename='results/S2_k_scan_comparison.png'))
```

---

## 🎯 執行選項

### 快速測試（推薦新手，2-4h）

```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

### 完整實驗（論文用，8-12h）

```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100
```

### 包含高風險 K=200（實驗性，+10-20h）

```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100 200
```

### 僅測試單一 K 值

```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 50
```

---

## ⚙️ 自訂 Epochs

```bash
# 設置環境變數控制 epochs
!export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

或直接使用 Python 訓練腳本：

```bash
!python scripts/train/train.py \
  --cfg configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml \
  --epochs 5000
```

---

## 📊 監控訓練

### 即時監控日誌

```bash
# 在新的 cell 中執行（訓練時）
!tail -f logs/s2_k_scan_colab_*.log
```

### 檢查 GPU 使用率

```python
!nvidia-smi -l 5  # 每 5 秒刷新
```

### 檢查 Checkpoint

```python
import os
import glob

checkpoints = glob.glob('checkpoints/experiments/S2_qr_K*/best_model.pth')
for ckpt in checkpoints:
    size = os.path.getsize(ckpt) / (1024**2)
    print(f"{ckpt}: {size:.1f} MB")
```

---

## 💾 備份策略

### 自動備份（腳本內建）

- 每個實驗完成後自動備份至 `/content/drive/MyDrive/pinns_checkpoints/`

### 手動備份

```bash
# 備份 checkpoints
!cp -r checkpoints/experiments /content/drive/MyDrive/pinns_checkpoints/

# 備份結果
!cp -r results /content/drive/MyDrive/pinns_results/

# 備份日誌
!cp -r logs /content/drive/MyDrive/pinns_logs/
```

### 打包下載

```bash
# 壓縮所有結果
!zip -r s2_results.zip checkpoints/experiments results logs

# 從 Colab 下載
from google.colab import files
files.download('s2_results.zip')
```

---

## 🔧 常見問題

### 1. OOM (Out of Memory)

**症狀**: CUDA out of memory 錯誤

**解決方案**:
```python
# 減少 batch size（修改 config 或使用環境變數）
!python scripts/train/train.py \
  --cfg configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml \
  --batch-size 5000  # 預設 10000
```

### 2. 會話斷開

**症狀**: Colab 斷線，訓練中斷

**解決方案**:
- 使用 Colab Pro（24h 連線）
- 分批執行（先跑 K=30,50）
- 使用較少 epochs 快速測試

**恢復訓練**:
```bash
# 檢查 checkpoint
!ls checkpoints/experiments/S2_qr_K50/

# 從 checkpoint 繼續（如果腳本支援）
!python scripts/train/train.py \
  --cfg configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml \
  --resume checkpoints/experiments/S2_qr_K50/latest_model.pth
```

### 3. 數據缺失

**症狀**: FileNotFoundError: dns_re50_t100.h5

**解決方案**:
```bash
# 從 Google Drive 複製數據
!cp /content/drive/MyDrive/pinns_data/dns_re50_t100.h5 \
   data/kolmogorov_dns/

# 或從伺服器下載（如果有 URL）
!wget https://your-server.com/dns_re50_t100.h5 \
  -O data/kolmogorov_dns/dns_re50_t100.h5
```

### 4. GPU 未使用

**症狀**: 訓練非常慢，`nvidia-smi` 無輸出

**解決方案**:
```python
# 檢查 PyTorch GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# 如果顯示 False，重新設置 Runtime
# Runtime → Change runtime type → GPU
```

---

## 📈 效能優化

### 加速訓練

1. **使用 Colab Pro**
   - 更快的 GPU（A100 / V100）
   - 更長的連線時間（24h）
   - 更高的 RAM

2. **減少 Epochs**
   ```bash
   COLAB_EPOCHS=3000 bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
   ```

3. **啟用 AMP（混合精度）**
   - 修改 config: `training.amp.enabled: true`
   - 可加速 30-50%

4. **減少驗證頻率**
   - 修改 config: `training.validation_freq: 200` (預設 100)

### 節省空間

```bash
# 清理快取
!rm -rf /root/.cache/pip
!rm -rf data/kolmogorov_dns_cache/*

# 只保留 best_model.pth
!find checkpoints -name "epoch_*.pth" -delete
```

---

## 📝 腳本說明

### run_s2_k_scan_colab.sh

**功能**:
- 自動偵測 Colab 環境
- 依序訓練指定 K 值
- 自動備份至 Google Drive
- 生成評估腳本

**參數**:
```bash
bash scripts/experiments/run_s2_k_scan_colab.sh [K1] [K2] [K3] ...
```

**環境變數**:
- `COLAB_EPOCHS`: 訓練 epochs（預設 5000）
- `WANDB_MODE=disabled`: 禁用 W&B

**輸出**:
- Checkpoints: `checkpoints/experiments/S2_qr_K*/`
- 日誌: `logs/s2_k_scan_colab_*.log`
- 評估腳本: `logs/evaluate_s2_results.sh`

---

## 🎓 進階用法

### 並行訓練（多個 Colab Notebook）

如果有多個 Colab 帳號或 Colab Pro，可以並行訓練：

**Notebook 1**:
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30
```

**Notebook 2**:
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 50
```

**Notebook 3**:
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 80
```

### 使用自訂配置

```python
# 複製並修改配置
!cp configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml \
   configs/my_custom_config.yml

# 編輯配置（在 Colab 中）
# 使用文本編輯器或 sed

# 執行訓練
!python scripts/train/train.py --cfg configs/my_custom_config.yml
```

### 進階評估

```python
# 使用完整評估工具
!python scripts/evaluate/comprehensive_evaluation.py \
  --checkpoint checkpoints/experiments/S2_qr_K50/best_model.pth \
  --reference-dir data/kolmogorov_dns \
  --output results/comprehensive_K50
```

---

## 📞 取得協助

遇到問題？

1. **檢查日誌**: `logs/s2_k_scan_colab_*.log`
2. **查看文檔**: `docs/QUICK_START.md`
3. **GitHub Issues**: 提交問題到專案 Issues

---

## ✅ 檢查清單

在開始訓練前，確認：

- [ ] GPU Runtime 已啟用
- [ ] Google Drive 已掛載
- [ ] 專案已 clone 並更新
- [ ] 依賴已安裝
- [ ] DNS 數據已準備
- [ ] Sensor 文件已存在
- [ ] 備份目錄已創建

---

**祝訓練順利！** 🚀
