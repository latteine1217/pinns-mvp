# Colab 快速使用指南

## 🚀 三步驟開始訓練

### Step 1: 掛載 Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2: 設置環境
```python
import os
os.chdir('/content/drive/MyDrive/pinns-sparse-flow')
%run scripts/setup_colab_env.py
```

### Step 3: 開始訓練
```bash
# 快速測試 K=30 和 K=50
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

---

## ⚙️ 自訂訓練參數

### 修改 Epochs

**方法 1: 使用環境變數（推薦）**
```bash
# 使用 3000 epochs 快速測試
!export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

**方法 2: 直接修改配置文件**
```python
# 編輯配置文件
config_file = 'configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml'

# 讀取配置
with open(config_file, 'r') as f:
    config = f.read()

# 修改 epochs
config = config.replace('epochs: 10000', 'epochs: 3000')

# 保存修改
with open(config_file, 'w') as f:
    f.write(config)

print(f"✓ 已修改 {config_file} 的 epochs 為 3000")
```

**方法 3: 單獨訓練**
```bash
# 直接使用訓練腳本（使用配置中的 epochs）
!python scripts/train/train.py --cfg configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml
```

---

## 📊 執行選項

### 快速測試（2-4 小時）
```bash
# 預設：K=30, 50；配置中的 epochs（通常 10000）
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50

# 或使用較少 epochs
!export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
```

### 完整實驗（8-12 小時）
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100
```

### 包含 K=200（+10-20 小時，高風險）
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100 200
```

### 單獨測試某個 K 值
```bash
!bash scripts/experiments/run_s2_k_scan_colab.sh 50
```

---

## 🔍 監控訓練

### 實時查看日誌
```python
# 新開一個 cell，在訓練時運行
!tail -f logs/s2_k_scan_colab_*.log
```

### 查看 GPU 狀態
```python
!watch -n 5 nvidia-smi  # 每 5 秒刷新
```

### 檢查訓練進度
```python
import glob
import os

# 列出所有 checkpoint
checkpoints = glob.glob('checkpoints/experiments/S2_qr_K*/best_model.pth')
for ckpt in sorted(checkpoints):
    if os.path.exists(ckpt):
        size = os.path.getsize(ckpt) / (1024**2)
        mtime = os.path.getmtime(ckpt)
        from datetime import datetime
        time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"✓ {ckpt}: {size:.1f} MB (最後更新: {time_str})")
```

---

## 💾 備份與下載

### 自動備份（內建）
腳本會自動將每個完成的實驗備份到：
```
/content/drive/MyDrive/pinns_checkpoints/S2_K{K}/
```

### 手動備份
```bash
# 備份 checkpoint
!cp -r checkpoints/experiments/S2_qr_K50 /content/drive/MyDrive/pinns_checkpoints/

# 備份結果
!cp -r results /content/drive/MyDrive/pinns_results/
```

### 打包下載
```python
# 壓縮結果
!zip -r s2_results.zip checkpoints/experiments/S2_qr_K* results logs

# 下載
from google.colab import files
files.download('s2_results.zip')
```

---

## 📈 評估結果

### 快速評估
```bash
!python scripts/evaluate_unified.py \
  --checkpoints \
    checkpoints/experiments/S2_qr_K30/best_model.pth \
    checkpoints/experiments/S2_qr_K50/best_model.pth \
  --labels "K=30" "K=50" \
  --output results/s2_comparison.png
```

### 查看結果圖表
```python
from IPython.display import Image, display
display(Image(filename='results/s2_comparison.png'))
```

---

## ⚠️ 常見問題

### 1. 訓練腳本不支援 --epochs 參數
**症狀**: `error: unrecognized arguments: --epochs`

**解決方案**: 
- 使用環境變數: `export COLAB_EPOCHS=3000`
- 或直接修改配置文件中的 `training.epochs`

### 2. ModuleNotFoundError: No module named 'pinnx'
**症狀**: 無法導入 pinnx 模組

**解決方案**:
```python
%run scripts/setup_colab_env.py
```
這會自動設置 PYTHONPATH 並安裝模組。

### 3. CUDA out of memory
**症狀**: GPU 記憶體不足

**解決方案**:
```python
# 清理 GPU 記憶體
import torch
torch.cuda.empty_cache()

# 或重啟 Runtime
# Runtime → Restart runtime
```

### 4. 訓練中斷
**症狀**: Colab 斷線，訓練停止

**解決方案**:
- 升級到 Colab Pro（24h 連線）
- 使用較少 epochs 快速測試
- 分批執行（先跑 K=30,50）

---

## 🎯 推薦配置

### 新手測試
```bash
# 3000 epochs，K=50 only
!export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 50
# 預估時間: 1-2 小時
```

### 快速驗證
```bash
# 5000 epochs，K=30 + K=50
!export COLAB_EPOCHS=5000 && bash scripts/experiments/run_s2_k_scan_colab.sh 30 50
# 預估時間: 3-5 小時
```

### 完整實驗
```bash
# 使用配置中的 epochs (10000)，K=30,50,80,100
!bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100
# 預估時間: 16-24 小時（Colab Pro 推薦）
```

---

## 📝 配置文件位置

所有 S2 K-scan 配置文件：
```
configs/experiments/S2_k_scan/
├── s2_qr_K30_2d_re50.yml   # epochs: 10000
├── s2_qr_K50_2d_re50.yml   # epochs: 10000
├── s2_qr_K80_2d_re50.yml   # epochs: 10000
├── s2_qr_K100_2d_re50.yml  # epochs: 10000
└── s2_qr_K200_2d_re50.yml  # epochs: 10000
```

---

## ✅ 完整工作流程範例

```python
# Cell 1: 環境設置
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/pinns-sparse-flow')
%run scripts/setup_colab_env.py

# Cell 2: 開始訓練（快速測試）
!export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 30 50

# Cell 3: 檢查結果
!ls -lh checkpoints/experiments/S2_qr_K*/best_model.pth

# Cell 4: 評估
!python scripts/evaluate_unified.py \
  --checkpoints \
    checkpoints/experiments/S2_qr_K30/best_model.pth \
    checkpoints/experiments/S2_qr_K50/best_model.pth \
  --labels "K=30" "K=50" \
  --output results/s2_comparison.png

# Cell 5: 顯示結果
from IPython.display import Image, display
display(Image(filename='results/s2_comparison.png'))

# Cell 6: 備份到 Drive（自動完成，可選）
!cp -r results /content/drive/MyDrive/pinns_results/
```

---

**準備就緒！現在可以開始訓練了 🚀**
