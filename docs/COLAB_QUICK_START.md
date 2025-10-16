# 🚀 PirateNet Colab 快速啟動指南

## 📋 背景

上次訓練（`piratenet_quick_test`）失敗原因：
- ❌ 壁面邊界錯誤：`y: [0, 2]`（應為 `[-1, 1]`）
- ❌ 學習率調度器未啟用
- ❌ 訓練不足：100 epochs（建議 1000+）

**結果**：L2 誤差 ~100%（目標 ≤15%）

---

## ✅ 已修復配置

### 文件列表
1. **`configs/colab_piratenet_1k.yml`**（380 行）
   - ✅ 壁面邊界：`y: [-1, 1]`（三處一致）
   - ✅ 學習率調度器：`warmup_exponential`
   - ✅ GradNorm 自適應權重
   - ✅ 因果權重：`epsilon=1.0`

2. **`PirateNet_Colab_Training.ipynb`**
   - 完整訓練流程
   - Google Drive 自動保存
   - TensorBoard 監控
   - 故障排除指南

3. **`scripts/evaluate_piratenet_vs_jhtdb.py`**（已修復）
   - 支援 VS-PINN 縮放因子載入

---

## 🎯 Colab 執行步驟

### **方法 1：使用 Notebook（推薦新手）**

#### 步驟 1：上傳到 Colab
```bash
# 選項 A：從 GitHub 克隆
1. 打開 Google Colab
2. 執行：
   !git clone https://github.com/your-username/pinns-mvp.git
   %cd pinns-mvp

# 選項 B：手動上傳文件
1. 上傳 PirateNet_Colab_Training.ipynb 到 Colab
2. 上傳 configs/colab_piratenet_1k.yml
3. 上傳整個 pinnx/ 資料夾
```

#### 步驟 2：執行 Notebook
```python
# 在 Colab 中打開 PirateNet_Colab_Training.ipynb
# 按順序執行所有 Cell（Shift+Enter）

# Notebook 會自動：
# 1. 檢查 GPU 可用性
# 2. 掛載 Google Drive
# 3. 安裝依賴
# 4. 下載 JHTDB 資料
# 5. 訓練模型
# 6. 評估結果
```

#### 步驟 3：監控訓練
```python
# 在新 Cell 中執行
%load_ext tensorboard
%tensorboard --logdir ./checkpoints/colab_piratenet_1k
```

---

### **方法 2：使用命令行（進階用戶）**

#### 步驟 1：環境設定
```python
# Cell 1: 檢查 GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### 步驟 2：掛載 Google Drive
```python
# Cell 2: 掛載 GDrive
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/drive/MyDrive/pinns-mvp/checkpoints
!mkdir -p /content/drive/MyDrive/pinns-mvp/results
```

#### 步驟 3：克隆專案
```bash
# Cell 3: 克隆專案
!git clone https://github.com/your-username/pinns-mvp.git
%cd pinns-mvp
```

#### 步驟 4：安裝依賴
```bash
# Cell 4: 安裝依賴
!pip install -q torch torchvision torchaudio
!pip install -q pyJHTDB h5py pyyaml tensorboard matplotlib scipy
```

#### 步驟 5：下載資料
```bash
# Cell 5: 下載 JHTDB 資料（首次運行）
!mkdir -p data/jhtdb/channel_flow_re1000
!python scripts/fetch_channel_flow.py --K 50 --output data/jhtdb/channel_flow_re1000

# 或從 Google Drive 複製（如果已下載）
!cp -r /content/drive/MyDrive/pinns-mvp/data/jhtdb ./data/
```

#### 步驟 6：開始訓練
```bash
# Cell 6: 訓練模型
!python scripts/train.py --cfg configs/colab_piratenet_1k.yml
```

#### 步驟 7：監控訓練
```python
# Cell 7: TensorBoard（新 Cell）
%load_ext tensorboard
%tensorboard --logdir ./checkpoints/colab_piratenet_1k
```

#### 步驟 8：評估結果
```bash
# Cell 8: 評估檢查點（訓練完成後）
!python scripts/evaluate_piratenet_vs_jhtdb.py \
  --checkpoint checkpoints/colab_piratenet_1k/best_model.pth \
  --config configs/colab_piratenet_1k.yml \
  --device cuda
```

#### 步驟 9：保存結果到 GDrive
```bash
# Cell 9: 保存到 Google Drive
!cp -r checkpoints/colab_piratenet_1k /content/drive/MyDrive/pinns-mvp/checkpoints/
!cp -r results/colab_piratenet_1k /content/drive/MyDrive/pinns-mvp/results/
```

---

## 📊 訓練監控指標

### **關鍵檢查點**（前 100 epochs）

| Epoch | 檢查項目 | 預期值 | 問題判斷 |
|-------|---------|--------|---------|
| 10 | `wall_loss` | > 0 | 若 = 0 → 壁面邊界未生效 |
| 20 | `learning_rate` | < 1e-3 | 若不變 → 調度器未啟用 |
| 50 | `total_loss` | < 初始值 50% | 若未降 → 訓練不收斂 |
| 100 | `pde_loss` | 穩定或下降 | 若暴增 → 權重失衡 |

### **最終驗收指標**（1000 epochs）

```bash
# 執行評估
!python scripts/evaluate_piratenet_vs_jhtdb.py \
  --checkpoint checkpoints/colab_piratenet_1k/best_model.pth \
  --config configs/colab_piratenet_1k.yml \
  --device cuda
```

**成功標準**：
- ✅ 速度場 L2：≤ 20%（可接受）/ ≤ 15%（理想）
- ✅ 壓力場 L2：≤ 25%（可接受）/ ≤ 20%（理想）
- ✅ `wall_loss` > 0（壁面約束生效）
- ✅ 訓練穩定收斂（無 NaN）

---

## ⚠️ 常見問題排除

### **問題 1：GPU 不可用**
```python
# 解決方式
Runtime → Change runtime type → Hardware accelerator: GPU
```

### **問題 2：記憶體不足**
```yaml
# 修改 configs/colab_piratenet_1k.yml
training:
  batch_size: 4096  # 降低批次大小（原 8192）
```

### **問題 3：訓練中斷**
```bash
# 從檢查點恢復
!python scripts/train.py \
  --cfg configs/colab_piratenet_1k.yml \
  --resume checkpoints/colab_piratenet_1k/latest.pth
```

### **問題 4：JHTDB 資料下載失敗**
```bash
# 選項 A：從 Google Drive 載入（如果已上傳）
!cp -r /content/drive/MyDrive/pinns-mvp/data/jhtdb ./data/

# 選項 B：使用預處理的感測點
# （需要先在本地運行 fetch_channel_flow.py）
```

### **問題 5：wall_loss = 0**
```bash
# 檢查配置文件
!rg "y_min:\s*-1\.0" configs/colab_piratenet_1k.yml
!rg "y:\s*\[-1\.0,\s*1\.0\]" configs/colab_piratenet_1k.yml

# 應該顯示 3 處匹配（data.domain, physics.domain, jhtdb_config.domain）
```

### **問題 6：學習率不變**
```python
# 檢查訓練日誌
!tail -100 log/colab_piratenet_1k/training.log | grep "learning_rate"

# 預期：learning_rate 應逐步下降
# Epoch 20: ~1e-3 → Epoch 100: ~5e-4 → Epoch 500: ~1e-4
```

---

## 📈 預期訓練時間

| GPU 型號 | Batch Size | Epochs | 訓練時間 | 預期 L2 |
|---------|------------|--------|---------|---------|
| T4 | 8192 | 1000 | 1.5-2 hrs | 20-25% |
| T4 | 8192 | 2000 | 3-4 hrs | 15-20% |
| V100 | 8192 | 1000 | 40-60 min | 20-25% |
| V100 | 8192 | 2000 | 1.5-2 hrs | 15-20% |
| A100 | 8192 | 1000 | 20-30 min | 20-25% |

---

## 🎯 成功驗證流程

### **步驟 1：訓練完成後檢查檢查點**
```bash
!ls -lh checkpoints/colab_piratenet_1k/
# 應包含：
# - best_model.pth
# - latest.pth
# - epoch_100.pth, epoch_200.pth, ...
```

### **步驟 2：執行評估**
```bash
!python scripts/evaluate_piratenet_vs_jhtdb.py \
  --checkpoint checkpoints/colab_piratenet_1k/best_model.pth \
  --config configs/colab_piratenet_1k.yml \
  --device cuda \
  --output results/colab_piratenet_1k
```

### **步驟 3：檢查結果**
```python
import json

# 讀取統計結果
with open('results/colab_piratenet_1k/vs_jhtdb_statistics.json') as f:
    stats = json.load(f)

print(f"速度場 L2 誤差：")
print(f"  u: {stats['statistics']['u']['relative_l2']*100:.2f}%")
print(f"  v: {stats['statistics']['v']['relative_l2']*100:.2f}%")
print(f"  w: {stats['statistics']['w']['relative_l2']*100:.2f}%")
print(f"壓力場 L2: {stats['statistics']['p']['relative_l2']*100:.2f}%")
print(f"\n整體評估: {'✅ 成功' if stats['success_criteria']['overall_success'] else '❌ 需改進'}")
```

### **步驟 4：視覺化結果**
```bash
!ls results/colab_piratenet_1k/visualizations/
# 應包含：
# - velocity_comparison.png
# - pressure_comparison.png
# - error_distribution.png
```

---

## 📂 檔案結構

```
pinns-mvp/
├── configs/
│   └── colab_piratenet_1k.yml          # ✅ 修復後的配置
├── scripts/
│   ├── train.py                         # 訓練腳本
│   ├── evaluate_piratenet_vs_jhtdb.py  # ✅ 修復後的評估腳本
│   └── fetch_channel_flow.py           # 資料下載
├── pinnx/                               # 核心模組
├── checkpoints/
│   └── colab_piratenet_1k/             # 訓練檢查點（自動創建）
├── results/
│   └── colab_piratenet_1k/             # 評估結果（自動創建）
├── data/
│   └── jhtdb/
│       └── channel_flow_re1000/        # JHTDB 資料（下載後）
└── PirateNet_Colab_Training.ipynb      # ✅ Colab Notebook
```

---

## 🔗 下一步行動

### **立即執行**
1. 打開 Google Colab
2. 上傳 `PirateNet_Colab_Training.ipynb`
3. 執行所有 Cell
4. 等待訓練完成（1-2 小時）
5. 檢查評估結果

### **若達標（L2 ≤ 15%）**
1. 保存檢查點到 Google Drive
2. 執行 K-scan 實驗（找出最少感測點數）
3. Ensemble 訓練（不確定性量化）

### **若未達標（L2 > 15%）**
1. 延長訓練時間（2000-5000 epochs）
2. 增加感測點（K=80-100）
3. 調整網路結構（width=1024）
4. 檢查訓練日誌（確認修復生效）

---

## 📧 支援與回報

若遇到問題，請提供：
1. 訓練日誌（`log/colab_piratenet_1k/training.log`）
2. 評估結果（`results/colab_piratenet_1k/vs_jhtdb_statistics.json`）
3. 配置文件（`configs/colab_piratenet_1k.yml`）
4. TensorBoard 截圖（損失曲線）

---

## ✅ 修復驗證清單

在開始訓練前，請確認：
- [ ] GPU 可用（`torch.cuda.is_available() == True`）
- [ ] Google Drive 已掛載
- [ ] 配置文件中 `y_min = -1.0`（三處）
- [ ] 學習率調度器已啟用（`scheduler.type: warmup_exponential`）
- [ ] GradNorm 已啟用（`adaptive_weights.enabled: true`）
- [ ] JHTDB 資料已下載或從 GDrive 載入

---

**祝訓練順利！🚀**
