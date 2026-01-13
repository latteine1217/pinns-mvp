# Experiment Scripts Index

本目錄包含所有實驗的執行腳本，支援 Local、SLURM 和 Google Colab 三種環境。

---

## 📋 腳本清單

### S1: Sensor Strategy Comparison
**目的**: 比較 QR-Pivot vs Random 感測器策略

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_s1_sensor_strategy_colab.sh` | 2 | 4-8h |

**配置檔案**:
- `configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re100.yml`
- `configs/experiments/S1_sensor_strategy/s1_random_K100_2d_re100.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_s1_sensor_strategy_colab.sh
```

---

### S2: K-Value Scan
**目的**: 掃描不同感測器數量 (K=30,50,80,100,200)

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_s2_k_scan_colab.sh` | 5 | 10-20h |
| SLURM | `run_s2_k_scan_slurm.sh` | 5 | Parallel |
| Local | `run_s2_k_scan_sequential.sh` | 5 | 10-20h |

**配置檔案**:
- `configs/experiments/S2_k_scan/s2_qr_K30_2d_re50.yml`
- `configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml`
- `configs/experiments/S2_k_scan/s2_qr_K80_2d_re50.yml`
- `configs/experiments/S2_k_scan/s2_qr_K100_2d_re50.yml`
- `configs/experiments/S2_k_scan/s2_qr_K200_2d_re50.yml` ⚠️ High Risk

**執行方式**:
```bash
# Colab - 全部執行
bash scripts/experiments/run_s2_k_scan_colab.sh

# Colab - 指定 K 值
bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80

# Colab - 快速測試 (自訂 epochs)
export COLAB_EPOCHS=3000 && bash scripts/experiments/run_s2_k_scan_colab.sh 50

# SLURM
sbatch scripts/experiments/run_s2_k_scan_slurm.sh

# Local
bash scripts/experiments/run_s2_k_scan_sequential.sh
```

---

### A1: Ablation Study - Fourier Features
**目的**: 測試 Fourier Features 對模型效能的影響

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_a1_ablation_fourier_colab.sh` | 2 | 4-8h |

**配置檔案**:
- `configs/experiments/A1_ablation_fourier/a1_with_fourier_K100_2d_re50.yml`
- `configs/experiments/A1_ablation_fourier/a1_without_fourier_K100_2d_re50.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_a1_ablation_fourier_colab.sh
```

---

### A2: Ablation Study - Adaptive Weighting
**目的**: 測試 Adaptive Weighting 對模型效能的影響

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_a2_ablation_weights_colab.sh` | 2 | 4-8h |

**配置檔案**:
- `configs/experiments/A2_ablation_weights/a2_with_adaptive_K100_2d_re50.yml`
- `configs/experiments/A2_ablation_weights/a2_without_adaptive_K100_2d_re50.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_a2_ablation_weights_colab.sh
```

---

### C1: Prior Comparison
**目的**: 比較有無 RANS Prior 對模型效能的影響

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_c1_prior_comparison_colab.sh` | 2 | 4-8h |

**配置檔案**:
- `configs/experiments/C1_prior_comparison/c1_with_prior_K100_2d_re50.yml`
- `configs/experiments/C1_prior_comparison/c1_no_prior_K100_2d_re50.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_c1_prior_comparison_colab.sh
```

---

### C2: Prior Weight Sweep
**目的**: 掃描不同 RANS Prior 權重 (0.1, 0.3, 0.5)

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_c2_prior_sweep_colab.sh` | 3 | 6-12h |

**配置檔案**:
- `configs/experiments/C2_prior_sweep/c2_prior_0.1_K100_2d_re50.yml`
- `configs/experiments/C2_prior_sweep/c2_prior_0.3_K100_2d_re50.yml`
- `configs/experiments/C2_prior_sweep/c2_prior_0.5_K100_2d_re50.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_c2_prior_sweep_colab.sh
```

---

### M1: Model Architecture Comparison
**目的**: 比較 Full Model vs Vanilla Model 效能差異

| 環境 | 腳本 | 配置數 | 預估時間 |
|------|------|--------|----------|
| Colab | `run_m1_model_comparison_colab.sh` | 2 | 4-8h |

**配置檔案**:
- `configs/experiments/M1_model_comparison/m1_full_K100_2d_re50.yml`
- `configs/experiments/M1_model_comparison/m1_vanilla_K100_2d_re50.yml`

**執行方式**:
```bash
# Colab
bash scripts/experiments/run_m1_model_comparison_colab.sh
```

---

## 🎯 實驗執行優先順序

### Phase 1: 基礎實驗 (Foundation)
1. **S2**: K-value scan - 找出最佳 K 值
2. **S1**: Sensor strategy - 驗證 QR-Pivot 優於 Random

### Phase 2: 消融研究 (Ablation Studies)
3. **A1**: Fourier features - 特徵工程影響
4. **A2**: Adaptive weighting - 損失函數平衡影響

### Phase 3: Prior 研究 (Prior Studies)
5. **C1**: Prior comparison - RANS prior 整體影響
6. **C2**: Prior sweep - 最佳 prior 權重

### Phase 4: 架構比較 (Architecture)
7. **M1**: Model comparison - Full vs Vanilla

---

## 🔧 通用使用方式

### Google Colab 環境設置

```python
# 1. 掛載 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 切換至專案目錄
import os
os.chdir('/content/drive/MyDrive/pinns-sparse-flow')

# 3. 執行環境設置
%run scripts/setup_colab_env.py

# 4. 執行實驗
!bash scripts/experiments/run_<experiment>_colab.sh
```

### 自訂 Epochs（快速測試）

```bash
# 使用環境變數覆蓋 epochs
export COLAB_EPOCHS=3000

# 執行任何實驗腳本
bash scripts/experiments/run_s2_k_scan_colab.sh
```

### 監控訓練進度

```bash
# 查看最新日誌
tail -f logs/experiments/<experiment>/*.log

# 查看 GPU 使用率
watch -n 1 nvidia-smi

# 查看 checkpoint 狀態
ls -lh checkpoints/experiments/
```

---

## 📊 評估與比較

所有實驗完成後，使用統一評估腳本：

```bash
# 單一模型評估
python scripts/evaluate_unified.py \
  --checkpoint checkpoints/experiments/<exp_name>/best_model.pth \
  --output results/<exp_name>_evaluation.png

# 多模型比較
python scripts/evaluate_unified.py \
  --checkpoints \
    checkpoints/experiments/<exp1>/best_model.pth \
    checkpoints/experiments/<exp2>/best_model.pth \
  --labels "Config 1" "Config 2" \
  --output results/comparison.png
```

---

## ⚠️ 重要注意事項

### K=200 高風險警告
- **Condition number**: 1.32×10⁸ (極高)
- **失敗機率**: 80-90%
- **建議**: 僅用於研究極限情況
- **監控**: 密切關注 NaN 和梯度爆炸

### Colab 時間限制
- **Free 版本**: 12 小時
- **Pro 版本**: 24 小時
- **建議**: 分批執行實驗，避免超時

### Google Drive 備份
所有腳本自動備份 checkpoints 至：
```
/content/drive/MyDrive/pinns_checkpoints/<experiment_name>/
```

---

## 📁 檔案結構

```
scripts/experiments/
├── README.md                              # 本文件
├── run_s1_sensor_strategy_colab.sh       # S1 實驗
├── run_s2_k_scan_colab.sh                # S2 實驗 (Colab)
├── run_s2_k_scan_sequential.sh           # S2 實驗 (Local)
├── run_s2_k_scan_slurm.sh                # S2 實驗 (SLURM)
├── run_a1_ablation_fourier_colab.sh      # A1 實驗
├── run_a2_ablation_weights_colab.sh      # A2 實驗
├── run_c1_prior_comparison_colab.sh      # C1 實驗
├── run_c2_prior_sweep_colab.sh           # C2 實驗
└── run_m1_model_comparison_colab.sh      # M1 實驗
```

---

## 🔗 相關文檔

- **實驗設計**: `configs/experiments/README.md`
- **配置指南**: `docs/CONFIG_GUIDE.md`
- **評估指南**: `docs/EVALUATION_GUIDE.md`
- **Colab 快速開始**: `docs/COLAB_QUICK_START.md`
- **完整 Colab 指南**: `docs/COLAB_GUIDE.md`

---

## 💡 提示與技巧

1. **快速驗證**: 先用 COLAB_EPOCHS=1000 快速測試流程
2. **分批執行**: Colab 每次執行 2-3 個實驗，避免超時
3. **定期備份**: 確認 Google Drive 空間充足
4. **監控資源**: 使用 `nvidia-smi` 確認 GPU 正常運作
5. **日誌檢查**: 訓練中定期檢查 logs 目錄，及早發現問題
