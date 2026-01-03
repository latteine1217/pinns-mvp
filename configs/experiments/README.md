# 實驗對比配置組織

本目錄包含系統性對比實驗的配置文件，依據論文需求設計。

**注意**: 本目錄的配置被 `configs/sweeps/` 中的 WandB sweep 定義引用，用於批次實驗執行。

---

## 📂 實驗目錄結構

```
experiments/
├── S1_sensor_strategy/      # 感測點策略對比（Random vs QR-pivot）
├── S2_k_scan/               # K 值掃描實驗（K=30/50/80/100）
├── M1_model_comparison/     # 模型表示能力對比（Vanilla vs Full）
├── A1_ablation_fourier/     # 消融實驗：Fourier Features
├── A2_ablation_weights/     # 消融實驗：動態權重（GradNorm）
├── C1_prior_comparison/     # RANS Prior 對比（有無先驗）
├── C2_prior_sweep/          # RANS Prior 權重掃描（0.1/0.3/0.5）
├── generate_experiment_configs.py  # 配置生成腳本
└── README.md                # 本文件
```

**總計**: 17 個實驗配置，對應 7 組對比實驗 (S1, S2, M1, A1, A2, C1, C2)

---

## 🎯 最少實驗集（Minimal Set）

根據論文需求，優先完成以下實驗以支撐主要 claims：

### Phase 1: 2D Kolmogorov（低成本驗證）

| 實驗 ID | 實驗名稱 | 配置文件位置 | 目的 |
|---------|---------|-------------|------|
| **S1** | 感測點策略對比 | `S1_sensor_strategy/` | 驗證 QR-pivot 優於 Random |
| **S2** | K 值掃描 | `S2_k_scan/` | 繪製 K-error 曲線 |
| **M1** | 模型對比 | `M1_model_comparison/` | Vanilla vs Full 性能差異 |
| **A1** | Fourier 消融 | `A1_ablation_fourier/` | 量化 Fourier Features 貢獻 |
| **A2** | 動態權重消融 | `A2_ablation_weights/` | 量化 GradNorm 貢獻 |

### Phase 2: Prior Experiments（2D Kolmogorov 先驗實驗）

| 實驗 ID | 實驗名稱 | 配置文件位置 | 目的 |
|---------|---------|-------------|------|
| **C1** | Prior 對比 | `C1_prior_comparison/` | 驗證 RANS Prior 改善重建 |
| **C2** | Prior 權重掃描 | `C2_prior_sweep/` | 找最佳 prior_weight |

> 3D Channel Flow（JHTDB）建議先用 `configs/templates/3d_slab_curriculum.yml` 跑通 pipeline；本目錄目前僅收錄 2D Kolmogorov 的系統性對比配置。

---

## 📋 實驗執行順序

### 🔹 Phase 0: 驗證基線（0.5 天）
```bash
# 從 repo root 執行：確認 Vanilla 與 Full 配置可正常訓練
python scripts/train/train.py --cfg configs/experiments/M1_model_comparison/m1_vanilla_K100_2d_re50.yml
python scripts/train/train.py --cfg configs/experiments/M1_model_comparison/m1_full_K100_2d_re50.yml
```

### 🔹 Phase 1: 2D 核心對比（1-2 天）
```bash
# S1: Random vs QR-pivot
for cfg in configs/experiments/S1_sensor_strategy/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# S2: K-scan
for cfg in configs/experiments/S2_k_scan/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# A1: Fourier ablation
for cfg in configs/experiments/A1_ablation_fourier/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# A2: Dynamic weights ablation
for cfg in configs/experiments/A2_ablation_weights/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

### 🔹 Phase 2: Prior（2D Kolmogorov）（2-3 天）
```bash
# C1: RANS Prior comparison
for cfg in configs/experiments/C1_prior_comparison/*.yml; do python scripts/train/train.py --cfg "$cfg"; done

# C2: Prior weight sweep
for cfg in configs/experiments/C2_prior_sweep/*.yml; do python scripts/train/train.py --cfg "$cfg"; done
```

---

## 📊 評估指標

所有實驗必須報告以下指標：

### 主要指標
- ✅ 全場相對 L2：u, v, (w)
- ✅ 壓力梯度誤差：∇p（避免 gauge 問題）
- ✅ 不可壓縮性：‖∇·u‖（均值 + 最大值）

### 3D Channel Flow 額外指標
- ✅ 壁面剪應力 τ_w 誤差
- ✅ 平均速度剖面 U⁺(y⁺) RMSE
- ✅ Reynolds 應力峰值位置與量級
- ✅ 能譜 E(k)（低/中頻重建品質）

### 魯棒性測試
- ✅ 噪聲：σ ∈ {0, 1%, 3%}
- ✅ 遺失：dropout ∈ {0, 10%}
- ✅ 統計：3 seeds 的平均 ± 標準差

---

## 🔬 公平對比原則

每組對比實驗必須遵守：

1. **固定變因**：
   - 相同 `seed: 42`
   - 相同感測器文件（不重抽）
   - 相同數據時間窗口
   - 相同 N_pde、batch_size

2. **單一變因**：
   - 每個實驗只改變一個變因
   - 其餘參數完全一致

3. **Loss 權重守恆**：
   - 若禁用自適應權重，需手動標準化
   - 避免「贏在 scale」

4. **以 Evaluation 為準**：
   - 不使用 training loss 作為主要指標
   - 使用獨立驗證集評估

---

## 📈 預期論文圖表

根據實驗結果，建議生成以下圖表：

1. **K-error 曲線**（S1, S2）
   - Random vs QR-pivot @ 不同 K 值
   - 2D 與 3D 各一張

2. **Ablation Bar Chart**（A1, A2）
   - Full vs 各項特徵關閉版本
   - 速度場與壓力場誤差並列

3. **Prior Weight Sweep**（C2）
   - Error vs prior_weight 曲線
   - 標示最佳 sweet spot

4. **Channel Mean Profile**（C1）
   - U⁺ vs y⁺ 對比（DNS vs PINN vs PINN+Prior）
   - τ_w 誤差柱狀圖

5. **能譜對比**（M1, C1）
   - E(k) 對比（Vanilla vs Full vs DNS）
   - 低/中/高頻區域標註

---

## 🚀 快速開始

```bash
# 1) 執行一個配置（從 repo root）
python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml

# 2) 評估一個 checkpoint（例如 best_model.pth）
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/S1_qr_K100/best_model.pth \
  --config configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml
```

---

## 📝 配置命名規範

```
<實驗ID>_<變因描述>_<benchmark>.yml

範例：
- S1_random_K100_2d_re50.yml       # S1 實驗，Random 感測器
- S1_qr_K100_2d_re50.yml          # S1 實驗，QR-pivot 感測器
- M1_vanilla_K100_2d_re50.yml     # M1 實驗，Vanilla 模型
- M1_full_K100_2d_re50.yml        # M1 實驗，Full 特徵
- C2_prior_0.1_3d_slab.yml        # C2 實驗，prior_weight=0.1
```

---

## ⚠️ 注意事項

1. **資源管理**：
   - 2D 實驗：T4 GPU 可能需數小時（視 epochs/設定而定）
   - 裝置選擇請在 YAML 內設定 `experiment.device`（`auto/cuda/cpu`），訓練 CLI 不提供 `--device` 旗標

2. **檢查點管理**：
   - 每個實驗使用獨立的 checkpoint 目錄
   - 定期備份最佳模型
   - 保留完整的訓練日誌

3. **結果追蹤**：
   - 使用 TensorBoard 監控訓練
   - 記錄所有超參數與結果到表格
   - 保存評估指標的 JSON 文件

---

**最後更新**：2025-12-13  
**維護者**：PINNs-MVP Team
