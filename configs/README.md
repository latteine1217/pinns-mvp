# PINNs 訓練配置管理指南

## 🚀 快速開始

**新手用戶**：直接使用標準化模板開始訓練 👉 **[`templates/` 目錄](./templates/README.md)**

**進階用戶**：參考下方完整指南進行自定義配置

---

## ⭐ 新預設配置摘要 (2025-10-20 更新)

為了提升基準性能與可重現性，所有模板 (`templates/`) 和主配置 (`main.yml`) 已更新至以下新預設值：

| 參數分類 | 參數 | 新預設值 |
|---|---|---|
| **優化器** | `optimizer.type` | `soap` |
| **損失權重** | `data_weight`, `boundary_weight` | `10.0` |
| | `momentum_*`, `continuity_weight` | `1.0` |
| **採樣點數** | `sampling.pde_points` | `10000` |
| | `sampling.boundary_points` | `5000` |
| **學習率** | `lr_scheduler.type` | `step` (每 1000 epochs 衰減 0.9 倍) |
| **感測器** | `sensors.K` | `500` |
| **自適應權重**| `weight_update_freq` | `1000` |
| **課程學習** | `curriculum.enable` | `true` (預設啟用 `Re_τ` 遞增課程) |

---

## 📋 目錄結構

```
configs/
├── README.md                    # 本文檔（配置管理總指南）
├── templates/                   # ⭐ 標準化模板目錄
│   ├── README.md                #    模板使用指引（詳細）
│   ├── 2d_quick_baseline.yml    #    2D 快速基線（5-10 min）
│   ├── 2d_medium_ablation.yml   #    2D 消融實驗（15-30 min）
│   ├── 3d_slab_curriculum.yml   #    3D 課程學習（30-60 min）
│   └── 3d_full_production.yml   #    3D 生產訓練（2-8 hrs）
│
├── main.yml                     # 生產級主配置
└── ... (其他實驗配置)
```

---

## 📦 模板快速選擇

| 場景 | 模板 | 時間 | 詳細說明 |
|---|---|---|---|
| **快速驗證想法** | [`2d_quick_baseline.yml`](./templates/2d_quick_baseline.yml) | 5-10 分鐘 | [使用指引](./templates/README.md#1️⃣-2d_quick_baselineyml---快速基準測試) |
| **特徵消融研究** | [`2d_medium_ablation.yml`](./templates/2d_medium_ablation.yml) | 15-30 分鐘 | [使用指引](./templates/README.md#2️⃣-2d_medium_ablationyml---中期消融實驗) |
| **課程式訓練** | [`3d_slab_curriculum.yml`](./templates/3d_slab_curriculum.yml) | 30-60 分鐘 | [使用指引](./templates/README.md#3️⃣-3d_slab_curriculumyml---3d-課程學習) |
| **論文級結果** | [`3d_full_production.yml`](./templates/3d_full_production.yml) | 2-8 小時 | [使用指引](./templates/README.md#4️⃣-3d_full_productionyml---生產級訓練) |

👉 **完整模板文檔**：[`templates/README.md`](./templates/README.md)

---

## 🎯 配置命名規範

建議使用 `{spatial}_{duration}_{type}_{variant}_{version}.yml` 格式，以利於實驗追蹤。

---

## 🔧 參數調優建議 (基於新預設)

### 1. **損失權重 (Loss Weights)**

新的預設權重方案為 **數據:物理 = 10:1**，這是一個更穩健的起點。

- **`data_weight: 10.0`**: 給予感測器數據足夠的監督力。
- **`momentum_x/y/z_weight: 1.0`**: 物理方程作為正則化項。
- **`wall_constraint_weight: 10.0`**: 強力執行壁面無滑移條件。

**調優建議**:
- 若 PDE Loss 過高，可考慮將 PDE 權重從 `1.0` 提升至 `2.0` 或 `3.0`。
- 若數據擬合不足，可將 `data_weight` 提升至 `15.0` 或 `20.0`。

### 2. **自適應權重 (GradNorm)**

- **`grad_norm_alpha: 1.5`**: 此參數控制權重調整的恢復力。`1.5` 是一個適用於大多數情況的穩健值。
- **`weight_update_freq: 1000`**: 更新頻率較低，給予優化器足夠的時間來響應權重變化，適用於 `SOAP` 或 `Adam` 等優化器。

### 3. **課程學習 (Curriculum)**

預設的 `Re_τ` 遞增課程是一個強大的工具，適用於從頭開始訓練。

- **階段設計**: 從低 `Re_τ` (如 500) 開始，讓模型先學習流場的基本結構，再逐步過渡到高 `Re_τ` (1000) 來捕捉更複雜的湍流細節。
- **調優**: 如果在階段轉換時遇到不穩定，可以考慮：
    1.  **平滑過渡**: 增加一個介於中間的階段。
    2.  **調整學習率**: 在進入新階段時，使用一個略微降低的學習率。

### 4. **優化器 (Optimizer)**

- **`SOAP`**: 作為新的預設，它在 PINN 問題中通常比 `Adam` 表現出更快的收斂速度和更好的最終精度。其內部的預處理步驟有助於處理 PINN 的不良條件 (ill-conditioned) 損失曲面。
- **`Adam`**: 仍然是一個可靠的選擇，特別是在需要快速進行原型設計或初步測試時。

---

## ✅ 使用流程

1.  **選擇模板**: `cp configs/templates/3d_full_production.yml configs/my_production_v1.yml`
2.  **修改配置**: 至少修改 `experiment.name`。
3.  **執行訓練**: `python scripts/train.py --cfg configs/my_production_v1.yml`
4.  **監控進度**: `tail -f log/my_production_v1/stdout.log`
5.  **評估結果**: `python scripts/evaluate_checkpoint.py --checkpoint checkpoints/my_production_v1/best_model.pth`

---

## 🚨 常見問題 (基於新預設)

### Q1: PDE Loss 在訓練後期停滯不前

**症狀**: 總損失下降緩慢，但 `data_loss` 已經很低。

**診斷**:
1.  **權重比例**: `10:1` 的數據-物理權重可能過於偏重數據擬合。
2.  **學習率**: `StepLR` 在後期可能過小。

**修正範例**:
```yaml
losses:
  momentum_x_weight: 2.0  # 提升 PDE 權重
  momentum_y_weight: 2.0
  ...

lr_scheduler:
  gamma: 0.95 # 減緩學習率衰減速度
```

### Q2: 模型在課程學習階段轉換時不穩定

**症狀**: 從 `Re_τ=750` 進入 `Re_τ=1000` 階段時，損失突然劇增。

**診斷**:
1.  **物理參數跳變**: `nu` 和 `pressure_gradient` 的變化可能過於劇烈。
2.  **學習率**: 進入新階段時，當前的學習率可能過高。

**修正範例**:
```yaml
curriculum:
  stages:
    - name: "Stage3_High_Re_Refinement"
      lr: 0.6e-3 # 手動為此階段設定一個較低的初始學習率
      ...
```

---

**維護者**：PINNs Research Team
**最後更新**：2025-10-20