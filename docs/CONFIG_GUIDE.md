# PINNs 配置文件完整指南

**最後更新**: 2025-10-20
**適用版本**: v1.0+（已移除 enhanced_fourier_mlp）

---

## 📋 目錄

1. [快速開始](#快速開始)
2. [配置結構總覽](#配置結構總覽)
3. [詳細配置選項](#詳細配置選項)
4. [推薦配置組合](#推薦配置組合)
5. [常見問題與調優](#常見問題與調優)
6. [範例配置](#範例配置)

---

## 快速開始

### 1. 複製模板

```bash
# 快速測試（5-10 分鐘）
cp configs/templates/2d_quick_baseline.yml configs/my_test.yml

# 標準訓練（1-2 小時）
cp configs/config_template_example.yml configs/my_experiment.yml

# 生產級訓練（4-8 小時）
cp configs/templates/3d_full_production.yml configs/my_production.yml
```

### 2. 必須修改的欄位

```yaml
experiment:
  name: "my_experiment_name"  # ⚠️ 改成你的實驗名稱

sensors:
  K: 500  # ⚠️ 感測點數量

output:
  checkpoint_dir: "./checkpoints/my_experiment_name"  # ⚠️ 改成你的路徑
  results_dir: "./results/my_experiment_name"
```

### 3. 執行訓練

```bash
python scripts/train.py --cfg configs/my_experiment.yml
```

---

## 配置結構總覽

配置文件分為 13 個主要區塊：

| 區塊 | 用途 | 重要性 |
|------|------|--------|
| **experiment** | 基本實驗設定 | ⭐⭐⭐ |
| **data** | 資料來源與預處理 | ⭐⭐⭐ |
| **normalization** | 資料標準化 | ⭐⭐⭐ 關鍵 |
| **sensors** | 感測點配置 | ⭐⭐⭐ |
| **model** | 模型架構 | ⭐⭐⭐ |
| **physics** | 物理設定 | ⭐⭐⭐ |
| **losses** | 損失函數權重 | ⭐⭐⭐ 關鍵 |
| **training** | 訓練設定 | ⭐⭐⭐ |
| **evaluation** | 評估設定 | ⭐⭐ |
| **physics_validation** | 物理驗證 | ⭐⭐ |
| **logging** | 日誌配置 | ⭐ |
| **curriculum** | 課程學習（可選） | ⭐ |
| **ensemble** | 不確定性量化（可選） | ⭐ |

---

## 詳細配置選項

### 1. 基本實驗設定 (experiment)

```yaml
experiment:
  name: "your_experiment_name"
  version: "v1.0"
  seed: 42                    # 隨機種子（確保可重現性）
  device: "auto"              # auto/cpu/cuda/mps
  precision: "float32"        # float32（推薦）/float64
  description: "實驗描述"
```

**device 選項說明**:
- `auto`: 自動檢測（CUDA > MPS > CPU）
- `cpu`: 強制 CPU（調試用）
- `cuda`: NVIDIA GPU
- `mps`: Apple Silicon GPU

---

### 2. 資料標準化配置 (normalization) ⭐ 關鍵

```yaml
normalization:
  type: 'training_data_norm'  # ⭐ 強烈推薦
  params: {}
```

**type 選項**:

| 類型 | 說明 | 適用場景 | 效果 |
|------|------|---------|------|
| **training_data_norm** | 基於訓練資料統計的 Z-score | ⭐ 所有場景（推薦） | 95-98% 損失降低 |
| **channel_flow** | 通道流特定標準化 | 通道流專用 | 良好 |
| **none** | 不標準化 | ❌ 不推薦 | 訓練不穩定 |

**⚠️ 重要**: 實驗證明 `training_data_norm` 可使損失降低 95-98%，是訓練成功的關鍵！

---

### 3. 感測點配置 (sensors)

```yaml
sensors:
  K: 500                        # 感測點數量
  selection_method: "qr_pivot"  # 選點策略
  spatial_coverage: "optimal"
```

**K（感測點數量）選擇**:

| 場景 | K 範圍 | 訓練時間 | 預期精度 |
|------|--------|---------|---------|
| 快速測試 | 50-100 | 5-10 min | 30-50% L2 error |
| 中等訓練 | 500-1000 | 1-2 hours | 15-30% L2 error |
| 生產級 | 500-1024 | 4-8 hours | <15% L2 error |

**selection_method 選項**:

| 策略 | 說明 | 品質 | 速度 |
|------|------|------|------|
| **qr_pivot** | QR 分解主元選點 | ⭐⭐⭐ 最優 | 中 |
| **stratified** | 分層採樣（wall/log/center） | ⭐⭐ 良好 | 快 |
| **random** | 隨機採樣 | ⭐ 基準 | 最快 |
| **uniform** | 均勻網格 | ⭐ 基準 | 快 |
| **hybrid** | 混合策略（wall + QR） | ⭐⭐ 平衡 | 中 |

---

### 4. 模型架構 (model) ⭐ 已更新

```yaml
model:
  type: "fourier_vs_mlp"  # ⚠️ 推薦（已移除 enhanced_fourier_mlp）
  in_dim: 3
  out_dim: 4
  width: 256              # 隱藏層寬度
  depth: 8                # 隱藏層深度
  activation: "sine"      # 激活函數
```

**⚠️ 架構變更（2025-10-20）**:
- ❌ 已移除: `enhanced_fourier_mlp`
- ✅ 統一使用: `fourier_vs_mlp`
- ✅ 向後兼容: `standard` 仍可用

**type 選項**:

| 類型 | 說明 | 適用場景 |
|------|------|---------|
| **fourier_vs_mlp** | Fourier + VS-PINN（⭐ 推薦） | 所有場景 |
| **standard** | 標準 MLP | 簡單問題 |
| **axis_selective_fourier** | 軸選擇性 Fourier | 高級用戶 |

**width & depth 選擇**:

| 場景 | width | depth | 參數量 | 訓練時間 |
|------|-------|-------|--------|---------|
| 快速測試 | 128 | 4-6 | ~0.5M | 快 |
| 標準訓練 | 256 | 8 | ~2M | 中等 |
| 高精度 | 512 | 10-12 | ~10M | 慢 |

**activation 選項**:

| 函數 | 適用場景 | PINNs 適合度 |
|------|---------|-------------|
| **sine** | PINNs（⭐ 推薦） | ⭐⭐⭐ |
| **tanh** | 標準 MLP | ⭐⭐ |
| **relu** | ❌ 不推薦用於 PINNs | ⭐ |
| **gelu** | Transformer 風格 | ⭐⭐ |

---

### 5. Fourier Features 配置 ⭐ 關鍵

```yaml
model:
  fourier_features:
    type: "axis_selective"  # 每軸獨立頻率（推薦）
    axes_config: {x: [1, 2], y: [], z: [1, 2]}  # 初始頻率
    full_axes_config: {x: [1, 2, 4, 8], y: [], z: [1, 2, 4, 8]}  # 完整頻譜
    domain_lengths: {x: 25.13, y: 2.0, z: 9.42}
    fourier_m: 32
    fourier_sigma: 5.0
```

**頻率配置說明**:

```yaml
[1, 2]        # 2 個八度音程（低頻）
[1, 2, 4]     # 3 個八度音程
[1, 2, 4, 8]  # 4 個八度音程（標準）
[1, 2, 4, 8, 16]  # 5 個八度音程（高解析度）
[]            # 不使用 Fourier（如壁面法向 y 軸）
```

**fourier_m 選擇**:

| 場景 | fourier_m | 特徵數 | 計算成本 |
|------|-----------|--------|---------|
| 快速測試 | 16 | 48 | 低 |
| 標準訓練 | 32 | 96 | 中等 |
| 高精度 | 64 | 192 | 高 |

---

### 6. 損失函數權重 (losses) ⭐⭐⭐ 最關鍵

```yaml
losses:
  # 基礎權重
  data_weight: 10.0
  boundary_weight: 10.0

  # PDE 權重
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  momentum_z_weight: 1.0
  continuity_weight: 1.0

  # 約束權重
  wall_constraint_weight: 10.0  # ⚠️ 重要

  # ⭐ 自適應權重（GradNorm）
  adaptive_weighting: true
  weight_update_freq: 50
  grad_norm_alpha: 1.5          # ⚠️ 關鍵參數
```

**GradNorm Alpha 調優**:

| alpha 值 | 行為 | 適用場景 | PDE Loss Ratio |
|---------|------|---------|---------------|
| 0.0 | 固定權重 | 簡單問題 | 可能 < 10% |
| 0.12 | 弱平衡 | ❌ 已證實過小 | < 10% |
| **1.5** | 強平衡（⭐ 推薦） | 所有場景 | 30-50% |
| 2.0 | 超強平衡 | 物理主導 | > 50% |

**⚠️ 關鍵發現**: 實驗證明 `alpha=0.12` 導致 PDE Loss Ratio < 10%，物理約束失效。**推薦使用 1.5-2.0**。

**weight_update_freq 選擇**:

| 頻率 | 行為 | 適用場景 |
|------|------|---------|
| 10-20 | 頻繁更新 | ❌ 不穩定 |
| **50-100** | 適中（⭐ 推薦） | 標準訓練 |
| 500-1000 | 緩慢更新 | 穩定性優先 |

---

### 7. 訓練設定 (training)

```yaml
training:
  optimizer:
    type: "soap"      # 優化器選擇
    lr: 1.0e-3        # 學習率

  epochs: 2000        # 總訓練輪數
  batch_size: 1024    # 批次大小

  sampling:
    pde_points: 10000      # PDE 碰撞點數
    boundary_points: 5000  # 邊界點數
    wall_clustering: 0.3   # 壁面聚類比例

  gradient_clip: 1.0  # 梯度裁剪
```

**optimizer 選擇**:

| 優化器 | 說明 | 學習率推薦 | 收斂速度 | 記憶體 |
|-------|------|-----------|---------|-------|
| **soap** | Shampoo-like 預條件（⭐ 推薦） | 1e-3 | 快 | 高 |
| **adam** | 標準 Adam | 1e-4 ~ 5e-4 | 中等 | 低 |
| **adamw** | Adam with weight decay | 1e-4 ~ 5e-4 | 中等 | 低 |
| **lbfgs** | 二階方法 | 1.0 | 慢但精確 | 最高 |

**epochs 選擇**:

| 場景 | epochs | 訓練時間（GPU） | 預期精度 |
|------|--------|---------------|---------|
| 快速測試 | 100-500 | 5-30 min | 30-50% |
| 標準訓練 | 2000-3000 | 1-2 hours | 15-30% |
| 生產級 | 5000-10000 | 4-8 hours | <15% |

**pde_points 選擇**:

| 場景 | pde_points | 計算成本 | 物理精度 |
|------|-----------|---------|---------|
| 快速測試 | 2048-4096 | 低 | 基準 |
| 標準訓練 | 10000-15000 | 中等 | 良好 |
| 生產級 | 20000-30000 | 高 | 最佳 |

**wall_clustering 調優**:

| 值 | 壁面點比例 | 適用場景 |
|----|-----------|---------|
| 0.1-0.2 | 10-20% | 低雷諾數 |
| **0.3-0.4** | 30-40%（⭐ 推薦） | Re_tau=1000 |
| 0.5-0.6 | 50-60% | 高雷諾數 |

---

### 8. 學習率調度器 (lr_scheduler)

```yaml
training:
  lr_scheduler:
    type: "step"          # 調度器類型
    step_size: 1000       # 步長
    gamma: 0.9            # 衰減率
```

**type 選項比較**:

| 類型 | 行為 | 適用場景 | 複雜度 |
|------|------|---------|-------|
| **step** | 階梯式衰減（⭐ 簡單穩定） | 所有場景 | 低 |
| **cosine** | 餘弦退火（平滑） | 長時間訓練 | 中 |
| **warmup_cosine** | 預熱+餘弦 | 大型訓練 | 高 |
| **exponential** | 指數衰減 | 穩定衰減 | 低 |
| **none** | 固定學習率 | 快速測試 | 最低 |

---

### 9. 混合精度訓練 (AMP)

```yaml
training:
  amp:
    enabled: false  # 是否啟用 AMP
```

**AMP 使用指南**:

| 設備 | 支援度 | 記憶體節省 | 速度提升 | 建議 |
|------|-------|-----------|---------|------|
| CUDA GPU | ✅ 完整支援 | 30-50% | 10-30% | ⭐ 推薦啟用 |
| Apple MPS | ❌ 不支援 | - | - | 必須禁用 |
| CPU | ❌ 不支援 | - | - | 必須禁用 |

**⚠️ 注意事項**:
- 快速測試建議關閉（簡化除錯）
- 生產訓練建議開啟（節省記憶體）
- MPS 設備必須禁用（會崩潰）

---

### 10. 物理驗證 (physics_validation)

```yaml
physics_validation:
  enabled: true  # ⭐ 推薦啟用
  save_metrics: true

  thresholds:
    mass_conservation: 1.0e-2      # 質量守恆誤差閾值
    momentum_conservation: 1.0e-1  # 動量守恆誤差閾值
    boundary_condition: 1.0e-3     # 邊界條件誤差閾值
```

**說明**:
- 在檢查點保存前自動驗證物理一致性
- 若驗證失敗，拒絕保存並記錄警告
- 可通過 `enabled: false` 禁用（僅用於除錯）

---

## 推薦配置組合

### 組合 1: 🏃 快速測試（5-10 分鐘）

```yaml
experiment:
  name: "quick_test"

sensors:
  K: 50

model:
  width: 128
  depth: 4
  fourier_features:
    fourier_m: 16

losses:
  adaptive_weighting: false  # 簡化變數

training:
  optimizer:
    type: "adam"
    lr: 5.0e-4
  epochs: 100
  batch_size: 512
  sampling:
    pde_points: 2048
  amp:
    enabled: false
```

**預期結果**:
- 訓練時間: 5-10 分鐘（GPU）
- 最終 L2: 30-50%
- PDE ratio: 10-20%

---

### 組合 2: 📊 標準訓練（1-2 小時）

```yaml
experiment:
  name: "standard_training"

sensors:
  K: 500

model:
  width: 256
  depth: 8
  fourier_features:
    fourier_m: 32

losses:
  adaptive_weighting: true
  grad_norm_alpha: 1.5         # ⚠️ 關鍵
  weight_update_freq: 50

training:
  optimizer:
    type: "soap"
    lr: 1.0e-3
  epochs: 2000
  batch_size: 1024
  sampling:
    pde_points: 10000
    wall_clustering: 0.3
  gradient_clip: 1.0
```

**預期結果**:
- 訓練時間: 1-2 小時（GPU）
- 最終 L2: 15-30%
- PDE ratio: 30-50%

---

### 組合 3: 🚀 生產級（4-8 小時）

```yaml
experiment:
  name: "production_run"

sensors:
  K: 1024

model:
  width: 512
  depth: 10
  fourier_features:
    fourier_m: 64
    full_axes_config: {x: [1, 2, 4, 8, 16], y: [], z: [1, 2, 4, 8, 16]}

losses:
  adaptive_weighting: true
  grad_norm_alpha: 1.5
  weight_update_freq: 50
  wall_constraint_weight: 15.0  # 增強壁面約束

training:
  optimizer:
    type: "soap"
    lr: 1.0e-3
  lr_scheduler:
    type: "warmup_cosine"
    warmup_epochs: 100
    min_lr: 1.0e-6
  epochs: 5000
  batch_size: 2048
  sampling:
    pde_points: 20000
    boundary_points: 10000
    wall_clustering: 0.4
  gradient_clip: 1.0
  amp:
    enabled: true  # 若使用 CUDA

curriculum:
  enable: true  # 多階段訓練
```

**預期結果**:
- 訓練時間: 4-8 小時（GPU）
- 最終 L2: <15%
- PDE ratio: >40%

---

## 常見問題與調優

### 問題 1: 訓練出現 NaN/Inf

**原因**:
- 學習率過高
- 標準化未啟用
- 梯度爆炸

**解決方案**:
```yaml
normalization:
  type: 'training_data_norm'  # ✅ 啟用標準化

training:
  optimizer:
    lr: 5.0e-4  # ✅ 降低學習率
  gradient_clip: 1.0  # ✅ 啟用梯度裁剪
```

---

### 問題 2: PDE Loss Ratio < 10%（物理約束失效）

**原因**:
- `grad_norm_alpha` 過小（0.12）
- `weight_update_freq` 過大（1000）
- 物理損失權重過低

**解決方案**:
```yaml
losses:
  grad_norm_alpha: 1.5  # ✅ 從 0.12 增加到 1.5
  weight_update_freq: 50  # ✅ 從 1000 降低到 50

  momentum_x_weight: 1.0  # ✅ 確保基準權重正確
  momentum_y_weight: 1.0
  momentum_z_weight: 1.0
  continuity_weight: 1.0
```

---

### 問題 3: 壁面剪應力歸零（τ_w ≈ 0）

**原因**:
- `wall_constraint_weight` 過低
- 壁面採樣點不足
- `wall_clustering` 過低

**解決方案**:
```yaml
losses:
  wall_constraint_weight: 15.0  # ✅ 從 5.0 增加到 15.0

training:
  sampling:
    boundary_points: 10000  # ✅ 增加邊界點
    wall_clustering: 0.4    # ✅ 從 0.2 增加到 0.4
```

---

### 問題 4: 訓練過慢

**原因**:
- `pde_points` 過多
- 網路過大
- 未啟用 AMP

**解決方案**:
```yaml
model:
  width: 256   # ✅ 從 512 降低到 256
  depth: 8     # ✅ 從 12 降低到 8

training:
  sampling:
    pde_points: 10000  # ✅ 從 20000 降低到 10000

  amp:
    enabled: true  # ✅ 啟用混合精度（若 CUDA）
```

---

### 問題 5: 記憶體不足（OOM）

**解決方案**:
```yaml
training:
  batch_size: 512    # ✅ 從 1024 降低到 512
  sampling:
    pde_points: 4096  # ✅ 從 10000 降低到 4096

  amp:
    enabled: true  # ✅ 啟用 AMP 節省 30-50% 記憶體
```

---

## 範例配置

### 範例 1: 2D 快速基準測試

```yaml
experiment:
  name: "2d_quick_test"
  seed: 42
  device: "auto"

normalization:
  type: 'training_data_norm'
  slice_config:
    plane: "xy"
    z_position: 4.71

sensors:
  K: 50
  selection_method: "qr_pivot"

model:
  type: "fourier_vs_mlp"
  width: 128
  depth: 4
  fourier_features:
    axes_config: {x: [1, 2], y: [], z: [1, 2]}
    fourier_m: 16

losses:
  data_weight: 10.0
  wall_constraint_weight: 10.0
  adaptive_weighting: false

training:
  optimizer:
    type: "soap"
    lr: 1.0e-3
  epochs: 100
  sampling:
    pde_points: 2048
```

---

### 範例 2: 3D 標準訓練

```yaml
experiment:
  name: "3d_standard_training"
  seed: 42
  device: "auto"

normalization:
  type: 'training_data_norm'
  slice_config:
    plane: "3d_full"

sensors:
  K: 500
  selection_method: "qr_pivot"

model:
  type: "fourier_vs_mlp"
  width: 256
  depth: 8
  fourier_features:
    axes_config: {x: [1, 2], y: [], z: [1, 2]}
    full_axes_config: {x: [1, 2, 4, 8], y: [], z: [1, 2, 4, 8]}
    fourier_m: 32

physics:
  vs_pinn:
    scaling_factors:
      N_x: 2.0
      N_y: 12.0
      N_z: 2.0

losses:
  data_weight: 10.0
  wall_constraint_weight: 10.0
  adaptive_weighting: true
  grad_norm_alpha: 1.5
  weight_update_freq: 50

training:
  optimizer:
    type: "soap"
    lr: 1.0e-3
  lr_scheduler:
    type: "step"
    step_size: 1000
    gamma: 0.9
  epochs: 2000
  sampling:
    pde_points: 10000
    wall_clustering: 0.3
  gradient_clip: 1.0
```

---

## 附錄：參數速查表

### 關鍵參數優先級

| 優先級 | 參數 | 推薦值 | 影響 |
|-------|------|-------|------|
| **P0** | `normalization.type` | `'training_data_norm'` | 95-98% 損失降低 |
| **P1** | `losses.grad_norm_alpha` | `1.5-2.0` | 物理約束強度 |
| **P2** | `losses.wall_constraint_weight` | `10.0-15.0` | 防止壁面剪應力歸零 |
| **P3** | `training.sampling.wall_clustering` | `0.3-0.4` | 壁面解析度 |
| **P4** | `sensors.K` | `500-1024` | 重建精度 |
| **P5** | `model.width` × `model.depth` | `256×8` | 表達能力 |

---

## 更多資源

- **完整文檔**: `CLAUDE.md`
- **技術細節**: `TECHNICAL_DOCUMENTATION.md`
- **模板目錄**: `configs/templates/`
- **訓練腳本**: `scripts/train.py`

---

**結束** 🎉
