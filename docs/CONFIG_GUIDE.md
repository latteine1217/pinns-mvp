# 配置指南

> **設計原則**: 單一真相來源 (Single Source of Truth)

本專案的配置系統採用 YAML 格式，所有配置範例與說明集中管理，避免重複維護。

## 快速導航

### 🎯 我要...

| 任務 | 文件 |
|------|------|
| **查看所有可用配置鍵與預設值** | [`configs/standard_config_template.yml`](../configs/standard_config_template.yml) |
| **學習如何寫配置** | 本文件（繼續往下讀） |
| **驗證配置文件（鍵名 + 結構）** | `python scripts/tools/validate_config_keys.py <config.yml>` |
| **驗證配置文件（語義 + 型別）** | `python scripts/tools/validate_config.py --config <config.yml>` |
| **複製範例配置** | 瀏覽 `configs/` 目錄 |
| **查看配置變更歷史** | [`CHANGELOG.md`](../CHANGELOG.md) + Git 歷史 |

## 配置結構概覽

以下為「最小可運行」的高層結構，完整鍵名請以 `standard_config_template.yml` 為準：

```yaml
experiment:          # 實驗基本資訊（名稱、種子、設備）
  name: "my_exp"
  seed: 42
  device: "auto"

reproducibility:     # 可重現性
  deterministic: true

data:               # 資料來源與預處理
  source: "jhtdb"
  dataset: "channel"

normalization:      # 輸入/輸出標準化與資料統計
  type: training_data_norm

sensors:            # 感測器設定
  K: 100
  selection_method: "qr_pivot"

model:              # 網路架構
  type: "fourier_vs_mlp"
  in_dim: 3
  out_dim: 4
  width: 256
  depth: 8
  activation: "sine"
  output_variables: [u, v, w, p]
  fourier_features:
    type: axis_selective
    fourier_m: 12
    fourier_sigma: 4.0

physics:            # 物理方程與邊界條件
  type: "vs_pinn_channel_flow"
  nu: 5.0e-5
  domain:
    x_range: [0.0, 25.13]
    y_range: [-1.0, 1.0]
    z_range: [0.0, 9.42]

losses:             # ⚠️ 注意：複數形式，不是 'loss'
  data_weight: 10.0
  momentum_x_weight: 1.0

training:           # 訓練超參數
  epochs: 1000
  optimizer:
    type: "soap"
    lr: 1e-3

output:             # 輸出路徑（TrainerBuilder 依此建立 CheckpointManager）
  checkpoint_dir: ./checkpoints
  results_dir: ./results
```

## 必做檢查（Fail Fast）

在訓練前務必執行：

```bash
python scripts/tools/validate_config_keys.py configs/my_exp.yml
python scripts/tools/validate_config.py --config configs/my_exp.yml
```

這兩步分別檢查：
- **鍵名一致性**（如 `loss` vs `losses`、舊版 Fourier 鍵名）
- **語義/型別錯誤**（如缺少必填段落、維度不一致、LR 類型錯誤）

## 重要段落說明

### 1. model（模型）

- **必填**: `model.type`, `in_dim`, `out_dim`, `width`, `depth`, `activation`
- **必填**: `model.output_variables`（長度需等於 `out_dim`）
- **Fourier features** 必須放在 `model.fourier_features`：
  - `type`: `standard` | `axis_selective` | `disabled`
  - 啟用時需提供 `fourier_m`, `fourier_sigma`

> 若舊配置缺少 `output_variables`，請執行：
> `python scripts/tools/add_output_variables.py --config <file>`

### 2. physics（物理方程）

- **Channel Flow**: `type: vs_pinn_channel_flow`
- **Kolmogorov 2D**: `type: kolmogorov_flow_2d`
- **標準 NS 2D**: `type: ns_2d`

VS-PINN 會讀取 `physics.vs_pinn.scaling_factors`：

```yaml
physics:
  type: vs_pinn_channel_flow
  vs_pinn:
    scaling_factors:
      N_x: 2.0
      N_y: 12.0
      N_z: 2.0
```

### 3. losses（損失項）

- 必須是 `losses`（複數）
- 常用權重：`data_weight`, `momentum_*_weight`, `continuity_weight`, `prior_weight`
- 自適應權重（GradNorm）：

```yaml
losses:
  adaptive_weighting: true
  adaptive:
    scheme: grad_norm
    init_weights:         # 可選：指定損失項的相對重要性（默認所有為 1.0）
      u_ic: 100.0         # 初始條件比 PDE 重要 100 倍
      v_ic: 100.0
      ru: 1.0             # PDE 基準
      rv: 1.0
      rc: 1.0
    momentum: 0.9         # EMA 平滑係數（對齊 grad_norm_momentum）
    update_every_steps: 1000  # 更新頻率（對齊 weight_update_freq）
  weight_update_freq: 1000    # 向後相容（優先使用 adaptive.update_every_steps）
  grad_norm_momentum: 0.9     # 向後相容（優先使用 adaptive.momentum）
  grad_norm_alpha: 1.5        # [已棄用] JaxPI 簡化版本不使用
  grad_norm_normalize: false  # [已棄用/無效] JaxPI 不使用權重正規化（設為 true 會觸發警告）
  adaptive_loss_terms: [data, momentum_x, momentum_y, continuity]
```

**⚠️ `grad_norm_normalize` 警告**（v1.1.0+ 修復）:
- **狀態**: 已棄用且無效（對齊 JaxPI 行為）
- **原因**: JaxPI-style GradNorm 不使用權重總和正規化
- **默認值**: `false`（v1.1.0+ 從 `true` 改為 `false`）
- **行為**: 設為 `true` 會觸發警告，但不執行正規化
- **建議**: 明確設為 `false` 或從配置中移除

**`init_weights` 語義說明**（v1.1.0+ 修復）:
- **作用**: 指定不同損失項的**相對重要性**（而非絕對權重）
- **默認**: 所有損失項初始權重為 `1.0`
- **示例**: `u_ic: 100.0` 表示初始條件比 PDE 重要 100 倍
- **動態範圍**: GradNorm 會在此基礎上動態調整（因子範圍 `[0.1, 10.0]`）
- **最終權重**: `applied_weight = base_weight × gradnorm_factor`
  - `base_weight`: 來自 `losses.data_weight`, `losses.pde_weight` 等
  - `gradnorm_factor`: GradNorm 動態計算的相對因子
- **別名映射**:
  - `ru` → `momentum_x`
  - `rv` → `momentum_y`
  - `rc` → `continuity` / `divergence`
  - `u_ic`, `v_ic` → `initial_condition`

**⚠️ GradNorm 相對權重裁剪**（v1.1.0+ 修復）:
- **裁剪語義**: `min_weight` 和 `max_weight` 是**相對比例**（非絕對值）
- **默認範圍**: `[0.1, 10.0]`（允許權重在初始值的 10% ~ 1000% 範圍內變化）
- **絕對邊界計算**: 
  ```python
  min_abs = init_weight * min_weight  # 例: 100.0 * 0.1 = 10.0
  max_abs = init_weight * max_weight  # 例: 100.0 * 10.0 = 1000.0
  ```
- **示例**: 
  - 若 `u_ic: 100.0`, `min_weight: 0.1`, `max_weight: 10.0`
  - 則 `u_ic` 的實際裁剪範圍為 `[10.0, 1000.0]`（相對範圍 100x）
  - 而 `ru: 1.0` 的裁剪範圍為 `[0.1, 10.0]`（相對範圍 100x）
- **配置方式**（可選，使用默認值即可）:
  ```yaml
  losses:
    adaptive:
      grad_norm_min_weight: 0.1   # 相對比例（10% 下界）
      grad_norm_max_weight: 10.0  # 相對比例（1000% 上界）
  ```
- **修復前行為**（v1.0.x bug）:
  - 使用絕對裁剪範圍 `[0.1, 10.0]`
  - 導致 `init_weight=100` 的項被過度裁剪（實際範圍僅 0.1x）
  - 觀測現象: WandB 權重圖在 step 1k 後出現階躍，權重卡在邊界不再動態調整

- 因果權重（Causal Weighting，對齊 JAX-PI 命名）：

```yaml
losses:
  causal_weighting: true
  causal_tol: 1.0
  num_chunks: 16
```

> 舊版鍵名如 `causal_eps/causal_n_bins` 會被 `validate_config_keys.py` 提示，請更新為 `causal_tol/num_chunks`。

### 4. training（訓練）

- 支援 Optimizer: `adam`, `adamw`, `soap`, `lbfgs`, `sgd`
- `soap` 專屬參數：`precondition_frequency`, `shampoo_beta`
- `lbfgs` 參數會直接傳入 `torch.optim.LBFGS`（如 `max_iter`, `history_size`）
- Time Window 訓練需設定：`num_time_windows`, `time_window_overlap`
- 支援 `sampling.adaptive_collocation`（重採樣策略）

### 5. lowfi_prior（低保真先驗）

```yaml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/channel_rans/rans_k_omega_sst.npz
  data_type: rans
  interpolation:
    method: linear
  consistency_weight: 0.3
  spatial_weighting:
    enabled: true
    strategy: distance_to_wall
```

### 6. output / checkpointing

- **TrainerBuilder** 使用 `output.checkpoint_dir` 建立 CheckpointManager
- Time Window 流程會讀取 `checkpointing.checkpoint_dir`（可覆蓋）

建議在專案內統一使用 `output.*`，並在需要時為 Time Window 額外指定 `checkpointing.*`。

## Registry 可用選項

來源：`pinnx/train/factories.py`, `pinnx/train/model_physics_factory.py`

- **model.type**: `fourier_vs_mlp`, `resnet`, `piratenet`, `axis_selective_fourier_mlp`
- **physics.type**: `vs_pinn_channel_flow`, `ns_2d`, `kolmogorov_flow_2d`
- **optimizer.type**: `adam`, `adamw`, `lbfgs`, `soap`, `sgd`
- **lr_scheduler.type**: `cosine`, `warmup_cosine`, `step`, `exponential`, `multistep`, `reduce_on_plateau`

## 常見錯誤與解決

### ❌ 錯誤 1: 使用 `loss` 而非 `losses`

```yaml
# ❌ 錯誤
loss:
  data_weight: 10.0

# ✅ 正確
losses:
  data_weight: 10.0
```

### ❌ 錯誤 2: 使用已移除的 Fourier 扁平鍵名

```yaml
# ❌ 已移除
model:
  fourier_m: 12
  fourier_sigma: 4.0

# ✅ 正確
model:
  fourier_features:
    type: axis_selective
    fourier_m: 12
    fourier_sigma: 4.0
```

### ❌ 錯誤 3: 缺少 output_variables

```yaml
# ❌ 缺少 model.output_variables
model:
  out_dim: 4

# ✅ 正確
model:
  out_dim: 4
  output_variables: [u, v, w, p]
```

### ❌ 錯誤 4: 模型維度與物理域不匹配

```yaml
# ❌ 3D 域但 2D 模型
physics:
  domain:
    z_range: [0, 9.42]
model:
  in_dim: 2

# ✅ 正確
model:
  in_dim: 3
```

## 配置覆蓋規則

本專案使用 **深度合併 (Deep Merge)** 策略：

```yaml
# base_config.yml
training:
  epochs: 1000
  optimizer:
    type: "adam"
    lr: 1e-3

# my_config.yml（繼承 base_config.yml）
training:
  optimizer:
    lr: 5e-4
```

最終結果：

```yaml
training:
  epochs: 1000
  optimizer:
    type: "adam"
    lr: 5e-4
```

## 最佳實踐

1. **先跑 validate_config_keys.py，再跑 validate_config.py**
2. **從 configs/ 中的範例配置開始，避免從空白檔案起手**
3. **確保 model.output_variables 與 out_dim 一致**
4. **重要實驗請保存配置快照**

```bash
cp configs/my_exp.yml configs/archive/my_exp_20260105.yml
```

## 配置完整參考

**主文件**: [`configs/standard_config_template.yml`](../configs/standard_config_template.yml)

該文件包含：
- 所有可用配置鍵與預設值
- 類型說明與範例

**不要直接修改該文件**，將它作為參考手冊使用。

---

**最後更新**: 2026-01-05
**版本**: v1.3.0（Registry Pattern + Schema Validation）
