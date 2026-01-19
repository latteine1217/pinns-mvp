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
  K: 400
  selection_method: "qr_pivot"
  sensor_file: ./data/kolmogorov_sensors/re100/sensors_temporal_K400_N256_t0-20.json
  dns_values_file: ./data/kolmogorov_sensors/re100/sensors_temporal_K400_N256_t0-20_dns_values.npz

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
  type: "kolmogorov_flow_2d"
  rho: 1.0
  # Kolmogorov 其餘物理參數由 DNS NPY 自動回補

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
  nu: 5.0e-5  # 運動黏度
  rho: 1.0    # 流體密度
  vs_pinn:
    scaling_factors:
      N_x: 2.0
      N_y: 12.0
      N_z: 2.0
  channel_flow:
    Re_tau: 1000.0  # 摩擦雷諾數（唯一必須配置的參數）
    # u_tau 和 pressure_gradient 將自動計算
  domain:
    x_range: [0.0, 25.13]
    y_range: [-1.0, 1.0]
    z_range: [0.0, 9.42]
```

**Channel Flow 參數計算**：
- 只需提供 `Re_tau`（摩擦雷諾數）
- 其他參數將自動計算：
  - `u_τ = Re_τ * ν / h`（摩擦速度，h 為通道半高）
  - `dP/dx = ρ * u_τ² / h`（壓力梯度）
- ❌ **已移除的參數**（v1.5.0+）：
  - `Re_bulk`：未被使用，配置時將被忽略
  - `u_tau`：從 Re_tau 自動計算，配置時將被忽略
  - `pressure_gradient`：從 Re_tau 自動計算，配置時將被忽略

**Kolmogorov Flow 2D 參數自動同步**：

當使用 `type: kolmogorov_flow_2d` 時，**強烈建議**使用 DNS NPY 自動同步參數，無需在 `physics` 區段重複配置。

**✅ 推薦配置**（最小化，參數從 DNS NPY 自動同步）：
```yaml
physics:
  type: kolmogorov_flow_2d
  nu: 0.01    # ⚠️ 可選：若省略則從 DNS NPY 讀取
  rho: 1.0    # 密度（DNS NPY 不提供，必須在此配置）
  forcing: {} # ✂️ 空字典（k_f 和 amplitude 從 DNS NPY 自動同步）
  domain:
    x_range: [0.0, 6.283185307179586]
    y_range: [0.0, 6.283185307179586]
  boundary_conditions:
    periodic_x: true
    periodic_y: true
```

**參數來源優先級**：
1. **DNS NPY**（最高優先）← 從 `data.kolmogorov_config.physics_params` 讀取
2. **YAML config**（次優先）← 從 `physics.forcing` 等讀取
3. **預設值**（最低優先）

**自動同步的參數**：
- `nu`（運動黏度）
- `k_f`（強迫波數）
- `forcing_amplitude`（強迫振幅）
- `L`（域大小，用於 domain range）

詳細說明請參閱：[Kolmogorov Flow 參數自動同步](#kolmogorov-flow-參數自動同步dns-npy--physics-module)

**❌ 已簡化/移除的區段**（v2.0+）：
- `physics.kolmogorov_flow`：重複配置，已移除
- `physics.forcing.{k_f, amplitude}`：從 DNS NPY 自動同步，可省略

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
- 支援 `sampling.adaptive_collocation`（重採樣策略）

#### Time Window 訓練配置

Time Window 模式支援分段訓練時間演化問題（目前僅支援 Kolmogorov Flow）：

```yaml
training:
  num_time_windows: 3           # > 1 啟用 Time Window
  time_window_overlap: 0.1      # 窗口重疊比例（可選，預設 0.0）
  transfer_learning: true       # 啟用遷移學習（可選）

data:
  kolmogorov_config:
    enabled: true               # ✅ 必須啟用
    time_range: [15.0, 25.0]   # ✅ 必須指定時間範圍
    domain:                     # ✅ 必須指定空間域
      x: [0.0, 12.566370614359172]
      y: [0.0, 6.283185307179586]
```

**配置驗證規則**:
1. `num_time_windows > 1` 時自動啟用 Time Window 模式
2. 必須啟用 `data.kolmogorov_config.enabled = true`
3. 必須指定 `data.kolmogorov_config.time_range`
4. 窗口持續時間 = (t_end - t_start) / num_windows，建議 > 0.1s

**常見錯誤**:
- ❌ 缺少 `time_range`: "Time Window 配置錯誤：缺少時間範圍"
- ❌ Kolmogorov Flow 未啟用: "Time Window mode requires Kolmogorov Flow enabled"
- ⚠️ 窗口過短: "窗口持續時間過短：建議減少窗口數量"

#### Kolmogorov Flow 參數自動同步（DNS NPY → Physics Module）

**🎯 設計原則：單一真相來源（Single Source of Truth）**

當 `data.kolmogorov_config.data_path` 指向 DNS NPY 檔案時，系統實現**完全自動化的參數同步**，無需在 YAML 中重複配置物理參數。

##### 自動回填機制

系統會從 DNS NPY 的 `config` 欄位自動回填以下參數到 `config['data']['kolmogorov_config']`：

**資料參數**：
- `dt`（時間步長）
- `L`（域大小）
- `resolution`（網格解析度：`{x: N, y: N}`）
- `variables`（變數列表）

**物理參數**（回填至 `physics_params`）：
- `nu`（運動黏度）
- `k_f`（強迫波數）
- `forcing_amplitude`（強迫振幅）

##### 三層優先級系統

物理模塊創建時，參數來源優先級為：**DNS NPY > YAML 配置 > 預設值**

```python
# 參數讀取流程
config['data']['kolmogorov_config']['physics_params']  # ← DNS NPY 自動回填
    ↓
create_physics() 讀取順序：
    ① DNS NPY（最高優先，來自 physics_params）
    ② YAML config（次優先，來自 physics.*）
    ③ Default values（最低優先，程式內建）
```

##### 簡化後的配置格式（v2.0+）

**✅ 推薦配置**（最小化，單一來源）：
```yaml
data:
  kolmogorov_config:
    enabled: true
    data_path: ./data/kolmogorov_dns/kolmogorov_dns_100.npy
    time_range: [15.0, 35.0]  # 訓練時間窗（必填）
    description: "Kolmogorov flow (Re=100, k_f=4) [物理參數由 DNS NPY 自動同步]"
    # ✂️ 以下參數已移除（DNS NPY 自動提供）：
    #   - physics_params（整個字典）
    #   - resolution
    #   - dt
    #   - L

physics:
  type: kolmogorov_flow_2d
  nu: 0.01    # ⚠️ 可選：若省略則從 DNS NPY 讀取
  rho: 1.0    # 密度（DNS NPY 不提供，需 YAML 配置）
  forcing: {} # ✂️ 空字典（DNS NPY 自動同步 k_f 和 amplitude）
  # ✂️ 已移除的冗餘區段：
  #   - kolmogorov_flow（重複配置）
  domain:
    x_range: [0.0, 6.283185307179586]  # 終點從 DNS NPY L 自動推導
    y_range: [0.0, 6.283185307179586]
  boundary_conditions:
    periodic_x: true
    periodic_y: true
```

**🔍 參數來源驗證**（訓練日誌會顯示）：
```
✅ 使用 Kolmogorov Flow 2D 求解器
   強迫參數: A=0.100 (DNS NPY), k_f=4 (DNS NPY)
   物理參數: ν=1.00e-02 (DNS NPY), ρ=1.0 (YAML)
   域範圍: L=6.2832 (DNS NPY)
   🔁 已從 DNS NPY 自動同步參數: amplitude, k_f, nu, L
```

##### 批量簡化工具

使用以下腳本批量簡化舊版配置文件：

```bash
# 預覽修改（試運行）
python scripts/tools/simplify_kolmogorov_configs.py --dry-run

# 執行簡化（帶備份）
python scripts/tools/simplify_kolmogorov_configs.py --backup

# 只處理特定目錄
python scripts/tools/simplify_kolmogorov_configs.py --dir configs/experiments
```

##### 向後相容性

- **舊配置仍可使用**：若 YAML 中仍有 `physics_params`，系統會優先使用 DNS NPY 值並發出提示
- **無縫遷移**：使用批量簡化工具可自動移除冗餘參數
- **參數衝突警告**：若 DNS NPY 與 YAML 參數不一致，系統會發出警告

##### 時間範圍配置

⚠️ **唯一必須在 YAML 手動配置的參數**：

```yaml
data:
  kolmogorov_config:
    time_range: [15.0, 35.0]  # ✅ 訓練時間窗（必填，DNS NPY 不控制）
```

原因：DNS NPY 包含完整時間序列，但訓練時需指定使用哪個時間段。

#### DDP Multi-GPU 訓練

建議使用 `torchrun` 啟動，系統會自動偵測並啟用 DDP：

```bash
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/your_config.yml
```

DDP 配置範例：

```yaml
training:
  ddp:
    enabled: null         # null=自動偵測, true=強制啟用, false=禁用
    split_data: true      # 分割訓練資料
    reduce_losses: true   # 同步損失供監控
```

#### logging（日誌頻率）

降低 `.item()` 同步造成的 CPU/GPU 等待：

```yaml
logging:
  log_freq: 10            # 基本日誌頻率
  loss_log_interval: 200  # loss 詳細記錄頻率（降低同步）
  wandb_sync_interval: 50 # WandB 同步頻率（降低同步）
```

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

## 已移除的配置項

以下配置項已從最新版本中移除，不再支援：

### 1. 混合精度訓練 (AMP)
```yaml
training:
  amp:              # ❌ 已移除（v1.4.0）
    enabled: false
```
**移除原因**: 未被實際使用，且增加配置複雜度。如需混合精度訓練，請使用 PyTorch 原生的 `torch.cuda.amp` 功能。

### 2. DataLoader 性能參數
```yaml
reproducibility:
  num_workers: 8           # ❌ 已移除（v1.4.0）
  pin_memory: true         # ❌ 已移除（v1.4.0）
  persistent_workers: true # ❌ 已移除（v1.4.0）
  prefetch_factor: 2       # ❌ 已移除（v1.4.0）
```
**移除原因**: 本專案未使用 PyTorch DataLoader，這些參數無實際作用。

### 3. 已棄用的 GradNorm 參數
```yaml
losses:
  grad_norm_normalize: false  # ❌ 已移除（v1.4.0）
```
**移除原因**: 該參數在 JaxPI 實現中未使用，保留會造成混淆。

### 4. 未實現的資料增強功能
```yaml
normalization:
  noise_sigma: 0.0      # ❌ 已移除（v1.4.0）
  dropout_prob: 0.0     # ❌ 已移除（v1.4.0）
```
**移除原因**: 僅在註解中提及但從未實現，保留會誤導用戶以為功能可用。如需資料增強，請在資料載入器中自行實現。

### 5. 未實現的權重追蹤功能
```yaml
logging:
  save_weight_evolution: false  # ❌ 已移除（v1.4.0）
```
**移除原因**: 配置項存在但功能完全未實現。如需追蹤權重演化，請使用 WandB 或自定義 hook。

### 6. 冗余的變量順序配置
```yaml
normalization:
  variable_order: [u, v, w, p]  # ❌ 已移除（v1.5.0）
```
**移除原因**: 變量順序現在自動從 `model.output_variables` 推斷，無需重複配置。配置時將被忽略。

**自動推斷優先級**：
1. `model.output_variables`（推薦）
2. `data.kolmogorov_config.variables`（Kolmogorov Flow）
3. 根據 `physics.type` 自動推斷（2D: [u, v, p], 3D: [u, v, w, p])

**遷移方式**：運行 `python scripts/tools/cleanup_redundant_configs.py` 自動移除所有冗余配置

## 配置完整參考

**主文件**: [`configs/standard_config_template.yml`](../configs/standard_config_template.yml)

該文件包含：
- 所有可用配置鍵與預設值
- 類型說明與範例

**不要直接修改該文件**，將它作為參考手冊使用。

---

**最後更新**: 2026-01-15
**版本**: v1.5.0（配置精簡：移除未使用/未實現的配置項 + 自動計算 Channel Flow 參數 + 自動推斷變量順序）
