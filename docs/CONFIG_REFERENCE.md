# 配置參考手冊

## 配置結構

```yaml
experiment:
  name: "your_exp"
  device: "cuda"

model:
  type: "fourier_vs_mlp"
  in_dim: 3
  out_dim: 4
  width: 256
  depth: 8
  activation: "sine"
  fourier_features:
    type: "standard"
    fourier_m: 32
    fourier_sigma: 5.0
    trainable_fourier: false
    fourier_use_2pi: true

training:
  optimizer:
    type: "SOAP"
    lr: 0.001
    lbfgs_switch:
      enabled: true
      switch_epoch: 800

  lr_scheduler:
    type: "CosineAnnealingWarmRestarts"
    T_0: 200
    T_mult: 2
    eta_min: 1.0e-6
  epochs: 1000
  batch_size: 1024
  n_collocation: 20000
  gradient_clip: 1.0
  enforce_pressure_data: true  # 壓力驅動流是否強制要求壓力場資料（新增）

data:
  dns:
    data_path: "./data/kolmogorov_dns/dns_re50_t100.h5"
    sensors_path: "./data/kolmogorov_dns/sensors_K100_v7.npz"
  lowfi_prior:
    use: true
    path: "./data/kolmogorov_dns/rans_re50_kf4.h5"

losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 1.0
  wall_constraint_weight: 10.0
  prior_weight: 0.1
```

## 關鍵參數說明

### 模型架構
- `width` / `depth`: 256×8 適用於高維度流場
- `activation`: "sine"（SIREN）對高階導數敏感
- `fourier_m`: 32–128 適用於週期問題

### 優化器
- **SOAP**: 前期穩定收斂
- **L-BFGS**: 後期精細調整（switch_epoch: 800）

### 學習率調度
| 類型 | 適用場景 | 關鍵參數 |
|------|----------|----------|
| CosineAnnealingWarmRestarts | 推薦 | T_0=200, T_mult=2 |
| StepLR | 簡單場景 | step_size=200, gamma=0.5 |
| ReduceLROnPlateau | 自適應 | patience=50, factor=0.5 |

### 損失權重
```yaml
# 推薦配置（Re=50-100）
losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 1.0
  wall_constraint_weight: 10.0
  prior_weight: 0.1
```

**權重調整原則**：
1. data loss 優先收斂
2. prior_weight < 0.5 避免過度依賴
3. 牆面約束過小會導致壁面剪應力偏差

### 標準化方法
| 方法 | 輸入 | 輸出 |
|------|------|------|
| none | 原始座標 | 原始物理量 |
| minmax | [0,1] | [0,1] |
| zscore | 標準化 | 標準化 |
| domain | [0,2π]→[-1,1] | 保留物理量 |

### 課程學習
```yaml
curriculum:
  stages:
    - Re: 30
      epochs: 300
    - Re: 50
      epochs: 300
      init_from_previous: true
    - Re: 100
      epochs: 400
      init_from_previous: true
```

### Kolmogorov Flow 物理參數
```yaml
physics:
  type: "kolmogorov_flow_2d"
  nu: 0.0125        # 動力黏度
  rho: 1.0
  forcing:
    k_f: 4          # 強迫波數
    amplitude: 1.0  # 強迫振幅
  domain:
    x_range: [0.0, 6.283185307179586]  # [0, 2π]
    y_range: [0.0, 6.283185307179586]
```

**雷諾數定義**：
```
Re = √f₀ × L^(3/2) / ν
L = 2π/k_f
```

### Channel Flow 物理參數
```yaml
physics:
  type: "vs_pinn_channel_flow"
  nu: 1.0e-3        # 運動黏度
  rho: 1.0
  pressure_driven: true  # ⚠️ 必須設為 true（壓力梯度驅動）
  domain_bounds:
    x: [0.0, 12.56] # 流向（2π）
    y: [-1.0, 1.0] # 壁面法向
    z: [0.0, 6.28] # 展向（π）
```

**⚠️ 新增配置（v2025-12-17）**：
- `pressure_driven`: 聲明流體是否由壓力梯度驅動
  - `true`: Channel Flow, Pipe Flow 等
  - `false`: Lid-Driven Cavity, Kolmogorov Flow 等
- 用途：控制壓力場資料缺失時的處理策略（參見下文 `training.enforce_pressure_data`）

### 低保真先驗策略
| 策略 | 適用場景 | prior_weight |
|------|----------|--------------|
| 無先驗 | 數據充足（K>200） | 0 |
| RANS 先驗 | 推薦（K=50-100） | 0.1-0.5 |
| 課程學習 | 高 Re（Re>100） | 動態調整 |

### 訓練行為控制（新增 v2025-12-17）

#### `training.enforce_pressure_data`
控制壓力場資料缺失時的行為。

```yaml
training:
  enforce_pressure_data: true   # 預設：與 physics.pressure_driven 相同
```

| 值 | 行為 | 適用場景 |
|----|------|----------|
| `true` (嚴格模式) | 壓力驅動流缺少壓力資料時拋出 `ValueError` | 生產環境（推薦） |
| `false` (允許模式) | 缺少壓力資料時發出警告並初始化為零 | 除錯或特殊情況 |
| 未設定 | 自動跟隨 `physics.pressure_driven` | 通用配置 |

**範例**：

```yaml
# 情況 1: Channel Flow（壓力驅動）- 嚴格模式
physics:
  pressure_driven: true
training:
  enforce_pressure_data: true  # 缺少壓力資料將報錯

# 情況 2: Kolmogorov Flow（體積力驅動）- 寬鬆模式
physics:
  pressure_driven: false
# enforce_pressure_data 預設為 false，缺少壓力資料僅記錄 Info 日誌

# 情況 3: Channel Flow 但允許無壓力資料（不推薦）
physics:
  pressure_driven: true
training:
  enforce_pressure_data: false  # 覆蓋預設，允許無壓力資料
```

**錯誤訊息範例**：
```
ValueError: ❌ 壓力場資料缺失！

當前配置：
  physics.pressure_driven = True
  training.enforce_pressure_data = True

檢測到問題：
  sensor NPZ 檔案中缺少 'p' (壓力) 欄位

對於壓力驅動流（Channel Flow），壓力場是必需的輸入資料。

解決方案（三選一）：
  1. 重新生成 sensor data 並確保包含壓力場
     → python scripts/generate/sensors/xxx.py --include-pressure
  
  2. 檢查 NPZ 檔案是否正確載入
     → data['sensor_data'] 應包含 'u', 'v', 'w', 'p' 鍵
  
  3. 若確實無壓力資料且理解風險，設定 config:
     training:
       enforce_pressure_data: false
```

## 常用配置模板

### 快速驗證（5-10 min）
```bash
python scripts/train.py --cfg configs/templates/2d_quick_baseline.yml
```

### 論文級結果（2-8 hrs）
```bash
python scripts/train.py --cfg configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml
```

### Ensemble（不確定性量化）
```bash
python scripts/train.py --cfg configs/your_config.yml --ensemble
```
