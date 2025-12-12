# 配置參考手冊

## 配置結構

```yaml
experiment:
  name: "your_exp"
  device: "cuda"

model:
  type: "FourierMLP"
  hidden_dims: [128, 128, 128, 128]
  activation: "sine"
  fourier:
    n_frequencies: 8
    omega_0: 30.0

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

data:
  dns:
    data_path: "./data/kolmogorov_dns/dns_re50_t100.h5"
    sensors_path: "./data/kolmogorov_dns/sensors_K100_v7.npz"
  lowfi_prior:
    use: true
    path: "./data/kolmogorov_dns/rans_re50_kf4.h5"

losses:
  data:
    weight: 1.0
  pde:
    weight: 0.01
    normalization: "adaptive"
  prior_consistency:
    weight: 0.1
  normalization:
    method: "sum_to_one"

training:
  epochs: 1000
  batch_size: 1024
  n_collocation: 20000
  gradient_clip: 1.0
```

## 關鍵參數說明

### 模型架構
- `hidden_dims`: [128, 128, 128, 128] 適用於 Re < 100
- `activation`: "sine"（SIREN）對高階導數敏感
- `n_frequencies`: 8 適用於週期問題

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
data: 1.0          # 基準
pde: 0.01-0.1      # 逐步增強
prior: 0.1-0.5     # 軟約束
```

**權重調整原則**：
1. 確保 `sum_to_one` 權重守恆
2. data loss 優先收斂
3. prior weight < 0.5 避免過度依賴

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
  Re: 50
  k_f: 4            # 強迫波數
  nu: 0.0125        # 動力黏度
  f0: 1.0           # 強迫振幅
  domain: [0, 6.283185307179586]  # [0, 2π]
```

**雷諾數定義**：
```
Re = √f₀ × L^(3/2) / ν
L = 2π/k_f
```

### 低保真先驗策略
| 策略 | 適用場景 | prior_weight |
|------|----------|--------------|
| 無先驗 | 數據充足（K>200） | 0 |
| RANS 先驗 | 推薦（K=50-100） | 0.1-0.5 |
| 課程學習 | 高 Re（Re>100） | 動態調整 |

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
