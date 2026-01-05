# PINNs 湍流逆重建技術文檔

**文檔版本**: v2.3  
**更新日期**: 2026-01-05  
**狀態**: 持續更新

---

## 目錄

1. [專案概述](#1-專案概述)
2. [核心技術模組](#2-核心技術模組)
3. [系統架構](#3-系統架構)
4. [驗證與指標](#4-驗證與指標)
5. [使用指南](#5-使用指南)
6. [已知限制](#6-已知限制)
7. [參考文獻](#7-參考文獻)

---

## 1. 專案概述

### 1.1 研究目標

本專案以 **稀疏感測資料**（K ≤ 100）結合 **物理先驗** 進行湍流場重建，面向逆問題場景：

- **研究驗證**：使用 DNS 生成感測點觀測值，並以 DNS 全場作為對照基準
- **工程類比**：用 RANS/LES 作為先驗，真實量測作為 sensor observations

驗收指標（維持原定義）：
- 流場誤差 ≤ 10–15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

### 1.2 技術路線（現行流程）

```
DNS / Low-Fi Prior
        ↓
Sensor Selection (QR-Pivot / Hybrid)
        ↓
Fourier-VS MLP + VS-PINN
        ↓
GradNorm / Causal Weighting / Curriculum
        ↓
Reconstruction + Physics Validation
```

### 1.3 目前系統狀態（以程式碼為準）

| 模組 | 狀態 | 說明 |
|------|------|------|
| Sensor Selection | ✅ 可用 | QR-Pivot + Hybrid/Stratified |
| Fourier-VS MLP | ✅ 可用 | axis-selective Fourier + SIREN + 可選 RWF |
| VS-PINN Channel Flow | ✅ 可用 | N=(2,12,2) scaling factors |
| 2D Kolmogorov Physics | ✅ 可用 | 正弦強迫項 + 週期邊界 |
| Loss Weighters | ✅ 可用 | GradNorm + Causal + Staged/Curriculum |
| TrainerBuilder | ✅ 可用 | 組件化訓練器建構 |
| Checkpoint/Validation Manager | ✅ 可用 | 依賴注入 + 策略模式 |

---

## 2. 核心技術模組

### 2.1 感測器選擇（QR-Pivot）

使用 QR 分解主元置換選擇資訊量最大的空間點：

```
X^T Π = QR
```

**檔案位置**: `pinnx/sensors/qr_pivot.py`

```python
from pinnx.sensors import create_sensor_selector

selector = create_sensor_selector(strategy='qr_pivot')
indices, metrics = selector.select_sensors(data_matrix, n_sensors=100)
```

支援策略：QR-Pivot、POD-based、Greedy、Hybrid/Stratified（見 `pinnx/sensors/`）。

---

### 2.2 Fourier-VS MLP + RWF

核心模型以 Fourier Features 提升高頻表達，並支援 RWF（Random Weight Factorization）：

```yaml
model:
  type: fourier_vs_mlp
  width: 256
  depth: 8
  activation: sine
  use_rwf: true
  rwf_scale_std: 0.1
  fourier_features:
    type: axis_selective
    fourier_m: 12
    fourier_sigma: 4.0
```

- **axis_selective** 可針對特定軸啟用頻率
- `fourier_features.type=disabled` 可關閉 Fourier Features

---

### 2.3 VS-PINN 變數尺度化

VS-PINN 為剛性方程提供變數尺度化，提升數值穩定性：

```yaml
physics:
  type: vs_pinn_channel_flow
  vs_pinn:
    scaling_factors:
      N_x: 2.0
      N_y: 12.0
      N_z: 2.0
```

**檔案位置**: `pinnx/physics/vs_pinn_channel_flow.py`

---

### 2.4 動態權重平衡（GradNorm / Causal）

GradNorm 以梯度範數自動平衡多損失項：

```yaml
losses:
  adaptive_weighting: true
  weight_update_freq: 1000
  grad_norm_momentum: 0.9
  grad_norm_alpha: 1.5
  grad_norm_normalize: true
```

Causal Weighting 用於時間依賴問題：

```yaml
losses:
  causal_weighting: true
  causal_tol: 1.0
  num_chunks: 16
```

---

### 2.5 Adaptive Collocation

依 PDE 殘差重採樣碰撞點：

```yaml
training:
  sampling:
    adaptive_collocation:
      enabled: true
      trigger:
        method: epoch_interval
        epoch_interval: 1000
      resampling_strategy: incremental_replace
```

**檔案位置**: `pinnx/train/adaptive_collocation.py`

---

### 2.6 低保真先驗（Low-Fi Prior）

支援 RANS/低保真資料作為軟約束：

```yaml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/channel_rans/rans_k_omega_sst.npz
  data_type: rans
  consistency_weight: 0.3
```

並支援空間權重（如壁面加權）與插值品質檢查。

---

### 2.7 訓練器架構（TrainerBuilder）

訓練流程透過 **TrainerBuilder** 組件化構建：

- `TrainerComponents`：封裝 model/physics/optimizer/scheduler 等組件
- `CheckpointManager`：統一 checkpoint 存取策略
- `ValidationManager`：策略式驗證（Data/Physics）

入口：`pinnx/train/trainer_builder.py`

---

## 3. 系統架構

### 3.1 檔案結構（核心路徑）

```
pinns-sparse-flow/
├── pinnx/
│   ├── models/                # Fourier-VS MLP, ResNet/PirateNet
│   ├── physics/               # VS-PINN, Kolmogorov 2D, NS 2D
│   ├── sensors/               # QR-Pivot / Hybrid / POD
│   ├── losses/                # GradNorm / Causal / Priors
│   ├── train/                 # TrainerBuilder / Trainer / Managers
│   ├── evals/                 # 指標與視覺化
│   └── optim/                 # SOAP optimizer
├── scripts/
│   ├── train/train.py          # 主訓練入口
│   ├── train_time_window.py    # Time Window 訓練
│   ├── evaluate/               # 評估腳本
│   └── tools/                  # validate_config / add_output_variables
├── configs/                    # 實驗配置與模板
└── tests/                      # 單元測試
```

### 3.2 訓練流程（實際路徑）

```
1. Config validation
   - validate_config_keys.py
   - validate_config.py
2. Data preparation
   - DNS / JHTDB / lowfi_prior
   - Sensor selection
3. Model + Physics creation
   - Registry factories
4. TrainerBuilder
   - CheckpointManager + ValidationManager
5. Training loop
   - loss → weight → backward → optimizer
6. Evaluation
   - relative L2, mass conservation, wall shear
```

---

## 4. 驗證與指標

### 4.1 驗證層級

- **P0**: Import/單元測試（`tests/`）
- **P1**: 物理正確性（守恆、邊界條件）
- **P2**: 端到端訓練與視覺化

### 4.2 常用指標

- 相對 L2
- 質量守恆誤差
- 壁面剪應力 / 速度剖面
- 能譜 / 壓力梯度（依實驗）

### 4.3 評估入口

```bash
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/<exp>/best_model.pth \
  --config configs/<exp>.yml
```

---

## 5. 使用指南

### 5.1 快速開始

```bash
# 1. 環境設置
conda env create -f environment.yml
conda activate pinns-sparse-flow

# 2. 配置驗證（必跑）
python scripts/tools/validate_config_keys.py configs/kolmogorov_re50_kf4_K100.yml
python scripts/tools/validate_config.py --config configs/kolmogorov_re50_kf4_K100.yml

# 3. 訓練
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml
```

### 5.2 Time Window 訓練

```bash
python scripts/train_time_window.py --cfg configs/experiments/time_window_kolmogorov.yml
```

---

## 6. 已知限制

1. **3D 訓練記憶體壓力高**：需調整 `pde_points` 與 batch size
2. **梯度檢查點**：高階導數下可能觸發不穩定，預設關閉
3. **Sensor 品質敏感**：感測資料異常時會觸發 Fail Fast
4. **JHTDB 依賴**：需有效 token + 穩定網路

---

## 7. 參考文獻

- Raissi et al., Physics-Informed Neural Networks (2019)
- Wang et al., Causality for PINNs (2022)
- JHTDB (Johns Hopkins Turbulence DB)
