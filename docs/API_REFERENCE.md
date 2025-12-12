# API 參考手冊

## 核心腳本

### 訓練
```bash
# 基本訓練
python scripts/train.py --cfg <config.yml> [--device cuda]

# 從檢查點恢復
python scripts/train.py --cfg <config.yml> --resume <checkpoint.pth>

# Ensemble 訓練
python scripts/train.py --cfg <config.yml> --ensemble
```

### 評估
```bash
# 快速評估
python scripts/evaluate_checkpoint.py \
  --checkpoint <path> --config <config.yml>

# 完整評估
python scripts/evaluate.py \
  --checkpoint <path> --config <config.yml>

# 課程學習評估
python scripts/evaluate_curriculum.py \
  --exp-dir <dir> --stages <n>
```

### 視覺化
```bash
# 訓練結果
python scripts/visualize_results.py \
  --checkpoint <path> --output <dir>

# 感測點品質
python scripts/visualize_qr_sensors.py \
  --input <sensors.npz> --output <dir>

# DNS 數據
python scripts/visualize_kolmogorov_dns.py \
  --input <dns.h5> --output <dir>
```

---

## DNS 生成

### Kolmogorov Flow
```bash
python scripts/generate_kolmogorov_dns.py \
  --Re <value> \
  --k_f <value> \
  --nu <value> \
  --T_max <value> \
  --dt <value> \
  --resolution <value> \
  --output <path>
```

**參數**：
- `--Re`: 雷諾數（30-100）
- `--k_f`: 強迫波數（4 或 8）
- `--nu`: 動力黏度（由 Re 計算）
- `--T_max`: 模擬時間（建議 ≥100）
- `--dt`: 時間步長（建議 0.05）
- `--resolution`: 空間解析度（512 或 1024）

### RANS 先驗
```bash
python scripts/generate_kolmogorov_rans.py \
  --Re <value> \
  --k_f <value> \
  --nu <value> \
  --T_avg_start <value> \
  --T_avg_end <value> \
  --output <path>
```

---

## 感測點生成

### V7 方法（推薦）
```bash
python scripts/generate_sensors_periodic_qr.py \
  --dns-path <dns.h5> \
  --K <n_sensors> \
  --output <sensors.npz> \
  [--t-start <value>] \
  [--t-end <value>] \
  [--oversample-factor <value>] \
  [--seam-weight <value>] \
  [--n-wrap-layers <value>]
```

**參數**：
- `--K`: 感測點數量（50-200）
- `--oversample-factor`: 過採樣倍數（推薦 3.0）
- `--seam-weight`: 邊界權重（推薦 1.0）
- `--n-wrap-layers`: 包裹層數（推薦 2）

### Channel Flow
```bash
python scripts/generate_channel_flow_sensors_qr.py \
  --input <jhtdb.h5> \
  --K <n_sensors> \
  --output <sensors.npz> \
  [--y-layers <value>]
```

---

## 驗證工具

### 雷諾數計算
```bash
# 驗證現有數據
python scripts/calculate_reynolds_parameters.py \
  --f0 <value> --nu <value> --k <value>

# 規劃新 DNS（求解 ν）
python scripts/calculate_reynolds_parameters.py \
  --target-Re <value> --f0 <value> --k <value> --solve-nu

# 批量掃描
python scripts/calculate_reynolds_parameters.py \
  --f0 <value> --k <value> --nu-range <start> <end> <step>
```

### DNS 驗證
```bash
# 物理守恆驗證
python scripts/validate_dns_physics.py --input <dns.h5>

# 能譜驗證
python scripts/validate_2d_turbulence_spectrum.py \
  --checkpoint <path> --reference <dns.h5>

# 解析度驗證
python scripts/validate_dns_resolution.py --input <dns.h5>
```

---

## 監控工具

### 訓練監控
```bash
# 腳本監控
python scripts/monitor_training_speed.py \
  --log-file <training.log>

# 直接查看
tail -f log/<exp>/training.log

# TensorBoard
tensorboard --logdir log/<exp>/tensorboard/
```

### DNS 生成監控
```bash
# 狀態檢查
python scripts/monitor_re50_training.sh

# 查看日誌
tail -f log/dns_generation.log
```

---

## 診斷工具

### 訓練失敗診斷
```bash
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint <path> --config <config.yml> --output <dir>
```

### 物理驗證
```bash
# NS 方程
python scripts/debug/diagnose_ns_equations.py \
  --checkpoint <path> --config <config.yml>

# 邊界條件
python scripts/debug/diagnose_boundary_conditions.py \
  --checkpoint <path>

# 壓力場
python scripts/debug/diagnose_pressure_failure.py \
  --checkpoint <path>
```

---

## 文件格式

### Sensors NPZ
```python
data = np.load("sensors.npz")
# 必要欄位
coords = data['coords']      # (K, 2) or (K, 3)
values = data['values']      # (K, n_vars)

# 可選欄位
condition_number = data['condition_number']  # float
energy_ratio = data['energy_ratio']          # float
unique_x_pct = data['unique_x_pct']          # float
```

### DNS HDF5
```python
import h5py
with h5py.File("dns.h5", "r") as f:
    u = f['u'][:]      # (nt, nx, ny)
    v = f['v'][:]
    p = f['p'][:]
    t = f['t'][:]      # (nt,)
    x = f['x'][:]      # (nx,)
    y = f['y'][:]      # (ny,)
```

### RANS HDF5
```python
with h5py.File("rans.h5", "r") as f:
    u_mean = f['u_mean'][:]      # (nx, ny)
    v_mean = f['v_mean'][:]
    p_mean = f['p_mean'][:]
    uu = f['uu'][:]              # Reynolds stress
    vv = f['vv'][:]
    uv = f['uv'][:]
```

---

## 配置模板

**位置**: `configs/templates/`

| 模板 | 用途 | 時間 |
|------|------|------|
| 2d_quick_baseline.yml | 快速驗證 | 5-10 min |
| 2d_medium_ablation.yml | 特徵消融 | 15-30 min |
| 3d_slab_curriculum.yml | 課程學習 | 30-60 min |
| 3d_full_production.yml | 論文級結果 | 2-8 hrs |

---

## 常用組合

### 完整訓練流程
```bash
# 1. 計算參數
python scripts/calculate_reynolds_parameters.py --target-Re 50 --solve-nu

# 2. 生成 DNS
python scripts/generate_kolmogorov_dns.py --Re 50 --k_f 4 --nu 0.0125

# 3. 驗證 DNS
python scripts/validate_dns_physics.py --input data/dns.h5

# 4. 生成感測點
python scripts/generate_sensors_periodic_qr.py --dns-path data/dns.h5 --K 100

# 5. 訓練
python scripts/train.py --cfg configs/your_config.yml

# 6. 評估
python scripts/evaluate.py --checkpoint checkpoints/best.pth

# 7. 視覺化
python scripts/visualize_results.py --checkpoint checkpoints/best.pth
```

### 快速測試
```bash
python scripts/train.py --cfg configs/quick_test_rans_prior.yml
```

### 除錯流程
```bash
# 1. 總體診斷
python scripts/debug/diagnose_piratenet_failure.py --checkpoint <path>

# 2. 感測點檢查
python scripts/visualize_qr_sensors.py --input data/sensors.npz

# 3. 物理驗證
python scripts/validation/physics_validation.py --checkpoint <path>
```
