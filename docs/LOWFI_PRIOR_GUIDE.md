# Low-Fidelity Prior 使用指南

> **⚠️ 專案範圍聲明**  
> 本專案**僅支援**以下兩種 low-fidelity prior 配置。  
> 其他模型（LES, 下採樣DNS, NetCDF格式等）的程式碼保留但**不提供支援**。

## 📋 支援的 Prior 模型

### 1. 2D Kolmogorov Flow → Leith Model ✅
- **適用場景**: 2D 週期性湍流（Kolmogorov Flow）
- **Prior 來源**: Leith turbulence model
- **資料格式**: HDF5 (.h5), 1D 座標 `x[N], y[N]`
- **變數**: `u, v, nu_t`（**無壓力場**）
- **特點**: 診斷模型，適合 2D 逆能量級串
- **配置範例**: `configs/kolmogorov_re50_kf4_K100.yml`

### 2. 3D Channel Flow → RANS k-ε ✅
- **適用場景**: 3D 通道流（Re_tau = 1000）
- **Prior 來源**: RANS k-ε turbulence model
- **資料格式**: HDF5 (.h5), 2D meshgrid `X[N,M], Y[N,M]` 或 1D `x[N], y[M]`
- **變數**: `u, v, w, p, k, epsilon, nu_t`（完整場）
- **特點**: 求解 k-ε 輸運方程，包含壓力場
- **配置範例**: `configs/channel_flow_re1000.yml`

### ❌ 不支援的模型（已移除）
- **LES (Large Eddy Simulation)** - ❌ 已刪除
- **下採樣 DNS** - ❌ 已刪除
- **NetCDF 格式** - ❌ 已刪除，僅支援 HDF5 (.h5)

---

## 🔧 配置範例

### 2D Kolmogorov + Leith

```yaml
# configs/kolmogorov_re50_kf4_K100.yml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5
  data_type: leith
  
  rans_structure:
    group_path: /mean_field
    field_mapping:
      u: u
      v: v
      nu_t: nu_t
    coord_mapping:
      x: x  # 1D座標
      y: y
  
  consistency_weight: 2.0  # Leith 品質高，權重適中
  variable_weights:
    u: 1.0
    v: 1.0
    p: 0.0  # 無壓力，不使用
```

### 3D Channel + RANS

```yaml
# configs/channel_flow_re1000.yml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/channel_rans/rans_retau1000.h5
  data_type: rans
  
  rans_structure:
    group_path: /mean_field
    field_mapping:
      u: u
      v: v
      w: w
      p: p
      k: k
      epsilon: epsilon
      nu_t: nu_t
    coord_mapping:
      x: x  # 可為1D或2D
      y: y
      z: z  # 3D專用
  
  consistency_weight: 5.0  # RANS 誤差較大，權重較高
  variable_weights:
    u: 1.0
    v: 1.0
    w: 1.0
    p: 0.5  # 壓力權重較低（RANS 壓力不準）
```

---

## 🏗️ 代碼架構決策

### 自動偵測邏輯

`scripts/train/train.py::load_rans_prior_data()` 會自動處理：

```python
# 1. 座標格式偵測（1D vs 2D vs 3D）
if 'x' in group:
    x_1d = group['x'][:]  # 直接讀取1D
elif 'X' in group:
    X_2d = group['X'][:]  # 轉換2D → 1D

# 2. 壓力有效性偵測
p_valid = False
if 'p' in group:
    p = group['p'][:]
    if not np.all(np.abs(p) < 1e-10):
        p_valid = True  # RANS: True, Leith: False

# 3. 返回 metadata
return {
    'u_pde': ...,
    'metadata': {
        'pressure_valid': p_valid,
        'model_type': 'leith' or 'rans'
    }
}
```

### Loss 計算邏輯

`pinnx/train/loss_manager.py::compute_prior_consistency_loss()`:

```python
# 根據 pressure_valid 動態選擇變數
if metadata['pressure_valid']:
    # RANS: 包含壓力
    variable_names = ['u', 'v', 'w', 'p']  # 3D
    # 或
    variable_names = ['u', 'v', 'p']        # 2D
else:
    # Leith: 跳過壓力
    variable_names = ['u', 'v']             # 2D only
```

---

## 📊 Prior Weight 建議

| 模型 | 典型誤差 | Stage 1 | Stage 2 | Stage 3 |
|------|---------|---------|---------|---------|
| **Leith** | ~15-20% | 2.0-3.0 | 1.0 | 0.1 |
| **RANS** | ~30-50% | 5.0-10.0 | 2.0 | 0.5 |

**原則**:
- Prior 品質越高 → 權重越低（避免過度約束）
- 訓練後期逐步降低權重（讓 PINN 主導）
- 目標：Prior loss 佔總 loss 的 20-30%

---

## ⚠️ 常見陷阱

### ❌ 錯誤示範 1: 混用模型
```yaml
# Kolmogorov Flow 使用 RANS（錯誤！）
lowfi_prior:
  data_path: ./data/lowfi/kolmogorov_rans/rans_re50_kf4.h5  # ❌ k-ε RANS
```
**問題**: RANS 不適合 2D 湍流（無法捕捉逆級串）  
**修正**: 使用 Leith 模型

### ❌ 錯誤示範 2: Prior 權重過高
```yaml
# Leith 使用過高權重
lowfi_prior:
  consistency_weight: 10.0  # ❌ 太高！
```
**問題**: Prior 主導訓練，PINN 無法學習 DNS 特徵  
**修正**: Leith 使用 2.0-3.0

### ❌ 錯誤示範 3: 強制使用壓力
```yaml
# Leith 配置中包含壓力權重
variable_weights:
  p: 0.5  # ❌ Leith 無壓力場！
```
**問題**: 雖然 `p: 0.5` 不會被使用，但造成困惑  
**修正**: 明確設為 `p: 0.0` 或移除

---

## 🧪 驗證 Checklist

新增 Prior 配置後，執行以下檢查：

### 1. 資料載入測試
```python
python3 << 'EOF'
from scripts.train.train import load_rans_prior_data
import yaml, torch

config = yaml.safe_load(open('configs/your_config.yml'))
training_data = {
    'x_pde': torch.rand(1000, 1) * 6.28,
    'y_pde': torch.rand(1000, 1) * 6.28,
    'x_sensors': torch.rand(100, 1) * 6.28,
    'y_sensors': torch.rand(100, 1) * 6.28,
}

prior = load_rans_prior_data(config, training_data, torch.device('cpu'))

# 檢查
assert 'metadata' in prior
assert prior['metadata']['pressure_valid'] in [True, False]
print(f"✅ Pressure valid: {prior['metadata']['pressure_valid']}")
print(f"✅ Model type: {prior['metadata']['model_type']}")
EOF
```

### 2. Loss 計算測試
```bash
# 執行短期訓練（10 epochs）
python scripts/train/train.py \
    --config configs/your_config.yml \
    --epochs 10

# 檢查 log 輸出
# ✅ 應看到「先驗一致性損失」
# ✅ Leith: 僅 u, v 損失
# ✅ RANS: u, v, (w), p 損失
```

### 3. 外插檢查
```bash
# 檢查訓練 log
grep "外插區域" logs/your_experiment.log

# ✅ 外插比例應 <5%
# ❌ 若 >5%，檢查座標範圍是否對齊
```

---

## 📚 相關文件

- **理論背景**: `docs/LEITH_PARAMETER_SELECTION_CRITERIA.md`
- **修復報告**: `tasks/task-001-leith-prior-audit/fix_summary.md`
- **測試腳本**: `tasks/task-001-leith-prior-audit/test_leith_fixes.py`
- **Leith 求解器**: `scripts/generate/dns/generate_kolmogorov_leith.py`

---

## 🔮 未來擴展（如需）

若未來需支援其他場景，需考慮：

1. **新湍流模型**: Smagorinsky, Dynamic Smagorinsky
2. **新幾何**: Cavity flow, Backward-facing step
3. **瞬態 Prior**: 時間相關的 RANS/LES

**但目前專案範圍僅限**:
- ✅ 2D Kolmogorov + Leith
- ✅ 3D Channel + RANS

---

**版本**: 1.0  
**更新日期**: 2025-12-17  
**維護者**: PINNs-MVP Team
