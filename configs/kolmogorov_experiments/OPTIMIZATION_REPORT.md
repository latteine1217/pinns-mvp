# Kolmogorov Flow 配置優化報告

**生成時間**: 2025-11-21  
**分析對象**: `configs/kolmogorov_experiments/` 中的 8 個配置檔案  
**目標**: 配置一致性檢查、最佳化建議、與當前 DNS 模擬對接

---

## 📊 配置一致性分析

### 1. **配置結構規範檢查**

| 檔案 | 結構 | Loss鍵 | Sensors | Physics | 備註 |
|------|------|---------|---------|---------|------|
| `baseline.yml` | ✅ | `losses` | ✅ K=50 | ✅ Re~62.8 | 標準逆問題 |
| `chaos_re30.yml` | ⚠️ | `loss` | ⚠️ 無sensors段 | ✅ Re=30 | **需修正** |
| `chaos_re30_full.yml` | ⚠️ | `loss` | ⚠️ 無sensors段 | ✅ Re=30 | **需修正** |
| `chaos_re30_quick.yml` | ⚠️ | `loss` | ⚠️ 無sensors段 | ✅ Re=30 | **需修正** |
| `turbulent_re60.yml` | ⚠️ | `loss` | ⚠️ 無sensors段 | ✅ Re=60 | **需修正** |
| `turbulent_pure_pde.yml` | ⚠️ | `loss` | ⚠️ 無sensors段 | ✅ Re~78.5 | **需修正** |
| `curriculum.yml` | ✅ | `loss` | ✅ K=0 | ✅ 課程學習 | 需補充sensors |
| `test_periodic.yml` | ⚠️ | `loss` | ✅ K=0 | ✅ Re=60 | 需補充device |

**關鍵問題**：
1. ❌ **7/8 檔案使用舊鍵 `loss`，應改為 `losses`**（`trainer.py` 讀取 `config['losses']`）
2. ⚠️ **6/8 檔案缺少 `sensors` 頂層段**（應明確設置 K=0 表示純PDE訓練）
3. ⚠️ **部分檔案 `compute.device` 設置不一致**（有些用 `mps`，有些缺失）

---

## 🔧 必須修正的問題

### **問題 1: 損失配置鍵名不一致**

**原因**：`pinnx/train/trainer.py` 第 107-109 行：
```python
loss_config = config.get('losses', {})  # ⚠️ 讀取 'losses'，不是 'loss'
```

**影響檔案**：
- `chaos_re30.yml`
- `chaos_re30_full.yml`
- `chaos_re30_quick.yml`
- `turbulent_re60.yml`
- `turbulent_pure_pde.yml`
- `curriculum.yml`
- `test_periodic.yml`

**修正方案**：
```yaml
# ❌ 錯誤
loss:
  weights:
    continuity: 1.0

# ✅ 正確
losses:
  weights:
    continuity: 1.0
```

---

### **問題 2: 缺少 `sensors` 頂層配置段**

**原因**：`scripts/train.py` 第 158-167 行檢查 `config['sensors']['K']`：
```python
if 'sensors' in config:
    K_sensors = config['sensors'].get('K', 0)
```

**影響檔案**：
- `chaos_re30.yml`（僅在 `data` 段有 `n_sensors: 0`）
- `chaos_re30_full.yml`
- `chaos_re30_quick.yml`
- `turbulent_re60.yml`
- `turbulent_pure_pde.yml`

**修正方案**：
```yaml
# ✅ 添加頂層 sensors 配置
sensors:
  K: 0                       # 純PDE訓練無監督資料
  selection_method: "none"
  noise_level: 0.0
```

---

### **問題 3: 與 DNS 模擬參數不匹配**

**當前 DNS 模擬配置**（`generate_kolmogorov_dns_re30_stationary.py`）：
```python
Grid: 256×256 (square domain)
Physics: ν=0.02, A=0.768, k_f=4, Re≈30
Domain: [0, 2π] × [0, 2π]  # ⚠️ 注意：方形域
Time: dt=0.001, T_max=200
```

**配置檔案中的域設定**：
- ❌ `chaos_re30.yml`: `x_max=12.566` (4π) → **與 DNS 不符**
- ❌ `chaos_re30_full.yml`: `x_max=12.566` (4π) → **與 DNS 不符**
- ❌ `chaos_re30_quick.yml`: `x_max=12.566` (4π) → **與 DNS 不符**

**修正方案**：
```yaml
# ✅ 匹配 DNS 模擬（方形域）
physics:
  domain:
    x_min: 0.0
    x_max: 6.283185307179586  # 2π（與 DNS 一致）
    y_min: 0.0
    y_max: 6.283185307179586  # 2π（與 DNS 一致）
```

---

## 🎯 配置最佳化建議

### **建議 1: 統一 `device` 設定**

**現狀**：
- `baseline.yml`: `device: "auto"`（推薦）
- `chaos_re30.yml`: `device: "mps"`（Apple Silicon 特定）
- `curriculum.yml`: `device: "mps"`
- `test_periodic.yml`: `device: "mps"`

**建議**：
```yaml
# ✅ 統一改為 auto（自動檢測）
compute:
  device: "auto"  # 自動選擇 CUDA > MPS > CPU
```

---

### **建議 2: 添加 `reproducibility` 段**

**現狀**：
- 僅 `chaos_re30_full.yml` 和 `curriculum.yml` 有此段
- 其他檔案缺失，可能導致結果不可重現

**建議**：
```yaml
# ✅ 添加到所有配置（確保可重現性）
reproducibility:
  deterministic: false  # MPS 不支援完全確定性
  benchmark: true       # 啟用 CuDNN benchmark（加速）
  num_workers: 4
```

---

### **建議 3: 標準化 `loss.weights` 結構**

**現狀**：
- `baseline.yml`: 使用嵌套結構（`losses.weights`）
- 其他檔案：扁平結構（`loss.weights.continuity`）

**建議**：統一為嵌套結構（與模板一致）
```yaml
# ✅ 標準結構
losses:
  weights:
    data: 0.0
    momentum_x: 1.0
    momentum_y: 1.0
    continuity: 1.0
    periodic_x: 10.0
    periodic_y: 10.0
  
  normalize_losses: true
  warmup_epochs: 10
  
  adaptive_weighting:
    enabled: true
    method: "gradnorm"
    alpha: 1.5
    update_interval: 50
```

---

### **建議 4: 移除或註解過時的實驗配置**

**問題**：
- `chaos_re30.yml`: 2000 epochs, elongated domain (4π×2π)
- `chaos_re30_full.yml`: 2000 epochs, L-BFGS + Adam 兩階段
- `chaos_re30_quick.yml`: 500 epochs, 快速測試版

**建議**：
- **保留 `chaos_re30_quick.yml`**（快速驗證）
- **保留 `chaos_re30_full.yml`**（完整訓練，修正域大小）
- **移除或歸檔 `chaos_re30.yml`**（與 full 版功能重複，且域大小不符 DNS）

---

### **建議 5: 課程學習配置改進**

**現狀**：`curriculum.yml` 的階段配置過於簡化

**建議**：添加階段性驗證與檢查點策略
```yaml
# ✅ 改進課程學習配置
curriculum:
  enabled: true
  
  phase1:
    name: "transition_onset"
    epochs: 1000
    validate_interval: 100  # ⭐ 每 100 epochs 驗證
    checkpoint_interval: 200
    physics:
      forcing:
        amplitude: 0.32  # Re≈50
    # ... 其他配置
  
  # Phase 1 完成後的驗收條件
  phase1_acceptance:
    min_epochs: 500           # 最少訓練 500 epochs
    max_loss: 1.0e-2          # 總損失 < 0.01
    continuity_ratio: 0.30    # 連續方程損失比例 > 30%
```

---

## 📝 與當前 DNS 模擬對接計畫

### **目標**：使用 DNS 資料訓練 PINN 模型

**步驟 1: DNS 模擬完成後處理資料**
```bash
# DNS 完成後，資料位於：
# - data/kolmogorov_re30_stationary.h5

# 驗證資料格式
python -c "
import h5py
with h5py.File('data/kolmogorov_re30_stationary.h5', 'r') as f:
    print('Keys:', list(f.keys()))
    print('u shape:', f['u'].shape)
    print('Time steps:', f['time'][:])
"
```

**步驟 2: 創建新配置 `kolmogorov_2d_dns_re30.yml`**
```yaml
# 基於 DNS 資料的逆問題訓練
experiment:
  name: "kolmogorov_2d_dns_re30"
  description: "Inverse problem using DNS Re=30 stationary data"

data:
  source: "hdf5"
  file: "data/kolmogorov_re30_stationary.h5"
  steady_state: true
  time_snapshot: -1  # 使用最後一個時間步（穩態）

sensors:
  K: 50
  selection_method: "qr_pivot"
  sensor_file: "data/kolmogorov_dns_re30_qr_sensors_K50.npz"

physics:
  domain:
    x_min: 0.0
    x_max: 6.283185307179586  # 2π（匹配 DNS）
    y_min: 0.0
    y_max: 6.283185307179586  # 2π（匹配 DNS）
```

**步驟 3: 生成 QR-Pivot 感測點**
```bash
# 從 DNS 資料中選擇最優感測點
python scripts/generate_sensors_from_dns.py \
  --input data/kolmogorov_re30_stationary.h5 \
  --K 50 \
  --method qr_pivot \
  --output data/kolmogorov_dns_re30_qr_sensors_K50.npz
```

**步驟 4: 訓練與評估**
```bash
# 快速測試（500 epochs）
python scripts/train.py \
  --cfg configs/kolmogorov_experiments/kolmogorov_2d_dns_re30_quick.yml

# 完整訓練（2000 epochs）
python scripts/train.py \
  --cfg configs/kolmogorov_experiments/kolmogorov_2d_dns_re30_full.yml

# 評估結果
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/kolmogorov_2d_dns_re30_full/best_model.pth \
  --config configs/kolmogorov_experiments/kolmogorov_2d_dns_re30_full.yml \
  --reference data/kolmogorov_re30_stationary.h5
```

---

## 🚀 優先級排序

### **立即修正**（Critical）
1. ✅ 將所有 `loss` 改為 `losses`（7 個檔案）
2. ✅ 添加 `sensors` 頂層段（6 個檔案）
3. ✅ 修正 Re=30 配置的域大小（3 個檔案）

### **高優先級**（High）
4. ✅ 統一 `device: "auto"`（4 個檔案）
5. ✅ 添加 `reproducibility` 段（6 個檔案）

### **中優先級**（Medium）
6. ✅ 標準化 `losses.weights` 結構
7. ✅ 改進 `curriculum.yml` 的階段驗收條件

### **低優先級**（Low）
8. 📝 創建 DNS 資料訓練配置（等待 DNS 完成）
9. 📝 歸檔冗餘配置（`chaos_re30.yml`）

---

## 📋 修正檢查清單

- [ ] 修正 `chaos_re30.yml`
- [ ] 修正 `chaos_re30_full.yml`
- [ ] 修正 `chaos_re30_quick.yml`
- [ ] 修正 `turbulent_re60.yml`
- [ ] 修正 `turbulent_pure_pde.yml`
- [ ] 修正 `curriculum.yml`
- [ ] 修正 `test_periodic.yml`
- [ ] 檢查 `baseline.yml`（已符合標準）
- [ ] 創建 `kolmogorov_2d_dns_re30.yml`（等待 DNS）

---

## 💡 後續工作建議

1. **自動化驗證腳本**：
   ```bash
   # 檢查所有配置合法性
   python scripts/validate_configs.py configs/kolmogorov_experiments/*.yml
   ```

2. **配置模板生成器**：
   ```bash
   # 從模板快速生成新配置
   python scripts/create_kolmogorov_config.py \
     --template configs/templates/kolmogorov_baseline.yml \
     --Re 30 --epochs 2000 --K 50 \
     --output configs/kolmogorov_experiments/my_exp.yml
   ```

3. **批次訓練管理**：
   ```bash
   # 順序執行多個實驗
   python scripts/run_experiment_batch.py \
     --configs configs/kolmogorov_experiments/*.yml \
     --max_parallel 2
   ```

---

**結論**：現有配置需要統一修正以符合 `trainer.py` 與 `config_loader.py` 的規範，修正後即可順利使用當前 DNS 模擬資料進行訓練。
