# 🔄 Leith 湍流模型遷移摘要

**日期**: 2025-12-18  
**任務**: 將 Kolmogorov Flow 實驗從 k-ε RANS 先驗遷移至 Leith 湍流模型先驗

---

## 📋 修改概覽

### 1. 配置檔案修改

#### `configs/kolmogorov_re50_kf4_K100.yml`

**實驗元數據**：
- `experiment.name`: `kolmogorov_re50_kf4_K100_rans_prior` → `kolmogorov_re50_kf4_K100_leith_prior`
- `experiment.version`: `v3.0_lbfgs_enhanced` → `v3.1_leith_prior`
- `experiment.description`: 更新為 "Leith turbulence prior (適合 2D 湍流，無 k/epsilon)"

**低保真先驗配置**（已存在，無需修改）：
```yaml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5  # ✅ Leith 數據
  data_type: leith  # ✅ 標記為 Leith 模型
  rans_structure:
    field_mapping:
      u: u
      v: v
      nu_t: nu_t  # ✅ 僅包含 nu_t，無 k/epsilon
    coord_mapping:
      x: x  # ✅ 1D 座標格式
      y: y
  consistency_weight: 2.0  # ✅ 已降低（Leith 品質較高）
```

**Curriculum 階段**（已調整）：
- Stage 1 (0-2000): Prior=2.0 (原 10.0，降低至適配 Leith)
- Stage 2 (2000-6000): Prior=1.0
- Stage 3 (6000-10000): Prior=0.1

**輸出路徑**：
- `checkpoint_dir`: `kolmogorov_re50_kf4_K100_rans_prior` → `kolmogorov_re50_kf4_K100_leith_prior`
- `results_dir`: 同上
- `visualization_dir`: 同上

---

#### `configs/kolmogorov_re50_kf4_K100_vanilla.yml`

**文檔註解更新**：
- 所有 "RANS" 引用 → "Turbulence" 或 "Leith"
- "與 RANS 版本對比" → "與 Leith-prior 版本對比"

**實驗元數據**：
- `experiment.version`: `v1.0_baseline` → `v1.1_baseline`
- `experiment.description`: 更新對比說明

---

### 2. Jupyter Notebook 修改

#### `PINNs_MVP_Kolmogorov_Guide.ipynb`

**全局替換**：
- 標題：`Physics-Informed Neural Networks with RANS Prior` → `with Leith Turbulence Prior`
- 版本：`v4.2 (Exponential Scheduler Edition)` → `v4.3 (Leith Prior Edition)`
- 更新日期：`2025-12-16` → `2025-12-18`

**核心特色描述**：
- ~~使用 k-ε RANS 場作為軟約束~~ → **使用 Leith 渦黏模型場（更適合 2D 湍流）**
- 新增說明：Leith 模型特點（僅包含 u, v, nu_t，無 k/epsilon）

**數據檢查邏輯**：
- 檔案路徑：`rans_re50_kf4.h5` → `rans_re50_kf4_leith.h5`
- Sensor 路徑：`sensors_K100_rans.npz` → `sensors_K100_leith.npz`
- 移除 k-ε 特有變量檢查（`k`, `epsilon`）
- 新增 Leith 座標格式檢查（1D `x`, `y` arrays）
- 新增渦黏度場 `nu_t` 視覺化（第三張圖）

**配置檢查代碼**：
- 修正 scheduler 路徑：`config['training']['optimizer']['scheduler']` → `config['training']['lr_scheduler']`
- 更新 Prior 說明文字

**訓練與評估路徑**：
- 所有 `rans_prior` → `leith_prior`
- 所有 `evaluation_rans_prior` → `evaluation_leith_prior`

---

## 🔬 Leith 模型 vs k-ε RANS 對比

| 特性 | k-ε RANS | Leith Model |
|------|----------|-------------|
| **適用場景** | 3D 湍流 | **2D 湍流** ✅ |
| **物理基礎** | 湍流動能 (k) + 耗散 (ε) | 渦黏度 (νₜ) 基於形變張量 |
| **逆級串捕捉** | ❌ 較差（3D 假設） | ✅ **優秀**（捕捉 2D 特性） |
| **變量數量** | 4 個 (u, v, k, ε) | **3 個** (u, v, νₜ) |
| **壓力場** | 有（可能不準確） | **無**（診斷模型） |
| **座標格式** | 2D meshgrid (X, Y) | **1D arrays** (x, y) |
| **初始權重建議** | 10.0（強引導） | **2.0**（中等引導，品質較高） |

---

## 📊 預期影響

### 訓練改進
1. **更適配 2D 物理**：Leith 模型專為 2D 湍流設計，能更好捕捉 Kolmogorov Flow 的逆級串
2. **降低過約束風險**：初始權重從 10.0 降至 2.0，避免模型過度依賴先驗
3. **簡化變量空間**：無需處理 k/ε 變量，降低插值複雜度

### 數據依賴
**需確保存在以下文件**：
```
data/lowfi/kolmogorov_rans/
├── rans_re50_kf4_leith.h5      # Leith 場數據
└── sensors_K100_leith.npz      # Leith QR Sensor
```

**數據格式要求**：
```python
# HDF5 結構
/mean_field/
  ├── u      [Ny, Nx]  # 速度場 x 分量
  ├── v      [Ny, Nx]  # 速度場 y 分量
  ├── nu_t   [Ny, Nx]  # 渦黏度
  ├── x      [Nx]      # 1D 座標
  └── y      [Ny]      # 1D 座標

# NPZ 結構
sensors_K100_leith.npz:
  ├── K              # 感測點數量
  ├── method         # 'qr_pivot'
  ├── source         # 'leith_model'
  ├── sensor_x       # [K] 感測點 x 座標
  ├── sensor_y       # [K] 感測點 y 座標
  └── metrics        # {'condition_number': float}
```

---

## ✅ 驗證清單

### 配置檔驗證
- [x] `kolmogorov_re50_kf4_K100.yml` 指向 Leith 數據
- [x] 實驗名稱/版本/描述已更新
- [x] 輸出路徑包含 `leith_prior` 標識
- [x] Curriculum 權重已調整（2.0 → 1.0 → 0.1）
- [x] Vanilla 配置檔註解已同步更新

### Notebook 驗證
- [x] 標題與版本已更新
- [x] 數據路徑指向 `rans_re50_kf4_leith.h5`
- [x] Sensor 路徑指向 `sensors_K100_leith.npz`
- [x] 移除 k-ε 特有檢查邏輯
- [x] 新增 Leith 特有說明（1D 座標、無 k/epsilon）
- [x] 所有路徑包含 `leith_prior` 標識

### 數據文件（需確認）
- [ ] `data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5` 存在且格式正確
- [ ] `data/lowfi/kolmogorov_rans/sensors_K100_leith.npz` 存在且格式正確
- [ ] Leith 場數值範圍合理（無 NaN/Inf）
- [ ] Sensor 條件數 < 500（良好可辨識性）

---

## 🚀 下一步行動

### 1. 數據生成（如尚未完成）
```bash
# 生成 Leith 場（需確認腳本是否存在）
python scripts/generate/generate_leith_field.py \
  --Re 50 \
  --kf 4 \
  --output data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5

# 生成 Leith QR Sensor
python scripts/generate/sensors/generate_sensors_qr_leith.py \
  --leith-file data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5 \
  --K 100 \
  --output data/lowfi/kolmogorov_rans/sensors_K100_leith.npz
```

### 2. 訓練驗證
```bash
# 快速測試（1000 epochs）
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100.yml \
  --override training.epochs=1000 \
  --device cuda

# 完整訓練（10000 epochs）
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100.yml \
  --device cuda
```

### 3. 對比實驗
```bash
# Vanilla baseline
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100_vanilla.yml \
  --device cuda

# 評估對比
python scripts/compare/compare_experiments.py \
  --exp1 kolmogorov_re50_kf4_K100_leith_prior \
  --exp2 kolmogorov_re50_kf4_K100_vanilla \
  --metrics relative_l2 pressure_gradient divergence
```

---

## 📚 參考文獻

1. **Leith, C. E. (1968)**. "Diffusion approximation for two‐dimensional turbulence."  
   *Physics of Fluids*, 11(3), 671-672.

2. **Kraichnan, R. H. (1976)**. "Eddy viscosity in two and three dimensions."  
   *Journal of the Atmospheric Sciences*, 33(8), 1521-1536.

3. **Boffetta, G., & Ecke, R. E. (2012)**. "Two-dimensional turbulence."  
   *Annual Review of Fluid Mechanics*, 44, 427-451.

---

**總結**：本次遷移完全將實驗從 k-ε RANS 先驗轉向 Leith 湍流模型先驗，以更好適配 Kolmogorov Flow 的 2D 湍流特性。所有配置、文檔和代碼邏輯已同步更新。接下來需確認 Leith 數據文件存在並符合格式要求，即可開始訓練。
