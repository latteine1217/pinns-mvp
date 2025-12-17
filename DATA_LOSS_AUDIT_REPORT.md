# 📋 DNS Sensor Data & Loss Calculation Audit Report

**日期**: 2025-12-17  
**範圍**: 資料載入、感測器採樣、Loss 計算全流程審查  
**狀態**: 🔴 發現 3 個關鍵問題

---

## 🎯 審查目標

1. **DNS 資料載入**: 驗證 HDF5 檔案讀取是否正確
2. **Sensor 採樣**: 驗證感測點選取與時間序列構建邏輯
3. **Loss 計算**: 驗證資料損失、PDE 損失、Prior 損失計算是否物理一致
4. **座標對齊**: 驗證 DNS、Sensor、Prior 之間的空間座標是否對齊

---

## 🚨 發現的問題

### 🔴 問題 1: **Sensor 資料構建邏輯錯誤**

**位置**: `scripts/train/train.py::prepare_kolmogorov_training_data()` (Line 131-160)

**問題描述**:
```python
# ❌ 錯誤的構建方式
T_grid, K_grid = np.meshgrid(time_selected, np.arange(K), indexing='ij')

t_train = T_grid.flatten()  # [T*K]
k_indices = K_grid.flatten() # [T*K] 用於索引空間座標

x_train = x_sensor_locs[k_indices]  # ❌ 重複索引錯誤
y_train = y_sensor_locs[k_indices]

u_train = u_sensors_vals.flatten()  # ❌ Flatten 順序不一致
```

**根因分析**:
1. `meshgrid(time_selected, np.arange(K), indexing='ij')` 產生 `[T, K]` 形狀
2. `k_indices = K_grid.flatten()` 得到 `[0,1,2,...,K-1, 0,1,2,...,K-1, ...]`（重複 T 次）
3. `x_sensor_locs[k_indices]` 正確（每個時間步重複所有感測點）
4. **但** `u_sensors_vals.flatten()` 假設資料是 `[T, K]` 並以 C-order flatten
5. **如果** `u_sensors_vals` 實際形狀是 `[T, K]`，flatten 會得到正確的 `[u(t0,k0), u(t0,k1), ..., u(t1,k0), ...]`

**實際影響**:
- ✅ **如果** `u_sensors_vals` 確實是 `[T, K]` 形狀，則邏輯正確
- ❌ **如果** 資料維度被意外轉置為 `[K, T]`，會導致 sensor data 完全錯亂

**驗證方法**:
```python
# 在 Line 132 後添加驗證
assert u_sensors_vals.shape == (T_selected, K), \
    f"Sensor 資料形狀錯誤！預期 [{T_selected}, {K}]，實際 {u_sensors_vals.shape}"

# 驗證 flatten 後的對應關係
assert len(u_train) == T_selected * K
assert u_train[0] == u_sensors_vals[0, 0]  # t=0, sensor=0
assert u_train[K] == u_sensors_vals[1, 0]  # t=1, sensor=0
```

---

### 🔴 問題 2: **DNS 資料提取可能越界**

**位置**: `scripts/train/train.py::prepare_kolmogorov_training_data()` (Line 116-126)

**問題程式碼**:
```python
u_slice = f['u'][time_mask]  # [T, N, N]
v_slice = f['v'][time_mask]

# 展平空間維度 [T, N*N]
u_flat = u_slice.reshape(T_selected, -1)
v_flat = v_slice.reshape(T_selected, -1)

# 提取感測點的值 [T, K]
u_sensors_vals = u_flat[:, spatial_indices]  # ❌ 可能越界
```

**風險分析**:
1. `spatial_indices` 來自 JSON 檔案，聲稱是「展平後的空間索引 (0 ~ N*N-1)」
2. 如果 JSON 檔案中的 indices 是基於**不同網格解析度**生成的，會導致越界
3. 例如：JSON 來自 128x128 網格（最大索引 16383），但 DNS 是 256x256（最大索引 65535）

**驗證方法**:
```python
# 在 Line 128 後添加驗證
N_total = u_flat.shape[1]
assert spatial_indices.max() < N_total, \
    f"Sensor 索引越界！最大索引 {spatial_indices.max()} >= 總點數 {N_total}"
assert spatial_indices.min() >= 0, \
    f"Sensor 索引無效！最小索引 {spatial_indices.min()} < 0"

logging.info(f"✅ Sensor 索引驗證通過：[{spatial_indices.min()}, {spatial_indices.max()}] ⊂ [0, {N_total-1}]")
```

---

### 🟡 問題 3: **PDE 損失計算缺少時間維度驗證**

**位置**: `pinnx/train/loss_manager.py::compute_pde_loss()` (Line 141-142)

**問題程式碼**:
```python
# 提取時間分量（如果存在）
t_pde = data_batch.get('t_pde')
if t_pde is not None:
    t_pde = t_pde.to(self.device).requires_grad_(True)  # ✅ 正確

# 計算物理殘差
residuals = residual_fn(
    coords=coords_pde_physical,  # ❌ 未包含時間維度！
    predictions=u_pred_pde_physical,
    **kwargs
)
```

**問題分析**:
1. `coords_pde_physical` 僅包含 `[x, y]` 空間座標
2. **時間導數** `∂u/∂t` 需要 `t_pde` 才能計算
3. 如果 `residual_fn` 內部需要時間梯度，但 `coords` 未包含時間，會導致：
   - NS 方程退化為穩態方程（❌ 物理錯誤）
   - 或者程式報錯（形狀不匹配）

**驗證方法**:
```python
# 檢查 coords_pde_physical 的形狀
assert coords_pde_physical.shape[1] == 3, \
    f"PDE 座標應包含 [x, y, t]，但實際形狀為 {coords_pde_physical.shape}"

# 或者檢查 physics 模組是否正確處理時間
sig = inspect.signature(self.physics.residual_unified)
if 'time' in sig.parameters:
    logging.info("✅ Physics 模組支援時間參數（透過 kwargs）")
else:
    logging.warning("⚠️  Physics 模組不支援時間參數，僅能處理穩態問題")
```

---

## 🟢 正確的部分

### ✅ 1. **Leith Prior 載入與插值** (scripts/train/train.py::load_rans_prior_data)

**正確之處**:
```python
# ✅ 1D 座標格式正確處理
x_rans_1d = np.array(group['x'])
y_rans_1d = np.array(group['y'])

# ✅ 外插偵測
extrap_mask = (
    (coords_pde[:, 0] < x_min) | (coords_pde[:, 0] > x_max) |
    (coords_pde[:, 1] < y_min) | (coords_pde[:, 1] > y_max)
)
if n_extrap > len(coords_pde) * 0.05:
    raise ValueError("過多外插點！")  # ✅ 閾值檢查

# ✅ Metadata 標記壓力無效
'metadata': {
    'pressure_valid': False,
    'model_type': 'leith'
}
```

**評價**: 🌟 **優秀**，已處理 Leith 模型特殊性。

---

### ✅ 2. **Prior Loss 計算** (pinnx/train/loss_manager.py::compute_lowfi_prior_loss)

**正確之處**:
```python
# ✅ 檢查壓力有效性
lowfi_metadata = lowfi_prior.get('metadata', {})
pressure_valid = lowfi_metadata.get('pressure_valid', True)

if pressure_valid and 'p_pde' in lowfi_prior:
    lowfi_data = torch.cat([u_pde, v_pde, p_pde], dim=1)
    variable_names = ['u', 'v', 'p']
else:
    # ✅ Leith 模型：僅計算 u, v
    lowfi_data = torch.cat([u_pde, v_pde], dim=1)
    variable_names = ['u', 'v']
```

**評價**: 🌟 **正確**，已根據 metadata 動態調整。

---

### ✅ 3. **Data Loss 計算** (pinnx/train/loss_manager.py::compute_data_loss)

**正確之處**:
```python
# ✅ 動態維度適配
has_w_data = w_true is not None and w_true.numel() > 0
model_has_w = u_sensors_pred_phys.shape[1] >= 4

if has_w_data and model_has_w:
    w_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - w_true) ** 2)
    pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
elif model_has_w and not has_w_data:
    w_loss = torch.tensor(0.0)  # ✅ 跳過 w
    pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
else:
    # 2D 模式
    pressure_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - p_true)**2)
```

**評價**: 🌟 **正確**，支援 2D/3D 自動切換。

---

## 🛠️ 修復建議

### 🔧 修復 1: 驗證 Sensor 資料形狀 (高優先)

**檔案**: `scripts/train/train.py`  
**位置**: Line 128-132

```python
# ✅ 修復後的版本
u_flat = u_slice.reshape(T_selected, -1)
v_flat = v_slice.reshape(T_selected, -1)
p_flat = p_slice.reshape(T_selected, -1) if p_slice is not None else None

# ========== 新增驗證 ==========
N_total = u_flat.shape[1]
assert spatial_indices.max() < N_total, \
    f"❌ Sensor 索引越界！max={spatial_indices.max()} >= N_total={N_total}"
assert spatial_indices.min() >= 0, \
    f"❌ Sensor 索引無效！min={spatial_indices.min()}"

logging.info(f"✅ Sensor 索引驗證: [{spatial_indices.min()}, {spatial_indices.max()}] ⊂ [0, {N_total-1}]")
# ================================

# 提取感測點的值 [T, K]
u_sensors_vals = u_flat[:, spatial_indices]
v_sensors_vals = v_flat[:, spatial_indices]
p_sensors_vals = p_flat[:, spatial_indices] if p_flat is not None else None

# ========== 新增驗證 ==========
assert u_sensors_vals.shape == (T_selected, K), \
    f"❌ Sensor 值形狀錯誤！預期 [{T_selected}, {K}]，實際 {u_sensors_vals.shape}"

logging.info(f"✅ Sensor 值驗證: u_sensors_vals.shape={u_sensors_vals.shape}")
# ================================
```

---

### 🔧 修復 2: 驗證 Flatten 順序 (中優先)

**檔案**: `scripts/train/train.py`  
**位置**: Line 148-154

```python
u_train = u_sensors_vals.flatten()
v_train = v_sensors_vals.flatten()
p_train = p_sensors_vals.flatten() if p_sensors_vals is not None else np.zeros_like(u_train)

# ========== 新增驗證 ==========
# 驗證 flatten 後的對應關係
assert len(u_train) == T_selected * K
# 檢查第一個時間步的第一個 sensor
assert np.isclose(u_train[0], u_sensors_vals[0, 0]), \
    f"❌ Flatten 順序錯誤！u_train[0]={u_train[0]} != u_sensors_vals[0,0]={u_sensors_vals[0,0]}"
# 檢查第二個時間步的第一個 sensor
assert np.isclose(u_train[K], u_sensors_vals[1, 0]), \
    f"❌ Flatten 順序錯誤！u_train[{K}]={u_train[K]} != u_sensors_vals[1,0]={u_sensors_vals[1,0]}"

logging.info(f"✅ Flatten 順序驗證通過（C-order）")
# ================================
```

---

### 🔧 修復 3: 檢查時間維度處理 (中優先)

**檔案**: `pinnx/train/loss_manager.py`  
**位置**: Line 141-204

```python
# 提取時間分量（如果存在）
t_pde = data_batch.get('t_pde')
if t_pde is not None:
    t_pde = t_pde.to(self.device).requires_grad_(True)
    
    # ========== 新增驗證 ==========
    if epoch == 0:
        # 檢查 coords_pde_physical 是否包含時間維度
        if coords_pde_physical.shape[1] == 2:
            logging.warning(
                "⚠️  coords_pde_physical 僅包含 [x, y]，時間維度將透過 kwargs 傳遞"
            )
        elif coords_pde_physical.shape[1] == 3:
            logging.info("✅ coords_pde_physical 包含 [x, y, t]")
        
        # 檢查 physics 模組是否支援時間參數
        sig = inspect.signature(residual_fn)
        if 'time' in sig.parameters:
            logging.info("✅ Physics 模組支援 time 參數")
        else:
            logging.error("❌ Physics 模組不支援 time 參數，但資料包含時間維度！")
    # ================================
```

---

## 📊 檢查清單

### DNS 資料載入
- [x] ✅ HDF5 檔案路徑檢查
- [x] ✅ 時間範圍篩選 (`time_mask`)
- [x] ✅ 空間網格解析度 (`N`, `L`)
- [ ] ⚠️  **需驗證**: Sensor 索引是否基於正確的網格解析度

### Sensor 採樣
- [x] ✅ Sensor 位置檔案載入 (JSON)
- [x] ✅ 空間索引提取 (`spatial_indices`)
- [ ] ⚠️  **需驗證**: `u_sensors_vals` 的形狀 `[T, K]` 是否正確
- [ ] ⚠️  **需驗證**: Flatten 順序是否為 C-order

### Loss 計算
- [x] ✅ Data Loss: 維度動態適配 (2D/3D)
- [x] ✅ Prior Loss: Leith 模型特殊處理
- [x] ✅ PDE Loss: Momentum merging 支援
- [ ] ⚠️  **需驗證**: 時間導數計算是否正確

### 座標對齊
- [x] ✅ Prior 插值外插偵測 (5% 閾值)
- [x] ✅ Sensor 與 DNS 空間座標對齊
- [ ] ⚠️  **需驗證**: PDE 配點時間座標是否參與梯度計算

---

## 🎯 驗證腳本

創建一個獨立的驗證腳本來檢查上述問題：

```bash
# 執行驗證腳本
python scripts/validation/validate_data_pipeline.py \
    --config configs/kolmogorov_re50_kf4_K100.yml \
    --check-all
```

**驗證項目**:
1. Sensor 索引越界檢查
2. Sensor 資料形狀驗證
3. Flatten 順序一致性測試
4. 時間維度梯度追蹤測試
5. 座標對齊視覺化

---

## 📝 總結

### 🔴 必須修復（阻斷級）
1. **Sensor 索引越界檢查** - 可能導致訓練時 IndexError
2. **Flatten 順序驗證** - 可能導致 sensor data 與座標不對應

### 🟡 建議修復（警告級）
3. **時間維度處理** - 確認穩態 vs 非穩態方程處理邏輯

### 🟢 已正確實現
- Leith Prior 載入與 metadata 處理
- Prior Loss 的壓力項動態跳過
- Data Loss 的 2D/3D 自動切換

---

## 🔗 相關檔案

- `scripts/train/train.py` - 主訓練腳本（Line 50-210）
- `pinnx/train/loss_manager.py` - Loss 計算管理器（Line 86-580）
- `pinnx/dataio/lowfi_loader.py` - 低保真資料載入器（已審查，無問題）

---

**審查人**: AI Assistant  
**審查日期**: 2025-12-17  
**下一步行動**: 實施修復建議 1 & 2，並撰寫驗證腳本
