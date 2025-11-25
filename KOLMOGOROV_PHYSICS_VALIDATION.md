# Kolmogorov Flow 物理設定深度驗證報告

**日期**: 2025-11-25
**檢查項目**: 外力設定、雷諾數定義、物理參數一致性、邊界條件、NS 方程實現
**狀態**: 🔍 檢查中

---

## 1. 外力項設定檢查 ✅

### 1.1 理論公式
Kolmogorov flow 的強迫項應為：
```
f_x = A sin(k_f * y)  （僅 x 方向）
f_y = 0                （y 方向無強迫）
```

### 1.2 實現檢查
**檔案**: `pinnx/physics/kolmogorov_flow_2d.py`

**實現位置**: `compute_forcing_term()` (第 276-288 行)
```python
def compute_forcing_term(self, coords: torch.Tensor) -> torch.Tensor:
    """計算正弦強迫項：f_x = A sin(k_f y)"""
    y = coords[:, 1:2]
    forcing = self.amplitude * torch.sin(self.wavenumber * y)
    return forcing
```

**在動量方程中使用** (第 390-403 行)：
```python
# 計算強迫項
forcing_u = self.compute_forcing_term(coords)

# x 方向動量方程（含正弦強迫項）
residual_x = time_deriv_u + conv_u + pressure_x - viscous_u - forcing_u

# y 方向動量方程（無強迫項）
residual_y = time_deriv_v + conv_v + pressure_y - viscous_v
```

### 1.3 配置參數
**檔案**: `configs/kolmogorov_re100_kf8_K50_initial.yml`

```yaml
physics:
  forcing:
    k_f: 8              # 強迫波數
    amplitude: 1.0      # 強迫振幅 A
```

**驗證結果**: ✅ **正確**
- 強迫項僅作用於 x 方向動量方程
- 強迫項使用標準的 `A sin(k_f y)` 形式
- 參數值與配置文件一致

---

## 2. 雷諾數定義檢查 ⚠️ **需要澄清**

### 2.1 理論基礎
Kolmogorov flow 的雷諾數有兩種常見定義：

#### 定義 A（標準文獻定義）
**來源**: Meshalkin & Sinai (1961), Boffetta et al. (2002)
```
Re = F / (ν² k³)
```
其中：
- F = A（強迫振幅）
- ν = 動力黏度
- k = k_f（強迫波數）

**推導邏輯**：
- 層流解：u_x(y) = (F / (ν k²)) sin(k y)
- 特徵速度：U = F / (ν k²)
- 特徵長度：L = 1/k
- Re = UL/ν = [F / (ν k²)] × (1/k) / ν = F / (ν² k³)

#### 定義 B（基於特徵尺度）
```
Re = U L / ν
```
其中：
- U = 特徵速度（通常取動能均方根或層流解峰值速度）
- L = 特徵長度（通常取 2π/k_f）
- ν = 動力黏度

### 2.2 程式碼實現
**檔案**: `pinnx/physics/kolmogorov_flow_2d.py`

**主要雷諾數計算** (第 182-211 行)：
```python
def compute_reynolds_number(self) -> float:
    """
    計算 Kolmogorov Flow 的雷諾數（標準定義）

    理論基礎：
        Re = F / (ν² k³)
    """
    F = float(self.amplitude.item())
    nu = float(self.nu.item())
    k = float(self.wavenumber.item())

    Re = F / (nu**2 * k**3)
    return Re
```

**有效雷諾數計算** (第 213-247 行)：
```python
def compute_effective_reynolds(self, predictions: torch.Tensor) -> float:
    """
    計算有效雷諾數（基於預測場的動能）

    定義：
        Re_eff = √(2E) × L / ν
    """
    KE = float(self.compute_kinetic_energy(predictions).item())
    U_eff = np.sqrt(2.0 * KE)
    L = 1.0 / float(self.wavenumber.item())
    nu = float(self.nu.item())
    Re_eff = U_eff * L / nu
    return Re_eff
```

### 2.3 配置參數驗證

**配置 1**: `kolmogorov_re100_kf8_K50_initial.yml`
```yaml
physics_params:
  Re: 100
  nu: 0.01  # 1/Re
  k_f: 8
  forcing_amplitude: 1.0
```

**計算驗證**：
```
Re = F / (ν² k³)
   = 1.0 / (0.01² × 8³)
   = 1.0 / (0.0001 × 512)
   = 1.0 / 0.0512
   = 19.53  ⚠️ 不等於 100！
```

**配置 2**: `kolmogorov_re100_kf8_K50_t20_2k.yml`
```yaml
physics_params:
  Re: 100
  nu: 0.01
  k_f: 8
  forcing_amplitude: 1.0
```

**同樣計算結果**: Re ≈ 19.53

### 2.4 問題診斷 ⚠️

**問題 1**: **雷諾數定義不一致**
- 配置文件標註 `Re: 100`
- 但根據標準定義 `Re = F / (ν² k³)` 計算得 Re ≈ 19.53
- 兩者相差約 5 倍

**可能原因**：
1. **定義不同**: 配置文件可能使用定義 B（Re = UL/ν），而程式碼使用定義 A
2. **參數衝突**: nu 值可能不應該是 `1/Re_config`，而應該從強迫參數倒推
3. **文獻差異**: 不同文獻對 Kolmogorov flow 的 Re 定義可能不同

**建議修正方案**：

#### 方案 A：修正 ν 值以匹配 Re=100（使用標準定義）
```
Re = F / (ν² k³) = 100
ν² = F / (Re × k³) = 1.0 / (100 × 512) = 1.953e-5
ν = 0.00442  （而非 0.01）
```

#### 方案 B：修正 A 值以匹配 Re=100（保持 ν=0.01）
```
Re = F / (ν² k³) = 100
F = Re × ν² × k³ = 100 × 0.0001 × 512 = 5.12
```

#### 方案 C：明確使用不同的 Re 定義並在文檔中說明
如果 DNS 數據使用不同的 Re 定義，需要：
1. 在配置文件中明確註明使用的定義
2. 在程式碼中添加轉換函數
3. 在文檔中清楚說明

---

## 3. 物理參數一致性檢查

### 3.1 配置文件之間的一致性

**檔案 1**: `kolmogorov_re100_kf8_K50_initial.yml`
```yaml
physics:
  nu: 0.01
  forcing:
    k_f: 8
    amplitude: 1.0
```

**檔案 2**: `kolmogorov_re100_kf8_K50_t20_2k.yml`
```yaml
physics:
  nu: 0.01
  forcing:
    k_f: 8
    amplitude: 1.0
```

✅ **兩個配置文件的物理參數一致**

### 3.2 DNS 數據參數比對

**配置文件宣稱**：
```yaml
kolmogorov_config:
  physics_params:
    Re: 100
    nu: 0.01
    k_f: 8
    forcing_amplitude: 1.0
```

**需要驗證**：
- DNS 數據檔案 `kolmogorov_dns_re100_512x512_kf8_midway.h5` 是否存在
- 檔案中的實際參數是否與配置匹配
- DNS 使用的 Re 定義是什麼

---

## 4. 邊界條件與域設定檢查

### 4.1 理論要求
Kolmogorov flow 應該使用**雙週期性邊界條件**：
```
u(0, y, t) = u(L_x, y, t)  （x 方向週期）
v(0, y, t) = v(L_x, y, t)
p(0, y, t) = p(L_x, y, t)

u(x, 0, t) = u(x, L_y, t)  （y 方向週期）
v(x, 0, t) = v(x, L_y, t)
p(x, 0, t) = p(x, L_y, t)
```

域範圍通常為：`[0, 2π] × [0, 2π]`

### 4.2 實現檢查

**域設定** (配置文件)：
```yaml
domain:
  x: [0.0, 6.283185307179586]  # 2π ✅
  y: [0.0, 6.283185307179586]  # 2π ✅
```

**週期性邊界條件實現** (`kolmogorov_flow_2d.py`, 第 499-552 行)：
```python
def compute_periodic_loss(
    self,
    coords: torch.Tensor,
    predictions: torch.Tensor,
    boundary_band_width: float = 5e-3
) -> Dict[str, torch.Tensor]:
    """計算嚴格週期性邊界約束損失"""
    x_min, x_max = self.domain_bounds['x']
    y_min, y_max = self.domain_bounds['y']

    # x 方向週期性
    mask_x_min = torch.abs(coords[:, 0] - x_min) < boundary_band_width
    mask_x_max = torch.abs(coords[:, 0] - x_max) < boundary_band_width

    # y 方向週期性
    mask_y_min = torch.abs(coords[:, 1] - y_min) < boundary_band_width
    mask_y_max = torch.abs(coords[:, 1] - y_max) < boundary_band_width

    # 計算週期性誤差
    # ...
```

**配置文件中的權重**：
```yaml
losses:
  periodicity_weight: 10.0
```

✅ **邊界條件實現正確**

---

## 5. NS 方程實現完整性檢查

### 5.1 理論方程
完整的 2D 不可壓縮 NS 方程（含強迫項）：

**x 方向動量**：
```
∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u + A sin(k_f y)
```

**y 方向動量**：
```
∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v
```

**連續性方程**：
```
∂u/∂x + ∂v/∂y = 0
```

### 5.2 實現檢查 (`compute_momentum_residuals`, 第 343-411 行)

**時間導數項** ✅:
```python
time_deriv_u = torch.zeros_like(u)
time_deriv_v = torch.zeros_like(v)

if time is not None:
    if time.requires_grad:
        time_deriv_u = compute_gradient_2d(u, time, component=0)
        time_deriv_v = compute_gradient_2d(v, time, component=0)
```

**對流項** ✅:
```python
conv_u = u * u_grads['x'] + v * u_grads['y']
conv_v = u * v_grads['x'] + v * v_grads['y']
```

**壓力項** ✅:
```python
pressure_x = p_grads['x'] / self.rho
pressure_y = p_grads['y'] / self.rho
```

**黏性項** ✅:
```python
laplacian_u = self.compute_laplacian(u, coords)
laplacian_v = self.compute_laplacian(v, coords)
viscous_u = self.nu * laplacian_u
viscous_v = self.nu * laplacian_v
```

**強迫項** ✅:
```python
forcing_u = self.compute_forcing_term(coords)
```

**組裝殘差** ✅:
```python
# x 方向動量方程（含正弦強迫項）
residual_x = time_deriv_u + conv_u + pressure_x - viscous_u - forcing_u

# y 方向動量方程
residual_y = time_deriv_v + conv_v + pressure_y - viscous_v
```

**連續性方程** (`compute_continuity_residual`, 第 413-440 行) ✅:
```python
def compute_continuity_residual(
    self,
    coords: torch.Tensor,
    predictions: torch.Tensor
) -> torch.Tensor:
    """計算連續方程殘差：∂u/∂x + ∂v/∂y = 0"""
    u = predictions[:, 0:1]
    v = predictions[:, 1:2]

    u_grads = self.compute_gradients(u, coords, order=1)
    v_grads = self.compute_gradients(v, coords, order=1)

    divergence = u_grads['x'] + v_grads['y']
    return divergence
```

✅ **NS 方程實現完整且正確**

---

## 6. 額外發現與建議

### 6.1 缺少的物理量驗證

程式碼中實現了以下物理量計算，但未在配置文件中啟用評估：

**渦度** (第 554-577 行):
```python
def compute_vorticity(self, coords, predictions):
    """計算 2D 渦度：ω = ∂v/∂x - ∂u/∂y"""
```

**Enstrophy** (第 579-599 行):
```python
def compute_enstrophy(self, coords, predictions):
    """計算 enstrophy：E = ∫ω² dA / A"""
```

**動能** (第 601-619 行):
```python
def compute_kinetic_energy(self, predictions):
    """計算動能：KE = ∫(u² + v²) dA / (2A)"""
```

**建議**: 在評估指標中添加渦度場比對、enstrophy 時間序列驗證

### 6.2 損失歸一化策略

程式碼實現了損失歸一化 (`normalize_loss_dict`, 第 621-667 行)：
```python
def normalize_loss_dict(self, loss_dict, epoch):
    """
    損失歸一化：將每個損失項除以其參考值

    策略：
    1. Warmup (epoch < warmup_epochs): 收集初始值
    2. Training: 使用參考值進行歸一化
    """
```

但配置文件中未明確指定 `warmup_epochs`，將使用預設值 5。

**建議**: 在配置文件中明確設定：
```yaml
losses:
  normalize_losses: true
  warmup_epochs: 10  # 明確指定
```

### 6.3 梯度計算的穩健性

程式碼中的梯度計算函數 (`compute_gradient_2d`) 包含錯誤處理：
```python
if grads is None:
    raise RuntimeError(
        f"計算梯度失敗：field 與 coords 之間沒有計算圖連接。"
    )
```

✅ **這是良好的實踐**，能及早發現計算圖斷裂問題。

---

## 總結與建議

### ✅ 正確的部分
1. **外力項實現正確**: `f_x = A sin(k_f y)` 僅作用於 x 方向
2. **NS 方程完整**: 包含所有必要項（時間導數、對流、壓力、黏性、強迫）
3. **邊界條件正確**: 實現了雙週期性邊界約束
4. **域設定正確**: `[0, 2π] × [0, 2π]`
5. **程式碼架構良好**: 模組化、有錯誤處理、包含物理量計算

### ⚠️ 需要修正的問題

#### 🔴 **嚴重問題：雷諾數定義不一致**
- **現象**: 配置標註 Re=100，但程式碼計算得 Re≈19.53
- **影響**: 訓練的實際物理條件與宣稱不符，可能導致：
  - 與 DNS 數據的物理尺度不匹配
  - 湍流特性與預期不同
  - 論文中的 Re 值報告錯誤

**必須採取的行動**：
1. **確認 DNS 數據的實際 Re 定義**
   ```bash
   python scripts/check_dns_re100.py  # 需要編寫檢查腳本
   ```

2. **選擇一種解決方案**：
   - 方案 A：修正 `ν = 0.00442`（匹配標準定義 Re=100）
   - 方案 B：修正 `A = 5.12`（保持 ν=0.01）
   - 方案 C：明確使用不同定義並在所有文檔中說明

3. **更新所有配置文件和文檔**

#### 🟡 **建議改進**
1. 在配置文件中明確指定 `warmup_epochs`
2. 添加渦度場、enstrophy 的評估指標
3. 編寫 DNS 數據驗證腳本
4. 在文檔中明確說明使用的 Re 定義

---

## 下一步行動

1. **立即行動**: 檢查 DNS 數據檔案中的實際物理參數
2. **修正配置**: 根據 DNS 數據調整 ν 或 A 值
3. **驗證訓練**: 用修正後的配置重新訓練並比對結果
4. **更新文檔**: 在所有相關文檔中明確說明 Re 定義

**優先級**: 🔴 **高（影響訓練的物理正確性）**
