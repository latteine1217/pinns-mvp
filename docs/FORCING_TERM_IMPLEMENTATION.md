# Kolmogorov Flow 外力項在 PINNs 訓練中的實現

**日期**: 2025-11-25
**適用**: Kolmogorov Flow 2D 訓練

---

## 📋 目錄

1. [物理背景](#物理背景)
2. [實現架構](#實現架構)
3. [代碼實現細節](#代碼實現細節)
4. [訓練流程](#訓練流程)
5. [配置設定](#配置設定)
6. [驗證與檢查](#驗證與檢查)
7. [常見問題](#常見問題)

---

## 物理背景

### Kolmogorov Flow 控制方程

2D 不可壓縮 Navier-Stokes 方程，含正弦強迫項：

```
∂u/∂t + u·∂u/∂x + v·∂u/∂y = -∂p/∂x + ν∇²u + f_x    (x-動量)
∂v/∂t + u·∂v/∂x + v·∂v/∂y = -∂p/∂y + ν∇²v          (y-動量)
∂u/∂x + ∂v/∂y = 0                                    (連續性)
```

### 外力項定義

**x 方向強迫**（Kolmogorov forcing）：
```
f_x = A sin(k_f y)
```

**y 方向強迫**：
```
f_y = 0  （無強迫）
```

### 物理參數

| 參數 | 符號 | 典型值 | 說明 |
|------|------|--------|------|
| 強迫振幅 | A (或 f₀) | 1.0 | 外力強度 |
| 強迫波數 | k_f | 4, 8, 16 | 空間頻率 |
| 動力黏度 | ν | 0.01-0.0125 | 黏性係數 |
| 雷諾數 | Re | 30-200 | Re = √f₀ × L^(3/2) / ν |

---

## 實現架構

### 整體流程圖

```
訓練循環 (trainer.py)
    │
    ├─► 1. 前向傳播：模型預測
    │   model(coords) → [u, v, p]
    │
    ├─► 2. 計算物理殘差
    │   └─► physics.residual_unified(coords, predictions)
    │       │
    │       ├─► compute_momentum_residuals()
    │       │   ├─► 對流項：u·∇u, u·∇v
    │       │   ├─► 壓力項：-∇p
    │       │   ├─► 黏性項：ν∇²u, ν∇²v
    │       │   └─► 強迫項：compute_forcing_term() ⭐
    │       │       └─► f_x = A sin(k_f y)
    │       │
    │       └─► compute_continuity_residual()
    │           └─► ∇·u = 0
    │
    └─► 3. 計算損失
        └─► L_pde = mean(residual²)
```

### 關鍵組件

| 組件 | 檔案 | 功能 |
|------|------|------|
| **物理模組** | `pinnx/physics/kolmogorov_flow_2d.py` | 定義外力項計算 |
| **訓練器** | `pinnx/train/trainer.py` | 調用物理殘差計算 |
| **配置** | `configs/*.yml` | 設定外力參數 |

---

## 代碼實現細節

### 1. 外力項計算（物理模組）

**檔案**: `pinnx/physics/kolmogorov_flow_2d.py`

#### 初始化外力參數

```python
class KolmogorovFlow2D(nn.Module):
    def __init__(
        self,
        forcing_params: Optional[Dict[str, float]] = None,
        physics_params: Optional[Dict[str, float]] = None,
        ...
    ):
        super().__init__()

        # 預設外力參數
        default_forcing = {
            'amplitude': 1.0,      # 強迫振幅 A
            'wavenumber': 4,       # 強迫波數 k_f
        }

        self.forcing_params = {**default_forcing, **(forcing_params or {})}

        # 註冊為緩衝區（不參與梯度計算）
        self.register_buffer('amplitude', torch.tensor(float(self.forcing_params['amplitude'])))
        self.register_buffer('wavenumber', torch.tensor(float(self.forcing_params['wavenumber'])))
```

**關鍵點**：
- ✅ 使用 `register_buffer` 而非 `nn.Parameter`
- ✅ 外力參數**不參與訓練**（固定值）
- ✅ 自動處理 CPU/GPU 遷移

---

#### 外力項計算函數

```python
def compute_forcing_term(self, coords: torch.Tensor) -> torch.Tensor:
    """
    計算正弦強迫項：f_x = A sin(k_f y)

    Args:
        coords: [batch, 2] = [x, y] 物理坐標

    Returns:
        forcing: [batch, 1] = x 方向強迫項
    """
    y = coords[:, 1:2]  # 提取 y 坐標
    forcing = self.amplitude * torch.sin(self.wavenumber * y)
    return forcing
```

**關鍵點**：
- ✅ 輸入：物理坐標 `[batch, 2]`
- ✅ 輸出：x 方向外力 `[batch, 1]`
- ✅ y 方向外力恆為 0（不需要計算）
- ✅ 使用 PyTorch 自動微分計算外力對座標的導數（如需）

---

#### 外力項整合到動量方程

```python
def compute_momentum_residuals(
    self,
    coords: torch.Tensor,
    predictions: torch.Tensor,
    time: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    計算動量殘差（含外力項）
    """
    u = predictions[:, 0:1]
    v = predictions[:, 1:2]
    p = predictions[:, 2:3]

    # === 計算各項 ===
    # 1. 對流項
    conv_u = u * u_grads['x'] + v * u_grads['y']
    conv_v = u * v_grads['x'] + v * v_grads['y']

    # 2. 壓力項
    pressure_x = p_grads['x'] / self.rho
    pressure_y = p_grads['y'] / self.rho

    # 3. 黏性項
    viscous_u = self.nu * self.compute_laplacian(u, coords)
    viscous_v = self.nu * self.compute_laplacian(v, coords)

    # 4. 外力項 ⭐
    forcing_u = self.compute_forcing_term(coords)
    # forcing_v = 0（y 方向無強迫）

    # 5. 時間導數（非穩態）
    time_deriv_u = ... if time is not None else 0
    time_deriv_v = ... if time is not None else 0

    # === 組裝殘差 ===
    # x 方向：包含外力項
    residual_x = time_deriv_u + conv_u + pressure_x - viscous_u - forcing_u

    # y 方向：無外力項
    residual_y = time_deriv_v + conv_v + pressure_y - viscous_v

    return {
        'momentum_x': residual_x,
        'momentum_y': residual_y,
    }
```

**關鍵點**：
- ✅ 外力項僅在 **x 方向動量方程**中出現
- ✅ 符號：方程右側為 `+f_x`，殘差中為 `-forcing_u`
- ✅ 外力項與座標相關（通過 `y` 計算），自動保持梯度連接

---

### 2. 訓練器調用（訓練流程）

**檔案**: `pinnx/train/trainer.py`

#### 計算 PDE 損失

```python
def step(self, data_batch: Dict[str, torch.Tensor], epoch: int) -> Dict[str, float]:
    """單步訓練"""

    # 1. 前向傳播
    coords_pde = data_batch['x_pde']  # [batch, 2]
    predictions = self.model(coords_pde)  # [batch, 3] = [u, v, p]

    # 2. 計算物理殘差（含外力項）
    residuals = self.physics.residual_unified(
        coords_pde,      # 物理座標
        predictions      # 模型預測
    )
    # residuals = {'momentum_x', 'momentum_y', 'continuity'}

    # 3. 計算損失
    momentum_x_loss = torch.mean(residuals['momentum_x']**2)
    momentum_y_loss = torch.mean(residuals['momentum_y']**2)
    continuity_loss = torch.mean(residuals['continuity']**2)

    # 4. 組合 PDE 損失
    pde_loss = (
        self.loss_weights['momentum_x'] * momentum_x_loss +
        self.loss_weights['momentum_y'] * momentum_y_loss +
        self.loss_weights['continuity'] * continuity_loss
    )

    return pde_loss
```

**關鍵點**：
- ✅ 外力項**自動包含**在 `momentum_x` 殘差中
- ✅ 不需要額外的損失項
- ✅ 使用統一的 `residual_unified` 接口

---

### 3. 統一殘差接口（兼容性）

```python
def residual_unified(
    self,
    coords: torch.Tensor,
    predictions: torch.Tensor,
    time: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    統一殘差計算介面（與 trainer.py 兼容）

    自動包含外力項計算
    """
    # 2D 座標切片（如果需要）
    coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords
    predictions_3d = predictions[:, :3] if predictions.shape[1] >= 3 else predictions

    # 計算動量殘差（自動包含外力項）
    momentum_residuals = self.compute_momentum_residuals(coords_2d, predictions_3d, time)

    # 計算連續性殘差
    continuity_residual = self.compute_continuity_residual(coords_2d, predictions_3d)

    # 統一格式（添加 momentum_z = 0 以兼容 3D 訓練器）
    return {
        'momentum_x': momentum_residuals['momentum_x'],
        'momentum_y': momentum_residuals['momentum_y'],
        'momentum_z': torch.zeros_like(continuity_residual),  # 2D 無 z 分量
        'continuity': continuity_residual,
    }
```

**關鍵點**：
- ✅ 自動維度兼容（2D/3D）
- ✅ 外力項透明整合（訓練器無需感知）
- ✅ 統一接口便於擴展

---

## 訓練流程

### 完整流程圖（含外力項）

```
[開始訓練]
    │
    ├─► 讀取配置
    │   ├─ physics.forcing.amplitude = 1.0
    │   ├─ physics.forcing.k_f = 8
    │   └─ physics.nu = 0.0125
    │
    ├─► 初始化物理模組
    │   KolmogorovFlow2D(
    │       forcing_params={'amplitude': 1.0, 'wavenumber': 8},
    │       physics_params={'nu': 0.0125}
    │   )
    │   └─► register_buffer('amplitude', 1.0)
    │   └─► register_buffer('wavenumber', 8)
    │
    ├─► 訓練循環
    │   │
    │   └─► 每個 epoch
    │       │
    │       ├─► 前向傳播
    │       │   model([x, y]) → [u, v, p]
    │       │
    │       ├─► 計算物理殘差
    │       │   │
    │       │   ├─► x-動量殘差
    │       │   │   = ∂u/∂t + conv - pressure - viscous - forcing ⭐
    │       │   │   其中 forcing = A sin(k_f y)
    │       │   │
    │       │   ├─► y-動量殘差
    │       │   │   = ∂v/∂t + conv - pressure - viscous
    │       │   │   （無外力項）
    │       │   │
    │       │   └─► 連續性殘差
    │       │       = ∂u/∂x + ∂v/∂y
    │       │
    │       ├─► 計算總損失
    │       │   = w_data * L_data +
    │       │     w_mx * L_momentum_x +
    │       │     w_my * L_momentum_y +
    │       │     w_cont * L_continuity
    │       │
    │       └─► 反向傳播 & 更新
    │
    └─► [訓練完成]
```

---

## 配置設定

### YAML 配置範例

**檔案**: `configs/kolmogorov_re56_kf8_K50.yml`

```yaml
# =============================================================================
# 物理參數配置
# =============================================================================
physics:
  type: "kolmogorov_flow_2d"

  # 動力學參數
  nu: 0.0125      # 動力黏度
  rho: 1.0        # 密度

  # 外力參數 ⭐
  forcing:
    amplitude: 1.0    # 強迫振幅 A (或 f₀)
    k_f: 8            # 強迫波數

  # 計算出的雷諾數
  kolmogorov_flow:
    Re: 55.7              # Musacchio & Boffetta (2014) 定義
    k_f: 8
    forcing_amplitude: 1.0

  # 域設定
  domain:
    x_range: [0.0, 6.283185307179586]  # 2π
    y_range: [0.0, 6.283185307179586]  # 2π

  # 邊界條件
  boundary_conditions:
    periodic_x: true
    periodic_y: true

# =============================================================================
# 損失權重配置
# =============================================================================
losses:
  # 資料項（感測點）
  data_weight: 100.0

  # PDE 項（含外力項在內）
  momentum_x_weight: 1.0    # x-動量（含外力）
  momentum_y_weight: 1.0    # y-動量（無外力）
  continuity_weight: 1.0    # 連續性

  # 週期性約束
  periodicity_weight: 10.0

  # 自適應權重
  adaptive_weighting: true
  adaptation_method: "gradnorm"
```

### 初始化代碼（訓練腳本）

**檔案**: `scripts/train.py`

```python
import yaml
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

# 讀取配置
with open('configs/kolmogorov_re56_kf8_K50.yml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化物理模組
physics = KolmogorovFlow2D(
    forcing_params={
        'amplitude': config['physics']['forcing']['amplitude'],
        'wavenumber': config['physics']['forcing']['k_f'],
    },
    physics_params={
        'nu': config['physics']['nu'],
        'rho': config['physics']['rho'],
    },
    domain_bounds={
        'x': config['physics']['domain']['x_range'],
        'y': config['physics']['domain']['y_range'],
    }
)

# 驗證外力參數
print(f"外力振幅 A = {physics.amplitude.item()}")
print(f"外力波數 k_f = {physics.wavenumber.item()}")
print(f"雷諾數 Re = {physics.compute_reynolds_number():.2f}")
```

---

## 驗證與檢查

### 1. 驗證外力項計算

```python
import torch
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

# 初始化物理模組
physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 1.0, 'wavenumber': 8}
)

# 測試外力項計算
coords = torch.tensor([[0.0, 0.0], [0.0, np.pi/8], [0.0, np.pi/4]], requires_grad=True)
forcing = physics.compute_forcing_term(coords)

print("座標 y:", coords[:, 1].detach().numpy())
print("外力 f_x:", forcing.detach().numpy())

# 預期結果：
# y = [0, π/8, π/4]
# f_x = A sin(k_f y) = [0, sin(π), sin(2π)] = [0, 0, 0]（因為 k_f=8）
```

### 2. 驗證殘差計算

```python
# 模擬模型預測
predictions = torch.tensor([[1.0, 0.5, 0.1]], requires_grad=True)  # [u, v, p]
coords = torch.tensor([[1.0, 0.5]], requires_grad=True)  # [x, y]

# 計算殘差
residuals = physics.residual_unified(coords, predictions)

print("Momentum X 殘差:", residuals['momentum_x'].item())
print("Momentum Y 殘差:", residuals['momentum_y'].item())
print("Continuity 殘差:", residuals['continuity'].item())

# 檢查：momentum_x 應包含外力項的影響
```

### 3. 驗證梯度連接

```python
# 確認外力項參與反向傳播
coords = torch.tensor([[1.0, 0.5]], requires_grad=True)
predictions = torch.rand(1, 3, requires_grad=True)

residuals = physics.residual_unified(coords, predictions)
loss = residuals['momentum_x'].pow(2).mean()

# 反向傳播
loss.backward()

# 檢查梯度
assert predictions.grad is not None, "預測值應有梯度"
assert coords.grad is not None, "座標應有梯度（外力項依賴 y）"
print("✅ 外力項梯度連接正確")
```

### 4. 物理一致性檢查

```python
# 檢查層流解
def check_laminar_solution():
    """驗證層流解是否滿足方程"""
    physics = KolmogorovFlow2D(
        forcing_params={'amplitude': 1.0, 'wavenumber': 8},
        physics_params={'nu': 0.0125}
    )

    # 層流解：u = (A/(ν k²)) sin(k y), v = 0, p = const
    A = 1.0
    nu = 0.0125
    k = 8.0
    U0 = A / (nu * k**2)

    y_test = torch.linspace(0, 2*np.pi, 100, requires_grad=True).view(-1, 1)
    x_test = torch.zeros_like(y_test)
    coords = torch.cat([x_test, y_test], dim=1)

    # 層流解預測
    u_laminar = U0 * torch.sin(k * y_test)
    v_laminar = torch.zeros_like(y_test)
    p_laminar = torch.zeros_like(y_test)
    predictions = torch.cat([u_laminar, v_laminar, p_laminar], dim=1)

    # 計算殘差
    residuals = physics.residual_unified(coords, predictions)

    # 檢查（層流解應使殘差接近 0）
    print(f"Momentum X 殘差 (應≈0): {residuals['momentum_x'].abs().mean():.6f}")
    print(f"Momentum Y 殘差 (應≈0): {residuals['momentum_y'].abs().mean():.6f}")
    print(f"Continuity 殘差 (應≈0): {residuals['continuity'].abs().mean():.6f}")

check_laminar_solution()
```

---

## 常見問題

### ❓ Q1: 外力項是否參與模型訓練？

**A**: **不參與**。外力項通過 `register_buffer` 註冊，**不是可學習參數**。

- ✅ 外力參數固定（A, k_f）
- ✅ 模型學習的是流場 (u, v, p)
- ✅ 外力項提供物理約束

```python
# 外力參數不會被優化器更新
self.register_buffer('amplitude', torch.tensor(1.0))  # 固定值
# vs.
# self.amplitude = nn.Parameter(torch.tensor(1.0))  # 可學習（錯誤！）
```

---

### ❓ Q2: 如何修改外力振幅？

**A**: 在配置文件中修改，然後重新初始化物理模組。

```yaml
# 方法 1：修改配置文件
physics:
  forcing:
    amplitude: 2.0  # 從 1.0 改為 2.0
```

```python
# 方法 2：程式碼中動態修改
physics.amplitude = torch.tensor(2.0)
physics.wavenumber = torch.tensor(16)

# 重新計算雷諾數
Re_new = physics.compute_reynolds_number()
print(f"新的 Re = {Re_new:.2f}")
```

---

### ❓ Q3: 為什麼 y 方向沒有外力項？

**A**: 這是 **Kolmogorov flow 的標準設定**。

- ✅ x 方向：正弦強迫 `f_x = A sin(k_f y)`
- ✅ y 方向：無強迫 `f_y = 0`

這種非對稱強迫會產生：
- 層流解：`u = U₀ sin(k_f y), v = 0`
- 隨 Re 增加出現不穩定 → 渦結構 → 湍流

如果需要 y 方向外力，需要修改 `compute_forcing_term` 函數。

---

### ❓ Q4: 外力項會影響哪些損失項？

**A**: **僅影響 x 方向動量損失** (`momentum_x_loss`)。

```
總損失 = w_data * L_data +
         w_mx * L_momentum_x +     ⭐ 包含外力項
         w_my * L_momentum_y +     （無外力項）
         w_cont * L_continuity +   （無外力項）
         w_bc * L_boundary
```

---

### ❓ Q5: 如何驗證外力項是否正確加入？

**A**: 使用層流解驗證：

```python
# 層流解應滿足：∂u/∂t + ... = ν∇²u + A sin(k_f y)
# 當 u = (A/(νk²))sin(k_f y), v=0 時，殘差應≈0

# 驗證腳本
python -c "
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
import torch
import numpy as np

physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 1.0, 'wavenumber': 8},
    physics_params={'nu': 0.0125}
)

# 層流解參數
A, nu, k = 1.0, 0.0125, 8.0
U0 = A / (nu * k**2)

# 測試點
y = torch.linspace(0, 2*np.pi, 100, requires_grad=True).view(-1, 1)
x = torch.zeros_like(y)
coords = torch.cat([x, y], dim=1)

# 層流解
u = U0 * torch.sin(k * y)
v = torch.zeros_like(y)
p = torch.zeros_like(y)
pred = torch.cat([u, v, p], dim=1)

# 計算殘差
res = physics.residual_unified(coords, pred)

print(f'Momentum X 殘差: {res[\"momentum_x\"].abs().mean():.6f} (應≈0)')
print(f'Momentum Y 殘差: {res[\"momentum_y\"].abs().mean():.6f} (應≈0)')
print(f'Continuity 殘差: {res[\"continuity\"].abs().mean():.6f} (應≈0)')
"
```

**預期輸出**（正確實現）：
```
Momentum X 殘差: 0.000001 (應≈0) ✅
Momentum Y 殘差: 0.000000 (應≈0) ✅
Continuity 殘差: 0.000000 (應≈0) ✅
```

---

### ❓ Q6: 外力項對訓練有何影響？

**A**: 外力項提供**額外的物理約束**：

| 影響 | 說明 |
|------|------|
| **收斂速度** | ✅ 提供明確的驅動源，加速收斂 |
| **物理正確性** | ✅ 確保模型學習到正確的流動模式 |
| **損失平衡** | ⚠️ 需要合適的權重（momentum_x_weight） |
| **穩定性** | ✅ 外力穩定，不會引入噪聲 |

---

### ❓ Q7: 可以訓練時學習外力參數嗎？

**A**: **可以，但不推薦**（除非是逆問題）。

```python
# 如果需要學習外力參數（逆問題）
self.amplitude = nn.Parameter(torch.tensor(1.0))  # 可學習
self.wavenumber = nn.Parameter(torch.tensor(8.0))  # 可學習

# 但需要額外的正則化
loss_reg = (self.amplitude - A_true)**2 + (self.wavenumber - k_true)**2
```

**標準 Kolmogorov flow 訓練**：外力參數應該是**已知的固定值**。

---

## 總結

### ✅ 外力項實現要點

1. **定義位置**: `pinnx/physics/kolmogorov_flow_2d.py`
2. **計算函數**: `compute_forcing_term(coords)`
3. **整合位置**: `compute_momentum_residuals()` 中的 x-動量方程
4. **參數類型**: `register_buffer`（固定值，不可學習）
5. **影響範圍**: 僅 x 方向動量損失

### 📋 檢查清單

訓練前確認：
- [ ] 配置文件中設定了 `physics.forcing.amplitude`
- [ ] 配置文件中設定了 `physics.forcing.k_f`
- [ ] 雷諾數計算正確（使用 `calculate_reynolds_parameters.py`）
- [ ] 物理模組初始化成功
- [ ] 外力參數與 DNS 數據一致（如果有）
- [ ] 層流解驗證通過

### 🔗 相關文檔

- **雷諾數計算**: `scripts/README_REYNOLDS_CALCULATOR.md`
- **物理驗證**: `KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`
- **配置模板**: `configs/templates/`

---

**最後更新**: 2025-11-25
**維護者**: PINNs-MVP 團隊
