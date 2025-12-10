# PINNs-MVP Loss Terms 完整參考手冊

> **文檔版本**: v1.0  
> **更新日期**: 2025-12-04  
> **適用專案**: 少量資料 × 物理先驗 PINNs 湍流逆重建

---

## 📋 目錄

1. [總覽](#總覽)
2. [物理殘差損失 (PDE Residual Losses)](#物理殘差損失-pde-residual-losses)
3. [資料一致性損失 (Data Consistency Losses)](#資料一致性損失-data-consistency-losses)
4. [邊界條件損失 (Boundary Condition Losses)](#邊界條件損失-boundary-condition-losses)
5. [先驗一致性損失 (Prior Consistency Losses)](#先驗一致性損失-prior-consistency-losses)
6. [正則化損失 (Regularization Losses)](#正則化損失-regularization-losses)
7. [權重策略 (Weighting Strategies)](#權重策略-weighting-strategies)
8. [總損失函數組合](#總損失函數組合)

---

## 總覽

### Loss Terms 分類

| 類別 | 損失項數量 | 主要功能 | 檔案位置 |
|------|-----------|---------|---------|
| **物理殘差** | 4-7 項 | 強制滿足 NS 方程 | `pinnx/losses/residuals.py` |
| **資料一致性** | 3-5 項 | 擬合稀疏觀測資料 | `pinnx/train/trainer.py` |
| **邊界條件** | 2-6 項 | 強制邊界約束 | `pinnx/losses/residuals.py` |
| **先驗一致性** | 1-4 項 | 整合低保真場 | `pinnx/losses/priors.py` |
| **正則化** | 1-3 項 | 避免過擬合/源項稀疏化 | `pinnx/losses/residuals.py` |
| **動態權重** | 多策略 | 平衡多項損失 | `pinnx/losses/weighting.py` |

### 符號約定

| 符號 | 含義 | 單位 |
|------|------|------|
| $\mathbf{u} = (u, v, w)$ | 速度場（2D 時為 $(u, v)$） | m/s |
| $p$ | 壓力場 | Pa |
| $\mathbf{S} = (S_x, S_y, S_z)$ | 源項（待辨識） | m/s² |
| $\nu$ | 動力黏性係數 | m²/s |
| $\rho$ | 密度 | kg/m³ |
| $\mathbf{x} = (x, y, z)$ | 空間座標 | m |
| $t$ | 時間 | s |
| $\Omega$ | 計算域 | - |
| $\partial\Omega$ | 邊界 | - |

---

## 物理殘差損失 (PDE Residual Losses)

### 1. 動量方程殘差 (Momentum Residual)

#### **2D 不可壓縮 NS 方程**

**X 方向動量**:
$$
\mathcal{L}_{\text{momentum}_x} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial x} - \nu\nabla^2 u - S_x \right|^2
$$

**Y 方向動量**:
$$
\mathcal{L}_{\text{momentum}_y} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \frac{\partial v}{\partial t} + u\frac{\partial v}{\partial x} + v\frac{\partial v}{\partial y} + \frac{1}{\rho}\frac{\partial p}{\partial y} - \nu\nabla^2 v - S_y \right|^2
$$

**程式實現** (`pinnx/losses/residuals.py:177-178`):
```python
momentum_x = u_t + (u * ux + v * uy) + px / density - nu * u_lap - sx
momentum_y = v_t + (u * vx + v * vy) + py / density - nu * v_lap - sy
```

#### **3D 不可壓縮 NS 方程**

**Z 方向動量**（額外項）:
$$
\mathcal{L}_{\text{momentum}_z} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \frac{\partial w}{\partial t} + u\frac{\partial w}{\partial x} + v\frac{\partial w}{\partial y} + w\frac{\partial w}{\partial z} + \frac{1}{\rho}\frac{\partial p}{\partial z} - \nu\nabla^2 w - S_z \right|^2
$$

**程式實現** (`pinnx/losses/residuals.py:242-244`):
```python
momentum_x = u_t + (u * ux + v * uy + w * uz) + px / density - nu * u_lap - sx
momentum_y = v_t + (u * vx + v * vy + w * vz) + py / density - nu * v_lap - sy
momentum_z = w_t + (u * wx + v * wy + w * wz) + pz / density - nu * w_lap - sz
```

---

### 2. 連續性方程殘差 (Continuity Residual)

**不可壓縮條件**:
$$
\mathcal{L}_{\text{continuity}} = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \nabla \cdot \mathbf{u} \right|^2 = \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left| \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} + \frac{\partial w}{\partial z} \right|^2
$$

**程式實現** (`pinnx/losses/residuals.py:181-247`):
```python
# 2D
continuity = ux + vy

# 3D
continuity = ux + vy + wz
```

---

### 3. VS-PINN 變數縮放修正

**各向異性縮放變換**:
$$
(X, Y, Z) = (N_x \cdot x, N_y \cdot y, N_z \cdot z)
$$

**梯度變換** (鏈式法則):
$$
\frac{\partial u}{\partial x} = N_x \frac{\partial v}{\partial X}, \quad \frac{\partial^2 u}{\partial x^2} = N_x^2 \frac{\partial^2 v}{\partial X^2}
$$

**Laplacian 變換**:
$$
\nabla^2 u = N_x^2 \frac{\partial^2 v}{\partial X^2} + N_y^2 \frac{\partial^2 v}{\partial Y^2} + N_z^2 \frac{\partial^2 v}{\partial Z^2}
$$

**推薦縮放因子** (通道流 Re_τ=1000):
- $N_x = 2.0$ (流向，週期性)
- $N_y = 12.0$ (壁法向，最剛性)
- $N_z = 2.0$ (展向，週期性)

**程式實現** (`pinnx/physics/vs_pinn_channel_flow.py:150-179`):
```python
self.scaling_factors = {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}
```

---

### 4. 源項正則化 (Source Regularization)

**L1 稀疏化**:
$$
\mathcal{L}_{\text{source}} = \lambda_{\text{reg}} \frac{1}{N_{\text{pde}}} \sum_{i=1}^{N_{\text{pde}}} \left( |S_{x,i}| + |S_{y,i}| + |S_{z,i}| \right)
$$

**參數設定**:
- $\lambda_{\text{reg}} = 10^{-6}$ 至 $10^{-4}$ (可調)

**程式實現** (`pinnx/losses/residuals.py:359-360`):
```python
source_l1 = torch.mean(torch.abs(source))
losses['source_l1'] = default_weights['source_reg'] * source_l1
```

---

## 資料一致性損失 (Data Consistency Losses)

### 5. 速度場擬合損失 (Velocity Data Loss)

**U 分量**:
$$
\mathcal{L}_{\text{data}_u} = \frac{1}{N_{\text{sensor}}} \sum_{i=1}^{N_{\text{sensor}}} \left| u_{\text{pred}}(\mathbf{x}_i) - u_{\text{obs}}(\mathbf{x}_i) \right|^2
$$

**V 分量**:
$$
\mathcal{L}_{\text{data}_v} = \frac{1}{N_{\text{sensor}}} \sum_{i=1}^{N_{\text{sensor}}} \left| v_{\text{pred}}(\mathbf{x}_i) - v_{\text{obs}}(\mathbf{x}_i) \right|^2
$$

**W 分量** (3D):
$$
\mathcal{L}_{\text{data}_w} = \frac{1}{N_{\text{sensor}}} \sum_{i=1}^{N_{\text{sensor}}} \left| w_{\text{pred}}(\mathbf{x}_i) - w_{\text{obs}}(\mathbf{x}_i) \right|^2
$$

**程式實現** (`pinnx/train/trainer.py:450-452`):
```python
u_loss = torch.mean((u_sensors_pred_phys[:, 0:1] - u_true)**2)
v_loss = torch.mean((u_sensors_pred_phys[:, 1:2] - v_true)**2)
w_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - w_true)**2)  # 3D only
```

---

### 6. 壓力場擬合損失 (Pressure Data Loss)

$$
\mathcal{L}_{\text{data}_p} = \frac{1}{N_{\text{sensor}}} \sum_{i=1}^{N_{\text{sensor}}} \left| p_{\text{pred}}(\mathbf{x}_i) - p_{\text{obs}}(\mathbf{x}_i) \right|^2
$$

**程式實現** (`pinnx/train/trainer.py:455`):
```python
pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
```

---

### 7. 噪聲加權資料損失 (Noise-Weighted Data Loss)

**不確定性權重**:
$$
\mathcal{L}_{\text{data}}^{\text{weighted}} = \frac{1}{N_{\text{sensor}}} \sum_{i=1}^{N_{\text{sensor}}} \frac{1}{\sigma_i^2 + \epsilon} \left| \mathbf{u}_{\text{pred}}(\mathbf{x}_i) - \mathbf{u}_{\text{obs}}(\mathbf{x}_i) \right|^2
$$

- $\sigma_i$: 觀測噪聲標準差
- $\epsilon = 10^{-8}$: 數值穩定項

**程式實現** (`pinnx/losses/residuals.py:886-890`):
```python
if noise_std is not None:
    weights = 1.0 / (noise_std.unsqueeze(-1) ** 2 + 1e-8)
    loss = torch.mean(weights * (residual ** 2))
```

---

## 邊界條件損失 (Boundary Condition Losses)

### 8. 無滑移邊界 (No-Slip Wall)

**壁面條件** ($y = \pm 1$):
$$
\mathcal{L}_{\text{wall}} = \frac{1}{N_{\text{wall}}} \sum_{i=1}^{N_{\text{wall}}} \left( |u_i|^2 + |v_i|^2 + |w_i|^2 \right)
$$

**程式實現** (`pinnx/train/trainer.py:475`):
```python
wall_loss = torch.mean(u_wall**2 + v_wall**2)
```

---

### 9. 週期邊界條件 (Periodic Boundary)

**X 方向週期性**:
$$
\mathcal{L}_{\text{periodic}_x} = \frac{1}{N_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \left| \mathbf{u}(x_{\min}, y_i, z_i) - \mathbf{u}(x_{\max}, y_i, z_i) \right|^2
$$

**Y 方向週期性** (如適用):
$$
\mathcal{L}_{\text{periodic}_y} = \frac{1}{N_{\text{bc}}} \sum_{i=1}^{N_{\text{bc}}} \left| \mathbf{u}(x_i, y_{\min}, z_i) - \mathbf{u}(x_i, y_{\max}, z_i) \right|^2
$$

**程式實現** (`pinnx/train/trainer.py:494-509`):
```python
periodic_x_loss = torch.mean((pred_left - pred_right) ** 2)
periodic_y_loss = torch.mean((pred_bottom - pred_top) ** 2)
```

**週期邊界詳細實現** (`pinnx/losses/residuals.py:569-695`):
```python
class PeriodicBoundaryLoss(nn.Module):
    def forward(self, coords_pair_1, coords_pair_2, predictions_1, predictions_2):
        # 速度場週期性
        vel_1 = predictions_1[:, :spatial_dim]
        vel_2 = predictions_2[:, :spatial_dim]
        losses['periodic_velocity'] = torch.mean((vel_1 - vel_2) ** 2)
        
        # 壓力場週期性
        p_1 = predictions_1[:, spatial_dim]
        p_2 = predictions_2[:, spatial_dim]
        losses['periodic_pressure'] = torch.mean((p_1 - p_2) ** 2)
```

---

### 10. Inlet 速度剖面損失 (Inlet Profile Loss)

**拋物線剖面** (層流):
$$
u_{\text{target}}(y) = u_{\max} \left(1 - \left(\frac{y}{h}\right)^2\right)
$$

**對數律剖面** (湍流):
$$
u^+(y^+) = \frac{1}{\kappa} \ln(y^+) + C^+
$$
- $\kappa = 0.41$ (von Karman 常數)
- $C^+ = 5.0$ (對數律常數)
- $y^+ = \frac{Re_\tau \cdot |y|}{h}$

**損失函數**:
$$
\mathcal{L}_{\text{inlet}} = \frac{1}{N_{\text{inlet}}} \sum_{i=1}^{N_{\text{inlet}}} \left( |u_i - u_{\text{target}}(y_i)|^2 + |v_i|^2 \right)
$$

**程式實現** (`pinnx/losses/residuals.py:425-503`):
```python
def inlet_velocity_profile_loss(self, inlet_coords, inlet_predictions, 
                                profile_type='log_law', Re_tau=1000.0):
    if profile_type == 'log_law':
        y_plus = Re_tau * y_abs / h
        u_plus = (1.0 / kappa) * torch.log(y_plus) + C
        u_target = u_tau * u_plus
    
    u_loss = torch.mean((u_pred - u_target) ** 2)
    v_loss = torch.mean(v_pred ** 2)
    return u_loss + v_loss
```

---

### 11. 初始條件損失 (Initial Condition Loss)

**非定常問題** ($t=0$):
$$
\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{i=1}^{N_{\text{IC}}} \left| \mathbf{u}(\mathbf{x}_i, t=0) - \mathbf{u}_0(\mathbf{x}_i) \right|^2
$$

**程式實現** (`pinnx/losses/residuals.py:543-549`):
```python
vel_pred = initial_predictions[:, :spatial_dim]
vel_true = initial_data[:, :spatial_dim]
losses['ic_velocity'] = torch.mean((vel_pred - vel_true) ** 2)

p_pred = initial_predictions[:, spatial_dim]
p_true = initial_data[:, spatial_dim]
losses['ic_pressure'] = torch.mean((p_pred - p_true) ** 2)
```

---

## 先驗一致性損失 (Prior Consistency Losses)

### 12. 低保真場一致性 (Low-Fidelity Consistency)

**RANS/LES 軟約束**:
$$
\mathcal{L}_{\text{prior}} = \lambda_{\text{prior}} \sum_{j \in \{u,v,w,p\}} w_j \frac{1}{N} \sum_{i=1}^{N} \left| \phi_j^{\text{HF}}(\mathbf{x}_i) - \phi_j^{\text{LF}}(\mathbf{x}_i) \right|^2
$$

- $\phi_j^{\text{HF}}$: PINNs 高保真預測
- $\phi_j^{\text{LF}}$: RANS/LES 低保真參考場
- $w_j$: 變數權重 (通常 $w_u = w_v = 1.0$, $w_p = 0.3 \sim 0.5$)

**距離度量選項**:
- MSE: `'mse'` → $\|\cdot\|^2$
- MAE: `'mae'` → $|\cdot|$
- Huber: `'huber'` → 結合 MSE 與 MAE 的魯棒損失

**程式實現** (`pinnx/losses/priors.py:101-119`):
```python
if self.distance_metric == 'mse':
    var_loss = torch.mean((pred_var - ref_var) ** 2)
elif self.distance_metric == 'mae':
    var_loss = torch.mean(torch.abs(pred_var - ref_var))
elif self.distance_metric == 'huber':
    var_loss = F.huber_loss(pred_var, ref_var, reduction='mean')

weighted_loss = var_weight * var_loss
losses[f'prior_consistency_{var_name}'] = weighted_loss
```

---

### 13. 統計矩一致性 (Statistical Consistency)

**均值約束** (1階矩):
$$
\mathcal{L}_{\text{stat}_1} = \sum_{j} \left| \frac{1}{N}\sum_{i=1}^{N} \phi_j(\mathbf{x}_i) - \langle \phi_j \rangle_{\text{ref}} \right|^2
$$

**方差約束** (2階矩):
$$
\mathcal{L}_{\text{stat}_2} = \sum_{j} \left| \frac{1}{N}\sum_{i=1}^{N} \phi_j^2(\mathbf{x}_i) - \langle \phi_j^2 \rangle_{\text{ref}} \right|^2
$$

**程式實現** (`pinnx/losses/priors.py:198-204`):
```python
for moment in self.moments:
    pred_moment = torch.mean(pred_var ** moment)
    ref_key = f'{var_name}_moment_{moment}'
    if ref_key in reference_stats:
        ref_moment = reference_stats[ref_key]
        moment_loss = torch.mean((pred_moment - ref_moment) ** 2)
        losses[f'stat_{var_name}_moment_{moment}'] = moment_loss
```

---

### 14. 守恆定律一致性 (Conservation Loss)

**質量守恆** (域積分):
$$
\mathcal{L}_{\text{conserve}_{\text{mass}}} = \left| \int_\Omega \nabla \cdot \mathbf{u} \, d\Omega \right|^2 \approx \left| \frac{1}{N}\sum_{i=1}^{N} (\nabla \cdot \mathbf{u})_i \right|^2
$$

**能量守恆** (穩態):
$$
\mathcal{L}_{\text{conserve}_{\text{energy}}} = \frac{\text{Var}(E_{\text{kinetic}})}{\langle E_{\text{kinetic}} \rangle + \epsilon}
$$

其中 $E_{\text{kinetic}} = \frac{1}{2}|\mathbf{u}|^2$

**程式實現** (`pinnx/losses/priors.py:239-293`):
```python
def mass_conservation_loss(self, velocity, coords):
    div_u = 0.0
    for i in range(velocity.shape[-1]):
        u_i = velocity[:, i]
        grad = torch.autograd.grad(u_i.sum(), coords, ...)[0]
        if grad is not None:
            div_u += grad[:, i]
    
    if self.domain_integration:
        integrated_div = torch.mean(div_u)
        loss = integrated_div ** 2
    else:
        loss = torch.mean(div_u ** 2)
    return loss

def energy_conservation_loss(self, velocity, coords):
    kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=-1)
    mean_energy = torch.mean(kinetic_energy)
    energy_variance = torch.var(kinetic_energy)
    loss = energy_variance / (mean_energy + 1e-8)
    return loss
```

---

### 15. 對稱性約束 (Symmetry Consistency)

**反射對稱** (如適用):
$$
\mathcal{L}_{\text{symmetry}} = \frac{1}{N} \sum_{i=1}^{N} \left| \mathbf{u}(\mathbf{x}_i) - \mathbf{u}(\mathbf{R}\mathbf{x}_i) \right|^2
$$

- $\mathbf{R}$: 反射變換算子

**程式實現** (`pinnx/losses/priors.py:350-364`):
```python
def reflection_loss(self, coords, predictions, axis=0):
    # 生成反射座標
    reflected_coords = coords.clone()
    reflected_coords[:, axis] = -reflected_coords[:, axis]
    
    # 對稱性損失
    loss = torch.mean((predictions - reflected_preds) ** 2)
    return loss
```

---

## 正則化損失 (Regularization Losses)

### 16. 物理約束正則化 (Physics Constraint Regularization)

**能量界限約束**:
$$
\mathcal{L}_{\text{energy}_{\text{bound}}} = \lambda_{\text{reg}} \frac{1}{N} \sum_{i=1}^{N} \max\left(0, E_{\text{kinetic},i} - E_{\max}\right)^2
$$

**場量值界限約束**:
$$
\mathcal{L}_{\text{magnitude}_{\text{bound}}} = \lambda_{\text{reg}} \frac{1}{N} \sum_{i=1}^{N} \max\left(0, |\mathbf{u}_i| - U_{\max}\right)^2
$$

**程式實現** (`pinnx/losses/priors.py:641-663`):
```python
def physics_constraint_loss(field, constraint_type='energy_bound', 
                           constraint_params=None, strength=1.0):
    if constraint_type == 'energy_bound':
        kinetic_energy = 0.5 * torch.sum(field ** 2, dim=-1)
        max_energy = constraint_params.get('max_energy', 10.0)
        excess_energy = torch.clamp(kinetic_energy - max_energy, min=0.0)
        loss = torch.mean(excess_energy ** 2)
    
    elif constraint_type == 'magnitude_bound':
        field_magnitude = torch.norm(field, dim=-1)
        max_magnitude = constraint_params.get('max_magnitude', 5.0)
        excess_magnitude = torch.clamp(field_magnitude - max_magnitude, min=0.0)
        loss = torch.mean(excess_magnitude ** 2)
    
    return strength * loss
```

---

### 17. SDF 幾何權重 (Signed Distance Function Weighting)

**距離基權重函數** (指數型):
$$
w(\mathbf{x}) = w_{\text{interior}} + (w_{\text{boundary}} - w_{\text{interior}}) \cdot \exp\left(-\beta \frac{d(\mathbf{x})}{\delta}\right)
$$

- $d(\mathbf{x})$: 到邊界的符號距離
- $\delta$: 邊界層寬度 (預設 0.15)
- $\beta$: 衰減率 (預設 1.5)
- $w_{\text{boundary}} = 3.0$, $w_{\text{interior}} = 1.0$

**通道流 SDF** (壁面主導):
$$
d_{\text{wall}}(y) = 1 - |y|
$$

**程式實現** (`pinnx/losses/sdf_weights.py:130-148`):
```python
def _sdf_channel_3d(self, coords):
    x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
    
    # 壁面距離（主要約束）
    wall_distance = 1.0 - torch.abs(y)
    
    # 週期邊界軟約束
    x_dist = torch.minimum(x - self.domain_bounds['x_min'], 
                          self.domain_bounds['x_max'] - x)
    z_dist = torch.minimum(z - self.domain_bounds['z_min'], 
                          self.domain_bounds['z_max'] - z)
    
    sdf = wall_distance - 0.1 * torch.minimum(torch.minimum(x_dist, z_dist), 
                                             torch.zeros_like(x_dist))
    return sdf

def _exponential_weights(self, distance):
    weights = torch.ones_like(distance) * self.weight_params['interior_weight']
    mask = distance <= self.weight_params['boundary_width']
    weights[mask] = self.weight_params['interior_weight'] + \
                    (self.weight_params['boundary_weight'] - 
                     self.weight_params['interior_weight']) * \
                    torch.exp(-self.weight_params['decay_rate'] * 
                             distance[mask] / self.weight_params['boundary_width'])
    return weights
```

---

## 權重策略 (Weighting Strategies)

### 18. GradNorm 動態權重 (GradNorm Adaptive Weighting)

**目標**: 平衡不同損失項對模型參數的梯度影響

**權重更新規則**:
$$
w_i^{(t+1)} = w_i^{(t)} \cdot \left(\frac{\bar{G}(t) \cdot r_i(t)}{G_i(t)}\right)^\alpha
$$

其中:
- $w_i$: 第 $i$ 項損失的權重
- $G_i = \|\nabla_{w_i \mathcal{L}_i} \theta\|_2$: 第 $i$ 項損失的梯度範數
- $\bar{G} = \frac{1}{K}\sum_{i=1}^K G_i$: 平均梯度範數
- $r_i = \mathcal{L}_i / \mathcal{L}_i^{(0)}$: 相對損失比例
- $\alpha = 1.5$: 梯度平衡更新率

**權重歸一化**:
$$
\sum_{i=1}^{K} w_i = \text{const} \quad \text{(保持權重總和守恆)}
$$

**比例限制**:
$$
\frac{\max(w_i)}{\min(w_i)} \leq \text{max\_ratio} = 50.0
$$

**程式實現** (`pinnx/losses/weighting.py:125-246`):
```python
def compute_gradients(self, losses):
    gradients = {}
    for name, loss in losses.items():
        weighted_loss = loss * self.weights[name]
        grads = torch.autograd.grad(
            outputs=weighted_loss,
            inputs=list(self.model.parameters()),
            retain_graph=True, create_graph=False
        )
        grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads if g is not None))
        gradients[name] = grad_norm
    return gradients

def update_weights(self, losses, total_loss=None):
    gradients = self.compute_gradients(losses)
    avg_grad = torch.mean(torch.stack(list(gradients.values())))
    
    for name in self.loss_names:
        target_grad = avg_grad * self.target_distribution[name]
        gradient_ratio = gradients[name] / (target_grad + self.eps)
        loss_ratio = losses[name].detach() / (self.initial_losses[name] + self.eps)
        
        weight_adjustment = (gradient_ratio * loss_ratio).pow(-self.alpha)
        weight_adjustment = torch.clamp(weight_adjustment, 0.5, 2.0)
        
        self.weights[name] = torch.clamp(
            self.weights[name] * weight_adjustment,
            self.min_weight, self.max_weight
        )
    
    self._normalize_weights()
```

**推薦配置**:
```yaml
adaptive_weights:
  method: gradnorm
  alpha: 1.5
  update_frequency: 1000
  min_weight: 0.1
  max_weight: 10.0
  max_ratio: 50.0
```

---

### 19. 時間因果權重 (Causal Weighting)

**核心機制** (Wang et al. 2022):
$$
w(t_i) = \exp\left(-\epsilon \int_0^{t_i} \mathcal{L}_{\text{PDE}}(\tau) \, d\tau\right)
$$

**離散近似** (時間分窗):
$$
w(t_i) \approx \exp\left(-\epsilon \sum_{k=0}^{k_i} \bar{\mathcal{L}}_k \cdot \Delta t\right)
$$

- $\epsilon$: 因果容差參數 (預設 1.0)
- $\bar{\mathcal{L}}_k$: 第 $k$ 個時間窗的平均 PDE 損失
- $\Delta t = (t_{\max} - t_{\min}) / N_{\text{bins}}$

**程式實現** (`pinnx/losses/weighting.py:368-426`):
```python
def compute_weights(self, pde_residuals, time_coords):
    # 1. 確定每個點所屬的 bin 索引
    t_norm = (time_coords - self.t_min) / (self.t_max - self.t_min)
    bin_indices = (t_norm * self.n_time_bins).long()
    
    # 2. 計算每個 bin 的平均損失
    bin_loss_sum = torch.zeros(self.n_time_bins, device=device)
    bin_counts = torch.zeros(self.n_time_bins, device=device)
    bin_loss_sum.index_add_(0, bin_indices, pde_residuals.detach().flatten())
    bin_counts.index_add_(0, bin_indices, torch.ones_like(pde_residuals.flatten()))
    bin_means = bin_loss_sum / (bin_counts + 1e-8)
    
    # 3. 計算累積損失（近似積分）
    cumulative_loss = torch.cumsum(bin_means, dim=0)
    cumulative_prev = torch.roll(cumulative_loss, 1)
    cumulative_prev[0] = 0.0
    
    # 4. 計算權重
    bin_weights = torch.exp(-self.epsilon * cumulative_prev)
    point_weights = bin_weights[bin_indices]
    
    return point_weights.unsqueeze(1)
```

**推薦配置**:
```yaml
causal_weighting:
  enabled: true
  epsilon: 1.0
  n_time_bins: 10
```

---

### 20. 損失歸一化 (Loss Normalization)

**目標**: 確保不同物理量級的損失項具有相同的數值範圍

**參考值計算** (滑動平均):
$$
\mathcal{L}_{\text{ref}, i}^{(t)} = \gamma \cdot \mathcal{L}_{\text{ref}, i}^{(t-1)} + (1-\gamma) \cdot \mathcal{L}_i^{(t)}
$$

- $\gamma = 0.9$: 動量係數 (滑動平均)

**歸一化損失**:
$$
\tilde{\mathcal{L}}_i = \frac{\mathcal{L}_i}{\mathcal{L}_{\text{ref}, i} + \epsilon}
$$

**Warmup 階段** (前 5 epochs):
- 不進行歸一化，直接累積參考值
- 避免初始損失波動導致歸一化失效

**程式實現** (`pinnx/physics/vs_pinn_channel_flow.py:400-430`):
```python
def normalize_loss(self, loss_name, loss_value, current_epoch):
    if current_epoch < self.warmup_epochs:
        # Warmup 階段：記錄但不歸一化
        if loss_name not in self.loss_normalizers:
            self.loss_normalizers[loss_name] = loss_value.detach().item()
        return loss_value
    
    # 更新參考值（滑動平均）
    if loss_name in self.loss_normalizers:
        self.loss_normalizers[loss_name] = (
            self.normalizer_momentum * self.loss_normalizers[loss_name] +
            (1 - self.normalizer_momentum) * loss_value.detach().item()
        )
    else:
        self.loss_normalizers[loss_name] = loss_value.detach().item()
    
    # 歸一化
    normalizer = self.loss_normalizers[loss_name] + 1e-8
    return loss_value / normalizer
```

---

### 21. 自適應殘差點權重 (Adaptive Collocation Point Weighting)

**新點權重衰減** (線性):
$$
w_{\text{new}}(\text{epoch}) = w_{\text{final}} + (w_{\text{initial}} - w_{\text{final}}) \cdot \max\left(0, 1 - \frac{\text{epoch} - \text{epoch}_{\text{resample}}}{\text{decay}_{\text{epochs}}}\right)
$$

**參數設定**:
- $w_{\text{initial}} = 2.5$: 新點初始權重
- $w_{\text{final}} = 1.0$: 衰減後權重
- $\text{decay}_{\text{epochs}} = 500$: 衰減週期

**程式實現** (`pinnx/train/loop.py:197-234`):
```python
def get_point_weights(self, epoch):
    epochs_since_resample = epoch - self.last_resample_epoch
    if epochs_since_resample >= self.new_point_decay_epochs:
        decay_factor = 0.0
    else:
        decay_factor = 1.0 - (epochs_since_resample / self.new_point_decay_epochs)
    
    weights = torch.ones(len(self.new_point_mask))
    new_point_weight = (
        self.new_point_final_weight +
        (self.new_point_initial_weight - self.new_point_final_weight) * decay_factor
    )
    weights[self.new_point_mask] = new_point_weight
    
    return weights
```

---

## 總損失函數組合

### 完整損失函數

$$
\begin{aligned}
\mathcal{L}_{\text{total}} = & \underbrace{w_{\text{data}} \left( \mathcal{L}_{\text{data}_u} + \mathcal{L}_{\text{data}_v} + \mathcal{L}_{\text{data}_w} + \mathcal{L}_{\text{data}_p} \right)}_{\text{資料一致性}} \\
& + \underbrace{w_{\text{pde}} \left( \mathcal{L}_{\text{momentum}_x} + \mathcal{L}_{\text{momentum}_y} + \mathcal{L}_{\text{momentum}_z} + \mathcal{L}_{\text{continuity}} \right)}_{\text{物理殘差}} \\
& + \underbrace{w_{\text{bc}} \left( \mathcal{L}_{\text{wall}} + \mathcal{L}_{\text{periodic}_x} + \mathcal{L}_{\text{periodic}_y} \right)}_{\text{邊界條件}} \\
& + \underbrace{w_{\text{prior}} \mathcal{L}_{\text{prior}}}_{\text{先驗一致性}} \\
& + \underbrace{w_{\text{reg}} \mathcal{L}_{\text{source}}}_{\text{正則化}}
\end{aligned}
$$

---

### 推薦權重配置

#### **階段 1: Warmup (Epochs 0-1000)**
```yaml
weights:
  data_u: 1.0
  data_v: 1.0
  data_w: 1.0
  data_p: 0.5
  pde_momentum_x: 0.1
  pde_momentum_y: 0.1
  pde_momentum_z: 0.1
  pde_continuity: 0.5
  bc_wall: 1.0
  bc_periodic_x: 0.3
  bc_periodic_y: 0.3
  prior_consistency: 0.1
  source_reg: 1e-6
```

#### **階段 2: Main Training (Epochs 1000-5000)**
使用 GradNorm 自適應調整，初始權重：
```yaml
weights:
  data: 1.0
  pde: 1.0
  bc: 0.5
  prior: 0.3
  reg: 1e-5
```

#### **階段 3: Refinement (Epochs 5000+)**
```yaml
weights:
  data: 0.5
  pde: 2.0  # 強化物理一致性
  bc: 1.0
  prior: 0.1  # 降低低保真依賴
  reg: 1e-4
```

---

### 配置範例

**標準配置** (`configs/channel_flow_re1000_K80.yml`):
```yaml
losses:
  # 資料一致性
  data_u: 1.0
  data_v: 1.0
  data_w: 1.0
  data_p: 0.5
  
  # 物理殘差
  pde_momentum_x: 1.0
  pde_momentum_y: 1.0
  pde_momentum_z: 1.0
  pde_continuity: 1.0
  
  # 邊界條件
  bc_wall: 1.0
  bc_periodic_x: 0.3
  bc_periodic_y: 0.3
  
  # 先驗與正則化
  prior_consistency: 0.1
  source_reg: 1e-6
  
  # 損失歸一化
  normalize_losses: true
  warmup_epochs: 5
  
  # 動態權重
  adaptive_weights:
    enabled: true
    method: gradnorm
    alpha: 1.5
    update_frequency: 1000
```

---

## 📊 Loss Terms 快速查詢表

| Loss Term | 公式簡記 | 典型權重 | 檔案位置 |
|-----------|---------|---------|---------|
| `data_u` | $\|\mathbf{u}_{\text{pred}} - \mathbf{u}_{\text{obs}}\|^2$ | 1.0 | `trainer.py:450` |
| `data_v` | $\|v_{\text{pred}} - v_{\text{obs}}\|^2$ | 1.0 | `trainer.py:451` |
| `data_w` | $\|w_{\text{pred}} - w_{\text{obs}}\|^2$ | 1.0 | `trainer.py:452` |
| `data_p` | $\|p_{\text{pred}} - p_{\text{obs}}\|^2$ | 0.5 | `trainer.py:455` |
| `pde_momentum_x` | NS X 方向殘差 | 1.0 | `residuals.py:177` |
| `pde_momentum_y` | NS Y 方向殘差 | 1.0 | `residuals.py:178` |
| `pde_momentum_z` | NS Z 方向殘差 | 1.0 | `residuals.py:244` |
| `pde_continuity` | $\|\nabla \cdot \mathbf{u}\|^2$ | 1.0 | `residuals.py:181` |
| `bc_wall` | $\|\mathbf{u}_{\text{wall}}\|^2$ | 1.0 | `trainer.py:475` |
| `bc_periodic_x` | X 週期性 | 0.3 | `trainer.py:494` |
| `bc_periodic_y` | Y 週期性 | 0.3 | `trainer.py:509` |
| `prior_consistency` | $\|\phi^{\text{HF}} - \phi^{\text{LF}}\|^2$ | 0.1-0.5 | `priors.py:101` |
| `source_reg` | $\|\mathbf{S}\|_1$ | 1e-6 | `residuals.py:359` |
| `conservation_mass` | $\|\int \nabla \cdot \mathbf{u}\|^2$ | 0.3 | `priors.py:256` |
| `conservation_energy` | $\text{Var}(E) / \langle E \rangle$ | 0.1 | `priors.py:286` |

---

## 🔗 相關文檔

- [技術文檔](TECHNICAL_DOCUMENTATION.md)
- [配置指南](CONFIG_GUIDE.md)
- [訓練指南](../scripts/README.md)
- [物理驗證指南](DNS_VALIDATION_GUIDE.md)

---

## 📝 變更記錄

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| v1.0 | 2025-12-04 | 初始版本，完整整理所有 loss terms |

---

**文檔維護**: PINNs-MVP 團隊  
**最後更新**: 2025-12-04
