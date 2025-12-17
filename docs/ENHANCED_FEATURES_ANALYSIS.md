# 湍流特徵擴充策略
## 目標：提升 QR-Pivot 感測器的資訊密度與可識別性

---

## 📊 當前特徵集（20 features）

### Primary Flow (4)
- u, v, w, p

### Velocity Gradients (6)
- ∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y

### Pressure Gradients (2)
- ∂p/∂x, ∂p/∂y

### Vorticity (1)
- ω_z = ∂v/∂x - ∂u/∂y

### Turbulence Quantities (5)
- k (TKE)
- τ_uu, τ_vv, τ_ww, τ_uv (Reynolds stresses)

### Velocity Gradient Tensor Eigenvalues (2)
- λ_1, λ_2 (2D)

---

## ✨ 新增特徵候選（分類）

### A. 渦動力學特徵 (Vortex Dynamics)

#### A1. 完整渦度分量 [2D: +0, 3D: +2]
```python
# 2D 只有 ω_z，3D 需要完整三分量
omega_x = ∂w/∂y - ∂v/∂z  # (not available in 2D slice)
omega_y = ∂u/∂z - ∂w/∂x  # (not available in 2D slice)
omega_z = ∂v/∂x - ∂u/∂y  # ✅ 已有
```
**2D 限制**：僅 z 軸切片，無法計算 x/y 方向渦度

#### A2. 渦量 (Enstrophy) [+1]
```python
enstrophy = 0.5 * (omega_x^2 + omega_y^2 + omega_z^2)
# 2D: enstrophy = 0.5 * omega_z^2
```
**物理意義**：渦度的"動能"，高值區域對應強渦旋結構

#### A3. Q-criterion [+1]
```python
Q = -0.5 * (||S||^2 + ||Ω||^2)
# S: strain rate tensor (對稱)
# Ω: vorticity tensor (反對稱)
```
**已實作** ✅（在 `compute_enhanced_turbulence_features_2d`）
**物理意義**：Q > 0 → 旋轉主導（渦旋），Q < 0 → 拉伸主導

#### A4. Lambda2 criterion (λ_2) [+1]
```python
# 速度梯度張量 S^2 + Ω^2 的第二特徵值
# λ_2 < 0 → 渦旋核心
```
**vs. 現有 grad_u_eig**：不同定義，專門用於渦旋識別

---

### B. 應變率張量特徵 (Strain Rate Tensor)

#### B1. 應變率不變量 [+2]
```python
# 已有 Q，再加：
S_invariant_I = tr(S) = ∂u/∂x + ∂v/∂y + ∂w/∂z  # = 0 for incompressible
S_invariant_II = -0.5 * tr(S^2)                  # = -||S||_F^2 / 2
S_invariant_III = det(S)
```
**推薦**：`S_invariant_II` (strain intensity)

#### B2. 應變率主方向 [+2]
```python
# S 張量的特徵向量（最大拉伸方向）
strain_direction_x, strain_direction_y
```
**物理意義**：流體拉伸的主方向（對應邊界層拉伸）

#### B3. 有效應變率 (Effective Strain Rate) [+1]
```python
S_eff = sqrt(2 * S_ij * S_ij)  # Frobenius norm
```
**vs. S_invariant_II**：直接物理量（單位：1/s）

---

### C. 湍流生成與耗散 (Production & Dissipation)

#### C1. 湍流生成率 (TKE Production) [+1]
```python
P_k = -τ_ij * ∂U_i/∂x_j  
    = -τ_uv * ∂U/∂y  (主要項，壁面剪切)
    - τ_uu * ∂U/∂x
    - τ_vv * ∂V/∂y
    - ...
```
**物理意義**：Reynolds stress 做功產生湍流動能
**重要性**：❗❗❗ 關鍵物理量，決定湍流強度空間分佈

#### C2. 湍流耗散率 (ε) [+1]
```python
# RANS 直接提供 (k-omega SST)
epsilon = C_mu * k * omega
# 或從黏性耗散計算
epsilon = 2 * nu * <s_ij * s_ij>
```
**檢查 RANS 資料**：是否有 `omega` 或 `epsilon` 欄位

#### C3. 生成-耗散比 (P_k / ε) [+1]
```python
Pk_epsilon_ratio = P_k / (epsilon + 1e-10)
```
**物理意義**：> 1 → 湍流增長區，< 1 → 衰減區

---

### D. 湍流長度尺度 (Turbulent Scales)

#### D1. 湍流長度尺度 [+1]
```python
L_t = k^(3/2) / epsilon
```
**物理意義**：湍流最大尺度

#### D2. Kolmogorov 尺度 [+1]
```python
eta = (nu^3 / epsilon)^(1/4)
```
**物理意義**：最小耗散尺度

#### D3. Taylor 微尺度 [+1]
```python
lambda_T = sqrt(15 * nu * k / epsilon)
```
**物理意義**：慣性與黏性尺度的過渡

---

### E. 無量綱數與混合指標 (Dimensionless Numbers)

#### E1. 局部雷諾數 [+1]
```python
Re_t = k^2 / (nu * epsilon)  # Turbulent Reynolds number
```
**物理意義**：湍流強度（高值 → 完全湍流，低值 → 層流化）

#### E2. 湍流強度 (Turbulence Intensity) [+1]
```python
TI = sqrt(2k/3) / U_mean
# 或各向異性：
TI_u = sqrt(tau_uu) / U_mean
TI_v = sqrt(tau_vv) / U_mean
```
**物理意義**：脈動與平均流的比例

#### E3. 湍流黏度比 [+1]
```python
mu_ratio = mu_t / mu  # 或 nu_t / nu
```
**物理意義**：湍流輸運 vs. 分子輸運（高值 → 湍流主導）

---

### F. 壓力相關 (Pressure Features)

#### F1. 壓力 Laplacian [+1]
```python
laplacian_p = ∂²p/∂x² + ∂²p/∂y²
```
**物理意義**：Poisson 方程右端（連結速度散度）

#### F2. 壓力-應變率相關 [+1]
```python
p_S_correlation = p * S_eff
```
**物理意義**：壓力做功

#### F3. 壓力梯度強度 [+1]
```python
grad_p_mag = sqrt((∂p/∂x)^2 + (∂p/∂y)^2)
```
**vs. 現有**：標量而非向量

---

### G. 各向異性指標 (Anisotropy)

#### G1. Reynolds Stress Anisotropy Tensor [+3]
```python
b_ij = (τ_ij / (2k)) - (1/3) * δ_ij
# 2D: b_11, b_22, b_12 (3 independent components)
```
**物理意義**：偏離各向同性湍流的程度
**重要性**：❗❗ 近壁區域強烈各向異性

#### G2. Anisotropy Invariants [+2]
```python
# Lumley triangle coordinates
II = b_ij * b_ji
III = b_ij * b_jk * b_ki
```
**物理意義**：表徵湍流狀態（1D/2D/3D 湍流）

---

### H. 壁面特徵 (Wall-Specific, for Channel Flow)

#### H1. 壁面法向距離 [+1]
```python
y_plus = y * u_tau / nu
# u_tau = sqrt(tau_w / rho) 摩擦速度
```
**物理意義**：無量綱壁距（y+ < 5 → 黏性底層）
**重要性**：❗❗❗ 通道流的核心參數

#### H2. 阻尼函數 (Van Driest Damping) [+1]
```python
D = 1 - exp(-y+ / A+)  # A+ ≈ 26
```
**物理意義**：近壁湍流抑制

#### H3. 壁面剪應力 [+1]
```python
tau_w = mu * (∂u/∂y)|_wall
# 或從 RANS: tau_w ≈ (mu + mu_t) * (∂u/∂y)
```
**已有**：`dudy` 隱含此資訊，但可顯式計算

---

### I. 二階導數 (Second Derivatives)

#### I1. 速度 Laplacians [+3]
```python
laplacian_u = ∂²u/∂x² + ∂²u/∂y²
laplacian_v = ∂²v/∂x² + ∂²v/∂y²
laplacian_w = ∂²w/∂x² + ∂²w/∂y²
```
**物理意義**：黏性擴散項（N-S 方程）

#### I2. 混合二階導數 [+6]
```python
∂²u/∂x∂y, ∂²v/∂x∂y, ...
```
**物理意義**：剪切層變化率
**警告**：數值二階導容易放大噪音

---

### J. 複合特徵 (Composite Features)

#### J1. 動能平衡項 [+3]
```python
# TKE transport equation 各項
Convection = U_j * ∂k/∂x_j
Diffusion = ∂/∂x_j ((nu + nu_t/sigma_k) * ∂k/∂x_j)
Production = P_k  # 已列
Dissipation = -epsilon  # 已列
```

#### J2. Lumley Flatness [+1]
```python
F = (tau_uu - tau_vv)^2 + (tau_vv - tau_ww)^2 + (tau_ww - tau_uu)^2
```
**物理意義**：各向異性的全局指標

---

## 🎯 推薦新增特徵（優先級）

### ⭐⭐⭐ 必加 (Critical for Physics)

1. **TKE Production (P_k)** [+1]  
   → 決定湍流生成的空間分佈，近壁最大

2. **y+ (Wall Distance)** [+1]  
   → 通道流最重要的無量綱參數

3. **Anisotropy Tensor (b_11, b_22, b_12)** [+3]  
   → 捕捉近壁各向異性

4. **Turbulent Reynolds Number (Re_t)** [+1]  
   → 湍流強度指標

5. **Enstrophy** [+1]  
   → 渦度強度

**小計**：+7 features → 總計 27

---

### ⭐⭐ 強烈推薦 (Highly Beneficial)

6. **Dissipation Rate (ε)** [+1]  
   → 如果 RANS 有 `omega` 可計算

7. **Effective Strain Rate (S_eff)** [+1]  
   → 拉伸強度

8. **Turbulence Intensity (TI)** [+1]  
   → 歸一化脈動強度

9. **Turbulent Viscosity Ratio (mu_t/mu)** [+1]  
   → 量化湍流輸運重要性

10. **Lambda2 criterion** [+1]  
    → 渦旋識別（vs. Q-criterion 互補）

**小計**：+5 features → 總計 32

---

### ⭐ 可選 (Useful but Lower Priority)

11. **Velocity Laplacians (∇²u, ∇²v, ∇²w)** [+3]  
    → N-S 黏性項，但二階導數值不穩定

12. **Pressure Laplacian (∇²p)** [+1]  
    → Poisson 方程相關

13. **TKE Turbulent Length Scale (L_t)** [+1]  
    → 需要 epsilon

14. **P_k / ε ratio** [+1]  
    → 湍流平衡指標

**小計**：+6 features → 總計 38

---

## 📋 實作檢查清單

### 1. 檢查 RANS 資料可用欄位
```bash
python3 -c "
import numpy as np
data = np.load('data/lowfi/channel_rans/rans_k_omega_sst.npz')
print('Available fields:', list(data.keys()))
print('\\nField shapes:')
for k in data.keys():
    if hasattr(data[k], 'shape'):
        print(f'  {k}: {data[k].shape}')
"
```

**需確認**：
- [ ] `omega` (specific dissipation rate)
- [ ] `epsilon` (dissipation rate)
- [ ] `y_wall` (wall distance)
- [ ] `nu` (kinematic viscosity)

### 2. 計算依賴關係

```
epsilon = C_mu * k * omega  (if omega available)
        = ...               (else estimate from velocity gradients)

y+ = y * u_tau / nu
   → u_tau = sqrt(tau_w / rho)
   → tau_w = mu * (∂u/∂y)|_wall  (need boundary condition)

P_k = -tau_ij * S_ij
    → 需要 Reynolds stresses (已有 ✅)
    → 需要 strain rate (可從速度梯度計算 ✅)
```

### 3. 數值穩定性考量

**二階導數**：需平滑化或 skip（容易放大噪音）
**小量相除**：加 epsilon 避免除零（如 P_k/ε）
**歸一化**：所有特徵標準化（已做 ✅）

---

## 🚀 建議實作順序

### Phase A: 必要特徵 (+7) → 27 total
1. TKE Production
2. Wall distance y+
3. Anisotropy tensor (b_ij)
4. Re_t
5. Enstrophy

### Phase B: 強化特徵 (+5) → 32 total
6. Dissipation rate ε
7. Effective strain rate
8. Turbulence intensity
9. Viscosity ratio
10. Lambda2

### Phase C: 實驗特徵 (+6) → 38 total
11-14. (根據 Phase A/B 效果決定)

---

## 📊 預期改善

| Feature Set | Count | Expected Rank | Expected Cond | 備註 |
|-------------|-------|---------------|---------------|------|
| Current (Minimal) | 10 | 10/10 | 7e5 | Baseline ✅ |
| + Phase A | 27 | ~25/27 | ? | 核心物理 |
| + Phase B | 32 | ~28/32 | ? | 完整湍流描述 |

**風險**：特徵過多可能導致：
- Condition number 上升（需監控）
- 冗餘特徵（如 P_k 與 tau_uv*S 相關）
- 計算成本增加

**緩解**：
- 使用 feature selection (e.g., mutual information)
- 監控 rank 與 condition number
- 保留多個版本（27, 32, 38）供對比

---

**下一步**：檢查 RANS 資料可用欄位，決定 Phase A 實作細節
