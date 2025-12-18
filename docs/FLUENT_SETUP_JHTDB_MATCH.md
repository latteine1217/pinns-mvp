# Fluent 設置：精確匹配 JHTDB Re_τ=1000

**基於 JHTDB Channel Flow 官方文檔**

---

## ⚠️ 關鍵差異

| 項目 | 之前的建議 | JHTDB 官方 | 差異 |
|------|-----------|-----------|------|
| 座標系統 | 有因次（h=0.1m） | **無因次（h=1）** | ❌ |
| 壓力梯度 | -2.5 Pa/m | **+0.0025** | ❌ |
| 摩擦速度 | 0.5 m/s | **0.05** | ❌ |
| 體平均速度 | ~9 m/s | **1.0** | ❌ |

**結論：必須使用無因次化設定！**

---

## 📋 JHTDB 精確參數

### 幾何尺寸（無因次，h=1）

```
Domain Size:
- 流向 (x): Lx = 8πh = 25.1327
- 壁向 (y): Ly = 2h = 2.0
- 展向 (z): Lz = 3πh = 9.4248
```

### 流體屬性（無因次）

```
Density: ρ = 1.0
Viscosity: μ = ν = 5×10⁻⁵
```

### 驅動條件（無因次）

```
Mean Pressure Gradient: dP/dx = +0.0025
（注意：正值，驅動正 x 方向流動）
```

### 預期統計量

```
Friction velocity: u_τ = 0.049968
Bulk velocity: U_b = 0.99994
Centerline velocity: U_c = 1.1312
Reynolds number: Re_τ = u_τ·h/ν = 999.35 ≈ 1000 ✓
```

---

## 🛠️ Fluent 完整設置步驟

### 1️⃣ General

```
Solver:
  Type: Pressure-Based
  Time: Steady

Velocity Formulation: Absolute
```

### 2️⃣ Models

```
Viscous Model: Laminar
（或 k-omega SST for RANS approximation）

⚠️ 注意：JHTDB 是 DNS，不是 RANS！
   如果要 RANS approximation，選 k-omega SST
   如果要嘗試 DNS-like（不推薦在 Fluent），選 Laminar + 極細網格
```

### 3️⃣ Materials

```
Fluid: custom-jhtdb-fluid

Properties:
  Density: 1.0 kg/m³
  Viscosity: 5e-05 kg/(m·s)  ← 關鍵！
```

### 4️⃣ Cell Zone Conditions

```
Zone: fluid

Source Terms:
  ☑ X Momentum

  Constant Value: 0.0025

⚠️ 注意：是正值 +0.0025，不是負值！
  JHTDB 的壓力梯度定義為驅動力，不是阻力
```

### 5️⃣ Boundary Conditions

```
inlet (x=0):
  Type: Periodic

outlet (x=Lx):
  Type: Periodic
  ───────────────
  Define → Periodic Conditions
    Translational Periodic
    Zone 1: inlet
    Zone 2: outlet
    Translation: (25.1327, 0, 0)

wall_bottom (y=0):
  Type: Wall
  No Slip

wall_top (y=2h):
  Type: Wall
  No Slip

periodic_z1 (z=0):
  Type: Periodic

periodic_z2 (z=Lz):
  Type: Periodic
  ───────────────
  Define → Periodic Conditions
    Translational Periodic
    Zone 1: periodic_z1
    Zone 2: periodic_z2
    Translation: (0, 0, 9.4248)
```

### 6️⃣ Solution Methods

**For RANS (推薦):**
```
Scheme: SIMPLE
Gradient: Least Squares Cell Based
Spatial Discretization: All Second Order
☑ Pseudo Transient
```

**For Laminar (DNS-like):**
```
Scheme: Coupled
Gradient: Least Squares Cell Based
Spatial Discretization: All Second Order
Transient Formulation: Second Order Implicit
```

### 7️⃣ Solution Initialization

```
Standard Initialization:

Gauge Pressure: 0
X Velocity: 1.0  ← U_b
Y Velocity: 0
Z Velocity: 0

If RANS:
  k: 0.003  ← ≈ 1.5(U·I)², I ≈ 5%
  ω: 50     ← 估算值
```

### 8️⃣ Run Calculation

```
Number of Iterations: 5000-10000
Reporting Interval: 100
Residuals Target: 1e-6
```

---

## 🎯 收斂後驗證

### 計算摩擦速度

```
Report → Surface Integrals
  Area-Weighted Average
  Wall Shear Stress
  Surfaces: wall_bottom, wall_top

得到 τ_w，計算：
u_τ = √(τ_w/ρ)

目標：u_τ ≈ 0.05 (無因次)
```

### 計算 Re_τ

```
Re_τ = u_τ × h / ν
     = 0.05 × 1.0 / 5e-5
     = 1000 ✓
```

### 調整壓力梯度（若需要）

```
如果 Re_τ 偏差 > 5%:

新壓力梯度 = 0.0025 × (Re_τ_target / Re_τ_actual)²

例：Re_τ = 950 → dP/dx = 0.0025 × (1000/950)² = 0.00277
```

---

## 📊 網格要求

### RANS（實用）

```
推薦網格：
  Nx × Ny × Nz = 128 × 80 × 64 ≈ 655k

網格分布：
  x: 均勻
  y: 壁面加密（y⁺ ≈ 30-50）
  z: 均勻

第一層網格高度：
  Δy₁ ≈ 0.03 (無因次)
  對應 y⁺ = Δy₁ × u_τ / ν ≈ 30
```

### DNS-like（不推薦在 Fluent）

```
JHTDB 實際網格：
  Nx × Ny × Nz = 2048 × 512 × 1536 ≈ 1.6B

這在 Fluent 中不實際！
建議直接使用 RANS approximation
```

---

## ✅ 後處理檢查清單

- [ ] 體積平均速度 U_b ≈ 1.0
- [ ] 摩擦速度 u_τ ≈ 0.05
- [ ] Re_τ = 1000 ± 50
- [ ] 中心線速度 U_c ≈ 1.13
- [ ] 速度剖面符合對數律（見 PDF Fig 2）
- [ ] 壁面速度 ≈ 0（無滑移）
- [ ] 殘差 < 1e-6
- [ ] 質量守恆（入口 = 出口）

---

## 📦 數據導出

**必須同時導出 Case & Data：**

```
File → Export → Case & Data...
  File Type: HDF5
  File Name: channel_re1000_jhtdb_rans

產生：
  ✓ channel_re1000_jhtdb_rans.cas.h5  ← 包含座標
  ✓ channel_re1000_jhtdb_rans.dat.h5  ← 包含流場
```

**導出變數清單：**
- ✓ 座標 (x, y, z)
- ✓ 速度 (u, v, w)
- ✓ 壓力 (p)
- ✓ 湍流量 (k, ω) [if RANS]
- ✓ 壁面距離

---

## 🔄 與 JHTDB DNS 的差異

| 項目 | JHTDB DNS | Fluent RANS | 影響 |
|------|-----------|-------------|------|
| 數值方法 | Spectral | Finite Volume | 精度 ↓ |
| 湍流處理 | 直接求解 | 模型化 (k-ω SST) | 小尺度細節 ↓ |
| 網格解析度 | 1.6B cells | ~1M cells | 解析度 ↓ |
| 瞬時場 | ✓ | ✗ (only mean) | 無時序資訊 |
| 計算成本 | 極高 | 可接受 | - |

**適用場景：**
- RANS 適合：提供 mean field prior
- DNS 適合：Ground truth validation

---

## 📝 重要提醒

1. **無因次化**：所有數值都是無因次的（h=1 為長度單位）
2. **壓力梯度符號**：+0.0025（正值驅動）
3. **座標導出**：必須同時導出 .cas.h5 和 .dat.h5
4. **RANS vs DNS**：Fluent RANS 只能提供平均場近似
5. **驗證**：收斂後務必檢查 Re_τ ≈ 1000

---

**參考文獻：**
- JHTDB Channel Flow README (thesis/Channel README.pdf)
- Lee et al., "Petascale DNS of turbulent channel flow", SC13, 2013
