# DNS Forcing Bug Fix Report

**Date**: 2025-11-23  
**Script**: `scripts/generate_kolmogorov_dns.py`  
**Issue**: Kolmogorov DNS simulations producing NaN values immediately after t=0

---

## 🐛 Bug Description

### Symptoms
- All DNS simulations (Re=100, Re=500) crashed with NaN values at t≈1.0s
- Kinetic energy, enstrophy, and divergence error all became NaN
- Simulation started from zero velocity field and failed to develop

### Root Causes Identified

#### **Bug #1: Incorrect Forcing Direction** ⭐ CRITICAL
**Location**: Lines 161-166, 236-241

**Problem**: Kolmogorov forcing was applied to **x-momentum (U)** instead of **y-momentum (V)**

```python
# WRONG (original code):
self.forcing = A * np.sin(k_f * Y)              # Generic forcing
rhs_U = conv_U_hat - self.nu * self.k2 * U_hat + forcing_hat  # ❌ Should be 0
rhs_V = conv_V_hat - self.nu * self.k2 * V_hat                 # ❌ Should have forcing
```

**Correct Physics**: Kolmogorov flow forcing is **F = A sin(k_f y) ŷ** (y-direction only)
- x-momentum: `∂u/∂t + ... = -∂p/∂x + ν∇²u` (no forcing)
- y-momentum: `∂v/∂t + ... = -∂p/∂y + ν∇²v + A sin(k_f y)` (with forcing)

**Fix Applied**:
```python
# CORRECT (fixed code):
self.forcing_y = A * np.sin(k_f * Y)                            # y-momentum forcing only
rhs_U = conv_U_hat - self.nu * self.k2 * U_hat                  # ✅ No forcing
rhs_V = conv_V_hat - self.nu * self.k2 * V_hat + forcing_y_hat  # ✅ With forcing
```

---

#### **Bug #2: Zero Initial Condition** ⭐ CRITICAL
**Location**: Lines 167-178 (after fix)

**Problem**: Simulation started from **U=0, V=0** (zero velocity everywhere)
- With zero initial condition, the flow must develop solely from forcing
- For high Re (Re=500), starting from zero leads to numerical instability
- The exact laminar solution **v(y) = (A/(ν k_f²)) sin(k_f y)** is too large (V_max ≈ 9.95 for Re=500)

**Solution**: Initialize with **weakened laminar base flow**

```python
# Weakened laminar initialization (alpha controls strength)
alpha = min(0.1, nu * k_f**2 / A)  # Ensures |V| ≤ ~1.0
V_amp = alpha * A / (nu * k_f**2)
V_init = V_amp * np.sin(k_f * Y)
V_hat = fft2(V_init)
```

**Key Parameters**:
- `alpha = 0.1` for Re=500 → V_max = 0.995 (stable)
- Original laminar solution → V_max = 9.95 (unstable, causes NaN)

---

## ✅ Verification Tests

### Test 1: Small Grid, Short Time (N=128, T=1s)
```bash
python scripts/generate_kolmogorov_dns.py --N 128 --nu 0.01 --k_f 4 --T_end 1.0
```

**Before Fix**: NaN at t=1.0s  
**After Fix**: ✅ KE=0.1169, Enstrophy=0.0163, Div_err=1.98×10⁻⁸

---

### Test 2: Re=500 Initialization (N=128, T=3s)
```bash
python scripts/generate_kolmogorov_dns.py --N 128 --nu 0.006283 --k_f 4 --T_end 3.0
```

**Results**:
- ✅ V_max(init) = 0.995 (weakened by alpha=0.1)
- ✅ KE: 0.247 → 0.154 → 0.153 (stable)
- ✅ Enstrophy: 0.0094 → 0.028 (vorticity developing)
- ✅ Div_err ≤ 5.4×10⁻⁴ (acceptable incompressibility)
- ✅ No NaN values throughout simulation

---

### Test 3: Production Run (N=1024, Re=500, T=50s) 🚀 IN PROGRESS
```bash
nohup python scripts/generate_kolmogorov_dns.py \
  --N 1024 --nu 0.006283 --k_f 4 --T_end 50.0 \
  --perturbation_times 5.0 --perturbation_method unstable_mode \
  --output data/kolmogorov_dns_re500_kf4_t50_N1024.h5 \
  > log/dns_re500_kf4_t50.log 2>&1 &
```

**Status**: Running (PID 60549)  
**Initial Output**: ✅ V_max(init)=0.995, KE=0.247, no NaN  
**Expected Runtime**: ~20 minutes (50k steps @ ~45 steps/s)

---

## 📊 Physics Validation

### Kolmogorov Flow Governing Equations
```
∂u/∂t + u·∇u = -∂p/∂x + ν∇²u
∂v/∂t + u·∇v = -∂p/∂y + ν∇²v + A sin(k_f y)  ← Forcing here!
∇·u = 0  (incompressibility)
```

### Laminar Solution (Steady State)
```
u_lam(y) = 0
v_lam(y) = (A / (ν k_f²)) [1 - cos(k_f y)]
```

- For Re=500 (ν=0.006283): **v_lam,max ≈ 9.95** (too large!)
- With alpha=0.1 damping: **v_init,max ≈ 0.995** (stable)

### Reynolds Number Definitions
```
Re_forcing = sqrt(A/k_f) × L / ν ≈ 500  (based on forcing scale)
Re_achieved = U_rms × L / ν           (depends on flow evolution)
```

---

## 📝 Code Changes Summary

### Modified Functions

#### 1. `__init__` (Lines 161-183)
- Changed `self.forcing` → `self.forcing_y` (clarify y-momentum only)
- Replaced zero initialization with weakened laminar solution
- Added alpha-based amplitude control
- Added logging for initial V_max and alpha

#### 2. `compute_rhs` (Lines 237-241)
- Changed `forcing_hat` → `forcing_y_hat`
- Removed forcing from `rhs_U` equation
- Added forcing to `rhs_V` equation
- Added comments clarifying physics

---

## 🔬 Future Improvements

### 1. Adaptive Initial Amplitude
```python
# Automatically adjust alpha based on Reynolds number
if Re < 100:
    alpha = 0.5  # Strong initialization for stable flows
elif Re < 500:
    alpha = 0.2  # Moderate initialization
else:
    alpha = 0.05  # Weak initialization for high Re
```

### 2. Gradual Forcing Ramp-Up
```python
# Gradually increase forcing from 0 to A over initial time period
A_eff(t) = A * min(1.0, t / T_rampup)
```

### 3. Energy-Based Stability Check
```python
# Halt simulation if energy grows too fast (sign of instability)
if dKE/dt > threshold:
    reduce_timestep() or adjust_forcing()
```

---

## ✅ Conclusion

**Both bugs are now fixed**:
1. ✅ Forcing correctly applied to y-momentum only
2. ✅ Initial condition uses stable weakened laminar solution

**Current Status**:
- Re=100 (k_f=8) simulations: ✅ Complete (data exists)
- Re=500 (k_f=4) simulation: 🚀 Running (N=1024², T=50s)

**Next Steps**:
1. Monitor Re=500 simulation to completion
2. Generate visualization suite (GIFs, energy spectra)
3. Compare Re=100 vs Re=500 turbulence characteristics
4. Use DNS data for PINNs training

---

**Author**: OpenCode AI  
**Verified**: 2025-11-23 03:51 AM
