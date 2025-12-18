# FLUENT RANS Version Comparison Report

**Date**: 2025-12-18  
**Purpose**: Compare two FLUENT RANS simulations to determine which provides better lowfi prior  
**Script**: `scripts/compare/compare_fluent_rans_versions.py`  
**Figure**: `results/fluent_rans_version_comparison.png`

---

## Executive Summary

### 🎯 Recommendation: **USE VERSION 1** (Keep Current)

**Reason**: Version 2 shows marginal differences (< 2%) but **slightly worse** Re_τ estimate compared to DNS target. Version 1 is already processed, validated, and integrated into the pipeline.

---

## Data Files

### Version 1 (Original - Currently In Use)
```
data/channel_fluent/FFF-Setup-Output.cas.h5   (13.74 MB)
data/channel_fluent/FFF-Setup-Output.dat.h5   (44.72 MB)
```
- **Converted NPZ**: `data/lowfi/channel_rans/rans_k_omega_sst.npz` (18.91 MB)
- **Validation**: `FLUENT_RANS_VALIDATION.md` (2025-12-16)
- **Status**: ✅ Validated and in use

### Version 2 (New Simulation)
```
data/channel_fluent/FFF-Setup-Output.cas_2.h5  (13.74 MB)
data/channel_fluent/FFF-Setup-Output.dat_2.h5  (38.79 MB)
```
- **Status**: ⏳ Comparison completed, conversion not needed
- **File Size Difference**: dat_2 is 5.93 MB **smaller** (13% reduction)

---

## Key Findings

### 1. Grid & Domain

| Property | Version 1 | Version 2 | Match? |
|----------|-----------|-----------|--------|
| Nodes | 502,740 | 502,740 | ✅ |
| Cells | 471,880 | 471,880 | ✅ |
| X range | [0, 25.13] | [0, 25.13] | ✅ |
| Y range | [0, 2.0] | [0, 2.0] | ✅ |
| Z range | [0, 9.425] | [0, 9.425] | ✅ |

**Conclusion**: Both versions use **identical mesh** (same resolution and domain size).

---

### 2. Flow Parameters

| Parameter | Version 1 | Version 2 | Difference | Target (DNS) | V1 Error | V2 Error |
|-----------|-----------|-----------|------------|--------------|----------|----------|
| **U_bulk** | 0.9844 | 0.9941 | +0.99% | ~1.0 | -1.6% | -0.6% |
| **u_τ** | 0.06720 | 0.06794 | +1.10% | ~0.05 | +34% | +36% |
| **Re_τ** | **1343.9** | **1358.7** | +1.10% | **983.7** | **+36.6%** | **+38.1%** |

**Critical Observation**:
- Version 2 has **1.5% higher Re_τ error** than Version 1
- Both over-predict Re_τ by ~37% (typical for k-ω SST on channel flow)
- Version 2's U_bulk is slightly closer to DNS (~1.0), but this doesn't compensate for worse Re_τ

---

### 3. Velocity Field Statistics

| Variable | Statistic | Version 1 | Version 2 | Difference |
|----------|-----------|-----------|-----------|------------|
| **u** (streamwise) | mean | 0.9844 | 0.9941 | +0.99% |
| | std | 0.1208 | 0.1200 | -0.62% |
| | min | 0.6839 | 0.6936 | +1.42% |
| | max | 1.0868 | 1.0937 | +0.63% |
| **v** (wall-normal) | mean | -2.1e-12 | 1.7e-13 | ~0% |
| | std | 1.4e-7 | **1.9e-9** | **-98.7%** ⚠️ |
| **w** (spanwise) | mean | -1.5e-11 | **0.0** | ~0% |
| | std | 4.9e-7 | **0.0** | **-100%** ⚠️ |

**Key Observations**:
- **v and w fluctuations nearly zero in V2**: Suggests stronger smoothing or different convergence criteria
- Version 2 may have tighter residual tolerances, reducing numerical noise
- This could indicate **over-converged/over-damped solution** in V2

---

### 4. Turbulence Statistics

| Variable | Statistic | Version 1 | Version 2 | Difference |
|----------|-----------|-----------|-----------|------------|
| **k** (TKE) | mean | 4.515e-3 | 4.615e-3 | +2.22% |
| | std | 1.833e-3 | 1.841e-3 | +0.42% |
| | min | 2.349e-3 | 2.505e-3 | +6.65% |
| | max | 7.848e-3 | 7.985e-3 | +1.75% |
| **p** (pressure) | mean | -2.7e-8 | 3.1e-9 | ~0 (gauge) |
| | std | 6.9e-8 | 9.5e-9 | -86% ⚠️ |

**Key Observations**:
- Version 2 has **slightly higher TKE** (+2.2%) → explains higher u_τ and Re_τ
- Pressure fluctuations **much lower** in V2 (-86%) → suggests different convergence or post-processing

---

## Visual Comparison

See `results/fluent_rans_version_comparison.png` for:

1. **Mean Velocity Profile (U vs Y)**: Nearly identical profiles, V2 slightly higher
2. **TKE Profile (k vs Y)**: V2 has marginally higher TKE across all wall-normal positions
3. **Velocity Difference Distribution**: Point-wise differences centered at ~+0.01 (V2 higher)
4. **TKE Difference Distribution**: Point-wise differences centered at ~+1e-4 (V2 higher)
5. **Velocity Magnitude Correlation**: Strong correlation (R² ~ 1.0), slight systematic offset
6. **Radar Chart**: V2 has slightly elevated U_bulk, u_τ, and TKE

---

## Interpretation

### What Changed Between V1 and V2?

**Likely Scenarios** (requires checking FLUENT case settings):

1. **Tighter Convergence Criteria**: 
   - V2's nearly-zero v/w fluctuations suggest residuals converged to machine precision
   - Could indicate more solver iterations or stricter tolerance

2. **Different Turbulence Model Parameters**:
   - TKE production/dissipation balance shifted slightly
   - Could be SST blending coefficient or wall treatment changes

3. **Post-Processing Differences**:
   - Pressure reference or normalization changed
   - Time-averaging window (if transient) might differ

### Why V2 is Not Better

1. **Re_τ Error Increased**: +1.5% higher error vs DNS target (36.6% → 38.1%)
2. **Marginal Improvements**: U_bulk closer to ~1.0, but difference negligible (<1%)
3. **Loss of Fluctuations**: Near-zero v/w std suggests over-smoothing (not physical)
4. **No Clear Physical Advantage**: Both versions have same grid, similar profiles

### Why Keep V1

1. **Already Validated**: Documented in `FLUENT_RANS_VALIDATION.md`
2. **Already Converted**: NPZ file integrated in training pipeline
3. **Phase A Sensors**: Generated using V1 grid (K=100 sensors)
4. **Stability**: V1 has realistic (non-zero) v/w fluctuations

---

## Decision Matrix

| Criterion | Version 1 | Version 2 | Winner |
|-----------|-----------|-----------|--------|
| **Re_τ Accuracy** | 36.6% error | 38.1% error | ✅ V1 |
| **U_bulk Accuracy** | 1.6% error | 0.6% error | V2 |
| **Physical Realism** | Non-zero v/w | Zero v/w ⚠️ | ✅ V1 |
| **Integration Status** | ✅ Ready | ⏳ Needs work | ✅ V1 |
| **Documentation** | ✅ Complete | ❌ None | ✅ V1 |
| **File Size** | 44.72 MB | 38.79 MB | V2 |

**Overall**: **Version 1 Wins** (4 vs 2)

---

## Recommendation

### ✅ Action: Keep Using Version 1

**Justification**:
1. Version 2 is **not significantly better** (< 2% differences)
2. Version 2 has **worse Re_τ estimate** (critical for wall-bounded flow)
3. Version 1 is **already validated and integrated** (no disruption to workflow)
4. Version 2's zero v/w fluctuations are **suspicious** (possibly numerical artifact)

### 🔒 Archive Version 2

Keep files for reference but do not convert to NPZ or integrate into training:

```bash
# Optional: Move to archive
mkdir -p data/channel_fluent/archive/
mv data/channel_fluent/FFF-Setup-Output.*_2.h5 data/channel_fluent/archive/
```

### 📋 If You Still Want to Try Version 2

**Only proceed if**:
1. You confirm V2 used **better physics settings** (check FLUENT case)
2. You can explain why v/w fluctuations vanished
3. You accept +1.5% higher Re_τ error

**Conversion steps**:
```bash
python scripts/data/extract_fluent_rans.py \
  --cas data/channel_fluent/FFF-Setup-Output.cas_2.h5 \
  --dat data/channel_fluent/FFF-Setup-Output.dat_2.h5 \
  --output data/lowfi/channel_rans/rans_k_omega_sst_v2.npz
```

**Re-generate sensors**:
```bash
python scripts/generate/sensors/generate_qr_sensors_phase_a.py \
  --rans-file data/lowfi/channel_rans/rans_k_omega_sst_v2.npz \
  --output data/lowfi/channel_rans/sensors_K100_rans_phase_a_v2.npz
```

---

## Technical Notes

### Observed Anomalies in Version 2

1. **Zero w-velocity everywhere**: Physically impossible in 3D channel flow
   - Likely: FLUENT exported 2D slice instead of 3D field
   - Or: Symmetry boundary condition over-constrained the solution

2. **Near-zero v-velocity fluctuations**: Unrealistic for turbulent channel
   - Likely: Over-converged RANS (residuals < 1e-12)
   - Or: Post-processing averaged out fluctuations

3. **Lower pressure std**: Consistent with over-smoothing hypothesis

### What to Check in FLUENT

If you want to investigate Version 2 further, check:

```
File → Export → Solution Data → Verify:
  ✓ 3D export (not 2D slice)
  ✓ Cell-centered data
  ✓ All velocity components included
  ✓ Unsteady statistics (if URANS) properly time-averaged

Solve → Monitors → Residuals:
  - Compare final residuals between V1 and V2
  - V2 likely has 2-3 orders tighter convergence

Models → Viscous → k-omega SST Options:
  - Compare production limiters
  - Check wall treatment (standard vs enhanced)
```

---

## Conclusion

**Version 1 remains the best choice** for lowfi prior in PINNs training:
- Acceptable Re_τ error (36.6% typical for RANS)
- Physically realistic flow field
- Already validated and integrated
- Compatible with existing Phase A sensors (K=100)

**Version 2 offers no advantage**:
- Marginally worse Re_τ estimate
- Suspicious zero fluctuations
- Would require re-generating all downstream data

**No action needed**: Continue with current pipeline using Version 1.

---

**Files Generated**:
- Comparison script: `scripts/compare/compare_fluent_rans_versions.py`
- Comparison figure: `results/fluent_rans_version_comparison.png` (804 KB)
- This report: `docs/FLUENT_RANS_V1_VS_V2_COMPARISON.md`

**Status**: ✅ Analysis complete, decision made
