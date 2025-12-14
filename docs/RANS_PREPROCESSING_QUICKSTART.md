# RANS Preprocessing Quick Start Guide

**5-Minute Setup for RANS Turbulent Viscosity Preprocessing**

---

## ✨ What You Get

Physics-based corrections for RANS turbulent viscosity (ν_t) data:
- **Van Driest Damping**: Suppress ν_t near walls (ν_t → 0 as y → wall)
- **Physical Clipping**: Remove negative/extreme values (0 ≤ ν_t ≤ 1000ν)
- **Spatial Smoothing**: Reduce RANS numerical noise (optional)

**Result**: More stable training + better physical consistency.

---

## 🚀 Quick Start

### Step 1: Enable Preprocessing in Config

Edit your config file (e.g., `configs/my_experiment.yml`):

```yaml
lowfi_prior:
  enabled: true
  data_path: ./data/rans_data.h5
  
  # 🆕 Add this section
  preprocessing:
    enabled: true            # Turn on physics-based corrections
    apply_damping: true      # Van Driest near-wall suppression
    apply_clipping: true     # Remove unphysical values
    apply_smoothing: false   # Skip for large datasets (N > 10k)
```

**That's it!** Preprocessing runs automatically during training.

---

### Step 2: Train

```bash
python scripts/train/train.py --config configs/my_experiment.yml
```

Every 100 epochs, check `logs/<experiment>/training.log`:

```
RANS preprocessing: raw_mean=0.0523, processed_mean=0.0489, damping_factor=0.935, n_clipped=12
```

**Good signs**:
- `processed_mean < raw_mean`: Damping working ✅
- `damping_factor ≈ 0.9-0.95`: Expected for bulk flow ✅  
- `n_clipped` low (< 1% of points): RANS data quality OK ✅

---

## 📊 Advanced Options

### Option 1: Adjust for Different Re_τ

```yaml
preprocessing:
  max_ratio: 500    # For Re_τ ~ 500
  max_ratio: 2000   # For Re_τ ~ 2000
  max_ratio: 5000   # For Re_τ ~ 5000
```

**Rule of thumb**: `max_ratio ≈ Re_τ`

### Option 2: Enable Spatial Smoothing (Small Datasets Only)

```yaml
preprocessing:
  apply_smoothing: true   # ⚠️ WARNING: O(N²) cost!
  smoothing_method: gaussian
  smoothing_radius: 0.1   # Adjust based on mesh size
```

**Use only if**:
- Dataset size N < 10,000
- RANS data is very noisy
- You have time to spare (5-50× slower)

### Option 3: Custom u_τ and Domain

```yaml
preprocessing:
  u_tau: 0.045             # Friction velocity [m/s]
  domain_bounds: [0, 6.28, 0, 2.0]  # [x_min, x_max, y_min, y_max]
```

If not provided, auto-inferred from `physics.domain` and `physics.nu`.

---

## 🔍 Troubleshooting

### Issue: Large n_clipped (> 10% of points)

**Cause**: RANS data has extreme ν_t values  
**Solution**:
```yaml
preprocessing:
  max_ratio: 2000  # Increase threshold
```

Or check RANS simulation quality.

---

### Issue: Damping factor too low (< 0.5)

**Cause**: Most points too close to walls  
**Solution**: Verify `domain_bounds` match your mesh:

```yaml
preprocessing:
  domain_bounds: [0.0, 6.28, 0.0, 2.0]  # Check y_max!
```

---

### Issue: Training too slow

**Cause**: Smoothing enabled on large dataset  
**Solution**:
```yaml
preprocessing:
  apply_smoothing: false  # Disable smoothing
```

---

## 📁 Full Example

See complete configuration:
```bash
cat configs/templates/rans_prior_with_preprocessing.yml
```

Run tests to verify installation:
```bash
pytest tests/test_turbulence_utils.py -v
```

---

## 🎯 Default Settings (Production Ready)

```yaml
preprocessing:
  enabled: true
  apply_damping: true     # ✅ Recommended
  apply_clipping: true    # ✅ Recommended
  apply_smoothing: false  # ❌ Skip for performance
  max_ratio: 1000.0       # For Re_τ ~ 1000
  A_plus: 26.0            # Standard van Driest constant
```

**Performance**: < 1% overhead (damping + clipping only)

---

## 📚 Learn More

- **Full documentation**: `docs/RANS_INTEGRATION_SUMMARY.md`
- **Theory & tests**: `tasks/RANS_NUT_REVIEW/physics_review.md`
- **API reference**: `pinnx/physics/turbulence_utils.py`

---

## ✅ Verification Checklist

Before production training:

- [ ] `preprocessing.enabled: true` in config
- [ ] Check first 100 epochs for reasonable `damping_factor` (0.8-0.95)
- [ ] Verify `n_clipped` is low (< 5% of points)
- [ ] Run `pytest tests/test_rans_*.py` (37/37 should pass)

**Ready to train!** 🚀
