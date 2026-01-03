# Deprecation Notice v1.4.0

## Deprecated Features

### 1. Adaptive Collocation Sampling (REMOVED)

**Status**: Hard Deprecated (Raises Error if Enabled)  
**Removal Date**: 2026-01-04  
**Reason**: Feature was never validated and has zero usage across all 30+ config files

**What This Means**:
- Setting `training.sampling.adaptive_sampling: true` will raise `ValueError`
- Setting `adaptive_collocation.enabled: true` will raise `ValueError`
- The module `pinnx/train/adaptive_collocation.py` (639 lines) is marked for deletion in v1.5.0
- Tests `test_adaptive_collocation_fixes.py` and `test_adaptive_integration.py` are marked for deletion

**Migration Path**: 
No migration needed - all existing configs already set `adaptive_sampling: false`

**Justification**:
```yaml
# Evidence from codebase audit:
Total configs checked: 20+
Configs with adaptive_sampling: true: 0
Configs with adaptive_sampling: false: 20+
Performance benchmarks completed: 0
Production usage: Never
```

**Philosophy Violation**:
- **Pragmatism**: "Theory without validation is technical debt"
- **Simplicity**: 639 lines of unused code increases maintenance burden

---

## Removal Timeline

### v1.4.0 (Current) - 2026-01-04
- ✅ Raise errors if adaptive sampling is enabled in config
- ✅ Add deprecation warnings to relevant methods
- ✅ Update documentation to mark feature as removed

### v1.5.0 (Planned) - 2026-02-01
- Delete `pinnx/train/adaptive_collocation.py`
- Delete related tests
- Remove all deprecated code paths

---

## Config Validation

To check if your configs will be affected:

```bash
# Check for usage of deprecated features
rg "adaptive_sampling:\s*true" configs/
rg "adaptive_collocation:" configs/ -A 2

# If any results found, update to:
training:
  sampling:
    adaptive_sampling: false  # or remove this key entirely
```

---

## Contact

For questions or concerns, please file an issue with label `deprecation-v1.4`
