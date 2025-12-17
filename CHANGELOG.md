# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Removed - 2025-12-17

**Phase 2: Aggressive Removal of Deprecated Low-Fidelity Features**

Removed deprecated classes and methods to align codebase with project scope:
- ✅ **2D Kolmogorov Flow** → Leith turbulence model only
- ✅ **3D Channel Flow** → RANS k-ε turbulence model only

#### Deleted Classes (6 total)

**pinnx/losses/priors.py (-393 lines, -49%)**
- `StatisticalConsistencyLoss` - Statistics handled by PDE residuals
- `ConservationLoss` - Conservation handled by PDE residuals  
- `SymmetryConsistencyLoss` - Symmetry handled by boundary conditions

**pinnx/dataio/lowfi_loader.py (-381 lines, -31%)**
- `NetCDFReader` - Project only uses HDF5 format
- `DownsampledDNSProcessor` - DNS downsampling not in project workflow
- `LESReader` - LES models not in project scope

#### Modified Files (4 total)

1. **pinnx/losses/__init__.py**
   - Removed deprecated imports
   - Updated `__all__` exports
   - Simplified `CompleteLossManager.__init__()`

2. **pinnx/dataio/__init__.py**
   - Removed deprecated imports (NetCDFReader, LESReader, DownsampledDNSProcessor)
   - Simplified `create_lowfi_loader()` helper function
   - Updated `__all__` exports

3. **tests/test_lowfi_loader.py**
   - Added `@pytest.mark.skip` to deprecated class tests
   - Removed deprecated imports

4. **docs/LOWFI_PRIOR_GUIDE.md**
   - Updated unsupported models section to reflect removal

#### Impact Analysis

**Code Reduction:**
- Total lines removed: **774 lines**
- Test results: **540/618 tests passed** (87.4%)
- All deprecated class tests properly skipped

**Supported Components (unchanged):**
- ✅ `LowFidelityConsistencyLoss` - Primary prior loss
- ✅ `PriorLossManager` - Simplified manager
- ✅ `HDF5Reader` - Primary data format reader
- ✅ `RANSReader` - RANS-specific wrapper
- ✅ `NPZReader` - Generic format support
- ✅ `SpatialInterpolator` - Spatial interpolation utility

**Breaking Changes:**
- ❌ Cannot import `NetCDFReader`, `DownsampledDNSProcessor`, `LESReader`
- ❌ Cannot import `StatisticalConsistencyLoss`, `ConservationLoss`, `SymmetryConsistencyLoss`
- ❌ `create_lowfi_loader()` no longer accepts `filter_type` parameter
- ❌ `LowFiLoader` no longer has `dns_processor` attribute

**Migration Guide:**
- Use `HDF5Reader` instead of `NetCDFReader`
- Use pre-generated RANS priors instead of DNS downsampling
- Use `LowFidelityConsistencyLoss` instead of deprecated loss classes

---

## [1.0.0] - 2025-12-17

### Added
- Initial stable release
- Support for 2D Kolmogorov Flow with Leith turbulence model
- Support for 3D Channel Flow with RANS k-ε turbulence model
- QR-Pivot sensor selection
- VS-PINN architecture with gradient caching
- Fourier feature networks
- Comprehensive test suite (618 tests)

### Documentation
- Complete API reference
- Configuration guide
- Quick start guide
- Troubleshooting guide
- Technical documentation

---

[Unreleased]: https://github.com/your-org/pinns-mvp/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/your-org/pinns-mvp/releases/tag/v1.0.0
