# 專案範圍與支援功能

> **最後更新**: 2025-12-17  
> **版本**: 1.0.0

---

## 🎯 專案目標

基於物理資訊神經網路 (PINNs) 與稀疏感測器資料，重建湍流場。

**核心能力**：
- ✅ 稀疏感測器數據 (K ≤ 100) 的全場重建
- ✅ 低保真物理先驗 (RANS/LES) 融合
- ✅ 高保真參考數據 (JHTDB DNS) 驗證
- ✅ 物理一致性保證 (守恆律、邊界條件)

---

## ✅ 支援的場景（僅 2 個）

### 1. 2D Kolmogorov Flow ✅

**物理配置**：
- 流體類型：不可壓縮 Navier-Stokes
- 幾何：2D 週期域 (4π × 2π)
- 雷諾數：Re = 50-100
- 強迫：Kolmogorov forcing (kf = 4)

**Low-Fidelity Prior**：
- 模型：LES turbulence model
- 變數：`u`, `v`, `nu_t` (無壓力場)
- 資料格式：NPY 目錄（memory-mapped）
- 特點：大尺度統計穩定，適合 2D 逆能量級串

**配置範例**：
```yaml
# configs/kolmogorov_re50_kf4_K100.yml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi_npy/kolmogorov_les/re50
  data_type: les
```

**參考文件**：
- `docs/archive/experiments/LES_PARAMETER_SELECTION_CRITERIA.md`（歷史參考）
- `scripts/generate/dns/generate_kolmogorov_les.py`

---

### 2. 3D Channel Flow (Re_tau = 1000) ✅

**物理配置**：
- 流體類型：不可壓縮 Navier-Stokes
- 幾何：3D 通道 (Lx × 2h × Lz = 8π × 2 × 3π)
- 雷諾數：Re_tau = 1000, Re_bulk ≈ 40000
- 邊界：壁面無滑移 + x/z 週期

**Low-Fidelity Prior**：
- 模型：RANS k-ε turbulence model
- 變數：`u`, `v`, `w`, `p`, `k`, `epsilon`, `nu_t`
- 資料格式：HDF5 (.h5)
- 特點：求解 k-ε 輸運方程，包含壓力場

**配置範例**：
```yaml
# configs/channel_flow_re1000.yml
lowfi_prior:
  enabled: true
  data_path: ./data/lowfi/channel_rans/rans_retau1000.h5
  data_type: rans
```

**參考文件**：
- `TECHNICAL_DOCUMENTATION.md`（系統架構）
- `CONFIG_GUIDE.md`（RANS 配置說明）

---

## ❌ 不支援的功能（已移除）

### 已刪除類別 (2025-12-17)

**pinnx/losses/priors.py**:
- ❌ `StatisticalConsistencyLoss` - 統計量由 PDE 殘差處理
- ❌ `ConservationLoss` - 守恆律由 PDE 殘差處理
- ❌ `SymmetryConsistencyLoss` - 對稱性由邊界條件處理

**pinnx/dataio/lowfi_loader.py**:
- ❌ `NetCDFReader` - 專案僅使用 HDF5 格式
- ❌ `DownsampledDNSProcessor` - 不進行 DNS 下採樣
- ❌ `LESReader` - LES 使用 NPY pipeline，reader 不再需要

### 不支援的場景

**湍流模型**：
- ❌ DES (Detached Eddy Simulation)
- ❌ DNS 下採樣作為 prior

**資料格式**：
- ❌ NetCDF (.nc, .nc4)
- ❌ OpenFOAM 原生格式
- ✅ 僅支援 HDF5 (.h5) 與 NumPy (.npz)

**幾何**：
- ❌ Cavity flow
- ❌ Backward-facing step
- ❌ Cylinder flow
- ❌ 任意幾何 (STL mesh)

---

## 🔧 核心技術棧

### 神經網路架構
- ✅ **Fourier Feature Network** - 高頻捕捉
- ✅ **VS-PINN** - 變數縮放 PINNs
- ✅ **SIREN** - 週期性激活函數
- ✅ **Axis-Selective Fourier** - 各向異性頻率

### 感測器選擇策略
- ✅ **QR-Pivot** - 基於物理的貪婪選擇
- ✅ **Random** - 基準對照
- ❌ ~~Optimal Sensor Placement (OSP)~~ - 未實作

### 優化器
- ✅ **SOAP** - 預條件 Adam
- ✅ **Adam** - 標準梯度下降
- ✅ **L-BFGS** - 二階優化（fine-tuning）
- ❌ ~~SGD~~ - 不適合 PINNs

### 損失權重策略
- ✅ **GradNorm** - 梯度平衡
- ✅ **Causal Weighting** - 時間因果
- ✅ **NTK Weighting** - 神經正切核
- ❌ ~~Multi-Task Learning (MTL)~~ - 未實作

---

## 📂 資料格式規範

### HDF5 結構 (RANS Prior)

```
rans_data.h5
├── /mean_field          # 主資料組
│   ├── u [Nx, Ny, Nz]  # 速度 u 分量
│   ├── v [Nx, Ny, Nz]  # 速度 v 分量
│   ├── w [Nx, Ny, Nz]  # 速度 w 分量 (3D only)
│   ├── p [Nx, Ny, Nz]  # 壓力場
│   ├── k [Nx, Ny, Nz]  # 湍動能
│   ├── epsilon [Nx, Ny, Nz]  # 耗散率
│   ├── nu_t [Nx, Ny, Nz]     # 渦黏度
│   ├── x [Nx] or X [Nx, Ny, Nz]  # 座標 (1D or 2D)
│   ├── y [Ny] or Y [Nx, Ny, Nz]
│   └── z [Nz] or Z [Nx, Ny, Nz]  # (3D only)
```

### HDF5 結構 (LES Prior)

```
les_data.h5
├── /mean_field
│   ├── u [Nx, Ny]      # 速度 u 分量
│   ├── v [Nx, Ny]      # 速度 v 分量
│   ├── nu_t [Nx, Ny]   # 渦黏度
│   ├── x [Nx]          # 座標 (1D)
│   └── y [Ny]
```

**⚠️ 注意**：
- 壓力場 `p` 在 LES 資料中可能不存在或為零
- 座標可以是 1D 或 2D meshgrid
- 所有變數必須為 float32 或 float64

---

## 🧪 測試與驗證

### 單元測試覆蓋
```bash
pytest tests/ -v --cov=pinnx
```

**核心模組測試**：
- ✅ `test_lowfi_loader.py` - 資料載入
- ✅ `test_rans_integration.py` - RANS 整合
- ✅ `test_losses.py` - 損失函數
- ✅ `test_physics.py` - 物理方程
- ✅ `test_sensors_integration.py` - 感測器選擇

### 物理驗證指標

**守恆律**：
- ✅ 質量守恆：∇·u < 1e-2
- ✅ 動量守恆：殘差 < 1e-1

**邊界條件**：
- ✅ 壁面無滑移：|u_wall| < 1e-3
- ✅ 週期性：|u(x=0) - u(x=L)| < 1e-3

**湍流統計**：
- ✅ 壁面剪應力 τ_w 誤差 < 20%
- ✅ 平均速度剖面相關性 > 0.8

---

## 📊 效能指標

### 訓練效率
- **2D Kolmogorov**: ~10-20 分鐘 (GPU, 1000 epochs)
- **3D Channel (slab)**: ~1-2 小時 (GPU, 5000 epochs)
- **3D Channel (full)**: ~6-8 小時 (GPU, 10000 epochs)

### 記憶體佔用
- **2D**: ~2-4 GB GPU memory
- **3D Slab**: ~8-12 GB GPU memory
- **3D Full**: ~16-24 GB GPU memory

### 重建精度
- **速度場 L2 誤差**: 10-15% (目標)
- **壓力場 L2 誤差**: 15-20% (目標)
- **相對於 RANS baseline**: >30% 改善

---

## 🚀 未來擴展（如需）

### Phase 3 可能功能
1. **新幾何支援**
   - Cavity flow
   - Backward-facing step
   - Pipe flow

2. **瞬態重建**
   - 時間相關 PINNs
   - 動態感測器重配置
   - 瞬態 RANS prior

3. **高階模型**
   - Ensemble PINNs
   - Multi-fidelity PINNs
   - Physics-guided CNNs

**但目前專案僅專注於**：
- ✅ 2D Kolmogorov + LES
- ✅ 3D Channel + RANS k-ε

---

## 📞 相關文件

- **快速開始**: [QUICK_START.md](QUICK_START.md)
- **技術文檔**: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- **配置指南**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **疑難排解**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **更新日誌**: `../CHANGELOG.md`

---

## 🔒 許可與引用

**License**: MIT  
**Authors**: PINNs-MVP Team  
**Contact**: [專案 GitHub 連結]

**引用格式**：
```bibtex
@software{pinns_mvp_2025,
  title={PINNs-MVP: Sparse Sensor Turbulence Reconstruction},
  author={PINNs-MVP Team},
  year={2025},
  version={1.0.0},
  url={https://github.com/your-org/pinns-mvp}
}
```

---

**文檔維護**: PINNs-MVP 團隊  
**版本**: 1.1.0  
**更新日期**: 2026-01-03
