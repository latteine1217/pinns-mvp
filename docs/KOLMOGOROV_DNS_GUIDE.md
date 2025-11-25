# Kolmogorov Flow DNS 完整指南

## 📋 目錄

1. [快速開始](#快速開始)
2. [物理背景](#物理背景)
3. [使用方法](#使用方法)
4. [實驗結果](#實驗結果)
5. [故障排除](#故障排除)
6. [技術細節](#技術細節)

---

## 🚀 快速開始

### 基本使用（自動檢測 GPU）

```bash
# 自動選擇最佳後端（MPS > CUDA > CPU > NumPy）
python scripts/generate_kolmogorov_dns.py \
  --N 512 \
  --nu 0.003 \
  --T_end 40.0 \
  --output data/kolmogorov_dns.h5

# 手動指定後端
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.003 --T_end 40.0 \
  --backend torch-mps    # 使用 Apple GPU
  
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.003 --T_end 40.0 \
  --backend numpy        # 強制使用 NumPy
```

### 典型參數配置

| 模擬目標 | Re | ν | N | T_end | 預計時間 |
|---------|-----|------|-----|-------|---------|
| **快速測試** | 99 | 0.01257 | 128 | 10 | 1 min |
| **瞬態湍流** | 99 | 0.01257 | 512 | 40 | 7 min |
| **自持湍流** | 1736 | 0.003 | 512 | 40 | 8 min |
| **高解析度** | 1736 | 0.003 | 1024 | 100 | 2-4 hrs |

---

## 🌊 物理背景

### Kolmogorov Flow 定義

二維 Navier-Stokes 方程 + 正弦強迫項：

```
∂u/∂t + (u·∇)u = -∇p + ν∇²u + F₀sin(k_f y)𝐞_x
∇·u = 0
```

**特點**：
- 強迫僅作用於 x 方向
- 波數 k_f 控制條紋數量（k_f=4 → 4 條）
- 經典的湍流轉捩基準問題

### 雷諾數定義

**標準公式**：
```
Re = F₀ / (ν² × k_f³)
```

**參數關係**：
- F₀ (forcing amplitude)：強迫振幅（通常 = 1.0）
- ν (kinematic viscosity)：運動黏滯係數
- k_f (forcing wavenumber)：強迫波數（通常 = 4）

**層流速度**：
```
U_laminar = F₀ / (ν × k_f²)
```

### 湍流轉捩

| 雷諾數範圍 | 流態 | 特徵 |
|-----------|------|------|
| **Re < 30** | 穩定層流 | 完美正弦剖面 |
| **30 ≤ Re < 60** | 弱不穩定 | 小擾動放大 |
| **Re ≥ 60** | 完全湍流 | 多尺度渦街 |
| **Re > 1000** | 深度湍流 | 能量串級明顯 |

---

## 💻 使用方法

### 命令行參數

```bash
python scripts/generate_kolmogorov_dns.py [OPTIONS]

必要參數：
  --N INT              網格大小 (N×N)
  --nu FLOAT           運動黏滯係數
  --T_end FLOAT        模擬時長

可選參數：
  --output PATH        輸出 HDF5 文件路徑 [default: data/kolmogorov_dns_*.h5]
  --dt FLOAT           時間步長 [default: auto-calculated]
  --save_interval INT  保存間隔 [default: 50]
  --k_f INT            強迫波數 [default: 4]
  --F0 FLOAT           強迫振幅 [default: 1.0]
  --backend STR        計算後端 [torch-mps|torch-cuda|torch-cpu|numpy]
  
擾動選項：
  --perturb_time FLOAT       擾動注入時刻
  --perturb_amplitude FLOAT  擾動振幅 [default: 0.1]
  --perturb_mode INT         不穩定模式 [default: 2]
```

### 實例：雷諾數掃描

```bash
# Re = 99 (瞬態湍流)
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.012566 --T_end 40.0 \
  --perturb_time 5.0 --perturb_amplitude 0.1 \
  --output data/dns_re99.h5

# Re = 500 (中等湍流)
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.005 --T_end 40.0 \
  --perturb_time 5.0 --perturb_amplitude 0.1 \
  --output data/dns_re500.h5

# Re = 1736 (自持湍流)
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.003 --T_end 40.0 \
  --perturb_time 5.0 --perturb_amplitude 0.1 \
  --output data/dns_re1736.h5
```

### 視覺化結果

```bash
# 生成分析圖表
python scripts/visualize_dns_results.py \
  --input data/dns_re1736.h5 \
  --output results/dns_re1736_analysis/

# 對比兩個雷諾數
python scripts/compare_reynolds_effects.py \
  --input1 data/dns_re99.h5 \
  --input2 data/dns_re1736.h5 \
  --output results/reynolds_comparison/
```

---

## 📊 實驗結果

### 關鍵發現（Re=99 vs Re=1736）

#### 動能演化

| 指標 | Re=99 | Re=1736 | 差異 |
|------|-------|---------|------|
| **擾動後峰值** | 21.80 | 7.62 | Re↓ 峰值反而更高 |
| **最終動能** | 1.67 | 5.48 | Re↑ 維持更高 |
| **最終/峰值比** | 7.7% | 71.9% | **關鍵差異** |

**物理解釋**：
- **Re=99**: 屬於「瞬態湍流」，擾動激發的湍流無法自我維持，最終衰退回弱湍流
- **Re=1736**: 屬於「自持湍流」，能量注入與耗散達到平衡，湍流結構自我維持

#### 渦度場對比

**Re=99 (瞬態湍流)**：
- ✅ 大尺度條紋結構清晰
- ❌ 缺乏小尺度精細結構
- 📊 黏滯耗散主導，小尺度渦度被快速抹平

**Re=1736 (持續湍流)**：
- ✅ 豐富的多尺度渦街結構
- ✅ 明顯的渦度細絲 (filaments)
- 📊 能量串級活躍，慣性力與耗散力平衡

#### 能譜分析

| 波數範圍 | Re=99 | Re=1736 | Kolmogorov 理論 |
|---------|-------|---------|----------------|
| **k=1-5 (大尺度)** | 較低 | 高 10× | - |
| **k=5-15 (慣性區)** | 快速衰減 | **k⁻⁵/³** 斜率 | ✅ 符合 |
| **k>15 (耗散區)** | 快速衰減 | 緩慢衰減 | - |

### GPU vs CPU 性能對比

#### 數值穩定性

| 模擬 | 框架 | Re | 結果 |
|------|------|-----|------|
| CPU | NumPy | 99 | ✅ 成功 |
| CPU | NumPy | 2073 | ❌ t=5.1 時 NaN |
| **GPU** | **PyTorch MPS** | **1736** | **✅ 成功** |

**關鍵發現**：PyTorch GPU 版本在高雷諾數下更穩健

#### 計算效能

| 網格大小 | NumPy CPU | PyTorch CPU | PyTorch MPS | PyTorch CUDA |
|---------|-----------|-------------|-------------|--------------|
| 128×128 | 500 steps/s | 400 steps/s | 600 steps/s | 1000 steps/s |
| 512×512 | 40 steps/s | 35 steps/s | **55 steps/s** | 150 steps/s |
| 1024×1024 | 10 steps/s | 8 steps/s | **30 steps/s** | 80 steps/s |

**注意**：
- MPS (Apple GPU) 在 ≥1024×1024 時優勢明顯（3-5× 加速）
- 小網格（≤256×256）性能差異不大

---

## 🛠️ 故障排除

### 常見問題

#### 1. NaN 或數值爆炸

**症狀**：模擬中途出現 `NaN` 或 `Inf`

**原因**：
- 時間步長過大
- 雷諾數過高（高 Re 需要更精細網格）
- 初始條件過於劇烈

**解決方案**：
```bash
# 方案 1: 減小時間步長
python scripts/generate_kolmogorov_dns.py --N 512 --nu 0.003 --dt 0.0005 ...

# 方案 2: 增加網格解析度
python scripts/generate_kolmogorov_dns.py --N 1024 --nu 0.003 ...

# 方案 3: 使用 GPU 後端（更穩健）
python scripts/generate_kolmogorov_dns.py --backend torch-mps ...
```

#### 2. 散度誤差過高

**症狀**：`div(u) > 10⁻¹`

**預期值**：
- NumPy: ~10⁻¹⁶（機器精度）
- PyTorch GPU: ~10⁻²（可接受）

**如果 > 10⁻¹**：
- 檢查網格解析度是否足夠
- 確認邊界條件設置正確
- 減小時間步長

#### 3. 內存不足

**症狀**：`MemoryError` 或 GPU OOM

**內存需求估算**：
- 512×512: ~500 MB
- 1024×1024: ~2 GB
- 2048×2048: ~8 GB

**解決方案**：
```bash
# 增加保存間隔（減少內存峰值）
python scripts/generate_kolmogorov_dns.py --save_interval 200 ...

# 減少網格大小
python scripts/generate_kolmogorov_dns.py --N 512 ...  # instead of 1024
```

#### 4. 湍流未激發

**症狀**：動能始終很低，無湍流轉捩

**可能原因**：
- 雷諾數太低（Re < 60）
- 未添加擾動或擾動過弱

**解決方案**：
```bash
# 添加擾動
python scripts/generate_kolmogorov_dns.py \
  --perturb_time 5.0 \
  --perturb_amplitude 0.1 \
  --perturb_mode 2 \
  ...

# 提高雷諾數（降低 ν）
python scripts/generate_kolmogorov_dns.py --nu 0.003 ...  # Re~1700
```

---

## 🔬 技術細節

### 數值方法

**Pseudo-Spectral 方法**：
- 空間離散：Fourier 譜方法（週期邊界）
- 時間積分：Semi-implicit Euler
- 去混疊：2/3 規則（防止高波數混疊）

**時間步長自動計算**：
```python
dt = min(0.1 / (ν * k_max²), 0.5 / U_max)
```
- CFL 條件：保證數值穩定
- 黏滯限制：防止過度耗散

### HDF5 數據結構

```python
kolmogorov_dns.h5
├── snapshots/
│   ├── U [n_frames, N, N]       # x 方向速度
│   ├── V [n_frames, N, N]       # y 方向速度
│   ├── P [n_frames, N, N]       # 壓力場
│   ├── vorticity [n_frames, N, N]  # 渦度
│   └── time [n_frames]          # 時間戳
├── diagnostics/
│   ├── kinetic_energy [n_frames]
│   ├── enstrophy [n_frames]
│   ├── divergence_error [n_frames]
│   └── energy_growth_rate [n_frames]
├── config/
│   ├── attrs['N']               # 網格大小
│   ├── attrs['nu']              # 黏滯係數
│   ├── attrs['Re']              # 雷諾數
│   ├── attrs['backend']         # 計算後端
│   └── ...
└── grid/
    ├── x [N, N]
    ├── y [N, N]
    └── k [N, N]                 # 波數網格
```

### 後端選擇邏輯

```
1. PyTorch 可用？
   ├─ 是 → 檢查 MPS (Apple GPU)
   │       ├─ 可用 → 使用 torch-mps ⭐
   │       └─ 不可用 → 檢查 CUDA
   │                  ├─ 可用 → 使用 torch-cuda
   │                  └─ 不可用 → 使用 torch-cpu
   └─ 否 → 使用 numpy
```

**手動覆蓋**：
```bash
--backend torch-mps    # 強制 Apple GPU
--backend torch-cuda   # 強制 NVIDIA GPU
--backend torch-cpu    # 強制 PyTorch CPU
--backend numpy        # 強制 NumPy（舊版）
```

---

## 📚 參考文獻

1. **Kolmogorov Flow 理論**：
   - Meshalkin & Sinai (1961): "Investigation of the stability of a stationary solution of a system of equations for the plane movement of an incompressible viscous liquid"
   
2. **Pseudo-Spectral 方法**：
   - Canuto et al. (2007): "Spectral Methods: Fundamentals in Single Domains"
   
3. **2D 湍流**：
   - Boffetta & Ecke (2012): "Two-dimensional turbulence"

---

## 🎯 下一步建議

### 1. 臨界雷諾數掃描
```bash
# 測試 Re = [100, 200, 500, 1000, 1500, 2000]
for nu in 0.01257 0.0063 0.0039 0.0031 0.0027; do
  python scripts/generate_kolmogorov_dns.py --N 512 --nu $nu --T_end 40.0
done

# 繪製湍流維持率 vs Re 曲線
python scripts/analyze_reynolds_transition.py
```

### 2. 長時間統計
```bash
# T_end = 200（驗證統計穩定性）
python scripts/generate_kolmogorov_dns.py \
  --N 1024 --nu 0.003 --T_end 200.0 \
  --output data/dns_re1736_long.h5

# 計算雙點相關、結構函數
python scripts/analyze_turbulence_statistics.py \
  --input data/dns_re1736_long.h5
```

### 3. 高解析度模擬
```bash
# 2048×2048（需要 GPU）
python scripts/generate_kolmogorov_dns.py \
  --N 2048 --nu 0.003 --T_end 40.0 \
  --backend torch-cuda \
  --output data/dns_re1736_hires.h5
```

---

**最後更新**：2025-11-22  
**維護者**：PINNs-MVP 團隊  
**版本**：2.0
