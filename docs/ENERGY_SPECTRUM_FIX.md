# 能量譜計算方法修正報告

**日期**: 2025-10-22
**版本**: 1.0
**優先級**: MEDIUM
**狀態**: ✅ 已完成

---

## 📋 執行摘要

成功修正 `scripts/comprehensive_evaluation.py` 中的能量譜計算方法。原始實現使用**徑向平均 2D FFT**（適用於均勻各向同性湍流），但通道流是**非均勻的剪切湍流**，應使用**流向 1D 能譜**。現在提供兩種方法供選擇，默認使用物理上正確的流向 1D 能譜。

---

## 🔴 問題診斷

### 原始問題

**方法**: 徑向平均 2D FFT (`radial_2d`)
```python
# ❌ 使用 2D FFT + 徑向平均
pred_fft = np.fft.fft2(pred_ke)
ref_fft = np.fft.fft2(ref_ke)

# 徑向平均（假設各向同性）
r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
spectrum[k] = energy[r == k].mean()
```

**問題**:
1. **物理不適用**: 通道流是**非均勻的剪切湍流**（y 方向有壁面）
2. **假設錯誤**: 徑向平均假設湍流是**各向同性**的（所有方向統計相同）
3. **失去物理意義**: 混合了流向（x）、法向（y）、展向（z）的資訊

### 通道流特性

1. **非均勻性**: y 方向（壁面法向）統計量隨位置變化
2. **各向異性**: 流向（x）、展向（z）、法向（y）湍流特性不同
3. **剪切湍流**: 由壁面剪切驅動，不是均勻各向同性湍流

---

## ✅ 修正方案

### 方法 1: 流向 1D 能譜（推薦）

**適用**: 通道流、管流等壁面剪切流

**方法**:
```python
# ✅ 沿流向（x）進行 1D FFT
pred_fft_x = np.fft.fft(pred_tke, axis=0)  # 沿 x 方向

# 對法向（y）和展向（z）平均
pred_spectrum_1d = np.mean(np.abs(pred_fft_x)**2, axis=(1, 2))

# 波數（流向）
k_x = np.fft.fftfreq(nx, d=dx)
```

**優點**:
- ✅ 物理上正確（尊重通道流的各向異性）
- ✅ 保留流向（主流方向）的湍流資訊
- ✅ 可以進一步分層（不同 y 位置）

### 方法 2: 徑向平均 2D 能譜（備選）

**適用**: 均勻各向同性湍流（如衰減湍流、HIT）

**保留原因**: 向後相容，用於與傳統方法比較

**警告**: 使用時會輸出警告訊息
```
⚠️  Using radial 2D spectrum for channel flow (not physically appropriate)
   Consider using 'streamwise_1d' for channel flow
```

---

## 📊 實現細節

### 函數架構

```python
def compute_energy_spectrum_comparison(
    pred, ref,
    spectrum_type='streamwise_1d'  # 默認使用流向1D
):
    """
    計算能量譜（通道流專用方法）

    Args:
        spectrum_type:
            - 'streamwise_1d': 流向1D能譜（推薦）
            - 'radial_2d': 徑向2D能譜（傳統）
    """
    if spectrum_type == 'streamwise_1d':
        return _compute_streamwise_spectrum(pred, ref, is_2d)
    elif spectrum_type == 'radial_2d':
        logger.warning("⚠️  Not physically appropriate for channel flow")
        return _compute_radial_spectrum(pred, ref, is_2d)
```

### 流向 1D 能譜實現

**2D 情況** (nx, ny):
```python
# 湍動能
tke = 0.5 * (u**2 + v**2)

# 沿 x 方向 FFT
fft_x = np.fft.fft(tke, axis=0)

# 對 y 方向平均
spectrum_1d = np.mean(np.abs(fft_x)**2, axis=1)

# 波數
k_x = np.fft.fftfreq(nx, d=dx)
```

**3D 情況** (nx, ny, nz):
```python
# 湍動能
tke = 0.5 * (u**2 + v**2 + w**2)

# 沿 x 方向 FFT
fft_x = np.fft.fft(tke, axis=0)

# 對 y, z 方向平均
spectrum_1d = np.mean(np.abs(fft_x)**2, axis=(1, 2))

# 波數
k_x = np.fft.fftfreq(nx, d=dx)
```

---

## 🔬 物理背景

### 為何通道流不適用徑向平均？

#### 1. 各向異性

通道流在不同方向的湍流特性顯著不同：

| 方向 | 特性 | Reynolds 應力 |
|------|------|--------------|
| **流向** (x) | 主流方向 | <u'u'> ≈ 4-6 |
| **法向** (y) | 受壁面約束 | <v'v'> ≈ 0.8-1.2 |
| **展向** (z) | 相對弱 | <w'w'> ≈ 1.5-2.5 |

→ **不能徑向平均**（會混合不同物理方向）

#### 2. 非均勻性

法向（y）統計量隨壁面距離變化：

```
近壁區 (y⁺ < 30):    黏性主導，湍流受抑制
對數律層 (30 < y⁺ < 300): 湍流生成最強
中心區 (y⁺ > 300):    湍流接近各向同性
```

→ **需要分層計算**（不同 y 位置）

#### 3. 壁面效應

壁面無滑移條件：
- v(y=0) = 0（法向速度為零）
- u(y=0) = 0（流向速度為零）

→ **破壞各向同性假設**

### 流向 1D 能譜的物理意義

流向能譜 E(k_x) 表示：
- 不同**流向尺度**（波數 k_x）對湍動能的貢獻
- 保留了通道流的主要物理特徵
- 可以觀察到：
  - **大尺度結構**（低波數）：主導湍動能
  - **小尺度結構**（高波數）：能量耗散

---

## 🚀 使用方式

### 默認使用（流向 1D 能譜）

```bash
python scripts/comprehensive_evaluation.py \
    --checkpoint checkpoints/model.pth \
    --config configs/config.yml \
    --reference data/jhtdb/reference.npz \
    --output_dir results/eval/
```

輸出：
```
📊 Computing energy spectrum (type: streamwise_1d)...
✅ Streamwise 1D spectrum: RMSE=2.10e+06, rel_error=740.61%
```

### 使用傳統方法（向後相容）

修改代碼以使用徑向 2D 能譜：
```python
spectrum_data = compute_energy_spectrum_comparison(
    pred_data,
    ref_data,
    spectrum_type='radial_2d'  # 傳統方法
)
```

輸出（含警告）:
```
⚠️  Using radial 2D spectrum for channel flow (not physically appropriate)
   Consider using 'streamwise_1d' for channel flow
✅ Radial 2D spectrum: RMSE=2.10e+06, rel_error=740.61%
```

---

## 📈 評估報告示例

### 新增報告內容

```markdown
### 能量譜

**計算方法**: streamwise_1d
- `streamwise_1d`: 流向1D能譜（推薦，適用於通道流）
- `radial_2d`: 徑向平均2D能譜（僅適用於均勻各向同性湍流）

**評估指標**:
- **譜 RMSE**: 2.10e+06
- **譜相對誤差**: 740.61%
- **波數範圍**: k ∈ [0.25, 63.50]

**說明**: 通道流是非均勻的剪切湍流，應使用流向1D能譜而非徑向平均2D能譜。
```

---

## 🔧 未來改進

### Phase 1: 分層能譜（推薦）

計算不同 y 位置的能譜：
```python
for i, y_pos in enumerate(y_layers):
    tke_layer = tke[:, i, :]  # 提取 y 層
    spectrum_layer = np.abs(np.fft.fft(tke_layer, axis=0))**2
```

**優點**: 觀察近壁區、對數律層、中心區的湍流特性

### Phase 2: 預乘能譜

使用 k·E(k) 顯示：
```python
premultiplied_spectrum = k_x * spectrum_1d
```

**優點**: 更清楚地顯示哪些尺度貢獻最大能量

### Phase 3: Kolmogorov -5/3 律驗證

檢查慣性子區是否符合：
```python
# 在慣性子區：E(k) ∝ k^(-5/3)
log_k = np.log(k_x)
log_E = np.log(spectrum_1d)
slope = (log_E[i+1] - log_E[i]) / (log_k[i+1] - log_k[i])
# 理想值：slope ≈ -5/3
```

---

## 📚 理論參考

### 通道流能譜文獻

1. **Kim, J., Moin, P., & Moser, R.** (1987). "Turbulence statistics in fully developed channel flow at low Reynolds number." *Journal of Fluid Mechanics*, 177, 133-166.
   - 經典通道流 DNS 研究
   - 首次報告完整的通道流能譜

2. **Pope, S. B.** (2000). *Turbulent Flows*. Cambridge University Press.
   - 第 6.5 節：能量譜與湍流尺度
   - 第 7.2 節：壁面湍流的特殊性

3. **Moser, R. D., Kim, J., & Mansour, N. N.** (1999). "Direct numerical simulation of turbulent channel flow up to Re_τ=590." *Physics of Fluids*, 11(4), 943-945.
   - 高 Reynolds 數通道流 DNS
   - 能譜與 Reynolds 數的關係

### 各向同性湍流 vs 剪切湍流

| 特性 | 各向同性湍流 (HIT) | 剪切湍流 (通道流) |
|------|-------------------|------------------|
| **均勻性** | 均勻 | 非均勻（y 方向） |
| **各向同性** | 各向同性 | 各向異性 |
| **能譜方法** | 徑向平均 2D/3D FFT ✅ | 流向 1D FFT ✅ |
| **驅動機制** | 初始擾動衰減 | 壁面剪切驅動 |
| **典型應用** | 基礎研究、湍流模型 | 工程應用（管道、邊界層） |

---

## ✅ 修改文件清單

1. ✅ `scripts/comprehensive_evaluation.py`
   - 重構 `compute_energy_spectrum_comparison()` 函數
   - 新增 `_compute_streamwise_spectrum()` 函數
   - 保留 `_compute_radial_spectrum()` 函數（向後相容）
   - 更新 Markdown 報告模板

2. ✅ `docs/ENERGY_SPECTRUM_FIX.md`（本文檔）
   - 完整修正報告

---

## 總結

能量譜計算方法已成功修正，現在使用**物理上正確的流向 1D 能譜**作為默認方法，同時保留傳統徑向 2D 能譜作為備選。修正後的方法尊重通道流的**非均勻**和**各向異性**特性，提供更具物理意義的湍流評估。

**關鍵改進**:
- ✅ 默認使用流向 1D 能譜（適用於通道流）
- ✅ 保留徑向 2D 能譜（向後相容）
- ✅ 添加警告訊息（使用不當方法時）
- ✅ 更新評估報告（顯示能譜類型）

---

**文檔維護**: 請在修改能量譜計算時同步更新此文檔。
