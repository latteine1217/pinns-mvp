# 週期性 Fourier 嵌入使用指南

## 📋 概述

週期性 Fourier 嵌入 (Periodic Fourier Embedding) 是針對週期邊界條件問題設計的特徵編碼方法，通過數學上保證的週期性映射，自動滿足週期邊界條件，無需額外的軟約束損失函數。

### 🎯 適用場景

- **Kolmogorov Flow**: 雙週期域 (x, y ∈ [0, 2π])
- **週期性槽道流**: x, z 週期，y 壁面
- **時空混合問題**: 空間週期 + 時間非週期

### ✨ 核心優勢

1. **嚴格週期性**: φ(0) = φ(L) 數學保證，無需軟約束
2. **頻率對齊**: 頻率與域週期對齊，避免頻譜洩漏
3. **簡化配置**: 移除 `periodicity_weight`，減少超參數調整
4. **軸向靈活**: 不同軸可使用不同編碼方式（週期/標準/無）

---

## 🔧 配置方式

### 1. 基本配置結構

```yaml
model:
  fourier_features:
    type: hybrid  # 使用混合型 Fourier 嵌入
    
    # 軸向配置：指定每個軸的編碼方式
    axes:
      # 軸 0 配置
      0:
        type: periodic | standard | none
        # ... 其他參數
      
      # 軸 1 配置
      1:
        type: periodic | standard | none
        # ... 其他參數
```

### 2. 軸編碼類型

#### 🔄 週期性嵌入 (`periodic`)

適用於週期邊界軸（如 Kolmogorov flow 的 x, y）

```yaml
1:
  type: periodic
  domain_size: 6.283185307179586  # 2π（週期域大小）
  n_modes: 8  # 傅立葉模態數量
```

**參數說明**:
- `domain_size`: 週期域大小 L，例如 2π
- `n_modes`: 使用的傅立葉模態數量 k = 1, 2, ..., n_modes
- 輸出維度: `2 * n_modes`（sin + cos）

**數學原理**:
```
φ(x) = [sin(2πkx/L), cos(2πkx/L)] for k = 1, 2, ..., n_modes
```

#### 🎲 標準 Fourier (`standard`)

適用於非週期軸（如時間 t）

```yaml
0:
  type: standard
  n_modes: 12
  sigma: 4.0
  use_2pi: true
```

**參數說明**:
- `n_modes`: Fourier 特徵數量
- `sigma`: 頻率尺度參數（控制頻譜寬度）
- `use_2pi`: 是否乘以 2π（預設 true）
- 輸出維度: `2 * n_modes`

**數學原理**:
```
φ(x) = [cos(2πωx), sin(2πωx)], 其中 ω ~ N(0, σ²)
```

#### 🚫 無編碼 (`none`)

直接透傳原始座標（不推薦）

```yaml
0:
  type: none
```

---

## 📝 完整配置範例

### Kolmogorov Flow 2D+T（推薦配置）

```yaml
model:
  type: fourier_vs_mlp
  in_dim: 3   # [t, x, y]
  out_dim: 3  # [u, v, p]
  width: 256
  depth: 6
  activation: swish
  block_type: resnet
  
  fourier_features:
    type: hybrid
    axes:
      # 時間軸：標準 Fourier（非週期）
      0:
        type: standard
        n_modes: 12
        sigma: 4.0
        use_2pi: true
      
      # 空間 x 軸：週期性嵌入
      1:
        type: periodic
        domain_size: 6.283185307179586  # 2π
        n_modes: 8
      
      # 空間 y 軸：週期性嵌入
      2:
        type: periodic
        domain_size: 6.283185307179586  # 2π
        n_modes: 8
    
    trainable_fourier: false

# 🔧 損失函數配置（移除 periodicity_weight）
losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 2.0
  prior_weight: 10.0
  # periodicity_weight: 0.0  # 🔧 不再需要週期性軟約束
```

### Kolmogorov Flow 2D 穩態

```yaml
model:
  in_dim: 2   # [x, y]
  
  fourier_features:
    type: hybrid
    axes:
      # 空間 x 軸
      0:
        type: periodic
        domain_size: 6.283185307179586
        n_modes: 8
      
      # 空間 y 軸
      1:
        type: periodic
        domain_size: 6.283185307179586
        n_modes: 8
```

---

## 🧪 驗證與測試

### 1. 配置驗證

使用提供的測試腳本驗證配置：

```bash
python test_periodic_fourier_config.py
```

**預期輸出**:
```
✅ 模型創建成功
✅ x 方向週期性 通過 (誤差: 1.8e-09)
✅ y 方向週期性 通過 (誤差: 2.9e-09)
✅ 梯度計算成功
```

### 2. 週期性檢查

```python
import torch
from pinnx.models import create_pinn_model

# 創建模型
model = create_pinn_model(config)

# 測試週期性
domain_size = 2 * 3.14159265
x_test = torch.tensor([
    [0.0, 0.0, 0.0],           # x=0, y=0
    [0.0, domain_size, 0.0],   # x=2π, y=0
])

with torch.no_grad():
    out = model(x_test)

diff = torch.abs(out[0] - out[1]).max().item()
print(f"週期性誤差: {diff:.2e}")  # 應 < 1e-6
```

---

## 📊 效果對比

| 特性 | 標準 Fourier + Soft Constraint | 週期性嵌入 |
|------|-------------------------------|-----------|
| 週期性保證 | ❌ 軟約束，依賴權重調整 | ✅ 數學保證 |
| 超參數 | 需調整 `periodicity_weight` | 無需週期性權重 |
| 訓練穩定性 | ⚠️ 權重衝突風險 | ✅ 更穩定 |
| 邊界精度 | ~10⁻⁴ - 10⁻⁶ | < 10⁻⁸ |

---

## ⚙️ 超參數調整指南

### `n_modes` (模態數量)

- **週期軸**: 建議 6-12
- **非週期軸**: 建議 12-24
- **原則**: 需捕捉的最高頻率特徵數量

```yaml
# 低頻主導（層流）
n_modes: 6

# 中頻特徵（過渡流）
n_modes: 8-12

# 高頻特徵（湍流）
n_modes: 12-16
```

### `sigma` (標準 Fourier 頻率尺度)

- **時間軸**: 3.0-5.0
- **非週期空間軸**: 2.0-4.0

```yaml
# 平滑變化
sigma: 2.0-3.0

# 中等頻率
sigma: 4.0-5.0

# 高頻捕捉
sigma: 6.0-8.0
```

### `domain_size` (週期域大小)

必須與物理域精確匹配：

```yaml
# Kolmogorov flow
domain_size: 6.283185307179586  # 2π

# 自定義週期域
domain_size: <your_domain_length>
```

---

## 🐛 常見問題

### Q1: 週期性誤差仍然較大（> 10⁻⁴）

**可能原因**:
- `domain_size` 與物理域不匹配
- 使用了 `normalize_input` 但未正確還原

**解決方案**:
```yaml
# 確保 domain_size 精確
domain_size: 6.283185307179586  # 使用完整精度

# 禁用輸入標準化（週期性嵌入不需要）
fourier_normalize_input: false
```

### Q2: 訓練不穩定

**可能原因**:
- `n_modes` 過多導致過度參數化
- `sigma` 設置不當

**解決方案**:
```yaml
# 減少模態數量
n_modes: 6-8  # 從較小值開始

# 調整頻率尺度
sigma: 3.0-4.0  # 保守值
```

### Q3: 與舊配置的遷移

**舊配置**:
```yaml
fourier_features:
  type: standard
  fourier_m: 16
  fourier_sigma: 4.0

losses:
  periodicity_weight: 10.0  # 需要週期性約束
```

**新配置**:
```yaml
fourier_features:
  type: hybrid
  axes:
    0: {type: standard, n_modes: 12, sigma: 4.0}
    1: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}
    2: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}

losses:
  # periodicity_weight: 移除！
  data_weight: 10.0
  momentum_x_weight: 1.0
  ...
```

---

## 📚 相關文件

- **示範配置**: `configs/kolmogorov_re50_kf4_K100_periodic_fourier.yml`
- **測試腳本**: `test_periodic_fourier_config.py`
- **源碼**:
  - 週期性嵌入: `pinnx/models/hybrid_fourier.py`
  - 模型整合: `pinnx/models/fourier_mlp.py`

---

## 🔬 技術細節

### 數學原理

週期性嵌入基於傅立葉級數展開：

```
對於週期函數 f(x) = f(x + L)，可展開為：
f(x) = Σ [aₖ cos(2πkx/L) + bₖ sin(2πkx/L)]
```

網路學習係數 aₖ, bₖ，天然滿足週期性。

### 實現細節

```python
# 週期性頻率
ω_k = 2πk / L  for k = 1, 2, ..., n_modes

# 特徵映射
φ(x) = [sin(ω₁x), ..., sin(ωₙx), cos(ω₁x), ..., cos(ωₙx)]
```

**關鍵特性**:
- φ(0) = φ(L) 自動滿足
- 頻率整數倍，避免頻譜洩漏
- 梯度連續，適合 PINNs 自動微分

---

## ✅ 檢查清單

在使用週期性嵌入前，確認：

- [ ] 確定哪些軸需要週期性（x, y, z）
- [ ] 確定哪些軸非週期（t, 壁法向）
- [ ] 正確設置 `domain_size`（與物理域匹配）
- [ ] 選擇合適的 `n_modes`（6-12 for 週期軸）
- [ ] 移除配置中的 `periodicity_weight`
- [ ] 執行 `test_periodic_fourier_config.py` 驗證
- [ ] 檢查週期性誤差 < 10⁻⁶

---

**最後更新**: 2026-01-06  
**作者**: PINNs-MVP 團隊
