# PINNs 評估最佳實踐指南

> **版本**: 1.0  
> **日期**: 2026-01-05  
> **作者**: OpenCode Agent

---

## 📋 目錄

1. [評估流程概述](#評估流程概述)
2. [反標準化策略](#反標準化策略)
3. [壓力場評估注意事項](#壓力場評估注意事項)
4. [物理指標計算](#物理指標計算)
5. [常見錯誤與解決方案](#常見錯誤與解決方案)
6. [評估腳本使用](#評估腳本使用)

---

## 評估流程概述

### 標準評估流程

```
1. 載入 Checkpoint → 2. 恢復模型架構 → 3. 推理預測 → 4. 反標準化 → 5. 計算誤差 → 6. 物理驗證
```

### 關鍵原則

1. **模型輸出≠物理量**：訓練時使用標準化（Z-score），評估時必須反標準化回物理空間
2. **Checkpoint 優先**：優先使用 checkpoint 中嵌入的配置，避免架構不匹配
3. **物理一致性**：除了數值誤差，還需檢查質量守恆、邊界條件、能量譜等物理約束
4. **壓力場特殊性**：壓力只定義到任意常數，應使用壓力梯度評估

---

## 反標準化策略

### 為什麼需要反標準化？

**訓練階段**（標準化空間）：
```python
# 感測器數據標準化（Z-score）
normalized_data = (data - mean) / std

# 模型學習標準化空間的映射
model_output = model(coords)  # 輸出也是標準化的
```

**評估階段**（物理空間）：
```python
# 模型輸出仍是標準化空間
model_output = model(coords)  # ❌ 不能直接與真實值比較

# 必須反標準化回物理空間
physical_output = model_output * std + mean  # ✅ 正確
```

### 統一反標準化介面

使用 `pinnx.utils.denormalization.denormalize_output()`：

```python
from pinnx.utils.denormalization import denormalize_output

# 模型推理
pred_normalized = model(coords)

# 反標準化
pred_physical = denormalize_output(
    predictions=pred_normalized.cpu().numpy(),
    config=config,
    output_norm_type='training_data_norm',
    checkpoint_path='checkpoints/model.pth',
    verbose=True  # 第一次呼叫時啟用，用於驗證
)
```

### 反標準化載入優先級

```
1. Checkpoint metadata（最高優先級）
   └─ checkpoint['normalization']['means/stds']

2. Config normalization.params
   └─ config['normalization']['params']['means/stds']

3. 硬編碼 JHTDB 預設值（最低優先級，僅用於後備）
   └─ u_mean=9.92, u_std=4.59, etc.
```

### 支援的標準化類型

| 類型 | 描述 | 反標準化公式 |
|------|------|--------------|
| `training_data_norm` | Z-score 標準化 | `x_phys = x_norm * std + mean` |
| `friction_velocity` | 摩擦速度縮放 | `u = u_norm * u_τ`, `p = p_norm * ρu_τ²` |
| `manual_scaling` | 手動範圍縮放 | `x_phys = x_norm * (max - min) + min` |
| `none` / `identity` | 無標準化 | `x_phys = x_norm` |

---

## 壓力場評估注意事項

### ⚠️ 壓力場的特殊性

**物理事實**：不可壓縮流體的壓力場只定義到任意常數 `C`：
```
p(x, y, z) = p_true(x, y, z) + C
```

### ❌ 錯誤做法

```python
# 絕對壓力誤差（無意義！）
p_error = np.linalg.norm(p_pred - p_true) / np.linalg.norm(p_true)
```

**問題**：即使 `p_pred` 和 `p_true` 只差一個常數，誤差也可能很大。

### ✅ 正確做法

#### 方法 1：壓力梯度評估（推薦）

```python
from comprehensive_evaluation import compute_pressure_gradient_error

# 計算壓力梯度誤差
dp_dx_pred = np.gradient(p_pred, x, axis=0)
dp_dy_pred = np.gradient(p_pred, y, axis=1)
dp_dz_pred = np.gradient(p_pred, z, axis=2)

dp_dx_true = np.gradient(p_true, x, axis=0)
dp_dy_true = np.gradient(p_true, y, axis=1)
dp_dz_true = np.gradient(p_true, z, axis=2)

# 相對誤差
grad_error_x = np.linalg.norm(dp_dx_pred - dp_dx_true) / np.linalg.norm(dp_dx_true)
grad_error_y = np.linalg.norm(dp_dy_pred - dp_dy_true) / np.linalg.norm(dp_dy_true)
grad_error_z = np.linalg.norm(dp_dz_pred - dp_dz_true) / np.linalg.norm(dp_dz_true)

grad_error_mean = (grad_error_x + grad_error_y + grad_error_z) / 3
```

#### 方法 2：減去平均值後比較

```python
# 移除任意常數（使兩個場均值為 0）
p_pred_centered = p_pred - np.mean(p_pred)
p_true_centered = p_true - np.mean(p_true)

p_error = np.linalg.norm(p_pred_centered - p_true_centered) / np.linalg.norm(p_true_centered)
```

### 參考實現

最完整的壓力評估實現位於 `comprehensive_evaluation.py:410-454`。

---

## 物理指標計算

### 1. 質量守恆（散度誤差）

```python
# 計算速度散度
div_u = np.gradient(u, x, axis=0) + np.gradient(v, y, axis=1) + np.gradient(w, z, axis=2)

# 平均散度（應接近 0）
div_mean = np.mean(np.abs(div_u))
div_max = np.max(np.abs(div_u))

# 驗證標準：
# - div_mean < 1e-2 （良好）
# - div_mean < 1e-3 （優秀）
```

### 2. 牆面剪應力（Wall Shear Stress）

**正確公式**（包含黏度係數）：
```python
# τ_w = μ * (∂u/∂y)|_wall = ρ * ν * (∂u/∂y)|_wall
nu = 5e-5  # JHTDB Re_τ=1000 的運動黏度
rho = 1.0  # 密度

# 計算牆面法向速度梯度
du_dy_wall = np.gradient(u[:, 0, :], y, axis=0)  # 下牆面 (y=0)
tau_w_pred = rho * nu * du_dy_wall

# 與理論值比較（JHTDB Re_τ=1000: τ_w ≈ 0.0025）
tau_w_theory = 0.0025
tau_w_error = np.abs(tau_w_pred - tau_w_theory) / tau_w_theory
```

**⚠️ 常見錯誤**：
```python
# ❌ 缺少黏度係數
tau_w_wrong = np.gradient(u[:, 0, :], y, axis=0)
```

### 3. 能量譜（Energy Spectrum）

#### 湍流類型判斷

| 流場類型 | 適用譜方法 | FFT 方向 |
|----------|-----------|----------|
| **均勻各向同性湍流** | 徑向 2D 譜 | 所有方向 |
| **通道流（非均勻剪切湍流）** | 流向 1D 譜 | 僅 x 方向 |

#### 正確實現（3D 通道流）

```python
# 流向 1D 能量譜（推薦）
def compute_streamwise_1d_spectrum(u, v, w, dx):
    """
    沿流向(x)計算 1D 能量譜，並對 y, z 方向平均
    """
    nx, ny, nz = u.shape
    
    # 計算 x 方向 FFT
    u_fft = np.fft.rfft(u, axis=0)
    v_fft = np.fft.rfft(v, axis=0)
    w_fft = np.fft.rfft(w, axis=0)
    
    # 能量密度
    E_k = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)
    
    # 對 y, z 平均
    E_k_avg = np.mean(E_k, axis=(1, 2))
    
    # 波數
    k_x = 2 * np.pi * np.fft.rfftfreq(nx, d=dx)
    
    return k_x, E_k_avg
```

#### ⚠️ 錯誤做法

```python
# ❌ 對非均勻湍流使用徑向 2D 譜
# 這會將 y 方向的非均勻性（剪切）錯誤地加入譜中
E_k_radial = np.sqrt(k_x**2 + k_y**2)  # 不適用於通道流
```

---

## 常見錯誤與解決方案

### 錯誤 1：未反標準化導致誤差異常

**症狀**：
- 速度場誤差 > 100%
- 預測值範圍在 [-3, 3]，而真實值在 [0, 20]

**原因**：模型輸出仍在標準化空間（Z-score），未反標準化回物理空間

**解決方案**：
```python
# 在所有評估腳本中加入反標準化步驟
from pinnx.utils.denormalization import denormalize_output

pred_physical = denormalize_output(
    pred.cpu().numpy(), config, 
    checkpoint_path=checkpoint_path
)
```

### 錯誤 2：Checkpoint 配置不匹配

**症狀**：
- `RuntimeError: size mismatch for fourier.B`
- 模型載入失敗

**原因**：評估時使用的配置與訓練時不一致（例如 Fourier 特徵維度改變）

**解決方案**：
```python
# 優先使用 checkpoint 中嵌入的配置
if 'config' in checkpoint:
    config = checkpoint['config']
    logger.info("✅ Using config from checkpoint")
else:
    logger.warning("⚠️ Using file config (may cause mismatch)")
```

### 錯誤 3：壓力場絕對誤差評估

**症狀**：
- 速度場誤差 < 10%，但壓力場誤差 > 50%
- 壓力場視覺上看起來很接近，但數值誤差很大

**原因**：壓力只定義到任意常數，不應使用絕對誤差

**解決方案**：
使用壓力梯度誤差或去中心化誤差（見上文「壓力場評估」）

### 錯誤 4：缺少 VS-PINN 座標縮放

**症狀**：
- 載入 VS-PINN checkpoint 後，推理結果異常
- `AttributeError: 'NoneType' object has no attribute 'scale_coordinates'`

**原因**：VS-PINN 在訓練時使用了座標縮放，評估時需恢復 physics 狀態

**解決方案**：
```python
# 使用統一的模型載入函數
from pinnx.utils.evaluation_utils import load_model_for_evaluation

model, physics = load_model_for_evaluation(checkpoint_path, config, device)

# 推理時應用座標縮放
if physics is not None and hasattr(physics, 'scale_coordinates'):
    coords_scaled = physics.scale_coordinates(coords)
    pred = model(coords_scaled)
```

---

## 評估腳本使用

### 1. `comprehensive_evaluation.py`（推薦）

**功能**：最完整的評估流程，包含所有物理指標和正確的反標準化

**使用**：
```bash
python scripts/evaluate/comprehensive_evaluation.py \
    --checkpoint checkpoints/model.pth \
    --config configs/channel_flow.yml \
    --data data/jhtdb/full_field.h5 \
    --output results/comprehensive_eval
```

**特點**：
- ✅ 自動反標準化
- ✅ 壓力梯度評估
- ✅ 完整物理指標（散度、牆面剪應力、能量譜）
- ✅ 豐富的可視化

### 2. `evaluate_checkpoint.py`（✅ 已修復）

**功能**：快速檢查 checkpoint 在感測點上的誤差

**使用**：
```bash
python scripts/evaluate/evaluate_checkpoint.py \
    --checkpoint checkpoints/model.pth \
    --config configs/model.yml \
    --data data/sensors.npz
```

**修復內容（2026-01-05）**：
- ✅ 加入 `denormalize_output()` 呼叫
- ✅ 使用 checkpoint 中的標準化統計量

### 3. `evaluate_curriculum.py`（✅ 已修復）

**功能**：評估課程訓練（Curriculum Learning）的 checkpoint

**使用**：
```bash
python scripts/evaluate/evaluate_curriculum.py \
    --checkpoint checkpoints/curriculum_latest.pth \
    --config configs/curriculum.yml \
    --sensor-file data/sensors.npz \
    --visualize
```

**修復內容（2026-01-05）**：
- ✅ 加入 `denormalize_output()` 呼叫
- ✅ 修改 `evaluate_model()` 和 `visualize_predictions()` 簽名

### 4. `evaluate_kolmogorov_2d.py`

**功能**：Kolmogorov 流（2D）專用評估

**特點**：
- ✅ 已正確使用 `OutputTransform.denormalize_batch()`
- ✅ 2D 能量譜計算正確

### 5. `evaluate.py`

**狀態**：⚠️ 未完成（無入口點，建議使用 `comprehensive_evaluation.py` 替代）

---

## 驗收檢查清單

在發布評估結果前，請確認：

- [ ] **反標準化已啟用**：檢查日誌中是否有 `denormalize_output` 的詳細輸出
- [ ] **數值範圍合理**：預測值範圍應與真實值接近（例如 u ∈ [0, 20]，而非 [-3, 3]）
- [ ] **壓力場使用梯度評估**：不使用絕對誤差
- [ ] **物理一致性通過**：散度 < 1e-2，牆面剪應力誤差 < 20%
- [ ] **能量譜方法正確**：通道流使用流向 1D 譜，均勻湍流使用徑向 2D 譜
- [ ] **Checkpoint 配置匹配**：優先使用 checkpoint 中的配置

---

## 參考資料

### 程式碼位置

- **反標準化工具**：`pinnx/utils/denormalization.py`
- **評估工具**：`pinnx/utils/evaluation_utils.py`
- **標準化配置**：`pinnx/utils/normalization/output_transform.py`
- **物理驗證**：`pinnx/physics/validators.py`

### 相關文檔

- `CONFIG_GUIDE.md`：配置文件規範
- `AGENTS.md`：專案開發準則
- `EXPERIMENT_COMPARISON_PLAN.md`：實驗對比計畫

### 理論參考

- **壓力場評估**：
  - Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press. (Section 6.2)
  - "Pressure in incompressible flow is defined up to an arbitrary constant."

- **能量譜**：
  - Kolmogorov, A. N. (1941). *The local structure of turbulence*.
  - 通道流能量譜：Kim, Moin & Moser (1987) *Turbulence statistics in fully developed channel flow at low Reynolds number*, JFM.

---

**文檔維護**：如發現評估相關問題或新的最佳實踐，請更新本文檔。

**最後更新**：2026-01-05 by OpenCode Agent
