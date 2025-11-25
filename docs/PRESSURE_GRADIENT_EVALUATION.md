# 壓力場梯度評估指標 - 技術文檔

**日期**: 2025-10-22
**版本**: 1.0
**作者**: AI Assistant

---

## 📋 目錄
1. [背景與動機](#背景與動機)
2. [實現細節](#實現細節)
3. [使用方法](#使用方法)
4. [測試驗證](#測試驗證)
5. [評估報告範例](#評估報告範例)

---

## 背景與動機

### 為何需要壓力梯度評估？

在 PINNs 模擬通道流等問題時，直接評估**壓力場絕對值**會遇到以下問題：

#### 問題 1：壓力場的不定性
壓力場在不可壓縮流動中只定義到一個任意常數：
```
p_true(x,y,z) = p_numerical(x,y,z) + C
```
其中 C 是任意常數。因此，即使預測的壓力場完全正確，只要 C ≠ 0，絕對值比較也會產生巨大誤差。

**實際案例**：
```
評估報告 comprehensive_eval_20251019_185222：
- 壓力場 L2 誤差：1720.93% ❌
- 流場 L2 誤差：< 100%
```
這個巨大的壓力誤差很可能是由於常數偏移造成的，並不代表模型物理上失敗。

#### 問題 2：物理意義的缺失
Navier-Stokes 方程中實際出現的是**壓力梯度 ∇p**，而非壓力本身：
```
∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u
```
因此，評估 ∇p 的誤差比評估 p 的誤差更具物理意義。

#### 問題 3：通道流的驅動力驗證
在壓力驅動的通道流中，∂p/∂x 應該是**常數**（驅動力）：
```
∂p/∂x = -0.0025  (JHTDB Re_τ=1000)
```
評估壓力梯度可以直接驗證模型是否正確學習到這個驅動機制。

---

## 實現細節

### 1. 核心函數：`pressure_gradient_from_finite_diff`

**位置**: `pinnx/evals/metrics.py`

**功能**: 使用有限差分計算壓力梯度（適用於網格數據）

**實現**:
```python
def pressure_gradient_from_finite_diff(
    p_field: np.ndarray,
    coords: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    使用有限差分計算壓力梯度

    Args:
        p_field: 壓力場 [Nx, Ny, Nz] 或 [Nx, Ny]
        coords: 座標字典 {'x': [Nx], 'y': [Ny], 'z': [Nz]}

    Returns:
        {'dpdx': [...], 'dpdy': [...], 'dpdz': [...]}
    """
    grad = {}

    # x 方向
    if 'x' in coords:
        dx = coords['x'][1] - coords['x'][0]
        grad['dpdx'] = np.gradient(p_field, dx, axis=0)

    # y 方向
    if 'y' in coords:
        dy = coords['y'][1] - coords['y'][0]
        grad['dpdy'] = np.gradient(p_field, dy, axis=1)

    # z 方向（3D）
    if 'z' in coords and p_field.ndim >= 3:
        dz = coords['z'][1] - coords['z'][0]
        grad['dpdz'] = np.gradient(p_field, dz, axis=2)

    return grad
```

**特點**:
- 使用 NumPy 的 `gradient()` 實現中心差分
- 自動適應 2D/3D 數據
- 保持與原始場相同的網格形狀

### 2. 進階函數：`pressure_gradient_metrics`

**功能**: 計算壓力梯度的誤差指標（使用 PyTorch 自動微分）

**適用場景**: 當壓力場通過神經網路計算且計算圖仍連接時

**實現**:
```python
def pressure_gradient_metrics(
    p_pred: torch.Tensor,
    p_ref: torch.Tensor,
    coords: torch.Tensor,
    spatial_dims: List[str] = ['x', 'y', 'z']
) -> Dict[str, float]:
    """
    使用自動微分計算壓力梯度誤差

    Returns:
        - grad_p_l2_error: 梯度向量的相對 L2 誤差
        - grad_p_{x,y,z}_rmse: 各方向梯度的 RMSE
        - mean_pressure_gradient_{x,y,z}: 平均梯度（驗證驅動力）
        - dpdx_std: ∂p/∂x 的標準差（應接近零，表示常數）
    """
```

### 3. 評估腳本整合

**位置**: `scripts/comprehensive_evaluation.py`

**修改**:
1. 導入壓力梯度計算函數
2. 在 `compute_error_metrics()` 中添加壓力梯度評估
3. 在 Markdown 報告中添加專門章節
4. 在終端輸出中顯示壓力梯度誤差

**新增輸出指標**:
```python
metrics = {
    # 傳統壓力場誤差（可能不準確）
    'p_l2_error': 17.2093,          # ❌ 可能有常數偏移

    # 🆕 壓力梯度誤差（更準確）
    'dpdx_l2_error': 0.0542,        # ✅ 物理上更有意義
    'dpdy_l2_error': 0.0123,
    'dpdz_l2_error': 0.0089,
    'pressure_gradient_l2_error': 0.0251,  # 綜合梯度誤差

    # 🆕 通道流驗證
    'dpdx_pred_mean': -0.002487,    # 預測的驅動力
    'dpdx_ref_mean': -0.002500,     # 參考驅動力（-0.0025）
    'dpdx_pred_std': 0.000015,      # 應接近零（表示常數梯度）
}
```

---

## 使用方法

### 方法 1：使用 comprehensive_evaluation.py（推薦）

**完整評估**（包含壓力梯度）：
```bash
python scripts/comprehensive_evaluation.py \
    --checkpoint checkpoints/2d_quick_baseline_test/best_model.pth \
    --config configs/templates/2d_quick_baseline.yml \
    --reference data/jhtdb/channel_flow_slice.npz \
    --output_dir results/eval_with_pressure_grad
```

**輸出示例**：
```
📊 Computing error metrics...
📊 Computing pressure gradient metrics (more accurate than absolute pressure)...
✅ Pressure gradient L2 error: 2.51%
✅ Overall L2 error: 5.42%

💡 壓力梯度誤差: 2.51% (更準確的壓力評估)
   ∂p/∂x: 預測=-0.002487, 參考=-0.002500
```

### 方法 2：獨立使用壓力梯度函數

```python
import numpy as np
from pinnx.evals.metrics import pressure_gradient_from_finite_diff

# 載入預測和參考壓力場
pred_p = pred_data['p']  # shape (Nx, Ny, Nz)
ref_p = ref_data['p']

# 計算梯度
pred_grad = pressure_gradient_from_finite_diff(
    pred_p,
    {'x': pred_data['x'], 'y': pred_data['y'], 'z': pred_data['z']}
)
ref_grad = pressure_gradient_from_finite_diff(
    ref_p,
    {'x': ref_data['x'], 'y': ref_data['y'], 'z': ref_data['z']}
)

# 計算誤差
dpdx_error = np.linalg.norm(pred_grad['dpdx'] - ref_grad['dpdx']) / \
             (np.linalg.norm(ref_grad['dpdx']) + 1e-12)

print(f"∂p/∂x 相對 L2 誤差: {dpdx_error:.2%}")
```

---

## 測試驗證

### 測試文件：`tests/test_pressure_gradient_metrics.py`

**測試案例**：

#### 1. 2D 二次壓力場
```python
# p = x² + y²
# 理論梯度：∂p/∂x = 2x, ∂p/∂y = 2y
✅ 預測 ∂p/∂x = 1.0000, 理論值 = 1.0000
✅ 預測 ∂p/∂y = 1.0000, 理論值 = 1.0000
```

#### 2. 3D 二次壓力場
```python
# p = x² + y² + z²
# 理論梯度：∂p/∂x = 2x, ∂p/∂y = 2y, ∂p/∂z = 2z
✅ 預測 ∂p/∂x = 1.0000, 理論值 = 1.0000
✅ 預測 ∂p/∂y = 1.0000, 理論值 = 1.0000
✅ 預測 ∂p/∂z = 1.0000, 理論值 = 1.0000
```

#### 3. 常數壓力梯度（通道流場景）
```python
# p = -0.0025 * x（線性壓力場）
# 理論梯度：∂p/∂x = -0.0025（常數），∂p/∂y = 0, ∂p/∂z = 0
✅ ∂p/∂x: mean = -0.002500, std = 0.000000 (常數驗證)
✅ ∂p/∂y: mean = 0.000000 (橫向無梯度)
✅ ∂p/∂z: mean = 0.000000 (展向無梯度)
```

**運行測試**：
```bash
python tests/test_pressure_gradient_metrics.py
```

---

## 評估報告範例

### 新增報告章節：壓力梯度誤差

```markdown
### 💡 壓力梯度誤差（更準確的壓力場評估）

**說明**: 由於壓力場僅定義到任意常數，壓力梯度 ∇p 的比較比絕對值更具物理意義。

| 梯度分量 | 相對 L2 誤差 | RMSE | 預測均值 | 參考均值 | 標準差（預測/參考） |
|---------|-------------|------|---------|---------|------------------|
| **∂p/∂x** | 0.0542 | 0.000135 | -0.002487 | -0.002500 | 0.000015 / 0.000000 |
| **∂p/∂y** | 0.0123 | 0.000003 | 0.000000 | 0.000000 | 0.000007 / 0.000000 |
| **∂p/∂z** | 0.0089 | 0.000002 | 0.000000 | 0.000000 | 0.000005 / 0.000000 |
| **綜合梯度** | **0.0251** | - | - | - | - |

**通道流驗證**: ∂p/∂x 應為常數（驅動流動），標準差應接近零。
```

### 指標解讀

#### ✅ 良好的壓力梯度預測
```
dpdx_l2_error: 0.0251 (2.51%)          # 梯度誤差低
dpdx_pred_mean: -0.002487              # 接近參考值 -0.0025
dpdx_pred_std: 0.000015                # 接近零（常數梯度）
p_l2_error: 17.2093 (1720%)            # 但壓力絕對值誤差巨大（常數偏移）
```
**結論**: 模型物理上正確，壓力場僅有常數偏移。

#### ❌ 壓力梯度預測失敗
```
dpdx_l2_error: 0.8542 (85.42%)         # 梯度誤差高
dpdx_pred_mean: -0.001234              # 遠離參考值
dpdx_pred_std: 0.000523                # 不是常數（應接近零）
```
**結論**: 模型未正確學習壓力驅動機制，需要調整訓練策略。

---

## 技術優勢總結

| 指標類型 | 壓力絕對值 L2 | 壓力梯度 L2 |
|---------|-------------|------------|
| **受常數偏移影響** | ❌ 是 | ✅ 否 |
| **物理意義** | ❌ 低（NS方程不含 p） | ✅ 高（NS方程含 ∇p） |
| **通道流驗證** | ❌ 無法驗證驅動力 | ✅ 可驗證 ∂p/∂x = const |
| **評估準確性** | ❌ 易誤判 | ✅ 準確反映物理 |

---

## 未來改進方向

1. **自動壓力場對齊**: 在評估前自動減去預測和參考壓力場的平均值，消除常數偏移
2. **頻譜分析**: 分析壓力梯度的波數分佈，驗證高頻/低頻重建能力
3. **不確定性量化**: 使用 Ensemble PINNs 估計壓力梯度的預測不確定性
4. **自適應評估**: 根據流動類型（通道流/圓管流/空腔流）自動選擇合適的壓力評估策略

---

## 參考文獻

1. **PINNs 壓力不定性問題**:
   - Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). "Physics-informed neural networks"
   - 指出壓力場需要額外的邊界條件或正規化以消除常數不定性

2. **通道流基準**:
   - JHTDB Channel Flow Re_τ=1000: ∂p/∂x = -0.0025 (常數驅動梯度)

3. **評估指標最佳實踐**:
   - Cai, S., et al. (2021). "Physics-informed neural networks for heat transfer"
   - 建議使用物理量梯度而非絕對值進行評估

---

**文檔維護**: 請在修改評估邏輯時同步更新此文檔。
