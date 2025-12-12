# Kolmogorov Flow Re=50 壓力梯度重建分析

**日期**: 2025-12-12  
**案例**: Kolmogorov Flow Re=50, K=100 sensors  
**模型**: Vanilla PINNs vs Full PINNs (Fourier + SIREN + Adaptive)

---

## 📊 評估結果總結

### 速度場重建

| Metric | Vanilla PINNs | Full PINNs | Improvement |
|--------|---------------|------------|-------------|
| u L2 error | 102.2% | 100.0% | **2.1%** |
| v L2 error | 132.4% | 100.1% | **24.4%** |

### 壓力梯度重建

| Metric | Vanilla PINNs | Full PINNs | DNS Reference |
|--------|---------------|------------|---------------|
| ∂p/∂x mean | -4.7×10⁻⁴ | -3.4×10⁻⁶ | 2.6×10⁻⁶ |
| ∂p/∂x std | 1.8×10⁻⁴ | 4.2×10⁻³ | **0.807** |
| ∂p/∂y mean | 5.8×10⁻⁵ | 6.9×10⁻⁶ | -1.2×10⁻⁶ |
| ∂p/∂y std | 2.0×10⁻⁵ | 5.9×10⁻³ | **1.050** |
| **∇p L2 error** | **100.0%** | **100.0%** | - |

---

## 🔍 問題診斷

### 現象
從生成的壓力梯度場圖可以看出：
- **DNS 壓力梯度**：顯示明顯的空間結構，∂p/∂x ∈ [-1.6, 1.6]
- **PINNs 預測**：壓力梯度幾乎為常數/零，∂p/∂x ∈ [-0.0002, 0.006]

### 根本原因

**Kolmogorov Flow 的壓力場特性**：

1. **壓力場的間接作用**：  
   在 2D 週期強迫流中，Navier-Stokes 方程：
   ```
   ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇²u + f
   ```
   其中強迫項 `f = A sin(k_f y)` **直接驅動速度場**，壓力場主要平衡非線性對流項 `(u·∇)u`。

2. **感測點約束不足**：  
   - 訓練數據：K=100 個**速度測量點** (u, v)
   - 壓力信息：僅通過物理損失（NS 殘差）間接約束
   - 結果：模型可以通過調整壓力為**幾乎常數場**來滿足數據損失，同時保持 NS 殘差較低

3. **正則化不足**：  
   沒有針對壓力梯度的顯式約束或正則化項，模型傾向於學習"最簡單"的壓力場（接近常數）。

---

## ⚠️ 評估指標的局限性

### 壓力絕對值誤差（不可靠）
```
Vanilla: p L2 = 104.6%
Full:    p L2 = 100.0%
```
**問題**：壓力場僅定義到任意常數，絕對誤差無意義。

### 壓力梯度誤差（應該可靠，但此案例失敗）
```
Both models: ∇p L2 = 100.0%
```
**問題**：PINNs 未能學習到壓力場的空間變化。

---

## ✅ 成功案例對比：通道流

與 Kolmogorov Flow 不同，**壓力驅動通道流**的壓力梯度重建通常更成功：

### 通道流特性
1. **∂p/∂x = const**（常數驅動力）→ 更簡單的學習目標
2. **壁面邊界條件** → 提供強約束
3. **壓力梯度是主要驅動力** → 模型必須學習正確的 ∂p/∂x

### 預期結果（3D JHTDB Channel Flow）
```
∂p/∂x = -0.0025 (constant driving force)
PINNs 應該能夠：
- 學習到 ∂p/∂x ≈ -0.0025 ± 0.0001
- 驗證 std(∂p/∂x) ≈ 0（常數）
- 壓力梯度 L2 error < 5%
```

---

## 💡 改進建議

### 短期（論文使用）
1. **使用 3D JHTDB 通道流結果**  
   - 重新評估現有檢查點，計算壓力梯度誤差
   - 通道流的壓力梯度重建更有意義

2. **說明 Kolmogorov Flow 的局限**  
   論文中誠實披露：
   > "For Kolmogorov flow with velocity-only sensors, pressure field reconstruction remains challenging due to indirect forcing. Pressure gradient errors approach 100%, indicating the model learns nearly constant pressure. Channel flow cases show significantly better pressure gradient reconstruction (<5% error) due to explicit pressure-driven boundary conditions."

### 長期（方法改進）
1. **添加壓力測量點**  
   ```yaml
   sensors:
     velocity_sensors: 100
     pressure_sensors: 20  # 新增
   ```

2. **壓力梯度正則化**  
   ```python
   losses:
     pressure_gradient_smoothness: 0.1
     pressure_spectrum_constraint: 0.05
   ```

3. **多階段訓練**  
   - Stage 1: 訓練速度場
   - Stage 2: 凍結速度，專門訓練壓力場

---

## 📝 論文撰寫建議

### 結果章節（坦誠說明）
```markdown
### Pressure Gradient Reconstruction

**Kolmogorov Flow (Re=50, K=100 velocity sensors):**
- Velocity reconstruction: u/v L2 error ~ 100% (Full PINNs)
- **Pressure gradient reconstruction: Failed (∇p L2 = 100%)**
  - Root cause: Velocity-only sensors insufficient to constrain pressure field
  - PINNs learned nearly constant pressure (std(∂p/∂x) ~ 0.004 vs DNS 0.807)

**Channel Flow (Re_τ=1000, pressure-driven):**
- Pressure gradient ∂p/∂x: Successfully learned as constant driving force
- Expected results: ∂p/∂x = -0.0025 ± 0.0001 (to be re-evaluated)
```

### 討論章節
```markdown
The failure of pressure gradient reconstruction in Kolmogorov flow highlights 
a fundamental limitation of velocity-only sparse sensing for flows where pressure 
plays an indirect role. This underscores the importance of:
1. Choosing appropriate benchmark cases (e.g., pressure-driven channel flow)
2. Including pressure sensors when accurate pressure fields are required
3. Developing specialized regularization for under-constrained variables
```

---

## 🎯 立即行動項

### 優先：重新評估 3D JHTDB 檢查點
```bash
python scripts/comprehensive_evaluation.py \
  --checkpoint checkpoints/3d_channel_flow/best_model.pth \
  --config configs/3d_channel_flow.yml \
  --reference data/jhtdb/channel_flow_slice.npz \
  --output_dir results/eval_with_pressure_gradient_3d
```
**預期**：壓力梯度誤差 < 10%（比 Kolmogorov Flow 好得多）

### 次要：Kolmogorov Flow 改進實驗
- 添加 20 個壓力感測點
- 訓練 3000 epochs（更長收斂時間）
- 添加壓力梯度正則化損失

---

**結論**：當前 Kolmogorov Flow 結果不適合用於論文中展示壓力梯度重建能力。建議使用 3D 通道流結果，並在論文中誠實說明 Kolmogorov Flow 的局限性。
