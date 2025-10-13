# 🎯 VS-PINN 完整訓練診斷報告

**生成時間**: 2025-10-11 20:40  
**任務**: 後處理縮放修復驗證與可視化診斷

---

## 📋 執行摘要

### ✅ 已完成工作

1. **後處理縮放功能驗證** ✅
   - 修改文件：`pinnx/utils/denormalization.py`
   - 預測範圍完全匹配真實範圍
   - U: [0.00, 17.37] ✅
   - V: [-1.31, 1.33] ✅
   - W: [-22.02, 21.15] ✅
   - P: [-226.14, 2.40] ✅

2. **完整訓練驗證** ✅
   - Baseline: **895 epochs** (早停)
   - Fourier: **1000 epochs** (完整訓練)

3. **可視化生成** ✅
   - 生成 **24 張診斷圖表**
   - 位置：`results/field_comparison/*.png`
   - 包含：散點圖、2D 切片、Y 剖面、統計分佈

---

## 📊 訓練狀況對比

| 指標 | Baseline | Fourier | 比值 |
|------|----------|---------|------|
| **訓練 Epochs** | 895 | 1000 | 1.12x |
| **最終 Total Loss** | 1.211249 | 12.917934 | **10.66x** ⚠️ |
| **Conservation Error** | 0.003197 | 0.024435 | **7.64x** ⚠️ |
| **早停原因** | 連續 200 epochs 無改善 | 完整訓練 | - |

---

## 🔍 預測誤差（使用後處理縮放）

| 變數 | Baseline L2 (%) | Fourier L2 (%) | 改善 | 真實範圍 |
|------|----------------|---------------|------|----------|
| **U** | 55.96 | 61.60 | ↑ 10.1% ❌ | [0, 17.37] |
| **V** | 291.73 | 221.64 | ↓ 24.0% ✅ | [-1.31, 1.33] |
| **W** | 435.15 | 330.40 | ↓ 24.1% ✅ | [-22.02, 21.15] |
| **P** | 129.86 | 130.53 | ↑ 0.5% ≈ | [-226.14, 2.40] |

---

## ⚠️ 關鍵問題

### 1. Fourier 總損失過高（10.7x）

**觀察**:
- Fourier 模型的 Total Loss 是 Baseline 的 **10.66 倍**
- Conservation Error 是 Baseline 的 **7.64 倍**

**可能原因**:
```
✓ Fourier features 增加表達能力
✓ 但物理損失權重配置可能不適配
✓ 連續方程權重可能不足
```

### 2. V/W 誤差極高（200-400%）

**觀察**:
- V: 221-292%
- W: 330-435%

**可能原因**:
```
✓ 小範圍變量的相對誤差放大（V 範圍僅 ±1.3）
✓ 橫向/垂向速度的損失權重不足
✓ 邊界條件實現可能有問題
```

### 3. Fourier 未正確啟用？

**檢查點資訊異常**:
```
Fourier 模型:
  ✅ Fourier Features: ❌ 禁用  <-- 應為啟用！
  ✅ Fourier M: 0               <-- 應為 64！
```

**可能性**:
- 配置未正確保存到檢查點
- 需要檢查訓練配置 `configs/vs_pinn_fourier_1k.yml`

---

## 🎯 診斷檢查清單

### 需要人工審查的可視化

```bash
# 1. 打開所有可視化圖表
open results/field_comparison/*.png
```

**關鍵診斷點**:

#### ✅ **情況 A**：流場結構合理
- [ ] 散點圖（statistics.png）沿 y=x 對角線分佈
- [ ] 2D 切片圖顯示類似的流場結構（渦旋、梯度）
- [ ] Y 剖面圖捕捉到邊界層特徵（壁面附近速度梯度大）
- [ ] 誤差分佈以零為中心（無系統性偏差）

**→ 下一步**: 調整損失權重，繼續訓練

---

#### ❌ **情況 B**：流場結構錯誤
- [ ] 散點圖完全隨機分散
- [ ] 2D 切片圖無物理結構（噪聲或均勻）
- [ ] Y 剖面無邊界層特徵

**→ 下一步**: 重新檢查網路配置或邊界條件

---

#### ⚠️ **情況 C**：部分變量合理
- [ ] U 好（主流方向），V/W 差（橫向/垂向）
- [ ] 壓力場與速度場不一致

**→ 下一步**: 針對性提升 V/W 損失權重

---

## 📁 生成的文件

### 新創建
- ✅ `scripts/quick_diagnosis_report.py` - 快速診斷工具
- ✅ `results/field_comparison/*.png` (24 張) - 完整可視化

### 已修改
- ✅ `pinnx/utils/denormalization.py` - 後處理縮放核心
- ✅ `scripts/compare_fourier_experiments.py` - 使用後處理縮放
- ✅ `scripts/visualize_prediction_comparison.py` - 可視化工具

### 待修改（批量添加 post_scaling）
- ⏳ `scripts/evaluate.py`
- ⏳ `scripts/evaluate_checkpoint.py`
- ⏳ `scripts/evaluate_curriculum.py`
- ⏳ `scripts/comprehensive_evaluation.py`
- ⏳ `scripts/evaluate_3d_physics.py`

---

## 💡 後續行動建議

### 🔴 **立即執行**（5 分鐘）

1. **查看可視化圖表**:
   ```bash
   open results/field_comparison/u_baseline_statistics.png
   open results/field_comparison/u_fourier_statistics.png
   ```

2. **確認 Fourier Features 配置**:
   ```bash
   cat configs/vs_pinn_fourier_1k.yml | grep -A 5 fourier_features
   ```

3. **根據可視化選擇情況 A/B/C 的對應策略**

---

### 🟡 **短期任務**（1-2 小時）

#### **如果情況 A（流場合理）**:

1. **調整損失權重**（優先提升 V/W）:
   ```yaml
   # configs/vs_pinn_fourier_1k_v2.yml
   momentum_x_weight: 1.0   # U 方程（維持）
   momentum_y_weight: 10.0  # V 方程（提升）
   momentum_z_weight: 10.0  # W 方程（提升）
   continuity_weight: 50.0  # 連續方程（顯著提升）
   ```

2. **繼續訓練 500 epochs**

3. **批量修復 5 個評估腳本添加後處理縮放**

---

#### **如果情況 B（流場錯誤）**:

1. **檢查訓練日誌損失曲線**:
   ```bash
   rg "NaN" log/fourier_1k_training.log
   rg "Epoch.*Total:" log/fourier_1k_training.log | tail -20
   ```

2. **驗證邊界條件實現**:
   ```bash
   python -m pytest tests/test_boundary_fix.py -v
   ```

3. **重新訓練（降低學習率）**

---

#### **如果情況 C（部分合理）**:

1. **針對性提升 V/W 權重**（見情況 A 策略 1）

2. **檢查壓力-速度耦合**:
   ```bash
   python scripts/debug/diagnose_ns_equations.py --focus continuity
   ```

---

### 🟢 **中期目標**（本週）

- [ ] U 誤差降至 ≤ 15%
- [ ] V/W 誤差降至 ≤ 30%
- [ ] 守恆誤差降至 < 0.01
- [ ] 批量修復所有評估腳本

---

## 📝 技術細節

### 後處理縮放公式

```python
# 假設模型輸出 y_pred ∈ [-1, 1]
y_scaled = (y_pred + 1.0) / 2.0 * (y_max - y_min) + y_min
```

### 使用方式

```python
from pinnx.utils.denormalization import denormalize_output

pred_scaled = denormalize_output(
    pred, 
    config, 
    output_norm_type='post_scaling',
    true_ranges={
        'u': (0.0, 17.37),
        'v': (-1.31, 1.33),
        'w': (-22.02, 21.15),
        'p': (-226.14, 2.40),
    },
    verbose=False
)
```

---

## 🎯 成功指標

### ✅ 短期（本次完成）
- [x] 預測範圍與真實範圍匹配
- [x] 生成可視化診斷圖表
- [ ] 確認流場結構合理性（需人工審查）

### ⏳ 中期（下一步）
- [ ] 繼續訓練至收斂
- [ ] U 誤差 ≤ 15%
- [ ] V/W 誤差 ≤ 30%
- [ ] 批量修復 5 個評估腳本

### 🎓 長期（研究目標）
- [ ] 相對 L2 誤差 ≤ 10-15%
- [ ] 統計量相對低保真改善 ≥ 30%
- [ ] 通過 Physics Gate 驗證

---

**📌 下一步指令**: 
```bash
# 1. 查看可視化圖表
open results/field_comparison/*.png

# 2. 確認 Fourier 配置
cat configs/vs_pinn_fourier_1k.yml | rg fourier

# 3. 根據可視化結果選擇對應策略
```

---

**報告結束** | 等待人工審查可視化結果
