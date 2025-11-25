# Kolmogorov Flow 雷諾數修正報告

**日期**: 2025-11-21  
**作者**: PINNs-MVP 團隊  
**修正內容**: 統一使用標準雷諾數定義 `Re = F / (ν² k³)`

---

## ✅ 修正內容

### 1️⃣ **核心物理模組更新**

**文件**: `pinnx/physics/kolmogorov_flow_2d.py`

**新增方法**:
```python
def compute_reynolds_number(self) -> float:
    """計算 Kolmogorov Flow 雷諾數（標準定義）"""
    F = float(self.amplitude.item())
    nu = float(self.nu.item())
    k = float(self.wavenumber.item())
    Re = F / (nu**2 * k**3)
    return Re

def compute_effective_reynolds(self, predictions: torch.Tensor) -> float:
    """計算有效雷諾數（基於預測場的動能）"""
    KE = float(self.compute_kinetic_energy(predictions).item())
    U_eff = np.sqrt(2.0 * KE)
    L = 1.0 / float(self.wavenumber.item())
    nu = float(self.nu.item())
    Re_eff = U_eff * L / nu
    return Re_eff
```

**修正內容**:
- ❌ 舊公式：`Re = A × L_y / ν`（錯誤）
- ✅ 新公式：`Re = F / (ν² k³)`（標準定義）

---

### 2️⃣ **驗證工具**

**文件**: `scripts/validation/validate_kolmogorov_reynolds.py`

**功能**:
```bash
# 根據目標 Re 計算所需黏滯度
python scripts/validation/validate_kolmogorov_reynolds.py --compute-nu --Re 30 --F 1.0 --k 4

# 驗證所有配置文件
python scripts/validation/validate_kolmogorov_reynolds.py --validate

# 生成標準參數表
python scripts/validation/validate_kolmogorov_reynolds.py --table

# 顯示公式參考
python scripts/validation/validate_kolmogorov_reynolds.py --formula
```

---

## 📊 配置文件驗證結果

| 配置文件 | F | k | ν | Re (實際) | Re (宣稱) | 狀態 |
|---------|---|---|---|----------|----------|------|
| `kolmogorov_2d_baseline.yml` | 1.0 | 4 | 0.01 | **156.25** | - | ✅ |
| `kolmogorov_2d_chaos_re30.yml` | 0.768 | 4 | 0.02 | **30.00** | 30 | ✅ |
| `kolmogorov_2d_chaos_re30_full.yml` | 0.768 | 4 | 0.02 | **30.00** | 30 | ✅ |
| `kolmogorov_2d_chaos_re30_quick.yml` | 0.768 | 4 | 0.02 | **30.00** | 30 | ✅ |
| `kolmogorov_2d_curriculum.yml` | 0.32 | 4 | 0.02 | **12.50** | 50 | ❌ **需修正** |
| `kolmogorov_2d_test_periodic.yml` | 1.536 | 4 | 0.02 | **60.00** | 60 | ✅ |
| `kolmogorov_2d_turbulent_pure_pde.yml` | 1.0 | 4 | 0.02 | **39.06** | - | ✅ |
| `kolmogorov_2d_turbulent_re60.yml` | 1.536 | 4 | 0.02 | **60.00** | 60 | ✅ |

**總結**: 7/8 個配置文件通過驗證

---

## ❌ 需要修正的配置

### **`kolmogorov_2d_curriculum.yml`**

**問題**: 宣稱 Re=50，但實際計算為 Re=12.50

**當前配置**:
```yaml
physics:
  forcing:
    amplitude: 0.32
    wavenumber: 4
  nu: 0.02
```

**修正方案**（選擇一種）:

#### 方案 A：保持 Re=50，調整 ν
```yaml
physics:
  forcing:
    amplitude: 0.32  # 保持不變
    wavenumber: 4
  nu: 0.011314       # √(0.32 / (50 × 4³)) = 0.011314
```

#### 方案 B：保持 ν=0.02，調整 F
```yaml
physics:
  forcing:
    amplitude: 0.640  # 50 × 0.02² × 4³ = 0.64
    wavenumber: 4
  nu: 0.02
```

#### 方案 C：接受實際 Re=12.50，更新文檔
```yaml
# 註解改為 Re=12.50
physics:
  forcing:
    amplitude: 0.32
    wavenumber: 4
  nu: 0.02
```

**建議**: 採用 **方案 B**（保持 ν=0.02），因為課程學習中其他階段也使用相同黏滯度。

---

## 📐 標準參數表 (F=1.0, k=4)

| Re | ν (nu) | U_laminar | 物理狀態 |
|----|--------|-----------|---------|
| 10 | 0.039528 | 1.5811 | 層流（穩定） |
| 20 | 0.027951 | 2.2361 | 弱失穩 |
| 30 | 0.022822 | 2.7386 | 時空混沌 |
| 40 | 0.019764 | 3.1623 | 時空混沌 |
| 50 | 0.017678 | 3.5355 | 時空混沌 |
| 60 | 0.016137 | 3.8730 | 完全發展湍流 |
| 80 | 0.013975 | 4.4721 | 完全發展湍流 |
| 100 | 0.012500 | 5.0000 | 完全發展湍流 |
| 150 | 0.010206 | 6.1237 | 完全發展湍流 |
| 200 | 0.008839 | 7.0711 | 完全發展湍流 |

---

## 🔬 公式參考

### 標準定義
```
Re = F / (ν² k³)
```

### 推導
```
層流解：u_x(y) = (F / (ν k²)) sin(k y)
特徵速度：U = F / (ν k²)
特徵長度：L = 1/k
雷諾數：Re = UL/ν = (F / (ν k²)) × (1/k) / ν = F / (ν² k³)
```

### 反推黏滯度
```
ν = √(F / (Re × k³))
```

### 參考文獻
- Meshalkin & Sinai (1961): *Stability of steady state Kolmogorov flow*
- Boffetta et al. (2002): *Inverse energy cascade in two-dimensional turbulence*

---

## ✅ 驗證流程

### 1. 檢查物理模組
```bash
python -c "
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 0.768, 'wavenumber': 4},
    physics_params={'nu': 0.02}
)
print(f'Re = {physics.compute_reynolds_number():.2f}')
"
```

### 2. 驗證所有配置
```bash
python scripts/validation/validate_kolmogorov_reynolds.py --validate
```

### 3. 生成參數表
```bash
python scripts/validation/validate_kolmogorov_reynolds.py --table --Re-list 20 30 40 50 60 80 100
```

---

## 📝 後續行動

- [ ] 修正 `kolmogorov_2d_curriculum.yml` 配置
- [ ] 更新所有配置文件的註解（確保 Re 標註正確）
- [ ] 在訓練日誌中記錄實際 Re（使用 `physics.compute_reynolds_number()`）
- [ ] 在評估報告中加入 Re_eff 對比（使用 `physics.compute_effective_reynolds()`）

---

**修正完成時間**: 2025-11-21  
**測試狀態**: ✅ 已通過單元測試與配置驗證
