# DNS 結果驗證指南

## 📋 為什麼要驗證 DNS 結果？

DNS (Direct Numerical Simulation) 是數值求解 Navier-Stokes 方程的「金標準」，但仍需驗證：

1. **數值誤差累積**：長時間積分可能導致誤差累積
2. **離散化誤差**：有限差分/譜方法的截斷誤差
3. **邊界條件**：週期性邊界實現正確性
4. **物理守恆律**：能量、動量守恆
5. **參數正確性**：雷諾數、強迫項等設定

---

## ✅ 驗證清單（6 項核心檢查）

### 1️⃣ **不可壓縮條件** ∇·u = 0

**物理意義**：
- 流體密度恆定（不可壓縮假設）
- 速度場無源無匯

**驗證方法**：
```python
div = ∂u/∂x + ∂v/∂y
max_div_error = max(|div|)
```

**通過標準**：
- `max_div_error < 1.0`（寬鬆）
- 理想值：O(10⁻⁶) - O(10⁻²)

**常見問題**：
- ❌ 散度 > 1.0：投影步驟失效，檢查 Poisson 求解器
- ❌ 散度持續增長：時間步長過大，減小 dt

---

### 2️⃣ **Navier-Stokes 方程殘差**

**物理意義**：
- 驗證動量方程是否被正確求解

**驗證方法**：
```python
# 2D NS 方程：
∂u/∂t + u·∇u = -∇p + ν∇²u + f

residual = |∂u/∂t + u·∇u + ∇p - ν∇²u - f|
```

**通過標準**：
- `max(residual) < 100`（寬鬆，取決於場量級）
- 理想值：O(10⁻³) - O(10)

**常見問題**：
- ❌ 殘差過大：時間導數計算不準確，或強迫項錯誤
- ❌ 壓力項異常：壓力 Poisson 方程求解問題

---

### 3️⃣ **能量平衡**

**物理意義**：
- 動能變化 = 能量注入 - 黏滯耗散

**驗證方法**：
```python
KE = 0.5 * ∫(u² + v²) dV  # 動能
P_in = ∫(u·f) dV          # 強迫功率
ε = ν * ∫(∇u : ∇u) dV    # 耗散率

dKE/dt ≈ P_in - ε
```

**通過標準**：
- `|dKE/dt - (P_in - ε)| < 10`（平均值）
- 理想值：誤差 < 5%

**常見問題**：
- ❌ 能量持續增長：強迫項過大或耗散不足
- ❌ 能量快速衰減：黏滯係數過大

---

### 4️⃣ **Kolmogorov 尺度檢查**

**物理意義**：
- 網格必須解析最小湍流尺度（Kolmogorov 尺度 η）

**驗證方法**：
```python
η = (ν³ / ε)^(1/4)        # Kolmogorov 長度尺度
τ_η = (ν / ε)^(1/2)       # Kolmogorov 時間尺度

dx/η < 10   # 空間解析度
dt/τ_η < 1  # 時間解析度
```

**通過標準**：
- 空間：`dx/η < 10`（粗略 DNS）、`< 2`（高精度 DNS）
- 時間：`dt/τ_η < 1`

**常見問題**：
- ❌ dx/η > 10：網格過粗，增加 N
- ❌ dt/τ_η > 1：時間步長過大，減小 dt

---

### 5️⃣ **雷諾數一致性**

**物理意義**：
- 配置參數與實際流場的雷諾數應一致

**驗證方法**：
```python
# Kolmogorov flow 定義：
Re_theory = √A * L^(3/2) / ν

# 從流場估計：
U_rms = √⟨u² + v²⟩
Re_actual = U_rms * L / ν
```

**通過標準**：
- `|Re_theory - Re_actual| / Re_theory < 50%`

**常見問題**：
- ❌ Re_actual << Re_theory：流動未充分發展，延長模擬時間
- ❌ Re_actual >> Re_theory：初始條件過強

---

### 6️⃣ **統計穩態性**（可選）

**物理意義**：
- 檢查流動是否達到統計穩態（對於強迫湍流）

**驗證方法**：
```python
# 後半段時間的動能統計
KE_mean = mean(KE[t > T/2])
KE_std = std(KE[t > T/2])
CV = KE_std / KE_mean  # 變異係數
```

**通過標準**：
- `CV < 0.2`（穩態）
- 趨勢斜率接近 0

**註**：Kolmogorov flow 在某些 Re 下會衰減，非穩態也正常

---

## 🛠️ 使用驗證工具

### 基本用法

```bash
# 完整驗證（包含所有 6 項檢查）
python scripts/validate_dns_physics.py \
  --input data/kolmogorov_Re1000_kf4_t100s.h5 \
  --output results/dns_validation/ \
  --verbose
```

### 輸出文件

1. **`validation_report.png`**：9 宮格可視化報告
   - 散度誤差時間序列
   - 能量演化
   - 能量平衡
   - 雷諾數比較
   - Kolmogorov 尺度檢查
   - NS 殘差
   - 驗證總結
   - 統計穩態性
   - 參數摘要

2. **`validation_results.json`**：詳細數值結果
   - 所有驗證項目的定量指標
   - Pass/Fail 狀態
   - 時間序列數據

---

## 📊 解讀驗證結果

### ✅ 理想情況（全部通過）

```
驗證總結
==============================
✅ incompressibility
✅ navier_stokes
✅ energy_balance
✅ kolmogorov_scales
✅ reynolds_number
✅ statistical_stationarity

通過: 6/6
成功率: 100%
```

**結論**：DNS 結果物理合理，可用於下游分析（PINNs 訓練、統計分析）

---

### ⚠️ 部分失敗情況

#### 情況 1：散度誤差過大

```
❌ incompressibility
   最大散度誤差: 5.23e+00  # > 1.0
```

**可能原因**：
1. 壓力投影不完全（檢查 Poisson 求解器）
2. 時間步長過大（減小 dt）
3. 強迫項過強（減小 A）

**補救措施**：
```bash
# 重新生成 DNS，減小時間步長
python scripts/generate_kolmogorov_dns.py \
  --dt 0.0001 \  # 原 0.0002
  ...
```

---

#### 情況 2：NS 殘差過大

```
❌ navier_stokes
   U-momentum 殘差: max=2.45e+02  # > 100
```

**可能原因**：
1. 時間導數計算不準確（單側差分誤差）
2. 壓力梯度計算錯誤
3. 強迫項設定錯誤

**補救措施**：
- 檢查強迫波數 k_f 是否正確
- 驗證強迫項公式：`f_x = A * sin(k_f * 2π/L * y)`

---

#### 情況 3：能量不守恆

```
❌ energy_balance
   平衡誤差: mean=15.23  # > 10
```

**可能原因**：
1. 耗散計算錯誤
2. 強迫功率計算錯誤
3. 數值耗散過大

**補救措施**：
- 檢查黏滯係數 ν 設定
- 啟用去混疊（`--dealias`）
- 增加網格解析度

---

#### 情況 4：網格解析度不足

```
⚠️  kolmogorov_scales
   空間解析度: dx/η = 15.2  # > 10
```

**可能原因**：
- 網格過粗，無法解析 Kolmogorov 尺度

**補救措施**：
```bash
# 重新生成 DNS，增加網格點數
python scripts/generate_kolmogorov_dns.py \
  --N 2048 \  # 原 1024
  ...
```

**權衡**：
- N = 1024：粗略 DNS，適合快速測試
- N = 2048：高精度 DNS，計算成本 4×
- N = 4096：非常高精度，計算成本 16×

---

## 🎯 針對不同用途的驗證標準

### 用途 1：PINNs 訓練數據

**最低要求**：
- ✅ 不可壓縮條件（必須）
- ✅ 能量平衡（推薦）
- ⚠️ NS 殘差（可放寬）
- ⚠️ Kolmogorov 尺度（可放寬至 dx/η < 20）

**理由**：PINNs 對數據噪聲有一定容忍度

---

### 用途 2：論文級別結果

**嚴格要求**：
- ✅ 所有 6 項必須通過
- ✅ 散度誤差 < 0.1
- ✅ dx/η < 5
- ✅ 能量平衡誤差 < 5%

**理由**：需要高精度 DNS 作為驗證基準

---

### 用途 3：湍流統計分析

**推薦要求**：
- ✅ 統計穩態性（必須）
- ✅ 能量平衡（必須）
- ✅ Kolmogorov 尺度（dx/η < 5）
- ⚠️ NS 殘差（可放寬）

**理由**：統計量對瞬時誤差不敏感，但需要足夠解析度

---

## 📚 理論背景

### Kolmogorov 理論

**適用範圍**：
- 高雷諾數湍流（Re > 1000）
- 慣性子範圍（η << r << L）

**關鍵尺度**：
```
η = (ν³/ε)^(1/4)      # 最小渦流尺度
λ = √(15νu²/ε)        # Taylor 微尺度
L ~ U³/ε              # 積分尺度
```

**能譜理論**：
```
E(k) ∝ ε^(2/3) k^(-5/3)  # Kolmogorov -5/3 law
```

---

### Kolmogorov Flow 特性

**定義**：
- 2D 週期強迫流動
- 強迫項：`f = A sin(k_f y) ê_x`

**雷諾數定義**：
```
Re = √A * (2π/k_f)^(3/2) / ν
```

**層流解**：
```
U(y) = A/(ν k_f²) sin(k_f y)
V(y) = 0
```

**穩定性**：
- Re < 30：層流穩定
- 30 < Re < 100：線性不穩定，出現二次流
- Re > 100：湍流

---

## 🔧 常見問題排查

### Q1: 驗證腳本運行很慢？

**A**: 正常，計算導數需要時間。加速方法：
- 使用更少快照驗證：`--snapshots 0 -1`
- 降低空間解析度（但會影響精度）

### Q2: 所有項都失敗？

**A**: 檢查基本設定：
```bash
# 查看 HDF5 文件結構
python -c "import h5py; f=h5py.File('data.h5','r'); print(list(f.keys()))"
```

確保包含：`u`, `v`, `p`, `time`, `config`, `diagnostics`

### Q3: 雷諾數不匹配？

**A**: Kolmogorov flow 的 Re 定義特殊：
- `Re_config = 1/ν`（配置值）
- `Re_theory = √A * L^(3/2) / ν`（理論值）
- `Re_actual = U_rms * L / ν`（實際值）

三者可能不完全一致，50% 誤差內可接受。

---

## 📖 參考文獻

1. **Kolmogorov 理論**：
   - Kolmogorov, A. N. (1941). "The local structure of turbulence in incompressible viscous fluid for very large Reynolds numbers." *Doklady Akademii Nauk SSSR*, 30, 301-305.

2. **Kolmogorov Flow**：
   - Musacchio, S., & Boffetta, G. (2014). "Split energy cascade in turbulent thin fluid layers." *Physics of Fluids*, 26(3), 035105.

3. **DNS 方法**：
   - Pope, S. B. (2000). *Turbulent Flows*. Cambridge University Press.
   - Canuto, C., et al. (2012). *Spectral Methods: Fundamentals in Single Domains*. Springer.

---

**最後更新**：2025-12-02
**作者**：PINNs-MVP 團隊
