# Kolmogorov Flow 2D 配置指南

## 📋 總覽

本指南說明如何設定 Kolmogorov Flow 的體力強迫以進入時空混沌/湍流態。

## 🔬 理論基礎

### 強迫公式
對於標準 2D Kolmogorov 流（$f_x = F_0 \sin(k_f y)$），層流解為：
```
u_x(y) = U sin(k_f y)
U = F0 / (ν × k_f²)
Re = U / (ν × k_f)
```

反推強迫幅值：
```
F0 = Re × ν² × k_f³
```

### 失穩與混沌門檻

| 雷諾數範圍 | 流態 | 現象 |
|-----------|------|------|
| Re < 1.414 (√2) | 層流穩定 | 無擾動發展 |
| 1.414 < Re < 20 | 線性失穩後 | 長波擾動發展 |
| 20 ≤ Re < 40 | 時空混沌 | 局域混沌斑、寬頻能譜 |
| Re ≥ 60 | 完全湍流 | 逆串級、強間歇性、統計穩定 |

**注意**: k_f=4 時，較低 Re 即可觀察到間歇爆發與混沌。

---

## 📦 可用配置

### 1️⃣ Re=30 時空混沌配置
**文件**: `configs/kolmogorov_2d_chaos_re30.yml`

#### 物理參數
```yaml
forcing:
  amplitude: 0.768        # F0 = 30 × 0.02² × 4³
  wavenumber: 4

nu: 0.02
domain:
  x_max: 12.566370614359172  # 4π (Lx/Ly = 2)
  y_max: 6.283185307179586   # 2π
```

#### 預期現象
- ✅ 局域混沌斑沿流向漂移
- ✅ 寬頻能譜（非單峰）
- ✅ 渦度場峰度 > 3（非高斯）
- ✅ 時間序列無週期性

#### 計算資源
- 配置點: 10,000
- 訓練輪數: 2,000 epochs
- 預估時間: ~40-60 分鐘
- 網路: 6 層 × 512 神經元

#### 適用場景
- 探索時空混沌動力學
- 研究混沌斑傳播機制
- 驗證混沌診斷方法

---

### 2️⃣ Re=60 完全湍流配置
**文件**: `configs/kolmogorov_2d_turbulent_re60.yml`

#### 物理參數
```yaml
forcing:
  amplitude: 1.536        # F0 = 60 × 0.02² × 4³
  wavenumber: 4

nu: 0.02
domain:
  x_max: 12.566370614359172  # 4π (Lx/Ly = 2)
  y_max: 6.283185307179586   # 2π
```

#### 預期現象
- ✅ 完全發展湍流
- ✅ 明顯逆能量串級（2D 湍流特徵）
- ✅ 寬頻能譜，k^(-5/3) 或 k^(-3) 慣性區
- ✅ 強渦度間歇性（峰度 >> 3）
- ✅ 複雜渦街結構

#### 計算資源
- 配置點: 15,000
- 訓練輪數: 5,000 epochs
- 預估時間: ~2-3 小時
- 網路: 8 層 × 512 神經元

#### 適用場景
- 統計湍流分析（PDF、結構函數）
- 逆串級動力學研究
- 能量譜與通量計算
- 論文級湍流模擬

---

## 🎯 關鍵設計決策

### 1. 損失函數權重（已優化）
```yaml
loss:
  weights:
    momentum_x: 1.0      # 與 continuity 等權重
    momentum_y: 1.0
    continuity: 1.0      # 降低（原 10.0）
    periodic_x: 10.0     # 提高（原 5.0）
    periodic_y: 10.0     # 提高（原 5.0）
```

**理由**:
- 降低 continuity 權重避免過度約束（PDE 自然滿足不可壓）
- 提高 boundary 權重確保週期性嚴格滿足（湍流對邊界敏感）

### 2. 域長比 Lx/Ly = 2
**理由**:
- 降低後續分岔門檻
- 促進局域混沌斑的出現與漂移
- 更接近實際湍流研究的設定

### 3. 高波數強迫 k_f=4
**理由**:
- 較低 Re 就能觀察到間歇爆發
- 更豐富的多尺度結構
- 減少所需計算資源

### 4. 初始擾動
```yaml
initial_perturbation:
  enabled: true
  amplitude: 0.02-0.03  # 2-3% 白噪
```

**理由**:
- 加速失穩過程
- 避免陷入層流解
- 促進湍流快速發展

---

## 🚀 使用方式

### 快速開始（Re=30）
```bash
python scripts/train_pure_pde.py \
  --cfg configs/kolmogorov_2d_chaos_re30.yml
```

### 背景運行（Re=60）
```bash
nohup python scripts/train_pure_pde.py \
  --cfg configs/kolmogorov_2d_turbulent_re60.yml \
  > log/kolmogorov_re60.log 2>&1 &
```

### 監控訓練
```bash
# 查看日誌
tail -f log/kolmogorov_re60.log

# 即時監控損失
watch -n 10 'tail -20 log/kolmogorov_re60.log | grep "Epoch"'
```

---

## 📊 湍流診斷指標

### 能譜分析
```python
# 檢查寬頻能譜（非單峰）
E(k) = FFT(u^2 + v^2)
# 預期: Re=30 → 多峰; Re=60 → k^(-5/3) 慣性區
```

### 渦度統計
```python
# 峰度（診斷非高斯）
kurtosis(ω) > 3  # 高斯分佈為 3
# 預期: Re=30 → 3-5; Re=60 → 5-10（強間歇性）

# 偏度（診斷對稱性破缺）
skewness(ω) ≠ 0
```

### 時間序列
```python
# 自相關（診斷非週期）
autocorr(u(x0, t)) → 快速衰減
# 預期: τ_corr ~ 1-2 (無量綱時間)
```

### 能量通量（Re=60）
```python
# 逆串級（2D 湍流特徵）
Π(k) = ∫ T(k,p,q) dk  # 轉移函數
# 預期: Π < 0 for k < k_forcing（能量向大尺度傳遞）
```

---

## ⚠️ 常見問題

### Q1: 訓練後仍呈噪聲狀，無清晰結構
**可能原因**:
1. 訓練時間不足（湍流需要 2000+ epochs）
2. 初始擾動過強或過弱
3. 損失權重失衡

**解決方案**:
```bash
# 延長訓練
n_epochs: 5000

# 調整初始擾動
initial_perturbation:
  amplitude: 0.02  # 從 0.01-0.05 掃描

# 檢查損失平衡
# momentum_x ≈ momentum_y ≈ continuity
```

### Q2: 動能過低或過高
**診斷**:
```python
# 檢查層流解
U_theory = F0 / (nu * k_f**2)
E_theory = U_theory**2 / 2

# 實際動能應在 0.5-2 × E_theory 範圍
```

**調整**:
```yaml
# 若動能過低 → 增加強迫
forcing:
  amplitude: 1.2  # 微調 ±20%

# 若動能過高 → 檢查是否數值不穩定
```

### Q3: 週期性邊界條件未滿足
**診斷**:
```python
# 檢查邊界誤差
BC_error = |u(x=0) - u(x=Lx)| + |u(y=0) - u(y=Ly)|
# 應 < 1e-3
```

**解決**:
```yaml
# 提高邊界權重
loss:
  weights:
    periodic_x: 20.0  # 從 10.0 提高
    periodic_y: 20.0
```

---

## 🔧 進階調參

### 掃描雷諾數
```python
Re_list = [20, 30, 40, 60, 80]
for Re in Re_list:
    F0 = Re * nu**2 * k_f**3
    # 修改 config 並訓練
```

### 掃描強迫波數
```yaml
# k_f=1: 大尺度注入（需更高 Re 湍流）
# k_f=2: 中等尺度
# k_f=4: 小尺度（推薦）
```

### 多種域長比
```yaml
# 方形域 Lx/Ly = 1
# 長域 Lx/Ly = 2 (推薦)
# 超長域 Lx/Ly = 4 (時空混沌更顯著)
```

---

## 📚 參考文獻

1. **線性失穩**: Meshalkin & Sinai (1961), "Investigation of the stability of a stationary solution of a system of equations for the plane movement of an incompressible viscous liquid"

2. **時空混沌**: Lucas & Kerswell (2015), "Spatiotemporal dynamics in two-dimensional Kolmogorov flow over large domains", J. Fluid Mech.

3. **2D 湍流統計**: Boffetta & Ecke (2012), "Two-dimensional turbulence", Annu. Rev. Fluid Mech.

4. **高波數強迫**: She (1987), "Large-scale dynamics and transition to turbulence in the two-dimensional Kolmogorov flow", Phys. Rev. E

---

## 📞 支援

若遇到問題，請檢查：
1. `log/<exp_name>/training.log` - 訓練日誌
2. `results/<exp_name>/` - 視覺化結果
3. `docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md` - 故障診斷

或運行診斷工具：
```bash
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint checkpoints/<exp_name>/latest.pth \
  --config configs/<config_name>.yml
```
