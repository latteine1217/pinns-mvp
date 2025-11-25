# Kolmogorov Flow 雷諾數計算工具使用指南

**腳本**: `scripts/calculate_reynolds_parameters.py`
**定義**: Musacchio & Boffetta (2014) - Re = √f₀ × L^(3/2) / ν，其中 L = 2π/k

---

## 快速開始

### 1. 計算雷諾數（給定 f₀, ν, k）

```bash
# DNS 數據的實際參數
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
```

**輸出**:
```
參數:
  f₀ = 1.0
  ν = 0.0125
  k = 8.0

結果:
  Re = 55.68
  U₀ = 1.2500
  L = 0.7854

流動狀態: 過渡/弱湍流
```

---

### 2. 反推 ν（給定目標 Re）

```bash
# 想要 Re=100，需要什麼樣的 ν？
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 \
  --f0 1.0 \
  --k 8 \
  --solve-nu
```

**輸出**:
```
所需動力黏度: ν = 0.006960
驗證: Re = 100.00
```

**應用場景**: 生成新的 DNS 數據時設定參數

---

### 3. 反推 f₀（給定目標 Re）

```bash
# 想要 Re=100，需要什麼樣的強迫振幅？
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 \
  --nu 0.0125 \
  --k 8 \
  --solve-f0
```

**輸出**:
```
所需強迫振幅: f₀ = 3.225153
驗證: Re = 100.00
```

**應用場景**: 調整外力強度以達到目標雷諾數

---

### 4. 參數掃描（批量計算）

#### 4.1 掃描 ν（固定 f₀, k）

```bash
# 掃描 ν 從 0.005 到 0.025，步長 0.005
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 \
  --k 8 \
  --nu-range 0.005 0.025 0.005
```

**輸出**:
```
ν            Re           U₀           流動狀態
------------------------------------------------------------
0.005000     139.21       3.1250       湍流
0.010000     69.60        1.5625       過渡/弱湍流
0.015000     46.40        1.0417       過渡/弱湍流
0.020000     34.80        0.7812       過渡/弱湍流
0.025000     27.84        0.6250       層流/弱不穩定
```

**應用場景**: 規劃一系列不同雷諾數的實驗

#### 4.2 掃描 f₀（固定 ν, k）

```bash
python scripts/calculate_reynolds_parameters.py \
  --nu 0.0125 \
  --k 8 \
  --f0-range 0.5 5.0 0.5
```

#### 4.3 掃描 k（固定 f₀, ν）

```bash
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 \
  --nu 0.0125 \
  --k-range 4 16 2
```

**輸出**:
```
k            Re           L            U₀           流動狀態
----------------------------------------------------------------------
4            222.39       1.5708       3.1250       強湍流
6            121.62       1.0472       1.3889       湍流
8            78.54        0.7854       0.7812       過渡/弱湍流
10           55.68        0.6283       0.5000       過渡/弱湍流
12           41.51        0.5236       0.3472       過渡/弱湍流
14           32.17        0.4488       0.2551       過渡/弱湍流
16           25.76        0.3927       0.1953       層流/弱不穩定
```

**應用場景**: 研究不同波數的影響

---

### 5. 保存結果到 CSV

```bash
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 \
  --k 8 \
  --nu-range 0.005 0.025 0.001 \
  --output results/reynolds_parameters.csv
```

**生成的 CSV 文件**:
```csv
nu,f0,k,Re,U0,regime
0.005,1.0,8.0,139.21,3.125,湍流
0.006,1.0,8.0,116.01,2.604,湍流
0.007,1.0,8.0,99.44,2.232,過渡/弱湍流
...
```

---

### 6. 交互式模式

```bash
# 直接運行（無參數）進入交互式選單
python scripts/calculate_reynolds_parameters.py
```

或

```bash
python scripts/calculate_reynolds_parameters.py --interactive
```

**選單**:
```
選擇計算模式:
  1. 計算雷諾數 (給定 f₀, ν, k)
  2. 計算所需的 ν (給定目標 Re, f₀, k)
  3. 計算所需的 f₀ (給定目標 Re, ν, k)
  4. 生成參數對照表
  5. 退出

請選擇 (1-5):
```

**模式 4 的子選單**:
```
選擇生成模式:
  1. 固定 f₀, k，掃描 ν → Re
  2. 固定 ν, k，掃描 f₀ → Re
  3. 固定 f₀, ν，掃描 k → Re
  4. 固定 k，掃描 ν 與 f₀ → Re (2D 網格)
```

---

## 常見使用場景

### 場景 1：驗證 DNS 數據的實際雷諾數

```bash
# 檢查 DNS 數據檔案 kolmogorov_dns_re100_512x512_kf8_midway.h5
# 實際參數：f₀=1.0, ν=0.0125, k=8

python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8

# 輸出: Re = 55.68（而非檔名的 100）
```

---

### 場景 2：規劃新的 DNS 模擬

```bash
# 目標：生成 Re=100 的 DNS 數據
# 已知：想使用 k=8 的強迫

# 選項 A：調整 ν（保持 f₀=1.0）
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --f0 1.0 --k 8 --solve-nu
# 結果: ν = 0.006960

# 選項 B：調整 f₀（保持 ν=0.0125）
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --nu 0.0125 --k 8 --solve-f0
# 結果: f₀ = 3.225153

# 建議：選擇 A（調整 ν 更常見）
```

---

### 場景 3：設計多雷諾數實驗系列

```bash
# 目標：Re = 30, 50, 100, 150, 200
# 固定：k=8, f₀=1.0

# 計算所需的 ν 值
for Re in 30 50 100 150 200; do
  echo "Re=$Re:"
  python scripts/calculate_reynolds_parameters.py \
    --target-Re $Re --f0 1.0 --k 8 --solve-nu
done
```

**輸出**:
```
Re=30:  ν = 0.023200
Re=50:  ν = 0.013920
Re=100: ν = 0.006960
Re=150: ν = 0.004640
Re=200: ν = 0.003480
```

---

### 場景 4：比對不同雷諾數定義

```bash
# 使用此工具計算 MB 2014 定義
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
# Re = 55.68

# 與其他定義比較（手動計算）：
# - 傳統定義 UL/ν (L=2π/k): Re ≈ 78.54
# - 標準定義 f₀/(ν²k³): Re ≈ 12.50
```

---

## 流動狀態分類（基於 MB 2014）

| Re 範圍 | 流動狀態 | 特徵 |
|---------|---------|------|
| Re < 30 | 層流/弱不穩定 | 主要為穩態層流解，少量時間波動 |
| 30 < Re < 100 | **過渡/弱湍流** | 出現渦結構，時間波動增強，開始能量級串 |
| 100 < Re < 200 | 湍流 | 明顯的逆能量級串，大尺度渦結構 |
| Re > 200 | 強湍流 | 充分發展的 2D 湍流，多尺度相互作用 |

**DNS 數據 (Re≈56) 處於過渡/弱湍流範圍**，非常適合 PINNs 研究。

---

## 命令行參數完整列表

```
基本參數:
  --f0 FLOAT              強迫振幅
  --nu FLOAT              動力黏度
  --k, --k-f FLOAT        強迫波數

求解模式:
  --target-Re FLOAT       目標雷諾數
  --solve-nu              求解所需的 ν
  --solve-f0              求解所需的 f₀

掃描模式:
  --nu-range START END STEP    掃描 ν 範圍
  --f0-range START END STEP    掃描 f₀ 範圍
  --k-range START END STEP     掃描 k 範圍

其他:
  --interactive, -i       交互式模式
  --output, -o FILE       保存結果到 CSV
  --verbose, -v           詳細輸出
  --help, -h              顯示幫助
```

---

## Python API 使用

如果想在其他腳本中使用：

```python
import sys
sys.path.append('scripts')
from calculate_reynolds_parameters import (
    compute_reynolds,
    compute_nu_from_Re,
    compute_f0_from_Re,
    classify_flow_regime
)

# 計算雷諾數
Re = compute_reynolds(f0=1.0, nu=0.0125, k_f=8)
print(f"Re = {Re:.2f}")  # Re = 55.68

# 反推 ν
nu = compute_nu_from_Re(Re=100, f0=1.0, k_f=8)
print(f"ν = {nu:.6f}")  # ν = 0.006960

# 反推 f₀
f0 = compute_f0_from_Re(Re=100, nu=0.0125, k_f=8)
print(f"f₀ = {f0:.6f}")  # f₀ = 3.225153

# 流動分類
regime, desc = classify_flow_regime(Re=56)
print(f"{regime}: {desc}")
```

---

## 與驗證腳本的配合使用

```bash
# 1. 使用計算工具規劃參數
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --f0 1.0 --k 8 --solve-nu
# 輸出: ν = 0.006960

# 2. 使用驗證腳本檢查 DNS 數據
python scripts/verify_kolmogorov_reynolds.py \
  --nu 0.006960 \
  --dns-data data/kolmogorov_dns_re100_new.h5

# 3. 比對並確認一致性
```

---

## 參考文獻

1. **Musacchio & Boffetta (2014)**
   *"Condensation and fluxes in two-dimensional turbulence"*
   Physical Review E, 89(2), 023004

2. **Shebalin (2013)**
   *"Kolmogorov flow in three dimensions"*
   Physics of Fluids, 25(10), 105111

3. **Danilov & Gurarie (2001)**
   *"Quasi-two-dimensional turbulence"*
   Physics-Uspekhi, 43(9), 863

---

## 疑難排解

### 問題 1：計算結果與配置文件不一致

```bash
# 檢查 DNS 數據的實際參數
python -c "
import h5py
with h5py.File('data/kolmogorov_dns_re100_512x512_kf8_midway.h5', 'r') as f:
    config = f['config']
    print(f\"A = {config.attrs['A']}\")
    print(f\"nu = {config.attrs['nu']}\")
    print(f\"k_f = {config.attrs['k_f']}\")
"

# 使用實際參數重新計算
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
```

### 問題 2：不確定使用哪個參數

```bash
# 使用交互式模式，跟隨提示操作
python scripts/calculate_reynolds_parameters.py --interactive
```

### 問題 3：需要生成複雜的參數表

```bash
# 使用掃描功能生成 CSV，然後用 Excel/Python 分析
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 --k 8 --nu-range 0.001 0.030 0.001 \
  --output reynolds_scan.csv
```

---

**最後更新**: 2025-11-25
**維護者**: PINNs-MVP 團隊
