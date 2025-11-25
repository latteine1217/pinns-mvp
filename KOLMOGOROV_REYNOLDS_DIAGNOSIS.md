# Kolmogorov Flow 雷諾數診斷報告

**日期**: 2025-11-25
**狀態**: 🔴 **發現嚴重問題：雷諾數定義不一致**

---

## 執行摘要

經過深入檢查，發現 **PINNs 訓練配置與 DNS 數據之間存在嚴重的物理參數不一致**：

1. **配置文件宣稱 Re=100**，但實際根據標準定義計算僅 **Re≈12.5–19.5**
2. **DNS 數據中的 ν=0.0125**，但配置文件誤用 **ν=0.01**
3. **不同雷諾數定義之間的結果相差 6-8 倍**

**影響**：
- ❌ PINNs 訓練使用的物理參數與 DNS 數據不匹配
- ❌ 模型學習到錯誤的物理尺度和黏度
- ❌ 論文中報告的雷諾數將與實際不符

---

## 問題詳情

### 1. DNS 數據的實際參數

**檔案**: `data/kolmogorov_dns_re100_512x512_kf8_midway.h5`

**元數據內容**（從 `config` group 讀取）：
```python
{
  'A': 1.0,                # 強迫振幅
  'nu': 0.0125,            # ⚠️ 實際黏度是 0.0125，而非配置文件的 0.01
  'k_f': 8,                # 強迫波數
  'L': 6.283185307179586,  # 域長度 (2π)
  'N': 512,                # 網格解析度
  'T_end': 20.0,           # 終止時間
  'dt': 0.0005             # 時間步長
}
```

**場統計**：
- 平均動能 (KE): 1.052370
- RMS 速度 (U_rms): 1.450772
- 平均 enstrophy: 137.605229
- 散度誤差: ~0 (滿足不可壓縮條件)

---

### 2. 雷諾數計算比對

#### 使用 DNS 實際參數 (A=1.0, ν=0.0125, k_f=8)

| 定義 | 公式 | 計算結果 | 與 Re_config=100 比值 | 狀態 |
|------|------|----------|----------------------|------|
| **配置文件宣稱** | - | **100.0** | 1.00x | 參考 |
| **標準定義** | F/(ν²k³) | **12.50** | 8.00x | ❌ 差異巨大 |
| **層流解定義** | UL/ν (L=2π/k) | **78.54** | 1.27x | ⚠️ 接近但不一致 |
| **DNS 有效 Re** | U_rms·L/ν (L=1/k) | **14.51** | 6.89x | ❌ 差異巨大 |

#### 使用配置文件參數 (A=1.0, ν=0.01, k_f=8) - **錯誤的參數**

| 定義 | 計算結果 | 與 Re_config=100 比值 | 狀態 |
|------|----------|----------------------|------|
| **標準定義** | 19.53 | 5.12x | ❌ |
| **層流解定義** | 122.72 | 0.81x | ⚠️ |

---

### 3. 根本原因分析

#### 問題 1：配置文件中 ν 值錯誤
**位置**: `configs/kolmogorov_re100_kf8_K50_*.yml`

**錯誤配置**：
```yaml
physics:
  nu: 0.01  # ❌ 錯誤！DNS 數據實際使用 0.0125
```

**正確配置應該是**：
```yaml
physics:
  nu: 0.0125  # ✅ 與 DNS 數據匹配
```

#### 問題 2：Re 定義不明確
配置文件標註 `Re: 100`，但未明確說明使用哪種定義：
- 標準定義：Re = F/(ν²k³)
- 層流解定義：Re = UL/ν
- 基於動能的有效 Re

#### 問題 3：Re=100 的來源可疑
根據任何定義計算，使用 DNS 數據的實際參數都無法得到 Re=100：
- 若 Re 來自層流解定義，應該是 **Re≈78.5**
- 若 Re 來自標準定義，應該是 **Re≈12.5**

**可能原因**：
1. DNS 數據的檔名誤導（`re100` 可能指的是不同定義）
2. 配置文件從其他實驗複製而未更新
3. Re=100 是目標值而非實際值

---

## 修正方案

### 🎯 推薦方案：修正配置文件以匹配 DNS 數據

**理由**：DNS 數據已經生成，無法更改。必須讓 PINNs 訓練配置與 DNS 一致。

#### Step 1：修正物理參數
**檔案**: `configs/kolmogorov_re100_kf8_K50_*.yml`

```yaml
physics:
  nu: 0.0125  # ✅ 修正為 DNS 實際值
  rho: 1.0

  forcing:
    k_f: 8
    amplitude: 1.0  # 保持不變
```

#### Step 2：明確標註實際雷諾數
```yaml
data:
  kolmogorov_config:
    physics_params:
      Re_standard: 12.50     # 標準定義：F/(ν²k³)
      Re_laminar: 78.54      # 層流解：UL/ν (L=2π/k)
      Re_effective: 14.51    # DNS 有效 Re (基於 U_rms)
      Re_definition: "laminar"  # 主要使用層流解定義
      nu: 0.0125
      k_f: 8
      forcing_amplitude: 1.0
    description: "Kolmogorov flow (Re_laminar≈78.5, k_f=8, 2D periodic)"
```

#### Step 3：更新實驗名稱
```yaml
experiment:
  name: "kolmogorov_re78_kf8_K50_initial"  # ✅ 反映實際 Re (層流解)
  # 或
  name: "kolmogorov_re12_kf8_K50_initial"  # ✅ 反映實際 Re (標準定義)
```

#### Step 4：更新文檔說明
在 `usage_notes` 中明確說明：
```yaml
usage_notes: |
  ⚠️ 雷諾數說明：
    - DNS 數據實際參數：ν=0.0125, A=1.0, k_f=8
    - 標準定義 Re = F/(ν²k³) = 12.5
    - 層流解定義 Re = UL/ν = 78.5 (主要使用)
    - 本研究採用層流解定義，Re≈78.5
    - 檔名 "re100" 是生成 DNS 時的目標值，實際略低
```

---

### 🔧 替代方案（不推薦）：重新生成 DNS 數據

**僅在必須使用 Re=100 的情況下考慮**

#### 方案 A：調整 ν 以匹配 Re=100 (層流解定義)
```python
# 層流解定義：Re = UL/ν = (A/(ν k²)) × (2π/k) / ν = 2πA / (ν² k³)
# 設 Re = 100, A = 1.0, k_f = 8
# 100 = 2π / (ν² × 512)
# ν² = 2π / 51200 = 0.000123
# ν = 0.0111
```

**新 DNS 參數**：
```yaml
nu: 0.0111
A: 1.0
k_f: 8
```

#### 方案 B：調整 A 以匹配 Re=100 (保持 ν=0.0125)
```python
# 使用層流解定義
# Re = 2πA / (ν² k³) = 100
# A = 100 × ν² × k³ / (2π) = 100 × 0.0125² × 512 / (2π) = 1.273
```

**新 DNS 參數**：
```yaml
nu: 0.0125
A: 1.273
k_f: 8
```

---

## 實施清單

### ✅ 立即行動（高優先級）

- [x] **驗證 DNS 數據的實際參數**（已完成）
- [ ] **修正所有配置文件的 `nu` 值**
  - `configs/kolmogorov_re100_kf8_K50_initial.yml`
  - `configs/kolmogorov_re100_kf8_K50_t20_2k.yml`
  - 其他相關配置

- [ ] **更新實驗名稱**（反映實際 Re）

- [ ] **更新文檔**
  - `README.md`: 修正 Re 描述
  - `TECHNICAL_DOCUMENTATION.md`: 添加 Re 定義說明
  - `CLAUDE.md`: 更新 Kolmogorov flow 設定章節

- [ ] **停止使用舊配置的訓練**

### 🔄 後續驗證

- [ ] **重新生成感測點**（如果 ν 改變會影響物理場）
  ```bash
  python scripts/generate_sensors_kolmogorov.py --nu 0.0125
  ```

- [ ] **重新訓練模型**
  ```bash
  python scripts/train.py --cfg configs/kolmogorov_re78_kf8_K50_initial.yml
  ```

- [ ] **驗證物理一致性**
  ```bash
  python scripts/verify_kolmogorov_reynolds.py --nu 0.0125
  python scripts/evaluate_kolmogorov_full.py
  ```

### 📝 文檔更新

- [ ] **創建 `docs/KOLMOGOROV_REYNOLDS_DEFINITION.md`**
  - 明確說明使用的 Re 定義
  - 提供不同定義之間的轉換公式
  - 記錄 DNS 數據的實際參數

- [ ] **更新論文草稿**
  - 修正所有提到 Re=100 的地方
  - 明確標註 Re 定義
  - 在 Methods 章節添加 Re 計算說明

---

## 影響評估

### 🔴 高影響

1. **已訓練的模型無效**
   - 使用錯誤的 ν=0.01 訓練的所有模型
   - 需要用正確的 ν=0.0125 重新訓練

2. **論文數據需修正**
   - Re 值報告錯誤
   - 物理分析基於錯誤的參數

3. **比較實驗失效**
   - 與其他 Re 的比較需重新評估
   - Re=500 實驗也需檢查類似問題

### 🟡 中等影響

1. **感測點可能需重新生成**
   - 如果使用物理場生成（如 DEIM/QR-pivot）
   - ν 改變會略微影響特徵值分佈

2. **文檔需要全面更新**
   - 配置模板
   - 技術文檔
   - README

### 🟢 低影響

1. **程式碼邏輯無需修改**
   - 物理方程實現正確
   - 僅需更新配置參數

2. **訓練腳本無需修改**
   - 自動讀取配置文件

---

## 驗證檢查表

修正後，執行以下驗證確保問題已解決：

### 1. 參數一致性驗證
```bash
python scripts/verify_kolmogorov_reynolds.py --nu 0.0125
```
**預期結果**：所有 Re 計算應該一致或差異 < 10%

### 2. 配置文件驗證
```bash
python -c "
import yaml
with open('configs/kolmogorov_re78_kf8_K50_initial.yml', 'r') as f:
    cfg = yaml.safe_load(f)
    assert cfg['physics']['nu'] == 0.0125, 'nu 應為 0.0125'
    print('✅ 配置文件驗證通過')
"
```

### 3. 物理模組驗證
```bash
python -c "
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 1.0, 'wavenumber': 8},
    physics_params={'nu': 0.0125, 'rho': 1.0}
)
Re = physics.compute_reynolds_number()
print(f'計算的 Re = {Re:.2f}')
assert 12 <= Re <= 13, f'標準定義 Re 應在 12-13 之間，實際 {Re:.2f}'
print('✅ 物理模組驗證通過')
"
```

### 4. DNS 數據載入驗證
```bash
python scripts/check_dns_re100.py  # 需要編寫
```

---

## 總結

這次診斷發現了一個**嚴重的物理參數不一致問題**，對專案有重大影響：

**核心問題**：
- PINNs 配置使用 ν=0.01
- DNS 數據實際使用 ν=0.0125
- 導致雷諾數差異 5-8 倍

**解決方案**：
- ✅ 修正所有配置文件的 ν 值為 0.0125
- ✅ 明確標註實際雷諾數（Re_laminar≈78.5 或 Re_standard≈12.5）
- ✅ 更新所有文檔和實驗名稱
- ✅ 重新訓練所有模型

**修正後的狀態**：
- PINNs 訓練將使用與 DNS 數據一致的物理參數
- 模型將學習正確的物理尺度
- 論文報告將基於正確的雷諾數

**優先級**：🔴 **最高** - 必須在任何後續訓練之前完成修正
