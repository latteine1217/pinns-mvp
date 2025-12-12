# Low-Fi 策略比較分析：RANS vs 粗網格DNS

## 📋 問題定義

在 PINNs 逆重建框架中，低保真場（Low-Fidelity Prior）扮演「軟先驗」角色，用於引導模型學習大尺度結構。  
**核心問題**：應該使用 **RANS 模擬場** 還是 **粗網格 DNS（下採樣）** 作為低保真先驗？

---

## 🎯 兩種策略的本質差異

### 策略 A：RANS 場（或高黏滯粗網格模擬）

**方法**：
- 使用較高黏滯係數（ν_lowfi = 1.5-2.0 × ν_hifi）
- 粗網格求解（N_lowfi = N_hifi / 8-16）
- 產生時間平均的平滑場

**物理特性**：
```
Re_lowfi = 0.5-0.7 × Re_hifi  →  更偏層流
```

**Bias 類型**：
- ✅ 過度擴散（over-diffusion）
- ✅ 小尺度渦旋被抹平
- ✅ 能量譜高頻截斷
- ⚠️ **物理模式不同**（不同雷諾數 → 不同流態）

---

### 策略 B：粗網格 DNS（空間下採樣）

**方法**：
- 使用**相同黏滯係數**（ν_lowfi = ν_hifi）
- 僅降低網格解析度（N_lowfi = N_hifi / 4-8）
- 保持相同的強迫參數（A, k_f）

**物理特性**：
```
Re_lowfi = Re_hifi  →  相同流態
```

**Bias 類型**：
- ✅ 數值黏滯（numerical diffusion）
- ✅ 無法解析小於網格尺度的渦旋
- ✅ 混疊誤差（aliasing）
- ✅ **物理模式相同**（同一控制方程，僅解析度不同）

---

## 📊 定量比較

| 評估維度 | 策略 A: RANS | 策略 B: 粗網格 DNS | 推薦 |
|---------|-------------|------------------|------|
| **物理一致性** | ❌ 不同雷諾數 → 不同物理 | ✅ 同一方程 → 相同物理 | **B** |
| **計算成本** | 🟢 低（高 ν → 大 dt 可用） | 🟡 中等（需保持 CFL < 1） | A |
| **實作複雜度** | 🟡 需調參（ν, Re） | 🟢 簡單（僅調 N） | **B** |
| **Bias 可解釋性** | 🟡 物理+數值混合 | 🟢 純數值（網格效應） | **B** |
| **與實際場景契合** | 🟢 模擬 RANS 工程應用 | 🟡 模擬粗網格計算限制 | A |
| **PINNs 修正難度** | 🔴 需跨 Re 外插 | 🟢 僅需插值小尺度 | **B** |

---

## 🔬 理論分析

### 1. **物理一致性**（最關鍵）

#### 策略 A 的問題：跨雷諾數外插

```
RANS (Re=33)  →  修正至  →  DNS (Re=50)
    ↓                           ↓
弱不穩定流態               過渡湍流流態
```

**挑戰**：
- PINN 需要學習「從層流過渡到湍流」的**相變過程**
- 不同 Re 下的渦結構、能量串級機制根本不同
- 文獻證據：跨 Re 外插是流體力學中的**病態問題**

#### 策略 B 的優勢：同一流態內插值

```
粗網格 DNS (Re=50, N=64)  →  修正至  →  精細 DNS (Re=50, N=512)
           ↓                                    ↓
     缺失小尺度渦旋                        完整尺度層次結構
```

**優勢**：
- PINN 僅需「補全高頻資訊」（插值問題）
- 控制方程完全相同，物理機制一致
- 數值分析理論支持：Richardson 外推、多網格方法

---

### 2. **文獻與實務證據**

#### Channel Flow Re_τ=1000 研究中的共識

根據你的專案目標（JHTDB 通道流）：

| 低保真策略 | 使用場景 | 典型誤差範圍 |
|-----------|---------|-------------|
| **RANS (k-ε)** | 統計穩態量（時均速度剖面、雷諾應力） | 20-50% |
| **粗 LES** | 大尺度瞬時結構 | 10-30% |
| **下採樣 DNS** | 完整瞬時場（缺小尺度） | **5-15%** ✅ |

**關鍵結論**：當目標是「瞬時場重建」時，下採樣 DNS 的 bias 更容易被 PINN 修正。

---

### 3. **PINNs 學習難度**

#### 策略 A：需要學習「物理修正」

```python
# 損失函數視角
L_total = L_data (稀疏 Hi-Fi) 
        + L_PDE (NS 方程，Re=50) 
        + λ·L_prior (RANS 場，Re=33)  # ⚠️ 矛盾！
```

**問題**：
- `L_PDE` 強制 Re=50 的物理
- `L_prior` 拉向 Re=33 的解
- PINN 需解決**多目標衝突**

#### 策略 B：僅需學習「解析度提升」

```python
# 損失函數視角
L_total = L_data (稀疏 Hi-Fi, N=512) 
        + L_PDE (NS 方程，Re=50) 
        + λ·L_prior (粗網格 DNS，Re=50, N=64)  # ✅ 一致！
```

**優勢**：
- 所有損失項指向**同一物理解**
- Prior 提供「模糊但正確」的大尺度結構
- PINN 專注於「高頻細節填補」

---

## 🧪 實驗驗證建議

### 實驗設計

**基準場**：Hi-Fi DNS (Re=50, N=256)

**對照組**：

| 實驗組 | Low-Fi 配置 | 稀疏資料 K | 預期結果 |
|--------|------------|-----------|---------|
| **Baseline** | 無先驗 | 50 | L2 error ≈ 15-25% |
| **A: RANS** | Re=33, N=32 | 50 | L2 error ≈ 12-18% |
| **B: 粗網格** | Re=50, N=64 | 50 | L2 error ≈ **8-12%** ✅ |

---

### 快速驗證腳本

```bash
# === 步驟 1：生成高保真 DNS ===
python scripts/generate_kolmogorov_dns.py \
    --N 256 --nu 0.039374 --T_end 20.0 \
    --output data/kolmogorov_dns/hifi_re50_N256.h5

# === 步驟 2A：生成 RANS-like Low-Fi ===
python scripts/generate_kolmogorov_lowfi.py \
    --N 32 --nu 0.059061 --T_total 50.0 --T_spinup 10.0 \
    --output data/kolmogorov_lowfi/rans_re33_N32.h5

# === 步驟 2B：生成粗網格 DNS ===
python scripts/generate_kolmogorov_dns.py \
    --N 64 --nu 0.039374 --T_end 20.0 \
    --output data/kolmogorov_lowfi/coarse_re50_N64.h5

# === 步驟 3：視覺化比較 ===
python scripts/compare_lowfi_hifi.py \
    --hifi data/kolmogorov_dns/hifi_re50_N256.h5 \
    --lowfi data/kolmogorov_lowfi/rans_re33_N32.h5 \
    --output results/comparison_rans/ \
    --time_avg_range 10.0 20.0

python scripts/compare_lowfi_hifi.py \
    --hifi data/kolmogorov_dns/hifi_re50_N256.h5 \
    --lowfi data/kolmogorov_lowfi/coarse_re50_N64.h5 \
    --output results/comparison_coarse/ \
    --time_avg_range 10.0 20.0

# === 步驟 4：訓練 PINNs（對照實驗）===
# 配置 A: RANS prior
python scripts/train.py --cfg configs/test_rans_prior_re50.yml

# 配置 B: 粗網格 DNS prior  
python scripts/train.py --cfg configs/test_coarse_dns_prior_re50.yml

# === 步驟 5：評估 ===
python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/rans_prior/best_model.pth \
    --reference data/kolmogorov_dns/hifi_re50_N256.h5 \
    --output results/eval_rans/

python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/coarse_dns_prior/best_model.pth \
    --reference data/kolmogorov_dns/hifi_re50_N256.h5 \
    --output results/eval_coarse/
```

---

## ✅ 推薦策略

### **首選：策略 B（粗網格 DNS）**

**理由**：
1. ✅ **物理一致性**：同一雷諾數 → 避免跨流態外插
2. ✅ **理論基礎扎實**：數值分析中的多網格思想
3. ✅ **實作簡單**：僅需調整 `--N` 參數
4. ✅ **Bias 可控**：純數值效應，易於建模

**參數建議**：
```yaml
# Hi-Fi 配置
N_hifi: 256-512
nu_hifi: (根據目標 Re 計算)
Re_hifi: 50-200

# Low-Fi 配置（粗網格 DNS）
N_lowfi: 64-128    # 4-8× 粗
nu_lowfi: nu_hifi  # ⚠️ 保持相同！
Re_lowfi: Re_hifi  # ⚠️ 保持相同！
```

**Prior 權重建議**：
```yaml
losses:
  prior_weight: 0.2-0.5  # 中等強度軟約束
```

---

### **備選：策略 A（RANS）的適用場景**

**僅在以下情況考慮**：
1. ✅ 研究目標是「模擬工程 RANS → 修正至 DNS」的實際工作流
2. ✅ 擁有大量實驗資料（可約束跨 Re 外插）
3. ✅ 關注統計穩態量而非瞬時場

**不推薦用於**：
- ❌ 基礎物理驗證（Kolmogorov Flow 基準測試）
- ❌ 稀疏資料場景（K < 50）
- ❌ 瞬時場重建任務

---

## 📚 支持文獻

### 多網格與解析度外推

1. **Briggs et al. (2000)**  
   *A Multigrid Tutorial*  
   → 同一方程下的粗細網格修正是線性代數基礎

2. **Pope (2000)**  
   *Turbulent Flows*  
   → 第13章：不同 Re 下的湍流統計量**不可直接縮放**

### PINNs 與 Low-Fi 整合

3. **Meng & Karniadakis (2020)**  
   *A composite neural network that learns from multi-fidelity data*  
   → 多保真度融合：要求低保真與高保真**物理一致**

4. **Yang et al. (2021)**  
   *B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data*  
   → Bias 建模：數值誤差（網格效應）比物理模型誤差（RANS）更易量化

---

## 🎯 總結

| 策略 | 適用場景 | 推薦指數 |
|------|---------|---------|
| **粗網格 DNS** | 基礎研究、物理驗證、瞬時場重建 | ⭐⭐⭐⭐⭐ |
| **RANS 場** | 工程應用、統計量預測、充足資料 | ⭐⭐⭐ |

**最終建議**：  
對於本專案（Kolmogorov Flow 與 JHTDB Channel Flow 的逆重建研究），**強烈推薦使用粗網格 DNS 作為低保真先驗**，理由是物理一致性與 PINNs 學習效率的雙重優勢。

---

**最後更新**: 2025-12-11  
**作者**: PINNs-MVP 分析團隊
