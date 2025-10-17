# PirateNet 訓練失敗診斷與修復指南

## 📋 問題摘要

**症狀**：
- 訓練損失：`nan`
- 速度場預測誤差：~100%
- 壓力場預測誤差：~60%
- 模型輸出幾乎為常數（預測均值偏離真實值 100 倍）

**評估日期**：2025-10-16  
**配置檔案**：`configs/colab_piratenet_2d_slice.yml`  
**訓練輪數**：1000 epochs  
**設備環境**：Google Colab (CUDA)

---

## 🔍 根本原因分析

### 1. **訓練損失為 NaN**（數值崩潰）

#### 可能原因：
| 原因 | 機率 | 證據 |
|------|------|------|
| **學習率過高** | ⭐⭐⭐⭐⭐ | 配置為 `1e-3`，對 PINNs 過於激進 |
| **梯度爆炸** | ⭐⭐⭐⭐⭐ | 梯度裁剪值 `1.0` 可能不夠保守 |
| **資料未歸一化** | ⭐⭐⭐⭐ | 配置未明確啟用歸一化 |
| **Batch size 過大** | ⭐⭐⭐ | `2048` 可能導致梯度估計不穩定 |
| **Fourier σ 過大** | ⭐⭐⭐ | `σ=2.0` 可能引入過大的高頻擾動 |

#### 診斷方法：
```bash
# 檢查訓練日誌中的梯度範數
grep "gradient_norm" log/colab_piratenet_2d_slice/training.log | head -20

# 預期：若 gradient_norm > 100 則為梯度爆炸
```

---

### 2. **模型預測完全失效**（權重崩潰）

#### 症狀細節：
```
預測統計量 vs 真實統計量：
  u: 0.145 vs 9.92  (偏離 98.5%)
  v: 0.028 vs -0.0001 (偏離 28000%)
  w: 0.003 vs -0.002 (偏離 250%)
```

#### 可能原因：
| 原因 | 機率 | 證據 |
|------|------|------|
| **權重初始化問題** | ⭐⭐⭐⭐ | swish + RWF 組合可能不穩定 |
| **輸出層未正確反歸一化** | ⭐⭐⭐⭐ | 預測範圍 [0.003, 0.145] 明顯偏小 |
| **物理損失權重過低** | ⭐⭐⭐ | `pde_loss_weight: 1.0` vs `data_loss_weight: 50.0` |
| **壁面邊界條件未生效** | ⭐⭐ | `wall_loss` 可能為 0 或未正確計算 |

#### 診斷方法：
```python
# 使用診斷腳本檢查模型權重
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint <checkpoint_path> \
  --config configs/colab_piratenet_2d_slice.yml \
  --output-dir results/diagnosis

# 關鍵檢查點：
# 1. 權重均值應接近 0
# 2. 權重標準差應在 [0.01, 0.5] 範圍
# 3. 不應包含 NaN/Inf
```

---

### 3. **配置問題清單**

| 問題 | 原始值 | 建議值 | 優先級 |
|------|--------|--------|--------|
| 學習率過高 | `1e-3` | `1e-4` | 🔴 高 |
| 梯度裁剪不足 | `1.0` | `0.5` | 🔴 高 |
| Batch size 過大 | `2048` | `1024` | 🟡 中 |
| Fourier σ 過大 | `2.0` | `1.0` | 🟡 中 |
| 資料未歸一化 | 未明確啟用 | 明確啟用 z-score | 🔴 高 |
| 激活函數風險 | `swish` | `tanh` (更穩定) | 🟡 中 |
| Warmup epochs 不足 | `20` | `50` | 🟢 低 |
| 自適應權重過早啟用 | 已啟用 | 初期關閉 | 🟡 中 |

---

## 🛠️ 修復方案

### **方案 A：保守修復**（推薦）

使用新配置檔案 `configs/colab_piratenet_2d_slice_fixed_v2.yml`

#### 關鍵修改：
```yaml
# 1. 降低學習率
optimizer:
  lr: 1.0e-4  # 原 1e-3

# 2. 強化梯度裁剪
gradient_clip: 0.5  # 原 1.0

# 3. 降低 batch size
batch_size: 1024  # 原 2048

# 4. 明確啟用資料歸一化
normalization:
  enabled: true
  method: "z_score"
  per_variable: true

# 5. 使用更穩定的激活函數
activation: "tanh"  # 原 swish

# 6. 降低 Fourier sigma
fourier_sigma: 1.0  # 原 2.0

# 7. 初期關閉自適應權重
adaptive_weights:
  enabled: false  # 等穩定後再啟用

# 8. 增加 warmup
warmup_epochs: 50  # 原 20
```

#### 訓練指令：
```bash
# Colab 中執行
!python scripts/train.py --cfg configs/colab_piratenet_2d_slice_fixed_v2.yml

# 監控訓練（每 10 epochs 執行）
!tail -30 log/colab_piratenet_2d_fixed_v2/training.log
```

#### 成功指標：
- ✅ 前 100 epochs 無 NaN loss
- ✅ `gradient_norm < 10.0`
- ✅ `total_loss` 穩定下降
- ✅ `wall_loss > 0` 且穩定

---

### **方案 B：階段式修復**（若方案 A 仍失敗）

#### 階段 1：最小可訓練配置（200 epochs）
```yaml
# 極保守設定
optimizer:
  lr: 5.0e-5  # 進一步降低

batch_size: 512  # 進一步降低

gradient_clip: 0.3  # 進一步降低

# 關閉所有高級特性
use_rwf: false
use_fourier: false
adaptive_weights:
  enabled: false
causal_weights:
  enabled: false
```

#### 階段 2：逐步啟用特性（200-500 epochs）
若階段 1 穩定（loss 持續下降），逐步啟用：
1. Fourier features（epoch 200）
2. RWF（epoch 300）
3. 提升 batch size 到 1024（epoch 400）

#### 階段 3：完整訓練（500-1000 epochs）
啟用所有特性並微調至目標誤差。

---

## 📊 診斷工具使用

### 1. **訓練失敗診斷腳本**

```bash
# 完整診斷（生成報告 + 損失曲線圖）
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint checkpoints/colab_piratenet_2d_slice/epoch_1000.pth \
  --config configs/colab_piratenet_2d_slice.yml \
  --output-dir results/piratenet_diagnosis

# 檢查輸出：
# - results/piratenet_diagnosis/diagnosis_report.json
# - results/piratenet_diagnosis/loss_history.png
```

### 2. **快速損失檢查**

```bash
# 提取最後 20 epochs 的損失
tail -20 log/colab_piratenet_2d_slice/training.log | grep "Epoch"

# 檢查是否包含 NaN
grep -i "nan" log/colab_piratenet_2d_slice/training.log
```

### 3. **模型權重檢查**

```python
import torch
import numpy as np

# 載入檢查點
ckpt = torch.load('checkpoints/colab_piratenet_2d_slice/epoch_100.pth', 
                  map_location='cpu')

# 檢查權重統計
state_dict = ckpt['model_state_dict']
for name, param in state_dict.items():
    if 'weight' in name:
        w = param.cpu().numpy()
        print(f"{name:40s} | mean: {w.mean():8.4f} | std: {w.std():8.4f} | "
              f"min: {w.min():8.4f} | max: {w.max():8.4f}")

# 預期：均值 ~0，標準差 ~0.1-0.5，無 NaN
```

---

## ✅ 驗證檢查清單

### **訓練前**
- [ ] 配置檔案中 `y_min: -1.0, y_max: 1.0`（壁面位置正確）
- [ ] 學習率 ≤ `1e-4`
- [ ] 梯度裁剪 ≤ `1.0`
- [ ] 資料歸一化已啟用
- [ ] 資料檔案存在且格式正確

### **訓練中**（每 50 epochs 檢查）
- [ ] `total_loss` 無 NaN/Inf
- [ ] `gradient_norm < 10.0`
- [ ] `data_loss` 逐步降低
- [ ] `wall_loss > 0` 且穩定
- [ ] GPU 記憶體使用 < 10 GB

### **訓練後**
- [ ] 最終損失 < 初始損失的 10%
- [ ] 預測場範圍合理（u ∈ [0, 20], v/w ∈ [-5, 5]）
- [ ] 相對 L2 誤差 < 30%（2D 切片寬鬆標準）
- [ ] 統計量均值誤差 < 50%

---

## 🔄 故障排除流程圖

```
訓練開始
    ↓
前 10 epochs 檢查
    ↓
是否出現 NaN? ──yes──> 降低學習率（1e-4 → 5e-5）─┐
    ↓ no                                          │
gradient_norm > 100? ──yes──> 強化梯度裁剪（0.5 → 0.3）─┤
    ↓ no                                              │
loss 是否下降? ──no──> 檢查資料歸一化 ────────────────┤
    ↓ yes                                             │
繼續訓練至 epoch 100                                  │
    ↓                                                 │
檢查預測場範圍 ──異常──> 檢查反歸一化邏輯 ───────────┤
    ↓ 正常                                            │
繼續訓練至完成 <──────────────────────────────────┘
```

---

## 📚 參考資料

### 相關檔案：
- **診斷腳本**：`scripts/debug/diagnose_piratenet_failure.py`
- **修正配置**：`configs/colab_piratenet_2d_slice_fixed_v2.yml`
- **原始配置**：`configs/colab_piratenet_2d_slice.yml`
- **評估腳本**：`scripts/evaluate_piratenet_vs_jhtdb.py`

### 類似問題案例：
- Issue #XX: "NaN loss in VS-PINN training"
- Issue #YY: "Gradient explosion with Fourier features"

---

## 🎯 下一步行動

### 立即執行：
1. ✅ 使用 `colab_piratenet_2d_slice_fixed_v2.yml` 重新訓練
2. ✅ 前 100 epochs 密切監控損失
3. ✅ 若穩定，繼續訓練至 1000 epochs

### 若仍失敗：
1. 執行診斷腳本並檢查報告
2. 採用方案 B（階段式修復）
3. 考慮使用更簡單的模型架構（降低寬度/深度）

### 成功後：
1. 評估 2D 切片結果
2. 若達標（L2 < 20%），升級至 3D 完整訓練
3. 記錄成功配置並更新文檔

---

## 📞 取得協助

若問題持續存在，請提供：
1. 完整的訓練日誌（前 50 + 最後 50 行）
2. 診斷腳本輸出
3. 檢查點檔案的權重統計
4. 使用的配置檔案

**文檔版本**：v2.0  
**更新日期**：2025-10-16
