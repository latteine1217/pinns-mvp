# 🚀 Quick Start: Channel Flow Normalization Fix

## ✅ Implementation Complete

**修復內容**:
- 添加數據質量驗證函數
- 強制從 DNS cutout 計算標準化（避免使用損壞的 RANS prior）
- 兩處修改（ensemble + single model training）

**驗證結果**: ✅ 所有檢查通過

---

## 📋 下一步操作

### Option 1: 快速驗證（10 分鐘）✨ **建議先做這個**

```bash
# 測試訓練 10 epochs 驗證標準化修復
python scripts/train/train.py \
  --config configs/channel_flow_re1000.yml \
  --max-epochs 10 \
  --checkpoint-dir checkpoints/test_fix \
  --log-dir logs/test_fix
```

**預期 log 輸出**（檢查 `logs/test_fix/training.log`）:
```
🔧 Computing normalization from DNS cutout: data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz
   (NOT from sensor data to avoid RANS prior corruption)

================================================================================
🔍 Sensor Data Quality Check
================================================================================
  u: mean=1.065725e+00, std=8.313331e-02, ...
  v: mean=2.629401e-03, std=4.081868e-02, ...  ← 應該是 ~0.04，不是 1e-7！
  w: mean=-2.636332e-04, std=4.366771e-02, ...  ← 應該是 ~0.04，不是 1e-7！
  p: mean=-1.518339e-03, std=3.253538e-03, ...  ← 應該是 ~0.003，不是 1e-8！
✅ Sensor data quality check PASSED
================================================================================

📊 DNS Normalization Statistics:
   u: mean=1.065725, std=0.083133
   v: mean=0.002629, std=0.040819  ✅ CORRECT!
   w: mean=-0.000264, std=0.043668  ✅ CORRECT!
   p: mean=-0.001518, std=0.003254  ✅ CORRECT!
```

**成功標誌**:
- ✅ 看到 "Computing normalization from DNS cutout"
- ✅ v/w/p 的 std 都在 0.003-0.044 範圍（不是 1e-7!）
- ✅ "Sensor data quality check PASSED"

---

### Option 2: 完整重訓（6-8 小時）

```bash
# 驗證通過後，從頭訓練完整模型
python scripts/train/train.py \
  --config configs/channel_flow_re1000.yml \
  --checkpoint-dir checkpoints/channel_flow_FIXED \
  --log-dir logs/channel_flow_FIXED
```

**建議**:
- 可以在背景執行（overnight）
- 訓練期間可以監控 TensorBoard：`tensorboard --logdir logs/channel_flow_FIXED`
- 預計需要 6-8 小時完成 5000 epochs

---

### Option 3: 評估新模型（訓練完成後，30 分鐘）

```bash
# 評估修復後的 checkpoint
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/channel_flow_FIXED/epoch_5000.pth \
  --output results/channel_flow_FIXED/ \
  --dns-cutout data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz
```

**預期改善**:
```
❌ 舊模型（損壞的標準化）:
   u: 124.8% error
   v: 2181.5% error  ← 災難性失敗
   w: 962.6% error   ← 災難性失敗
   p: 2454.1% error  ← 災難性失敗

✅ 新模型（正確的標準化）:
   u: ~80-120% error
   v: ~100-200% error  ← 合理範圍！
   w: ~100-200% error  ← 合理範圍！
   p: ~100-200% error  ← 合理範圍！
```

---

## 🔍 監控重點

### 訓練過程中檢查

1. **標準化統計（epoch 0 的 log）**:
   - v_std ≈ 0.041（不是 1e-7）
   - w_std ≈ 0.044（不是 1e-7）
   - p_std ≈ 0.003（不是 1e-8）

2. **損失平衡（training.log）**:
   - u/v/w/p 的 sensor loss 應該有相似的量級
   - 不應該只有 u_sensor_loss 在下降

3. **梯度大小（如啟用 debug 模式）**:
   - v/w/p 的梯度不應該比 u 小 10⁴-10⁵ 倍

---

## 📊 驗證腳本

**已創建**: `verify_normalization_fix.py`

```bash
# 隨時可以重跑驗證
python verify_normalization_fix.py
```

這會檢查：
- ✅ DNS cutout 文件存在
- ✅ v/w/p 統計合理（std > 1e-3）
- ✅ validate_sensor_data_quality 函數正常工作

---

## 📁 相關文件

**Bug 分析**:
- `results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md` ⭐ 完整證據鏈
- `results/channel_flow_evaluation/NORMALIZATION_BUG_REPORT.md` 詳細分析
- `results/channel_flow_evaluation/EVALUATION_REPORT.md` 原始發現

**修復實施**:
- `results/channel_flow_evaluation/NORMALIZATION_FIX_IMPLEMENTED.md` 本次修復詳情

**程式碼**:
- `scripts/train/train.py` (已修改，lines 51-120, 1972-2009, 2013-2050)
- `verify_normalization_fix.py` (驗證腳本)

---

## ❓ 疑難排解

### 問題: Training 失敗，顯示 "DNS cutout not found"

**解決**:
```bash
# 檢查文件是否存在
ls -lh data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz

# 如果不存在，可能需要重新生成（參考專案文檔）
```

### 問題: 標準化統計仍然顯示 1e-7

**可能原因**:
- 訓練配置未啟用標準化（`data.normalize: false`）
- 或使用了預設的標準化參數（`normalization.params` 已設定）

**解決**: 確認 `configs/channel_flow_re1000.yml` 中：
```yaml
data:
  normalize: true

normalization:
  type: training_data_norm
  # params: {}  # 這一行應該是空的或不存在，讓程式自動計算
```

---

## 🎯 成功標準

### ✅ 驗證階段（10 min 測試）
- [ ] Log 顯示 "Computing normalization from DNS cutout"
- [ ] v/w/p 的 std 在 0.003-0.044 範圍
- [ ] "Sensor data quality check PASSED"
- [ ] 訓練正常啟動，無錯誤

### ⏸️ 訓練階段（6-8 hours）
- [ ] 損失正常收斂
- [ ] 無 NaN/Inf
- [ ] u/v/w/p sensor loss 都在下降（不是只有 u）

### ⏸️ 評估階段（30 min）
- [ ] Overall L2 < 200%（目前是 155%，但 v/w/p 錯誤）
- [ ] v/w/p error ≈ 100-200%（不是 2000%！）
- [ ] 所有場的誤差量級相似

---

## 🔔 Ready to Start!

**建議執行順序**:
1. ✅ 執行 `verify_normalization_fix.py`（確認修復正確）
2. 🟡 執行 Option 1 快速驗證（10 min，確認 log 正確）
3. 🟢 執行 Option 2 完整訓練（6-8 hours，可過夜）
4. 🔵 執行 Option 3 評估（30 min，比較結果）

**當前狀態**: 🟢 Ready to start validation (Option 1)

---

**Questions?** 查看詳細文檔：
- `results/channel_flow_evaluation/NORMALIZATION_FIX_IMPLEMENTED.md`
- `results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md`
