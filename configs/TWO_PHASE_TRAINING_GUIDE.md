# 兩階段訓練方案：SOAP → L-BFGS

## 📌 概述

根據上一次訓練結果分析（訓練1000 epochs後仍有物理約束不滿足問題），採用**兩階段訓練策略**：

- **Phase 1 (SOAP)**: 0-10000 epochs，穩定收斂階段
- **Phase 2 (L-BFGS)**: 10000-12000 epochs，精細優化階段

---

## 🔍 上次訓練問題診斷

### 訓練日誌分析（1000 epochs）
```
初始 loss: 6.829 → 最終 loss: 6.508 → 最佳(epoch 942): 6.499
```

### 關鍵問題：
1. **質量守恆誤差過高**: 0.55-0.65 (目標 < 0.001) ❌
2. **動量守恆誤差超標**: 0.013-0.017 (目標 < 0.01) ❌  
3. **學習率未衰減**: 日誌顯示全程 `lr: 1.00e-03`，exponential scheduler 未生效
4. **Loss 平台期**: epoch 600+ 開始在 6.50-6.52 震盪

---

## ✅ 優化方案

### Phase 1 改進（`kolmogorov_re50_kf4_K100.yml`）

#### 1. 學習率調度修正
```yaml
lr_scheduler:
  type: step        # 從 exponential 改為 step（更可靠）
  gamma: 0.5
  step_size: 2000
  # 時程表：
  # 0-2000: 1e-3
  # 2000-4000: 5e-4
  # 4000-6000: 2.5e-4
  # 6000-8000: 1.25e-4
  # 8000-10000: 6.25e-5
```

#### 2. 強化 Continuity 約束
```yaml
curriculum:
  stages:
    - Stage 1 (0-2000):    continuity: 2.0  (⬆️ 從 1.0)
    - Stage 2 (2000-6000): continuity: 5.0  (⬆️ 從 3.0)
    - Stage 3 (6000-10000): continuity: 10.0 (⬆️ 從 8.0)
```

#### 3. 延長訓練週期
- 從 **1000 → 10000 epochs**（給模型足夠時間收斂）

---

### Phase 2 新增（`kolmogorov_re50_kf4_K100_lbfgs.yml`）

#### 1. L-BFGS 精細優化
```yaml
optimizer:
  type: lbfgs
  lr: 1.0
  max_iter: 20
  line_search_fn: "strong_wolfe"  # 自適應步長搜索
```

#### 2. 嚴格物理約束
```yaml
losses:
  continuity_weight: 15.0  # ⭐ 極嚴格質量守恆
  prior_weight: 0.0        # ⭐ 移除 RANS 先驗依賴
```

#### 3. 增強採樣
```yaml
sampling:
  N_pde: 20000  # ⬆️ 從 18000 增加
  batch_size: 20000
```

---

## 🚀 執行步驟

### Step 1: Phase 1 訓練（SOAP，10000 epochs）

```bash
cd /Users/latteine/Documents/coding/pinns-mvp

python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100.yml

# 預計耗時：~10-12 小時（取決於硬體）
```

**檢查點保存位置**：
- Best model: `checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth`
- Latest: `checkpoints/kolmogorov_re50_kf4_K100_rans_prior/latest.pth`

---

### Step 2: Phase 2 微調（L-BFGS，2000 epochs）

```bash
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100_lbfgs.yml \
  --resume checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth

# 預計耗時：~3-4 小時
# 說明：L-BFGS 每步耗時更長（內部迭代），但收斂更精確
```

**最終檢查點**：
- `checkpoints/kolmogorov_re50_kf4_K100_rans_prior_lbfgs/best_model.pth`

---

## 📊 預期改進

| 指標 | 上次結果 (1000 epochs) | 目標 (12000 epochs) |
|------|----------------------|-------------------|
| **質量守恆誤差** | 0.55-0.65 ❌ | < 0.01 ✅ |
| **動量守恆誤差** | 0.013-0.017 ❌ | < 0.005 ✅ |
| **Total Loss** | 6.499 (最佳) | < 6.0 |
| **訓練穩定性** | 震盪 (600+ epoch) | 單調收斂 |

---

## 🔧 監控關鍵指標

### Phase 1 (SOAP)
監控是否改善：
```bash
# 即時查看
tail -f log/kolmogorov_re50_kf4_K100_rans_prior/training.log | grep -E "Epoch|質量守恆"

# 檢查學習率是否衰減
grep "lr:" log/*/training.log | tail -20
# 應該看到：lr: 1.00e-03 → 5.00e-04 → 2.50e-04 → ...
```

### Phase 2 (L-BFGS)
```bash
# L-BFGS 日誌會顯示 line search 迭代
tail -f log/kolmogorov_re50_kf4_K100_rans_prior_lbfgs/training.log
```

---

## ⚠️ 注意事項

### 1. Curriculum 權重覆蓋機制
配置中的 `curriculum.stages[].weights` **會覆蓋** `losses.*_weight`。
- 實際生效的是 **curriculum 中的值**
- 基礎 `losses` 權重僅作為 fallback

### 2. L-BFGS 特性
- **慢但準**: 每步耗時更長，但精度更高
- **自適應**: 內建 line search，不需要手動調 LR
- **記憶體需求**: `history_size=100` 會儲存過去 100 步的梯度訊息

### 3. Resume 檢查
確保 Phase 2 能正確載入：
```bash
# 驗證 checkpoint 存在
ls -lh checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth

# 檢查 checkpoint 內容
python3 << 'EOF'
import torch
ckpt = torch.load('checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth', 
                  map_location='cpu', weights_only=False)
print(f"Epoch: {ckpt['epoch']}")
print(f"Loss: {ckpt['loss']:.6f}")
print(f"Keys: {list(ckpt.keys())}")
EOF
```

---

## 📈 評估結果

### Phase 1 完成後
```bash
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth \
  --config configs/kolmogorov_re50_kf4_K100.yml
```

### Phase 2 完成後
```bash
python scripts/evaluate/comprehensive_evaluation.py \
  --checkpoint checkpoints/kolmogorov_re50_kf4_K100_rans_prior_lbfgs/best_model.pth \
  --config configs/kolmogorov_re50_kf4_K100_lbfgs.yml
```

---

## 🎯 成功標準

### Phase 1 (SOAP) 通過條件
- [ ] 完成 10000 epochs
- [ ] 質量守恆誤差 < 0.05
- [ ] Loss 穩定下降（無大幅震盪）
- [ ] 學習率正確衰減（檢查日誌）

### Phase 2 (L-BFGS) 通過條件
- [ ] 完成 2000 epochs
- [ ] 質量守恆誤差 < 0.01 ✅
- [ ] 動量守恆誤差 < 0.005 ✅
- [ ] 相對 L2 誤差 < 15% ✅

---

## 🐛 Troubleshooting

### 問題 1: Phase 1 學習率未衰減
**症狀**: 日誌顯示 `lr: 1.00e-03` 不變
**解決**: 
```python
# 在 trainer 中添加 debug 日誌
logging.info(f"Scheduler state: {self.scheduler.state_dict()}")
```

### 問題 2: L-BFGS OOM (Out of Memory)
**症狀**: Phase 2 啟動後記憶體不足
**解決**:
```yaml
# 調整 kolmogorov_re50_kf4_K100_lbfgs.yml
training:
  optimizer:
    history_size: 50  # 從 100 降低
  sampling:
    N_pde: 15000      # 從 20000 降低
```

### 問題 3: Checkpoint 載入失敗
**症狀**: `KeyError: 'model_state_dict'`
**解決**: 檢查 checkpoint 格式
```python
ckpt = torch.load('...pth')
print(ckpt.keys())  # 確認包含 'model_state_dict', 'optimizer_state_dict'
```

---

## 📚 相關文檔

- [CONFIG_REFERENCE.md](../docs/CONFIG_REFERENCE.md) - 完整配置說明
- [TECHNICAL_DOCUMENTATION.md](../docs/TECHNICAL_DOCUMENTATION.md) - 架構細節
- [TROUBLESHOOTING.md](../docs/TROUBLESHOOTING.md) - 常見問題

---

**Created**: 2025-12-17  
**Last Updated**: 2025-12-17  
**Author**: AI Assistant  
**Status**: Ready for Testing ✅
