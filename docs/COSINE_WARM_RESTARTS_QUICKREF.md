# 學習率調度器快速參考

## ✅ 是的！專案已支援 Cosine Annealing with Warm Restarts

---

## 🚀 快速使用

### 在配置文件中添加

```yaml
training:
  epochs: 1000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'cosine_warm_restarts'  # ⬅️ 關鍵字
    T_0: 100        # 第一個週期長度（epochs）
    T_mult: 1       # 週期倍增因子（1=固定週期，2=遞增週期）
    eta_min: 1e-6   # 最小學習率
```

### 效果示意

```
Learning Rate
    ^
1e-3|     ╱╲       ╱╲           ╱╲
    |    ╱  ╲     ╱  ╲         ╱  ╲
    |   ╱    ╲   ╱    ╲       ╱    ╲
    |  ╱      ╲ ╱      ╲     ╱      ╲
1e-6| ╱        ╲        ╲   ╱        ╲
    |----------╲--------╲--╱----------╲-------> Epoch
              100      200         300
              ⚡       ⚡          ⚡
            (重啟)   (重啟)      (重啟)
```

**關鍵特性**：
- ✅ 每 `T_0` epochs **學習率跳回最大值**
- ✅ 週期性探索參數空間，逃離局部最小值
- ✅ 適合 PINNs 非凸優化問題

---

## 📊 所有支援的調度器

| 類型 | 關鍵字 | 推薦場景 |
|------|--------|---------|
| Warmup + Cosine | `warmup_cosine` | 訓練初期不穩定 ⭐⭐⭐⭐⭐ |
| **Cosine Warm Restarts** | `cosine_warm_restarts` | 逃離局部最小值 ⭐⭐⭐⭐ |
| Cosine Annealing | `cosine` | 標準訓練 ⭐⭐⭐⭐ |
| Exponential Decay | `exponential` | 漸進式收斂 ⭐⭐⭐ |
| Step Decay | `step` | 階段式訓練 ⭐⭐⭐ |
| Reduce on Plateau | `plateau` | 自適應調整 ⭐⭐⭐⭐ |
| Fixed LR | `none` | 簡單任務 ⭐⭐ |

---

## 🎯 參數推薦

### T_0（首次重啟週期）

| 訓練長度 | 推薦 T_0 | 重啟次數 |
|---------|---------|---------|
| 500 epochs | 100 | ~5 次 |
| 1000 epochs | 200 | ~5 次 |
| 2000 epochs | 400 | ~5 次 |
| 5000 epochs | 500 | ~10 次 |

**經驗公式**：`T_0 = max_epochs // 5`

### T_mult（週期倍增因子）

| T_mult | 週期序列 | 適用場景 |
|--------|---------|---------|
| **1** | 100, 100, 100, ... | 均勻探索（推薦初學者）|
| **2** | 100, 200, 400, ... | 後期更穩定（推薦進階）|

### eta_min（最小學習率）

```yaml
# 保守策略（推薦）
eta_min: 1e-6

# 激進策略
eta_min: 1e-7

# 經驗公式
eta_min: initial_lr / 1000
```

---

## 📝 實戰範例

### 範例 1：Kolmogorov Flow（1000 epochs）

```yaml
# configs/my_kolmogorov_experiment.yml
training:
  epochs: 1000
  optimizer:
    name: 'adam'
    lr: 3e-4
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 200        # 每 200 epochs 重啟（共 5 次）
    T_mult: 1       # 固定週期
    eta_min: 1e-6
```

**學習率時間線**：
```
Epoch 0:     3e-4 (開始)
Epoch 100:   ~5e-5 (下降)
Epoch 200:   3e-4 ⚡ (重啟!)
Epoch 300:   ~5e-5 (下降)
Epoch 400:   3e-4 ⚡ (重啟!)
...
```

### 範例 2：長期訓練（遞增週期）

```yaml
training:
  epochs: 5000
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 500        # 第一個週期 500 epochs
    T_mult: 2       # 週期遞增（500 → 1000 → 2000）
    eta_min: 1e-7
```

**週期劃分**：
```
週期 1: Epoch 0-500      (500 epochs)
週期 2: Epoch 500-1500   (1000 epochs)
週期 3: Epoch 1500-3500  (2000 epochs)
週期 4: Epoch 3500-5000  (1500 epochs, 未完成)
```

---

## ⚠️ 注意事項

### ✅ 適合使用的場景

- PINNs 訓練陷入局部最小值
- 長期訓練（> 1000 epochs）
- 需要探索多組學習率組合

### ❌ 不適合的場景

- 快速實驗（< 500 epochs）- 重啟次數不足
- 訓練初期極不穩定 - 先用 `warmup_cosine`
- 已知最佳學習率 - 用標準 `cosine` 即可

### 常見問題

**Q: 重啟時 Loss 會上升嗎？**  
A: ✅ 正常現象！學習率跳回高值，Loss 會短暫上升，幾個 epoch 內會恢復。

**Q: T_0 設多少合適？**  
A: 經驗公式：`T_0 = max_epochs // 5`（總訓練期間約 5 次重啟）

**Q: T_mult 用 1 還是 2？**  
A: 
- 初學者/探索期：`T_mult = 1`（均勻重啟）
- 進階訓練/追求極致：`T_mult = 2`（後期更穩定）

---

## 📚 完整文檔

詳細使用指南、參數調整、視覺化範例：

👉 **`docs/LR_SCHEDULER_GUIDE.md`**

---

## 🧪 快速測試

```bash
# 創建測試配置
cat > configs/test_warm_restarts.yml << EOF
experiment:
  name: "test_warm_restarts"

training:
  epochs: 400
  optimizer:
    name: 'adam'
    lr: 1e-3
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 100
    T_mult: 1
    eta_min: 1e-6
# ... 其他配置
EOF

# 執行訓練
python scripts/train.py --cfg configs/test_warm_restarts.yml

# 觀察學習率變化
tail -f log/test_warm_restarts/training.log | grep "LR:"
```

**預期輸出**：
```
Epoch 100 | LR: 1.0e-06 ⬇
Epoch 101 | LR: 1.0e-03 ⚡ (重啟!)
Epoch 200 | LR: 1.0e-06 ⬇
Epoch 201 | LR: 1.0e-03 ⚡ (重啟!)
...
```

---

## 🔗 相關資源

- **PyTorch 官方文檔**: [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)
- **原始論文**: [SGDR: Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)](https://arxiv.org/abs/1608.03983)
- **實作位置**: 
  - `pinnx/train/factory.py:1052`
  - `pinnx/train/trainer.py:401`
- **測試代碼**: `tests/test_lr_schedulers.py:372`

---

**總結**：專案已完整支援，配置簡單，適合 PINNs 訓練！🚀
