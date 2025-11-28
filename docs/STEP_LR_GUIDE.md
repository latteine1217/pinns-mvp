# StepLR 學習率調度器配置指南

## 📋 需求

> 學習率以每 2,000 步（epochs）以 0.9 的係數指數衰減

## ✅ 解決方案

使用 `StepLR` 調度器（已支援）

---

## 🚀 配置方式

### 基本配置

```yaml
training:
  epochs: 10000
  
  optimizer:
    name: 'adam'
    lr: 1e-3      # 初始學習率
  
  lr_scheduler:
    type: 'step'
    step_size: 2000   # 每 2000 epochs 衰減一次
    gamma: 0.9        # 衰減係數
```

### 學習率變化軌跡

```
Epoch         Learning Rate
-------------------------------
0-1999        1e-3           (初始學習率)
2000-3999     9e-4           (× 0.9)
4000-5999     8.1e-4         (× 0.9²)
6000-7999     7.29e-4        (× 0.9³)
8000-9999     6.56e-4        (× 0.9⁴)
10000+        5.90e-4        (× 0.9⁵)
```

**數學公式**：
```
lr(epoch) = initial_lr × gamma^(floor(epoch / step_size))
```

---

## 📊 完整配置範例

### 範例 1：Kolmogorov Flow（長期訓練）

```yaml
# configs/kolmogorov_step_decay.yml
experiment:
  name: "kolmogorov_re100_step_decay"

training:
  epochs: 10000
  batch_size: 4000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
    weight_decay: 1e-5
  
  lr_scheduler:
    type: 'step'
    step_size: 2000   # 每 2000 epochs 衰減
    gamma: 0.9        # 衰減 10%

data:
  # ... 你的資料配置
```

**預期行為**：
```
Epoch 0:     lr = 1.000e-03
Epoch 2000:  lr = 9.000e-04  (↓ 10%)
Epoch 4000:  lr = 8.100e-04  (↓ 10%)
Epoch 6000:  lr = 7.290e-04  (↓ 10%)
Epoch 8000:  lr = 6.561e-04  (↓ 10%)
Epoch 10000: lr = 5.905e-04  (↓ 10%)
```

### 範例 2：兩階段訓練（快速衰減後穩定）

```yaml
training:
  epochs: 5000
  
  optimizer:
    name: 'adam'
    lr: 5e-4
  
  lr_scheduler:
    type: 'step'
    step_size: 1000   # 更頻繁的衰減
    gamma: 0.8        # 更激進的衰減（20%）
```

**學習率軌跡**：
```
Epoch 0:    5.00e-04
Epoch 1000: 4.00e-04  (↓ 20%)
Epoch 2000: 3.20e-04  (↓ 20%)
Epoch 3000: 2.56e-04  (↓ 20%)
Epoch 4000: 2.05e-04  (↓ 20%)
Epoch 5000: 1.64e-04  (↓ 20%)
```

### 範例 3：與 SOAP → L-BFGS 兩階段優化器結合

```yaml
training:
  epochs: 4000
  
  # 第一階段：SOAP + StepLR
  optimizer:
    name: 'soap'
    lr: 1e-3
  
  lr_scheduler:
    type: 'step'
    step_size: 1000
    gamma: 0.9
  
  # 第二階段：L-BFGS（不使用學習率調度器）
  switch_to_lbfgs: true
  lbfgs_switch_epoch: 3000
  lbfgs_cfg:
    max_iter: 20
    history_size: 10
```

---

## 🆚 與其他調度器對比

### StepLR vs. ExponentialLR

```yaml
# StepLR（階梯式衰減）
lr_scheduler:
  type: 'step'
  step_size: 2000   # 每 2000 epochs 衰減
  gamma: 0.9
# 行為：學習率在 [0, 2000) 內保持不變，
#       在 epoch 2000 突然降至 0.9 倍

# ExponentialLR（連續指數衰減）
lr_scheduler:
  type: 'exponential'
  gamma: 0.9999     # 每個 epoch 衰減
# 行為：學習率每個 epoch 都平滑下降
```

**視覺化對比**：
```
Learning Rate
    ^
    |
1e-3| ▓▓▓▓▓▓▓▓▓▓▓▓▓╲
    |                 ▓▓▓▓▓▓▓▓▓▓▓╲
9e-4|                             ▓▓▓▓▓▓▓▓▓▓╲
    |                                         ▓▓▓▓▓▓▓
8e-4|-----------------------------------------------------> Epoch
    0              2000         4000        6000
         ⬆ StepLR (階梯狀)

Learning Rate
    ^
1e-3|╲
    | ╲╲
    |   ╲╲
    |     ╲╲
    |       ╲╲
    |         ╲╲____________________________________
    |-----------------------------------------------------> Epoch
    0              2000         4000        6000
         ⬆ ExponentialLR (平滑下降)
```

### StepLR vs. Cosine Annealing

```yaml
# StepLR（固定步長，單調遞減）
lr_scheduler:
  type: 'step'
  step_size: 2000
  gamma: 0.9
# 優點：可預測、易調整
# 缺點：無法逃離局部最小值

# Cosine Annealing（平滑曲線，單調遞減）
lr_scheduler:
  type: 'cosine'
# 優點：平滑收斂
# 缺點：需要預知總訓練 epochs
```

---

## ⚙️ 參數調整指南

### step_size（衰減間隔）

**選擇策略**：

```python
# 經驗公式
step_size = max_epochs // 5  # 總訓練期間衰減 5 次

# 範例：
# 5000 epochs  → step_size = 1000
# 10000 epochs → step_size = 2000
# 20000 epochs → step_size = 4000
```

**推薦值**：

| 總訓練 epochs | 推薦 step_size | 衰減次數 |
|--------------|---------------|---------|
| 1000 | 200 | ~5 次 |
| 5000 | 1000 | ~5 次 |
| **10000** | **2000** | **~5 次** ⭐ |
| 20000 | 4000 | ~5 次 |

### gamma（衰減係數）

**常用值**：

| gamma | 衰減幅度 | 適用場景 |
|-------|---------|---------|
| **0.9** | 10% | 溫和衰減（推薦）⭐ |
| 0.8 | 20% | 中等衰減 |
| 0.5 | 50% | 激進衰減 |
| 0.1 | 90% | 極端衰減（慎用）|

**選擇邏輯**：

```yaml
# 保守策略（穩定收斂）
gamma: 0.9   # 每次只降 10%

# 激進策略（快速收斂）
gamma: 0.7   # 每次降 30%

# 極端策略（快速探索最小值）
gamma: 0.5   # 每次降 50%
```

### 計算最終學習率

```python
# 公式
final_lr = initial_lr × gamma^(max_epochs / step_size)

# 範例：你的配置
initial_lr = 1e-3
gamma = 0.9
step_size = 2000
max_epochs = 10000

# 衰減次數
n_decays = max_epochs / step_size  # 10000 / 2000 = 5

# 最終學習率
final_lr = 1e-3 × 0.9^5
         = 1e-3 × 0.59049
         ≈ 5.9e-4
```

**驗證計算**：
```python
# Python 快速驗證
initial_lr = 1e-3
gamma = 0.9
n_decays = 5
final_lr = initial_lr * (gamma ** n_decays)
print(f"最終學習率: {final_lr:.6f}")
# 輸出: 最終學習率: 0.000590
```

---

## 📈 監控與驗證

### 訓練時實時監控

```bash
# 使用 monitor_training.py
python scripts/monitor_training.py \
  --checkpoint checkpoints/my_exp/ \
  --refresh 10
```

**預期輸出**：
```
Epoch 1999/10000 | Loss: 0.0234 | LR: 1.000e-03
Epoch 2000/10000 | Loss: 0.0198 | LR: 9.000e-04 ⬇ (衰減!)
Epoch 2001/10000 | Loss: 0.0195 | LR: 9.000e-04
...
Epoch 3999/10000 | Loss: 0.0156 | LR: 9.000e-04
Epoch 4000/10000 | Loss: 0.0145 | LR: 8.100e-04 ⬇ (衰減!)
```

### 後處理視覺化

```python
import yaml
import matplotlib.pyplot as plt

# 讀取訓練歷史
with open('checkpoints/my_exp/training_history.yaml', 'r') as f:
    history = yaml.safe_load(f)

epochs = history['epochs']
lr_history = history['learning_rate']
loss_history = history['total_loss']

# 雙軸圖：Loss + Learning Rate
fig, ax1 = plt.subplots(figsize=(12, 5))

# 左軸：Loss
ax1.plot(epochs, loss_history, 'b-', linewidth=2, label='Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('log')

# 右軸：Learning Rate
ax2 = ax1.twinx()
ax2.plot(epochs, lr_history, 'r-', linewidth=2, label='Learning Rate')
ax2.set_ylabel('Learning Rate', color='r')
ax2.tick_params(axis='y', labelcolor='r')
ax2.set_yscale('log')

# 標記衰減點
for epoch in range(2000, max(epochs), 2000):
    ax1.axvline(x=epoch, color='gray', linestyle='--', alpha=0.5)

plt.title('Training Progress: Loss & Learning Rate (StepLR)')
plt.tight_layout()
plt.savefig('training_steplr.png', dpi=150)
```

---

## ⚠️ 常見問題與解決方案

### 問題 1：衰減後 Loss 突然上升

**現象**：
```
Epoch 1999 | Loss: 0.0123 | LR: 1e-3
Epoch 2000 | Loss: 0.0156 | LR: 9e-4 ⬇ (衰減)
Epoch 2001 | Loss: 0.0145 | LR: 9e-4
```

**原因**：學習率降低，梯度更新幅度變小，短期內可能探索到稍差的區域

**解決方案**：
```yaml
# ✅ 正常現象！通常 10-20 epochs 內會恢復
# 若持續上升超過 50 epochs，檢查：
# 1. gamma 是否過小（試試 0.9 → 0.95）
# 2. step_size 是否過短（試試增加間隔）
```

### 問題 2：學習率衰減過快導致訓練停滯

**症狀**：
```yaml
# 問題配置
lr_scheduler:
  type: 'step'
  step_size: 500    # 太頻繁
  gamma: 0.5        # 太激進
```

**結果**：
```
Epoch 0:    1e-3
Epoch 500:  5e-4   (↓ 50%)
Epoch 1000: 2.5e-4 (↓ 50%)
Epoch 1500: 1.25e-4 (已經很小)
Epoch 2000: 6.25e-5 (過小，訓練停滯)
```

**解決方案**：
```yaml
# ✅ 修正配置
lr_scheduler:
  type: 'step'
  step_size: 2000   # 增加間隔
  gamma: 0.9        # 溫和衰減
```

### 問題 3：不確定應該用 StepLR 還是其他調度器

**決策樹**：

```
是否需要週期性"熱重啟"探索參數空間？
├─ 是 → 使用 cosine_warm_restarts
└─ 否 → 繼續

訓練初期是否極不穩定（梯度爆炸）？
├─ 是 → 使用 warmup_cosine
└─ 否 → 繼續

是否已知最佳學習率範圍？
├─ 是 → 使用 step（簡單可控）⭐
└─ 否 → 使用 plateau（自適應）
```

---

## 🎯 推薦配置（你的需求）

### 標準配置（符合需求）

```yaml
training:
  epochs: 10000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'step'
    step_size: 2000   # 每 2000 epochs 衰減
    gamma: 0.9        # 衰減係數 0.9
```

### 進階配置（配合課程學習）

```yaml
training:
  epochs: 10000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'step'
    step_size: 2000
    gamma: 0.9

curriculum:
  stages:
    - name: "stage1"
      max_epochs: 2000   # 學習率保持 1e-3
      # ... 簡單物理參數
    
    - name: "stage2"
      max_epochs: 4000   # 學習率降至 9e-4
      # ... 中等難度
    
    - name: "stage3"
      max_epochs: 6000   # 學習率降至 8.1e-4
      # ... 接近目標參數
```

---

## 📚 參考資源

- **PyTorch 官方文檔**: [StepLR](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html)
- **實作位置**:
  - `pinnx/train/factory.py:1078-1086` (剛剛新增)
  - `pinnx/train/trainer.py:433-439`
- **測試代碼**: `tests/test_lr_schedulers.py:416-434`

---

## 🧪 快速測試

```bash
# 創建測試配置
cat > configs/test_step_decay.yml << 'EOF'
experiment:
  name: "test_step_decay"

training:
  epochs: 5000
  batch_size: 1000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'step'
    step_size: 2000
    gamma: 0.9

# ... 其他配置（data, physics, model）
EOF

# 執行訓練
python scripts/train.py --cfg configs/test_step_decay.yml

# 觀察學習率變化
tail -f log/test_step_decay/training.log | grep "LR:"
```

**預期學習率變化**：
```
Epoch 0:    LR: 1.000e-03
Epoch 1999: LR: 1.000e-03 (保持)
Epoch 2000: LR: 9.000e-04 ⬇ (衰減!)
Epoch 3999: LR: 9.000e-04 (保持)
Epoch 4000: LR: 8.100e-04 ⬇ (衰減!)
Epoch 5000: LR: 8.100e-04 (保持)
```

---

## ✅ 總結

| 項目 | 配置值 |
|------|--------|
| **調度器類型** | `step` |
| **衰減間隔** | `step_size: 2000` epochs |
| **衰減係數** | `gamma: 0.9` (每次降 10%) |
| **最終學習率** | ~5.9e-4 (在 10000 epochs) |
| **衰減次數** | 5 次 |

**配置簡潔**，**行為可預測**，**適合長期訓練**！🚀
