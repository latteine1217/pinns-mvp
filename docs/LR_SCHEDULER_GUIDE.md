# 學習率調度器使用指南

## 📋 支援的調度器類型

專案已支援 **7 種**學習率調度器，包括 **Cosine Annealing with Warm Restarts**！

| 調度器類型 | 關鍵字 | 適用場景 | 推薦度 |
|-----------|--------|---------|--------|
| **Warmup + Cosine** | `warmup_cosine` | 訓練初期不穩定 | ⭐⭐⭐⭐⭐ |
| **Cosine Warm Restarts** | `cosine_warm_restarts` | 逃離局部最小值 | ⭐⭐⭐⭐ |
| **Cosine Annealing** | `cosine` | 標準訓練 | ⭐⭐⭐⭐ |
| **Exponential Decay** | `exponential` | 漸進式收斂 | ⭐⭐⭐ |
| **Step Decay** | `step` | 階段式訓練 | ⭐⭐⭐ |
| **Reduce on Plateau** | `plateau` | 自適應調整 | ⭐⭐⭐⭐ |
| **Fixed Learning Rate** | `none` | 簡單任務 | ⭐⭐ |

---

## 🔥 Cosine Annealing with Warm Restarts

### 原理

**數學公式**：
```
η_t = η_min + 0.5 × (η_max - η_min) × (1 + cos(π × T_cur / T_i))
```

**Warm Restart** 機制：
- 每 `T_0` epochs 重啟一次，學習率跳回最大值
- 重啟後週期長度 × `T_mult`（如 T_mult=2：100, 200, 400, ...）

**優點**：
- ✅ 週期性跳出局部最小值
- ✅ 多次"熱啟動"機會探索參數空間
- ✅ 適合非凸優化（PINNs 訓練）

**示意圖**：
```
Learning Rate
    ^
    |     ╱╲       ╱╲           ╱╲
    |    ╱  ╲     ╱  ╲         ╱  ╲
    |   ╱    ╲   ╱    ╲       ╱    ╲
    |  ╱      ╲ ╱      ╲     ╱      ╲
    | ╱        ╲        ╲   ╱        ╲
    |----------╲--------╲--╱----------╲-------> Epoch
              T_0      2T_0         3T_0
              (重啟)    (重啟)        (重啟)
```

### 配置範例

#### 基本配置
```yaml
training:
  epochs: 1000
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 100        # 第一個週期長度（epochs）
    T_mult: 1       # 週期倍增因子（1=固定週期）
    eta_min: 1e-6   # 最小學習率
```

#### 進階配置（遞增週期）
```yaml
training:
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 50         # 第一個週期 50 epochs
    T_mult: 2       # 每次週期長度 ×2 (50→100→200→400...)
    eta_min: 1e-7   # 更小的最小學習率
```

#### 配合 Curriculum Learning
```yaml
training:
  epochs: 2000
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 200        # 與課程階段對齊
    T_mult: 1       # 固定週期
    eta_min: 1e-6

curriculum:
  stages:
    - name: "stage1"
      max_epochs: 400   # 2 個 Warm Restart 週期
      # ...
    - name: "stage2"
      max_epochs: 800   # 4 個 Warm Restart 週期
      # ...
```

---

## 📊 參數調整指南

### T_0（首次重啟週期）

**推薦值**：
- **小型實驗**（< 500 epochs）：`T_0 = 50-100`
- **中型訓練**（500-2000 epochs）：`T_0 = 100-200`
- **長期訓練**（> 2000 epochs）：`T_0 = 200-500`

**選擇邏輯**：
```python
# 經驗公式
T_0 = max_epochs // 5  # 總訓練期間約 5 次重啟
```

### T_mult（週期倍增因子）

| T_mult | 週期序列（T_0=100） | 適用場景 |
|--------|-------------------|---------|
| **1** | 100, 100, 100, ... | 均勻探索，推薦初學者 ⭐ |
| **2** | 100, 200, 400, ... | 後期更穩定，推薦進階訓練 ⭐⭐ |
| **3** | 100, 300, 900, ... | 快速收斂，慎用（可能過早收斂） |

**推薦策略**：
- **探索期**（前期不確定架構是否合適）：`T_mult = 1`
- **精煉期**（架構已驗證，追求最佳性能）：`T_mult = 2`

### eta_min（最小學習率）

**推薦範圍**：
```yaml
# 保守策略（避免學習停滯）
eta_min: 1e-6

# 激進策略（更徹底探索最小值附近）
eta_min: 1e-7

# 極端策略（幾乎完全衰減）
eta_min: 1e-8
```

**與初始學習率的比例**：
```python
eta_min = initial_lr / 1000  # 通常為初始學習率的 1/1000
```

---

## 🆚 與其他調度器對比

### Warm Restarts vs. 標準 Cosine Annealing

```yaml
# 標準 Cosine Annealing（單次下降）
lr_scheduler:
  type: 'cosine'
  # 學習率單調下降，適合：
  # - 凸優化問題
  # - 已知模型架構穩定

# Cosine Warm Restarts（週期性重啟）
lr_scheduler:
  type: 'cosine_warm_restarts'
  T_0: 100
  T_mult: 1
  # 學習率週期性跳躍，適合：
  # - 非凸優化（PINNs）✓
  # - 容易陷入局部最小值 ✓
  # - 需探索多個參數配置 ✓
```

### Warm Restarts vs. Warmup + Cosine

```yaml
# Warmup + Cosine（漸進啟動）
lr_scheduler:
  type: 'warmup_cosine'
  warmup_epochs: 50
  max_lr: 1e-3
  min_lr: 1e-6
  # 適合：訓練初期不穩定、梯度爆炸風險

# Cosine Warm Restarts（週期性探索）
lr_scheduler:
  type: 'cosine_warm_restarts'
  T_0: 100
  T_mult: 1
  eta_min: 1e-6
  # 適合：訓練中期陷入局部最小值
```

**組合使用**（不支援，需手動實作）：
```yaml
# 理想但目前不支援的配置
# lr_scheduler:
#   type: 'warmup_cosine_warm_restarts'  # ❌ 未實作
```

---

## 🔬 實驗配置範例

### 範例 1：Kolmogorov Flow（快速實驗）

```yaml
# configs/kolmogorov_cosine_warm_restarts.yml
experiment:
  name: "kolmogorov_re100_kf4_cosine_warm_restarts"

training:
  epochs: 1000
  batch_size: 4000
  
  optimizer:
    name: 'adam'
    lr: 3e-4
    weight_decay: 1e-5
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 200        # 每 200 epochs 重啟（共 5 次重啟）
    T_mult: 1       # 固定週期
    eta_min: 1e-6

# 預期學習率時間線：
# Epoch 0-200:   3e-4 → 1e-6 (第1個週期)
# Epoch 200:     跳回 3e-4 ⚡ (重啟)
# Epoch 200-400: 3e-4 → 1e-6 (第2個週期)
# Epoch 400:     跳回 3e-4 ⚡ (重啟)
# ...
```

### 範例 2：Channel Flow（長期訓練）

```yaml
# configs/channel_flow_cosine_warm_restarts_progressive.yml
experiment:
  name: "channel_re1000_progressive_restart"

training:
  epochs: 5000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 500        # 第一個週期 500 epochs
    T_mult: 2       # 週期遞增（500 → 1000 → 2000）
    eta_min: 1e-7

# 預期學習率時間線：
# Epoch 0-500:    1e-3 → 1e-7 (第1個週期，500 epochs)
# Epoch 500:      跳回 1e-3 ⚡
# Epoch 500-1500: 1e-3 → 1e-7 (第2個週期，1000 epochs)
# Epoch 1500:     跳回 1e-3 ⚡
# Epoch 1500-3500: 1e-3 → 1e-7 (第3個週期，2000 epochs)
# Epoch 3500:     跳回 1e-3 ⚡
# Epoch 3500-5000: 1e-3 → ? (第4個週期未完成)
```

### 範例 3：兩階段優化（SOAP → L-BFGS）

```yaml
# configs/two_stage_cosine_warm_restarts.yml
training:
  epochs: 2000
  
  # 第一階段：SOAP + Cosine Warm Restarts
  optimizer:
    name: 'soap'
    lr: 1e-3
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 200
    T_mult: 1
    eta_min: 1e-6
  
  # 第二階段設定
  switch_to_lbfgs: true
  lbfgs_switch_epoch: 1800  # 在第 9 個重啟週期後切換
  lbfgs_cfg:
    max_iter: 20
    history_size: 10
```

---

## 📈 監控與視覺化

### 訓練時監控

```bash
# 使用 monitor_training.py 實時查看學習率
python scripts/monitor_training.py \
  --checkpoint checkpoints/my_exp/ \
  --refresh 10
```

**預期輸出**（Cosine Warm Restarts）：
```
Epoch 100/1000 | Loss: 0.0234 | LR: 5.2e-05 ⬇
Epoch 200/1000 | Loss: 0.0198 | LR: 1.0e-06 ⬇
Epoch 201/1000 | Loss: 0.0195 | LR: 3.0e-04 ⚡ (重啟!)
Epoch 300/1000 | Loss: 0.0156 | LR: 5.2e-05 ⬇
```

### 後處理視覺化

```python
# 從訓練歷史提取學習率
import yaml
import matplotlib.pyplot as plt

with open('checkpoints/my_exp/training_history.yaml', 'r') as f:
    history = yaml.safe_load(f)

epochs = history['epochs']
lr_history = history['learning_rate']

plt.figure(figsize=(12, 4))
plt.plot(epochs, lr_history, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Cosine Annealing with Warm Restarts')
plt.yscale('log')  # 對數尺度更清楚
plt.grid(True, alpha=0.3)
plt.savefig('lr_schedule.png', dpi=150)
```

---

## ⚠️ 常見陷阱與解決方案

### 陷阱 1：重啟時 Loss 突然上升

**現象**：
```
Epoch 199 | Loss: 0.0123 | LR: 1e-6
Epoch 200 | Loss: 0.0456 | LR: 3e-4 ⚡ (重啟)
Epoch 201 | Loss: 0.0389 | LR: 2.8e-4
```

**原因**：學習率跳回高值，梯度更新幅度變大

**解決方案**：
```yaml
# ✅ 正常現象！Loss 會在幾個 epoch 內恢復
# 若持續上升，降低 initial_lr：
optimizer:
  lr: 5e-4  # 原本 1e-3 → 5e-4
```

### 陷阱 2：週期過短導致訓練不穩定

**現象**：
```yaml
# ❌ 問題配置
lr_scheduler:
  type: 'cosine_warm_restarts'
  T_0: 10  # 太短！
```

**結果**：每 10 epochs 重啟一次，模型無法充分收斂

**解決方案**：
```yaml
# ✅ 修正配置
lr_scheduler:
  T_0: 100  # 增加至合理長度
```

**經驗法則**：`T_0 ≥ 50`

### 陷阱 3：eta_min 過大導致無法充分探索

**現象**：
```yaml
# ❌ 問題配置
optimizer:
  lr: 1e-3
lr_scheduler:
  eta_min: 1e-4  # 只下降 10 倍
```

**結果**：學習率始終較高，無法精細調整參數

**解決方案**：
```yaml
# ✅ 修正配置
lr_scheduler:
  eta_min: 1e-6  # 下降 1000 倍
```

**經驗法則**：`eta_min ≤ initial_lr / 1000`

---

## 🎯 推薦使用場景

### ✅ 適合使用 Cosine Warm Restarts

1. **PINNs 訓練陷入局部最小值**
   - 物理損失 & 資料損失不平衡
   - 訓練中期 loss 停滯

2. **超參數敏感度高**
   - 需要探索多組學習率組合
   - 不確定最佳訓練策略

3. **長期訓練（> 1000 epochs）**
   - 有足夠時間進行多次重啟
   - 可受益於多次「熱啟動」

### ❌ 不適合使用的場景

1. **快速實驗（< 500 epochs）**
   - 重啟次數不足，效果不明顯
   - 建議使用 `warmup_cosine`

2. **已知最佳學習率**
   - 模型架構穩定、數據充足
   - 使用標準 `cosine` 即可

3. **訓練初期極不穩定**
   - 梯度爆炸風險高
   - 先使用 `warmup_cosine` 穩定後再考慮

---

## 📚 相關文檔

- **PyTorch 官方文檔**: [CosineAnnealingWarmRestarts](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html)
- **原始論文**: Loshchilov & Hutter (2017), [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/abs/1608.03983)
- **專案測試**: `tests/test_lr_schedulers.py:372`
- **實作位置**: `pinnx/train/factory.py:1052`, `pinnx/train/trainer.py:401`

---

## 🧪 快速測試

建立一個最小配置測試 Cosine Warm Restarts：

```yaml
# configs/test_cosine_warm_restarts.yml
experiment:
  name: "test_cosine_warm_restarts"

training:
  epochs: 400
  batch_size: 1000
  
  optimizer:
    name: 'adam'
    lr: 1e-3
  
  lr_scheduler:
    type: 'cosine_warm_restarts'
    T_0: 100
    T_mult: 1
    eta_min: 1e-6

data:
  # ... 你的資料配置
```

**執行**：
```bash
python scripts/train.py --cfg configs/test_cosine_warm_restarts.yml

# 觀察 log 中的學習率變化
tail -f log/test_cosine_warm_restarts/training.log | grep "LR:"
```

**預期行為**：
- Epoch 1-100: 學習率逐漸下降
- Epoch 100: 學習率跳回 1e-3 ⚡
- Epoch 101-200: 重複衰減
- Epoch 200: 再次重啟 ⚡
- ...

---

**總結**：專案已完整支援 Cosine Annealing with Warm Restarts，配置簡單，適合 PINNs 非凸優化問題！🚀
