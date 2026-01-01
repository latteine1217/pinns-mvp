# 📊 WandB 追蹤數據完整清單

## 🎯 概述

WandB 目前追蹤**超過 50+ 種指標**，涵蓋損失、訓練動態、模型內部狀態與驗證指標。

---

## 📋 詳細追蹤項目

### 1️⃣ **主要損失 (Main Losses)**

| 指標名稱 | WandB Key | 來源 | 說明 |
|---------|-----------|------|------|
| 總損失 | `Loss/total` | `total_loss` | 所有損失的加權總和 |
| 數據損失 | `Loss/data` | `data_loss` | 感測器數據擬合損失 |
| PDE 損失 | `Loss/pde` | `pde_loss` | 物理方程殘差損失 |
| 邊界損失 | `Loss/boundary` | `bc_loss` | 邊界條件損失 |

---

### 2️⃣ **PDE 子項 (Physics Residuals)**

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| X 動量方程 | `Loss/PDE/momentum_x` | N-S 方程 x 分量殘差 |
| Y 動量方程 | `Loss/PDE/momentum_y` | N-S 方程 y 分量殘差 |
| Z 動量方程 | `Loss/PDE/momentum_z` | N-S 方程 z 分量殘差 (3D) |
| 連續性方程 | `Loss/PDE/continuity` | 質量守恆 (∇·u = 0) |
| 散度損失 | `Loss/PDE/divergence` | 不可壓縮性約束 |

---

### 3️⃣ **數據擬合損失 (Data Fitting)**

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| U 速度 | `Loss/Data/u` | 主流向速度擬合 |
| V 速度 | `Loss/Data/v` | 橫向速度擬合 |
| W 速度 | `Loss/Data/w` | 展向速度擬合 (3D) |
| 壓力 | `Loss/Data/pressure` | 壓力場擬合 |

---

### 4️⃣ **加權損失 (Weighted Losses)**

用於監控 GradNorm 等動態權重策略的效果。

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| 加權數據損失 | `Loss/Weighted/data` | 權重 × 數據損失 |
| 加權 PDE 損失 | `Loss/Weighted/pde` | 權重 × PDE 損失 |
| 加權連續性損失 | `Loss/Weighted/continuity` | 權重 × 散度損失 |
| 加權邊界損失 | `Loss/Weighted/boundary` | 權重 × 邊界損失 |

---

### 5️⃣ **RANS Prior 損失 (Low-Fidelity Prior)**

追蹤低保真 RANS 場的軟約束效果。

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| Prior 總損失 | `Loss/Prior/total` | 先驗一致性總損失 |
| Prior U | `Loss/Prior/u` | U 速度先驗損失 |
| Prior V | `Loss/Prior/v` | V 速度先驗損失 |
| Prior P | `Loss/Prior/p` | 壓力先驗損失 |

---

### 6️⃣ **邊界條件損失 (Boundary Conditions)**

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| X 週期邊界 | `Loss/BC/periodic_x` | X 方向週期性 |
| Y 週期邊界 | `Loss/BC/periodic_y` | Y 方向週期性 |
| 入口邊界 | `Loss/BC/inlet` | 入流邊界條件 |
| 出口邊界 | `Loss/BC/outlet` | 出流邊界條件 |
| 壁面邊界 | `Loss/BC/wall` | 無滑移壁面條件 |

---

### 7️⃣ **正則化項 (Regularization)**

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| 正則化總損失 | `Loss/Regularization/total` | 所有正則化項總和 |
| L2 正則化 | `Loss/Regularization/l2` | 權重 L2 懲罰 |
| 梯度懲罰 | `Loss/Regularization/gradient` | 梯度平滑懲罰 |

---

### 8️⃣ **驗證指標 (Validation Metrics)**

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| 相對 L2 誤差 | `Validation/relative_l2` | 全場相對誤差 |
| MSE | `Validation/mse` | 均方誤差 |

---

### 9️⃣ **訓練超參數 (Training Hyperparameters)**

| 指標名稱 | WandB Key | 頻率 | 說明 |
|---------|-----------|------|------|
| 學習率 | `Training/learning_rate` | 每 epoch | 當前學習率 (scheduler 後) |

---

### 🔟 **梯度與權重統計 (Gradients & Weights)**

**⚠️ 注意**: 這些指標較耗時，通常僅每 N epochs 記錄一次。

#### 梯度統計
- `Gradients/norm/{layer_name}` - 每層梯度範數
- `Gradients/hist/{layer_name}` - 每層梯度直方圖 (WandB Histogram)

#### 權重統計
- `Weights/{layer_name}` - 每層權重直方圖 (WandB Histogram)

**層名稱範例**:
- `feature_layers.0.weight`
- `feature_layers.0.bias`
- `output_layer.weight`
- `fourier_features.B` (Fourier 特徵矩陣)

---

### 1️⃣1️⃣ **非線性係數 (Nonlinearities)**

追蹤 PirateNet/ResNet 的自適應非線性係數。

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| Alpha 係數 | `Nonlinearity/alpha_{idx}` | 各層的 alpha 參數 |

---

### 1️⃣2️⃣ **最終摘要指標 (Final Summary)**

訓練結束時記錄到 `wandb.run.summary`：

| 指標名稱 | WandB Key | 說明 |
|---------|-----------|------|
| 最終損失 | `hparam/final_loss` | 最後一個 epoch 的 total_loss |
| 最佳損失 | `hparam/best_loss` | 訓練過程中的最低 total_loss |
| 最終相對誤差 | `hparam/final_relative_l2` | 最後驗證的相對 L2 誤差 |
| 訓練時長 | `hparam/training_time` | 總訓練時間 (秒) |

---

## 🔧 實際使用範例

### 在 Trainer 中的調用

```python
# 每個 epoch
loop_helper.log_losses_to_wandb(loss_dict, epoch)           # 記錄損失
loop_helper.log_hyperparameters(current_lr, epoch)           # 記錄學習率

# 每 N epochs (較耗時)
if epoch % 50 == 0:
    loop_helper.log_gradients_and_weights(model, epoch)      # 梯度/權重
    loop_helper.log_nonlinearities(model, epoch)             # 非線性係數

# 訓練結束
loop_helper.finalize_wandb(final_metrics, hparams)          # 摘要
```

---

## 📊 WandB Dashboard 可視化

### 默認視圖建議

1. **損失概覽 (Main Losses)**
   - `Loss/total`, `Loss/data`, `Loss/pde`, `Loss/boundary`

2. **PDE 平衡 (Physics Balance)**
   - `Loss/PDE/momentum_x`, `Loss/PDE/momentum_y`, `Loss/PDE/continuity`

3. **數據擬合 (Data Fitting)**
   - `Loss/Data/u`, `Loss/Data/v`, `Loss/Data/w`

4. **權重動態 (Weight Dynamics)**
   - `Loss/Weighted/data`, `Loss/Weighted/pde`

5. **訓練動態 (Training Dynamics)**
   - `Training/learning_rate`
   - `Gradients/norm/*` (梯度爆炸/消失監控)

6. **驗證指標 (Validation)**
   - `Validation/relative_l2`

---

## 🎯 關鍵追蹤策略

### 高頻追蹤 (每 epoch)
✅ 所有損失項  
✅ 學習率  
✅ 驗證指標  

### 中頻追蹤 (每 10-50 epochs)
⚠️ 梯度統計 (監控訓練穩定性)  
⚠️ 非線性係數 (PirateNet)  

### 低頻追蹤 (每 100+ epochs)
🔻 權重直方圖 (視覺化權重分布)  

### 一次性追蹤 (訓練結束)
🏁 最終摘要指標  
🏁 超參數配置 (WandB config)  

---

## 📈 統計摘要

| 類別 | 指標數量 | 頻率 |
|------|---------|------|
| 主要損失 | 4 | 每 epoch |
| PDE 子項 | 5 | 每 epoch |
| 數據擬合 | 4 | 每 epoch |
| 加權損失 | 4 | 每 epoch |
| Prior 損失 | 4 | 每 epoch |
| 邊界條件 | 5 | 每 epoch |
| 正則化 | 3 | 每 epoch |
| 驗證指標 | 2 | 每 epoch |
| 訓練參數 | 1 | 每 epoch |
| 梯度/權重 | 2N (N=層數) | 每 N epochs |
| 非線性 | M (M=alpha數) | 每 N epochs |
| 最終摘要 | 4+ | 訓練結束 |

**總計**: **30+ 常規指標 + 可變數量的梯度/權重統計**

---

## 🔗 相關文件

- **配置設定**: `configs/main.yml` → `logging.wandb: true`
- **實作代碼**: `pinnx/train/training_loop_manager.py`
- **初始化邏輯**: `pinnx/train/trainer.py` (line 149-180)
- **測試腳本**: `scripts/tools/test_wandb_integration.py`
- **遷移指南**: `docs/WANDB_MIGRATION_GUIDE.md`

