# 📊 TensorBoard 監控內容清單

**配置檔**: `configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml`  
**TensorBoard 狀態**: ✅ **已啟用**  
**日誌目錄**: `runs/kolmogorov_re50_kf4_K100_rans_prior/`

---

## 🎯 目前會記錄的內容

### 1. 損失函數 (Loss Metrics)
**頻率**: 每 10 epochs  
**類別**: `Loss/`

```
✅ Loss/total                    - 總損失
✅ Loss/data_loss                - 數據擬合損失
✅ Loss/pde_loss                 - PDE 殘差損失
✅ Loss/div_loss                 - 散度損失（質量守恆）
✅ Loss/weighted_data_loss       - 加權數據損失
✅ Loss/weighted_pde_loss        - 加權 PDE 損失
✅ Loss/weighted_div_loss        - 加權散度損失
✅ Loss/weighted_bc_loss         - 加權邊界條件損失
✅ Loss/momentum_x_loss          - x 方向動量殘差
✅ Loss/momentum_y_loss          - y 方向動量殘差
✅ Loss/continuity_loss          - 連續方程損失
✅ Loss/u_loss                   - u 速度擬合誤差
✅ Loss/v_loss                   - v 速度擬合誤差
✅ Loss/pressure_loss            - 壓力擬合誤差
✅ Loss/periodic_x_loss          - x 方向週期邊界
✅ Loss/periodic_y_loss          - y 方向週期邊界
✅ Loss/prior_consistency_loss   - RANS 先驗一致性
✅ Loss/prior_loss_u             - u 先驗損失
✅ Loss/prior_loss_v             - v 先驗損失
✅ Loss/prior_loss_p             - p 先驗損失
```

### 2. 驗證指標 (Validation Metrics)
**頻率**: 每 50 epochs  
**類別**: `Validation/`

```
✅ Validation/relative_l2        - 相對 L2 誤差
✅ Validation/mse                - 均方誤差
```

### 3. 訓練超參數 (Training Hyperparameters)
**頻率**: 每 10 epochs  
**類別**: `Training/`

```
✅ Training/learning_rate        - 學習率（含調度器衰減）
```

### 4. 梯度與權重統計 (Gradients & Weights)
**頻率**: 每 20 epochs  
**類別**: `Gradients/`, `Weights/`

```
✅ Gradients/{layer_name}        - 梯度分布直方圖（所有層）
✅ Weights/{layer_name}          - 權重分布直方圖（所有層）
```

**範例層名稱**:
- `hidden_layers.0.weight`
- `hidden_layers.1.fc1.weight`
- `hidden_layers.1.fc2.weight`
- `output_layer.weight`
- ...（共約 40+ 個張量）

---

## ❌ 目前**沒有**記錄的內容

### 物理守恆指標（僅在終端顯示，每 100 epochs）
```
❌ 質量守恆誤差 (mass_conservation_error)
❌ 動量守恆誤差 (momentum_conservation_error)
❌ 邊界條件誤差 (boundary_condition_error)
```

### 場預測可視化
```
❌ 速度場 u/v 圖像
❌ 壓力場 p 圖像
❌ 散度場圖像
❌ 誤差分布熱圖
```

### 高級統計
```
❌ 能譜 (energy_spectrum)
❌ 渦度場 (enstrophy)
❌ 壓力梯度誤差 (pressure_gradient)
❌ 訓練/驗證樣本可視化
```

---

## 🔧 建議增強功能

### 優先級 HIGH：物理守恆指標
**原因**: 這是當前最關鍵的問題，必須實時監控

```python
# 在 trainer.py 中添加
if self.writer is not None and epoch % checkpoint_freq == 0:
    # 計算物理指標
    metrics = self.checkpointing.validate_physics(...)
    
    self.writer.add_scalar('Physics/mass_conservation', 
                          metrics['mass_conservation_error'], self.global_step)
    self.writer.add_scalar('Physics/momentum_conservation', 
                          metrics['momentum_conservation_error'], self.global_step)
    self.writer.add_scalar('Physics/boundary_condition', 
                          metrics['boundary_condition_error'], self.global_step)
```

### 優先級 MEDIUM：場可視化
**原因**: 直觀理解模型預測品質

```python
# 每 500 epochs 生成場圖
if epoch % 500 == 0:
    # 預測場
    u_pred, v_pred, p_pred = predict_on_grid(...)
    
    # 添加圖像
    self.writer.add_image('Fields/u_velocity', u_pred_image, self.global_step)
    self.writer.add_image('Fields/v_velocity', v_pred_image, self.global_step)
    self.writer.add_image('Fields/pressure', p_pred_image, self.global_step)
    self.writer.add_image('Fields/divergence', div_image, self.global_step)
```

### 優先級 LOW：損失權重演化
**原因**: 理解自適應權重策略效果

```python
# 如果使用 GradNorm 或自適應權重
if hasattr(self, 'loss_weights'):
    for name, weight in self.loss_weights.items():
        self.writer.add_scalar(f'LossWeights/{name}', weight, self.global_step)
```

---

## 📈 如何啟動 TensorBoard

### 方法 1：本地查看
```bash
# 啟動 TensorBoard
tensorboard --logdir=runs/kolmogorov_re50_kf4_K100_rans_prior --port=6006

# 打開瀏覽器
open http://localhost:6006
```

### 方法 2：多實驗對比
```bash
# 對比多個實驗
tensorboard --logdir=runs --port=6006
```

### 方法 3：遠程查看
```bash
# SSH 端口轉發
ssh -L 6006:localhost:6006 user@remote_server

# 在遠程服務器上啟動 TensorBoard
tensorboard --logdir=runs --bind_all
```

---

## 🎨 TensorBoard 面板預覽

### SCALARS 面板（最常用）
- **Loss**：20+ 條損失曲線（總損失、分項損失、先驗損失等）
- **Training**：學習率衰減曲線
- **Validation**：驗證誤差演化

### HISTOGRAMS 面板
- **Gradients**：每層梯度分布（檢測梯度消失/爆炸）
- **Weights**：每層權重分布（檢測初始化與訓練健康度）

### HPARAMS 面板（訓練結束時）
- 超參數與最終指標對比表

---

## 💡 實際使用建議

### 訓練時重點監控
1. **Loss/total** - 確保下降且穩定
2. **Loss/continuity_loss** - 質量守恆約束
3. **Loss/momentum_x_loss, momentum_y_loss** - 動量守恆約束
4. **Loss/prior_consistency_loss** - RANS 先驗引導效果
5. **Training/learning_rate** - 學習率衰減是否正常

### 診斷問題時
1. **Gradients/** - 檢查是否有梯度消失/爆炸
2. **Loss/weighted_*_loss** - 確認權重平衡
3. **Loss/u_loss vs v_loss** - 檢查變量學習不平衡

### 對比實驗時
1. 在同一個窗口查看不同配置的曲線
2. 使用 TensorBoard 的 smoothing 功能減少噪聲
3. 使用正則表達式過濾特定 loss（例如：`.*momentum.*`）

---

## 📝 配置總結

```yaml
logging:
  tensorboard: true             # ✅ 已啟用
  log_freq: 10                  # 每 10 epochs 記錄標量
  
training:
  checkpoint_freq: 100          # 每 100 epochs 保存檢查點（物理診斷）
  validation_freq: 50           # 每 50 epochs 驗證
  log_interval: 10              # 終端輸出頻率

output:
  tensorboard_dir: runs/kolmogorov_re50_kf4_K100_rans_prior  # 自動生成
```

---

**總結**: 當前 TensorBoard 配置已涵蓋基本訓練監控（損失、學習率、梯度），但**缺少關鍵的物理守恆指標**。建議在下次訓練時添加物理指標記錄，以便實時監控當前最關鍵的問題。
