# Checkpoint Resume 功能完整性報告

## ✅ 核心功能確認

### 1. Checkpoint 載入機制

**位置**: `pinnx/train/checkpointing.py` + `pinnx/train/trainer.py`

#### 支持的功能：
- ✅ 模型參數載入 (`model_state_dict`)
- ✅ 優化器狀態載入 (`optimizer_state_dict`)
- ✅ LR Scheduler 狀態載入 (`lr_scheduler_state_dict`)
- ✅ 訓練歷史載入 (`history`)
- ✅ Physics 模組狀態載入 (`physics_state_dict`)
- ✅ 標準化器狀態載入 (`normalization`)
- ✅ Epoch 計數恢復 (`self.epoch = checkpoint['epoch']`)

#### 實際 Checkpoint 內容（已驗證）：
```python
checkpoint.keys() = [
    'epoch',
    'model_state_dict',
    'optimizer_state_dict',
    'history',
    'config',
    'physics_state_dict',
    'normalization',
    'physics_metrics',
    'metrics',
    'lr_scheduler_state_dict'
]
```

---

## 🔧 Optimizer 切換可行性

### 場景：SOAP (10000 epochs) → L-BFGS (2000 epochs)

#### Trainer.load_checkpoint() 行為：
```python
def load_checkpoint(self, checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location=self.device)
    
    self.model.load_state_dict(checkpoint['model_state_dict'])  # ✅ 載入模型
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # ⚠️ 載入舊優化器狀態
    self.epoch = checkpoint['epoch']  # ⚠️ 繼續 epoch 計數
    
    if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])  # ⚠️ 載入舊 scheduler
```

#### ⚠️ 潛在問題：

1. **Optimizer 狀態不兼容**
   - SOAP 的 `optimizer_state_dict` 包含 momentum 等狀態
   - L-BFGS 的 `optimizer_state_dict` 結構完全不同
   - **結果**：`optimizer.load_state_dict()` 可能失敗或產生警告

2. **Epoch 計數繼續**
   - Phase 1 結束於 epoch 10000
   - Phase 2 會從 epoch 10001 開始訓練
   - 配置的 `epochs: 2000` 是**相對值還是絕對值**？
   - **結果**：可能只訓練到 epoch 10001-12000，符合預期 ✅

3. **LR Scheduler 錯配**
   - L-BFGS 配置 `lr_scheduler: type: none`
   - 但 checkpoint 包含 StepLR 的狀態
   - **結果**：trainer 可能創建 None scheduler，忽略舊狀態 ✅

---

## 🧪 實際測試結論

### 測試 Checkpoint：
- **路徑**: `checkpoints/kolmogorov_re50_kf4_K100_vanilla/epoch_1000.pth`
- **Epoch**: 1000
- **Optimizer**: SOAP (LR: 0.000631)
- **Scheduler**: StepLR (gamma=0.7943, step_size=500)

### 測試結果：
```
✅ Checkpoint 結構完整
✅ 包含所有必要的 state_dict
✅ 可以使用 --resume 載入
```

---

## 📋 使用建議

### ✅ 安全做法（推薦）

#### Phase 2 配置修改（已實現）：
```yaml
# configs/kolmogorov_re50_kf4_K100_lbfgs.yml
training:
  optimizer:
    type: lbfgs  # 新優化器
  lr_scheduler:
    type: none   # 禁用 scheduler
  epochs: 2000   # 額外訓練 2000 epochs
```

#### 執行命令：
```bash
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100_lbfgs.yml \
  --resume checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth
```

#### Trainer 內部流程：
1. **載入配置** → 創建 L-BFGS optimizer (不載入狀態)
2. **載入 checkpoint**:
   - ✅ `model.load_state_dict()` → 成功
   - ⚠️ `optimizer.load_state_dict()` → 失敗（忽略警告）
   - ✅ `self.epoch = 10000` → 從 epoch 10000 繼續
3. **訓練循環**:
   ```python
   for epoch in range(start_epoch, max_epochs):  # range(10000, 12000)
       # 使用 L-BFGS 訓練
   ```

---

### ⚠️ 潛在風險與解決

#### 風險 1: Optimizer State 載入失敗
**現象**：
```
WARNING: optimizer state loading failed: ... (將使用初始狀態)
```

**解決**：
- Trainer 已有 try-except 處理（line 186）
- **影響**：無，L-BFGS 本來就不需要舊的 SOAP state
- **狀態**：✅ 已處理

#### 風險 2: Epoch 計數混淆
**現象**：
- Checkpoint: epoch 10000
- Config: epochs 2000
- 實際訓練：10000 → 12000 ✅

**確認**：
```python
# trainer.py line 1175
for epoch in range(start_epoch, max_epochs):
    # start_epoch = 10000 (from checkpoint)
    # max_epochs = 12000 (config: epochs=2000會被trainer解讀為總epochs)
```

**⚠️ 關鍵問題**：Config 的 `epochs` 是**總 epochs** 還是**額外 epochs**？

讓我檢查：

---

## 🔍 Epochs 計數邏輯確認

### 檢查 _setup_training_config()：
```python
def _setup_training_config(self):
    max_epochs = self.train_cfg.get('epochs', 1000)
    # ...
    return max_epochs, ...
```

### 訓練循環：
```python
start_epoch = self.epoch  # 從 checkpoint 恢復的 epoch
for epoch in range(start_epoch, max_epochs):
    # 如果 start_epoch=10000, max_epochs=2000
    # 則不會執行任何訓練！❌
```

### ❌ **發現嚴重問題！**

如果：
- Checkpoint: `self.epoch = 10000`
- Config: `epochs: 2000`
- 則：`range(10000, 2000)` = 空範圍，**不會訓練**！

---

## 🛠️ 解決方案

### 方案 A：修改 Phase 2 配置（立即可用）

```yaml
# configs/kolmogorov_re50_kf4_K100_lbfgs.yml
training:
  epochs: 12000  # ⚠️ 改為總 epochs（10000 + 2000）
  max_epochs: 12000
```

### 方案 B：手動重置 epoch（需修改代碼）

在 `Trainer.load_checkpoint()` 後手動：
```python
trainer.load_checkpoint(args.resume)
trainer.epoch = 0  # 重置 epoch 計數
```

### 方案 C：修改 Trainer 邏輯（最佳但需時間）

```python
# pinnx/train/trainer.py
def _setup_training_config(self):
    max_epochs = self.train_cfg.get('epochs', 1000)
    
    # 若從 checkpoint 恢復，epochs 視為「額外訓練」
    if self.epoch > 0:
        max_epochs = self.epoch + max_epochs  # ⭐ 關鍵修改
    
    return max_epochs, ...
```

---

## 📊 最終建議

### ✅ 立即可用方案（無需改代碼）

#### 1. 修改 Phase 2 配置：
```yaml
training:
  epochs: 12000  # 總 epochs = 10000 (Phase 1) + 2000 (Phase 2)
```

#### 2. 執行命令：
```bash
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100_lbfgs.yml \
  --resume checkpoints/kolmogorov_re50_kf4_K100_rans_prior/best_model.pth
```

#### 3. 監控 epoch 範圍：
```
應該看到：
Epoch 10000/12000 | ...
Epoch 10001/12000 | ...
...
Epoch 12000/12000 | ... (完成)
```

---

## ⚠️ 關鍵警告

**如果使用當前配置（`epochs: 2000`），訓練將不會執行！**

**必須修改為 `epochs: 12000` 才能正常工作。**

---

## 📝 待辦事項

- [ ] 修改 `kolmogorov_re50_kf4_K100_lbfgs.yml` 的 `epochs: 2000 → 12000`
- [ ] 更新 `TWO_PHASE_TRAINING_GUIDE.md` 中的說明
- [ ] (可選) 提交 PR 修改 Trainer 邏輯支持「額外 epochs」語義

---

**Created**: 2025-12-17  
**Status**: ⚠️ **Critical Issue Found - Config Needs Update**
