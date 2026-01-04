# TrainerBuilder 使用指南

> **版本**: v1.3.4 (2026-01-05)
> **狀態**: 推薦使用 ✅

---

## 概述

**TrainerBuilder** 是 v1.3.4 引入的建構器模式（Builder Pattern），用於簡化 `Trainer` 的創建和配置。

### 為什麼使用 TrainerBuilder？

**❌ 舊方式（已棄用）**:
```python
# 需要手動創建和配置所有組件
weighters = create_weighters(config, model, device, physics=physics)
normalizer = setup_output_normalization(config, training_data, logger)

trainer = Trainer(model, physics, losses, config, device,
                  weighters=weighters,
                  input_normalizer=input_normalizer,
                  training_data=training_data)

# 還需要手動設置更多屬性
trainer.data_normalizer = normalizer
```

**✅ 新方式（推薦）**:
```python
from pinnx.train.trainer_builder import TrainerBuilder

builder = TrainerBuilder(config, device)
builder.with_model(model)
builder.with_physics(physics)
builder.with_losses(losses)
builder.with_training_data(training_data)

trainer = builder.build()  # 所有組件自動創建 ✨
```

### 優勢

| 特性 | 舊方式 | TrainerBuilder |
|------|--------|----------------|
| **代碼量** | ~50 行 | ~6 行 |
| **組件創建** | 手動 | 自動 |
| **錯誤風險** | 高（容易遺漏） | 低（自動檢查） |
| **可測試性** | 低（緊耦合） | 高（依賴注入） |
| **可維護性** | 低（分散配置） | 高（集中管理） |

---

## 基本用法

### 1. 最簡單的例子

```python
from pinnx.train.trainer_builder import TrainerBuilder

# 創建 builder
builder = TrainerBuilder(config, device)

# 配置必需組件
builder.with_model(model)
builder.with_physics(physics)
builder.with_losses(losses)

# 構建 Trainer（所有可選組件自動創建）
trainer = builder.build()

# 開始訓練
trainer.train()
```

### 2. 完整例子（包含訓練數據）

```python
from pinnx.train.trainer_builder import TrainerBuilder
from pinnx.train.model_physics_factory import create_model, create_physics, get_device
from pinnx.train.loss_factory import create_loss_functions
from pinnx.dataio.loaders.kolmogorov import prepare_kolmogorov_training_data

# 1. 設置設備
device = get_device(config['experiment']['device'])

# 2. 創建基本組件
model = create_model(config, device)
physics = create_physics(config, device)
losses = create_loss_functions(config, device)
training_data = prepare_kolmogorov_training_data(config, device)

# 3. 使用 TrainerBuilder
builder = TrainerBuilder(config, device)
builder.with_model(model)
builder.with_physics(physics)
builder.with_losses(losses)
builder.with_training_data(training_data)

# 4. 構建並訓練
trainer = builder.build()
result = trainer.train()
```

---

## 自動創建的組件

TrainerBuilder 會根據配置自動創建以下組件：

### 訓練組件
- ✅ **Optimizer**: Adam/SGD/AdamW（從 config.training.optimizer）
- ✅ **LR Scheduler**: Cosine/Step/Exponential（從 config.training.lr_scheduler）
- ✅ **AMP Scaler**: 混合精度訓練（從 config.training.amp）

### 策略組件
- ✅ **CheckpointManager**: Checkpoint 管理
- ✅ **PeriodicCheckpointStrategy**: 保存策略
- ✅ **ValidationManager**: 驗證管理

### 功能組件
- ✅ **PriorLossManager**: RANS 先驗損失（如果啟用）
- ✅ **FourierAnnealing**: Fourier 特徵退火（如果啟用）
- ✅ **AdaptiveSampler**: 自適應採樣（如果啟用）
- ✅ **EarlyStopping**: 早停配置（如果啟用）

### 監控組件
- ✅ **Timer**: 訓練時間追蹤
- ✅ **MemoryTracker**: 記憶體使用追蹤
- ✅ **PhysicsValidator**: 物理約束驗證
- ✅ **WandB**: 實驗追蹤（如果啟用）

### 數據組件
- ✅ **InputNormalizer**: 輸入標準化
- ✅ **OutputNormalizer**: 輸出標準化
- ✅ **Loss Weighters**: GradNorm/NTK/Adaptive/Causal

---

## 配置文件示例

TrainerBuilder 會從配置文件自動讀取所有設置：

```yaml
training:
  epochs: 1000
  lr: 0.001
  optimizer: adam  # 自動創建 Adam
  lr_scheduler:
    type: cosine   # 自動創建 Cosine Annealing
    T_max: 1000
  amp:
    enabled: true  # 自動創建 AMP Scaler
  checkpoint:
    enabled: true  # 自動創建 CheckpointManager
    save_every: 100

weighting:
  method: gradnorm  # 自動創建 GradNorm Weighter
  enabled: true
  alpha: 1.5

normalization:
  type: training_data_norm  # 自動創建 Normalizer
  variable_order: [u, v, p]

lowfi_prior:
  enabled: true   # 自動創建 PriorLossManager
  data_path: ./data/rans_prior.h5

wandb:
  enabled: true   # 自動初始化 WandB
  project: my-project
```

---

## 進階用法

### 1. 自定義組件（如果需要）

雖然 TrainerBuilder 會自動創建組件，但你仍可以提供自定義組件：

```python
# 創建自定義 optimizer
custom_optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

builder = TrainerBuilder(config, device)
builder.with_model(model)
builder.with_physics(physics)
builder.with_losses(losses)

# 構建時會優先使用配置中的設置
# 如果需要完全自定義，可以在 build() 後手動替換
trainer = builder.build()
trainer.optimizer = custom_optimizer  # 替換為自定義 optimizer
```

### 2. 檢查創建的組件

```python
trainer = builder.build()

# 查看創建的組件
print(f"Optimizer: {type(trainer.optimizer).__name__}")
print(f"LR Scheduler: {type(trainer.lr_scheduler).__name__}")
print(f"AMP Enabled: {trainer.use_amp}")
print(f"Checkpoint Manager: {trainer.checkpoint_manager is not None}")
print(f"Validation Manager: {trainer.validation_manager is not None}")
```

### 3. Ensemble 訓練

```python
models = []
for seed in [42, 43, 44]:
    # 為每個 seed 創建新模型
    set_random_seed(seed)
    member_model = create_model(config, device)

    # 使用 TrainerBuilder
    builder = TrainerBuilder(config, device)
    builder.with_model(member_model)
    builder.with_physics(physics)
    builder.with_losses(losses)
    builder.with_training_data(training_data)

    trainer = builder.build()
    trainer.train()

    models.append(member_model)
```

---

## 遷移指南

### 從舊代碼遷移

**Before**:
```python
# 舊代碼（約 50 行）
weighters = create_weighters(config, model, device, physics=physics)
normalizer = setup_output_normalization(config, training_data, logger)
trainer = Trainer(model, physics, losses, config, device,
                  weighters=weighters,
                  input_normalizer=input_normalizer,
                  training_data=training_data)
trainer.data_normalizer = normalizer
# ... 更多手動設置
```

**After**:
```python
# 新代碼（6 行）
builder = TrainerBuilder(config, device)
builder.with_model(model)
builder.with_physics(physics)
builder.with_losses(losses)
builder.with_training_data(training_data)
trainer = builder.build()
```

### 棄用警告

如果你仍然直接實例化 `Trainer`，會看到以下警告：

```
DeprecationWarning:
⚠️  直接實例化 Trainer 已棄用，建議使用 TrainerBuilder。

推薦用法：
  from pinnx.train.trainer_builder import TrainerBuilder

  builder = TrainerBuilder(config, device)
  builder.with_model(model)
  builder.with_physics(physics)
  builder.with_losses(losses)
  builder.with_training_data(training_data)

  trainer = builder.build()

此警告將在未來版本變為錯誤。
```

---

## TrainerComponents 內部結構

TrainerBuilder 使用 `TrainerComponents` 數據類封裝所有組件：

```python
@dataclass
class TrainerComponents:
    # 訓練組件
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[Any] = None
    amp_scaler: Optional[GradScaler] = None

    # 策略組件
    checkpoint_manager: Optional[Any] = None
    validation_manager: Optional[Any] = None

    # 功能組件
    prior_loss_manager: Optional[Any] = None
    fourier_annealing: Optional[Any] = None
    adaptive_sampler: Optional[Any] = None

    # 監控組件
    timer: Optional[Any] = None
    physics_validator: Optional[Any] = None
    wandb_run: Optional[Any] = None

    # 數據組件
    input_normalizer: Optional[Any] = None
    data_normalizer: Optional[Any] = None
    training_data: Optional[Dict] = None

    # ... 更多組件
```

---

## 故障排除

### Q: TrainerBuilder 不創建某個組件？
**A**: 檢查配置文件中對應的 `enabled` 設置。例如：
- WandB: `wandb.enabled: true`
- Prior Loss: `lowfi_prior.enabled: true`
- AMP: `training.amp.enabled: true`

### Q: 需要自定義組件創建邏輯？
**A**: 可以在 `build()` 後手動替換：
```python
trainer = builder.build()
trainer.my_custom_component = MyComponent()
```

### Q: 向後兼容性？
**A**: 舊的直接實例化方式仍然可用，但會顯示棄用警告。建議盡快遷移。

---

## 參考資料

- **源代碼**: `pinnx/train/trainer_builder.py`
- **組件封裝**: `pinnx/train/trainer_components.py`
- **版本說明**: `README.md` v1.3.4
- **設計文檔**: `docs/TECHNICAL_DOCUMENTATION.md`

---

## 更新日誌

### v1.3.4 (2026-01-05)
- ✅ 初始發布 TrainerBuilder
- ✅ TrainerComponents 數據類
- ✅ 自動組件創建
- ✅ Dual-Path Architecture（新路徑 + 舊路徑）
- ✅ Deprecation Warning 添加
