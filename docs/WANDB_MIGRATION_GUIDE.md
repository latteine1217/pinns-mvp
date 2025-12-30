# WandB 遷移指南

## 📋 概述

本專案已完全從 TensorBoard 遷移至 Weights & Biases (WandB)，不再向後兼容 TensorBoard。

## ⚠️ 重大變更

### 移除的功能
- ❌ TensorBoard SummaryWriter
- ❌ `logging.tensorboard` 配置選項
- ❌ `output.tensorboard_dir` 配置選項
- ❌ `runs/` 目錄（TensorBoard 日誌）

### 新增的功能
- ✅ WandB 集成
- ✅ `logging.wandb` 配置選項
- ✅ `.wandb_config` 配置文件
- ✅ `wandb/` 目錄（WandB 日誌）

## 🔧 配置設定

### 1. WandB API Key 配置

創建 `.wandb_config` 文件於專案根目錄：

```bash
# WandB Configuration
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=pinns-turbulence-reconstruction
WANDB_ENTITY=  # 留空使用預設 entity
```

**注意**：`.wandb_config` 已加入 `.gitignore`，不會提交到版本控制。

### 2. 配置文件更新

將配置文件中的 `tensorboard` 改為 `wandb`：

```yaml
# 舊配置（不再支援）
logging:
  tensorboard: true
  
output:
  tensorboard_dir: "./runs"

# 新配置
logging:
  wandb: true
```

## 📊 日誌記錄功能對比

### TensorBoard → WandB 映射

| TensorBoard | WandB |
|------------|-------|
| `SummaryWriter.add_scalar()` | `wandb.log()` |
| `SummaryWriter.add_histogram()` | `wandb.Histogram()` |
| `SummaryWriter.add_hparams()` | `wandb.run.summary` |

### 日誌結構

所有日誌項目保持相同的命名結構：

- `Loss/total` - 總損失
- `Loss/PDE/momentum_x` - x 方向動量損失
- `Loss/Data/u` - u 變量數據損失
- `Training/learning_rate` - 學習率
- `Validation/relative_l2` - 驗證 L2 誤差

## 🚀 使用方法

### 基本使用

1. 確保 WandB 已安裝：
```bash
pip install wandb>=0.16
```

2. 設定 API key（選項一：環境變數）：
```bash
export WANDB_API_KEY=your_api_key_here
```

3. 設定 API key（選項二：`.wandb_config` 文件）：
```bash
echo "WANDB_API_KEY=your_api_key_here" > .wandb_config
```

4. 在配置文件中啟用 WandB：
```yaml
logging:
  wandb: true
```

5. 正常運行訓練：
```bash
python scripts/train/train.py --config configs/main.yml
```

### 查看實驗結果

- WandB 儀表板：https://wandb.ai/your-entity/pinns-turbulence-reconstruction
- 本地日誌：`wandb/` 目錄

## 🔍 故障排除

### 問題：無法連接 WandB

**解決方案**：
1. 檢查 `.wandb_config` 文件是否存在且包含有效的 API key
2. 檢查網路連接
3. 嘗試手動登入：`wandb login your_api_key`

### 問題：找不到 wandb 模組

**解決方案**：
```bash
pip install wandb>=0.16
# 或
conda install -c conda-forge wandb>=0.16
```

### 問題：配置文件仍使用 tensorboard

**解決方案**：
批量更新所有配置文件：
```bash
find configs -name "*.yml" -exec sed -i 's/tensorboard: true/wandb: true/g' {} \;
```

## 📝 程式碼變更摘要

### 主要修改文件

1. **pinnx/train/trainer.py**
   - 移除 `from torch.utils.tensorboard import SummaryWriter`
   - 新增 `import wandb`
   - 將 `self.writer` 改為 `self.wandb_run`
   - 更新初始化邏輯

2. **pinnx/train/training_loop_manager.py**
   - 移除所有 `self.writer.add_scalar()` 調用
   - 改用 `wandb.log()` 統一記錄
   - 將 `finalize_tensorboard()` 改為 `finalize_wandb()`

3. **配置文件**
   - `configs/templates/standard_config_template.yml` - 更新預設配置
   - 所有實驗配置 - 批量更新 `tensorboard` → `wandb`

## 🎯 優勢

### WandB vs TensorBoard

| 特性 | TensorBoard | WandB |
|------|-------------|-------|
| 雲端同步 | ❌ | ✅ |
| 實驗比較 | 有限 | 強大 |
| 協作功能 | ❌ | ✅ |
| 超參數掃描 | 需手動 | 內建 |
| 模型版本控制 | ❌ | ✅ |
| 資料視覺化 | 基礎 | 豐富 |

## 📚 參考資源

- [WandB 官方文檔](https://docs.wandb.ai/)
- [WandB Python API](https://docs.wandb.ai/ref/python)
- [從 TensorBoard 遷移](https://docs.wandb.ai/guides/integrations/tensorboard)

## 🆘 支援

如有問題，請：
1. 查閱本文檔
2. 檢查 [WandB 文檔](https://docs.wandb.ai/)
3. 提交 Issue 到專案 repository
