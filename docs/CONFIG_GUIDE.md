# 配置指南

> **設計原則**: 單一真相來源 (Single Source of Truth)

本專案的配置系統採用 YAML 格式，所有配置範例和說明都集中管理，避免重複維護。

## 快速導航

### 🎯 我要...

| 任務 | 文件 |
|------|------|
| **查看所有可用配置鍵與預設值** | [`configs/standard_config_template.yml`](../configs/standard_config_template.yml) |
| **學習如何寫配置** | 本文件（繼續往下讀） |
| **驗證配置文件** | 執行 `python scripts/tools/validate_config.py --config <your_config.yml>` |
| **複製範例配置** | 瀏覽 `configs/` 目錄 |
| **查看配置變更歷史** | [`CHANGELOG.md`](../CHANGELOG.md) + Git 歷史 |

## 配置結構概覽

```yaml
experiment:          # 實驗基本信息（名稱、種子、設備）
  name: "my_exp"
  seed: 42
  device: "auto"

data:               # 資料來源與預處理
  source: "jhtdb"
  dataset: "channel"

physics:            # 物理方程與邊界條件
  nu: 5.0e-5
  domain: {...}

model:              # 網路架構
  type: "fourier_vs_mlp"
  width: 256
  depth: 8

losses:             # ⚠️ 注意：複數形式，不是 'loss'
  data_weight: 10.0
  momentum_x_weight: 1.0

training:           # 訓練超參數
  epochs: 1000
  optimizer:
    type: "adam"
    lr: 1e-3

lowfi_prior:        # 低保真先驗（可選）
  enabled: false
  data_path: ""
```

## 常見錯誤與解決

### ❌ 錯誤 1: 使用 `loss` 而非 `losses`

```yaml
# ❌ 錯誤：會被忽略，導致使用預設權重
loss:
  data_weight: 10.0

# ✅ 正確
losses:
  data_weight: 10.0
```

**檢測方法**:
```bash
python scripts/tools/validate_config.py --config your_config.yml
```

### ❌ 錯誤 2: 模型維度與物理域不匹配

```yaml
# ❌ 錯誤：3D 域但 2D 模型
physics:
  domain:
    z_range: [0, 9.42]  # 3D
model:
  in_dim: 2             # 2D

# ✅ 正確：維度一致
model:
  in_dim: 3
```

### ❌ 錯誤 3: 啟用 lowfi_prior 但未提供路徑

```yaml
# ❌ 錯誤：會在訓練開始時失敗
lowfi_prior:
  enabled: true
  data_path: ""         # 空路徑

# ✅ 正確
lowfi_prior:
  enabled: true
  data_path: "./data/lowfi/rans_retau1000.h5"
```

## 配置覆蓋規則

本專案使用 **深度合併 (Deep Merge)** 策略：

```yaml
# base_config.yml
training:
  epochs: 1000
  optimizer:
    type: "adam"
    lr: 1e-3

# my_config.yml（繼承 base_config.yml）
training:
  optimizer:
    lr: 5e-4  # 只覆蓋 lr，保留 type="adam"
```

最終結果：
```yaml
training:
  epochs: 1000      # 繼承自 base
  optimizer:
    type: "adam"    # 繼承自 base
    lr: 5e-4        # 覆蓋
```

## 最佳實踐

### 1. 使用配置驗證工具

每次修改配置後，執行驗證：

```bash
python scripts/tools/validate_config.py --config my_config.yml
```

嚴格模式（警告視為錯誤）：
```bash
python scripts/tools/validate_config.py --config my_config.yml --strict
```

### 2. 從範例配置開始

不要從空白文件開始，複製現有範例：

```bash
# 2D 快速實驗
cp configs/kolmogorov_re50_kf4_K100.yml configs/my_exp.yml

# 3D 生產環境
cp configs/main.yml configs/my_prod_exp.yml
```

### 3. 註解你的修改

```yaml
losses:
  momentum_y_weight: 0.5  # 降低以平衡壁面法向剛性（實驗 #42）
```

### 4. 版本控制配置快照

每次重要實驗都保存配置副本：

```bash
cp configs/my_exp.yml configs/archive/my_exp_20260103.yml
git add configs/archive/my_exp_20260103.yml
git commit -m "Archive config for run #42"
```

## Losses 配置重點說明

### GradNorm 自適應權重（對齊 JaxPI）

GradNorm 動態調整多項損失的權重，確保梯度平衡：

```yaml
losses:
  # 基本權重設定
  data_weight: 10.0
  momentum_x_weight: 1.0
  continuity_weight: 1.0
  
  # GradNorm 自適應權重
  adaptive_weighting: true            # 啟用 GradNorm
  weight_update_freq: 1000            # 每 1000 步更新權重（JaxPI 默認）
  grad_norm_momentum: 0.9             # EMA 平滑係數（0.9=JaxPI 默認，0=不使用）
  grad_norm_alpha: 1.5                # 權重更新激進程度（非 JaxPI 參數）
  grad_norm_normalize: true           # 權重正規化開關（見下方說明）
  
  adaptive_loss_terms:                # 參與自適應調整的損失項
    - data
    - momentum_x
    - continuity
```

#### `grad_norm_normalize` 參數詳解

控制是否正規化權重總和，影響訓練穩定性與行為：

| 值 | 行為 | 特點 | 適用場景 |
|----|------|------|---------|
| `true` (默認) | **PINNx 穩定模式**<br>保持權重總和恆定 | ✅ 訓練穩定<br>✅ 損失尺度可預測<br>✅ 易於調試<br>❌ 非完全 JaxPI 對齊 | 一般生產使用，需要穩定訓練 |
| `false` | **JaxPI 精確對齊**<br>權重反映純梯度比率 | ✅ 完全對齊 JaxPI<br>✅ 程式碼更簡潔<br>❌ 權重總和漂移<br>❌ 損失尺度變化 | 與 JaxPI 論文比較，或追求純理論實現 |

**技術細節**:

- **True**: 每次更新後調整權重，使其總和等於初始總和（如 `data=10, residual=1` → 總和保持 11）
- **False**: 權重直接由梯度比率決定（`w_i = mean_grad / grad_i`），總和隨時間變化

**建議**:
- 實驗對比時設為 `false` 以精確對齊 JaxPI
- 生產訓練時設為 `true` 以提升穩定性
- 兩者數學核心相同，僅後處理方式不同

### Causal Weighting（對齊 JaxPI）

適用於時間依賴問題（如 Burgers、Navier-Stokes）：

```yaml
losses:
  causal_weighting: true              # 啟用因果權重
  causal_tol: 1.0                     # 權重衰減容忍度（JaxPI 默認）
  num_chunks: 32                      # 時間分塊數（Burgers=32, NS=16）
```

## 進階主題

### 動態配置（程式化生成）

對於超參數掃描，使用 Python 生成配置：

```python
import yaml

base_config = yaml.safe_load(open('configs/base.yml'))

for lr in [1e-3, 5e-4, 1e-4]:
    config = base_config.copy()
    config['training']['optimizer']['lr'] = lr
    config['experiment']['name'] = f'lr_sweep_{lr}'
    
    with open(f'configs/generated/lr_{lr}.yml', 'w') as f:
        yaml.dump(config, f)
```

### 配置模板變數（未實現，規劃中）

未來版本可能支援：

```yaml
experiment:
  name: "exp_${timestamp}"  # 自動替換
  
training:
  epochs: ${env:MAX_EPOCHS}  # 從環境變數讀取
```

## 配置完整參考

**主文件**: [`configs/standard_config_template.yml`](../configs/standard_config_template.yml)

該文件包含：
- 所有可用配置鍵（450+ 行）
- 預設值
- 類型說明
- 使用範例

**不要直接修改該文件**，將它作為參考手冊使用。

---

## 已棄用文檔（v1.2.1+）

以下文檔已整合至本文件或 `standard_config_template.yml`：

- ~~`CONFIG_MANAGEMENT_GUIDE.md`~~ → 本文件 + 驗證工具
- ~~`CONFIG_REFERENCE.md`~~ → `standard_config_template.yml`

這些文件保留在 `docs/archive/` 供歷史參考。

---

**最後更新**: 2026-01-05  
**版本**: v1.2.2 (新增 GradNorm normalize_weights 參數說明)
