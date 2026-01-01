# 配置管理指南 (Configuration Management Guide)

**更新日期**: 2025-12-19  
**版本**: v1.0

---

## 🎯 目的

本指南說明如何正確管理 PINNx 專案的配置檔案，避免常見的配置錯誤，特別是鍵名不一致導致的問題。

---

## 📋 快速開始

### 1. 使用標準配置模板

所有新配置應基於標準模板創建：

```bash
# 複製標準模板
cp configs/templates/standard_config_template.yml configs/my_experiment.yml

# 編輯配置
vim configs/my_experiment.yml
```

### 2. 驗證配置正確性

在運行訓練前，務必驗證配置：

```bash
# 單個文件驗證
python scripts/tools/validate_config_keys.py configs/my_experiment.yml

# 批量驗證
python scripts/tools/validate_config_keys.py configs/*.yml
```

---

## ⚠️ 常見錯誤與修復

### 錯誤 1: 使用 `loss:` 而非 `losses:`

**症狀**：
```yaml
# ❌ 錯誤配置
loss:
  momentum_y_weight: 0.3
  continuity_weight: 2.0
```

**結果**：
- 配置被忽略，所有權重預設為 `1.0`
- 訓練日誌顯示 `Momentum Y: 1.00e+00` 而非 `3.00e-01`

**修復**：
```yaml
# ✅ 正確配置
losses:  # ← 改為複數
  momentum_y_weight: 0.3
  continuity_weight: 2.0
```

**驗證**：
```bash
python scripts/tools/validate_config_keys.py configs/my_experiment.yml
```

應輸出：
```
✅ 配置檢查通過，未發現問題
```

---

### 錯誤 2: 損失權重拼寫錯誤

**症狀**：
```yaml
losses:
  momemtum_y_weight: 0.3  # ❌ 拼寫錯誤: momemtum
```

**結果**：
- 該權重被忽略
- 使用預設值 `2.0`

**修復**：
```yaml
losses:
  momentum_y_weight: 0.3  # ✅ 正確拼寫
```

**驗證工具會提示**：
```
⚠️  發現未知的損失權重鍵: momemtum_y_weight
   這些鍵可能不會被使用，請確認拼寫是否正確
```

---

## 📖 標準配置鍵名參考

### 頂層配置段落

| 鍵名 | 類型 | 必填 | 預設值 | 說明 |
|------|------|------|--------|------|
| `experiment` | dict | ✅ | - | 實驗基本設定 |
| `data` | dict | ✅ | - | 資料配置 |
| `model` | dict | ✅ | - | 模型架構 |
| `physics` | dict | ✅ | - | 物理設定 |
| **`losses`** | dict | ✅ | - | **損失函數權重**（必須用複數） |
| `training` | dict | ✅ | - | 訓練設定 |
| `lowfi_prior` | dict | ❌ | - | 低保真先驗（可選） |
| `physics_validation` | dict | ❌ | `{enabled: true}` | 物理驗證 |
| `logging` | dict | ❌ | - | 日誌配置 |
| `output` | dict | ❌ | - | 輸出路徑 |

### Losses 段落配置鍵

#### 基礎損失權重
```yaml
losses:
  data_weight: 10.0              # 資料損失 (預設: 10.0)
```

#### PDE 損失權重
```yaml
losses:
  momentum_x_weight: 1.0         # x 動量方程 (預設: 2.0)
  momentum_y_weight: 1.0         # y 動量方程 (預設: 2.0)
  momentum_z_weight: 1.0         # z 動量方程 (預設: 2.0)
  continuity_weight: 1.0         # 連續方程 (預設: 2.0)
```

#### 約束損失權重
```yaml
losses:
  wall_constraint_weight: 10.0   # 壁面約束 (預設: 10.0)
  periodicity_weight: 10.0       # 週期性約束 (預設: 5.0)
  pressure_gradient_weight: 1.0  # 壓力梯度 (預設: 1.0)
```

#### 先驗損失權重
```yaml
losses:
  prior_weight: 0.1              # 先驗一致性 (預設: 0.1)
```

#### 正則化權重
```yaml
losses:
  source_l1: 1.0e-6              # 源項 L1 正則 (預設: 1e-6)
  gradient_penalty: 1.0e-4       # 梯度懲罰 (預設: 1e-4)
```

#### 自適應權重配置
```yaml
losses:
  adaptive_weighting: false      # 啟用自適應權重 (預設: false)
  weight_update_freq: 1000       # 權重更新頻率 (預設: 1000)
  grad_norm_alpha: 1.5           # GradNorm alpha (預設: 1.5)
  adaptive_loss_terms:           # 可自適應調整的損失項
    - data
    - momentum_x
    - momentum_y
    - momentum_z
    - continuity
```

---

## 🔧 配置檢查器使用

### 基本用法

```bash
# 檢查單個檔案
python scripts/tools/validate_config_keys.py configs/my_experiment.yml

# 檢查多個檔案
python scripts/tools/validate_config_keys.py \
  configs/phase_a_qr_baseline_fixed.yml \
  configs/phase_a_qr_quick_test.yml

# 檢查所有配置檔案
python scripts/tools/validate_config_keys.py configs/*.yml

# 檢查模板
python scripts/tools/validate_config_keys.py configs/templates/*.yml
```

### 輸出解讀

#### ✅ 通過檢查
```
======================================================================
📋 配置檔案: configs/my_experiment.yml
======================================================================
✅ 配置檢查通過，未發現問題
```

#### ❌ 發現錯誤
```
======================================================================
📋 配置檔案: configs/bad_config.yml
======================================================================

❌ 發現 1 個錯誤（阻斷性問題）:

1. ❌ 使用了錯誤的鍵名 'loss'，應改為 'losses' (複數)
   位置: 配置檔案頂層
   影響: 損失權重配置將被忽略，所有權重預設為 1.0
   修復: 將 'loss:' 改為 'losses:'

❌ 整體狀態: 失敗（存在阻斷性錯誤，需要修復）
```

#### ⚠️ 有警告
```
======================================================================
📋 配置檔案: configs/warning_config.yml
======================================================================

⚠️  發現 1 個警告（非阻斷性問題）:

1. ⚠️  發現未知的損失權重鍵: momemtum_y_weight
   這些鍵可能不會被使用，請確認拼寫是否正確

✅ 整體狀態: 通過（有警告但不影響運行）
```

---

## 📚 程式碼行為參考

### LossManager 配置讀取邏輯

**位置**: `pinnx/train/loss_manager.py` (line 73-74)

```python
# 正確的實現（已修復）
self.loss_cfg = config.get('losses', {})
```

**讀取順序**：
1. 讀取 `config['losses']`（標準鍵名）
2. 若不存在則返回空字典 `{}`

**預設權重**（當配置為空時）：
```python
DEFAULT_WEIGHTS = {
    'data': 10.0,
    'momentum_x': 2.0,
    'momentum_y': 2.0,
    'momentum_z': 2.0,
    'continuity': 2.0,
    'wall_constraint': 10.0,
    'periodicity': 5.0,
    # ...
}
```

### Trainer 配置讀取邏輯

**位置**: `pinnx/train/trainer.py` (line 98)

```python
self.loss_cfg = config.get('losses', {})
```

### Factory 配置讀取邏輯

**位置**: `pinnx/train/factory.py` (line 816, 867)

```python
# Physics 創建時
loss_config=config.get('losses', {})

# Loss 權重推導時
loss_cfg = config.get('losses', {})
```

---

## 🎓 最佳實踐

### 1. 始終使用模板

```bash
# ✅ 好習慣
cp configs/templates/standard_config_template.yml configs/my_new_exp.yml

# ❌ 壞習慣
touch configs/my_new_exp.yml  # 從空白開始
```

### 2. 訓練前驗證

```bash
# 將驗證加入訓練腳本
python scripts/tools/validate_config_keys.py configs/my_experiment.yml && \
python scripts/train/train.py --cfg configs/my_experiment.yml
```

### 3. 版本控制配置

```bash
# 提交前檢查所有配置
python scripts/tools/validate_config_keys.py configs/*.yml

# 若有錯誤則拒絕提交
if [ $? -ne 0 ]; then
    echo "❌ 配置驗證失敗，請修復後再提交"
    exit 1
fi

git add configs/
git commit -m "Add new experiment config"
```

### 4. 文檔化非標準配置

若使用非標準配置鍵，請在配置檔案中註解說明：

```yaml
losses:
  # 🔬 實驗性配置：測試新的損失項
  # 注意：此鍵名未被識別，需手動在 LossManager 中處理
  experimental_divergence_penalty: 0.5
```

---

## 🐛 疑難排解

### 問題：權重配置不生效

**症狀**：
```
訓練日誌:
  Momentum Y: 1.00e+00  ← 應該是 0.3
```

**診斷**：
```bash
# 1. 檢查配置鍵名
python scripts/tools/validate_config_keys.py configs/your_config.yml

# 2. 檢查 YAML 格式
python -c "import yaml; yaml.safe_load(open('configs/your_config.yml'))"

# 3. 手動驗證配置載入
python << 'EOF'
import yaml
with open('configs/your_config.yml') as f:
    cfg = yaml.safe_load(f)
    loss_cfg = cfg.get('losses', {})
    print(f"momentum_y_weight: {loss_cfg.get('momentum_y_weight', 'NOT FOUND')}")
EOF
```

**可能原因**：
1. 使用 `loss:` 而非 `losses:` ✅ 最常見
2. 鍵名拼寫錯誤（`momemtum_y_weight`）
3. YAML 縮排錯誤
4. 配置值類型錯誤（字串而非數字）

---

## 📊 驗證清單

在運行訓練前，確認以下項目：

- [ ] 配置基於標準模板創建
- [ ] 使用 `losses:` 而非 `loss:`
- [ ] 所有損失權重鍵名正確拼寫
- [ ] 通過 `validate_config_keys.py` 驗證
- [ ] 關鍵參數已填寫（`nu`, `Re_tau`, 等）
- [ ] 路徑正確且檔案存在（`data_path`, `cache_dir`）
- [ ] YAML 格式正確（無縮排錯誤）

---

## 🔗 相關文件

- **標準配置模板**: `configs/templates/standard_config_template.yml`
- **配置檢查器**: `scripts/tools/validate_config_keys.py`
- **配置載入器**: `pinnx/train/config_loader.py`
- **損失管理器**: `pinnx/train/loss_manager.py`
- **修復報告**: `context/config_key_fix_report.md`

---

## 📝 更新日誌

### v1.0 (2025-12-19)
- ✅ 建立標準配置模板
- ✅ 創建配置檢查器工具
- ✅ 修復 `losses`/`loss` 鍵名不一致問題
- ✅ 添加所有預設值註解
- ✅ 編寫配置管理指南

---

**問題回報**: 若發現配置相關問題，請在 `context/decisions_log.md` 中記錄。
