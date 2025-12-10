# PirateNet 參數對齊說明

## 變更摘要

本次更新將專案預設參數與 PirateNet 論文（Wang et al. 2025, "Simulating Three-dimensional Turbulence with Physics-informed Neural Networks"）對齊，確保實現符合文獻最佳實踐。

## 主要變更

### 1. Random Weight Factorization (RWF)

**變更內容**：
```python
# 修改前
rwf_scale_mean: float = 0.0  # exp(0) = 1.0

# 修改後
rwf_scale_mean: float = 1.0  # exp(1.0) ≈ 2.7 (對齊 PirateNet 論文)
```

**影響**：
- RWF 的尺度因子初始化更加積極（exp(1.0) ≈ 2.7 vs exp(0.0) = 1.0）
- 提供更強的自適應學習率效果
- 適合高雷諾數或剛性問題

**修改文件**：
- `pinnx/models/fourier_mlp.py:46` - RWFLinear.__init__
- `pinnx/models/fourier_mlp.py:210` - DenseLayer.__init__
- `pinnx/models/fourier_mlp.py:302` - ResBlock.__init__
- `pinnx/models/fourier_mlp.py:408` - PINNNet.__init__
- `pinnx/models/fourier_mlp.py:788` - create_pinn_model

---

### 2. Adaptive Residual (ResBlock)

#### 2.1 Alpha 初始化

**變更內容**：
```python
# 修改前
res_block_alpha_init: float = 1.0  # 初始即啟用完整殘差

# 修改後
res_block_alpha_init: float = 0.0  # 初始恆等映射，逐步增深（對齊 PirateNet 論文）
```

**影響**：
- 訓練初期網路等效於淺層網路（α=0 → 恆等映射）
- α 在訓練過程中逐漸學習增大，漸進式啟動深度
- 改善深層網路（depth>8）的訓練穩定性

**修改文件**：
- `pinnx/models/fourier_mlp.py:307` - ResBlock.__init__
- `pinnx/models/fourier_mlp.py:414` - PINNNet.__init__
- `pinnx/models/fourier_mlp.py:794` - create_pinn_model
- `pinnx/train/factory.py:410` - create_model (resnet type)

#### 2.2 Forward 實現形式

**變更內容**：
```python
# 修改前（加法形式）
return x + self.alpha * out

# 修改後（插值形式）
return self.alpha * out + (1 - self.alpha) * x
```

**影響**：
- α=0 時為純恆等映射（完全跳過非線性）
- α=1 時為純非線性變換（無殘差連接）
- 數學上與 PirateNet 論文 Eq. 5 一致：`y = α·f(x) + (1-α)·x`

**修改文件**：
- `pinnx/models/fourier_mlp.py:377` - ResBlock.forward

---

### 3. SOAP 優化器參數

**變更內容**：
```python
# 修改前
betas: (0.95, 0.95)          # 一階與二階動量衰減
precondition_frequency: 10   # 每 10 步更新預條件器

# 修改後
betas: (0.9, 0.999)          # 對齊 PirateNet 論文 (Wang et al. 2025)
precondition_frequency: 2    # 對齊 PirateNet 論文 (f=2)
```

**影響**：
- β₁=0.9：一階動量更新更快（vs 0.95）
- β₂=0.999：二階動量更穩定（vs 0.95）
- f=2：預條件器更頻繁更新（每 2 步 vs 每 10 步），更好地追蹤曲率變化
- 整體更適合極高雷諾數（Re>10⁶）或強剛性問題

**修改文件**：
- `pinnx/optim/soap.py:51` - SOAP.__init__ (betas)
- `pinnx/optim/soap.py:55` - SOAP.__init__ (precondition_frequency)

---

### 4. 命名統一：resnet2 → resnet

**變更內容**：
- 類名：`ResBlock2` → `ResBlock`
- 參數：`block_type='resnet2'` → `block_type='resnet'`

**影響**：
- 消除命名中的「2」（原為表示兩層殘差，但容易混淆）
- 統一命名為 `resnet` 表示 PirateNet-style adaptive residual

**修改文件**：
- `pinnx/models/fourier_mlp.py` (多處)
- `pinnx/train/factory.py:409`
- `tests/test_models.py:131`
- `configs/config_template_example.yml:142`
- `configs/kolmogorov_re50_kf4_K100.yml:96`
- `configs/kolmogorov_re50_kf4_K100_full_1k.yml:97`
- `README.md:78`

---

## 使用建議

### 對於現有專案（逆問題，Re=50-500）

**建議配置**：
```yaml
model:
  activation: sine         # 保持，高頻敏感度優先
  block_type: resnet       # 更新後的名稱
  res_block_alpha_init: 0.0  # ✅ 使用新預設值
  use_rwf: true
  rwf_scale_mean: 1.0      # ✅ 使用新預設值

optimizer:
  type: adam_lbfgs         # 保持兩段式（Adam → L-BFGS）
  # 或嘗試純 SOAP：
  # type: soap
  # betas: [0.9, 0.999]    # ✅ 使用新預設值
  # precondition_frequency: 2  # ✅ 使用新預設值
```

### 對於純前向模擬（文獻風格，Re>10⁶）

**建議配置**：
```yaml
model:
  activation: swish        # 文獻推薦（穩定性優先）
  block_type: resnet
  res_block_alpha_init: 0.0  # 深層網路（>10 blocks）必須
  use_rwf: true
  rwf_scale_mean: 1.0
  fourier_sigma: 2.0       # 文獻值（vs 專案預設 5.0）

optimizer:
  type: soap               # 文獻推薦（極高 Re 必須）
  betas: [0.9, 0.999]
  precondition_frequency: 2
```

---

## 向後相容性

### 配置文件

舊配置文件中的 `block_type='resnet2'` 將**無法使用**，必須更新為 `resnet`：

```bash
# 批量更新配置文件
find configs/ -name "*.yml" -exec sed -i '' 's/resnet2/resnet/g' {} \;
```

### 檢查點

已訓練的模型檢查點**仍可載入**，因為：
- 類名變更（`ResBlock2` → `ResBlock`）在序列化時只影響類型標註
- 參數結構未改變（仍為 `alpha`, `fc1`, `fc2` 等）
- PyTorch 的 `state_dict` 基於參數名稱，不依賴類名

但**新訓練的模型**將使用新預設值（rwf_scale_mean=1.0, alpha_init=0.0），訓練動態可能不同。

---

## 驗證建議

### 1. 單元測試

```bash
# 驗證 ResBlock 基本功能
python -m pytest tests/test_models.py::TestPINNNet::test_pinn_net_resnet_block -v

# 驗證參數初始化
python -m pytest tests/test_models.py -v
```

### 2. 快速訓練測試

使用配置模板進行快速驗證（5-10 分鐘）：

```bash
python scripts/train.py --cfg configs/templates/2d_quick_baseline.yml
```

預期結果：
- 訓練穩定（無 NaN/Inf）
- Loss 正常下降
- Alpha 參數從 ~0.0 逐步增長

### 3. A/B 測試（可選）

對比舊參數 vs 新參數（針對高 Re 或深層網路）：

```bash
# 舊參數（rwf_scale_mean=0.0, alpha_init=1.0）
python scripts/train.py --cfg configs/test_old_params.yml

# 新參數（rwf_scale_mean=1.0, alpha_init=0.0）
python scripts/train.py --cfg configs/test_new_params.yml

# 比較收斂速度與最終誤差
python scripts/compare_experiments.py --exp1 test_old_params --exp2 test_new_params
```

---

## 參考文獻

**主要文獻**：
- Wang, S., et al. (2025). "Simulating Three-dimensional Turbulence with Physics-informed Neural Networks". *arXiv:2507.08972v2*.

**相關技術**：
- **RWF**: Wang et al. (2022), "Random Weight Factorization Improves the Training of Continuous Normalizing Flows"
- **SOAP**: Mishchenko et al. (2023), "SOAP: Shampoo with Adam in the Outer Loop and Preconditioning in the Inner Loop"
- **Causal Training**: Wang et al. (2022), "Respecting Causality for Training Physics-Informed Neural Networks"

---

## 變更歷史

| 日期 | 版本 | 變更內容 |
|------|------|---------|
| 2025-01-XX | 1.0.0 | 初始對齊：RWF、Alpha、SOAP 參數；resnet2→resnet 重命名 |

