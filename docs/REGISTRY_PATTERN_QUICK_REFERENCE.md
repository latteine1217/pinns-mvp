# Schema Validation Quick Reference

> **快速參考卡**：5 分鐘學會 Schema Validation

---

## 🚀 快速開始

### 註冊帶 Schema 的工廠函數

```python
from pinnx.train.model_physics_factory import _model_factory

@_model_factory.register(
    'my_model',
    required_fields=['in_dim', 'out_dim'],              # 必要欄位
    field_types={'in_dim': int, 'out_dim': int},        # 型別約束
    validators={'in_dim': lambda x: x > 0}               # 自訂驗證
)
def create_my_model(config, device):
    return MyModel(**config)
```

---

## 📚 Schema 參數速查

| 參數 | 用途 | 範例 |
|------|------|------|
| `required_fields` | 必要欄位 | `['in_dim', 'out_dim']` |
| `optional_fields` | 預設值 | `{'activation': 'tanh'}` |
| `field_types` | 型別檢查 | `{'nu': (int, float)}` |
| `validators` | 自訂邏輯 | `{'depth': lambda x: x >= 2}` |

---

## 🎯 常見使用場景

### 1️⃣ 基本驗證（必要欄位 + 型別）

```python
@_model_factory.register(
    'simple_model',
    required_fields=['x', 'y'],
    field_types={'x': int, 'y': int}
)
def create_simple(config, device):
    return config['x'] + config['y']

# ✅ create_simple({'x': 1, 'y': 2}, device)  → 3
# ❌ create_simple({'x': 1}, device)          → ValueError: Missing 'y'
# ❌ create_simple({'x': '1', 'y': 2}, device) → ValueError: Wrong type
```

---

### 2️⃣ Union Types（接受多種型別）

```python
@_physics_factory.register(
    'ns_2d',
    field_types={
        'nu': (int, float),   # ✅ 可以是 int 或 float
        'rho': (int, float)
    }
)
def create_ns_2d(config):
    return NSEquations2D(**config)

# ✅ create_ns_2d({'nu': 0.001, 'rho': 1.0})  # float
# ✅ create_ns_2d({'nu': 1, 'rho': 1})        # int
# ❌ create_ns_2d({'nu': '0.001'})            # str → ValueError
```

---

### 3️⃣ 自訂驗證邏輯（Validators）

```python
@_model_factory.register(
    'resnet',
    field_types={'depth': int, 'width': int},
    validators={
        'depth': lambda x: x >= 2,              # 至少 2 層
        'width': lambda x: x % 16 == 0          # 必須是 16 的倍數
    }
)
def create_resnet(config, device):
    return ResNet(**config)

# ✅ create_resnet({'depth': 18, 'width': 256}, device)
# ❌ create_resnet({'depth': 1, 'width': 256}, device)   # depth < 2
# ❌ create_resnet({'depth': 18, 'width': 100}, device)  # width 不是 16 的倍數
```

---

### 4️⃣ 預設值（Optional Fields）

```python
@_model_factory.register(
    'flexible_model',
    required_fields=['in_dim', 'out_dim'],
    optional_fields={
        'activation': 'tanh',      # 預設 'tanh'
        'dropout': 0.0             # 預設 0.0
    }
)
def create_flexible(config, device):
    return FlexibleModel(**config)

# 最小配置（自動填充預設值）
config = {'in_dim': 3, 'out_dim': 4}
validated = _model_factory.validate_config('flexible_model', config)
# → {'in_dim': 3, 'out_dim': 4, 'activation': 'tanh', 'dropout': 0.0}
```

---

## 🔍 錯誤訊息解讀

### ❌ `Missing required fields: ['out_dim']`
**原因**: 配置缺少必要欄位  
**解法**: 添加缺少的欄位到配置

### ❌ `Field 'x' has wrong type. Expected int, got str`
**原因**: 型別不匹配  
**解法**: 檢查配置值的型別（常見於 YAML 解析錯誤）

### ❌ `Field 'x' failed validation`
**原因**: Validator 返回 False  
**解法**: 檢查 validator 的邏輯約束（例如 `x > 0`）

---

## 💡 最佳實踐

### ✅ DO

```python
# 1. 用 field_types 檢查型別
field_types={'nu': (int, float)}

# 2. 用 validators 檢查邏輯
validators={'nu': lambda x: x > 0}

# 3. 使用清晰的驗證條件
validators={'activation': lambda x: x in ['tanh', 'relu']}
```

### ❌ DON'T

```python
# ❌ 不要在 validator 中檢查型別（用 field_types）
validators={'nu': lambda x: isinstance(x, (int, float)) and x > 0}

# ❌ 不要過度複雜的 validator
validators={'x': lambda x: (x > 0 and x < 100) or (x > 200 and x < 300)}

# ❌ 不要忽略 Union Types（當需要多型別時）
field_types={'nu': float}  # ← 拒絕 int
# 應該用:
field_types={'nu': (int, float)}
```

---

## 🧪 測試你的 Schema

```python
# 直接測試驗證邏輯
from pinnx.train.model_physics_factory import _model_factory

# 測試正確配置
config_good = {'in_dim': 3, 'out_dim': 4, 'width': 256, 'depth': 8, 'activation': 'tanh'}
validated = _model_factory.validate_config('fourier_vs_mlp', config_good)
print("✅ 驗證通過")

# 測試錯誤配置
config_bad = {'in_dim': 'invalid', 'out_dim': 4}
try:
    _model_factory.validate_config('fourier_vs_mlp', config_bad)
except ValueError as e:
    print(f"❌ 預期的錯誤: {e}")
```

---

## 📖 完整文件

- **詳細技術指南**: `SCHEMA_VALIDATION_TECHNICAL_GUIDE.md`
- **實作報告**: `SESSION_SUMMARY_2026-01-03_Phase3_Schema_Validation.md`
- **Registry Pattern**: `REGISTRY_PATTERN_PHASE2_COMPLETE.md`

---

## 🆘 常見問題

**Q: Schema 驗證會影響效能嗎？**  
A: 不會。每次驗證 < 0.01ms，可忽略不計。

**Q: 可以跳過驗證嗎？**  
A: 不建議。驗證是為了早期發現配置錯誤，避免訓練到一半才發現問題。

**Q: 如何驗證巢狀結構？**  
A: 使用 `field_types={'config': dict}` + validator 檢查內部鍵值。

**Q: 可以動態生成 Schema 嗎？**  
A: 可以。在註冊前根據條件創建 `ConfigSchema` 物件。

---

**最後更新**: 2026-01-03  
**版本**: 1.0.0
