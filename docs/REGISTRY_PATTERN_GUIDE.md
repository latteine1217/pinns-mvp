# Schema Validation 技術指南

**Author**: Registry Pattern Implementation Team  
**Date**: 2026-01-03  
**Version**: 1.0.0

---

## 📚 目錄

1. [概述](#概述)
2. [核心概念](#核心概念)
3. [API 參考](#api-參考)
4. [使用範例](#使用範例)
5. [最佳實踐](#最佳實踐)
6. [故障排除](#故障排除)

---

## 概述

Schema Validation 是 Registry Pattern 的擴展，提供宣告式配置驗證。它允許你在註冊工廠函數時定義：

- ✅ **必要欄位** - 哪些參數是必須的
- ✅ **型別約束** - 參數應該是什麼型別（支援 Union Types）
- ✅ **自訂驗證** - 任意的驗證邏輯（Lambda 函數）
- ✅ **預設值** - 可選參數的預設值

### 設計原則

1. **宣告式優於命令式** - 用資料結構描述驗證規則，而非 if-else
2. **早期失敗** - 在工廠函數執行前就發現配置錯誤
3. **Zero Branches** - 驗證邏輯透過字典查找，不增加條件分支

---

## 核心概念

### 1. ConfigSchema 類別

```python
class ConfigSchema:
    """配置 Schema 定義（用於驗證必要欄位）"""
    
    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[Dict[str, Any]] = None,
        field_types: Optional[Dict[str, Union[type, Tuple[type, ...]]]] = None,
        validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ):
        ...
```

#### 參數說明

| 參數 | 型別 | 描述 | 範例 |
|------|------|------|------|
| `required_fields` | `List[str]` | 必要欄位列表 | `['in_dim', 'out_dim']` |
| `optional_fields` | `Dict[str, Any]` | 可選欄位及預設值 | `{'activation': 'tanh'}` |
| `field_types` | `Dict[str, Union[type, Tuple[type, ...]]]` | 型別約束 | `{'nu': (int, float)}` |
| `validators` | `Dict[str, Callable]` | 自訂驗證函數 | `{'depth': lambda x: x >= 2}` |

### 2. Registry.register() 裝飾器

```python
@registry.register(
    type_name='my_type',
    # 選項 1: 完整 schema
    schema=ConfigSchema(...),
    
    # 選項 2: 快捷參數（內部轉為 ConfigSchema）
    required_fields=['x', 'y'],
    optional_fields={'z': 0},
    field_types={'x': int, 'y': (int, float)},
    validators={'x': lambda x: x > 0}
)
def create_my_type(config, device):
    ...
```

### 3. 驗證流程

```
配置輸入
    ↓
[1. 檢查必要欄位]
    ↓ missing fields? → ValueError
[2. 檢查型別約束]
    ↓ type mismatch? → ValueError
[3. 執行自訂驗證]
    ↓ validation failed? → ValueError
[4. 填充預設值]
    ↓
標準化配置輸出
```

---

## API 參考

### ConfigSchema.validate()

**簽名**:
```python
def validate(self, config: Dict[str, Any], context: str = "") -> None
```

**參數**:
- `config`: 待驗證的配置字典
- `context`: 上下文訊息（用於錯誤報告）

**拋出**:
- `ValueError`: 若配置不符合 schema

**範例**:
```python
schema = ConfigSchema(
    required_fields=['x'],
    field_types={'x': int},
    validators={'x': lambda x: x > 0}
)

# ✅ 正確
schema.validate({'x': 10})

# ❌ 缺少欄位
schema.validate({})  # ValueError: Missing required fields: ['x']

# ❌ 型別錯誤
schema.validate({'x': '10'})  # ValueError: Expected int, got str

# ❌ 驗證失敗
schema.validate({'x': -1})  # ValueError: Field 'x' failed validation
```

### ConfigSchema.apply_defaults()

**簽名**:
```python
def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]
```

**功能**: 為缺少的可選欄位填充預設值

**範例**:
```python
schema = ConfigSchema(
    optional_fields={'activation': 'tanh', 'dropout': 0.0}
)

config = {'activation': 'relu'}
result = schema.apply_defaults(config)
# 結果: {'activation': 'relu', 'dropout': 0.0}
```

### Registry.validate_config()

**簽名**:
```python
def validate_config(self, type_name: str, config: Dict[str, Any]) -> Dict[str, Any]
```

**功能**: 驗證並標準化配置

**回傳**: 標準化後的配置（已填充預設值）

**範例**:
```python
from pinnx.train.model_physics_factory import _model_factory

config = {'in_dim': 3, 'out_dim': 4, 'width': 256, 'depth': 8, 'activation': 'tanh'}
validated = _model_factory.validate_config('fourier_vs_mlp', config)
```

---

## 使用範例

### 範例 1: 基本驗證

```python
from pinnx.train.model_physics_factory import _Registry

registry = _Registry("MyRegistry")

@registry.register(
    'basic_type',
    required_fields=['x', 'y'],
    field_types={'x': int, 'y': int}
)
def create_basic(config):
    return config['x'] + config['y']

# 使用
try:
    validated = registry.validate_config('basic_type', {'x': 1, 'y': 2})
    result = create_basic(validated)  # 3
except ValueError as e:
    print(f"Validation failed: {e}")
```

### 範例 2: Union Types（物理參數）

```python
@_physics_factory.register(
    'ns_2d',
    field_types={
        'nu': (int, float),   # ✅ 接受 int 或 float
        'rho': (int, float)
    },
    validators={
        'nu': lambda x: x > 0,   # 必須為正數
        'rho': lambda x: x > 0
    }
)
def create_ns_2d(config):
    return NSEquations2D(nu=config['nu'], rho=config['rho'])

# ✅ 正確使用
create_ns_2d({'nu': 0.001, 'rho': 1.0})  # float
create_ns_2d({'nu': 1, 'rho': 1})        # int

# ❌ 錯誤使用
create_ns_2d({'nu': '0.001'})  # ValueError: Expected int or float, got str
create_ns_2d({'nu': -0.001})   # ValueError: Field 'nu' failed validation
```

### 範例 3: 複雜驗證邏輯

```python
@_model_factory.register(
    'resnet',
    required_fields=['depth', 'width'],
    field_types={'depth': int, 'width': int},
    validators={
        'depth': lambda x: x >= 2,              # ResNet 至少 2 層
        'width': lambda x: x % 16 == 0 and x > 0  # 寬度必須是 16 的倍數
    }
)
def create_resnet(config, device):
    return ResNet(depth=config['depth'], width=config['width'])

# ✅ 正確
create_resnet({'depth': 18, 'width': 256}, device)

# ❌ depth < 2
create_resnet({'depth': 1, 'width': 256}, device)  # ValueError

# ❌ width 不是 16 的倍數
create_resnet({'depth': 18, 'width': 100}, device)  # ValueError
```

### 範例 4: 預設值與可選參數

```python
@_model_factory.register(
    'flexible_model',
    required_fields=['in_dim', 'out_dim'],
    optional_fields={
        'activation': 'tanh',
        'dropout': 0.0,
        'use_batch_norm': False
    },
    field_types={
        'in_dim': int,
        'out_dim': int,
        'dropout': (int, float)
    }
)
def create_flexible_model(config, device):
    # config 已包含所有預設值
    return FlexibleModel(**config)

# 最小配置（使用預設值）
config = {'in_dim': 3, 'out_dim': 4}
validated = _model_factory.validate_config('flexible_model', config)
# 結果: {'in_dim': 3, 'out_dim': 4, 'activation': 'tanh', 'dropout': 0.0, 'use_batch_norm': False}
```

### 範例 5: 巢狀結構驗證

```python
@_physics_factory.register(
    'vs_pinn_channel_flow',
    required_fields=['domain'],
    field_types={
        'nu': (int, float),
        'domain': dict,      # 巢狀字典
        'vs_pinn': dict
    },
    validators={
        'nu': lambda x: x > 0,
        'domain': lambda d: 'x' in d and 'y' in d and 'z' in d  # 確保有 x, y, z
    }
)
def create_vs_pinn(config):
    return VSPINNChannelFlow(**config)

# ✅ 正確
config = {
    'nu': 1e-5,
    'domain': {'x': [0, 6.28], 'y': [0, 2], 'z': [0, 3.14]},
    'vs_pinn': {'scaling_factors': {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}}
}
create_vs_pinn(config)

# ❌ domain 缺少 'z'
config_bad = {
    'nu': 1e-5,
    'domain': {'x': [0, 6.28], 'y': [0, 2]}
}
create_vs_pinn(config_bad)  # ValueError: Field 'domain' failed validation
```

---

## 最佳實踐

### 1. 選擇適當的驗證層級

```python
# ❌ 過度驗證（不必要）
validators={
    'activation': lambda x: x == 'tanh' or x == 'relu' or x == 'sine'
}

# ✅ 簡潔的驗證
validators={
    'activation': lambda x: x in ['tanh', 'relu', 'sine']
}
```

### 2. 使用 Union Types 而非多個 validator

```python
# ❌ 用 validator 檢查型別（不推薦）
validators={
    'nu': lambda x: isinstance(x, (int, float)) and x > 0
}

# ✅ 用 field_types 檢查型別
field_types={'nu': (int, float)},
validators={'nu': lambda x: x > 0}
```

### 3. 提供清晰的錯誤訊息

```python
# ❌ 不清楚的驗證
validators={
    'depth': lambda x: x >= 2
}

# ✅ 帶註解的驗證（在 docstring 說明）
"""
Args:
    depth: 網路深度，ResNet 至少需要 2 層
"""
validators={
    'depth': lambda x: x >= 2  # ResNet minimum depth
}
```

### 4. 分離物理約束與工程約束

```python
# 物理約束（必須滿足）
validators={
    'nu': lambda x: x > 0,      # 黏度必須為正
    'rho': lambda x: x > 0       # 密度必須為正
}

# 工程約束（建議但非必須，可放在 warning）
optional_fields={
    'reynolds': 1000  # 典型值，但可以自訂
}
```

---

## 故障排除

### 問題 1: TypeError: int() argument must be a string...

**原因**: `field_types` 定義中使用了 `None` 作為型別

```python
# ❌ 錯誤
field_types={'nu': (int, float, None)}

# ✅ 正確（使用 Optional）
from typing import Optional
field_types={'nu': (int, float)}  # 不需要 None，缺少欄位由 required_fields 控制
```

### 問題 2: ValueError: Field 'x' has wrong type. Expected int, got str

**原因**: 配置中的值是字串而非數字

```python
# ❌ YAML 配置未正確解析
config = {'x': '10'}  # 字串

# ✅ 確保 YAML loader 正確解析型別
config = {'x': 10}    # 整數
```

**解決方案**: 檢查配置來源（YAML/JSON）的型別轉換

### 問題 3: Type hint error: Dict[str, type] is not assignable to Dict[str, Union[type, Tuple]]

**原因**: Python 的 `Dict` 型別是不變的（invariant）

```python
# ❌ 不匹配
field_types: Dict[str, type] = {'x': int}
# 傳給 field_types: Dict[str, Union[type, Tuple[type, ...]]]

# ✅ 解決方案 1: 使用 Union
field_types: Dict[str, Union[type, Tuple[type, ...]]] = {'x': int}

# ✅ 解決方案 2: 使用 Mapping（協變）
from typing import Mapping
field_types: Mapping[str, Union[type, Tuple[type, ...]]] = {'x': int}
```

### 問題 4: Validator 拋出 Exception 而非返回 bool

**原因**: Validator 中的錯誤未被捕捉

```python
# ❌ 可能拋出 AttributeError
validators={
    'domain': lambda d: d['x'] is not None  # 若 'x' 不存在會拋錯
}

# ✅ 防禦性檢查
validators={
    'domain': lambda d: 'x' in d and d['x'] is not None
}
```

### 問題 5: 錯誤訊息不清楚

**解決方案**: 使用 `context` 參數

```python
# Registry 內部自動添加 context
schema.validate(config, context=f"{self.name}[{type_name}]")
# 錯誤訊息: "ModelFactory[fourier_vs_mlp]: Missing required fields: ['width']"
```

---

## 進階主題

### 1. 自訂 Schema 類別

```python
class PhysicsSchema(ConfigSchema):
    """物理模組專用的 Schema（額外檢查量綱一致性）"""
    
    def validate(self, config, context=""):
        # 先執行基本驗證
        super().validate(config, context)
        
        # 額外檢查量綱
        if 'nu' in config and 'L' in config and 'U' in config:
            Re = config['L'] * config['U'] / config['nu']
            if Re < 1:
                logging.warning(f"Reynolds number too low: Re={Re}")
```

### 2. 動態 Schema（根據配置生成 Schema）

```python
def create_adaptive_schema(model_type: str) -> ConfigSchema:
    """根據模型類型動態生成 schema"""
    if model_type == 'resnet':
        return ConfigSchema(
            required_fields=['depth', 'width'],
            validators={'depth': lambda x: x >= 2}
        )
    elif model_type == 'mlp':
        return ConfigSchema(
            required_fields=['hidden_dim', 'num_layers']
        )
    else:
        return ConfigSchema()
```

### 3. Schema 組合

```python
# 基礎 Schema
base_schema = ConfigSchema(
    required_fields=['in_dim', 'out_dim'],
    field_types={'in_dim': int, 'out_dim': int}
)

# 擴展 Schema（手動合併）
advanced_schema = ConfigSchema(
    required_fields=base_schema.required_fields + ['activation'],
    field_types={**base_schema.field_types, 'activation': str}
)
```

---

## 效能考量

### Schema 驗證的開銷

```python
import time

# 測試 10000 次驗證的時間
configs = [{'in_dim': 3, 'out_dim': 4, 'width': 256, 'depth': 8} for _ in range(10000)]

start = time.time()
for config in configs:
    validated = _model_factory.validate_config('fourier_vs_mlp', config)
elapsed = time.time() - start

print(f"10000 次驗證耗時: {elapsed:.4f} 秒")
print(f"平均每次: {elapsed/10000*1000:.4f} 毫秒")
```

**典型結果**:
- 10000 次驗證: ~0.05 秒
- 平均每次: ~0.005 毫秒

**結論**: Schema 驗證開銷可忽略（< 0.01ms），遠小於模型建立時間（通常 > 100ms）

---

## 參考資料

### 相關文件
- **Registry Pattern Phase 1+2**: `REGISTRY_PATTERN_PHASE2_COMPLETE.md`
- **Phase 3 完成報告**: `SESSION_SUMMARY_2026-01-03_Phase3_Schema_Validation.md`

### Python Type Hints
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [PEP 604 - Union Types](https://peps.python.org/pep-0604/)
- [typing.Union](https://docs.python.org/3/library/typing.html#typing.Union)

### 設計模式
- [Registry Pattern](https://en.wikipedia.org/wiki/Service_locator_pattern)
- [Declarative Programming](https://en.wikipedia.org/wiki/Declarative_programming)

---

**最後更新**: 2026-01-03  
**維護者**: Registry Pattern Implementation Team
