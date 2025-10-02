# 失敗案例目錄 - PINNs API不匹配問題詳細分析

**建立時間**: 2025-09-30 23:02  
**案例範圍**: Task-001 所有測試失敗案例  
**總失敗數**: 8個 (從63個測試中)

## 📋 失敗案例分類總覽

| 類別 | 數量 | 優先級 | 修復複雜度 | 影響範圍 |
|------|------|--------|------------|----------|
| API介面不匹配 | 4 | 🔴 高 | 中等 | 核心功能 |
| 參數類型錯誤 | 2 | 🔴 高 | 低 | 模型包裝器 |
| 方法名稱錯誤 | 1 | 🟡 中 | 低 | 權重策略 |
| 統計接口不完整 | 1 | 🟡 中 | 低 | 集成模型 |

## 🔍 詳細失敗案例分析

### 案例 #1: MultiHeadWrapper API介面錯誤
**測試**: `test_models.py::TestMultiHeadWrapper::test_multihead_wrapper_basic`  
**錯誤類型**: `AttributeError`  
**錯誤訊息**: `AttributeError: 'list' object has no attribute 'encode'`

#### 根本原因分析
```python
# 失敗位置: pinnx/models/wrappers.py:187
x_scaled = self.input_scaler.encode(x)

# 問題: input_scaler 被設為 list []，但程式期待有 encode() 方法的物件
# 追蹤: 測試中隱含傳入錯誤的參數類型
```

#### 調用堆疊
1. `test_multihead_wrapper_basic()` → 建立 wrapper
2. `wrapper(x)` → 呼叫 forward()
3. `self.input_scaler.encode(x)` → 失敗點

#### 最小重現案例
```python
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import MultiHeadWrapper

net = PINNNet(in_dim=2, out_dim=4, width=32, depth=2)
head_configs = [{"name": "velocity", "dim": 2, "activation": "tanh"}]

# 這會失敗 - input_scaler 應該是 None 或 Scaler 物件
wrapper = MultiHeadWrapper(net, head_configs, input_scaler=[])
```

#### 修復方案
```python
# 方案 1: 修改測試以傳入正確參數
wrapper = MultiHeadWrapper(net, head_configs, input_scaler=None, output_scaler=None)

# 方案 2: 在 ScaledPINNWrapper.__init__ 中加入參數驗證
if input_scaler is not None and not hasattr(input_scaler, 'encode'):
    raise TypeError(f"input_scaler must have 'encode' method, got {type(input_scaler)}")
```

---

### 案例 #2: EnsembleWrapper 統計接口不完整
**測試**: `test_models.py::TestEnsembleWrapper::test_ensemble_wrapper_basic`  
**錯誤類型**: `AssertionError`  
**錯誤訊息**: `AssertionError: assert 'min' in {'mean': ..., 'std': ..., 'uncertainty': ..., 'var': ...}`

#### 根本原因分析
```python
# 失敗位置: tests/test_models.py:239
assert 'min' in stats

# 問題: EnsemblePINNWrapper.forward() 在 mode='stats' 時只返回 
# ['mean', 'std', 'var', 'uncertainty']，但測試期待 ['mean', 'std', 'min', 'max']
```

#### 期待vs實際對比
```python
# 測試期待的統計鍵
expected = ['mean', 'std', 'min', 'max']

# 實際提供的統計鍵  
actual = ['mean', 'std', 'var', 'uncertainty']

# 缺失鍵
missing = ['min', 'max']
```

#### 修復方案
```python
# 檔案: pinnx/models/wrappers.py，第376行附近
elif mode == 'stats':
    mean = torch.einsum('m,mbo->bo', self.weights, stacked)
    var = torch.var(stacked, dim=0)
    std = torch.std(stacked, dim=0)
    min_vals = torch.min(stacked, dim=0)[0]  # 新增
    max_vals = torch.max(stacked, dim=0)[0]  # 新增
    
    return {
        'mean': mean,
        'var': var,
        'std': std,
        'uncertainty': std,
        'min': min_vals,     # 新增
        'max': max_vals      # 新增
    }
```

---

### 案例 #3: CausalWeighter 方法名稱錯誤
**測試**: `test_losses.py::TestWeightingStrategies::test_causal_weighting`  
**錯誤類型**: `AttributeError`  
**錯誤訊息**: `'CausalWeighter' object has no attribute 'compute_temporal_weights'. Did you mean: 'compute_causal_weights'?`

#### 根本原因分析
```python
# 失敗位置: tests/test_losses.py:360
weights = causal.compute_temporal_weights(time_losses)

# 問題: 方法名稱不匹配
# 測試期待: compute_temporal_weights()
# 實際實作: compute_causal_weights()
```

#### API不一致追蹤
```python
# 實際可用方法
>>> causal = CausalWeighter()
>>> [m for m in dir(causal) if not m.startswith('_')]
['accumulated_errors', 'apply_temporal_decay', 'causality_strength', 
 'compute_causal_weights', 'decay_rate', 'time_window_size']

# 測試期待的方法
compute_temporal_weights  # ❌ 不存在
```

#### 修復方案
```python
# 方案 1: 添加方法別名 (推薦，向後相容)
class CausalWeighter:
    def compute_temporal_weights(self, *args, **kwargs):
        """向後相容性別名"""
        return self.compute_causal_weights(*args, **kwargs)

# 方案 2: 修改測試使用正確方法名 (破壞性變更)
weights = causal.compute_causal_weights(time_losses)
```

---

### 案例 #4: MultiObjectiveWeighting 初始化錯誤
**測試**: `test_losses.py::TestWeightingStrategies::test_multiobjective_weighting`  
**錯誤類型**: `NameError`  
**錯誤訊息**: `NameError: name 'model' is not defined`

#### 根本原因分析
```python
# 失敗位置: pinnx/losses/weighting.py:662
self.weighters['gradnorm'] = GradNormWeighter(model, loss_names)

# 問題: 在 MultiObjectiveWeighting.__init__ 中，變數 'model' 未定義
# 原因: 參數處理邏輯錯誤，測試模式下 model=None 但仍嘗試使用
```

#### 變數作用域分析
```python
class MultiWeightManager:
    def __init__(self, objectives_or_model=None, loss_names=None, ...):
        if isinstance(objectives_or_model, list):
            # 測試模式
            self.model = None  # ← model 被設為 None
        else:
            # 正常模式  
            self.model = objectives_or_model
        
        # 但這裡仍使用 'model' 變數 (未定義)
        if 'gradnorm' in strategies:
            self.weighters['gradnorm'] = GradNormWeighter(model, loss_names)  # ❌
```

#### 修復方案
```python
# 修復: 使用 self.model 而非 model
if 'gradnorm' in strategies:
    if self.model is not None:  # 檢查 model 可用性
        self.weighters['gradnorm'] = GradNormWeighter(self.model, self.loss_names)
    else:
        # 測試模式或無模型情況的處理
        self.weighters['gradnorm'] = None
```

---

### 案例 #5-8: 其他 ScaledPINNWrapper 參數錯誤

#### 共同模式分析
所有剩餘的失敗都遵循相同模式：
1. 測試傳入不正確的 scaler 參數類型
2. `ScaledPINNWrapper.forward()` 嘗試呼叫 `scaler.encode()`
3. 因為 scaler 是 list/dict/str 而失敗

#### 影響的測試
- `TestMultiHeadWrapper::test_multihead_wrapper_gradients`
- `TestResidualWrapper::test_residual_wrapper_basic`  
- `TestResidualWrapper::test_residual_vs_base`
- `test_models_integration`

#### 統一修復策略
1. **參數驗證強化** (最佳方案)
2. **測試檔案修正** (快速方案)
3. **預設值改善** (輔助方案)

## 🎯 修復優先級排序

### 第一波 (立即修復，0-30分鐘)
1. **EnsembleWrapper統計接口** - 一行程式碼修復，無風險
2. **CausalWeighter方法別名** - 三行程式碼，無風險

### 第二波 (短期修復，30-90分鐘)  
3. **MultiObjectiveWeighting NameError** - 變數作用域修正
4. **參數驗證強化** - 預防類似問題

### 第三波 (清理，90-120分鐘)
5. **測試檔案標準化** - 確保所有測試使用正確API
6. **API文檔同步** - 預防未來不一致

## 📊 影響評估矩陣

| 案例 | 使用者影響 | 開發影響 | 修復成本 | 回歸風險 |
|------|------------|----------|----------|----------|
| EnsembleWrapper統計 | 低 | 高 | 極低 | 極低 |
| CausalWeighter方法 | 低 | 中 | 極低 | 極低 |
| MultiObj NameError | 中 | 高 | 低 | 低 |
| 參數驗證 | 高 | 中 | 低 | 中 |
| ScaledWrapper API | 高 | 高 | 中 | 中 |

## 🔄 後續預防措施

### 1. API合約測試
```python
# 建立合約測試確保API穩定性
def test_wrapper_api_contract():
    """確保wrapper API符合契約"""
    assert hasattr(ScaledPINNWrapper, '__init__')
    assert hasattr(ScaledPINNWrapper, 'forward')
    # ... 更多契約檢查
```

### 2. 參數類型檢查
```python
# 在關鍵類別中加入runtime類型檢查
from typing import Union, Optional
def __init__(self, scaler: Optional[Union[VSScaler, StandardScaler, MinMaxScaler]]):
    # 編譯時和運行時類型檢查
```

### 3. 持續監控
```bash
# 每次提交後自動執行API相容性檢查
git hook: pytest tests/test_api_compatibility.py
```

---
**建立者**: Debugger sub-agent  
**最後更新**: 2025-09-30 23:02  
**案例狀態**: 分析完成，等待修復