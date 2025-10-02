# Debug Playbook - PINNs 測試失敗根本原因分析與修復策略

**分析時間**: 2025-09-30 23:02  
**分析對象**: Task-001 API不匹配問題  
**測試通過率**: 81% (51/63)，12個失敗案例  
**分析者**: Debugger sub-agent

## 📊 背景
基於對當前專案的全面分析，識別出主要的API不匹配問題影響了測試通過率。本手冊提供系統性的根本原因分析、修復策略和預防措施。

## 🔍 方法/依據
1. **靜態程式碼分析**: 檢查模組間接口一致性
2. **動態執行分析**: 透過最小可重現例子分析運行時錯誤
3. **測試失敗分類**: 依據錯誤類型和影響範圍分類
4. **相依性追蹤**: 分析模組間相依關係和潛在衝突

## 🎯 關鍵結論

### 失敗案例分類與優先級

#### 🔴 高優先級 - API介面不匹配 (8個失敗)
**影響範圍**: 核心功能阻斷，直接影響模型使用

1. **模型包裝器類別混淆** (4個失敗)
   - 測試期待: `MultiHeadWrapper`
   - 實際實作: `MultiHeadWrapper` 存在但行為與 `ScaledPINNWrapper` 不同
   - 根本原因: 測試中使用別名但實際類別行為不一致

2. **輸入參數類型錯誤** (3個失敗)
   - 錯誤訊息: `AttributeError: 'list' object has no attribute 'encode'`
   - 根本原因: `ScaledPINNWrapper` 期待 Scaler 物件，但測試傳入 list/dict
   
3. **EnsembleWrapper統計介面不完整** (1個失敗)
   - 測試期待: `stats` 字典包含 `['mean', 'std', 'min', 'max']`
   - 實際提供: `['mean', 'std', 'var', 'uncertainty']`

#### 🟡 中優先級 - 方法名稱不匹配 (2個失敗)
4. **CausalWeighter方法名錯誤** (1個失敗)
   - 測試呼叫: `compute_temporal_weights()`
   - 實際方法: `compute_causal_weights()`

5. **MultiObjectiveWeighting初始化錯誤** (1個失敗)
   - 錯誤: `NameError: name 'model' is not defined`
   - 根本原因: 變數作用域問題

#### 🟢 低優先級 - 資料不一致 (2個失敗)
**影響**: 功能可用但結果格式不符期待

## 🛠️ 具體修復策略

### Phase 1: 核心API修復 (預估2小時)

#### 1.1 修復 MultiHeadWrapper API不匹配
```python
# 問題: wrapper(x) 期待字典輸出，但實際傳入錯誤的scaler類型
# 解決方案 1: 修正測試中的scaler參數
# 檔案: tests/test_models.py

# 原始程式碼 (問題)
wrapper = MultiHeadWrapper(base_net, head_configs)  # input_scaler=[] 隱含傳入

# 修復程式碼
wrapper = MultiHeadWrapper(base_net, head_configs, input_scaler=None, output_scaler=None)
```

#### 1.2 統一 EnsembleWrapper 統計介面
```python
# 檔案: pinnx/models/wrappers.py，第376-387行
# 在 mode='stats' 分支中添加 'min' 和 'max'

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

#### 1.3 修正方法名稱不匹配
```python
# 方案1: 在 CausalWeighter 添加方法別名
class CausalWeighter:
    # 現有方法保持不變...
    
    def compute_temporal_weights(self, *args, **kwargs):
        """向後相容性別名"""
        return self.compute_causal_weights(*args, **kwargs)
```

### Phase 2: 參數驗證與錯誤處理 (預估1小時)

#### 2.1 強化 ScaledPINNWrapper 參數驗證
```python
# 檔案: pinnx/models/wrappers.py，第140-149行
def __init__(self, base_model, input_scaler=None, output_scaler=None, variable_names=None):
    # 添加參數類型檢查
    if input_scaler is not None:
        if not hasattr(input_scaler, 'encode'):
            raise TypeError(f"input_scaler must have 'encode' method, got {type(input_scaler)}")
    
    if output_scaler is not None:
        if not hasattr(output_scaler, 'decode'):
            raise TypeError(f"output_scaler must have 'decode' method, got {type(output_scaler)}")
```

### Phase 3: 測試相容性修復 (預估30分鐘)

#### 3.1 更新測試檔案的期待值
```python
# 檔案: tests/test_models.py
# 修正測試參數和期待值，確保與實際實作一致
```

## 🧪 最小可重現例子 (MRE)

### MRE 1: 模型包裝器API問題
```python
# 檔案: debug_mre_wrapper.py
import torch
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ScaledPINNWrapper

def test_wrapper_api_issue():
    """重現wrapper API問題"""
    net = PINNNet(in_dim=2, out_dim=3)
    
    # 這會失敗: input_scaler 期待 Scaler 物件，不是 list
    try:
        wrapper = ScaledPINNWrapper(net, input_scaler=[], output_scaler={})
        x = torch.randn(5, 2)
        y = wrapper(x)  # AttributeError: 'list' object has no attribute 'encode'
        print("❌ 未捕獲到預期錯誤")
    except AttributeError as e:
        print(f"✅ 成功重現錯誤: {e}")
    
    # 正確的用法
    wrapper = ScaledPINNWrapper(net, input_scaler=None, output_scaler=None)
    x = torch.randn(5, 2)
    y = wrapper(x)
    print(f"✅ 正確用法成功，輸出形狀: {y.shape}")

if __name__ == "__main__":
    test_wrapper_api_issue()
```

### MRE 2: EnsembleWrapper 統計接口問題
```python
# 檔案: debug_mre_ensemble.py
import torch
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import EnsemblePINNWrapper

def test_ensemble_stats_issue():
    """重現ensemble統計接口問題"""
    models = [PINNNet(2, 3) for _ in range(3)]
    ensemble = EnsemblePINNWrapper(models)
    
    x = torch.randn(5, 2)
    stats = ensemble(x, mode='stats')
    
    print(f"現有統計鍵: {list(stats.keys())}")
    print(f"測試期待鍵: ['mean', 'std', 'min', 'max']")
    
    # 檢查缺失的鍵
    expected_keys = ['mean', 'std', 'min', 'max']
    missing_keys = [k for k in expected_keys if k not in stats]
    print(f"缺失鍵: {missing_keys}")

if __name__ == "__main__":
    test_ensemble_stats_issue()
```

## ⚠️ 風險評估

### 高風險操作
1. **修改核心wrapper類別**: 可能影響現有功能，需要完整回歸測試
2. **API介面變更**: 可能破壞其他依賴模組

### 中等風險操作  
1. **添加方法別名**: 向後相容性良好，風險較低
2. **參數驗證增強**: 可能使原本"寬鬆"的程式碼失效

### 低風險操作
1. **測試檔案修正**: 只影響測試，不影響生產程式碼
2. **錯誤訊息改善**: 純粹改善除錯體驗

## 📈 可驗證指標

### 成功指標
1. **測試通過率**: 從 81% 提升至 ≥95%
2. **API一致性**: 所有wrapper類別通過類型檢查
3. **錯誤處理**: 所有輸入參數錯誤能產生清晰錯誤訊息

### 驗證方法
```bash
# 執行修復驗證腳本
python debug_mre_wrapper.py      # 驗證MRE 1修復
python debug_mre_ensemble.py     # 驗證MRE 2修復

# 執行完整測試套件
pytest tests/test_models.py -v   # 模型測試
pytest tests/test_losses.py -v   # 損失測試
pytest tests/ -x --tb=short      # 完整測試套件(快速失敗)

# 特定問題驗證
pytest tests/test_models.py::TestMultiHeadWrapper -v
pytest tests/test_models.py::TestEnsembleWrapper -v
```

## 🔄 下一步行動

### 立即執行 (0-2小時)
1. **修復EnsembleWrapper統計接口** - 低風險，高影響
2. **添加CausalWeighter方法別名** - 低風險，立即生效
3. **強化參數類型驗證** - 中等風險，長期受益

### 後續規劃 (2-4小時)
4. **全面測試檔案審查** - 確保測試期待值與實作一致
5. **建立API規範文檔** - 預防未來的API不匹配問題
6. **實作自動化API相容性檢查** - 持續整合

### 預防措施 (長期)
7. **建立API變更管理流程** - 任何介面變更需要影響分析
8. **實作契約測試** - 確保模組間介面穩定性
9. **定期API一致性稽核** - 每週自動檢查API相容性

## 📚 參考
- [測試失敗完整記錄](../context/context_session_001.md)
- [物理模組審查](./physics_review.md)  
- [風險登記簿](./physics_risks_register.md)
- [任務簡述](./task_brief.md)

---
**建立者**: Debugger sub-agent  
**最後更新**: 2025-09-30 23:02  
**狀態**: 分析完成，等待修復實施