# Debug 觀測協議 - PINNs API不匹配問題監控

**建立時間**: 2025-09-30 23:02  
**適用範圍**: Task-001 API修復過程監控  
**負責人**: Debugger sub-agent

## 🎯 監控目標
本協議制定系統性的偵錯觀測方法，確保API修復過程的可追蹤性和問題的早期發現。

## 📊 關鍵監控指標

### 1. 測試通過率追蹤
```bash
# 基線測試通過率檢查
pytest tests/ --tb=no -q | grep -E "(passed|failed|error)"

# 預期基線: 51 passed, 12 failed
# 目標: ≥95% 通過率 (≥60 passed, ≤3 failed)
```

### 2. API相容性監控
```python
# 檔案: monitor_api_compatibility.py
import torch
from pinnx.models.wrappers import ScaledPINNWrapper, MultiHeadWrapper, EnsemblePINNWrapper
from pinnx.losses.weighting import CausalWeighter

def check_api_compatibility():
    """檢查關鍵API相容性"""
    results = {}
    
    # 1. 檢查wrapper類別可用性
    try:
        from pinnx.models.wrappers import MultiHeadWrapper
        results['MultiHeadWrapper_import'] = True
    except ImportError:
        results['MultiHeadWrapper_import'] = False
    
    # 2. 檢查方法可用性  
    try:
        causal = CausalWeighter()
        results['compute_temporal_weights'] = hasattr(causal, 'compute_temporal_weights')
        results['compute_causal_weights'] = hasattr(causal, 'compute_causal_weights')
    except Exception:
        results['CausalWeighter_methods'] = False
    
    # 3. 檢查參數驗證
    try:
        from pinnx.models.fourier_mlp import PINNNet
        net = PINNNet(2, 3)
        wrapper = ScaledPINNWrapper(net, input_scaler=[], output_scaler={})
        results['parameter_validation'] = False  # 應該要失敗
    except (TypeError, AttributeError):
        results['parameter_validation'] = True   # 正確捕獲錯誤
    
    return results

if __name__ == "__main__":
    results = check_api_compatibility()
    for test, status in results.items():
        print(f"{test}: {'✅' if status else '❌'}")
```

### 3. 錯誤模式追蹤
```bash
# 檔案: track_error_patterns.sh
#!/bin/bash

echo "=== 錯誤模式追蹤 ==="
echo "時間: $(date)"

# 執行測試並擷取錯誤模式
pytest tests/ -v --tb=short 2>&1 | tee test_output.log

echo "=== 常見錯誤統計 ==="
grep -c "AttributeError.*encode" test_output.log || echo "AttributeError encode: 0"
grep -c "AttributeError.*compute_temporal_weights" test_output.log || echo "method name error: 0"  
grep -c "AssertionError.*min.*in" test_output.log || echo "missing stats keys: 0"
grep -c "NameError.*model.*not defined" test_output.log || echo "NameError model: 0"

echo "=== 失敗測試清單 ==="
grep "FAILED" test_output.log | cut -d' ' -f1 | sort
```

## 🔍 實時監控檢查點

### Phase 1 檢查點: EnsembleWrapper修復
**觸發條件**: 修改 `pinnx/models/wrappers.py` 後
```bash
# 驗證腳本
python -c "
import torch
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import EnsemblePINNWrapper

models = [PINNNet(2, 3) for _ in range(3)]
ensemble = EnsemblePINNWrapper(models)
x = torch.randn(5, 2)
stats = ensemble(x, mode='stats')

expected_keys = ['mean', 'std', 'min', 'max']
actual_keys = list(stats.keys())
missing = [k for k in expected_keys if k not in stats]

print(f'期待鍵: {expected_keys}')
print(f'實際鍵: {actual_keys}')
print(f'缺失鍵: {missing}')
print(f'修復狀態: {\"✅\" if not missing else \"❌\"}')
"

# 執行相關測試
pytest tests/test_models.py::TestEnsembleWrapper::test_ensemble_wrapper_basic -v
```

### Phase 2 檢查點: CausalWeighter方法別名
**觸發條件**: 修改 `pinnx/losses/weighting.py` 後
```bash
# 驗證腳本
python -c "
from pinnx.losses.weighting import CausalWeighter
import torch

causal = CausalWeighter()
has_old_method = hasattr(causal, 'compute_temporal_weights')
has_new_method = hasattr(causal, 'compute_causal_weights')

print(f'compute_temporal_weights: {\"✅\" if has_old_method else \"❌\"}')
print(f'compute_causal_weights: {\"✅\" if has_new_method else \"❌\"}')

# 測試方法呼叫
try:
    time_losses = [torch.tensor(1.0), torch.tensor(2.0)]
    result1 = causal.compute_temporal_weights(time_losses)
    result2 = causal.compute_causal_weights(time_losses)
    print(f'方法別名功能: {\"✅\" if result1 == result2 else \"❌\"}')
except Exception as e:
    print(f'方法呼叫錯誤: {e}')
"

# 執行相關測試
pytest tests/test_losses.py::TestWeightingStrategies::test_causal_weighting -v
```

### Phase 3 檢查點: 參數驗證強化
**觸發條件**: 修改參數驗證邏輯後
```bash
# 驗證腳本  
python -c "
import torch
from pinnx.models.fourier_mlp import PINNNet
from pinnx.models.wrappers import ScaledPINNWrapper

net = PINNNet(2, 3)

# 測試錯誤參數應該被拒絕
test_cases = [
    ('list_input_scaler', {'input_scaler': []}),
    ('dict_input_scaler', {'input_scaler': {}}), 
    ('str_output_scaler', {'output_scaler': 'invalid'})
]

for name, kwargs in test_cases:
    try:
        wrapper = ScaledPINNWrapper(net, **kwargs)
        print(f'{name}: ❌ 應該失敗但沒有')
    except (TypeError, AttributeError) as e:
        print(f'{name}: ✅ 正確拒絕 - {type(e).__name__}')
    except Exception as e:
        print(f'{name}: ⚠️  意外錯誤 - {type(e).__name__}: {e}')

# 測試正確參數應該成功
try:
    wrapper = ScaledPINNWrapper(net, input_scaler=None, output_scaler=None)
    x = torch.randn(3, 2)
    y = wrapper(x)
    print(f'正確參數: ✅ 成功 - 輸出形狀 {y.shape}')
except Exception as e:
    print(f'正確參數: ❌ 意外失敗 - {e}')
"
```

## 📈 進度追蹤儀表板

### 測試進度追蹤
```bash
# 檔案: progress_dashboard.sh
#!/bin/bash

echo "╔══════════════════════════════════════╗"
echo "║        PINNs Debug Dashboard         ║"
echo "╚══════════════════════════════════════╝"
echo ""

echo "📊 測試通過率趨勢:"
echo "基線 ($(date -v-1H '+%H:%M')): 51/63 (81%)"

# 當前測試狀態
current_results=$(pytest tests/ --tb=no -q 2>/dev/null | tail -1)
echo "當前 ($(date '+%H:%M')): $current_results"

echo ""
echo "🎯 關鍵修復目標:"
echo "  1. EnsembleWrapper stats interface    : $([[ -f .checkpoint_1 ]] && echo '✅' || echo '⏳')"
echo "  2. CausalWeighter method alias        : $([[ -f .checkpoint_2 ]] && echo '✅' || echo '⏳')"  
echo "  3. Parameter validation enhancement   : $([[ -f .checkpoint_3 ]] && echo '✅' || echo '⏳')"
echo "  4. MultiObjectiveWeighting NameError : $([[ -f .checkpoint_4 ]] && echo '✅' || echo '⏳')"

echo ""
echo "🔥 當前失敗測試:"
pytest tests/ --tb=no -q 2>/dev/null | grep FAILED | head -5
```

## 🚨 異常警報條件

### 紅色警報 (立即處理)
- 測試通過率下降超過5%
- 新增任何ImportError或ModuleNotFoundError
- 原本通過的測試開始失敗

### 黃色警報 (需要關注)
- 測試執行時間增加超過30%
- 出現新的警告訊息
- Memory usage 異常增長

### 監控腳本
```bash
# 檔案: alert_monitor.sh
#!/bin/bash

# 執行測試並檢查異常情況
pytest tests/ --tb=no -q > current_test_result.txt 2>&1

# 檢查是否有新的導入錯誤
import_errors=$(grep -c "ImportError\|ModuleNotFoundError" current_test_result.txt)
if [ $import_errors -gt 0 ]; then
    echo "🚨 紅色警報: 發現 $import_errors 個導入錯誤"
fi

# 檢查測試通過率
passed=$(grep -o "[0-9]* passed" current_test_result.txt | cut -d' ' -f1)
failed=$(grep -o "[0-9]* failed" current_test_result.txt | cut -d' ' -f1)

if [ -n "$passed" ] && [ -n "$failed" ]; then
    total=$((passed + failed))
    pass_rate=$((passed * 100 / total))
    
    if [ $pass_rate -lt 76 ]; then  # 81% - 5%
        echo "🚨 紅色警報: 測試通過率下降至 ${pass_rate}%"
    elif [ $pass_rate -lt 81 ]; then
        echo "⚠️  黃色警報: 測試通過率下降至 ${pass_rate}%"
    fi
fi

# 檢查執行時間 (如果有歷史基線)
if [ -f baseline_time.txt ]; then
    baseline_time=$(cat baseline_time.txt)
    current_time=$(grep -o "in [0-9.]*s" current_test_result.txt | cut -d' ' -f2 | cut -d's' -f1)
    
    if [ -n "$current_time" ] && [ -n "$baseline_time" ]; then
        time_increase=$(echo "scale=2; ($current_time - $baseline_time) / $baseline_time * 100" | bc)
        if [ $(echo "$time_increase > 30" | bc) -eq 1 ]; then
            echo "⚠️  黃色警報: 測試執行時間增加 ${time_increase}%"
        fi
    fi
fi
```

## 🔧 自動化監控設置

### 定期檢查 (建議每30分鐘)
```bash
# 加入 crontab 或設置定期執行
*/30 * * * * cd /path/to/pinns-mvp && bash monitor/alert_monitor.sh
```

### 修復驗證清單
- [ ] EnsembleWrapper 統計接口完整性
- [ ] CausalWeighter 方法別名可用性  
- [ ] ScaledPINNWrapper 參數驗證
- [ ] MultiObjectiveWeighting 初始化修復
- [ ] 所有 wrapper 類別 API 一致性
- [ ] 測試期待值與實作同步

---
**建立者**: Debugger sub-agent  
**最後更新**: 2025-09-30 23:02  
**執行狀態**: 監控協議就緒