# Task-001 技術審查報告與 Gate 決議

**審查時間**: 2025-09-30 23:15  
**審查範圍**: PINNs 逆重建專案 API 不匹配問題修復  
**審查員**: Reviewer Sub-Agent  
**審查階段**: 基礎修復階段 Gate 決議

---

## 📋 背景

本審查針對 Task-001「API不匹配問題修復任務」進行全面技術評估，基於 Physics、Debug、Performance 三個 sub-agents 的詳細報告，制定統一的 API 標準化方案，並做出最終 Gate 通過/失敗決議。

### 專案現況
- **測試通過率**: 81% (51/63)，12個失敗案例
- **核心問題**: API介面不一致，參數類型錯誤，方法名稱不匹配
- **影響範圍**: 模型包裝器、權重策略、物理模組

---

## 🔍 方法/依據

### 審查標準框架
本審查採用四維度評估框架：

1. **物理正確性** (Physics Gate)
   - 理論完整度：N-S 方程式實作準確性
   - 守恆律滿足：質量、動量守恆驗證
   - 數值穩定性：梯度計算魯棒性

2. **程式碼品質** (Debug Gate)
   - API一致性：介面命名與參數標準化
   - 測試覆蓋率：關鍵功能驗證完整性
   - 錯誤處理：異常情況處理能力

3. **效能基線** (Performance Gate)
   - 計算效率：基準建立與監控
   - 記憶體管理：資源使用最佳化
   - 擴展性：大規模問題支援能力

4. **整體可維護性** (Reviewer Gate)
   - API設計標準：統一的命名規範
   - 向後相容性：既有功能保護
   - 文檔完整性：使用說明與範例

### 評估依據
- **國際標準**: PEP 8 Python 編碼規範
- **科學計算慣例**: NumPy/SciPy API 設計原則  
- **深度學習框架**: PyTorch 最佳實踐
- **物理模擬標準**: 守恆律與量綱一致性要求

---

## 🎯 關鍵結論

### API 標準化方案

基於三個 sub-agents 的報告，制定以下 API 統一標準：

#### 1. 模型包裝器標準化 (高優先級)

**問題**: `MultiHeadWrapper` vs `ScaledPINNWrapper` API 不一致

**解決方案**:
```python
# 統一的包裝器基類設計
class BasePINNWrapper(nn.Module):
    """所有 PINN 包裝器的基類"""
    def __init__(self, base_model, input_scaler=None, output_scaler=None):
        # 標準化參數驗證
        self._validate_scalers(input_scaler, output_scaler)
    
    def _validate_scalers(self, input_scaler, output_scaler):
        """統一的參數驗證邏輯"""
        if input_scaler is not None and not hasattr(input_scaler, 'encode'):
            raise TypeError(f"input_scaler must implement 'encode' method")
        if output_scaler is not None and not hasattr(output_scaler, 'decode'):
            raise TypeError(f"output_scaler must implement 'decode' method")
```

**標準化要求**:
- 所有包裝器必須繼承 `BasePINNWrapper`
- 參數命名統一：`input_scaler`, `output_scaler`, `variable_names`
- 必須實作 `_validate_scalers()` 方法

#### 2. 權重策略接口統一 (高優先級)

**問題**: 方法命名不一致 (`compute_temporal_weights` vs `compute_causal_weights`)

**解決方案**:
```python
# 統一的權重策略基類
class BaseWeighter:
    """所有權重策略的基類"""
    def compute_weights(self, losses, **kwargs):
        """統一的權重計算接口"""
        raise NotImplementedError
    
    def update_weights(self, losses):
        """統一的權重更新接口"""
        raise NotImplementedError

# 向後相容性別名策略
class CausalWeighter(BaseWeighter):
    def compute_weights(self, losses, **kwargs):
        return self.compute_causal_weights(losses, **kwargs)
    
    def compute_temporal_weights(self, *args, **kwargs):
        """向後相容別名"""
        return self.compute_causal_weights(*args, **kwargs)
```

#### 3. 集成模型統計接口標準 (中優先級)

**問題**: `EnsembleWrapper` 統計輸出格式不統一

**解決方案**:
```python
# 標準化統計輸出格式
ENSEMBLE_STATS_SCHEMA = {
    'mean': 'torch.Tensor',    # 集成平均
    'std': 'torch.Tensor',     # 標準差
    'var': 'torch.Tensor',     # 方差
    'uncertainty': 'torch.Tensor',  # 不確定性 (= std)
    'min': 'torch.Tensor',     # 最小值
    'max': 'torch.Tensor',     # 最大值
    'quantile': 'Dict[str, torch.Tensor]'  # 分位數 (可選)
}
```

#### 4. 物理模組接口規範 (高優先級)

**問題**: `NSEquations2D` 類別缺失

**解決方案**:
```python
class NSEquations2D:
    """2D Navier-Stokes 方程式統一接口"""
    def __init__(self, viscosity=1e-3, density=1.0):
        self.viscosity = viscosity
        self.density = density
    
    def residual(self, u, v, p, x, y, t=None):
        """計算 N-S 方程式殘差"""
        return ns_residual_2d(u, v, p, x, y, self.viscosity, t)
    
    def check_conservation(self, u, v, p):
        """檢查守恆律"""
        return check_conservation_laws(u, v, p)
```

### 向後相容性策略

**原則**: 所有修改必須保持向後相容性，不破壞現有功能

**實施方法**:
1. **別名機制**: 舊方法名保留作為新方法的別名
2. **漸進式遷移**: 新舊API並存，逐步過渡
3. **清晰的棄用警告**: 使用 `warnings.warn()` 提示API變更

---

## ⚠️ 風險評估

### 高風險項目 🔴
1. **核心API變更風險**
   - **風險**: 修改 `ScaledPINNWrapper` 可能影響依賴模組
   - **緩解**: 完整回歸測試 + 分階段實施
   - **回滾計畫**: Git分支保護 + 自動化測試門控

2. **物理正確性風險**
   - **風險**: API修改可能影響數值計算精度
   - **緩解**: 物理驗證測試套件 + 基準比對
   - **監控**: PDE殘差、守恆律偏差實時監控

### 中等風險項目 🟡
3. **測試失敗擴散風險**
   - **風險**: 修復過程中可能引入新的測試失敗
   - **緩解**: 每次修改後立即執行測試
   - **限制**: 單次只修改一個模組

4. **性能退化風險**
   - **風險**: 參數驗證可能影響計算效率
   - **緩解**: 效能基準測試 + 最佳化驗證
   - **閾值**: 性能下降不得超過5%

### 低風險項目 🟢
5. **文檔同步風險**
   - **影響**: 使用者體驗，不影響功能
   - **處理**: 後續任務中統一更新

---

## 📈 可驗證指標

### 成功指標
1. **測試通過率**: 從 81% → ≥95%
2. **API一致性**: 100% 核心類別通過類型檢查
3. **物理正確性**: PDE 殘差 < 1e-3, 守恆律偏差 < 1e-5
4. **向後相容性**: 所有現有使用方式正常運作

### 驗證腳本
```bash
# 基本功能驗證
python -c "from pinnx.models.wrappers import MultiHeadWrapper, ScaledPINNWrapper"
python -c "from pinnx.physics.ns_2d import NSEquations2D"
python -c "from pinnx.losses.weighting import CausalWeighter; w=CausalWeighter(); w.compute_temporal_weights({})"

# 完整測試套件
pytest tests/ -v --tb=short
pytest tests/test_models.py::TestMultiHeadWrapper -v
pytest tests/test_physics.py::TestNSEquations -v

# 物理正確性驗證
python scripts/physics_verification.py --conservation --pde-residual
```

### 監控儀表板
```python
dashboard_metrics = {
    "api_consistency": {
        "wrapper_type_check": "PASS/FAIL",
        "weighter_interface": "PASS/FAIL", 
        "physics_interface": "PASS/FAIL"
    },
    "physics_correctness": {
        "pde_residual_l2": "< 1e-3",
        "mass_conservation": "< 1e-5",
        "momentum_conservation": "< 1e-4"
    },
    "test_coverage": {
        "unit_tests": ">= 95%",
        "integration_tests": ">= 90%",
        "physics_tests": "100%"
    }
}
```

---

## 🔄 下一步修復計畫

### Phase 1: 立即修復 (0-2小時)
**目標**: 解決阻斷性問題，提升測試通過率至90%+

#### 1.1 EnsembleWrapper 統計接口修復 (30分鐘)
```python
# 檔案: pinnx/models/wrappers.py L376-387
# 風險: 極低 | 影響: 高
elif mode == 'stats':
    mean = torch.einsum('m,mbo->bo', self.weights, stacked)
    var = torch.var(stacked, dim=0)
    std = torch.std(stacked, dim=0)
    min_vals = torch.min(stacked, dim=0)[0]  # 新增
    max_vals = torch.max(stacked, dim=0)[0]  # 新增
    
    return {
        'mean': mean, 'var': var, 'std': std, 'uncertainty': std,
        'min': min_vals, 'max': max_vals  # 新增
    }
```

#### 1.2 CausalWeighter 方法別名 (15分鐘)
```python
# 檔案: pinnx/losses/weighting.py
# 風險: 極低 | 影響: 中
class CausalWeighter:
    def compute_temporal_weights(self, *args, **kwargs):
        """向後相容性別名"""
        return self.compute_causal_weights(*args, **kwargs)
```

#### 1.3 MultiObjectiveWeighting NameError 修復 (45分鐘)
```python
# 檔案: pinnx/losses/weighting.py L662
# 風險: 低 | 影響: 中
if 'gradnorm' in strategies:
    if self.model is not None:
        self.weighters['gradnorm'] = GradNormWeighter(self.model, self.loss_names)
    else:
        self.weighters['gradnorm'] = None  # 測試模式
```

### Phase 2: 系統修復 (2-4小時)
**目標**: 建立穩固的API基礎，測試通過率至95%+

#### 2.1 參數驗證強化 (90分鐘)
- 實作 `BasePINNWrapper` 基類
- 統一 scaler 參數驗證邏輯
- 改善錯誤訊息可讀性

#### 2.2 NSEquations2D 實作 (90分鐘)
- 創建面向對象的物理方程式接口
- 包裝現有函數為統一API
- 確保與測試期待一致

### Phase 3: 品質提升 (後續任務)
**目標**: 長期可維護性和穩定性

#### 3.1 API文檔標準化
- 統一所有類別的 docstring 格式
- 建立 API 使用範例
- 建立類型提示標準

#### 3.2 持續監控機制
- 實作 API 相容性自動檢查
- 建立回歸測試自動化
- 建立效能監控儀表板

---

## 🚪 Gate 決議

基於對 Physics、Debug、Performance 三個 sub-agents 報告的綜合分析：

### Physics Gate: ✅ **有條件通過**
- **核心物理理論**: 正確無誤，N-S方程式實作符合標準
- **守恆律驗證**: 方法論正確，實作穩健
- **數值穩定性**: VS-PINN尺度化理論實作正確
- **條件**: 需完成 `NSEquations2D` API 封裝

### Debug Gate: 🔴 **暫不通過**
- **當前測試通過率**: 81% (未達95%門檻)
- **主要問題**: 8個API不匹配失敗案例
- **修復可行性**: 高，修復方案明確且低風險
- **預期**: Phase 1修復後可達95%通過率

### Performance Gate: 🟡 **待評估**
- **基線建立**: 已制定詳細的三波優化策略
- **無退化保證**: 修復過程包含效能監控
- **條件**: 需在修復完成後建立正式基線

### Reviewer Gate: 🔴 **暫不通過**
- **API標準化**: 方案完整，實施計畫明確
- **向後相容性**: 策略穩健，風險可控
- **可維護性**: 架構設計符合最佳實踐
- **阻斷因素**: Debug Gate 未通過

---

## 🎯 最終決議

### 決議結果: **條件通過 - 須完成 Phase 1 修復**

**理由**:
1. **物理正確性已驗證**: 核心理論實作無誤
2. **修復方案成熟**: Debug 問題有明確解決路徑
3. **風險可控**: 修復操作風險低，影響範圍明確
4. **時程合理**: Phase 1 預計2小時內完成

### 進入實作階段的前置條件:
1. ✅ **Physics Gate**: 已通過，條件：完成 NSEquations2D 封裝
2. ❌ **Debug Gate**: Phase 1修復後必須達到95%測試通過率  
3. ⏳ **Performance Gate**: 基線建立，確保無性能退化
4. ⏳ **Reviewer Gate**: 所有條件滿足後自動通過

### 下一步行動:
1. **立即執行**: Phase 1 修復計畫 (2小時內)
2. **驗證確認**: 執行完整測試套件 + 物理驗證
3. **重新評估**: 所有Gates重新檢查
4. **正式啟動**: 滿足條件後進入大規模實作階段

---

## 📚 參考

1. **技術標準**:
   - PEP 8: Python Code Style Guide
   - PyTorch API Design Guidelines
   - SciPy API Standards

2. **Sub-agents 報告**:
   - [Physics Review](./physics_review.md) - 物理正確性詳細分析
   - [Debug Playbook](./debug_playbook.md) - 系統性修復策略
   - [Performance Playbook](./perf_playbook.md) - 三波優化計畫
   - [Failure Catalog](./failure_catalog.md) - 詳細失敗案例分析

3. **專案文檔**:
   - [Context Session 001](../context/context_session_001.md) - 專案全域狀態
   - [Task Brief](./task_brief.md) - 任務目標與範圍

---

**審查員**: Reviewer Sub-Agent  
**最後更新**: 2025-09-30 23:15  
**決議狀態**: 條件通過 - 等待 Phase 1 修復完成  
**下次審查**: Phase 1 完成後自動觸發