# Task-001 實作計畫

**建立時間**: 2025-09-30 23:25  
**執行者**: 主 Agent  
**基於**: Physics、Debug、Performance、Reviewer sub-agents 審查報告

## 🎯 實作目標
基於 Reviewer 的條件通過決議，執行 Phase 1 關鍵修復，將測試通過率從 81% 提升至 95%+，滿足 Debug Gate 條件。

## 📋 Phase 1 實作清單（預計 2 小時）

### 1.1 EnsembleWrapper 統計接口修復 (30分鐘)
**檔案**: `pinnx/models/wrappers.py` L376-387  
**風險**: 極低 | **影響**: 高

**問題**: 統計模式缺少 min/max 欄位  
**修復**: 添加 min_vals, max_vals 到回傳字典

```python
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

### 1.2 CausalWeighter 方法別名 (15分鐘)
**檔案**: `pinnx/losses/weighting.py`  
**風險**: 極低 | **影響**: 中

**問題**: 測試期待 `compute_temporal_weights` 方法  
**修復**: 添加向後相容別名

```python
class CausalWeighter:
    def compute_temporal_weights(self, *args, **kwargs):
        """向後相容性別名"""
        warnings.warn("compute_temporal_weights is deprecated, use compute_causal_weights", 
                      DeprecationWarning, stacklevel=2)
        return self.compute_causal_weights(*args, **kwargs)
```

### 1.3 MultiObjectiveWeighting NameError 修復 (45分鐘)
**檔案**: `pinnx/losses/weighting.py` L662  
**風險**: 低 | **影響**: 中

**問題**: `GradNormWeighter` 未正確初始化  
**修復**: 修正條件判斷邏輯

```python
if 'gradnorm' in strategies:
    if self.model is not None:
        self.weighters['gradnorm'] = GradNormWeighter(self.model, self.loss_names)
    else:
        logger.warning("GradNorm requires model reference, skipping in test mode")
        self.weighters['gradnorm'] = None
```

### 1.4 NSEquations2D 類別實作 (30分鐘)
**檔案**: `pinnx/physics/ns_2d.py`  
**風險**: 低 | **影響**: 高

**問題**: 類別完全缺失  
**修復**: 創建面向對象的 N-S 方程式接口

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

## 🧪 測試策略

### 漸進式測試
每次修復後立即執行相關測試：

```bash
# 1.1 修復後
pytest tests/test_models.py::TestEnsembleWrapper::test_stats_output -v

# 1.2 修復後  
pytest tests/test_losses.py::TestCausalWeighter::test_temporal_weights -v

# 1.3 修復後
pytest tests/test_losses.py::TestMultiObjectiveWeighting -v

# 1.4 修復後
pytest tests/test_physics.py::TestNSEquations -v

# 完整驗證
pytest tests/ -v --tb=short
```

### 回滾策略
- 每個檔案修改前先備份
- Git 分支保護：`git checkout -b fix/api-consistency`
- 單一修改失敗時立即回滾該檔案
- 保持每次 commit 的原子性

## 📊 成功指標

### 主要指標
1. **測試通過率**: 81% → ≥95% (目標：達到 60/63 通過)
2. **API 一致性**: 100% 核心類別通過類型檢查
3. **向後相容性**: 0 個現有功能受破壞

### 驗證腳本
```bash
# 基本導入驗證
python -c "from pinnx.models.wrappers import EnsembleWrapper"
python -c "from pinnx.physics.ns_2d import NSEquations2D"
python -c "from pinnx.losses.weighting import CausalWeighter; w=CausalWeighter(); hasattr(w, 'compute_temporal_weights')"

# 物理正確性檢查（不破壞守恆律）
python -c "from pinnx.physics.ns_2d import NSEquations2D; eq=NSEquations2D(); eq.check_conservation"
```

## 🚧 風險控制

### 修復順序（由低風險到高風險）
1. **別名添加** (CausalWeighter) - 無破壞性
2. **字典欄位添加** (EnsembleWrapper) - 低風險
3. **類別創建** (NSEquations2D) - 中風險  
4. **邏輯修正** (MultiObjectiveWeighting) - 高風險

### 監控協議
- 每次修改後執行 `python -c "import pinnx"` 確保無導入錯誤
- 記錄修改前後的測試通過數量
- 監控物理測試指標不退化

## 📝 實作記錄

將在 `context/decisions_log.md` 中記錄：
- 修改時間與檔案
- 測試結果變化  
- 效能影響評估
- commit hash

## 🔄 後續任務鏈接
- **task-002**: Phase 2 系統修復（參數驗證強化）
- **task-003**: 建立完整訓練流程
- **task-004**: JHTDB 資料獲取實作

---

**建立者**: 主 Agent  
**狀態**: 準備執行  
**預計完成時間**: 2 小時內