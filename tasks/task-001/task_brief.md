# Task-001: API不匹配問題修復任務簡述

**任務ID**: task-001  
**建立時間**: 2025-09-30  
**執行者**: 主 Agent + sub-agents 協作  
**時間盒**: 2 小時

## 🎯 任務目標
修復測試失敗中的 API 不匹配問題，統一模型包裝器和權重策略的接口命名，確保所有核心模組能正常協作。

## 📋 具體工作項目

### 1. 模型包裝器API統一 (高優先級)
**問題描述**: 測試期待 `MultiHeadWrapper` 類別，但實作為 `ScaledPINNWrapper`
- **影響範圍**: `pinnx/models/wrappers.py`、相關測試檔案
- **修復策略**: 標準化類別命名，或提供向後相容的別名

### 2. 權重策略API修正 (高優先級)  
**問題描述**: `GradNormWeighter.update_weights()` 方法簽名與測試不符
- **影響範圍**: `pinnx/losses/weighting.py`
- **修復策略**: 統一方法簽名，確保參數接口一致

### 3. 缺失類別補全 (高優先級)
**問題描述**: `NSEquations2D` 類別在測試中被引用但找不到定義
- **影響範圍**: `pinnx/physics/ns_2d.py`
- **修復策略**: 實作完整的 N-S 方程式類別

## 📊 輸入條件
- 現有的基礎模組架構 ✅
- 失敗的測試案例清單 ✅ 
- 核心功能需求規格 ✅

## 🎯 預期輸出與驗收標準

### 成功指標
1. **API一致性**: 所有模型包裝器、權重策略類別命名統一
2. **測試通過率**: 從 81% 提升至 ≥95%
3. **模組完整性**: 所有核心類別能正常導入和實例化
4. **向後相容**: 現有使用方式不受破壞

### 驗收測試
```bash
# 基本模組導入測試
python -c "from pinnx.models.wrappers import MultiHeadWrapper"
python -c "from pinnx.physics.ns_2d import NSEquations2D"

# 權重策略測試
python -c "from pinnx.losses.weighting import GradNormWeighter; w=GradNormWeighter(); w.update_weights({})"

# 完整測試套件
pytest test_models.py::test_model_wrappers -v
pytest test_physics.py::test_ns_equations -v
pytest test_losses.py::test_weighting_strategies -v
```

## 🚧 風險與限制

### 潛在風險
1. **破壞現有功能**: API變更可能影響已運作的模組
2. **測試期待不明確**: 某些測試的預期行為需要澄清
3. **相依性衝突**: 修改可能引發其他模組的問題

### 風險緩解
- 所有修改前先備份關鍵檔案
- 採用漸進式修復，每次只改動一個模組
- 修改後立即執行回歸測試

## 🔄 後續任務鏈接
- **task-002**: 完善 N-S 方程式物理模組
- **task-003**: 建立測試數據生成函數
- **task-004**: 實作完整訓練流程

## 📝 委派計畫

### Phase 1: Physics Gate 驗證
**委派對象**: Physicist sub-agent
- 審查 N-S 方程式實作的物理正確性
- 驗證量綱一致性和守恆律
- 產出：`physics_review.md`, `physics_test_spec.md`

### Phase 2: Debug 分析
**委派對象**: Debugger sub-agent  
- 分析所有測試失敗的根本原因
- 建立系統性的修復策略
- 產出：`debug_playbook.md`, `failure_catalog.md`

### Phase 3: Code Review
**委派對象**: Reviewer sub-agent
- 制定API標準化方案
- 審查修復後的程式碼品質
- 產出：`review_report.md`

---
**建立者**: 主 Agent  
**最後更新**: 2025-09-30 22:30  
**狀態**: 待開始