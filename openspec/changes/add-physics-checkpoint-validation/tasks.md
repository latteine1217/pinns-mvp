# Implementation Tasks

## 1. Core Validation Logic
- [x] 1.1 創建 `pinnx/physics/validators.py` 模組
- [x] 1.2 實作 `validate_mass_conservation()` 函數（檢查 ∇·u ≈ 0）
- [x] 1.3 實作 `validate_momentum_conservation()` 函數（檢查 NS 方程殘差）
- [x] 1.4 實作 `validate_boundary_conditions()` 函數（檢查壁面無滑移條件）
- [x] 1.5 實作 `compute_physics_metrics()` 整合函數

## 2. Checkpoint Integration
- [x] 2.1 在 `pinnx/train/checkpointing.py` 新增 `validate_physics_before_save()` 函數
- [x] 2.2 修改 `Trainer.save_checkpoint()` 調用物理驗證
- [x] 2.3 在檢查點元數據中記錄物理指標（`physics_metrics` 欄位）
- [x] 2.4 實作驗證失敗時的警告日誌與拒絕保存邏輯

## 3. Configuration Support
- [x] 3.1 在配置 schema 新增 `physics_validation` 區塊
- [x] 3.2 新增驗證閾值參數（`mass_conservation_threshold`, `boundary_error_threshold`）
- [x] 3.3 新增 `enable_physics_validation` 開關（預設 `true`）
- [x] 3.4 更新所有模板配置檔案（`configs/templates/*.yml`）

## 4. Testing
- [x] 4.1 撰寫 `tests/test_physics_validators.py` 單元測試
- [x] 4.2 撰寫 `tests/test_checkpoint_physics_validation.py` 整合測試
- [x] 4.3 測試驗證失敗情境（故意違反物理約束）
- [x] 4.4 測試驗證通過情境（正常訓練檢查點）

## 5. Documentation
- [ ] 5.1 更新 `TECHNICAL_DOCUMENTATION.md` 新增物理驗證章節
- [ ] 5.2 更新 `configs/README.md` 說明 `physics_validation` 配置
- [ ] 5.3 在 `AGENTS.md` 更新 Physics Gate 流程說明
- [ ] 5.4 新增範例配置與使用說明
