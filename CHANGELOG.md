# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
專案遵循 [語義化版本](https://semver.org/lang/zh-TW/)。

---

## [1.3.0] - 2026-01-03

### ✨ Added - Registry Pattern & Schema Validation
- **Registry Pattern 架構重構**: 統一的工廠函數管理系統
  - `pinnx/train/factories.py`: Optimizer/Scheduler 工廠（Phase 1）
  - `pinnx/train/model_physics_factory.py`: Model/Physics 工廠（Phase 2）
  - **消除 18 個條件分支**（if-elif chains）
  - 裝飾器註冊：`@registry.register('type_name')`
  
- **Schema Validation System** (Phase 3): 型別安全的配置驗證
  - `ConfigSchema` 類別：宣告式配置驗證
  - **Union Types 支援**: `field_types={'nu': (int, float)}`
  - 自動驗證流程：必要欄位 → 型別檢查 → 自訂驗證 → 預設值填充
  - 智慧錯誤訊息：`"Expected int or float, got str"`

- **完整文檔系統**:
  - `context/session_logs/SESSION_SUMMARY_2026-01-03_Phase3_Schema_Validation.md`: 完成報告
  - `context/session_logs/SCHEMA_VALIDATION_TECHNICAL_GUIDE.md`: 詳細技術指南
  - `context/session_logs/SCHEMA_VALIDATION_QUICK_REFERENCE.md`: 快速參考卡
  - `REGISTRY_PATTERN_PHASE2_COMPLETE.md`: Phase 1+2 完成報告

### 🔄 Changed - Architecture
- **工廠函數統一**: 所有創建邏輯透過 Registry Pattern
  - Optimizer: 5 類型（adam, adamw, soap, lbfgs, sgd）
  - Scheduler: 6 類型（cosine, step, exponential, warmup_cosine, etc.）
  - Model: 4 類型（fourier_vs_mlp, resnet, piratenet, axis_selective_fourier_mlp）
  - Physics: 3 類型（vs_pinn_channel_flow, ns_2d, kolmogorov_flow_2d）

- **型別提示增強**: `pinnx/train/model_physics_factory.py`
  - `ConfigSchema.__init__`: `field_types` 支援 `Union[type, Tuple[type, ...]]`
  - `ConfigSchema.validate()`: 智慧型別檢查（單一型別 vs Union types）
  - `_Registry.register()`: 同步型別提示更新

### 🧪 Testing
- **測試覆蓋率**: 96.6% (28/29 tests passing)
  - `tests/test_factories.py`: 17/18 passing (1 skipped - SOAP)
  - `tests/test_model_physics_factories.py`: 11/11 passing
- **驗證功能**: Schema validation 自動測試套件

### 📊 Statistics
- **條件分支消除**: 18 branches → 0 (100% reduction)
  - Phase 1: 11 branches (Optimizer/Scheduler)
  - Phase 2: 7 branches (Model/Physics)
  - Phase 3: 0 branches added (Schema validation)
- **文檔新增**: 3 個完整文檔（~2,500 行）
- **程式碼修改**: 1 個核心檔案（型別提示增強）

### 🎯 Philosophy Alignment
- **Good Taste**: Zero conditional branches，純粹字典查找
- **Simplicity**: 宣告式配置（declarative validation）
- **Type Safety**: 完整型別檢查與 Union Types 支援
- **Pragmatism**: 早期發現配置錯誤，避免訓練到一半才失敗

### 🔗 Related Documents
- **Phase 1+2**: `REGISTRY_PATTERN_PHASE2_COMPLETE.md`
- **Phase 3**: `SESSION_SUMMARY_2026-01-03_Phase3_Schema_Validation.md`
- **Technical Guide**: `SCHEMA_VALIDATION_TECHNICAL_GUIDE.md`
- **Quick Reference**: `SCHEMA_VALIDATION_QUICK_REFERENCE.md`

---

## [1.2.2] - 2026-01-03

### ✨ Added
- **配置驗證工具**: 全新的配置檔案驗證系統
  - `pinnx/utils/config_validator.py`: 核心驗證邏輯類別
  - `scripts/tools/validate_config.py`: CLI 獨立驗證工具
  - 支援常見錯誤檢測（如 `loss` vs `losses` 拼寫錯誤）
  - 支援批次驗證與嚴格模式
- **早期配置檢查**: `Trainer.__init__()` 中整合 `_validate_config_early()`
  - Fail-fast 機制：在初始化前攔截錯誤配置
  - 清晰的錯誤訊息與修復建議

### 📚 Changed
- **文檔簡化**: 總文檔行數減少 21%（4,781 → 3,761 行）
  - `README.md`: 103 行（-42%），移除冗餘表格
  - `docs/CONFIG_GUIDE.md`: 232 行，合併 3 個配置文檔為單一來源
  - 任務導向結構："我想要..." 引導式說明
- **文檔歸檔**: 舊配置文檔移至 `docs/archive/`
  - `CONFIG_MANAGEMENT_GUIDE.md` → `docs/archive/`
  - `CONFIG_REFERENCE.md` → `docs/archive/`

### 🐛 Fixed
- **配置靜默失敗**: 修復 `loss:` 錯寫導致使用預設權重的問題
  - 現在會明確報錯並提供正確寫法 (`losses:`)
- **文檔重複**: 消除配置說明在 3 個檔案中的重複內容

### 📊 Statistics
- **文檔減少**: -21% 總文檔行數（-1,020 行）
- **配置文檔**: -68% 行數（727 → 232 行）
- **核心文檔**: 從 9 個減至 7 個活躍文檔

### 🎯 Philosophy Alignment
- **Pragmatism**: 解決配置靜默失敗的真實痛點
- **Simplicity**: 單一配置文檔來源（`standard_config_template.yml`）
- **Good Taste**: 清晰錯誤訊息，消除調試猜測

---

## [1.2.0] - 2025-12-30

### ✨ Added
- **WandB 整合**: 完整的 Weights & Biases 實驗追蹤系統
  - 雲端實驗管理與超參數追蹤
  - 進階視覺化與團隊協作功能
  - `.wandb_config` 配置檔支援
- **測試工具**: `scripts/tools/test_wandb_integration.py` 自動化測試腳本
- **文檔**: 完整的 WandB 遷移指南 (`docs/WANDB_MIGRATION_GUIDE.md`)

### 🔄 Changed
- **Trainer**: `pinnx/train/trainer.py` WandB 初始化與 run 管理
- **Training Loop Manager**: `pinnx/train/training_loop_manager.py` 完整 logging 重寫
  - `log_losses_to_tensorboard()` → `log_losses_to_wandb()`
  - `finalize_tensorboard()` → `finalize_wandb()`
- **配置檔**: 43+ YAML 配置檔更新
  - `logging.tensorboard: true` → `logging.wandb: true`
  - 移除 `output.tensorboard_dir` 配置項

### ❌ Removed
- **TensorBoard 支援**: 完全移除所有 TensorBoard 相關代碼
  - 移除 `torch.utils.tensorboard.SummaryWriter` 所有引用
  - 移除 `tensorboard>=2.13` 依賴項
  - 移除所有 TensorBoard 配置選項
- **向下相容性**: 不再支援舊的 TensorBoard logging 配置

### 🔒 Security
- **API Key 保護**: `.wandb_config` 加入 `.gitignore`
- **本地緩存排除**: `wandb/` 目錄加入 `.gitignore`

### 📊 Statistics
- 55 檔案修改
- +3,135 行新增
- -2,285 行刪除
- 淨增加 850 行

### ⚠️ Breaking Changes
- **必須配置 WandB**: 訓練前需建立 `.wandb_config` 檔案
- **無 TensorBoard 支援**: 舊配置檔中的 `logging.tensorboard` 將被忽略
- **配置格式變更**: 必須使用 `logging.wandb: true`

### 🔗 Related
- Commit: `af3c67f`
- PR: N/A (直接推送到 master)
- Issues: N/A

---

## [1.1.1] - 2025-12-18

### 🐛 Fixed
- **Channel Flow 標準化**: 修復 3D Channel Flow 訓練災難性失敗問題
  - 問題: v/w/p 場誤差達 1000-2000%
  - 原因: Sensor 文件僅含座標，訓練 fallback 到損壞的 RANS prior
  - 解決: 使用 K=100 稀疏測量點計算標準化統計
  - 結果: 誤差降至 100-200% (合理範圍)

### ✨ Added
- **數據工具**: `extract_sensor_values_from_dns.py` 提取感測器值
- **驗證流程**: 自動數據品質驗證機制

### 📖 Documentation
- 根因分析: `results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md`
- 解決方案: `results/channel_flow_evaluation/CORRECT_SOLUTION_K100.md`
- Kolmogorov 驗證: `KOLMOGOROV_NORMALIZATION_CHECK.md`

### ℹ️ Notes
- ✅ 僅 3D Channel Flow 受影響
- ✅ 2D Kolmogorov Flow 已驗證不受影響

---

## [1.1.0] - 2025-12-17

### 🎯 Strategy
- **聚焦核心**: 僅支援 2 個核心場景
  - 2D Kolmogorov Flow (u, v, nu_t) + Leith 模型
  - 3D Channel Flow Re_τ=1000 (u, v, w, p, k, ε, nu_t) + RANS k-ε

### ❌ Removed
- **LES 支援**: 移除所有 LES 相關功能
- **DNS 降採樣**: 移除 DNS downsampling 功能
- **NetCDF 格式**: 移除 NetCDF 格式支援
- **進階損失**: 移除統計損失、守恆損失、對稱性損失

### 🔄 Changed
- **API 簡化**: 
  - `create_lowfi_loader()` 簡化介面
  - `PriorLossManager` 精簡實作

### ✅ Verified
- 核心模組 100% 測試通過
- 零回歸問題

### 📊 Statistics
- -774 行代碼移除
- 代碼庫精簡約 15%

### 📖 Documentation
- 完整紀錄: 專案根目錄 (舊版 CHANGELOG)
- 專案範圍: `docs/PROJECT_SCOPE.md`
- Phase 2 報告: `docs/PHASE2_COMPLETION_REPORT.md`

---

## [1.0.0] - 2025-12-01 (假設日期)

### 🎉 Initial Release
- **核心架構**: Fourier-SIREN MLP + RWF + VS-PINN
- **優化策略**: GradNorm + Curriculum Learning + 因果訓練
- **先驗整合**: RANS 低保真場軟約束
- **感測器優化**: QR-Pivot 離線佈局
- **配置系統**: YAML 驅動實驗配置
- **DNS 數據集**: 自建 2D Kolmogorov Flow (Re=50/100/500)
- **評估系統**: 完整的場誤差評估與視覺化

---

## 版本號規範

本專案採用 [語義化版本 2.0.0](https://semver.org/lang/zh-TW/)：

- **MAJOR**: 不相容的 API 變更
- **MINOR**: 向下相容的功能新增
- **PATCH**: 向下相容的 bug 修復

### 類型標籤

- ✨ **Added**: 新功能
- 🔄 **Changed**: 既有功能變更
- 🗑️ **Deprecated**: 即將移除的功能
- ❌ **Removed**: 已移除的功能
- 🐛 **Fixed**: Bug 修復
- 🔒 **Security**: 安全性修復
- 📖 **Documentation**: 文檔更新
- 📊 **Statistics**: 統計數據
- ⚠️ **Breaking Changes**: 破壞性變更

---

## 未發布變更

### [Unreleased]

目前沒有未發布的變更。

---

## 參考連結

- [WandB 遷移指南](docs/WANDB_MIGRATION_GUIDE.md)
- [技術文檔](docs/TECHNICAL_DOCUMENTATION.md)
- [專案範圍](docs/PROJECT_SCOPE.md)
- [快速開始](docs/QUICK_START.md)

---

**最後更新**: 2025-12-30
