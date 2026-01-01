# Changelog

所有重要的專案變更都會記錄在此文件中。

格式基於 [Keep a Changelog](https://keepachangelog.com/zh-TW/1.0.0/)，
專案遵循 [語義化版本](https://semver.org/lang/zh-TW/)。

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
