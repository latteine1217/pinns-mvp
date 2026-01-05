# 🌊 PINNs-SparseFlow: 稀疏測量湍流重建

基於物理資訊神經網路 (PINNs) 的湍流逆問題求解器，從極少量感測器觀測 (K ≤ 100) 重建高保真流場。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/WandB-Logging-yellow)](https://wandb.ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

## 核心特性

- **架構**: Fourier-SIREN MLP + Random Weight Factorization (RWF) + VS-PINN
- **優化**: GradNorm 動態權重 + Curriculum Learning + 因果訓練
- **先驗整合**: RANS 低保真場作為軟約束
- **感測器**: QR-Pivot 離線優化佈局 (K ≤ 100)
- **實驗管理**: WandB 雲端追蹤

## 快速開始

```bash
# 1. 環境設置
conda env create -f environment.yml && conda activate pinns-sparse-flow

# 2. 配置 WandB（必須，僅需一次）
echo "WANDB_API_KEY=your_key_here" > .wandb_config

# 3. 驗證配置（推薦）
python scripts/tools/validate_config.py --config configs/main.yml

# 4. 訓練
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 5. 評估
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/<exp>/best_model.pth \
  --config configs/<exp>.yml
```

## 支援場景

### 2D Kolmogorov Flow ✅
- **用途**: 設計驗證、快速實驗（5-10 分鐘）
- **物理**: 不可壓縮 NS + Leith 湍流模型
- **域**: 4π × 2π 週期域

### 3D Channel Flow Re_τ=1000 ✅
- **用途**: 工程級驗證、生產環境（2-8 小時）
- **物理**: 不可壓縮 NS + RANS k-ε 模型
- **域**: 8π × 2 × 3π（x/y/z）

## 文檔索引

| 文檔 | 用途 | 讀者 |
|------|------|------|
| [QUICK_START.md](docs/QUICK_START.md) | 完整工作流程 | 新用戶 |
| [TRAINERBUILDER_GUIDE.md](docs/TRAINERBUILDER_GUIDE.md) | TrainerBuilder 使用指南 ✨ | 開發者 |
| [CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) | 配置參數說明與管理 | 配置調整 |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API 文檔 | 開發者 |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 問題診斷 | 調試 |
| [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | 系統架構 | 深入理解 |

## 驗收指標

- 流場誤差 ≤ 10-15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

## 重要更新

### ✅ v1.3.4 (2026-01-05): P2-3 Init Simplification
- **架構重構**: TrainerBuilder + TrainerComponents 完整實施
- **TrainerComponents**: 新增數據類封裝所有可選組件（24+ 組件，5 大類別）
- **TrainerBuilder 擴展**: +500 行組件創建邏輯（從 Trainer._setup_* 遷移）
- **Trainer 雙路徑**: 支持新路徑（TrainerComponents）+ 舊路徑（向後兼容）
- **代碼簡化**: Trainer 初始化邏輯從 17 個方法簡化為依賴注入
- **測試覆蓋**: 28/28 現有測試通過，完全向後兼容 ✅
- **設計模式**: Builder Pattern（複雜對象創建）+ Dependency Injection（組件外部注入）+ Dual-Path Architecture（新舊並存）

### ✅ v1.3.3 (2026-01-04): P2-2 ValidationManager 解耦
- **Strategy Pattern**: 將 Trainer 中 150+ 行驗證邏輯提取到獨立 ValidationManager
- **責任分離**: DataBasedValidation 處理數據驗證，PhysicsBasedValidation 處理物理驗證
- **依賴注入**: Trainer 通過建構子接收 ValidationManager 實例，支援自定義驗證策略
- **可組合性**: 支援多種驗證策略組合使用（數據 + 物理）
- **測試覆蓋**: 15/15 ValidationManager 單元測試 + 2/2 Trainer 集成測試通過
- **代碼減少**: Trainer.py -154 行（移除冗餘驗證方法），新增 validation_manager.py +463 行

### ✅ v1.3.2 (2026-01-04): P2-1 CheckpointManager 解耦
- **Manager Pattern**: 將 Trainer 中 300+ 行 checkpoint 邏輯提取到獨立 CheckpointManager
- **責任分離**: StandardCheckpointManager 處理 I/O，PeriodicCheckpointStrategy 管理保存策略
- **依賴注入**: Trainer 通過建構子接收 CheckpointManager 實例，支援自定義實現
- **向後兼容**: 保留 `best_model.pth` 文件格式，所有現有測試通過
- **測試覆蓋**: 13/13 CheckpointManager 單元測試 + 2/2 Trainer 集成測試通過
- **代碼減少**: Trainer.py -54 行（移除 `_build_checkpoint_data`），新增 checkpoint_manager.py +491 行

### ✅ v1.3.1 (2026-01-04): P1-3a Weighter 接口統一
- **接口重構**: 統一所有 Loss Weighter 接口，消除方法名和參數簽名不一致
- **抽象基類**: 新建 `LossWeighter` 和 `PointWeighter` 基類
- **純多態**: GradNorm/NTK/Adaptive/Causal 全部遵循統一接口
- **Context 模式**: 通過字典傳遞可選參數，避免簽名膨脹
- **測試驗證**: 6/6 單元測試通過，P0 驗證完成

### ✅ v1.3.0 (2026-01-03): Registry Pattern + Schema Validation
- **架構重構**: 完整遷移至 Registry Pattern（消除 18 個條件分支）
- **型別安全**: Schema Validation with Union Types 支援
- **工廠函數**: Model/Physics/Optimizer/Scheduler 統一管理
- **測試覆蓋**: 96.6% (28/29 tests passing)
- **文檔**: 完整技術指南與快速參考卡

### ✅ v1.2.2 (2026-01-03): 配置驗證與文檔簡化
- **新功能**: 配置檔案驗證系統（攔截常見錯誤如 `loss` vs `losses`）
- **CLI 工具**: `scripts/tools/validate_config.py` 獨立驗證工具
- **文檔整合**: 合併配置文檔為單一 `CONFIG_GUIDE.md`（-68% 行數）
- **早期檢查**: Trainer 初始化前 Fail-fast 驗證

### ✅ v1.2.1 (2026-01-02): 標準化清理
- 統一配置鍵名（`losses` 為標準）
- 感測資料一致化處理
- **破壞性變更**: 移除向後相容分支

### 🚀 v1.2.0 (2025-12-30): WandB 遷移完成
- 完全遷移至 Weights & Biases
- 移除所有 TensorBoard 支援
- **破壞性變更**: 無向下相容

## 已知限制

- **記憶體**: MPS (Apple Silicon) 在 10K PDE 點時可能 OOM，建議降至 5K 或使用 CUDA
- **兼容性**: v1.2.0 完全移除 TensorBoard，必須使用 WandB
- **向下相容**: v1.1.0 移除多項已棄用功能，詳見 [CHANGELOG.md](CHANGELOG.md)

## 貢獻與授權

歡迎 Issue/PR，遵循標準 Fork & PR 流程。授權：MIT。

研究引用請註明本專案：
```bibtex
@software{pinns_sparse_flow_2026,
  title={PINNs-SparseFlow: Sparse Sensor Turbulence Reconstruction},
  author={Li, JunYi},
  year={2026},
  version={1.3.4},
  url={https://github.com/latteine1217/pinns-sparse-flow}
}
```

---

**完整文檔**: [docs/README.md](docs/README.md) | **更新日誌**: [CHANGELOG.md](CHANGELOG.md)
