# 🌊 PINNs-MVP: 稀疏測量湍流重建

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
conda env create -f environment.yml && conda activate pinns-mvp

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
@software{pinns_mvp_2026,
  title={PINNs-MVP: Sparse Sensor Turbulence Reconstruction},
  author={PINNs-MVP Team},
  year={2026},
  version={1.2.2},
  url={https://github.com/your-org/pinns-mvp}
}
```

---

**完整文檔**: [docs/README.md](docs/README.md) | **更新日誌**: [CHANGELOG.md](CHANGELOG.md)
