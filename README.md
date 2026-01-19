# 🌊 PINNs-SparseFlow: 稀疏測量湍流重建

基於物理資訊神經網路 (PINNs) 的湍流逆問題求解器，從極少量感測器觀測 (K ≤ 400) 重建高保真流場。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![WandB](https://img.shields.io/badge/WandB-Logging-yellow)](https://wandb.ai/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)

## 核心特性

- **架構**: Fourier-VS MLP（SIREN 激活 + 軸向選擇 Fourier Features + 🚀 Lazy Evaluation）+ VS-PINN 變數尺度化 + 可選 RWF
- **效能優化**: 🚀 Phase 3 Lazy Evaluation（時間軸零計算）+ 記憶體預分配 + 效能監控
- **優化器**: SOAP 為預設，支援 Adam/AdamW/SGD/L-BFGS + 多種 scheduler
- **權重策略**: GradNorm 動態權重 + Causal Weighting + Curriculum/Staged Weights
- **先驗整合**: RANS/LES 等低保真場作為軟約束（可加權/空間加權）
- **感測器**: QR-Pivot + 時間序列佈點 (K ≤ 400)
- **訓練系統**: TrainerBuilder/TrainerComponents + Checkpoint/Validation Manager + 配置驗證工具
- **分散式訓練**: 🆕 自動 DDP 支援（多 GPU 加速 ~1.7x）

## 快速開始

```bash
# 1. 環境設置（uv）
uv sync

# 2. 配置 WandB（必須，僅需一次）
echo "WANDB_API_KEY=your_key_here" > .wandb_config

# 3. 感測器生成（LES 選點 + DNS values）
uv run python scripts/generate/sensors/generate_kolmogorov_temporal_qr.py \
  --input data/kolmogorov_les/kolmogorov_les_re100.npy \
  --output data/kolmogorov_sensors/re100 \
  --K 400 --time-range 0 20 --time-stride 10 \
  --include-dns-values

# 4. 驗證配置（必跑，Fail Fast）
uv run python scripts/tools/validate_config_keys.py configs/kolmogorov_re50_kf4_K100.yml
uv run python scripts/tools/validate_config.py --config configs/main.yml

# 5. 訓練
# 5a. 單 GPU 訓練
uv run python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 5b. 多 GPU DDP 訓練（🆕 自動加速 ~1.7x）
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 6. 評估
# 6a. 快速評估（訓練中驗證，1-2 分鐘）
uv run python scripts/evaluate_unified.py \
  --checkpoint checkpoints/<exp>/best_model.pth \
  --output results/evaluation

# 6b. 多模型比較
uv run python scripts/evaluate_unified.py \
  --checkpoints checkpoints/model1.pth checkpoints/model2.pth checkpoints/model3.pth \
  --labels "RANS Prior" "Vanilla" "Proposed" \
  --output results/comparison

# 6c. 進階科學分析（論文前評估，5-10 分鐘，含能譜/壁剪應力）
uv run python scripts/evaluate/comprehensive_evaluation.py \
  --checkpoint checkpoints/<exp>/best_model.pth \
  --reference_dir data/jhtdb \
  --output results/comprehensive_eval
```

### Time Window 快速驗證

```bash
MPLCONFIGDIR=./.mplconfig PYTHONPATH=. \
  uv run python scripts/train/train.py --cfg configs/quick_test_full.yml
```

## 支援場景

### 2D Kolmogorov Flow ✅
- **用途**: 設計驗證、快速實驗（5-10 分鐘）
- **物理**: 不可壓縮 NS + 正弦強迫項
- **低保真先驗**: 2D LES（hyperviscosity + Ekman friction）
- **域**: 4π × 2π 週期域

### 3D Channel Flow Re_τ=1000 ✅
- **用途**: 工程級驗證、生產環境（2-8 小時）
- **物理**: 不可壓縮 NS + RANS k-ε 模型
- **域**: 8π × 2 × 3π（x/y/z）

## 感測器資料格式

- **JSON**: 感測器索引與座標（`sensor_file`）
- **NPZ**: DNS time series values（`dns_values_file`）

## 🚀 效能優化

### TorchScript Kernel Fusion (2026-01-16)
- **加速效果**: 1.035x（在 Tesla P100 上提升 3.5%）
- **優化目標**: SiLU/Swish 激活函數的融合優化
- **狀態**: ✅ 已部署並驗證
- **使用方式**: 自動啟用（配置 `activation: 'swish'` 時）
- **詳細說明**: [TORCHSCRIPT_OPTIMIZATION_GUIDE.md](docs/TORCHSCRIPT_OPTIMIZATION_GUIDE.md)

### P100 GPU 優化指南
- **硬體限制與已驗證方案**
  - ✅ **有效**: TorchScript kernel fusion（+3.5%）
  - ❌ **無效**: AMP/FP16（P100 無 Tensor Cores）
  - ❌ **不支援**: torch.compile()（需 Compute Capability ≥ 7.0）
- **硬體升級 ROI 分析**（V100 vs A100）
- **未來優化方向**（Larger batch size, architectural changes）
- **詳細說明**: [P100_OPTIMIZATION_GUIDE.md](docs/P100_OPTIMIZATION_GUIDE.md)

### 累積優化成果
| 優化項目 | 效果 | 狀態 |
|---------|------|------|
| WandB 同步頻率 | 50x 減少 | ✅ 已部署 |
| Loss 日誌頻率 | 2x 減少 | ✅ 已部署 |
| 配置驗證 | ~100x 加速 | ✅ 已部署 |
| 日誌文件大小 | 6.7x 減少 | ✅ 已完成 |
| **TorchScript Fusion** | **+3.5% 加速** | ✅ **已部署** |
| **總體預期** | **~3-5% 更快** | ⏳ **驗證中** |

## 文檔索引

| 文檔 | 用途 | 讀者 |
|------|------|------|
| [QUICK_START.md](docs/QUICK_START.md) | 完整工作流程 | 新用戶 |
| [DDP_GUIDE.md](docs/DDP_GUIDE.md) | 多 GPU 分散式訓練 🆕 | 效能優化 |
| [TORCHSCRIPT_OPTIMIZATION_GUIDE.md](docs/TORCHSCRIPT_OPTIMIZATION_GUIDE.md) | TorchScript 優化指南 🚀 | 效能優化 |
| [P100_OPTIMIZATION_GUIDE.md](docs/P100_OPTIMIZATION_GUIDE.md) | P100 硬體優化策略 🔧 | 效能優化 |
| [TRAINERBUILDER_GUIDE.md](docs/TRAINERBUILDER_GUIDE.md) | TrainerBuilder 使用指南 ✨ | 開發者 |
| [CONFIG_GUIDE.md](docs/CONFIG_GUIDE.md) | 配置參數說明與管理 | 配置調整 |
| [EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) | 評估工具使用指南 🔬 | 所有用戶 |
| [API_REFERENCE.md](docs/API_REFERENCE.md) | API 文檔 | 開發者 |
| [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | 問題診斷 | 調試 |
| [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md) | 系統架構 | 深入理解 |
| [TIME_WINDOW_TRAINING_GUIDE.md](docs/TIME_WINDOW_TRAINING_GUIDE.md) | Time Window 訓練指南 | 長時間序列 |

## 驗收指標

- 流場誤差 ≤ 10-15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 400 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

## 重要更新

### 🚀 Phase 3 優化：Fourier Features Lazy Evaluation (2025-01-07)

針對 `HybridFourierFeatures` 實現智能計算優化：
- **Lazy Evaluation**: `type='none'` 的軸（如時間軸 t）實現零計算開銷
- **記憶體優化**: 預分配輸出 tensor，避免動態 `torch.cat` 的記憶體複製
- **效能監控**: 內建統計功能追蹤實際計算節省
- **預期效益**: 對於典型 2D+T 配置（t, x, y），節省 ~33% Fourier 計算開銷

**快速啟用**:
```yaml
model:
  fourier_features:
    type: hybrid
    axes:
      0: {type: 'none'}        # 時間軸：零計算
      1: {type: 'periodic', ...}  # 空間軸：完整 Fourier
      2: {type: 'periodic', ...}
```

詳見: [Phase 3 優化報告](context/session_logs/PHASE3_OPTIMIZATION_REPORT_2025-01-07.md)

---

完整變更請見 [CHANGELOG.md](CHANGELOG.md)。

## 已知限制

- **記憶體**: 3D 訓練對 GPU/CPU 記憶體要求高，請從較低 PDE 點數開始
- **梯度檢查點**: 高階導數下可能不穩定，預設關閉 `model.use_gradient_checkpointing`
- **JHTDB 存取**: 需可用的 API token 與網路連線

## 貢獻與授權

歡迎 Issue/PR，遵循標準 Fork & PR 流程。授權：MIT。

研究引用請註明本專案：
```bibtex
@software{pinns_sparse_flow_2026,
  title={PINNs-SparseFlow: Sparse Sensor Turbulence Reconstruction},
  author={Li, JunYi},
  year={2026},
  version={1.4.0},
  url={https://github.com/latteine1217/pinns-sparse-flow}
}
```

---

**完整文檔**: [docs/README.md](docs/README.md) | **更新日誌**: [CHANGELOG.md](CHANGELOG.md)
