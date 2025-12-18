# 🌊 PINNs-MVP: 稀疏測量湍流重建

基於物理資訊神經網路 (PINNs) 的湍流逆問題求解器，從極少量感測器觀測 (K ≤ 100) 重建高保真流場。

[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![硬體](https://img.shields.io/badge/硬體-CUDA%20%7C%20MPS%20%7C%20CPU-yellow)](A100_DEPLOYMENT_GUIDE.md)
[![狀態](https://img.shields.io/badge/狀態-積極開發-success)](README.md)

**應用場景**: 工程實務中僅有低保真模擬 (RANS/LES) 與稀疏測量，需重建全場流動。
**驗證基準**: 2D Kolmogorov Flow (設計驗證) → 3D JHTDB Channel Flow Re_τ=1000 (工程級驗證)。

---

## 🔄 重要更新

### 🚨 v1.1.1 (2025-12-18): Channel Flow 標準化修復 ✅

**問題**: 3D Channel Flow 訓練後 v/w/p 場誤差達 1000-2000% (災難性失敗)  
**原因**: Sensor 文件僅含座標，訓練 fallback 到損壞的 RANS prior (穩態 k-ω SST 導致 v/w≈0)  
**修復**: 使用 K=100 稀疏測量點計算標準化統計，誤差降至 100-200% (合理範圍)  
**工具**: 新增 `extract_sensor_values_from_dns.py` + 自動數據質量驗證  
**影響**: ✅ 僅 3D Channel Flow 受影響；**2D Kolmogorov Flow 已驗證不受影響**

📖 **詳細分析**: [ROOT_CAUSE_FINAL.md](results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md) | [CORRECT_SOLUTION_K100.md](results/channel_flow_evaluation/CORRECT_SOLUTION_K100.md)  
📊 **Kolmogorov 驗證**: [KOLMOGOROV_NORMALIZATION_CHECK.md](KOLMOGOROV_NORMALIZATION_CHECK.md)

---

### ✅ v1.1.0 (2025-12-17): 程式碼清理 (-774 行)

**聚焦策略**: 僅支援 2 個核心場景
- **2D Kolmogorov Flow** (u, v, nu_t) + Leith 模型
- **3D Channel Flow Re_τ=1000** (u, v, w, p, k, ε, nu_t) + RANS k-ε

**移除功能**: LES 支援、DNS 降採樣、NetCDF 格式、統計/守恆/對稱性損失
**API 變更**: 簡化 `create_lowfi_loader()` 與 `PriorLossManager`
**測試**: 核心模組 100% 通過，零回歸

📖 **完整紀錄**: [CHANGELOG.md](CHANGELOG.md) | [PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md) | [PHASE2_COMPLETION_REPORT.md](docs/PHASE2_COMPLETION_REPORT.md)

---

## 📊 DNS 數據集

自建 2D Kolmogorov Flow DNS 作為設計驗證基準 (Ground Truth)，涵蓋 Transitional → Turbulence → Inverse Cascade 全流態。

| Dataset | Grid | Re (實際) | 狀態 | 用途 |
|---------|------|----------|------|------|
| `dns_re50_t100.h5` | 256² | 35.7 | Transitional | 低 Re 基準 |
| `dns_re100_t100.h5` | 256² | 105.9 | **Turbulence** | **主要基準** |
| `dns_re500_t100.h5` | 512² | 1617.7 | Strong Turbulence | 高 Re 挑戰 |

**物理驗證** (已通過): 不可壓縮性 ∇·u<10⁻⁴ | 解析度 Δx/η<0.5 | 準穩態統計
**詳細文檔**: [KOLMOGOROV_DNS_GUIDE.md](docs/archive/KOLMOGOROV_DNS_GUIDE.md)

---

## ⚡ 核心特性

- **架構**: Fourier-SIREN MLP + Random Weight Factorization (RWF) + VS-PINN
- **優化**: GradNorm 動態權重 + Curriculum Learning + 因果訓練
- **先驗整合**: RANS 低保真場作為軟約束 (`lowfi_prior` 配置)
- **感測器**: QR-Pivot 離線優化佈局 (K ≤ 100)
- **配置驅動**: YAML 控制實驗，自動裝置選擇 (CUDA/MPS/CPU)

---

## 🚀 快速開始

```bash
# 1. 環境設置
conda env create -f environment.yml && conda activate pinns-mvp

# 2. 訓練 (使用現有 DNS 數據)
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 3. 評估
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/<exp>/best_model.pth \
  --config configs/<exp>.yml
```

**完整工作流程** (DNS 生成 → 感測器優化 → 訓練 → 評估): 參見 [QUICK_START.md](docs/QUICK_START.md)

---

## 🔬 技術架構

結合多項技術處理湍流重建的高頻、多尺度與梯度剛性挑戰。

### 核心元件
1. **Fourier-SIREN MLP + RWF**: 傅立葉特徵消除頻譜偏差 + SIREN 平滑高階導數 + 權重分解改善優化
2. **VS-PINN**: 非等向縮放 (N_x, N_y, N_z) 平衡各向梯度，特別針對壁面法向剛性
3. **QR-Pivot 感測器**: 列主元 QR 分解挑選最大資訊量測點
4. **動態訓練**: GradNorm 權重平衡 + 因果權重 + Curriculum Learning + RANS 先驗衰減

### 訓練流程
```
DNS 數據 → QR-Pivot 感測點 → Fourier-SIREN+RWF → VS-PINN 物理殘差
→ GradNorm 動態權重 → 反向傳播 (Adam/LBFGS/SOAP) → 全場重建
```

**詳細架構**: [TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)

---

## 📚 文檔導航

### 核心文檔
- 📖 [快速入門](docs/QUICK_START.md) - 完整工作流程
- 📚 [技術文檔](docs/TECHNICAL_DOCUMENTATION.md) - 系統架構
- ⚙️ [配置參考](docs/CONFIG_REFERENCE.md) - YAML 配置
- 🔧 [腳本參考](scripts/README.md) - 工具使用
- 🐛 [疑難排解](docs/TROUBLESHOOTING.md) - 問題診斷

### 專題指南
DNS 生成 · 感測器優化 · 雷諾數計算 · 課程學習 · A100 部署 → 詳見 `docs/archive/`

### 配置模板
`configs/templates/`: 2D 基準 (5-10min) | 2D 消融 (15-30min) | 3D Slab (30-60min) | 3D Production (2-8hrs)

---

## ⚠️ 已知限制

- **記憶體**: MPS (Apple Silicon) 在 10K PDE 點時可能 OOM (20GB 限制)，建議降至 5K 或使用 CUDA GPU
- **兼容性**: v1.1.0 移除多項已棄用功能，詳見 [CHANGELOG.md](CHANGELOG.md)

---

## 📈 Roadmap

- **架構消融**: 系統比較 Fourier/RWF/Adaptive Residual 各元件貢獻
- **不確定性量化**: 整合 B-PINNs/NN-aPC/Ensemble 輸出置信區間
- **進階架構**: 探索 KAN 等新型 backbone
- **自適應採樣**: QR-DEIM 自適應 collocation 與動態感測器佈局
- **RANS 先驗評估**: 完整評估先驗衰減策略效果

---

## 貢獻與授權

歡迎 Issue/PR，遵循標準 Fork & PR 流程。授權：MIT。研究引用請註明本專案網址。
