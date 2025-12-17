# 🌊 PINNs-MVP: 基於物理資訊神經網路的湍流場重建

**少量資料 × 物理先驗：Kolmogorov Flow 與 JHTDB 通道流的 PINNs 逆重建**

[![研究](https://img.shields.io/badge/研究-PINNs逆問題-blue)](https://github.com/latteine/pinns-mvp)
[![資料來源](https://img.shields.io/badge/資料-Kolmogorov_Flow_DNS-green)](docs/KOLMOGOROV_DNS_GUIDE.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![硬體支援](https://img.shields.io/badge/硬體-CUDA%20%7C%20MPS%20%7C%20CPU-yellow)](A100_DEPLOYMENT_GUIDE.md)
[![狀態](https://img.shields.io/badge/狀態-積極開發中-success)](README.md)

> **使命**：從極度稀疏的感測器觀測中，重建高保真 2D/3D 湍流場；第一階段以自建 **Kolmogorov Flow DNS** 作為設計與收斂驗證基準，第二階段擴展至 **JHTDB 通道流 ($Re_\tau \approx 1000$)**，並結合 RANS 低保真場作為物理軟先驗。

---

## 🔄 最新重大更新 (v1.1.0 - 2025-12-17)

**✅ Phase 2 程式碼清理完成** - 移除已棄用的低保真功能 (-774 行)

### 📌 專案聚焦策略
本專案現聚焦於 **2 個核心場景**，確保程式碼簡潔且可維護：

#### ✅ 支援場景
1. **2D Kolmogorov Flow** + Leith 湍流模型
   - 變數：u, v, nu_t（無壓力場）
   - 格式：HDF5 (.h5)
   
2. **3D Channel Flow (Re_tau=1000)** + RANS k-ε 湍流模型
   - 變數：u, v, w, p, k, epsilon, nu_t
   - 格式：HDF5 (.h5)

#### ❌ 已移除功能（v1.1.0）
- LES 模型支援
- DNS 降採樣處理
- NetCDF 格式 (.nc) 支援
- 統計一致性損失（Statistical Consistency Loss）
- 守恆定律損失（Conservation Loss）
- 對稱性一致性損失（Symmetry Consistency Loss）

### 📝 變更詳情
- **移除類別**：6 個已棄用類別（-774 行代碼）
- **簡化 API**：`create_lowfi_loader()` 移除 `filter_type` 參數
- **損失管理器簡化**：`PriorLossManager` 現僅管理 `LowFidelityConsistencyLoss`
- **測試結果**：核心模組 100% 測試通過，零回歸

### 🔗 完整文檔
- 📖 [CHANGELOG.md](CHANGELOG.md) - 詳細變更紀錄與遷移指南
- 📚 [docs/PROJECT_SCOPE.md](docs/PROJECT_SCOPE.md) - 完整專案範圍說明
- 📊 [docs/PHASE2_COMPLETION_REPORT.md](docs/PHASE2_COMPLETION_REPORT.md) - Phase 2 完整報告

### ⚠️ 向後不相容變更
若您從 v1.0.0 升級，以下導入將產生 `ImportError`：
```python
# ❌ 已移除
from pinnx.losses.priors import StatisticalConsistencyLoss
from pinnx.dataio import NetCDFReader, LESReader, DownsampledDNSProcessor
```

詳見 [CHANGELOG.md](CHANGELOG.md) 的遷移指南。

---

## 🌊 Phase 1: Kolmogorov Flow DNS Datasets (2D Benchmark)

We have generated and validated a comprehensive suite of Direct Numerical Simulation (DNS) datasets for 2D Kolmogorov Flow, serving as the sandbox and "Ground Truth" for PINNs reconstruction and training-stability studies.

### Dataset Overview

| Dataset | Grid Resolution | Re (Target) | Re (Actual) | State | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `dns_re50_t100.h5` | 256x256 | 50 | 35.7 | Transitional | Bursting phenomena, quasi-periodic |
| `dns_re70_t100.h5` | 256x256 | 70 | 60.6 | Weak Turbulence | Richer vortex structures |
| `dns_re100_t100.h5` | 256x256 | 100 | 105.9 | **Turbulence** | **Primary Benchmark**. Fully developed turbulence. |
| `dns_re500_t100.h5` | 512x512 | 500 | 1617.7 | **Strong Turbulence** | **Inverse Cascade**. Large-scale structures dominate. |

### Physics Validation ✅

All datasets have passed rigorous physics validation (`scripts/validation/validate_dns_physics.py`):
- **Incompressibility**: Divergence error $\nabla \cdot u \approx 10^{-14}$ (Spectral), $< 10^{-4}$ (Finite Difference).
- **Resolution**: $\Delta x / \eta < 0.5$ for all cases (Standard requires $< 2.5$), ensuring high-fidelity capture of small scales.
- **Stationarity**: Verified quasi-steady state statistics for t > 40s.

### Visualization
Detailed visualization reports (animations, energy spectra) are available in `results/dns_re*_viz/`.
Use `scripts/visualize/visualize_kolmogorov_dns.py` for custom visualizations.

---

## 📌 項目速覽
- 針對稀疏觀測的湍流逆問題，結合物理約束與神經網路實現穩健重建：先在 2D Kolmogorov Flow 上驗證設計，再遷移到 3D JHTDB 通道流（含 RANS 低保真軟先驗）。
- Fourier-SIREN MLP + VS-PINN + GradNorm/因果訓練，兼顧高頻細節與梯度穩定。
- **Random Weight Factorization (RWF)**：提升深層網路訓練穩定性。
- **RANS Prior Integration**：支援低保真 RANS 場作為軟先驗，透過 `lowfi_prior` 配置動態調整一致性權重。
- **Curriculum Learning**：階段式損失權重調整，支援先驗衰減策略（強→弱），改善壓力梯度重建。
- 配置驅動：所有實驗由 YAML 控制；自動裝置選擇支援 CUDA/MPS/CPU。
- 感測器佈局離線優化（QR-Pivot/DEIM）；DNS 生成與 Re 校準工具內建。

---

## 🚀 快速開始

```bash
# 1) 取得程式碼與環境
git clone https://github.com/latteine/pinns-mvp.git
cd pinns-mvp
conda env create -f environment.yml
conda activate pinns-mvp

# 2) 生成或下載 Kolmogorov Flow DNS
python scripts/generate/dns/generate_kolmogorov_dns.py \
  --N 512 --nu 0.0125 --k_f 8 --T_end 40.0 \
  --output data/kolmogorov_dns_re56_512x512_kf8_midway.h5

# 3) 驗證 DNS 物理守恆
python scripts/validation/validate_dns_physics.py \
  --input data/kolmogorov_dns_re56_512x512_kf8_midway.h5

# 4) 生成感測點（V7 QR-Pivot）
python scripts/generate/sensors/generate_sensors_periodic_qr.py \
  --dns-path data/kolmogorov_dns_re56_512x512_kf8_midway.h5 \
  --K 100 --oversample-factor 3.0 \
  --output data/sensors_K100_v7.npz

# 5) 執行訓練（配置驅動）
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 6) 評估與視覺化
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --config configs/kolmogorov_re50_kf4_K100.yml

python scripts/visualize/visualize_results.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --output results/visualizations/
```

**詳細文檔**：
- 📖 [快速入門指南](docs/QUICK_START.md) - 完整工作流程
- 📚 [技術文檔](docs/TECHNICAL_DOCUMENTATION.md) - 系統架構
- 🛠️ [腳本使用指南](scripts/README.md) - 30 核心工具（已分類至 8 個子目錄）
- ⚙️ [配置參考](docs/CONFIG_REFERENCE.md) - YAML 配置說明
- 🐛 [疑難排解](docs/TROUBLESHOOTING.md) - 常見問題

---

## 核心技術深度解析

本專案結合多項技術，解決湍流重建的高頻、多尺度與梯度剛性挑戰，並在 2D Kolmogorov Flow 與 3D JHTDB 通道流上共用同一套架構設計。

### 1. 模型架構: Fourier-SIREN MLP + RWF
- **傅立葉特徵**：先將 `(t, x, y, z)` 映射到高維頻域，消除 MLP 的頻譜偏差。
- **正弦激活 (SIREN)**：`sin(ωx)` 使高階導數平滑，適合 PDE 殘差計算。
- **Random Weight Factorization (RWF)**：分解權重為方向與長度 ($W=\exp(s) \cdot V$)，改善深層網路的優化風景與收斂性（PirateNet 論文）。
- **Adaptive Residual 機制**：透過 `block_type='resnet'` 啟用 PirateNet-style Adaptive Skip Connection (`y = α·f(x) + (1-α)·x`)，從淺層（α≈0）逐步增深網路。

### PINNs 內部運作流 (Internal Workflow)
1. **輸入準備**：原始座標與監測點物理量 → `UnifiedNormalizer` 標準化；VS-PINN 依各向異性縮放座標。
2. **神經網路**：`fourier_mlp.PINNNet` (with RWFLinear) 進行 Fourier 特徵映射與 SIREN MLP；內建 `fourier_normalize_input` 確保頻率穩定。
3. **輸出後處理**：`OutputTransform` 將標準化輸出還原至物理量 (u, v, w, p)。
4. **損失與權重**：`residuals.py` 計算 PDE/邊界/數據損失，`priors.py` 提供均值約束；`weighting.py` 中的 GradNorm/因果/自適應排程平衡權重。
5. **優化流程**：彙總總損失 → 反向傳播 → `Adam/LBFGS/SOAP` 更新；學習率由 `WarmupCosineScheduler` 等策略調度。
6. **檢查點與物理驗證**：保存模型與標準化參數前執行物理一致性檢查。

### 2. 物理引擎: 變數縮放 PINN (VS-PINN)
- **非等向縮放**：`(X, Y, Z) = (N_x·x, N_y·y, N_z·z)`，以大幅放大壁法向梯度 (例 `N_y=12`)，平衡剛性。
- **鏈式法則修正**：PDE 殘差計算時對導數加入縮放係數，如 `∂u/∂x = N_x · ∂u/∂X`。
- **效果**：反向傳播時各向梯度均衡，提升收斂性與穩定度。

### 3. 數據策略: QR 分解感測器佈局
- **QR-Pivoting**：對 DNS 快照矩陣進行列主元 QR，挑選資訊量最高的行作為感測器位置。
- **離線生成**：訓練前產生感測器位置文件，訓練僅讀取這些點作監督信號。

### 4. 訓練策略: 自適應權重、課程學習、因果訓練、RANS 先驗
- **GradNorm**：動態平衡各損失項的梯度範數，避免單一損失主導。
- **因果權重**：`w(t) = exp(-ε × ∫₀ᵗ Loss(τ)dτ)`，強化早期時間點的物理約束，配置中啟用 `losses.causal_weighting`。
- **課程學習**：多階段調整學習率與損失權重，先擬合數據再強化物理，再精修細節。支援 RANS 先驗衰減策略（Stage 1: prior_weight=1.0 → Stage 2: 0.3 → Stage 3: 0.1）。
- **RANS 低保真先驗**：`PriorLossManager` 動態調整 `consistency_weight`，將 RANS 平均場作為軟約束，改善壓力梯度重建（目標 ∇p L2 < 30%）。

---

## 總體工作流程

```mermaid
graph TD
    A[Kolmogorov DNS 數據] --> B[QR-Pivot 離線分析]
    B --> C[生成最優感測器位置]
    C --> D[訓練數據載入器]
    A --> D
    D --> E[模型訓練循環]

    E --> E1[座標輸入]
    E1 --> E2[Fourier-SIREN MLP + RWF]
    E2 --> E3[預測流場 u,v,w,p]
    E3 --> L1[數據損失]
    E3 --> L2[物理殘差 VS-PINN]
    E3 --> L3[邊界條件]
    L1 --> E5[GradNorm 動態加權]
    L2 --> E5
    L3 --> E5
    E5 --> E6[總損失]
    E6 --> E7[反向傳播與優化]
    E7 --> E8{收斂?}
    E8 -->|否| E1
    E8 -->|是| G[重建的完整流場]

    F[課程學習調度器] -.-> E7
```

---

## 📚 文檔導航

### 核心文檔（5 個精簡指南）
- 📖 [快速入門](docs/QUICK_START.md) - 完整工作流程與常用命令
- 📚 [技術文檔](docs/TECHNICAL_DOCUMENTATION.md) - 系統架構與設計原理
- ⚙️ [配置參考](docs/CONFIG_REFERENCE.md) - YAML 配置說明
- 🔧 [API 參考](docs/API_REFERENCE.md) - 腳本使用說明
- 🐛 [疑難排解](docs/TROUBLESHOOTING.md) - 常見問題與診斷流程

### 專題指南
- 🔬 DNS 生成：`docs/archive/KOLMOGOROV_DNS_GUIDE.md`
- 🎯 感測器優化：`docs/archive/QR_SENSOR_VISUALIZATION_GUIDE.md`
- 🧮 雷諾數計算：`scripts/README_REYNOLDS_CALCULATOR.md`
- 📈 課程學習：`docs/archive/KOLMOGOROV_CURRICULUM_GUIDE.md`
- 🖥️ A100 部署：`A100_DEPLOYMENT_GUIDE.md`

### 配置模板
- 🏃 2D 快速基準：`configs/templates/2d_quick_baseline.yml` (5-10 min)
- 🔍 2D 消融實驗：`configs/templates/2d_medium_ablation.yml` (15-30 min)
- 📚 3D 課程學習：`configs/templates/3d_slab_curriculum.yml` (30-60 min)
- 🎯 3D 論文級結果：`configs/templates/3d_full_production.yml` (2-8 hrs)

詳見 `configs/templates/README.md` 獲取完整模板說明。

## 📈 更新歷程

### ✅ Phase 2: 程式碼清理與專案聚焦 (2025-12-17)
- **移除已棄用功能**：6 個類別 (-774 行)，聚焦 2D Kolmogorov + 3D Channel Flow
- **簡化 API**：移除 `filter_type`、`statistical_weight`、`conservation_weight` 等參數
- **文檔完善**：新增 `CHANGELOG.md`、`PROJECT_SCOPE.md`、`PHASE2_COMPLETION_REPORT.md`
- **測試驗證**：核心模組 100% 測試通過，零回歸
- **詳細內容**：見上方「最新重大更新」區塊

### ✅ Momentum Merging 功能上線 (2025-12-15)
- **損失項簡化**：針對各向同性流動（如 Kolmogorov Flow）新增 `merge_momentum` 參數
  - **損失項減少**：PDE 損失從 3 項 → 2 項（-33%）
  - **物理動機**：合併 X/Y 動量方程為單一向量範數，強制各向同性約束
  - **配置支援**：所有 17 個 Kolmogorov Flow 實驗配置已啟用 `losses.merge_momentum: true`
  - **測試驗證**：100% 測試覆蓋率，包含集成測試與單元測試
- **使用方式**：
  ```yaml
  losses:
    merge_momentum: true  # Kolmogorov Flow (各向同性)
    # merge_momentum: false  # Channel Flow (各向異性)
  ```
- **效益**：
  - 簡化 GradNorm 權重調整（2 項 vs 3 項）
  - 強制 X/Y 動量同步收斂，符合物理對稱性
  - 訓練穩定性提升（減少權重失衡風險）
- **文檔**：詳見 `docs/MOMENTUM_MERGING_GUIDE.md` 與 `MOMENTUM_MERGING_TEST_REPORT.md`

### ✅ Trainer 重構完成 (2025-12-14)
- **代碼質量提升**：`Trainer` 類別完成 Phase 1-4 全面重構
  - **74% 行數減少**：4 個核心方法從 971 行 → 251 行
  - **模組化設計**：新增 `TrainingLoopManager` 類別 + 17 個 helper methods
  - **零回歸**：所有測試通過，功能完全保留
  - **可維護性**：單一職責原則，每個方法聚焦單一任務
- **重構細節**：詳見 `REFACTORING_COMPLETE_GUIDE.md` 完整文檔
  - Phase 1: `step()` (785 → 92 lines, -88%)
  - Phase 2: `train()` (371 → 92 lines, -75%)
  - Phase 3: `validate()` (71 → 21 lines, -70%)
  - Phase 4: `save_checkpoint()` (158 → 46 lines, -71%)

### ✅ 專案整理與文檔重構 (2025-12-13)
- **文檔精簡**：33 個冗長文檔 → 5 個核心文檔（85% 減量）
  - 📖 快速入門、📚 技術文檔、⚙️ 配置參考、🔧 API 參考、🐛 疑難排解
  - 舊文檔移至 `docs/archive/` 供參考
- **腳本分類**：30 核心腳本組織至 8 個功能子目錄
  - `train/` (1), `evaluate/` (4), `visualize/` (6), `generate/dns/` (3), `generate/sensors/` (2)
  - `calculate/` (2), `compare/` (3), `tools/` (3), `validation/` (7), `debug/` (5)
  - 23 個歷史腳本歸檔至 `scripts/archive/`（實驗性/一次性/重複功能）
- **路徑更新**：所有文檔與配置已更新為新目錄結構

### ✅ RANS Prior Loss Integration (2025-12-12)
- **實作完成**：`PriorLossManager` 完整整合至 `trainer.py`，支援動態一致性權重調整
- **配置支援**：`lowfi_prior` 配置區塊，可設定 `consistency_weight` 與 `variable_weights`
- **訓練驗證**：已驗證 prior loss 正確計算並記錄至訓練日誌與 TensorBoard

### ✅ Curriculum Learning with Prior Decay (2025-12-12)
- **策略設計**：3 階段損失權重調整（固定 Re=50）
  - Stage 1 (0-300): 強先驗引導 (prior_weight=1.0, PDE=0.5)
  - Stage 2 (300-700): 平衡先驗與物理 (prior_weight=0.3, PDE=1.0)
  - Stage 3 (700-1000): 弱先驗精煉 (prior_weight=0.1, PDE=1.0)
- **配置檔案**：`configs/kolmogorov_re50_kf4_K100_rans_curriculum.yml`
- **目標指標**：壓力梯度 L2 error < 30% (vs 目前 ~100%)

### ⚠️ 已知限制
- **記憶體需求**：MPS (Apple Silicon) 在 10K PDE 點時可能 OOM（20GB 限制），建議降低至 5K 或使用 CUDA GPU

## 📈 Roadmap / Future Work

- **架構消融實驗**：系統性比較 Fourier-VS-PINN baseline、僅 RWF、僅 adaptive residual、與完整組合，在 2D Kolmogorov 與 3D JHTDB 通道流上量化各元件貢獻（收斂速度與最終誤差）。
- **不確定性量化 (UQ)**：在現有 RWF + VS-PINN 架構上整合 B-PINNs、NN-aPC 或 ensemble PINNs，輸出帶置信區間的重建結果。
- **進階架構**：探索 Kolmogorov–Arnold Networks（KAN）等更具表達力的 backbone，並與現有 Fourier-SIREN + RWF 組合比較。
- **自適應採樣與感測**：將 QR-DEIM 型自適應 collocation 與 randomized QRCP 感測設計整合進訓練流程，提升在更高 $Re_\tau$ 與更嚴苛感測預算下的可擴展性。
- **RANS Prior 完整評估**：完成 1000 epoch 訓練並與固定權重基準（`kolmogorov_re50_kf4_K100_rans_prior_1k.yml`）進行對比分析。

## 貢獻與授權
- 歡迎 Issue/PR，遵循一般 Fork & PR 流程。
- 授權：MIT。若在研究中引用，請註明本專案網址。
