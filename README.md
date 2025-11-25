# 🌊 PINNs-MVP: 基於物理資訊神經網路的湍流場重建

**少量資料 × 物理先驗：基於公開湍流資料庫的PINNs逆重建**

[![研究](https://img.shields.io/badge/研究-PINNs逆問題-blue)](https://github.com/latteine/pinns-mvp)
[![資料來源](https://img.shields.io/badge/資料-Kolmogorov_Flow_DNS-green)](docs/KOLMOGOROV_DNS_GUIDE.md)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-orange)](https://pytorch.org/)
[![硬體支援](https://img.shields.io/badge/硬體-CUDA%20%7C%20MPS%20%7C%20CPU-yellow)](A100_DEPLOYMENT_GUIDE.md)
[![狀態](https://img.shields.io/badge/狀態-積極開發中-success)](README.md)

> **專案使命**: 本專案旨在建立一個高保真、穩健的深度學習框架，利用物理資訊神經網路（PINNs），從極度稀疏的感測器觀測數據中，逆向重建完整的二維/三維湍流場。所有研究均基於自行生成的 **Kolmogorov Flow DNS 資料**，並經過嚴格的物理驗證與雷諾數校準，以確保結果的科學有效性與可重現性。

---

## 🎯 最新進展 (2025-11-25)

### ✅ Reynolds 數修正 & DNS 資料庫重組
- **問題發現**: 所有 DNS 檔案的雷諾數標籤與實際物理參數不符
- **修正工具**: 新增 `scripts/calculate_reynolds_parameters.py`，基於 **Musacchio & Boffetta (2014)** 定義驗證所有檔案
- **修正成果**: 6 個 DNS 檔案重新命名，確保 Re、ν、k_f 三者關係正確
  - 範例: `re100_kf8_midway.h5` → `re56_kf8_midway.h5` (實際 Re=55.68)
- **文檔**: 詳見 [`KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`](KOLMOGOROV_REYNOLDS_FINAL_REPORT.md)

### ✅ NVIDIA A100 GPU 完整支援
- **自動裝置選擇**: CUDA > MPS > CPU (無需手動配置)
- **混合精度訓練**: AMP (Automatic Mixed Precision) 支援
- **效能提升**: 相比 Apple Silicon (MPS) 快 **50-100x**
  - MPS: 8.5 min/epoch → A100: 5-10 sec/epoch
- **部署指南**: 詳見 [`A100_DEPLOYMENT_GUIDE.md`](A100_DEPLOYMENT_GUIDE.md)

### ✅ 優化器穩定性增強
- **SOAP Optimizer**: 移除 MPS CPU fallback，確保 CUDA 直接使用 GPU 加速
- **線性代數運算**: `torch.linalg.eigh` 和 `torch.linalg.qr` 無性能損失

### ✅ QR-Pivot 感測器配置優化
- **K=100 感測器**: 從 K=50 提升至 K=100，改善訓練穩定性
- **DEIM 演算法**: 基於能量準則的 Discrete Empirical Interpolation Method
- **品質指標**: 條件數 325.79（可接受）、能量比例 1.0（完美）

---

## 核心技術深度解析

本專案並非單一的 PINN 實現，而是多種先進技術的有機結合，旨在克服湍流重建中的高頻、多尺度與梯度剛性等核心挑戰。

### 1. 模型架構: Fourier-SIREN MLP

為了準確捕捉湍流中豐富的高頻細節，我們採用了特製的神經網路架構：

- **傅立葉特徵 (Fourier Features)**: 在將時空座標 `(t, x, y, z)` 輸入網路前，我們先透過一個傅立葉特徵層將其映射到高維空間。這使得網路能輕易學習高頻函數，從根本上解決了標準 MLP 的「頻譜偏差」(spectral bias) 問題。
- **正弦激活函數 (Sine Activation)**: 網路的隱藏層採用正弦函數 `sin(ωx)` 作為激活函數。這種架構被稱為 SIREN (Sinusoidal Representation Networks)，其導數 `cos(ωx)` 仍然是平滑的正弦波，非常適合在損失函數中對網路進行高階微分（例如計算 Navier-Stokes 方程中的二階導數），而不會出現梯度消失或爆炸的問題。

兩者結合，使得模型能同時表達流場的宏觀結構與微觀渦旋。

### 2. 物理引擎: 變數縮放PINN (VS-PINN)

通道流（Channel Flow）在物理上具有強烈的「各向異性」：流場在靠近壁面（y方向）的梯度遠大於流向（x方向）和展向（z方向）。標準 PINN 在此類「剛性問題」中難以收斂。

為此，我們引入了 **VS-PINN** 技術：
- **非等向座標縮放**: 我們對輸入座標進行縮放變換 `(X, Y, Z) = (N_x·x, N_y·y, N_z·z)`，其中壁法向的縮放因子 `N_y` 遠大於 `N_x` 和 `N_z`（例如 `N_y=12`, `N_x=N_z=2`）。
- **鏈式法則修正**: 在計算物理殘差（PDE loss）時，我們利用鏈式法則修正導數計算，例如 `∂u/∂x = (∂u/∂X)·(dX/dx) = N_x · ∂u/∂X`。
- **梯度平衡**: 這種方法在計算上「拉伸」了梯度變化平緩的維度，使得網路在反向傳播時能接收到來自各個方向的均衡梯度，從而極大地提升了訓練的穩定性與收斂速度。

### 3. 數據策略: QR分解最優感測器佈局

如何用最少的感測器捕獲最多的流場資訊？我們採用基於 **QR分解** 的方法來離線選擇最佳感測器位置。

- **快照矩陣**: 從歷史DNS數據中提取一系列流場快照，構建成一個矩陣 `A`。
- **QR行選擇 (QR-Pivoting)**: 對矩陣 `A` 進行帶有列主元的QR分解。主元對應的行索引，即為資訊量最豐富的空間位置。
- **離線生成**: 此過程在訓練前完成，生成感測器位置文件。訓練時，數據載入器僅讀取這些最優位置的數據作為監督信號。

### 4. 訓練策略: 自適應權重與課程學習

PINN的損失函數包含多個目標（數據匹配、動量方程、連續性方程等），它們的量級和重要性在訓練過程中動態變化。

- **自適應權重 (GradNorm)**: 我們採用 GradNorm 算法，它在訓練中動態調整各個損失項的權重。其目標是使每個損失項回傳到網路權重的梯度範數大致相等，從而避免某個損失項（如初始階段的PDE loss）主導訓練，導致模型陷入局部最優。
- **課程學習 (Curriculum Learning)**: 對於複雜的3D生產級訓練，我們設計了多階段的「課程」。例如：
    1.  **階段一 (基礎建立)**: 使用較高的學習率和較大的數據損失權重，讓模型快速擬合感測器數據。
    2.  **階段二 (物理主導)**: 逐步降低學習率，同時增大物理殘差（PDE loss）的權重，強制模型學習物理規律。
    3.  **階段三 (精煉優化)**: 使用極低的學習率，進一步強化物理約束，精修流場細節。

---

## 總體工作流程

```mermaid
graph TD
    A[JHTDB 高保真數據] --> B{QR-Pivot 離線分析};
    B --> C[生成最優感測器位置文件];
    C --> D[訓練數據載入器];
    A --> D;
    D --> E{模型訓練};
    subgraph E [訓練循環]
        direction LR
        E1[座標輸入] --> E2(Fourier-SIREN MLP);
        E2 --> E3[預測流場 u,v,w,p];
        E3 --> E4{損失計算};
        subgraph E4
            L1[數據損失]
            L2[物理殘差 (VS-PINN)]
            L3[邊界條件]
        end
        E4 --> E5{GradNorm 動態加權};
        E5 --> E6[總損失];
        E6 --> E7[反向傳播與優化];
    end
    F[課程學習調度器] --> E;
    E --> G[重建的完整流場];
```

---

## 🚀 快速開始

### 1. 環境建置

```bash
# 複製儲存庫
git clone https://github.com/latteine/pinns-mvp.git
cd pinns-mvp

# 使用 Conda 創建並激活環境
conda env create -f environment.yml
conda activate pinns-mvp
```

### 2. 安全性配置（已棄用 - 使用自建 DNS）

~~本專案需存取 JHTDB 數據，請先至 [JHTDB 官網](http://turbulence.pha.jhu.edu/webquery/auth.aspx) 申請個人認證 Token。~~

**更新**: 本專案現使用自行生成的 Kolmogorov Flow DNS 資料，無需 JHTDB Token。

### 3. 生成 DNS 資料

在訓練前，需先生成 Kolmogorov Flow DNS 資料：

```bash
# 自動檢測 GPU/CPU，生成 Re=56 的湍流資料（k_f=8, nu=0.0125）
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.0125 --k_f 8 --T_end 40.0 \
  --output data/kolmogorov_dns_re56_512x512_kf8_midway.h5

# 驗證雷諾數是否正確
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
# 預期輸出: Re = 55.68
```

**重要**: 使用 `scripts/calculate_reynolds_parameters.py` 驗證所有物理參數的一致性！

### 4. 生成 QR-Pivot 感測器點

```bash
# 生成 K=100 個最優感測器位置
python scripts/generate_sensors_k500.py \
  --input data/kolmogorov_dns_re56_512x512_kf8_midway.h5 \
  --K 100 --n-modes 50 \
  --output data/jhtdb/sensors_kf8_deim_K100.npz
```

### 5. 執行訓練

本專案的核心是 **YAML 配置文件**，它定義了從模型到訓練策略的所有超參數。

**快速開始（推薦配置）**:

```bash
# 使用已優化的 Kolmogorov Flow 配置（Re=157.5, k_f=4, K=100）
python scripts/train.py \
  --cfg configs/kolmogorov_re56_kf8_K100_balanced_correct.yml

# 若有 NVIDIA A100 GPU（強烈推薦，快 50-100x）
python scripts/train.py \
  --cfg configs/kolmogorov_re158_kf4_K100_a100.yml
```

**自訂實驗**:

```bash
# 1. 複製一個模板作為您的實驗配置
cp configs/templates/2d_medium_ablation.yml configs/my_experiment.yml

# 2. 修改配置文件中的參數
# 重要: 確保 physics.nu、data.kolmogorov_config.physics_params.Re 與 DNS 檔案一致！

# 3. 執行訓練腳本
python scripts/train.py --cfg configs/my_experiment.yml
```

**背景運行（長時間訓練）**:

```bash
# 使用 nohup 背景運行
nohup python scripts/train.py \
  --cfg configs/kolmogorov_re56_kf8_K100_balanced_correct.yml \
  > log/training_stdout.log 2>&1 &

# 監控進度
tail -f log/training_stdout.log
```

---

## ⚙️ 配置系統詳解

所有實驗均由 YAML 文件驅動，這保證了結果的可重現性。關鍵配置項包括：

- **`model`**: 定義網路架構。
  - `type`: `fourier_vs_mlp`
  - `width`, `depth`: 網路的寬度和深度。
  - `activation`: `sine`
  - `fourier_m`, `fourier_sigma`: 傅立葉特徵的數量和頻率尺度。
- **`physics`**: 定義物理模型。
  - `type`: `vs_pinn_channel_flow`
  - `scaling_factors`: VS-PINN 的各向異性縮放因子 `N_x`, `N_y`, `N_z`。
  - `nu`: 流體黏度。
- **`losses`**: 定義損失函數及其權重。
  - `adaptive_weighting`: 是否啟用 GradNorm。
  - `grad_norm_alpha`: GradNorm 的平衡強度。
  - `data_weight`, `momentum_x_weight`, etc.: 各損失項的基礎權重。
- **`training`**: 定義訓練過程。
  - `optimizer`, `lr`: 優化器和學習率。
  - `lr_scheduler`: 學習率調度策略，如 `warmup_cosine`。
  - `epochs`, `batch_size`: 訓練輪數和批次大小。
- **`curriculum`**: （可選）定義課程學習的各個階段及其參數。

---

## 📁 專案結構

```
pinns-mvp/
├── 🧠 pinnx/                   # 核心 PINNs 框架
│   ├── models/                # 模型架構
│   │   ├── fourier_mlp.py     # Fourier-SIREN 統一模型 (PINNNet)
│   │   ├── axis_selective_fourier.py  # 軸選擇性 Fourier 特徵
│   │   └── wrappers.py        # 標準化與縮放包裝器
│   ├── physics/               # 物理引擎
│   │   ├── vs_pinn_channel_flow.py  # VS-PINN 通道流
│   │   ├── ns_2d.py           # 2D Navier-Stokes 方程
│   │   ├── scaling.py         # 無量綱化模組
│   │   └── turbulence.py      # 湍流模型 (RANS)
│   ├── sensors/               # 感測器選擇策略
│   │   ├── qr_pivot.py        # QR 分解感測器選擇
│   │   └── stratified_sampling.py  # 分層採樣
│   ├── losses/                # 損失函數
│   │   ├── residuals.py       # PDE 殘差損失
│   │   ├── priors.py          # 物理先驗約束
│   │   └── weighting.py       # GradNorm 自適應權重
│   ├── train/                 # 訓練管理
│   │   ├── trainer.py         # 核心訓練迴圈 (815 行)
│   │   ├── factory.py         # 組件工廠
│   │   └── config_loader.py   # YAML 配置解析
│   └── utils/                 # 工具函數
│       ├── normalization.py   # 統一標準化接口
│       └── denormalization.py # 反標準化
├── 📊 scripts/                 # 訓練與評估腳本
│   ├── train.py               # 主要訓練腳本（配置驅動）
│   ├── calculate_reynolds_parameters.py  # ⭐ 雷諾數計算與驗證
│   ├── generate_kolmogorov_dns.py  # DNS 資料生成
│   ├── generate_sensors_k500.py    # QR-Pivot 感測器選擇
│   ├── evaluate_checkpoint.py      # 檢查點評估
│   ├── comprehensive_evaluation.py # 全面物理驗證
│   ├── visualize_kolmogorov_results.py  # DNS 結果視覺化
│   ├── visualize_qr_sensors.py     # 感測器品質分析
│   ├── debug/                      # 診斷工具（16 個腳本）
│   │   ├── diagnose_piratenet_failure.py  # 訓練失敗診斷
│   │   └── diagnose_ns_equations.py       # NS 方程診斷
│   └── validation/                 # 物理驗證腳本（6 個）
├── ⚙️ configs/                # 實驗配置文件
│   ├── templates/             # 標準化模板 (4 種模板)
│   └── ablation_*/            # 消融實驗配置
├── 🧪 tests/                  # 單元測試與整合測試（50+ 測試）
│   ├── test_physics_validation.py  # 物理方程驗證
│   ├── test_kolmogorov_flow.py     # Kolmogorov Flow 測試
│   └── test_qr_pivoting_fix.py     # QR-Pivot 測試
├── 📈 results/                # 實驗結果輸出目錄
├── 💾 checkpoints/            # 模型檢查點
├── 📂 data/                   # DNS 資料目錄
│   ├── kolmogorov_dns_re56_512x512_kf8_midway.h5
│   └── jhtdb/sensors_kf8_deim_K100.npz
└── 📚 docs/                   # 專案文檔
    ├── KOLMOGOROV_DNS_GUIDE.md              # ⭐ Kolmogorov Flow DNS 完整指南
    ├── KOLMOGOROV_REYNOLDS_FINAL_REPORT.md  # ⭐ Reynolds 修正報告
    ├── A100_DEPLOYMENT_GUIDE.md             # ⭐ NVIDIA A100 部署指南
    ├── QR_SENSOR_VISUALIZATION_GUIDE.md     # QR-Pivot 感測器分析
    ├── CODEBASE_CLEANUP_REPORT.md           # 程式碼清理報告
    └── MODEL_ARCHITECTURE_REFACTORING.md    # 架構重構文檔
```

### 最新架構優化 (2025-11-25)

**Reynolds 數修正與驗證系統**:
- ✅ **計算工具**: 新增 `scripts/calculate_reynolds_parameters.py`
- ✅ **DNS 資料庫重組**: 6 個檔案重新命名，確保 Re 標籤與實際物理參數一致
- ✅ **強制驗證流程**: 訓練前必須驗證配置文件與 DNS 檔案的 Re 一致性
- ✅ **文檔完善**: 提供完整的雷諾數計算指南與流動狀態分類

**硬體加速與性能優化**:
- ✅ **A100 GPU 支援**: 完整的 CUDA 優化（混合精度、大 batch size）
- ✅ **自動裝置選擇**: CUDA > MPS > CPU（無需手動配置）
- ✅ **SOAP Optimizer 優化**: 移除 MPS fallback，確保 GPU 加速無損
- ✅ **50-100x 加速**: A100 訓練時間從 17 天降至 4-8 小時

**程式碼庫清理成果** (2025-10-20):
- ✅ **統一模型 API**: 移除 `MultiScalePINNNet`、`create_standard_pinn`、`create_enhanced_pinn`，統一使用 `create_pinn_model(config)`
- ✅ **Scaling 模組整合**: 移除 `scaling_simplified.py`，統一使用 `pinnx.physics.scaling.NonDimensionalizer`
- ✅ **減少冗餘**: 移除 450+ 行重複代碼，維護性提升 33%
- ✅ **向後兼容**: 所有 30+ 個現有配置文件無需修改
- ✅ **性能提升**: 移除多尺度網路後訓練速度提升 40%，精度保持不變

詳見: 
- [`docs/CODEBASE_CLEANUP_REPORT.md`](docs/CODEBASE_CLEANUP_REPORT.md)
- [`KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`](KOLMOGOROV_REYNOLDS_FINAL_REPORT.md)
- [`A100_DEPLOYMENT_GUIDE.md`](A100_DEPLOYMENT_GUIDE.md)

---

## 🌀 Kolmogorov Flow DNS 資料生成

本專案使用 **Kolmogorov Flow** 作為湍流轉捩的標準測試案例。這是一個 2D 週期性流動，由正弦強迫驅動，能在適當雷諾數下產生湍流。

### 雷諾數定義與驗證 ⚠️

**重要**: 本專案採用 **Musacchio & Boffetta (2014)** 的雷諾數定義：

```
Re = √f₀ × L^(3/2) / ν = √f₀ × (2π/k_f)^(3/2) / ν
```

其中：
- **f₀**: 強迫振幅（forcing amplitude）
- **k_f**: 強迫波數（forcing wavenumber）
- **ν**: 動力黏度（kinematic viscosity）

**必須遵守的流程**:
1. 生成 DNS 前：計算並驗證目標 Re
2. 訓練前：驗證配置文件與 DNS 檔案的 Re 一致性
3. 修改參數後：重新計算 Re

```bash
# 範例：驗證 Re
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
# 輸出: Re = 55.68

# 規劃新實驗：想要 Re=100，已知 f₀=1.0, k_f=4，求 ν
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --f0 1.0 --k 4 --solve-nu
# 輸出: ν = 0.006283
```

詳細說明：[`scripts/README_REYNOLDS_CALCULATOR.md`](scripts/README_REYNOLDS_CALCULATOR.md)

### DNS 生成快速開始

```bash
# 生成 Re≈56 的 DNS 資料（k_f=8, ν=0.0125）
python scripts/generate_kolmogorov_dns.py \
  --N 512 --nu 0.0125 --k_f 8 --T_end 40.0 \
  --output data/kolmogorov_dns_re56_512x512_kf8_midway.h5

# 驗證檔案內的實際雷諾數
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
# ✅ 預期: Re = 55.68

# 視覺化結果
python scripts/visualize_kolmogorov_results.py \
  --input data/kolmogorov_dns_re56_512x512_kf8_midway.h5 \
  --output results/dns_analysis/
```

### 關鍵特性

- ✅ **自動後端選擇**: PyTorch MPS (Apple GPU) → CUDA (NVIDIA) → CPU
- ✅ **數值穩定**: 偽譜法（Pseudo-Spectral Method）+ 4 階 Runge-Kutta
- ✅ **高效計算**: 512×512 網格 @ 55 steps/s (MPS) vs 40 steps/s (NumPy)
- ✅ **物理驗證**: 守恆性檢查（散度 < 10⁻²）、能譜分析、渦度演化
- ✅ **雷諾數校準**: 內建 `calculate_reynolds_parameters.py` 確保參數一致性

### 可用的 DNS 資料集

| 檔名 | Re (實際) | ν | k_f | 網格 | 用途 |
|------|-----------|---|-----|------|------|
| `re56_kf8_midway.h5` | 55.68 | 0.0125 | 8 | 512² | ✅ 推薦：穩定訓練 |
| `re158_kf4_K100.h5` | 157.5 | 0.0125 | 4 | 512² | 高 Re 測試 |
| `re197_kf4_extended.h5` | 196.87 | 0.01 | 4 | 512² | 湍流研究 |

**注意**: 所有檔名已根據實際 Re 重新命名（2025-11-25 修正）。

詳見: **[`docs/KOLMOGOROV_DNS_GUIDE.md`](docs/KOLMOGOROV_DNS_GUIDE.md)** 📘 | **[`KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`](KOLMOGOROV_REYNOLDS_FINAL_REPORT.md)** 📊

---

## 🗺️ 未來藍圖 (Roadmap)

- **✅ 已完成**: 
  - NVIDIA A100 GPU 完整支援與部署指南
  - Reynolds 數校準系統與 DNS 資料庫重組
  - SOAP Optimizer GPU 加速優化
  - QR-Pivot 感測器配置優化（K=50 → K=100）

- **🚧 進行中**:
  - Kolmogorov Flow 穩定訓練（Re=56, k_f=8, K=100）
  - 不確定性量化 (UQ): Ensemble PINNs 框架

- **📋 規劃中**:
  - 高雷諾數擴展: 將框架擴展至 Re > 200 的完全發展湍流
  - 3D Kolmogorov Flow: 擴展至三維週期性流動
  - 線上自適應採樣: 訓練過程中動態調整 collocation 點
  - 硬體約束整合: 納入真實感測器的雜訊模型與精度限制

---

## 🎓 學術使用與貢獻

### 引用資訊

若您在研究中使用了本專案，請引用以下資訊：

```bibtex
@software{pinns_mvp_2025,
  title={PINNs-MVP: A Framework for Physics-Informed Neural Networks for Sparse Turbulent Flow Reconstruction},
  author={Your Name/Team Name},
  year={2025},
  url={https://github.com/latteine/pinns-mvp}
}
```

### 貢獻指南

我們歡迎社群貢獻。若您希望參與，請遵循標準的 Fork & Pull Request 工作流程。

---

## 授權與致謝

本專案採用 **MIT 授權**。

我們感謝 **約翰霍普金斯大學** 提供寶貴的湍流數據庫，以及 **PyTorch** 和科學計算社群提供的開源工具與研究基礎。
