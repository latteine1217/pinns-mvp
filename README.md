# 🌊 PINNs-MVP: 基於物理資訊神經網路的湍流場重建

**少量資料 × 物理先驗：基於公開湍流資料庫的PINNs逆重建**

[![研究](https://img.shields.io/badge/研究-PINNs逆問題-blue)](https://github.com/latteine/pinns-mvp)
[![資料來源](https://img.shields.io/badge/資料-JHTDB通道流-green)](http://turbulence.pha.jhu.edu/)
[![狀態](https://img.shields.io/badge/狀態-積極開發中-success)](README.md)

> **專案使命**: 本專案旨在建立一個高保真、穩健的深度學習框架，利用物理資訊神經網路（PINNs），從極度稀疏的感測器觀測數據中，逆向重建完整的三維湍流場。所有研究均基於約翰霍普金斯湍流資料庫（JHTDB）的公開基準數據，以確保結果的科學有效性與可重現性。

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

### 2. 安全性配置 (JHTDB Token)

本專案需存取 JHTDB 數據，請先至 [JHTDB 官網](http://turbulence.pha.jhu.edu/webquery/auth.aspx) 申請個人認證 Token。

```bash
# 複製環境變數範本
cp .env.example .env

# 編輯 .env 文件，填入您的 JHTDB token
# JHTDB_AUTH_TOKEN=your-actual-token-here
```

### 3. 執行訓練

本專案的核心是 **YAML 配置文件**，它定義了從模型到訓練策略的所有超參數。我們提供了一系列模板。

```bash
# 1. 複製一個模板作為您的實驗配置
cp configs/templates/2d_quick_baseline.yml configs/my_first_experiment.yml

# 2. (可選) 修改配置文件中的參數
# vim configs/my_first_experiment.yml

# 3. 執行訓練腳本
python scripts/train.py --cfg configs/my_first_experiment.yml
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
│   ├── train.py               # 主要訓練腳本
│   ├── comprehensive_evaluation.py  # 全面評估
│   ├── debug/                 # 診斷工具
│   └── validation/            # 物理驗證腳本
├── ⚙️ configs/                # 實驗配置文件
│   ├── templates/             # 標準化模板 (4 種模板)
│   └── ablation_*/            # 消融實驗配置
├── 🧪 tests/                  # 單元測試與整合測試 (90%+ 覆蓋率)
├── 📈 results/                # 實驗結果輸出目錄
└── 📚 docs/                   # 專案文檔
    ├── CODEBASE_CLEANUP_REPORT.md      # 程式碼清理報告
    ├── MODEL_ARCHITECTURE_REFACTORING.md  # 架構重構文檔
    └── SCALING_MODULE_CONSOLIDATION.md # Scaling 模組整合
```

### 最新架構優化 (2025-10-20)

**程式碼庫清理成果**:
- ✅ **統一模型 API**: 移除 `MultiScalePINNNet`、`create_standard_pinn`、`create_enhanced_pinn`，統一使用 `create_pinn_model(config)`
- ✅ **Scaling 模組整合**: 移除 `scaling_simplified.py`，統一使用 `pinnx.physics.scaling.NonDimensionalizer`
- ✅ **減少冗餘**: 移除 450+ 行重複代碼，維護性提升 33%
- ✅ **向後兼容**: 所有 30+ 個現有配置文件無需修改
- ✅ **性能提升**: 移除多尺度網路後訓練速度提升 40%，精度保持不變

詳見: `docs/CODEBASE_CLEANUP_REPORT.md`

---

## 🗺️ 未來藍圖 (Roadmap)

- **不確定性量化 (UQ)**: 實現基於 Ensemble 的 PINNs 訓練，量化預測結果的不確定性。
- **高雷諾數擴展**: 將當前框架擴展至更高雷諾數（`Re > 2000`）的湍流場景。
- **即時處理優化**: 針對模型與算法進行性能優化，探索線上即時重建的可能性。
- **硬體約束整合**: 研究如何將真實世界的硬體（如感測器類型、精度限制）約束納入模型。

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
