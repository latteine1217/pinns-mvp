# 🎯 Agent 角色定位

- **Role**: 資深 AI Engineer & 物理資訊機器學習 (SciML) 專家
- **Specialty**: PyTorch 架構設計、流體力學逆問題、高維度優化策略

---

# 🎯 專案目標

## 研究主題

- 稀疏測量湍流場重建（Sparse-Data Turbulence Reconstruction）

## 工程場景（研究實驗對應）

- 現實工程：RANS/LES + 極少量真實量測 → 全場逆推
- 研究驗證：用 DNS 取樣生成 sensor observations（作為「量測真值」的替代），並以 DNS 全場作為對照基準

## 驗收指標（維持原定義）

- 流場誤差 ≤ 10–15%（相對 L2）
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點（QR-Pivot）
- 收斂速度提升 ≥ 30%

# 專案重要規則
- 本專案將在伺服器上運行，使用指令 `ssh junyi@140.114.120.128` 來登入伺服器
- 使用的伺服器環境為：
    - #SBATCH --time=14-00:00:00
    - #SBATCH --partition=r740
    - #SBATCH --mem=108G
    - #SBATCH --gres=gpu:2 (兩張Nvidia P100)

## 技術規範（摘要版）

- 架構：8×256 MLP + Fourier (m=12, σ=4.0) + RWF
- 優化：SOAP → L-BFGS
- 物理：VS-PINN N=(2,12,2) + Source Term + 無因次化

## 🔄 標準工程流程（Canonical Workflow）

0. 任務接收
   - 理解意圖 → 確認邊界 → 提出計畫 → 等待確認
   - 原則：未經確認不得直接動手
1. 問題診斷（Fail Fast）
   - 訓練穩定性（NaN / 梯度 / 權重）
   - 感測器品質（覆蓋率 / 條件數）
   - 物理一致性（BC / 守恆 / ∇·u）
2. 實驗設計
   - 嚴格控制變因（seed / sensor / steps）
   - 先 2D（Kolmogorov）→ 再 3D（Channel）
   - 所有對比依據 `EXPERIMENT_COMPARISON_PLAN.md`
3. 程式修改
   - Read → Plan → Incremental Edit → Immediate Test → Commit
   - 禁止：多點修改後才測試
4. 驗證層級
   - P0：Import / 單元測試
   - P1：物理正確性
   - P2：端到端訓練與視覺化
5. 文件同步
   - 配置變更一定更新：
     - `standard_config_template.yml`
     - `docs/CONFIG_GUIDE.md`
   - 重要決策一定記錄：`context/decisions/decisions_log.md`
6. 會話記錄（Session Documentation）
   - **所有會話總結必須保存至 `context/session_logs/`**
   - 命名格式：
     - `SESSION_SUMMARY_YYYY-MM-DD_<主題>.md`（會話總結）
     - `SESSION_LOG_YYYY-MM-DD_<主題>.md`（詳細日誌）
     - `<主題>_REPORT_YYYY-MM-DD.md`（技術報告）
   - 內容要求：
     - 問題診斷與解決方案
     - 關鍵決策與理由
     - 修改的文件清單
     - 量化成果（行數、效能等）
     - 下一步建議
   - **禁止將會話文件留在專案根目錄**

## ⚠️ 不可違反的工程鐵律（Hard Rules）

- Loss ≠ Accuracy，一切以後驗指標（L2 / RMSE / 物理量）為準
- DNS 的角色（研究設定）：
  1. DNS 提供感測點的真實觀測值（作為 sensor observations / pseudo-measurements）
  2. DNS 也同時用於 全場對照評估
  3. 真正工程場景下，DNS 由「真實量測」取代，但流程不變：prior + sparse sensors → reconstruction
- Config 是系統的一部分，只能用 losses:（複數），訓練前必跑 `validate_config_keys.py`
- 一次只修一個原因
- 所有修正必須可回溯、可解釋
- 縮進是語法：大改動後必做 import + AST；避免 Trainer 方法「消失」的歷史 bug 重演

---

## 📂 關鍵專案結構

### 核心模組 (`pinnx/`)

```
pinnx/
├── constants.py              # 物理常數管理（JHTDB 統計量、參考值）
├── README.md                 # 模組說明文檔
│
├── models/                   # 神經網路架構
│   ├── fourier_mlp.py       # 基礎 MLP (Fourier Features + SIREN)
│   ├── axis_selective_fourier.py # 軸向選擇性 Fourier Embedding
│   ├── hybrid_fourier.py    # 混合 Fourier 架構
│   ├── resnet.py            # ResNet 架構
│   └── wrappers.py          # 模型包裝器
│
├── physics/                  # 物理方程式與微分算子
│   ├── base/                # 基礎微分算子
│   │   ├── gradient_ops.py  # 梯度算子 (∇u, ∇p)
│   │   ├── laplacian_ops.py # Laplacian 算子 (∇²u)
│   │   ├── ns_base.py       # Navier-Stokes 基礎類別
│   │   └── pde_base.py      # PDE 基礎抽象類別
│   ├── gradient_cache.py    # 梯度快取優化 (Gradient Caching)
│   ├── ns_2d.py             # 2D NS 方程（Kolmogorov Flow）
│   ├── ns_3d_temporal.py    # 3D 時域 NS 方程
│   ├── kolmogorov_flow_2d.py # Kolmogorov Flow 專用物理
│   ├── vs_pinn_channel_flow.py # VS-PINN Channel Flow
│   ├── turbulence.py        # 湍流物理約束
│   ├── turbulence_utils.py  # 湍流工具函數
│   └── validators.py        # 物理驗證器
│
├── train/                    # 訓練系統
│   ├── trainer.py           # 核心 Trainer（主要訓練邏輯）
│   ├── trainer_builder.py   # TrainerBuilder (Builder Pattern)
│   ├── trainer_components.py # Trainer 組件
│   ├── time_window_trainer.py # Time Window 訓練器
│   ├── training_loop_manager.py # 訓練迴圈管理器
│   ├── validation_manager.py # 驗證管理器
│   ├── checkpoint_manager.py # Checkpoint 管理
│   ├── loss_manager.py      # 損失函數管理
│   ├── loss_factory.py      # 損失函數工廠
│   ├── model_physics_factory.py # 模型與物理工廠
│   ├── weighter_factory.py  # 權重調度器工廠
│   └── schedulers/          # 學習率與權重調度
│       ├── warmup_cosine.py # Warmup + Cosine Annealing
│       ├── warmup_exponential.py # Warmup + Exponential Decay
│       ├── curriculum.py    # Curriculum Learning
│       └── staged_weights.py # 階段式權重調度
│
├── losses/                   # 損失函數
│   ├── residuals.py         # PDE 殘差計算（主版本）
│   ├── residuals_vectorized.py # 向量化殘差
│   ├── residuals_vorticity.py # 渦度法殘差
│   ├── causal_weighter_v2.py # 因果加權 (Causal Weighting)
│   ├── priors.py            # Prior 損失（RANS/LES）
│   ├── weighting.py         # 權重調度策略
│   └── sdf_weights.py       # SDF 基礎權重
│
├── optim/                    # 優化器
│   ├── soap.py              # SOAP 優化器（主要優化器）
│   └── soap_utils.py        # SOAP 工具函數
│
├── sensors/                  # 感測器佈點策略
│   ├── qr_pivot/            # QR-Pivot 稀疏採樣
│   │   ├── base.py          # QR-Pivot 基礎類別
│   │   ├── factory.py       # QR-Pivot 工廠
│   │   ├── features.py      # 特徵提取
│   │   └── selectors/       # 選點策略
│   └── adaptive_collocation.py # 自適應配點
│
├── dataio/                   # 資料載入與介面
│   ├── jhtdb_client.py      # JHTDB 資料庫介面
│   ├── channel_flow_loader.py # Channel Flow 資料載入
│   ├── jhtdb_cutout_loader.py # JHTDB Cutout 載入
│   ├── lowfi_loader.py      # Low-fidelity 資料載入
│   ├── nondimensionalization.py # 無因次化
│   ├── loaders/             # 資料集 Loaders
│   │   ├── kolmogorov.py    # Kolmogorov Flow 載入器
│   │   └── rans_prior.py    # RANS Prior 載入器
│   └── sampling/            # 採樣策略
│       ├── boundary.py      # 邊界採樣
│       └── interior.py      # 內部配點採樣
│
├── evals/                    # 評估模組
│   ├── metrics.py           # 評估指標（L2, RMSE, Conservation）
│   └── visualizer.py        # 視覺化工具
│
└── utils/                    # 工具函數
    ├── config_loader.py     # 配置載入
    ├── config_validator.py  # 配置驗證
    ├── config_snapshot.py   # 配置快照
    ├── physics_validator.py # 物理驗證
    ├── evaluation_utils.py  # 評估工具
    ├── memory_tracker.py    # 記憶體追蹤
    ├── timer.py             # 計時器
    ├── training_monitor.py  # 訓練監控
    └── normalization/       # 正規化模組
        ├── base_normalizer.py # 基礎正規化器
        ├── input_transform.py # 輸入轉換
        ├── output_transform.py # 輸出轉換
        └── kolmogorov_transform.py # Kolmogorov 專用轉換
```

### 工具腳本 (`scripts/`)

```
scripts/
├── train/
│   └── train.py             # 🚀 訓練主程式 (Entry Point)
│
├── evaluate_unified.py      # 🔍 快速評估工具 (訓練中/後驗證)
│   - 用途: Checkpoint 快速檢查 (L2, RMSE, Conservation)
│   - 支援多模型比較
│
├── evaluate/
│   ├── comprehensive_evaluation.py # 🔬 進階科學分析 (論文級)
│   │   - 用途: 能量譜、壁剪應力、速度剖面、湍流統計
│   └── README.md            # 評估指南
│
├── generate/                # 資料生成工具
│   ├── dns/                 # DNS/LES 地面真值生成
│   │   ├── generate_kolmogorov_dns.py # Kolmogorov DNS
│   │   └── generate_kolmogorov_les.py # Kolmogorov LES
│   └── sensors/             # 感測器佈點生成
│       ├── generate_kolmogorov_temporal_qr.py # QR-Pivot 採樣
│       └── generate_kolmogorov_temporal_random.py # 隨機採樣
│
├── tools/                   # 開發工具
│   ├── validate_config.py   # 配置完整性驗證
│   ├── validate_config_keys.py # ⚡ 配置鍵值檢查 (Fail Fast)
│   ├── extract_dns_snapshot.py # DNS 快照提取
│   ├── convert_h5_to_npy.py # 資料格式轉換
│   └── batch_update_experiment_configs.py # 批次更新配置
│
├── validation/              # 物理與資料驗證
│   ├── physics_validation.py # 物理守恆驗證
│   ├── validate_dns_physics.py # DNS 物理驗證
│   ├── validate_kolmogorov_reynolds.py # Kolmogorov Re 驗證
│   └── validate_ns_conservation.py # NS 守恆驗證
│
├── visualize/               # 視覺化工具
│   ├── visualize_kolmogorov_dns.py # Kolmogorov DNS 視覺化
│   └── visualize_kolmogorov_sensors.py # 感測器佈點視覺化
│
├── experiments/             # 實驗腳本
│   ├── run_s1_sensor_sweep_slurm.sh # S1: 感測器策略掃描
│   ├── run_s2_k_scan_slurm.sh # S2: K 值掃描
│   └── run_B1_series.sh     # B1: Baseline 系列
│
└── calculate/               # 參數計算工具
    ├── calculate_reynolds_parameters.py # Reynolds 參數計算
    └── calculate_lowfi_parameters.py # Low-fi 參數計算
```

### 配置與文檔

```
configs/                      # 實驗配置文件 (YAML)
├── standard_config_template.yml # 標準配置模板（必讀）
├── kolmogorov_re50_kf4_K100.yml # Kolmogorov 標準配置
├── experiments/             # 實驗配置系列
│   ├── S1_sensor_strategy/  # 感測器策略實驗
│   ├── S2_k_scan/           # K 值掃描實驗
│   ├── A1_ablation_fourier/ # Fourier 消融實驗
│   ├── A2_ablation_weights/ # 權重消融實驗
│   ├── C1_prior_comparison/ # Prior 對比實驗
│   └── M1_model_comparison/ # 模型對比實驗
└── sweeps/                  # WandB Sweep 配置
    ├── sweep_s1_sensor_strategy.yaml
    └── sweep_s2_k_scan.yaml

docs/                         # 完整開發文檔
├── README.md                # 文檔索引
├── QUICK_START.md           # 🚀 快速入門
├── CONFIG_GUIDE.md          # 配置參數詳解
├── TRAINERBUILDER_GUIDE.md  # TrainerBuilder 架構說明
├── EVALUATION_GUIDE.md      # 評估策略指南
├── EXPERIMENT_DESIGN.md     # 實驗設計指南
├── TIME_WINDOW_TRAINING_GUIDE.md # Time Window 訓練
├── DDP_GUIDE.md             # 分散式訓練指南
├── COLAB_QUICK_START.md     # Colab 快速開始
├── P100_OPTIMIZATION_GUIDE.md # P100 GPU 優化
├── TROUBLESHOOTING.md       # 故障排除
└── API_REFERENCE.md         # API 參考文檔

context/                      # 會話與技術紀錄
├── session_logs/            # 會話記錄 (Session Logs & Decision Records)
├── technical_reviews/       # 技術審查文檔
├── DDP_INTEGRATION_PLAN.md  # DDP 整合計畫
└── JAXPI_LR_SCHEDULER_COMPARISON.md # 學習率調度器對比

data/                         # 資料目錄（.gitignore）
├── jhtdb/                   # JHTDB Channel Flow 資料
├── kolmogorov/              # Kolmogorov Flow 資料
│   ├── dns/                 # DNS 地面真值
│   ├── les/                 # LES 地面真值
│   └── sensors/             # 感測器佈點
└── rans/                    # RANS Prior 資料

checkpoints/                  # 模型 Checkpoint（.gitignore）
results/                      # 實驗結果（.gitignore）
wandb/                        # WandB 日誌（.gitignore）
```

### 測試目錄 (`tests/`)

```
tests/                        # 單元測試與整合測試
├── test_*.py                # 單元測試檔案
└── integration/             # 整合測試
```

---

## 🔧 常用命令快查

### 評估
```bash
# 快速評估（訓練中）
python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth

# 多模型比較
python scripts/evaluate_unified.py \
  --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
  --labels "RANS" "Vanilla" "Proposed"

# 進階科學分析（論文前）
python scripts/evaluate/comprehensive_evaluation.py \
  --checkpoint checkpoints/best_model.pth \
  --reference_dir data/jhtdb \
  --output results/final_eval
```

### 配置驗證
```bash
# 鍵驗證（Fail Fast）
python scripts/tools/validate_config_keys.py configs/your_config.yml

# 完整驗證
python scripts/tools/validate_config.py --config configs/your_config.yml
```

### 訓練
```bash
# 標準訓練
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# Time Window 訓練
python scripts/train/train.py --cfg configs/quick_test_full.yml
```
