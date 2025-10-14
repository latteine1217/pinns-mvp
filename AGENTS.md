# 開發者指引 👨‍💻

## 🎯 角色扮演準則
> 當執行專案任務時，請扮演一位資深的artificial intellegence engineer，具備以下特質：
> - 專精python語言，尤其擅長使用pytorch設計網路架構
> - 🔍 **類型安全優先**
> - ⚡ **效能導向**
> - 🧪 **測試驅動**: 重視程式碼品質，推崇文檔覆蓋
> - 🔄 **簡潔架構**
> - 仔細思考，只執行我給你的具體任務，用最簡潔優雅的解決方案，盡可能少的修改程式碼

# 專案研究題目
少量資料 × 物理先驗：基於公開湍流資料庫的PINNs逆重建
Sparse-Data, Physics-Informed Inversion on Public Turbulence Benchmarks: Reconstruction

# 專案目標
本研究試著建立「**少量資料點 × 低保真引導 × 源項還原**」之 PINNs 逆問題框架；以 JHTDB/UT Austin 的湍流資料（通道流 Re_\tau=1000)為基準場，從 **K（≤30–80）個空間點**與/或短時間序列出發，在**低保真場（RANS／粗 LES／下採樣 DNS）作為軟先驗下，重建瞬時或統計穩態的速度與壓力場**。

量化門檻係參考下列已證實的里程碑與最新進展而制定：

- **稀疏觀測→全場重建與壓力場可推回**：以 HFM/PINN 脈絡證實可由稀疏可視化/速度資料重建速度與壓力（作為本研究「少量點重建」的可行性下限參考）
- **感測點選擇與穩健性**：採用 **QR-pivot 最適感測配置**作為「最少點數 K 」的設計與驗證基準。
- **訓練效率與剛性處理**：引入 **VS-PINN（變數尺度化）** 與動態權重，將其作為「收斂效率」與「硬問題穩定化」的可重現強基線。
- **資料來源與重現性**：使用 **JHTDB**（官方建議的 SciServer/pyJHTDB 取數路徑）做為高保真基準與對照資料的權威來源。

據此背景設定以下**可驗證成功門檻**（對應上列基準的能力）：

1. **流場誤差**：速度/壓力場相對 L2 error ≤ **10–15%**（依案例分層），且相較低保真之**統計/能譜/壁面剪應力**的 RMSE **下降 ≥30%**（對齊「低→高保真校正」的主流做法）
2. **可識別性（MPS, 最少點數 K）**：在量測噪聲 **σ=1–3%**、隨機遺失 **10%** 下，達標點數 **K ≤ 50**；K–誤差曲線以 **QR-pivot 佈局**為最小可行上界的對照線
3. **效率/穩健**：相較固定權重基線，採 **VS-PINN + 動態權重**使收斂 epoch **下降 ≥30%**；UQ 方差與真實誤差相關 **r ≥ 0.6**（以 ensemble PINN 量測）。
4. **資料重現性**：所有評測以 **JHTDB cutout/散點取樣**為標準流程，並遵循官方引用與再現規範。

# 目錄結構（實際）

```
pinns-mvp/
  ├─ configs/                      # 訓練配置 (30+ YAML 檔案)
  │   ├─ defaults.yml              # 預設配置
  │   ├─ channel_flow_*.yml        # 通道流系列配置
  │   ├─ vs_pinn_*.yml             # VS-PINN 系列配置
  │   ├─ inverse_*.yml             # 逆問題配置
  │   └─ curriculum_*.yml          # 課程學習配置
  │
  ├─ data/
  │   ├─ lowfi/                    # RANS/粗LES（NetCDF/HDF5/npz）
  │   └─ jhtdb/                    # 高保真 cutout/散點取樣快取
  │
  ├─ scripts/                      # 可執行腳本 (30 核心 + 歸檔)
  │   ├─ train.py ⭐               # 主訓練器（支援 curriculum）
  │   ├─ evaluate.py               # 主評估腳本
  │   ├─ evaluate_checkpoint.py    # 檢查點評估
  │   ├─ evaluate_curriculum.py    # 課程學習評估
  │   ├─ comprehensive_evaluation.py # 完整物理驗證評估
  │   ├─ evaluate_3d_physics.py    # 3D 物理場評估
  │   │
  │   ├─ fetch_channel_flow.py     # JHTDB 資料獲取
  │   ├─ verify_jhtdb_data.py      # 資料驗證
  │   │
  │   ├─ visualize_results.py      # 增強視覺化工具
  │   ├─ visualize_adaptive_sampling.py # 自適應採樣視覺化
  │   ├─ generate_jhtdb_field_plots.py # JHTDB 場圖生成
  │   │
  │   ├─ monitor_training_progress.py # 通用訓練監控
  │   ├─ monitor_warmup_test.py    # Warmup 監控
  │   ├─ monitor_curriculum.sh     # 課程學習監控腳本
  │   ├─ monitor_curriculum_ic.sh  # IC 課程監控
  │   │
  │   ├─ parameter_sensitivity_experiment.py # 參數敏感度實驗
  │   ├─ k_scan_experiment.py      # K 掃描實驗
  │   ├─ analyze_k_scan.py         # K 掃描分析
  │   ├─ run_longterm_training.py  # 長期訓練管理
  │   ├─ benchmark.py              # 性能基準測試
  │   ├─ quick_benchmark.py        # 快速基準測試
  │   ├─ activation_benchmark.py   # 激活函數測試
  │   ├─ analyze_full_field_data.py # 全場資料分析
  │   ├─ detailed_field_analysis.py # 詳細場分析
  │   ├─ diagnose_channel_flow_characteristics.py # 通道流診斷
  │   │
  │   ├─ validate_constraints.py   # 約束條件驗證
  │   ├─ verify_model_scaling.py   # 模型尺度驗證
  │   ├─ verify_weights.py         # 損失權重驗證
  │   │
  │   ├─ debug/                    # 除錯工具 (15 個診斷腳本)
  │   │   ├─ diagnose_ns_equations.py ⭐ # 主要 NS 方程診斷
  │   │   ├─ diagnose_boundary_conditions.py
  │   │   ├─ diagnose_pressure_failure.py
  │   │   ├─ debug_autograd_issue.py
  │   │   ├─ debug_derivatives_computation.py
  │   │   ├─ debug_gradient_computation.py
  │   │   ├─ debug_physics_residuals.py
  │   │   ├─ diagnose_conservation_error.py
  │   │   ├─ diagnose_training_data.py
  │   │   ├─ diagnose_sensor_overfitting.py
  │   │   └─ ... (其他診斷工具)
  │   │
  │   ├─ validation/               # 物理驗證測試 (6 個)
  │   │   ├─ physics_validation.py
  │   │   ├─ test_channel_flow_experiment.py
  │   │   ├─ test_channel_flow_physics.py
  │   │   ├─ test_conservation_with_model.py
  │   │   ├─ validate_hybrid_sensors.py
  │   │   └─ validate_ns_conservation.py
  │   │
  │   └─ archive_*/                # 歸檔舊腳本 (45 個已棄用)
  │       ├─ archive_demos/        # 演示腳本 (3)
  │       ├─ archive_eval/         # 階段性評估 (11)
  │       ├─ archive_monitors/     # 階段性監控 (9)
  │       ├─ archive_sensors/      # 感測點生成 (10)
  │       ├─ archive_diagnostics/  # 過時診斷 (6)
  │       └─ archive_shell_scripts/ # 過時 Shell (9)
  │
  ├─ pinnx/                        # 核心模組
  │   ├─ __init__.py
  │   ├─ physics/
  │   │   ├─ channel_flow_3d.py    # 3D 通道流物理
  │   │   ├─ vs_pinn_channel_flow.py # VS-PINN 通道流（含縮放）
  │   │   ├─ navier_stokes_3d.py   # 通用 3D NS 方程
  │   │   ├─ hit_turbulence.py     # 均勻各向同性湍流
  │   │   └─ ... (其他物理模組)
  │   │
  │   ├─ models/
  │   │   ├─ fourier_mlp.py        # Fourier feature + sine MLP
  │   │   ├─ enhanced_fourier_mlp.py # 增強版 Fourier MLP
  │   │   └─ siren.py              # SIREN 模型
  │   │
  │   ├─ losses/
  │   │   ├─ adaptive_weights.py   # 自適應權重（GradNorm/NTK）
  │   │   ├─ curriculum_weights.py # 課程學習權重
  │   │   ├─ physics_residuals.py  # 物理殘差損失
  │   │   └─ data_losses.py        # 資料一致性損失
  │   │
  │   ├─ sensors/
  │   │   └─ qr_sampling.py        # QR-pivot 感測點選擇
  │   │
  │   ├─ dataio/
  │   │   ├─ jhtdb_loader.py       # JHTDB 資料載入
  │   │   ├─ channel_flow_loader.py # 通道流專用載入器
  │   │   └─ lowfi_loader.py       # 低保真資料載入
  │   │
  │   ├─ train/                    # 訓練管理模組 ⭐ [重構完成]
  │   │   ├─ trainer.py (815 行)   # 核心訓練器類別
  │   │   │   └─ Trainer: 管理完整訓練循環（優化器、動態權重、檢查點、驗證）
  │   │   ├─ ensemble.py           # Ensemble 訓練 + UQ
  │   │   ├─ loop.py               # 訓練循環工具函數
  │   │   ├─ adaptive_collocation.py # 自適應採樣
  │   │   ├─ checkpointing.py      # 檢查點管理
  │   │   ├─ config_loader.py      # 配置載入器
  │   │   └─ factory.py            # 模型/優化器工廠
  │   │
  │   └─ evals/
  │       ├─ metrics.py            # 評估指標
  │       └─ visualizers.py        # 視覺化工具
  │
  ├─ tests/                        # 單元測試與整合測試
  │   ├─ test_3d_physics.py        # 3D 物理測試
  │   ├─ test_losses.py            # 損失函數測試
  │   ├─ test_models.py            # 模型架構測試
  │   ├─ test_physics_validation.py # 物理驗證測試
  │   ├─ test_physics.py           # 物理模組測試
  │   ├─ test_metrics.py           # 指標計算測試
  │   └─ ... (其他測試)
  │
  ├─ context/                      # 主 Agent 決策記錄
  │   ├─ context_session_*.md      # 會話上下文
  │   └─ decisions_log.md          # 決策日誌
  │
  ├─ tasks/                        # 任務管理
  │   └─ TASK-<id>/               # 各任務目錄
  │       ├─ task_brief.md         # 任務簡述
  │       ├─ physics_review.md     # 物理審查
  │       ├─ impl_plan.md          # 實作計畫
  │       └─ ... (其他產出)
  │
  ├─ results/                      # 訓練結果
  ├─ log/                          # 訓練日誌
  ├─ checkpoints/                  # 模型檢查點
  ├─ environment.yml               # Conda 環境
  ├─ README.md                     # 專案說明
  ├─ TECHNICAL_DOCUMENTATION.md    # 技術文檔
  └─ AGENTS.md                     # 開發者指引（本文檔）
```

詳見 `scripts/README.md` 獲取完整腳本使用說明。

# 物理訊息神經網路(PINNs)架構、邏輯設定
- 網路大小: 8 x 200
- 輸入: x, y, z, t, (低精度RANS結果), (fourier feature)
- 輸出：u, v, w, p
- loss term基礎設置(詳細由adaptive weighting決定): w_{pde} 為基準設為1，規則為w_{data} = w_{boundary} = w_{initial} > w_{pde} 
- 數據集：JHTDB channel flow Re_\tau=1000
- 數據集大小：128 x 32 x 128
- 使用完整數據集中的QR pivoting選點作為監督訓練，訓練完成後用來預測完整數據集的流場誤差

# JHTDB channel flow Re_\tau=1000設定
| 類別        | 參數                                                            | 數值 / 說明                                                                                                                           |
| --------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| 幾何/網格     | 計算域                                                           | (L_x\times L_y\times L_z=8\pi h\times 2h\times 3\pi h)（(h=1)）【】                                                                   |
|           | 網格（頻域/實域）                                                     | (N_x\times N_y\times N_z=2048\times 512\times 1536)（wavemodes）；對應 collocation (3072\times 512\times 2304)；**儲存**以 wavemode 解析度。【】 |
| 流體/強迫     | 黏滯係數                                                          | (\nu=5\times10^{-5})（無因次）。【】                                                                                                      |
|           | 平均壓力梯度                                                        | (dP/dx=0.0025)（無因次）。【】                                                                                                            |
| 時間        | DNS 時步 / 資料庫時步                                                | (\Delta t=0.0013)；(\delta t=0.0065)。【】                                                                                            |
|           | 儲存時間範圍                                                        | (t\in[0,,25.9935])。【】                                                                                                             |
| 統計量       | 體積平均 / 中心線 / 摩擦速度                                             | (U_b=0.99994)、(U_c=1.1312)、(u_\tau=4.9968\times10^{-2})。【】                                                                        |
|           | 黏長尺度                                                          | (\delta_\nu=\nu/u_\tau=1.0006\times10^{-3})。【】                                                                                    |
|           | 雷諾數                                                           | (Re_b=3.9998\times10^4)，(Re_c=2.2625\times10^4)，(Re_\tau=9.9935\times10^2)。【】                                                     |
| 解析度（黏長單位） | (\Delta x^+) / (\Delta y_1^+) / (\Delta y_c^+) / (\Delta z^+) | (12.2639) / (1.65199\times10^{-2}) / (6.15507) / (6.13196)。【】                                                                     |
| 寫出/筆數     | Frame 數                                                       | 速度與壓力每 5 步寫出，共 **4000** frames。【】                                                                                                 |


# 評測與成敗門檻檢查

`scripts/evaluate.py` 需輸出：

1. **相對 L2**（u,v,p）≤ 10–15%（依案例分層）。
2. 相對於低保真：**統計（均值/二階動量）、能譜、壁面剪應力** 的 RMSE 下降 ≥ 30%。
3. **K–誤差曲線**（對比 QR-pivot 上界、隨機/等距佈點）。

`pinnx/evals/metrics.py` 可提供：

* `relative_L2(pred, ref)`
* `spectrum_rmse(pred, ref)`（1D/2D FFT 能譜）
* `wall_shear_stress(u,v)`（通道或壁面法向）
* `k_curve(exps)`（掃 K=4..16，畫 error vs. K）
* `uncertainty_correlation(mean, var, ref)`（var 的平方根 vs. |err| 的相關）

---

# 預期實現內容

* **QR-pivot**：用低保真或歷史快照矩陣就能挑 K 點，能穩健地縮減感測點需求
* **VS-PINN + 動態權重 + 因果權重**：同時改善收斂與剛性，對高 Re 或逆問題穩定度很關鍵。
* **軟先驗**：`prior_consistency` 權重不要太大（避免把 PINN 綁死在低保真）；建議 0.1–0.5 做 sweep。
* **源項還原**：先從空間平滑 + L1 稀疏啟動，觀察殘差圖樣再逐步放鬆正則。
* **重現性**：固定所有 seed，資料管線（cutout/散點）存檔，配置寫入 `configs/`。
* **資料標準化**：標準化網路輸入輸出，利於訓練穩定
* **學習率策略**：前期Adam後期L-BFGS，收斂更快速，不會一直震盪
* **權重標準化＋自適應**：標準化loss權重，使loss大小保持穩定，並透過自適應調整各項權重
* **啟動函數**：使用sine函數作為activation function，相比tanh更能保持高階項導數的靈敏度
* **課程式學習**：使用階段式學習，逐步降低學習率以及提升雷諾數，使模型緩步收斂更為精準
* **傅立葉特徵**：引入fourier feature當作輸入之一，使模型對於高頻特徵更為敏感

---

# 訓練架構設計
專案採用分層架構，將訓練邏輯清晰分離：

### 1. **腳本層** (`scripts/train.py` - 1232 行)
**職責**: 輕量級協調器與入口點
- 參數解析與配置載入
- 資料載入與預處理協調
- 模型/物理/損失函數初始化
- 訓練器實例化與調用
- 結果保存與日誌管理

**關鍵特性**:
- 不包含訓練循環邏輯（已移至 `Trainer`）
- 專注於「組裝」而非「執行」
- 支援單模型與 Ensemble 兩種模式
- 保持與所有現有 30+ 配置檔案的向後相容

### 2. **核心訓練器** (`pinnx/train/trainer.py` - 815 行)
**職責**: 可重用的訓練循環管理
- 單步訓練 (`step()`)：前向傳播、損失計算、梯度更新
- 驗證循環 (`validate()`)：計算驗證集指標
- 完整訓練 (`train()`)：epoch 循環、早停、檢查點管理
- 動態權重調度（GradNorm、因果權重、課程學習）
- 學習率調度（Adam → L-BFGS 切換）

**關鍵特性**:
- 設備無關（支援 CPU/CUDA）
- 可獨立測試（單元測試友好）
- 支援 VS-PINN 與標準 PINN
- 完整的訓練歷史記錄

### 3. **工具模組** (`pinnx/train/`)
- `loop.py`: 訓練循環工具函數（權重應用、殘差計算）
- `adaptive_collocation.py`: 自適應採樣策略
- `checkpointing.py`: 檢查點保存/載入
- `ensemble.py`: Ensemble 訓練與不確定性量化
- `factory.py`: 模型/優化器/損失函數工廠
- `config_loader.py`: 配置管理與驗證

**使用範例**:
```python
# scripts/train.py 中的簡化調用
from pinnx.train.trainer import Trainer

# 初始化訓練器
trainer = Trainer(model, physics, losses, config, device)
trainer.training_data = training_data_sample

# 執行訓練（一行搞定）
train_result = trainer.train()
```
---

## 📦 配置模板快速開始

**新手用戶**：我們提供 4 個標準化模板，涵蓋從快速測試到生產訓練的完整流程。

### **模板選擇指南**

| 場景 | 模板 | 時間 | K 點數 | Epochs |
|------|------|------|--------|--------|
| **快速驗證想法** | [`2d_quick_baseline.yml`](configs/templates/2d_quick_baseline.yml) | 5-10 min | 50 | 100 |
| **特徵消融研究** | [`2d_medium_ablation.yml`](configs/templates/2d_medium_ablation.yml) | 15-30 min | 100 | 1000 |
| **課程式訓練** | [`3d_slab_curriculum.yml`](configs/templates/3d_slab_curriculum.yml) | 30-60 min | 100 | 1000 |
| **論文級結果** | [`3d_full_production.yml`](configs/templates/3d_full_production.yml) | 2-8 hrs | 500 | 5000 |

👉 **完整模板文檔**：[`configs/templates/README.md`](configs/templates/README.md)

### **快速使用範例**

```bash
# 1. 複製模板到 configs/ 目錄
cp configs/templates/2d_quick_baseline.yml configs/my_experiment.yml

# 2. 修改必要參數（實驗名稱、輸出路徑）
vim configs/my_experiment.yml

# 3. 執行訓練
python scripts/train.py --cfg configs/my_experiment.yml

# 4. 監控訓練進度
tail -f log/my_experiment/training.log
```

**必改參數**：
- `experiment.name`: 改為你的實驗名稱（如 `my_ablation_fourier_v1`）
- `output.checkpoint_dir`: 改為對應路徑（如 `./checkpoints/my_ablation_fourier_v1`）
- `output.results_dir`: 改為對應路徑（如 `./results/my_ablation_fourier_v1`）

**配置命名規範**：參見 [`configs/README.md`](configs/README.md#-配置命名規範)

---

## 🚀 訓練腳本使用方式

### **基本訓練指令**
```bash
# 基本訓練（使用配置文件）
python scripts/train.py --cfg configs/<config_name>.yml

# Ensemble 訓練（不確定性量化）
python scripts/train.py --cfg configs/<config_name>.yml --ensemble

# 從檢查點恢復訓練
python scripts/train.py --cfg configs/<config_name>.yml --resume checkpoints/<exp_name>/epoch_X.pth
```

### **命令行參數說明**
| 參數 | 類型 | 預設值 | 說明 |
|------|------|--------|------|
| `--cfg` | str | `configs/defaults.yml` | 配置文件路徑 |
| `--ensemble` | flag | False | 啟用 Ensemble 訓練 |
| `--resume` | str | None | 從檢查點恢復訓練的路徑 |

### **配置文件必要欄位**

#### 1. **輸出路徑配置** (必須在 YAML 中定義)
```yaml
output:
  checkpoint_dir: "./checkpoints/<experiment_name>"  # 檢查點保存位置
  results_dir: "./results/<experiment_name>"         # 結果輸出位置
  visualization_dir: "./results/<experiment_name>/visualizations"  # 視覺化輸出

logging:
  level: "info"              # 日誌等級: debug/info/warning/error
  log_freq: 10              # 日誌輸出頻率（每 N 個 epoch）
  save_predictions: true    # 是否保存預測結果
  tensorboard: true         # 啟用 TensorBoard
  wandb: false              # 啟用 Weights & Biases（選用）
```

#### 2. **實驗基本設定**
```yaml
experiment:
  name: "<experiment_name>"  # 實驗名稱（用於日誌識別）
  version: "v1.0"           # 版本號
  seed: 42                  # 隨機種子（可重現性）
  device: "auto"            # 設備：auto/cpu/cuda/cuda:0
  precision: "float32"      # 精度：float32/float64
  description: "實驗描述"   # 實驗說明（選用）
```

#### 3. **訓練設定**
```yaml
training:
  optimizer: "adam"         # 優化器：adam/lbfgs/sgd
  lr: 1.0e-3               # 學習率
  epochs: 1000             # 訓練輪數
  batch_size: 1024         # 批次大小
  checkpoint_freq: 100     # 檢查點保存頻率
  log_interval: 10         # 日誌輸出間隔
  validation_freq: 50      # 驗證頻率
```

### **實際使用範例**

#### 快速測試（100 epochs）
```bash
python scripts/train.py --cfg configs/test_rans_quick.yml
```

#### 完整訓練（1000+ epochs）
```bash
# 前台運行
python scripts/train.py --cfg configs/test_physics_fix_1k.yml

# 背景運行（推薦）
nohup python scripts/train.py --cfg configs/test_physics_fix_1k.yml \
    > log/<exp_name>/training_stdout.log 2>&1 &
```

#### 監控訓練進度
```bash
# 即時監控訓練日誌
tail -f log/<exp_name>/training.log

# 監控標準輸出
tail -f log/<exp_name>/training_stdout.log

# 使用監控腳本（如果有）
./scripts/monitor_curriculum.sh
```

### **⚠️ 常見錯誤與修正**

| 錯誤類型 | 錯誤原因 | 正確方式 |
|---------|---------|---------|
| ❌ `--log_dir` | 不存在的參數 | ✅ 在 YAML 中配置 `output.checkpoint_dir` |
| ❌ `--checkpoint` | 不存在的參數 | ✅ 使用 `--resume <path>` |
| ❌ `--name` | 不存在的參數 | ✅ 在 YAML 中配置 `experiment.name` |
| ❌ 直接修改 `train.py` | 破壞可維護性 | ✅ 通過 YAML 配置所有參數 |
| ❌ 硬編碼路徑 | 不利於複用 | ✅ 使用相對路徑並遵循目錄規範 |

### **目錄結構規範**
```
實驗名稱建議格式: test_<feature>_<variant>_<epochs>
範例: test_rans_phase6c_v3

對應目錄結構:
├── configs/test_rans_phase6c_v3.yml          # 配置文件
├── checkpoints/test_rans_phase6c_v3/         # 檢查點輸出
│   ├── epoch_100.pth
│   ├── best_model.pth
│   └── latest.pth
├── results/test_rans_phase6c_v3/             # 結果輸出
│   ├── metrics.json
│   ├── predictions.npz
│   └── visualizations/                       # 視覺化圖表
└── log/test_rans_phase6c_v3/                 # 日誌文件（可選）
    ├── training.log
    └── training_stdout.log
```

### **檢查點管理**
```bash
# 從特定 epoch 恢復
python scripts/train.py --cfg configs/my_exp.yml --resume checkpoints/my_exp/epoch_500.pth

# 從最佳模型恢復（warm start）
python scripts/train.py --cfg configs/phase2.yml --resume checkpoints/phase1/best_model.pth

# 從最新檢查點恢復
python scripts/train.py --cfg configs/my_exp.yml --resume checkpoints/my_exp/latest.pth
```

---
