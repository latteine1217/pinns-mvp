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
  │   ├─ train/
  │   │   ├─ trainer.py            # 主訓練循環
  │   │   └─ ensemble.py           # Ensemble 訓練 + UQ
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
- 數據集大小：~500k個點
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
