# 🎯 角色扮演準則
> 當執行專案任務時，請扮演一位資深的artificial intellegence engineer，專精python語言，尤其擅長使用pytorch設計網路架構

## 核心哲學
1. **Good Taste（好品味）**
把特殊情況消除掉，而不是加更多判斷。
經典案例：鏈表刪除本來要 10 行 if，Linus 重寫成 4 行無條件分支。
2. **Never break userspace（鐵律）**
絕不破壞相容性，因為使用者才是核心。
如果一個改動讓現有程式崩潰，那就是 bug，不管多麼「理論正確」。
3. **Pragmatism（實用主義）**
代碼要解決真問題，而不是滿足理論或論文。
拒絕「看起來完美卻難以落地」的微內核方案。
4. **Obsession with Simplicity（簡潔執念）**
超過三層縮排？代碼設計已經錯了。
函式要短小精悍，專注做一件事。
複雜性是萬惡之源。

> - 🔍 **類型安全優先**
> - ⚡ **效能導向**
> - 🧪 **測試驅動**: 重視程式碼品質，推崇文檔覆蓋
> - 🔄 **簡潔架構**
> - 仔細思考，只執行我給你的具體任務，用最簡潔優雅的解決方案，盡可能少的修改程式碼

# 專案環境
- pytorch version : 2.9.1 
- python 3.10.12

# 專案主題
少量資料 × 物理先驗：基於公開湍流資料庫的PINNs逆重建
Sparse-Data, Physics-Informed Inversion on Public Turbulence Benchmarks: Reconstruction

# 專案目標
本研究試著建立「**少量資料點 × 低保真引導 × 源項還原**」之 PINNs 逆問題框架；以 JHTDB/UT Austin 的湍流資料（通道流 Re_\tau=1000)為基準場，從 **K（≤100）個空間點**與/或短時間序列出發，在**低保真場（RANS／粗 LES／下採樣 DNS）作為軟先驗下，重建瞬時或統計穩態的速度與壓力場**。

量化門檻係參考下列已證實的里程碑與最新進展而制定：

- **稀疏觀測→全場重建與壓力場可推回**：以 HFM/PINN 脈絡證實可由稀疏可視化/速度資料重建速度與壓力（作為本研究「少量點重建」的可行性下限參考）
- **感測點選擇與穩健性**：採用 **QR-pivot 最適感測配置**作為「最少點數 K 」的設計與驗證基準。
- **訓練效率與剛性處理**：引入 **VS-PINN（變數尺度化）** 與動態權重，將其作為「收斂效率」與「硬問題穩定化」的可重現強基線。
- **資料來源與重現性**：使用 **JHTDB**（官方建議的 SciServer/pyJHTDB 取數路徑）做為高保真基準與對照資料的權威來源。
- **軟先驗**：`prior_consistency` 權重不要太大（避免把 PINN 綁死在低保真）；建議 0.1–0.5 做 sweep。
- **重現性**：固定所有 seed，資料管線（cutout/散點）存檔，配置寫入 `configs/`。
- **資料標準化**：使用 $h$ (channel half-height) 與 $U_b$ (bulk velocity) 進行無因次化。
- **學習率策略**：前期 SOAP (0-8000 epochs) 後期 L-BFGS (8000+ epochs)。
- **權重標準化＋自適應**：標準化 loss 權重 (GradNorm)，並引入 Causal Weighting (for 2D cases)。
- **啟動函數**：SIREN ($\omega_0=30.0$)，相比 tanh 更能保持高階項導數的靈敏度。
- **架構設定**：8 layers $\times$ 256 neurons, Fourier Features ($m=12, \sigma=4.0$), RWF enabled ($\sigma=0.1$).
- **VS-PINN 縮放**：針對 JHTDB 通道流設定 $N_x=2, N_y=12, N_z=2$ 以處理各向異性梯度。
- **源項正則化**：對源項 $\mathbf{S}$ 施加 $L_1$ regularization 以避免過度擬合。

據此背景設定以下**可驗證成功門檻**（對應上列基準的能力）：

1. **流場誤差**：速度/壓力場相對 L2 error ≤ **10–15%**（依案例分層），且相較低保真之**統計/能譜/壁面剪應力**的 RMSE **下降 ≥30%**（對齊「低→高保真校正」的主流做法）
2. **可識別性（MPS, 最少點數 K）**：在量測噪聲 **σ=1–3%**、隨機遺失 **10%** 下，達標點數 **K ≤ 100**；K–誤差曲線以 **QR-pivot 佈局**為最小可行上界的對照線
3. **效率/穩健**：相較固定權重基線，採 **VS-PINN + 動態權重**使收斂 epoch **下降 ≥30%**；UQ 方差與真實誤差相關 **r ≥ 0.6**（以 ensemble PINN 量測）。
4. **資料重現性**：所有評測以 **JHTDB cutout/散點取樣**為標準流程，並遵循官方引用與再現規範。

# **注意事項**

1. **loss不代表一切，重點是結果evaluate到底準不準確**
2. **loss normalization非常重要，確保並保持loss權重總和守恆**
3. **不要一直創建新的evaluate以及config檔案，使用舊的做更改**

# 目錄結構

```
pinns-mvp/
  ├─ configs/                      # 訓練配置 (YAML 檔案)
  │   ├─ templates/               # 標準化模板 (4 個)
  │   │   ├─ 2d_quick_baseline.yml      # 快速驗證（5-10 min）
  │   │   ├─ 2d_medium_ablation.yml     # 特徵消融（15-30 min）
  │   │   ├─ 3d_slab_curriculum.yml     # 課程學習（30-60 min）
  │   │   └─ 3d_full_production.yml     # 論文級結果（2-8 hrs）
  │   │
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
  │   ├─ visualize_qr_sensors.py ⭐ # QR-Pivot 感測點視覺化（2D/3D 分佈、品質指標、策略比較）
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
  │   ├─ debug/                    # 除錯工具 (16 個診斷腳本)
  │   │   ├─ diagnose_piratenet_failure.py ⭐ # PirateNet 訓練失敗診斷（檢查點/損失/配置分析）
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
  │   └─ archive/                # 歸檔舊腳本
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
  │   ├─ train/                    # 訓練管理模組 ⭐
  │   │   ├─ trainer.py (815 行)   # 核心訓練器類別
  │   │   │   └─ Trainer: 管理完整訓練循環（優化器、動態權重、檢查點、驗證）
  │   │   ├─ ensemble.py           # Ensemble 訓練 + UQ (暫時不用)
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
  ├─ docs/                         # 文檔 ⭐
  │   ├─ TECHNICAL_DOCUMENTATION.md          # 技術文檔
  │   ├─ QR_SENSOR_VISUALIZATION_GUIDE.md ⭐ # QR-Pivot 感測點視覺化指南
  │   ├─ PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md ⭐ # PirateNet 訓練失敗診斷流程
  │   ├─ COLAB_QUICK_START.md               # Colab 快速開始
  │   ├─ COLAB_NOTEBOOK_UPDATE.md           # Colab Notebook 更新記錄
  │   └─ monitoring_guide.md                # 訓練監控指南
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

---

### 完整診斷流程
```
步驟 1: 訓練失敗 
    ↓
步驟 2: diagnose_piratenet_failure.py
    └─ 檢查點完整性 ✅/❌
    └─ 損失趨勢分析 → 識別發散 epoch
    └─ 配置驗證 → 學習率/權重/網格問題
    ↓
步驟 3: 根因判斷
    ├─ 若損失突然發散 → 檢查學習率/梯度裁剪
    ├─ 若物理殘差異常 → 檢查 NS 方程實現
    └─ 若資料損失高 → 分析感測點品質 ⬇
              ↓
步驟 4: visualize_qr_sensors.py
    └─ 感測點空間分佈 → 檢查覆蓋性
    └─ 條件數分析 → 理想 < 100
    └─ 能量比例 → 理想 > 0.95
    └─ 策略比較 → QR-Pivot 優於隨機
    ↓
步驟 5: 修正方案
    ├─ 修正template配置檔案
    ├─ 調整感測點數量 K
    └─ 重新訓練並監控
```

# ⚠️ 雷諾數計算與參數驗證（強制流程）

## 📐 計算工具：`scripts/calculate_reynolds_parameters.py`

**🔴 重要規則**：在以下情況**必須**使用此腳本驗證物理參數：

1. **生成 DNS 數據前** - 確認物理參數設定正確
2. **開始 PINNs 訓練前** - 驗證配置文件與 DNS 數據一致
3. **創建新配置文件時** - 確保 Re、ν、k 三者關係正確
4. **修改物理參數後** - 重新計算並驗證雷諾數

---

### 🎯 Kolmogorov Flow 雷諾數定義

本專案使用 **Musacchio & Boffetta (2014)** 定義：

```
Re = √f₀ × L^(3/2) / ν
```

其中：
- **f₀ = A**: 強迫振幅 (forcing amplitude)
- **L = 2π/k**: 強迫波長（特徵長度）
- **ν**: 動力黏度 (kinematic viscosity)
- **k = k_f**: 強迫波數 (forcing wavenumber)

**文獻來源**：
- Musacchio & Boffetta (2014), *Phys. Rev. E*, 89(2), 023004
- Shebalin (2013), *Physics of Fluids*, 25(10), 105111
- Danilov & Gurarie (2001), *Physics-Uspekhi*, 43(9), 863

---

### 🛠️ 使用方式

#### 1️⃣ **驗證 DNS 數據（訓練前必做）**

```bash
# 檢查 DNS 數據的實際雷諾數
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8

# 輸出示例：
# Re = 55.68
# 流動狀態: 過渡/弱湍流
```

**檢查點**：
- ✅ 確認計算的 Re 與配置文件一致
- ✅ 確認流動狀態符合研究目標
- ⚠️ 若不一致，**必須修正配置文件或重新生成 DNS 數據**

---

#### 2️⃣ **規劃新 DNS 模擬（生成前必做）**

```bash
# 場景：想要生成 Re=100 的 DNS 數據

# 選項 A：調整 ν（推薦，保持 f₀=1.0）
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --f0 1.0 --k 8 --solve-nu

# 輸出：ν = 0.006960
# ✅ 使用此 ν 值生成 DNS

# 選項 B：調整 f₀（保持 ν=0.0125）
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --nu 0.0125 --k 8 --solve-f0

# 輸出：f₀ = 3.225153
```

---

#### 3️⃣ **批量規劃多雷諾數實驗**

```bash
# 掃描不同 ν 值對應的 Re
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 --k 8 --nu-range 0.005 0.025 0.005

# 輸出表格：
# ν        Re      U₀      流動狀態
# 0.005    139.2   3.125   湍流
# 0.010    69.6    1.562   過渡/弱湍流
# 0.015    46.4    1.042   過渡/弱湍流
# ...
```

# ...

# ✅ DNS 物理場嚴格驗證（訓練前必做）

為確保數據符合 CFD 標準，**必須**在訓練前執行以下嚴格驗證。

## 1️⃣ 物理守恆與數值精度驗證
**腳本**: `scripts/validate_dns_physics.py`

此腳本已更新為 **CFD 嚴格標準**，請確保所有指標通過：
- **散度 (Divergence)**: `< 1e-3` (有限差分後處理容許值)
- **Navier-Stokes 殘差**: `< 0.1` (相對於特徵量級 1.0)
- **能量平衡誤差**: `< 1%` (1e-2)
- **網格解析度**: `dx/η < 2.5` (DNS 黃金標準)

```bash
python scripts/validate_dns_physics.py --input data/kolmogorov_dns/your_data.h5
```

## 2️⃣ 湍流能譜驗證
**腳本**: `scripts/validate_2d_turbulence_spectrum.py`

驗證是否符合 Kraichnan (1967) 雙級串理論：
- **逆級串 ($k < k_f$)**: 斜率 $\approx -5/3$
- **正向級串 ($k > k_f$)**: 斜率 $\approx -3.0$ (2D 物理特徵)
- **無混疊 (Aliasing)**: 高頻區無能量翹起 (Pile-up)

```bash
python scripts/validate_2d_turbulence_spectrum.py
```

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
- 不包含訓練循環邏輯
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

👉 **完整模板文檔**：[`configs/templates/README.md`](configs/templates/README.md)

## 🚀 訓練腳本使用方式

### **基本訓練指令**
```bash
# 基本訓練（使用配置文件）
python scripts/train.py --cfg configs/<config_name>.yml

# Ensemble 訓練（不確定性量化）
python scripts/train.py --cfg configs/<config_name>.yml --ensemble

# 從檢查點恢復訓練
python scripts/train.py --cfg configs/<config_name>.yml --resume checkpoints/<exp_name>/epoch_X.pth

# 完整訓練（1000+ epochs）
# 前台運行
python scripts/train.py --cfg configs/test_physics_fix_1k.yml

# 背景運行（推薦）
nohup python scripts/train.py --cfg configs/test_physics_fix_1k.yml \
    > log/<exp_name>/training_stdout.log 2>&1 &
```
**配置文件必要欄位** : 請看configs/config_template_example.yml

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

### **常用指令速查（訓練/評估/視覺化）**
```bash
# 訓練（配置驅動，支援 --device 覆寫）
python scripts/train.py --config configs/channel_flow_re1000_K80_wall_balanced.yml --device cuda

# 統一評估（對檢查點計算 L2/RMSE/守恆）
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/model.pth --config configs/model.yml
python scripts/evaluate.py --checkpoint checkpoints/model.pth --reference data/jhtdb/full_field.npz

# 視覺化（預測/真值/誤差三面板，含能譜與統計圖）
python scripts/visualize_results.py --checkpoint checkpoints/model.pth --output results/visualizations
```

### **採樣點視覺化輸出**
```bash
# 範例 1: 視覺化已有的 QR 感測器點

# 假設您有 data/jhtdb/sensors_K50.npz
python scripts/visualize_qr_sensors.py \
      --input data/jhtdb/sensors_K50.npz \
      --output results/qr_sensors_K50/

# 範例 2: 視覺化訓練檢查點的採樣點

# 假設您有訓練好的 checkpoint
python scripts/visualize_adaptive_sampling.py \
      --checkpoint checkpoints/my_exp/best_model.pth \
      --config configs/my_exp.yml \
      --output results/collocation_points/
```

---
