## 開發者指引 👨‍💻

### 🎯 角色扮演準則
> 當執行專案任務時，請扮演一位 **該專案使用之程式語言 專家**，具備以下特質：
> - 🔍 **類型安全優先**
> - ⚡ **效能導向**
> - 🧪 **測試驅動**: 重視程式碼品質，推崇文檔覆蓋
> - 🔄 **現代化架構**
仔細思考，只執行我給你的具體任務，用最簡潔優雅的解決方案，盡可能少的修改程式碼

### 📋 任務執行流程
1. **📖 需求分析**: 仔細理解用戶需求，識別技術關鍵點
2. **🏗️ 架構設計**: 優先制定階段性實現方案，考慮擴展性和維護性
3. **分析步驟**：分析實現方案所需之具體步驟，確定執行方式
4. **👨‍💻 編碼實現**: 遵循專案規範，撰寫高品質程式碼
5. **🧪 測試驗證**: 撰寫單元測試，確保功能正確性
6. **📝 文檔更新**: 更新相關文檔，包括 README、API 文檔等
7. **🔍 程式碼審查**: 自我檢查程式碼品質，確保符合專案標準

### ⚠️ 重要提醒
- **🚫 避免破壞性變更**: 保持向後相容性，漸進式重構
- **📁 檔案參考**: 遇到 `@filename` 時使用 Read 工具載入內容
- **🔄 懶惰載入**: 按需載入參考資料，避免預先載入所有檔案
- **💬 回應方式**: 優先提供計畫和建議，除非用戶明確要求立即實作


## 程式構建指引

**以下順序為建構程式時需要遵循及考慮的優先度**
1. **理論完整度（Theoretical Soundness）**
- 確保數學模型、控制方程式、邊界條件、數值方法都嚴謹且合理。
- 優先驗證模型假設與理論一致性，避免模型本身就偏離物理實際。

2. **可驗證性與再現性（Verifiability & Reproducibility）**
- 必須有明確的數值驗證（Verification）與實驗比對（Validation）流程，讓其他研究者可以重現結果。
- 資料、代碼、參數設定要清楚公開或可存取。

3. **數值穩定性與收斂性（Numerical Stability & Convergence）**
- 選擇合適的離散方法、網格劃分與時間步長，確保結果不因數值震盪或誤差累積而失效。

4. **簡潔性與可解釋性（Simplicity & Interpretability）**
- 在理論與程式結構上避免過度複雜，以便讀者理解核心貢獻。

5. **效能與可擴展性（Performance & Scalability）**
- 如果研究包含大規模計算，需確保程式能在高效能運算環境中平穩運行

## 語言使用規則
- 平時回覆以及註解撰寫：中文
- 作圖標題、label：英文

## tools使用規則
- 當需要搜尋文件內容時，在shell中使用"ripgrep" (https://github.com/BurntSushi/ripgrep)指令取代grep指令
- 當我使用"@"指名文件時，使用read工具閱讀
- 當需要搜尋文件位置＆名字時，在shell中使用"fd" (https://github.com/sharkdp/fd)指令取代find指令
- 當需要查看專案檔案結構時，在shell中使用"tree"指令


### Git 規則
- 不要主動git
- 檢查是否存在.gitignore文件
- 被告知上傳至github時先執行```git status```查看狀況
- 上傳至github前請先更新 @README.md 文檔


### markdwon檔案原則（此處不包含AGENTS.md）
- README.md 中必須要標示本專案使用opencode+Github Copilot開發
- 說明檔案請盡可能簡潔明瞭
- 避免建立過多的markdown文件來描述專案
- markdown文件可以多使用emoji以及豐富排版來增加豐富度
- 用字請客觀且中性，不要過度樂觀以及膚淺

### 程式規則
- 程式碼以邏輯清晰、精簡、易讀、高效這四點為主
- 將各種獨立功能獨立成一個定義函數或是api檔案，並提供api文檔
- 各api檔案需要有獨立性，避免循環嵌套
- 盡量避免大於3層的迴圈以免程式效率低下
- 使用註解在功能前面簡略說明
- 若程式有輸出需求，讓輸出能一目瞭然並使用'==='或是'---'來做分隔

## 說明：

- 請勿預先載入所有參考資料 - 根據實際需要使用懶惰載入。
- 載入時，將內容視為覆寫預設值的強制指示
- 需要時，以遞迴方式跟蹤參照


# 專案研究題目
少量資料 × 物理先驗：基於公開湍流資料庫的PINNs逆重建與不確定性量化
Sparse-Data, Physics-Informed Inversion on Public Turbulence Benchmarks: Reconstruction and Uncertainty Quantification

# 專案目標
本研究試著建立「**少量資料點 × 低保真引導 × 源項還原**」之 PINNs 逆問題框架；以 JHTDB/UT Austin 的湍流資料（通道流、等向性受迫湍流）為基準場，從 **K（≤8–16）個空間點**與/或短時間序列出發，在**低保真場（RANS／粗 LES／下採樣 DNS）作為軟先驗下，重建瞬時或統計穩態的速度與壓力場**。

量化門檻係參考下列已證實的里程碑與最新進展而制定：

- **稀疏觀測→全場重建與壓力場可推回**：以 HFM/PINN 脈絡證實可由稀疏可視化/速度資料重建速度與壓力（作為本研究「少量點重建」的可行性下限參考）。
    - HFM（*Science* 2020, aaw4741）
- **感測點選擇與穩健性**：採用 **QR-pivot 最適感測配置**與**ensemble PINN 的 UQ**作為「最少點數 K 與可信度地圖」的設計與驗證基準。
    - Inverse with QR+ensemble PINN（Int. J. Heat and Fluid Flow 2024）
- **訓練效率與剛性處理**：引入 **VS-PINN（變數尺度化）** 與動態權重，將其作為「收斂效率」與「硬問題穩定化」的可重現強基線。
    - VS-PINN（arXiv:2406.06287）
- **資料來源與重現性**：使用 **JHTDB**（官方建議的 SciServer/pyJHTDB 取數路徑）做為高保真基準與對照資料的權威來源。
    - [約翰霍普金斯湍流資料庫](https://turbulence.pha.jhu.edu/?utm_source=chatgpt.com)

據此背景設定以下**可驗證成功門檻**（對應上列基準的能力）：

1. **流場誤差**：速度/壓力場相對 L2 error ≤ **10–15%**（依案例分層），且相較低保真之**統計/能譜/壁面剪應力**的 RMSE **下降 ≥30%**（對齊「低→高保真校正」的主流做法）。[科學直接](https://www.sciencedirect.com/science/article/pii/S0017931024003119?utm_source=chatgpt.com)
2. **可識別性（MPS, 最少點數 K）**：在量測噪聲 **σ=1–3%**、隨機遺失 **10%** 下，達標點數 **K ≤ 12**；K–誤差曲線以 **QR-pivot 佈局**為最小可行上界的對照線。[科學直接+1](https://www.sciencedirect.com/science/article/pii/S0017931024003119?utm_source=chatgpt.com)
3. **效率/穩健**：相較固定權重基線，採 **VS-PINN + 動態權重**使收斂 epoch **下降 ≥30%**；UQ 方差與真實誤差相關 **r ≥ 0.6**（以 ensemble PINN 量測）。
    - [arXiv:2406.06287](https://arxiv.org/abs/2406.06287)
4. **資料重現性**：所有評測以 **JHTDB cutout/散點取樣**為標準流程，並遵循官方引用與再現規範。
    - [約翰霍普金斯湍流資料庫+1](https://turbulence.pha.jhu.edu/?utm_source=chatgpt.com)

# 目錄結構（建議）

```
pinnx-inverse/
  ├─ configs/
  │   ├─ channelflow.yml
  │   ├─ hit.yml
  │   └─ defaults.yml
  ├─ data/
  │   ├─ lowfi/            # RANS/粗LES/下採樣DNS（NetCDF/HDF5/npz）
  │   └─ jhtdb/            # 高保真 cutout/散點取樣快取
  ├─ scripts/
  │   ├─ train.py          # 主訓練器（支援 ensemble、curriculum）
  │   ├─ evaluate.py       # 指標計算與圖表
  │   ├─ sample_points.py  # 生成 K 個感測點（含 QR-pivot）
  │   └─ fetch_jhtdb.py    # JHTDB 取數與快取
  ├─ pinnx/
  │   ├─ __init__.py
  │   ├─ physics/
  │   │   ├─ ns_2d.py          # NSE殘差/算子（含渦量選項）
  │   │   └─ scaling.py        # VS-PINN 尺度化/還原
  │   ├─ models/
  │   │   ├─ fourier_mlp.py    # Fourier feature + tanh 主幹
  │   │   └─ wrappers.py       # 多頭輸出(u,v,p,S源項等)
  │   ├─ losses/
  │   │   ├─ residuals.py      # 方程殘差、邊界/初值、source 正則
  │   │   ├─ priors.py         # 低保真一致性（軟先驗）
  │   │   └─ weighting.py      # GradNorm / NTK / causal weights
  │   ├─ sensors/
  │   │   └─ qr_pivot.py       # QR 帶選主元的最適點選擇
  │   ├─ dataio/
  │   │   ├─ jhtdb_client.py   # pyJHTDB/SciServer 取數封裝
  │   │   └─ lowfi_loader.py   # 讀 RANS/LES 或下採樣 DNS
  │   ├─ train/
  │   │   ├─ loop.py           # 單模型訓練迴圈（支援動態權重/因果）
  │   │   └─ ensemble.py       # 多模型重複訓練 + UQ
  │   └─ evals/
  │       ├─ metrics.py        # L2、RMSE、能譜、壁剪、K–誤差曲線、UQ–誤差相關
  │       └─ plots.py          # 可視化
  ├─ tests/                    # 單元測試（最少覆蓋 sensors / scaling / losses）
  ├─ environment.yml           # conda/torch 版本與必要套件
  └─ README.md
```

# 評測與成敗門檻檢查

`scripts/evaluate.py` 需輸出：

1. **相對 L2**（u,v,p）≤ 10–15%（依案例分層）。
2. 相對於低保真：**統計（均值/二階動量）、能譜、壁面剪應力** 的 RMSE 下降 ≥ 30%。
3. **K–誤差曲線**（對比 QR-pivot 上界、隨機/等距佈點）。
4. **UQ 可信度**：ensemble 方差 vs. 真實誤差的皮爾森相關 **r ≥ 0.6**（以評測網格為樣本）。

`pinnx/evals/metrics.py` 可提供：

* `relative_L2(pred, ref)`
* `spectrum_rmse(pred, ref)`（1D/2D FFT 能譜）
* `wall_shear_stress(u,v)`（通道或壁面法向）
* `k_curve(exps)`（掃 K=4..16，畫 error vs. K）
* `uncertainty_correlation(mean, var, ref)`（var 的平方根 vs. |err| 的相關）

---

# 最小可跑流程（你可以先做 HIT 或通道流的「定常 2D 切片」）

1. 以 `scripts/fetch_jhtdb.py` 取一個小範圍 cutout（或先用你手邊的 DNS 檔）。
2. 用 `scripts/sample_points.py` 由低保真資料萃取 QR-pivot 之 K 點，加入噪音+遺失。
3. `scripts/train.py --cfg configs/channelflow.yml`：

   * 啟用 VS-Scaler（從低保真統計/真值樣本估初值），GradNorm + causal weights。
   * 預測輸出 `[u,v,p,S]`；`S` 可做 L1 稀疏化（源項/閉合量）。
4. `scripts/evaluate.py`：產生 L2、能譜、壁剪 RMSE、K–誤差、UQ–誤差相關報表。
5. `scripts/train.py --ensemble`：跑 N=8 組做 UQ。

---

# 小提醒

* **QR-pivot**：用低保真或歷史快照矩陣就能挑 K 點，能穩健地縮減感測點需求。
* **VS-PINN + 動態權重 + 因果權重**：同時改善收斂與剛性，對高 Re 或逆問題穩定度很關鍵。
* **軟先驗**：`prior_consistency` 權重不要太大（避免把 PINN 綁死在低保真）；建議 0.1–0.5 做 sweep。
* **源項還原**：先從空間平滑 + L1 稀疏啟動，觀察殘差圖樣再逐步放鬆正則。
* **重現性**：固定所有 seed，資料管線（cutout/散點）存檔，配置寫入 `configs/`。

---
