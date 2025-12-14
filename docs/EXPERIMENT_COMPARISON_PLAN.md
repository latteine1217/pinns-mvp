# 稀疏量測 × 物理先驗 PINNs：對比實驗設計計畫（依據 `thesis/JY Thesis.tex`）

**目的**：把論文中的「QR 感測佈點 + VS-PINN 基線」擴充成可驗證的對比實驗矩陣，系統性回答哪些設計（模型表示、穩定化機制、感測點策略、低保真先驗/源項、訓練策略）在 **極少感測點 K（≤30–100）** 下真正帶來可量化的改善。

---

## 1. 論文基線（已在 thesis 中出現的比較）

`thesis/JY Thesis.tex` 已包含以下關鍵基線，後續所有對比建議以此為 anchor：

1. **2D Kolmogorov flow（Re=50, K=100）**
   - **Random vs QR-pivot**：感測點空間分佈對比（質性）
   - **Vanilla vs Full configuration**：全場相對 L2（u, v, ∇p）對比，並指出 **資料擬合 vs 不可壓縮性** 的 trade-off
2. **3D JHTDB channel flow（Re\_τ≈1000, K=100）**
   - **QR-pivot 感測器分佈**（近壁聚集）
   - **Full-field reconstruction baseline（無啟用 RANS prior loss）**：相對 L2(u,v,w,∇p) 仍在 O(100%)，用來凸顯高 Re 稀疏逆問題之剛性

> 註：論文亦明確指出未開啟 `RANS prior loss`，因此「低保真融合」與「源項/先驗協同」仍屬後續關鍵實驗缺口。

---

## 2. 核心研究問題 → 對應實驗軸（What to compare）

為了讓每個貢獻能在論文中被「單獨驗證」，建議把實驗拆成 6 條主軸（每次只改 1 個變因）：

### A) 感測點策略（Identifiability）
- Random vs QR-pivot（必要）
- K-scan：K ∈ {30, 50, 80, 100}（必要，對應 thesis 動機與工程可行性）

### B) 表示能力：Vanilla MLP vs（Enhanced / Fourier / SIREN）
- **Vanilla MLP**（tanh、無 Fourier、無殘差、無 RWF）
- **Fourier MLP**（Fourier features + MLP）
- **SIREN / sine MLP**（sine activation + SIREN init）
- **Enhanced（本專案實作上等價於 “Fourier-VS-MLP + ResNet block + RWF + dynamic weights” 的組合開關）**

### C) 穩定化機制拆解（Ablation）
以「Full（論文架構）」為母體，逐一關閉：
- Fourier features
- RWF（Random Weight Factorization）
- Adaptive residual / ResNet block（PirateNet-style）
- VS-PINN 變數尺度化（variable scaling）
- 動態權重（GradNorm / AGB）與 loss normalization（權重總和守恆）

### D) 訓練策略（Stiffness / Efficiency）
- Adam vs SOAP vs（SOAP→L-BFGS）等 schedule（以 **達標 epoch / wall-time** 比較）

### E) 低保真先驗融合（Prior consistency）
- prior weight sweep：**0.0 / 0.1 / 0.3 / 0.5**（避免把 PINN 綁死在 low-fi）
- prior 點數 N\_prior 與空間權重（例如 near-wall 加權）對結果的影響

### F) 源項（Source term）與先驗的協同/解耦
對應 thesis 的 “Synergy between Source Term and RANS Prior”：
- 無源項 vs 有源項（含 L1 sparsity）
- prior-only vs prior+source（觀察是否出現「RANS + 大源項」的作弊解）

---

## 3. Benchmarks 與推薦跑法（先 2D 再 3D，逐步加剛性）

### 3.1 2D Kolmogorov（低成本、適合做消融）
- **用途**：快速驗證「感測點策略 / 表示能力 / 權重平衡 / 因果訓練」的相對貢獻
- **建議設定**：
  - 固定 K=100 做機制消融（先把變因釐清）
  - 再做 K-scan（K=30/50/80/100）畫出 K–error 曲線
- **可直接重用配置（避免新增 config 檔）**：
  - `configs/kolmogorov_re50_kf4_K100_vanilla_1k.yml`
  - `configs/kolmogorov_re50_kf4_K100_full_1k.yml`
  - `configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml`（建議把 prior weight 調回 0.1–0.5 sweep）

### 3.2 3D Channel flow（高成本、優先做「最有信息量」的對比）
- **用途**：驗證方法在高 Re、近壁多尺度下是否能把 error 從 O(100%) 拉下來
- **建議跑法**：
  - 先用 `configs/templates/3d_slab_curriculum.yml`（slab）做快速篩選
  - 再把最有希望的組合升級到 `configs/templates/3d_full_production.yml`
  - 最後補上「Random vs QR」與「K-scan」的關鍵點（不用全組合爆炸）

---

## 4. 評估指標（避免只看 loss）

### 4.1 主要指標（必報）
- **全場相對 L2**：u, v（2D）；u, v, w（3D）
- **壓力可識別性**：用 **∇p**（而非 p）做誤差，避免 gauge 問題（thesis 已採用）
- **不可壓縮性**：‖∇·u‖（均值 + 最大值都要看；thesis 已指出 DNS 也有 operator floor）

### 4.2 湍流/工程關鍵量（Channel flow 建議必報）
- **壁面剪應力 τ\_w** / friction velocity 誤差
- **Mean profile**：U⁺(y⁺) correlation / RMSE
- **Reynolds stresses / TKE**：⟨u′v′⟩、k(y)（至少抓住近壁峰值位置與量級）
- **能譜**：E(k)（低/中頻是否被重建；避免只靠 smooth field 取巧）

### 4.3 魯棒性（必要）
- 噪聲：σ ∈ {0, 1%, 3%}
- 遺失：dropout ∈ {0, 10%}
- 報告：誤差平均 ± 標準差（至少 3 seeds；3D 可先做 1 seed + 2D 補統計）

---

## 5. 實驗矩陣（最少集 → 擴充集）

### 5.1 最少集（建議 thesis 必做，能支撐主要 claim）

| ID | 對比 | Benchmark | 控制變因（固定） | 主要輸出 |
|---|---|---|---|---|
| S1 | Random vs QR-pivot | 2D Re50, K=100 | 同模型/同訓練/同噪聲 | L2(u,v,∇p), ‖∇·u‖, sensor quality（cond. #） |
| S2 | K-scan（QR） | 2D Re50, K=30/50/80/100 | 同模型/同訓練 | K–error 曲線 + 可達門檻 K |
| M1 | Vanilla MLP vs Full | 2D Re50, K=100 | 同感測器（QR） | L2 與 divergence trade-off（對齊 thesis） |
| A1 | Full w/ vs w/o Fourier | 2D Re50, K=100 | 只切 Fourier | L2 與能譜差異 |
| A2 | Full w/ vs w/o dynamic weights | 2D Re50, K=100 | 只切 GradNorm/權重標準化 | 收斂速度（epochs/time）+ 最終 L2 |
| C1 | Prior comparison（無 prior） vs 有 prior | 2D Re50（先）→ 3D slab（後） | 同感測器（QR）、同訓練 | L2(u,v,∇p), ‖∇·u‖（3D 再加 τ\_w, U⁺(y⁺)） |
| C2 | prior weight sweep | 2D Re50（先）→ 3D slab（後） | prior\_weight ∈ {0.1,0.3,0.5} | error vs prior\_weight（找 sweet spot） |

### 5.2 擴充集（用來定位為什麼 3D 會 O(100%)）

| ID | 對比 | Benchmark | 目的（假說） |
|---|---|---|---|
| A3 | RWF on/off | 2D→3D slab | 降低剛性、改善收斂穩定 |
| A4 | ResNet block on/off | 2D→3D slab | 深網可訓練性提升（PirateNet 方向） |
| A5 | VS-PINN on/off | 3D slab | 減少變數尺度差導致的 loss imbalance |
| O1 | Adam vs SOAP vs SOAP→L-BFGS | 2D→3D slab | 用 wall-time/達標 epoch 量化效率 |
| R1 | 噪聲/遺失魯棒性 | 2D（統計）+ 3D slab（抽樣） | 工程可用性：σ=1–3%, dropout=10% |
| P1 | prior-only vs prior+source | 2D→3D slab | 檢查是否需要源項吸收 model bias，同時避免作弊解 |

### 5.3 研究級擴展（放 future work 也合理）
- QR-DEIM / randomized QR pivot（sensor / collocation 可擴充）
- Adaptive collocation（residual-based / QR-DEIM）
- UQ：ensemble PINN / Bayesian PINN（對應 thesis future work）

---

## 6. 執行順序（避免組合爆炸）

1. **Phase-0（半天）**：把 thesis 2D baseline（Vanilla vs Full）完整跑通，確保 `evaluate` 指標與圖表一致
2. **Phase-1（1–2 天）**：2D 上做 S1/S2/M1/A1/A2（把關鍵機制貢獻釐清）
3. **Phase-2（2–3 天）**：3D slab 做 C1/C2 +（A3/A4/A5 擇 1–2 個最可能改善的）
4. **Phase-3（算力允許再做）**：把 slab 最佳組合升級到 full domain，補上必要的統計/能譜/近壁指標

---

## 7. 重現性與公平性檢查清單（每組對比都要符合）

- 固定 `seed`；感測器檔案與資料切窗固定（不重抽）
- 只改一個變因（其餘包含：K、感測器、N\_pde、batch、optimizer schedule 盡量固定）
- **loss 權重總和守恆**：若禁用自適應權重，仍需手動標準化，避免「贏在 scale」
- 報告指標以 evaluation 為準（不是 training loss）

---

## 8. 建議輸出到論文/簡報的圖表（最值回報）

- K–error 曲線（Random vs QR；2D 與 3D 至少各一張）
- Ablation bar chart：Full vs（-Fourier/-GradNorm/-RWF/-ResNet/-VS）
- prior weight sweep 曲線（error vs prior\_weight）
- Channel mean profile（U⁺ vs y⁺）+ τ\_w 對比
- 能譜 E(k) 對比（至少 low/mid-k 是否被重建）

---

## 9. 結果圖/示意圖清單與繪圖規格（避免「圖畫得漂亮但不可比」）

### 9.1 全域繪圖規則（所有對比都必須遵守）
- **同一組比較要同一個 reference**：同一 snapshot / 同一 time window / 同一切片平面與位置。
- **同一組比較要同一個色階**：
  - DNS vs PINN 的「場圖」使用 **相同 vmin/vmax**（建議以 DNS 的 min/max 或 1–99% percentile 決定，並固定在整組比較）。
  - 「誤差圖」用獨立色階，但同一組方法也要一致（避免靠色階掩飾）。
- **同一組比較要同一個視角/裁切**：座標範圍、aspect ratio、相機角度（3D）固定。
- **感測點標記規格固定**：marker 大小/顏色/透明度/邊框一致；避免「看起來比較密」其實只是 marker 比較大。
- **標註必含**：`K`、選點方法（Random/QR/Hybrid）、變數名稱與無因次化（例如 u/u\_τ）、以及色條單位。
- **壓力相關圖**：以 **∇p** 或去均值的 p′ 呈現（避免 gauge 造成假差異）。

### 9.2 必備示意圖（方法/設定）
1. **Pipeline schematic（1 張）**
   - Low-fi（RANS/粗 LES）→ QR 選點 → 虛擬量測（DNS/JHTDB）→ PINN assimilation（data+PDE+BC+prior+source）→ evaluation（L2/τ\_w/U⁺/spectrum）。
2. **Loss terms schematic（1 張）**
   - 清楚列出 data / PDE / BC / prior / source L1 的角色與「只改一個變因」的 ablation 思路。

### 9.3 感測器佈點圖（最重要、最容易畫錯）
**目標**：直觀展示「Random vs QR」在同一背景場上，QR 會聚焦於高梯度/近壁資訊區域。

- **F-S1：Random vs QR 兩張並排（同一色階）**
  - 左：Random sensors；右：QR-pivot sensors。
  - 背景：同一張 reference flow 圖（建議用 |u|、|∇u|、vorticity 或 Q-criterion；2D 用 vorticity 很直覺）。
  - 兩張 **共用同一個 colorbar**（同 vmin/vmax），且標註同一時間/切片位置。
- **F-S2：近壁放大圖（channel flow 必做）**
  - 在 y≈±h 的薄層區域做 zoom-in，避免主圖看不出近壁差異。
  - 可加 y⁺ 標尺或標示 y-range。

### 9.4 全場重建圖（DNS vs PINN vs Error）
- **F-R1：三聯圖（DNS / PINN / |Error|）**
  - 2D：u、v（必要）+ ∇p（可選但建議）。
  - 3D：至少 u、w + ∇p（或 dp/dx）在固定切面（如 y–z at fixed x）。
  - DNS 與 PINN：**同色階**；誤差：單獨色階（但跨方法一致）。
- **F-R2：統計圖（channel flow 必做）**
  - U⁺(y⁺) mean profile：DNS / RANS / PINN 同圖（可加 near-wall inset）。
  - τ\_w 分佈：沿 x–z 的 map 或 histogram（同 bins），並報告相對誤差。
- **F-R3：能譜圖（避免 smooth-cheat）**
  - log–log E(k)，同一段 k 範圍；加參考斜率線（-5/3 等，依案例）。

### 9.5 對比實驗專用圖（支撐 claim）
- **F-K1：K–error 曲線**
  - Random vs QR（兩條曲線），y 軸為 relative L2（或 overall error），含目標門檻線（10–15%）。
  - 2D 建議做 3 seeds：畫 mean ± std（shaded band）。
- **F-P1：prior weight sweep 曲線**
  - x：prior\_weight；y：overall error +（可選）‖∇·u‖、τ\_w error。
  - 目的：找 sweet spot（避免 prior 太大綁死）。
- **F-A1：Ablation bar chart**
  - 以 Full 為 0，畫 Δerror（-Fourier/-GradNorm/-RWF/-ResNet/-VS）。
  - 同時報告收斂速度（epochs / wall-time）可用第二座標軸或表格。
- **F-RB1：魯棒性圖**
  - σ 與 dropout 的 error 曲線或 heatmap（至少 2D 做統計）。

### 9.6 圖表輸出與命名（建議）
- 同一實驗輸出集中到 `results/<exp>/visualizations/`，命名固定：`F-S1_random_vs_qr.png`、`F-R1_u_dns_pred_err.png`、`F-K1_k_scan.png` 等。
- 同一張圖的所有版本（不同方法）用同一套函式/腳本產出，避免手工調色造成不可比。
