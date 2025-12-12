# PINNs-MVP 結果與實驗補強計畫

本文件說明專案接下來在「結果與論文支持」上應該完成的實驗、需要掃描的變數，以及要輸出的圖表與指標。所有項目都以目前 repo 內已存在的腳本與資料夾為基礎，不假設尚未實作的功能。

---

## 1. 2D Kolmogorov Flow（低 Re 基線）

目標：把 Kolmogorov Re=50（`configs/kolmogorov_re50_kf4_K100*.yml`）變成完整可重現的 baseline，展示：
- Vanilla vs Full 模型的重建誤差差異
- 物理約束（散度）與數據擬合之間的 trade-off

### 1.1 必要實驗與變數

- **固定設定**
  - Re = 50, $k_f = 4$, domain $[0,2\pi]^2$
  - 感測點數：`K = 100`
  - 使用已存在配置：
    - Vanilla: `kolmogorov_re50_kf4_K100_vanilla_1k.yml`
    - Full:    `kolmogorov_re50_kf4_K100_full_1k.yml`

- **需系統記錄的指標**
  - 相對 L2 error：$u, v, p$（全場）
  - RMSE：$u, v, p$
  - 散度指標：mean |∇·u|、max |∇·u|
  - 訓練效率：每 epoch 時間、總訓練時間、收斂到某個 L2 門檻所需 epoch（若適用）

### 1.2 之後可掃描的變數（可選）

在同一條 pipeline 下，逐步增加實驗：

- 感測點數 K：
  - $K \in \{25, 50, 100, 200\}$（固定 QR layout）
  - 量化：L2 error vs K 曲線（Vanilla / Full 各一條）

- 感測噪聲（`data_noise_std`）：
  - $\sigma \in \{0, 0.01, 0.03\}$（相對均方噪聲）
  - 量化：L2 error vs noise；檢查 Full 模型是否比 Vanilla 對噪聲更敏感或更穩健

- Prior / Source term 開關（若使用 Kolmogorov 對應的設定）：
  - 無 prior / 無 S（純 PINN）
  - + prior（若有 Kolmogorov 低保真）
  - + S（源項）
  - full（prior + S）

### 1.3 建議輸出圖表

利用現有腳本：
- `scripts/visualize_kolmogorov_dns.py`
- `scripts/visualize_results.py`
- `scripts/comprehensive_evaluation.py`（若已支援 2D case）

**圖表清單**
- 場圖：
  - Ground truth vs prediction vs error（$u$, $v$）  
    - 檔名範例：`results/evaluation_re50_kf4_K100/field_u.png`, `field_v.png`

- 收斂行為：
  - Loss vs epoch（Vanilla vs Full，若已有 log 可從 TensorBoard 或 log 重新繪製）

- K-scan / 噪聲掃描（完成後）：
  - L2 error vs K（Vanilla / Full）
  - L2 error vs noise level

---

## 2. 3D JHTDB Channel Flow（高 Re 基線）

目標：把目前的 3D VS-PINN run（`results/comprehensive_eval_20251019_185222`）整理成：
- 清楚的「早期 baseline」（尚未達標）
- 與 JHTDB DNS／RANS baseline 的對照
- 為後續改進提供明確指標

### 2.1 DNS 與 RANS baseline（已存在資料）

現有結果：
- DNS 驗證與可視化：
  - `results/kolmogorov_dns/*`（2D）
  - channel DNS 可用：`scripts/verify_jhtdb_data.py`, `scripts/validate_dns_physics.py`, `scripts/validate_dns_resolution.py`
- RANS vs DNS：
  - `results/rans_vs_dns_re50/*`
  - `results/rans_vs_dns_re100/*`
  - `results/rans_vs_dns_re500/*`

**需要整理進論文或報告的內容**
- 統計對比圖：`statistics_comparison.png`
  - RANS vs DNS 的 mean velocity profile、RMS、TKE（目前圖中已有）
- 能量譜對比：`spectrum_comparison.png`
  - RANS 在高頻是否嚴重衰減
- 場圖對比：`field_comparison_u.png`, `field_comparison_v.png`

> 建議：在論文 Results 裡加入一小節，引用這些圖作「低保真先驗的偏差基準」，但不需增加新實驗。

### 2.2 VS-PINN run 的完整診斷（已存在資料）

現有 run：`results/comprehensive_eval_20251019_185222`
- 指標：`evaluation_metrics.json`
- 報告：`evaluation_report.md`
- 圖表：
  - `field_comparison_{u,v,w,p}.png`
  - `velocity_profiles_comparison.png`
  - `energy_spectrum_comparison.png`
  - `wall_shear_stress_comparison.png`
  - `statistics_comparison.png`
  - `error_distribution.png`

**需要在論文中引用的指標與圖**
- 相對 L2 error（目前約 99–100%）：
  - $u, v, w, p$（已寫入 Table 4.1）
- 速度剖面：
  - `velocity_profiles_comparison.png`：DNS vs PINN 在 $y$ 方向的 mean profile
- 牆面剪應力：
  - `wall_shear_stress_comparison.png`
  - keep: `tau_rel_error ≈ 100%`，說明目前 baseline 與 DNS 差距
- 能譜：
  - `energy_spectrum_comparison.png`
  - 当前 `spectrum_rel_error ≈ 740%`，可在 Discussion 中誠實呈現

> 這一階段的重點不是追數字，而是把「目前 run 場圖看起來還可以，但所有量化指標都遠未達標」講清楚，當作後續改進的起點。

### 2.3 之後建議的變數掃描（待未來實驗時用）

**感測點數 K（3D）**
- 固定其他設定，掃描：
  - $K \in \{32, 64, 100, 200\}$；
  - 使用 QR sensors（`scripts/visualize_qr_sensors.py`, `scripts/compare_sensor_strategies.py`）。
- 指標：
  - overall relative L2 error（`comprehensive_evaluation.py`）
  - 速度剖面 RMSE
- 圖表：
  - error vs K 曲線；
  - QR vs random sensor layout 的對比（已有 `compare_sensor_strategies.py` 及對應圖可以復用）。

**prior_weight / source term / VS-PINN scaling**
- 掃描 `configs/main.yml` 中的：
  - `prior_weight`（例如 0.0, 0.1, 0.3）
  - `source_l1`（是否啟用學習源項，及其強度）
  - `vs_pinn.scaling_factors`（目前 $(N_x,N_y,N_z)=(2,12,2)$，可做小幅調整）
- 每組配置：
  - 利用 `scripts/train.py` + `scripts/comprehensive_evaluation.py` 再跑一次短訓練；
  - 只需紀錄：overall L2、速度剖面 RMSE、能譜誤差三個核心指標。

> 這一段屬於「未來改進時」的參考計畫，不需要馬上寫進論文，只要實作時遵循即可。

---

## 3. 討論章節需要用到的素材整理清單

為了讓 Discussion 可以「逐條回答」論文目標，建議事先整理以下素材，全部都可以從現有結果或輕量級實驗得到：

1. **2D Kolmogorov**
   - Vanilla vs Full 的 L2 表（已整理）
   - 散度指標表（已整理）
   - 至少一張 Loss vs epoch 圖（可以從現有訓練 log 重繪）

2. **RANS vs DNS（Channel Flow）**
   - 一張統計比較圖（`statistics_comparison.png`，Re=100 case 即可）  
   - 一張能譜比較圖（`spectrum_comparison.png`）

3. **VS-PINN vs DNS（Channel Flow）**
   - 圖：`field_comparison_u/v/w/p.png`  
   - 圖：`velocity_profiles_comparison.png`  
   - 圖：`energy_spectrum_comparison.png`  
   - 圖：`wall_shear_stress_comparison.png`  
   - 表：`evaluation_metrics.json` 中的 L2、譜誤差、τ_w 誤差（已部分寫入論文）

> 建議：把這些圖表挑一小部分（每類 1–2 張）放進論文，其餘保留在補充材料或 repo 的 `results/` 中，讓讀者可以自行查閱。

---

## 4. 總結：執行順序建議

1. 整理 Kolmogorov 的 Vanilla vs Full 結果（已完成，論文部分已更新）。  
2. 在論文 Results 中加入 RANS vs DNS 的「定性」對比描述（引用現有圖，不新跑）。  
3. 利用現有 `comprehensive_eval_20251019_185222` 完整描述 VS-PINN channel run 的不足與之後要改善的指標。  
4. 若時間允許，再逐步加入 K-scan、prior_weight 掃描等實驗結果，優先更新 repo 的 `results/` 與技術文檔，最後再回填到論文。  

以上步驟都是建立在目前已有的腳本與結果之上，不需要修改核心程式碼，只是把現有能力「展示清楚」，並為後續實驗留出清楚的路線圖。

