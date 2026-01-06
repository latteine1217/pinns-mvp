# 📐 實驗設計文檔（Experiment Design Document）

**版本**: v2.1  
**日期**: 2025-01-05  
**狀態**: 完整實驗設計（包含 WandB Sweep 與新增實驗組）  
**依據**: thesis/main.tex 第4章、configs/sweeps/ 掃描配置、研究目標

**更新日誌**：
- v2.1 (2025-01-05): 新增 O1 (Optimizer)、RE (Reynolds)、E1 (Efficiency)、P1 (Physics) 實驗組，整合 WandB Sweep 配置
- v2.0 (2025-01-05): 基於論文需求的完整實驗設計

---

## 🎯 實驗設計總覽（Executive Summary）

### 研究目標（Research Objectives）

根據論文第1.4節（Objectives），本研究的核心目標為：

1. **開發穩定的逆問題 PINN 框架**：針對探測器受限的湍流重建
2. **定義可測試的稀疏度預算與成功標準**：$K/N$、$N_t$、$N_{\text{obs}}$ 標準化報告
3. **量化感測器佈置對極端稀疏度的影響**：QR-pivot vs Random，$K=100$
4. **建立可重現的 2D/3D 湍流基準評估**：Kolmogorov + JHTDB Channel ($Re_\tau \approx 1000$)

### 關鍵研究問題（Research Questions）

| ID | 問題 | 對應實驗 | 論文章節 |
|----|----|---------|---------|
| **RQ1** | QR-pivot 感測器佈置是否優於隨機佈置？ | S1, S2, J2 | §3.6, §4.1 |
| **RQ2** | 極端稀疏度（$K=100$，$K/N \sim 10^{-4}$）下重建的極限在哪？ | S2, M1, J3 | §1.3, §4.2 |
| **RQ3** | 低保真 RANS 先驗如何影響重建品質？ | C1, C2, J1 | §3.9, §4.1 |
| **RQ4** | Fourier Features、VS-PINN、RWF 等架構元件的貢獻為何？ | A1, A2, M1 | §3.4, §3.5 |
| **RQ5** | 2D Kolmogorov 的結論是否可推廣至 3D Channel Flow？ | Phase 1 → Phase 2 | §4.1, §4.2 |
| **RQ6** | 不同優化器（Adam vs SOAP）對收斂速度的影響？ | O1 | §3.7, §3.7.1 |
| **RQ7** | 方法在不同 Reynolds 數下的可擴展性如何？ | RE | §1.3, §5 |
| **RQ8** | 計算成本與精度的權衡關係如何？ | E1 | §5 (Discussion) |
| **RQ9** | 模型是否真正滿足物理約束（不只是低 $L_2$ 誤差）？ | P1 | §3.7.2, §4 |

### 驗收指標（Acceptance Criteria）

根據論文第1.4節與第3章，定義以下驗收標準：

#### 主要指標（Primary Metrics）
- ✅ **全場相對 $L_2$ 誤差**：$\leq 10\%$（長期目標），當前基線 $\sim 130\%$（需改善）
- ✅ **優於 RANS Baseline**：$\geq 30\%$ 改善
- ✅ **感測器數量**：$K \leq 100$（QRCP 選點）
- ✅ **收斂速度提升**：$\geq 30\%$（相較 vanilla Adam）

#### 物理一致性指標（Physics Constraints）
- ✅ **不可壓縮性**：$\|\nabla \cdot \mathbf{u}\|_2 < 10^{-3}$（均值），$\max < 10^{-2}$
- ✅ **邊界條件**：壁面 $\|\mathbf{u}\|_{\text{wall}} < 10^{-4}$，週期性誤差 $< 10^{-3}$
- ✅ **質量守恆**：相對誤差 $< 1\%$

#### 工程診斷指標（Engineering Diagnostics，3D Channel Flow）
- ✅ **壁面剪應力**：$\tau_w$ 相對誤差 $< 20\%$
- ✅ **平均速度剖面**：$U^+(y^+)$ RMSE $< 0.5$
- ✅ **能譜重建**：低頻（$k < k_f$）誤差 $< 30\%$

---

## 📊 實驗矩陣（Experiment Matrix）

### Phase 0：環境驗證（Infrastructure Validation）

| ID | 實驗名稱 | 目的 | 配置文件 | 預估時間 | 優先級 |
|----|---------|-----|---------|---------|--------|
| **E0.1** | Import 測試 | 驗證所有模組可正常匯入 | `scripts/validate_imports.py` | 5 min | P0 |
| **E0.2** | Quick Train 測試 | 100 epochs 快速訓練驗證 | `configs/quick_test.yml` | 30 min | P0 |
| **E0.3** | Data Pipeline 測試 | JHTDB 與 Kolmogorov 資料載入 | `scripts/validate_data_pipeline.py` | 10 min | P0 |

**成功標準**：無 ImportError、NaN、或 OOM 錯誤

---

### Phase 1：2D Kolmogorov Flow（低成本驗證）

#### 實驗組 S1：感測器策略對比（Sensor Strategy Comparison）

**研究問題**：RQ1  
**論文依據**：§3.6（Sensor Placement via QR Pivoting）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **S1.1** | QR-pivot | K=100, Re=50, Vanilla MLP | `S1_sensor_strategy/s1_qr_K100_2d_re50.yml` | 3-5 h |
| **S1.2** | Random | K=100, Re=50, Vanilla MLP | `S1_sensor_strategy/s1_random_K100_2d_re50.yml` | 3-5 h |

**對比指標**：
- 條件數（Condition Number）：$\kappa(\mathbf{C}\mathbf{U}_r)$
- 全場 $L_2$ 誤差：$u$, $v$, $p$
- 壓力梯度誤差：$\|\nabla p\|$
- 不可壓縮性：$\|\nabla \cdot \mathbf{u}\|$

**預期結果**：QR-pivot 應優於 Random $\geq 20\%$

---

#### 實驗組 S2：感測器數量掃描（K-Scan）

**研究問題**：RQ2  
**論文依據**：§1.3（Motivation - Gap 1）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **S2.1** | K=30 | QR-pivot, Re=50, Vanilla | `S2_k_scan/s2_qr_K30_2d_re50.yml` | 3 h |
| **S2.2** | K=50 | QR-pivot, Re=50, Vanilla | `S2_k_scan/s2_qr_K50_2d_re50.yml` | 3 h |
| **S2.3** | K=80 | QR-pivot, Re=50, Vanilla | `S2_k_scan/s2_qr_K80_2d_re50.yml` | 4 h |
| **S2.4** | K=100 | QR-pivot, Re=50, Vanilla | `S2_k_scan/s2_qr_K100_2d_re50.yml` | 5 h |

**對比指標**：
- K-error 曲線：$\text{rel}\,L_2(u)$ vs $K$
- 稀疏度指標：$K/N$ 與 $N_{\text{obs}}$

**預期結果**：識別出「最小可行感測器數量」$K_{\min}$

---

#### 實驗組 M1：模型能力對比（Model Capacity Comparison）

**研究問題**：RQ4  
**論文依據**：§3.4（Network Architecture）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **M1.1** | Vanilla MLP | K=100, QR-pivot, no prior | `M1_model_comparison/m1_vanilla_K100_2d_re50.yml` | 5 h |
| **M1.2** | Full (Fourier+VS+RWF) | K=100, QR-pivot, no prior | `M1_model_comparison/m1_full_K100_2d_re50.yml` | 6 h |

**對比指標**：
- 收斂速度（至目標 loss）
- 最終重建誤差
- 訓練穩定性（loss 波動）

**預期結果**：Full 版本應優於 Vanilla $\geq 30\%$

---

#### 實驗組 A1：Fourier Features 消融（Ablation: Fourier）

**研究問題**：RQ4  
**論文依據**：§3.4.1（Fourier Feature Embeddings）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **A1.1** | 啟用 Fourier ($m=16$, $\sigma=4$) | K=100, QR-pivot | `A1_ablation_fourier/a1_with_fourier_K100_2d_re50.yml` | 5 h |
| **A1.2** | 禁用 Fourier | K=100, QR-pivot | `A1_ablation_fourier/a1_without_fourier_K100_2d_re50.yml` | 5 h |

**對比指標**：
- 高頻重建能力（能譜高頻區）
- 壓力梯度誤差
- 收斂穩定性

**預期結果**：Fourier 應改善高頻重建 $\geq 15\%$

---

#### 實驗組 A2：自適應權重消融（Ablation: Adaptive Weighting）

**研究問題**：RQ4  
**論文依據**：§3.8（Adaptive Weighting Strategy）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **A2.1** | GradNorm 啟用 | K=100, Full model | `A2_ablation_weights/a2_with_adaptive_K100_2d_re50.yml` | 6 h |
| **A2.2** | 固定權重 | K=100, Full model | `A2_ablation_weights/a2_without_adaptive_K100_2d_re50.yml` | 5 h |

**對比指標**：
- Loss 平衡度（各項 loss 的動態範圍）
- 收斂速度
- 最終誤差

**預期結果**：自適應權重應改善收斂速度 $\geq 20\%$

---

#### 實驗組 C1：RANS Prior 對比（Prior Comparison）

**研究問題**：RQ3  
**論文依據**：§3.9（Low-Fidelity Prior）、Appendix D（Kolmogorov RANS Setup）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **C1.1** | 無 Prior | K=100, Full model | `C1_prior_comparison/c1_no_prior_K100_2d_re50.yml` | 5 h |
| **C1.2** | Leith Prior (weight=1.0) | K=100, Full model | `C1_prior_comparison/c1_with_prior_K100_2d_re50.yml` | 6 h |

**對比指標**：
- 全場 $L_2$ 改善
- 能譜結構保留
- 不可壓縮性滿足

**預期結果**：Prior 應改善重建 $\geq 10\%$（對比論文 Table 5）

---

#### 實驗組 C2：Prior 權重掃描（Prior Weight Sweep）

**研究問題**：RQ3  
**論文依據**：§3.9（Eq. 3.27 - Prior Loss）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **C2.1** | prior_weight=0.1 | K=100, Full model | `C2_prior_sweep/c2_prior_0.1_K100_2d_re50.yml` | 5 h |
| **C2.2** | prior_weight=0.3 | K=100, Full model | `C2_prior_sweep/c2_prior_0.3_K100_2d_re50.yml` | 5 h |
| **C2.3** | prior_weight=0.5 | K=100, Full model | `C2_prior_sweep/c2_prior_0.5_K100_2d_re50.yml` | 5 h |

**對比指標**：
- Prior-error 曲線
- 最佳 sweet spot 識別
- Bias injection 風險評估

**預期結果**：識別出最佳 $w_{\text{prior}} \approx 0.3$

---

#### 實驗組 O1：優化器對比（Optimizer Comparison）

**研究問題**：不同優化器對收斂速度與最終精度的影響  
**論文依據**：§3.7（Optimization Strategy）、§3.7.1（SOAP Optimizer）

| Exp ID | 變因 | 固定參數 | 配置方式 | 預估時間 |
|--------|-----|---------|---------|---------|
| **O1.1** | Adam (lr=1e-3) | K=100, QR-pivot, Full model | WandB Sweep | 5 h |
| **O1.2** | SOAP (lr=5e-4) | K=100, QR-pivot, Full model | WandB Sweep | 6 h |
| **O1.3** | AdamW (lr=1e-3) | K=100, QR-pivot, Full model | WandB Sweep | 5 h |
| **O1.4-O1.9** | 學習率掃描 (3 optimizers × 3 LRs) | K=100, QR-pivot, Full model | WandB Sweep | 15 h |

**對比指標**：
- **收斂速度**：達到目標精度（$L_2 < 15\%$）所需的 epochs 與 wall-time
- **最終精度**：best $L_2$ error
- **訓練穩定性**：loss 曲線波動程度
- **記憶體開銷**：SOAP 的 preconditioner 記憶體成本

**預期結果**：
- SOAP 應在收斂速度上優於 Adam $\geq 30\%$（論文主張）
- 最佳學習率：Adam/AdamW ~1e-3，SOAP ~5e-4

**配置文件**：`configs/sweeps/sweep_o1_optimizer.yaml`（WandB Bayesian/Grid Sweep）

---

#### 實驗組 RE：Reynolds 數可擴展性驗證（Reynolds Number Scalability）

**研究問題**：方法在不同 Reynolds 數下的泛化能力  
**論文依據**：§1.3（Gap 1 - 可擴展性）、審稿必問問題

| Exp ID | 變因 | 固定參數 | 配置方式 | 預估時間 |
|--------|-----|---------|---------|---------|
| **RE.1** | Re=50 | K=100, QR-pivot, Full model | WandB Sweep | 5 h |
| **RE.2** | Re=100 | K=100, QR-pivot, Full model | WandB Sweep | 6 h |
| **RE.3** | Re=200 | K=100, QR-pivot, Full model | WandB Sweep | 8 h |
| **RE.4** | Re=500 | K=100, QR-pivot, Full model | WandB Sweep | 12 h |

**對比指標**：
- **Re-error 曲線**：$L_2$ error vs Reynolds number
- **收斂難度**：訓練時間 vs Re
- **物理約束滿足**：散度 vs Re（高 Re 更難滿足）
- **能譜重建**：高 Re 的高頻成分重建能力

**預期結果**：
- 量化誤差增長率：$\text{Error} \propto Re^\alpha$（預期 $\alpha \approx 0.3\text{-}0.5$）
- Re=500 應仍可收斂（雖然誤差增加）
- 證明方法在「中等 Re」下的可擴展性

**注意事項**：
- ⚠️ **需要預先生成不同 Re 的 DNS 數據**（Re=100/200/500）
- ⚠️ **需要為每個 Re 生成對應的 QR-pivot 感測器**
- 建議先跑 Re=50, 100 驗證流程，再跑 Re=200, 500

**配置文件**：`configs/sweeps/sweep_re_scan.yaml`（WandB Grid Sweep）

---

#### 實驗組 E1：計算效率分析（Efficiency Analysis）

**研究問題**：不同架構的計算成本（訓練時間、記憶體、推理速度）  
**論文依據**：§5（Discussion - 計算成本考量）

| Exp ID | 變因 | 固定參數 | 配置方式 | 預估時間 |
|--------|-----|---------|---------|---------|
| **E1.1-E1.18** | 模型寬度 × 深度 × Batch size | K=100, QR-pivot | WandB Sweep | 20 h |
|  | width ∈ {128, 256, 512} | | | |
|  | depth ∈ {4, 6, 8} | | | |
|  | batch_size ∈ {5000, 10000, 15000} | | | |

**追蹤指標**：
- **訓練時間**：total_wall_time, time_per_epoch, time_to_convergence
- **記憶體使用**：peak_gpu_memory, model_parameters
- **推理速度**：inference_time_per_sample, throughput
- **效率分數**：$\text{Efficiency} = \frac{\text{Accuracy}}{\text{Training Time}}$

**對比指標**：
- **Pareto Frontier**：精度 vs 訓練時間（找最佳權衡點）
- **Memory Scaling**：模型參數量 vs GPU 記憶體
- **Inference Efficiency**：模型大小 vs 推理速度

**預期結果**：
- Full 模型雖參數多，但訓練時間增加有限（< 2×）
- 量化 Fourier Features 的計算成本 vs 精度收益
- 提供實務部署的模型選型建議

**注意事項**：
- ⚠️ **必須在相同硬體執行**（避免 wall-time 比較失真）
- 建議單 GPU 執行（`CUDA_VISIBLE_DEVICES=0`）
- 記錄硬體資訊：GPU 型號、CUDA 版本、PyTorch 版本

**配置文件**：`configs/sweeps/sweep_e1_efficiency_analysis.yaml`（WandB Grid Sweep）

---

#### 實驗組 P1：物理約束驗證（Physics Validation）

**研究問題**：模型是否真正滿足物理約束（守恆定律、邊界條件）  
**論文依據**：§3.7.2（Physics-Informed Loss）、§4（Results 驗證）

| Exp ID | 變因 | 固定參數 | 配置方式 | 預估時間 |
|--------|-----|---------|---------|---------|
| **P1.1** | Vanilla (no prior) | K=100, QR-pivot | 重用 M1.1 結果 | - |
| **P1.2** | Full (no prior) | K=100, QR-pivot | 重用 M1.2 結果 | - |
| **P1.3** | Full + Prior | K=100, QR-pivot | 重用 C1.2 結果 | - |

**追蹤指標**：

1. **質量守恆（Incompressibility）**
   - $\|\nabla \cdot \mathbf{u}\|$ (mean, max, 99th percentile, L2 norm)
   - 相對散度：$\|\nabla \cdot \mathbf{u}\| / \|\mathbf{u}\|$

2. **動量守恆（Momentum Conservation）**
   - NS 方程殘差（x/y/z 方向）
   - 最大動量殘差

3. **能量守恆（Energy Conservation）**
   - TKE 誤差（vs DNS）
   - 能量耗散率誤差
   - 擾度誤差（2D Kolmogorov）

4. **邊界條件（Boundary Conditions）**
   - 週期性誤差（Kolmogorov）
   - 壁面速度違反（Channel flow）

5. **綜合物理違反分數（Composite Score）**
   - $\text{Score} = w_1 \cdot \|\nabla \cdot \mathbf{u}\|_2 + w_2 \cdot \text{Momentum Residual} + w_3 \cdot \text{BC Violation}$
   - 建議權重：$w_1=10$, $w_2=1$, $w_3=5$

**對比指標**：
- Radar Chart：各架構的物理指標（質量、動量、能量、BC）
- Scatter Plot：$L_2$ error vs physics_violation_score（檢驗 data-PDE trade-off）
- Heatmap：散度空間分佈（DNS vs Vanilla vs Full）

**預期結果**：
- 證明「低 $L_2$ 誤差」≠「物理合理」
- Full 模型在物理約束滿足度上優於 Vanilla
- RANS 先驗有助於改善物理合理性

**注意事項**：
- 本實驗主要是**後處理分析**，重用已有實驗結果
- 需在高解析度網格（512×512）評估物理指標
- DNS 本身也有散度 floor（operator discretization），需報告

**配置文件**：`configs/sweeps/sweep_p1_physics_validation.yaml`（WandB Grid Sweep）

---

### Phase 1.5：WandB Sweep 實驗總覽（Sweep Experiments Summary）

**說明**：部分實驗使用 WandB Sweep 進行參數空間探索，相關配置位於 `configs/sweeps/`。

#### Sweep 配置映射表

| Sweep 文件 | 對應實驗組 | 方法 | 參數範圍 | 預期 Runs |
|-----------|-----------|------|---------|----------|
| `sweep_s1_sensor_strategy.yaml` | S1 | Grid | 2 configs | 2 |
| `sweep_s2_k_scan.yaml` | S2 | Grid | 4 configs | 4 |
| `sweep_m1_model_comparison.yaml` | M1 | Grid | 2 configs | 2 |
| `sweep_a1_fourier_ablation.yaml` | A1 | Grid | 2 configs | 2 |
| `sweep_a2_weights_ablation.yaml` | A2 | Grid | 2 configs | 2 |
| `sweep_c1_prior_comparison.yaml` | C1 | Grid | 2 configs | 2 |
| `sweep_c2_prior_weight.yaml` | C2 | **Bayes** | prior_weight ∈ [0.0, 0.5] | 10-15 |
| `sweep_o1_optimizer.yaml` | **O1** | Grid | 3 optimizers × 3 LRs | 9 |
| `sweep_re_scan.yaml` | **RE** | Grid | 4 Reynolds numbers | 4 |
| `sweep_e1_efficiency_analysis.yaml` | **E1** | Grid | width × depth × batch | 18+ |
| `sweep_p1_physics_validation.yaml` | **P1** | Grid | 4 configs | 4 |
| `sweep_r1_robustness.yaml` | R1 | Grid | 3 noise × 2 dropout × 3 seeds | 18 |

**Sweep vs 單獨配置的使用場景**：

- **單獨配置**（`configs/experiments/`）：
  - 用於核心對比實驗（S1, M1, C1）
  - 每個配置文件獨立可執行
  - 適合精確控制與重現

- **WandB Sweep**（`configs/sweeps/`）：
  - 用於參數空間探索（C2, O1, E1）
  - 自動並行執行多組參數
  - Bayesian 優化找最佳超參數（C2）
  - 方便 WandB 可視化與對比

**執行方式**：

```bash
# 方法 1：執行單獨配置
python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml

# 方法 2：執行 WandB Sweep
wandb sweep configs/sweeps/sweep_s1_sensor_strategy.yaml  # 建立 sweep
wandb agent <entity>/<project>/<sweep_id>                  # 啟動 agent
```

---

### Phase 2：3D Channel Flow (JHTDB, $Re_\tau = 1000$)

#### 實驗組 J1：JHTDB Baseline（3D Baseline）

**研究問題**：RQ5  
**論文依據**：§4.2（3D Channel Flow Reconstruction Results）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **J1.1** | Vanilla (no prior) | K=100, QR-pivot, 2D slice | `experiments/J1_jhtdb_baseline/j1_vanilla_K100_3d.yml` | 12 h |
| **J1.2** | +RANS Prior | K=100, QR-pivot, 2D slice | `experiments/J1_jhtdb_baseline/j1_with_prior_K100_3d.yml` | 15 h |

**對比指標**：
- 全場 $L_2$ 誤差（$u$, $v$, $w$, $p$）
- 壁面剪應力誤差
- 平均速度剖面 $U^+(y^+)$ RMSE
- 能譜 $E(k)$ 對比（低/中/高頻）

**預期結果**：
- Vanilla: $L_2 \sim 100\%$（過度平滑，論文 §4.2）
- +Prior: 改善 $\geq 10\%$

---

#### 實驗組 J2：3D QR vs Random（3D Sensor Strategy）

**研究問題**：RQ1（3D 推廣）  
**論文依據**：§3.6、Fig. 3.6（QR vs Random Sensors）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **J2.1** | QR-pivot | K=100, 2D slice, Full model | `experiments/J2_jhtdb_sensor/j2_qr_K100_3d.yml` | 12 h |
| **J2.2** | Random | K=100, 2D slice, Full model | `experiments/J2_jhtdb_sensor/j2_random_K100_3d.yml` | 12 h |

**對比指標**：
- 條件數對比
- 壁面附近感測器覆蓋率
- 重建誤差（特別是近壁區）

**預期結果**：QR-pivot 優於 Random $\geq 15\%$（3D 更顯著）

---

#### 實驗組 J3：3D K-Scan（尋找 3D 最小感測器數）

**研究問題**：RQ2（3D 推廣）  
**論文依據**：§1.3（Gap 1 - Minimal sensing under extreme sparsity）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **J3.1** | K=50 | QR-pivot, 2D slice | `experiments/J3_jhtdb_kscan/j3_K50_3d.yml` | 10 h |
| **J3.2** | K=100 | QR-pivot, 2D slice | `experiments/J3_jhtdb_kscan/j3_K100_3d.yml` | 12 h |
| **J3.3** | K=200 | QR-pivot, 2D slice | `experiments/J3_jhtdb_kscan/j3_K200_3d.yml` | 15 h |
| **J3.4** | K=500 | QR-pivot, 2D slice | `experiments/J3_jhtdb_kscan/j3_K500_3d.yml` | 20 h |

**對比指標**：
- K-error 曲線（3D）
- $K_{\min}$ 識別（達到 $L_2 < 20\%$ 的最小 K）

**預期結果**：$K_{\min}^{3D} \geq 200$（對比 2D $K_{\min} \sim 100$）

---

#### 實驗組 J4：3D 完整體積重建（Full 3D Volume，選做）

**研究問題**：RQ5（完整 3D 推廣）  
**論文依據**：§5（Future Work）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **J4.1** | 完整 3D 體積 | K=100, QR-pivot, Full model | `experiments/J4_jhtdb_volume/j4_full_volume_K100.yml` | 48 h |

**對比指標**：
- 3D 全體積 $L_2$ 誤差
- 3D 能譜 $E(k_x, k_y, k_z)$
- Reynolds 應力張量 $\langle u_i' u_j' \rangle$

**狀態**：選做（資源允許）

---

### Phase 3：魯棒性測試（Robustness Analysis）

#### 實驗組 R1：噪聲敏感度測試（Noise Sensitivity）

**研究問題**：實驗設定的魯棒性  
**論文依據**：§1.2（Literature Review - Observation modality and noise）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **R1.1** | 無噪聲（$\sigma=0$） | K=100, QR, 2D Re=50 | `experiments/R1_noise/r1_noise_0_K100_2d.yml` | 5 h |
| **R1.2** | 1% 噪聲（$\sigma=0.01$） | K=100, QR, 2D Re=50 | `experiments/R1_noise/r1_noise_1pct_K100_2d.yml` | 5 h |
| **R1.3** | 3% 噪聲（$\sigma=0.03$） | K=100, QR, 2D Re=50 | `experiments/R1_noise/r1_noise_3pct_K100_2d.yml` | 5 h |

**對比指標**：
- 誤差隨噪聲的增長率
- 收斂穩定性
- 不可壓縮性約束滿足

**預期結果**：$L_2$ 誤差增長 $\propto \sigma$，3% 噪聲下仍可收斂

---

#### 實驗組 R2：資料遺失測試（Dropout Sensitivity）

**研究問題**：感測器失效魯棒性  
**論文依據**：實驗設定補充

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **R2.1** | 無遺失（dropout=0） | K=100, QR, 2D Re=50 | `experiments/R2_dropout/r2_dropout_0_K100_2d.yml` | 5 h |
| **R2.2** | 10% 遺失（dropout=0.1） | K=100, QR, 2D Re=50 | `experiments/R2_dropout/r2_dropout_10pct_K100_2d.yml` | 5 h |

**對比指標**：
- 有效感測器數量 $K_{\text{eff}} = K \times (1 - \text{dropout})$
- 重建誤差增長

**預期結果**：10% dropout 應對應 $\sim 15\%$ 誤差增長

---

#### 實驗組 R3：隨機種子測試（Reproducibility）

**研究問題**：結果可重現性  
**論文依據**：§1.2（Gap 4 - Reproducible benchmarking）

| Exp ID | 變因 | 固定參數 | 配置文件 | 預估時間 |
|--------|-----|---------|---------|---------|
| **R3.1** | seed=42 | K=100, QR, Full, 2D Re=50 | `experiments/R3_seeds/r3_seed_42_K100_2d.yml` | 5 h |
| **R3.2** | seed=123 | K=100, QR, Full, 2D Re=50 | `experiments/R3_seeds/r3_seed_123_K100_2d.yml` | 5 h |
| **R3.3** | seed=456 | K=100, QR, Full, 2D Re=50 | `experiments/R3_seeds/r3_seed_456_K100_2d.yml` | 5 h |

**對比指標**：
- 均值 ± 標準差（3 seeds）
- 變異係數 CV $< 10\%$

**預期結果**：結果應穩定（$\text{CV} < 5\%$）

---

## 📈 預期論文圖表（Expected Figures for Thesis）

根據論文結構，建議生成以下圖表：

### Chapter 3: Methodology
- **Fig. 3.1**: Framework 流程圖（已有：`result_figures/framework/framework.png`）
- **Fig. 3.2**: VS-PINN 架構圖（已有：`result_figures/framework/PINNs_structure.png`）
- **Fig. 3.3**: QR vs Random 感測器佈置（2D Kolmogorov）→ **實驗 S1**
- **Fig. 3.4**: RANS vs DNS 速度剖面（已有：`result_figures/channel_flow/rans_dns_velocity_profiles.png`）

### Chapter 4: Experiments and Results

#### 4.1 2D Kolmogorov Flow
- **Fig. 4.1**: Leith 模型誤差 vs Reynolds 數（已有：`result_figures/kolmogorov/fig_leith_error_scaling.png`）
- **Fig. 4.2**: Leith vs DNS 能譜對比（已有：`result_figures/kolmogorov/leith_dns_spectrum_re50.png`）
- **Fig. 4.3**: Vanilla vs Soft Prior 場重建對比 → **實驗 C1**
  - 生成自：`C1_prior_comparison/` 結果
  - 包含：$u$, $v$, $p$, $\omega_z$, $|\mathbf{u}|$, $|\nabla p|$
- **Fig. 4.4**: 訓練動態對比（loss curves）→ **實驗 C1**
  - 生成自：TensorBoard logs
- **Table 4.1**: Vanilla vs Soft Prior 誤差表（已有：論文 Table 5）→ **實驗 C1**
- **Table 4.2**: 物理約束滿足（已有：論文 Table 6）→ **實驗 C1**
- **Fig. 4.5**: K-error 曲線（2D）→ **實驗 S2**
  - X軸：$K \in \{30, 50, 80, 100\}$
  - Y軸：rel $L_2$ error
- **Fig. 4.6**: Ablation Bar Chart → **實驗 A1, A2**
  - Vanilla vs +Fourier vs +Adaptive
- **Fig. 4.7 (新增)**: Optimizer Comparison → **實驗 O1**
  - Bar chart: Adam vs SOAP vs AdamW 的收斂速度與最終精度
  - Convergence curves: Loss vs epoch（3 條曲線）
- **Fig. 4.8 (新增)**: Reynolds Scalability → **實驗 RE**
  - Line chart: Re (x) vs $L_2$ error (y)
  - 量化 scaling law: $\text{Error} \propto Re^\alpha$
- **Fig. 4.9 (新增)**: Efficiency Pareto Frontier → **實驗 E1**
  - Scatter plot: Training time (x) vs $L_2$ error (y), size=model_params
  - 標示最佳權衡點
- **Fig. 4.10 (新增)**: Physics Validation Radar Chart → **實驗 P1**
  - Radar chart: 質量守恆、動量守恆、能量守恆、BC 滿足度
  - 對比 Vanilla vs Full vs Full+Prior

#### 4.2 3D Channel Flow (JHTDB)
- **Fig. 4.7**: JHTDB 參考場（已有：`result_figures/channel_flow/channel_jhtdb_reference_u.png`）
- **Fig. 4.8**: QR vs Random 感測器佈置（3D）（已有：`result_figures/sensors/sensor_comparison_jhtdb_K100.png`）
- **Fig. 4.9**: 3D Channel 場重建（$u$, $v$, $w$, $p$）→ **實驗 J1**
  - 已有初步結果：`channel_field_{u,v,w,p}.png`
- **Fig. 4.10**: 3D 能譜對比（已有：`result_figures/channel_flow/channel_energy_spectrum_comparison.png`）
- **Fig. 4.11**: 平均速度剖面 $U^+(y^+)$ → **實驗 J1**
- **Fig. 4.12**: K-error 曲線（3D）→ **實驗 J3**

### Chapter 5: Discussion & Conclusion
- **Fig. 5.1**: 2D vs 3D 對比總結
- **Table 5.1**: 所有實驗結果總表

---

## 🗓️ 執行時程（Execution Timeline）

### 總時程估計（基於雙 P100 GPU）

| Phase | 實驗組 | 預估時間 | 累積時間 |
|-------|-------|---------|---------|
| **Phase 0** | E0.1-E0.3 | 1 小時 | 1 h |
| **Phase 1** | S1, S2, M1, A1, A2, C1, C2 | 80 小時 | 81 h (~3.5 天) |
| **Phase 1.5 (新增)** | O1, RE, E1, P1 | 65 小時 | 146 h (~6 天) |
| **Phase 2** | J1, J2, J3 | 120 小時 | 266 h (~11 天) |
| **Phase 3** | R1, R2, R3 | 50 小時 | 316 h (~13 天) |

**總計**：約 **13-14 天**（連續運算）或 **4 週**（考慮除錯與分析）

#### 新增實驗時間明細（Phase 1.5）

| 實驗組 | 描述 | 預估時間 | 備註 |
|-------|-----|---------|------|
| **O1** | Optimizer Comparison (3 opt × 3 LR) | 16 小時 | WandB Sweep, 9 runs |
| **RE** | Reynolds Scan (Re=50/100/200/500) | 31 小時 | 需預先生成 DNS 數據 |
| **E1** | Efficiency Analysis (width × depth × batch) | 20 小時 | WandB Sweep, 18+ runs |
| **P1** | Physics Validation (4 configs) | 0 小時 | 後處理分析，重用已有結果 |

**注意**：
- **RE** 實驗需要預先生成 Re=100/200/500 的 DNS 數據與感測器（準備時間另計）
- **P1** 為後處理分析，主要是計算物理指標，不需額外訓練
- **E1** 與其他實驗可並行執行（使用不同 GPU）

### 優先級排序（Priority Ordering）

根據論文撰寫需求，建議執行順序：

#### Week 1：核心基線（論文 §4.1 必要）
1. **E0.1-E0.3**：環境驗證（P0）
2. **S1**：QR vs Random（P0）
3. **C1**：Vanilla vs Soft Prior（P0）
4. **M1**：Vanilla vs Full（P1）

#### Week 2：消融與掃描（論文 §4.1 補充）
5. **S2**：K-scan（P1）
6. **A1**：Fourier 消融（P1）
7. **A2**：自適應權重消融（P2）
8. **C2**：Prior 權重掃描（P2）
9. **O1**：Optimizer 對比（P1，新增）
10. **E1**：Efficiency 分析（P2，新增，可並行）

#### Week 3：可擴展性驗證（論文 §5 必要）
11. **RE**：Reynolds 掃描（P1，新增，審稿必問）
12. **P1**：Physics Validation 後處理（P1，新增）

#### Week 4：3D 推廣（論文 §4.2 必要）
13. **J1**：JHTDB Baseline（P0）
14. **J2**：3D QR vs Random（P1）
15. **J3**：3D K-scan（P2，時間允許）

#### Week 5：魯棒性（選做）
16. **R1**：噪聲敏感度（P2）
17. **R2**：Dropout（P3）
18. **R3**：隨機種子（P3）

---

## 🔧 配置文件生成（Configuration Generation）

### 當前狀態

#### 已有單獨配置（位於 `configs/experiments/`）：
- ✅ **S1**：`S1_sensor_strategy/` (2 configs)
- ✅ **S2**：`S2_k_scan/` (4 configs)
- ✅ **M1**：`M1_model_comparison/` (2 configs)
- ✅ **A1**：`A1_ablation_fourier/` (2 configs)
- ✅ **A2**：`A2_ablation_weights/` (2 configs)
- ✅ **C1**：`C1_prior_comparison/` (2 configs)
- ✅ **C2**：`C2_prior_sweep/` (3 configs)

**Phase 1 完成度**: ✅ 100% (17 configs)

#### 已有 WandB Sweep 配置（位於 `configs/sweeps/`）：
- ✅ **sweep_s1_sensor_strategy.yaml** → S1 實驗組
- ✅ **sweep_s2_k_scan.yaml** → S2 實驗組
- ✅ **sweep_m1_model_comparison.yaml** → M1 實驗組
- ✅ **sweep_a1_fourier_ablation.yaml** → A1 實驗組
- ✅ **sweep_a2_weights_ablation.yaml** → A2 實驗組
- ✅ **sweep_c1_prior_comparison.yaml** → C1 實驗組
- ✅ **sweep_c2_prior_weight.yaml** → C2 實驗組（Bayesian 優化）
- ✅ **sweep_o1_optimizer.yaml** → **O1 實驗組**（新增）
- ✅ **sweep_re_scan.yaml** → **RE 實驗組**（新增）
- ✅ **sweep_e1_efficiency_analysis.yaml** → **E1 實驗組**（新增）
- ✅ **sweep_p1_physics_validation.yaml** → **P1 實驗組**（新增）
- ✅ **sweep_r1_robustness.yaml** → R1 實驗組（部分完成）

**Sweep 配置完成度**: ✅ 100% (12 sweep 文件)

### 需要新增的配置

#### Phase 2: 3D Channel Flow (JHTDB)

需要建立以下新配置：

```
configs/experiments/
├── J1_jhtdb_baseline/
│   ├── j1_vanilla_K100_3d.yml          # 基於 standard_config_template.yml
│   └── j1_with_prior_K100_3d.yml       # 啟用 lowfi_prior.enabled: true
├── J2_jhtdb_sensor/
│   ├── j2_qr_K100_3d.yml               # 預設 QR-pivot
│   └── j2_random_K100_3d.yml           # sensors.selection_method: random
├── J3_jhtdb_kscan/
│   ├── j3_K50_3d.yml                   # sensors.K: 50
│   ├── j3_K100_3d.yml                  # sensors.K: 100
│   ├── j3_K200_3d.yml                  # sensors.K: 200
│   └── j3_K500_3d.yml                  # sensors.K: 500
└── J4_jhtdb_volume/                    # 選做
    └── j4_full_volume_K100.yml         # 3D 全體積
```

#### Phase 3: 魯棒性測試

```
configs/experiments/
├── R1_noise/
│   ├── r1_noise_0_K100_2d.yml          # normalization.noise_sigma: 0.0
│   ├── r1_noise_1pct_K100_2d.yml       # normalization.noise_sigma: 0.01
│   └── r1_noise_3pct_K100_2d.yml       # normalization.noise_sigma: 0.03
├── R2_dropout/
│   ├── r2_dropout_0_K100_2d.yml        # normalization.dropout_prob: 0.0
│   └── r2_dropout_10pct_K100_2d.yml    # normalization.dropout_prob: 0.1
└── R3_seeds/
    ├── r3_seed_42_K100_2d.yml          # experiment.seed: 42
    ├── r3_seed_123_K100_2d.yml         # experiment.seed: 123
    └── r3_seed_456_K100_2d.yml         # experiment.seed: 456
```

### 配置生成腳本

可使用現有的 `configs/experiments/generate_experiment_configs.py` 為模板，擴充以下功能：

```python
# 建議新增函數：
def generate_jhtdb_configs():
    """生成 J1-J4 的 JHTDB 配置"""
    pass

def generate_robustness_configs():
    """生成 R1-R3 的魯棒性測試配置"""
    pass
```

---

## 📊 評估指標計算（Metrics Calculation）

### 主要指標實作位置

| 指標 | 計算公式 | 實作位置 | 備註 |
|-----|---------|---------|------|
| **rel $L_2$** | $\frac{\|\phi_{\text{pred}} - \phi_{\text{DNS}}\|_2}{\|\phi_{\text{DNS}}\|_2}$ | `src/evaluation/metrics.py::compute_relative_l2()` | 論文 Eq. 3.28 |
| **Divergence** | $\|\nabla \cdot \mathbf{u}\|$ (mean, max) | `src/evaluation/physics_metrics.py::compute_divergence()` | 論文 §3.7.2 |
| **$\nabla p$ error** | $\text{rel}\,L_2(\nabla p_x, \nabla p_y)$ | `src/evaluation/metrics.py::compute_pressure_gradient_error()` | 避免 gauge 問題 |
| **$U^+(y^+)$ RMSE** | $\sqrt{\frac{1}{N_y} \sum (\langle u \rangle - \langle u \rangle_{\text{DNS}})^2}$ | `src/evaluation/channel_metrics.py::compute_profile_rmse()` | 3D Channel only |
| **$\tau_w$ error** | $\frac{\|\tau_w - \tau_{w,\text{DNS}}\|}{\tau_{w,\text{DNS}}}$ | `src/evaluation/channel_metrics.py::compute_wall_shear_stress()` | 3D Channel only |
| **Energy spectrum** | $E(k) = \int \hat{u}(k) \cdot \hat{u}^*(k) dk$ | `src/evaluation/spectrum.py::compute_energy_spectrum()` | 論文 Fig. 4.10 |

### 評估腳本範例

```bash
# 評估單一 checkpoint
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/experiments/C1_with_prior_K100/best_model.pth \
  --config configs/experiments/C1_prior_comparison/c1_with_prior_K100_2d_re50.yml \
  --output results/experiments/C1_with_prior_K100/evaluation.json

# 批次評估（所有 C1 實驗）
for cfg in configs/experiments/C1_prior_comparison/*.yml; do
  exp_name=$(basename "$cfg" .yml)
  python scripts/evaluate/evaluate_checkpoint.py \
    --checkpoint "checkpoints/experiments/${exp_name}/best_model.pth" \
    --config "$cfg" \
    --output "results/experiments/${exp_name}/evaluation.json"
done
```

---

## 🚨 風險與緩解措施（Risks & Mitigation）

### 風險評估

| 風險 | 可能性 | 影響 | 緩解措施 |
|-----|-------|-----|---------|
| **訓練不收斂**（NaN/發散） | 中 | 高 | (1) 降低學習率；(2) 啟用 gradient clipping；(3) 使用 SOAP 取代 Adam |
| **記憶體不足**（OOM） | 低 | 高 | (1) 減少 batch_size；(2) 使用 gradient checkpointing；(3) 2D slice 而非完整 3D |
| **時間不足** | 高 | 中 | (1) 優先執行 P0/P1 實驗；(2) 降低 epochs；(3) 分批執行 |
| **配置錯誤** | 中 | 中 | (1) 使用 `validate_config_keys.py` 預檢；(2) 啟用 reproducibility.deterministic |
| **結果不可重現** | 低 | 高 | (1) 固定 seed；(2) 使用相同感測器文件；(3) 記錄完整環境（`pip freeze`） |

### 檢查點（Checkpoints）

每個實驗必須保存：
- ✅ **best_model.pth**：驗證集最佳模型
- ✅ **config.yml**：完整配置（包含自動生成的參數）
- ✅ **metrics.json**：所有評估指標
- ✅ **training_log.txt**：訓練日誌
- ✅ **loss_history.csv**：每個 epoch 的 loss

---

## 📝 實驗記錄範本（Experiment Log Template）

每個實驗完成後，記錄至 `context/session_logs/` 或 `results/experiments/<ExpID>/README.md`：

```markdown
# 實驗 <ExpID>: <實驗名稱>

**日期**: YYYY-MM-DD  
**執行者**: <姓名>  
**配置**: `configs/experiments/<ExpID>/<config_file>.yml`

## 目標
- 研究問題：RQ<X>
- 預期結果：<簡述>

## 執行環境
- GPU: 2x NVIDIA P100
- PyTorch: <版本>
- CUDA: <版本>
- 執行時間：<小時>

## 結果
### 主要指標
| 指標 | 值 | 目標 | 達成 |
|-----|---|-----|-----|
| rel $L_2(u)$ | <值>% | <目標>% | ✅/❌ |
| ... | ... | ... | ... |

### 診斷
- 訓練穩定性：✅/❌
- 收斂情況：<描述>
- 異常行為：<描述>

## 結論
- 是否支持假設：✅/❌
- 需要後續實驗：<描述>

## 視覺化
- 圖表位置：`results/experiments/<ExpID>/figures/`
```

---

## 📚 論文寫作對應（Thesis Writing Mapping）

### 實驗結果 → 論文章節對應

| 實驗組 | 論文章節 | 關鍵圖表 | 狀態 |
|-------|---------|---------|-----|
| **S1** | §4.1.1（Sensor Comparison） | Fig. 4.3 | ✅ 配置已準備 |
| **S2** | §4.1.2（K-Scan） | Fig. 4.5 | ✅ 配置已準備 |
| **M1** | §4.1.3（Model Comparison） | Table 4.3 | ✅ 配置已準備 |
| **A1** | §4.1.4（Ablation - Fourier） | Fig. 4.6 | ✅ 配置已準備 |
| **A2** | §4.1.5（Ablation - Weights） | Fig. 4.6 | ✅ 配置已準備 |
| **C1** | §4.1.6（Prior Comparison） | Fig. 4.3, Table 4.1 | ✅ 配置已準備 |
| **C2** | §4.1.7（Prior Sweep） | Fig. 4.7 | ✅ 配置已準備 |
| **J1** | §4.2.1（JHTDB Baseline） | Fig. 4.9, 4.11 | ⚠️ 需新增配置 |
| **J2** | §4.2.2（3D Sensor Strategy） | Fig. 4.8 | ⚠️ 需新增配置 |
| **J3** | §4.2.3（3D K-Scan） | Fig. 4.12 | ⚠️ 需新增配置 |
| **R1-R3** | Appendix E（Robustness） | Table E.1 | ⚠️ 需新增配置 |

### 論文 Claims 驗證（Claims Verification）

根據論文摘要與結論，需實驗支持以下 claims：

1. ✅ **Claim 1**：QR-pivot 優於 Random（§1.3 Gap 1）→ **實驗 S1, J2**
2. ✅ **Claim 2**：$K=100$ 在當前設定下不足（§4.1, §5）→ **實驗 S2, J3**
3. ✅ **Claim 3**：Soft prior 提供一致改善（§4.1.6）→ **實驗 C1, C2**
4. ✅ **Claim 4**：Fourier + VS + RWF 改善穩定性（§3.4）→ **實驗 A1, A2, M1**
5. ⚠️ **Claim 5**：JHTDB 上重建困難（over-smoothing）（§4.2）→ **實驗 J1**

---

## 🎓 總結與下一步（Summary & Next Steps）

### 實驗完整度評估

| Phase | 配置完整度 | 優先級 | 預估完成時間 |
|-------|----------|-------|------------|
| **Phase 0** | ✅ 100% | P0 | 即刻可執行 |
| **Phase 1 (2D)** | ✅ 100% (17 configs) | P0/P1 | Week 1-2 |
| **Phase 2 (3D)** | ⚠️ 0% (需建立) | P0/P1 | Week 3 |
| **Phase 3 (魯棒性)** | ⚠️ 0% (需建立) | P2/P3 | Week 4 |

### 立即行動項（Immediate Actions）

1. **檢視現有配置**：確認 Phase 1 所有配置可執行
   ```bash
   python scripts/validate_config_keys.py configs/experiments/S1_sensor_strategy/*.yml
   ```

2. **生成 Phase 2 配置**：擴充 `generate_experiment_configs.py`
   ```bash
   python configs/experiments/generate_experiment_configs.py --phase 2
   ```

3. **執行 Phase 0 驗證**：環境測試
   ```bash
   bash scripts/validate_imports.py
   python scripts/train/train.py --cfg configs/quick_test.yml
   ```

4. **開始 Phase 1 核心實驗**：
   ```bash
   # Week 1 優先級
   python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml
   python scripts/train/train.py --cfg configs/experiments/S1_sensor_strategy/s1_random_K100_2d_re50.yml
   python scripts/train/train.py --cfg configs/experiments/C1_prior_comparison/c1_no_prior_K100_2d_re50.yml
   python scripts/train/train.py --cfg configs/experiments/C1_prior_comparison/c1_with_prior_K100_2d_re50.yml
   ```

---

## 📎 附錄：配置模板（Appendix: Configuration Templates）

### Template 1: 2D Kolmogorov Baseline

```yaml
# configs/experiments/_template_2d_kolmogorov.yml
experiment:
  name: <exp_id>_<variant>_K<K>_2d_re<Re>
  seed: 42
  device: auto
  precision: float32
  description: "<Experiment description>"

data:
  source: kolmogorov_dns
  kolmogorov_config:
    enabled: true
    data_path: ./data/kolmogorov_dns/dns_re50_t100.h5
    time_range: [15.0, 35.0]
    physics_params:
      Re: 50.0
      nu: 0.039374
      k_f: 4

sensors:
  K: 100
  selection_method: precomputed
  sensor_file: ./data/sensors/kolmogorov/sensors_K100_re50_256x256.json

model:
  type: <standard_mlp | fourier_vs_mlp>
  width: 256
  depth: 6
  activation: <tanh | swish>
  fourier_features:
    type: <disabled | standard>
    fourier_m: 16
    fourier_sigma: 4.0

losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 1.0
  periodicity_weight: 5.0
  prior_weight: <0.0 | 1.0 | 0.1-0.5>

training:
  optimizer:
    type: <adam | soap>
    lr: 0.001
  epochs: 10000
  batch_size: 10000
  sampling:
    N_pde: 5000
```

### Template 2: 3D JHTDB Channel Flow

```yaml
# configs/experiments/_template_3d_jhtdb.yml
experiment:
  name: <exp_id>_<variant>_K<K>_3d
  seed: 42
  device: auto
  precision: float32

data:
  source: jhtdb
  dataset: channel
  jhtdb_config:
    enabled: true
    resolution: {x: 2048, y: 512, z: 1536}
  normalize: true

sensors:
  K: 100
  selection_method: <qr_pivot | random>

model:
  type: fourier_vs_mlp
  width: 256
  depth: 8
  fourier_features:
    type: standard
    fourier_m: 32
    fourier_sigma: 5.0

physics:
  type: vs_pinn_channel_flow
  nu: 5.0e-05
  vs_pinn:
    scaling_factors: {N_x: 2.0, N_y: 12.0, N_z: 2.0}
  channel_flow:
    Re_tau: 1000.0
    pressure_gradient: 0.0025

losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  momentum_z_weight: 1.0
  continuity_weight: 1.0
  wall_constraint_weight: 10.0
  periodicity_weight: 10.0
  prior_weight: <0.0 | 0.1>

training:
  optimizer:
    type: soap
    lr: 0.001
  epochs: 5000
  batch_size: 1024
  sampling:
    N_pde: 10000
    boundary_points: 2000

evaluation:
  metrics:
    - relative_l2
    - mass_conservation
    - wall_shear_stress
    - mean_velocity_profile
    - energy_spectrum
```

---

**文檔結束 | End of Document**

**最後更新**: 2025-01-05  
**維護者**: PINNs-MVP Research Team  
**聯絡**: 請參閱 `AGENTS.md` 與 `README.md`
