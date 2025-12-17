# QR vs Random Sensor Placement 實驗進度追蹤

**目的**：回應審稿人 Major Concern #2  
**開始日期**：2025-12-17  
**預計完成**：2025-12-31  
**狀態**：進行中

---

## 📋 任務檢查清單

### Phase 1: Random Baseline ✅ (完成 2025-12-17)

- [x] **Step 1.1**: 創建隨機感測器生成腳本
  - 文件：`scripts/generate/sensors/generate_channel_random_sensors.py`
  - 功能：支援 2D/3D、Stratified/Uniform、可重現（seed=42）
  - 格式兼容：與 QR sensor 文件格式一致

- [x] **Step 1.2**: 生成隨機感測器文件
  - 輸出：`data/jhtdb/channel_flow_re1000/sensors_K100_random_stratified.npz`
  - 配置：K=100, seed=42, stratified sampling
  - 域範圍：與 QR sensor 匹配（X: [9.32, 15.96], Y: [-0.78, 0.76]）
  
- [x] **Step 1.3**: 視覺化對比
  - 輸出：`results/qr_vs_random_sensor_layouts.png`
  - 確認：兩種布局在相同域內，Random 無極端聚集

### Phase 1.5: Enhanced QR with RANS Features ✅ (完成 2025-12-17)

- [x] **Step 1.5.1**: 創建增強型 QR-Pivot 生成器
  - 文件：`scripts/generate/sensors/generate_channel_rans_qr_enhanced.py`
  - 功能：從 RANS 資料提取湍流特徵（TKE, Reynolds stresses, 壓力梯度等）
  - 特徵模式：`original` (6), `minimal` (10), `physics_guided` (15), `full` (20)

- [x] **Step 1.5.2**: 生成增強感測器集
  - 輸出檔案：
    - `sensors_K100_rans_enhanced_minimal.npz` (10 features) ✅ **Best stability**
    - `sensors_K100_rans_enhanced_physics_guided.npz` (15 features) ⚠️ Rank deficient (13/15)
    - `sensors_K100_rans_enhanced_full.npz` (20 features) ⚠️ Rank deficient (17/20)
  - 指標對比：
    ```
    Minimal:        Cond=7.0e5,   Rank=10/10  ← Best
    Physics-Guided: Cond=4.4e15,  Rank=13/15  ← Redundancy issues
    Full:           Cond=1.0e16,  Rank=17/20  ← High correlation
    ```

- [x] **Step 1.5.3**: 可視化增強版本
  - 輸出：`results/qr_enhanced_sensor_comparison.png`
  - 觀察：Minimal 具備最佳數值穩定性，Full 過度豐富導致 rank 下降

### **Phase A: Advanced Turbulence Features** ✅ (完成 2025-12-17 23:45)

- [x] **Step A.1**: 分析額外特徵需求
  - 文件：`docs/ENHANCED_FEATURES_ANALYSIS.md`
  - 識別：18 種額外特徵類別 (~40+ features)
  - 優先級分級：Phase A (critical, +7), Phase B (beneficial, +5), Phase C (experimental, +6)

- [x] **Step A.2**: 實作 Phase A 特徵計算
  - 新增函數：`compute_phase_a_features()` (in generate_channel_rans_qr_enhanced.py)
  - 新增特徵（8 個）：
    1. **P_k**: TKE 生成率 (`-tau_ij * S_ij`)
    2. **y_plus**: 壁面距離 (`y * u_tau / nu`)
    3. **b_11, b_22, b_12**: 各向異性張量 (`tau_ij/(2k) - δ_ij/3`)
    4. **Re_t**: 湍流雷諾數 (`k^2 / (nu * epsilon)`)
    5. **epsilon**: 耗散率 (從速度梯度估算: `2*nu*<s_ij*s_ij>`)
    6. **enstrophy**: 渦度動能 (`0.5 * omega_z^2`)

- [x] **Step A.3**: 生成 Phase A 感測器
  - 輸出：`data/lowfi/channel_rans/sensors_K100_rans_phase_a.npz`
  - 特徵數：18 (Minimal 10 + Phase A 8)
  - **關鍵成就**：
    ```
    Condition Number: 2.46e6   (3.5× worse than Minimal, but acceptable)
    Matrix Rank:      18 / 18  ✅ FULL RANK (unlike Physics-Guided and Full)
    Energy Ratio:     1.000000 (100% variance captured)
    ```

- [x] **Step A.4**: 驗證數值穩定性
  - 檢查：無 NaN/Inf 值 ✅
  - 特徵範圍：所有 18 特徵在有限界內
  - Y 分佈：保持近壁聚集（7% at y < 0.1）

- [x] **Step A.5**: 更新可視化比較
  - 輸出：`results/qr_enhanced_sensor_comparison.png` (更新為 6 策略)
  - 包含：Random, Original, Minimal, Physics-Guided, **Phase A** (新增), Full
  - 觀察：Phase A 在豐富性與穩定性間達成最佳平衡

- [x] **Step A.6**: 創建完成報告
  - 文件：`docs/PHASE_A_COMPLETION_REPORT.md`
  - 內容：實作細節、特徵物理意義、數值指標、下一步建議

### Phase 2: 訓練配置 (待執行)

- [ ] **Step 2.1**: 創建 Random baseline 訓練配置
  - 文件：`configs/channel_flow_random_K100.yml`
  - 修改：僅 `sensors.file` 路徑，其他與 QR 完全相同
  
- [ ] **Step 2.2**: 驗證配置文件
  - 檢查：model, optimizer, loss weights 與 QR 一致
  - 確認：只有 sensor file 不同

### Phase 3: 訓練執行 (待執行)

- [ ] **Step 3.1**: 訓練 Random baseline
  - 命令：`python scripts/train/train.py --config configs/channel_flow_random_K100.yml --device mps`
  - 預計時長：8-12 小時 (M1/M2)
  - 輸出：`checkpoints/channel_random_K100/`
  
- [ ] **Step 3.2**: 監控訓練
  - 檢查：loss curves, divergence, NaN
  - 比較：與 QR baseline 的收斂行為

### Phase 4: 評估對比 (待執行)

- [ ] **Step 4.1**: 實現對比腳本
  - 文件：`scripts/evaluate/compare_qr_vs_random.py`
  - 功能：L2 error, divergence, energy spectrum, sensor info metrics
  
- [ ] **Step 4.2**: 運行評估
  - 生成：4-panel 對比圖、metrics table
  - 輸出：`results/qr_vs_random_2d/`

### Phase 5: 論文更新 (待執行)

- [ ] **Step 5.1**: 撰寫審稿回應
  - 文件：`docs/REVIEWER_RESPONSE_MAJOR_CONCERN_2.md`
  - 內容：實驗結果、定量對比、理論justification
  
- [ ] **Step 5.2**: 更新 thesis
  - Abstract：添加 QR vs Random 驗證
  - Results：新增 subsection
  - Conclusion：移除 "future work" 語氣

---

## 📊 已生成文件

### 代碼
```
scripts/generate/sensors/generate_channel_random_sensors.py  [✅ 完成]
configs/channel_flow_random_K100.yml                          [⏸️  待創建]
scripts/evaluate/compare_qr_vs_random.py                      [⏸️  待創建]
```

### 數據
```
data/jhtdb/channel_flow_re1000/sensors_K100_random_stratified.npz  [✅ 完成]
checkpoints/channel_random_K100/                                    [⏸️  待訓練]
```

### 結果
```
results/qr_vs_random_sensor_layouts.png    [✅ 完成]
results/qr_vs_random_2d/comparison.png     [⏸️  待生成]
results/qr_vs_random_2d/metrics_table.csv  [⏸️  待生成]
```

---

## 🔍 技術細節

### Sensor 域範圍匹配
```
QR Sensor:
  X: [9.6211, 15.6589] (range: 6.04)
  Y: [-0.7070, 0.6938]  (range: 1.40)

Random Sensor (with 5% margin):
  X: [9.3401, 15.9188] (range: 6.58)
  Y: [-0.7531, 0.7481]  (range: 1.50)

✅ Random 覆蓋 QR 範圍，確保公平對比
```

### 感測器統計
```
QR-Pivot:
  Strategy: qr_pivot_periodic
  Periodic axes: [0] (x 方向)
  Condition number: 323.38
  
Random Stratified:
  Strategy: random_stratified
  Grid: 10×10 = 100 cells
  Min nearest-neighbor distance: 0.0525
  Mean nearest-neighbor distance: 0.1921
```

---

## ⏭️ 下一步

**立即執行**：創建 Random baseline 訓練配置文件

**待命令**：
```bash
# 創建配置
cd /Users/latteine/Documents/coding/pinns-mvp
# (待創建 configs/channel_flow_random_K100.yml)

# 開始訓練
python scripts/train/train.py \
    --config configs/channel_flow_random_K100.yml \
    --device mps \
    --output checkpoints/channel_random_K100
```

---

## 📝 註記

- **重要**：Random sensor 使用 stratified sampling（10×10 grid）避免極端聚集，這比 pure uniform 更公平
- **種子固定**：seed=42 確保可重現
- **域範圍**：從 QR sensor 的實際分佈提取（+5% margin），而非使用理論全域
- **格式兼容**：Random sensor file 包含所有必要欄位，可直接用於訓練

---

## ✨ Phase 1.5: Enhanced QR-Pivot Sensors (RANS Features) ✅ (完成 2025-12-17)

### 背景
用戶要求重新生成 QR-Pivot sensors，使用更豐富的湍流特徵來改善矩陣 rank 和可識別性。

### 完成項目

- [x] **Step 1.5.1**: 實作增強版 QR-Pivot 生成器
  - 文件：`scripts/generate/sensors/generate_channel_rans_qr_enhanced.py`
  - 數據源：使用 RANS (k-omega SST) 而非 DNS
  - 特徵增強：從 6 → 10/15/20 features
  
- [x] **Step 1.5.2**: 特徵集設計
  ```
  Original (6):
    - u, v, w, omega_z, grad_u_eig1, grad_u_eig2
  
  Minimal (10): ✅ 最佳數值穩定性
    - u, v, w, p
    - dudy (壁面剪切)
    - omega_z
    - k (TKE)
    - tau_uv (Reynolds stress)
    - grad_u_eig1, grad_u_eig2
  
  Physics-Guided (15):
    - u, v, w, p
    - dudx, dudy, dvdx, dvdy (速度梯度)
    - dpdx, dpdy (壓力梯度)
    - omega_z, k, tau_uv
    - grad_u_eig1, grad_u_eig2
  
  Full (20):
    - Physics-Guided (15)
    - dwdx, dwdy (W 梯度)
    - tau_uu, tau_vv, tau_ww (所有 Reynolds stresses)
  ```

- [x] **Step 1.5.3**: Reynolds Stress 計算
  - 方法：Boussinesq 假設
  - 公式：`τ_ij = μ_t (∂U_i/∂x_j + ∂U_j/∂x_i) - (2/3)ρk δ_ij`
  - 輸入：RANS k 和 μ_t (eddy viscosity)

- [x] **Step 1.5.4**: 生成增強版 sensors
  ```
  data/lowfi/channel_rans/sensors_K100_rans_enhanced_minimal.npz
    K=100, features=10, rank=10/10, cond=7.0e5 ✅
  
  data/lowfi/channel_rans/sensors_K100_rans_enhanced_physics_guided.npz
    K=100, features=15, rank=13/15, cond=4.4e15
  
  data/lowfi/channel_rans/sensors_K100_rans_enhanced_full.npz
    K=100, features=20, rank=17/20, cond=1.0e16
  ```

- [x] **Step 1.5.5**: 視覺化對比
  - 輸出：`results/qr_enhanced_sensor_comparison.png`
  - 對比 5 種策略：Random, QR-Original, QR-Minimal, QR-Physics, QR-Full

### 關鍵發現

| 策略 | Features | Rank | Cond Number | Near-Wall % |
|------|----------|------|-------------|-------------|
| Random | 0 | - | N/A | 27% |
| QR-Original (DNS) | 6 | - | ~3e2 (舊) | 25% |
| **QR-Minimal (RANS)** | **10** | **10/10** | **7e5** ✅ | 10% |
| QR-Physics (RANS) | 15 | 13/15 | 4e15 | 12% |
| QR-Full (RANS) | 20 | 17/20 | 1e16 | 14% |

**推薦**：使用 **QR-Minimal (10 features)** 作為最佳平衡
- ✅ 滿秩（10/10）
- ✅ 最佳 conditioning (7e5 vs. 1e16)
- ✅ 包含核心物理特徵（壓力、TKE、Reynolds stress）
- ✅ 近壁覆蓋率適中（10%）

### 技術細節

**RANS 資料**：
- 檔案：`data/lowfi/channel_rans/rans_k_omega_sst.npz`
- 模型：k-omega SST (Fluent)
- 網格：251 × 20 × 94 (3D) → 251 × 20 (2D slice at z=center)
- Re_τ estimate: 1343 (vs. DNS Re_τ=1000)

**特徵標準化**：
- 所有特徵經過 z-score 標準化（critical for QR rank）
- `data_matrix = (data - mean) / std`

**無量綱化**：
- 速度梯度：乘以特徵長度 Lx/Ly
- 確保不同方向梯度可比較

---

**最後更新**：2025-12-17 23:25  
**當前狀態**：Phase 1 + 1.5 完成，準備進入 Phase 2
