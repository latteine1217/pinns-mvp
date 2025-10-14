# PINNs 訓練配置管理指南

## 🚀 快速開始

**新手用戶**：直接使用標準化模板開始訓練 👉 **[`templates/` 目錄](./templates/README.md)**

**進階用戶**：參考下方完整指南進行自定義配置

---

## 📋 目錄結構

```
configs/
├── README.md                    # 本文檔（配置管理總指南）
├── templates/                   # ⭐ 標準化模板目錄
│   ├── README.md                #    模板使用指引（詳細）
│   ├── 2d_quick_baseline.yml    #    2D 快速基線（5-10 min）
│   ├── 2d_medium_ablation.yml   #    2D 消融實驗（15-30 min）
│   ├── 3d_slab_curriculum.yml   #    3D 課程學習（30-60 min）
│   └── 3d_full_production.yml   #    3D 生產訓練（2-8 hrs）
│
├── experiment_tracking.md       # 實驗追蹤記錄
├── main.yml                     # 生產級主配置
├── monitoring.yml              # 監控配置
├── config_template_example.yml # 詳細範例模板
└── test_physics_1k.yml         # v3 修正版配置（1000 epochs）
```

---

## 📦 模板快速選擇

| 場景 | 模板 | 時間 | 詳細說明 |
|------|------|------|---------|
| **快速驗證想法** | [`2d_quick_baseline.yml`](./templates/2d_quick_baseline.yml) | 5-10 分鐘 | [使用指引](./templates/README.md#1️⃣-2d_quick_baselineyml---快速基準測試) |
| **特徵消融研究** | [`2d_medium_ablation.yml`](./templates/2d_medium_ablation.yml) | 15-30 分鐘 | [使用指引](./templates/README.md#2️⃣-2d_medium_ablationyml---中期消融實驗) |
| **課程式訓練** | [`3d_slab_curriculum.yml`](./templates/3d_slab_curriculum.yml) | 30-60 分鐘 | [使用指引](./templates/README.md#3️⃣-3d_slab_curriculumyml---3d-課程學習) |
| **論文級結果** | [`3d_full_production.yml`](./templates/3d_full_production.yml) | 2-8 小時 | [使用指引](./templates/README.md#4️⃣-3d_full_productionyml---生產級訓練) |

👉 **完整模板文檔**：[`templates/README.md`](./templates/README.md)

---

---

## 🎯 配置命名規範

### 標準格式
```
{spatial}_{duration}_{type}_{variant}_{version}.yml
```

### 欄位說明

| 欄位 | 取值範圍 | 說明 | 範例 |
|------|---------|------|------|
| **spatial** | `2d` / `3d_slab` / `3d_full` | 空間維度與解析度 | `2d`: XY 切片<br>`3d_slab`: 128×64×16<br>`3d_full`: 128×64×64 |
| **duration** | `quick` / `medium` / `full` / `1k` / `5k` | 訓練時長等級 | `quick`: 50-100 epochs (5-10 min)<br>`medium`: 500-1k epochs (15-30 min)<br>`full`: 2k-5k epochs (1-4 hrs) |
| **type** | `baseline` / `ablation` / `curriculum` / `sensitivity` / `comparison` | 實驗類型 | `baseline`: 基線對照<br>`ablation`: 消融研究<br>`curriculum`: 課程學習<br>`sensitivity`: 參數敏感度<br>`comparison`: 方法對比 |
| **variant** | `fourier` / `rans` / `adaptive` / `vs_pinn` / 自定義 | 特徵變體 | `fourier`: 傅立葉特徵<br>`rans`: RANS 低保真<br>`adaptive`: 自適應權重<br>`vs_pinn`: 變數尺度化 |
| **version** | `v1` / `v2` / `v3` | 版本號 | `v1`: 初始版本<br>`v2`: 修正版本 |

### 命名範例

| 配置檔名 | 解讀 |
|---------|------|
| `2d_quick_baseline_adam_v1.yml` | 2D 切片 + 快速測試（50-100 epochs）+ Adam 優化器基線 + 版本 1 |
| `3d_slab_medium_ablation_fourier_v2.yml` | 3D 薄片 + 中等時長（500-1k）+ 傅立葉消融實驗 + 版本 2 |
| `3d_full_curriculum_vs_pinn_v1.yml` | 3D 完整 + 課程學習 + VS-PINN + 版本 1 |
| `2d_1k_sensitivity_alpha_v1.yml` | 2D + 1000 epochs + GradNorm alpha 敏感度測試 + 版本 1 |

---

## 📐 實驗設計框架

### 1. **空間維度選擇**

| 類型 | 解析度 | K 點數範圍 | 訓練時長 | 適用場景 |
|------|--------|-----------|---------|---------|
| **2D 切片** | 128×64 (XY) | 20-100 | 5-30 min | 快速驗證、參數調優 |
| **3D 薄片** | 128×64×16 | 50-200 | 15-60 min | 中等規模驗證 |
| **3D 完整** | 128×64×64 | 80-500 | 1-8 hrs | 生產級訓練、最終評估 |

### 2. **訓練時長分級**

| 等級 | Epochs | 時長 | Checkpoint 頻率 | 適用場景 |
|------|--------|------|----------------|---------|
| **Quick** | 50-100 | 5-10 min | 每 50 | 功能測試、快速驗證 |
| **Medium** | 500-1k | 15-30 min | 每 100 | 消融研究、參數掃描 |
| **Full** | 2k-5k | 1-4 hrs | 每 500 | 生產訓練、最終模型 |
| **Extended** | 5k-10k | 4-12 hrs | 每 1000 | 長期收斂、極限性能 |

### 3. **實驗類型定義**

#### A. **Baseline（基線）**
- **目的**：建立性能參考點
- **特徵狀態**：
  - Fourier: ✅ 啟用（標準配置）
  - Adaptive Weighting: ✅ 啟用（alpha=1.5, freq=50）
  - Curriculum: ❌ 關閉（單階段訓練）
  - RANS: ❌ 關閉
  - VS-PINN: ✅ 啟用
- **驗收標準**：
  - 無 NaN/Inf（必須）
  - Loss 收斂（必須）
  - L2 < 30%（目標）

#### B. **Ablation（消融）**
- **目的**：評估單一特徵貢獻
- **實驗組**：
  - `ablation_fourier`: 開啟/關閉 Fourier 特徵
  - `ablation_adaptive`: 固定權重 vs 自適應權重
  - `ablation_curriculum`: 單階段 vs 課程學習
  - `ablation_rans`: 無先驗 vs RANS 引導
- **驗收標準**：
  - 相對基線 L2 變化 ≥ 10%（有效性證據）
  - 訓練時長變化記錄

#### C. **Curriculum（課程學習）**
- **目的**：逐步提升訓練難度
- **階段設計**：
  - Stage1: 強 PDE 約束（epochs 0-30%）
  - Stage2: PDE 主導（epochs 30-100%）
- **驗收標準**：
  - 階段切換損失變化 < 20%
  - 最終 PDE ratio ≥ 30%

#### D. **Sensitivity（敏感度）**
- **目的**：評估參數穩健性
- **掃描參數**：
  - `alpha`: [0.5, 1.0, 1.5, 2.0, 3.0]（GradNorm）
  - `K`: [20, 30, 50, 80, 100, 200]（感測點數）
  - `freq`: [25, 50, 100, 200]（權重更新頻率）
  - `lr`: [1e-4, 5e-4, 1e-3, 5e-3]（學習率）
- **驗收標準**：
  - K-誤差曲線斜率（可識別性）
  - 參數-性能熱圖

#### E. **Comparison（方法對比）**
- **目的**：比較不同方法優劣
- **對比組**：
  - 優化器：Adam vs SOAP vs L-BFGS
  - 調度器：Cosine vs Warm Restarts vs 固定
  - 網路：Fourier MLP vs SIREN
- **驗收標準**：
  - 收斂速度對比（達到相同 L2 所需 epochs）
  - 最終性能對比

---

## 🔧 快速選擇指南

### 使用場景 → 推薦模板

| 場景 | 推薦模板 | 時長 | 關鍵配置 |
|------|---------|------|---------|
| **快速功能測試** | `2d_quick_baseline_v1.yml` | 5-10 min | K=20-50, epochs=50-100 |
| **參數調優** | `2d_medium_ablation_v1.yml` | 15-30 min | K=50-100, epochs=500-1k |
| **課程訓練驗證** | `3d_slab_curriculum_v1.yml` | 30-60 min | K=50-100, 2 stages |
| **生產級訓練** | `3d_full_production_v1.yml` | 2-8 hrs | K=200-500, epochs=2k-5k |
| **敏感度實驗** | `2d_1k_sensitivity_*.yml` | 20-40 min | 掃描單一參數 |

### 特徵組合建議

| 組合 | Fourier | Adaptive | Curriculum | RANS | VS-PINN | 適用場景 |
|------|---------|----------|-----------|------|---------|---------|
| **最小配置** | ✅ | ❌ | ❌ | ❌ | ✅ | 基線對照 |
| **標準配置** | ✅ | ✅ | ❌ | ❌ | ✅ | 快速訓練 |
| **完整配置** | ✅ | ✅ | ✅ | ❌ | ✅ | 生產訓練 |
| **低保真引導** | ✅ | ✅ | ✅ | ✅ | ✅ | 少量資料場景 |

---

## 📊 參數調優建議

### 1. **自適應權重（GradNorm）**

| 參數 | 推薦範圍 | 說明 |
|------|---------|------|
| `grad_norm_alpha` | **1.5**（標準）<br>0.5-3.0（掃描） | 控制權重調整幅度<br>α↑ → 更激進的權重調整 |
| `weight_update_freq` | **50**（標準）<br>25-200（掃描） | 權重更新頻率（epochs）<br>freq↓ → 更頻繁調整 |

**⚠️ 警告**：
- `alpha < 1.0`：可能導致權重更新過慢（如 v2 的 0.12 導致失效）
- `freq > 100`：權重調整滯後，可能錯過最佳時機

### 2. **初始權重配置**

| 損失項 | 2D 快速 | 3D 中等 | 3D 完整 | 說明 |
|--------|---------|---------|---------|------|
| `data_weight` | 5.0 | 5.0-10.0 | 10.0 | 資料監督 |
| `momentum_*_weight` | 2.0-5.0 | 5.0-8.0 | 8.0-10.0 | PDE 殘差 |
| `wall_constraint_weight` | 8.0 | 10.0 | 10.0-15.0 | 壁面約束 |

**原則**：
- 保持 `data ≈ PDE`（避免 v2 的 data=93% 問題）
- 壁面約束應強於 PDE（wall > PDE × 1.5-2.0）
- 課程 Stage2 提升 PDE 權重 60-100%

### 3. **採樣策略**

| 參數 | 2D | 3D Slab | 3D Full | 說明 |
|------|----|---------|---------|----- |
| `pde_points` | 1024-2048 | 2048-4096 | 4096-8192 | PDE 殘差採樣點 |
| `boundary_points` | 500-1000 | 1000-2000 | 2000-5000 | 邊界條件點 |
| `wall_clustering` | 0.3 | 0.3-0.5 | 0.5 | 壁面聚類比例 |

---

## ✅ 驗收標準（基於 AGENTS.md）

### Gate 1: 穩定性（必須通過）
- [ ] 無 NaN/Inf
- [ ] Loss 有限且收斂
- [ ] 梯度穩定（無爆炸）
- [ ] 無模態崩塌（動態範圍 > 80%）

### Gate 2: 物理一致性（核心目標）
- [ ] **PDE Loss Ratio ≥ 30%**（PDE/(PDE+Data) ≥ 0.30）
- [ ] **壁面剪應力非零**（τ_w > 0.0005，理論 0.0025）
- [ ] 質量守恆誤差 < 1%（∇·u < 0.01）
- [ ] 壁面違反 < 1.0（|u_wall| < 1.0）

### Gate 3: 預測精度（成功門檻）
- [ ] **相對 L2 ≤ 15%**（u,v,p）
- [ ] 統計 RMSE 改善 ≥ 30%（vs 低保真）
- [ ] 能譜 RMSE ≤ 25%

### Gate 4: 效率（優化目標）
- [ ] 收斂加速 ≥ 30%（vs 固定權重）
- [ ] 訓練時長 < 8 小時（3D 完整）

---

## 📝 使用流程

### 1. 選擇模板
```bash
# 快速測試（推薦初次使用）
cp configs/templates/2d_quick_baseline.yml configs/my_experiment_v1.yml

# 中等規模消融實驗
cp configs/templates/2d_medium_ablation.yml configs/my_ablation_v1.yml

# 生產級訓練
cp configs/templates/3d_full_production.yml configs/my_production_v1.yml
```

### 2. 修改配置
```yaml
# 必改欄位
experiment:
  name: "my_experiment_v1"  # 實驗名稱（影響輸出路徑）
  description: "簡短描述實驗目的"

# 建議改欄位
sensors:
  K: 50  # 感測點數（根據場景調整）

training:
  epochs: 100  # 訓練輪數
```

### 3. 執行訓練
```bash
# 基本訓練
python scripts/train.py --cfg configs/my_experiment_v1.yml

# 背景執行（推薦）
nohup python scripts/train.py --cfg configs/my_experiment_v1.yml \
    > log/my_experiment_v1/stdout.log 2>&1 &
```

### 4. 監控進度
```bash
# 即時監控
tail -f log/my_experiment_v1/training.log

# 檢查 PDE ratio（每 20 epochs）
grep "PDE ratio" log/my_experiment_v1/training.log
```

### 5. 評估結果
```bash
# 檢查點評估
python scripts/evaluate_checkpoint.py \
    --checkpoint checkpoints/my_experiment_v1/best_model.pth \
    --cfg configs/my_experiment_v1.yml
```

### 6. 記錄實驗
在 `experiment_tracking.md` 中記錄：
- 實驗 ID、配置檔名、日期
- 關鍵結果（L2, PDE ratio, τ_w）
- 決策（通過/失敗 → 下一步）

---

## 🚨 常見問題

### Q1: PDE Loss Ratio 過低（< 10%）
**症狀**：`weighted_pde_loss / total_loss < 0.10`

**診斷**：
1. 檢查 `grad_norm_alpha`：若 < 1.0，提升至 1.5-2.0
2. 檢查 `weight_update_freq`：若 > 100，降至 50
3. 檢查初始權重：確保 `data_weight ≈ momentum_*_weight`

**修正範例**：
```yaml
losses:
  data_weight: 5.0          # 降低（原 10.0）
  momentum_x_weight: 5.0    # 提升（原 1.0）
  grad_norm_alpha: 1.5      # 標準化（原 0.12）
  weight_update_freq: 50    # 加速（原 100）
```

### Q2: 壁面剪應力為零（τ_w ≈ 0）
**症狀**：`tau_w_ratio < 0.1`（理論應 > 1.0）

**診斷**：
1. 檢查 `wall_constraint_weight`：應 > PDE 權重 1.5-2.0 倍
2. 檢查 `boundary_points`：增加至 1000-2000
3. 檢查 `wall_clustering`：提升至 0.3-0.5

**修正範例**：
```yaml
losses:
  wall_constraint_weight: 10.0  # 提升（原 5.0）

training:
  sampling:
    boundary_points: 1000  # 加倍（原 500）
    wall_clustering: 0.3   # 聚焦壁面
```

### Q3: 階段切換損失跳變
**症狀**：Stage1→Stage2 時 loss 增加 > 20%

**診斷**：
1. 檢查權重變化幅度：Stage2 PDE 權重不應超過 Stage1 的 2 倍
2. 考慮學習率同步降低：Stage2 lr = Stage1 lr × 0.3-0.5

**修正範例**：
```yaml
curriculum:
  stages:
    - name: "Stage1"
      weights:
        momentum_x: 5.0
    - name: "Stage2"
      weights:
        momentum_x: 8.0   # 僅提升 1.6×（非 3×）
      lr: 3.0e-4          # 降低學習率（原 1e-3）
```

---

## 📚 參考文獻

1. **GradNorm 調參**：Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss Balancing" (2018)
   - 推薦 alpha=1.0-1.5（中等任務）
   - 更新頻率 50-100 steps

2. **VS-PINN 縮放**：Schiassi et al., "Extreme Theory of Functional Connections" (2021)
   - 通道流 N_y=10-20（壁面法向）
   - 流向/展向 N_x,z=1-5

3. **QR-Pivot 感測**：Manohar et al., "Data-Driven Sparse Sensor Placement" (2018)
   - K ≥ 2×rank(POD) 保證重建
   - 通道流建議 K=50-100（2D），K=200-500（3D）

4. **Fourier Features**：Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions" (2020)
   - 週期邊界：使用域長頻率 [1,2,4,8]×(2π/L)
   - 壁面法向：不使用 Fourier（保持梯度精度）

---

## 🔄 版本歷史

| 版本 | 日期 | 變更內容 |
|------|------|---------|
| v1.0 | 2025-10-15 | 初始版本：建立命名規範、模板框架、驗收標準 |

---

**維護者**：PINNs Research Team  
**最後更新**：2025-10-15
