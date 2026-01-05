# 評估工具說明

本目錄包含兩個層級的評估工具，各有不同定位。

## 📊 評估工具對比

| 特性 | `evaluate_unified.py` | `evaluate_kolmogorov_physics.py` | `comprehensive_evaluation.py` |
|------|----------------------|----------------------------------|-------------------------------|
| **定位** | 🚀 快速評估工具 | 🌀 Kolmogorov 物理量分析 | 🔬 進階科學分析工具 |
| **用途** | 日常訓練驗證 | Kolmogorov Flow 專用評估 | 論文級深度分析 |
| **物理場景** | ✅ Kolmogorov 2D<br>✅ Channel 3D | ✅ Kolmogorov 2D（專精） | ✅ Channel 3D（專精） |
| **基礎指標** | ✅ L2, RMSE, 守恆 | ✅ L2, RMSE | ✅ L2, RMSE, 守恆 |
| **進階分析** | ❌ 無 | ✅ 動能演化<br>✅ 擾動度<br>✅ 能量譜 | ✅ 能量譜<br>✅ 壁剪應力<br>✅ 速度剖面<br>✅ 統計分布 |
| **視覺化** | 場對比、誤差分布 | 4-panel 物理量對比 | 多層級科學圖表 |
| **輸出格式** | JSON + Markdown + PNG | JSON + PNG | JSON + 高品質科學圖 |
| **執行時間** | 快（1-2 分鐘） | 中（2-3 分鐘） | 慢（5-10 分鐘） |
| **適用階段** | 訓練中 checkpoint 快速檢查 | Kolmogorov 專案物理驗證 | 最終模型完整評估 |

---

## 🚀 快速評估工具: `evaluate_unified.py`

### 使用場景
- ✅ 訓練過程中驗證 checkpoint
- ✅ 快速比較多個模型
- ✅ 自動檢測物理類型
- ✅ 入門使用者

### 使用方式
```bash
# 單一模型評估
python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth

# 多模型比較
python scripts/evaluate_unified.py \
    --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
    --labels "RANS Prior" "Vanilla" "Proposed"
```

### 輸出內容
- 基礎誤差指標（L2, RMSE）
- 守恆誤差（散度）
- 場對比視覺化
- 誤差分布直方圖

---

## 🌀 Kolmogorov Flow 物理量分析: `evaluate_kolmogorov_physics.py`

### 使用場景
- ✅ Kolmogorov Flow 專案的物理一致性驗證
- ✅ 動能、擾動度時間演化分析
- ✅ 能量譜 k-space 分析
- ✅ 與參考論文結果對比（4-panel 圖）

### 使用方式
```bash
# 單一 snapshot 評估
python scripts/evaluate/evaluate_kolmogorov_physics.py \
    --checkpoint checkpoints/kolmogorov_model.pth \
    --reference data/kolmogorov_dns/snapshot_re50_for_eval.npz \
    --output results/physics_eval

# 時間序列評估（多個 checkpoint）
python scripts/evaluate/evaluate_kolmogorov_physics.py \
    --checkpoint checkpoints/time_windows/ \
    --reference data/kolmogorov_dns/snapshot_re50_for_eval.npz \
    --output results/time_series_physics \
    --time-series
```

### 獨特功能

#### 1. **動能演化** (`compute_kinetic_energy`)
- 計算公式: KE = 0.5 * ∫∫ (u² + v²) dx dy
- 時間演化對比（預測 vs 參考）
- 誤差百分比分析

#### 2. **擾動度分析** (`compute_enstrophy`)
- 計算公式: Ω = 0.5 * ∫∫ ω² dx dy
- 渦度場計算: ω = ∂v/∂x - ∂u/∂y
- 湍流強度指標

#### 3. **能量譜分析** (`compute_energy_spectrum`)
- 2D FFT 徑向平均
- 理論標度線對比:
  - k^(-5/3): 慣性子範圍（Kolmogorov scaling）
  - k^(-3): 耗散範圍
- 預測 vs 參考能譜對比

#### 4. **4-panel 物理量對比圖**
- (a) 相對 L2 誤差（u, v 分量）
- (b) 動能對比（Reference vs Prediction）
- (c) 擾動度對比
- (d) 能量譜（log-log plot with scaling lines）

### 輸出內容
- 基礎誤差指標（L2, RMSE）
- 物理量時間序列 JSON
- 4-panel 對比圖（PNG, 高解析度）
- 守恆檢查報告

### 物理量計算驗證
- ✅ 渦度計算與 DNS 參考一致（0% 誤差）
- ✅ 動能典型值: KE ≈ 0.65 (Re=50, kf=4)
- ✅ 能量譜波數範圍: k ∈ [1, 181] for 256×256 網格

---

## 🔬 進階科學工具: `comprehensive_evaluation.py`

### 使用場景
- ✅ 論文投稿前的完整評估
- ✅ 深入物理一致性驗證
- ✅ 湍流特性分析（能譜、壁剪應力）
- ✅ 高品質科學繪圖

### 使用方式
```bash
python scripts/evaluate/comprehensive_evaluation.py \
    --checkpoint checkpoints/final_model.pth \
    --reference_dir data/jhtdb \
    --output results/comprehensive_eval
```

### 獨特功能

#### 1. **能量譜分析** (`compute_energy_spectrum_comparison`)
- 流向 1D 能譜（適用於通道流剪切湍流）
- 徑向 2D 能譜（適用於各向同性湍流）
- 預測 vs 參考能譜對比
- 慣性子範圍斜率驗證（-5/3 定律）

#### 2. **壁剪應力分析** (`compute_wall_shear_stress_comparison`)
- 上下壁面剪應力計算
- 空間分布比較
- 統計量誤差（mean, std, max）

#### 3. **速度剖面分析** (`plot_velocity_profiles`)
- 壁面法向速度分布
- 對數律驗證（u+ vs y+）
- 不同流向位置的剖面比較

#### 4. **場統計分析** (`compute_field_statistics`)
- 多階統計矩（mean, std, skewness, kurtosis）
- 預測 vs 參考統計量對比
- 高階矩物理合理性檢查

### 輸出內容
- 所有基礎指標（同 unified）
- 能量譜圖（PNG, 高解析度）
- 壁剪應力分布圖
- 速度剖面圖（含對數律擬合）
- 統計量比較表
- 完整 JSON 報告

---

## 📁 目錄結構

```
scripts/evaluate/
├── README.md                              # 本文檔
├── evaluate_kolmogorov_physics.py         # Kolmogorov Flow 物理量分析（NEW）
├── comprehensive_evaluation.py            # 進階科學評估工具
└── archived/                              # 已歸檔的舊腳本
    ├── evaluate.py
    ├── evaluate_checkpoint.py
    ├── evaluate_curriculum.py
    └── evaluate_kolmogorov_2d.py
```

---

## 🎯 使用建議

### 訓練階段（Kolmogorov Flow）
使用 `evaluate_kolmogorov_physics.py` 驗證物理一致性：
```bash
# 檢查物理量演化
python scripts/evaluate/evaluate_kolmogorov_physics.py \
    --checkpoint checkpoints/kolmogorov_step_10k.pth \
    --reference data/kolmogorov_dns/snapshot_re50_for_eval.npz \
    --output results/physics_check_10k
```

### 訓練階段（通用）
使用 `evaluate_unified.py` 快速驗證每個 checkpoint：
```bash
# 每 5000 steps 快速評估
python scripts/evaluate_unified.py \
    --checkpoint checkpoints/step_5000.pth \
    --output results/quick_eval_5k
```

### 最終評估
使用 `comprehensive_evaluation.py` 完整分析最佳模型：
```bash
# 論文投稿前完整評估
python scripts/evaluate/comprehensive_evaluation.py \
    --checkpoint checkpoints/best_model.pth \
    --reference_dir data/jhtdb \
    --output results/final_evaluation \
    --all_metrics
```

---

## 🔄 未來規劃

### 短期（1-2 週）
- [x] **已完成**: Kolmogorov Flow 物理量分析工具
- [ ] 時間窗口訓練的時間序列評估
- [ ] 統一兩個工具的視覺化風格
- [ ] 添加 `--quick` 和 `--comprehensive` 模式切換

### 中期（1 個月）
- [ ] 將 comprehensive 的獨特功能提取到 `pinnx.evals.advanced_metrics`
- [ ] 創建統一的配置系統（選擇評估指標）
- [ ] 添加自動化測試

### 長期（3 個月）
- [ ] 考慮完全整合兩個工具
- [ ] 添加互動式評估儀表板（streamlit/dash）

---

## 📞 常見問題

**Q: 應該使用哪個工具？**
A: 
- 訓練中快速檢查 → `evaluate_unified.py`
- Kolmogorov Flow 物理驗證 → `evaluate_kolmogorov_physics.py`
- 論文投稿前完整分析 → `comprehensive_evaluation.py`

**Q: Kolmogorov 物理量分析與 unified 有何不同？**
A: `evaluate_kolmogorov_physics.py` 專注於 Kolmogorov Flow 特有的物理量（動能、擾動度、能量譜），而 `evaluate_unified.py` 提供通用的場誤差指標。

**Q: 能否刪除 comprehensive_evaluation.py？**
A: 不建議。它提供的科學分析功能（能譜、壁剪應力）是論文必需的。

**Q: 兩個工具的輸出一致嗎？**
A: 基礎指標（L2, RMSE）完全一致，comprehensive 額外提供進階分析。

**Q: 執行時間差多少？**
A: 
- `evaluate_unified.py`: ~1-2 分鐘
- `evaluate_kolmogorov_physics.py`: ~2-3 分鐘
- `comprehensive_evaluation.py`: ~5-10 分鐘（取決於網格解析度）

**Q: 能量譜的理論標度線是什麼？**
A: 
- k^(-5/3): Kolmogorov 慣性子範圍標度（各向同性湍流）
- k^(-3): 耗散範圍標度
- 用於驗證預測能量譜是否符合物理理論

---

**最後更新**: 2026-01-05  
**維護者**: PINNs-Sparse-Flow Team
