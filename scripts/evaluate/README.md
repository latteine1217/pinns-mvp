# 評估工具說明

本目錄包含兩個層級的評估工具，各有不同定位。

## 📊 評估工具對比

| 特性 | `evaluate_unified.py` | `comprehensive_evaluation.py` |
|------|----------------------|-------------------------------|
| **定位** | 🚀 快速評估工具 + Kolmogorov 物理分析 | 🔬 進階科學分析工具 |
| **用途** | 日常訓練驗證 + 物理量驗證 | 論文級深度分析 |
| **物理場景** | ✅ Kolmogorov 2D<br>✅ Channel 3D | ✅ Channel 3D（專精） |
| **基礎指標** | ✅ L2, RMSE, 守恆 | ✅ L2, RMSE, 守恆 |
| **Kolmogorov 專用** | ✅ 動能演化<br>✅ 擾動度<br>✅ 能量譜<br>✅ 4-panel 對比圖 | ❌ 無 |
| **Channel 專用** | ❌ 無 | ✅ 能量譜<br>✅ 壁剪應力<br>✅ 速度剖面<br>✅ 統計分布 |
| **視覺化** | 場對比、誤差分布、物理量 4-panel | 多層級科學圖表 |
| **輸出格式** | JSON + Markdown + PNG | JSON + 高品質科學圖 |
| **執行時間** | 快（1-2 分鐘） | 慢（5-10 分鐘） |
| **適用階段** | 訓練中 checkpoint 快速檢查 | 最終模型完整評估 |

---

## 🚀 快速評估工具: `evaluate_unified.py`

### 使用場景
- ✅ 訓練過程中驗證 checkpoint
- ✅ 快速比較多個模型
- ✅ 自動檢測物理類型
- ✅ Kolmogorov Flow 物理量分析（可選）
- ✅ 入門使用者

### 使用方式
```bash
# 單一模型評估
python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth

# Kolmogorov Flow 物理量分析（添加 --physics-analysis）
python scripts/evaluate_unified.py \
    --checkpoint checkpoints/kolmogorov_model.pth \
    --physics-analysis

# 多模型比較
python scripts/evaluate_unified.py \
    --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
    --labels "RANS Prior" "Vanilla" "Proposed"
```

### 輸出內容
**基礎評估**（所有模型）:
- 基礎誤差指標（L2, RMSE）
- 守恆誤差（散度）
- 場對比視覺化
- 誤差分布直方圖

**物理量分析**（Kolmogorov + `--physics-analysis`）:
- 動能 (Kinetic Energy): KE = 0.5 * ∫∫ (u² + v²) dx dy
- 擾動度 (Enstrophy): Ω = 0.5 * ∫∫ ω² dx dy
- 能量譜 (Energy Spectrum): E(k) with k^(-5/3) and k^(-3) scaling
- 4-panel 對比圖（匹配學術論文風格）

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
使用 `evaluate_unified.py` 快速驗證，添加 `--physics-analysis` 檢查物理量：
```bash
# 快速評估（基礎指標）
python scripts/evaluate_unified.py \
    --checkpoint checkpoints/kolmogorov_step_10k.pth \
    --output results/quick_eval_10k

# 物理量驗證（動能、擾動度、能量譜）
python scripts/evaluate_unified.py \
    --checkpoint checkpoints/kolmogorov_step_10k.pth \
    --output results/physics_check_10k \
    --physics-analysis
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
- [x] **已完成**: Kolmogorov Flow 物理量分析整合到 `evaluate_unified.py`
- [ ] Channel Flow 專用物理量分析（壁剪應力、速度剖面）
- [ ] 時間窗口訓練的時間序列評估
- [ ] 統一兩個工具的視覺化風格

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
- Kolmogorov Flow 物理驗證 → `evaluate_unified.py --physics-analysis`
- Channel Flow 論文投稿前分析 → `comprehensive_evaluation.py`

**Q: Kolmogorov 物理量分析如何使用？**
A: 在 `evaluate_unified.py` 中添加 `--physics-analysis` flag：
```bash
python scripts/evaluate_unified.py \
    --checkpoint checkpoints/kolmogorov_model.pth \
    --physics-analysis
```
這會自動計算動能、擾動度、能量譜，並生成 4-panel 對比圖。

**Q: 物理量分析支持哪些流場？**
A: 目前僅支持 Kolmogorov 2D flow。Channel 3D flow 請使用 `comprehensive_evaluation.py`。

**Q: 能否刪除 comprehensive_evaluation.py？**
A: 不建議。它提供的科學分析功能（能譜、壁剪應力）是論文必需的。

**Q: 兩個工具的輸出一致嗎？**
A: 基礎指標（L2, RMSE）完全一致，comprehensive 額外提供進階分析。

**Q: 執行時間差多少？**
A: 
- `evaluate_unified.py`: ~1-2 分鐘
- `evaluate_unified.py --physics-analysis`: ~2-3 分鐘（Kolmogorov）
- `comprehensive_evaluation.py`: ~5-10 分鐘（取決於網格解析度）

**Q: 能量譜的理論標度線是什麼？**
A: 
- k^(-5/3): Kolmogorov 慣性子範圍標度（各向同性湍流）
- k^(-3): 耗散範圍標度
- 用於驗證預測能量譜是否符合物理理論

---

**最後更新**: 2026-01-05  
**維護者**: PINNs-Sparse-Flow Team
