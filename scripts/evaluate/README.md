# 評估工具說明

本目錄包含兩個層級的評估工具，各有不同定位。

## 📊 評估工具對比

| 特性 | `evaluate_unified.py` | `comprehensive_evaluation.py` |
|------|----------------------|-------------------------------|
| **定位** | 🚀 快速評估工具 | 🔬 進階科學分析工具 |
| **用途** | 日常訓練驗證 | 論文級深度分析 |
| **物理場景** | ✅ Kolmogorov 2D<br>✅ Channel 3D | ✅ Channel 3D（專精） |
| **基礎指標** | ✅ L2, RMSE, 守恆 | ✅ L2, RMSE, 守恆 |
| **進階分析** | ❌ 無 | ✅ 能量譜<br>✅ 壁剪應力<br>✅ 速度剖面<br>✅ 統計分布 |
| **視覺化** | 場對比、誤差分布 | 多層級科學圖表 |
| **輸出格式** | JSON + Markdown + PNG | JSON + 高品質科學圖 |
| **執行時間** | 快（1-2 分鐘） | 慢（5-10 分鐾） |
| **適用階段** | 訓練中 checkpoint 快速檢查 | 最終模型完整評估 |

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
├── README.md                         # 本文檔
├── comprehensive_evaluation.py       # 進階科學評估工具
└── archived/                         # 已歸檔的舊腳本
    ├── evaluate.py
    ├── evaluate_checkpoint.py
    ├── evaluate_curriculum.py
    └── evaluate_kolmogorov_2d.py
```

---

## 🎯 使用建議

### 訓練階段
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
- [ ] 將能譜分析整合到 `evaluate_unified.py`（可選模式）
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
A: 訓練中用 `evaluate_unified.py`，論文前用 `comprehensive_evaluation.py`。

**Q: 能否刪除 comprehensive_evaluation.py？**
A: 不建議。它提供的科學分析功能（能譜、壁剪應力）是論文必需的。

**Q: 兩個工具的輸出一致嗎？**
A: 基礎指標（L2, RMSE）完全一致，comprehensive 額外提供進階分析。

**Q: 執行時間差多少？**
A: unified ~1-2 分鐘，comprehensive ~5-10 分鐘（取決於網格解析度）。

---

**最後更新**: 2026-01-05  
**維護者**: PINNs-Sparse-Flow Team
