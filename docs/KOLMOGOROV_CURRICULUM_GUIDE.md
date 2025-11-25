# Kolmogorov Flow 2D 課程學習指南

## 📚 概述

本指南說明如何使用**課程學習（Curriculum Learning）**策略訓練 Kolmogorov Flow 2D 模型，從簡單的轉捩流動（Re≈50）逐步增加到完全發展的湍流（Re≈240）。

## 🎯 為什麼需要課程學習？

### 問題：直接訓練 Re=240 的挑戰

**Re=240 是完全發展湍流**：
- Re/Re_c1 = 8× （臨界雷諾數 Re_c1≈30）
- Re/Re_c2 = 5× （完全湍流轉捩 Re_c2≈50）

**訓練困難**：
- ⚠️ 強非線性使優化困難
- ⚠️ 多尺度渦旋結構需高解析度
- ⚠️ 對權重平衡極度敏感
- ⚠️ 容易陷入局部最優

**觀察到的症狀**（1000 epochs 直接訓練）：
- Momentum X 殘差較高（1.24）
- 局部散度波動大（max=0.42）
- 連續性殘差中等（0.315）

### 解決方案：課程學習

**核心思想**：讓網絡從簡單問題開始學習，逐步增加難度

```
Phase 1: Re=50  (轉捩初期)   → 建立基礎流場結構
Phase 2: Re=100 (弱湍流)     → 增加非線性複雜度
Phase 3: Re=240 (發展湍流)   → 達成最終目標
```

**優勢**：
- ✅ 更快收斂（避免從隨機初始化）
- ✅ 更好泛化（逐步學習物理規律）
- ✅ 更低殘差（階段性優化）
- ✅ 更穩定訓練（避免梯度爆炸）

## 📋 課程配置說明

### 階段 1: Re≈50（轉捩初期）

**物理參數**：
- 強迫振幅：A = 0.32
- 雷諾數：Re ≈ 50
- 訓練 epochs：1500

**特點**：
- 流場相對穩定
- 開始出現次級不穩定
- 保持時間平均對稱性

**損失權重**：
```yaml
momentum_x: 1.0
momentum_y: 1.0
continuity: 1.0
periodic_x: 10.0
periodic_y: 10.0
```

**優化器**：
- Adam, lr=0.001

### 階段 2: Re≈100（弱湍流）

**物理參數**：
- 強迫振幅：A = 0.64
- 雷諾數：Re ≈ 100
- 訓練 epochs：2000

**特點**：
- 混沌時間依賴行為
- 渦旋產生與消散
- 能量在不同尺度轉移

**損失權重**（提高物理約束）：
```yaml
momentum_x: 1.5   # ↑ 增加 x 方向權重
continuity: 2.0   # ↑ 強化不可壓縮約束
```

**優化器**（降低學習率）：
- Adam, lr=0.0005

### 階段 3: Re≈240（發展湍流）

**物理參數**：
- 強迫振幅：A = 1.536
- 雷諾數：Re ≈ 240
- 訓練 epochs：3000

**特點**：
- 完全發展湍流
- 寬範圍渦旋尺度
- 強非線性相互作用

**損失權重**（最強約束）：
```yaml
momentum_x: 2.0   # ↑↑ 最高 x 方向權重
continuity: 3.0   # ↑↑ 最強不可壓縮約束
```

**優化器**（最低學習率 + 正則化）：
- Adam, lr=0.0002, weight_decay=1e-5

## 🚀 使用方法

### 方法 1：使用專用腳本（推薦）

```bash
# 執行完整課程學習（自動執行 3 個階段）
python scripts/train_curriculum_kolmogorov.py
```

**輸出**：
```
checkpoints/kolmogorov_2d_curriculum/
├── phase1_final.pth  # Re=50  最終檢查點
├── phase2_final.pth  # Re=100 最終檢查點
└── phase3_final.pth  # Re=240 最終檢查點
```

### 方法 2：手動分階段訓練

```bash
# 階段 1: Re=50
python scripts/train.py --cfg configs/kolmogorov_2d_curriculum_phase1.yml

# 階段 2: Re=100（從階段 1 繼續）
python scripts/train.py --cfg configs/kolmogorov_2d_curriculum_phase2.yml \
  --resume checkpoints/kolmogorov_2d_curriculum/phase1_final.pth

# 階段 3: Re=240（從階段 2 繼續）
python scripts/train.py --cfg configs/kolmogorov_2d_curriculum_phase3.yml \
  --resume checkpoints/kolmogorov_2d_curriculum/phase2_final.pth
```

## 📊 預期結果

### 訓練時間

**總訓練時間**（估計）：
- Phase 1: 1500 epochs ≈ 20 分鐘
- Phase 2: 2000 epochs ≈ 30 分鐘
- Phase 3: 3000 epochs ≈ 50 分鐘
- **總計**：≈ 100 分鐘（1.7 小時）

### 預期損失收斂

| 階段 | 最終 PDE Loss | 最終 Continuity | 最終 Momentum X |
|------|---------------|-----------------|-----------------|
| Phase 1 (Re=50) | < 0.05 | < 0.01 | < 0.05 |
| Phase 2 (Re=100) | < 0.1 | < 0.05 | < 0.2 |
| Phase 3 (Re=240) | < 0.2 | < 0.1 | < 0.5 |

**對比直接訓練**（Re=240, 1000 epochs）：
- PDE Loss: 0.047 → 預期 < 0.2 ✅
- Continuity: 0.315 → 預期 < 0.1 ✅ (改善 3×)
- Momentum X: 1.24 → 預期 < 0.5 ✅ (改善 2.5×)

## 🔬 評估與視覺化

### 評估各階段結果

```bash
# 評估階段 1（Re=50）
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_2d_curriculum/phase1_final.pth \
  --output results/curriculum_phase1

# 評估階段 2（Re=100）
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_2d_curriculum/phase2_final.pth \
  --output results/curriculum_phase2

# 評估階段 3（Re=240）
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_2d_curriculum/phase3_final.pth \
  --output results/curriculum_phase3
```

### 比較視覺化

觀察雷諾數增加時的流場演化：
- **Re=50**：清晰條紋結構，輕微波動
- **Re=100**：條紋開始扭曲，渦旋形成
- **Re=240**：複雜湍流結構，多尺度渦旋

## ⚙️ 進階調整

### 調整網絡容量

如果 Re=240 仍然困難，可以進一步增強網絡：

```yaml
model:
  width: 512        # 384 → 512
  depth: 8          # 6 → 8
  fourier_m: 128    # 64 → 128
```

### 增加採樣點

```yaml
data:
  n_collocation: 30000  # 20000 → 30000
  n_boundary: 6000      # 4000 → 6000
```

### 延長訓練時間

```yaml
curriculum:
  phase3:
    epochs: 5000  # 3000 → 5000
```

## 📈 監控訓練

### 實時監控

```bash
# 監控訓練日誌
tail -f log/kolmogorov_2d_curriculum/training.log

# 查看檢查點
ls -lht checkpoints/kolmogorov_2d_curriculum/
```

### 關鍵指標

**Phase 1（Re=50）**：
- ✅ Periodic loss < 1e-4
- ✅ PDE loss 穩定下降
- ✅ 散度接近 0

**Phase 2（Re=100）**：
- ✅ 損失不發散
- ✅ Momentum 殘差合理
- ⚠️ 可能出現輕微震盪（正常）

**Phase 3（Re=240）**：
- ✅ 損失繼續下降（即使緩慢）
- ⚠️ 允許更高殘差（湍流固有特性）
- 🎯 重點：流場結構正確

## 🎓 理論背景

### Kolmogorov Flow 雷諾數階段

| Re 範圍 | 階段 | 特徵 |
|---------|------|------|
| < 30 | 穩定層流 | 平行條紋，無波動 |
| 30-50 | 第一次不穩定 | 次級流動開始 |
| 50-100 | 轉捩 | 混沌時間依賴 |
| 100-200 | 弱湍流 | 多尺度渦旋 |
| **> 200** | **發展湍流** | **完全湍流級聯** |

### 課程學習的神經網絡視角

```
簡單問題 → 網絡學習基本模式（條紋結構）
         ↓
中等問題 → 網絡學習非線性交互（渦旋形成）
         ↓
複雜問題 → 網絡學習多尺度耦合（湍流級聯）
```

**關鍵**：每個階段都從前一階段的**良好初始化**開始，而非隨機權重。

## 📚 參考文獻

1. Meshalkin & Sinai (1961) - Kolmogorov flow 穩定性理論
2. Lucas & Kerswell (2014) - Re=240 湍流態研究
3. Raissi et al. (2019) - Physics-Informed Neural Networks
4. Bengio et al. (2009) - Curriculum Learning

## 💡 常見問題

**Q: 為什麼不直接訓練 Re=240？**
A: Re=240 是完全湍流（8× 臨界雷諾數），強非線性使直接訓練困難且容易陷入局部最優。

**Q: 可以跳過某個階段嗎？**
A: 不建議。每個階段都為下一階段提供良好的初始化。跳過階段會失去課程學習的優勢。

**Q: 如果某階段不收斂怎麼辦？**
A: 延長該階段訓練時間，或降低學習率。也可以考慮降低該階段的雷諾數。

**Q: 訓練完成後如何驗證結果？**
A: 使用 `evaluate_kolmogorov_quick.py` 評估流場重建品質，檢查物理殘差和守恆性。

---

**建立時間**: 2025-11-20
**版本**: 1.0
**作者**: PINNx Team
