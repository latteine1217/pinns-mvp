# Kolmogorov Flow 2D 配置文件指南

本目錄包含 **Kolmogorov Flow 2D** 的訓練配置，用於研究週期性強迫驅動的二維湍流流動。

---

## 🎯 配置分類

### 1️⃣ 基準配置

| 文件名 | 用途 | 訓練時間 | K 點數 | Epochs |
|--------|------|---------|--------|--------|
| `kolmogorov_2d_baseline.yml` | 稀疏感測點逆問題基準 | 10-20 min | 50 | 100 |

**特性**:
- 雙週期性邊界條件
- 正弦強迫項 (A=1.0, k_f=4)
- Re ≈ 62.8

---

### 2️⃣ Re=30 時空混沌態系列

探討 Kolmogorov flow 在 **Re=30** 時進入時空混沌/局域混沌斑狀態的物理現象。

| 文件名 | 變體 | 訓練時間 | Epochs | 用途 |
|--------|------|---------|--------|------|
| `kolmogorov_2d_chaos_re30.yml` | 標準版 | 中等 | - | 標準混沌態研究 |
| `kolmogorov_2d_chaos_re30_full.yml` | 完整訓練 | 1-2 hrs | 2000 | 論文級結果 |
| `kolmogorov_2d_chaos_re30_quick.yml` | 快速測試 | 10-15 min | 500 | 快速驗證 |

**物理參數**:
- 強迫公式: F₀ = Re × ν² × k³ = 30 × 0.02² × 4³ = 0.768
- 層流解: U = F₀/(ν×k²) = 2.4
- 線性失穩門檻: Re_c ≈ 1.414
- 時空混沌區間: Re ≈ 30-40 (k_f=4 時)

---

### 3️⃣ Re=60 完全發展湍流系列

探討 Kolmogorov flow 在 **Re=60** 時的完全發展湍流態。

| 文件名 | 變體 | 訓練時間 | Epochs | 用途 |
|--------|------|---------|--------|------|
| `kolmogorov_2d_turbulent_re60.yml` | 標準版 | 長期 | 5000 | 統計湍流分析 |
| `kolmogorov_2d_turbulent_pure_pde.yml` | 純 PDE | 中等 | 500 | 無 DNS 資料驅動 |

**物理參數**:
- 強迫公式: F₀ = Re × ν² × k³ = 60 × 0.02² × 4³ = 1.536
- 層流解: U = F₀/(ν×k²) = 4.8
- 統計湍流區間: Re ≥ 60

**差異說明**:
- `turbulent_re60.yml`: 使用感測器數據 + PDE 聯合訓練
- `turbulent_pure_pde.yml`: **僅使用物理方程**（無觀測資料），測試 PINNs 純逆問題能力

---

### 4️⃣ 課程學習

| 文件名 | 階段 | 用途 |
|--------|------|------|
| `kolmogorov_2d_curriculum.yml` | Re=50 → 100 → 240 | 漸進式湍流強度訓練 |

**策略**: 從簡單流動開始（Re=50），逐步提高雷諾數至完全湍流（Re=240）

---

### 5️⃣ 測試/除錯

| 文件名 | 用途 | Epochs |
|--------|------|--------|
| `kolmogorov_2d_test_periodic.yml` | 週期性邊界損失驗證 | 100 |

**目的**:
- 驗證週期性邊界條件實現正確性
- 檢查邊界處的連續性誤差

---

## 🚀 快速開始

### 基準測試（推薦新手）
```bash
python scripts/train.py --cfg configs/kolmogorov_2d_baseline.yml
```

### Re=30 混沌態快速測試
```bash
python scripts/train.py --cfg configs/kolmogorov_2d_chaos_re30_quick.yml
```

### Re=60 完全湍流
```bash
python scripts/train.py --cfg configs/kolmogorov_2d_turbulent_re60.yml
```

### 課程學習
```bash
python scripts/train_curriculum_kolmogorov.py --cfg configs/kolmogorov_2d_curriculum.yml
```

---

## 📊 評估與視覺化

### 評估訓練結果
```bash
# 快速評估
python scripts/evaluate_kolmogorov_quick.py --checkpoint checkpoints/<exp_name>/best_model.pth

# 完整評估
python scripts/evaluate_kolmogorov_full.py --checkpoint checkpoints/<exp_name>/best_model.pth
```

### 視覺化流場
```bash
python scripts/visualize_kolmogorov_results.py \
    --checkpoint checkpoints/<exp_name>/best_model.pth \
    --output results/<exp_name>/
```

---

## 🔧 參數調優建議

### 提高 Re 數（增強湍流）
```yaml
physics:
  nu: 0.01  # 降低黏度 → 提高 Re
```

### 調整強迫強度
```yaml
physics:
  forcing:
    amplitude: 1.5  # 增加強迫振幅
    wavenumber: 4   # 強迫波數
```

### 加速訓練（測試用）
```yaml
training:
  n_epochs: 500     # 降低 epochs
  warmup_epochs: 10 # 減少 warmup
```

---

## 📚 理論背景

**Kolmogorov Flow** 是一種經典的二維湍流基準問題，由週期性正弦強迫驅動：

\[
\mathbf{f} = A \sin(k_f y) \hat{\mathbf{x}}
\]

**控制方程**（2D Navier-Stokes）:
\[
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}
\]
\[
\nabla \cdot \mathbf{u} = 0
\]

**邊界條件**: 雙週期性（x, y 方向）

**物理現象**:
- **Re < 5**: 層流穩定
- **5 ≤ Re < 30**: 週期性解、擬週期解
- **30 ≤ Re < 60**: 時空混沌、局域混沌斑
- **Re ≥ 60**: 完全發展湍流

---

## 🗑️ 已移除的冗餘配置

以下配置已移至 `configs/archive/redundant_kolmogorov/`:

| 文件名 | 移除原因 |
|--------|---------|
| `kolmogorov_2d_baseline_fixed.yml` | 與 `config_template_example.yml` 重複（通用模板） |
| `kolmogorov_2d_test_quick.yml` | 與 `turbulent_re60.yml` 重複（僅 epochs 不同） |
| `kolmogorov_2d_quick_test.yml` | 與 `test_periodic.yml` 重複 |

---

## 📖 相關文檔

- [主配置管理指南](./README.md)
- [標準化模板](./templates/README.md)
- [技術文檔](../docs/TECHNICAL_DOCUMENTATION.md)

---

**維護者**: PINNs Research Team  
**最後更新**: 2025-11-21
