# 對比實驗圖表生成指南

本指南說明如何使用 `scripts/visualize/generate_comparison_figures.py` 生成論文所需的對比圖表。

## 📋 目錄

- [可用圖表類型](#可用圖表類型)
- [通用使用方式](#通用使用方式)
- [具體圖表生成指令](#具體圖表生成指令)
- [輸出規範](#輸出規範)
- [常見問題](#常見問題)

---

## 可用圖表類型

根據 `docs/EXPERIMENT_COMPARISON_PLAN.md` 第 9 節，目前已實作以下圖表：

| 圖表 ID | 圖表名稱 | 用途 | 狀態 |
|---------|----------|------|------|
| **F-S1** | Random vs QR 感測器佈點對比圖 | 展示感測點策略差異 | ✅ 已實作 |
| **F-K1** | K-scan Error 曲線圖 | K 值掃描與誤差關係 | ✅ 已實作 |
| **F-P1** | Prior Weight Sweep 曲線圖 | 找出最佳 prior weight | ✅ 已實作 |
| **F-A1** | Ablation Bar Chart | 消融實驗對比 | ✅ 已實作 |
| F-R1 | 場重建三聯圖 | DNS / PINN / Error 對比 | 📝 使用 `visualize_results.py` |
| F-R2 | Channel flow 統計圖 | U⁺(y⁺), τ_w 分析 | 📝 使用 `comprehensive_evaluation.py` |

---

## 通用使用方式

### 基本語法

```bash
python scripts/visualize/generate_comparison_figures.py \
    --mode <圖表模式> \
    [特定模式參數] \
    --output-dir results/figures \
    --output <輸出檔名>
```

### 參數說明

- `--mode`：圖表模式，可選值：
  - `sensor_comparison`：F-S1 感測器佈點對比
  - `k_scan`：F-K1 K 值掃描曲線
  - `prior_sweep`：F-P1 Prior weight 掃描
  - `ablation`：F-A1 消融實驗條形圖

- `--output-dir`：輸出目錄（預設：`results/figures`）
- `--output`：輸出檔名（若不指定則使用預設名稱）

---

## 具體圖表生成指令

### F-S1: Random vs QR 感測器佈點對比圖

**功能**：在同一背景場上並排顯示 Random 與 QR-Pivot 感測點分佈。

**必要資料**：
- Random 感測器資料檔（`.npz`）
- QR 感測器資料檔（`.npz`）
- 背景場資料檔（DNS 切片，`.npz`）

**指令**：

```bash
python scripts/visualize/generate_comparison_figures.py \
    --mode sensor_comparison \
    --random-sensors data/sensors/random_K100.npz \
    --qr-sensors data/sensors/qr_K100.npz \
    --background-field data/jhtdb/slice_z0.npz \
    --field-name vorticity \
    --view xy \
    --output F-S1_random_vs_qr.png
```

**參數說明**：
- `--field-name`：背景場變數（可選：`vorticity`, `|u|`, `u`, `v`, `w`, `p`）
- `--view`：視角（可選：`xy`, `xz`, `yz`）

**輸出範例**：
- 左圖：Random sensors（紅色標記）
- 右圖：QR-Pivot sensors（藍色標記）
- 背景：渦度場或速度場
- 共用色階：確保可比性

---

### F-K1: K-scan Error 曲線圖

**功能**：繪製不同感測點數量 K 與重建誤差的關係曲線。

**必要資料**：
- K-scan 實驗結果目錄（包含多個 K 值的子目錄）
- 每個子目錄需包含 `metrics.json`

**目錄結構範例**：
```
results/experiments/S2_k_scan/
├── random_K30/
│   └── metrics.json
├── random_K50/
│   └── metrics.json
├── qr_K30/
│   └── metrics.json
├── qr_K50/
│   └── metrics.json
...
```

**指令**：

```bash
python scripts/visualize/generate_comparison_figures.py \
    --mode k_scan \
    --results-dir results/experiments/S2_k_scan \
    --k-values 30 50 80 100 \
    --output F-K1_k_scan.png
```

**參數說明**：
- `--k-values`：K 值列表（若不指定則自動偵測）

**輸出範例**：
- Random 曲線（紅色，含誤差帶）
- QR-Pivot 曲線（藍色，含誤差帶）
- 目標門檻線（15%，橙色虛線）
- 支援多 seed 統計（mean ± std）

---

### F-P1: Prior Weight Sweep 曲線圖

**功能**：繪製不同 prior_weight 與重建誤差、散度誤差、τ_w 誤差的關係。

**必要資料**：
- Prior weight sweep 實驗結果目錄
- 每個子目錄需包含 `metrics.json`

**目錄結構範例**：
```
results/experiments/C2_prior_sweep/
├── prior_weight_0.0/
│   └── metrics.json
├── prior_weight_0.1/
│   └── metrics.json
├── prior_weight_0.3/
│   └── metrics.json
...
```

**指令**：

```bash
python scripts/visualize/generate_comparison_figures.py \
    --mode prior_sweep \
    --results-dir results/experiments/C2_prior_sweep \
    --prior-weights 0.0 0.1 0.3 0.5 \
    --output F-P1_prior_sweep.png
```

**參數說明**：
- `--prior-weights`：prior weight 列表（若不指定則自動偵測）

**輸出範例**：
- 三個子圖：Overall Error / Divergence Error / τ_w Error
- 幫助找出「sweet spot」（避免 prior 過大綁死 PINN）

---

### F-A1: Ablation Bar Chart（消融實驗）

**功能**：繪製消融實驗的條形圖，展示各組件貢獻度。

**必要資料**：
- Ablation 實驗結果目錄
- 需包含 baseline（Full）與各消融版本的結果

**目錄結構範例**：
```
results/experiments/A1_ablation_fourier/
├── full/
│   └── metrics.json
├── without_fourier/
│   └── metrics.json
├── without_gradnorm/
│   └── metrics.json
├── without_rwf/
│   └── metrics.json
...
```

**指令**：

```bash
python scripts/visualize/generate_comparison_figures.py \
    --mode ablation \
    --results-dir results/experiments/A1_ablation_fourier \
    --baseline-name full \
    --output F-A1_ablation.png
```

**參數說明**：
- `--baseline-name`：基線實驗名稱（預設：`full`）

**輸出範例**：
- 左圖：Δerror（相對於 Full 的誤差變化百分比）
- 右圖：收斂速度（達到 15% 門檻所需 epochs）
- 正值（灰色）：性能下降
- 負值（綠色）：性能提升

---

## 輸出規範

所有圖表遵循 `docs/EXPERIMENT_COMPARISON_PLAN.md` 第 9.1 節規範：

### 1. 色階統一原則
- **同一組比較必須使用相同色階**（vmin, vmax）
- DNS vs PINN 場圖：基於 DNS 的 1-99% percentile
- 誤差圖：獨立色階，但同組方法一致

### 2. 感測點標記規格（固定）
- Marker 大小：40
- 透明度：0.8
- 邊框顏色：黑色
- 邊框寬度：1.0

### 3. 顏色方案
- Random：紅色 (#E74C3C)
- QR-Pivot：藍色 (#3498DB)
- Prior：綠色 (#2ECC71)
- Baseline：灰色 (#95A5A6)
- 目標門檻：橙色 (#F39C12)

### 4. 必含標註
- K 值（感測點數量）
- 選點方法（Random / QR / Hybrid）
- 變數名稱與無因次化（如 u/u_τ）
- 色條單位
- 壓力：使用 **∇p** 或 p′（去均值）

### 5. 圖片品質
- 螢幕顯示：150 DPI
- 儲存輸出：300 DPI
- 格式：PNG（含透明背景支援）

---

## 常見問題

### Q1: 如何生成 2D Kolmogorov 的感測器對比圖？

**A**: 假設你已經有 Random 和 QR 感測器，並且有 DNS 場資料：

```bash
# 步驟1：確保有 DNS 資料
# 資料路徑範例：data/kolmogorov/re50_kf4_snapshot.npz

# 步驟2：生成圖表
python scripts/visualize/generate_comparison_figures.py \
    --mode sensor_comparison \
    --random-sensors data/sensors/kolmogorov_random_K100.npz \
    --qr-sensors data/sensors/kolmogorov_qr_K100.npz \
    --background-field data/kolmogorov/re50_kf4_snapshot.npz \
    --field-name vorticity \
    --view xy \
    --output results/figures/F-S1_kolmogorov_K100.png
```

### Q2: 如何自動批次生成所有圖表？

**A**: 創建一個 shell 腳本：

```bash
#!/bin/bash
# batch_generate_figures.sh

OUTPUT_DIR="results/figures/thesis"
mkdir -p $OUTPUT_DIR

echo "生成 F-S1: Sensor comparison..."
python scripts/visualize/generate_comparison_figures.py \
    --mode sensor_comparison \
    --random-sensors data/sensors/random_K100.npz \
    --qr-sensors data/sensors/qr_K100.npz \
    --background-field data/jhtdb/slice_z0.npz \
    --output-dir $OUTPUT_DIR \
    --output F-S1_random_vs_qr.png

echo "生成 F-K1: K-scan curve..."
python scripts/visualize/generate_comparison_figures.py \
    --mode k_scan \
    --results-dir results/experiments/S2_k_scan \
    --output-dir $OUTPUT_DIR \
    --output F-K1_k_scan.png

echo "生成 F-P1: Prior sweep..."
python scripts/visualize/generate_comparison_figures.py \
    --mode prior_sweep \
    --results-dir results/experiments/C2_prior_sweep \
    --output-dir $OUTPUT_DIR \
    --output F-P1_prior_sweep.png

echo "生成 F-A1: Ablation chart..."
python scripts/visualize/generate_comparison_figures.py \
    --mode ablation \
    --results-dir results/experiments/A1_ablation_fourier \
    --output-dir $OUTPUT_DIR \
    --output F-A1_ablation.png

echo "✅ 所有圖表生成完成！"
echo "輸出目錄: $OUTPUT_DIR"
```

執行：
```bash
chmod +x batch_generate_figures.sh
./batch_generate_figures.sh
```

### Q3: metrics.json 應該包含哪些欄位？

**A**: 標準 metrics.json 格式範例：

```json
{
  "relative_l2_overall": 0.1234,
  "relative_l2_u": 0.1100,
  "relative_l2_v": 0.1200,
  "relative_l2_p": 0.1400,
  "divergence_error_mean": 1.23e-3,
  "divergence_error_max": 5.67e-3,
  "tau_w_relative_error": 0.0890,
  "epochs_to_threshold": 3500,
  "seeds": [
    {"relative_l2_overall": 0.1200},
    {"relative_l2_overall": 0.1250},
    {"relative_l2_overall": 0.1250}
  ]
}
```

若有多 seed 運行，加入 `seeds` 欄位（可選）。

### Q4: 如何修改顏色方案或標記樣式？

**A**: 編輯 `scripts/visualize/generate_comparison_figures.py` 中的全域配置：

```python
# 顏色方案
COLORS = {
    'random': '#E74C3C',      # 紅色
    'qr': '#3498DB',          # 藍色
    'prior': '#2ECC71',       # 綠色
    'baseline': '#95A5A6',    # 灰色
    'target': '#F39C12',      # 橙色
}

# 感測點標記規格
SENSOR_MARKER_CONFIG = {
    'size': 40,
    'alpha': 0.8,
    'edgecolors': 'black',
    'linewidths': 1.0,
}
```

### Q5: 如何處理缺失的實驗數據？

**A**: 腳本會自動跳過不存在的檔案，並在日誌中警告。例如：

```
[WARNING] 未找到 random_K30/metrics.json，跳過該數據點
```

確保至少有一組數據（Random 或 QR）可用，否則會產生空圖。

---

## 進階功能

### 自訂背景場計算

若要使用自訂的背景場（例如 Q-criterion, λ2），可在 `_extract_background_field` 方法中添加：

```python
elif field_name == 'Q':
    # 計算 Q-criterion
    u = data.get('u')
    v = data.get('v')
    # ... 實作 Q-criterion 計算
    field = Q_value
```

### 多視角並排

若要同時生成 xy, xz, yz 三個視角：

```bash
for view in xy xz yz; do
    python scripts/visualize/generate_comparison_figures.py \
        --mode sensor_comparison \
        --random-sensors data/sensors/random_K100.npz \
        --qr-sensors data/sensors/qr_K100.npz \
        --background-field data/jhtdb/channel_flow_full.npz \
        --view $view \
        --output F-S1_random_vs_qr_${view}.png
done
```

---

## 與其他腳本的配合

### 完整工作流程

1. **感測器生成**：
   ```bash
   python scripts/generate/sensors/generate_sensors_periodic_qr.py \
       --input data/jhtdb/channel_flow_re1000.h5 \
       --k 100 \
       --output data/sensors/qr_K100.npz
   ```

2. **訓練模型**（各實驗配置）：
   ```bash
   python scripts/train/train.py --config configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml
   ```

3. **評估與生成 metrics.json**：
   ```bash
   python scripts/evaluate/evaluate_checkpoint.py \
       --checkpoint checkpoints/qr_K50/best_model.pth \
       --config configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml \
       --output results/experiments/S2_k_scan/qr_K50
   ```

4. **生成對比圖表**：
   ```bash
   python scripts/visualize/generate_comparison_figures.py \
       --mode k_scan \
       --results-dir results/experiments/S2_k_scan \
       --output results/figures/F-K1_k_scan.png
   ```

---

## 引用與相關文檔

- **實驗計畫**：`docs/EXPERIMENT_COMPARISON_PLAN.md`
- **配置參考**：`docs/CONFIG_REFERENCE.md`
- **快速開始**：`docs/QUICK_START.md`
- **腳本說明**：`scripts/README.md`

---

## 支援與問題回報

若遇到問題或需要新功能，請：

1. 檢查此文檔與 `EXPERIMENT_COMPARISON_PLAN.md`
2. 查看日誌輸出（logger 會提供詳細錯誤訊息）
3. 確認資料格式符合預期
4. 提交 issue 或聯繫專案維護者

---

**最後更新**：2025-12-14  
**腳本版本**：v1.0  
**維護者**：PINNs-MVP 專案團隊
