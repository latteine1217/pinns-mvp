# Scripts 使用指南

**最後更新**: 2025-11-26  
**狀態**: ✅ 已清理冗餘腳本 (49 → 22 核心腳本)

---

## 📁 目錄結構

```
scripts/
├── 🎯 訓練 (2)
│   ├── train.py ⭐
│   └── train_curriculum_kolmogorov.py
│
├── 📊 評估 (3)
│   ├── evaluate.py ⭐
│   ├── evaluate_checkpoint.py
│   ├── evaluate_curriculum.py
│   └── comprehensive_evaluation.py
│
├── 📈 視覺化 (3)
│   ├── visualize_results.py ⭐
│   ├── visualize_qr_sensors.py ⭐
│   └── visualize_kolmogorov_dns.py
│
├── 🔍 監控 (2)
│   ├── monitor_training.py ⭐
│   └── monitor_kolmogorov_dns_status.py ⭐
│
├── 🧬 感測器生成 (2)
│   ├── generate_sensors_k500.py ⭐
│   └── compare_qr_strategies.py
│
├── 🌊 DNS 生成 (1)
│   └── generate_kolmogorov_dns.py ⭐
│
├── ✅ 數據驗證 (3)
│   ├── verify_kolmogorov_reynolds.py ⭐
│   ├── validate_2d_turbulence_spectrum.py
│   ├── validate_constraints.py
│   └── verify_jhtdb_data.py
│
├── 🛠️ 工具 (4)
│   ├── calculate_reynolds_parameters.py ⭐
│   ├── create_dns_animation.py
│   ├── fetch_channel_flow.py
│   └── generate_jhtdb_field_plots.py
│
├── 🐛 除錯 (子目錄)
│   └── debug/ (48 診斷腳本)
│
├── 🧪 測試驗證 (子目錄)
│   └── validation/ (7 物理驗證腳本)
│
└── 📦 歸檔
    └── archive/ (36 舊腳本 + shell 腳本)
```

---

## 🎯 核心腳本快速參考

### 1️⃣ 訓練 PINNs

#### 主訓練器
```bash
# 基本訓練
python scripts/train.py --cfg configs/your_config.yml --device cuda

# 從檢查點恢復
python scripts/train.py --cfg configs/your_config.yml \
  --resume checkpoints/exp/epoch_500.pth

# Ensemble 訓練（不確定性量化）
python scripts/train.py --cfg configs/your_config.yml --ensemble
```

**說明**: 支援所有配置驅動訓練，包含 VS-PINN、標準 PINN、Ensemble

---

#### 課程學習訓練
```bash
# Kolmogorov flow 課程學習（Re 梯度）
python scripts/train_curriculum_kolmogorov.py \
  --base-config configs/kolmogorov_base.yml \
  --stages configs/curriculum_stages.yml
```

**說明**: 階段式訓練（Re=30 → 55 → 100），每階段使用前一階段權重

---

### 2️⃣ 評估與驗證

#### 統一評估入口
```bash
# 評估檢查點（自動載入配置）
python scripts/evaluate.py \
  --checkpoint checkpoints/exp/best_model.pth \
  --config configs/exp.yml

# 完整物理驗證（守恆定律、統計量）
python scripts/comprehensive_evaluation.py \
  --checkpoint checkpoints/exp/best_model.pth \
  --config configs/exp.yml \
  --output results/evaluation/
```

**輸出**:
- 相對 L2 誤差（u, v, w, p）
- 質量守恆（∇·u）
- 動量守恆（NS 殘差）
- 統計量（均值、二階動量、雷諾應力）
- 能譜分析（Kolmogorov flow）

---

#### 檢查點評估
```bash
# 快速評估單一檢查點
python scripts/evaluate_checkpoint.py \
  --checkpoint checkpoints/exp/epoch_1000.pth \
  --config configs/exp.yml
```

---

#### 課程學習評估
```bash
# 評估課程學習各階段
python scripts/evaluate_curriculum.py \
  --exp-dir checkpoints/curriculum_exp/ \
  --stages 3
```

---

### 3️⃣ 視覺化

#### 統一視覺化工具
```bash
# 視覺化訓練結果（預測/真值/誤差）
python scripts/visualize_results.py \
  --checkpoint checkpoints/exp/best_model.pth \
  --output results/visualizations/

# 支援多種數據源：
# - Kolmogorov flow (2D)
# - Channel flow (3D)
# - 切片數據
```

**輸出**:
- 3 面板圖（預測/真值/誤差）
- 能譜比較（Kolmogorov）
- 統計剖面（通道流）
- 時間演化（如有時間序列）

---

#### QR-Pivot 感測點視覺化
```bash
# 視覺化感測點分佈與品質
python scripts/visualize_qr_sensors.py \
  --input data/sensors_K100.npz \
  --output results/sensor_analysis/

# 從 JHTDB 數據重新計算並比較策略
python scripts/visualize_qr_sensors.py \
  --jhtdb-data data/jhtdb/channel_flow.h5 \
  --n-sensors 100 --compare-strategies \
  --output results/comparison/
```

**輸出**:
- 2D/3D 空間分佈圖
- 品質指標（條件數、能量比例、覆蓋率）
- 策略比較（QR-Pivot vs POD vs Random）

**參考**: `docs/QR_SENSOR_VISUALIZATION_GUIDE.md`

---

#### DNS 數據視覺化
```bash
# 視覺化 Kolmogorov flow DNS 數據
python scripts/visualize_kolmogorov_dns.py \
  --input data/kolmogorov_dns_re100_kf8_T100.h5 \
  --output results/dns_visualization/ \
  --snapshots 5  # 選擇幾個時間點

# 創建動畫 GIF
python scripts/create_dns_animation.py \
  --input data/kolmogorov_dns_re100_kf8_T100.h5 \
  --output results/animations/re100_kf8.gif \
  --fps 10
```

---

### 4️⃣ 監控

#### 訓練監控
```bash
# 實時監控訓練進度
python scripts/monitor_training.py --exp your_exp_name

# 或直接查看日誌
tail -f log/your_exp/training.log
```

**顯示**:
- 最新 loss 值與趨勢
- PDE ratio（物理約束佔比）
- 梯度範數
- 學習率
- 預估剩餘時間

---

#### DNS 生成監控
```bash
# 監控 Kolmogorov DNS 生成狀態
python scripts/monitor_kolmogorov_dns_status.py --details

# 持續監控
watch -n 10 'python scripts/monitor_kolmogorov_dns_status.py'
```

**輸出**: `DNS_STATUS_SUMMARY.md`（包含物理量、流動狀態、進度）

---

### 5️⃣ 感測器生成

#### QR-Pivot 感測點生成
```bash
# 為 Kolmogorov flow 生成 QR-pivot 感測點
python scripts/generate_sensors_k500.py \
  --input data/kolmogorov_dns_re100_kf8.h5 \
  --snapshot-idx 750 \
  --K 100 \
  --output data/sensors_K100.npz \
  --n-modes 50
```

**品質指標**:
- 條件數 < 50（優秀）
- 能量比例 > 0.95
- 空間覆蓋均勻

---

#### 策略比較
```bash
# 比較不同感測器選擇策略
python scripts/compare_qr_strategies.py \
  --input data/kolmogorov_dns.h5 \
  --K-values 50,100,200 \
  --strategies qr_pivot,random,greedy,pod \
  --output results/strategy_comparison/
```

---

### 6️⃣ DNS 生成

#### Kolmogorov Flow DNS 生成
```bash
# 生成 DNS 數據（參數化）
python scripts/generate_kolmogorov_dns.py \
  --Re 100 \
  --k_f 8 \
  --T_max 100 \
  --dt 0.1 \
  --resolution 512 \
  --output data/kolmogorov_dns_re100_kf8_T100.h5

# 背景運行（長時間模擬）
nohup python scripts/generate_kolmogorov_dns.py \
  --Re 100 --k_f 8 --T_max 100 \
  > log/dns_generation.log 2>&1 &
```

**物理參數**:
- `Re`: 雷諾數（基於 Musacchio & Boffetta 2014）
- `k_f`: 強迫波數（典型值 4 或 8）
- `T_max`: 最大時間（建議 ≥100 for 統計穩定）
- `dt`: 時間步長
- `resolution`: 空間解析度（512 或 1024）

---

### 7️⃣ 數據驗證

#### 雷諾數驗證與計算
```bash
# 驗證 DNS 數據的實際雷諾數
python scripts/verify_kolmogorov_reynolds.py \
  --input data/kolmogorov_dns.h5 \
  --expected-Re 100

# 計算特定參數的雷諾數
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 --nu 0.0125 --k 8

# 規劃新 DNS（計算所需 nu）
python scripts/calculate_reynolds_parameters.py \
  --target-Re 100 --f0 1.0 --k 8 --solve-nu

# 批量掃描
python scripts/calculate_reynolds_parameters.py \
  --f0 1.0 --k 8 --nu-range 0.005 0.025 0.005
```

**參考**: `scripts/README_REYNOLDS_CALCULATOR.md`

---

#### 能譜驗證
```bash
# 驗證 2D 湍流能譜（k^(-5/3), k^(-3)）
python scripts/validate_2d_turbulence_spectrum.py \
  --checkpoint checkpoints/exp/best_model.pth \
  --reference data/kolmogorov_dns.h5
```

---

#### JHTDB 數據驗證
```bash
# 驗證從 JHTDB 獲取的數據
python scripts/verify_jhtdb_data.py \
  --input data/jhtdb/channel_flow.h5
```

---

### 8️⃣ 工具腳本

#### JHTDB 數據獲取
```bash
# 從 JHTDB 獲取通道流數據
python scripts/fetch_channel_flow.py \
  --dataset channel \
  --time-range 0.0 26.0 \
  --output data/jhtdb/channel_flow.h5
```

---

#### 場圖生成
```bash
# 生成 JHTDB 場圖
python scripts/generate_jhtdb_field_plots.py \
  --input data/jhtdb/channel_flow.h5 \
  --output results/field_plots/
```

---

## 🐛 除錯工具 (scripts/debug/)

當訓練出現問題時，使用完整診斷工具鏈：

### 訓練失敗總體診斷
```bash
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint checkpoints/exp/epoch_100.pth \
  --config configs/exp.yml \
  --output results/diagnosis/
```

**診斷內容**:
- 檢查點完整性分析
- Loss 趨勢圖（識別發散點）
- 配置參數驗證
- 建議修正方案

**參考**: `docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md`

---

### NS 方程診斷
```bash
python scripts/debug/diagnose_ns_equations.py \
  --checkpoint checkpoints/exp/latest.pth \
  --config configs/exp.yml
```

---

### 其他診斷工具 (48 個)
- `diagnose_boundary_conditions.py` - 邊界條件檢查
- `diagnose_pressure_failure.py` - 壓力場分析
- `debug_gradient_computation.py` - 梯度計算驗證
- `diagnose_sensor_overfitting.py` - 感測點過擬合診斷
- ... 等 (詳見 `scripts/debug/`)

---

## 🧪 物理驗證 (scripts/validation/)

### 物理驗證測試
```bash
python scripts/validation/physics_validation.py \
  --checkpoint checkpoints/exp/best_model.pth
```

### 通道流物理測試
```bash
python scripts/validation/test_channel_flow_physics.py \
  --checkpoint checkpoints/exp/best_model.pth
```

### 守恆定律測試
```bash
python scripts/validation/test_conservation_with_model.py \
  --checkpoint checkpoints/exp/best_model.pth
```

---

## 📦 歸檔腳本 (scripts/archive/)

已移動 **36 個過時/冗餘腳本** 至 `archive/`：

- **評估冗餘** (5): `evaluate_kolmogorov_full.py`, `evaluate_kolmogorov_quick.py`, ...
- **監控冗餘** (4): `monitor_dns_re100_T100.py`, `quick_monitor.py`, ...
- **視覺化冗餘** (6): `visualize_kolmogorov_results.py`, `visualize_dns_comparison.py`, ...
- **數據驗證冗餘** (6): `check_dns_re100.py`, `check_dns_re60.py`, ...
- **訓練冗餘** (2): `train_kolmogorov_helper.py`, `train_pure_pde.py`
- **其他** (4): DNS 生成、感測器、工具冗餘
- **Shell 腳本** (9): 過時的監控與處理腳本

**注意**: 如需使用歸檔腳本，請檢查是否有更新的通用版本。

---

## 📊 整理前後對比

| 類別 | 整理前 | 整理後 | 減少 | 說明 |
|------|--------|--------|------|------|
| 訓練 | 6 | 2 | -4 | 合併至 `train.py` |
| 評估 | 9 | 4 | -5 | 統一至 `evaluate.py` |
| 視覺化 | 9 | 3 | -6 | 統一至 `visualize_results.py` |
| 監控 | 5 | 2 | -3 | 合併至 `monitor_*.py` |
| 感測器 | 3 | 2 | -1 | 統一至 `generate_sensors_k500.py` |
| DNS 生成 | 2 | 1 | -1 | 參數化至單一腳本 |
| 數據驗證 | 8 | 4 | -4 | 合併功能 |
| 工具 | 7 | 4 | -3 | 保留核心工具 |
| **總計** | **49** | **22** | **-27** | **減少 55%** |

---

## 🎯 常用工作流程

### 完整訓練流程
```bash
# 1. 驗證雷諾數（DNS 生成前）
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8

# 2. 生成 DNS 數據
python scripts/generate_kolmogorov_dns.py --Re 100 --k_f 8 --T_max 100

# 3. 監控 DNS 生成
python scripts/monitor_kolmogorov_dns_status.py --details

# 4. 視覺化 DNS（檢查物理正確性）
python scripts/visualize_kolmogorov_dns.py --input data/kolmogorov_dns.h5

# 5. 生成 QR-pivot 感測點
python scripts/generate_sensors_k500.py --input data/kolmogorov_dns.h5 --K 100

# 6. 視覺化感測點品質
python scripts/visualize_qr_sensors.py --input data/sensors_K100.npz

# 7. 訓練 PINNs
python scripts/train.py --cfg configs/my_exp.yml --device cuda

# 8. 監控訓練（另開終端）
python scripts/monitor_training.py --exp my_exp

# 9. 評估結果
python scripts/comprehensive_evaluation.py \
  --checkpoint checkpoints/my_exp/best_model.pth \
  --config configs/my_exp.yml

# 10. 視覺化結果
python scripts/visualize_results.py \
  --checkpoint checkpoints/my_exp/best_model.pth
```

---

### 快速除錯流程
```bash
# 1. 訓練失敗診斷
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint checkpoints/exp/epoch_100.pth

# 2. 感測點品質檢查
python scripts/visualize_qr_sensors.py --input data/sensors.npz

# 3. 物理驗證
python scripts/validation/physics_validation.py \
  --checkpoint checkpoints/exp/latest.pth

# 4. 梯度計算檢查
python scripts/debug/debug_gradient_computation.py \
  --checkpoint checkpoints/exp/latest.pth
```

---

## 📚 相關文檔

- **技術文檔**: `docs/TECHNICAL_DOCUMENTATION.md`
- **配置指南**: `configs/README.md`, `docs/CONFIG_GUIDE.md`
- **QR 感測點**: `docs/QR_SENSOR_VISUALIZATION_GUIDE.md`
- **診斷流程**: `docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md`
- **雷諾數計算**: `scripts/README_REYNOLDS_CALCULATOR.md`
- **Kolmogorov 指南**: `docs/KOLMOGOROV_CURRICULUM_GUIDE.md`

---

## ⚠️ 重要提醒

1. **配置驅動**: 所有腳本優先使用配置文件，避免硬編碼參數
2. **物理驗證**: DNS 生成後必須執行 `verify_kolmogorov_reynolds.py`
3. **感測點品質**: 訓練前檢查條件數 < 50，能量比 > 0.95
4. **訓練監控**: 長時間訓練建議背景運行 + `monitor_training.py`
5. **檢查點評估**: 定期評估中間檢查點，避免過擬合

---

**維護者**: PINNs-MVP 團隊  
**更新日期**: 2025-11-26  
**清理狀態**: ✅ 已完成（49 → 22 核心腳本）
