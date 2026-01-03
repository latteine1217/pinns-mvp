# Scripts 使用指南

**活躍腳本**: 30 個核心腳本（已分類至 8 個子目錄）  
**歸檔腳本**: 23 個（實驗性/一次性/重複功能）

---

## 目錄結構

```
scripts/
├── train/           (1)  訓練腳本
├── evaluate/        (4)  評估工具
├── visualize/       (6)  視覺化工具
├── generate/        (5)  數據生成
│   ├── dns/         (3)  DNS 數據生成
│   └── sensors/     (2)  感測點生成
├── calculate/       (2)  參數計算
├── compare/         (3)  對比分析
├── tools/           (3)  實用工具
├── validation/      (7)  物理驗證
├── debug/           (5)  診斷工具
└── archive/        (23)  歸檔腳本
```

---

## 核心腳本分類

### 🚀 訓練 (train/)
```bash
python scripts/train/train.py --cfg <config.yml> [--device cuda] [--ensemble]
```

### 📊 評估 (evaluate/)
```bash
# 主要評估
python scripts/evaluate/evaluate.py --checkpoint <path> --config <config.yml>

# 檢查點評估
python scripts/evaluate/evaluate_checkpoint.py --checkpoint <path> --config <config.yml>

# 課程學習評估
python scripts/evaluate/evaluate_curriculum.py --exp-dir <dir> --stages <n>

# 完整物理驗證
python scripts/evaluate/comprehensive_evaluation.py --checkpoint <path> --config <config.yml>
```

### 📈 視覺化 (visualize/)
```bash
# 訓練結果
python scripts/visualize/visualize_results.py --checkpoint <path> --output <dir>

# 感測點分析（V7 品質檢查）
python scripts/visualize/visualize_qr_sensors.py --input <sensors.npz> --output <dir>

# DNS 視覺化
python scripts/visualize/visualize_kolmogorov_dns.py --input <dns.h5> --output <dir>

# JHTDB Channel 2D slice（可重現圖）
python scripts/visualize/plot_jhtdb_channel_2d_slice.py --field u --outdir thesis/result_figures

# 通道流 3D
python scripts/visualize/visualize_channel_3d.py --input <channel.h5> --output <dir>

# 自適應採樣
python scripts/visualize/visualize_adaptive_resampling.py --checkpoint <path> --output <dir>

# DNS 動畫
bash scripts/visualize/view_dns_results.sh <dns.h5>
```

### 🔬 DNS 生成 (generate/dns/)
```bash
# Kolmogorov Flow DNS
python scripts/generate/dns/generate_kolmogorov_dns.py \
  --Re <value> --k_f <value> --nu <value> \
  --T_max <value> --resolution <value> --output <path>

# 低保真數據
python scripts/generate/dns/generate_kolmogorov_lowfi.py \
  --Re <value> --k_f <value> --resolution <value> --output <path>

# RANS 先驗
python scripts/generate/dns/generate_kolmogorov_rans.py \
  --Re <value> --k_f <value> --nu <value> \
  --T_avg_start <value> --T_avg_end <value> --output <path>
```

### 📍 感測點生成 (generate/sensors/)
```bash
# Kolmogorov Flow (V7 推薦)
python scripts/generate/sensors/generate_sensors_periodic_qr.py \
  --dns-path <dns.h5> --K <n_sensors> \
  --oversample-factor 3.0 --output <sensors.npz>

# Channel Flow
python scripts/generate/sensors/generate_channel_flow_sensors_qr.py \
  --input <channel.h5> --K <n_sensors> --output <sensors.npz>
```

### ✅ 驗證 (validation/)
```bash
# DNS 物理守恆
python scripts/validation/validate_dns_physics.py --input <dns.h5>

# DNS 解析度
python scripts/validation/validate_dns_resolution.py --input <dns.h5>

# 2D 湍流能譜
python scripts/validation/validate_2d_turbulence_spectrum.py --checkpoint <path> --reference <dns.h5>

# 約束條件
python scripts/validation/validate_constraints.py --checkpoint <path>

# RANS 能量平衡
python scripts/validation/validate_rans_energy_balance.py --input <rans.h5>

# Kolmogorov 雷諾數
python scripts/validation/verify_kolmogorov_reynolds.py --input <dns.h5> --expected-Re <value>

# NS 守恆
python scripts/validation/validate_ns_conservation.py --checkpoint <path>
```

### 🧮 參數計算 (calculate/)
```bash
# 雷諾數計算
python scripts/calculate/calculate_reynolds_parameters.py \
  --f0 <value> --nu <value> --k <value>

# 規劃新 DNS（求解 ν）
python scripts/calculate/calculate_reynolds_parameters.py \
  --target-Re <value> --f0 <value> --k <value> --solve-nu

# 低保真參數計算
python scripts/calculate/calculate_lowfi_parameters.py \
  --Re <value> --target-resolution <value>
```

### 🔄 比較工具 (compare/)
```bash
# 低保真 vs 高保真
python scripts/compare/compare_lowfi_hifi.py \
  --lowfi <path> --hifi <path> --output <dir>

# QR 感測點策略對比
python scripts/compare/compare_qr_strategies.py \
  --input <dns.h5> --K-values 50,100,200 --output <dir>

# 感測點策略對比
python scripts/compare/compare_sensor_strategies.py \
  --input <dns.h5> --strategies qr,random,greedy --output <dir>
```

### 🛠️ 工具 (tools/)
```bash
# JHTDB 數據獲取
python scripts/tools/fetch_channel_flow.py \
  --dataset channel --time-range 0.0 26.0 --output <path>

# 訓練監控
python scripts/tools/monitor_training_speed.py --log-file <training.log>

# JHTDB 數據驗證
python scripts/tools/verify_jhtdb_data.py --input <jhtdb.h5>
```

---

## 常用工作流程

### 完整訓練流程
```bash
# 1. 計算參數
python scripts/calculate/calculate_reynolds_parameters.py --target-Re 50 --solve-nu

# 2. 生成 DNS
python scripts/generate/dns/generate_kolmogorov_dns.py --Re 50 --k_f 4 --nu 0.0125 \
  --T_max 100 --resolution 512 --output data/kolmogorov_dns/re50_kf4.h5

# 3. 驗證 DNS
python scripts/validation/validate_dns_physics.py --input data/kolmogorov_dns/re50_kf4.h5

# 4. 生成感測點（V7）
python scripts/generate/sensors/generate_sensors_periodic_qr.py \
  --dns-path data/kolmogorov_dns/re50_kf4.h5 --K 100 \
  --oversample-factor 3.0 --output data/kolmogorov_dns/sensors_K100_v7.npz

# 5. 生成 RANS 先驗（可選）
python scripts/generate/dns/generate_kolmogorov_rans.py --Re 50 --k_f 4 --nu 0.0125 \
  --T_avg_start 50.0 --T_avg_end 100.0 --output data/kolmogorov_dns/rans_re50_kf4.h5

# 6. 訓練
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml

# 7. 評估
python scripts/evaluate/evaluate.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --config configs/your_exp.yml

# 8. 視覺化
python scripts/visualize/visualize_results.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --output results/visualizations/
```

### 快速測試
```bash
python scripts/train/train.py --cfg configs/quick_test_rans_prior.yml
```

---

## Debug 與 Validation 子目錄

### debug/ (5 個診斷工具)
用於訓練問題診斷，詳見 `docs/TROUBLESHOOTING.md`

### validation/ (7 個物理驗證)
用於物理場驗證，詳見 `docs/QUICK_START.md`

---

## Archive 目錄

### archive/ (23 個歸檔腳本)

**結構**:
```
archive/
└── debug_2024_11/           舊版診斷腳本（保留作參考）
```

**注意**: Archive 腳本可能與當前代碼不相容,僅供參考。

---

## 腳本命名規範

- `train.py` - 訓練
- `evaluate_*.py` - 評估相關
- `visualize_*.py` - 視覺化
- `generate_*.py` - 數據/感測點生成
- `validate_*.py` / `verify_*.py` - 驗證
- `calculate_*.py` - 參數計算
- `compare_*.py` - 對比分析
- `monitor_*.py` - 監控
- `fetch_*.py` - 數據獲取

---

**維護**: PINNs-MVP 團隊  
**更新**: 2025-12-13  
**狀態**: ✅ 已分類至子目錄 (30 核心腳本 → 8 類別)
