# 🚀 Leith Prior 快速使用指南

本指南說明如何使用更新後的 Leith 湍流模型先驗進行 Kolmogorov Flow 訓練。

---

## 📋 前置檢查

### 1. 驗證 Leith 數據文件

運行驗證腳本檢查數據完整性：

```bash
python scripts/validation/verify_leith_data.py
```

**預期輸出**：
```
✅ Leith 場數據:   通過
✅ Leith Sensor:   通過
🎉 所有驗證通過！可以開始訓練。
```

如果驗證失敗，請參考 `LEITH_MIGRATION_SUMMARY.md` 中的數據生成步驟。

---

## 🎯 訓練流程

### 方案 A：使用 Leith Prior（推薦）

**配置文件**：`configs/kolmogorov_re50_kf4_K100.yml`

**特點**：
- ✅ Leith 湍流模型引導（適合 2D 湍流）
- ✅ 3-Stage Curriculum Learning
- ✅ Step LR Scheduler (gamma=0.5, step=2000)

**訓練命令**：
```bash
# 完整訓練 (10000 epochs, ~8-12 小時 on A100)
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100.yml \
  --device cuda

# 快速驗證 (1000 epochs, ~1 小時)
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100.yml \
  --device cuda \
  --override training.epochs=1000
```

**輸出位置**：
```
checkpoints/kolmogorov_re50_kf4_K100_leith_prior/
results/kolmogorov_re50_kf4_K100_leith_prior/
```

---

### 方案 B：Vanilla Baseline（對照組）

**配置文件**：`configs/kolmogorov_re50_kf4_K100_vanilla.yml`

**特點**：
- ❌ 無湍流先驗
- ❌ 無 Curriculum
- ✅ Exponential LR Scheduler (gamma=0.7943, step=500)

**訓練命令**：
```bash
python scripts/train/train.py \
  --cfg configs/kolmogorov_re50_kf4_K100_vanilla.yml \
  --device cuda
```

**輸出位置**：
```
checkpoints/kolmogorov_re50_kf4_K100_vanilla/
results/kolmogorov_re50_kf4_K100_vanilla/
```

---

## 📊 評估與對比

### 單模型評估

```bash
# 評估 Leith Prior 版本
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/kolmogorov_re50_kf4_K100_leith_prior/best_model.pth \
  --config configs/kolmogorov_re50_kf4_K100.yml \
  --output results/evaluation_leith_prior/

# 評估 Vanilla 版本
python scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint checkpoints/kolmogorov_re50_kf4_K100_vanilla/best_model.pth \
  --config configs/kolmogorov_re50_kf4_K100_vanilla.yml \
  --output results/evaluation_vanilla/
```

### 對比實驗

```bash
# 生成對比報告
python scripts/compare/compare_experiments.py \
  --exp1 kolmogorov_re50_kf4_K100_leith_prior \
  --exp2 kolmogorov_re50_kf4_K100_vanilla \
  --metrics relative_l2 pressure_gradient divergence \
  --output results/comparison_leith_vs_vanilla/
```

---

## 📈 視覺化

### 場重建對比

```bash
python scripts/visualize/visualize_results.py \
  --checkpoint checkpoints/kolmogorov_re50_kf4_K100_leith_prior/best_model.pth \
  --reference data/kolmogorov_dns/dns_re50_t100.h5 \
  --output results/evaluation_leith_prior/visualizations/
```

### Leith 場與 Sensor 視覺化

```bash
# 視覺化 Leith 場
python scripts/visualize/visualize_leith_field.py \
  --leith-file data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5 \
  --output results/leith_visualization/

# 視覺化 Sensor 分佈
python scripts/visualize/visualize_qr_sensors.py \
  --sensor-file data/lowfi/kolmogorov_rans/sensors_K100_leith.npz \
  --background-field data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5 \
  --output results/sensor_visualization/
```

---

## 📓 使用 Jupyter Notebook

### Google Colab（推薦）

1. 上傳專案至 Google Drive：`/MyDrive/pinns-mvp/`
2. 開啟 Notebook：`PINNs_MVP_Kolmogorov_Guide.ipynb`
3. 選擇 GPU Runtime（T4/V100/A100）
4. 按順序執行 cells

**注意事項**：
- ✅ Notebook 已更新為 Leith Prior 版本（v4.3）
- ✅ 所有數據路徑已指向 Leith 文件
- ⚠️ 確保 Google Drive 中存在 Leith 數據文件

### 本地 Jupyter

```bash
# 啟動 Jupyter
jupyter notebook PINNs_MVP_Kolmogorov_Guide.ipynb

# 或使用 JupyterLab
jupyter lab PINNs_MVP_Kolmogorov_Guide.ipynb
```

---

## 🔧 常見問題

### Q1: 驗證腳本報告文件不存在

**A**: 運行數據生成腳本：
```bash
# 如果專案包含 Leith 生成腳本
python scripts/generate/generate_leith_field.py --Re 50 --kf 4

# 或者從現有 RANS 轉換（如果可行）
python scripts/tools/convert_rans_to_leith.py \
  --input data/lowfi/kolmogorov_rans/rans_re50_kf4.h5 \
  --output data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5
```

### Q2: 訓練時報錯找不到 Leith 文件

**A**: 檢查配置檔路徑是否正確：
```bash
# 查看配置
grep "data_path" configs/kolmogorov_re50_kf4_K100.yml

# 應顯示：
# data_path: ./data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5
```

### Q3: Leith Prior 與 RANS Prior 的主要差異？

**A**: 
- **物理模型**：Leith 專為 2D 湍流設計，無 k/ε 變量
- **初始權重**：Leith 使用 2.0（vs RANS 10.0），避免過約束
- **座標格式**：Leith 使用 1D 座標 arrays（vs RANS 2D meshgrid）
- **適用性**：Leith 更適合 Kolmogorov Flow 的 2D 特性

詳見：`LEITH_MIGRATION_SUMMARY.md`

### Q4: 如何切換回 RANS Prior？

**A**: 
```bash
# 恢復到舊版本配置（如有備份）
git checkout <commit-hash> configs/kolmogorov_re50_kf4_K100.yml

# 或手動修改配置檔：
# lowfi_prior:
#   data_path: ./data/lowfi/kolmogorov_rans/rans_re50_kf4.h5  # 舊 RANS
#   data_type: rans
#   consistency_weight: 10.0
```

---

## 📚 相關文檔

- **完整遷移報告**：[LEITH_MIGRATION_SUMMARY.md](LEITH_MIGRATION_SUMMARY.md)
- **技術文檔**：[docs/TECHNICAL_DOCUMENTATION.md](docs/TECHNICAL_DOCUMENTATION.md)
- **實驗對比計畫**：[docs/EXPERIMENT_COMPARISON_PLAN.md](docs/EXPERIMENT_COMPARISON_PLAN.md)
- **配置模板**：[configs/templates/README.md](configs/templates/README.md)

---

## 🎉 預期結果

使用 Leith Prior 訓練後，預期達成：

### 場重建誤差
- **u 速度**：相對 L2 誤差 < 15%
- **v 速度**：相對 L2 誤差 < 15%
- **壓力 p**：相對 L2 誤差 < 20%

### 物理守恆
- **連續性**：散度誤差 < 1e-3
- **週期性**：邊界殘差 < 1e-4

### 相比 Vanilla 改進
- **壓力梯度重建**：改善 ≥ 30%
- **收斂速度**：加快 ≥ 30%
- **感測點需求**：K=100（穩定收斂）

---

**最後更新**：2025-12-18  
**版本**：v4.3 (Leith Prior Edition)
