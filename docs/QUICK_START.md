# 快速開始指南

> **版本**: 2.0.0  
> **最後更新**: 2026-01-03  
> **狀態**: 反映最新標準化配置

---

## ⚠️ 重要提醒

1. **配置已標準化**（2025-12-30）：舊版別名已移除，請使用標準鍵名
2. **訓練損失≠場誤差**（2025-12-19）：必須用後驗指標評估模型
3. **Causal Training 已優化**（2026-01-03）：推薦啟用，性能提升 15x

---

## 環境設定

```bash
# 安裝依賴
pip install -r requirements.txt

# 或使用 conda
conda env create -f environment.yml
```

## 完整流程（Kolmogorov Flow）

### 1. 計算雷諾數參數
```bash
python scripts/calculate_reynolds_parameters.py \
  --target-Re 50 --f0 1.0 --k 4 --solve-nu
# 輸出: ν = 0.0125
```

### 2. 生成 DNS 數據
```bash
python scripts/generate_kolmogorov_dns.py \
  --Re 50 --k_f 4 --nu 0.0125 \
  --T_max 100 --dt 0.05 --resolution 512 \
  --output data/kolmogorov_dns/re50_kf4.h5
```

### 3. 驗證 DNS
```bash
python scripts/validate_dns_physics.py \
  --input data/kolmogorov_dns/re50_kf4.h5
# 檢查: 散度 < 1e-3, NS 殘差 < 0.1, 能量平衡 < 1%
```

### 4. 生成感測點（V7 方法）
```bash
python scripts/generate_sensors_periodic_qr.py \
  --dns-path data/kolmogorov_dns/re50_kf4.h5 \
  --K 100 --oversample-factor 3.0 \
  --output data/kolmogorov_dns/sensors_K100_v7.npz
```

### 5. 生成 RANS 先驗（可選）
```bash
python scripts/generate_kolmogorov_rans.py \
  --Re 50 --k_f 4 --nu 0.0125 \
  --T_avg_start 50.0 --T_avg_end 100.0 \
  --output data/kolmogorov_dns/rans_re50_kf4.h5
```

### 6. 訓練
```bash
# Baseline（no prior）
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100_vanilla.yml

# With LES prior（推薦）
python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml

# 啟用 Causal Training（推薦，2026-01-03）
# 在配置文件中加入：
# losses:
#   causal_weighting: true
#   causal_eps: 1.5
#   causal_n_bins: 20
```

**訓練監控重點**：
- ✅ 監控 **field_l2_error**（主要指標）
- ✅ 監控 velocity_correlation
- ⚠️ 訓練損失僅供參考，不代表實際場誤差

### 7. 評估
```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --config configs/your_exp.yml
```

### 8. 視覺化
```bash
python scripts/visualize_results.py \
  --checkpoint checkpoints/your_exp/best_model.pth \
  --output results/visualizations/
```

## Colab 設定

```python
# 1. 安裝依賴
!pip install torch numpy h5py pyyaml

# 2. 掛載 Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. 克隆專案
!git clone https://github.com/latteine1217/pinns-sparse-flow.git
%cd pinns-sparse-flow

# 4. 執行訓練
!python scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100_vanilla.yml
```

## 檢查清單

**DNS 生成前**
- [ ] 確認 Re/ν/k_f 關係正確
- [ ] 解析度足夠（Re < 100 使用 512×512）
- [ ] 統計時間 T ≥ 100

**訓練前**
- [ ] DNS 通過物理驗證
- [ ] 感測點品質：唯一座標 > 15%, 最大聚集 < 15%
- [ ] 配置啟用 loss normalization

**訓練中**
- [ ] 監控 field_l2_error（主要指標，不是訓練損失！）
- [ ] 檢查物理一致性（∇·u, NS 殘差）
- [ ] 定期保存檢查點

**訓練後**
- [ ] 場 L2 誤差 < 15%（相對 DNS）
- [ ] ∇·u < 1e-3, NS 殘差 < 0.1
- [ ] 能譜符合理論（k^(-5/3), k^(-3)）

---

## 📚 相關文檔

- **配置參考**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **API 參考**: [API_REFERENCE.md](API_REFERENCE.md)
- **故障排除**: [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- **技術文檔**: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)

---

**文檔維護**: PINNs-SparseFlow 專案  
**版本**: 2.0.0  
**最後更新**: 2026-01-05
