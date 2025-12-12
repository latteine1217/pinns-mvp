# 快速開始指南

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
# 快速測試（100 epochs）
python scripts/train.py --cfg configs/quick_test_rans_prior.yml

# 完整訓練（1000 epochs）
python scripts/train.py --cfg configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml
```

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
!git clone https://github.com/your-repo/pinns-mvp.git
%cd pinns-mvp

# 4. 執行訓練
!python scripts/train.py --cfg configs/quick_test_rans_prior.yml --device cuda
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
- [ ] 監控 loss 趨勢（無劇烈震盪）
- [ ] PDE ratio > 30%
- [ ] 定期保存檢查點

**訓練後**
- [ ] L2 誤差 < 15%
- [ ] ∇·u < 1e-3, NS 殘差 < 0.1
- [ ] 能譜符合理論（k^(-5/3), k^(-3)）
