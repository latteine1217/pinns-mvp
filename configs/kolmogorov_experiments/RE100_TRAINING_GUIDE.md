# Kolmogorov Flow Re=100 高解析度訓練指南

**建立日期**: 2025-11-21  
**目標**: 使用正確黏滯度 (ν=0.0125) 訓練 Re=100 的 Kolmogorov Flow（512×512 網格）

---

## ✅ 配置驗證

### 雷諾數計算
```bash
# 驗證黏滯度設定
python scripts/validation/validate_kolmogorov_reynolds.py --compute-nu --Re 100 --F 1.0 --k 4
```

**輸出**:
```
所需黏滯度：ν = 0.012500
層流速度：U = 5.0000
驗證：Re = 100.00 (✅ 正確)
```

### 配置文件驗證
```bash
python scripts/validation/validate_kolmogorov_reynolds.py --validate
```

**結果**:
| 配置文件 | F | k | ν | Re (實際) | Re (宣稱) | 狀態 |
|---------|---|---|---|----------|----------|------|
| `kolmogorov_2d_re100_highres.yml` | 1.0 | 4 | **0.0125** | **100.00** | 100 | ✅ |

---

## 📊 當前進度

### 1️⃣ DNS 資料生成（進行中）
```bash
# 監控進度
tail -f log/dns_generation_re100.log

# 預估完成時間：10-20 分鐘
# 當前進度：~3% (Step 600/20000)
```

**參數**:
- 網格：512×512
- 雷諾數：Re=100
- 黏滯度：ν=0.0125
- 強迫：F=1.0, k=4
- 模擬時間：T=20.0

**輸出文件**: `data/kolmogorov_dns_re100_512x512.h5`

---

## 🚀 訓練執行流程

### 自動化腳本（推薦）
```bash
# 等待 DNS 生成完成後執行
./scripts/train_kolmogorov_re100_highres.sh
```

**腳本功能**:
1. ✅ 檢查 DNS 資料完整性
2. ✅ 生成 QR-Pivot 感測點 (K=100)
3. ✅ 驗證配置文件雷諾數
4. ✅ 建立輸出目錄
5. ✅ 啟動訓練

### 手動執行步驟

#### 步驟 1：生成 QR-Pivot 感測點
```bash
python scripts/generate_2d_slice_qr_sensors_fixed_v2.py \
    --dns-file data/kolmogorov_dns_re100_512x512.h5 \
    --K 100 \
    --output data/kolmogorov_qr_sensors_re100_K100.npz \
    --snapshot-key "t_10.0" \
    --method "qr_pivot"
```

#### 步驟 2：啟動訓練（前台）
```bash
python scripts/train.py --cfg configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml
```

#### 步驟 3：啟動訓練（背景）
```bash
nohup python scripts/train.py \
    --cfg configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml \
    > log/kolmogorov_2d_re100_highres/training.log 2>&1 &

# 監控訓練
tail -f log/kolmogorov_2d_re100_highres/training.log
```

---

## 📋 配置摘要

### 物理參數
```yaml
physics:
  forcing:
    amplitude: 1.0              # F
    wavenumber: 4               # k
  nu: 0.0125                    # ν (Re=100)
  rho: 1.0
```

### 網路架構
```yaml
model:
  type: "fourier_mlp"
  width: 512                    # 提升至 512（匹配高解析度）
  depth: 8                      # 8 層（增強容量）
  fourier_features:
    n_features: 256             # 256 個 Fourier 特徵
    sigma: 15.0                 # 適應高 Re
```

### 訓練設定
```yaml
training:
  n_epochs: 500                 # 500 輪
  n_collocation: 20000          # 20000 配置點
  optimizer:
    phase1:
      type: "adam"
      lr: 5.0e-4                # 降低學習率（更穩定）
      epochs: 400
    phase2:
      type: "lbfgs"
      lr: 1.0
      epochs: 100
```

### 感測點配置
```yaml
sensors:
  K: 100                        # 100 個感測點
  selection_method: "qr_pivot"
```

---

## 📊 預期結果

### 收斂指標
- **相對 L2 誤差**: ≤ 10-15%
- **散度誤差**: < 1e-4
- **能譜匹配**: RMSE 下降 ≥ 30%（相對低保真）

### 輸出文件
```
checkpoints/kolmogorov_2d_re100_highres/
├── best_model.pth
├── latest.pth
├── epoch_100.pth
├── epoch_200.pth
└── ...

results/kolmogorov_2d_re100_highres/
├── metrics.json
├── predictions.npz
└── visualizations/
    ├── u_field.png
    ├── v_field.png
    ├── p_field.png
    └── vorticity.png
```

---

## 🛠️ 故障排除

### DNS 生成卡住
```bash
# 檢查進程
ps aux | grep generate_kolmogorov_dns

# 如果需要重啟
kill <PID>
python scripts/generate_kolmogorov_dns.py \
    --N 512 --nu 0.0125 --A 1.0 --k_f 4 \
    --T_end 20.0 --dt 0.001 \
    --output data/kolmogorov_dns_re100_512x512.h5
```

### 感測點生成失敗
```bash
# 檢查 DNS 文件
python -c "
import h5py
with h5py.File('data/kolmogorov_dns_re100_512x512.h5', 'r') as f:
    print(list(f.keys()))
    print(list(f['snapshots'].keys()))
"

# 如果缺少 t=10.0 快照，使用其他時間點
python scripts/generate_2d_slice_qr_sensors_fixed_v2.py \
    --dns-file data/kolmogorov_dns_re100_512x512.h5 \
    --K 100 \
    --snapshot-key "t_15.0"  # 改用其他時間
```

### 訓練 OOM（記憶體不足）
```yaml
# 降低批次大小（修改配置文件）
training:
  n_collocation: 10000  # 從 20000 降為 10000
  batch_size: 10000
```

---

## 📐 物理驗證

### 檢查雷諾數
```python
from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D

physics = KolmogorovFlow2D(
    forcing_params={'amplitude': 1.0, 'wavenumber': 4},
    physics_params={'nu': 0.0125}
)

print(f"Re = {physics.compute_reynolds_number():.2f}")
# 輸出：Re = 100.00
```

### 評估物理一致性
```bash
python scripts/validation/physics_validation.py \
    --checkpoint checkpoints/kolmogorov_2d_re100_highres/best_model.pth \
    --config configs/kolmogorov_experiments/kolmogorov_2d_re100_highres.yml
```

---

## 📞 快速參考

| 項目 | 值 |
|------|---|
| **目標雷諾數** | Re = 100 |
| **黏滯度** | ν = 0.0125 |
| **強迫參數** | F = 1.0, k = 4 |
| **層流速度** | U = 5.0 |
| **網格解析度** | 512×512 |
| **感測點數** | K = 100 |
| **訓練輪數** | 500 epochs |
| **預估時間** | 1-2 小時 (GPU) |

---

**下一步**：等待 DNS 生成完成（約 10-15 分鐘），然後執行：
```bash
./scripts/train_kolmogorov_re100_highres.sh
```
