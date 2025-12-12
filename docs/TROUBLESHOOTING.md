# 問題排查指南

## 常見問題診斷

### 1. Loss 突然發散

**症狀**：Loss 在訓練中期突然飆升

**診斷**：
```bash
python scripts/debug/diagnose_piratenet_failure.py \
  --checkpoint checkpoints/exp/epoch_100.pth \
  --config configs/exp.yml
```

**解決方案**：
```yaml
# 降低學習率
optimizer:
  lr: 0.0005  # 從 0.001 降低

# 添加梯度裁剪
training:
  gradient_clip: 1.0

# 延遲 L-BFGS 切換
optimizer:
  lbfgs_switch:
    switch_epoch: 900  # 從 800 延遲
```

---

### 2. 感測點過擬合

**症狀**：Data loss 很低，但全場誤差很高

**診斷**：
```bash
python scripts/visualize_qr_sensors.py \
  --input data/sensors.npz \
  --output results/sensor_check/
```

**品質標準**：
- ✅ 唯一 X/Y 座標 > 15%
- ✅ 最大聚集 < 15%
- ✅ 條件數 < 100

**解決方案**：
```bash
# 使用 V7 方法重新生成
python scripts/generate_sensors_periodic_qr.py \
  --dns-path data/kolmogorov_dns/re50_kf4.h5 \
  --K 100 --oversample-factor 3.0
```

或調整配置：
```yaml
# 增加 PDE 權重
losses:
  pde:
    weight: 0.1  # 從 0.01 提高

# 增加採樣點
training:
  n_collocation: 20000  # 從 10000 提高
```

---

### 3. 壓力場不準確

**症狀**：速度場收斂，但壓力場發散

**診斷**：
```bash
python scripts/debug/diagnose_pressure_failure.py \
  --checkpoint checkpoints/exp/latest.pth
```

**解決方案**：
```yaml
# 增加壓力相關權重
losses:
  pde:
    momentum:
      weight: 0.1

# 使用 VS-PINN（變數尺度化）
model:
  type: "VSPINNFourierMLP"
  variable_scaling: true
```

---

### 4. 週期邊界不連續

**症狀**：x=0 與 x=2π 處不連續

**解決方案**：
```yaml
# V7 感測點強化週期處理
sensors:
  seam_weight: 2.0      # 提高到 2.0
  n_wrap_layers: 3      # 提高到 3

# 訓練時添加週期約束
losses:
  periodic_boundary:
    weight: 0.1
    boundaries: ['x']
```

---

## 感測點方法比較

| 方法 | 唯一 X | 唯一 Y | 最大聚集 | 狀態 |
|------|--------|--------|----------|------|
| V5 (單時間) | 4.5% | - | 48% | ❌ 已棄用 |
| V6 (5 時間) | 8.2% | - | 43% | ❌ 已棄用 |
| **V7 (Oversample+Filter)** | **17.8%** | **16.6%** | **10%** | ✅ 推薦 |

**V7 方法**：
```bash
python scripts/generate_sensors_periodic_qr.py \
  --dns-path data/kolmogorov_dns/re50_kf4.h5 \
  --K 100 \
  --oversample-factor 3.0 \
  --seam-weight 1.0 \
  --n-wrap-layers 2
```

**核心改進**：
1. Oversample 3× (請求 300 選 100)
2. 貪婪最小距離過濾
3. 多時間快照 (101 snapshots)

---

## DNS 驗證標準

```bash
python scripts/validate_dns_physics.py --input data/dns.h5
```

**通過標準**：
- ✅ 散度 (∇·u): < 1e-3
- ✅ NS 殘差: < 0.1
- ✅ 能量平衡: < 1%
- ✅ 雷諾數誤差: < 5%

---

## 訓練監控

```bash
# 實時監控
python scripts/monitor_training_speed.py \
  --log-file log/your_exp/training.log

# TensorBoard
tensorboard --logdir log/your_exp/tensorboard/
```

**關鍵指標**：
- Total Loss: 應穩定下降
- Data Loss: < 1e-3
- PDE Loss: < 1e-2
- PDE Ratio: > 30%

---

## 週期邊界處理

**V7 感測點參數**：
```yaml
seam_weight: 1.0       # 邊界權重
n_wrap_layers: 2       # 包裹層數
oversample_factor: 3.0 # 過採樣倍數
```

**原理**：
1. 將週期邊界附近點複製到對側
2. 增加邊界點權重
3. QR 分解時考慮週期連續性

---

## 診斷腳本速查

```bash
# 訓練失敗總診斷
python scripts/debug/diagnose_piratenet_failure.py

# NS 方程檢查
python scripts/debug/diagnose_ns_equations.py

# 邊界條件檢查
python scripts/debug/diagnose_boundary_conditions.py

# 感測點過擬合
python scripts/debug/diagnose_sensor_overfitting.py

# 梯度計算驗證
python scripts/debug/debug_gradient_computation.py
```
