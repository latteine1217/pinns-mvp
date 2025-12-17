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

---

## 新增錯誤處理（v2025-12-17）

### 4. 座標維度不匹配警告

**症狀**：訓練時出現以下警告之一：

```
⚠️ VS-PINN (3D) 模式但 z 座標為常數 (z=4.7100)
   可能原因：
   1. NPZ 檔案來自 2D 切片但誤用 3D 模式
   2. z_default 設定錯誤
   解決方案：
   1. 檢查 sensor NPZ 檔案是否為 2D 資料
   2. 若為 2D 問題，在 config 設定 model.enable_vs_pinn=false
   3. 若為 3D 切片，確認 z_default=4.71 是否正確
```

或

```
⚠️ 2D 物理模式但 z 座標非零（範圍: [3.1400, 4.7100]）
   可能原因：
   1. NPZ 檔案來自 3D 資料但誤用 2D 模式
   2. 應使用 VS-PINN 但未啟用
   解決方案：
   1. 檢查是否應啟用 VS-PINN：config 設定 model.enable_vs_pinn=true
   2. 若確實為 2D 問題，重新生成 sensor NPZ 確保 z=0
   3. z 座標將被忽略並強制為 0
```

**診斷步驟**：

1. **檢查 NPZ 檔案維度**：
   ```python
   import numpy as np
   data = np.load('sensors.npz', allow_pickle=True)
   coords = data['coords'] if 'coords' in data else data['coords_2d']
   print(f"Shape: {coords.shape}")
   print(f"z range: [{coords[:, 2].min():.4f}, {coords[:, 2].max():.4f}]")
   print(f"z std: {coords[:, 2].std():.6f}")  # < 1e-6 視為常數
   ```

2. **檢查配置一致性**：
   ```yaml
   # 確認物理模式
   model:
     enable_vs_pinn: true  # 3D 模式
   
   # 或
   model:
     enable_vs_pinn: false  # 2D 模式
   ```

3. **修正方案**：
   - **2D 資料 + 3D 模式** → 設定 `enable_vs_pinn: false`
   - **3D 資料 + 2D 模式** → 設定 `enable_vs_pinn: true`
   - **2D 切片 + 3D 模式** → 確認 `z_default` 正確

---

### 5. 壓力場資料缺失錯誤

**症狀**：訓練初始化時拋出錯誤：

```
ValueError: ❌ 壓力場資料缺失！

當前配置：
  physics.pressure_driven = True
  training.enforce_pressure_data = True

檢測到問題：
  sensor NPZ 檔案中缺少 'p' (壓力) 欄位

對於壓力驅動流（Channel Flow），壓力場是必需的輸入資料。

解決方案（三選一）：
  1. 重新生成 sensor data 並確保包含壓力場
     → python scripts/generate/sensors/xxx.py --include-pressure
  
  2. 檢查 NPZ 檔案是否正確載入
     → data['sensor_data'] 應包含 'u', 'v', 'w', 'p' 鍵
  
  3. 若確實無壓力資料且理解風險，設定 config:
     training:
       enforce_pressure_data: false
```

**診斷步驟**：

1. **檢查 NPZ 檔案內容**：
   ```python
   import numpy as np
   data = np.load('sensors.npz', allow_pickle=True)
   print("Available keys:", data.files)
   
   # 檢查 sensor_data 結構
   if 'sensor_data' in data:
       sensor_data = data['sensor_data'].item()
       print("Sensor fields:", sensor_data.keys())  # 應包含 'p'
   ```

2. **確認物理類型**：
   ```yaml
   physics:
     type: "channel_flow"  # 壓力驅動
     pressure_driven: true  # 明確聲明
   ```

3. **修正方案**（優先級排序）：

   **方案 1（推薦）**：重新生成包含壓力場的 sensor data
   ```bash
   python scripts/generate/sensors/generate_channel_sensors_qr.py \
     --input data/channel_flow_re1000.h5 \
     --K 100 \
     --include-pressure  # 確保包含壓力
   ```

   **方案 2**：檢查資料載入邏輯
   - 確認 NPZ 格式正確（`sensor_data['p']` 存在）
   - 檢查 `channel_flow_loader.py` 是否正確提取壓力欄位

   **方案 3（不推薦）**：允許無壓力資料
   ```yaml
   training:
     enforce_pressure_data: false  # 壓力場將初始化為零
   ```
   ⚠️ **風險**：初期訓練完全依賴 PDE residual，收斂慢且可能陷入局部最優

**流體類型對照表**：

| 流體類型 | `pressure_driven` | 壓力資料要求 | 範例 |
|---------|-------------------|--------------|------|
| Channel Flow | `true` | ✅ 必需 | Re_tau=1000 通道流 |
| Pipe Flow | `true` | ✅ 必需 | 圓管流 |
| Kolmogorov Flow | `false` | ⭕ 可選 | 體積力驅動 |
| Lid-Driven Cavity | `false` | ⭕ 可選 | 速度驅動 |

---

### 6. 標準化統計量無效錯誤

**症狀**：Trainer 初始化時拋出錯誤：

```
RuntimeError: ❌ DataNormalizer 統計量無效！
   可能原因:
   1. 標準化類型為 'training_data_norm' 但未提供 training_data
   2. training_data 中某些變量為常數（std ≈ 0）
   3. config['normalization']['params'] 中的統計量包含 NaN/Inf
   解決方案:
   1. 檢查 config['normalization']['params'] 是否正確設定
   2. 確認 training_data 傳遞給 Trainer.__init__()
   3. 檢查 sensor data 的數值範圍是否合理
   4. 若某變量確實為常數，考慮使用 normalization.type='none'
```

**診斷步驟**：

1. **檢查標準化配置**：
   ```yaml
   normalization:
     type: 'training_data_norm'  # 或 'manual', 'friction_velocity'
     params:  # 僅 type='manual' 時需要
       u_mean: 0.5
       u_std: 1.2   # ⚠️ 必須 > 1e-12
       v_mean: 0.0
       v_std: 0.8
       # ...
   ```

2. **檢查 sensor data 統計量**：
   ```python
   import numpy as np
   data = np.load('sensors.npz', allow_pickle=True)
   sensor_data = data['sensor_data'].item()
   
   for var in ['u', 'v', 'w', 'p']:
       if var in sensor_data:
           values = sensor_data[var]
           print(f"{var}: mean={values.mean():.6f}, std={values.std():.6f}")
           # 檢查是否為常數
           if values.std() < 1e-12:
               print(f"  ⚠️ {var} 為常數！")
           # 檢查是否有 NaN/Inf
           if not np.isfinite(values).all():
               print(f"  ⚠️ {var} 包含 NaN/Inf！")
   ```

3. **修正方案**：

   **問題 1: std 過小（常數變量）**
   ```yaml
   # 方案 A: 使用不需標準化的方法
   normalization:
     type: 'none'
   
   # 方案 B: 排除常數變量
   normalization:
     type: 'training_data_norm'
     variable_order: ['u', 'v', 'p']  # 移除常數的 'w'
   ```

   **問題 2: 缺少統計量（type='manual' 時）**
   ```yaml
   normalization:
     type: 'manual'
     params:
       u_mean: 0.5   # ✅ 必須提供
       u_std: 1.2    # ✅ 必須提供且 > 1e-12
       v_mean: 0.0
       v_std: 0.8
       p_mean: 0.0
       p_std: 2.5
       # 若有 w 分量也需提供
   ```

   **問題 3: 資料包含 NaN/Inf**
   ```python
   # 清理資料
   import numpy as np
   data = np.load('sensors.npz', allow_pickle=True)
   sensor_data = data['sensor_data'].item()
   
   for var in sensor_data:
       values = sensor_data[var]
       if not np.isfinite(values).all():
           print(f"Cleaning {var}...")
           sensor_data[var] = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
   
   # 重新儲存
   np.savez('sensors_cleaned.npz', sensor_data=sensor_data, ...)
   ```

**標準化閾值說明**：

| 統計量 | 有效範圍 | 失敗範圍 | 原因 |
|--------|----------|----------|------|
| `mean` | 任何有限值 | `NaN`, `±Inf` | 數值錯誤 |
| `std` | `≥ 1e-12` | `< 1e-12`, `NaN`, `±Inf` | 常數或數值不穩定 |

**檢查清單**：
- [ ] 確認 `normalization.type` 設定正確
- [ ] 若 `type='manual'`，確認所有變量的 `mean`/`std` 已提供
- [ ] 確認 sensor data 不包含 NaN/Inf
- [ ] 確認沒有常數變量（std < 1e-12）
- [ ] 確認 `training_data` 正確傳遞給 `Trainer.__init__()`

---
