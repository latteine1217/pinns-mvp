# 故障排除指南

> **版本**: 2.0.0  
> **最後更新**: 2026-01-03  
> **狀態**: 包含最新已知問題與解決方案

---

## 🎯 快速診斷

| 症狀 | 可能原因 | 跳至章節 |
|------|----------|---------|
| 訓練損失降但場誤差大 | **Normalization Mismatch** | [#0](#0-訓練損失與場誤差不匹配-critical) |
| 訓練出現 NaN | 學習率過大 / 權重失衡 | [#1](#1-loss-突然發散) |
| 感測器過擬合 | 感測點品質差 | [#2](#2-感測點過擬合) |
| 壓力場發散 | Prior 缺失 / 權重不當 | [#3](#3-壓力場不準確) |
| 配置驗證失敗 | 使用舊版別名 | [#4](#4-配置錯誤) |

---

## 0. 訓練損失與場誤差不匹配 🚨 CRITICAL

### 問題描述（2025-12-19 發現）

**現象**：
- 訓練損失持續下降，但實際場重建誤差反而增加
- 不同實驗間訓練損失無法比較
- 模型預測方差遠小於 DNS ground truth

### 根本原因

使用 `normalization.type = training_data_norm` 時：
- 標準化統計從 **RANS prior** 計算（平滑場）
- 但評估用 **DNS ground truth**（湍流場）
- 兩者方差差異 **10-60 倍**！

```python
# RANS Prior 統計（用於訓練標準化）
u_mean = 1.076, u_std = 0.115  # 小方差
v_mean = 0.017, v_std = 0.039
p_mean = -0.003, p_std = 0.008

# DNS Ground Truth（用於評估）
u_mean = 0.000, u_std = 1.140  # 大方差（10x）
v_mean = 0.000, v_std = 0.212  # (5x)
p_mean = 0.000, p_std = 0.445  # (60x)
```

### 實驗證據

| 實驗 | 訓練損失 | 場 L2 誤差 | 預測 σ / DNS σ |
|------|---------|-----------|---------------|
| A3   | 1.764 (最低) | 104% (最差) | 1-13% |
| B1.4 | 3.368 (較高) | 128% (稍好) | 6% |
| A1/A2| 5.923 (最高) | 未知 | 未知 |

**結論**：訓練損失改善 ≠ 場重建品質改善

### 解決方案

1. **不依賴訓練損失評估模型**
   ```python
   # ❌ 錯誤做法
   if train_loss < best_loss:
       save_model()
   
   # ✅ 正確做法
   if field_l2_error < best_error:
       save_model()
   ```

2. **使用後驗指標**
   ```yaml
   # 在 WandB 或 TensorBoard 中追蹤
   metrics:
     - field_l2_error        # 必須
     - field_rmse            # 必須
     - velocity_correlation  # 推薦
     - pressure_gradient_error  # 推薦
   ```

3. **考慮替代標準化方案**（待驗證）
   ```yaml
   normalization:
     type: "zscore"
     # 從 DNS 計算統計量（需要修改程式碼）
   ```

詳見：`context/decisions/decisions_log.md` (2025-12-19 CRITICAL FINDING)

---

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
training:
  optimizer:
    lr: 0.0005  # 從 0.001 降低

# 添加梯度裁剪
training:
  gradient_clip: 1.0
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
  momentum_x_weight: 1.0  # 從 0.1 提高
  continuity_weight: 1.0

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
  continuity_weight: 2.0  # 提高連續性約束

# 啟用低保真先驗
data:
  lowfi_prior:
    use: true
    path: "./data/rans_prior.h5"

losses:
  prior_weight: 0.1  # 提供壓力場先驗
```

---

### 4. 配置錯誤

**症狀**：配置驗證失敗或訓練無法啟動

**常見錯誤**（2025-12-30 移除向後相容後）：

```yaml
# ❌ 錯誤：使用舊版別名
loss:  # 應該是 losses (複數)
  boundary_weight: 1.0  # 應該是 wall_constraint_weight

model:
  fourier_features:
    enabled: true  # 應該是 type: "standard"

# ✅ 正確：標準配置
losses:
  wall_constraint_weight: 1.0

model:
  fourier_features:
    type: "standard"
```

**解決方案**：
```bash
# 驗證配置
python scripts/tools/validate_config_keys.py --config your_config.yml

# 自動修正（如果有工具）
python scripts/tools/migrate_config_to_v2.py --input old.yml --output new.yml
```

---

## 性能優化

### Causal Training 加速（2026-01-03）

如果使用 Causal Training，確保使用預計算優化版本：

```yaml
losses:
  causal_weighting: true
  causal_eps: 1.5       # 標準值
  causal_n_bins: 20     # 標準值
```

**性能提升**：~15x 加速（從 15-20ms → 1.12ms per iteration）

詳見：`context/session_logs/session_summary_2026-01-03_stage1.md`

---

## 📚 相關資源

- **配置指南**: [CONFIG_GUIDE.md](CONFIG_GUIDE.md)
- **技術文檔**: [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md)
- **決策日誌**: `context/decisions/decisions_log.md`
- **會話記錄**: `context/session_logs/`

---

**文檔維護**: PINNs-MVP 團隊  
**版本**: 2.0.0  
**最後更新**: 2026-01-03
