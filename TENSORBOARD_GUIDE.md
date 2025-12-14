# TensorBoard 使用指南

## 📊 啟動 TensorBoard

```bash
# 方法 1: 啟動特定實驗
tensorboard --logdir=runs/kolmogorov_re50_kf4_K100_rans_prior --port=6006

# 方法 2: 啟動所有實驗（比較模式）
tensorboard --logdir=runs/ --port=6006

# 訪問地址
http://localhost:6006
```

## 📈 可視化內容總覽

### 1. **Loss/**（主要損失）
| 指標 | 說明 | 目標值 |
|------|------|--------|
| `Loss/total` | 總損失 | 持續下降 |
| `Loss/data` | 感測器數據擬合損失 | < 0.05 |
| `Loss/pde` | 物理方程殘差損失 | < 0.01 |
| `Loss/boundary` | 邊界條件損失 | < 0.001 |

### 2. **Loss/PDE/**（物理方程子項）
| 指標 | 說明 | 診斷用途 |
|------|------|----------|
| `Loss/PDE/momentum_x` | x方向動量守恆 | 檢查 Navier-Stokes 方程收斂 |
| `Loss/PDE/momentum_y` | y方向動量守恆 | 檢查 Navier-Stokes 方程收斂 |
| `Loss/PDE/momentum_z` | z方向動量守恆（3D） | 檢查 Navier-Stokes 方程收斂 |
| `Loss/PDE/continuity` | 連續性方程（∇·u=0） | **關鍵**：質量守恆，應 < 0.001 |
| `Loss/PDE/divergence` | 散度損失（同上） | 同 continuity |

### 3. **Loss/Data/**（各變量數據擬合）
| 指標 | 說明 | 期望趨勢 |
|------|------|----------|
| `Loss/Data/u` | u速度擬合誤差 | 快速下降 |
| `Loss/Data/v` | v速度擬合誤差 | 快速下降 |
| `Loss/Data/w` | w速度擬合誤差（3D） | 快速下降 |
| `Loss/Data/pressure` | 壓力擬合誤差 | 相對較高，正常 |

### 4. **Loss/Weighted/**（權重平衡分析）
| 指標 | 說明 | 用途 |
|------|------|------|
| `Loss/Weighted/data` | data_weight × data_loss | 分析損失權重是否平衡 |
| `Loss/Weighted/pde` | pde_weight × pde_loss | 應與 data 同數量級 |
| `Loss/Weighted/continuity` | continuity_weight × continuity_loss | **關鍵**：不應被忽略 |
| `Loss/Weighted/boundary` | bc_weight × bc_loss | 邊界約束強度 |

### 5. **Loss/Prior/**（RANS 先驗損失）
| 指標 | 說明 | 目標 |
|------|------|------|
| `Loss/Prior/total` | RANS 先驗總損失 | 逐漸下降 |
| `Loss/Prior/u` | u 速度先驗一致性 | 穩定在低值 |
| `Loss/Prior/v` | v 速度先驗一致性 | 穩定在低值 |
| `Loss/Prior/p` | 壓力先驗一致性 | 通常很小 |

### 6. **Loss/BC/**（邊界條件）
| 指標 | 說明 | 適用場景 |
|------|------|----------|
| `Loss/BC/periodic_x` | x方向週期邊界 | Kolmogorov flow |
| `Loss/BC/periodic_y` | y方向週期邊界 | Kolmogorov flow |
| `Loss/BC/inlet` | 入口邊界 | Channel flow |
| `Loss/BC/outlet` | 出口邊界 | Channel flow |
| `Loss/BC/wall` | 壁面無滑移 | Channel flow |

### 7. **Loss/Regularization/**（正則化）
| 指標 | 說明 |
|------|------|
| `Loss/Regularization/total` | 總正則化損失 |
| `Loss/Regularization/l2` | L2 權重衰減 |
| `Loss/Regularization/gradient` | 梯度懲罰 |

### 8. **Training/**（訓練超參數）
| 指標 | 說明 | 監控重點 |
|------|------|----------|
| `Training/learning_rate` | 學習率 | 確認指數衰減正常 |

### 9. **Validation/**（驗證指標）
| 指標 | 說明 |
|------|------|
| `Validation/relative_l2` | 驗證集相對L2誤差 |
| `Validation/mse` | 驗證集均方誤差 |

### 10. **Gradients/**（梯度診斷）
| 指標 | 說明 | 診斷用途 |
|------|------|----------|
| `Gradients/norm/{layer}` | 各層梯度範數 | 檢查梯度消失/爆炸 |
| `Gradients/hist/{layer}` | 梯度分佈直方圖 | 每 20 epochs 記錄 |
| `Weights/{layer}` | 權重分佈直方圖 | 檢查權重分佈健康度 |

---

## 🔍 常見問題診斷

### ❌ 問題 1：Training Loss 下降但物理約束失敗
**症狀**：
- `Loss/total` 下降
- `Loss/PDE/continuity` 不下降或上升
- `Loss/Weighted/continuity` 遠小於 `Loss/Weighted/data`

**原因**：損失權重失衡，模型忽略物理方程

**解決方案**：
```yaml
# 修改 config.yml
losses:
  data_weight: 10.0
  continuity_weight: 10.0  # ← 提升至與 data_weight 同級
  momentum_x_weight: 5.0   # ← 提升
  momentum_y_weight: 5.0   # ← 提升
```

### ❌ 問題 2：Learning Rate 太高或太低
**症狀**：
- 太高：Loss 震盪不收斂
- 太低：Loss 下降極慢

**診斷方法**：查看 `Training/learning_rate` 曲線

**解決方案**：
```yaml
training:
  optimizer:
    lr: 1.0e-3  # 調整初始學習率
    scheduler:
      gamma: 0.9998  # 調整衰減率
```

### ❌ 問題 3：梯度消失/爆炸
**症狀**：
- `Gradients/norm/{layer}` 趨近 0（消失）
- `Gradients/norm/{layer}` > 100（爆炸）

**解決方案**：
```yaml
training:
  gradient_clip: 1.0  # 啟用梯度裁剪

model:
  initialization:
    type: siren  # 使用 SIREN 初始化
```

---

## 📊 記錄頻率

| 類別 | 頻率 | 配置參數 |
|------|------|----------|
| 標量損失 | 每 `log_freq` epochs | `training.log_interval: 10` |
| 梯度直方圖 | 每 `log_freq × 2` epochs | 硬編碼 |
| 權重直方圖 | 每 `log_freq × 2` epochs | 硬編碼 |

## 🛠️ 自定義記錄

如需新增自定義指標，修改 `pinnx/train/trainer.py` 中的 TensorBoard 記錄區塊：

```python
# 在 train() 方法的 TensorBoard 記錄部分新增
if self.writer is not None and epoch % log_freq == 0:
    # 新增自定義指標
    self.writer.add_scalar('Custom/my_metric', my_value, epoch)
```

---

## 📌 最佳實踐

1. **訓練前檢查**：啟動訓練後立即開啟 TensorBoard，確認指標正常記錄
2. **多實驗比較**：使用 `--logdir=runs/` 模式比較不同配置效果
3. **關鍵指標**：重點監控 `Loss/PDE/continuity` 和 `Loss/Weighted/*` 的平衡
4. **梯度健康**：定期查看 `Gradients/norm` 確保無異常
5. **學習率調整**：根據 `Loss/total` 收斂速度調整 `scheduler.gamma`

---

## 🔗 相關文檔

- **配置參考**：`docs/CONFIG_REFERENCE.md`
- **訓練監控**：`TENSORBOARD_MONITORING.md`（舊版，本文檔為增強版）
- **評估指標**：`docs/API_REFERENCE.md`

---

**更新日期**：2025-12-13  
**版本**：v2.0（完整記錄增強版）
