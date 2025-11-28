# Kolmogorov Flow 評估指南

## 📋 問題診斷與解決方案

### 🔴 常見錯誤：輸入維度不匹配

**錯誤訊息**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (5000x2 and 3x64)
```

**根本原因**:
- **模型期望**: 3D 輸入 `(t, x, y)` - 時空座標
- **腳本提供**: 2D 輸入 `(x, y)` - 僅空間座標
- **Fourier 矩陣**: `B.shape = [3, 64]` (3D → 64 modes)

**配置驗證**:
```yaml
# configs/kolmogorov_re100_kf4_K100.yml
model:
  in_dim: 3       # ✅ 3D 輸入: (t, x, y)
  out_dim: 3      # ✅ 3D 輸出: (u, v, p)
```

---

## ✅ 解決方案：使用修正版評估腳本

### 1️⃣ 腳本位置
```
scripts/evaluate_kolmogorov_quick.py
```

### 2️⃣ 核心修正

**✅ 正確的網格生成**（第 157-171 行）:
```python
def create_evaluation_grid(x, y, t, n_points=256):
    """創建評估網格 - 注意順序：(t, x, y)"""
    coords = []
    for ti in t:
        for yi in y_sub:
            for xi in x_sub:
                coords.append([ti, xi, yi])  # ✅ 3D: (t, x, y)
    
    grid_coords = np.array(coords, dtype=np.float32)
    # grid_coords.shape = [N, 3]
    return grid_coords
```

**❌ 錯誤示範**（舊腳本）:
```python
# ❌ 只有 2D 座標
coords = [[xi, yi] for yi in y for xi in x]
# coords.shape = [N, 2] → 導致維度不匹配
```

---

## 🚀 使用方式

### 基本評估（單一時間快照）

```bash
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf4_K100/epoch_4000.pth \
  --config configs/kolmogorov_re100_kf4_K100.yml \
  --output results/evaluation_re100_kf4/ \
  --n-points 256 \
  --time-snapshot 30.0  # 使用 t=30.0 時刻（穩態區域）
```

### 完整評估（時間平均）

```bash
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf4_K100/epoch_4000.pth \
  --config configs/kolmogorov_re100_kf4_K100.yml \
  --output results/evaluation_re100_kf4/ \
  --n-points 256
  # 不指定 --time-snapshot 則使用時間窗內所有快照平均
```

### 高解析度評估

```bash
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf4_K100/epoch_4000.pth \
  --config configs/kolmogorov_re100_kf4_K100.yml \
  --output results/evaluation_re100_kf4_highres/ \
  --n-points 512  # 提高解析度至 512×512
  --time-snapshot 30.0
```

---

## 📊 輸出內容

### 1. 評估指標（`metrics.yaml`）

```yaml
u_l2: 0.1234        # 速度 u 相對 L2 誤差
v_l2: 0.1456        # 速度 v 相對 L2 誤差
p_l2: 0.2012        # 壓力 p 相對 L2 誤差
u_rmse: 0.0234      # 速度 u RMSE
v_rmse: 0.0312      # 速度 v RMSE
p_rmse: 0.0456      # 壓力 p RMSE
mass_conservation: 0.0012  # 質量守恆誤差
```

### 2. 視覺化圖表

- `field_u.png`: u 速度場（參考/預測/誤差三面板）
- `field_v.png`: v 速度場（參考/預測/誤差三面板）
- `field_p.png`: 壓力場（參考/預測/誤差三面板）

**範例圖表結構**:
```
┌─────────────┬─────────────┬─────────────┐
│ Reference   │ Prediction  │ Error       │
│ (DNS)       │ (PINNs)     │ |pred-ref|  │
├─────────────┼─────────────┼─────────────┤
│   u field   │   u field   │  error map  │
│   colorbar  │   colorbar  │  colorbar   │
└─────────────┴─────────────┴─────────────┘
```

---

## 🔍 驗證檢查清單

### ✅ 執行前檢查

1. **檢查點存在**:
   ```bash
   ls -lh checkpoints/kolmogorov_re100_kf4_K100/epoch_4000.pth
   ```

2. **DNS 數據存在**:
   ```bash
   ls -lh data/kolmogorov_dns_re100_512x512_kf4.h5
   ```

3. **配置文件正確**:
   ```bash
   grep "in_dim: 3" configs/kolmogorov_re100_kf4_K100.yml
   ```

### ✅ 執行後驗證

1. **檢查輸出文件**:
   ```bash
   ls -lh results/evaluation_re100_kf4/
   # 應包含:
   # - metrics.yaml
   # - field_u.png
   # - field_v.png
   # - field_p.png
   ```

2. **驗證指標範圍**:
   ```bash
   cat results/evaluation_re100_kf4/metrics.yaml
   # L2 誤差應在 10-20% 範圍（湍流重建目標）
   # RMSE 應為合理物理量級
   ```

3. **檢查視覺化品質**:
   - 預測場應呈現合理的渦結構
   - 誤差場應低於參考場量級
   - 無異常的 NaN/Inf 區域

---

## ⚙️ 參數調整建議

### 解析度權衡

| `--n-points` | 評估點數 | 記憶體需求 | 計算時間 | 適用場景 |
|--------------|---------|----------|---------|---------|
| 64           | 4K      | ~100MB   | <1 min  | 快速驗證 |
| **128**      | **16K** | ~200MB   | 2-3 min | **推薦** |
| 256          | 65K     | ~500MB   | 5-10 min | 標準評估 |
| 512          | 262K    | ~2GB     | 20-30 min | 高解析度 |

### 時間採樣策略

**選項 A: 單一快照**（推薦，快速）
```bash
--time-snapshot 30.0  # 穩態區域中點
```
- ✅ 快速計算
- ✅ 減少記憶體需求
- ⚠️ 可能受瞬時波動影響

**選項 B: 時間平均**（更準確）
```bash
# 不指定 --time-snapshot
```
- ✅ 統計穩定
- ✅ 捕捉平均流場
- ⚠️ 計算時間 × N_snapshots

---

## 🐛 故障排除

### 問題 1: 維度不匹配錯誤

**症狀**:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (Nx2 and 3x64)
```

**解決方案**:
1. 確認使用最新版本的 `evaluate_kolmogorov_quick.py`
2. 檢查腳本第 165 行：
   ```python
   coords.append([ti, xi, yi])  # ✅ 必須是 3D
   ```

### 問題 2: DNS 數據路徑錯誤

**症狀**:
```
FileNotFoundError: DNS 數據檔案不存在
```

**解決方案**:
```bash
# 檢查配置文件中的路徑
grep "data_path" configs/kolmogorov_re100_kf4_K100.yml

# 更新為正確路徑（Colab 範例）
sed -i 's|./data/|/content/drive/MyDrive/pinns-mvp/data/|g' \
  configs/kolmogorov_re100_kf4_K100.yml
```

### 問題 3: 記憶體不足（OOM）

**症狀**:
```
CUDA out of memory
```

**解決方案**:
```bash
# 降低解析度
--n-points 128  # 從 256 降至 128

# 或使用 CPU（較慢但穩定）
--device cpu
```

### 問題 4: 模型輸出維度錯誤

**症狀**:
```
IndexError: index 2 is out of bounds for dimension 1 with size 2
```

**解決方案**:
檢查模型配置是否正確：
```yaml
model:
  out_dim: 3  # ✅ 必須是 3 (u, v, p)
```

---

## 📈 預期結果基準

### Re=100, k_f=4, K=100 訓練目標

| 指標 | 目標值 | 及格標準 | 優秀標準 |
|------|--------|---------|---------|
| `u_l2` | ≤ 15% | < 20% | < 10% |
| `v_l2` | ≤ 15% | < 20% | < 10% |
| `p_l2` | ≤ 20% | < 25% | < 15% |
| `mass_conservation` | ≤ 0.01 | < 0.05 | < 0.005 |

### 結果解讀

**✅ 良好訓練**（L2 < 15%）:
- 預測場呈現清晰的 Kolmogorov 流特徵（交錯渦對）
- 誤差場分佈均勻，無系統性偏差
- 質量守恆誤差 < 1%

**⚠️ 需要改進**（15% < L2 < 25%）:
- 檢查訓練是否收斂（查看 loss 曲線）
- 考慮延長訓練或調整超參數
- 增加感測點數量 K

**❌ 訓練失敗**（L2 > 25%）:
- 檢查物理參數是否正確（Re, ν, k_f）
- 檢查損失權重平衡
- 使用診斷工具分析：
  ```bash
  python scripts/debug/diagnose_piratenet_failure.py \
    --checkpoint checkpoints/kolmogorov_re100_kf4_K100/epoch_4000.pth
  ```

---

## 📚 相關文檔

- **配置指南**: `docs/KOLMOGOROV_CONFIG_GUIDE.md`
- **訓練指南**: `docs/KOLMOGOROV_CURRICULUM_GUIDE.md`
- **物理驗證**: `docs/KOLMOGOROV_PHYSICS_VALIDATION.md`
- **診斷工具**: `docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md`
- **雷諾數計算**: `scripts/README_REYNOLDS_CALCULATOR.md`

---

## 🎓 進階使用

### 批次評估多個檢查點

```bash
#!/bin/bash
# 評估訓練歷程中的多個檢查點

for epoch in 1000 2000 3000 4000; do
  python scripts/evaluate_kolmogorov_quick.py \
    --checkpoint checkpoints/kolmogorov_re100_kf4_K100/epoch_${epoch}.pth \
    --config configs/kolmogorov_re100_kf4_K100.yml \
    --output results/evaluation_re100_kf4/epoch_${epoch}/ \
    --n-points 256 \
    --time-snapshot 30.0
done

# 比較不同 epoch 的誤差演變
python scripts/compare_checkpoints.py \
  --results-dir results/evaluation_re100_kf4/
```

### 不同雷諾數比較

```bash
# 評估不同 Re 訓練結果
for Re in 56 100 158 197; do
  python scripts/evaluate_kolmogorov_quick.py \
    --checkpoint checkpoints/kolmogorov_re${Re}_kf4_K100/best_model.pth \
    --config configs/kolmogorov_re${Re}_kf4_K100.yml \
    --output results/evaluation_re${Re}_kf4/ \
    --n-points 256
done
```

---

**更新日期**: 2025-11-27  
**版本**: v1.0  
**作者**: PINNs-MVP Team
