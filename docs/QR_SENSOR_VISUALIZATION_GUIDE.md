# QR-Pivot 感測點視覺化工具使用指南

## 📋 功能概述

`scripts/visualize_qr_sensors.py` 提供完整的感測點分析與視覺化功能：

1. **載入感測點資料**（.npz 或 .h5 格式）
2. **3D/2D 空間分佈圖**（多視角投影）
3. **統計分析**（距離分佈、座標範圍、速度場統計）
4. **策略比較**（QR-Pivot vs POD vs Greedy）
5. **資料匯出**（JSON + 文字表格）

---

## 🚀 快速開始

### 1. 從已有感測點資料視覺化

```bash
# 基本用法（標準 QR-Pivot）
python scripts/visualize_qr_sensors.py \
  --input data/jhtdb/sensors_K50.npz \
  --output results/sensor_analysis

# 分層 QR-Pivot 結果（自動檢測）⭐
python scripts/visualize_qr_sensors.py \
  --input results/qr_pivot_stratified/stratified_qr_K50.npz \
  --output results/stratified_analysis

# 指定視角
python scripts/visualize_qr_sensors.py \
  --input data/jhtdb/sensors_K50.npz \
  --output results/sensor_analysis \
  --views xy xz yz
```

### 2. 從 JHTDB 資料重新計算感測點

```bash
# 使用 QR-Pivot 選擇 50 個感測點
python scripts/visualize_qr_sensors.py \
  --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
  --n-sensors 50 \
  --strategy qr_pivot \
  --output results/sensor_analysis

# 使用 POD-based 策略
python scripts/visualize_qr_sensors.py \
  --jhtdb-data data/jhtdb/channel_flow_re1000.npz \
  --n-sensors 30 \
  --strategy pod_based \
  --output results/sensor_pod
```

### 3. 比較多種策略

```bash
python scripts/visualize_qr_sensors.py \
  --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
  --n-sensors 50 \
  --compare-strategies \
  --output results/sensor_comparison
```

---

## 📂 輸入檔案格式

### NPZ 格式（推薦）

```python
# 選項 1: 僅感測點資料
{
    'sensor_indices': np.array([...]),      # 感測點索引
    'sensor_coords': np.array([N, 3]),      # 座標 (x, y, z)
    'sensor_values': np.array([N, 3])       # 速度 (u, v, w)
}

# 選項 2: 完整 JHTDB 資料
{
    'x': np.array([Nx]),                    # x 座標
    'y': np.array([Ny]),                    # y 座標  
    'z': np.array([Nz]),                    # z 座標
    'u': np.array([Nx, Ny, Nz]),           # u 速度分量
    'v': np.array([Nx, Ny, Nz]),           # v 速度分量
    'w': np.array([Nx, Ny, Nz]),           # w 速度分量
    'indices': np.array([...])              # 選定的感測點索引（選用）
}
```

### HDF5 格式

```python
{
    'sensor_indices': Dataset([...]),
    'sensor_coords': Dataset([N, 3]),
    'sensor_values': Dataset([N, 3])
}
```

---

## 📊 輸出檔案說明

執行後會在輸出目錄生成以下檔案：

| 檔案名稱 | 說明 |
|---------|------|
| `sensor_distribution_2d_xy.png` | XY 平面 2D 分佈圖 |
| `sensor_distribution_2d_xz.png` | XZ 平面 2D 分佈圖 |
| `sensor_distribution_2d_yz.png` | YZ 平面 2D 分佈圖 |
| `sensor_distribution_3d.png` | 3D 立體分佈圖 |
| `sensor_statistics.png` | 統計分析圖表 |
| `sensor_table.txt` | 感測點座標與數值表格 |
| `sensor_data.json` | JSON 格式資料（程式可讀） |
| `strategy_comparison.png` | 策略比較圖（如啟用） |

---

## 🔧 命令行參數

### 必要參數（二擇一）

| 參數 | 說明 | 範例 |
|------|------|------|
| `--input` | 感測點資料檔案路徑 | `--input data/sensors.npz` |
| `--jhtdb-data` | JHTDB 資料路徑（重新計算） | `--jhtdb-data data/jhtdb/channel.h5` |

### 選用參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--n-sensors` | 感測點數量 | 50 |
| `--strategy` | 選點策略 (`qr_pivot`, `pod_based`, `greedy`) | `qr_pivot` |
| `--compare-strategies` | 比較多種策略 | False |
| `--output` | 輸出目錄 | `results/sensor_analysis` |
| `--views` | 2D 視角列表 | `xy xz yz` |

---

## 📈 視覺化圖表說明

### 1. 2D 分佈圖

- **左圖**：按索引顏色編碼，顯示選點順序
- **右圖**：按速度大小顏色編碼，顯示流場特性

### 2. 3D 分佈圖

- 立體顯示感測點空間位置
- 顏色編碼：速度大小或索引
- 標註前 20 個感測點編號

### 3. 統計分析圖

包含四個子圖：
- **座標分佈**：x, y, z 座標的直方圖
- **距離分佈**：感測點間成對距離
- **速度分佈**：u, v, w 速度分量
- **統計摘要**：數值統計與 QR-Pivot 指標

---

## 🔍 QR-Pivot 指標解讀

輸出中會顯示以下指標：

| 指標 | 說明 | 理想值 | 計算方式 |
|------|------|--------|---------|
| **速度場條件數** (Velocity Condition Number) | 評估速度場矩陣 V 的數值尺度平衡 | < 50（越小越好） | κ(V) = σ_max / σ_min |
| **加權條件數** (Weighted Condition Number) | 物理量尺度平衡後的條件數 | < 10（越小越好） | κ(W⊙V)，W 為物理尺度權重 |
| **能量比例** (Energy Ratio) | 捕捉的 POD 能量比例 | > 0.95（越高越好） | Σσ²(selected) / Σσ²(all) |
| **子空間覆蓋率** (Subspace Coverage) | 覆蓋的流場特徵空間 | > 0.90（越高越好） | rank(V) / min(K, d) |

### ⚠️ 條件數計算注意事項

**正確方式**：使用速度場矩陣 V 的條件數
```python
_, s, _ = svd(V, full_matrices=False)
kappa = s[0] / s[-1]  # 正確：評估數值尺度
```

**錯誤方式**：對低秩矩陣使用 Gram 矩陣條件數
```python
G = V @ V.T + regularization * I
kappa = np.linalg.cond(G)  # ❌ 錯誤：零特徵值導致數值誤差放大
```

**為何 Gram 矩陣條件數會誤導？**
- 對於 K=50, d=3 的速度場矩陣：秩 ≤ 3（受特徵數限制）
- Gram 矩陣 G = V @ V^T 的秩 ≤ 3 → 47 個零特徵值是**固有的**
- 數值誤差在零特徵值處被放大，產生 1e+10 的誤導性條件數
- 實際速度場條件數可能僅 10-50（優秀品質）

**適用場景**：
- ✅ 速度場條件數：**所有場景**（2D/3D，低秩/滿秩）
- ✅ Gram 矩陣條件數：僅適用於**滿秩矩陣** (K ≤ d) 或訓練穩定性分析
- ✅ 加權條件數：評估**物理量平衡**後的品質

---

## 💡 實用案例

### 案例 1：驗證訓練用的感測點品質

```bash
# 1. 從訓練配置提取感測點
python scripts/extract_sensors_from_checkpoint.py \
  --checkpoint checkpoints/my_exp/best_model.pth \
  --output data/sensors_from_training.npz

# 2. 視覺化分析
python scripts/visualize_qr_sensors.py \
  --input data/sensors_from_training.npz \
  --output results/training_sensors_check
```

### 案例 2：感測點數量敏感性分析

```bash
# 比較不同 K 值
for K in 20 30 50 80 100; do
  python scripts/visualize_qr_sensors.py \
    --jhtdb-data data/jhtdb/channel_flow.h5 \
    --n-sensors $K \
    --output results/k_sensitivity/K${K}
done
```

### 案例 3：選點策略對比實驗

```bash
# 同時比較三種策略
python scripts/visualize_qr_sensors.py \
  --jhtdb-data data/jhtdb/channel_flow.h5 \
  --n-sensors 50 \
  --compare-strategies \
  --output results/strategy_comparison
```

---

## ⚠️ 常見問題

### Q1: 載入 NPZ 檔案時報錯 "鍵不存在"

**原因**：NPZ 檔案的鍵名與腳本預期不符。

**解決**：腳本會自動嘗試多種常見鍵名：
- 索引: `sensor_indices`, `indices`, `selected_indices`
- 座標: `sensor_coords`, `coordinates`, `coords`, `positions`
- 數值: `sensor_values`, `values`, `u`, `velocity`

如仍報錯，請檢查檔案鍵名：
```python
import numpy as np
data = np.load('your_file.npz')
print(list(data.keys()))
```

### Q2: 3D 圖顯示不完整

**原因**：matplotlib 版本問題。

**解決**：更新 matplotlib：
```bash
pip install --upgrade matplotlib
```

### Q3: 從 JHTDB 計算時記憶體不足

**原因**：資料矩陣過大。

**解決**：
1. 使用子集資料（空間下採樣）
2. 降低感測點數量
3. 使用 HDF5 格式逐塊讀取

---

## 🔗 相關文件

- **感測點選擇演算法**：`pinnx/sensors/qr_pivot.py`
- **PirateNet 訓練失敗診斷**：`docs/PIRATENET_TRAINING_FAILURE_DIAGNOSIS.md`
- **技術文檔**：`TECHNICAL_DOCUMENTATION.md`

---

## 📝 輸出範例

### 終端輸出

```
================================================================================
  🎯 QR-Pivot 感測點分佈視覺化工具
================================================================================
📂 載入感測點資料: data/jhtdb/sensors_K50.npz
  ✅ 載入成功
     包含鍵: ['sensor_indices', 'sensor_coords', 'sensor_values']

================================================================================
  📊 生成視覺化圖表
================================================================================

📊 繪製 2D 分佈圖 (xy 平面)...
  ✅ 已保存: results/sensor_analysis/sensor_distribution_2d_xy.png

📊 繪製 2D 分佈圖 (xz 平面)...
  ✅ 已保存: results/sensor_analysis/sensor_distribution_2d_xz.png

📊 繪製 2D 分佈圖 (yz 平面)...
  ✅ 已保存: results/sensor_analysis/sensor_distribution_2d_yz.png

📊 繪製 3D 分佈圖...
  ✅ 已保存: results/sensor_analysis/sensor_distribution_3d.png

📊 繪製統計資訊...
  ✅ 已保存: results/sensor_analysis/sensor_statistics.png

💾 保存感測點資料表格...
  ✅ 已保存: results/sensor_analysis/sensor_table.txt
  ✅ 已保存: results/sensor_analysis/sensor_data.json

================================================================================
  ✅ 完成
================================================================================

結果已保存至: /path/to/results/sensor_analysis

包含檔案:
  - sensor_data.json
  - sensor_distribution_2d_xy.png
  - sensor_distribution_2d_xz.png
  - sensor_distribution_2d_yz.png
  - sensor_distribution_3d.png
  - sensor_statistics.png
  - sensor_table.txt
```

### 表格輸出範例

```
====================================================================================================
QR-Pivot 感測點座標與數值表格
====================================================================================================

 Index  Global_ID            x            y            z            u            v            w          |U|
----------------------------------------------------------------------------------------------------
     0        123     0.523599     0.157080     1.047198     0.850234    -0.012456     0.234567     0.882145
     1        456     1.570796     0.314159     2.094395     0.912345     0.045678     0.123456     0.925678
   ...
```

---

最後更新：2025-10-16
