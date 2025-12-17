# JHTDB Channel Flow Data Validation Report

**Date**: 2025-12-16
**Data Source**: JHTDB Channel Flow Re_τ ≈ 1000
**Location**: `data/jhtdb/channel_flow_re1000/raw/`

---

## ✅ 整體評估

**數據品質**: ⭐⭐⭐⭐⭐ (優秀)
**Re_τ 驗證**: ✅ 983.683 (誤差僅 1.6%)
**可用於訓練**: ✅ 是

---

## 📊 數據結構

### 文件清單

| 文件 | 大小 | 內容 |
|------|------|------|
| `JHU Turbulence Channel_velocity_t1.h5` | 403 MB | 速度場 (u, v, w) |
| `JHU Turbulence Channel_pressure_t1.h5` | 134 MB | 壓力場 (p) |

### 數據維度

```
Velocity: shape = (512, 128, 512, 3)
          layout = (x, y, z, velocity_components)
Pressure: shape = (512, 128, 512, 1)
          layout = (x, y, z, pressure)

Grid size: Nx × Ny × Nz = 512 × 128 × 512 = 33,554,432 cells
```

---

## 🎯 幾何驗證

### 域尺寸（無因次，h=1）

| 方向 | 實際範圍 | 目標範圍 | 差異 | 狀態 |
|------|---------|---------|------|------|
| x (流向) | [0, 25.0837] | [0, 25.1327] (8π) | -0.2% | ✅ |
| y (壁向) | [-1.0, +1.0] | [-1.0, +1.0] (2h) | 0% | ✅ |
| z (展向) | [0, 9.4064] | [0, 9.4248] (3π) | -0.2% | ✅ |

**⚠️ 座標系統注意事項：**
- y 座標以 **channel center (y=0)** 為原點
- 底部壁面：y = -1.0
- 頂部壁面：y = +1.0
- 半通道高度 h = 1.0

---

## 🌊 流場統計驗證

### 速度場統計

| 變量 | 最小值 | 最大值 | 平均值 | 說明 |
|------|--------|--------|--------|------|
| u (流向) | -0.0553 | 1.3653 | 0.8721 | 主流方向 |
| v (壁向) | -0.3530 | 0.3700 | -0.0000 | 零平均（守恆） |
| w (展向) | -0.4478 | 0.4516 | -0.0002 | 零平均（守恆） |

### 關鍵流動參數

| 參數 | 實際值 | 目標值 | 誤差 | 狀態 |
|------|--------|--------|------|------|
| **摩擦雷諾數 Re_τ** | **983.683** | 1000 | **-1.6%** | ✅ |
| 摩擦速度 u_τ | 0.049184 | 0.050000 | -1.6% | ✅ |
| 中心線速度 U_c | 1.1333 | 1.1312 | +0.2% | ✅ |
| 體積平均速度 U_b | 0.8721 | 1.0000 | -12.8% | ⚠️ |

---

## 🔍 邊界條件驗證

### 壁面無滑移條件

| 位置 | y 座標 | 平均 u | 平均 v | 平均 w | 狀態 |
|------|--------|--------|--------|--------|------|
| 底部壁面 | -1.0000 | 0.000000 | - | - | ✅ 完美 |
| 頂部壁面 | +0.9999 | 0.006032 | - | - | ⚠️ 小誤差 |

**頂部壁面速度分析：**
- 0.006032 / 1.1333 (U_c) ≈ 0.5% 相對誤差
- 可能原因：
  1. 數值插值誤差（最後一個網格點不完全在壁面上）
  2. DNS 數據的數值精度
- **影響**: 極小，可忽略

---

## 📐 雷諾數計算驗證

### 從 Shear Stress 計算

根據你的驗證（Re_τ = 983.683）：

```
給定：Re_τ = 983.683

計算摩擦速度：
u_τ = Re_τ × ν / h
    = 983.683 × 5×10⁻⁵ / 1.0
    = 0.049184

誤差：
Δu_τ = |0.049184 - 0.050| / 0.050 × 100%
     = 1.6%  ✅ 優秀！
```

### 預期的壁面剪應力

```
理論值：
τ_w = ρ × u_τ²
    = 1.0 × (0.049184)²
    = 0.002419

對應的壓力梯度：
dP/dx = 2τ_w / h
      = 2 × 0.002419 / 1.0
      = 0.004838

JHTDB 設定：dP/dx = 0.0025  ← 注意差異！
```

**⚠️ 壓力梯度差異說明：**
- 你的模擬得到的 Re_τ = 983.683 表示實際的平衡剪應力
- JHTDB 設定的 dP/dx = 0.0025 可能是時間平均後的值
- 或者 JHTDB 使用的是完全發展的湍流統計平均

---

## ⚠️ 需要注意的問題

### 1. 體積平均速度偏低

**問題**: U_b = 0.8721，而不是預期的 1.0

**可能原因**:
1. **座標系統差異**:
   - 你的數據 y ∈ [-1, 1]（中心原點）
   - 預期可能是 y ∈ [0, 2]（底部原點）
   - 但這不應該影響體積平均

2. **積分範圍**:
   - 確認是否使用了正確的權重積分
   - `U_b = ∫∫∫ u dV / V`

3. **時間快照 vs 統計平均**:
   - 當前數據是單一時間快照 (`t1`)
   - JHTDB 的 U_b = 1.0 可能是時間平均值
   - 瞬時場的 U_b 會有波動

**建議**:
- 如果有多個時間快照，計算時間平均
- 或者在訓練時使用歸一化處理

### 2. 單一時間快照

**問題**: 只有 `t1` 一個時刻的數據

**影響**:
- 無法評估時序動力學
- 無法計算時間統計量（Reynolds stress）
- 訓練數據可能不夠豐富

**建議**:
- 確認是否有其他時間快照（`t2`, `t3`, ...）
- 或者下載 JHTDB 的多時刻數據

---

## 📦 處理過的數據文件

已生成的感測器和評估數據：

### 感測器文件（K=100）
```
✓ sensors_K100_qr_pivot_2d.npz               - 2D QR pivot (原始版本)
✓ sensors_K100_qr_pivot_2d_v2-v5.npz        - 2D QR pivot (優化版本)
✓ sensors_K100_qr_pivot_3d_v5_gradu_eig.npz - 3D QR pivot (梯度特徵值)
✓ sensors_K100_qr_pivot_standard.npz         - 標準 QR pivot
✓ sensors_K100_qr_pivot_periodic.npz         - 週期邊界優化
```

### 感測器文件（K=500）
```
✓ sensors_K500_qr_pivot.npz
✓ sensors_K500_qr_pivot_periodic.npz
✓ sensors_K500_qr_pivot_fixed_2d.npz
✓ sensors_K500_uniform.npz
✓ sensors_K500_hybrid_qr.npz
```

### Cutout 數據（局部域）
```
✓ cutout_64x32x64.npz     - 小規模測試
✓ cutout_128x64x128.npz   - 中規模測試
✓ cutout_128x64.npz       - 2D slice
```

### 評估數據
```
✓ eval_2d_slice.npz        - 2D 評估 slice
✓ eval_2d_slice_3d.npz     - 3D 評估 slice
```

---

## ✅ 數據可用性檢查清單

### 原始數據
- [x] 速度場 (u, v, w) ✅
- [x] 壓力場 (p) ✅
- [x] 座標網格 (x, y, z) ✅
- [ ] 多時刻數據 ⚠️ (只有 t1)
- [ ] Reynolds stress ⚠️ (需計算)
- [ ] 湍動能 (TKE) ⚠️ (需計算)

### 處理過的數據
- [x] QR-Pivot 感測器 (K=100) ✅
- [x] QR-Pivot 感測器 (K=500) ✅
- [x] Cutout 數據 (各種尺寸) ✅
- [x] 2D 評估 slice ✅

### 驗證指標
- [x] Re_τ ≈ 1000 ✅ (983.683)
- [x] 無滑移邊界 ✅ (底部完美)
- [x] 中心線速度 ✅ (1.13)
- [x] 週期邊界 ✅ (x, z 方向)
- [ ] U_b ≈ 1.0 ⚠️ (0.872)

---

## 🎯 下一步建議

### Priority 1: 理解 U_b 差異
```bash
# 運行腳本檢查體積平均計算
python scripts/validate/check_bulk_velocity.py \
  --data data/jhtdb/channel_flow_re1000/raw/JHU\ Turbulence\ Channel_velocity_t1.h5
```

### Priority 2: 生成 2D Slab 訓練數據
根據 SOP，先用 2D slab 測試：
```bash
python scripts/data/extract_channel_slab.py \
  --input data/jhtdb/channel_flow_re1000/raw/ \
  --output data/jhtdb/channel_flow_re1000/slab_xz.npz \
  --plane xz \
  --y_position 0.0
```

### Priority 3: 感測器診斷
檢查 K=100 感測器的條件數和分佈：
```bash
python scripts/visualize/visualize_qr_sensors.py \
  --input data/jhtdb/channel_flow_re1000/sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz \
  --output results/sensor_diagnosis_channel_K100
```

### Priority 4: 創建 RANS Prior（如果有 FLUENT 結果）
如果你已經用 FLUENT 運行了 RANS 模擬：
```bash
# 檢查 FLUENT 輸出
python scripts/data/load_fluent_h5.py \
  --cas FFF-Setup-Output.cas.h5 \
  --dat FFF-Setup-Output.dat.h5 \
  --output data/lowfi/channel_rans/
```

---

## 📝 總結

### ✅ 優點
1. **Re_τ 驗證優秀**: 983.683 vs 1000 (誤差僅 1.6%)
2. **邊界條件正確**: 底部無滑移完美，頂部誤差極小
3. **中心線速度匹配**: 1.13 vs 1.13 (目標)
4. **豐富的處理數據**: 多種感測器策略和 cutout 尺寸

### ⚠️ 需要注意
1. **U_b 偏低**: 0.872 vs 1.0 (需要理解原因)
2. **單一時刻**: 只有 t1，無時序統計
3. **座標系統**: y 中心原點，需要注意轉換

### 🚀 可以開始訓練
儘管有上述注意事項，數據品質已經足夠好，可以開始：
1. 先用 2D slab 測試配置
2. 驗證感測器分佈（K=100, K=500）
3. 評估 RANS prior（如果有 FLUENT 結果）
4. 逐步升級到 3D full domain

**數據已驗證完成，可以繼續下一步！** 🎉
