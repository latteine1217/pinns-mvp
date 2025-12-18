# 🗂️ Channel Flow 數據清理計畫

**日期**: 2025-12-18  
**目標**: 整理 Channel Flow 數據，刪除冗餘的 2D slice 和舊版 sensor 文件

---

## 📊 當前數據庫狀況

### `data/jhtdb/channel_flow_re1000/` (543 MB)

#### 1. 3D Cutout 數據（3 個文件，~20 MB）
```
✓ cutout_128x64x128.npz          (17 MB)   - 全 3D 切塊（主要使用）
✓ cutout_64x32x64.npz            (2.1 MB)  - 小型 3D 切塊
? cutout_128x64.npz              (1.0 MB)  - 2D 切塊（可能冗餘）
```

#### 2. 2D Slice 評估數據（5 個文件，~8.2 MB）
```
✗ eval_2d_slice.npz              (1.0 MB)  - 早期評估（已過時）
✗ eval_2d_slice_3d.npz           (1.8 MB)  - 早期評估（已過時）
? slab_xz_center.npz             (3.6 MB)  - 中心切面（用於 2D 實驗？）
? slab_xz_nearwall.npz           (3.6 MB)  - 近壁切面（用於 2D 實驗？）
```

#### 3. K=100 Sensors（13 個文件，~87 KB）
**QR-Pivot 版本（多版本迭代）**：
```
? sensors_K100_qr_pivot_2d.npz              (5.4 KB)  - v1
? sensors_K100_qr_pivot_2d_v2.npz           (5.1 KB)  - v2
? sensors_K100_qr_pivot_2d_v3.npz           (5.1 KB)  - v3
? sensors_K100_qr_pivot_2d_v4.npz           (5.1 KB)  - v4
? sensors_K100_qr_pivot_2d_v5_gradu_eig.npz (5.1 KB)  - v5 (2D)
? sensors_K100_qr_pivot_3d_v5_gradu_eig.npz (5.8 KB)  - v5 (3D)
✓ sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz (5.8 KB)  - 最終版本？
? sensors_K100_qr_pivot_periodic.npz        (4.1 KB)  - 週期邊界
? sensors_K100_qr_pivot_standard.npz        (4.0 KB)  - 標準版本
```

**RANS-based Sensors**：
```
✓ sensors_K100_rans_phase_a_with_data.npz   (11 KB)   - Phase A 使用（保留）
✗ sensors_K100_rans_phase_a.npz             (symlink)  - 符號連結（可刪除）
✗ sensors_K100_random_rans_grid.npz         (symlink)  - 符號連結（可刪除）
? sensors_K100_random_rans_grid_with_data.npz (10 KB)  - 隨機 RANS
? sensors_K100_random_stratified.npz        (3.7 KB)  - 分層隨機
```

#### 4. K=500 Sensors（5 個文件，~282 KB）
```
✗ sensors_K500_qr_pivot.npz              (37 KB)   - 早期版本
✗ sensors_K500_hybrid_qr.npz             (39 KB)   - 混合策略
✗ sensors_K500_uniform.npz               (38 KB)   - 均勻分佈
✗ sensors_K500_qr_pivot_fixed_2d.npz     (169 KB)  - 2D 固定版本
? sensors_K500_qr_pivot_periodic.npz     (18 KB)   - 週期版本（可能有用）
```

#### 5. 其他文件
```
✗ sensor_strategies_comparison_K500.png  (329 KB)  - 舊對比圖（可刪除）
✓ raw/                                   (目錄)    - 原始數據（保留）
✓ reports/                               (目錄)    - 報告（保留）
? slices/                                (目錄)    - 切面數據（檢查）
```

---

### `data/lowfi/channel_rans/` (19 MB)

```
✓ rans_k_omega_sst.npz                   (19 MB)   - 主要 RANS 數據（保留）
✓ sensors_K100_rans_phase_a.npz          (8.4 KB)  - Phase A sensor（保留）
✓ sensors_K100_random_rans_grid.npz      (7.5 KB)  - 隨機 RANS sensor（保留）
? sensors_K100_per_feature_phase_a.npz   (15 KB)   - Per-feature sensor
? sensors_per_feature_5_phase_a.npz      (12 KB)   - 5-feature sensor
✓ archive/                               (目錄)    - 歸檔（保留）
```

---

## 🎯 清理策略

### 目標
1. **刪除冗餘的 2D slice 評估數據**（早期實驗產物）
2. **刪除舊版 sensor 迭代版本**（保留最終版）
3. **刪除符號連結**（指向 lowfi 的 symlinks）
4. **刪除舊對比圖**（可重新生成）
5. **保留核心數據**：3D cutouts, 最終 sensors, RANS 數據

### 決策原則
- ✅ **保留**：生產級數據、最終版本、RANS 數據、原始數據
- ⚠️ **需確認**：slab_xz_*.npz（如用於 2D 實驗則保留）
- ✗ **刪除**：eval_*.npz, 舊版本 sensor, K=500 sensors（未使用）

---

## 🗑️ 刪除清單（建議）

### data/jhtdb/channel_flow_re1000/

#### 低風險（可安全刪除）
```bash
# 早期評估數據
✗ eval_2d_slice.npz                      (1.0 MB)
✗ eval_2d_slice_3d.npz                   (1.8 MB)

# 舊版 sensor 迭代（v1-v4）
✗ sensors_K100_qr_pivot_2d.npz           (5.4 KB)
✗ sensors_K100_qr_pivot_2d_v2.npz        (5.1 KB)
✗ sensors_K100_qr_pivot_2d_v3.npz        (5.1 KB)
✗ sensors_K100_qr_pivot_2d_v4.npz        (5.1 KB)
✗ sensors_K100_qr_pivot_2d_v5_gradu_eig.npz (5.1 KB)  # 保留 3D 版本即可
✗ sensors_K100_qr_pivot_3d_v5_gradu_eig.npz (5.8 KB)  # 保留 large 版本

# K=500 sensors（未使用）
✗ sensors_K500_qr_pivot.npz              (37 KB)
✗ sensors_K500_hybrid_qr.npz             (39 KB)
✗ sensors_K500_uniform.npz               (38 KB)
✗ sensors_K500_qr_pivot_fixed_2d.npz     (169 KB)
✗ sensors_K500_qr_pivot_periodic.npz     (18 KB)

# 舊對比圖
✗ sensor_strategies_comparison_K500.png  (329 KB)

# 符號連結
✗ sensors_K100_rans_phase_a.npz          (symlink)
✗ sensors_K100_random_rans_grid.npz      (symlink)
```

#### 需確認（根據使用情況）
```bash
# 2D 實驗切面（如果不做 2D 實驗可刪除）
? slab_xz_center.npz                     (3.6 MB)
? slab_xz_nearwall.npz                   (3.6 MB)
? cutout_128x64.npz                      (1.0 MB)  # 2D cutout

# 其他 sensor 策略
? sensors_K100_qr_pivot_periodic.npz     (4.1 KB)
? sensors_K100_qr_pivot_standard.npz     (4.0 KB)
? sensors_K100_random_rans_grid_with_data.npz (10 KB)
? sensors_K100_random_stratified.npz     (3.7 KB)
```

---

## ✅ 保留清單

### data/jhtdb/channel_flow_re1000/
```
✓ cutout_128x64x128.npz                  (17 MB)   - 主要 3D 數據
✓ cutout_64x32x64.npz                    (2.1 MB)  - 小型 3D 數據
✓ sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz (5.8 KB)  - 最終 sensor
✓ sensors_K100_rans_phase_a_with_data.npz (11 KB)  - Phase A sensor
✓ raw/                                   - 原始數據
✓ reports/                               - 報告文檔
```

### data/lowfi/channel_rans/
```
✓ rans_k_omega_sst.npz                   (19 MB)   - RANS 數據
✓ sensors_K100_rans_phase_a.npz          (8.4 KB)  - Phase A sensor
✓ sensors_K100_random_rans_grid.npz      (7.5 KB)  - 隨機 sensor
✓ archive/                               - 歸檔數據
```

---

## 📈 預期清理效果

### 空間釋放（保守估計）
```
早期評估數據:           ~2.8 MB
舊版 sensor (v1-v5):    ~31 KB
K=500 sensors:          ~282 KB
舊對比圖:               ~329 KB
符號連結:               ~0 KB
─────────────────────────────
總計:                   ~3.4 MB
```

### 空間釋放（激進策略 - 含 2D slice）
```
上述 + 2D slice:        ~10.6 MB
```

### 簡化效果
```
清理前: ~28 個 sensor 文件
清理後: ~5-8 個 sensor 文件（保留核心版本）
簡化率: ~71-82%
```

---

## 📝 執行步驟

### Step 1: 備份重要數據（可選）
```bash
cd data/jhtdb/
tar -czf channel_flow_re1000_backup_$(date +%Y%m%d).tar.gz \
  channel_flow_re1000/sensors_K100_rans_phase_a_with_data.npz \
  channel_flow_re1000/cutout_*.npz
```

### Step 2: 刪除低風險文件
```bash
cd data/jhtdb/channel_flow_re1000/

# 早期評估數據
rm -f eval_2d_slice.npz eval_2d_slice_3d.npz

# 舊版 sensor 迭代
rm -f sensors_K100_qr_pivot_2d*.npz
rm -f sensors_K100_qr_pivot_3d_v5_gradu_eig.npz  # 保留 large 版本

# K=500 sensors
rm -f sensors_K500_*.npz

# 舊對比圖
rm -f sensor_strategies_comparison_K500.png

# 符號連結
rm -f sensors_K100_rans_phase_a.npz
rm -f sensors_K100_random_rans_grid.npz
```

### Step 3: 確認是否刪除 2D slice（需用戶決定）
```bash
# 如果不做 2D 實驗，執行：
rm -f slab_xz_*.npz cutout_128x64.npz
```

### Step 4: 驗證保留文件
```bash
cd data/jhtdb/channel_flow_re1000/
ls -lh

# 應該看到：
# - cutout_128x64x128.npz
# - cutout_64x32x64.npz
# - sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz
# - sensors_K100_rans_phase_a_with_data.npz
# - raw/, reports/
```

---

## ⚠️ 風險評估

### 零風險（可直接刪除）
- ✅ eval_*.npz（早期測試數據）
- ✅ sensors_K100_qr_pivot_2d_v1-v4.npz（舊迭代）
- ✅ sensors_K500_*.npz（K=500 未使用）
- ✅ sensor_strategies_comparison_K500.png（可重新生成）
- ✅ 符號連結（指向 lowfi 目錄）

### 低風險（建議刪除，但需確認）
- ⚠️ sensors_K100_qr_pivot_3d_v5_gradu_eig.npz（已有 large 版本）
- ⚠️ sensors_K100_random_*.npz（如未用於實驗）

### 中風險（需使用者確認）
- ⚠️ slab_xz_*.npz（如用於 2D 實驗則保留）
- ⚠️ cutout_128x64.npz（2D cutout）
- ⚠️ sensors_K100_qr_pivot_periodic/standard.npz（其他策略）

---

## 🔍 使用者確認問題

在執行清理前，請回答：

1. **是否還會進行 2D Channel Flow 實驗？**
   - 是 → 保留 `slab_xz_*.npz`, `cutout_128x64.npz`
   - 否 → 可刪除，節省 ~8.2 MB

2. **Phase A 實驗是否已完成？**
   - 是 → 可刪除相關測試 sensors
   - 否 → 保留所有 Phase A sensors

3. **是否需要 K=500 的 sensor 數據？**
   - 是 → 保留
   - 否 → 可刪除，節省 ~282 KB

4. **是否需要測試不同 sensor 策略？**
   - 是 → 保留 periodic/standard/random 版本
   - 否 → 僅保留最終使用版本

---

**準備執行？** 請確認以上問題後，我將生成自動化清理腳本。
