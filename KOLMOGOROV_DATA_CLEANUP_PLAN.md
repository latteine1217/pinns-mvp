# 🗂️ Kolmogorov 數據清理計畫

**日期**: 2025-12-18  
**目標**: 刪除所有 k-ε RANS 結果，僅保留 Leith 和 DNS 數據

---

## 📊 當前數據庫狀況

### `data/lowfi/kolmogorov_rans/` (約 18 MB)

**需要刪除的文件**：

#### 1. k-ε RANS 原始數據 (6 個文件, ~5.4 MB)
```
✗ rans_re50_kf4.h5                    (916 KB)  - k-ε RANS Re=50
✗ rans_re100_kf4.h5                   (916 KB)  - k-ε RANS Re=100
✗ rans_re500_kf4.h5                   (920 KB)  - k-ε RANS Re=500
✗ rans_re50_kf4_corrected.h5          (920 KB)  - k-ε RANS Re=50 修正版
✗ rans_re100_kf4_corrected.h5         (920 KB)  - k-ε RANS Re=100 修正版
✗ rans_re500_kf4_corrected.h5         (920 KB)  - k-ε RANS Re=500 修正版
```

#### 2. k-ε RANS Backup 目錄 (4 個目錄, ~14 MB)
```
✗ backup_20251217_110426/             (2.7 MB)  - k-ε RANS 備份 1
✗ backup_20251217_110506/             (2.7 MB)  - k-ε RANS 備份 2
✗ backup_leith_20251217_112721/       (5.4 MB)  - 混合備份（含 k-ε）
✗ backup_leith_20251217_114227/       (5.4 MB)  - 混合備份（含 k-ε）
```

#### 3. k-ε RANS Sensors (3 個文件, ~24 KB)
```
✗ sensors_K100_rans.npz               (8 KB)    - 基於 k-ε RANS 的 QR sensors
✗ sensors_K100_rans_re100_kf4.npz    (8 KB)    - Re=100 k-ε sensors
✗ sensors_K100_rans_re500_kf4.npz    (8 KB)    - Re=500 k-ε sensors
```

#### 4. Leith 舊版本 (6 個文件, ~2.1 MB)
```
✗ rans_re50_kf4_leith_OLD_UNIFORM.h5  (456 KB)  - Leith 舊版 Re=50
✗ rans_re100_kf4_leith_OLD_UNIFORM.h5 (404 KB)  - Leith 舊版 Re=100
✗ rans_re500_kf4_leith_OLD_UNIFORM.h5 (408 KB)  - Leith 舊版 Re=500
✗ rans_re50_kf4_leith_optimized.h5    (136 KB)  - Leith 優化版 Re=50
✗ rans_re100_kf4_leith_optimized.h5   (404 KB)  - Leith 優化版 Re=100
✗ rans_re500_kf4_leith_optimized.h5   (408 KB)  - Leith 優化版 Re=500
```

#### 5. Leith 測試文件 (2 個文件, ~288 KB)
```
✗ rans_re50_kf4_leith_test_conservative.h5  (112 KB)  - Leith 保守測試
✗ test_leith_re50.h5                         (176 KB)  - Leith 測試文件
```

#### 6. Leith Backup (1 個目錄, ~1.2 MB)
```
✗ backup_uniform_params_20251217_142218/  (1.2 MB)  - Leith 舊參數備份
```

---

**需要保留的文件** (僅 3 個, ~1.2 MB)：
```
✓ rans_re50_kf4_leith.h5              (404 KB)  - ✅ 正式 Leith Re=50
✓ rans_re100_kf4_leith.h5             (404 KB)  - ✅ 正式 Leith Re=100
✓ rans_re500_kf4_leith.h5             (408 KB)  - ✅ 正式 Leith Re=500
```

---

### `data/kolmogorov_dns/` (約 17 GB)

**全部保留** - DNS 真值數據：
```
✓ dns_re50_t100.h5                    (751 MB)  - ✅ DNS Re=50
✓ dns_re70_t100.h5                    (751 MB)  - ✅ DNS Re=70
✓ dns_re100_t100.h5                   (751 MB)  - ✅ DNS Re=100
✓ dns_re500_t100.h5                   (2.9 GB)  - ✅ DNS Re=500
✓ dns_re2000_t100.h5                  (12 GB)   - ✅ DNS Re=2000
✓ debug_re50_t10_v2.h5                (76 MB)   - ✅ Debug 數據
✓ snapshot_re50_mid.npz               (689 KB)  - ✅ 快照數據
✓ snapshot_re50_for_eval.npz          (689 KB)  - ✅ 評估快照
✓ qr_sensors_K100_v7_standard.npz     (5.4 KB)  - ✅ QR Sensors
```

---

### `data/sensors/kolmogorov/` (約 30 KB)

**全部保留** - 基於 DNS 的 sensors（非 RANS）：
```
✓ sensors_K100_re50_256x256.json      (8 KB)    - ✅ DNS QR K=100
✓ sensors_K30_re50_256x256.json       (3 KB)    - ✅ DNS QR K=30
✓ sensors_K50_re50_256x256.json       (4 KB)    - ✅ DNS QR K=50
✓ sensors_K80_re50_256x256.json       (6 KB)    - ✅ DNS QR K=80
✓ sensors_K100_re50_256x256_random_seed42.json  (8 KB)  - ✅ 隨機 sensors
```

---

## 🗑️ 刪除操作

### 總計刪除：
- **文件數量**: ~27 個文件/目錄
- **釋放空間**: ~17 MB

### 保留：
- **Leith 數據**: 3 個文件 (~1.2 MB)
- **DNS 數據**: 9 個文件 (~17 GB)
- **Sensors**: 5 個文件 (~30 KB)

---

## ⚠️ 風險評估

### 低風險（可安全刪除）
- ✅ k-ε RANS 原始數據（已完全棄用）
- ✅ k-ε RANS Backup（無需保留）
- ✅ k-ε RANS Sensors（不再使用）

### 中風險（可刪除，但需確認）
- ⚠️ Leith 舊版本（OLD_UNIFORM, optimized）
  - 如果當前 Leith 數據穩定，可刪除
  - 建議先驗證當前 Leith 數據可用性

### 零風險（不應刪除）
- 🔒 `rans_re*_kf4_leith.h5`（正式 Leith 數據）
- 🔒 `dns_*.h5`（DNS 真值）
- 🔒 `sensors_*.json`（基於 DNS 的 sensors）

---

## 📝 執行步驟

### Step 1: 備份當前狀態（可選）
```bash
cd data/lowfi/
tar -czf kolmogorov_rans_backup_full_$(date +%Y%m%d).tar.gz kolmogorov_rans/
```

### Step 2: 刪除 k-ε RANS 數據
```bash
cd data/lowfi/kolmogorov_rans/

# 刪除 k-ε RANS 原始數據
rm -f rans_re*_kf4.h5
rm -f rans_re*_kf4_corrected.h5

# 刪除 k-ε RANS Backup
rm -rf backup_20251217_*
rm -rf backup_leith_2025*

# 刪除 k-ε RANS Sensors
rm -f sensors_K100_rans*.npz
```

### Step 3: 刪除 Leith 舊版本
```bash
cd data/lowfi/kolmogorov_rans/

# 刪除 OLD_UNIFORM 版本
rm -f *_OLD_UNIFORM.h5

# 刪除 optimized 版本
rm -f *_optimized.h5

# 刪除測試文件
rm -f *_test_*.h5
rm -f test_leith_*.h5

# 刪除舊參數備份
rm -rf backup_uniform_params_*
```

### Step 4: 驗證保留文件
```bash
cd data/lowfi/kolmogorov_rans/
ls -lh

# 應該僅剩下：
# rans_re50_kf4_leith.h5
# rans_re100_kf4_leith.h5
# rans_re500_kf4_leith.h5
```

---

## ✅ 清理後的目錄結構

```
data/
├── kolmogorov_dns/                     (保留，17 GB)
│   ├── dns_re50_t100.h5
│   ├── dns_re100_t100.h5
│   ├── dns_re500_t100.h5
│   ├── dns_re2000_t100.h5
│   ├── ... (其他 DNS 文件)
│
├── lowfi/
│   └── kolmogorov_rans/                (清理後，1.2 MB)
│       ├── rans_re50_kf4_leith.h5      ✅ 保留
│       ├── rans_re100_kf4_leith.h5     ✅ 保留
│       └── rans_re500_kf4_leith.h5     ✅ 保留
│
└── sensors/
    └── kolmogorov/                     (保留，30 KB)
        ├── sensors_K100_re50_256x256.json
        └── ... (其他 sensor 文件)
```

---

## 🔍 驗證清單

清理完成後執行：

```bash
# 驗證 Leith 數據完整性
python scripts/validation/verify_leith_data.py

# 確認配置指向正確文件
grep "data_path" configs/kolmogorov_re50_kf4_K100.yml
# 應顯示: ./data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5

# 檢查目錄大小
du -sh data/lowfi/kolmogorov_rans/
# 應為: ~1.2 MB
```

---

**準備執行？** 確認以上計畫後，運行刪除腳本開始清理。
