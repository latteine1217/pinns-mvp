# 數據同步指南

本文檔說明如何將本地 `data` 目錄同步到伺服器 `junyi@140.114.120.128`。

---

## 📊 數據目錄結構

```
data/                           (~12 GB 總大小)
├── kolmogorov_dns/            (1.4 MB, ✅ 必要)
│   ├── snapshot_re50_for_eval.npz       # DNS 參考場
│   └── qr_sensors_K100_v7_standard.npz  # 感測器位置
│
├── jhtdb/                     (532 MB, 可選)
│   ├── channel_*.h5                     # Channel flow 數據
│   └── sensors_*.npz                    # 各種感測器配置
│
├── sensors/                   (124 KB, 可選)
│   └── *.json                          # 感測器配置文件
│
├── lowfi_npy/                 (1.3 MB, 可選)
│   ├── u.npy                           # RANS u 分量
│   ├── v.npy                           # RANS v 分量
│   └── p.npy                           # RANS 壓力
│
├── kolmogorov_dns_npy/        (5.9 GB, 可選)
│   └── *.npy                           # 完整時間序列數據
│
└── archived_h5/               (5.1 GB, 可選)
    └── *.h5                            # 歷史歸檔數據
```

---

## 🚀 快速同步（推薦）

### 方法 1：只傳輸必要文件（~1 GB，推薦）

```bash
# 在本地執行
cd /path/to/pinns-sparse-flow
./scripts/tools/sync_data_to_server.sh --essential-only
```

**說明**：
- ✅ 包含 `kolmogorov_dns`, `jhtdb`, `sensors`, `lowfi_npy`
- ❌ 跳過 `archived_h5` (5.1 GB) 和 `kolmogorov_dns_npy` (5.9 GB)
- ⏱️ 傳輸時間：約 5-10 分鐘（取決於網速）

### 方法 2：完整同步（~12 GB）

```bash
# 在本地執行
cd /path/to/pinns-sparse-flow
./scripts/tools/sync_data_to_server.sh --full
```

**說明**：
- ✅ 包含所有文件
- ⏱️ 傳輸時間：約 30-60 分鐘（取決於網速）

---

## 🔍 模擬執行（不實際傳輸）

```bash
# 查看將傳輸哪些文件
./scripts/tools/sync_data_to_server.sh --dry-run

# 查看必要文件將傳輸哪些
./scripts/tools/sync_data_to_server.sh --dry-run --essential-only
```

---

## 🛠️ 進階選項

### 斷點續傳

如果傳輸中斷，可以使用 `--resume` 選項繼續：

```bash
./scripts/tools/sync_data_to_server.sh --essential-only --resume
```

### 自訂伺服器路徑

編輯 `scripts/tools/sync_data_to_server.sh`，修改這些變數：

```bash
SERVER_USER="junyi"
SERVER_HOST="140.114.120.128"
SERVER_PATH="/home/junyi/pinns-sparse-flow"  # 修改為實際路徑
```

---

## ✅ 驗證數據完整性

### 在伺服器上執行驗證

```bash
# SSH 登入伺服器
ssh junyi@140.114.120.128

# 進入專案目錄
cd /home/junyi/pinns-sparse-flow  # 請根據實際路徑調整

# 驗證數據
python scripts/tools/verify_data_integrity.py
```

**預期輸出**：

```
======================================================================
數據完整性驗證
======================================================================

數據目錄: /home/junyi/pinns-sparse-flow/data

【1】Kolmogorov DNS 數據
----------------------------------------------------------------------
  ✓ snapshot_re50_for_eval.npz (0.69 MB)
    包含 5 個變數: ['X', 'Y', 'u', 'v', 'p']
  ✓ qr_sensors_K100_v7_standard.npz (0.01 MB)
    包含 3 個變數: ['x', 'y', 'indices']

【2】JHTDB Channel Flow 數據
----------------------------------------------------------------------
  ✓ channel_47afd90366124d2fd19a340f16d808d5.h5 (0.02 MB)
  ...

✅ 數據完整性驗證通過
```

---

## 🧪 測試訓練

### 單 GPU 訓練測試

```bash
# 在伺服器上執行
python scripts/train/train.py --cfg configs/quick_test.yml --epochs 2
```

### 多 GPU DDP 訓練測試

```bash
# 在伺服器上執行（假設有 2 張 GPU）
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/quick_test.yml --epochs 10
```

---

## 🔧 手動同步（備選方案）

如果自動腳本無法使用，可以手動執行 rsync：

### 必要文件同步

```bash
# 在本地執行
cd /path/to/pinns-sparse-flow

# 同步 kolmogorov_dns（必要）
rsync -avz --progress data/kolmogorov_dns/ \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/data/kolmogorov_dns/

# 同步 jhtdb（可選）
rsync -avz --progress data/jhtdb/ \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/data/jhtdb/

# 同步 sensors（可選）
rsync -avz --progress data/sensors/ \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/data/sensors/

# 同步 lowfi_npy（可選）
rsync -avz --progress data/lowfi_npy/ \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/data/lowfi_npy/
```

---

## ❓ 常見問題

### Q1: SSH 連線失敗

**問題**：`ssh: connect to host 140.114.120.128 port 22: Connection refused`

**解決方案**：
1. 確認伺服器 IP 正確
2. 確認 SSH 服務運行中：`ssh junyi@140.114.120.128 "echo '連線成功'"`
3. 檢查 SSH 金鑰設定：`ssh-copy-id junyi@140.114.120.128`

### Q2: 權限拒絕

**問題**：`rsync: permission denied`

**解決方案**：
```bash
# 在伺服器上創建目錄並設置權限
ssh junyi@140.114.120.128 "mkdir -p /home/junyi/pinns-sparse-flow/data && chmod 755 /home/junyi/pinns-sparse-flow/data"
```

### Q3: 傳輸速度慢

**問題**：傳輸速度只有幾百 KB/s

**解決方案**：
1. 使用 `--essential-only` 模式
2. 壓縮傳輸（rsync 已預設使用 `-z` 壓縮）
3. 考慮使用校內網路

### Q4: 伺服器磁碟空間不足

**問題**：`rsync: write failed on "...": No space left on device`

**解決方案**：
```bash
# 檢查伺服器磁碟空間
ssh junyi@140.114.120.128 "df -h"

# 清理不必要的文件
ssh junyi@140.114.120.128 "du -sh /home/junyi/pinns-sparse-flow/* | sort -h"
```

---

## 📋 檢查清單

同步前：
- [ ] 確認本地 data 目錄存在且完整
- [ ] 確認可以 SSH 連線到伺服器
- [ ] 確認伺服器有足夠磁碟空間（至少 2 GB，完整需 13 GB）
- [ ] 編輯腳本設置正確的伺服器路徑

同步後：
- [ ] 執行 `verify_data_integrity.py` 驗證數據
- [ ] 執行 `quick_test.yml` 測試訓練
- [ ] 執行 DDP 訓練測試（如果有多 GPU）

---

## 📞 技術支援

如有問題，請檢查：
1. `scripts/tools/sync_data_to_server.sh` 腳本輸出
2. `scripts/tools/verify_data_integrity.py` 驗證結果
3. 伺服器日誌：`/home/junyi/pinns-sparse-flow/logs/`

---

**文檔版本**：1.0  
**最後更新**：2026-01-07  
**相關文檔**：[README.md](../../README.md), [DDP_GUIDE.md](../DDP_GUIDE.md)
