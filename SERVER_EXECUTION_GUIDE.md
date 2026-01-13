# S2 K-Scan 伺服器執行指南

**日期**: 2026-01-13  
**任務**: 在伺服器上執行 S2 K-scan 實驗（5 個 K 值平行訓練）

---

## ✅ 準備工作完成

### 1. 配置文件已更新
所有 5 個 S2 config 已對齊 Wang et al. (2025) 參數：

| Config File | Status | Parameters |
|-------------|--------|------------|
| `s2_qr_K30_2d_re50.yml` | ✅ | width:768, depth:2, epochs:300K |
| `s2_qr_K50_2d_re50.yml` | ✅ | width:768, depth:2, epochs:300K |
| `s2_qr_K80_2d_re50.yml` | ✅ | width:768, depth:2, epochs:300K |
| `s2_qr_K100_2d_re50.yml` | ✅ | width:768, depth:2, epochs:300K |
| `s2_qr_K200_2d_re50.yml` | ✅ | width:768, depth:2, epochs:300K |

### 2. 配置驗證通過
```bash
✅ K=30:  通過
✅ K=50:  通過
✅ K=80:  通過
✅ K=100: 通過
✅ K=200: 通過
```

### 3. SLURM 腳本配置
```bash
Script: scripts/experiments/run_s2_k_scan_slurm.sh
Resources per job:
  - Time: 14 days
  - Partition: r740
  - Memory: 108G
  - GPU: 1× Nvidia P100
  - Array: 0-4 (5 jobs)
```

---

## 🚀 伺服器執行步驟

### Step 1: 登入伺服器
```bash
ssh junyi@140.114.120.128
```

### Step 2: 進入專案目錄
```bash
cd /home/junyi/pinns-sparse-flow
```

### Step 3: 確認 Git 狀態（可選）
```bash
# 檢查是否有未同步的修改
git status

# 如果本地有更新，拉取最新代碼
git pull origin main
```

### Step 4: 確認配置文件存在
```bash
# 列出所有 S2 config
ls -lh configs/experiments/S2_k_scan/

# 檢查關鍵參數（應該顯示 width:768, depth:2, epochs:300000）
for K in 30 50 80 100 200; do
  echo "=== K=${K} ==="
  grep -E "^\s+(width|depth|epochs):" configs/experiments/S2_k_scan/s2_qr_K${K}_2d_re50.yml
done
```

### Step 5: 創建必要目錄
```bash
# 創建日誌目錄
mkdir -p logs/slurm

# 創建 checkpoint 目錄
mkdir -p checkpoints/experiments

# 創建結果目錄
mkdir -p results/experiments
```

### Step 6: 提交 SLURM 任務
```bash
# 提交陣列任務（5 個 K 值平行執行）
sbatch scripts/experiments/run_s2_k_scan_slurm.sh
```

**預期輸出**:
```
Submitted batch job 123456
```

### Step 7: 監控任務狀態
```bash
# 查看任務佇列
squeue -u junyi

# 查看陣列任務詳細狀態
squeue -u junyi -j 123456

# 持續監控（每 5 秒更新）
watch -n 5 'squeue -u junyi'
```

**預期顯示**:
```
JOBID    PARTITION  NAME        USER   ST  TIME  NODES  NODELIST(REASON)
123456_0 r740       s2_k_scan   junyi  R   0:05  1      node01
123456_1 r740       s2_k_scan   junyi  R   0:05  1      node02
123456_2 r740       s2_k_scan   junyi  PD  0:00  1      (Resources)
123456_3 r740       s2_k_scan   junyi  PD  0:00  1      (Resources)
123456_4 r740       s2_k_scan   junyi  PD  0:00  1      (Resources)
```

說明：
- `R` = Running（正在執行）
- `PD` = Pending（等待資源）
- 由於只有 2 個 GPU，最多同時執行 2 個任務

### Step 8: 監控訓練日誌

**即時監控（選擇一個 K 值）**:
```bash
# 監控 K=100 的訓練日誌
tail -f logs/slurm/s2_k_scan_123456_2.out

# 或使用 less 瀏覽
less +F logs/slurm/s2_k_scan_123456_2.out
# 按 Ctrl+C 停止滾動，按 F 恢復
```

**檢查錯誤日誌**:
```bash
# 查看是否有錯誤
tail -n 50 logs/slurm/s2_k_scan_123456_2.err

# 檢查所有任務的錯誤
for log in logs/slurm/s2_k_scan_123456_*.err; do
  echo "=== $log ==="
  tail -n 10 "$log"
done
```

### Step 9: 監控 GPU 使用

**在執行節點上監控**:
```bash
# 查看任務在哪個節點執行
squeue -u junyi -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R"

# SSH 到執行節點（假設是 node01）
ssh node01

# 監控 GPU 狀態
watch -n 1 nvidia-smi

# 或持續輸出
nvidia-smi dmon -s mu -d 5
```

**預期 GPU 使用**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.xx.xx    Driver Version: 470.xx.xx    CUDA Version: 11.4    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla P100-PCIE...  On   | 00000000:00:04.0 Off |                    0 |
| N/A   65C    P0   180W / 250W |  15000MiB / 16280MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+
```

### Step 10: 檢查 Checkpoint 生成

**每個任務每 2000 epochs 會保存一次 checkpoint**:
```bash
# 查看已生成的 checkpoint
ls -lh checkpoints/experiments/S2_K*/

# 持續監控 checkpoint 數量
watch -n 60 'for K in 30 50 80 100 200; do
  echo "K=${K}: $(ls checkpoints/experiments/S2_K${K}/ 2>/dev/null | wc -l) checkpoints"
done'
```

**預期結構**:
```
checkpoints/experiments/
├── S2_K30/
│   ├── epoch_2000.pth
│   ├── epoch_4000.pth
│   └── ...
├── S2_K50/
├── S2_K80/
├── S2_K100/
└── S2_K200/
```

---

## 📊 訓練進度估算

### 時間估算
| K Value | Total Epochs | Per Window | Est. Time (單任務) | Parallel |
|---------|--------------|------------|-------------------|----------|
| K=30    | 300,000      | 100,000    | ~90-120 hours     | 同時 |
| K=50    | 300,000      | 100,000    | ~90-120 hours     | 同時 |
| K=80    | 300,000      | 100,000    | ~100-130 hours    | 排隊 |
| K=100   | 300,000      | 100,000    | ~100-130 hours    | 排隊 |
| K=200   | 300,000      | 100,000    | ~110-140 hours    | 排隊 |

**總計**: 約 **5-7 天**（考慮平行執行與排隊）

### Checkpoint 數量
- 每個任務: 300,000 / 2,000 = **150 checkpoints**
- 每個 checkpoint: ~50MB
- 每個任務總大小: ~7.5GB
- **5 個任務總計**: ~37.5GB

### 磁碟空間檢查
```bash
# 檢查可用空間
df -h /home/junyi/pinns-sparse-flow

# 預留至少 50GB 空間（包含日誌和結果）
```

---

## 🛠️ 故障排除

### 問題 1: 任務提交失敗
```bash
# 檢查 SLURM 腳本語法
bash -n scripts/experiments/run_s2_k_scan_slurm.sh

# 查看 SLURM 配額
sinfo -p r740
```

### 問題 2: 配置文件找不到
```bash
# 確認當前目錄
pwd
# 應該在: /home/junyi/pinns-sparse-flow

# 列出配置文件
ls configs/experiments/S2_k_scan/

# 如果缺少，從本地同步
# (在本地執行)
rsync -avz configs/experiments/S2_k_scan/ \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/configs/experiments/S2_k_scan/
```

### 問題 3: 訓練中斷或失敗
```bash
# 檢查錯誤日誌
tail -n 100 logs/slurm/s2_k_scan_123456_X.err

# 檢查磁碟空間
df -h

# 檢查記憶體使用
free -h

# 檢查 GPU 狀態
nvidia-smi

# 取消特定任務
scancel 123456_X

# 取消所有陣列任務
scancel 123456
```

### 問題 4: Python 環境問題
```bash
# 檢查 Python 版本
python3 --version

# 檢查 PyTorch
python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# 檢查 PINNx
python3 -c "import pinnx; print(pinnx.__version__)"

# 如果缺少套件，安裝
pip3 install -r requirements.txt
```

---

## 📈 訓練完成後驗證

### Step 1: 確認所有任務完成
```bash
# 檢查任務狀態（應該都消失或顯示 COMPLETED）
squeue -u junyi

# 檢查退出碼
sacct -j 123456 --format=JobID,JobName,State,ExitCode
```

**預期輸出**:
```
JobID           JobName      State    ExitCode
------------ ---------- ---------- -----------
123456_0     s2_k_scan  COMPLETED      0:0
123456_1     s2_k_scan  COMPLETED      0:0
123456_2     s2_k_scan  COMPLETED      0:0
123456_3     s2_k_scan  COMPLETED      0:0
123456_4     s2_k_scan  COMPLETED      0:0
```

### Step 2: 檢查 Checkpoint
```bash
# 統計每個 K 值的 checkpoint 數量
for K in 30 50 80 100 200; do
  COUNT=$(ls checkpoints/experiments/S2_K${K}/*.pth 2>/dev/null | wc -l)
  echo "K=${K}: ${COUNT} checkpoints"
done

# 預期: 每個 K 應該有 ~150 個 checkpoint
```

### Step 3: 快速評估（可選）
```bash
# 對每個 K 值運行快速評估
for K in 30 50 80 100 200; do
  echo "=== 評估 K=${K} ==="
  
  # 找到最後一個 checkpoint
  LAST_CKPT=$(ls checkpoints/experiments/S2_K${K}/epoch_*.pth | sort -V | tail -1)
  
  # 運行評估
  python scripts/evaluate_unified.py \
    --checkpoint "$LAST_CKPT" \
    --output results/experiments/S2_K${K}/eval_final
done
```

### Step 4: 同步結果到本地（可選）
```bash
# 在本地執行
rsync -avz --progress \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/checkpoints/experiments/S2_K*/ \
  ./checkpoints/experiments/

rsync -avz --progress \
  junyi@140.114.120.128:/home/junyi/pinns-sparse-flow/results/experiments/S2_K*/ \
  ./results/experiments/
```

---

## 📝 重要提醒

### 訓練期間注意事項
1. **不要關閉終端機**: SLURM 任務在背景執行，關閉終端機不影響
2. **定期檢查狀態**: 建議每天檢查一次訓練進度
3. **監控磁碟空間**: 確保至少保留 50GB 空間
4. **檢查錯誤日誌**: 如有異常，及時查看 `.err` 文件

### 預期訓練時間
- **開始時間**: 提交任務後立即開始（或排隊）
- **預計完成**: 5-7 天後
- **最長時限**: 14 天（SLURM 設定）

### 資源使用
- **GPU**: 每個任務 1 個 P100
- **記憶體**: 每個任務 108GB
- **磁碟**: 總計 ~37.5GB checkpoint + ~10GB 日誌

---

## 🎯 成功標準

任務成功的指標：
- ✅ 所有 5 個任務狀態為 `COMPLETED`
- ✅ 每個 K 值有 ~150 個 checkpoint
- ✅ 錯誤日誌 (`.err`) 為空或只有警告
- ✅ 最後的 checkpoint 能成功載入
- ✅ 訓練日誌顯示 loss 下降

---

## 📞 聯絡資訊

如遇問題，請檢查：
1. SLURM 日誌: `logs/slurm/s2_k_scan_*.out` 和 `.err`
2. 訓練日誌: 檢查 PINNx 輸出
3. 系統日誌: `dmesg` 或 `/var/log/messages`

---

**準備就緒！** 可以開始執行伺服器訓練了 🚀
