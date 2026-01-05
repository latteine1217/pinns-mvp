# 🖥️ PINNs-SparseFlow 伺服器環境狀態報告

**伺服器**: junyi@140.114.120.128  
**節點**: acmt0 (登入節點) / acmt20 (計算節點 r740)

---

## ✅ 環境配置狀態

### 1. Python 與套件
- **Python 版本**: 3.10.12 ✅
- **PyTorch**: 2.7.1+cu118 ✅ (CUDA 版本)
- **關鍵依賴**: NumPy, SciPy, h5py, yaml, wandb, matplotlib ✅

### 2. CUDA 環境
- **NVCC 版本**: CUDA 11.5 ✅
- **PyTorch CUDA**: 11.8 (兼容) ✅
- **GPU (登入節點)**: GTX 1050 (2GB) - 僅供測試
- **GPU (計算節點 r740)**: 2x NVIDIA P100 (16GB each) ✅

### 3. SLURM 配置
- **可用分區**: r620, r630a, r630b, r630c, r630l, r740
- **目標分區**: r740 ✅
  - 時間限制: 14 天
  - 節點: acmt20
  - GPU: 2x P100
  - CPU: 48 cores
  - RAM: 108GB+

### 4. 專案結構
```
~/pinns-sparse-flow/
├── scripts/
│   ├── train/train.py          ✅ (已上傳)
│   └── check_server_env.sh     ✅
├── pinnx/                       ✅ (模組已上傳)
├── configs/
│   └── kolmogorov_re50_kf4_K100.yml  ✅
├── data/
│   ├── kolmogorov_dns/
│   │   └── dns_re50_t100.h5    ⚠️  (正在上傳: ~49%)
│   ├── lowfi/kolmogorov_leith/
│   │   └── rans_re50_kf4_leith.h5  ✅ (401KB)
│   └── sensors/kolmogorov/
│       └── sensors_K100_re50_256x256.json  ✅ (1.1KB)
├── slurm_train.sh               ✅ (已改寫)
└── .wandb_config                ✅ (API key 已設置)
```

---

## ⚠️ 當前狀況

### 狀況 1: DNS 資料正在上傳
- **本地檔案大小**: 751MB
- **已上傳大小**: ~382MB (49%)
- **狀態**: 正在後台上傳 (rsync 進程 ID: 23924)
- **預計完成**: 取決於網路速度

**監控命令**:
```bash
# 本地機器查看上傳日誌
tail -f /tmp/rsync_dns.log

# 伺服器端查看檔案大小
ssh junyi@140.114.120.128 "watch -n 5 'ls -lh ~/pinns-sparse-flow/data/kolmogorov_dns/dns_re50_t100.h5'"
```

### 狀況 2: 計算節點 GPU 被佔用
- **當前作業**: Job 2580 (kf_soap_2gpu)
- **佔用資源**: 2x GPU, 8 CPUs
- **運行時間**: 1 天 1 小時+
- **排隊作業**: Job 2581 (test_gpu) - 等待 GPU 可用

**影響**: 測試作業需等待，但不影響正式訓練提交（會自動排隊）

---

## 📋 下一步行動清單

### ✅ 已完成
1. [x] 專案結構上傳到伺服器
2. [x] 環境檢查腳本執行
3. [x] PyTorch CUDA 版本安裝
4. [x] SLURM 訓練腳本改寫
5. [x] WandB 配置設置
6. [x] 感測器和 Leith 先驗上傳

### 🔄 進行中
- [ ] DNS 資料上傳 (49% 完成)

### 📝 待執行

#### 當 DNS 資料上傳完成後：

**步驟 1: 驗證資料完整性**
```bash
ssh junyi@140.114.120.128
cd ~/pinns-sparse-flow

# 測試 HDF5 檔案
python3 -c "import h5py; f = h5py.File('data/kolmogorov_dns/dns_re50_t100.h5', 'r'); print('✅ 檔案完整'); print('Keys:', list(f.keys())); f.close()"
```

**步驟 2: 提交正式訓練作業**
```bash
cd ~/pinns-sparse-flow
sbatch slurm_train.sh
```

**步驟 3: 監控訓練**
```bash
# 查看作業狀態
squeue -u junyi

# 即時查看輸出
JOB_ID=$(squeue -u junyi -h -o "%i" | tail -1)
tail -f ~/logs/pinns_sparse_kflow_${JOB_ID}.out

# 查看錯誤日誌（如果有問題）
tail -f ~/logs/pinns_sparse_kflow_${JOB_ID}.err
```

**步驟 4: 訓練中監控**
```bash
# GPU 使用率（在計算節點上訓練開始後）
watch -n 2 nvidia-smi

# 查看 WandB 同步狀態（訓練完成後）
cd ~/pinns-sparse-flow
wandb sync $(ls -td wandb/offline-run-* | head -1)
```

---

## 📊 預期訓練指標

基於配置 `kolmogorov_re50_kf4_K100.yml`:
- **實驗**: Kolmogorov Flow Re=50, K=100 感測點
- **Epochs**: 10,000 (3 個 curriculum stages)
- **模型**: Fourier-SIREN MLP (8×256, depth=6)
- **優化**: SOAP → L-BFGS fine-tuning
- **物理**: VS-PINN N=(2,12,2) + Leith 先驗

**預估訓練時間**:
- 單 GPU (P100): ~48-72 小時
- 雙 GPU (如支援): ~24-36 小時

**驗收目標**:
- 流場誤差 ≤ 10-15% (相對 L2)
- 優於 RANS Baseline ≥ 30%
- K ≤ 100 感測點
- 收斂速度提升 ≥ 30%

---

## 🐛 常見問題排查

### Q1: DNS 上傳太慢怎麼辦？
```bash
# 檢查 rsync 進程
ps aux | grep rsync

# 如果被中斷，重新啟動
cd ~/Documents/coding/pinns-sparse-flow  # 本地機器
rsync -avz --progress data/kolmogorov_dns/dns_re50_t100.h5 \
  junyi@140.114.120.128:~/pinns-sparse-flow/data/kolmogorov_dns/
```

### Q2: 訓練時 CUDA Out of Memory
**解決方案**: 降低 batch size
```bash
# 編輯配置
vim ~/pinns-sparse-flow/configs/kolmogorov_re50_kf4_K100.yml

# 修改:
training:
  batch_size: 10000  # 從 20000 降低
  sampling:
    N_pde: 10000     # 從 15000 降低
```

### Q3: ModuleNotFoundError
**原因**: PYTHONPATH 未設置
**解決方案**: slurm_train.sh 中已自動設置，檢查是否正確執行
```bash
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"
```

### Q4: WandB 連接失敗
**解決方案**: 使用 offline 模式（slurm_train.sh 已配置）
```bash
# 訓練完成後手動同步
cd ~/pinns-sparse-flow
wandb sync wandb/offline-run-*
```

### Q5: 檢查訓練是否正常
```bash
# 查看最新日誌
tail -100 ~/logs/pinns_sparse_kflow_*.out

# 查看訓練 loss
grep -E "Epoch|Loss|loss" ~/logs/pinns_sparse_kflow_*.out | tail -50

# 查看是否有錯誤
cat ~/logs/pinns_sparse_kflow_*.err
```

---

## 📞 快速命令參考

```bash
# === 登入伺服器 ===
ssh junyi@140.114.120.128

# === 檢查環境 ===
cd ~/pinns-sparse-flow
./scripts/check_server_env.sh

# === 提交訓練 ===
sbatch slurm_train.sh

# === 監控作業 ===
squeue -u junyi                    # 查看作業狀態
sinfo -p r740                      # 查看分區狀態
tail -f ~/logs/*.out               # 即時查看輸出

# === 取消作業 ===
scancel <JOB_ID>

# === 檢查結果 ===
ls -lh ~/pinns-sparse-flow/checkpoints/kolmogorov_re50_kf4_K100_leith_prior/
ls -lh ~/pinns-sparse-flow/results/kolmogorov_re50_kf4_K100_leith_prior/

# === 同步 WandB ===
cd ~/pinns-sparse-flow
wandb sync $(ls -td wandb/offline-run-* | head -1)

# === 下載結果（本地機器） ===
rsync -avz junyi@140.114.120.128:~/pinns-sparse-flow/checkpoints/ \
  ~/Documents/coding/pinns-sparse-flow/checkpoints/
```

---

## 📈 訓練完成後

### 評估模型
```bash
cd ~/pinns-sparse-flow

# 找到最佳模型
BEST_MODEL="checkpoints/kolmogorov_re50_kf4_K100_leith_prior/best_model.pth"

# 執行評估
python3 scripts/evaluate/evaluate_checkpoint.py \
  --checkpoint ${BEST_MODEL} \
  --config configs/kolmogorov_re50_kf4_K100.yml
```

### 視覺化結果
```bash
# 結果圖片位於
ls -lh results/kolmogorov_re50_kf4_K100_leith_prior/visualizations/
```

### 下載到本地
```bash
# 本地機器執行
rsync -avz --progress \
  junyi@140.114.120.128:~/pinns-sparse-flow/results/ \
  ~/Documents/coding/pinns-sparse-flow/results/

rsync -avz --progress \
  junyi@140.114.120.128:~/pinns-sparse-flow/checkpoints/ \
  ~/Documents/coding/pinns-sparse-flow/checkpoints/
```

---

**更新時間**: 2026-01-05 15:00 (GMT+8)  
**狀態**: 環境就緒，等待 DNS 資料上傳完成
