# 🖥️ 伺服器環境設置指南

> **伺服器資訊**: `junyi@140.114.120.128`  
> **Partition**: r740 (2x Nvidia P100, 108GB RAM)

---

## 📋 快速設置步驟

### 1️⃣ 登入伺服器並上傳專案

```bash
# 本地電腦執行：上傳專案到伺服器
rsync -avz --progress \
  --exclude 'results/' \
  --exclude 'checkpoints/' \
  --exclude 'wandb/' \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  ~/Documents/coding/pinns-sparse-flow \
  junyi@140.114.120.128:~/
```

### 2️⃣ SSH 登入伺服器

```bash
ssh junyi@140.114.120.128
```

### 3️⃣ 執行環境檢查腳本

```bash
cd ~/pinns-sparse-flow
chmod +x scripts/check_server_env.sh
./scripts/check_server_env.sh
```

**預期輸出**：檢查 Python、PyTorch、CUDA、資料檔案、WandB 配置等

---

## 🔧 環境配置（如果檢查失敗）

### A. 創建 Conda 環境（推薦）

```bash
cd ~/pinns-sparse-flow

# 創建環境
conda env create -f environment.yml

# 啟動環境
conda activate pinns-sparse-flow

# 驗證安裝
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### B. 使用現有 Python 環境（如果無 Conda）

```bash
# 安裝依賴
pip install --user -r requirements.txt

# 驗證
python3 -c "import torch, wandb, h5py; print('✅ 依賴安裝成功')"
```

### C. 設置 WandB API Key

```bash
cd ~/pinns-sparse-flow

# 方法 1: 創建配置檔案（推薦，避免 key 洩漏到日誌）
echo "export WANDB_API_KEY=your_wandb_api_key_here" > .wandb_config
chmod 600 .wandb_config  # 設置權限保護

# 方法 2: 直接登入（會儲存到 ~/.netrc）
wandb login your_wandb_api_key_here
```

**取得 WandB API Key**: https://wandb.ai/authorize

---

## 📦 資料檔案檢查

### 必需檔案清單

```bash
# 檢查檔案是否存在
ls -lh ~/pinns-sparse-flow/data/kolmogorov_dns/dns_re50_t100.h5
ls -lh ~/pinns-sparse-flow/data/lowfi/kolmogorov_leith/rans_re50_kf4_leith.h5
ls -lh ~/pinns-sparse-flow/data/sensors/kolmogorov/sensors_K100_re50_256x256.json
```

### 如果檔案缺失

**選項 1: 從本地上傳**
```bash
# 本地執行
rsync -avz --progress \
  ~/Documents/coding/pinns-sparse-flow/data/ \
  junyi@140.114.120.128:~/pinns-sparse-flow/data/
```

**選項 2: 在伺服器上生成**
```bash
cd ~/pinns-sparse-flow

# 生成 DNS 資料（如果有原始資料）
python scripts/generate/generate_kolmogorov_dns.py

# 生成 Leith 先驗
python scripts/generate/generate_leith_prior.py

# 生成感測器位置
python scripts/generate/sensors/generate_qr_sensors.py
```

---

## 🚀 提交訓練作業

### 1. 檢查 SLURM 配置

```bash
cd ~/pinns-sparse-flow

# 查看腳本配置
head -20 slurm_train.sh

# 確認配置檔案路徑
cat configs/kolmogorov_re50_kf4_K100.yml | head -30
```

### 2. 提交作業

```bash
# 提交訓練作業
sbatch slurm_train.sh

# 查看作業狀態
squeue -u junyi

# 取得 JOB_ID
JOB_ID=$(squeue -u junyi -h -o "%i" | head -1)
echo "Job ID: ${JOB_ID}"
```

### 3. 監控訓練

```bash
# 即時查看輸出（按 Ctrl+C 退出）
tail -f ~/logs/pinns_sparse_kflow_${JOB_ID}.out

# 查看錯誤日誌
tail -f ~/logs/pinns_sparse_kflow_${JOB_ID}.err

# 檢查 GPU 使用率（如果作業在運行）
watch -n 1 nvidia-smi
```

### 4. 取消作業（如果需要）

```bash
scancel ${JOB_ID}
```

---

## 📊 訓練完成後

### 同步 WandB 結果（如果使用 offline 模式）

```bash
cd ~/pinns-sparse-flow

# 同步最新 run
wandb sync $(ls -td wandb/offline-run-* | head -1)

# 或同步所有離線 runs
wandb sync wandb/offline-run-*
```

### 查看結果

```bash
# Checkpoints
ls -lh ~/pinns-sparse-flow/checkpoints/kolmogorov_re50_kf4_K100_leith_prior/

# 訓練日誌
less ~/pinns-sparse-flow/log/training.log

# 可視化結果
ls -lh ~/pinns-sparse-flow/results/kolmogorov_re50_kf4_K100_leith_prior/visualizations/
```

### 下載結果到本地

```bash
# 本地執行
rsync -avz --progress \
  junyi@140.114.120.128:~/pinns-sparse-flow/checkpoints/ \
  ~/Documents/coding/pinns-sparse-flow/checkpoints/

rsync -avz --progress \
  junyi@140.114.120.128:~/pinns-sparse-flow/results/ \
  ~/Documents/coding/pinns-sparse-flow/results/
```

---

## 🐛 常見問題排查

### 問題 1: `CUDA out of memory`

**解決方案**：修改配置降低 batch size
```bash
# 編輯配置檔案
vim configs/kolmogorov_re50_kf4_K100.yml

# 找到 training.batch_size，降低數值（例如從 20000 改為 10000）
# 或降低 sampling.N_pde（例如從 15000 改為 10000）
```

### 問題 2: `ModuleNotFoundError: No module named 'pinnx'`

**解決方案**：確認 PYTHONPATH
```bash
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"

# 或在 slurm_train.sh 中已經自動設置，檢查是否正確執行
```

### 問題 3: WandB 無法連接

**解決方案**：使用 offline 模式
```bash
# 在 slurm_train.sh 中已設置
export WANDB_MODE=offline

# 訓練完成後手動同步
wandb sync wandb/offline-run-*
```

### 問題 4: 資料檔案讀取錯誤

**解決方案**：檢查檔案權限和路徑
```bash
# 檢查檔案是否存在且可讀
ls -l ~/pinns-sparse-flow/data/kolmogorov_dns/dns_re50_t100.h5

# 測試 HDF5 檔案完整性
python3 -c "import h5py; f = h5py.File('data/kolmogorov_dns/dns_re50_t100.h5', 'r'); print(list(f.keys())); f.close()"
```

---

## 📝 快速命令參考

```bash
# 環境檢查
./scripts/check_server_env.sh

# 提交作業
sbatch slurm_train.sh

# 查看作業狀態
squeue -u junyi
sinfo -p r740

# 監控輸出
tail -f ~/logs/pinns_sparse_kflow_*.out

# 監控 GPU
watch -n 1 nvidia-smi

# 取消作業
scancel <JOB_ID>

# 同步 WandB
wandb sync $(ls -td wandb/offline-run-* | head -1)
```

---

## 🎯 下一步

1. ✅ 執行環境檢查腳本
2. ✅ 確認所有依賴已安裝
3. ✅ 設置 WandB API Key
4. ✅ 驗證資料檔案存在
5. ✅ 提交訓練作業
6. ✅ 監控訓練進度
7. ✅ 同步結果到 WandB

---

**需要幫助？** 查看詳細文檔：
- 配置說明: `docs/CONFIG_GUIDE.md`
- 訓練指南: `docs/QUICK_START.md`
- 問題排查: `docs/TROUBLESHOOTING.md`
