# NVIDIA A100 部署指南

## 📋 執行摘要

**目的**: 將 Kolmogorov Flow PINNs 訓練從 MPS (17 天) 遷移到 NVIDIA A100 (~4-8 小時)  
**配置文件**: `configs/kolmogorov_re158_kf4_K100_a100.yml`  
**預期加速**: **50-100x**

---

## ✅ 專案 CUDA 支持狀態

| 項目 | 狀態 | 說明 |
|------|------|------|
| **PyTorch CUDA** | ✅ 支持 | PyTorch 2.9.1 內建 CUDA 支持 |
| **自動設備選擇** | ✅ 已實現 | `cuda > mps > cpu` |
| **混合精度訓練** | ✅ 支持 | AMP (Automatic Mixed Precision) |
| **配置靈活性** | ✅ 完整 | 單一參數切換設備 |

**程式碼驗證**:
```python
# pinnx/__init__.py
if torch.cuda.is_available():
    default_device = "cuda"  # ✅ 自動偵測 CUDA
```

---

## 🚀 部署步驟

### 1️⃣ 環境檢查 (A100 伺服器上執行)

```bash
# 檢查 GPU 狀態
nvidia-smi

# 預期輸出:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 525.xx.xx    Driver Version: 525.xx.xx    CUDA Version: 12.0   |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  On   | 00000000:00:04.0 Off |                    0 |
# | N/A   30C    P0    50W / 400W |      0MiB / 40960MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# 檢查 PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# 預期輸出:
# CUDA available: True
# GPU: NVIDIA A100-SXM4-40GB
```

### 2️⃣ 上傳專案與數據

```bash
# 假設使用 scp/rsync 上傳到 A100 伺服器
rsync -avz --progress \
  /Users/latteine/Documents/coding/pinns-mvp/ \
  user@a100-server:/path/to/pinns-mvp/

# 確認關鍵文件存在
cd /path/to/pinns-mvp
ls -lh configs/kolmogorov_re158_kf4_K100_a100.yml
ls -lh data/kolmogorov_dns_re56_512x512_kf8_midway.h5
ls -lh data/jhtdb/sensors_kf8_deim_K100.npz
```

### 3️⃣ 安裝依賴 (如需要)

```bash
# 檢查 Python 環境
python --version  # 需要 Python 3.8+

# 安裝 PyTorch (CUDA 版本)
pip install torch==2.9.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 安裝專案依賴
pip install -r requirements.txt
```

### 4️⃣ 啟動訓練

```bash
# 前台運行（測試用）
python scripts/train.py --cfg configs/kolmogorov_re158_kf4_K100_a100.yml

# 背景運行（正式訓練）
nohup python scripts/train.py \
  --cfg configs/kolmogorov_re158_kf4_K100_a100.yml \
  > log/kolmogorov_re158_kf4_K100_a100.log 2>&1 &

echo $! > log/kolmogorov_re158_kf4_K100_a100.pid
echo "Training started with PID: $(cat log/kolmogorov_re158_kf4_K100_a100.pid)"
```

### 5️⃣ 實時監控

```bash
# 監控訓練日誌
tail -f log/kolmogorov_re158_kf4_K100_a100.log

# 監控 GPU 使用率（另一終端）
watch -n 2 nvidia-smi

# 檢查檢查點
ls -lht checkpoints/kolmogorov_re158_kf4_K100_a100/
```

---

## ⚙️ A100 優化配置詳解

### 配置對比

| 參數 | MPS (當前) | A100 (優化) | 改變 |
|------|-----------|------------|------|
| **device** | mps | cuda | ✅ |
| **batch_size** | 512 | 2048 | 4x ⬆ |
| **pde_points** | 50,000 | 80,000 | 1.6x ⬆ |
| **boundary_points** | 2,000 | 4,000 | 2x ⬆ |
| **amp.enabled** | false | true | ✅ |
| **benchmark** | false | true | ✅ |
| **num_workers** | 4 | 8 | 2x ⬆ |

### 關鍵優化

1. **混合精度訓練 (AMP)**
   ```yaml
   training:
     amp:
       enabled: true  # A100 Tensor Cores 加速
   ```
   - 使用 FP16 計算，FP32 儲存
   - A100: ~2x 速度提升
   - 記憶體使用減半

2. **cuDNN Benchmark**
   ```yaml
   reproducibility:
     benchmark: true  # 自動選擇最快卷積算法
   ```
   - 首次運行較慢（尋找最佳算法）
   - 後續 epoch 加速明顯

3. **更大 Batch Size**
   ```yaml
   training:
     batch_size: 2048  # 充分利用 A100 40GB 記憶體
   ```
   - 提高 GPU 利用率
   - 更穩定的梯度更新

---

## ⚡ 效能預估

### 訓練時間對比

| 環境 | 每 Epoch 時間 | 3000 Epochs | 加速比 |
|------|--------------|-------------|--------|
| **MPS (實測)** | 8.5 分鐘 | 17.7 天 | 1x |
| **A100 (保守)** | 10 秒 | 8.3 小時 | 50x ⬆ |
| **A100 (樂觀)** | 5 秒 | 4.2 小時 | 100x ⬆ |

### 記憶體使用預估

```
模型參數: 296,963 (~1.2 MB)
PDE 配點: 80,000 × 2 (xy) × 4 bytes = 0.64 MB
感測器: 100 × (u,v,p) × 4 bytes = 1.2 KB
梯度緩存: ~10-20 GB (AMP 模式)

預計 A100 記憶體使用: 15-25 GB / 40 GB ✅
```

---

## 🔍 故障排除

### 問題 1: CUDA Out of Memory (OOM)

**症狀**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**解決方案**:
```yaml
# 降低 batch size
training:
  batch_size: 1024  # 減半

# 或降低 PDE 配點
training:
  sampling:
    pde_points: 50000  # 減少 37.5%
```

### 問題 2: CUDA 不可用

**症狀**:
```
CUDA available: False
```

**解決方案**:
```bash
# 檢查 NVIDIA Driver
nvidia-smi

# 檢查 PyTorch CUDA 版本
python -c "import torch; print(torch.version.cuda)"

# 重新安裝 PyTorch CUDA 版本
pip install torch==2.9.1+cu118 --force-reinstall
```

### 問題 3: AMP 導致 NaN

**症狀**:
```
Epoch 50 | total_loss: nan
```

**解決方案**:
```yaml
# 關閉混合精度
training:
  amp:
    enabled: false

# 或降低學習率
training:
  optimizer:
    lr: 0.0005  # 減半
```

---

## 📊 訓練完成後評估

### 1. 檢查最終模型

```bash
# 查看檢查點
ls -lh checkpoints/kolmogorov_re158_kf4_K100_a100/best_model.pth

# 檢查訓練歷史
grep "新最佳指標" log/kolmogorov_re158_kf4_K100_a100.log | tail -5
```

### 2. 評估模型性能

```bash
# 快速評估
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re158_kf4_K100_a100/best_model.pth \
  --config configs/kolmogorov_re158_kf4_K100_a100.yml

# 完整評估
python scripts/evaluate_kolmogorov_full.py \
  --checkpoint checkpoints/kolmogorov_re158_kf4_K100_a100/best_model.pth \
  --reference data/kolmogorov_dns_re56_512x512_kf8_midway.h5
```

### 3. 下載結果

```bash
# 從 A100 伺服器下載結果
rsync -avz --progress \
  user@a100-server:/path/to/pinns-mvp/results/kolmogorov_re158_kf4_K100_a100/ \
  ./results_a100/

# 下載檢查點
rsync -avz --progress \
  user@a100-server:/path/to/pinns-mvp/checkpoints/kolmogorov_re158_kf4_K100_a100/best_model.pth \
  ./checkpoints/
```

---

## ✅ 檢查清單

### 部署前
- [ ] A100 伺服器可訪問
- [ ] `nvidia-smi` 顯示 A100 GPU
- [ ] PyTorch CUDA 可用
- [ ] 專案文件已上傳
- [ ] DNS 數據已上傳 (1.7GB)
- [ ] 感測器數據已上傳 (8.3KB)

### 訓練中
- [ ] 訓練進程運行中
- [ ] GPU 使用率 > 80%
- [ ] 無 OOM 錯誤
- [ ] 損失穩定下降
- [ ] 檢查點定期保存

### 訓練後
- [ ] best_model.pth 存在
- [ ] 評估指標達標 (L2 < 15%)
- [ ] 物理約束滿足
- [ ] 結果已下載到本地

---

## 📚 參考資料

- **配置文件**: `configs/kolmogorov_re158_kf4_K100_a100.yml`
- **當前訓練**: `configs/kolmogorov_re56_kf8_K100_balanced_correct.yml` (MPS, 進行中)
- **雷諾數驗證**: `KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`
- **DNS 重命名**: `KOLMOGOROV_DNS_RENAME_REPORT.md`
- **訓練監控**: `/tmp/monitor_training.sh`

---

**文檔生成**: 2025-11-25  
**作者**: PINNs-MVP 自動化系統  
**狀態**: ✅ 準備就緒，可立即部署
