# 多 GPU 分散式訓練指南 (DDP Guide)

## 📋 功能概述

PINNx 已整合自動 GPU 環境偵測與 DistributedDataParallel (DDP) 訓練支援：

- ✅ **自動偵測**：載入 `pinnx` 時自動偵測 GPU 數量
- ✅ **自動啟用 DDP**：偵測到多張 GPU 時自動配置 DDP
- ✅ **單 GPU 相容**：單張 GPU 或 CPU 環境使用標準訓練模式
- ✅ **透明整合**：無需修改訓練腳本，自動處理模型包裝與通訊

---

## 🚀 快速開始

### 方法 1：使用 `torchrun`（推薦）

```bash
# 使用 2 張 GPU 訓練
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/your_config.yml

# 使用伺服器上所有 GPU（專案環境：2 張 P100）
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/kolmogorov_re50_kf4_K100.yml
```

### 方法 2：使用 `torch.distributed.launch`（向後相容）

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --use_env \
    scripts/train/train.py --cfg configs/your_config.yml
```

### 方法 3：標準訓練（單 GPU / CPU）

```bash
# 自動使用單張 GPU 或 CPU
python scripts/train/train.py --cfg configs/your_config.yml
```

---

## 🔍 環境偵測機制

### 自動偵測流程

當您執行 `import pinnx` 時，系統會自動執行以下偵測：

```python
# pinnx/__init__.py 中的偵測邏輯
def detect_gpu_environment():
    """偵測 GPU 環境並配置分散式訓練"""
    
    # 1. CUDA 環境偵測
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        
        # 2. 多 GPU 自動啟用 DDP
        if num_gpus > 1:
            env_info['use_ddp'] = True
            env_info['backend'] = 'nccl'  # NVIDIA GPU 最佳化後端
            logger.info(f"🎯 偵測到 {num_gpus} 張 GPU，自動啟用 DDP 訓練")
        else:
            logger.info(f"✅ 偵測到單張 GPU，使用標準訓練模式")
    
    # 3. macOS MPS 環境
    elif torch.backends.mps.is_available():
        env_info['device'] = 'mps'
        logger.info("✅ 偵測到 Apple Silicon GPU (MPS)")
    
    # 4. CPU 環境
    else:
        logger.info("⚠️  未偵測到 GPU，使用 CPU 訓練")
```

### 偵測結果示例

#### 多 GPU 環境（2 張 P100）

```
🎯 偵測到 2 張 GPU，自動啟用 DDP 訓練
   Backend: nccl
   World Size: 2
   Devices: [0, 1]
```

#### 單 GPU 環境

```
✅ 偵測到單張 GPU，使用標準訓練模式
```

#### CPU 環境

```
⚠️  未偵測到 GPU，使用 CPU 訓練
```

---

## ⚙️ 配置參數

### 存取 GPU 環境資訊

```python
import pinnx

# 讀取偵測結果
device = pinnx.Config.default_device    # 'cuda', 'mps', 或 'cpu'
num_gpus = pinnx.Config.num_gpus        # GPU 數量
use_ddp = pinnx.Config.use_ddp          # 是否啟用 DDP
backend = pinnx.Config.ddp_backend      # DDP 後端（'nccl' 或 None）
world_size = pinnx.Config.world_size    # 總程序數
```

### 訓練腳本中的 DDP 初始化

```python
# scripts/train/train.py 中的初始化
def main():
    # 1. 初始化 DDP 環境（自動根據 pinnx.Config 配置）
    ddp_config = init_distributed_mode()
    
    # 2. 獲取當前程序資訊
    is_main_process = (ddp_config['rank'] == 0)
    device = ddp_config['device']
    
    # 3. 只在主程序輸出日誌
    if is_main_process:
        logger.info("訓練開始...")
    
    # 4. 模型自動包裝為 DDP（如果啟用）
    if ddp_config['is_distributed']:
        model = DDP(model, device_ids=[ddp_config['local_rank']])
```

---

## 📊 效能預期

### 理論加速比

| GPU 數量 | 理論加速 | 實際預期 | 備註 |
|---------|---------|---------|------|
| 1 GPU   | 1.0x    | 1.0x    | 基準 |
| 2 GPU   | 2.0x    | 1.7-1.8x | 通訊開銷 ~10-15% |
| 4 GPU   | 4.0x    | 3.2-3.6x | 通訊開銷 ~10-20% |

### 專案環境（2 × P100）

```bash
# 單 GPU 訓練（基準）
python scripts/train/train.py --cfg configs/quick_test.yml
# 預期: ~30 秒/epoch

# 多 GPU DDP 訓練
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/quick_test.yml
# 預期: ~17 秒/epoch（1.76x 加速）
```

---

## 🐛 故障排除

### 問題 1：DDP 未啟用

**症狀**：偵測到多張 GPU 但仍使用單 GPU 訓練

**解決方案**：

```bash
# 確認使用 torchrun 啟動
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/your_config.yml

# 檢查環境變數
echo $WORLD_SIZE  # 應為 GPU 數量
echo $RANK        # 應為 0, 1, ..., N-1
```

### 問題 2：NCCL 初始化失敗

**症狀**：`RuntimeError: NCCL error in: ...`

**解決方案**：

```bash
# 設定 NCCL 除錯模式
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 檢查 CUDA 版本與 NCCL 相容性
nvidia-smi
python -c "import torch; print(torch.cuda.nccl.version())"
```

### 問題 3：記憶體不足 (OOM)

**症狀**：`RuntimeError: CUDA out of memory`

**原因**：DDP 會在每張 GPU 上複製完整模型

**解決方案**：

```yaml
# 配置文件中減少批次大小
training:
  sampling:
    batch_size: 512  # 原本 1024，減半以配合 DDP
```

### 問題 4：訓練速度沒有提升

**可能原因**：

1. **批次大小太小**：通訊開銷超過計算節省
   - 解決：增加 `batch_size` 至 1024 以上

2. **資料傳輸瓶頸**：CPU→GPU 傳輸成為瓶頸
   - 解決：啟用 `pin_memory=True`（Wave 1 優化）

3. **梯度同步開銷**：頻繁的梯度同步
   - 解決：啟用梯度累積（Wave 2 優化）

---

## 📈 效能優化建議

### 與其他優化的組合

DDP 與以下優化組合效果最佳：

1. **混合精度訓練 (AMP)**
   ```yaml
   training:
     amp:
       enabled: true  # 減少記憶體佔用 30%，加速 15-25%
   ```

2. **Pin Memory**
   ```python
   dataloader = DataLoader(..., pin_memory=True)  # 加速 CPU→GPU 傳輸
   ```

3. **梯度累積**
   ```yaml
   training:
     gradient_accumulation_steps: 2  # 有效 batch size × 2
   ```

### 最佳配置範例

```yaml
# configs/ddp_optimized.yml
training:
  sampling:
    batch_size: 2048  # DDP × 2 GPU = 4096 有效 batch size
  
  amp:
    enabled: true
    
  gradient_accumulation_steps: 2  # 進一步擴大 batch size
```

```bash
# 啟動命令
torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/ddp_optimized.yml
```

**預期效能**：
- 單 GPU 基準：30 秒/epoch
- DDP (2 GPU)：17 秒/epoch (1.76x)
- DDP + AMP：13 秒/epoch (2.31x)
- DDP + AMP + 梯度累積：11 秒/epoch (2.73x)

---

## 🔗 相關文檔

- **性能優化總覽**：`tasks/perf-analysis-001/perf_playbook.md`
- **Wave 3 Advanced 優化**：包含完整 DDP 實作細節
- **PyTorch DDP 官方文檔**：https://pytorch.org/tutorials/intermediate/ddp_tutorial.html

---

## ✅ 驗證清單

使用以下清單確認 DDP 正確運行：

- [ ] 載入 pinnx 時顯示 "🎯 偵測到 N 張 GPU，自動啟用 DDP 訓練"
- [ ] 使用 `torchrun` 或 `torch.distributed.launch` 啟動訓練
- [ ] 日誌顯示 "🚀 DDP 初始化完成"
- [ ] `nvidia-smi` 顯示所有 GPU 都有程序運行
- [ ] 訓練速度相比單 GPU 有明顯提升（1.7-1.8x for 2 GPUs）
- [ ] Checkpoint 正確保存且只在 rank 0 保存
- [ ] TensorBoard 日誌正常記錄

---

**最後更新**：2026-01-07  
**作者**：PINNx Performance Optimization Team
