# Batch Size Sweep 效能優化實驗

## 📋 實驗目標

測試不同 batch size 對 P100 GPU 訓練效能的影響，找出最佳配置以提升訓練速度。

**背景**: 
- 先前 AMP 測試顯示 P100 無法從 FP16 獲得加速（0.98x speedup）
- cProfile 分析顯示 backward pass 佔 62.8% 的訓練時間
- P100 有 16GB 記憶體，當前 batch size (5k-8k) 可能未充分利用 GPU

**預期效果**: 1.2-1.5x 訓練加速

---

## 🗂️ 檔案結構

```
.
├── configs/                          # 實驗配置文件
│   ├── batch_test_8k.yml            # Baseline (8k batch, 5k PDE points)
│   ├── batch_test_16k.yml           # 2x batch size
│   ├── batch_test_24k.yml           # 3x batch size
│   └── batch_test_32k.yml           # 4x batch size (可能 OOM)
│
├── run_batch_sweep.sh               # 批次提交 SLURM 任務腳本
├── scripts/analyze_batch_sweep.py   # 結果分析工具
│
├── logs/                            # SLURM 輸出日誌（自動生成）
│   ├── profile_simple_XXXX.log     # 標準輸出
│   └── profile_simple_XXXX.err     # 錯誤輸出
│
└── results/                         # 分析結果（自動生成）
    ├── batch_sweep_analysis.txt    # 文字報告
    └── batch_sweep_analysis.json   # JSON 結果
```

---

## 🚀 快速開始

### 1. 提交實驗任務

```bash
# 在伺服器上執行（junyi@140.114.120.128）
cd ~/pinns-sparse-flow

# 確保已同步最新代碼
git pull

# 提交所有 batch size 測試（8k, 16k, 24k, 32k）
bash run_batch_sweep.sh

# Dry run（只顯示命令不執行）
bash run_batch_sweep.sh --dry
```

**預期輸出**:
```
========================================
🔍 Batch Size Sweep 實驗提交腳本
========================================

實驗設定:
  - 測試 batch sizes: 8k, 16k, 24k, 32k
  - 每個配置訓練 10 epochs
  - 使用 cProfile 進行效能分析
  - 預期時間: 每個任務 ~5 分鐘

📝 檢查配置文件...
✅ configs/batch_test_8k.yml
✅ configs/batch_test_16k.yml
✅ configs/batch_test_24k.yml
✅ configs/batch_test_32k.yml

📤 提交 SLURM 任務...

[1/4] 提交: 8k (baseline)
  Config: configs/batch_test_8k.yml
  ✅ 已提交 Job ID: 2751

[2/4] 提交: 16k (2x)
  Config: configs/batch_test_16k.yml
  ✅ 已提交 Job ID: 2752

...
```

### 2. 監控任務狀態

```bash
# 實時監控任務狀態
watch -n 5 'squeue -u junyi'

# 查看特定任務日誌
tail -f logs/profile_simple_2751.log

# 檢查記憶體使用
ssh junyi@140.114.120.128 "nvidia-smi"
```

### 3. 分析結果

```bash
# 所有任務完成後，運行分析工具
python3 scripts/analyze_batch_sweep.py

# 指定 log 目錄與輸出路徑
python3 scripts/analyze_batch_sweep.py \
  --log-dir logs \
  --output results/batch_sweep_analysis.txt \
  --json
```

**預期輸出**:
```
====================================================================================================
🔍 Batch Size Sweep 效能分析報告
====================================================================================================

Baseline (8k) 平均 Epoch 時間: 3.521s

----------------------------------------------------------------------------------------------------
Batch Size   N_PDE      Avg Epoch (s)   Speedup    Memory (MB)     Status     Job ID    
----------------------------------------------------------------------------------------------------
8000         5000       3.521           1.00x      1450            ✅ OK       2751      
16000        10000      2.834           1.24x      2890            ✅ OK       2752      
24000        15000      2.456           1.43x      4120            ✅ OK       2753      
32000        20000      N/A             N/A        N/A             ❌ OOM      2754      
----------------------------------------------------------------------------------------------------

🏆 最佳配置:
   Batch Size: 24000
   Speedup: 1.43x
   Avg Epoch Time: 2.456s

💡 建議:
   - Batch size 24000 為最佳平衡點（效能增益遞減）
   - Batch size 32000 及以上可能導致 OOM，避免使用
====================================================================================================
```

---

## 📊 實驗配置細節

### Batch Size 與 PDE Points 配置表

| Config File | Batch Size | N_PDE | N_PDE/Batch Ratio | 預期記憶體 | 狀態 |
|------------|-----------|-------|------------------|----------|------|
| `batch_test_8k.yml` | 8,000 | 5,000 | 0.625 | ~1.5 GB | Baseline |
| `batch_test_16k.yml` | 16,000 | 10,000 | 0.625 | ~3 GB | ✅ |
| `batch_test_24k.yml` | 24,000 | 15,000 | 0.625 | ~4.5 GB | ✅ |
| `batch_test_32k.yml` | 32,000 | 20,000 | 0.625 | ~6 GB | ⚠️ 可能 OOM |

**設計原則**:
- 保持 `N_PDE / batch_size ≈ 0.625` 比例（經驗最佳實踐）
- 逐步增加至 P100 記憶體上限（16GB）
- 使用相同的模型架構（width=256, depth=6）確保可比性

### 共用配置參數

所有測試使用相同的超參數（除了 batch_size 與 N_pde）：

```yaml
model:
  type: fourier_vs_mlp
  width: 256
  depth: 6
  activation: swish

training:
  optimizer:
    type: adam
    lr: 0.001
  epochs: 10  # 快速測試

losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 2.0
```

---

## 🔬 技術細節

### 為什麼增加 Batch Size 會加速？

1. **GPU 並行計算**
   - 更大的 batch → 更多並行計算 → 更好的 GPU 利用率
   - P100 有 3584 CUDA cores，小 batch 無法充分利用

2. **減少 Kernel 啟動開銷**
   - 每次 forward/backward 需要啟動 CUDA kernel
   - 較大 batch → 較少的 kernel 啟動次數

3. **記憶體帶寬優化**
   - 連續記憶體訪問更有效率
   - 較大 batch 提升快取命中率

### 為什麼 N_PDE/batch_size 保持 0.625？

- **物理點採樣密度**: N_PDE 是 PDE 殘差計算的採樣點數
- **資料點數量**: batch_size 包含 sensor data points + PDE points + boundary points
- **平衡原則**: 太多 PDE points → 計算成本高；太少 → 物理約束不足
- **0.625 比例**: 經驗上對 Kolmogorov flow 效果最好

### 記憶體估算

**公式**:
```
Memory ≈ batch_size × (model_width × depth × 4 bytes) × gradient_factor
```

**實際測量** (Job 2746):
- batch_size = 5000, N_pde = 3000
- Peak memory = 168.5 MB (模型計算)
- Total memory = ~1.5 GB (含 optimizer state)

**預測**:
- 16k batch → ~3 GB
- 24k batch → ~4.5 GB
- 32k batch → ~6 GB (可能 OOM，取決於其他記憶體開銷)

---

## 📈 預期結果

### Conservative Estimates

| Batch Size | Expected Speedup | Risk Level |
|-----------|-----------------|-----------|
| 8k (baseline) | 1.00x | ✅ Low |
| 16k | 1.15-1.25x | ✅ Low |
| 24k | 1.35-1.50x | ✅ Low |
| 32k | 1.40-1.60x | ⚠️ Medium (OOM risk) |

### Success Criteria

**Minimum Goals** (達成即為成功):
- ✅ 找到最佳 batch size（無 OOM）
- ✅ 獲得 ≥ 1.2x 加速比

**Stretch Goals** (額外獎勵):
- ✅ 獲得 ≥ 1.5x 加速比
- ✅ 記憶體利用率 > 30%（5 GB / 16 GB）

---

## 🛠️ 故障排除

### 常見問題

#### 1. OOM Error (Out of Memory)

**症狀**:
```
RuntimeError: CUDA out of memory. Tried to allocate XXX MiB
```

**解決方案**:
- 使用較小的 batch size（如 24k 而非 32k）
- 檢查是否有記憶體洩漏（重啟 SLURM job）
- 使用 gradient checkpointing（需要修改代碼）

#### 2. Job 卡住不動

**症狀**:
- `squeue` 顯示任務在運行但日誌無更新

**解決方案**:
```bash
# 取消任務
scancel <JOB_ID>

# 檢查節點狀態
sinfo -N -l

# 重新提交到不同節點
sbatch --exclude=acmt20 --export=CONFIG=configs/batch_test_16k.yml slurm_profile_simple.sh
```

#### 3. 找不到配置文件

**症狀**:
```
FileNotFoundError: configs/batch_test_8k.yml
```

**解決方案**:
```bash
# 確認配置文件存在
ls configs/batch_test_*.yml

# 確認 git 同步
git status
git pull
```

#### 4. 分析腳本無法找到 logs

**症狀**:
```
❌ 找不到 batch sweep 相關的 log 文件
```

**解決方案**:
```bash
# 確認 logs 目錄存在
ls logs/profile_simple_*.log

# 手動指定 log 目錄
python3 scripts/analyze_batch_sweep.py --log-dir ~/pinns-sparse-flow/logs

# 檢查 log 內容是否包含 batch_test 標記
grep "batch_test" logs/profile_simple_*.log
```

---

## 📝 後續優化方向

完成 batch size sweep 後，根據結果可以繼續嘗試：

### Priority 1: torch.compile (如果 batch size 有效)
- 可與 batch size 優化疊加
- 預期額外 1.2x 加速
- **總加速**: 1.43x (batch) × 1.2x (compile) = **1.72x**

### Priority 2: Gradient Checkpointing（如果遇到 OOM）
- 犧牲 10-15% 速度換取 30-40% 記憶體節省
- 允許更大的 batch size
- **淨效果**: 可能仍有 1.2-1.3x 加速

### Priority 3: Model Architecture Simplification
- 僅在不影響精度（< 5% L2 error 增加）的前提下進行
- 需要 100+ epochs 驗證
- 預期 1.3-1.8x 加速

---

## 📚 參考資料

### 相關文檔
- **P100 Optimization Guide**: `docs/P100_OPTIMIZATION_GUIDE.md`
- **Config Guide**: `docs/CONFIG_GUIDE.md`
- **Profiling README**: `scripts/PROFILING_README.md`

### 先前實驗結果
- **Job 2746**: cProfile baseline (62.8% backward pass)
- **Job 2750**: AMP test (0.98x speedup, 失敗)

### 技術參考
- NVIDIA P100 Specs: 16GB HBM2, 732 GB/s bandwidth, 9.3 TFLOPS FP32
- PyTorch Batch Size Tuning: https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- PINNs Batch Size Best Practices: N_PDE/batch_size ≈ 0.5-0.7

---

## 🎯 預期時間表

| 步驟 | 時間 | 說明 |
|-----|------|------|
| 配置準備 | ✅ 完成 | 已創建 4 個配置文件 |
| 任務提交 | ~5 分鐘 | 提交 4 個 SLURM jobs |
| 訓練執行 | ~20 分鐘 | 每個任務 ~5 分鐘（10 epochs） |
| 結果分析 | ~5 分鐘 | 運行分析腳本 |
| **總計** | **~30 分鐘** | 從提交到獲得結果 |

---

## ✅ Checklist

**實驗前**:
- [ ] Git 同步最新代碼
- [ ] 確認配置文件存在（4 個）
- [ ] 檢查 SLURM 可用資源（`sinfo`）
- [ ] 確認資料文件存在（Kolmogorov DNS, sensors）

**實驗中**:
- [ ] 提交所有 4 個任務
- [ ] 記錄 Job IDs
- [ ] 定期檢查任務狀態（`squeue`）
- [ ] 監控記憶體使用（`nvidia-smi`）

**實驗後**:
- [ ] 所有任務完成（成功或失敗）
- [ ] 運行分析腳本
- [ ] 保存分析報告
- [ ] 更新 `docs/P100_OPTIMIZATION_GUIDE.md`
- [ ] Commit 結果與更新文檔

---

## 📧 聯絡資訊

**專案**: pinns-sparse-flow  
**伺服器**: junyi@140.114.120.128  
**節點**: acmt20 (P100 GPU)

**如遇問題**:
1. 檢查本文檔的「故障排除」章節
2. 查看 SLURM logs 錯誤訊息
3. 確認 git 同步與配置文件正確

---

**最後更新**: 2026-01-17 (Batch Sweep 實驗設計)
