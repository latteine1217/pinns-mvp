# Job 2774 監控指南

## 🎯 目標
測試 Gradient Clipping + GradNorm 是否能防止 Epoch 650 的 loss 爆炸

## 🔍 關鍵時刻
- **Epoch 650** (預計 09:26 CST): 原始配置的爆炸點
- **Epoch 1000** (預計 09:37 CST): GradNorm 首次更新權重
- **Epoch 2000+**: GradNorm 權重收斂觀察

## 📊 監控工具

### 1. 自動監控 (後台運行中)
```bash
# 狀態: ✅ Running (PID 34599)
# 腳本: ~/watch_critical_epochs.sh
# 輸出: ~/epoch_alerts_2774.txt

# 查看結果
ssh junyi@140.114.120.128 "cat ~/epoch_alerts_2774.txt"
```

### 2. 快速檢查
```bash
# 本地執行
bash /tmp/quick_check_2774.sh
```

### 3. Epoch 650 分析
```bash
# 當訓練到達 Epoch 650 後執行
ssh junyi@140.114.120.128 "python3 ~/analyze_epoch650.py ~/logs/test_K100_stable_2774.out"
```

### 4. 手動檢查命令

#### 檢查最新進度
```bash
ssh junyi@140.114.120.128 "tail -50 ~/logs/test_K100_stable_2774.out | grep 'Epoch' | tail -5"
```

#### 檢查 Epoch 650 附近
```bash
ssh junyi@140.114.120.128 "grep 'Epoch 6[4-6]0/20000' ~/logs/test_K100_stable_2774.out"
```

#### 檢查 GradNorm 權重更新
```bash
ssh junyi@140.114.120.128 "grep -i 'gradnorm.*weight' ~/logs/test_K100_stable_2774.out"
```

#### 檢查 GPU 狀態
```bash
ssh junyi@140.114.120.128 "srun --jobid=2774 nvidia-smi"
```

#### 檢查 Job 狀態
```bash
ssh junyi@140.114.120.128 "squeue -j 2774"
```

## 📈 預期結果

### 成功標準 (Epoch 650)
- ✅ total_loss < 1.5 (baseline: 2.515)
- ✅ momentum_x_loss < 1.0 (baseline: 2.262)
- ✅ 無 NaN / Inf

### GradNorm 行為 (Epoch 1000)
- 權重自動調整（日誌中應出現 "GradNorm weights @ step 1000"）
- 權重範圍: [0.1, 10.0]（相對初始值）
- 預期: data weight ↑, momentum_x weight ↓

## 🚨 異常處理

### 如果 Epoch 650 仍然爆炸
1. Cancel job: `ssh junyi@140.114.120.128 "scancel 2774"`
2. 分析失敗模式
3. 調整參數:
   - Option A: `gradient_clip: 0.5` (降低裁剪閾值)
   - Option B: `peak_lr: 5e-4` (降低學習率)
   - Option C: `warmup_epochs: 1000` (延長 warmup)
   - Option D: `batch_size: 9000` (減小 batch)

### 如果 Job 意外結束
```bash
# 檢查日誌
ssh junyi@140.114.120.128 "tail -100 ~/logs/test_K100_stable_2774.out"

# 檢查錯誤
ssh junyi@140.114.120.128 "tail -100 ~/logs/test_K100_stable_2774.err"
```

## 📝 相關文件

- **配置**: `configs/experiments/S2_k_scan/s2_qr_K100_2d_re100.yml`
- **SLURM 腳本**: `slurm_test_K100_stable.sh`
- **訓練日誌**: `~/logs/test_K100_stable_2774.out` (伺服器)
- **會話摘要**: `context/session_logs/SESSION_SUMMARY_2026-01-17_GradientClipping_GradNorm.md`

## 🎓 技術細節

### Gradient Clipping
- Method: `torch.nn.utils.clip_grad_norm_`
- Threshold: 1.0 (JAX-PI aligned)
- Location: `pinnx/train/trainer.py:1140-1144`

### GradNorm
- Formula: `w_i = mean(G) / (G_i + eps * mean(G))`
- Update frequency: 1000 epochs
- Momentum: 0.9 (EMA smoothing)
- Location: `pinnx/losses/weighting.py:32-170`

---

**Last Updated**: 2026-01-17 09:20 CST  
**Job ID**: 2774  
**Current Epoch**: ~300/20000  
**ETA to Epoch 650**: ~10 minutes
