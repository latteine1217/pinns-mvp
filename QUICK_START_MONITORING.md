# Quick Start Guide: Kolmogorov Re=100 Training Monitoring

This guide provides immediate commands to monitor and manage the ongoing training.

---

## 🚀 Quick Status Check

### Check if Training is Still Running
```bash
ps aux | grep "train.py --cfg configs/kolmogorov_re100_kf8_K50_t20_2k" | grep -v grep
```

**Expected output**: Process with PID 86504 (or similar)  
**If empty**: Training has stopped or crashed

---

## 📊 Monitoring Commands

### 1. Full Dashboard (Recommended)
```bash
bash scripts/monitor_kolmogorov_re100_training.sh
```

**Shows**:
- Process status (PID, running/stopped)
- Latest 15 log lines
- Checkpoint count
- Loss trend (last 5 epochs)
- Training speed estimate
- Quick action commands

### 2. Auto-Refresh Dashboard (Every 5 Minutes)
```bash
watch -n 300 bash scripts/monitor_kolmogorov_re100_training.sh
```

**Usage**: Press `Ctrl+C` to stop watching

### 3. Live Log Tail
```bash
tail -f log/kolmogorov_re100_kf8_K50_t20_2k/training.log
```

**Shows**: Real-time training updates as they happen  
**Usage**: Press `Ctrl+C` to stop

### 4. Check Latest Epoch
```bash
grep "total_loss:" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -1
```

**Shows**: Most recent epoch number and loss value

---

## 🔬 Evaluation Commands

### Quick Evaluation (64×64 grid, ~30s)
```bash
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf8_K50_t20_2k/best_model.pth \
  --config configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
  --output results/eval_latest/ \
  --n-points 64
```

**Outputs**:
- `results/eval_latest/evaluation_results.npz` - Numerical results
- `results/eval_latest/fields_visualization.png` - Flow field plots

### High-Res Evaluation (128×128 grid, ~2min)
```bash
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf8_K50_t20_2k/best_model.pth \
  --config configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
  --output results/eval_latest_highres/ \
  --n-points 128
```

### Evaluate Specific Epoch
```bash
# Replace XXX with actual epoch number (e.g., 50, 100, 200)
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf8_K50_t20_2k/epoch_XXX.pth \
  --config configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
  --output results/eval_epochXXX/ \
  --n-points 64
```

---

## 🎛️ Control Commands

### Stop Training
```bash
# Find PID first
PID=$(ps aux | grep "train.py --cfg configs/kolmogorov_re100_kf8_K50_t20_2k" | grep -v grep | awk '{print $2}')

# Kill the process
kill $PID
```

**Note**: Checkpoints are saved automatically; you can resume from the latest one

### Resume Training from Latest Checkpoint
```bash
nohup python scripts/train.py \
  --cfg configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
  --resume checkpoints/kolmogorov_re100_kf8_K50_t20_2k/best_model.pth \
  > log/kolmogorov_re100_kf8_K50_t20_2k/training_restart.log 2>&1 &

# Save the new PID
echo $! > log/kolmogorov_re100_kf8_K50_t20_2k.pid
```

### Check System Resources
```bash
# CPU/Memory usage
top -pid $(ps aux | grep "train.py --cfg configs/kolmogorov_re100_kf8_K50_t20_2k" | grep -v grep | awk '{print $2}')
```

**Press `q` to exit top**

---

## 📈 Expected Timeline

Based on current speed (~6.8 min/epoch):

| Epochs | Time from Start | Calendar Time (from 18:13) | What to Check |
|--------|-----------------|----------------------------|---------------|
| **10** | ~1.1 hours | ~19:20 | Loss should be < 7.0 |
| **50** | ~5.7 hours | ~00:00 (midnight) | Loss should be < 5.0 |
| **100** | ~11.3 hours | ~05:30 (next morning) | **First major checkpoint** |
| **200** | ~22.7 hours | ~17:00 (next day evening) | Loss should be < 3.0 |
| **500** | ~2.4 days | - | Target completion for baseline |
| **2000** | ~9.4 days | - | Full training (if needed) |

---

## ✅ Decision Points

### @ Epoch 100 (~11 hours from start)

1. **Run evaluation**:
   ```bash
   python scripts/evaluate_kolmogorov_quick.py \
     --checkpoint checkpoints/kolmogorov_re100_kf8_K50_t20_2k/epoch_100.pth \
     --config configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
     --output results/eval_epoch100/
   ```

2. **Check loss trend**:
   ```bash
   grep "total_loss:" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -10
   ```

3. **Decide**:
   - **If loss < 5.0**: Good convergence → continue to 500
   - **If loss > 6.0**: Slow convergence → consider optimization
   - **If loss stagnating**: May need hyperparameter adjustment

### @ Epoch 200 (~23 hours from start)

1. **Evaluate convergence rate**
2. **If loss < 3.0**: Excellent → continue to 500-1000
3. **If loss > 4.0**: Re-evaluate time investment

---

## 🆘 Troubleshooting

### Training Stopped Unexpectedly
```bash
# Check last 50 lines of log for errors
tail -50 log/kolmogorov_re100_kf8_K50_t20_2k/training.log

# Look for error messages
grep -i "error\|exception\|failed" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -20
```

### NaN Losses
```bash
# Check if any NaN losses occurred
grep "nan\|NaN" log/kolmogorov_re100_kf8_K50_t20_2k/training.log

# If found, training needs to restart with lower learning rate
```

### Checkpoints Not Saving
```bash
# Check checkpoint directory permissions
ls -ld checkpoints/kolmogorov_re100_kf8_K50_t20_2k/

# Check disk space
df -h .
```

---

## 📊 Key Metrics to Track

### Loss Trend
- **Epoch 0**: 9.149
- **Epoch 2**: 7.551 (↓ 17.5%)
- **Target @ 100**: < 5.0
- **Target @ 500**: < 2.0

### Physics Residuals
- **Early training** (< 50 epochs): 1.0-10.0 (normal)
- **Mid training** (50-200): 0.1-1.0
- **Late training** (> 200): < 0.1

### Checkpoint Size
- **Expected**: ~40 MB per checkpoint
- **Frequency**: Every 2 epochs (configurable)

---

## 📁 Important Files

| File/Directory | Purpose |
|----------------|---------|
| `configs/kolmogorov_re100_kf8_K50_t20_2k.yml` | Training configuration |
| `checkpoints/kolmogorov_re100_kf8_K50_t20_2k/` | Saved model weights |
| `log/kolmogorov_re100_kf8_K50_t20_2k/training.log` | Full training log |
| `results/quick_eval_epoch2/` | Epoch 2 evaluation results |
| `scripts/monitor_kolmogorov_re100_training.sh` | Monitoring script |
| `scripts/evaluate_kolmogorov_quick.py` | Evaluation tool |
| `docs/SESSION_SUMMARY_2025-11-24.md` | Full session documentation |

---

## 💡 Quick Tips

1. **Don't check too frequently**: Every 5 minutes is sufficient
2. **Use `watch`** for auto-refresh instead of manual checking
3. **Save evaluation results** with descriptive names (e.g., `eval_epoch100`)
4. **Monitor disk space**: Checkpoints can accumulate (40 MB each)
5. **Keep terminal open**: Don't close the shell that started training

---

## 🎯 Next Steps (Recommended)

### Today (2025-11-24)
- [x] Launch training (PID 86504)
- [x] Setup monitoring tools
- [x] Test evaluation pipeline
- [ ] Check status in 1 hour (epoch ~10)
- [ ] Let run overnight

### Tomorrow (2025-11-25 Morning)
- [ ] Check epoch ~100 status (~05:30)
- [ ] Run full evaluation on epoch 100
- [ ] Analyze loss trend
- [ ] Decide: continue, optimize, or reduce epochs

### Day 3 (2025-11-26)
- [ ] Check epoch ~200 status
- [ ] If converging well, continue to 500
- [ ] If slow, create optimized config

---

**Current Status** (as of 18:39 2025-11-24):  
✅ Training running (PID 86504)  
✅ Epoch 2/2000 completed  
✅ Loss trending down (9.15 → 7.55)  
⏱️ Next checkpoint expected: Epoch 4 (~18:45)

**Recommended Action**:  
Monitor in 1 hour, then let run overnight. Evaluate at epoch 100.
