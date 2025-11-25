# 訓練持續運行計劃
**開始時間**: 2025-11-24 18:13  
**當前狀態**: 2025-11-24 23:07 - Epoch 22/2000  
**決策**: 繼續訓練至 Epoch 100，然後重新評估

---

## 📊 當前訓練狀態

### **進程信息**
- **PID**: 86504 ✅ 運行中
- **當前 Epoch**: 22 / 2000 (1.1%)
- **最佳 Loss**: 1.068 (epoch 22)
- **訓練時長**: 4小時47分

### **訓練速度**
- **時間/Epoch**: ~12.2 分鐘
- **預估到 Epoch 100**: ~20 小時 (明天 19:00)
- **預估到 Epoch 500**: ~4.1 天
- **預估到 Epoch 2000**: ~16.8 天

### **收斂表現** ⭐⭐⭐
```
Epoch  0: Loss = 9.149
Epoch 10: Loss = 1.728  (↓ 81.1%)
Epoch 22: Loss = 1.068  (↓ 88.0%)
```

**物理殘差改善**:
- PDE 損失: 8.528 → 0.520 (↓ 93.9%) ✅
- 動量 Y: 5.021 → 0.036 (↓ 99.3%) ✅
- 連續性: 1.435 → 0.016 (↓ 98.9%) ✅

---

## 🎯 評估檢查點計劃

### **自動評估系統**
已設置自動評估腳本: `scripts/auto_evaluate_checkpoints.sh`

**目標 Epochs**: 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 300, 400, 500

**使用方式**:
```bash
# 手動執行檢查
bash scripts/auto_evaluate_checkpoints.sh

# 查看已評估的 epochs
ls -1 results/kolmogorov_re100_kf8_K50_t20_2k_auto_eval/
```

### **關鍵檢查點時間表**

| Epoch | 預計時間 | 預計日期 | 重要性 | 決策點 |
|-------|----------|----------|--------|--------|
| **30** | 02:00 | 2025-11-25 凌晨 | ⭐ | 趨勢確認 |
| **50** | 08:00 | 2025-11-25 早上 | ⭐⭐ | 初步評估 |
| **100** | 19:00 | 2025-11-25 晚上 | ⭐⭐⭐ | **主要決策點** |
| **150** | 09:00 | 2025-11-26 早上 | ⭐⭐ | 中期評估 |
| **200** | 22:00 | 2025-11-26 晚上 | ⭐⭐⭐ | **第二決策點** |

---

## 🔍 Epoch 100 評估計劃 (主要決策點)

**預計時間**: 2025-11-25 19:00

### **評估內容**
```bash
# 1. 完整評估 (高解析度)
python scripts/evaluate_kolmogorov_quick.py \
  --checkpoint checkpoints/kolmogorov_re100_kf8_K50_t20_2k/epoch_100.pth \
  --config configs/kolmogorov_re100_kf8_K50_t20_2k.yml \
  --output results/eval_epoch100_detailed/ \
  --n-points 256

# 2. 查看損失歷史
grep "total_loss:" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -20

# 3. 繪製訓練曲線
# (自動完成)
```

### **決策標準**

#### **情況 1: 優秀收斂** ✅ → 縮減至 500 Epochs
**條件**:
- Total Loss < 0.3
- 連續性殘差 < 0.01
- 動量殘差 < 0.1
- 物理診斷全部通過

**行動**:
```bash
# 停止當前訓練
kill 86504

# 修改配置
# epochs: 2000 → 500
# switch_epoch: 1600 → 400

# 從 epoch 100 繼續訓練
nohup python scripts/train.py \
  --cfg configs/kolmogorov_re100_kf8_K50_500ep.yml \
  --resume checkpoints/kolmogorov_re100_kf8_K50_t20_2k/epoch_100.pth \
  > log/kolmogorov_re100_kf8_K50_500ep/training.log 2>&1 &
```

#### **情況 2: 良好收斂** ✅ → 繼續至 200-500
**條件**:
- Total Loss 0.3 - 0.5
- 連續性殘差 0.01 - 0.05
- 動量殘差 0.1 - 0.3
- 損失仍在穩定下降

**行動**: 繼續訓練至 epoch 200，再次評估

#### **情況 3: 收斂緩慢** ⚠️ → 優化配置
**條件**:
- Total Loss > 0.5
- 物理殘差 > 0.3
- 損失下降趨緩

**行動**: 調整學習率或優化器設定

#### **情況 4: 已收斂** ⭐ → 可停止
**條件**:
- Loss 連續 20 epochs 變化 < 1%
- 所有物理診斷通過

**行動**: 停止訓練，進行完整評估

---

## 📈 監控指令

### **即時監控** (推薦)
```bash
# 每 5 分鐘自動刷新
watch -n 300 bash scripts/monitor_kolmogorov_re100_training.sh
```

### **查看最新狀態**
```bash
# 快速檢查
bash scripts/monitor_kolmogorov_re100_training.sh

# 查看最新 10 個 epochs
grep "total_loss:" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -10

# 查看實時日誌
tail -f log/kolmogorov_re100_kf8_K50_t20_2k/training.log
```

### **檢查檢查點**
```bash
# 列出所有檢查點
ls -lt checkpoints/kolmogorov_re100_kf8_K50_t20_2k/epoch_*.pth | head -10

# 檢查最新評估
ls -lt results/kolmogorov_re100_kf8_K50_t20_2k_auto_eval/
```

---

## 🎯 成功指標追蹤

### **目標 @ Epoch 100**
| 指標 | 當前 (Epoch 22) | 目標 (Epoch 100) | 狀態 |
|------|----------------|------------------|------|
| **總損失** | 1.068 | < 0.5 | 🔄 進行中 |
| **連續性殘差** | 0.016 | < 0.01 | 🔄 接近 |
| **動量 X 殘差** | 0.468 | < 0.1 | 🔄 進行中 |
| **動量 Y 殘差** | 0.036 | < 0.05 | ✅ 已達標 |
| **散度 (平均)** | 7.6e-05 | < 1e-04 | ✅ 已達標 |

### **最終目標 (項目要求)**
- 相對 L2 誤差: ≤ 10-15%
- RMSE 改善: ≥ 30% vs 低保真
- K 感測點: ≤ 50 (當前 K=50 ✅)
- 條件數: < 50 (當前 20.43 ✅)

---

## 🔔 警報條件

### **需要立即處理**
- ❌ 進程意外停止
- ❌ Loss 出現 NaN
- ❌ 磁碟空間不足
- ❌ 記憶體不足

### **檢查方式**
```bash
# 檢查進程
ps aux | grep 86504 | grep -v grep

# 檢查 NaN
grep -i "nan" log/kolmogorov_re100_kf8_K50_t20_2k/training.log

# 檢查磁碟空間
df -h .

# 檢查記憶體
top -pid 86504
```

---

## 📁 重要檔案位置

### **配置與日誌**
- 配置: `configs/kolmogorov_re100_kf8_K50_t20_2k.yml`
- 訓練日誌: `log/kolmogorov_re100_kf8_K50_t20_2k/training.log`
- PID 檔案: `log/kolmogorov_re100_kf8_K50_t20_2k.pid` (需手動創建)

### **檢查點**
- 目錄: `checkpoints/kolmogorov_re100_kf8_K50_t20_2k/`
- 最佳模型: `checkpoints/kolmogorov_re100_kf8_K50_t20_2k/best_model.pth`
- 大小: ~42 MB / 檢查點

### **評估結果**
- 即時評估: `results/eval_epoch*/`
- 自動評估: `results/kolmogorov_re100_kf8_K50_t20_2k_auto_eval/`
- 訓練進度圖: `results/training_progress/`

### **文檔**
- 完整總結: `docs/SESSION_SUMMARY_2025-11-24.md`
- 快速指南: `QUICK_START_MONITORING.md`
- 本計劃: `docs/TRAINING_CONTINUATION_PLAN.md` (本檔案)

---

## 💡 建議的檢查排程

### **每日檢查** (最小頻率)
- **早上** (08:00): 檢查訓練狀態，查看最新 epoch
- **晚上** (20:00): 查看損失趨勢，決定是否繼續

### **關鍵時刻檢查**
- **Epoch 50** (2025-11-25 08:00): 初步評估
- **Epoch 100** (2025-11-25 19:00): **主要決策點** ⭐⭐⭐
- **Epoch 200** (2025-11-26 22:00): 第二決策點

### **自動化建議**
```bash
# 將監控腳本加入每小時執行 (可選)
# 編輯 crontab: crontab -e
0 * * * * cd /Users/latteine/Documents/coding/pinns-mvp && bash scripts/auto_evaluate_checkpoints.sh >> log/auto_eval.log 2>&1

# 每 6 小時發送狀態報告 (可選，需配置郵件)
0 */6 * * * cd /Users/latteine/Documents/coding/pinns-mvp && bash scripts/monitor_kolmogorov_re100_training.sh | mail -s "Training Status" your@email.com
```

---

## 🎊 預期結果

基於當前優秀的收斂表現 (22 epochs 降 88%)，我們預期：

### **Epoch 50**
- Total Loss: ~0.5 - 0.7
- 連續性殘差: ~0.005 - 0.01
- 動量殘差: ~0.1 - 0.2

### **Epoch 100** (目標)
- Total Loss: **< 0.5** ✅
- 連續性殘差: **< 0.01** ✅
- 動量殘差: **< 0.1** ✅
- 物理診斷: **大部分通過** ✅

### **Epoch 200-500** (如需要)
- Total Loss: < 0.3
- 所有物理約束: 通過
- 流場重建誤差: ≤ 15%

---

## 🔄 下次會話快速啟動

```bash
# 1. 檢查訓練狀態
bash scripts/monitor_kolmogorov_re100_training.sh

# 2. 查看損失趨勢
grep "total_loss:" log/kolmogorov_re100_kf8_K50_t20_2k/training.log | tail -10

# 3. 評估最新檢查點
bash scripts/auto_evaluate_checkpoints.sh

# 4. 查看評估結果
ls -lt results/kolmogorov_re100_kf8_K50_t20_2k_auto_eval/

# 5. 根據 Epoch 100 評估結果決定下一步
```

---

**最後更新**: 2025-11-24 23:07  
**訓練進程**: PID 86504 ✅ 運行中  
**下一檢查點**: Epoch 30 (預計 2025-11-25 02:00)  
**主要決策點**: Epoch 100 (預計 2025-11-25 19:00)
