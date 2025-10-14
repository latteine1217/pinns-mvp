# 通用訓練監控系統使用指南

## 📋 概述

通用訓練監控系統提供統一的介面來監控所有 PINNs 訓練任務，自動檢測活躍訓練、解析日誌、計算趨勢，並提供異常警告。

## 🎯 核心特性

### 1. **自動檢測訓練進程**
- 掃描所有正在執行的 `train.py` 進程
- 自動識別配置文件名稱
- 顯示 PID、CPU、Memory 使用情況

### 2. **配置驅動**
- 通過 `configs/monitoring.yml` 統一管理
- 自定義監控指標與閾值
- 靈活的警告規則

### 3. **智能日誌解析**
- 支援多種日誌格式
- 自動提取 loss 指標
- 異常值檢測（NaN、Inf、閾值超標）

### 4. **趨勢分析**
- 計算短/中/長期趨勢
- 趨勢指示器（↑ 上升、↓ 下降、→ 持平）
- ETA 預估（基於歷史速度）

---

## 🚀 快速開始

### 基本用法

#### 1. 監控特定訓練
```bash
python scripts/monitor_training.py --config test_rans_phase6c_v1
```

**輸出範例：**
```
================================================================================
📊 Training Status: test_rans_phase6c_v1
================================================================================
🟢 Active Process:
   PID: 50485
   CPU: 133.7%
   Memory: 9 MB

📈 Progress:
   Epoch: 2650/3000 (88.3%)
   ETA: 0:05:30

📉 Key Metrics:
   total_loss          :   375.2722 ↓
   data_loss           :     0.3030 ↓
   pde_loss            :     6.4857 →
   wall_loss           :    17.0364 ↑

⚠️  Warnings:
   ⚠️  total_loss = 3.75e+02 (超過警告閾值)
   ⚠️  wall_loss = 1.70e+01 (超過警告閾值)
================================================================================
```

#### 2. 監控所有活躍訓練
```bash
python scripts/monitor_training.py --all
```

#### 3. 持續監控模式
```bash
# 每 5 秒刷新（預設）
python scripts/monitor_training.py --all --watch

# 自定義刷新間隔
python scripts/monitor_training.py --all --watch --interval 10
```

#### 4. 詳細模式（顯示所有指標）
```bash
python scripts/monitor_training.py --config phase6c_v1 --verbose
```

**額外輸出：**
```
📊 All Metrics:
   continuity_loss               :     0.248822
   data_loss                     :     0.302961
   epsilon_equation_loss         :   108.227783
   k_equation_loss               :     5.656868
   momentum_x_loss               :     4.756660
   ... (所有指標)
```

---

## ⚙️ 配置文件

### 位置
`configs/monitoring.yml`

### 關鍵配置項

#### 1. **路徑設定**
```yaml
paths:
  log_base_dir: "log"                 # 日誌目錄
  checkpoint_base_dir: "checkpoints"  # 檢查點目錄
  config_dir: "configs"               # 配置目錄
```

#### 2. **監控指標**
```yaml
metrics:
  primary:
    - name: "total_loss"
      display_name: "Total Loss"
      threshold_warning: 100.0      # 警告閾值
      threshold_error: 500.0        # 錯誤閾值
  
  rans:
    - name: "epsilon_equation_loss"
      threshold_warning: 20.0
```

#### 3. **趨勢分析窗口**
```yaml
trend_analysis:
  short_window: 10      # 短期（10 epochs）
  medium_window: 50     # 中期（50 epochs）
  long_window: 100      # 長期（100 epochs）
```

#### 4. **監控行為**
```yaml
monitoring:
  refresh_interval: 5                # 刷新間隔（秒）
  
  display:
    show_all_metrics: false          # 只顯示主要指標
    show_trend_indicators: true      # 顯示趨勢箭頭
    show_eta: true                   # 顯示 ETA
    show_process_info: true          # 顯示進程資訊
  
  alerts:
    enable_warnings: true            # 啟用警告
    enable_nan_detection: true       # NaN 檢測
```

---

## 🔧 進階用法

### 1. 自定義閾值

編輯 `configs/monitoring.yml`，添加新指標：

```yaml
metrics:
  constraints:
    - name: "wall_loss"
      display_name: "Wall BC"
      threshold_warning: 5.0        # 當 > 5.0 時警告
      threshold_error: 20.0         # 當 > 20.0 時標記為錯誤
```

### 2. 添加新的日誌格式

如果你的日誌格式不同，可在配置中添加新模式：

```yaml
log_parsing:
  loss_patterns:
    - "([\\w_]+):\\s*([\\d.e+-]+)"      # key: value
    - "([\\w_]+)=([\\d.e+-]+)"          # key=value
    - "Loss_([\\w_]+)\\s*([\\d.e+-]+)"  # 自定義格式
```

### 3. 在腳本中使用

```python
from pinnx.utils.training_monitor import TrainingMonitor

# 初始化
monitor = TrainingMonitor(config_path="configs/monitoring.yml")

# 取得所有訓練狀態
all_status = monitor.get_all_training_status()

# 取得特定訓練
status = monitor.get_training_status("test_rans_phase6c_v1")

# 格式化報告
report = monitor.format_status_report(status)
print(report)
```

---

## 📊 輸出解讀

### 1. **進程資訊**
- **🟢 Active Process**: 訓練正在運行
- **🔴 No Active Process**: 訓練已結束或未啟動
- **PID**: 進程 ID（可用於 `kill`）
- **CPU**: CPU 使用率（正常應 > 100%，多核）
- **Memory**: 記憶體使用量

### 2. **訓練進度**
- **Epoch**: 當前/總 epochs
- **進度百分比**: 訓練完成度
- **ETA**: 預計剩餘時間（基於最近 20 epochs）

### 3. **趨勢指示器**
- **↓** (下降): Loss 在降低（好）
- **↑** (上升): Loss 在增加（需關注）
- **→** (持平): Loss 穩定

### 4. **警告類型**
- **⚠️  NaN**: 數值變為 NaN（需立即處理）
- **⚠️  Inf**: 數值溢出
- **⚠️  超過警告閾值**: 超過 `threshold_warning`
- **🔥 超過錯誤閾值**: 超過 `threshold_error`

---

## 🛠️ 故障排除

### 問題 1: 找不到訓練
```
⚠️  找不到訓練: test_rans_phase6c_v1
```

**解決方案：**
1. 檢查訓練是否正在執行：`ps aux | grep train.py`
2. 確認配置名稱正確（不含 `.yml`）
3. 檢查日誌文件是否存在：`ls log/test_rans_phase6c_v1/`

### 問題 2: 沒有指標顯示
```
⚠️  No metrics available
```

**原因：** 日誌文件可能尚未寫入或格式不匹配

**解決方案：**
1. 檢查日誌文件內容：`tail -50 log/<config_name>/training_stdout.log`
2. 確認日誌包含 `Epoch X/Y` 行
3. 調整 `configs/monitoring.yml` 的 `loss_patterns`

### 問題 3: ETA 不準確
**原因：** 訓練速度波動大

**解決方案：** 
- 等待更多 epochs（ETA 基於最近 20 epochs）
- 調整 `eta.rolling_average_window` 增加窗口

---

## 📚 相關文件

- **配置文件**: `configs/monitoring.yml`
- **核心模組**: `pinnx/utils/training_monitor.py`
- **腳本文檔**: `scripts/README.md`
- **專案指引**: `AGENTS.md`

---

## 🎯 最佳實踐

### 1. 訓練開始前
```bash
# 啟動訓練
nohup python scripts/train.py --cfg configs/phase6c.yml > training.log 2>&1 &

# 立即監控（持續模式）
python scripts/monitor_training.py --config phase6c --watch
```

### 2. 多實驗並行
```bash
# 同時訓練多個實驗
python scripts/train.py --cfg configs/exp1.yml &
python scripts/train.py --cfg configs/exp2.yml &
python scripts/train.py --cfg configs/exp3.yml &

# 監控所有實驗
python scripts/monitor_training.py --all --watch
```

### 3. 定期檢查
```bash
# 添加到 cron 或在另一終端運行
watch -n 10 'python scripts/monitor_training.py --all'
```

---

## 🔮 未來計畫

- [ ] Web Dashboard（實時圖表）
- [ ] Email/Slack 通知
- [ ] 歷史趨勢資料庫
- [ ] 自動檢查點評估觸發
- [ ] GPU 使用監控整合

---

**最後更新**: 2025-10-14  
**版本**: 1.0.0
