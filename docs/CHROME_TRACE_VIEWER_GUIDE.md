# Chrome Trace Viewer 使用指南

**目標**: 視覺化分析 PyTorch Profiler 的詳細時間軸，識別 GPU 閒置時間和 CPU-GPU 同步瓶頸。

---

## 🚀 快速開始

### 1. 打開 Chrome Trace Viewer
在 Chrome 瀏覽器網址列輸入：
```
chrome://tracing
```

### 2. 載入 Trace 檔案
- 點擊左上角 **"Load"** 按鈕
- 選擇檔案：`profiler_results/trace.json` (14 MB)
- 等待載入完成（約 5-10 秒）

---

## 🔍 如何分析

### A. 時間軸概覽
載入後你會看到：
- **橫軸**: 時間（微秒）
- **縱軸**: 不同的執行緒和 CUDA Stream
  - `CPU Thread` - Python 主執行緒
  - `CUDA Stream 0` - GPU 計算
  - `autograd` - PyTorch 自動微分引擎

### B. 導航操作
| 操作 | 快捷鍵 | 說明 |
|-----|--------|------|
| **放大** | `W` 或滑鼠滾輪向上 | 查看更細節的操作 |
| **縮小** | `S` 或滑鼠滾輪向下 | 查看全局視野 |
| **左移** | `A` | 向左平移時間軸 |
| **右移** | `D` | 向右平移時間軸 |
| **點擊事件** | 滑鼠左鍵 | 查看該操作的詳細資訊 |

---

## 🎯 關鍵分析目標

### 1. 找出 GPU 閒置時間
**步驟**:
1. 放大到 `CUDA Stream 0` 軌道
2. 尋找 **空白區域**（GPU 沒有任何操作的時段）
3. 點擊前後的操作，檢查是否有 CPU 等待或數據傳輸

**預期發現**:
- 梯度計算之間的 CPU-GPU 同步點
- Optimizer step 之前的短暫空窗
- DDP all-reduce 通訊期間 GPU 閒置

**如何識別**:
```
Good (GPU 滿載):
  |████████████████████████████████|
  
Bad (GPU 閒置):
  |████    |    |████    |████|    |
       ↑ 空白 = 閒置
```

---

### 2. 分析 Gradient Computation
**步驟**:
1. 搜尋 `compute_all_gradients`（使用 Ctrl+F）
2. 點擊該事件，查看右側面板：
   - **Duration**: 持續時間（應為 ~30.5 ms）
   - **Args**: 傳入參數
3. 檢查該時段內的 CUDA kernels：
   - `aten::mm` (GEMM)
   - `aten::mul` (Element-wise)
   - `autograd::engine::evaluate_function`

**優化線索**:
- 若 GPU 使用率 <80%，代表計算未飽和
- 若有大量 `cudaStreamSynchronize`，代表同步過多
- 若 `aten::mm` 連續執行，代表 GEMM 優化良好

---

### 3. 檢查 CPU-GPU 同步瓶頸
**步驟**:
1. 尋找 `cudaStreamSynchronize` 或 `cudaDeviceSynchronize`
2. 點擊查看呼叫堆疊（Call Stack）
3. 識別呼叫來源：
   - `loss.item()` - 將 Tensor 轉為 Python float
   - `wandb.log()` - 日誌同步
   - `model.eval()` / `model.train()` - 模式切換

**警告信號**:
- 每次迭代有 >5 次同步
- 單次同步時間 >1 ms
- 同步操作不在關鍵路徑（如 logging）

---

### 4. DDP 通訊分析
**步驟**:
1. 搜尋 `ncclAllReduce` 或 `c10d::allreduce`
2. 測量通訊時間（應 <10% 總時間）
3. 檢查是否與計算重疊：
   - ✅ **良好**: 通訊與下一次 Forward Pass 重疊
   - ⚠️ **不良**: 通訊期間 GPU 完全閒置

**優化目標**:
- DDP bucket 大小調整（目前預設 25 MB）
- 通訊與計算重疊（使用 `gradient_as_bucket_view=True`）

---

## 📊 參考指標

### 理想的時間分配（DDP 雙 GPU）
| 階段 | 目標佔比 | 當前值 (Job 2703) |
|-----|---------|------------------|
| Forward Pass | 5-10% | 1.4% ✅ |
| Gradient Computation | 30-40% | 26.4% ✅ |
| Backward Pass | 10-15% | <0.01% ✅ |
| Optimizer Step | 5-10% | 0.2% ✅ |
| DDP Communication | <10% | 未直接測量 |
| CPU-GPU Sync | <5% | 未直接測量 ⚠️ |

### GPU 利用率目標
- **目標**: >85% 時間在執行 CUDA kernels
- **測量方法**: 
  1. 在 Trace Viewer 中選取一個完整的訓練 step
  2. 查看 CUDA Stream 的 **佔空比**（Duty Cycle）
  3. 計算：`(CUDA kernel 總時間) / (Wall Clock 時間)`

---

## 🛠️ 常見問題排查

### Q1: Trace 檔案載入失敗
**原因**: 檔案過大（>100 MB）或格式錯誤

**解決**:
```bash
# 檢查檔案大小
ls -lh profiler_results/trace.json

# 若過大，重新生成時減少 Profiler 步數
# 在 train_with_profiler.py 中修改:
activities=[...],
schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1)  # 減少 active 步數
```

---

### Q2: 看不到 CUDA kernels
**原因**: Profiler 未啟用 CUDA 追蹤

**檢查**:
```python
# train_with_profiler.py 應包含:
activities=[
    torch.profiler.ProfilerActivity.CPU,
    torch.profiler.ProfilerActivity.CUDA,  # ← 必須啟用
]
```

---

### Q3: 自訂標記 (custom markers) 未出現
**原因**: 使用 `prof.record_function()` 而非 `torch.autograd.profiler.record_function()`

**修正**:
```python
# 正確寫法
with torch.autograd.profiler.record_function("### My Custom Label"):
    # your code
```

---

## 📚 進階分析

### A. 匯出統計資料
在 Trace Viewer 中：
1. 點擊右上角 **"Export"**
2. 選擇格式：`JSON` 或 `HTML`
3. 用 Python 解析 JSON 進行程式化分析

### B. 對比兩次 Profiler Run
```bash
# 使用 diff 工具對比
diff profiler_results_baseline/stacks.txt profiler_results_optimized/stacks.txt

# 或用 Python 腳本解析並生成對比表格
```

### C. 自動化瓶頸檢測
創建腳本來：
1. 解析 `trace.json`
2. 計算 GPU 空閒比例
3. 識別最耗時的前 10 個操作
4. 生成優化建議報告

---

## 🎓 學習資源

- [Chrome Tracing Format 規範](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/)
- [PyTorch Profiler 官方教學](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) - 更強大的替代工具

---

**提示**: 第一次使用 Trace Viewer 可能會被大量資訊淹沒。建議先專注在：
1. **找出最大的色塊** → 最耗時的操作
2. **找出空白區域** → GPU 閒置
3. **找出重複模式** → 優化機會

祝分析順利！ 🚀
