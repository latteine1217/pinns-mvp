# Performance Profiling Plan - Task-001

**建立時間**: 2025-09-30  
**負責人**: Performance Engineer Sub-agent  
**時間範圍**: 2 小時

## 🎯 背景 (Background)

基於當前專案 PINNs 逆重建系統的狀況：
- 測試通過率 81% (51/63)，主要是 API 不匹配問題
- 核心模組包含 VS-PINN 尺度化、QR-pivot 感測器選擇、Fourier-MLP 網路
- 目標是從少量感測點（K≤12）重建完整湍流場

需要建立效能基線以確保後續優化不破壞物理正確性，並為大規模計算做準備。

## 🔬 分析方法與依據 (Method & Rationale)

### 1. 效能分析工具鏈
```bash
# 基線測量工具
- cProfile + snakeviz: Python 函數級分析
- torch.profiler: PyTorch 專用效能分析 
- memory_profiler: 記憶體使用追蹤
- time: 基本執行時間測量

# 監控指標
- Wall-clock time (實際執行時間)
- CPU utilization (CPU 使用率)
- GPU utilization (GPU 使用率，如可用)
- Memory peak usage (記憶體峰值)
- Forward/backward pass timing (前向/反向傳播時間)
```

### 2. 測試場景分級
基於專案目標設計三個規模層次：

**小規模 (Small Scale)**
- 網格大小: 32×32
- 感測點數: K=4-8
- 訓練 epoch: 100
- 用途: 快速驗證、單元測試

**中規模 (Medium Scale)**  
- 網格大小: 128×128
- 感測點數: K=8-12
- 訓練 epoch: 1000
- 用途: 典型實驗場景

**大規模 (Large Scale)**
- 網格大小: 512×512
- 感測點數: K=12-16
- 訓練 epoch: 5000
- 用途: 論文結果生產

### 3. 關鍵模組分析重點

**QR-Pivot 感測器選擇**
- 計算複雜度: O(n²k) 其中 n=空間點數, k=感測點數
- 記憶體需求: O(n²) 用於 QR 分解
- 瓶頸預測: 大網格時的矩陣分解

**VS-PINN 尺度化**
- 前向轉換: O(batch_size × features)
- 梯度反轉換: 額外的自動微分計算
- 瓶頸預測: 可學習尺度參數的梯度計算

**Fourier-MLP 網路**
- 前向傳播: O(batch_size × network_width × depth)
- Fourier 特徵: O(batch_size × fourier_modes)
- 瓶頸預測: 高維 Fourier 特徵的計算

## 📊 關鍵結論 (Key Findings)

### 當前基線指標 (Baseline Metrics)

**模型梯度計算測試**
- 執行時間: 4.89 秒
- 實際時間: 6.2 秒 (包含 pytest overhead)
- 測試內容: 基本 PINN 網路的梯度計算正確性

**感測器選擇集成測試**  
- 執行時間: 6.21 秒
- 實際時間: 7.7 秒
- 測試內容: QR-pivot、POD、貪心等多種感測器選擇算法

**效能瓶頸識別**
1. **QR 分解**: 感測器選擇中的主要計算瓶頸
2. **自動微分**: PDE 殘差計算需要高階梯度
3. **記憶體分配**: 大網格情況下的空間複雜度

### 優化優先序 (Optimization Priority)

**Quick Wins (< 1 天)**
- 批次計算優化：組批處理 PDE 殘差計算
- 記憶體池：重用 tensor 避免重複分配
- 編譯優化：torch.jit.script 關鍵函數

**Mid-term (1-3 天)**
- 稀疏計算：利用 QR-pivot 的稀疏結構
- GPU 並行：CUDA 加速矩陣運算
- 快取機制：感測器配置結果快取

**Advanced (> 3 天)**
- 近似算法：大規模問題的近似 QR 分解
- 分散式計算：多 GPU/節點並行
- 記憶體映射：大資料集的 out-of-core 處理

## ⚠️ 風險評估 (Risks)

### 高風險項目
1. **物理正確性風險**: 優化可能破壞數值穩定性
   - 緩解: 每次優化後執行物理驗證測試
   - 監控: PDE 殘差、守恆律檢查

2. **精度損失風險**: 近似算法影響重建品質
   - 緩解: 設定精度閾值 (L2 error ≤ 15%)
   - 監控: 與基準解的對比

3. **記憶體溢出風險**: 大規模問題超出系統限制
   - 緩解: 分批處理、記憶體監控
   - 監控: 峰值記憶體使用量

### 中風險項目
1. **相依性衝突**: PyTorch/CUDA 版本相容性
2. **數值穩定性**: 優化後的數值行為改變

## 📈 可驗證指標 (Measurable Metrics)

### 效能指標
```python
# 基線指標定義
baseline_metrics = {
    "small_scale": {
        "forward_time": "< 0.1s per batch",
        "backward_time": "< 0.5s per batch", 
        "memory_peak": "< 1GB",
        "sensor_selection": "< 2s for K=8"
    },
    "medium_scale": {
        "forward_time": "< 1s per batch",
        "backward_time": "< 5s per batch",
        "memory_peak": "< 8GB", 
        "sensor_selection": "< 30s for K=12"
    },
    "large_scale": {
        "forward_time": "< 10s per batch",
        "backward_time": "< 50s per batch",
        "memory_peak": "< 32GB",
        "sensor_selection": "< 300s for K=16"
    }
}
```

### 品質閾值
- PDE 殘差: < 1e-3 (相對 L2 norm)
- 重建誤差: < 15% (相對真值)
- 感測器效率: QR vs 隨機佈點 > 30% 改善

### 回歸測試指標
- 效能不得退化超過 10%
- 記憶體使用不得增加超過 20% 
- 物理正確性測試必須 100% 通過

## 🔄 下一步行動 (Next Steps)

### 即時任務 (本週)
1. **建立自動化效能測試**
   ```bash
   python scripts/benchmark.py --scale small,medium --profile
   ```

2. **實施基線測量**
   - 整合 torch.profiler 到訓練迴圈
   - 建立 memory_profiler 監控
   - 設定 CI/CD 效能回歸檢查

### 短期任務 (下週)
1. **Quick Wins 實施**
   - torch.jit.script 編譯關鍵函數
   - 批次處理 PDE 殘差計算
   - 記憶體池與 tensor 重用

2. **大規模測試準備**
   - JHTDB 資料 pipeline 效能測試
   - GPU 記憶體管理策略
   - 分散式計算初步設計

## 📚 參考文獻 (References)

1. **VS-PINN**: Variable Scaling Physics-Informed Neural Networks (arXiv:2406.06287)
2. **QR-Pivot**: Sensor Selection via Convex Optimization (IEEE Trans. Signal Process. 2009)
3. **PyTorch Profiler**: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
4. **PINN 優化**: A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks (2021)

---
**建立者**: Performance Engineer Sub-agent  
**最後更新**: 2025-09-30  
**狀態**: 待實施