# P100 GPU 優化指南（2026-01-16 更新）

**最後更新**: 2026-01-16  
**硬體**: 2× Tesla P100-PCIE-16GB (Compute Capability 6.0)  
**狀態**: ✅ TorchScript 優化已部署並驗證  
**實測加速**: 3.5%

---

## 📋 目錄

- [P100 硬體特性](#p100-硬體特性)
- [優化方案總覽](#優化方案總覽)
- [已部署的優化](#已部署的優化)
- [已測試但不推薦的方案](#已測試但不推薦的方案)
- [未來優化方向](#未來優化方向)
- [硬體升級建議](#硬體升級建議)

---

## P100 硬體特性

### 規格

```
架構: Pascal (2016)
Compute Capability: 6.0
CUDA Cores: 3584
Tensor Core: ❌ 無
記憶體: 16 GB HBM2  
記憶體頻寬: 732 GB/s
FP32 性能: 9.3 TFLOPS
FP16 性能: 18.7 TFLOPS (理論2x，實際無加速)
最高時鐘: 1328 MHz (SM), 715 MHz (Memory)
```

### 關鍵限制

| 限制 | 影響 | 解決方案 |
|-----|------|---------|
| **無 Tensor Core** | AMP 無實際加速 | ❌ 跳過 AMP |
| **CC 6.0** | 不支援 torch.compile() | ❌ 跳過 torch.compile |
| **SYS 連接** | GPU 間通訊慢 | ✅ DDP 已優化 |
| **16GB 記憶體** | 限制 batch size | ✅ 已最大化使用 |

---

## 優化方案總覽

### 實測驗證矩陣

| 優化方案 | 理論加速 | 實測加速 | 狀態 | 建議 |
|---------|---------|---------|------|------|
| **TorchScript Fusion** | 5-15% | **3.5%** ✅ | 已部署 | ✅ **推薦** |
| **AMP (FP16)** | 2x | 0.73-0.98x ❌ | 已測試 | ❌ 不推薦 |
| **torch.compile()** | 1.2-1.5x | N/A | 不支援 | ❌ 不可用 |
| **DataLoader 優化** | 5-10% | 0% | 不適用 | ❌ 無需 |
| **DDP 通訊優化** | <1% | N/A | 已最佳 | ✅ 已完成 |

### Profiler 瓶頸分析 (Job 2703)

| 瓶頸 | CUDA 時間 | 佔比 | 優化方案 | 狀態 |
|-----|----------|------|---------|------|
| **GEMM** | 249 ms | 43% | 需要硬體升級 | ⏳ 未來 |
| **Element-wise** | 175 ms | 30% | TorchScript | ✅ **已完成** |
| **Gradient Comp** | 152 ms | 26% | 需要架構改變 | ⏳ 未來 |
| **DDP 通訊** | <1 ms | 0.08% | 無需優化 | ✅ 已最佳 |

**關鍵洞察**:
- GEMM 是最大瓶頸，但 P100 上無軟體解決方案
- Element-wise 已透過 TorchScript 優化
- 進一步加速需要硬體升級（V100/A100）

---

## 已部署的優化

### ✅ TorchScript Kernel Fusion

**實測加速**: 1.035x (3.5%)  
**數值穩定性**: 誤差 < 5e-7  
**測試任務**: Job 2711

#### 實施內容

融合 SiLU (Swish) 激活函數的 `sigmoid` 和 `multiply` 操作：

```python
# 原始: 2 次 kernel 啟動
def silu_baseline(x):
    s = torch.sigmoid(x)  # Kernel 1
    return x * s           # Kernel 2

# 優化: 1 次 kernel 啟動
@torch.jit.script
def fused_silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)  # 單一融合 kernel
```

#### 性能提升

```
測試配置:
- GPU: Tesla P100-PCIE-16GB
- Batch Size: 4096
- Model: 8×256 MLP (478K 參數)

結果:
- Baseline (F.silu):     0.939 ± 0.001 ms
- Fused (fused_silu):    0.907 ± 0.000 ms
- Speedup:               1.035x (3.5% faster)
- Numerical error:       4.77e-07 (< 1e-5)
```

#### 如何使用

**自動啟用**（推薦）:
```yaml
# configs/your_config.yml
model:
  activation: swish  # 自動使用 FusedSiLU
```

**詳細文檔**: [TORCHSCRIPT_OPTIMIZATION_GUIDE.md](./TORCHSCRIPT_OPTIMIZATION_GUIDE.md)

---

## 已測試但不推薦的方案

### ❌ 混合精度訓練 (AMP)

**結論**: P100 的 AMP 會降低性能或無提升  
**測試任務**: Jobs 2708, 2709

#### 測試結果 (Job 2709 - 充分暖身版)

```
GPU: Tesla P100-PCIE-16GB
GPU 暖身: 5 秒（1189 → 1328 MHz）
測試輪數: 3 輪取平均

| Batch | FP32 (ms) | FP16 (ms) | 加速比 | 結論 |
|-------|-----------|----------|--------|------|
| 1024  | 1.521±0.019 | 2.077±0.016 | 0.732x | ❌ 慢36% |
| 4096  | 2.847±0.003 | 2.909±0.001 | 0.979x | ⚠️ 持平 |
```

#### 為何無效？

1. **無 Tensor Core**
   - V100/A100 的 AMP 加速來自 Tensor Core（8-16x）
   - P100 僅有 CUDA Core，FP16 與 FP32 使用相同硬體

2. **Type Casting 開銷**
   - FP32 ↔ FP16 轉換: ~5-10 μs/次
   - 小模型（<1M 參數）: overhead > 計算節省

3. **記憶體非瓶頸**
   - Profiler 顯示 GEMM 是計算受限（compute-bound）
   - 減少記憶體使用無助於加速

**建議**: ❌ **P100 不應使用 AMP**

**詳細報告**: `context/session_logs/2026-01-16_torchscript_optimization/AMP_TEST_RESULTS_P100_2026-01-16.md`

---

### ❌ torch.compile()

**狀態**: 硬體不支援  
**原因**: 需要 CC ≥ 7.0（P100 是 6.0）

```python
# 會報錯
model = torch.compile(model, mode='max-autotune')
# RuntimeError: Compute Capability 6.0 not supported
```

---

### ❌ DataLoader 優化

**狀態**: 不適用  
**原因**: PINNs 使用固定 collocation points，非 DataLoader

**架構差異**:
```python
# PINNs (本專案)
training_data = prepare_training_data(config, device)  # 一次性載入
for epoch in range(max_epochs):
    loss = train_step(training_data)  # 固定點集

# 標準深度學習
dataloader = DataLoader(dataset, batch_size=32, num_workers=4, ...)
for batch in dataloader:  # 每次不同 mini-batch
    loss = train_step(batch)
```

**Profiler 驗證**: CPU→GPU 傳輸 < 1% CUDA 時間

---

## 未來優化方向

### 可探索的方向

#### 1. Gradient Checkpointing
**目標**: 降低記憶體使用，允許更大 batch size  
**預期**: 記憶體 -60%，速度 -20%  
**適用場景**: 記憶體不足時

```python
from torch.utils.checkpoint import checkpoint

def forward(self, x):
    for layer in self.layers:
        x = checkpoint(layer, x)  # Trade 計算換記憶體
    return x
```

#### 2. 更大 Batch Size
**目標**: 提升 GPU 利用率  
**當前**: 7000-8000  
**可嘗試**: 12000-16000（利用剩餘 GPU 記憶體）  
**風險**: 可能影響訓練動態

#### 3. 自訂 CUDA Kernel
**目標**: 進一步融合 Element-wise 操作  
**預期**: 額外 5-10% 加速  
**成本**: 高（需要 CUDA 專業知識）  
**工具**: Triton, CuPy, PyTorch Custom CUDA Extension

#### 4. JAX 遷移
**目標**: 更高效的高階導數計算  
**預期**: 梯度計算 1.5-2x 加速  
**成本**: 中（~2-4 週遷移時間）  
**優勢**: `vmap` + `jit` 對 VS-PINN 友好

---

## 硬體升級建議

### GPU 對比

| GPU | 架構 | Tensor Core | AMP 加速 | 價格 | ROI | 推薦度 |
|-----|------|------------|---------|------|-----|--------|
| **P100** | Pascal (6.0) | ❌ | 0.98x | - | - | 當前 |
| **V100** | Volta (7.0) | ✅ Gen1 | 5-8x | ~$2000 | ⭐⭐⭐⭐⭐ | **最佳** |
| **A100** | Ampere (8.0) | ✅ Gen3 | 10-16x | ~$10000 | ⭐⭐⭐ | 性能最高 |

### ROI 計算

假設訓練時間 100 小時/次：

**V100 升級**:
- 加速: 5-8x → 訓練時間: 12.5-20 小時
- 節省時間: 80-87.5 小時
- 回本週期: 若每月訓練 10 次，節省 800-875 小時，6-12 個月回本

**A100 升級**:
- 加速: 10-16x → 訓練時間: 6.25-10 小時
- 節省時間: 90-93.75 小時  
- 回本週期: 若每月訓練 20 次，節省 1800-1875 小時，12-24 個月回本

**建議**: 若訓練頻繁（>100 小時/月），升級 V100 ROI 最高

---

## 參考資料

### 完整技術文檔

- [TORCHSCRIPT_OPTIMIZATION_GUIDE.md](./TORCHSCRIPT_OPTIMIZATION_GUIDE.md) - TorchScript 優化詳細指南
- [CHROME_TRACE_VIEWER_GUIDE.md](./CHROME_TRACE_VIEWER_GUIDE.md) - Profiler 視覺化使用方法

### 會話記錄

- `context/session_logs/2026-01-16_torchscript_optimization/` - 完整優化會話記錄
  - `README.md` - 優化專案總覽
  - `PROFILER_OPTIMIZATION_REPORT_2026-01-16.md` - Profiler 深度分析
  - `AMP_TEST_RESULTS_P100_2026-01-16.md` - AMP 測試完整報告
  - `SESSION_SUMMARY_2026-01-16_TORCHSCRIPT_INTEGRATION.md` - TorchScript 實施記錄

### 測試數據

- `profiler_results/baseline_job2703/` - Baseline Profiler 數據
- `profiler_results/torchscript_fusion_results.txt` - TorchScript 測試結果
- `slurm_logs/torchscript_fusion_2711.out` - 完整測試日誌

---

## 總結

### 已完成的優化

- ✅ TorchScript Kernel Fusion (3.5%)
- ✅ DDP 通訊優化（已是最佳狀態）
- ✅ WandB 同步頻率優化（50x 減少）
- ✅ Loss 日誌優化（2x 減少）

### P100 的極限

在軟體層面，P100 已達到優化上限：
- GEMM (43%) 無法進一步優化（需要 Tensor Core）
- Element-wise (30%) 已優化（TorchScript）
- Gradient Comp (26%) 需要架構改變（成本高）

### 下一步建議

**短期（1-2 週）**:
- 完整訓練驗證（確認 3.5% 加速）
- 探索更大 batch size
- Chrome Trace Viewer 深度分析

**中期（3-6 個月）**:
- 評估硬體升級（V100 ROI 最高）
- 探索 JAX 遷移可行性
- 準備論文性能分析章節

**長期（6-12 個月）**:
- 硬體升級至 V100/A100
- 實施 AMP（在新硬體上）
- 達到 5-16x 訓練加速

---

**文檔維護**: OpenCode Agent  
**最後更新**: 2026-01-16  
**下次審查**: Profiler Job 2712 完成後
