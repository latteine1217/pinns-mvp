# 效能分析工具使用說明

## 📊 概述

本目錄包含兩個效能分析腳本，用於診斷訓練循環的時間瓶頸。

---

## 🔧 工具列表

### 1. `profile_training_loop.py` - 訓練循環總體分析

**用途**: 測量訓練循環各主要組件的時間分布

**測量項目**:
- 數據傳輸時間
- 前向傳播時間
- 損失計算時間
- 損失組合與權重調整時間
- 反向傳播與優化時間

**使用方法**:
```bash
python scripts/profile_training_loop.py \
    --config configs/profiling_test.yml \
    --epochs 10
```

**SLURM 提交**:
```bash
sbatch slurm_profile_training.sh
```

---

### 2. `profile_residual_detailed.py` - PDE Residual 詳細分析

**用途**: 深入測量 PDE residual 計算的詳細時間

**測量項目**:
- 速度與壓力提取時間
- 梯度計算時間 (`compute_all_gradients_2d`)
- 動量方程 x 方向計算時間
- 動量方程 y 方向計算時間
- 連續性方程計算時間
- PDE residual 總時間

**使用方法**:
```bash
python scripts/profile_residual_detailed.py \
    --config configs/profiling_test.yml \
    --epochs 20
```

**SLURM 提交**:
```bash
sbatch slurm_profile_residual_detailed.sh
```

---

## 📈 輸出範例

### 訓練循環總體分析輸出

```
================================================================================
⏱️  訓練循環效能分析報告
================================================================================
組件名稱                              平均時間(ms)     佔比(%)    調用次數
--------------------------------------------------------------------------------
0_total_step                              530.25        100.0         10
5_backward_optimize                       180.32         34.0         10
2_forward_pass                            150.18         28.3         10
3_compute_losses                          120.45         22.7         10
4_combine_losses                           60.12         11.3         10
1_data_transfer                            19.18          3.6         10
--------------------------------------------------------------------------------
總時間                                  5302.50 ms
================================================================================
```

### PDE Residual 詳細分析輸出

```
====================================================================================================
⏱️  PDE Residual 詳細效能分析報告
====================================================================================================
組件名稱                                平均(ms)      標準差(ms)    佔比(%)    調用次數
----------------------------------------------------------------------------------------------------
step_backward                             180.320      5.120        34.0         20
step_forward_pass                         150.180      3.450        28.3         20
residual_total                             45.230      1.230         8.5         20
residual_gradient_computation              38.120      1.050         7.2         20
step_combine_losses                        60.120      2.340        11.3         20
residual_momentum_x                         3.450      0.120         0.7         20
residual_momentum_y                         2.890      0.090         0.5         20
residual_continuity                         0.770      0.030         0.1         20
----------------------------------------------------------------------------------------------------
總時間                                  5302.500 ms
====================================================================================================
```

---

## 🔍 分析建議

### 如何判斷 PDE Residual 是否為瓶頸？

1. **查看 `residual_total` 的佔比**
   - 如果 < 10%：PDE residual 不是主要瓶頸
   - 如果 10-30%：中等影響，優化有價值
   - 如果 > 30%：主要瓶頸，優化效果顯著

2. **對比 `residual_gradient_computation` 與其他 residual 組件**
   - 如果梯度計算佔 residual 總時間 > 80%：向量化優化有效
   - 如果梯度計算佔 residual 總時間 < 50%：瓶頸在其他地方

3. **查看 `step_backward` 的佔比**
   - 如果反向傳播佔比 > 40%：考慮優化模型架構或使用混合精度訓練
   - 如果前向傳播 + 反向傳播 > 70%：模型計算是主要瓶頸，PDE residual 優化效果有限

---

## 📝 實驗建議

### 對比實驗設計

1. **Baseline vs Vectorized**
   ```bash
   # 1. 切換到 baseline 版本
   cp pinnx/losses/residuals_baseline.py pinnx/losses/residuals.py
   sbatch slurm_profile_residual_detailed.sh  # Job A
   
   # 2. 切換回 vectorized 版本
   git checkout pinnx/losses/residuals.py
   sbatch slurm_profile_residual_detailed.sh  # Job B
   
   # 3. 比較兩次運行的 residual_gradient_computation 時間
   ```

2. **不同批次大小測試**
   - 修改 `configs/profiling_test.yml` 中的 `N_pde`, `N_sensors`, `N_bc`
   - 測試 10k, 20k, 50k, 100k 點
   - 觀察 PDE residual 佔比是否隨批次大小變化

3. **不同模型大小測試**
   - 測試不同網路深度與寬度
   - 觀察前向/反向傳播時間變化
   - 確定是否模型計算是主導瓶頸

---

## 🚨 注意事項

1. **GPU 同步**: 腳本已自動插入 `torch.cuda.synchronize()`，確保計時準確
2. **預熱**: 建議至少運行 10 個 epochs，忽略前 2-3 個 epoch 的計時（GPU 預熱）
3. **測量噪聲**: 多次運行取平均值，減少測量誤差
4. **WandB 禁用**: 腳本自動禁用 WandB，避免日誌上傳影響計時

---

## 📊 結果解讀範例

### 場景 1: PDE Residual 不是瓶頸（當前情況）

```
residual_total:           45 ms  (8.5%)
step_forward_pass:       150 ms  (28.3%)
step_backward:           180 ms  (34.0%)
```

**結論**: 
- PDE residual 只佔 8.5%，優化它最多帶來 ~8% 的端到端加速
- 向量化將 residual 從 80ms → 45ms (1.77x)，但端到端只從 530ms → 500ms (1.06x)
- **建議**: 優化前向傳播（模型架構）或反向傳播（混合精度訓練）

### 場景 2: PDE Residual 是主要瓶頸

```
residual_total:          250 ms  (47%)
step_forward_pass:       100 ms  (19%)
step_backward:           120 ms  (23%)
```

**結論**:
- PDE residual 佔 47%，優化它可帶來顯著加速
- 向量化將 residual 從 250ms → 140ms (1.77x)，端到端從 530ms → 420ms (1.26x)
- **建議**: 向量化優化有價值，應該採用

---

## 🎯 下一步行動建議

基於當前實驗結果（E2E 無加速），建議執行：

1. ✅ **運行詳細效能分析** (已完成腳本準備)
   ```bash
   sbatch slurm_profile_residual_detailed.sh
   ```

2. 📊 **分析結果並確定真正瓶頸**
   - 如果前向傳播 > 30%：優化模型架構
   - 如果反向傳播 > 40%：考慮混合精度訓練或梯度累積
   - 如果 PDE residual < 10%：接受當前向量化（代碼質量提升，效能無損）

3. 🔬 **撰寫技術報告**
   - 記錄完整的效能分析結果
   - 解釋為何微基準測試與端到端結果不一致
   - 提出未來優化方向

---

## 📚 參考資料

- PyTorch Profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- CUDA Events Timing: https://pytorch.org/docs/stable/notes/cuda.html#cuda-semantics
- Autograd Performance: https://pytorch.org/docs/stable/notes/autograd.html
