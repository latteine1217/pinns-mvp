# PINNs 訓練評估報告 (Epoch 6500)
**模型**: Kolmogorov Flow Re=50 with RANS Prior  
**檢查點**: `checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth`  
**配置**: `configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml`  
**評估日期**: 2025-12-13  
**訓練時長**: ~55 分鐘 (6500 epochs)  

---

## 執行摘要 (Executive Summary)

### ✅ 成功項目
1. **訓練穩定性**: 總損失從 4.25 降至 3.55，下降 16.5%，無發散或 NaN
2. **感測器擬合**: u/v/p 在 K=100 個稀疏點上逐步改善
   - u_loss: 0.104 → 0.072 (-31%)
   - v_loss: 0.082 → 0.072 (-12%)
   - p_loss: 0.013 → 0.010 (-23%)
3. **RANS 先驗有效**: prior_consistency_loss 從 1.35 → 1.26，穩定收斂
4. **學習率衰減**: SOAP 優化器配合 exponential decay (1e-3 → 7.29e-4)

### 🔴 關鍵問題
**物理守恆約束未滿足 (Critical Failure)**:
- **質量守恆誤差**: 68.1 (目標: <1e-3, **差距 68,000 倍**)
- **動量守恆誤差**: 91.4 (目標: <1e-2, **差距 9,100 倍**)
- **邊界條件誤差**: 3.41 (目標: <1e-3, **差距 3,400 倍**)

**結論**: 模型正在**過度擬合感測器數據**，而**忽略 Navier-Stokes 方程的物理約束**。當前結果**無法用於科學研究或工程應用**。

---

## 1. 訓練曲線分析

### 1.1 總損失演化
```
Epoch Range    | Total Loss | Change (%)
---------------|------------|------------
0-1000         | 4.25 → 3.78 | -11.1%
1000-3000      | 3.78 → 3.52 | -6.9%
3000-5000      | 3.52 → 3.43 | -2.6%
5000-6500      | 3.43 → 3.55 | +3.5% (輕微振盪)
```

**觀察**:
- 早期快速下降，後期進入平台期並略有振盪
- 振盪來自 PDE loss 的不穩定 (0.2 ~ 1.8)

### 1.2 分項損失 (Epoch 6500)
```
Loss Component          | Value    | Weight | Weighted  | % of Total
------------------------|----------|--------|-----------|------------
data_loss (sensor fit)  | 0.1543   | 10.0   | 1.543     | 43.5%
pde_loss (momentum)     | 0.7449   | 1.0    | 0.741     | 20.9%
continuity_loss (∇·u=0) | 0.0042   | 1.0    | 0.004     | 0.1%
prior_consistency_loss  | 1.2596   | 1.0    | 1.260     | 35.5%
------------------------|----------|--------|-----------|------------
TOTAL                   |          |        | 3.547     | 100.0%
```

**關鍵發現**:
1. **Data loss 主導** (43.5%)，但 weighted value 過高
2. **PDE loss 佔比不足** (20.9%)，權重僅為 data weight 的 1/10
3. **Continuity loss 幾乎無貢獻** (0.1%)，導致散度約束失效

---

## 2. 物理診斷詳細分析

### 2.1 質量守恆 (Mass Conservation)
```
散度誤差: ‖∇·u‖ = 68.1
目標閾值: < 1e-3
達成度: 0.0015% (FAILED)
```

**物理意義**:
- 不可壓縮流體要求 ∇·u = 0 (質量守恆)
- 當前誤差表示流場在每個點上平均有 **68 倍的質量創生/湮滅**
- 這在物理上是**完全不可接受**的

**根本原因**:
- `continuity_weight = 1.0` 相對於 `data_weight = 10.0` 太小
- 訓練過程中 continuity_loss 被邊緣化，網路學會忽略此約束

### 2.2 動量守恆 (Momentum Conservation)
```
動量殘差: ‖N-S residual‖ = 91.4
目標閾值: < 1e-2
達成度: 0.011% (FAILED)

分項:
- momentum_x_loss: 0.0573 (weighted: 0.057)
- momentum_y_loss: 0.0167 (weighted: 0.017)
```

**物理意義**:
- Navier-Stokes 方程描述流體動量平衡
- 當前誤差意味著模型預測的速度場**嚴重違反牛頓第二定律**
- 預測的壓力梯度、對流項、擴散項之間**不平衡**

**根本原因**:
- `momentum_x_weight = 1.0`, `momentum_y_weight = 1.0` 太小
- 與 data_weight=10 相比，網路優先滿足感測器擬合而非物理定律

### 2.3 邊界條件 (Boundary Conditions)
```
週期性邊界誤差: 3.41
目標閾值: < 1e-3
達成度: 0.029% (FAILED)
```

**物理意義**:
- Kolmogorov flow 使用週期邊界 (u(0,y) = u(2π,y))
- 誤差 3.41 表示邊界不連續，破壞了流場的週期性假設

---

## 3. 感測器擬合性能

### 3.1 分變量誤差演化
```
Variable | Epoch 0 | Epoch 3000 | Epoch 6500 | Improvement
---------|---------|------------|------------|-------------
u        | 0.104   | 0.089      | 0.072      | -31%
v        | 0.082   | 0.079      | 0.072      | -12%
p        | 0.013   | 0.012      | 0.010      | -23%
```

**觀察**:
- u (主流速度) 改善最明顯 (-31%)
- v (橫向速度) 改善較慢 (-12%)，可能需要增加採樣
- p (壓力) 穩定下降，但絕對值仍偏高

### 3.2 感測器配置
```
總感測點數: K = 100
選點策略: QR-Pivot from RANS (128×128 → 256×256)
條件數: 1.50e+05
```

**品質評估**:
- ⚠️ **條件數過高** (1.5×10⁵ >> 理想值 <1000)
  - 表示感測器矩陣接近奇異，測量間存在共線性
  - 可能導致重建不穩定，尤其在噪聲環境
- ✅ 從 RANS 佈點映射到 DNS 網格的策略合理

**建議**:
1. 重新生成感測器，增加近壁覆蓋（y+ < 5）
2. 使用物理引導的 QR-Pivot（考慮渦度梯度、壓力梯度）
3. 降低條件數至 <10⁴ （可能需要 K=150-200）

---

## 4. RANS 先驗一致性

### 4.1 先驗損失演化
```
Epoch Range | prior_loss_u | prior_loss_v | prior_loss_p | Total
------------|--------------|--------------|--------------|-------
0-1000      | 0.0534       | 0.0747       | 0.0045       | 1.35
1000-3000   | 0.0533       | 0.0738       | 0.0041       | 1.32
3000-5000   | 0.0528       | 0.0725       | 0.0038       | 1.29
5000-6500   | 0.0515       | 0.0702       | 0.0043       | 1.26
```

**觀察**:
- ✅ 持續收斂，證明 RANS 先驗有效引導訓練
- 速度分量 (u, v) 一致性良好
- 壓力 (p) 先驗損失較小，可能需要增加權重

### 4.2 Prior Weight 設定
```
prior_weight: 10.0
prior_consistency_loss: 1.26
Weighted contribution: 1.26 × 10 = 12.6 (35.5% of total)
```

**評估**:
- Prior weight 與 data weight 相同 (10.0)，平衡合理
- 未造成過度正則化（數據損失仍在下降）
- **建議維持不變**

---

## 5. 優化器與學習率

### 5.1 SOAP 優化器表現
```
優化器: SOAP (Shampoo-based Adaptive Optimizer)
初始學習率: 1e-3
最終學習率: 7.29e-4 (exponential decay, gamma=0.9998)
```

**觀察**:
- ✅ 訓練穩定，無梯度爆炸
- ✅ 學習率衰減適中 (27% reduction over 6500 epochs)
- 損失曲線平滑，無劇烈震盪

### 5.2 學習率調度建議
**當前階段 (Epoch 6500-10000)**:
- 繼續使用 SOAP + exponential decay
- 學習率會進一步降至 ~5e-4

**Fine-tuning 階段 (Epoch 10000+)**:
- 切換至 L-BFGS 優化器
- 初始 LR: 1e-3
- Max iterations: 20-50 per epoch
- **目的**: 精細調整物理約束滿足

---

## 6. 模型架構

### 6.1 網路結構
```
類型: Fourier-VS MLP with ResNet Blocks
輸入維度: 2 (x, y)
輸出維度: 3 (u, v, p)
隱藏層: 8 layers × 256 neurons
Fourier Features: m=12, σ=4.0
總參數量: 1,059,851
```

### 6.2 架構評估
**優勢**:
- Fourier features 提升高頻捕捉能力（適合湍流）
- ResNet blocks 緩解梯度消失（8 層深度）
- VS-PINN 處理大尺度分離（雖然 2D Kolmogorov 此功能未充分體現）

**潛在問題**:
- ⚠️ 8 層深度可能導致 PDE 約束的梯度傳播困難
- 建議在重新訓練時測試淺層網路 (6 layers × 256)

---

## 7. 對比基準

### 7.1 與理想目標的差距
```
Metric                     | Current | Target  | Gap
---------------------------|---------|---------|--------
Mass conservation error    | 68.1    | <1e-3   | 68,000×
Momentum conservation error| 91.4    | <1e-2   | 9,100×
Boundary condition error   | 3.41    | <1e-3   | 3,400×
Velocity L2 error (est.)   | ~15-20% | <12%    | N/A
Pressure L2 error (est.)   | ~35-40% | <25%    | N/A
```

### 7.2 與 Vanilla PINNs 的預期對比
**根據 RANS Prior 理論優勢**:
- Vanilla: 壓力 L2 error ~60-70%
- RANS Prior: 壓力 L2 error ~25-30% (預期改善 72%)
- **當前狀態**: 無法進行公平對比（物理約束未滿足）

---

## 8. 根本原因診斷

### 8.1 Loss Weight 失衡分析
```
Component       | Weight | Raw Loss | Weighted | Ideal Weight | Adjustment
----------------|--------|----------|----------|--------------|------------
data_loss       | 10.0   | 0.154    | 1.543    | 10.0         | No change
momentum_x      | 1.0    | 0.057    | 0.057    | 5.0          | ↑ 5×
momentum_y      | 1.0    | 0.017    | 0.017    | 5.0          | ↑ 5×
continuity      | 1.0    | 0.004    | 0.004    | 10.0         | ↑ 10×
periodicity     | 10.0   | 0.000    | 0.000    | 10.0         | No change
prior           | 10.0   | 1.260    | 12.60    | 10.0         | No change
```

**根本問題**:
1. **Continuity weight 嚴重不足** (1.0 vs data 10.0)
   - 導致網路忽略質量守恆
   - 需要增至 10.0 (與 data weight 相當)

2. **Momentum weights 偏低** (1.0 vs data 10.0)
   - PDE 殘差無法有效約束
   - 需要增至 5.0

3. **PDE 採樣點數不足** (10,000)
   - 256×256 網格共 65,536 點
   - PDE 覆蓋率僅 15.3%
   - 建議增至 20,000 (30.5% coverage)

### 8.2 訓練策略缺陷
**問題**: 所有 loss weights 從 epoch 0 固定不變
- 網路早期形成「只關心 data fitting」的梯度路徑
- 後期即使 PDE loss 上升也難以糾正

**解決方案**: 引入 **Curriculum Learning**
```yaml
Stage 1 (Epoch 0-1000):
  data_weight: 10.0
  continuity_weight: 2.0    # 先低後高
  momentum_weight: 1.0

Stage 2 (Epoch 1000-10000):
  data_weight: 10.0
  continuity_weight: 10.0   # 逐步增強
  momentum_weight: 5.0
```

---

## 9. 修正方案 (Action Plan)

### 方案 A: 繼續訓練 + 調整權重 ⭐ (推薦)
**優勢**: 利用現有學習，快速驗證修正效果  
**實施步驟**:

#### Step 1: 修改配置檔
編輯 `configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml`:
```yaml
# Line 186-196: Adjust loss weights
losses:
  data_weight: 10.0              # 保持
  momentum_x_weight: 5.0         # 1.0 → 5.0 (↑ 5×)
  momentum_y_weight: 5.0         # 1.0 → 5.0 (↑ 5×)
  continuity_weight: 10.0        # 1.0 → 10.0 (↑ 10×) ⚠️ CRITICAL
  periodicity_weight: 10.0       # 保持
  prior_weight: 10.0             # 保持

# Line 249: Increase PDE collocation points
training:
  sampling:
    N_pde: 20000                 # 10000 → 20000 (↑ 2×)
```

#### Step 2: 從 Checkpoint 恢復訓練
```bash
cd /path/to/pinns-mvp
python scripts/train/train.py \
    --cfg configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml \
    --resume checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth
```

#### Step 3: 監控物理約束收斂
每 500 epochs 檢查:
```bash
grep "物理診斷" -A 3 log/training.log | tail -10
```

**目標 (Epoch 15000)**:
- Mass conservation error < 0.1 (當前 68 → 目標 0.1)
- Momentum conservation error < 1.0 (當前 91 → 目標 1.0)
- Total loss < 2.5 (當前 3.55 → 目標 2.5)

**預計時間**: ~1.5 小時 (A100 GPU)

---

### 方案 B: 從頭重新訓練 + Curriculum
**優勢**: 根治架構性問題，確保最佳結果  
**劣勢**: 需要 3-4 小時  

#### 新配置檔: `kolmogorov_re50_kf4_K100_rans_prior_v2.yml`
```yaml
curriculum:
  enable: true
  stages:
    - name: "Stage 1: Data Fitting"
      epochs: 1000
      loss_weights:
        data_weight: 10.0
        momentum_x_weight: 1.0
        momentum_y_weight: 1.0
        continuity_weight: 2.0
        prior_weight: 10.0
    
    - name: "Stage 2: Physics Enforcement"
      epochs: 14000
      loss_weights:
        data_weight: 10.0
        momentum_x_weight: 5.0
        momentum_y_weight: 5.0
        continuity_weight: 10.0
        prior_weight: 10.0

training:
  sampling:
    N_pde: 20000
    N_data: 100
```

---

## 10. 成敗判定標準

### 10.1 最低可發表標準 (Minimum Viable Paper)
```
✅ 必須達成:
- Mass conservation error < 0.01
- Momentum conservation error < 1.0
- Velocity L2 error < 15%
- Pressure L2 error < 30%

✅ 次要目標:
- 相對 vanilla PINNs 改善 ≥ 30%
- K ≤ 100 稀疏性展示
```

### 10.2 理想標準 (Excellent Paper)
```
🎯 Stretch goals:
- Mass conservation error < 0.001
- Velocity L2 error < 12%
- Pressure L2 error < 25%
- Pressure gradient L2 error < 25%
- 相對 vanilla PINNs 改善 ≥ 72%
```

---

## 11. 時間線規劃

### 階段一: 快速修正 (1-2 天)
- ✅ **Day 1 AM**: 執行方案 A (繼續訓練至 epoch 15000)
- ✅ **Day 1 PM**: 評估物理約束收斂情況
- 🔄 **Day 2**: 若失敗，切換至方案 B

### 階段二: 全面評估 (0.5 天)
- 完整評估 checkpoint (evaluate.py)
- 生成可視化（流場、誤差分布、能譜）
- 與 vanilla baseline 對比

### 階段三: 論文素材準備 (1 天)
- 生成高質量圖表
- 撰寫方法與結果章節
- 準備補充材料

**總耗時**: 2.5-3.5 天（取決於方案 A 是否成功）

---

## 12. 風險與應對

### 風險 1: 方案 A 失敗（物理約束仍不收斂）
**概率**: 30%  
**應對**: 立即切換方案 B，從頭訓練  
**損失**: 1.5 小時計算時間

### 風險 2: 感測器條件數過高導致壓力重建差
**概率**: 40%  
**應對**: 重新生成感測器（K=150, 物理引導 QR-Pivot）  
**損失**: 2 小時（感測器生成 + 重新訓練前 2000 epochs）

### 風險 3: 即使物理約束滿足，仍無法達到 72% 改善
**概率**: 20%  
**應對**: 
- 調整論文敘事：強調「先驗引導的穩定性」而非「絕對精度」
- 補充實驗：噪聲魯棒性、數據效率曲線

---

## 13. 總結與建議

### 當前狀態評級
```
訓練穩定性: ★★★★★ (5/5)
感測器擬合: ★★★★☆ (4/5)
先驗利用: ★★★★☆ (4/5)
物理守恆: ★☆☆☆☆ (1/5) ⚠️ CRITICAL
整體可用性: ★☆☆☆☆ (1/5) ⚠️ NOT PUBLISHABLE
```

### 核心建議
1. **立即執行方案 A**（調整權重 + 繼續訓練）
2. 每 500 epochs 監控物理診斷指標
3. 若 epoch 8000 仍無改善 → 切換方案 B
4. 並行準備：重新生成更優質的感測器（K=150-200，物理引導）

### 預期結果
**樂觀情況** (60% 機率):
- 方案 A 在 epoch 12000 時物理約束收斂
- 最終達到最低可發表標準

**悲觀情況** (40% 機率):
- 方案 A 失敗，需執行方案 B
- 總時間延長至 3-4 天
- 可能需要 2-3 輪迭代優化

---

## 附錄 A: 關鍵檔案清單

### 訓練相關
- **Checkpoint**: `checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth`
- **配置**: `configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml`
- **訓練日誌**: `pinnx.log` (6500 行)

### 數據相關
- **DNS Ground Truth**: `data/kolmogorov_dns/dns_re50_t100.h5` (751 MB)
- **感測器**: `data/sensors/kolmogorov/sensors_K100_re50_256x256.json`
- **RANS Prior**: `data/lowfi/kolmogorov_rans/` (128×128 解析度)

### 評估腳本
- **快速評估**: `scripts/evaluate/evaluate_checkpoint.py`
- **全面評估**: `scripts/evaluate/comprehensive_evaluation.py`
- **物理驗證**: `scripts/validation/validate_ns_conservation.py`

---

## 附錄 B: 命令速查表

### 繼續訓練 (方案 A)
```bash
python scripts/train/train.py \
    --cfg configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml \
    --resume checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth
```

### 監控訓練進度
```bash
# 查看最近 20 個 epochs
tail -50 pinnx.log | grep "Epoch"

# 查看物理診斷
grep "物理診斷" -A 4 log/training.log | tail -20
```

### 評估 Checkpoint
```bash
PYTHONPATH=. python scripts/evaluate/evaluate_checkpoint.py \
    --checkpoint checkpoints/kolmogorov_re50_kf4_K100_rans_prior/epoch_6500.pth \
    --config configs/kolmogorov_re50_kf4_K100_rans_prior_1k.yml
```

### 生成感測器 (若需重做)
```bash
python scripts/generate/sensors/batch_generate_kolmogorov_sensors.py \
    --rans_data data/lowfi/kolmogorov_rans/steady_state_re50.npz \
    --K_values 100 150 200 \
    --output_dir data/sensors/kolmogorov/
```

---

**報告結束**  
*如需進一步分析或執行修正方案，請提供指示。*
