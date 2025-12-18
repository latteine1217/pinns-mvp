# ✅ Kolmogorov Flow 標準化驗證總結

**日期**: 2025-12-18  
**驗證原因**: 確認 Channel Flow 標準化 bug (v1.1.1) 是否影響 Kolmogorov Flow  
**結論**: **不受影響，無需修復**

---

## 🎯 快速結論

| 項目 | Channel Flow | Kolmogorov Flow |
|------|--------------|-----------------|
| **感測器數值** | ❌ 僅座標 | ✅ 含實際 u/v |
| **Prior 模型** | ❌ RANS 穩態 (退化) | ✅ Leith 非穩態 (正常) |
| **v/w 退化問題** | ❌ std~10⁻⁷ | ✅ std~0.2-5.0 |
| **需要修復** | ✅ 是 | ❌ 否 |

---

## 📊 驗證數據

### 感測器檔案統計

```
data/jhtdb/sensors_kf8_qr_K100.npz:
  sensor_u: mean=0.789, std=5.570 ✅
  sensor_v: mean=-0.906, std=4.377 ✅

data/jhtdb/sensors_kf8_physical_K100.npz:
  sensor_u: mean=0.029, std=0.241 ✅
  sensor_v: mean=-0.034, std=0.758 ✅
```

**結論**: 所有感測器檔案都包含實際數值，統計正常（std > 0.2）

### Leith Prior 統計

```
data/lowfi/kolmogorov_leith/rans_re50_kf4_leith.h5:
  u: mean=2.92e-05, std=0.424 ✅
  v: mean=-3.85e-06, std=0.219 ✅
```

**結論**: Leith 模型未退化，u/v 標準差遠大於閾值（> 1e-3）

---

## 🔍 關鍵差異分析

### 為何 Channel Flow 受影響而 Kolmogorov Flow 不受影響？

**Channel Flow 的問題**:
1. **感測器檔案僅含座標** → fallback 到 RANS prior
2. **RANS k-ω SST 是穩態模型** → v/w 收斂到接近零
3. **v/w std ~ 10⁻⁷** → 標準化時梯度被抑制 10⁵ 倍
4. **訓練失敗**: v/w/p 誤差 1000-2000%

**Kolmogorov Flow 為何正常**:
1. ✅ **感測器檔案包含實際 u/v 數值**
2. ✅ **Leith 是非穩態模型** → u/v 不會退化
3. ✅ **u/v std ~ 0.2-5.0** → 標準化梯度正常
4. ✅ **即使 fallback 到 Leith，統計也正確**

### Leith vs RANS k-ω SST 的根本差異

| 特性 | RANS k-ω SST | Leith Model |
|------|--------------|-------------|
| 狀態 | 穩態 (steady-state) | 非穩態 (time-averaged) |
| v/w 行為 | 收斂到 ~0 (對流被渦黏抵消) | 保持湍流統計 |
| 適用流動 | 3D 通道流 | 2D 湍流 (逆級串) |
| 標準差 | v/w ~ 10⁻⁷ ❌ | u/v ~ 0.2-0.4 ✅ |

---

## 📂 相關文檔

### Channel Flow 修復文檔
- [ROOT_CAUSE_FINAL.md](../results/channel_flow_evaluation/ROOT_CAUSE_FINAL.md) - 根因分析
- [CORRECT_SOLUTION_K100.md](../results/channel_flow_evaluation/CORRECT_SOLUTION_K100.md) - 解決方案

### Kolmogorov Flow 配置
```
configs/kolmogorov_re50_kf4_K100.yml
configs/kolmogorov_re50_kf4_K100_lbfgs.yml
configs/kolmogorov_re50_kf4_K100_vanilla.yml
```

### 診斷工具
```
scripts/diagnose_field_quality.py
scripts/quick_sensor_diagnosis.py
```

---

## ✅ 驗證清單

- [x] 檢查所有 Kolmogorov 感測器檔案
- [x] 驗證感測器包含實際 u/v 數值
- [x] 檢查 u/v 標準差 > 1e-3 閾值
- [x] 檢查 Leith prior 資料品質
- [x] 確認 Leith u/v 標準差正常
- [x] 分析 Leith 與 RANS 的根本差異
- [x] 確認訓練配置使用正確的資料來源
- [x] 撰寫驗證報告
- [x] 更新 README.md

---

## 🎓 結論

**Kolmogorov Flow 不受 Channel Flow 標準化 bug 影響的原因**:

1. **資料管線設計較完善**: 感測器檔案生成時已包含實際數值
2. **Prior 模型選擇正確**: Leith 模型適合 2D 湍流，不會退化
3. **自動 fallback 機制有效**: 即使 fallback 到 Leith，統計也正確

**無需採取行動**:
- ✅ 可以繼續使用現有 Kolmogorov 配置訓練
- ✅ v1.1.1 修復已包含通用驗證機制（未來保護）
- ✅ 所有文檔已更新，說明影響範圍

---

**驗證者**: Main Agent  
**驗證時間**: 2025-12-18  
**相關版本**: v1.1.1 (Channel Flow normalization fix)  
**狀態**: ✅ VERIFIED - No action required
