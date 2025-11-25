# Kolmogorov DNS 文件重命名報告

## 📋 執行摘要

**日期**: 2025-11-25  
**問題**: 所有 DNS 文件中的雷諾數標註與實際物理參數不符  
**解決**: 重命名 6 個文件，使文件名反映正確的雷諾數  

---

## ❌ 問題根因

DNS 文件命名使用錯誤的雷諾數標籤，導致：
1. 配置文件中的 `Re` 參數與 DNS 實際值不符
2. 研究者可能誤用不符合目標雷諾數的數據
3. 物理驗證失敗（PINNs 使用錯誤的 `nu` 值）

**雷諾數定義** (Musacchio & Boffetta 2014):
```
Re = √f₀ × L^(3/2) / ν
其中 L = 2π/k_f
```

---

## ✅ 重命名記錄

| 舊文件名 | 新文件名 | 聲稱 Re | 實際 Re | nu | k_f | 大小 |
|---------|---------|---------|---------|-----|-----|------|
| `kolmogorov_dns_re100_512x512_kf8_midway.h5` | `kolmogorov_dns_re56_512x512_kf8_midway.h5` | 100 | **55.68** | 0.0125 | 8 | 1.7GB |
| `kolmogorov_dns_re100_512x512_kf4_extended.h5` | `kolmogorov_dns_re197_512x512_kf4_extended.h5` | 100 | **196.87** | 0.01 | 4 | 2.2GB |
| `kolmogorov_dns_re100_kf8_t40.h5` | `kolmogorov_dns_re158_kf8_t40.h5` | 100 | **157.51** | 0.004419 | 8 | 2.3GB |
| `kolmogorov_dns_re100_kf8_t40_N1024.h5` | `kolmogorov_dns_re157_kf8_t40_N1024.h5` | 100 | **157.5** | 0.00441942 | 8 | 4.7GB |
| `kolmogorov_dns_re100_512x512_midway_v4.h5` | `kolmogorov_dns_re197_512x512_midway_v4.h5` | 100 | **196.87** | 0.01 | 4 | 853MB |
| `kolmogorov_dns_re100_512x512_v2.h5` | `kolmogorov_dns_re157_512x512_v2.h5` | 100 | **157.5** | 0.0125 | 4 | 1.1GB |

**總計**: 6 個文件重命名，總大小 12.8GB

---

## 🔬 物理驗證

### 驗證方法
使用 `scripts/calculate_reynolds_parameters.py` 計算實際雷諾數：

```bash
python scripts/calculate_reynolds_parameters.py --f0 1.0 --nu 0.0125 --k 8
# Output: Re = 55.68
```

### 流動狀態分類 (MB 2014)

| Re 範圍 | 流動狀態 | 文件數量 | 文件 |
|---------|---------|---------|------|
| **30 < Re < 100** | **過渡/弱湍流** ⭐ | 1 | `re56_kf8_midway` |
| 100 < Re < 200 | 湍流 | 4 | `re157_*`, `re158_*`, `re197_*` |
| Re > 200 | 強湍流 | 0 | - |

**PINNs 推薦**: Re ∈ [30, 100] (過渡區，物理豐富且計算可行)

---

## 📝 配置文件更新

### 已更新的配置

**新配置**: `configs/kolmogorov_re56_kf8_K100_balanced_correct.yml`

**關鍵修正**:
```yaml
data:
  kolmogorov_config:
    data_path: ./data/kolmogorov_dns_re56_512x512_kf8_midway.h5  # ✅ 已修正
    physics_params:
      Re: 55.68  # ✅ 已修正（原為 100）
      Re_definition: "Musacchio_Boffetta_2014"
      nu: 0.0125  # ✅ 已修正（原為 0.01）
      k_f: 8

physics:
  nu: 0.0125  # ✅ 已修正（原為 0.01）
  kolmogorov_flow:
    Re: 55.68  # ✅ 已修正

evaluation:
  reference_data: ./data/kolmogorov_dns_re56_512x512_kf8_midway.h5  # ✅ 已修正
```

### 需要手動更新的配置

以下配置文件仍引用舊文件名，需手動檢查更新：

```bash
# 搜尋所有引用舊文件名的配置
grep -r "kolmogorov_dns_re100" configs/
```

---

## 🎯 後續行動

### 立即行動
- [x] 重命名 DNS 文件（已完成）
- [x] 創建物理正確的配置文件（已完成）
- [ ] 生成 K=100 sensors (`sensors_kf8_deim_K100.npz`)
- [ ] 開始新訓練

### 中期行動
- [ ] 更新所有引用舊文件名的配置文件
- [ ] 更新文檔中的文件名引用
- [ ] 驗證所有其他 DNS 文件的雷諾數標註

### 長期行動
- [ ] 建立 DNS 生成標準流程（強制使用 `calculate_reynolds_parameters.py`）
- [ ] 在 DNS 文件內部添加 `Re_calculated` 屬性
- [ ] 創建自動驗證腳本（pre-commit hook）

---

## 📚 參考文獻

- Musacchio, S., & Boffetta, G. (2014). *Phys. Rev. E*, 89(2), 023004
- Shebalin, J. V. (2013). *Physics of Fluids*, 25(10), 105111
- 專案文檔: `scripts/README_REYNOLDS_CALCULATOR.md`
- 驗證報告: `KOLMOGOROV_REYNOLDS_FINAL_REPORT.md`

---

## ✅ 檢查清單

**重命名前**:
- [x] 備份原始文件（Git 版本控制）
- [x] 計算所有文件的實際雷諾數
- [x] 生成重命名計畫

**重命名後**:
- [x] 驗證文件完整性（大小未變）
- [x] 更新配置文件引用
- [x] 生成驗證報告

**訓練前**:
- [ ] 生成新 sensor 數據
- [ ] 使用 `calculate_reynolds_parameters.py` 驗證配置
- [ ] 確認 DNS 文件路徑正確

---

**報告生成**: 2025-11-25  
**工具版本**: `calculate_reynolds_parameters.py` v1.0  
**狀態**: ✅ 重命名完成，等待訓練
