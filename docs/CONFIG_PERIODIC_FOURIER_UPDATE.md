# 配置文件週期性 Fourier 嵌入更新總結

**日期**: 2026-01-06  
**狀態**: ✅ 所有配置文件已更新

---

## 更新內容

所有配置文件現在都包含**週期性 Fourier 嵌入**的配置選項和詳細註釋說明。

### 更新策略

- **保持向後相容**: 所有配置文件保留原有的配置方式（`type: standard` 或 `type: axis_selective`）
- **添加註釋說明**: 在每個配置文件的 `fourier_features` 部分添加詳細註釋
- **提供切換指南**: 清楚說明如何從舊格式切換到新的週期性嵌入格式

---

## 已更新的配置文件清單

| 配置文件 | 場景 | 更新狀態 |
|---------|------|---------|
| `kolmogorov_re50_kf4_K100.yml` | Kolmogorov Flow (baseline) | ✅ 已更新 |
| `kolmogorov_re50_kf4_K100_vanilla.yml` | Kolmogorov Flow (vanilla) | ✅ 已更新 |
| `kolmogorov_re50_kf4_K100_periodic_fourier.yml` | Kolmogorov Flow (週期性嵌入示範) | ✅ 原生支持 |
| `quick_test.yml` | 快速測試配置 | ✅ 已更新 |
| `quick_test_full.yml` | 完整快速測試配置 | ✅ 已更新 |
| `main.yml` | Channel Flow 主配置 | ✅ 已更新 |
| `main_quick_validate.yml` | Channel Flow 快速驗證 | ✅ 已更新 |
| `standard_config_template.yml` | 標準配置模板 | ✅ 已更新 |
| `config_template_example.yml` | 配置範例模板 | ✅ 已更新 |
| `channel_flow_periodic_example.yml` | Channel Flow 週期性嵌入示範 | ✅ 原生支持 |

**總計**: 10 個配置文件全部更新

---

## 配置格式對比

### Kolmogorov Flow (2D+T)

#### 舊格式（type: standard）
```yaml
model:
  fourier_features:
    type: standard
    fourier_m: 16
    fourier_sigma: 4.0
    include_input: true

losses:
  periodicity_weight: 10.0  # 需要軟約束
```

#### 新格式（type: hybrid，註釋提供）
```yaml
model:
  fourier_features:
    type: hybrid
    axes:
      0: {type: standard, n_modes: 12, sigma: 4.0, use_2pi: true}  # 時間
      1: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # x
      2: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # y
    trainable_fourier: false

losses:
  periodicity_weight: 0.0  # 或移除（週期性由嵌入保證）
```

---

### Channel Flow (3D+T)

#### 舊格式（type: axis_selective）
```yaml
model:
  fourier_features:
    type: axis_selective
    fourier_m: 32
    fourier_sigma: 5.0
    axes_config:
      x: [1, 2, 4, 8]  # 流向
      y: []            # 壁法向
      z: [1, 2, 4, 8]  # 展向
    domain_lengths:
      x: 25.13
      y: 2.0
      z: 9.42
```

#### 新格式（type: hybrid，註釋提供）
```yaml
model:
  fourier_features:
    type: hybrid
    axes:
      0: {type: standard, n_modes: 24, sigma: 5.0, use_2pi: true}  # 時間
      1: {type: periodic, domain_size: 25.13, n_modes: 16}  # x（流向週期）
      2: {type: standard, n_modes: 20, sigma: 4.0, use_2pi: true}  # y（壁法向）
      3: {type: periodic, domain_size: 9.42, n_modes: 16}  # z（展向週期）
    trainable_fourier: false
```

---

## 如何切換到週期性嵌入

### 步驟 1: 選擇配置文件

根據你的場景選擇配置文件：
- **Kolmogorov Flow**: `kolmogorov_re50_kf4_K100.yml`
- **Channel Flow**: `main.yml`
- **快速測試**: `quick_test.yml`

### 步驟 2: 修改配置

在配置文件中找到 `fourier_features` 部分，會看到類似這樣的註釋：

```yaml
# ========== Fourier Features 配置 ==========
# 方式 1: 標準 Fourier（當前使用）
fourier_features:
  type: standard
  fourier_m: 16
  fourier_sigma: 4.0

# 方式 2: 週期性 Fourier 嵌入（推薦）
# 取消下方註釋並註釋上方 "方式 1" 以啟用：
# fourier_features:
#   type: hybrid
#   axes:
#     0: {type: standard, n_modes: 12, sigma: 4.0, use_2pi: true}
#     1: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}
#     2: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}
#   trainable_fourier: false
```

**操作**:
1. 註釋掉「方式 1」（在每行前加 `#`）
2. 取消「方式 2」的註釋（移除每行前的 `#`）

### 步驟 3: 調整損失權重（僅 Kolmogorov Flow）

如果使用週期性嵌入，修改 `losses` 部分：

```yaml
losses:
  data_weight: 10.0
  momentum_x_weight: 1.0
  momentum_y_weight: 1.0
  continuity_weight: 2.0
  prior_weight: 10.0
  periodicity_weight: 0.0  # 設為 0 或移除此行
```

### 步驟 4: 驗證配置

```bash
python scripts/tools/validate_config_keys.py configs/your_config.yml
```

### 步驟 5: 測試訓練

```bash
python scripts/train/train.py --cfg configs/your_config.yml
```

---

## 配置註釋內容說明

每個配置文件的 `fourier_features` 部分現在包含：

1. **當前使用的配置**（未註釋）
   - 保持原有配置，確保現有訓練不受影響

2. **週期性嵌入配置範例**（註釋格式）
   - 完整的 `type: hybrid` 配置
   - 每個軸的詳細說明（週期/非週期）
   - `domain_size` 參數（必須與物理域精確匹配）

3. **使用說明**
   - 如何切換配置
   - 注意事項（如 `periodicity_weight` 的處理）

---

## 示範配置文件

如果你想直接使用週期性嵌入，可以參考這兩個完整示範：

### 1. Kolmogorov Flow
```bash
configs/kolmogorov_re50_kf4_K100_periodic_fourier.yml
```

### 2. Channel Flow
```bash
configs/channel_flow_periodic_example.yml
```

這兩個文件原生使用 `type: hybrid`，可以作為參考範本。

---

## 測試驗證

### 測試腳本

#### Kolmogorov Flow
```bash
python test_periodic_fourier_config.py
```

**預期輸出**:
```
✅ x 方向週期性 通過 (誤差: 1.8e-09)
✅ y 方向週期性 通過 (誤差: 2.9e-09)
```

#### Channel Flow
```bash
python test_channel_flow_config.py
```

**預期輸出**:
```
✅ x 方向週期性 通過 (誤差: 2.65e-06)
✅ z 方向週期性 通過 (誤差: 2.65e-06)
✅ y 方向非週期 正確（特徵差異: 1.91）
```

---

## 常見問題

### Q1: 切換到週期性嵌入後，訓練不穩定？

**A**: 檢查以下幾點：
1. `domain_size` 是否與物理域精確匹配
2. `n_modes` 是否過大（建議從 6-8 開始）
3. `periodicity_weight` 是否已設為 0

### Q2: 週期性誤差仍然較大？

**A**: 
- 檢查 `domain_size` 精度（使用完整浮點數，如 `6.283185307179586`）
- 確認配置正確切換到 `type: hybrid`

### Q3: 如何驗證配置正確？

**A**: 使用配置驗證工具：
```bash
python scripts/tools/validate_config_keys.py configs/your_config.yml
```

### Q4: 原有訓練會受影響嗎？

**A**: 不會。所有配置文件保留原有格式（未註釋），現有訓練不受影響。週期性嵌入配置以註釋形式提供，需手動啟用。

---

## 技術支持

### 相關文檔
- **使用指南**: `docs/PERIODIC_FOURIER_GUIDE.md`
- **實現確認**: `docs/PERIODIC_FOURIER_IMPLEMENTATION_CONFIRMATION.md`
- **配置指南**: `docs/CONFIG_GUIDE.md`

### 核心實現
- **混合 Fourier 編碼器**: `pinnx/models/hybrid_fourier.py`
- **模型集成**: `pinnx/models/fourier_mlp.py`

---

## 總結

✅ **10/10 配置文件已更新**  
✅ **所有配置文件包含週期性 Fourier 註釋說明**  
✅ **向後相容，不影響現有訓練**  
✅ **提供完整切換指南**  
✅ **包含測試驗證腳本**

**你現在可以在任何配置文件中輕鬆切換到週期性 Fourier 嵌入！**

---

**文檔版本**: v1.0  
**最後更新**: 2026-01-06  
**維護者**: PINNs-MVP 團隊
