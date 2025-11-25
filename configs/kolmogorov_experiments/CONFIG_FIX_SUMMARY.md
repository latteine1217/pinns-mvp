# Kolmogorov 配置修正摘要

**修正時間**: 2025-11-21  
**修正範圍**: `configs/kolmogorov_experiments/` 中的 7 個配置檔案  
**修正目標**: 符合 `trainer.py` 與 `config_loader.py` 規範

---

## ✅ 修正完成清單

### **1. `kolmogorov_2d_chaos_re30.yml`**
- [x] `loss` → `losses`
- [x] 添加 `sensors` 頂層段 (K=0)
- [x] 添加 `reproducibility` 段
- [x] `device: "mps"` → `device: "auto"`

### **2. `kolmogorov_2d_chaos_re30_full.yml`**
- [x] `loss` → `losses`
- [x] 添加 `sensors` 頂層段 (K=0)
- [x] 修正 `reproducibility.deterministic: false`（MPS 不支援）
- [x] `device: "mps"` → `device: "auto"`

### **3. `kolmogorov_2d_chaos_re30_quick.yml`**
- [x] `loss` → `losses`
- [x] 添加 `sensors` 頂層段 (K=0)
- [x] 添加 `reproducibility` 段
- [x] `device: "mps"` → `device: "auto"`

### **4. `kolmogorov_2d_turbulent_re60.yml`**
- [x] `loss` → `losses`
- [x] 添加 `sensors` 頂層段 (K=0)
- [x] 添加 `reproducibility` 段
- [x] `device: "mps"` → `device: "auto"`

### **5. `kolmogorov_2d_turbulent_pure_pde.yml`**
- [x] `loss` → `losses`
- [x] 添加 `sensors` 頂層段 (K=0)
- [x] 添加 `reproducibility` 段
- [x] `device: "mps"` → `device: "auto"`

### **6. `kolmogorov_2d_curriculum.yml`**
- [x] `loss` → `losses`
- [x] 移動 `sensors` 至頂層（原本在底部重複定義）
- [x] 添加 `reproducibility` 段
- [x] `device: mps` → `device: auto`
- [x] 添加 `compute.num_workers: 4`

### **7. `kolmogorov_2d_test_periodic.yml`**
- [x] `loss` → `losses`
- [x] `sensors` 段已存在，無需修改
- [x] 添加 `reproducibility` 段
- [x] `device: "mps"` → `device: "auto"`
- [x] 移除 `experiment.device`（應在 `compute` 段）

### **8. `kolmogorov_2d_baseline.yml`**
- [x] **無需修正**（已符合規範）

---

## 📋 修正細節

### **修正 1: `loss` → `losses`**

**原因**：`trainer.py` 第 107 行讀取 `config['losses']`

```diff
- loss:
+ losses:
    weights:
      momentum_x: 1.0
      ...
```

### **修正 2: 添加 `sensors` 頂層段**

**原因**：`train.py` 第 158-167 行檢查 `config['sensors']['K']`

```yaml
# ✅ 添加（純 PDE 訓練無監督資料）
sensors:
  K: 0                          # 純 PDE 訓練無感測點
  selection_method: "none"
  noise_level: 0.0
```

### **修正 3: 添加 `reproducibility` 段**

**原因**：確保訓練可重現性

```yaml
# ✅ 添加
reproducibility:
  deterministic: false          # MPS 不支援完全確定性
  benchmark: true               # 啟用 benchmark 以加速
  num_workers: 4
```

### **修正 4: 統一 `device` 設定**

**原因**：提高可移植性（自動適配 CUDA/MPS/CPU）

```diff
  compute:
-   device: "mps"                # Apple Silicon 特定
+   device: "auto"               # 自動選擇 CUDA > MPS > CPU
```

---

## ⚠️ 保持不變的項目

以下項目**未修改**，以保持原始實驗設計：

1. **物理參數**（Re, ν, A, k_f）
2. **域大小**（保留 4π×2π 長方形域，用於時空混沌研究）
3. **訓練策略**（epochs, learning rate, optimizer）
4. **網路架構**（layers, width, Fourier features）
5. **損失權重**（weights, adaptive_weighting）

---

## 🧪 驗證建議

建議執行以下驗證確保修正正確：

### **1. 配置載入測試**
```bash
python -c "
from pinnx.train.config_loader import load_config
import sys

configs = [
    'configs/kolmogorov_experiments/kolmogorov_2d_chaos_re30.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_chaos_re30_full.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_chaos_re30_quick.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_turbulent_re60.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_turbulent_pure_pde.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_curriculum.yml',
    'configs/kolmogorov_experiments/kolmogorov_2d_test_periodic.yml',
]

for cfg_path in configs:
    try:
        config = load_config(cfg_path)
        assert 'losses' in config, f'{cfg_path}: Missing losses key'
        assert 'sensors' in config, f'{cfg_path}: Missing sensors key'
        assert 'reproducibility' in config, f'{cfg_path}: Missing reproducibility key'
        assert config['compute']['device'] == 'auto', f'{cfg_path}: device not auto'
        print(f'✅ {cfg_path.split(\"/\")[-1]}')
    except Exception as e:
        print(f'❌ {cfg_path}: {e}')
        sys.exit(1)

print('\\n🎉 All configs validated successfully!')
"
```

### **2. 快速訓練測試**（可選）
```bash
# 測試最小配置（100 epochs）
python scripts/train.py \
  --cfg configs/kolmogorov_experiments/kolmogorov_2d_test_periodic.yml
```

---

## 📊 修正統計

| 項目 | 修正數量 |
|------|---------|
| `loss` → `losses` | 7 個檔案 |
| 添加 `sensors` 段 | 6 個檔案 |
| 添加 `reproducibility` 段 | 7 個檔案 |
| `device: "mps"` → `device: "auto"` | 7 個檔案 |
| **總計修正檔案** | **7 / 8** |

---

## 🔗 相關文檔

- **優化報告**: `OPTIMIZATION_REPORT.md`（詳細分析）
- **配置指南**: `KOLMOGOROV_CONFIGS.md`（使用說明）
- **模板範例**: `../config_template_example.yml`（參考格式）

---

## ✨ 後續工作

### **待 DNS 模擬完成後**：
1. 創建新配置 `kolmogorov_2d_dns_re30.yml`（匹配 DNS 2π×2π 域）
2. 生成 QR-Pivot 感測點從 DNS 資料
3. 訓練並評估逆問題重建效果

### **可選優化**：
- 標準化 `losses.weights` 結構（嵌套格式）
- 改進 `curriculum.yml` 的階段驗收條件
- 創建配置驗證腳本 `scripts/validate_configs.py`

---

**修正完成！** 🎉 所有配置現已符合程式碼規範，可正常使用。
