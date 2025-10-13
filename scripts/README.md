# Scripts 目錄說明

本目錄包含 PINNs 專案的所有可執行腳本，已按功能分類整理。

## 📁 目錄結構

```
scripts/
├── 核心腳本 (主要工作流程)
├── 資料處理
├── 評估與可視化
├── 監控工具
├── 實驗與分析
├── 驗證測試
├── debug/ (除錯工具)
├── validation/ (物理驗證)
└── archive_*/ (已歸檔的舊版腳本)
```

---

## 🎯 核心腳本 (主要工作流程)

### `train.py` ⭐
**主訓練腳本** - 負責完整的 PINNs 訓練流程

**功能：**
- 配置驅動的模型建立
- 支援標準/增強 PINN 模型
- 階段式學習與動態權重調整
- 自動檢查點保存
- 整合物理損失、資料損失、先驗損失

**使用範例：**
```bash
python scripts/train.py --config configs/channel_flow_re1000_K80_wall_balanced.yml
python scripts/train.py --config configs/vs_pinn_3d_full_training.yml --device cuda
```

---

## 📊 資料處理

### `fetch_channel_flow.py`
從 JHTDB 獲取通道流資料並快取

**使用範例：**
```bash
python scripts/fetch_channel_flow.py --output data/jhtdb/channel_flow.npz
```

### `verify_jhtdb_data.py`
驗證 JHTDB 資料完整性與正確性

---

## 📈 評估與可視化

### `evaluate.py` / `evaluate_checkpoint.py` ⭐
**統一評估腳本** - 配置驅動的模組化評估

**功能：**
- 載入訓練檢查點
- 計算各項指標（L2誤差、RMSE、守恆誤差）
- 生成標準可視化圖表
- 支援參考資料對比

**使用範例：**
```bash
python scripts/evaluate_checkpoint.py --checkpoint checkpoints/model.pth --config configs/model.yml
python scripts/evaluate.py --checkpoint checkpoints/model.pth --reference data/jhtdb/full_field.npz
```

### `evaluate_curriculum.py`
課程學習專用評估工具

### `comprehensive_evaluation.py`
完整物理驗證評估

### `evaluate_3d_physics.py`
3D 物理場評估分析

### `visualize_results.py`
**增強視覺化工具** - 生成高品質分析圖表

**功能：**
- 三面板對比圖（預測/真實/誤差）
- 能譜分析
- 統計分佈圖
- 不確定性量化視覺化

**使用範例：**
```bash
python scripts/visualize_results.py --checkpoint checkpoints/model.pth
```

### `visualize_adaptive_sampling.py`
自適應採樣過程可視化

### `generate_jhtdb_field_plots.py`
生成 JHTDB 場圖

---

## 🔍 監控工具

### `monitor_training_progress.py` ⭐
通用訓練進度監控

### `monitor_warmup_test.py`
Warmup Cosine 學習率策略專用監控

### `monitor_curriculum.sh` / `monitor_curriculum_ic.sh`
課程學習訓練監控腳本

**使用範例：**
```bash
# 在另一個終端持續監控訓練
python scripts/monitor_training_progress.py --checkpoint_dir checkpoints --interval 60
```

---

## 🔬 實驗與分析

### `parameter_sensitivity_experiment.py`
參數敏感度分析實驗

### `k_scan_experiment.py`
感測點數量 K 的掃描實驗

### `analyze_k_scan.py`
分析 K-掃描實驗結果

### `run_longterm_training.py`
長期訓練實驗管理

### `benchmark.py` / `quick_benchmark.py`
性能基準測試

### `activation_benchmark.py`
激活函數性能測試

### 場分析工具
- `analyze_full_field_data.py` - 全場資料分析
- `detailed_field_analysis.py` - 詳細場分析
- `diagnose_channel_flow_characteristics.py` - 通道流特性診斷

---

## ✅ 驗證測試

### `validation/`
物理與數值驗證測試：

- `physics_validation.py` - 物理方程驗證
- `test_channel_flow_experiment.py` - 通道流實驗測試
- `test_channel_flow_physics.py` - 通道流物理測試
- `test_conservation_with_model.py` - 守恆性測試
- `validate_hybrid_sensors.py` - 混合感測點驗證
- `validate_ns_conservation.py` - NS方程守恆性驗證

**使用範例：**
```bash
python scripts/validation/physics_validation.py
python scripts/validation/test_channel_flow_physics.py
```

### 其他驗證腳本

- `validate_constraints.py` - 約束條件驗證
- `verify_model_scaling.py` - 模型尺度驗證
- `verify_weights.py` - 損失權重驗證

---

## 🐛 除錯工具 (`debug/`)

用於診斷訓練問題的專用工具：

### NS 方程診斷
- `diagnose_ns_equations.py` - **主要診斷工具** - 完整的 NS 方程驗證
- `diagnose_boundary_conditions.py` - 邊界條件診斷
- `diagnose_pressure_failure.py` - 壓力場失效診斷

### 梯度與導數問題
- `debug_autograd_issue.py` - 自動微分問題診斷
- `debug_derivatives_computation.py` - 導數計算診斷
- `debug_gradient_computation.py` - 梯度計算診斷
- `test_derivative_function_direct.py` - 導數函數直接測試

### 物理殘差問題
- `debug_physics_residuals.py` - 物理殘差診斷
- `diagnose_conservation_error.py` - 守恆誤差診斷

### 資料與感測點問題
- `diagnose_training_data.py` - 訓練資料診斷
- `diagnose_sensor_overfitting.py` - 感測點過擬合診斷
- `analyze_sensor_overfitting.py` - 感測點過擬合分析
- `compare_training_vs_fullfield_divergence.py` - 訓練點與全場散度對比

### 綜合診斷
- `verify_remaining_hypotheses.py` - 驗證剩餘假設（大型診斷工具）
- `monitor_training.py` - 訓練過程監控

**使用範例：**
```bash
# 診斷 NS 方程實現
python scripts/debug/diagnose_ns_equations.py --config configs/vs_pinn_3d_warmup_test.yml

# 檢查邊界條件
python scripts/debug/diagnose_boundary_conditions.py
```

---

## 📦 歸檔目錄

### `archive_demos/`
已歸檔的演示腳本：
- `demo_vs_pinn.py` - VS-PINN 基礎演示
- `demo_vs_pinn_fixed.py` - VS-PINN 修復版演示
- `demo_vs_pinn_quickstart.py` - VS-PINN 快速開始

### `archive_eval/`
已過時的階段性評估腳本：
- `eval_phase2.py`
- `evaluate_phase3b_fullfield.py`
- `evaluate_training_result.py`
- `evaluate_training_simple.py`
- `quick_eval.py`, `quick_eval_phase3.py`, `quick_eval_sine.py`
- `run_evaluate_simple.py`
- `evaluate_jhtdb_comparison.py`
- `load_and_evaluate.py`

### `archive_monitors/`
已過時的階段性監控腳本：
- `monitor_phase3_training.py`, `monitor_phase4b_*.py`
- `monitor_500epochs.py`
- `monitor_test_training.py`, `monitor_stable_training.py`
- `monitor_cosine_restarts.py`
- `continuous_monitor.py`, `simple_monitor.py`

### `archive_sensors/`
已歸檔的感測點生成腳本：
- `generate_3d_sensors_k30.py`, `generate_3d_sensors_k500.py`
- `generate_k80_*.py` (5 個變體)
- `generate_sensors_wall_balanced.py`
- `fix_sensor_generation.py`
- `generate_sensor_cache_from_existing_data.py`

### `archive_diagnostics/`
已過時的診斷腳本：
- `diagnose_sine_training.py`, `diagnose_training_failure.py`
- `quick_diagnostic_phase2.py`, `quick_diagnostic_phase3.py`
- `quick_test_sine.py`, `quick_test_train_fixed.py`

### `archive_shell_scripts/`
已過時的 Shell 腳本：
- `auto_evaluate_phase3.sh`, `auto_monitor_and_evaluate.sh`
- `enhanced_phase3_monitor.sh`, `monitor_phase2.sh`
- `monitor_phase4b_retrain.sh`, `simple_training_monitor.sh`
- `watch_phase3_training.sh`, `watch_training.sh`

**注意：** 歸檔腳本僅供歷史參考，不建議在新實驗中使用。

---

## 📝 使用建議

### 典型工作流程

1. **準備資料**
   ```bash
   python scripts/fetch_channel_flow.py
   ```

2. **訓練模型**
   ```bash
   python scripts/train.py --config configs/vs_pinn_3d_full_training.yml
   ```

3. **監控訓練** (另一個終端)
   ```bash
   python scripts/monitor_training_progress.py --checkpoint_dir checkpoints
   ```

4. **評估結果**
   ```bash
   python scripts/evaluate_checkpoint.py --checkpoint checkpoints/latest.pth
   python scripts/visualize_results.py --checkpoint checkpoints/latest.pth
   ```

5. **驗證物理** (可選)
   ```bash
   python scripts/validation/physics_validation.py
   python scripts/debug/diagnose_ns_equations.py --config configs/model.yml
   ```

### 遇到問題時

1. **診斷 NS 方程**: `python scripts/debug/diagnose_ns_equations.py`
2. **檢查物理殘差**: `python scripts/debug/debug_physics_residuals.py`
3. **驗證守恆性**: `python scripts/validation/validate_ns_conservation.py`
4. **檢查邊界條件**: `python scripts/debug/diagnose_boundary_conditions.py`

---

## 🔄 最近更新

- **2025-10-09**: Task-9 腳本整理完成
  - 歸檔 7 個重複評估腳本至 `archive_eval/`
  - 歸檔 5 個重複監控腳本至 `archive_monitors/`
  - 歸檔 10 個感測點生成腳本至 `archive_sensors/`
  - 移動 8 個測試腳本至 `tests/`
  - 歸檔 6 個過時診斷腳本至 `archive_diagnostics/`
  - 歸檔 9 個過時 Shell 腳本至 `archive_shell_scripts/`
  - **目前根目錄腳本數**: 30 個（目標 ≤ 20 個，持續優化中）

- **2025-10-09**: VS-PINN 計算圖斷裂問題修復完成
  - `diagnose_ns_equations.py` 通過所有測試
  - 梯度計算誤差 < 1e-7

- **2025-10-08**: 整合 `enhanced_fourier_mlp.py` 到統一模型架構
- **2025-10-08**: 創建此 README 文檔

---

## 📌 注意事項

- 所有腳本都支援 `--help` 參數查看詳細使用說明
- 建議使用配置文件驅動訓練，避免硬編碼參數
- debug 工具僅在遇到問題時使用，不影響正常流程
- 歸檔腳本可能依賴已棄用的 API，使用時需謹慎
- 測試腳本已移至 `tests/` 資料夾，使用 `pytest` 執行

---

需要更多幫助？請參閱專案根目錄的 `README.md` 和 `TECHNICAL_DOCUMENTATION.md`。
