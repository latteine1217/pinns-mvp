# scripts/ 指南

這裡列出仍在維護的腳本，按用途分組，方便快速找到入口。不再維護的臨時監控工具已移除，避免干擾。

## 核心訓練
- `train.py`：主訓練入口，配置驅動，支援 VS-PINN 與標準 PINN。
- `train_curriculum_kolmogorov.py`：Kolmogorov 課程式訓練流程。
- `train_pure_pde.py`：僅 PDE 監督的基線訓練。

## 資料與前處理
- `fetch_channel_flow.py`、`fetch_temporal_snapshots.py`：JHTDB 資料與時間序列擷取。
- `generate_kolmogorov_dns.py`、`generate_kolmogorov_dns_re30_stationary.py`：Kolmogorov DNS / RE30 場生成。
- `generate_2d_slice_qr_sensors_fixed_v2.py`、`generate_sensors_k500.py`：感測點生成（QR-pivot 變體）。
- `auto_process_re100.py` ⭐：Re=100 DNS 完成後自動處理（湍流驗證、比較分析、QR 生成）。

## 評估與視覺化
- 評估：`evaluate.py`、`evaluate_checkpoint.py`、`evaluate_curriculum.py`、`evaluate_3d_physics.py`、`evaluate_piratenet_vs_jhtdb.py`、`evaluate_sensor_ablation.py`、`evaluate_kolmogorov_full.py`、`evaluate_kolmogorov_quick.py`、`comprehensive_evaluation.py`。
- 視覺化：`visualize_results.py`、`visualize_qr_sensors.py`、`visualize_sensors_comparison.py`、`visualize_ablation_heatmaps.py`、`visualize_adaptive_sampling.py`、`visualize_kolmogorov_results.py`。
- 補助分析：`compare_circular_indexing.py`、`compare_qr_strategies.py`、`validate_circular_indexing_jhtdb.py`、`generate_jhtdb_field_plots.py`.

## 監控
- `monitor_training.py`：推薦的通用訓練監控。
- `monitor_kolmogorov_dns.sh`：Kolmogorov DNS 生成監控。
- `check_dns_re60.py`、`check_dns_re100.py`：快速檢查 DNS 模擬進度。
- `monitor_dns_re60.sh`、`monitor_dns_enhanced.sh`：DNS 完整視覺化監控。

## 驗證/校核
- `validate_constraints.py`、`verify_jhtdb_data.py`、`verify_model_scaling.py`、`verify_weights.py`。
- `validation/`：物理解算與數值一致性測試集合。

### 使用建議
1. 資料→訓練→監控→評估/視覺化→驗證，依序呼叫上述腳本。
2. 任何腳本加 `--help` 先看參數；如需課程式或 Kolmogorov 專案，優先使用對應入口。
3. 若需舊版或一次性實驗腳本，請在 git 歷史查找，不再保留於此目錄。
