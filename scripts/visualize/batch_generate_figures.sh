#!/bin/bash
# 批次生成論文所需的所有對比圖表
# 根據 docs/EXPERIMENT_COMPARISON_PLAN.md Section 9

set -e  # 遇到錯誤即停止

# 設定路徑
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/results/figures"

echo "=========================================="
echo "批次生成論文對比圖表"
echo "=========================================="
echo "專案根目錄: $PROJECT_ROOT"
echo "輸出目錄: $OUTPUT_DIR"
echo ""

# 確保輸出目錄存在
mkdir -p "$OUTPUT_DIR"

# ===========================
# F-S1: 感測器佈點對比圖
# ===========================
echo ">>> 生成 F-S1: Random vs QR 感測器佈點對比圖..."
python "$SCRIPT_DIR/generate_comparison_figures.py" \
    --mode sensor_comparison \
    --random-sensors data/jhtdb/sensors_kf8_random_K100.npz \
    --qr-sensors data/jhtdb/sensors_kf8_qr_K100.npz \
    --background-field data/kolmogorov_dns/snapshot_re50_mid.npz \
    --field-name vorticity \
    --output "$OUTPUT_DIR/F-S1_random_vs_qr_vorticity.png"

echo ""

# ===========================
# F-K1: K-scan Error 曲線
# ===========================
echo ">>> 生成 F-K1: K-scan error 曲線圖..."
python "$SCRIPT_DIR/generate_comparison_figures.py" \
    --mode k_scan \
    --results-dir results/experiments/S2_k_scan \
    --k-values 30 50 80 100 \
    --output "$OUTPUT_DIR/F-K1_k_scan_qr_vs_random.png"

echo ""

# ===========================
# F-P1: Prior Weight Sweep
# ===========================
echo ">>> 生成 F-P1: Prior weight sweep 曲線圖..."
python "$SCRIPT_DIR/generate_comparison_figures.py" \
    --mode prior_sweep \
    --results-dir results/experiments/C2_prior_sweep \
    --prior-weights 0.0 0.1 0.3 0.5 \
    --output "$OUTPUT_DIR/F-P1_prior_weight_sweep.png"

echo ""

# ===========================
# F-A1: Ablation Study
# ===========================
echo ">>> 生成 F-A1: Ablation study 條形圖..."
python "$SCRIPT_DIR/generate_comparison_figures.py" \
    --mode ablation \
    --results-dir results/experiments/A1_ablation_fourier \
    --baseline-name with_fourier \
    --output "$OUTPUT_DIR/F-A1_ablation_fourier.png"

echo ""

# ===========================
# 總結
# ===========================
echo "=========================================="
echo "✅ 圖表生成完成！"
echo "=========================================="
echo "輸出檔案位置:"
ls -lh "$OUTPUT_DIR"/F-*.png | awk '{print "  📊", $9, "(" $5 ")"}'
echo ""
echo "後續步驟:"
echo "  1. 檢查圖表品質與格式"
echo "  2. 使用 scripts/visualize/visualize_results.py 生成 F-R1 (場重建圖)"
echo "  3. 使用 scripts/evaluate/comprehensive_evaluation.py 生成 F-R2 (統計圖)"
echo ""
