#!/bin/bash
# Fourier 對比實驗 - 自動化評估腳本

echo "========================================="
echo "🔍 檢查訓練狀態..."
echo "========================================="
echo ""

# 檢查兩個訓練是否都已完成
baseline_running=$(pgrep -f "train.py.*baseline_1k" | wc -l)
fourier_running=$(pgrep -f "train.py.*fourier_1k" | wc -l)

if [ $baseline_running -gt 0 ]; then
    echo "⏳ Baseline 訓練仍在進行中..."
    baseline_latest=$(grep "Epoch" log/baseline_1k_training.log | tail -1)
    echo "   $baseline_latest"
fi

if [ $fourier_running -gt 0 ]; then
    echo "⏳ Fourier 訓練仍在進行中..."
    fourier_latest=$(grep "Epoch" log/fourier_1k_training.log | tail -1)
    echo "   $fourier_latest"
fi

if [ $baseline_running -gt 0 ] || [ $fourier_running -gt 0 ]; then
    echo ""
    echo "❌ 訓練尚未完成，無法執行評估"
    echo "   請等待訓練完成後再執行此腳本"
    echo ""
    echo "💡 提示: 使用 './monitor_both_trainings.sh' 監控進度"
    exit 1
fi

echo "✅ 兩個訓練都已完成！"
echo ""

# 檢查檢查點是否存在
echo "========================================="
echo "📂 檢查檢查點文件..."
echo "========================================="
echo ""

if [ ! -f "checkpoints/vs_pinn_baseline_1k_latest.pth" ]; then
    echo "❌ Baseline 檢查點不存在"
    exit 1
fi

if [ ! -f "checkpoints/vs_pinn_fourier_1k_latest.pth" ]; then
    echo "❌ Fourier 檢查點不存在"
    exit 1
fi

if [ ! -f "data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz" ]; then
    echo "❌ 測試資料不存在"
    exit 1
fi

echo "✅ Baseline: checkpoints/vs_pinn_baseline_1k_latest.pth"
echo "✅ Fourier:  checkpoints/vs_pinn_fourier_1k_latest.pth"
echo "✅ 測試集:   data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz"
echo ""

# 執行評估
echo "========================================="
echo "🚀 開始對比評估..."
echo "========================================="
echo ""

python scripts/compare_fourier_experiments.py

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✅ 評估完成！"
    echo "========================================="
    echo ""
    echo "📊 查看結果:"
    echo "   報告: results/fourier_comparison/comparison_report.md"
    echo "   圖表: results/fourier_comparison/training_comparison.png"
    echo ""
    echo "📖 快速預覽報告:"
    echo "========================================="
    head -50 results/fourier_comparison/comparison_report.md
else
    echo ""
    echo "❌ 評估失敗，請檢查錯誤訊息"
    exit 1
fi
