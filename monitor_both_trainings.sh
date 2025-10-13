#!/bin/bash
# 同時監控 Baseline 和 Fourier 訓練

clear
echo "========================================="
echo "📊 Fourier 對比實驗 - 雙訓練監控"
echo "========================================="
echo ""

# 檢查進程
echo "🔍 訓練進程狀態："
echo "-----------------------------------"
if pgrep -f "train.py.*baseline_1k" > /dev/null; then
    baseline_pid=$(pgrep -f 'train.py.*baseline_1k')
    echo "✅ Baseline 運行中 (PID: $baseline_pid)"
else
    echo "❌ Baseline 未運行"
fi

if pgrep -f "train.py.*fourier_1k" > /dev/null; then
    fourier_pid=$(pgrep -f 'train.py.*fourier_1k')
    echo "✅ Fourier 運行中 (PID: $fourier_pid)"
else
    echo "❌ Fourier 未運行"
fi

echo ""
echo "========================================="
echo "📈 Baseline (無 Fourier) - 最新進度"
echo "========================================="
baseline_latest=$(grep "Epoch" log/baseline_1k_training.log | tail -1)
echo "$baseline_latest"
baseline_best=$(grep "New best conservation" log/baseline_1k_training.log | tail -1)
echo "$baseline_best"

echo ""
echo "========================================="
echo "📈 Fourier (Fourier Features) - 最新進度"
echo "========================================="
fourier_latest=$(grep "Epoch" log/fourier_1k_training.log | tail -1)
echo "$fourier_latest"
fourier_best=$(grep "New best conservation" log/fourier_1k_training.log | tail -1)
echo "$fourier_best"

echo ""
echo "========================================="
echo "📊 對比分析"
echo "========================================="

# 提取當前 epoch
baseline_epoch=$(echo "$baseline_latest" | grep -oE "Epoch +[0-9]+" | grep -oE "[0-9]+")
fourier_epoch=$(echo "$fourier_latest" | grep -oE "Epoch +[0-9]+" | grep -oE "[0-9]+")

# 提取最佳 conservation error
baseline_cons=$(echo "$baseline_best" | grep -oE "[0-9.]+$")
fourier_cons=$(echo "$fourier_best" | grep -oE "[0-9.]+$")

echo "當前 Epoch:"
echo "  Baseline: $baseline_epoch / 1000 ($(echo "scale=1; $baseline_epoch * 100 / 1000" | bc)%)"
echo "  Fourier:  $fourier_epoch / 1000 ($(echo "scale=1; $fourier_epoch * 100 / 1000" | bc)%)"
echo ""
echo "最佳 Conservation Error:"
echo "  Baseline: $baseline_cons"
echo "  Fourier:  $fourier_cons"

echo ""
echo "========================================="
echo "💾 已保存檢查點"
echo "========================================="
echo "Baseline:"
ls -lt checkpoints/*baseline_1k*.pth 2>/dev/null | head -3 | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "Fourier:"
ls -lt checkpoints/*fourier_1k*.pth 2>/dev/null | head -3 | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "========================================="
echo "⏱️  執行 'watch -n 10 ./monitor_both_trainings.sh' 可持續監控"
echo "========================================="
