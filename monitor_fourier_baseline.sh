#!/bin/bash
# 監控 Baseline 訓練進度

echo "========================================="
echo "📊 Baseline 訓練監控"
echo "========================================="
echo ""

# 檢查進程
if pgrep -f "train.py.*baseline_1k" > /dev/null; then
    echo "✅ 訓練進程運行中 (PID: $(pgrep -f 'train.py.*baseline_1k'))"
else
    echo "❌ 訓練進程未運行"
fi

echo ""
echo "📈 最新損失（最後 10 個 epoch）："
echo "-----------------------------------"
grep "Epoch" log/baseline_1k_training.log | tail -10

echo ""
echo "🎯 最佳指標："
echo "-----------------------------------"
grep "New best" log/baseline_1k_training.log | tail -5

echo ""
echo "💾 已保存檢查點："
echo "-----------------------------------"
ls -lth checkpoints/*baseline_1k*.pth 2>/dev/null | head -5

echo ""
echo "========================================="
