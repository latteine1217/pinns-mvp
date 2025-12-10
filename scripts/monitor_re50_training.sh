#!/bin/bash
# Re=50 K=100 對比實驗監控腳本

echo "======================================================================"
echo "📊 Kolmogorov Re=50 K=100 訓練監控"
echo "======================================================================"
echo ""

# 完整版監控
echo "🔷 完整版 (Full Features)"
echo "----------------------------------------------------------------------"
if [ -f "log/kolmogorov_re50_kf4_K100_full_1k/training.log" ]; then
    echo "📁 日誌文件: log/kolmogorov_re50_kf4_K100_full_1k/training.log"

    # 檢查進程
    FULL_PID=$(ps aux | grep "kolmogorov_re50_kf4_K100_full_1k.yml" | grep -v grep | awk '{print $2}')
    if [ -n "$FULL_PID" ]; then
        echo "✅ 狀態: 運行中 (PID: $FULL_PID)"
    else
        echo "❌ 狀態: 未運行"
    fi

    # 最新 epoch
    LATEST_EPOCH=$(grep "💾 檢查點已保存" log/kolmogorov_re50_kf4_K100_full_1k/training.log | tail -1 | grep -oE "epoch_[0-9]+" | grep -oE "[0-9]+")
    if [ -n "$LATEST_EPOCH" ]; then
        echo "📊 當前 Epoch: $LATEST_EPOCH / 1000"
        PROGRESS=$((LATEST_EPOCH * 100 / 1000))
        echo "📈 進度: $PROGRESS%"
    fi

    # 最新 loss
    LATEST_LOSS=$(grep "Epoch.*total_loss" log/kolmogorov_re50_kf4_K100_full_1k/training.log | tail -1 | grep -oE "total_loss: [0-9.]+" | grep -oE "[0-9.]+")
    if [ -n "$LATEST_LOSS" ]; then
        echo "📉 最新 Loss: $LATEST_LOSS"
    fi

    # 檢查點文件
    if [ -d "checkpoints/kolmogorov_re50_kf4_K100_full" ]; then
        CHECKPOINT_COUNT=$(ls checkpoints/kolmogorov_re50_kf4_K100_full/epoch_*.pth 2>/dev/null | wc -l | xargs)
        echo "💾 檢查點數量: $CHECKPOINT_COUNT"
    fi
else
    echo "❌ 日誌文件不存在"
fi

echo ""
echo "----------------------------------------------------------------------"
echo ""

# Vanilla 版監控
echo "🔶 Vanilla 版 (Baseline)"
echo "----------------------------------------------------------------------"
if [ -f "log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log" ]; then
    echo "📁 日誌文件: log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log"

    # 檢查進程
    VANILLA_PID=$(ps aux | grep "kolmogorov_re50_kf4_K100_vanilla_1k.yml" | grep -v grep | awk '{print $2}')
    if [ -n "$VANILLA_PID" ]; then
        echo "✅ 狀態: 運行中 (PID: $VANILLA_PID)"
    else
        echo "❌ 狀態: 未運行"
        # 檢查是否有錯誤
        if grep -q "Error\|failed assertion" log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log; then
            echo "⚠️  發現錯誤，請檢查日誌"
            tail -5 log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log | grep -E "Error|failed"
        fi
    fi

    # 最新 epoch
    LATEST_EPOCH=$(grep "💾 檢查點已保存" log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log | tail -1 | grep -oE "epoch_[0-9]+" | grep -oE "[0-9]+")
    if [ -n "$LATEST_EPOCH" ]; then
        echo "📊 當前 Epoch: $LATEST_EPOCH / 1000"
        PROGRESS=$((LATEST_EPOCH * 100 / 1000))
        echo "📈 進度: $PROGRESS%"
    fi

    # 最新 loss
    LATEST_LOSS=$(grep "Epoch.*total_loss" log/kolmogorov_re50_kf4_K100_vanilla_1k/training.log | tail -1 | grep -oE "total_loss: [0-9.]+" | grep -oE "[0-9.]+")
    if [ -n "$LATEST_LOSS" ]; then
        echo "📉 最新 Loss: $LATEST_LOSS"
    fi

    # 檢查點文件
    if [ -d "checkpoints/kolmogorov_re50_kf4_K100_vanilla" ]; then
        CHECKPOINT_COUNT=$(ls checkpoints/kolmogorov_re50_kf4_K100_vanilla/epoch_*.pth 2>/dev/null | wc -l | xargs)
        echo "💾 檢查點數量: $CHECKPOINT_COUNT"
    fi
else
    echo "❌ 日誌文件不存在"
fi

echo ""
echo "======================================================================"
echo ""
echo "💡 提示："
echo "   • 查看完整日誌: tail -f log/kolmogorov_re50_kf4_K100_full_1k/training.log"
echo "   • 停止訓練: kill <PID>"
echo "   • 重新啟動監控: watch -n 30 ./scripts/monitor_re50_training.sh"
echo ""
echo "======================================================================"
