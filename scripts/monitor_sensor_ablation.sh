#!/bin/bash
# 監控感測點策略對比訓練進度

echo "🔍 感測點策略對比實驗監控"
echo "=========================================="
echo ""

# QR-Pivot 基線
echo "📊 QR-Pivot 基線 (K=50, 中心區 0%)"
echo "----------------------------------------"
if [ -f "log/ablation_sensor_qr_K50.pid" ]; then
    PID_QR=$(cat log/ablation_sensor_qr_K50.pid)
    if ps -p $PID_QR > /dev/null 2>&1; then
        echo "✅ 訓練進程運行中 (PID: $PID_QR)"
    else
        echo "❌ 訓練進程已停止"
    fi
fi

# 最新 epoch
LATEST_QR=$(grep -E "Epoch [0-9]+/500" log/ablation_sensor_qr_K50.log | tail -1)
if [ -n "$LATEST_QR" ]; then
    echo "📈 $LATEST_QR" | sed 's/^.*Epoch/Epoch/'
fi

# 最佳 loss
BEST_QR=$(grep "🎯 新最佳指標" log/ablation_sensor_qr_K50.log | tail -1)
if [ -n "$BEST_QR" ]; then
    echo "$BEST_QR" | sed 's/^.*INFO - //'
fi

echo ""

# Stratified 改進
echo "📊 Stratified 改進 (K=50, 中心區 32%)"
echo "----------------------------------------"
if [ -f "log/ablation_sensor_stratified_K50.pid" ]; then
    PID_STRAT=$(cat log/ablation_sensor_stratified_K50.pid)
    if ps -p $PID_STRAT > /dev/null 2>&1; then
        echo "✅ 訓練進程運行中 (PID: $PID_STRAT)"
    else
        echo "❌ 訓練進程已停止"
    fi
fi

# 最新 epoch
LATEST_STRAT=$(grep -E "Epoch [0-9]+/500" log/ablation_sensor_stratified_K50.log | tail -1)
if [ -n "$LATEST_STRAT" ]; then
    echo "📈 $LATEST_STRAT" | sed 's/^.*Epoch/Epoch/'
fi

# 最佳 loss
BEST_STRAT=$(grep "🎯 新最佳指標" log/ablation_sensor_stratified_K50.log | tail -1)
if [ -n "$BEST_STRAT" ]; then
    echo "$BEST_STRAT" | sed 's/^.*INFO - //'
fi

echo ""
echo "=========================================="
echo "💡 使用方法："
echo "   watch -n 10 ./scripts/monitor_sensor_ablation.sh  # 每 10 秒刷新"
echo "   tail -f log/ablation_sensor_qr_K50.log           # 即時查看 QR 日誌"
echo "   tail -f log/ablation_sensor_stratified_K50.log   # 即時查看 Stratified 日誌"
