#!/bin/bash
# 課程訓練監控腳本

LOG_FILE="/Users/latteine/Documents/coding/pinns-mvp/logs/curriculum_4stage_training.log"

echo "================================================================================"
echo "📊 課程訓練即時監控"
echo "================================================================================"
echo ""

# 檢查訓練是否還在運行
if pgrep -f "train.py.*curriculum_4stage" > /dev/null; then
    echo "✅ 訓練進程運行中 (PID: $(pgrep -f "train.py.*curriculum_4stage"))"
else
    echo "❌ 訓練進程未運行"
fi

echo ""
echo "📈 最新訓練日誌（最後30行）："
echo "--------------------------------------------------------------------------------"
tail -30 "$LOG_FILE"

echo ""
echo "================================================================================"
echo "🎯 階段切換記錄："
echo "--------------------------------------------------------------------------------"
grep "CURRICULUM STAGE TRANSITION" "$LOG_FILE" | tail -5

echo ""
echo "================================================================================"
echo "💾 檢查點保存記錄："
echo "--------------------------------------------------------------------------------"
grep "Stage checkpoint saved" "$LOG_FILE" | tail -5

echo ""
echo "================================================================================"
echo "📊 當前統計："
echo "--------------------------------------------------------------------------------"
LAST_EPOCH_LINE=$(grep "Epoch" "$LOG_FILE" | grep "Total:" | tail -1)
if [ -n "$LAST_EPOCH_LINE" ]; then
    # 使用 awk 提取數字
    CURRENT_EPOCH=$(echo "$LAST_EPOCH_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="Epoch") {print $(i+1); exit}}')
    CURRENT_LOSS=$(echo "$LAST_EPOCH_LINE" | awk '{for(i=1;i<=NF;i++) if($i=="Total:") {print $(i+1); exit}}')
    
    echo "當前 Epoch: $CURRENT_EPOCH / 4000"
    echo "當前總損失: $CURRENT_LOSS"
    
    # 計算階段
    if [ "$CURRENT_EPOCH" -ge 0 ] 2>/dev/null && [ "$CURRENT_EPOCH" != "" ]; then
        if [ "$CURRENT_EPOCH" -lt 1000 ]; then
            echo "當前階段: Stage 1 (Laminar, Re_tau=100)"
        elif [ "$CURRENT_EPOCH" -lt 2000 ]; then
            echo "當前階段: Stage 2 (Transition, Re_tau=300)"
        elif [ "$CURRENT_EPOCH" -lt 3000 ]; then
            echo "當前階段: Stage 3 (Turbulent, Re_tau=550)"
        else
            echo "當前階段: Stage 4 (HighRe, Re_tau=1000)"
        fi
    else
        echo "⚠️  無法解析當前階段"
    fi
else
    echo "⚠️  無法解析訓練進度"
fi

echo "================================================================================"
echo ""
echo "💡 提示："
echo "  - 查看完整日誌: tail -f $LOG_FILE"
echo "  - 停止訓練: pkill -f 'train.py.*curriculum_4stage'"
echo "  - 重新監控: bash scripts/monitor_curriculum.sh"
echo ""
