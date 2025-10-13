#!/bin/bash
# 監控 Initial Condition 課程訓練進度
# 用法: ./scripts/monitor_curriculum_ic.sh

LOG_FILE="logs/curriculum_ic_training.log"
CHECKPOINT_DIR="checkpoints/"

echo "=================================================="
echo "  PINNs Curriculum IC Training Monitor"
echo "=================================================="
echo ""

# 檢查日誌文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  日誌文件不存在: $LOG_FILE"
    echo "   訓練可能尚未啟動或已完成"
    exit 1
fi

echo "📊 最新訓練狀態："
echo "--------------------------------------------------"

# 提取最新的 epoch 資訊
LATEST_EPOCH=$(grep -E "Epoch [0-9]+" "$LOG_FILE" | tail -1)
echo "$LATEST_EPOCH"
echo ""

# 提取最新的損失分析
echo "📉 最新損失分析："
echo "--------------------------------------------------"
grep -A 15 "損失分析：" "$LOG_FILE" | tail -16
echo ""

# 提取階段切換資訊
echo "🔄 課程階段："
echo "--------------------------------------------------"
grep "切換到" "$LOG_FILE" | tail -5
echo ""

# 檢查點狀態
echo "💾 檢查點狀態："
echo "--------------------------------------------------"
ls -lht "$CHECKPOINT_DIR" | grep "pinnx_channel_flow_curriculum_ic" | head -5
echo ""

# 評估結果（如果有）
if grep -q "Evaluation completed" "$LOG_FILE"; then
    echo "✅ 評估結果："
    echo "--------------------------------------------------"
    grep -A 20 "Evaluation completed" "$LOG_FILE" | tail -21
    echo ""
fi

# 訓練時間估計
START_TIME=$(grep "開始訓練" "$LOG_FILE" | head -1 | awk '{print $1, $2}')
CURRENT_EPOCH=$(echo "$LATEST_EPOCH" | grep -oE "Epoch [0-9]+" | grep -oE "[0-9]+")

if [ ! -z "$CURRENT_EPOCH" ]; then
    PROGRESS=$(echo "scale=2; $CURRENT_EPOCH / 4000 * 100" | bc)
    echo "⏱️  訓練進度："
    echo "--------------------------------------------------"
    echo "  當前 Epoch: $CURRENT_EPOCH / 4000 ($PROGRESS%)"
    
    # 估計剩餘時間（粗略）
    if [ "$CURRENT_EPOCH" -gt 100 ]; then
        echo "  （完整時間估計需要更多數據點）"
    fi
fi

echo ""
echo "=================================================="
echo "💡 提示："
echo "  - 即時查看日誌: tail -f $LOG_FILE"
echo "  - 查看完整日誌: cat $LOG_FILE"
echo "  - 重新運行監控: ./scripts/monitor_curriculum_ic.sh"
echo "=================================================="
