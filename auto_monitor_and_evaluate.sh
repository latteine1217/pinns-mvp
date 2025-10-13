#!/bin/bash
# 智能監控腳本：持續監控訓練，完成後自動評估

CHECK_INTERVAL=60  # 每 60 秒檢查一次
MAX_WAIT_TIME=7200  # 最多等待 2 小時

elapsed=0
both_completed=false

echo "========================================="
echo "🤖 智能監控啟動"
echo "========================================="
echo ""
echo "⏱️  檢查間隔: ${CHECK_INTERVAL} 秒"
echo "⏱️  最長等待: $((MAX_WAIT_TIME / 60)) 分鐘"
echo ""

while [ $elapsed -lt $MAX_WAIT_TIME ]; do
    clear
    echo "========================================="
    echo "🤖 智能監控 - $(date '+%H:%M:%S')"
    echo "========================================="
    echo ""
    echo "已運行時間: $((elapsed / 60)) 分鐘"
    echo ""
    
    # 執行監控腳本
    ./monitor_both_trainings.sh
    
    # 檢查訓練是否都已完成
    baseline_running=$(pgrep -f "train.py.*baseline_1k" | wc -l)
    fourier_running=$(pgrep -f "train.py.*fourier_1k" | wc -l)
    
    if [ $baseline_running -eq 0 ] && [ $fourier_running -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "🎉 兩個訓練都已完成！"
        echo "========================================="
        echo ""
        both_completed=true
        break
    fi
    
    echo ""
    echo "⏳ 下次檢查: ${CHECK_INTERVAL} 秒後..."
    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

if [ "$both_completed" = true ]; then
    echo "🚀 自動啟動評估..."
    echo ""
    sleep 2
    ./run_fourier_evaluation.sh
else
    echo ""
    echo "⏰ 已達最長等待時間，監控結束"
    echo "   請手動執行 './run_fourier_evaluation.sh' 進行評估"
fi
