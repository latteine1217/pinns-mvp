#!/bin/bash
# Re=60 DNS 模擬即時監控腳本
# 用法: bash scripts/monitor_dns_re60.sh

LOG_FILE="log/dns_re60_20251121_125441.log"
INTERVAL=${1:-10}  # 預設 10 秒更新一次

echo "========================================"
echo "  Re=60 DNS 即時監控 (每 ${INTERVAL}s)"
echo "========================================"
echo "按 Ctrl+C 退出"
echo ""

while true; do
    clear
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║               Re=60 Kolmogorov Flow DNS 模擬監控                   ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo ""
    date "+更新時間: %Y-%m-%d %H:%M:%S"
    echo ""
    
    # 進程狀態
    echo "【進程狀態】"
    ps aux | grep "generate_kolmogorov_dns_re30_stationary.py" | grep -v grep | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CPU=$(echo $line | awk '{print $3}')
        MEM=$(echo $line | awk '{print $4}')
        TIME=$(echo $line | awk '{print $10}')
        
        echo "  PID: $PID | CPU: ${CPU}% | MEM: ${MEM}% | Time: $TIME"
    done
    echo ""
    
    # 最新 5 行日誌
    echo "【最新日誌】（最後 5 步）"
    tail -5 "$LOG_FILE" | grep "Step" | while read line; do
        echo "  $line"
    done
    echo ""
    
    # 提取關鍵指標
    LATEST=$(tail -1 "$LOG_FILE" | grep "Step")
    if [ ! -z "$LATEST" ]; then
        STEP=$(echo "$LATEST" | grep -oP 'Step\s+\K\d+')
        TOTAL=200000
        TIME=$(echo "$LATEST" | grep -oP 't=\s+\K[\d.]+')
        KE=$(echo "$LATEST" | grep -oP 'KE=\K[\de.+-]+')
        BALANCE=$(echo "$LATEST" | grep -oP 'Balance=\K[\d.]+')
        
        PROGRESS=$(echo "scale=1; $STEP * 100 / $TOTAL" | bc)
        
        echo "【關鍵指標】"
        echo "  進度:       $STEP / $TOTAL ($PROGRESS%)"
        echo "  模擬時間:   t = $TIME / 200.0"
        echo "  動能:       KE = $KE"
        echo "  能量平衡:   $BALANCE"
        
        # 狀態判斷
        if (( $(echo "$BALANCE < 1.05" | bc -l) )) && (( $(echo "$BALANCE > 0.95" | bc -l) )); then
            echo "  狀態:       ✅ 已達穩態"
        elif (( $(echo "$BALANCE < 1.5" | bc -l) )); then
            echo "  狀態:       ⏳ 接近穩態"
        else
            echo "  狀態:       📈 收斂中"
        fi
    fi
    echo ""
    
    # 預估完成時間
    echo "【時間預估】"
    if [ ! -z "$STEP" ]; then
        # 計算速率（從啟動時間 12:54:41）
        START_EPOCH=$(date -j -f "%Y-%m-%d %H:%M:%S" "2025-11-21 12:54:41" "+%s" 2>/dev/null || echo 0)
        NOW_EPOCH=$(date "+%s")
        
        if [ "$START_EPOCH" != "0" ]; then
            ELAPSED=$((NOW_EPOCH - START_EPOCH))
            RATE=$(echo "scale=1; $STEP / $ELAPSED" | bc)
            REMAINING=$((TOTAL - STEP))
            REMAINING_SEC=$(echo "scale=0; $REMAINING / $RATE" | bc)
            REMAINING_MIN=$(echo "scale=1; $REMAINING_SEC / 60" | bc)
            
            ETA_EPOCH=$((NOW_EPOCH + REMAINING_SEC))
            ETA=$(date -r $ETA_EPOCH "+%H:%M:%S")
            
            echo "  計算速率:   ${RATE} 步/秒"
            echo "  已運行:     $((ELAPSED / 60)) 分鐘"
            echo "  預計剩餘:   ${REMAINING_MIN} 分鐘"
            echo "  預計完成:   $ETA"
        fi
    fi
    echo ""
    
    echo "────────────────────────────────────────────────────────────────────"
    echo "下次更新: ${INTERVAL}s 後 | 按 Ctrl+C 退出"
    
    sleep $INTERVAL
done
