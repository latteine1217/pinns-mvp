#!/bin/bash
# 監控 Kolmogorov DNS 運行進度

LOG_FILE="log/kolmogorov_re30_dns.log"

echo "=========================================="
echo "Kolmogorov Re=30 DNS 運行監控"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "⏱️  最後 20 行日誌"
    echo "=========================================="
    tail -20 "$LOG_FILE"
    echo ""
    echo "=========================================="
    echo "📊 運行統計"
    echo "=========================================="
    
    # 提取最新一行
    LAST_LINE=$(tail -1 "$LOG_FILE")
    
    if [[ $LAST_LINE == *"Step"* ]]; then
        echo "$LAST_LINE"
    fi
    
    # 檢查是否完成
    if grep -q "完成" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "✅ DNS 模擬已完成！"
        break
    fi
    
    if grep -q "達到穩態" "$LOG_FILE" 2>/dev/null; then
        echo ""
        echo "🎯 已達到統計穩態！"
    fi
    
    echo ""
    echo "按 Ctrl+C 退出監控（不會停止模擬）"
    echo "=========================================="
    
    sleep 5
done
