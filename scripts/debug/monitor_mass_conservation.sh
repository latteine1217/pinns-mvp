#!/bin/bash
# 實時監控質量守恆誤差

LOG_FILE="log/kolm_balanced_10k.log"

echo "==========================================="
echo "實時監控質量守恆誤差"
echo "==========================================="
echo "按 Ctrl+C 停止監控"
echo ""

while true; do
    clear
    echo "最新 20 epochs 的訓練狀態："
    echo "-------------------------------------------"
    
    # 提取最近的 epoch 資訊
    tail -500 "$LOG_FILE" | grep "Epoch [0-9]" | tail -20
    
    echo ""
    echo "最近 10 次質量守恆誤差："
    echo "-------------------------------------------"
    
    # 提取質量守恆誤差
    tail -200 "$LOG_FILE" | grep "質量守恆誤差" | tail -10
    
    echo ""
    echo "連續性損失趨勢："
    echo "-------------------------------------------"
    
    # 提取最近的 continuity_loss
    tail -500 "$LOG_FILE" | grep -o "continuity_loss: [0-9.e+-]*" | tail -10
    
    sleep 5
done
