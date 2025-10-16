#!/bin/bash
# ========================================
# 感測點策略對比實驗 - 增強監控腳本
# ========================================

echo "========================================"
echo "  感測點策略對比訓練監控"
echo "  時間: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# 顏色定義
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ========================================
# 1. QR-Pivot 基線
# ========================================
echo -e "\n${GREEN}[1] QR-Pivot 基線${NC}"
echo "----------------------------------------"

QR_PID=$(pgrep -f "ablation_sensor_qr_K50.yml")
if [ -n "$QR_PID" ]; then
    echo -e "狀態: ${GREEN}運行中${NC} (PID: $QR_PID)"
    
    # 提取最新 epoch 和 loss
    QR_LOG="log/ablation_sensor_qr_K50.log"
    if [ -f "$QR_LOG" ]; then
        echo -e "\n📊 最新進度:"
        tail -1 "$QR_LOG" | grep -oE "Epoch [0-9]+/[0-9]+" | head -1
        
        echo -e "\n📉 Loss 趨勢 (最近 5 個 Best):"
        grep "新最佳指標" "$QR_LOG" | tail -5 | \
            awk '{print $NF, $(NF-1)}' | \
            awk '{printf "  Epoch %-4s: %.2f\n", substr($2, 1, length($2)-1), $1}'
        
        echo -e "\n🔍 當前 Total Loss:"
        grep "total_loss" "$QR_LOG" | tail -1 | \
            grep -oE "total_loss: [0-9.]+" | \
            awk '{printf "  %.2f\n", $2}'
    fi
else
    echo -e "狀態: ${RED}已停止${NC}"
    echo "檢查點目錄: checkpoints/ablation_sensor_qr_K50/"
    ls -lh checkpoints/ablation_sensor_qr_K50/best_model.pth 2>/dev/null || echo "  未找到最佳模型"
fi

# ========================================
# 2. Stratified 改進
# ========================================
echo -e "\n${YELLOW}[2] Stratified 改進${NC}"
echo "----------------------------------------"

STRAT_PID=$(pgrep -f "ablation_sensor_stratified_K50.yml")
if [ -n "$STRAT_PID" ]; then
    echo -e "狀態: ${GREEN}運行中${NC} (PID: $STRAT_PID)"
    
    # 提取最新 epoch 和 loss
    STRAT_LOG="log/ablation_sensor_stratified_K50.log"
    if [ -f "$STRAT_LOG" ]; then
        echo -e "\n📊 最新進度:"
        tail -1 "$STRAT_LOG" | grep -oE "Epoch [0-9]+/[0-9]+" | head -1
        
        echo -e "\n📉 Loss 趨勢 (最近 5 個 Best):"
        grep "新最佳指標" "$STRAT_LOG" | tail -5 | \
            awk '{print $NF, $(NF-1)}' | \
            awk '{printf "  Epoch %-4s: %.2f\n", substr($2, 1, length($2)-1), $1}'
        
        echo -e "\n🔍 當前 Total Loss:"
        grep "total_loss" "$STRAT_LOG" | tail -1 | \
            grep -oE "total_loss: [0-9.]+" | \
            awk '{printf "  %.2f\n", $2}'
        
        echo -e "\n⚠️  子損失項分解 (最新 Epoch):"
        grep "total_loss" "$STRAT_LOG" | tail -1 | \
            grep -oE "(data_loss|pde_loss|wall_loss|pressure_loss): [0-9.]+" | \
            awk '{printf "  %-15s: %8.2f\n", $1, $2}'
    fi
else
    echo -e "狀態: ${RED}已停止${NC}"
    echo "檢查點目錄: checkpoints/ablation_sensor_stratified_K50/"
    ls -lh checkpoints/ablation_sensor_stratified_K50/best_model.pth 2>/dev/null || echo "  未找到最佳模型"
fi

# ========================================
# 3. 對比分析
# ========================================
echo -e "\n========================================"
echo -e "${GREEN}[3] 即時對比分析${NC}"
echo "========================================"

if [ -f "$QR_LOG" ] && [ -f "$STRAT_LOG" ]; then
    QR_BEST=$(grep "新最佳指標" "$QR_LOG" | tail -1 | sed 's/.*新最佳指標: \([0-9.]*\).*/\1/')
    STRAT_BEST=$(grep "新最佳指標" "$STRAT_LOG" | tail -1 | sed 's/.*新最佳指標: \([0-9.]*\).*/\1/')
    
    echo "Best Loss 比較:"
    printf "  %-12s: %10.2f\n" "QR-Pivot" "$QR_BEST"
    printf "  %-12s: %10.2f\n" "Stratified" "$STRAT_BEST"
    
    if [ -n "$QR_BEST" ] && [ -n "$STRAT_BEST" ]; then
        RATIO=$(echo "scale=2; $STRAT_BEST / $QR_BEST" | bc)
        printf "  %-12s: %10sx\n" "比值" "$RATIO"
        
        if (( $(echo "$RATIO > 5" | bc -l) )); then
            echo -e "  ${RED}⚠️  Stratified 當前 loss 遠高於 QR（>5x）${NC}"
        elif (( $(echo "$RATIO > 2" | bc -l) )); then
            echo -e "  ${YELLOW}⚡ Stratified 仍在收斂中（2-5x）${NC}"
        else
            echo -e "  ${GREEN}✅ Stratified 接近 QR 水平（<2x）${NC}"
        fi
    fi
fi

echo -e "\n========================================"
echo "  使用方式: watch -n 10 $0"
echo "========================================"
