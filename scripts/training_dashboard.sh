#!/bin/bash

# 3D VS-PINN 訓練即時監控面板

LOG_FILE="log/vs_pinn_3d_full_training.log"
PID=67302

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

clear

echo "========================================="
echo -e "${BLUE}🚀 3D VS-PINN 訓練監控面板${NC}"
echo "========================================="
echo ""

# 檢查進程狀態
if ps -p $PID > /dev/null; then
    echo -e "${GREEN}✅ 訓練進程運行中 (PID: $PID)${NC}"
else
    echo -e "${RED}❌ 訓練進程已結束${NC}"
    exit 1
fi

echo ""
echo "=== 📊 訓練統計 ==="

# 獲取初始和當前狀態
FIRST_LINE=$(grep "Epoch" $LOG_FILE | head -1)
LAST_LINE=$(grep "Epoch" $LOG_FILE | tail -1)

echo "起始: $FIRST_LINE"
echo "當前: $LAST_LINE"

echo ""
echo "=== 🎯 Conservation Error 趨勢 (最近10次) ==="
tail -500 $LOG_FILE | grep "New best conservation_error" | tail -10 | \
    awk '{print "Epoch " $17 ": " $15}' | sed 's/://g'

echo ""
echo "=== 💾 檢查點列表 ==="
ls -lht checkpoints/vs_pinn_3d_full_training_*.pth 2>/dev/null | \
    awk '{print $9, "(" $5 ")", $6, $7, $8}'

echo ""
echo "=== ⏱️ 時間估算 ==="

# 計算已訓練的 epoch 數
CURRENT_EPOCH=$(grep "Epoch" $LOG_FILE | tail -1 | sed 's/.*Epoch *\([0-9]*\).*/\1/')
TOTAL_EPOCHS=1000

if [ -n "$CURRENT_EPOCH" ] && [ "$CURRENT_EPOCH" -gt 0 ] 2>/dev/null; then
    REMAINING=$((TOTAL_EPOCHS - CURRENT_EPOCH))
    PROGRESS=$(echo "scale=1; $CURRENT_EPOCH*100/$TOTAL_EPOCHS" | bc)
    EST_TIME=$(echo "scale=0; $REMAINING*0.6/60" | bc)
    
    echo "當前進度: ${CURRENT_EPOCH}/${TOTAL_EPOCHS} (${PROGRESS}%)"
    echo "剩餘 epochs: ${REMAINING}"
    echo "預估剩餘時間: ~${EST_TIME} 分鐘"
else
    echo "無法解析當前 epoch 數"
fi

echo ""
echo "========================================="
echo -e "${YELLOW}🔄 持續監控中... (每30秒刷新)${NC}"
echo "========================================="
