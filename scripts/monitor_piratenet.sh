#!/bin/bash
# PirateNet 訓練監控腳本

LOG_FILE="log/piratenet_quick_test/training_stdout.log"
CHECKPOINT_DIR="checkpoints/piratenet_quick_test"

echo "========================================"
echo "PirateNet 訓練監控"
echo "========================================"
echo ""

# 檢查訓練進程
PID=$(pgrep -f "train.py.*piratenet_quick_test")
if [ -z "$PID" ]; then
    echo "⚠️  訓練進程未運行"
else
    echo "✅ 訓練進程運行中 (PID: $PID)"
fi
echo ""

# 顯示最新訓練進度
echo "📊 最新訓練進度:"
echo "----------------------------------------"
tail -15 "$LOG_FILE" | grep "Epoch" | tail -5
echo ""

# 損失趨勢分析
echo "📈 損失趨勢 (最近 20 epochs):"
echo "----------------------------------------"
grep "Epoch" "$LOG_FILE" | tail -20 | awk '{
    match($0, /Epoch ([0-9]+)\/([0-9]+)/, epoch_arr);
    match($0, /total_loss: ([0-9.]+)/, total_arr);
    match($0, /data_loss: ([0-9.]+)/, data_arr);
    match($0, /pde_loss: ([0-9.]+)/, pde_arr);
    printf "Epoch %3d: Total=%.2e | Data=%.2e | PDE=%.2e\n", 
           epoch_arr[1], total_arr[1], data_arr[1], pde_arr[1];
}'
echo ""

# 檢查點狀態
echo "💾 檢查點狀態:"
echo "----------------------------------------"
if [ -d "$CHECKPOINT_DIR" ]; then
    ls -lth "$CHECKPOINT_DIR" | head -6
    echo ""
    echo "檢查點數量: $(ls -1 $CHECKPOINT_DIR/*.pth 2>/dev/null | wc -l)"
else
    echo "⚠️  檢查點目錄不存在"
fi
echo ""

# 訓練時間估算
echo "⏱️  訓練時間估算:"
echo "----------------------------------------"
START_TIME=$(head -20 "$LOG_FILE" | grep "開始訓練" | head -1 | awk '{print $1, $2}')
LATEST_EPOCH=$(grep "Epoch" "$LOG_FILE" | tail -1 | awk '{match($0, /Epoch ([0-9]+)\/([0-9]+)/, arr); print arr[1]}')
TOTAL_EPOCHS=$(grep "Epoch" "$LOG_FILE" | tail -1 | awk '{match($0, /Epoch ([0-9]+)\/([0-9]+)/, arr); print arr[2]}')

if [ ! -z "$LATEST_EPOCH" ] && [ ! -z "$TOTAL_EPOCHS" ]; then
    echo "當前進度: $LATEST_EPOCH / $TOTAL_EPOCHS epochs"
    PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($LATEST_EPOCH/$TOTAL_EPOCHS)*100}")
    echo "完成度: $PROGRESS%"
    
    # 計算平均 epoch 時間
    FIRST_EPOCH_TIME=$(grep "Epoch 0/" "$LOG_FILE" | head -1 | awk '{print $1, $2}')
    LAST_EPOCH_TIME=$(grep "Epoch" "$LOG_FILE" | tail -1 | awk '{print $1, $2}')
    
    if [ ! -z "$FIRST_EPOCH_TIME" ] && [ ! -z "$LAST_EPOCH_TIME" ]; then
        echo "訓練開始: $START_TIME"
        echo "最新更新: $LAST_EPOCH_TIME"
    fi
fi
echo ""

# 警告/錯誤檢查
echo "⚠️  警告與錯誤:"
echo "----------------------------------------"
WARNING_COUNT=$(grep -c "WARNING" "$LOG_FILE" 2>/dev/null || echo 0)
ERROR_COUNT=$(grep -c "ERROR" "$LOG_FILE" 2>/dev/null || echo 0)
echo "警告數: $WARNING_COUNT"
echo "錯誤數: $ERROR_COUNT"

if [ $ERROR_COUNT -gt 0 ]; then
    echo ""
    echo "最近錯誤:"
    grep "ERROR" "$LOG_FILE" | tail -3
fi
echo ""

echo "========================================"
echo "使用 'tail -f $LOG_FILE' 實時監控"
echo "========================================"
