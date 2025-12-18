#!/bin/bash
# =============================================================================
# Phase A Training Monitor - 實時追蹤訓練進度
# =============================================================================

echo "================================================================================"
echo "🔍 Phase A Training Monitor - QR vs Random Baseline"
echo "================================================================================"
echo ""

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 檢查訓練進程
echo "📊 Training Processes:"
ps aux | grep "train.py" | grep -v grep | while read line; do
    echo "  ✅ $line"
done
echo ""

# QR-Pivot 訓練進度
echo "================================================================================"
echo "${BLUE}📈 QR-Pivot Baseline Progress${NC}"
echo "================================================================================"
QR_LOG=$(ls -t logs/phase_a_qr_baseline_*.log 2>/dev/null | head -1)
if [ -f "$QR_LOG" ]; then
    echo "Log: $QR_LOG"
    echo ""
    
    # 提取最新 epoch 信息
    echo "最新 5 個 Epochs:"
    grep "Epoch [0-9]*/500" "$QR_LOG" | tail -5 | while read line; do
        epoch=$(echo "$line" | grep -oP 'Epoch \K[0-9]+')
        loss=$(echo "$line" | grep -oP 'total_loss: \K[0-9.]+')
        data_loss=$(echo "$line" | grep -oP 'data_loss: \K[0-9.]+')
        pde_loss=$(echo "$line" | grep -oP 'pde_loss: \K[0-9.]+')
        echo "  Epoch $epoch: Loss=$loss (data=$data_loss, pde=$pde_loss)"
    done
    echo ""
    
    # 檢查錯誤
    errors=$(grep -i "error\|exception\|traceback\|nan\|inf" "$QR_LOG" | tail -3)
    if [ ! -z "$errors" ]; then
        echo "${RED}⚠️ Recent Errors/Warnings:${NC}"
        echo "$errors"
    else
        echo "${GREEN}✅ No errors detected${NC}"
    fi
else
    echo "${RED}❌ No log file found${NC}"
fi

echo ""
echo "================================================================================"
echo "${BLUE}📈 Random Baseline Progress${NC}"
echo "================================================================================"
RANDOM_LOG=$(ls -t logs/phase_a_random_baseline_*.log 2>/dev/null | head -1)
if [ -f "$RANDOM_LOG" ]; then
    echo "Log: $RANDOM_LOG"
    echo ""
    
    # 提取最新 epoch 信息
    echo "最新 5 個 Epochs:"
    grep "Epoch [0-9]*/500" "$RANDOM_LOG" | tail -5 | while read line; do
        epoch=$(echo "$line" | grep -oP 'Epoch \K[0-9]+')
        loss=$(echo "$line" | grep -oP 'total_loss: \K[0-9.]+')
        data_loss=$(echo "$line" | grep -oP 'data_loss: \K[0-9.]+')
        pde_loss=$(echo "$line" | grep -oP 'pde_loss: \K[0-9.]+')
        echo "  Epoch $epoch: Loss=$loss (data=$data_loss, pde=$pde_loss)"
    done
    echo ""
    
    # 檢查錯誤
    errors=$(grep -i "error\|exception\|traceback\|nan\|inf" "$RANDOM_LOG" | tail -3)
    if [ ! -z "$errors" ]; then
        echo "${RED}⚠️ Recent Errors/Warnings:${NC}"
        echo "$errors"
    else
        echo "${GREEN}✅ No errors detected${NC}"
    fi
else
    echo "${RED}❌ No log file found${NC}"
fi

echo ""
echo "================================================================================"
echo "📁 Checkpoints & Results"
echo "================================================================================"

# QR checkpoints
if [ -d "checkpoints/phase_a_qr_baseline" ]; then
    qr_ckpts=$(ls checkpoints/phase_a_qr_baseline/*.pth 2>/dev/null | wc -l)
    qr_best=$(ls -lh checkpoints/phase_a_qr_baseline/best_model.pth 2>/dev/null)
    echo "${GREEN}QR-Pivot:${NC} $qr_ckpts checkpoints"
    [ ! -z "$qr_best" ] && echo "  Best model: $qr_best"
fi

# Random checkpoints
if [ -d "checkpoints/phase_a_random_baseline" ]; then
    random_ckpts=$(ls checkpoints/phase_a_random_baseline/*.pth 2>/dev/null | wc -l)
    random_best=$(ls -lh checkpoints/phase_a_random_baseline/best_model.pth 2>/dev/null)
    echo "${GREEN}Random:${NC} $random_ckpts checkpoints"
    [ ! -z "$random_best" ] && echo "  Best model: $random_best"
fi

echo ""
echo "================================================================================"
echo "⏱️  Estimated Completion"
echo "================================================================================"

# 估算完成時間（基於 50 epoch 快速測試的時間）
# 50 epochs = ~300 seconds, so 500 epochs ≈ 3000 seconds ≈ 50 minutes
if [ -f "$QR_LOG" ]; then
    qr_epoch=$(grep "Epoch [0-9]*/500" "$QR_LOG" | tail -1 | grep -oP 'Epoch \K[0-9]+')
    if [ ! -z "$qr_epoch" ]; then
        remaining=$((500 - qr_epoch))
        minutes=$((remaining * 6 / 60))
        echo "QR-Pivot: Epoch $qr_epoch/500 → ~$minutes minutes remaining"
    fi
fi

if [ -f "$RANDOM_LOG" ]; then
    random_epoch=$(grep "Epoch [0-9]*/500" "$RANDOM_LOG" | tail -1 | grep -oP 'Epoch \K[0-9]+')
    if [ ! -z "$random_epoch" ]; then
        remaining=$((500 - random_epoch))
        minutes=$((remaining * 6 / 60))
        echo "Random: Epoch $random_epoch/500 → ~$minutes minutes remaining"
    fi
fi

echo ""
echo "================================================================================"
echo "💡 Commands"
echo "================================================================================"
echo "  📊 TensorBoard: tensorboard --logdir runs/"
echo "  📝 Tail logs: tail -f $QR_LOG"
echo "  🛑 Stop training: pkill -f train.py"
echo "  🔄 Re-run monitor: bash monitor_training.sh"
echo "================================================================================"
