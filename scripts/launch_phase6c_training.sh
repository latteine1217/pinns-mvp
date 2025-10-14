#!/bin/bash
# TASK-008 Phase 6C 訓練啟動腳本
# 用途：啟動均值約束修正訓練（Warm start from Phase 6B）

set -e  # 遇錯即停

echo "╔════════════════════════════════════════════════════════════╗"
echo "║       TASK-008 Phase 6C: 均值約束修正訓練                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📋 實驗配置："
echo "  • 配置檔: configs/test_rans_phase6c_v1.yml"
echo "  • Warm Start: checkpoints/test_rans_phase6b_extended_2k/best_model.pth"
echo "  • 輸出目錄: checkpoints/test_rans_phase6c_v1"
echo "  • 訓練 epochs: 3000"
echo ""
echo "🎯 目標："
echo "  ✅ 修正 +665% 均值偏移 → < 10%"
echo "  ✅ 保留 Fourier Features 高頻學習能力"
echo "  ✅ Mean Constraint Loss 穩定收斂"
echo ""
echo "⏰ 預計訓練時間: 12-24 小時"
echo "════════════════════════════════════════════════════════════"
echo ""

# 檢查配置檔是否存在
if [ ! -f "configs/test_rans_phase6c_v1.yml" ]; then
    echo "❌ 錯誤: 配置檔不存在"
    exit 1
fi

# 檢查 warm start checkpoint
if [ ! -f "checkpoints/test_rans_phase6b_extended_2k/best_model.pth" ]; then
    echo "⚠️  警告: Warm start checkpoint 不存在"
    echo "   將從頭開始訓練（不建議）"
fi

# 創建輸出目錄
mkdir -p checkpoints/test_rans_phase6c_v1
mkdir -p log/test_rans_phase6c_v1

echo "🚀 啟動訓練..."
echo ""

# 執行訓練（背景運行）
nohup python scripts/train.py \
    --cfg configs/test_rans_phase6c_v1.yml \
    > log/test_rans_phase6c_v1/training_stdout.log 2>&1 &

TRAIN_PID=$!
echo "✅ 訓練已在背景啟動 (PID: ${TRAIN_PID})"
echo ""
echo "📊 監控指令："
echo "  • 即時監控: ./scripts/monitor_phase6c.sh"
echo "  • 查看日誌: tail -f log/test_rans_phase6c_v1/training.log"
echo "  • 查看輸出: tail -f log/test_rans_phase6c_v1/training_stdout.log"
echo "  • 停止訓練: kill ${TRAIN_PID}"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "訓練已啟動！使用上述指令監控進度。"
echo "════════════════════════════════════════════════════════════"
