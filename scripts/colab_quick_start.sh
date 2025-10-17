#!/bin/bash
# ====================================================================
# Google Colab A100 快速啟動腳本
# ====================================================================
# 用途：在 Colab 環境中一鍵設置並啟動訓練
# 使用：bash scripts/colab_quick_start.sh
# ====================================================================

set -e  # 遇到錯誤立即停止

echo "🚀 Google Colab A100 快速啟動腳本"
echo "===================================="
echo ""

# ====================================================================
# 1. 環境檢查
# ====================================================================
echo "📋 步驟 1/4：環境檢查"
echo "--------------------"
python scripts/colab_setup_check.py
if [ $? -ne 0 ]; then
    echo "❌ 環境檢查失敗，請修正問題後重試"
    exit 1
fi
echo ""

# ====================================================================
# 2. 創建必要目錄
# ====================================================================
echo "📁 步驟 2/4：創建輸出目錄"
echo "------------------------"
mkdir -p checkpoints/normalization_main_2000epochs
mkdir -p log/normalization_main_2000epochs
mkdir -p results/normalization_main_2000epochs
echo "✅ 目錄創建完成"
echo ""

# ====================================================================
# 3. 驗證關鍵文件
# ====================================================================
echo "🔍 步驟 3/4：驗證關鍵文件"
echo "------------------------"

# 檢查配置文件
if [ ! -f "configs/test_normalization_main_2000epochs.yml" ]; then
    echo "❌ 配置文件缺失：configs/test_normalization_main_2000epochs.yml"
    exit 1
fi
echo "✅ 配置文件存在"

# 檢查感測點文件
if [ ! -f "data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz" ]; then
    echo "❌ 感測點文件缺失：data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot.npz"
    echo "   請上傳該文件或從其他位置複製"
    exit 1
fi
echo "✅ 感測點文件存在"
echo ""

# ====================================================================
# 4. 啟動訓練
# ====================================================================
echo "🎯 步驟 4/4：啟動訓練"
echo "-------------------"
echo "配置：configs/test_normalization_main_2000epochs.yml"
echo "模式：背景運行（nohup）"
echo "日誌：log/normalization_main_2000epochs/training.log"
echo ""

# 詢問用戶確認
read -p "是否開始訓練？(y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 訓練已取消"
    exit 0
fi

# 啟動背景訓練
nohup python scripts/train.py \
    --cfg configs/test_normalization_main_2000epochs.yml \
    > log/normalization_main_2000epochs_stdout.log 2>&1 &

TRAIN_PID=$!
echo "✅ 訓練已啟動（PID: $TRAIN_PID）"
echo ""

# ====================================================================
# 使用提示
# ====================================================================
echo "📝 監控與管理指令"
echo "================="
echo ""
echo "查看訓練進程："
echo "  ps aux | grep train.py"
echo ""
echo "實時監控日誌："
echo "  tail -f log/normalization_main_2000epochs/training.log"
echo ""
echo "查看最近 50 行："
echo "  tail -n 50 log/normalization_main_2000epochs/training.log"
echo ""
echo "檢查檢查點："
echo "  ls -lh checkpoints/normalization_main_2000epochs/"
echo ""
echo "停止訓練："
echo "  kill $TRAIN_PID"
echo ""
echo "從檢查點恢復："
echo "  python scripts/train.py --cfg configs/test_normalization_main_2000epochs.yml \\"
echo "    --resume checkpoints/normalization_main_2000epochs/latest.pth"
echo ""
echo "======================================================================"
echo "🎉 訓練已成功啟動！預計完成時間：2-4 小時"
echo "======================================================================"
