#!/bin/bash
#
# 感測點策略對比實驗腳本
# Sensor Strategy Comparison Experiment
# 
# 功能:
# 1. 對比 QR-Pivot vs Wall-Clustered Random 兩種策略
# 2. 自動執行訓練（可選）
# 3. 生成對比報告
#
# 使用方式:
#   bash scripts/run_sensor_strategy_comparison.sh [--train] [--epochs N]
#
# 參數:
#   --train       執行訓練（預設只生成感測點與報告）
#   --epochs N    訓練輪數（預設 1000）
#
# 作者: PINNs-MVP
# 日期: 2025-10-16

set -e  # 遇到錯誤立即退出

# 預設參數
RUN_TRAINING=false
EPOCHS=1000

# 解析命令列參數
while [[ $# -gt 0 ]]; do
    case $1 in
        --train)
            RUN_TRAINING=true
            shift
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "未知參數: $1"
            echo "使用方式: bash scripts/run_sensor_strategy_comparison.sh [--train] [--epochs N]"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "感測點策略對比實驗"
echo "============================================================"
echo "時間: $(date)"
echo "訓練模式: $RUN_TRAINING"
if [ "$RUN_TRAINING" = true ]; then
    echo "訓練輪數: $EPOCHS"
fi
echo "============================================================"

# 步驟 1: 檢查感測點檔案是否存在
echo ""
echo "【步驟 1】檢查感測點檔案"
echo "-----------------------------------------------------------"

QR_SENSOR_FILE="data/jhtdb/channel_flow_re1000/sensors_K50_velocity_qr_pivot.npz"
WALL_SENSOR_FILE="data/jhtdb/channel_flow_re1000/sensors_K50_wall_clustered.npz"

if [ ! -f "$QR_SENSOR_FILE" ]; then
    echo "❌ QR-Pivot 感測點不存在: $QR_SENSOR_FILE"
    echo "請先執行: python scripts/generate_2d_slice_qr_sensors.py"
    exit 1
fi

if [ ! -f "$WALL_SENSOR_FILE" ]; then
    echo "❌ Wall-Clustered 感測點不存在: $WALL_SENSOR_FILE"
    echo "自動生成..."
    python scripts/generate_wall_clustered_sensors.py \
        --input "data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_velocity_t1.h5" \
        --output "$WALL_SENSOR_FILE" \
        --K 50 \
        --visualize \
        --viz-output results/sensors_wall_clustered/sensors_K50_wall_clustered_analysis.png
fi

echo "✅ 感測點檔案檢查完成"
echo "  QR-Pivot: $(ls -lh $QR_SENSOR_FILE | awk '{print $5}')"
echo "  Wall-Clustered: $(ls -lh $WALL_SENSOR_FILE | awk '{print $5}')"

# 步驟 2: 生成對比報告
echo ""
echo "【步驟 2】生成感測點對比報告"
echo "-----------------------------------------------------------"

python - <<EOF
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 載入資料
qr_data = np.load('$QR_SENSOR_FILE', allow_pickle=True)
wall_data = np.load('$WALL_SENSOR_FILE', allow_pickle=True)

# 創建對比表格
print("\n" + "="*60)
print("感測點策略對比")
print("="*60)

print("\n【策略 A: QR-Pivot】")
print(f"  條件數: {qr_data['condition_number']:.2e}")
print(f"  能量比例: {qr_data['energy_ratio']:.4f}")
coords_qr = qr_data['coords_2d']
y_qr = coords_qr[:, 1]
print(f"  壁面層 (|y|>0.95): {np.sum(np.abs(y_qr) > 0.95)} 點 ({np.sum(np.abs(y_qr) > 0.95)/50*100:.1f}%)")
print(f"  對數層 (0.3<|y|<0.95): {np.sum((np.abs(y_qr) > 0.3) & (np.abs(y_qr) <= 0.95))} 點")
print(f"  中心層 (|y|<0.3): {np.sum(np.abs(y_qr) <= 0.3)} 點")
print(f"  速度均值: u={qr_data['u'].mean():.4f}, v={qr_data['v'].mean():.4f}, w={qr_data['w'].mean():.4f}")

print("\n【策略 B: Wall-Clustered Random】")
print(f"  條件數: N/A (無矩陣分解)")
print(f"  能量比例: N/A")
coords_wall = wall_data['coords_2d']
y_wall = coords_wall[:, 1]
print(f"  壁面層 (|y|>0.95): {np.sum(np.abs(y_wall) > 0.95)} 點 ({np.sum(np.abs(y_wall) > 0.95)/50*100:.1f}%)")
print(f"  對數層 (0.3<|y|<0.95): {np.sum((np.abs(y_wall) > 0.3) & (np.abs(y_wall) <= 0.95))} 點")
print(f"  中心層 (|y|<0.3): {np.sum(np.abs(y_wall) <= 0.3)} 點")
print(f"  速度均值: u={wall_data['u'].mean():.4f}, v={wall_data['v'].mean():.4f}, w={wall_data['w'].mean():.4f}")

print("\n【關鍵差異】")
print(f"  壁面層覆蓋: QR={np.sum(np.abs(y_qr) > 0.95)}點 vs Wall={np.sum(np.abs(y_wall) > 0.95)}點")
print(f"  中心層覆蓋: QR={np.sum(np.abs(y_qr) <= 0.3)}點 vs Wall={np.sum(np.abs(y_wall) <= 0.3)}點")
print(f"  x 標準差: QR={coords_qr[:, 0].std():.2f} vs Wall={coords_wall[:, 0].std():.2f}")
print(f"  y 標準差: QR={y_qr.std():.4f} vs Wall={y_wall.std():.4f}")

print("\n" + "="*60)
EOF

echo "✅ 對比報告生成完成"

# 步驟 3: 訓練（可選）
if [ "$RUN_TRAINING" = true ]; then
    echo ""
    echo "【步驟 3】執行訓練"
    echo "-----------------------------------------------------------"
    
    # 實驗 A: QR-Pivot
    echo ""
    echo "▶ 實驗 A: QR-Pivot 策略訓練"
    nohup python scripts/train.py \
        --cfg configs/test_velocity_qr_pivot_K50.yml \
        > log/qr_pivot_K50_training.log 2>&1 &
    QR_PID=$!
    echo "  PID: $QR_PID"
    echo "  日誌: log/qr_pivot_K50_training.log"
    
    # 等待 5 秒
    sleep 5
    
    # 實驗 B: Wall-Clustered
    echo ""
    echo "▶ 實驗 B: Wall-Clustered 策略訓練"
    nohup python scripts/train.py \
        --cfg configs/test_wall_clustered_random_K50.yml \
        > log/wall_clustered_K50_training.log 2>&1 &
    WALL_PID=$!
    echo "  PID: $WALL_PID"
    echo "  日誌: log/wall_clustered_K50_training.log"
    
    echo ""
    echo "✅ 訓練啟動完成"
    echo ""
    echo "監控訓練進度:"
    echo "  QR-Pivot:        tail -f log/qr_pivot_K50_training.log"
    echo "  Wall-Clustered:  tail -f log/wall_clustered_K50_training.log"
    echo ""
    echo "檢查訓練狀態:"
    echo "  ps aux | grep train.py"
else
    echo ""
    echo "【步驟 3】跳過訓練 (使用 --train 啟用)"
fi

echo ""
echo "============================================================"
echo "感測點策略對比實驗設置完成！"
echo "============================================================"
echo ""
echo "下一步操作:"
echo "  1. 檢查感測點對比報告（見上方輸出）"
echo "  2. 啟動訓練: bash scripts/run_sensor_strategy_comparison.sh --train"
echo "  3. 監控訓練進度（見日誌檔案）"
echo "  4. 訓練完成後對比結果"
echo ""
