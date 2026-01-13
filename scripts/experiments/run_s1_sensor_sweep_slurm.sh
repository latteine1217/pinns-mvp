#!/bin/bash
#SBATCH --job-name=s1_sweep
#SBATCH --output=logs/slurm/s1_sweep_%A_%a.out
#SBATCH --error=logs/slurm/s1_sweep_%A_%a.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --mem=108G
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

# ===================================================================
# S1 Sensor Strategy Sweep - SLURM 陣列任務（K=200）
# ===================================================================
# 用途：比較 QR-Pivot vs Random 感測器策略 (K=200)
# 
# 提交方式：
#   sbatch scripts/experiments/run_s1_sensor_sweep_slurm.sh
#
# 說明：
#   - 兩個策略會分配到獨立的 GPU 平行執行
#   - QR-Pivot vs Random
#   - 每個任務最多執行 14 天
# ===================================================================

# 設定環境
source ~/.bashrc
cd /home/junyi/pinns-sparse-flow || exit 1

# 策略列表
STRATEGIES=(qr random)

# 根據 SLURM_ARRAY_TASK_ID 選擇策略
STRATEGY=${STRATEGIES[$SLURM_ARRAY_TASK_ID]}

# 配置檔案
CONFIG_FILE="configs/experiments/S1_sensor_strategy/s1_${STRATEGY}_K200_2d_re100.yml"

# 建立日誌目錄
mkdir -p logs/slurm

echo "================================"
echo "🚀 S1 Sensor Strategy: ${STRATEGY^^}"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Strategy: $STRATEGY"
echo "Config: $CONFIG_FILE"
echo "Start time: $(date)"
echo ""

# 檢查配置檔案
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 啟動訓練
python3 scripts/train/train.py --cfg "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "================================"
echo "Training Completed: ${STRATEGY^^}"
echo "================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful"
    
    # 檢查 checkpoint
    CKPT_DIR="checkpoints/experiments/S1_${STRATEGY}_K200"
    if [ -d "$CKPT_DIR" ]; then
        echo "📁 Checkpoint saved to: $CKPT_DIR"
        ls -lh "$CKPT_DIR"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
