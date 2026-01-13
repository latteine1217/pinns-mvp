#!/bin/bash
#SBATCH --job-name=s2_k_scan
#SBATCH --output=logs/slurm/s2_k_scan_%A_%a.out
#SBATCH --error=logs/slurm/s2_k_scan_%A_%a.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --mem=108G
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-3

# ===================================================================
# S2 K-Scan SLURM 陣列任務（平行執行）
# ===================================================================
# 用途：使用 SLURM 陣列任務平行執行所有 K 值的訓練
# 
# 提交方式：
#   sbatch scripts/experiments/run_s2_k_scan_slurm.sh
#
# 說明：
#   - 每個 K 值會分配到獨立的 GPU
#   - 可同時執行 4 個實驗（如果資源允許）
#   - 每個任務最多執行 14 天
#   - 使用伺服器上可用的感測器文件: K=50, 100, 200, 400
# ===================================================================

# 設定環境
source ~/.bashrc
cd /home/junyi/pinns-sparse-flow || exit 1

# 設定 Python 路徑
export PYTHONPATH="/home/junyi/pinns-sparse-flow:$PYTHONPATH"

# 添加 torchrun 到 PATH
export PATH="/home/junyi/.local/bin:$PATH"

# 禁用 WandB（伺服器無法連接外網）
export WANDB_MODE=disabled

# K 值列表（使用實際可用的感測器文件）
K_VALUES=(50 100 200 400)

# 根據 SLURM_ARRAY_TASK_ID 選擇 K 值
K=${K_VALUES[$SLURM_ARRAY_TASK_ID]}

# 配置檔案
CONFIG_FILE="configs/experiments/S2_k_scan/s2_qr_K${K}_2d_re100.yml"

# 建立日誌目錄
mkdir -p logs/slurm

echo "================================"
echo "🚀 S2 K-Scan Training: K=${K}"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "Config: $CONFIG_FILE"
echo "Start time: $(date)"
echo ""

# 檢查配置檔案
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 啟動訓練（使用 torchrun 啟動 DDP）
echo "🚀 Launching DDP training with 2 GPUs..."
torchrun --standalone --nnodes=1 --nproc_per_node=2 \
    scripts/train/train.py --cfg "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "================================"
echo "Training Completed: K=${K}"
echo "================================"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training successful"
    
    # 檢查 checkpoint
    CKPT_DIR="checkpoints/experiments/S2_K${K}"
    if [ -d "$CKPT_DIR" ]; then
        echo "📁 Checkpoint saved to: $CKPT_DIR"
        ls -lh "$CKPT_DIR"
    fi
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi

exit $EXIT_CODE
