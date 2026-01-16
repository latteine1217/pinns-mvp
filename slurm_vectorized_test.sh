#!/bin/bash
#SBATCH --job-name=vec_test_100ep
#SBATCH --output=logs/slurm_vectorized_test_%j.out
#SBATCH --error=logs/slurm_vectorized_test_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=r740
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# 向量化殘差優化測試 (100 epochs, Kolmogorov Re=50)
# 目標：驗證 1.77x 加速是否反映在端到端訓練時間

echo "========================================"
echo "🚀 Vectorized Residual Test (100 Epochs)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo ""

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"

# 確認 GPU
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# 工作目錄
cd ~/pinns-sparse-flow

# 環境檢查
echo "Environment Check:"
python3 --version
echo ""

# 備份檢查
echo "Checking residuals.py modification:"
grep -n "向量化優化版本" pinnx/losses/residuals.py | head -2
echo ""

# 執行訓練
echo "Starting training..."
echo "Config: configs/vectorized_test_100ep.yml"
echo ""

srun python3 scripts/train/train.py --cfg configs/vectorized_test_100ep.yml

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Training completed with exit code: $EXIT_CODE"
echo "End: $(date)"
echo "========================================"

# 如果成功，顯示關鍵指標
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "📊 Training Summary:"
    echo "-------------------"
    
    # 查找最新的訓練日誌
    LOG_DIR="./results/vectorized_test_100ep"
    if [ -d "$LOG_DIR" ]; then
        echo "Results saved to: $LOG_DIR"
        
        # 顯示最終 epoch 的時間統計（如果有）
        LATEST_LOG=$(ls -t logs/slurm_vectorized_test_*.out 2>/dev/null | head -1)
        if [ -n "$LATEST_LOG" ]; then
            echo ""
            echo "Last 10 epochs timing:"
            grep -E "Epoch.*Time:" "$LATEST_LOG" | tail -10
        fi
    fi
fi

exit $EXIT_CODE
