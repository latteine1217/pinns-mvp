#!/bin/bash
#SBATCH --job-name=amp_ultra
#SBATCH --output=logs/amp_ultra_%j.log
#SBATCH --error=logs/amp_ultra_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=r740
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"

# 項目目錄
cd ~/pinns-sparse-flow
mkdir -p logs results

# 打印環境
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="
python3 --version
echo "=========================================="

# 執行超簡化 AMP profiling
echo "🚀 超簡化 AMP Profiling (SimpleMLP)"
echo "=========================================="

python3 scripts/profile_amp_ultra_minimal.py \
    --batch-size 8000 \
    --iterations 100 \
    --width 768 \
    --depth 2

EXIT_CODE=$?

echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 完成"
else
    echo "❌ 失敗 (Exit: $EXIT_CODE)"
fi
echo "End: $(date)"
echo "=========================================="

exit $EXIT_CODE
