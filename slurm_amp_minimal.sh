#!/bin/bash
#SBATCH --job-name=amp_minimal
#SBATCH --output=logs/amp_minimal_%j.log
#SBATCH --error=logs/amp_minimal_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=r740
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"

# 項目目錄
cd ~/pinns-sparse-flow

# 創建目錄
mkdir -p logs results

# 打印環境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "=========================================="
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo "=========================================="

# 執行最小化 AMP profiling
echo ""
echo "=========================================="
echo "🚀 開始最小化 AMP Profiling 測試"
echo "=========================================="
echo "Batch Size: 8000"
echo "Iterations: 50"
echo ""

python3 scripts/profile_amp_minimal.py \
    --batch-size 8000 \
    --iterations 50

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ AMP Profiling 完成"
    echo "結果已保存至: results/amp_minimal_profiling.json"
else
    echo "❌ AMP Profiling 失敗 (Exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
