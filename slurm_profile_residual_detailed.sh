#!/bin/bash
#SBATCH --job-name=profile_detail
#SBATCH --output=logs/profile_residual_detailed_%j.log
#SBATCH --error=logs/profile_residual_detailed_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"

# 項目目錄
cd ~/pinns-sparse-flow

# 創建日誌目錄
mkdir -p logs

# 打印環境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=========================================="
python3 --version
echo "=========================================="

# 運行詳細效能分析
echo "🔍 運行 PDE Residual 詳細效能分析..."
python3 scripts/profile_residual_detailed.py \
    --config configs/profiling_test.yml \
    --epochs 20

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
