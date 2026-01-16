#!/bin/bash
#SBATCH --job-name=profile_simple
#SBATCH --output=logs/profile_simple_%j.log
#SBATCH --error=logs/profile_simple_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"
export WANDB_MODE=disabled  # 強制禁用 WandB 以避免初始化超時

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

# 複製配置文件並修改 epochs + 確保 wandb 禁用
cp configs/profiling_test.yml /tmp/profiling_simple_$SLURM_JOB_ID.yml
sed -i 's/epochs:.*/epochs: 10/' /tmp/profiling_simple_$SLURM_JOB_ID.yml
sed -i 's/wandb:.*/wandb: false/g' /tmp/profiling_simple_$SLURM_JOB_ID.yml

echo "配置文件檢查 (確認 wandb 已禁用):"
grep -E 'wandb:|epochs:' /tmp/profiling_simple_$SLURM_JOB_ID.yml
echo ""

# 運行標準訓練腳本
echo "🔍 運行效能分析（10 epochs）..."
python3 -m cProfile -o profile_output_$SLURM_JOB_ID.prof scripts/train/train.py \
    --cfg /tmp/profiling_simple_$SLURM_JOB_ID.yml

# 分析 profile 輸出
echo "=========================================="
echo "Top 30 Time-Consuming Functions:"
echo "=========================================="
python3 << EOF
import pstats
from pstats import SortKey

p = pstats.Stats('profile_output_$SLURM_JOB_ID.prof')
p.strip_dirs()
p.sort_stats(SortKey.CUMULATIVE)
p.print_stats(30)
EOF

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
