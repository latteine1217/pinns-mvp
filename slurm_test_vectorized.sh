#!/bin/bash
#SBATCH --job-name=test_vectorized
#SBATCH --output=/home/junyi/logs/test_vectorized_%j.out
#SBATCH --error=/home/junyi/logs/test_vectorized_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

# ===========================
# 向量化 PDE 殘差性能測試
# ===========================

PROJECT_DIR="${HOME}/pinns-sparse-flow"

mkdir -p ${HOME}/logs

echo "==========================="
echo "Vectorized Residual Benchmark"
echo "==========================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "==========================="

# 環境變數設定
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

cd ${PROJECT_DIR}

echo ""
echo "環境檢查:"
echo "---"
python3 --version
echo ""
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

echo ""
echo "GPU 狀態:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

echo ""
echo "========================================="
echo "執行 Benchmark"
echo "========================================="

# 執行測試
srun python3 scripts/test_vectorized_residual.py

EXIT_CODE=$?

echo ""
echo "==========================="
echo "測試完成"
echo "Exit Code: ${EXIT_CODE}"
echo "End Time: $(date)"
echo "==========================="

echo ""
echo "--- 最終 GPU 狀態 ---"
nvidia-smi
