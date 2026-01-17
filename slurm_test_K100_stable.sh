#!/bin/bash
#SBATCH --job-name=K100_stable
#SBATCH --output=/home/junyi/logs/test_K100_stable_%j.out
#SBATCH --error=/home/junyi/logs/test_K100_stable_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=108G

# ========================================
# 測試目標：Gradient Clipping + Adaptive Weighting
# ========================================
# 配置：
#   - gradient_clip: 1.0
#   - adaptive_weighting: true (GradNorm)
#   - weight_update_freq: 1000
#   - batch_size: 12000
#   - N_pde: 9000
# ========================================

PROJECT_DIR="${HOME}/pinns-sparse-flow"
CONFIG_FILE="configs/experiments/S2_k_scan/s2_qr_K100_2d_re100.yml"

mkdir -p ${HOME}/logs

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export WANDB_MODE=offline
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29501
export PINNX_DETECT_ANOMALY=0

# NCCL 配置
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0

# WandB 配置
export WANDB_API_KEY=daf43f72d9f4f636dc69479c446ace76a4a3eb92
export WANDB_PROJECT=pinns-turbulence-reconstruction

# Python 環境
if [ -f "${HOME}/python/bin/activate" ]; then
    source "${HOME}/python/bin/activate"
    export PATH="${HOME}/bin:${HOME}/python/bin:${PATH}"
    export PATH="${HOME}/.local/bin:${PATH}"
else
    echo "❌ Python 環境不存在"
    exit 1
fi

cd ${PROJECT_DIR}

echo "===================================="
echo "K100 穩定性測試 (Gradient Clipping + GradNorm)"
echo "===================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "配置:"
echo "  - Config: ${CONFIG_FILE}"
echo "  - Gradient clip: 1.0"
echo "  - Adaptive weighting: GradNorm (update_freq=1000)"
echo "  - Batch size: 12000"
echo "  - N_pde: 9000"
echo "===================================="
echo ""

python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

echo ""
nvidia-smi

echo ""
echo "===================================="
echo "開始訓練..."
echo "===================================="

START_TIME=$(date +%s)

torchrun --nproc_per_node=2 scripts/train/train.py --cfg ${CONFIG_FILE}
EXIT_CODE=$?

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "===================================="
echo "測試完成"
echo "===================================="
echo "Exit code: ${EXIT_CODE}"
echo "End time: $(date)"
echo "總耗時: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "===================================="

nvidia-smi
