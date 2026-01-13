#!/bin/bash
#SBATCH --job-name=pinns_ddp_profiler
#SBATCH --output=/home/junyi/logs/pinns_ddp_profiler_%j.out
#SBATCH --error=/home/junyi/logs/pinns_ddp_profiler_%j.err
#SBATCH --time=14-00:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=108G

# ===========================
# PINNs-SparseFlow DDP + Profiler 測試
# ===========================

PROJECT_DIR="${HOME}/pinns-sparse-flow"
CONFIG_FILE="configs/experiments/S2_k_scan/s2_qr_K100_2d_re100.yml"
WORKDIR="${PROJECT_DIR}/results/ddp_profiler_${SLURM_JOB_ID}"

mkdir -p ${HOME}/logs
mkdir -p ${WORKDIR}

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export WANDB_MODE=offline
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

if [ -f "${PROJECT_DIR}/.wandb_config" ]; then
    source "${PROJECT_DIR}/.wandb_config"
    echo "✅ WandB API Key loaded from .wandb_config"
else
    echo "⚠️  .wandb_config not found, WandB may not work"
fi

# === Python 環境 ===
REQUIREMENTS_FILE="${PROJECT_DIR}/requirements.txt"

if [ -f "${HOME}/python/bin/activate" ]; then
    source "${HOME}/python/bin/activate"
    export PATH="${HOME}/bin:${HOME}/python/bin:${PATH}"
else
    echo "❌ 找不到 Python 環境: ${HOME}/python/bin/activate"
    exit 1
fi

python -m pip install --upgrade pip
if [ -f "${REQUIREMENTS_FILE}" ]; then
    python3 -m pip install -r "${REQUIREMENTS_FILE}"
else
    echo "❌ 找不到 requirements.txt: ${REQUIREMENTS_FILE}"
    exit 1
fi

cd ${PROJECT_DIR}

echo "==========================="
echo "PINNs-SparseFlow DDP + Profiler"
echo "==========================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "Config: ${CONFIG_FILE}"
echo "Workdir: ${WORKDIR}"
echo "==========================="

python3 --version
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None"

nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv

START_TIME=$(date +%s)

# === DDP 訓練 ===
echo "🚀 開始 DDP 訓練..."
torchrun --nproc_per_node=2 scripts/train/train.py --cfg ${CONFIG_FILE}
TRAIN_EXIT_CODE=$?

# === Profiler 分析 ===
# 使用簡化版 profiler 觀察訓練算子時間分佈
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo "🔬 啟動 Profiler..."
    python3 scripts/train/train_with_profiler.py
    PROFILER_EXIT_CODE=$?
else
    echo "❌ 訓練失敗，跳過 profiler"
    PROFILER_EXIT_CODE=1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "==========================="
echo "作業完成"
echo "Train Exit Code: ${TRAIN_EXIT_CODE}"
echo "Profiler Exit Code: ${PROFILER_EXIT_CODE}"
echo "End Time: $(date)"
echo "總耗時: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "==========================="

echo "Profiler 輸出: ${PROJECT_DIR}/profiler_results"

echo "--- 最終 GPU 狀態 ---"
nvidia-smi
