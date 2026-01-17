#!/bin/bash
#SBATCH --job-name=pinns_kflow_re50
#SBATCH --output=/home/junyi/logs/pinns_sparse_kflow_%j.out
#SBATCH --error=/home/junyi/logs/pinns_sparse_kflow_%j.err
#SBATCH --time=336:00:00
#SBATCH --partition=r740
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=108G

# ===========================
# PINNs-SparseFlow: Kolmogorov Flow 稀疏測量重建
# 專案：pinns-sparse-flow
# 場景：Kolmogorov Flow Re=50, K=100 感測點
# ===========================

PROJECT_DIR="${HOME}/pinns-sparse-flow"
CONFIG_FILE="configs/experiments/S2_k_scan/s2_qr_K100_2d_re100.yml"
WORKDIR="${PROJECT_DIR}/results/kflow_re50_K100_${SLURM_JOB_ID}"

mkdir -p ${HOME}/logs
mkdir -p ${WORKDIR}

echo "==========================="
echo "PINNs-SparseFlow Training"
echo "==========================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start Time: $(date)"
echo "==========================="

# 環境變數設定
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export WANDB_MODE=offline
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# 啟動虛擬環境
source ~/python/bin/activate
export PATH="${HOME}/bin:${HOME}/python/bin:${PATH}"

# WandB 配置（從專案 .wandb_config 讀取或手動設置）
if [ -f "${PROJECT_DIR}/.wandb_config" ]; then
    source "${PROJECT_DIR}/.wandb_config"
    echo "✅ WandB API Key loaded from .wandb_config"
else
    echo "⚠️  .wandb_config not found, WandB may not work"
fi

# WandB 模式：offline（訓練完手動同步）或 online（即時上傳）
# 推薦長時間訓練使用 offline，避免網路問題

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
echo "訓練配置"
echo "========================================="
echo "專案: PINNs-SparseFlow v1.3.4"
echo "場景: Kolmogorov Flow Re=50"
echo "配置: ${CONFIG_FILE}"
echo ""
echo "模型架構:"
echo "  - Fourier-SIREN MLP (8×256)"
echo "  - Fourier features: m=16, σ=4.0"
echo "  - Random Weight Factorization (RWF)"
echo ""
echo "物理設定:"
echo "  - Re=50, nu=0.039374, k_f=4"
echo "  - VS-PINN 變分尺度 N=(2,12,2)"
echo "  - LES 湍流先驗"
echo ""
echo "優化策略:"
echo "  - SOAP optimizer (lr=1e-3)"
echo "  - L-BFGS fine-tuning"
echo "  - Curriculum learning (3 stages)"
echo ""
echo "監督資料:"
echo "  - K=100 感測點 (QR-Pivot)"
echo "  - 時間範圍: [15.0, 35.0]"
echo "  - PDE 配點: 15,000-18,000"
echo ""
echo "WandB: ${WANDB_MODE} mode"
echo "========================================="
echo ""

START_TIME=$(date +%s)

# 開始訓練
echo "🚀 開始訓練..."
torchrun --nproc_per_node=2 scripts/train/train.py --cfg ${CONFIG_FILE}

TRAIN_EXIT_CODE=$?
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "==========================="
echo "訓練完成"
echo "Exit Code: ${TRAIN_EXIT_CODE}"
echo "End Time: $(date)"
echo "總耗時: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "==========================="

# 如果訓練成功，執行評估
if [ ${TRAIN_EXIT_CODE} -eq 0 ]; then
    echo ""
    echo "✅ 訓練成功！開始評估模型..."
    
    # 尋找最佳模型 checkpoint
    CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/kolmogorov_re50_kf4_K100_les_prior"
    BEST_MODEL="${CHECKPOINT_DIR}/best_model.pth"
    
    if [ -f "${BEST_MODEL}" ]; then
        echo "📊 評估最佳模型: ${BEST_MODEL}"
        python3 scripts/evaluate/evaluate_checkpoint.py \
            --checkpoint ${BEST_MODEL} \
            --config ${CONFIG_FILE}
        
        EVAL_EXIT_CODE=$?
        echo "評估完成 (Exit Code: ${EVAL_EXIT_CODE})"
    else
        echo "⚠️  最佳模型未找到: ${BEST_MODEL}"
    fi
    
    # 如果使用 offline 模式，提示如何同步
    if [ "${WANDB_MODE}" = "offline" ]; then
        echo ""
        echo "========================================="
        echo "WandB 同步指令"
        echo "========================================="
        echo "訓練使用 offline 模式，請手動同步到 WandB:"
        echo ""
        echo "cd ${PROJECT_DIR}"
        echo "wandb sync wandb/offline-run-*"
        echo ""
        echo "或同步最新一個 run:"
        echo "wandb sync \$(ls -td wandb/offline-run-* | head -1)"
        echo "========================================="
    fi
else
    echo ""
    echo "❌ 訓練失敗 Exit Code: ${TRAIN_EXIT_CODE}"
    echo "請檢查錯誤日誌: ${HOME}/logs/pinns_sparse_kflow_${SLURM_JOB_ID}.err"
fi

echo ""
echo "==========================="
echo "結果檔案位置"
echo "==========================="
echo "工作目錄: ${WORKDIR}"
echo "Checkpoints: ${PROJECT_DIR}/checkpoints/kolmogorov_re50_kf4_K100_les_prior/"
echo "Results: ${PROJECT_DIR}/results/kolmogorov_re50_kf4_K100_les_prior/"
echo "日誌檔案:"
echo "  - stdout: ${HOME}/logs/pinns_sparse_kflow_${SLURM_JOB_ID}.out"
echo "  - stderr: ${HOME}/logs/pinns_sparse_kflow_${SLURM_JOB_ID}.err"
if [ "${WANDB_MODE}" = "online" ]; then
    echo "WandB: https://wandb.ai/<your-entity>/pinns-sparse-flow"
else
    echo "WandB: (offline - 需手動同步)"
fi
echo "==========================="

echo ""
echo "--- 最終 GPU 狀態 ---"
nvidia-smi

echo ""
echo "--- 磁碟使用 ---"
du -sh ${PROJECT_DIR}/checkpoints/ 2>/dev/null || echo "Checkpoint 目錄不存在"
du -sh ${PROJECT_DIR}/results/ 2>/dev/null || echo "Results 目錄不存在"

echo ""
echo "==========================="
echo "作業完成"
echo "==========================="
