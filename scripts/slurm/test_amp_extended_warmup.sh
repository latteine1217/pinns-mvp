#!/bin/bash
#SBATCH --job-name=amp_extended_warmup
#SBATCH --output=logs/amp_extended_warmup_%j.log
#SBATCH --error=logs/amp_extended_warmup_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# AMP 擴展暖身測試
# 目標：通過充分的 GPU 暖身，消除初期性能測量誤差

echo "=================================="
echo "AMP Extended Warmup Test"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# 啟動虛擬環境
if [ -f "${HOME}/python/bin/activate" ]; then
    source "${HOME}/python/bin/activate"
    export PATH="${HOME}/bin:${HOME}/python/bin:${PATH}"
else
    echo "❌ 找不到 Python 環境: ${HOME}/python/bin/activate"
    exit 1
fi

# 顯示環境資訊
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo ""

# 檢查 GPU
nvidia-smi
echo ""

# 執行測試
echo "=================================="
echo "開始測試..."
echo "=================================="
echo ""

python scripts/test_amp_p100_extended_warmup.py

echo ""
echo "=================================="
echo "測試完成"
echo "=================================="
echo "End time: $(date)"
