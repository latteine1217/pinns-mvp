#!/bin/bash
#SBATCH --job-name=test_torchscript_fusion
#SBATCH --partition=r740
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/torchscript_fusion_%j.out
#SBATCH --error=slurm_logs/torchscript_fusion_%j.err

# ============================================================
# TorchScript Kernel Fusion 性能測試 (P100)
# ============================================================

echo "========================================"
echo "TorchScript Kernel Fusion 測試"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "========================================"

# 環境設置
source ~/python/bin/activate

# 檢查 GPU
echo ""
echo "🖥️  GPU 資訊:"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader
echo ""

# 檢查 Python 環境
echo "🐍 Python 環境:"
which python3
python3 --version
echo ""

echo "📦 PyTorch 版本:"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
echo ""

# 進入專案目錄
cd /home/junyi/pinns-sparse-flow || exit 1

# 創建輸出目錄
mkdir -p profiler_results
mkdir -p slurm_logs

echo "========================================"
echo "🚀 開始測試..."
echo "========================================"
echo ""

# 運行測試腳本
python3 scripts/test_torchscript_fusion.py

TEST_EXIT_CODE=$?

echo ""
echo "========================================"
echo "測試完成"
echo "========================================"
echo "Exit Code: $TEST_EXIT_CODE"
echo "End Time: $(date)"

# 顯示結果文件
echo ""
echo "📊 結果文件:"
if [ -f profiler_results/torchscript_fusion_results.txt ]; then
    echo "✅ profiler_results/torchscript_fusion_results.txt"
    echo ""
    echo "--- 結果預覽 ---"
    cat profiler_results/torchscript_fusion_results.txt
else
    echo "❌ 未找到結果文件"
fi

exit $TEST_EXIT_CODE
