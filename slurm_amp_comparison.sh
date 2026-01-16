#!/bin/bash
#SBATCH --job-name=amp_compare
#SBATCH --output=logs/amp_comparison_%j.log
#SBATCH --error=logs/amp_comparison_%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=r740
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# 環境設定
export PYTHONPATH="${HOME}/pinns-sparse-flow:${PYTHONPATH}"
export WANDB_MODE=disabled  # 禁用 WandB

# 項目目錄
cd ~/pinns-sparse-flow

# 創建日誌目錄
mkdir -p logs
mkdir -p results

# 打印環境信息
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=========================================="
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
echo "=========================================="
python3 --version
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'cuDNN: {torch.backends.cudnn.version()}')"
echo "=========================================="

# 檢查數據文件是否存在
echo "檢查數據文件..."
if [ ! -f "./data/kolmogorov_dns/kolmogorov_dns_100.npy" ]; then
    echo "⚠️  警告: DNS 數據文件不存在"
    echo "路徑: ./data/kolmogorov_dns/kolmogorov_dns_100.npy"
fi

if [ ! -f "./data/kolmogorov_sensors/re100/sensors_temporal_K200_N256_t0-20.json" ]; then
    echo "⚠️  警告: 感測器數據文件不存在"
    echo "路徑: ./data/kolmogorov_sensors/re100/sensors_temporal_K200_N256_t0-20.json"
fi

if [ ! -f "./data/kolmogorov_sensors/re100/sensors_temporal_K200_N256_t0-20_dns_values.npz" ]; then
    echo "⚠️  警告: DNS values 數據文件不存在"
    echo "路徑: ./data/kolmogorov_sensors/re100/sensors_temporal_K200_N256_t0-20_dns_values.npz"
fi

echo ""

# 執行 AMP 對比測試
echo "=========================================="
echo "🚀 開始 FP32 vs FP16 (AMP) 效能對比測試"
echo "=========================================="
echo "Epochs: 10"
echo "FP32 Config: configs/amp_test_fp32.yml"
echo "FP16 Config: configs/amp_test_fp16.yml"
echo ""

python3 scripts/profile_amp_comparison.py \
    --fp32_cfg configs/amp_test_fp32.yml \
    --fp16_cfg configs/amp_test_fp16.yml \
    --epochs 10

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ AMP 對比測試完成"
    echo "結果已保存至: results/amp_comparison_results.json"
else
    echo "❌ AMP 對比測試失敗 (Exit code: $EXIT_CODE)"
fi
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
