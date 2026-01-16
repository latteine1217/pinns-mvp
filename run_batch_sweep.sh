#!/bin/bash
# =============================================================================
# Batch Size Sweep 實驗 - 批次提交腳本
# =============================================================================
# 目的: 測試不同 batch size 對訓練效能的影響（8k, 16k, 24k, 32k）
# 
# 使用方式:
#   bash run_batch_sweep.sh        # 提交所有任務
#   bash run_batch_sweep.sh --dry  # 只顯示命令不執行
# =============================================================================

set -e  # 遇到錯誤立即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 解析參數
DRY_RUN=false
if [[ "$1" == "--dry" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}⚠️  Dry Run 模式：只顯示命令不實際執行${NC}"
    echo ""
fi

# 配置文件列表
CONFIGS=(
    "configs/batch_test_8k.yml"
    "configs/batch_test_16k.yml"
    "configs/batch_test_24k.yml"
    "configs/batch_test_32k.yml"
)

# Batch Size 列表（用於顯示）
BATCH_SIZES=(
    "8k (baseline)"
    "16k (2x)"
    "24k (3x)"
    "32k (4x, may OOM)"
)

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🔍 Batch Size Sweep 實驗提交腳本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "實驗設定:"
echo "  - 測試 batch sizes: 8k, 16k, 24k, 32k"
echo "  - 每個配置訓練 10 epochs（SLURM script 自動修改）"
echo "  - 使用 cProfile 進行效能分析"
echo "  - 預期時間: 每個任務 ~5 分鐘"
echo ""

# 檢查配置文件是否存在
echo -e "${YELLOW}📝 檢查配置文件...${NC}"
ALL_EXIST=true
for config in "${CONFIGS[@]}"; do
    if [[ ! -f "$config" ]]; then
        echo -e "${RED}❌ 配置文件不存在: $config${NC}"
        ALL_EXIST=false
    else
        echo -e "${GREEN}✅ $config${NC}"
    fi
done

if [[ "$ALL_EXIST" == false ]]; then
    echo ""
    echo -e "${RED}❌ 部分配置文件缺失，請先創建配置文件${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}📤 提交 SLURM 任務...${NC}"
echo ""

# 提交任務
JOB_IDS=()
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    batch_size="${BATCH_SIZES[$i]}"
    
    echo -e "${BLUE}[$(($i+1))/${#CONFIGS[@]}] 提交: $batch_size${NC}"
    echo "  Config: $config"
    
    if [[ "$DRY_RUN" == true ]]; then
        echo "  Command: sbatch --export=CONFIG=$config slurm_profile_simple.sh"
        JOB_ID="DRYRUN_$i"
    else
        # 實際提交任務
        output=$(sbatch --export=CONFIG=$config slurm_profile_simple.sh 2>&1)
        
        if [[ $? -eq 0 ]]; then
            # 提取 Job ID
            JOB_ID=$(echo $output | grep -oP 'Submitted batch job \K\d+')
            echo -e "  ${GREEN}✅ 已提交 Job ID: $JOB_ID${NC}"
            JOB_IDS+=($JOB_ID)
        else
            echo -e "  ${RED}❌ 提交失敗: $output${NC}"
        fi
    fi
    
    echo ""
    sleep 1  # 避免過快提交
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ 所有任務已提交完成${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [[ "$DRY_RUN" == false ]]; then
    echo "已提交的 Job IDs:"
    for i in "${!JOB_IDS[@]}"; do
        echo "  - ${JOB_IDS[$i]} (${BATCH_SIZES[$i]})"
    done
    echo ""
    
    echo "監控命令:"
    echo "  watch -n 5 'squeue -u \$USER'"
    echo ""
    echo "查看日誌:"
    echo "  tail -f logs/profile_simple_<JOB_ID>.log"
    echo ""
    echo "結果分析:"
    echo "  python3 scripts/analyze_batch_sweep.py"
fi
