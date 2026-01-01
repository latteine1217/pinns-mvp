#!/bin/bash
# ====================================================================
# B1 系列實驗執行腳本
# ====================================================================
# 目的: 批次執行 B1.1/B1.2/B1.3 三個 continuity weight sweep 實驗
# 時間: ~60 分鐘 (3 experiments × 20 min)
# 設備: 建議使用 CUDA GPU
# ====================================================================

set -e  # Exit on error

# 設定環境
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1  # For MPS compatibility

# 顏色輸出
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}B1 Series: Continuity Weight Sweep${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo ""
echo "Starting time: $(date)"
echo ""

# B1.1: continuity_weight = 5.0
echo -e "${GREEN}▶ Running B1.1: continuity_weight = 5.0 (2.5× from A3)${NC}"
python scripts/train/train.py \
  --cfg configs/experiments/loss_balance/B1_continuity_5.0/config.yml
echo -e "${GREEN}✓ B1.1 Complete${NC}"
echo ""

# B1.2: continuity_weight = 10.0
echo -e "${GREEN}▶ Running B1.2: continuity_weight = 10.0 (5× from A3)${NC}"
python scripts/train/train.py \
  --cfg configs/experiments/loss_balance/B1_continuity_10.0/config.yml
echo -e "${GREEN}✓ B1.2 Complete${NC}"
echo ""

# B1.3: continuity_weight = 20.0
echo -e "${GREEN}▶ Running B1.3: continuity_weight = 20.0 (10× from A3)${NC}"
python scripts/train/train.py \
  --cfg configs/experiments/loss_balance/B1_continuity_20.0/config.yml
echo -e "${GREEN}✓ B1.3 Complete${NC}"
echo ""

echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}B1 Series Complete!${NC}"
echo -e "${BLUE}=====================================================================${NC}"
echo "End time: $(date)"
echo ""
echo "Results saved in:"
echo "  - checkpoints/loss_balance/B1_continuity_5.0/"
echo "  - checkpoints/loss_balance/B1_continuity_10.0/"
echo "  - checkpoints/loss_balance/B1_continuity_20.0/"
echo ""
echo "Next steps:"
echo "  1. Check final losses in pinnx.log"
echo "  2. Compare with A3 baseline (total_loss=1.759, div=3.4e-4)"
echo "  3. Select best configuration for B4 combined experiments"
