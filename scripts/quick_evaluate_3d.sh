#!/bin/bash

# 快速 3D 物理評估腳本

# 設定參數
CHECKPOINT="${1:-checkpoints/vs_pinn_3d_full_training_latest.pth}"
CONFIG="${2:-configs/vs_pinn_3d_full_training.yml}"
OUTPUT_DIR="${3:-results/3d_evaluation_$(date +%Y%m%d_%H%M%S)}"

# 顏色定義
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo -e "${BLUE}🔬 3D VS-PINN 物理評估${NC}"
echo "========================================="
echo ""
echo "📦 檢查點: $CHECKPOINT"
echo "⚙️  配置檔: $CONFIG"
echo "📁 輸出目錄: $OUTPUT_DIR"
echo ""

# 檢查檔案存在
if [ ! -f "$CHECKPOINT" ]; then
    echo -e "${RED}❌ 檢查點不存在: $CHECKPOINT${NC}"
    exit 1
fi

if [ ! -f "$CONFIG" ]; then
    echo -e "${RED}❌ 配置檔不存在: $CONFIG${NC}"
    exit 1
fi

# 創建輸出目錄
mkdir -p "$OUTPUT_DIR"

echo -e "${YELLOW}🚀 開始評估...${NC}"
echo ""

# 執行評估
python scripts/evaluate_3d_physics.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output_dir "$OUTPUT_DIR"

EVAL_STATUS=$?

echo ""
echo "========================================="

if [ $EVAL_STATUS -eq 0 ]; then
    echo -e "${GREEN}✅ 評估完成！${NC}"
    echo ""
    echo "📊 結果文件："
    ls -lh "$OUTPUT_DIR"/*.png "$OUTPUT_DIR"/*.npz "$OUTPUT_DIR"/*.md 2>/dev/null | \
        awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
    echo "📖 查看完整報告："
    echo "   cat $OUTPUT_DIR/evaluation_report.md"
else
    echo -e "${RED}❌ 評估失敗 (退出碼: $EVAL_STATUS)${NC}"
    exit $EVAL_STATUS
fi

echo "========================================="
