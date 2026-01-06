#!/bin/bash
# ===================================================================
# S2 K-Scan 批次執行腳本（順序執行）
# ===================================================================
# 用途：依序執行所有 K 值的訓練實驗
# 
# 執行方式：
#   bash scripts/experiments/run_s2_k_scan_sequential.sh
# ===================================================================

set -e  # 遇到錯誤立即停止

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
PROJECT_ROOT="/Users/latteine/Documents/coding/pinns-sparse-flow"
TRAIN_SCRIPT="scripts/train/train.py"
EXPERIMENT_DIR="configs/experiments/S2_k_scan"

# K 值列表（按推薦順序）
K_VALUES=(30 50 80 100 200)

# 記錄開始時間
START_TIME=$(date +%s)
LOG_FILE="logs/s2_k_scan_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 S2 K-Scan 批次訓練開始${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "開始時間: $(date)"
echo "K 值列表: ${K_VALUES[@]}"
echo "日誌檔案: $LOG_FILE"
echo ""

# 記錄成功/失敗的實驗
SUCCESS_LIST=()
FAILED_LIST=()

# 依序執行每個 K 值
for K in "${K_VALUES[@]}"; do
    CONFIG_FILE="$EXPERIMENT_DIR/s2_qr_K${K}_2d_re50.yml"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}▶ 開始訓練: K=${K}${NC}"
    echo -e "${YELLOW}================================${NC}"
    echo "配置檔案: $CONFIG_FILE"
    echo "開始時間: $(date)"
    echo ""
    
    # 執行訓練
    EXP_START_TIME=$(date +%s)
    
    if python "$TRAIN_SCRIPT" --cfg "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${GREEN}✅ K=${K} 訓練完成${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        SUCCESS_LIST+=("K=$K")
    else
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${RED}❌ K=${K} 訓練失敗${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        FAILED_LIST+=("K=$K")
        
        # 詢問是否繼續
        echo ""
        echo -e "${YELLOW}是否繼續下一個實驗？ (y/n)${NC}"
        read -r CONTINUE
        if [[ "$CONTINUE" != "y" ]]; then
            echo "用戶中止執行"
            break
        fi
    fi
    
    echo ""
    echo "---"
    echo ""
done

# 計算總時間
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

# 輸出總結
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}📊 S2 K-Scan 批次訓練完成${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "結束時間: $(date)"
echo "總耗時: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""

if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ 成功的實驗 (${#SUCCESS_LIST[@]}):${NC}"
    for exp in "${SUCCESS_LIST[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo -e "${RED}❌ 失敗的實驗 (${#FAILED_LIST[@]}):${NC}"
    for exp in "${FAILED_LIST[@]}"; do
        echo "  - $exp"
    done
    echo ""
fi

# 檢查 checkpoint 位置
echo "📁 Checkpoints 位置:"
for K in "${K_VALUES[@]}"; do
    CKPT_DIR="checkpoints/experiments/S2_qr_K${K}"
    if [ -d "$CKPT_DIR" ]; then
        echo "  ✓ $CKPT_DIR"
    else
        echo "  ✗ $CKPT_DIR (未找到)"
    fi
done

echo ""
echo "📊 下一步: 執行評估比較"
echo "  python scripts/evaluate_unified.py \\"
echo "    --checkpoints \\"
for K in "${K_VALUES[@]}"; do
    echo "      checkpoints/experiments/S2_qr_K${K}/best_model.pth \\"
done
echo "    --labels K=30 K=50 K=80 K=100 K=200"

echo ""
echo "✨ 完成！"
