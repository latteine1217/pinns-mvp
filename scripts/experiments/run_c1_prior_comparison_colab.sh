#!/bin/bash
# ===================================================================
# C1 Prior Comparison 實驗 - Colab 執行腳本
# ===================================================================
# 實驗目的：比較有無 RANS Prior 對模型效能的影響
# 配置數量：2 個 (With Prior vs No Prior)
# 預估時間：4-8 小時（K=100, 2 個配置）
# ===================================================================

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 C1 Prior Comparison 實驗${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# 專案路徑設置
if [ -d "/content" ]; then
    if [ -d "/content/drive/MyDrive/pinns-sparse-flow" ]; then
        PROJECT_ROOT="/content/drive/MyDrive/pinns-sparse-flow"
    elif [ -d "/content/pinns-sparse-flow" ]; then
        PROJECT_ROOT="/content/pinns-sparse-flow"
    elif [ -f "scripts/train/train.py" ]; then
        PROJECT_ROOT="$(pwd)"
    else
        echo -e "${RED}❌ 找不到專案目錄${NC}"
        exit 1
    fi
else
    PROJECT_ROOT="$(pwd)"
fi

cd "$PROJECT_ROOT" || exit 1
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo -e "${GREEN}✓ 專案目錄: $PROJECT_ROOT${NC}"
echo ""

# 實驗配置
EXPERIMENT_DIR="configs/experiments/C1_prior_comparison"
CONFIGS=("with_prior" "no_prior")
K_VALUE=100

# 日誌設置
LOG_DIR="logs/experiments/C1_prior_comparison"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/c1_run_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}實驗配置：${NC}"
echo "  - 比較目標：RANS Prior 影響"
echo "  - 配置數量：${#CONFIGS[@]}"
echo "  - 感測器數量：K=$K_VALUE"
echo "  - 日誌檔案：$LOG_FILE"
echo ""

# 檢查 epochs 設置
if [ -n "$COLAB_EPOCHS" ]; then
    echo -e "${YELLOW}⚠ 使用自訂 epochs: $COLAB_EPOCHS${NC}"
else
    echo -e "${BLUE}使用配置文件中的 epochs (預設 10000)${NC}"
fi
echo ""

# 記錄開始時間
START_TIME=$(date +%s)
SUCCESS_LIST=()
FAILED_LIST=()

# 執行每個配置
for config_name in "${CONFIGS[@]}"; do
    CONFIG_FILE="$EXPERIMENT_DIR/c1_${config_name}_K${K_VALUE}_2d_re50.yml"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}▶ 開始訓練: ${config_name^^}${NC}"
    echo -e "${YELLOW}================================${NC}"
    echo "配置檔案: $CONFIG_FILE"
    echo "開始時間: $(date)"
    echo ""
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ 配置檔案不存在: $CONFIG_FILE${NC}"
        FAILED_LIST+=("$config_name")
        continue
    fi
    
    # 執行訓練
    EXP_START_TIME=$(date +%s)
    
    # 處理 epochs 覆蓋
    ACTUAL_CONFIG="$CONFIG_FILE"
    if [ -n "$COLAB_EPOCHS" ]; then
        TEMP_CONFIG="/tmp/c1_${config_name}_temp.yml"
        cp "$CONFIG_FILE" "$TEMP_CONFIG"
        sed -i "s/epochs: [0-9]*/epochs: $COLAB_EPOCHS/" "$TEMP_CONFIG"
        ACTUAL_CONFIG="$TEMP_CONFIG"
        echo -e "${GREEN}✓ 使用臨時配置 (epochs=$COLAB_EPOCHS)${NC}"
    fi
    
    if python scripts/train/train.py --cfg "$ACTUAL_CONFIG" 2>&1 | tee -a "$LOG_FILE"; then
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${GREEN}✅ ${config_name^^} 訓練完成${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        SUCCESS_LIST+=("$config_name")
        
        # 備份到 Google Drive
        if [ -d "/content/drive/MyDrive" ]; then
            BACKUP_DIR="/content/drive/MyDrive/pinns_checkpoints/C1_${config_name}"
            mkdir -p "$BACKUP_DIR"
            cp -r "checkpoints/experiments/C1_${config_name}_K${K_VALUE}" "$BACKUP_DIR/" 2>/dev/null && \
                echo -e "${GREEN}✓ 已備份至 Google Drive${NC}"
        fi
    else
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${RED}❌ ${config_name^^} 訓練失敗${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        FAILED_LIST+=("$config_name")
    fi
    
    # 清理臨時配置
    [ -n "$COLAB_EPOCHS" ] && rm -f "$TEMP_CONFIG"
    
    echo ""
done

# 總結
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}📊 C1 實驗完成${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "結束時間: $(date)"
echo "總耗時: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""

if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ 成功的實驗 (${#SUCCESS_LIST[@]}):${NC}"
    for exp in "${SUCCESS_LIST[@]}"; do
        echo "  - ${exp^^}"
    done
    echo ""
fi

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo -e "${RED}❌ 失敗的實驗 (${#FAILED_LIST[@]}):${NC}"
    for exp in "${FAILED_LIST[@]}"; do
        echo "  - ${exp^^}"
    done
    echo ""
fi

# 檢查 checkpoints
echo "📁 Checkpoints:"
for config_name in "${CONFIGS[@]}"; do
    CKPT_DIR="checkpoints/experiments/C1_${config_name}_K${K_VALUE}"
    if [ -d "$CKPT_DIR" ]; then
        echo "  ✓ $CKPT_DIR"
    else
        echo "  ✗ $CKPT_DIR (未找到)"
    fi
done

echo ""
echo "📊 評估指令："
echo "python scripts/evaluate_unified.py \\"
echo "  --checkpoints \\"
echo "    checkpoints/experiments/C1_with_prior_K${K_VALUE}/best_model.pth \\"
echo "    checkpoints/experiments/C1_no_prior_K${K_VALUE}/best_model.pth \\"
echo "  --labels \"With RANS Prior\" \"No Prior\" \\"
echo "  --output results/C1_prior_comparison.png"

echo ""
echo "✨ 完成！"
