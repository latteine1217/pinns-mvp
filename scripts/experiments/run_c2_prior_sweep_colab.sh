#!/bin/bash
# ===================================================================
# C2 Prior Sweep 實驗 - Colab 執行腳本
# ===================================================================
# 實驗目的：掃描不同 RANS Prior 權重對模型效能的影響
# 配置數量：3 個 (Prior weights: 0.1, 0.3, 0.5)
# 預估時間：6-12 小時（K=100, 3 個配置）
# ===================================================================

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 C2 Prior Sweep 實驗${NC}"
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
EXPERIMENT_DIR="configs/experiments/C2_prior_sweep"
PRIOR_WEIGHTS=("0.1" "0.3" "0.5")
K_VALUE=100

# 日誌設置
LOG_DIR="logs/experiments/C2_prior_sweep"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/c2_run_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}實驗配置：${NC}"
echo "  - 掃描目標：RANS Prior 權重"
echo "  - 權重值：${PRIOR_WEIGHTS[@]}"
echo "  - 配置數量：${#PRIOR_WEIGHTS[@]}"
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

# 執行每個 prior weight
for weight in "${PRIOR_WEIGHTS[@]}"; do
    CONFIG_FILE="$EXPERIMENT_DIR/c2_prior_${weight}_K${K_VALUE}_2d_re50.yml"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}▶ 開始訓練: Prior Weight = $weight${NC}"
    echo -e "${YELLOW}================================${NC}"
    echo "配置檔案: $CONFIG_FILE"
    echo "開始時間: $(date)"
    echo ""
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ 配置檔案不存在: $CONFIG_FILE${NC}"
        FAILED_LIST+=("$weight")
        continue
    fi
    
    # 執行訓練
    EXP_START_TIME=$(date +%s)
    
    # 處理 epochs 覆蓋
    ACTUAL_CONFIG="$CONFIG_FILE"
    if [ -n "$COLAB_EPOCHS" ]; then
        TEMP_CONFIG="/tmp/c2_prior_${weight}_temp.yml"
        cp "$CONFIG_FILE" "$TEMP_CONFIG"
        sed -i "s/epochs: [0-9]*/epochs: $COLAB_EPOCHS/" "$TEMP_CONFIG"
        ACTUAL_CONFIG="$TEMP_CONFIG"
        echo -e "${GREEN}✓ 使用臨時配置 (epochs=$COLAB_EPOCHS)${NC}"
    fi
    
    if python scripts/train/train.py --cfg "$ACTUAL_CONFIG" 2>&1 | tee -a "$LOG_FILE"; then
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${GREEN}✅ Prior Weight = $weight 訓練完成${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        SUCCESS_LIST+=("$weight")
        
        # 備份到 Google Drive
        if [ -d "/content/drive/MyDrive" ]; then
            BACKUP_DIR="/content/drive/MyDrive/pinns_checkpoints/C2_prior_${weight}"
            mkdir -p "$BACKUP_DIR"
            cp -r "checkpoints/experiments/C2_prior_${weight}_K${K_VALUE}" "$BACKUP_DIR/" 2>/dev/null && \
                echo -e "${GREEN}✓ 已備份至 Google Drive${NC}"
        fi
    else
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${RED}❌ Prior Weight = $weight 訓練失敗${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        FAILED_LIST+=("$weight")
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
echo -e "${GREEN}📊 C2 實驗完成${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "結束時間: $(date)"
echo "總耗時: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m"
echo ""

if [ ${#SUCCESS_LIST[@]} -gt 0 ]; then
    echo -e "${GREEN}✅ 成功的實驗 (${#SUCCESS_LIST[@]}):${NC}"
    for exp in "${SUCCESS_LIST[@]}"; do
        echo "  - Prior Weight = $exp"
    done
    echo ""
fi

if [ ${#FAILED_LIST[@]} -gt 0 ]; then
    echo -e "${RED}❌ 失敗的實驗 (${#FAILED_LIST[@]}):${NC}"
    for exp in "${FAILED_LIST[@]}"; do
        echo "  - Prior Weight = $exp"
    done
    echo ""
fi

# 檢查 checkpoints
echo "📁 Checkpoints:"
for weight in "${PRIOR_WEIGHTS[@]}"; do
    CKPT_DIR="checkpoints/experiments/C2_prior_${weight}_K${K_VALUE}"
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
echo "    checkpoints/experiments/C2_prior_0.1_K${K_VALUE}/best_model.pth \\"
echo "    checkpoints/experiments/C2_prior_0.3_K${K_VALUE}/best_model.pth \\"
echo "    checkpoints/experiments/C2_prior_0.5_K${K_VALUE}/best_model.pth \\"
echo "  --labels \"Prior=0.1\" \"Prior=0.3\" \"Prior=0.5\" \\"
echo "  --output results/C2_prior_sweep_comparison.png"

echo ""
echo "✨ 完成！"
