#!/bin/bash
# ===================================================================
# S1 Sensor Strategy Sweep - Colab 執行腳本 (K=200)
# ===================================================================
# 實驗目的：比較 QR-Pivot vs Random 感測器策略
# 配置數量：2 個 (QR, Random)
# 感測器數量：K=200
# 預估時間：6-12 小時
# ===================================================================

set -e

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 S1 Sensor Strategy Sweep (K=200)${NC}"
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
EXPERIMENT_DIR="configs/experiments/S1_sensor_strategy"
STRATEGIES=("qr" "random")
K_VALUE=200

# 日誌設置
LOG_DIR="logs/experiments/S1_sensor_sweep"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/s1_K${K_VALUE}_run_$(date +%Y%m%d_%H%M%S).log"

echo -e "${BLUE}實驗配置：${NC}"
echo "  - 感測器策略：QR-Pivot vs Random"
echo "  - 感測器數量：K=$K_VALUE"
echo "  - 配置數量：${#STRATEGIES[@]}"
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

# 執行每個策略
for strategy in "${STRATEGIES[@]}"; do
    CONFIG_FILE="$EXPERIMENT_DIR/s1_${strategy}_K${K_VALUE}_2d_re50.yml"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}▶ 開始訓練: ${strategy^^} 策略 (K=${K_VALUE})${NC}"
    echo -e "${YELLOW}================================${NC}"
    echo "配置檔案: $CONFIG_FILE"
    echo "開始時間: $(date)"
    echo ""
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ 配置檔案不存在: $CONFIG_FILE${NC}"
        FAILED_LIST+=("$strategy")
        continue
    fi
    
    # 執行訓練
    EXP_START_TIME=$(date +%s)
    
    # 處理 epochs 覆蓋
    ACTUAL_CONFIG="$CONFIG_FILE"
    if [ -n "$COLAB_EPOCHS" ]; then
        TEMP_CONFIG="/tmp/s1_${strategy}_K${K_VALUE}_temp.yml"
        cp "$CONFIG_FILE" "$TEMP_CONFIG"
        sed -i "s/epochs: [0-9]*/epochs: $COLAB_EPOCHS/" "$TEMP_CONFIG"
        ACTUAL_CONFIG="$TEMP_CONFIG"
        echo -e "${GREEN}✓ 使用臨時配置 (epochs=$COLAB_EPOCHS)${NC}"
    fi
    
    if python scripts/train/train.py --cfg "$ACTUAL_CONFIG" 2>&1 | tee -a "$LOG_FILE"; then
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${GREEN}✅ ${strategy^^} 策略訓練完成${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        SUCCESS_LIST+=("$strategy")
        
        # 備份到 Google Drive
        if [ -d "/content/drive/MyDrive" ]; then
            BACKUP_DIR="/content/drive/MyDrive/pinns_checkpoints/S1_${strategy}_K${K_VALUE}"
            mkdir -p "$BACKUP_DIR"
            cp -r "checkpoints/experiments/S1_${strategy}_K${K_VALUE}" "$BACKUP_DIR/" 2>/dev/null && \
                echo -e "${GREEN}✓ 已備份至 Google Drive${NC}"
        fi
    else
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${RED}❌ ${strategy^^} 策略訓練失敗${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m"
        FAILED_LIST+=("$strategy")
    fi
    
    echo ""
done

# 總結
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}📊 S1 實驗完成 (K=${K_VALUE})${NC}"
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
for strategy in "${STRATEGIES[@]}"; do
    CKPT_DIR="checkpoints/experiments/S1_${strategy}_K${K_VALUE}"
    if [ -d "$CKPT_DIR" ]; then
        echo "  ✓ $CKPT_DIR"
        du -sh "$CKPT_DIR"
    else
        echo "  ✗ $CKPT_DIR (未找到)"
    fi
done

echo ""
echo "📊 評估指令："
echo "python scripts/evaluate_unified.py \\"
echo "  --checkpoints \\"
echo "    checkpoints/experiments/S1_qr_K${K_VALUE}/best_model.pth \\"
echo "    checkpoints/experiments/S1_random_K${K_VALUE}/best_model.pth \\"
echo "  --labels \"QR-Pivot (K=${K_VALUE})\" \"Random (K=${K_VALUE})\" \\"
echo "  --output results/S1_K${K_VALUE}_sensor_strategy_comparison.png"

echo ""
echo "✨ 完成！"
