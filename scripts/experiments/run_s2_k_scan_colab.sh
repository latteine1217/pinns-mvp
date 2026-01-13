#!/bin/bash
# ===================================================================
# S2 K-Scan Google Colab 執行腳本
# ===================================================================
# 用途：在 Google Colab 環境中執行 S2 K-scan 實驗
# 
# 使用方式：
#   1. 上傳此腳本到 Colab
#   2. 在 Colab cell 中執行: !bash scripts/experiments/run_s2_k_scan_colab.sh
# ===================================================================

set -e  # 遇到錯誤立即停止

# 顏色輸出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🚀 S2 K-Scan Colab 環境設置${NC}"
echo -e "${GREEN}================================${NC}"
echo ""

# 檢查是否在 Colab 環境並找到專案目錄
if [ -d "/content" ]; then
    echo -e "${GREEN}✓ 偵測到 Google Colab 環境${NC}"
    
    # 嘗試多個可能的專案位置
    if [ -d "/content/drive/MyDrive/pinns-sparse-flow" ]; then
        PROJECT_ROOT="/content/drive/MyDrive/pinns-sparse-flow"
        echo -e "${GREEN}✓ 找到專案於 Google Drive: $PROJECT_ROOT${NC}"
    elif [ -d "/content/pinns-sparse-flow" ]; then
        PROJECT_ROOT="/content/pinns-sparse-flow"
        echo -e "${GREEN}✓ 找到專案於 /content: $PROJECT_ROOT${NC}"
    elif [ -f "scripts/train/train.py" ]; then
        PROJECT_ROOT="$(pwd)"
        echo -e "${GREEN}✓ 使用當前目錄: $PROJECT_ROOT${NC}"
    else
        echo -e "${RED}❌ 錯誤：找不到專案目錄${NC}"
        echo -e "${YELLOW}請確認：${NC}"
        echo -e "${YELLOW}  1. 已掛載 Google Drive${NC}"
        echo -e "${YELLOW}  2. 專案位於 /content/drive/MyDrive/pinns-sparse-flow${NC}"
        echo -e "${YELLOW}  3. 或先 cd 到專案目錄再執行此腳本${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}⚠ 非 Colab 環境，使用當前目錄${NC}"
    PROJECT_ROOT="$(pwd)"
fi

# 切換到專案目錄（如果尚未在專案目錄中）
if [ "$(pwd)" != "$PROJECT_ROOT" ]; then
    echo -e "${BLUE}切換到專案目錄: $PROJECT_ROOT${NC}"
    cd "$PROJECT_ROOT" || {
        echo -e "${RED}❌ 無法切換到專案目錄${NC}"
        exit 1
    }
else
    echo -e "${GREEN}✓ 已在專案目錄中${NC}"
fi

# 設置 PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}✓ PYTHONPATH 已設置: $PYTHONPATH${NC}"

# 驗證 pinnx 模組是否可導入
if python -c "import pinnx" 2>/dev/null; then
    echo -e "${GREEN}✓ pinnx 模組可正常導入${NC}"
else
    echo -e "${YELLOW}⚠ 嘗試安裝 pinnx 模組...${NC}"
    pip install -e . -q || {
        echo -e "${RED}❌ 無法安裝 pinnx 模組${NC}"
        echo -e "${YELLOW}請手動執行: pip install -e .${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ pinnx 模組安裝完成${NC}"
fi

# K 值列表（Colab 建議順序：先跑小的測試）
K_VALUES=(30 50)  # 預設只跑 K=30 和 K=50

# 檢查是否傳入參數來決定要跑哪些 K 值
if [ $# -gt 0 ]; then
    K_VALUES=("$@")
    echo -e "${BLUE}使用自訂 K 值: ${K_VALUES[@]}${NC}"
else
    echo -e "${BLUE}使用預設 K 值（快速測試）: ${K_VALUES[@]}${NC}"
    echo -e "${YELLOW}提示：如需測試更多 K 值，執行：${NC}"
    echo -e "${YELLOW}  bash scripts/experiments/run_s2_k_scan_colab.sh 30 50 80 100${NC}"
    echo -e "${YELLOW}  或測試高風險 K=200：${NC}"
    echo -e "${YELLOW}  bash scripts/experiments/run_s2_k_scan_colab.sh 200${NC}"
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}📊 實驗配置${NC}"
echo -e "${GREEN}================================${NC}"
echo "K 值列表: ${K_VALUES[@]}"
echo "專案目錄: $PROJECT_ROOT"
echo "開始時間: $(date)"
echo ""

# 記錄開始時間
START_TIME=$(date +%s)
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/s2_k_scan_colab_$(date +%Y%m%d_%H%M%S).log"

echo "日誌檔案: $LOG_FILE"
echo ""

# 記錄成功/失敗的實驗
SUCCESS_LIST=()
FAILED_LIST=()

# 依序執行每個 K 值
for K in "${K_VALUES[@]}"; do
    CONFIG_FILE="configs/experiments/S2_k_scan/s2_qr_K${K}_2d_re100.yml"
    
    echo -e "${YELLOW}================================${NC}"
    echo -e "${YELLOW}▶ 開始訓練: K=${K}${NC}"
    echo -e "${YELLOW}================================${NC}"
    echo "配置檔案: $CONFIG_FILE"
    
    # 檢查配置檔案是否存在
    if [ ! -f "$CONFIG_FILE" ]; then
        echo -e "${RED}❌ 錯誤：配置檔案不存在: $CONFIG_FILE${NC}"
        FAILED_LIST+=("K=$K (config not found)")
        continue
    fi
    
    echo "開始時間: $(date)"
    echo ""
    
    # 執行訓練
    EXP_START_TIME=$(date +%s)
    
    # 檢查是否需要修改 epochs（透過環境變數）
    if [ -n "$COLAB_EPOCHS" ]; then
        echo -e "${YELLOW}⚠ 自訂 epochs: $COLAB_EPOCHS (修改配置文件)${NC}"
        # 創建臨時配置（修改 epochs）
        TEMP_CONFIG="/tmp/temp_config_K${K}.yml"
        cp "$CONFIG_FILE" "$TEMP_CONFIG"
        # 使用 sed 修改 epochs
        sed -i "s/epochs: [0-9]*/epochs: $COLAB_EPOCHS/" "$TEMP_CONFIG"
        CONFIG_FILE="$TEMP_CONFIG"
        echo -e "${GREEN}✓ 使用臨時配置: $CONFIG_FILE${NC}"
    else
        # 讀取配置中的 epochs
        EPOCHS=$(grep "epochs:" "$CONFIG_FILE" | head -1 | awk '{print $2}')
        echo -e "${BLUE}訓練 epochs: ${EPOCHS:-10000} (來自配置文件)${NC}"
    fi
    
    echo -e "${BLUE}GPU 狀態：${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "無法檢測 GPU"
    echo ""
    
    if python scripts/train/train.py --cfg "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"; then
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${GREEN}✅ K=${K} 訓練完成${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m $((EXP_DURATION % 60))s"
        SUCCESS_LIST+=("K=$K")
        
        # 立即備份 checkpoint 到 Google Drive（如果掛載）
        if [ -d "/content/drive/MyDrive" ]; then
            DRIVE_BACKUP="/content/drive/MyDrive/pinns_checkpoints/S2_K${K}"
            mkdir -p "$DRIVE_BACKUP"
            cp -r "checkpoints/experiments/S2_qr_K${K}" "$DRIVE_BACKUP/" 2>/dev/null && \
                echo -e "${GREEN}✓ Checkpoint 已備份至 Google Drive${NC}" || \
                echo -e "${YELLOW}⚠ 無法備份至 Google Drive${NC}"
        fi
    else
        EXP_END_TIME=$(date +%s)
        EXP_DURATION=$((EXP_END_TIME - EXP_START_TIME))
        
        echo ""
        echo -e "${RED}❌ K=${K} 訓練失敗${NC}"
        echo "耗時: $((EXP_DURATION / 3600))h $((EXP_DURATION % 3600 / 60))m $((EXP_DURATION % 60))s"
        FAILED_LIST+=("K=$K")
        
        # Colab 上不詢問，直接繼續下一個
        echo -e "${YELLOW}自動繼續下一個實驗...${NC}"
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
echo "總耗時: $((TOTAL_DURATION / 3600))h $((TOTAL_DURATION % 3600 / 60))m $((TOTAL_DURATION % 60))s"
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
        CKPT_SIZE=$(du -sh "$CKPT_DIR" 2>/dev/null | cut -f1)
        echo "  ✓ $CKPT_DIR ($CKPT_SIZE)"
    else
        echo "  ✗ $CKPT_DIR (未找到)"
    fi
done

echo ""
echo "📊 下一步: 執行評估比較"
echo "在 Colab cell 中執行："
echo ""
echo "!python scripts/evaluate_unified.py \\"
echo "  --checkpoints \\"
for K in "${K_VALUES[@]}"; do
    echo "    checkpoints/experiments/S2_qr_K${K}/best_model.pth \\"
done
echo "  --labels ${K_VALUES[@]/#/K=}"
echo ""

# 自動產生評估腳本（Colab 友善）
EVAL_SCRIPT="$LOG_DIR/evaluate_s2_results.sh"
cat > "$EVAL_SCRIPT" << 'EOF'
#!/bin/bash
# 自動生成的評估腳本
python scripts/evaluate_unified.py \
  --checkpoints \
EOF

for K in "${K_VALUES[@]}"; do
    echo "    checkpoints/experiments/S2_qr_K${K}/best_model.pth \\" >> "$EVAL_SCRIPT"
done

echo "  --labels ${K_VALUES[@]/#/K=} \\" >> "$EVAL_SCRIPT"
echo "  --output results/S2_k_scan_comparison.png" >> "$EVAL_SCRIPT"

chmod +x "$EVAL_SCRIPT"
echo -e "${GREEN}✓ 評估腳本已生成: $EVAL_SCRIPT${NC}"
echo ""

# 顯示 GPU 使用記憶體
echo "🖥️  GPU 最終狀態："
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader || echo "無法檢測 GPU"
echo ""

echo "✨ 完成！"
