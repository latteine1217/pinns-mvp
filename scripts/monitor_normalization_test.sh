#!/bin/bash
# =============================================================================
# 標準化訓練監控腳本
# =============================================================================
# 用途：監控 normalization_baseline_test 實驗的訓練進度
# 
# 用法：
#   ./scripts/monitor_normalization_test.sh          # 單次查看
#   ./scripts/monitor_normalization_test.sh --watch  # 持續監控
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

EXPERIMENT_NAME="normalization_baseline_test"
LOG_DIR="log/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="checkpoints/${EXPERIMENT_NAME}"

# 顏色設定
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 函數：顯示標題
show_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}📊 標準化訓練監控 ${NC}"
    echo -e "${BLUE}實驗: ${EXPERIMENT_NAME}${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

# 函數：檢查訓練狀態
check_status() {
    show_header
    
    # 檢查日誌目錄
    if [ -d "$LOG_DIR" ]; then
        echo -e "${GREEN}✅ 日誌目錄存在: ${LOG_DIR}${NC}"
        
        # 查找最新的訓練日誌
        LATEST_LOG=$(find "$LOG_DIR" -name "training*.log" -type f | head -1)
        
        if [ -n "$LATEST_LOG" ]; then
            echo -e "${GREEN}✅ 訓練日誌: ${LATEST_LOG}${NC}"
            echo ""
            
            # 顯示最後 30 行
            echo -e "${BLUE}📝 最新訓練日誌 (最後 30 行):${NC}"
            echo "----------------------------------------"
            tail -30 "$LATEST_LOG"
            echo "----------------------------------------"
            echo ""
        else
            echo -e "${YELLOW}⚠️  未找到訓練日誌${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️  日誌目錄不存在: ${LOG_DIR}${NC}"
        echo -e "${YELLOW}💡 訓練可能尚未開始${NC}"
    fi
    
    # 檢查檢查點
    if [ -d "$CHECKPOINT_DIR" ]; then
        echo -e "${GREEN}✅ 檢查點目錄存在: ${CHECKPOINT_DIR}${NC}"
        
        CHECKPOINT_COUNT=$(find "$CHECKPOINT_DIR" -name "*.pth" -type f | wc -l | tr -d ' ')
        echo -e "${GREEN}📦 檢查點數量: ${CHECKPOINT_COUNT}${NC}"
        
        if [ "$CHECKPOINT_COUNT" -gt 0 ]; then
            echo ""
            echo -e "${BLUE}📦 最新檢查點:${NC}"
            find "$CHECKPOINT_DIR" -name "*.pth" -type f -exec ls -lh {} \; | tail -5
        fi
    else
        echo -e "${YELLOW}⚠️  檢查點目錄不存在: ${CHECKPOINT_DIR}${NC}"
    fi
    
    echo ""
    
    # 使用通用監控工具（如果可用）
    if [ -f "scripts/monitor_training.py" ]; then
        echo -e "${BLUE}🔍 詳細訓練狀態:${NC}"
        echo "----------------------------------------"
        python scripts/monitor_training.py --config "$EXPERIMENT_NAME" 2>/dev/null || echo -e "${YELLOW}⚠️  無法獲取詳細狀態（訓練可能未啟動）${NC}"
    fi
}

# 主邏輯
if [ "$1" == "--watch" ]; then
    echo -e "${GREEN}🔄 持續監控模式啟動（每 5 秒刷新）${NC}"
    echo -e "${GREEN}按 Ctrl+C 停止監控${NC}"
    echo ""
    
    while true; do
        clear
        check_status
        sleep 5
    done
else
    check_status
fi
