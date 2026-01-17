#!/bin/bash
# =============================================================================
# Batch Size Sweep 監控腳本
# =============================================================================
# 自動監控所有 batch size 實驗任務的狀態與進度
# =============================================================================

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Job IDs（從 run_batch_sweep.sh 提交的任務）
JOBS=(2751 2752 2753 2754)
BATCH_SIZES=("8k" "16k" "24k" "32k")

clear
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}📊 Batch Size Sweep 任務監控${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 顯示當前任務狀態
echo -e "${CYAN}🔄 SLURM 任務狀態:${NC}"
squeue -u $USER -o "%.10i %.9P %.20j %.8u %.2t %.10M %.6D %R" | head -20
echo ""

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}📈 訓練進度詳情:${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# 檢查每個任務的進度
for i in "${!JOBS[@]}"; do
    JOB_ID="${JOBS[$i]}"
    BATCH="${BATCH_SIZES[$i]}"
    LOG_FILE="logs/profile_simple_${JOB_ID}.log"
    
    echo -e "${YELLOW}[Job ${JOB_ID}] Batch Size: ${BATCH}${NC}"
    
    if [[ ! -f "$LOG_FILE" ]]; then
        echo -e "  ${RED}❌ Log 文件不存在（任務尚未開始）${NC}"
        echo ""
        continue
    fi
    
    # 檢查任務狀態
    JOB_STATUS=$(squeue -j $JOB_ID -h -o "%T" 2>/dev/null)
    
    if [[ -z "$JOB_STATUS" ]]; then
        # 任務已完成或失敗
        if grep -q "End Time:" "$LOG_FILE" 2>/dev/null; then
            echo -e "  ${GREEN}✅ 狀態: 已完成${NC}"
            
            # 提取總時間
            TOTAL_TIME=$(grep "Total time:" "$LOG_FILE" | tail -1 | grep -oP '\d+\.\d+s' || echo "N/A")
            echo -e "  ${GREEN}⏱️  總時間: ${TOTAL_TIME}${NC}"
            
            # 提取最後的 loss
            LAST_LOSS=$(grep "Epoch.*total_loss" "$LOG_FILE" | tail -1 | grep -oP 'total_loss: \K[0-9.]+' || echo "N/A")
            echo -e "  ${GREEN}📉 最終 Loss: ${LAST_LOSS}${NC}"
            
        elif grep -qi "out of memory\|cuda error" "$LOG_FILE" 2>/dev/null; then
            echo -e "  ${RED}❌ 狀態: OOM Error${NC}"
        else
            echo -e "  ${RED}❌ 狀態: 失敗（檢查 logs/profile_simple_${JOB_ID}.err）${NC}"
        fi
    elif [[ "$JOB_STATUS" == "RUNNING" ]]; then
        echo -e "  ${GREEN}🟢 狀態: 運行中${NC}"
        
        # 提取當前 epoch
        CURRENT_EPOCH=$(grep -oP "Epoch \K\d+" "$LOG_FILE" | tail -1 || echo "0")
        TOTAL_EPOCHS=10
        PROGRESS=$((CURRENT_EPOCH * 100 / TOTAL_EPOCHS))
        
        echo -e "  ${CYAN}📊 進度: Epoch ${CURRENT_EPOCH}/${TOTAL_EPOCHS} (${PROGRESS}%)${NC}"
        
        # 顯示最新的 loss
        LATEST_LOSS=$(grep "total_loss:" "$LOG_FILE" | tail -1 | grep -oP 'total_loss: \K[0-9.]+' || echo "N/A")
        echo -e "  ${CYAN}📉 當前 Loss: ${LATEST_LOSS}${NC}"
        
        # 估算剩餘時間
        if [[ $CURRENT_EPOCH -gt 0 ]]; then
            START_TIME=$(grep "Timer started at" "$LOG_FILE" | head -1 | awk -F'at ' '{print $2}')
            if [[ -n "$START_TIME" ]]; then
                START_SEC=$(date -d "$START_TIME" +%s 2>/dev/null || echo "0")
                NOW_SEC=$(date +%s)
                ELAPSED=$((NOW_SEC - START_SEC))
                
                if [[ $ELAPSED -gt 0 && $CURRENT_EPOCH -gt 0 ]]; then
                    AVG_EPOCH_TIME=$((ELAPSED / CURRENT_EPOCH))
                    REMAINING_EPOCHS=$((TOTAL_EPOCHS - CURRENT_EPOCH))
                    ETA=$((AVG_EPOCH_TIME * REMAINING_EPOCHS))
                    
                    ETA_MIN=$((ETA / 60))
                    ETA_SEC=$((ETA % 60))
                    
                    echo -e "  ${CYAN}⏱️  預估剩餘: ${ETA_MIN}m ${ETA_SEC}s${NC}"
                fi
            fi
        fi
        
    elif [[ "$JOB_STATUS" == "PENDING" ]]; then
        echo -e "  ${YELLOW}⏳ 狀態: 排隊中${NC}"
        REASON=$(squeue -j $JOB_ID -h -o "%R" 2>/dev/null)
        echo -e "  ${YELLOW}📌 原因: ${REASON}${NC}"
    else
        echo -e "  ${YELLOW}❓ 狀態: ${JOB_STATUS}${NC}"
    fi
    
    echo ""
done

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}💡 監控命令:${NC}"
echo -e "   持續監控: ${GREEN}watch -n 5 'bash monitor_batch_sweep.sh'${NC}"
echo -e "   查看日誌: ${GREEN}tail -f logs/profile_simple_2751.log${NC}"
echo -e "   分析結果: ${GREEN}python3 scripts/analyze_batch_sweep.py${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${YELLOW}⏰ 更新時間: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
