#!/bin/bash
# ====================================================================
# 數據同步腳本 - 傳輸 data 目錄到伺服器
# ====================================================================
# 用途：將本地 data 目錄（~12 GB）傳輸到遠端伺服器
# 使用方式：./scripts/tools/sync_data_to_server.sh [選項]
# ====================================================================

set -e  # 遇到錯誤即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 配置參數
SERVER_USER="junyi"
SERVER_HOST="140.114.120.128"
SERVER_PATH="/home/junyi/pinns-sparse-flow"  # 請根據實際路徑調整
LOCAL_DATA_DIR="data"

# 顯示使用說明
function show_usage() {
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}數據同步腳本 - 傳輸 data 目錄到伺服器${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    echo ""
    echo "使用方式："
    echo "  $0 [選項]"
    echo ""
    echo "選項："
    echo "  --dry-run        : 模擬執行，不實際傳輸（查看將傳輸的文件）"
    echo "  --essential-only : 只傳輸必要文件（~1 GB，跳過大型 archived_h5）"
    echo "  --full          : 傳輸完整 data 目錄（~12 GB，預設）"
    echo "  --resume        : 斷點續傳模式"
    echo "  --help          : 顯示此說明"
    echo ""
    echo "範例："
    echo "  $0 --dry-run              # 查看將傳輸的文件"
    echo "  $0 --essential-only       # 只傳輸必要文件（建議）"
    echo "  $0 --full                 # 傳輸完整數據（較慢）"
    echo ""
}

# 解析參數
DRY_RUN=""
ESSENTIAL_ONLY=false
RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --essential-only)
            ESSENTIAL_ONLY=true
            shift
            ;;
        --full)
            ESSENTIAL_ONLY=false
            shift
            ;;
        --resume)
            RESUME="--partial"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}錯誤：未知選項 $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# 檢查本地 data 目錄是否存在
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo -e "${RED}錯誤：本地 data 目錄不存在${NC}"
    exit 1
fi

# 顯示配置資訊
echo -e "${GREEN}=====================================================================${NC}"
echo -e "${GREEN}數據同步配置${NC}"
echo -e "${GREEN}=====================================================================${NC}"
echo -e "來源目錄: ${YELLOW}$LOCAL_DATA_DIR${NC}"
echo -e "目標伺服器: ${YELLOW}$SERVER_USER@$SERVER_HOST:$SERVER_PATH${NC}"
echo -e "傳輸模式: ${YELLOW}$([ "$ESSENTIAL_ONLY" = true ] && echo "必要文件" || echo "完整數據")${NC}"
echo -e "模擬執行: ${YELLOW}$([ -n "$DRY_RUN" ] && echo "是" || echo "否")${NC}"
echo ""

# 統計數據大小
if [ "$ESSENTIAL_ONLY" = true ]; then
    echo -e "${BLUE}正在計算必要文件大小...${NC}"
    ESSENTIAL_SIZE=$(du -sh data/kolmogorov_dns data/jhtdb data/sensors data/lowfi_npy 2>/dev/null | awk '{sum+=$1} END {print sum}')
    echo -e "預計傳輸大小: ${YELLOW}~1 GB${NC}"
else
    echo -e "${BLUE}正在計算完整數據大小...${NC}"
    TOTAL_SIZE=$(du -sh data 2>/dev/null | awk '{print $1}')
    echo -e "預計傳輸大小: ${YELLOW}$TOTAL_SIZE${NC}"
fi

echo ""

# 確認執行
if [ -z "$DRY_RUN" ]; then
    read -p "$(echo -e ${YELLOW}是否繼續？[y/N]: ${NC})" -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${RED}取消同步${NC}"
        exit 0
    fi
fi

# 測試 SSH 連線
echo -e "${BLUE}測試 SSH 連線...${NC}"
if ! ssh -o ConnectTimeout=5 "$SERVER_USER@$SERVER_HOST" "echo '連線成功'" > /dev/null 2>&1; then
    echo -e "${RED}錯誤：無法連線到伺服器${NC}"
    echo -e "${YELLOW}請確認：${NC}"
    echo "  1. 伺服器地址正確"
    echo "  2. SSH 金鑰已設定"
    echo "  3. 網路連線正常"
    exit 1
fi
echo -e "${GREEN}✓ SSH 連線正常${NC}"
echo ""

# 在伺服器上創建目標目錄
echo -e "${BLUE}在伺服器上創建 data 目錄...${NC}"
ssh "$SERVER_USER@$SERVER_HOST" "mkdir -p $SERVER_PATH/data" || {
    echo -e "${RED}錯誤：無法在伺服器上創建目錄${NC}"
    exit 1
}
echo -e "${GREEN}✓ 目錄創建成功${NC}"
echo ""

# 執行 rsync 傳輸
echo -e "${BLUE}=====================================================================${NC}"
echo -e "${BLUE}開始數據傳輸...${NC}"
echo -e "${BLUE}=====================================================================${NC}"

RSYNC_OPTIONS="-avz --progress $RESUME $DRY_RUN"

if [ "$ESSENTIAL_ONLY" = true ]; then
    # 只傳輸必要文件
    echo -e "${YELLOW}傳輸必要文件（跳過 archived_h5 和 kolmogorov_dns_npy）${NC}"
    echo ""
    
    # 傳輸關鍵目錄
    for dir in kolmogorov_dns jhtdb sensors lowfi_npy; do
        if [ -d "data/$dir" ]; then
            echo -e "${BLUE}傳輸 $dir...${NC}"
            rsync $RSYNC_OPTIONS \
                "data/$dir/" \
                "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/data/$dir/"
        fi
    done
else
    # 傳輸完整 data 目錄
    echo -e "${YELLOW}傳輸完整 data 目錄（包含所有文件）${NC}"
    echo ""
    
    rsync $RSYNC_OPTIONS \
        --exclude='.DS_Store' \
        "data/" \
        "$SERVER_USER@$SERVER_HOST:$SERVER_PATH/data/"
fi

# 驗證傳輸結果
if [ -z "$DRY_RUN" ]; then
    echo ""
    echo -e "${BLUE}=====================================================================${NC}"
    echo -e "${BLUE}驗證傳輸結果...${NC}"
    echo -e "${BLUE}=====================================================================${NC}"
    
    # 檢查伺服器上的目錄
    echo -e "${BLUE}伺服器 data 目錄結構：${NC}"
    ssh "$SERVER_USER@$SERVER_HOST" "ls -lh $SERVER_PATH/data/"
    
    echo ""
    echo -e "${GREEN}=====================================================================${NC}"
    echo -e "${GREEN}✅ 數據同步完成！${NC}"
    echo -e "${GREEN}=====================================================================${NC}"
    
    if [ "$ESSENTIAL_ONLY" = true ]; then
        echo -e "${YELLOW}提示：已跳過非必要文件（archived_h5, kolmogorov_dns_npy）${NC}"
        echo -e "${YELLOW}      如需完整數據，請執行：$0 --full${NC}"
    fi
else
    echo ""
    echo -e "${YELLOW}=====================================================================${NC}"
    echo -e "${YELLOW}這是模擬執行，沒有實際傳輸文件${NC}"
    echo -e "${YELLOW}=====================================================================${NC}"
    echo -e "若要執行實際傳輸，請移除 --dry-run 選項"
fi

echo ""
echo -e "${BLUE}後續步驟：${NC}"
echo "  1. 在伺服器上驗證數據：python scripts/tools/verify_data_integrity.py"
echo "  2. 執行訓練測試：python scripts/train/train.py --cfg configs/quick_test.yml --epochs 2"
echo "  3. 執行 DDP 訓練：torchrun --nproc_per_node=2 scripts/train/train.py --cfg configs/main.yml"
echo ""
