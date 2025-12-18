#!/bin/bash
#==============================================================================
# Channel Flow 數據清理腳本
# Date: 2025-12-18
# Purpose: 移除冗餘的 2D slice 和舊版 sensor 文件
#==============================================================================

set -e  # 遇到錯誤立即停止

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 基礎路徑
BASE_DIR="/Users/latteine/Documents/coding/pinns-mvp"
CHANNEL_DIR="$BASE_DIR/data/jhtdb/channel_flow_re1000"
BACKUP_DIR="$BASE_DIR/data/jhtdb/channel_flow_backup_$(date +%Y%m%d_%H%M%S)"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Channel Flow 數據清理腳本${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 檢查目錄是否存在
if [ ! -d "$CHANNEL_DIR" ]; then
    echo -e "${RED}❌ 錯誤: $CHANNEL_DIR 不存在${NC}"
    exit 1
fi

cd "$CHANNEL_DIR"

#==============================================================================
# Step 1: 顯示清理前狀態
#==============================================================================
echo -e "${YELLOW}[1/5] 清理前狀態檢查${NC}"
echo "---"
BEFORE_SIZE=$(du -sh . | awk '{print $1}')
BEFORE_FILES=$(find . -type f | wc -l | tr -d ' ')
echo "總大小: $BEFORE_SIZE"
echo "文件數: $BEFORE_FILES"
echo ""

#==============================================================================
# Step 2: 創建備份（僅備份即將刪除的文件）
#==============================================================================
echo -e "${YELLOW}[2/5] 創建備份${NC}"
echo "---"
mkdir -p "$BACKUP_DIR"

# 需要備份的文件列表
FILES_TO_BACKUP=(
    "eval_2d_slice.npz"
    "eval_2d_slice_3d.npz"
    "slab_xz_center.npz"
    "slab_xz_nearwall.npz"
    "cutout_128x64.npz"
    "sensors_K100_qr_pivot_2d.npz"
    "sensors_K100_qr_pivot_2d_v2.npz"
    "sensors_K100_qr_pivot_2d_v3.npz"
    "sensors_K100_qr_pivot_2d_v4.npz"
    "sensors_K100_qr_pivot_2d_v5_gradu_eig.npz"
    "sensors_K100_qr_pivot_3d_v5_gradu_eig.npz"
    "sensors_K500_qr_pivot.npz"
    "sensors_K500_hybrid_qr.npz"
    "sensors_K500_uniform.npz"
    "sensors_K500_qr_pivot_fixed_2d.npz"
    "sensors_K500_qr_pivot_periodic.npz"
    "sensor_strategies_comparison_K500.png"
    "sensors_K100_rans_phase_a.npz"
    "sensors_K100_random_rans_grid.npz"
    "sensors_K100_qr_pivot_periodic.npz"
    "sensors_K100_qr_pivot_standard.npz"
    "sensors_K100_random_stratified.npz"
)

BACKUP_COUNT=0
for file in "${FILES_TO_BACKUP[@]}"; do
    if [ -e "$file" ]; then
        cp -p "$file" "$BACKUP_DIR/"
        BACKUP_COUNT=$((BACKUP_COUNT + 1))
    fi
done

echo "✓ 已備份 $BACKUP_COUNT 個文件到: $BACKUP_DIR"
echo ""

#==============================================================================
# Step 3: 執行清理
#==============================================================================
echo -e "${YELLOW}[3/5] 執行清理${NC}"
echo "---"

DELETED_COUNT=0

# 3.1 刪除早期評估數據
echo "► 刪除早期評估數據..."
for file in eval_2d_slice.npz eval_2d_slice_3d.npz; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

# 3.2 刪除 2D Slice 文件
echo "► 刪除 2D Slice 文件..."
for file in slab_xz_center.npz slab_xz_nearwall.npz cutout_128x64.npz; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

# 3.3 刪除舊版 sensor 迭代 (v1-v5)
echo "► 刪除舊版 K=100 sensor 迭代..."
for file in sensors_K100_qr_pivot_2d*.npz sensors_K100_qr_pivot_3d_v5_gradu_eig.npz; do
    if [ -e "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

# 3.4 刪除 K=500 sensors
echo "► 刪除 K=500 sensors..."
for file in sensors_K500_*.npz; do
    if [ -e "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

# 3.5 刪除舊對比圖
echo "► 刪除舊對比圖..."
if [ -f "sensor_strategies_comparison_K500.png" ]; then
    rm -f sensor_strategies_comparison_K500.png
    echo "  ✓ 已刪除: sensor_strategies_comparison_K500.png"
    DELETED_COUNT=$((DELETED_COUNT + 1))
fi

# 3.6 刪除符號連結和其他冗餘 sensors
echo "► 刪除符號連結和其他冗餘 sensors..."
for file in sensors_K100_rans_phase_a.npz sensors_K100_random_rans_grid.npz \
            sensors_K100_qr_pivot_periodic.npz sensors_K100_qr_pivot_standard.npz \
            sensors_K100_random_stratified.npz; do
    if [ -e "$file" ]; then
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        DELETED_COUNT=$((DELETED_COUNT + 1))
    fi
done

echo ""
echo "✓ 總計刪除 $DELETED_COUNT 個文件"
echo ""

#==============================================================================
# Step 4: 驗證保留文件
#==============================================================================
echo -e "${YELLOW}[4/5] 驗證保留文件${NC}"
echo "---"

# 關鍵文件檢查
CRITICAL_FILES=(
    "cutout_128x64x128.npz"
    "cutout_64x32x64.npz"
    "sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz"
    "sensors_K100_rans_phase_a_with_data.npz"
    "sensors_K100_random_rans_grid_with_data.npz"
)

ALL_PRESERVED=true
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✓ $file ($SIZE)"
    else
        echo -e "  ${RED}✗ 缺失: $file${NC}"
        ALL_PRESERVED=false
    fi
done

# 檢查目錄
for dir in raw reports; do
    if [ -d "$dir" ]; then
        echo "  ✓ $dir/ (目錄完整)"
    else
        echo -e "  ${RED}✗ 缺失: $dir/${NC}"
        ALL_PRESERVED=false
    fi
done

echo ""

#==============================================================================
# Step 5: 顯示清理後狀態
#==============================================================================
echo -e "${YELLOW}[5/5] 清理後狀態${NC}"
echo "---"
AFTER_SIZE=$(du -sh . | awk '{print $1}')
AFTER_FILES=$(find . -type f | wc -l | tr -d ' ')
echo "總大小: $AFTER_SIZE (原: $BEFORE_SIZE)"
echo "文件數: $AFTER_FILES (原: $BEFORE_FILES)"
echo ""

#==============================================================================
# 總結
#==============================================================================
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}清理完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "📊 清理統計:"
echo "  - 刪除文件: $DELETED_COUNT"
echo "  - 備份位置: $BACKUP_DIR"
echo "  - 保留關鍵文件: ${#CRITICAL_FILES[@]}"
echo ""

if [ "$ALL_PRESERVED" = true ]; then
    echo -e "${GREEN}✅ 所有關鍵文件已驗證完整${NC}"
else
    echo -e "${RED}⚠️  部分關鍵文件缺失，請檢查！${NC}"
fi

echo ""
echo "📂 當前保留文件:"
ls -lh | grep -E "cutout_|sensors_K100" | awk '{print "  - "$9, "("$5")"}'
echo ""

echo "💾 如需恢復，請使用:"
echo "  cp $BACKUP_DIR/* $CHANNEL_DIR/"
echo ""
