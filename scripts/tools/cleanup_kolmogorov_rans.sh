#!/bin/bash
# ============================================================================
# Kolmogorov 數據清理腳本
# 刪除所有 k-ε RANS 結果，僅保留 LES 和 DNS 數據
# ============================================================================

set -e  # 遇到錯誤立即退出

# 顏色定義
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 工作目錄
RANS_DIR="data/lowfi/kolmogorov_rans"

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  🗑️  Kolmogorov 數據清理工具${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# 檢查目錄是否存在
if [ ! -d "$RANS_DIR" ]; then
    echo -e "${RED}❌ 錯誤：目錄不存在 $RANS_DIR${NC}"
    exit 1
fi

cd "$RANS_DIR"

echo -e "${YELLOW}📊 當前目錄大小：${NC}"
du -sh .
echo ""

echo -e "${YELLOW}📋 將要刪除的文件：${NC}"
echo ""

# 列出將要刪除的文件
echo -e "${RED}[k-ε RANS 原始數據]${NC}"
ls -lh rans_re*_kf4.h5 2>/dev/null || echo "  (無文件)"
ls -lh rans_re*_kf4_corrected.h5 2>/dev/null || echo "  (無文件)"
echo ""

echo -e "${RED}[k-ε RANS Backup 目錄]${NC}"
ls -d backup_20251217_* 2>/dev/null || echo "  (無目錄)"
ls -d backup_les_2025* 2>/dev/null || echo "  (無目錄)"
echo ""

echo -e "${RED}[k-ε RANS Sensors]${NC}"
ls -lh sensors_K100_rans*.npz 2>/dev/null || echo "  (無文件)"
echo ""

echo -e "${RED}[LES 舊版本]${NC}"
ls -lh *_OLD_UNIFORM.h5 2>/dev/null || echo "  (無文件)"
ls -lh *_optimized.h5 2>/dev/null || echo "  (無文件)"
ls -lh *_test_*.h5 test_les_*.h5 2>/dev/null || echo "  (無文件)"
ls -d backup_uniform_params_* 2>/dev/null || echo "  (無目錄)"
echo ""

echo -e "${GREEN}✅ 將要保留的文件：${NC}"
ls -lh rans_re*_kf4_les.h5 2>/dev/null | grep -v "OLD\|optimized\|test" || echo "  ⚠️  未找到正式 LES 文件！"
echo ""

# 確認提示
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
read -p "確定要刪除以上文件嗎？(yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo -e "${YELLOW}❌ 操作已取消${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}🚀 開始清理...${NC}"
echo ""

# 計數器
deleted_count=0
deleted_size=0

# ============================================================================
# 刪除 k-ε RANS 原始數據
# ============================================================================
echo -e "${YELLOW}[1/6] 刪除 k-ε RANS 原始數據...${NC}"
for file in rans_re*_kf4.h5 rans_re*_kf4_corrected.h5; do
    if [ -f "$file" ]; then
        size=$(du -k "$file" | cut -f1)
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 刪除 k-ε RANS Backup
# ============================================================================
echo -e "${YELLOW}[2/6] 刪除 k-ε RANS Backup 目錄...${NC}"
for dir in backup_20251217_* backup_les_2025*; do
    if [ -d "$dir" ]; then
        size=$(du -sk "$dir" | cut -f1)
        rm -rf "$dir"
        echo "  ✓ 已刪除: $dir/"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 刪除 k-ε RANS Sensors
# ============================================================================
echo -e "${YELLOW}[3/6] 刪除 k-ε RANS Sensors...${NC}"
for file in sensors_K100_rans*.npz; do
    if [ -f "$file" ]; then
        size=$(du -k "$file" | cut -f1)
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 刪除 LES OLD_UNIFORM 版本
# ============================================================================
echo -e "${YELLOW}[4/6] 刪除 LES OLD_UNIFORM 版本...${NC}"
for file in *_OLD_UNIFORM.h5; do
    if [ -f "$file" ]; then
        size=$(du -k "$file" | cut -f1)
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 刪除 LES optimized 版本
# ============================================================================
echo -e "${YELLOW}[5/6] 刪除 LES optimized 版本...${NC}"
for file in *_optimized.h5; do
    if [ -f "$file" ]; then
        size=$(du -k "$file" | cut -f1)
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 刪除 LES 測試文件與備份
# ============================================================================
echo -e "${YELLOW}[6/6] 刪除 LES 測試文件與舊備份...${NC}"
for file in *_test_*.h5 test_les_*.h5; do
    if [ -f "$file" ]; then
        size=$(du -k "$file" | cut -f1)
        rm -f "$file"
        echo "  ✓ 已刪除: $file"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done

for dir in backup_uniform_params_*; do
    if [ -d "$dir" ]; then
        size=$(du -sk "$dir" | cut -f1)
        rm -rf "$dir"
        echo "  ✓ 已刪除: $dir/"
        ((deleted_count++))
        deleted_size=$((deleted_size + size))
    fi
done
echo ""

# ============================================================================
# 清理完成，顯示結果
# ============================================================================
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  ✅ 清理完成！${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${BLUE}📊 清理統計：${NC}"
echo "  已刪除文件/目錄數: $deleted_count"
echo "  釋放空間: $(echo "scale=2; $deleted_size/1024" | bc) MB"
echo ""

echo -e "${BLUE}📂 清理後目錄大小：${NC}"
du -sh .
echo ""

echo -e "${GREEN}✅ 保留的 LES 文件：${NC}"
ls -lh rans_re*_kf4_les.h5 2>/dev/null | grep -v "OLD\|optimized\|test" || echo -e "${RED}  ⚠️  警告：未找到正式 LES 文件！${NC}"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}📝 後續建議：${NC}"
echo ""
echo "1. 驗證 LES 數據完整性："
echo "   python scripts/validation/verify_les_data.py"
echo ""
echo "2. 檢查配置文件路徑："
echo "   grep 'data_path' configs/kolmogorov_re50_kf4_K100.yml"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
