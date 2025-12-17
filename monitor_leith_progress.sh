#!/bin/bash
# Leith RANS 生成進度監控

LOG_FILE="rans_leith_generation.log"
OUTPUT_DIR="data/lowfi/kolmogorov_rans"

echo "========================================================================"
echo "Leith RANS 生成進度監控"
echo "========================================================================"
echo ""

# 檢查進程
RUNNING=$(ps aux | grep "generate_kolmogorov_leith" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "✅ 進程運行中"
    ps aux | grep "generate_kolmogorov_leith" | grep -v grep | awk '{print "   PID: "$2", CPU: "$3"%, MEM: "$4"%"}'
else
    echo "⚠️  進程已結束或未啟動"
fi

echo ""
echo "========================================================================"
echo "最新日誌（最後 5 行）"
echo "========================================================================"
tail -5 "$LOG_FILE"

echo ""
echo "========================================================================"
echo "已生成檔案"
echo "========================================================================"
ls -lht "${OUTPUT_DIR}/"*_leith.h5 2>/dev/null | head -5 || echo "  (尚未生成)"

echo ""
echo "========================================================================"
echo "預估時間"
echo "========================================================================"
COMPLETED=$(grep "✅.*completed!" "$LOG_FILE" 2>/dev/null | wc -l | tr -d ' ')
echo "  已完成: ${COMPLETED}/3"
echo "  Re=50:  ~10 分鐘"
echo "  Re=100: ~15 分鐘"
echo "  Re=500: ~45 分鐘"
echo "  總計:   ~70 分鐘"
echo ""
