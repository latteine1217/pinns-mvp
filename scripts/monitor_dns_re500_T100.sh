#!/bin/bash
# DNS Re=500 (T=100s, 擾動@t=10s) 監控腳本

PID=63849
LOG="log/dns_re500_kf4_T100_generation.log"

echo "=========================================="
echo "  DNS Re=500 (T=100s) 監控"
echo "=========================================="
echo ""

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ 進程運行中 (PID: $PID)"
    ps -p $PID -o pid,etime,%cpu,%mem | tail -1
    echo ""
    echo "📊 最新進度:"
    tail -1 "$LOG" | grep -E "Step"
    echo ""
    echo "🔍 配置摘要:"
    echo "  - 雷諾數: Re = 500"
    echo "  - 動力黏度: ν = 0.003937"
    echo "  - 網格: N = 512 × 512"
    echo "  - 時間步: dt = 0.001"
    echo "  - 總時長: T = 100.0 秒"
    echo "  - 擾動時刻: t = 10.0 秒"
    echo "  - 計算後端: torch-mps"
else
    echo "✅ 進程已完成 (或已停止)"
    echo ""
    echo "📁 輸出檔案:"
    ls -lh data/kolmogorov_dns_re500_kf4_T100_pert10.h5 2>/dev/null || echo "  檔案尚未生成"
fi

echo ""
echo "🔄 持續監控指令:"
echo "  watch -n 10 ./scripts/monitor_dns_re500_T100.sh"
echo "  python scripts/monitor_dns_re500_T100.py"
echo "=========================================="
