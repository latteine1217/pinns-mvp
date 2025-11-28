#!/bin/bash
# Kolmogorov DNS 監控腳本 (Re=100 & Re=500)
# 使用方式: ./scripts/monitor_dns_re100_re500.sh

echo "========================================================================"
echo "Kolmogorov Flow DNS 模擬監控 (Re=100 & Re=500, k_f=4, T=100s)"
echo "========================================================================"
echo ""

# Re=100 監控
echo "🔵 Re=100 模擬狀態:"
echo "------------------------------------------------------------------------"
if [ -f "log/dns_re100_kf4_T100.log" ]; then
    tail -5 log/dns_re100_kf4_T100.log | grep -E "Step|完成|模擬"
    echo ""

    # 檢查是否完成
    if grep -q "模擬完成" log/dns_re100_kf4_T100.log; then
        echo "✅ Re=100 模擬已完成！"
    else
        echo "⏳ Re=100 模擬進行中..."
    fi
else
    echo "❌ 日誌文件不存在"
fi
echo ""

# Re=500 監控
echo "🟣 Re=500 模擬狀態:"
echo "------------------------------------------------------------------------"
if [ -f "log/dns_re500_kf4_T100.log" ]; then
    tail -5 log/dns_re500_kf4_T100.log | grep -E "Step|完成|模擬"
    echo ""

    # 檢查是否完成
    if grep -q "模擬完成" log/dns_re500_kf4_T100.log; then
        echo "✅ Re=500 模擬已完成！"
    else
        echo "⏳ Re=500 模擬進行中..."
    fi
else
    echo "❌ 日誌文件不存在"
fi
echo ""

# 檢查輸出文件
echo "📁 輸出文件檢查:"
echo "------------------------------------------------------------------------"
if [ -f "data/kolmogorov_dns/kolmogorov_re100_kf4_T100.h5" ]; then
    size_re100=$(ls -lh data/kolmogorov_dns/kolmogorov_re100_kf4_T100.h5 | awk '{print $5}')
    echo "✅ Re=100 HDF5: $size_re100"
else
    echo "❌ Re=100 HDF5 不存在"
fi

if [ -f "data/kolmogorov_dns/kolmogorov_re500_kf4_T100.h5" ]; then
    size_re500=$(ls -lh data/kolmogorov_dns/kolmogorov_re500_kf4_T100.h5 | awk '{print $5}')
    echo "✅ Re=500 HDF5: $size_re500"
else
    echo "❌ Re=500 HDF5 不存在"
fi

echo ""
echo "========================================================================"
echo "提示："
echo "  - 持續監控: watch -n 10 './scripts/monitor_dns_re100_re500.sh'"
echo "  - 完整日誌: tail -f log/dns_re100_kf4_T100.log"
echo "  - 完整日誌: tail -f log/dns_re500_kf4_T100.log"
echo "========================================================================"
