#!/bin/bash
# Task-10 訓練監控腳本
# 實時追蹤 1000 epochs 訓練進度

LOG_FILE="/Users/latteine/Documents/coding/pinns-mvp/log/train_task10_1k_20251010_060501.log"

echo "============================================"
echo "  Task-10 訓練監控"
echo "  配置: vs_pinn_channel_flow_1k.yml"
echo "  目標: 1000 epochs"
echo "============================================"
echo ""

# 檢查訓練進程
if pgrep -f "train.py.*vs_pinn_channel_flow_1k" > /dev/null; then
    echo "✅ 訓練進程運行中 (PID: $(pgrep -f 'train.py.*vs_pinn_channel_flow_1k'))"
else
    echo "❌ 訓練進程未運行"
    exit 1
fi

echo ""
echo "=== 最新進度 ==="
tail -3 "$LOG_FILE" | grep "Epoch"

echo ""
echo "=== 關鍵指標趨勢 ==="
echo "📊 總損失 (Total Loss):"
grep "Total:" "$LOG_FILE" | tail -10 | awk '{print "  Epoch", $4, ":", $7}'

echo ""
echo "📊 PDE 殘差 (Residual):"
grep "Residual:" "$LOG_FILE" | tail -10 | awk '{print "  Epoch", $3, ":", $6}'

echo ""
echo "📊 邊界條件 (BC):"
grep "BC:" "$LOG_FILE" | tail -10 | awk '{print "  Epoch", $3, ":", $8}'

echo ""
echo "📊 資料擬合 (Data):"
grep "Data:" "$LOG_FILE" | tail -10 | awk '{print "  Epoch", $3, ":", $10}'

echo ""
echo "=== 新增 Loss 項狀態 ==="
echo "📌 流量守恆 (bulk_velocity):"
grep "bulk_velocity:" "$LOG_FILE" | tail -5 | awk '{print "  ", $3}'

echo ""
echo "📌 中心線對稱 (centerline_dudy):"
grep "centerline_dudy:" "$LOG_FILE" | tail -5 | awk '{print "  ", $3}'

echo ""
echo "📌 中心線 v (centerline_v):"
grep "centerline_v:" "$LOG_FILE" | tail -5 | awk '{print "  ", $3}'

echo ""
echo "📌 壓力參考 (pressure_reference):"
grep "pressure_reference:" "$LOG_FILE" | tail -5 | awk '{print "  ", $3}'

echo ""
echo "=== 異常檢測 ==="
if grep -q "nan\|inf\|NaN\|Inf" "$LOG_FILE"; then
    echo "⚠️  檢測到 NaN/Inf（需要 Sub-Agent 分析）"
    grep -n "nan\|inf\|NaN\|Inf" "$LOG_FILE" | tail -5
else
    echo "✅ 無 NaN/Inf"
fi

echo ""
echo "=== 訓練速度 ==="
grep "Time:" "$LOG_FILE" | tail -5 | awk '{print "  Epoch", $3, "耗時:", $13}'

echo ""
echo "============================================"
echo "實時日誌: tail -f $LOG_FILE"
echo "============================================"
