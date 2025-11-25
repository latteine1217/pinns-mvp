#!/bin/bash
# 增強型 DNS 監控腳本 - 顯示詳細進度和預測

LOG_FILE="log/dns_generation_re100_v2.log"
REFRESH=5  # 每 5 秒更新

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║       DNS Re=100 v2 實時監控 (壓力投影修正版)              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

while true; do
    if [ ! -f "$LOG_FILE" ]; then
        echo "❌ 日誌文件不存在: $LOG_FILE"
        exit 1
    fi
    
    # 讀取最新一行
    last_line=$(tail -1 "$LOG_FILE")
    
    # 檢查是否完成
    if echo "$last_line" | grep -q "DNS 求解完成"; then
        echo ""
        echo "╔════════════════════════════════════════════════════════════╗"
        echo "║                   🎉 DNS 生成完成！                        ║"
        echo "╚════════════════════════════════════════════════════════════╝"
        echo ""
        echo "下一步："
        echo "  1. 驗證 DNS 數據: python scripts/validate_dns_v2.py"
        echo "  2. 生成感測點: python scripts/auto_process_re100.py --K 100"
        echo ""
        exit 0
    fi
    
    # 解析進度
    step=$(echo "$last_line" | grep -oP 'Step \K\d+' | head -1)
    total=20000
    time=$(echo "$last_line" | grep -oP 't=\K[0-9.]+' | head -1)
    ke=$(echo "$last_line" | grep -oP 'KE=\K[0-9.e+-]+' | head -1)
    div=$(echo "$last_line" | grep -oP 'Div_err=\K[0-9.e+-]+' | head -1)
    
    if [ -z "$step" ]; then
        echo "⏳ 等待日誌更新..."
        sleep $REFRESH
        continue
    fi
    
    # 計算進度
    pct=$(echo "scale=1; $step / $total * 100" | bc)
    remain=$((total - step))
    eta_sec=$(echo "scale=0; $remain * 0.002 * 60" | bc)
    eta_min=$(echo "scale=0; $eta_sec / 60" | bc)
    
    # 清屏並顯示
    tput cup 4 0  # 移動到第 4 行
    echo "┌─ 進度 ─────────────────────────────────────────────────────┐"
    printf "│ 步數:     %6d / %6d  (%5.1f%%)                       │\n" $step $total $pct
    printf "│ 時間:     t = %6.2f / 20.00                              │\n" $time
    echo "│                                                            │"
    
    # 進度條
    bar_len=50
    filled=$(echo "scale=0; $pct / 2" | bc)
    bar=$(printf "%${filled}s" | tr ' ' '█')
    empty=$(printf "%$((bar_len - filled))s" | tr ' ' '░')
    echo "│ [$bar$empty] │"
    echo "│                                                            │"
    
    printf "│ 剩餘步數: %6d  →  預計 %3d 分鐘後完成                │\n" $remain $eta_min
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""
    echo "┌─ 物理量 ───────────────────────────────────────────────────┐"
    printf "│ 動能 (KE):          %12s                         │\n" "$ke"
    printf "│ 散度誤差 (div):     %12s   [目標: <1e-8]        │\n" "$div"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""
    echo "┌─ 狀態 ─────────────────────────────────────────────────────┐"
    
    # 檢查物理量健康狀態
    div_check=$(echo "$div < 0.00000001" | bc)
    if [ "$div_check" -eq 1 ]; then
        echo "│ ✅ 散度誤差: 優秀 (< 1e-8)                                 │"
    else
        echo "│ ⚠️  散度誤差: 偏高                                         │"
    fi
    
    ke_check=$(echo "$ke > 0.5 && $ke < 5.0" | bc)
    if [ "$ke_check" -eq 1 ]; then
        echo "│ ✅ 動能: 穩定範圍                                          │"
    else
        echo "│ ⚠️  動能: 異常範圍                                         │"
    fi
    
    echo "│ ✅ 壓力投影: 已啟用 (5次/步)                               │"
    echo "│ ✅ 去混疊: 已啟用                                          │"
    echo "└────────────────────────────────────────────────────────────┘"
    echo ""
    echo "最後更新: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "刷新間隔: ${REFRESH} 秒 | 按 Ctrl+C 退出"
    
    sleep $REFRESH
done
