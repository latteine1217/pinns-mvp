#!/bin/bash
# 持續監控 Leith 模擬進度，完成後自動更新論文

set -e

PROJECT_ROOT="/Users/latteine/Documents/coding/pinns-mvp"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "👀 Watching Leith Simulations"
echo "=========================================="
echo ""
echo "Monitoring files:"
echo "  - data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5"
echo "  - data/lowfi/kolmogorov_rans/rans_re100_kf4_leith.h5"
echo "  - data/lowfi/kolmogorov_rans/rans_re500_kf4_leith.h5"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

check_interval=30  # Check every 30 seconds
last_status=""

while true; do
    # Check file existence
    re50_exists=false
    re100_exists=false
    re500_exists=false
    
    [ -f "data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5" ] && re50_exists=true
    [ -f "data/lowfi/kolmogorov_rans/rans_re100_kf4_leith.h5" ] && re100_exists=true
    [ -f "data/lowfi/kolmogorov_rans/rans_re500_kf4_leith.h5" ] && re500_exists=true
    
    # Count active processes
    active_processes=$(ps aux | grep "generate_kolmogorov_leith.py" | grep -v grep | wc -l | xargs)
    
    # Build status string
    current_time=$(date "+%H:%M:%S")
    status="[$current_time] "
    
    if [ "$re50_exists" = true ]; then
        size=$(ls -lh data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5 | awk '{print $5}')
        status+="Re50: ✅ ($size) | "
    else
        status+="Re50: ⏳ | "
    fi
    
    if [ "$re100_exists" = true ]; then
        size=$(ls -lh data/lowfi/kolmogorov_rans/rans_re100_kf4_leith.h5 | awk '{print $5}')
        status+="Re100: ✅ ($size) | "
    else
        status+="Re100: ⏳ | "
    fi
    
    if [ "$re500_exists" = true ]; then
        size=$(ls -lh data/lowfi/kolmogorov_rans/rans_re500_kf4_leith.h5 | awk '{print $5}')
        status+="Re500: ✅ ($size) | "
    else
        status+="Re500: ⏳ | "
    fi
    
    status+="Active: $active_processes"
    
    # Only print if status changed
    if [ "$status" != "$last_status" ]; then
        echo "$status"
        last_status="$status"
    fi
    
    # Check if all complete
    if [ "$re50_exists" = true ] && [ "$re100_exists" = true ] && [ "$re500_exists" = true ]; then
        echo ""
        echo "=========================================="
        echo "🎉 All Leith Simulations Complete!"
        echo "=========================================="
        echo ""
        
        # Wait for processes to fully finish
        if [ "$active_processes" -gt 0 ]; then
            echo "Waiting for processes to finish writing..."
            sleep 5
        fi
        
        echo "🚀 Starting automatic thesis update..."
        echo ""
        
        # Run the update script
        ./scripts/thesis/update_leith_results.sh
        
        echo ""
        echo "=========================================="
        echo "✅ Thesis update complete!"
        echo "=========================================="
        echo ""
        echo "📝 Next steps:"
        echo "1. Check the printed table values above"
        echo "2. Update thesis/main.tex Table 7.1"
        echo "3. Remove figure placeholder notes"
        echo "4. Compile: cd thesis && pdflatex main.tex"
        echo ""
        
        break
    fi
    
    sleep $check_interval
done
