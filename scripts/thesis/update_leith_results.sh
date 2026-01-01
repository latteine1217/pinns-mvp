#!/bin/bash
# 當 Leith 模擬完成後，自動更新論文中的數值與圖表

set -e

PROJECT_ROOT="/Users/latteine/Documents/coding/pinns-mvp"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "📊 Leith Results Update Pipeline"
echo "=========================================="

# 1. 驗證所有 Leith 文件存在
echo ""
echo "Step 1: Checking Leith simulation files..."
required_files=(
    "data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5"
    "data/lowfi/kolmogorov_rans/rans_re100_kf4_leith.h5"
    "data/lowfi/kolmogorov_rans/rans_re500_kf4_leith.h5"
)

all_exist=true
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        all_exist=false
    else
        echo "✅ Found: $file"
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "⚠️  Not all Leith files ready. Run this script again when simulations complete."
    exit 1
fi

# 2. 生成對比數據與圖表
echo ""
echo "Step 2: Generating Leith vs DNS comparison..."
python3 scripts/compare/compare_ke_vs_leith.py --re 50 100 500

# 3. 創建單獨的 Leith 誤差縮放圖
echo ""
echo "Step 3: Generating Leith error scaling figure..."
python3 << 'EOF'
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# Read Leith results
re_values = [50, 100, 500]
leith_errors = []
leith_rmse_u = []
leith_rmse_v = []

for re in re_values:
    result_dir = Path(f"results/rans_validation/re{re}")
    
    # Try to load from comparison results
    try:
        # Parse from generated comparison file
        metrics_file = result_dir / "metrics_ke_vs_leith.txt"
        if metrics_file.exists():
            with open(metrics_file) as f:
                lines = f.readlines()
                for line in lines:
                    if "Leith L2 Total" in line:
                        error = float(line.split(':')[1].strip().replace('%', ''))
                        leith_errors.append(error)
                    elif "Leith RMSE(u)" in line:
                        rmse = float(line.split(':')[1].strip())
                        leith_rmse_u.append(rmse)
                    elif "Leith RMSE(v)" in line:
                        rmse = float(line.split(':')[1].strip())
                        leith_rmse_v.append(rmse)
    except Exception as e:
        print(f"Warning: Could not parse Re={re}: {e}")
        leith_errors.append(np.nan)
        leith_rmse_u.append(np.nan)
        leith_rmse_v.append(np.nan)

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(re_values, leith_errors, 'o-', linewidth=2, markersize=8, 
        label='Leith Model', color='#2E86AB')
ax.axhline(100, color='red', linestyle='--', alpha=0.5, label='100% error')
ax.set_xlabel('Reynolds Number (Re)', fontsize=12)
ax.set_ylabel('Relative L2 Error (%)', fontsize=12)
ax.set_title('Leith Model Error Scaling with Reynolds Number', fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)

# Save
output_dir = Path("thesis/result_figures/kolmogorov")
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / "fig_leith_error_scaling.png", dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'fig_leith_error_scaling.png'}")

# Print summary for thesis table
print("\n" + "="*50)
print("📋 VALUES FOR THESIS TABLE 7.1:")
print("="*50)
for i, re in enumerate(re_values):
    print(f"Re{re}  & {re}.0  & {leith_errors[i]:.1f}  & {leith_rmse_u[i]:.3f} & {leith_rmse_v[i]:.3f} \\\\")
print("="*50)
EOF

# 4. 複製能譜圖（如果需要）
echo ""
echo "Step 4: Copying spectrum comparison figure..."
if [ -f "results/rans_validation/re50_comparison_ke_vs_leith.png" ]; then
    cp results/rans_validation/re50_comparison_ke_vs_leith.png \
       thesis/result_figures/kolmogorov/leith_dns_spectrum_re50.png
    echo "✅ Copied spectrum figure to thesis/result_figures/"
fi

echo ""
echo "=========================================="
echo "✅ Pipeline Complete!"
echo "=========================================="
echo ""
echo "📝 Next Steps:"
echo "1. Check the printed table values above"
echo "2. Update thesis/main.tex Table 7.1 (line ~701-703)"
echo "3. Remove placeholder notes from figures"
echo "4. Compile thesis to verify figures appear correctly"
echo ""
echo "Run:"
echo "  cd thesis && pdflatex main.tex"
echo ""
