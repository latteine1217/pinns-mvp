#!/usr/bin/env python3
"""
生成 Leith 模型誤差縮放圖（thesis Figure）
目前只有 Re=50 數據，Re=100 和 Re=500 用 placeholder
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 設置字體
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11

# 數據
re_values = [50, 100, 500]
leith_errors = [
    86.9,    # Re=50 實際數據
    np.nan,  # Re=100 待補充
    np.nan,  # Re=500 待補充
]

# 創建圖表
fig, ax = plt.subplots(figsize=(8, 6))

# 繪製 Re=50 實際數據點
ax.plot([50], [86.9], 'o', markersize=10, color='#2E86AB', 
        label='Leith Model (measured)', zorder=3)

# 繪製 Re=100, 500 placeholder（半透明）
ax.plot([100, 500], [90, 95], 'o', markersize=8, color='#2E86AB', 
        alpha=0.3, label='Leith Model (pending)', zorder=2)

# 繪製趨勢線（虛線，半透明）
ax.plot([50, 100, 500], [86.9, 90, 95], '--', linewidth=1.5, 
        color='#2E86AB', alpha=0.3, zorder=1)

# 參考線：100% error
ax.axhline(100, color='red', linestyle='--', alpha=0.4, linewidth=1.5, 
           label='100% error (total failure)', zorder=0)

# 標註 Re=50 數值
ax.annotate(f'{86.9:.1f}%', xy=(50, 86.9), xytext=(50, 82),
            ha='center', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# 軸設置
ax.set_xlabel('Reynolds Number (Re)', fontsize=13)
ax.set_ylabel('Relative L2 Error (%)', fontsize=13)
ax.set_title('Leith Model Error Scaling with Reynolds Number', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xscale('log')
ax.set_xlim(40, 600)
ax.set_ylim(75, 105)
ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
ax.legend(loc='lower right', fontsize=11, framealpha=0.95)

# 添加說明文字
ax.text(0.02, 0.98, 
        'Note: Re=100 and Re=500 values are pending simulation completion.\n'
        'Trend line is illustrative only.',
        transform=ax.transAxes, fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# 保存
output_dir = Path("thesis/result_figures/kolmogorov")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "fig_leith_error_scaling.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_file}")

# 顯示統計
print("\n" + "="*60)
print("Leith Model Error Summary")
print("="*60)
print(f"Re=50:  {leith_errors[0]:.1f}% (measured)")
print(f"Re=100: pending")
print(f"Re=500: pending")
print("="*60)
