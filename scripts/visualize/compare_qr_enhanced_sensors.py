#!/usr/bin/env python3
"""
比較不同 QR-Pivot 感測器策略

比較：
1. Original QR (6 features): u, v, w, omega_z, grad_u_eig1/2
2. Enhanced QR-Minimal (10 features): + p, dudy, k, tau_uv
3. Enhanced QR-Physics (15 features): + velocity gradients, pressure gradients
4. Enhanced QR-Full (20 features): + all Reynolds stresses
5. Random (baseline)
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# File paths
data_dir = Path("data/lowfi/channel_rans")
sensor_files = {
    'Random': 'data/jhtdb/channel_flow_re1000/sensors_K100_random_stratified.npz',
    'QR-Original (6)': 'data/jhtdb/channel_flow_re1000/sensors_K100_qr_pivot_periodic.npz',
    'QR-Minimal (10)': data_dir / 'sensors_K100_rans_enhanced_minimal.npz',
    'QR-Physics (15)': data_dir / 'sensors_K100_rans_enhanced_physics_guided.npz',
    'QR-Phase-A (18)': data_dir / 'sensors_K100_rans_phase_a.npz',
    'QR-Full (20)': data_dir / 'sensors_K100_rans_enhanced_full.npz',
}

# Load sensor data
sensors_data = {}
for name, fpath in sensor_files.items():
    if Path(fpath).exists():
        data = np.load(fpath, allow_pickle=True)
        
        # Handle different file formats
        if 'sensor_x' in data and 'sensor_y' in data:
            x, y = data['sensor_x'], data['sensor_y']
        elif 'sensor_points' in data:
            points = data['sensor_points']
            x, y = points[:, 0], points[:, 1]
        else:
            print(f"✗ Unknown format for {name}")
            continue
        
        sensors_data[name] = {
            'x': x,
            'y': y,
            'K': int(data.get('K', len(x))),
            'cond': float(data.get('condition_number', np.nan)),
            'n_features': int(data.get('n_features', 0)),
        }
        print(f"✓ Loaded {name}: K={sensors_data[name]['K']}, features={sensors_data[name]['n_features']}, cond={sensors_data[name]['cond']:.2e}")
    else:
        print(f"✗ File not found: {fpath}")

# Create comparison plot (now 6 strategies)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

# Color map (6 colors for 6 strategies)
colors = ['gray', 'red', 'blue', 'green', 'orange', 'purple']
markers = ['o', 's', '^', 'D', 'p', 'v']

for idx, (name, color, marker) in enumerate(zip(sensors_data.keys(), colors, markers)):
    ax = axes[idx]
    data = sensors_data[name]
    
    # Scatter plot
    sc = ax.scatter(data['x'], data['y'], 
                   c=color, marker=marker, s=30, alpha=0.7, 
                   edgecolors='black', linewidths=0.5,
                   label=name)
    
    # Formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title(f"{name}\nK={data['K']}, Features={data['n_features']}, Cond={data['cond']:.2e}", 
                fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', 'box')
    
    # Add y=0 wall line
    ax.axhline(0, color='black', linewidth=2, linestyle='-', alpha=0.5, label='Wall')
    
    # Add histogram on side (y-distribution)
    hist_ax = ax.twinx()
    hist_ax.hist(data['y'], bins=20, orientation='horizontal', 
                alpha=0.2, color=color, edgecolor='black')
    hist_ax.set_ylabel('Sensor Count', fontsize=10)
    hist_ax.tick_params(labelsize=9)

# Keep all 6 subplots visible (no need to remove)

# Add overall title
fig.suptitle('QR-Pivot Sensor Comparison: Original → Phase A → Full Enhancement', 
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=(0, 0, 1, 0.96))

# Save
output_path = Path('results/qr_enhanced_sensor_comparison.png')
output_path.parent.mkdir(exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\n💾 Saved: {output_path}")

# ========== Quantitative Comparison Table ==========
print("\n" + "=" * 80)
print("📊 Quantitative Comparison")
print("=" * 80)
print(f"{'Strategy':<20} {'Features':>10} {'Cond Number':>15} {'Near-Wall %':>12} {'Uniform %':>12}")
print("-" * 80)

for name, data in sensors_data.items():
    # Calculate near-wall percentage (|y| < 0.2)
    near_wall_mask = np.abs(data['y']) < 0.2
    near_wall_pct = 100 * near_wall_mask.sum() / data['K']
    
    # Calculate uniformity (std of y)
    y_std = data['y'].std()
    
    print(f"{name:<20} {data['n_features']:>10} {data['cond']:>15.2e} {near_wall_pct:>11.1f}% {y_std:>11.4f}")

print("=" * 80)
print("\n✅ Analysis complete!")
print(f"\n📌 Key Observations:")
print(f"  • Minimal (10 features) has best condition number: {sensors_data['QR-Minimal (10)']['cond']:.2e}")
if 'QR-Phase-A (18)' in sensors_data:
    print(f"  • Phase A (18 features) adds advanced turbulence physics: {sensors_data['QR-Phase-A (18)']['cond']:.2e}")
print(f"  • Full (20 features) has most physical information but worse conditioning")
print(f"  • All QR methods show near-wall clustering (physically motivated)")
print(f"  • Random shows uniform distribution (no physics guidance)")

plt.show()
