"""
Visualize comparison between global QR-pivot (100 sensors) 
and per-feature QR-pivot (16 sensors) strategies.
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--global-sensors', type=str,
                       default='data/lowfi/channel_rans/sensors_K100_rans_phase_a.npz',
                       help="Global QR sensor file")
    parser.add_argument('--perfeature-sensors', type=str,
                       default='data/lowfi/channel_rans/sensors_per_feature_5_phase_a.npz',
                       help="Per-feature QR sensor file")
    parser.add_argument('--output', type=str,
                       default='results/sensor_comparison_global_vs_perfeature.png',
                       help="Output figure path")
    args = parser.parse_args()
    
    # Load sensor data
    global_data = np.load(args.global_sensors, allow_pickle=True)
    perfeature_data = np.load(args.perfeature_sensors, allow_pickle=True)
    
    global_pts = global_data['sensor_points']
    perfeature_pts = perfeature_data['sensor_points']
    
    print("="*80)
    print("Global vs Per-Feature Sensor Comparison")
    print("="*80)
    print(f"\n✅ Loaded:")
    print(f"   Global QR:       {len(global_pts)} sensors")
    print(f"   Per-Feature QR:  {len(perfeature_pts)} sensors")
    
    # Compute statistics
    global_y = global_pts[:, 1]
    perfeature_y = perfeature_pts[:, 1]
    
    domain_Ly = float(global_data['domain_Ly'])
    
    print(f"\n📊 Y-Distribution Statistics:")
    print(f"   Global QR:")
    print(f"      Mean y:   {np.mean(global_y):.4f}")
    print(f"      Median y: {np.median(global_y):.4f}")
    print(f"      Std y:    {np.std(global_y):.4f}")
    print(f"      Min y:    {np.min(global_y):.4f}  ({np.min(global_y)/domain_Ly*100:.1f}% of domain)")
    print(f"      Max y:    {np.max(global_y):.4f}  ({np.max(global_y)/domain_Ly*100:.1f}% of domain)")
    
    print(f"\n   Per-Feature QR:")
    print(f"      Mean y:   {np.mean(perfeature_y):.4f}")
    print(f"      Median y: {np.median(perfeature_y):.4f}")
    print(f"      Std y:    {np.std(perfeature_y):.4f}")
    print(f"      Min y:    {np.min(perfeature_y):.4f}  ({np.min(perfeature_y)/domain_Ly*100:.1f}% of domain)")
    print(f"      Max y:    {np.max(perfeature_y):.4f}  ({np.max(perfeature_y)/domain_Ly*100:.1f}% of domain)")
    
    # Count sensors in wall region (y < 0.2 * Ly)
    wall_threshold = 0.2 * domain_Ly
    global_near_wall = np.sum(global_y < wall_threshold)
    perfeature_near_wall = np.sum(perfeature_y < wall_threshold)
    
    print(f"\n🧱 Near-Wall Concentration (y < {wall_threshold:.3f}):")
    print(f"   Global QR:       {global_near_wall}/{len(global_pts)} ({global_near_wall/len(global_pts)*100:.1f}%)")
    print(f"   Per-Feature QR:  {perfeature_near_wall}/{len(perfeature_pts)} ({perfeature_near_wall/len(perfeature_pts)*100:.1f}%)")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Sensor Placement: Global QR vs Per-Feature QR', fontsize=16, weight='bold')
    
    # --- Row 1: Global QR ---
    # Spatial distribution
    ax = axes[0, 0]
    ax.scatter(global_pts[:, 0], global_pts[:, 1], c='blue', s=50, alpha=0.6, edgecolors='k', linewidths=0.5)
    ax.set_xlabel('X (streamwise)', fontsize=11)
    ax.set_ylabel('Y (wall-normal)', fontsize=11)
    ax.set_title(f'Global QR: K={len(global_pts)} sensors\n' +
                 f'Cond = {global_data["condition_number"]:.2e}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Y-histogram
    ax = axes[0, 1]
    ax.hist(global_y, bins=20, color='blue', alpha=0.7, edgecolor='black')
    ax.axvline(wall_threshold, color='red', linestyle='--', linewidth=2, label=f'Wall region (y<{wall_threshold:.2f})')
    ax.set_xlabel('Y (wall-normal)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Y-Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Condition & rank
    ax = axes[0, 2]
    ax.axis('off')
    info_text = f"""Global QR-Pivot Strategy
    
Sensors (K): {len(global_pts)}
Features (n): {global_data['n_features']}
K/n ratio: {len(global_pts)/global_data['n_features']:.2f}

Condition #: {global_data['condition_number']:.2e}
Matrix Rank: {global_data['n_features']}/{global_data['n_features']} (full rank)

Approach:
• Selects K sensors maximizing
  global information across all
  {global_data['n_features']} features simultaneously
• Optimal for overall variance
  but may under-sample some
  individual features
"""
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # --- Row 2: Per-Feature QR ---
    # Spatial distribution
    ax = axes[1, 0]
    ax.scatter(perfeature_pts[:, 0], perfeature_pts[:, 1], 
               c='red', s=100, alpha=0.7, edgecolors='k', linewidths=1.0, marker='s')
    ax.set_xlabel('X (streamwise)', fontsize=11)
    ax.set_ylabel('Y (wall-normal)', fontsize=11)
    ax.set_title(f'Per-Feature QR: K={len(perfeature_pts)} sensors\n' +
                 f'Cond = {perfeature_data["condition_number"]:.2e}', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Y-histogram
    ax = axes[1, 1]
    ax.hist(perfeature_y, bins=20, color='red', alpha=0.7, edgecolor='black')
    ax.axvline(wall_threshold, color='darkred', linestyle='--', linewidth=2, label=f'Wall region (y<{wall_threshold:.2f})')
    ax.set_xlabel('Y (wall-normal)', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Y-Distribution', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Condition & rank
    ax = axes[1, 2]
    ax.axis('off')
    info_text = f"""Per-Feature QR-Pivot Strategy

Sensors (K): {len(perfeature_pts)}
Features (n): {perfeature_data['n_features']}
K/n ratio: {len(perfeature_pts)/perfeature_data['n_features']:.2f}

Condition #: {perfeature_data['condition_number']:.2e}
Dedup Rate: {perfeature_data['deduplication_rate']*100:.1f}%
Multi-Feat: {perfeature_data['multi_feature_count']} sensors

Approach:
• Each feature independently
  selects {perfeature_data['n_per_feature']} most informative points
• Total: {perfeature_data['n_features']} × {perfeature_data['n_per_feature']} = {perfeature_data['n_features'] * perfeature_data['n_per_feature']} → {len(perfeature_pts)} unique
• Ensures all features represented
• {perfeature_data['deduplication_rate']*100:.1f}% overlap → spatial co-location
"""
    ax.text(0.05, 0.95, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\n💾 Saved: {args.output}")
    
    # Show
    # plt.show()
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()
