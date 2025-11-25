"""
對比視覺化：舊版本 vs 新版本（修正後）K=500 感測點

顯示：
1. 空間分佈對比（2D/3D）
2. 指標對比表格
3. x 座標直方圖
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_sensors(filepath):
    """載入感測點資料"""
    data = np.load(filepath)

    # 檢測舊版本或新版本格式
    if 'sensor_indices' in data:
        # 新版本格式
        coords = data['sensor_coords']
        indices = data['sensor_indices']
    elif 'indices' in data and 'coords' in data:
        # 舊版本格式
        indices = data['indices']
        coords = data['coords']
    else:
        raise ValueError(f"無法識別的檔案格式: {filepath}")

    return coords, indices, data

def compute_metrics(coords):
    """計算空間分佈指標"""
    from scipy.spatial.distance import pdist

    # x 聚集度
    x_range = coords[:, 0].max() - coords[:, 0].min()
    x_std = coords[:, 0].std()
    x_clustering = x_std / x_range if x_range > 0 else 0

    # 壁面覆蓋率
    y_norm = (coords[:, 1] - coords[:, 1].min()) / (coords[:, 1].max() - coords[:, 1].min())
    wall_coverage = np.sum((y_norm < 0.1) | (y_norm > 0.9)) / len(coords)

    # 最小距離
    min_dist = pdist(coords).min() if len(coords) > 1 else 0

    # 平均距離
    mean_dist = pdist(coords).mean() if len(coords) > 1 else 0

    return {
        'x_clustering': x_clustering,
        'wall_coverage': wall_coverage,
        'min_distance': min_dist,
        'mean_distance': mean_dist
    }

def visualize_2d_comparison(old_path, new_path, output_dir):
    """2D 感測點對比視覺化"""
    old_coords, old_indices, old_data = load_sensors(old_path)
    new_coords, new_indices, new_data = load_sensors(new_path)

    old_metrics = compute_metrics(old_coords)
    new_metrics = compute_metrics(new_coords)

    fig = plt.figure(figsize=(18, 12))

    # Row 1: x-y 平面分佈
    ax1 = plt.subplot(3, 3, 1)
    ax1.scatter(old_coords[:, 0], old_coords[:, 1], c='gray', s=10, alpha=0.6, label='Old (before fix)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Old QR-Pivot\n(x-y plane)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 3, 2)
    ax2.scatter(new_coords[:, 0], new_coords[:, 1], c='steelblue', s=10, alpha=0.6, label='New (fixed)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('New QR-Pivot (Fixed)\n(x-y plane)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = plt.subplot(3, 3, 3)
    ax3.scatter(old_coords[:, 0], old_coords[:, 1], c='gray', s=10, alpha=0.4, label='Old')
    ax3.scatter(new_coords[:, 0], new_coords[:, 1], c='steelblue', s=10, alpha=0.4, label='New')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Overlay\n(x-y plane)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Row 2: x 座標直方圖
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(old_coords[:, 0], bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax4.axvline(old_coords[:, 0].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax4.set_xlabel('x coordinate')
    ax4.set_ylabel('Count')
    old_x_clust = old_metrics['x_clustering']
    ax4.set_title(f'Old: x distribution\n(clustering={old_x_clust:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(new_coords[:, 0], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax5.axvline(new_coords[:, 0].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax5.set_xlabel('x coordinate')
    ax5.set_ylabel('Count')
    new_x_clust_val = new_metrics['x_clustering']
    ax5.set_title(f'New: x distribution\n(clustering={new_x_clust_val:.3f})')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 改善百分比
    x_clust_improvement = (new_metrics['x_clustering'] - old_metrics['x_clustering']) / old_metrics['x_clustering'] * 100
    ax6 = plt.subplot(3, 3, 6)
    ax6.text(0.5, 0.7, f'x Clustering Improvement:\n{x_clust_improvement:+.1f}%',
             ha='center', va='center', fontsize=16, fontweight='bold',
             color='green' if x_clust_improvement < 0 else 'red')
    old_x_clust = old_metrics["x_clustering"]
    new_x_clust = new_metrics["x_clustering"]
    ax6.text(0.5, 0.3, f"Target: < 0.3\nOld: {old_x_clust:.3f}\nNew: {new_x_clust:.3f}",
             ha='center', va='center', fontsize=12)
    ax6.axis('off')
    ax6.set_title('Summary')

    # Row 3: 指標對比
    metrics_names = ['x Clustering\n(lower better)', 'Wall Coverage\n(higher better)', 'Min Distance\n(higher better)']
    old_values = [old_metrics['x_clustering'], old_metrics['wall_coverage'], old_metrics['min_distance']]
    new_values = [new_metrics['x_clustering'], new_metrics['wall_coverage'], new_metrics['min_distance']]

    x = np.arange(len(metrics_names))
    width = 0.35

    for i, (name, old_val, new_val) in enumerate(zip(metrics_names, old_values, new_values)):
        ax = plt.subplot(3, 3, 7 + i)
        ax.bar([0, 1], [old_val, new_val], color=['gray', 'steelblue'], alpha=0.7, edgecolor='black')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Old', 'New'])
        ax.set_ylabel('Value')
        ax.set_title(name)
        ax.grid(True, alpha=0.3, axis='y')

        # 顯示改善百分比
        if old_val > 0:
            improvement = (new_val - old_val) / old_val * 100
            ax.text(0.5, max(old_val, new_val) * 1.1, f'{improvement:+.1f}%',
                   ha='center', fontsize=10, fontweight='bold',
                   color='green' if (i == 0 and improvement < 0) or (i > 0 and improvement > 0) else 'red')

    plt.tight_layout()
    output_path = Path(output_dir) / '2d_sensors_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存 2D 對比圖: {output_path}")
    plt.close()

def visualize_3d_comparison(old_path, new_path, output_dir):
    """3D 感測點對比視覺化"""
    old_coords, old_indices, old_data = load_sensors(old_path)
    new_coords, new_indices, new_data = load_sensors(new_path)

    old_metrics = compute_metrics(old_coords)
    new_metrics = compute_metrics(new_coords)

    fig = plt.figure(figsize=(20, 14))

    # Row 1: x-y 平面
    ax1 = plt.subplot(3, 4, 1)
    ax1.scatter(old_coords[:, 0], old_coords[:, 1], c='gray', s=5, alpha=0.5)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Old: x-y plane')
    ax1.grid(True, alpha=0.3)

    ax2 = plt.subplot(3, 4, 2)
    ax2.scatter(new_coords[:, 0], new_coords[:, 1], c='steelblue', s=5, alpha=0.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('New: x-y plane')
    ax2.grid(True, alpha=0.3)

    # Row 1: x-z 平面
    ax3 = plt.subplot(3, 4, 3)
    ax3.scatter(old_coords[:, 0], old_coords[:, 2], c='gray', s=5, alpha=0.5)
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    ax3.set_title('Old: x-z plane')
    ax3.grid(True, alpha=0.3)

    ax4 = plt.subplot(3, 4, 4)
    ax4.scatter(new_coords[:, 0], new_coords[:, 2], c='steelblue', s=5, alpha=0.5)
    ax4.set_xlabel('x')
    ax4.set_ylabel('z')
    ax4.set_title('New: x-z plane')
    ax4.grid(True, alpha=0.3)

    # Row 2: y-z 平面
    ax5 = plt.subplot(3, 4, 5)
    ax5.scatter(old_coords[:, 1], old_coords[:, 2], c='gray', s=5, alpha=0.5)
    ax5.set_xlabel('y')
    ax5.set_ylabel('z')
    ax5.set_title('Old: y-z plane')
    ax5.grid(True, alpha=0.3)

    ax6 = plt.subplot(3, 4, 6)
    ax6.scatter(new_coords[:, 1], new_coords[:, 2], c='steelblue', s=5, alpha=0.5)
    ax6.set_xlabel('y')
    ax6.set_ylabel('z')
    ax6.set_title('New: y-z plane')
    ax6.grid(True, alpha=0.3)

    # Row 2: 3D 散點圖
    from mpl_toolkits.mplot3d import Axes3D

    ax7 = plt.subplot(3, 4, 7, projection='3d')
    ax7.scatter(old_coords[:, 0], old_coords[:, 1], old_coords[:, 2],
               c='gray', s=2, alpha=0.3)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    ax7.set_title('Old: 3D view')

    ax8 = plt.subplot(3, 4, 8, projection='3d')
    ax8.scatter(new_coords[:, 0], new_coords[:, 1], new_coords[:, 2],
               c='steelblue', s=2, alpha=0.3)
    ax8.set_xlabel('x')
    ax8.set_ylabel('y')
    ax8.set_zlabel('z')
    ax8.set_title('New: 3D view')

    # Row 3: 直方圖與指標
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(old_coords[:, 0], bins=30, alpha=0.7, color='gray', edgecolor='black')
    ax9.axvline(old_coords[:, 0].mean(), color='red', linestyle='--', linewidth=2)
    ax9.set_xlabel('x coordinate')
    ax9.set_ylabel('Count')
    old_x_clust = old_metrics["x_clustering"]
    ax9.set_title(f'Old: x dist (clust={old_x_clust:.3f})')
    ax9.grid(True, alpha=0.3, axis='y')

    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(new_coords[:, 0], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax10.axvline(new_coords[:, 0].mean(), color='red', linestyle='--', linewidth=2)
    ax10.set_xlabel('x coordinate')
    ax10.set_ylabel('Count')
    new_x_clust = new_metrics["x_clustering"]
    ax10.set_title(f'New: x dist (clust={new_x_clust:.3f})')
    ax10.grid(True, alpha=0.3, axis='y')

    # 指標對比
    ax11 = plt.subplot(3, 4, 11)
    metrics_names = ['x\nClust', 'Wall\nCov', 'Min\nDist']
    old_vals = [old_metrics['x_clustering'], old_metrics['wall_coverage'], old_metrics['min_distance'] * 100]
    new_vals = [new_metrics['x_clustering'], new_metrics['wall_coverage'], new_metrics['min_distance'] * 100]

    x_pos = np.arange(len(metrics_names))
    width = 0.35
    ax11.bar(x_pos - width/2, old_vals, width, label='Old', color='gray', alpha=0.7, edgecolor='black')
    ax11.bar(x_pos + width/2, new_vals, width, label='New', color='steelblue', alpha=0.7, edgecolor='black')
    ax11.set_xticks(x_pos)
    ax11.set_xticklabels(metrics_names)
    ax11.set_ylabel('Value')
    ax11.set_title('Metrics Comparison')
    ax11.legend()
    ax11.grid(True, alpha=0.3, axis='y')

    # 改善摘要
    ax12 = plt.subplot(3, 4, 12)
    x_clust_imp = (new_metrics['x_clustering'] - old_metrics['x_clustering']) / old_metrics['x_clustering'] * 100
    wall_imp = (new_metrics['wall_coverage'] - old_metrics['wall_coverage']) / old_metrics['wall_coverage'] * 100
    dist_imp = (new_metrics['min_distance'] - old_metrics['min_distance']) / old_metrics['min_distance'] * 100

    summary_text = f'''
Improvements:

x Clustering: {x_clust_imp:+.1f}%
    Old: {old_metrics["x_clustering"]:.3f}
    New: {new_metrics["x_clustering"]:.3f}

Wall Coverage: {wall_imp:+.1f}%
    Old: {old_metrics["wall_coverage"]:.3f}
    New: {new_metrics["wall_coverage"]:.3f}

Min Distance: {dist_imp:+.1f}%
  Old: {old_metrics["min_distance"]:.4f}
  New: {new_metrics["min_distance"]:.4f}
    '''

    ax12.text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=9, family='monospace')
    ax12.axis('off')

    plt.tight_layout()
    output_path = Path(output_dir) / '3d_sensors_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 已保存 3D 對比圖: {output_path}")
    plt.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="對比視覺化：舊 vs 新感測點")
    parser.add_argument("--mode", type=str, required=True, choices=['2d', '3d', 'both'],
                       help="模式: 2d, 3d, 或 both")
    parser.add_argument("--output", type=str, default="results/sensors_comparison",
                       help="輸出目錄")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ['2d', 'both']:
        old_2d = "data/jhtdb/channel_flow_re1000/sensors_K500_qr_pivot.npz"
        new_2d = "data/jhtdb/channel_flow_re1000/sensors_K500_qr_pivot_fixed_2d.npz"

        if Path(old_2d).exists() and Path(new_2d).exists():
            print("生成 2D 對比視覺化...")
            visualize_2d_comparison(old_2d, new_2d, output_dir)
        else:
            print(f"⚠️ 缺少 2D 感測點檔案")

    if args.mode in ['3d', 'both']:
        old_3d = "data/jhtdb/channel_flow_re1000/sensors_K500_hybrid_qr.npz"  # 或其他舊版本
        new_3d = "data/jhtdb/channel_flow_re1000/sensors_K500_qr_pivot_fixed_3d.npz"

        if Path(old_3d).exists() and Path(new_3d).exists():
            print("生成 3D 對比視覺化...")
            visualize_3d_comparison(old_3d, new_3d, output_dir)
        elif not Path(new_3d).exists():
            print(f"⚠️ 3D 新版本尚未生成，請等待 generate_sensors_k500.py 完成")
        else:
            print(f"⚠️ 缺少 3D 舊版本檔案: {old_3d}")

    print(f"\n✅ 完成！輸出目錄: {output_dir}")
