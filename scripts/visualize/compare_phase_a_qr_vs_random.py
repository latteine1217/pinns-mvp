#!/usr/bin/env python3
"""
Phase A QR-Pivot vs Random Sensors 空間分布對比圖

比較：
- Phase A QR-Pivot (K=100, 18 features, global QR on standardized features)
- Random Stratified (K=100, 10×10 strata)

用途：
- 論文/thesis 的 sensor 佈局展示圖
- 證明 QR-pivot 的物理導向採樣 vs 均勻隨機
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path


def load_rans_background(rans_file: str, z_slice: int = 47):
    """
    載入 RANS 背景場（速度大小）
    
    Returns
    -------
    x_grid, y_grid, vel_mag : ndarray
        網格座標和速度大小場
    """
    data = np.load(rans_file)
    
    x_grid = data['x']  # [251]
    y_grid = data['y']  # [20]
    
    u = data['u'][:, :, z_slice]  # [251, 20]
    v = data['v'][:, :, z_slice]
    w = data['w'][:, :, z_slice]
    
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    
    return x_grid, y_grid, vel_mag


def compute_sensor_statistics(sensor_x, sensor_y):
    """
    計算 sensor 空間分布統計
    
    Returns
    -------
    stats : dict
        near_wall_fraction: 近壁面比例 (y < 0.2)
        y_mean: Y 方向平均位置
        y_std: Y 方向標準差
    """
    near_wall = np.sum(sensor_y < 0.2) / len(sensor_y)
    
    return {
        'near_wall_fraction': near_wall,
        'y_mean': sensor_y.mean(),
        'y_std': sensor_y.std(),
        'x_mean': sensor_x.mean(),
        'x_std': sensor_x.std()
    }


def create_comparison_figure(
    rans_file: str = 'data/lowfi/channel_rans/rans_k_omega_sst.npz',
    qr_file: str = 'data/lowfi/channel_rans/sensors_K100_rans_phase_a.npz',
    random_file: str = 'data/lowfi/channel_rans/sensors_K100_random_rans_grid.npz',
    output_file: str = 'results/phase_a_qr_vs_random_sensor_layouts.png',
    z_slice: int = 47
):
    """
    創建 Phase A QR vs Random sensors 對比圖
    """
    
    # 載入背景場
    print("Loading RANS background field...")
    x_grid, y_grid, vel_mag = load_rans_background(rans_file, z_slice)
    
    # 載入 QR sensors
    print("Loading Phase A QR sensors...")
    qr_data = np.load(qr_file)
    qr_x = qr_data['sensor_x']
    qr_y = qr_data['sensor_y']
    qr_K = qr_data['K']
    qr_cond = qr_data['condition_number']
    qr_n_features = qr_data['n_features']
    
    qr_stats = compute_sensor_statistics(qr_x, qr_y)
    
    # 載入 Random sensors
    print("Loading random sensors...")
    rand_data = np.load(random_file)
    rand_x = rand_data['sensor_x']
    rand_y = rand_data['sensor_y']
    rand_K = rand_data['K']
    
    rand_stats = compute_sensor_statistics(rand_x, rand_y)
    
    # 創建圖形
    print("Creating figure...")
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # ========== 左圖：QR-Pivot Sensors ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 背景速度場
    im1 = ax1.contourf(x_grid, y_grid, vel_mag.T, 
                       levels=30, cmap='gray', alpha=0.5)
    
    # QR sensors
    ax1.scatter(qr_x, qr_y, c='red', s=60, 
               edgecolors='darkred', linewidths=1.8, 
               alpha=0.85, label=f'QR-Pivot (K={qr_K})', 
               zorder=5)
    
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_title(
        f'(a) QR-Pivot Sensors (Phase A)\n'
        f'{qr_n_features} features, Condition: {qr_cond:.2e}',
        fontsize=13, fontweight='bold'
    )
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(x_grid.min(), x_grid.max())
    ax1.set_ylim(y_grid.min(), y_grid.max())
    ax1.grid(True, alpha=0.2, linestyle='--')
    
    # 添加統計資訊
    stats_text = (
        f"Near-wall (y<0.2): {qr_stats['near_wall_fraction']*100:.1f}%\n"
        f"Mean y: {qr_stats['y_mean']:.3f} ± {qr_stats['y_std']:.3f}"
    )
    ax1.text(0.02, 0.98, stats_text, 
            transform=ax1.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # 添加 colorbar
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('|U| (m/s)', fontsize=10)
    
    # ========== 右圖：Random Sensors ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 背景速度場
    im2 = ax2.contourf(x_grid, y_grid, vel_mag.T, 
                       levels=30, cmap='gray', alpha=0.5)
    
    # Random sensors
    ax2.scatter(rand_x, rand_y, c='blue', s=60, 
               edgecolors='navy', linewidths=1.8, 
               alpha=0.85, label=f'Random (K={rand_K})', 
               zorder=5)
    
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_title(
        f'(b) Random Sensors (Stratified Sampling)\n'
        f'10×10 strata',
        fontsize=13, fontweight='bold'
    )
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(x_grid.min(), x_grid.max())
    ax2.set_ylim(y_grid.min(), y_grid.max())
    ax2.grid(True, alpha=0.2, linestyle='--')
    
    # 添加統計資訊
    stats_text = (
        f"Near-wall (y<0.2): {rand_stats['near_wall_fraction']*100:.1f}%\n"
        f"Mean y: {rand_stats['y_mean']:.3f} ± {rand_stats['y_std']:.3f}"
    )
    ax2.text(0.02, 0.98, stats_text, 
            transform=ax2.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # 添加 colorbar
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('|U| (m/s)', fontsize=10)
    
    # 保存圖形
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"\n✅ Figure saved to: {output_file}")
    print(f"   Resolution: 300 DPI")
    print(f"   Size: {output_path.stat().st_size / 1024:.1f} KB")
    
    # 顯示對比統計
    print("\n" + "="*60)
    print("COMPARISON STATISTICS")
    print("="*60)
    print(f"QR-Pivot (Phase A):")
    print(f"  K = {qr_K}")
    print(f"  Features = {qr_n_features}")
    print(f"  Condition number = {qr_cond:.2e}")
    print(f"  Near-wall sensors = {qr_stats['near_wall_fraction']*100:.1f}%")
    print(f"  Mean Y = {qr_stats['y_mean']:.4f} ± {qr_stats['y_std']:.4f}")
    
    print(f"\nRandom (Stratified):")
    print(f"  K = {rand_K}")
    print(f"  Near-wall sensors = {rand_stats['near_wall_fraction']*100:.1f}%")
    print(f"  Mean Y = {rand_stats['y_mean']:.4f} ± {rand_stats['y_std']:.4f}")
    
    print("\nKey Observations:")
    print(f"  - QR has {qr_stats['near_wall_fraction']/rand_stats['near_wall_fraction']:.1f}× more near-wall sensors")
    print(f"  - QR Y-spread: {qr_stats['y_std']:.4f} vs Random: {rand_stats['y_std']:.4f}")
    print("="*60)
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate Phase A QR vs Random sensor comparison figure'
    )
    parser.add_argument('--rans-file', type=str,
                       default='data/lowfi/channel_rans/rans_k_omega_sst.npz',
                       help='RANS data file')
    parser.add_argument('--qr-file', type=str,
                       default='data/lowfi/channel_rans/sensors_K100_rans_phase_a.npz',
                       help='Phase A QR sensor file')
    parser.add_argument('--random-file', type=str,
                       default='data/lowfi/channel_rans/sensors_K100_random_rans_grid.npz',
                       help='Random sensor file')
    parser.add_argument('--output', type=str,
                       default='results/phase_a_qr_vs_random_sensor_layouts.png',
                       help='Output figure path')
    parser.add_argument('--z-slice', type=int, default=47,
                       help='Z-slice index for background field')
    
    args = parser.parse_args()
    
    create_comparison_figure(
        rans_file=args.rans_file,
        qr_file=args.qr_file,
        random_file=args.random_file,
        output_file=args.output,
        z_slice=args.z_slice
    )


if __name__ == '__main__':
    main()
