#!/usr/bin/env python3
"""
使用真實 Channel Flow 數據測試週期性感測點生成
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_channel_flow import ChannelFlowDataFetcher, ChannelFlowConfig


def extract_2d_slice_from_3d(data_3d_path, z_index=64):
    """從 3D cutout 提取 2D 切片"""
    print(f"📂 載入 3D cutout: {data_3d_path}")
    data = np.load(data_3d_path)

    x = data['x']
    y = data['y']
    z = data['z']

    nx, ny, nz = len(x), len(y), len(z)

    print(f"   3D 形狀: x={nx}, y={ny}, z={nz}")
    print(f"   選擇 z 索引: {z_index} (z={z[z_index]:.4f})")

    # 重新整形扁平化的數據為 3D
    u_3d = data['u'].reshape(nx, ny, nz)
    v_3d = data['v'].reshape(nx, ny, nz)
    w_3d = data['w'].reshape(nx, ny, nz)
    p_3d = data['p'].reshape(nx, ny, nz)

    # 提取 2D 切片
    field_2d = {
        'x': x,
        'y': y,
        'u': u_3d[:, :, z_index],
        'v': v_3d[:, :, z_index],
        'w': w_3d[:, :, z_index],
        'p': p_3d[:, :, z_index]
    }

    print(f"   2D 切片形狀: u={field_2d['u'].shape}")
    return field_2d


def main():
    print("=" * 70)
    print("🧪 真實 Channel Flow 數據：週期性感測點測試")
    print("=" * 70)

    # 載入 3D cutout
    data_3d_path = Path("data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz")

    if not data_3d_path.exists():
        print(f"❌ 找不到 3D cutout: {data_3d_path}")
        print("   請先執行: python scripts/fetch_channel_flow.py --cutout3d_mode")
        return 1

    # 提取 2D 切片
    field_data = extract_2d_slice_from_3d(data_3d_path, z_index=64)
    x = field_data['x']
    y = field_data['y']

    # 創建 fetcher
    config = ChannelFlowConfig()
    fetcher = ChannelFlowDataFetcher(config, cache_dir="./data/jhtdb/channel_flow_re1000")

    K = 100

    # ============ 測試 1: 標準 QR-Pivot ============
    print(f"\n🔹 測試 1: 標準 QR-Pivot（無週期性）")
    result_standard = fetcher.generate_sensor_points_with_periodicity(
        field_data,
        K=K,
        use_periodic=False,
        noise_sigma=0.0,
        dropout_prob=0.0
    )

    # ============ 測試 2: 週期性 QR-Pivot ============
    print(f"\n🔹 測試 2: 週期性 QR-Pivot")
    result_periodic = fetcher.generate_sensor_points_with_periodicity(
        field_data,
        K=K,
        periodic_axes=[0],
        use_periodic=True,
        n_wrap_layers=2,
        noise_sigma=0.0,
        dropout_prob=0.0
    )

    # ============ 統計分析 ============
    print("\n" + "=" * 70)
    print("📈 統計分析")
    print("=" * 70)

    def analyze_distribution(sensor_points, label):
        """分析感測點分佈"""
        x_sensors = sensor_points[:, 0]
        y_sensors = sensor_points[:, 1]

        x_min, x_max = x[0], x[-1]
        y_min, y_max = y[0], y[-1]

        seam_threshold = 0.05 * (x_max - x_min)
        near_x0 = np.sum(x_sensors < (x_min + seam_threshold))
        near_xL = np.sum(x_sensors > (x_max - seam_threshold))

        # 壁面區域分析（|y| > 0.9）
        near_wall = np.sum(np.abs(y_sensors) > 0.9)

        print(f"\n{label}:")
        print(f"  x 方向接縫覆蓋:")
        print(f"    x≈0: {near_x0}/{K} ({100*near_x0/K:.1f}%)")
        print(f"    x≈L: {near_xL}/{K} ({100*near_xL/K:.1f}%)")
        print(f"    總計: {near_x0+near_xL}/{K} ({100*(near_x0+near_xL)/K:.1f}%)")
        print(f"  y 方向壁面覆蓋:")
        print(f"    |y| > 0.9: {near_wall}/{K} ({100*near_wall/K:.1f}%)")

        return near_x0, near_xL, near_wall

    std_x0, std_xL, std_wall = analyze_distribution(result_standard['sensor_points'], "標準 QR-Pivot")
    per_x0, per_xL, per_wall = analyze_distribution(result_periodic['sensor_points'], "週期性 QR-Pivot")

    # ============ 品質指標 ============
    print("\n" + "=" * 70)
    print("🎯 品質指標")
    print("=" * 70)

    for key in ['condition_number', 'energy_ratio']:
        if key in result_standard['selection_info'] and key in result_periodic['selection_info']:
            std_val = result_standard['selection_info'][key]
            per_val = result_periodic['selection_info'][key]
            print(f"\n{key}:")
            print(f"  標準: {std_val:.6f}")
            print(f"  週期性: {per_val:.6f}")

    # ============ 視覺化 ============
    print("\n📊 生成視覺化比較...")

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    X, Y = np.meshgrid(x, y, indexing='ij')

    # Row 1: 速度場 u 的分佈
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.7)
    ax1.scatter(result_standard['sensor_points'][:, 0],
               result_standard['sensor_points'][:, 1],
               c='black', s=20, marker='o', alpha=0.8, edgecolors='white', linewidths=0.3)
    ax1.set_title(f'Standard QR-Pivot (K={K})\nSeam: {std_x0+std_xL}/{K}, Wall: {std_wall}/{K}', fontsize=10)
    ax1.set_xlabel('x (streamwise)')
    ax1.set_ylabel('y (wall-normal)')
    plt.colorbar(im, ax=ax1, label='u velocity')
    ax1.axvline(x[0], color='lime', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axvline(x[-1], color='lime', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y[0], color='yellow', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y[-1], color='yellow', linestyle='--', linewidth=1, alpha=0.5)

    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.7)
    ax2.scatter(result_periodic['sensor_points'][:, 0],
               result_periodic['sensor_points'][:, 1],
               c='red', s=20, marker='^', alpha=0.8, edgecolors='white', linewidths=0.3)
    ax2.set_title(f'Periodic QR-Pivot (K={K})\nSeam: {per_x0+per_xL}/{K}, Wall: {per_wall}/{K}', fontsize=10)
    ax2.set_xlabel('x (streamwise)')
    ax2.set_ylabel('y (wall-normal)')
    plt.colorbar(im, ax=ax2, label='u velocity')
    ax2.axvline(x[0], color='lime', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axvline(x[-1], color='lime', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y[0], color='yellow', linestyle='--', linewidth=1, alpha=0.5)
    ax2.axhline(y[-1], color='yellow', linestyle='--', linewidth=1, alpha=0.5)

    # 重疊比較
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.3)
    ax3.scatter(result_standard['sensor_points'][:, 0],
               result_standard['sensor_points'][:, 1],
               c='blue', s=30, marker='o', alpha=0.6, label='Standard', edgecolors='white', linewidths=0.5)
    ax3.scatter(result_periodic['sensor_points'][:, 0],
               result_periodic['sensor_points'][:, 1],
               c='red', s=30, marker='^', alpha=0.6, label='Periodic', edgecolors='white', linewidths=0.5)
    ax3.set_title('Overlay Comparison', fontsize=10)
    ax3.set_xlabel('x (streamwise)')
    ax3.set_ylabel('y (wall-normal)')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.axvline(x[0], color='lime', linestyle='--', linewidth=1, alpha=0.5)
    ax3.axvline(x[-1], color='lime', linestyle='--', linewidth=1, alpha=0.5)

    # Row 2: x 方向分佈
    ax4 = fig.add_subplot(gs[1, :2])
    bins = 40
    ax4.hist(result_standard['sensor_points'][:, 0], bins=bins, alpha=0.5,
             label='Standard', color='blue', edgecolor='black')
    ax4.hist(result_periodic['sensor_points'][:, 0], bins=bins, alpha=0.5,
             label='Periodic', color='red', edgecolor='black')
    ax4.axvline(x[0], color='lime', linestyle='--', linewidth=2, label=f'x=0 (seam)')
    ax4.axvline(x[-1], color='lime', linestyle='--', linewidth=2, label=f'x=L (seam)')
    # 標示 5% 邊界區域
    seam_threshold = 0.05 * (x[-1] - x[0])
    ax4.axvspan(x[0], x[0]+seam_threshold, alpha=0.1, color='green', label='5% seam zone')
    ax4.axvspan(x[-1]-seam_threshold, x[-1], alpha=0.1, color='green')
    ax4.set_xlabel('x coordinate (streamwise)', fontsize=10)
    ax4.set_ylabel('Sensor count', fontsize=10)
    ax4.set_title('x-direction Distribution (Periodicity Test)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)

    # Row 2-3: y 方向分佈
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(result_standard['sensor_points'][:, 1], bins=30, alpha=0.5,
             label='Standard', color='blue', orientation='horizontal', edgecolor='black')
    ax5.hist(result_periodic['sensor_points'][:, 1], bins=30, alpha=0.5,
             label='Periodic', color='red', orientation='horizontal', edgecolor='black')
    ax5.axhline(0.9, color='yellow', linestyle='--', linewidth=2, label='Wall (y=±0.9)')
    ax5.axhline(-0.9, color='yellow', linestyle='--', linewidth=2)
    ax5.set_ylabel('y coordinate (wall-normal)', fontsize=10)
    ax5.set_xlabel('Sensor count', fontsize=10)
    ax5.set_title('y-direction\nDistribution', fontsize=10)
    ax5.legend(fontsize=7)
    ax5.grid(True, alpha=0.3)

    # Row 3: 壓力場分佈
    ax6 = fig.add_subplot(gs[2, 0])
    im = ax6.contourf(X, Y, field_data['p'], levels=50, cmap='viridis', alpha=0.7)
    ax6.scatter(result_standard['sensor_points'][:, 0],
               result_standard['sensor_points'][:, 1],
               c='black', s=15, marker='o', alpha=0.6)
    ax6.set_title('Pressure field - Standard', fontsize=10)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    plt.colorbar(im, ax=ax6, label='pressure')

    ax7 = fig.add_subplot(gs[2, 1])
    im = ax7.contourf(X, Y, field_data['p'], levels=50, cmap='viridis', alpha=0.7)
    ax7.scatter(result_periodic['sensor_points'][:, 0],
               result_periodic['sensor_points'][:, 1],
               c='red', s=15, marker='^', alpha=0.6)
    ax7.set_title('Pressure field - Periodic', fontsize=10)
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    plt.colorbar(im, ax=ax7, label='pressure')

    # 統計摘要
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    summary_text = f"""Quality Metrics Comparison

Standard QR-Pivot:
  • x-seam: {std_x0+std_xL}/{K} ({100*(std_x0+std_xL)/K:.1f}%)
  • y-wall: {std_wall}/{K} ({100*std_wall/K:.1f}%)
  • Condition#: {result_standard['selection_info'].get('condition_number', 'N/A'):.2f}
  • Energy: {result_standard['selection_info'].get('energy_ratio', 'N/A'):.4f}

Periodic QR-Pivot:
  • x-seam: {per_x0+per_xL}/{K} ({100*(per_x0+per_xL)/K:.1f}%)
  • y-wall: {per_wall}/{K} ({100*per_wall/K:.1f}%)
  • Condition#: {result_periodic['selection_info'].get('condition_number', 'N/A'):.2f}
  • Energy: {result_periodic['selection_info'].get('energy_ratio', 'N/A'):.4f}

Domain Info:
  • x: [0, {x[-1]:.2f}] (periodic)
  • y: [{y[0]:.2f}, {y[-1]:.2f}] (wall-bounded)
  • Grid: {len(x)} × {len(y)}
"""

    ax8.text(0.05, 0.95, summary_text, transform=ax8.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Real Channel Flow Re1000: Periodic vs Standard QR-Pivot Sensor Placement',
                 fontsize=13, fontweight='bold', y=0.995)

    # 保存
    output_dir = Path("results/periodic_sensor_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"real_channel_flow_K{K}_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 視覺化已保存: {output_path}")

    # 總結
    print("\n" + "=" * 70)
    print("✅ 測試完成")
    print("=" * 70)

    seam_improvement = (per_x0 + per_xL) - (std_x0 + std_xL)
    print(f"\n接縫覆蓋變化: {seam_improvement:+d} 個點")

    if (per_x0 + per_xL) > (std_x0 + std_xL):
        print(f"✅ 週期性處理提升接縫覆蓋 {abs(seam_improvement)} 點")
    else:
        print(f"⚠️ 週期性處理減少接縫覆蓋 {abs(seam_improvement)} 點")

    print(f"\n輸出文件:")
    print(f"  - 視覺化: {output_path}")
    print(f"  - 標準感測點: data/jhtdb/channel_flow_re1000/sensors_K{K}_qr_pivot_standard.npz")
    print(f"  - 週期性感測點: data/jhtdb/channel_flow_re1000/sensors_K{K}_qr_pivot_periodic.npz")


if __name__ == "__main__":
    sys.exit(main())
