#!/usr/bin/env python3
"""
最小距離因子對比可視化腳本
比較不同 min_dist_factor (0.3, 0.5, 0.7) 的 QR-Pivot 感測點分佈
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path
import argparse
from scipy.spatial import distance_matrix


def load_dns_field(dns_path, time_idx=0):
    """載入 DNS 流場數據"""
    with h5py.File(dns_path, 'r') as f:
        u = f['u'][time_idx, :, :]
        v = f['v'][time_idx, :, :]
        velocity_magnitude = np.sqrt(u**2 + v**2)

        Ny, Nx = u.shape
        x = np.linspace(0, 2 * np.pi, Nx)
        y = np.linspace(0, 2 * np.pi, Ny)

    return x, y, velocity_magnitude


def load_qr_sensors(sensor_path):
    """載入 QR-Pivot 感測點（支援 .json 和 .npz 格式）"""
    if sensor_path.endswith('.npz'):
        # 載入 .npz 格式
        data = np.load(sensor_path)
        x_sensors = data['sensor_x']
        y_sensors = data['sensor_y']
        return x_sensors, y_sensors
    else:
        # 載入 .json 格式
        with open(sensor_path, 'r') as f:
            data = json.load(f)
        coordinates = np.array(data['coordinates'])
        return coordinates[:, 0], coordinates[:, 1]


def calculate_metrics(x_sensors, y_sensors):
    """計算感測點品質指標"""
    coords = np.column_stack([x_sensors, y_sensors])
    dist_mat = distance_matrix(coords, coords)
    np.fill_diagonal(dist_mat, np.inf)

    min_distance = dist_mat.min()
    avg_distance = dist_mat[dist_mat < np.inf].mean()

    # 計算理論特徵距離
    area = (2 * np.pi) ** 2
    K = len(x_sensors)
    characteristic_dist = np.sqrt(area / K)

    return {
        'min_distance': min_distance,
        'avg_distance': avg_distance,
        'characteristic_dist': characteristic_dist,
        'actual_factor': min_distance / characteristic_dist
    }


def plot_three_way_comparison(x, y, field, sensors_list, labels, factors,
                                output_path, title_prefix="", cmap='RdBu_r'):
    """繪製三因子對比圖"""

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    vmin, vmax = np.percentile(field, [2, 98])
    X, Y = np.meshgrid(x, y)

    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']

    for idx, (ax, (sx, sy), label, factor, color, marker) in enumerate(
        zip(axes, sensors_list, labels, factors, colors, markers)
    ):
        # 背景流場
        im = ax.contourf(X, Y, field, levels=50, cmap=cmap,
                        vmin=vmin, vmax=vmax, alpha=0.8)
        ax.contour(X, Y, field, levels=10, colors='k',
                  linewidths=0.5, alpha=0.3)

        # 感測點
        ax.scatter(sx, sy, c=color, s=80, marker=marker,
                  edgecolors='white', linewidths=1.5,
                  label=f'K={len(sx)}', zorder=10)

        # 計算指標
        metrics = calculate_metrics(sx, sy)

        # 標題（包含指標）
        title_lines = [
            f'{title_prefix}min_dist_factor = {factor}',
            f'Actual min_dist = {metrics["min_distance"]:.4f}',
            f'(factor = {metrics["actual_factor"]:.2f})'
        ]
        ax.set_title('\n'.join(title_lines),
                    fontsize=13, fontweight='bold')

        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    # 共用色條
    fig.subplots_adjust(right=0.95, wspace=0.3)
    cbar_ax = fig.add_axes([0.96, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=12)

    # 總標題
    fig.suptitle(f'{title_prefix}QR-Pivot: Min Distance Factor Comparison',
                fontsize=17, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 對比圖已保存: {output_path}")
    plt.close()


def print_metrics_table(sensors_list, factors):
    """打印指標對比表"""
    print("\n" + "="*80)
    print("📊 最小距離因子對比指標")
    print("="*80)

    print(f"\n{'Factor':<10} {'Min Dist':<12} {'Avg Dist':<12} {'Actual Factor':<15} {'評估':<20}")
    print("-"*80)

    for (sx, sy), factor in zip(sensors_list, factors):
        metrics = calculate_metrics(sx, sy)

        # 評估狀態
        if metrics['actual_factor'] < 0.4:
            status = "⚠️ 可能聚集"
        elif metrics['actual_factor'] < 0.6:
            status = "✅ 適中"
        else:
            status = "✅ 良好分散"

        print(f"{factor:<10.1f} {metrics['min_distance']:<12.4f} "
              f"{metrics['avg_distance']:<12.4f} "
              f"{metrics['actual_factor']:<15.2f} {status:<20}")

    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="最小距離因子對比可視化")
    parser.add_argument('--dns', type=str, required=True,
                        help='DNS 數據文件路徑')
    parser.add_argument('--sensors-list', type=str, nargs='+', required=True,
                        help='QR 感測點 JSON 文件列表（按順序：0.3, 0.5, 0.7）')
    parser.add_argument('--factors', type=float, nargs='+', required=True,
                        help='對應的 min_dist_factor 值')
    parser.add_argument('--output', type=str, required=True,
                        help='輸出圖片路徑')
    parser.add_argument('--time-idx', type=int, default=100,
                        help='DNS 時間步索引')
    parser.add_argument('--title-prefix', type=str, default='',
                        help='圖片標題前綴')
    parser.add_argument('--cmap', type=str, default='RdBu_r',
                        help='色圖')

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("🎨 最小距離因子對比可視化")
    print("="*80)
    print(f"DNS 數據: {args.dns}")
    print(f"因子數量: {len(args.factors)}")
    print(f"輸出路徑: {args.output}")
    print("="*80 + "\n")

    # 載入 DNS 流場
    print("📂 載入 DNS 流場數據...")
    x, y, velocity_mag = load_dns_field(args.dns, args.time_idx)
    print(f"   域範圍: x ∈ [0, {2*np.pi:.3f}], y ∈ [0, {2*np.pi:.3f}]")
    print(f"   速度範圍: [{velocity_mag.min():.3f}, {velocity_mag.max():.3f}] m/s\n")

    # 載入所有感測點配置
    sensors_list = []
    labels = []

    for sensor_file, factor in zip(args.sensors_list, args.factors):
        print(f"📥 載入 factor={factor} 感測點: {sensor_file}")
        sx, sy = load_qr_sensors(sensor_file)
        sensors_list.append((sx, sy))
        labels.append(f'QR-Pivot (factor={factor})')
        print(f"   ✅ 已載入 {len(sx)} 個感測點\n")

    # 打印指標對比
    print_metrics_table(sensors_list, args.factors)

    # 繪製對比圖
    print("🎨 繪製三因子對比圖...")
    plot_three_way_comparison(x, y, velocity_mag, sensors_list, labels,
                              args.factors, args.output,
                              args.title_prefix, args.cmap)

    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
