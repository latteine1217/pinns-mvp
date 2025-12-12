#!/usr/bin/env python3
"""
感測點策略對比可視化腳本
比較 Latin Hypercube Sampling (LHS) vs QR-Pivot 感測點分佈
背景為實際流場（如 Kolmogorov flow）
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from pathlib import Path
from scipy.stats import qmc
import argparse


def load_dns_field(dns_path, time_idx=0):
    """載入 DNS 流場數據"""
    with h5py.File(dns_path, 'r') as f:
        # 讀取速度場
        u = f['u'][time_idx, :, :]  # (Ny, Nx)
        v = f['v'][time_idx, :, :]

        # 計算速度幅值
        velocity_magnitude = np.sqrt(u**2 + v**2)

        # 創建網格（Kolmogorov flow 域為 [0, 2π] × [0, 2π]）
        Ny, Nx = u.shape
        x = np.linspace(0, 2 * np.pi, Nx)
        y = np.linspace(0, 2 * np.pi, Ny)

    return x, y, velocity_magnitude, u, v


def generate_lhs_sensors(K, domain_x, domain_y, seed=42):
    """使用 Latin Hypercube Sampling 生成感測點"""
    np.random.seed(seed)

    # 使用 scipy 的 LHS
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    samples = sampler.random(n=K)

    # 縮放到域範圍
    x_min, x_max = domain_x
    y_min, y_max = domain_y

    x_sensors = samples[:, 0] * (x_max - x_min) + x_min
    y_sensors = samples[:, 1] * (y_max - y_min) + y_min

    return x_sensors, y_sensors


def load_qr_sensors(sensor_json_path):
    """從 JSON 載入 QR-Pivot 感測點"""
    with open(sensor_json_path, 'r') as f:
        data = json.load(f)

    # 提取感測點座標（[x, y, z] 格式，忽略 z）
    coordinates = np.array(data['coordinates'])
    x_sensors = coordinates[:, 0]
    y_sensors = coordinates[:, 1]

    return x_sensors, y_sensors


def plot_comparison(x, y, field, lhs_x, lhs_y, qr_x, qr_y,
                    output_path, title_prefix="", cmap='RdBu_r'):
    """繪製對比圖"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 計算統一的顏色範圍
    vmin, vmax = np.percentile(field, [2, 98])

    # === 左圖：LHS ===
    ax1 = axes[0]

    # 背景流場
    X, Y = np.meshgrid(x, y)
    im1 = ax1.contourf(X, Y, field, levels=50, cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=0.8)
    ax1.contour(X, Y, field, levels=10, colors='k',
                linewidths=0.5, alpha=0.3)

    # 感測點
    ax1.scatter(lhs_x, lhs_y, c='red', s=80, marker='o',
                edgecolors='white', linewidths=1.5,
                label=f'LHS Sensors (K={len(lhs_x)})', zorder=10)

    ax1.set_xlabel('x', fontsize=14)
    ax1.set_ylabel('y', fontsize=14)
    ax1.set_title(f'{title_prefix}Latin Hypercube Sampling (Random)',
                  fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)

    # === 右圖：QR-Pivot ===
    ax2 = axes[1]

    # 背景流場
    im2 = ax2.contourf(X, Y, field, levels=50, cmap=cmap,
                       vmin=vmin, vmax=vmax, alpha=0.8)
    ax2.contour(X, Y, field, levels=10, colors='k',
                linewidths=0.5, alpha=0.3)

    # 感測點
    ax2.scatter(qr_x, qr_y, c='blue', s=80, marker='s',
                edgecolors='white', linewidths=1.5,
                label=f'QR-Pivot Sensors (K={len(qr_x)})', zorder=10)

    ax2.set_xlabel('x', fontsize=14)
    ax2.set_ylabel('y', fontsize=14)
    ax2.set_title(f'{title_prefix}QR-Pivoting (Optimized)',
                  fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)

    # === 共用色條 ===
    fig.subplots_adjust(right=0.92, wspace=0.25)
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Velocity Magnitude (m/s)', fontsize=12)

    # === 總標題 ===
    fig.suptitle(f'{title_prefix}Sensor Placement Strategy Comparison',
                 fontsize=17, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 對比圖已保存: {output_path}")
    plt.close()


def calculate_coverage_metrics(x_sensors, y_sensors, domain_x, domain_y):
    """計算感測點覆蓋度指標"""
    from scipy.spatial import distance_matrix

    # 計算最小間距
    if len(x_sensors) > 1:
        coords = np.column_stack([x_sensors, y_sensors])
        dist_mat = distance_matrix(coords, coords)
        np.fill_diagonal(dist_mat, np.inf)
        min_distance = dist_mat.min()
        avg_distance = dist_mat[dist_mat < np.inf].mean()
    else:
        min_distance = 0
        avg_distance = 0

    # 計算空間均勻度（標準差）
    x_std = np.std(x_sensors)
    y_std = np.std(y_sensors)

    # 計算域覆蓋率（邊緣區域的點數）
    x_min, x_max = domain_x
    y_min, y_max = domain_y

    margin = 0.1 * (x_max - x_min)  # 10% 邊界
    edge_points = np.sum(
        (x_sensors < x_min + margin) | (x_sensors > x_max - margin) |
        (y_sensors < y_min + margin) | (y_sensors > y_max - margin)
    )
    edge_coverage = edge_points / len(x_sensors) * 100

    return {
        'min_distance': min_distance,
        'avg_distance': avg_distance,
        'x_std': x_std,
        'y_std': y_std,
        'edge_coverage': edge_coverage
    }


def print_comparison_metrics(lhs_metrics, qr_metrics):
    """打印對比指標"""
    print("\n" + "="*70)
    print("📊 感測點策略對比指標")
    print("="*70)

    print(f"\n{'指標':<25} {'LHS (隨機)':<20} {'QR-Pivot (優化)':<20}")
    print("-"*70)
    print(f"{'最小間距':<25} {lhs_metrics['min_distance']:<20.4f} {qr_metrics['min_distance']:<20.4f}")
    print(f"{'平均間距':<25} {lhs_metrics['avg_distance']:<20.4f} {qr_metrics['avg_distance']:<20.4f}")
    print(f"{'X 軸標準差':<25} {lhs_metrics['x_std']:<20.4f} {qr_metrics['x_std']:<20.4f}")
    print(f"{'Y 軸標準差':<25} {lhs_metrics['y_std']:<20.4f} {qr_metrics['y_std']:<20.4f}")
    print(f"{'邊界覆蓋率 (%)':<25} {lhs_metrics['edge_coverage']:<20.2f} {qr_metrics['edge_coverage']:<20.2f}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="感測點策略對比可視化")
    parser.add_argument('--dns', type=str, required=True,
                        help='DNS 數據文件路徑（HDF5 格式）')
    parser.add_argument('--qr-sensors', type=str, required=True,
                        help='QR-Pivot 感測點 JSON 文件路徑')
    parser.add_argument('--K', type=int, required=True,
                        help='感測點數量')
    parser.add_argument('--output', type=str, required=True,
                        help='輸出圖片路徑')
    parser.add_argument('--time-idx', type=int, default=100,
                        help='DNS 時間步索引（默認 100）')
    parser.add_argument('--title-prefix', type=str, default='',
                        help='圖片標題前綴')
    parser.add_argument('--cmap', type=str, default='RdBu_r',
                        help='色圖（默認 RdBu_r）')
    parser.add_argument('--seed', type=int, default=42,
                        help='LHS 隨機種子（默認 42）')

    args = parser.parse_args()

    # 確保輸出目錄存在
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("🎨 感測點策略對比可視化")
    print("="*70)
    print(f"DNS 數據: {args.dns}")
    print(f"QR 感測點: {args.qr_sensors}")
    print(f"感測點數量 K: {args.K}")
    print(f"時間步索引: {args.time_idx}")
    print(f"輸出路徑: {args.output}")
    print("="*70 + "\n")

    # 載入 DNS 流場
    print("📂 載入 DNS 流場數據...")
    x, y, velocity_mag, u, v = load_dns_field(args.dns, args.time_idx)
    domain_x = (x.min(), x.max())
    domain_y = (y.min(), y.max())
    print(f"   域範圍: x ∈ [{domain_x[0]:.3f}, {domain_x[1]:.3f}], "
          f"y ∈ [{domain_y[0]:.3f}, {domain_y[1]:.3f}]")
    print(f"   速度範圍: [{velocity_mag.min():.3f}, {velocity_mag.max():.3f}] m/s\n")

    # 生成 LHS 感測點
    print(f"🎲 生成 Latin Hypercube Sampling 感測點 (K={args.K})...")
    lhs_x, lhs_y = generate_lhs_sensors(args.K, domain_x, domain_y, args.seed)
    print(f"   ✅ 已生成 {len(lhs_x)} 個 LHS 感測點\n")

    # 載入 QR-Pivot 感測點
    print(f"📥 載入 QR-Pivot 感測點...")
    qr_x, qr_y = load_qr_sensors(args.qr_sensors)
    print(f"   ✅ 已載入 {len(qr_x)} 個 QR-Pivot 感測點\n")

    # 計算對比指標
    print("📊 計算覆蓋度指標...")
    lhs_metrics = calculate_coverage_metrics(lhs_x, lhs_y, domain_x, domain_y)
    qr_metrics = calculate_coverage_metrics(qr_x, qr_y, domain_x, domain_y)
    print_comparison_metrics(lhs_metrics, qr_metrics)

    # 繪製對比圖
    print("🎨 繪製對比圖...")
    plot_comparison(x, y, velocity_mag, lhs_x, lhs_y, qr_x, qr_y,
                    args.output, args.title_prefix, args.cmap)

    print("\n✅ 完成！")


if __name__ == '__main__':
    main()
