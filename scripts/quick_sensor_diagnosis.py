#!/usr/bin/env python3
"""快速感測器覆蓋診斷"""

import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def diagnose_sensor_coverage(sensor_json, dns_h5, output_dir):
    """診斷感測器覆蓋率與分布"""
    
    # 讀取感測器數據
    with open(sensor_json) as f:
        sensor_data = json.load(f)
    
    K = sensor_data['K']
    indices = sensor_data['indices']
    
    print(f"="*70)
    print(f"感測器覆蓋診斷")
    print(f"="*70)
    print(f"\n📊 基本信息:")
    print(f"   感測器數量 (K): {K}")
    print(f"   方法: {sensor_data.get('method', 'N/A')}")
    if 'condition_number' in sensor_data:
        print(f"   條件數: {sensor_data['condition_number']:.2e}")
    
    # 讀取 DNS 參考場
    with h5py.File(dns_h5, 'r') as f:
        u = f['u'][0]  # 第一個時間步 [Nx, Ny]
        
        # 對於 Kolmogorov flow，domain 通常是 [0, 2π] × [0, 2π]
        Nx, Ny = u.shape
        Lx = Ly = 2.0 * np.pi
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
    
    total_points = Nx * Ny
    
    print(f"\n🌐 DNS 網格:")
    print(f"   網格尺寸: {Nx} × {Ny} = {total_points} points")
    print(f"   覆蓋率: {K}/{total_points} = {K/total_points*100:.3f}%")
    
    # 將索引轉為 2D 坐標
    sensor_i = np.array(indices) // Ny
    sensor_j = np.array(indices) % Ny
    
    sensor_x = x[sensor_i]
    sensor_y = y[sensor_j]
    
    print(f"\n📍 感測器分布:")
    print(f"   x 範圍: [{sensor_x.min():.4f}, {sensor_x.max():.4f}] (domain: [{x.min():.4f}, {x.max():.4f}])")
    print(f"   y 範圍: [{sensor_y.min():.4f}, {sensor_y.max():.4f}] (domain: [{y.min():.4f}, {y.max():.4f}])")
    
    # 計算最近鄰距離
    from scipy.spatial.distance import pdist
    coords = np.stack([sensor_x, sensor_y], axis=1)
    distances = pdist(coords)
    
    min_dist = distances.min()
    mean_dist = distances.mean()
    max_dist = distances.max()
    
    grid_spacing = np.mean([np.diff(x).mean(), np.diff(y).mean()])
    
    print(f"\n📏 感測器間距:")
    print(f"   最小間距: {min_dist:.4f} ({min_dist/grid_spacing:.1f} × 網格間距)")
    print(f"   平均間距: {mean_dist:.4f} ({mean_dist/grid_spacing:.1f} × 網格間距)")
    print(f"   最大間距: {max_dist:.4f} ({max_dist/grid_spacing:.1f} × 網格間距)")
    print(f"   網格間距: {grid_spacing:.4f}")
    
    # 檢查覆蓋空洞
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # 計算每個網格點到最近感測器的距離
    all_coords = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    from scipy.spatial import cKDTree
    tree = cKDTree(coords)
    distances_to_sensors, _ = tree.query(all_coords)
    
    max_dist_to_sensor = distances_to_sensors.max()
    mean_dist_to_sensor = distances_to_sensors.mean()
    
    print(f"\n🕳️  覆蓋空洞分析:")
    print(f"   最遠點距感測器: {max_dist_to_sensor:.4f} ({max_dist_to_sensor/grid_spacing:.1f} × 網格間距)")
    print(f"   平均距離: {mean_dist_to_sensor:.4f}")
    
    coverage_radius_5 = np.sum(distances_to_sensors < 5*grid_spacing) / total_points
    coverage_radius_10 = np.sum(distances_to_sensors < 10*grid_spacing) / total_points
    
    print(f"\n📡 覆蓋半徑分析:")
    print(f"   5  網格半徑內: {coverage_radius_5*100:.1f}%")
    print(f"   10 網格半徑內: {coverage_radius_10*100:.1f}%")
    
    # 視覺化
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左圖：感測器分布
    ax1 = axes[0]
    ax1.scatter(sensor_x, sensor_y, c='red', s=50, alpha=0.7, label=f'Sensors (K={K})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f'Sensor Distribution (K={K}, Coverage={K/total_points*100:.3f}%)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal')
    
    # 右圖：距離場（每個網格點到最近感測器的距離）
    ax2 = axes[1]
    dist_field = distances_to_sensors.reshape(X.shape)
    im = ax2.imshow(dist_field.T, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
                    cmap='viridis', aspect='equal')
    ax2.scatter(sensor_x, sensor_y, c='red', s=20, marker='x', alpha=0.8)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Distance to Nearest Sensor')
    plt.colorbar(im, ax=ax2, label='Distance')
    
    plt.tight_layout()
    fig_path = output_dir / 'sensor_coverage_diagnosis.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 診斷圖已保存: {fig_path}")
    plt.close()
    
    # 保存數值結果
    results = {
        'K': int(K),
        'total_points': int(total_points),
        'coverage_ratio': float(K/total_points),
        'min_sensor_distance': float(min_dist),
        'mean_sensor_distance': float(mean_dist),
        'max_sensor_distance': float(max_dist),
        'grid_spacing': float(grid_spacing),
        'max_distance_to_sensor': float(max_dist_to_sensor),
        'mean_distance_to_sensor': float(mean_dist_to_sensor),
        'coverage_within_5grid': float(coverage_radius_5),
        'coverage_within_10grid': float(coverage_radius_10)
    }
    
    json_path = output_dir / 'sensor_coverage_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 數值結果已保存: {json_path}")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='快速感測器覆蓋診斷')
    parser.add_argument('--sensor-file', required=True, help='感測器 JSON 文件')
    parser.add_argument('--reference', required=True, help='DNS 參考數據 (.h5)')
    parser.add_argument('--output', required=True, help='輸出目錄')
    
    args = parser.parse_args()
    
    diagnose_sensor_coverage(args.sensor_file, args.reference, args.output)
