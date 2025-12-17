#!/usr/bin/env python3
"""
隨機感測點生成器（對照 QR-Pivot）
針對 JHTDB Channel Flow Re_tau=1000 設計

目的：
- 為 QR vs Random 對比實驗生成對照組
- 支援 2D (x-y 平面) 與 3D (x-y-z 體積) 模式
- 支援 Stratified Sampling（避免極端聚集）

使用方法：
    # 2D Random (Stratified)
    python generate_channel_random_sensors.py \\
        --K 100 --seed 42 --stratified \\
        --output data/jhtdb/channel_flow_re1000/sensors_K100_random_stratified.npz
    
    # 2D Random (Pure Uniform)
    python generate_channel_random_sensors.py \\
        --K 100 --seed 42 \\
        --output data/jhtdb/channel_flow_re1000/sensors_K100_random_uniform.npz
    
    # 3D Random (Stratified)
    python generate_channel_random_sensors.py \\
        --K 100 --seed 42 --stratified --mode 3d \\
        --output data/jhtdb/channel_flow_re1000/sensors_K100_random_3d.npz
"""

import numpy as np
import argparse
from pathlib import Path
import sys

def generate_random_sensors(K, domain_bounds, seed=42, stratified=False, mode='2d'):
    """
    生成隨機感測點佈局
    
    Args:
        K: 感測點數量
        domain_bounds: {'x': [xmin, xmax], 'y': [ymin, ymax], 'z': [zmin, zmax] (3D only)}
        seed: 隨機種子（確保可重現）
        stratified: 是否使用分層採樣（避免極端聚集）
        mode: '2d' or '3d'
    
    Returns:
        sensor_points: [K, 2] (2D) or [K, 3] (3D) 坐標
        metadata: 生成資訊字典
    """
    np.random.seed(seed)
    
    xmin, xmax = domain_bounds['x']
    ymin, ymax = domain_bounds['y']
    
    if mode == '3d':
        if 'z' not in domain_bounds:
            raise ValueError("3D mode requires 'z' key in domain_bounds")
        zmin, zmax = domain_bounds['z']
        ndim = 3
    else:
        ndim = 2
        zmin, zmax = None, None  # 2D mode doesn't use z
    
    print("=" * 70)
    print("🎲 隨機感測點生成")
    print("=" * 70)
    print(f"\n模式: {mode.upper()}")
    print(f"目標點數: {K}")
    print(f"隨機種子: {seed}")
    print(f"策略: {'Stratified Sampling' if stratified else 'Pure Uniform Random'}")
    
    if stratified:
        # 分層採樣：將域切成網格，在每個格子內隨機選點
        if ndim == 2:
            # 2D: sqrt(K) × sqrt(K) 格子
            n_grid = int(np.ceil(np.sqrt(K)))
            x_bins = np.linspace(xmin, xmax, n_grid + 1)
            y_bins = np.linspace(ymin, ymax, n_grid + 1)
            
            print(f"\n網格劃分: {n_grid} × {n_grid} = {n_grid**2} 格子")
            
            sensors = []
            for i in range(min(K, n_grid * n_grid)):
                ix = i % n_grid
                iy = i // n_grid
                # 在每個格子內隨機選點
                x = np.random.uniform(x_bins[ix], x_bins[ix+1])
                y = np.random.uniform(y_bins[iy], y_bins[iy+1])
                sensors.append([x, y])
            
            sensor_points = np.array(sensors[:K], dtype=np.float32)
            
        else:
            # 3D: K^(1/3) × K^(1/3) × K^(1/3) 格子
            n_grid = int(np.ceil(K ** (1/3)))
            x_bins = np.linspace(xmin, xmax, n_grid + 1)
            y_bins = np.linspace(ymin, ymax, n_grid + 1)
            z_bins = np.linspace(zmin, zmax, n_grid + 1)
            
            print(f"\n網格劃分: {n_grid} × {n_grid} × {n_grid} = {n_grid**3} 格子")
            
            sensors = []
            for i in range(min(K, n_grid ** 3)):
                ix = i % n_grid
                iy = (i // n_grid) % n_grid
                iz = i // (n_grid ** 2)
                # 在每個格子內隨機選點
                x = np.random.uniform(x_bins[ix], x_bins[ix+1])
                y = np.random.uniform(y_bins[iy], y_bins[iy+1])
                z = np.random.uniform(z_bins[iz], z_bins[iz+1])
                sensors.append([x, y, z])
            
            sensor_points = np.array(sensors[:K], dtype=np.float32)
    
    else:
        # 純隨機（Uniform）
        if ndim == 2:
            x = np.random.uniform(xmin, xmax, K)
            y = np.random.uniform(ymin, ymax, K)
            sensor_points = np.column_stack([x, y]).astype(np.float32)
        else:
            x = np.random.uniform(xmin, xmax, K)
            y = np.random.uniform(ymin, ymax, K)
            z = np.random.uniform(zmin, zmax, K)
            sensor_points = np.column_stack([x, y, z]).astype(np.float32)
    
    # 構建 metadata（兼容 QR sensor 格式）
    metadata = {
        'strategy': f"random_{'stratified' if stratified else 'uniform'}",
        'K_requested': K,
        'K_actual': len(sensor_points),
        'seed': seed,
        'mode': mode,
        'domain_bounds': domain_bounds,
        'periodic_axes': [],  # Random 不考慮週期性
        'use_periodic': False,
        'circular_indexing_enabled': False
    }
    
    # 統計分析
    print(f"\n📊 生成結果:")
    print(f"   實際點數: {len(sensor_points)}")
    print(f"   X 範圍: [{sensor_points[:, 0].min():.4f}, {sensor_points[:, 0].max():.4f}]")
    print(f"   Y 範圍: [{sensor_points[:, 1].min():.4f}, {sensor_points[:, 1].max():.4f}]")
    if ndim == 3:
        print(f"   Z 範圍: [{sensor_points[:, 2].min():.4f}, {sensor_points[:, 2].max():.4f}]")
    
    # 計算最近鄰距離（診斷聚集程度）
    if len(sensor_points) > 1:
        from scipy.spatial.distance import cdist
        dists = cdist(sensor_points, sensor_points)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        
        print(f"\n📏 最近鄰距離分析:")
        print(f"   Min: {min_dists.min():.4f}")
        print(f"   Mean: {min_dists.mean():.4f}")
        print(f"   Max: {min_dists.max():.4f}")
        print(f"   Std: {min_dists.std():.4f}")
        
        # 警告：過度聚集檢測
        if min_dists.min() < 0.01:
            print(f"   ⚠️  警告：存在極近的點對 (min_dist={min_dists.min():.4f})")
    
    return sensor_points, metadata


def main():
    parser = argparse.ArgumentParser(
        description="生成隨機感測點佈局（QR-Pivot 對照組）"
    )
    parser.add_argument('--K', type=int, default=100,
                       help='感測點數量')
    parser.add_argument('--seed', type=int, default=42,
                       help='隨機種子（確保可重現）')
    parser.add_argument('--stratified', action='store_true',
                       help='使用分層採樣（避免極端聚集）')
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='2d',
                       help='2D (x-y 平面) 或 3D (x-y-z 體積)')
    parser.add_argument('--output', type=str, required=True,
                       help='輸出 NPZ 文件路徑')
    
    # 可選：自訂域範圍（預設使用 JHTDB Channel Flow）
    parser.add_argument('--x-range', type=float, nargs=2, default=None,
                       help='X 範圍 [xmin, xmax]（預設：JHTDB）')
    parser.add_argument('--y-range', type=float, nargs=2, default=None,
                       help='Y 範圍 [ymin, ymax]（預設：JHTDB）')
    parser.add_argument('--z-range', type=float, nargs=2, default=None,
                       help='Z 範圍 [zmin, zmax]（僅 3D，預設：JHTDB）')
    
    args = parser.parse_args()
    
    # JHTDB Channel Flow 標準域範圍
    if args.x_range is not None:
        x_range = args.x_range
    else:
        # x: 4π (streamwise, periodic)
        x_range = [0, 12.566]
    
    if args.y_range is not None:
        y_range = args.y_range
    else:
        # y: 2h (wall-normal, centered at y=0)
        y_range = [-1.0, 1.0]
    
    domain_bounds = {
        'x': x_range,
        'y': y_range
    }
    
    if args.mode == '3d':
        if args.z_range is not None:
            z_range = args.z_range
        else:
            # z: 4π/3 (spanwise, periodic)
            z_range = [0, 4.188]
        domain_bounds['z'] = z_range
    
    print(f"\n域範圍:")
    print(f"   X: [{domain_bounds['x'][0]:.4f}, {domain_bounds['x'][1]:.4f}] (Streamwise)")
    print(f"   Y: [{domain_bounds['y'][0]:.4f}, {domain_bounds['y'][1]:.4f}] (Wall-Normal)")
    if args.mode == '3d':
        print(f"   Z: [{domain_bounds['z'][0]:.4f}, {domain_bounds['z'][1]:.4f}] (Spanwise)")
    
    # 生成感測點
    sensor_points, metadata = generate_random_sensors(
        K=args.K,
        domain_bounds=domain_bounds,
        seed=args.seed,
        stratified=args.stratified,
        mode=args.mode
    )
    
    # 保存（格式兼容 QR sensor 文件）
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'sensor_points': sensor_points,
        'selection_info': metadata,
        'noise_sigma': 0.0,
        'dropout_prob': 0.0,
        # 兼容性欄位（訓練腳本可能需要）
        'sensor_x': sensor_points[:, 0],
        'sensor_y': sensor_points[:, 1],
        'K': args.K
    }
    
    if args.mode == '3d':
        save_dict['sensor_z'] = sensor_points[:, 2]
    
    np.savez(args.output, **save_dict)
    
    print(f"\n💾 已保存: {args.output}")
    print("=" * 70)
    print("✅ 完成")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
