#!/usr/bin/env python3
"""
生成 RANS Grid 上的隨機 Sensors（用於與 Phase A QR-Pivot 對比）

目的：
- 在與 Phase A QR sensors 相同的 2D domain 上生成隨機採樣點
- 確保公平比較（相同 K=100, 相同空間範圍）
- 使用 stratified sampling 提高空間覆蓋均勻性

輸出：
- data/lowfi/channel_rans/sensors_K100_random_rans_grid.npz
"""

import numpy as np
import argparse
from pathlib import Path


def generate_random_sensors_rans_grid(
    rans_file: str = 'data/lowfi/channel_rans/rans_k_omega_sst.npz',
    K: int = 100,
    z_slice: int = 47,
    seed: int = 42,
    output_file: str = 'data/lowfi/channel_rans/sensors_K100_random_rans_grid.npz',
    stratified: bool = True
):
    """
    在 RANS 2D grid 上生成隨機 sensors
    
    Parameters
    ----------
    rans_file : str
        RANS 數據文件路徑
    K : int
        Sensor 數量
    z_slice : int
        Z 方向切片索引（與 Phase A 一致）
    seed : int
        隨機種子（可重現性）
    output_file : str
        輸出文件路徑
    stratified : bool
        是否使用分層採樣（提高均勻性）
    """
    
    # 設置隨機種子
    rng = np.random.RandomState(seed)
    
    # 載入 RANS grid
    print(f"Loading RANS data from: {rans_file}")
    rans_data = np.load(rans_file)
    
    x_grid = rans_data['x']  # [251]
    y_grid = rans_data['y']  # [20]
    z_grid = rans_data['z']  # [94]
    
    nx, ny = len(x_grid), len(y_grid)
    total_points = nx * ny
    
    print(f"\nGrid info:")
    print(f"  Shape: ({nx}, {ny}) = {total_points} points")
    print(f"  X range: [{x_grid.min():.4f}, {x_grid.max():.4f}]")
    print(f"  Y range: [{y_grid.min():.4f}, {y_grid.max():.4f}]")
    print(f"  Z slice: {z_slice} (z={z_grid[z_slice]:.4f})")
    
    # 生成隨機採樣索引
    if stratified:
        print(f"\n使用分層採樣策略 (K={K})...")
        
        # 將 2D grid 分成 sqrt(K) × sqrt(K) 的區塊
        n_strata = int(np.sqrt(K))
        samples_per_stratum = K // (n_strata ** 2)
        remaining = K - samples_per_stratum * (n_strata ** 2)
        
        print(f"  Strata: {n_strata} × {n_strata} = {n_strata**2} blocks")
        print(f"  Samples per block: {samples_per_stratum}")
        print(f"  Remaining samples: {remaining}")
        
        # X 和 Y 方向的分層邊界
        x_edges = np.linspace(0, nx, n_strata + 1, dtype=int)
        y_edges = np.linspace(0, ny, n_strata + 1, dtype=int)
        
        flat_indices = []
        
        # 從每個區塊採樣
        for i in range(n_strata):
            for j in range(n_strata):
                # 當前區塊的索引範圍
                x_start, x_end = x_edges[i], x_edges[i + 1]
                y_start, y_end = y_edges[j], y_edges[j + 1]
                
                # 當前區塊的所有點
                block_x_indices = np.arange(x_start, x_end)
                block_y_indices = np.arange(y_start, y_end)
                
                # 生成區塊內所有點的 flat indices
                block_flat = []
                for yi in block_y_indices:
                    for xi in block_x_indices:
                        block_flat.append(yi * nx + xi)
                
                block_flat = np.array(block_flat)
                
                # 從區塊中隨機採樣
                n_samples = samples_per_stratum
                if len(flat_indices) < K and remaining > 0:
                    n_samples += 1
                    remaining -= 1
                
                if len(block_flat) >= n_samples:
                    sampled = rng.choice(block_flat, size=n_samples, replace=False)
                    flat_indices.extend(sampled.tolist())
        
        flat_indices = np.array(flat_indices[:K])
        
    else:
        print(f"\n使用純隨機採樣 (K={K})...")
        flat_indices = rng.choice(total_points, size=K, replace=False)
    
    # 轉換為 2D 索引
    y_idx = flat_indices // nx
    x_idx = flat_indices % nx
    
    # 提取座標
    sensor_x = x_grid[x_idx]
    sensor_y = y_grid[y_idx]
    sensor_points = np.column_stack([sensor_x, sensor_y])
    
    # 統計分析
    near_wall_threshold = 0.2
    near_wall_count = np.sum(sensor_y < near_wall_threshold)
    near_wall_frac = near_wall_count / K
    
    print(f"\n採樣統計:")
    print(f"  總數: {K}")
    print(f"  X 覆蓋: [{sensor_x.min():.4f}, {sensor_x.max():.4f}]")
    print(f"  Y 覆蓋: [{sensor_y.min():.4f}, {sensor_y.max():.4f}]")
    print(f"  近壁面 (y<{near_wall_threshold}): {near_wall_count} ({near_wall_frac*100:.1f}%)")
    
    # 保存
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_file,
        sensor_points=sensor_points,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_indices=flat_indices,
        K=K,
        method='random_stratified' if stratified else 'random',
        grid_shape=np.array([nx, ny]),
        z_slice=z_slice,
        z_coordinate=z_grid[z_slice],
        domain_Lx=x_grid.max() - x_grid.min(),
        domain_Ly=y_grid.max() - y_grid.min(),
        seed=seed,
        near_wall_fraction=near_wall_frac,
        near_wall_threshold=near_wall_threshold
    )
    
    print(f"\n✅ Saved to: {output_file}")
    print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
    
    return sensor_points, flat_indices


def main():
    parser = argparse.ArgumentParser(description='Generate random sensors on RANS grid')
    parser.add_argument('--rans-file', type=str, 
                       default='data/lowfi/channel_rans/rans_k_omega_sst.npz',
                       help='RANS data file')
    parser.add_argument('--K', type=int, default=100,
                       help='Number of sensors')
    parser.add_argument('--z-slice', type=int, default=47,
                       help='Z-slice index (match Phase A)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str,
                       default='data/lowfi/channel_rans/sensors_K100_random_rans_grid.npz',
                       help='Output file path')
    parser.add_argument('--no-stratified', action='store_true',
                       help='Use pure random sampling (not stratified)')
    
    args = parser.parse_args()
    
    generate_random_sensors_rans_grid(
        rans_file=args.rans_file,
        K=args.K,
        z_slice=args.z_slice,
        seed=args.seed,
        output_file=args.output,
        stratified=not args.no_stratified
    )


if __name__ == '__main__':
    main()
