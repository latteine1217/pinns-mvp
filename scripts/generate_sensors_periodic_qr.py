#!/usr/bin/env python3
"""
週期性 Kolmogorov Flow QR-Pivot 感測點生成器 (修正版)
正確處理週期性邊界條件，使用真實的空間 QR Pivot，並加入空間均勻性約束
"""

import sys
from pathlib import Path
# 添加專案根目錄到路徑，以便導入 pinnx
sys.path.append(str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from pinnx.sensors.qr_pivot import QRPivotSelector, PeriodicBoundaryHandler
import argparse
import logging

# 設定 logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_periodic_distance_squared(coords1, coords2, domain_size=2*np.pi):
    """計算週期性距離的平方 (針對 2D 域 [0, domain_size]^2)"""
    dx = np.abs(coords1[0] - coords2[:, 0])
    dy = np.abs(coords1[1] - coords2[:, 1])
    
    dx = np.minimum(dx, domain_size - dx)
    dy = np.minimum(dy, domain_size - dy)
    
    return dx**2 + dy**2

def generate_periodic_qr_sensors(
    dns_file: str,
    time_range: tuple = (20.0, 40.0),
    K: int = 100,
    output_file: str = None,
    n_wrap_layers: int = 2,
    seam_weight: float = 1.0,
    min_dist_factor: float = 0.5
):
    """
    為週期性 Kolmogorov Flow 生成 QR-Pivot 感測點
    
    改進特點：
    1. Spatial QR: 使用物理空間快照進行 QR 分解
    2. Circular Indexing: 正確處理週期性邊界 (QRPivotSelector)
    3. Spatial Filtering: 強制最小距離約束，防止點聚集在單一軸線上
    """
    
    print("=" * 70)
    print("🔬 週期性 Kolmogorov Flow QR-Pivot 感測點生成 (Spatial QR + Filter)")
    print("=" * 70)
    
    # 1. 讀取 DNS 數據
    # ------------------------------------------------------
    with h5py.File(dns_file, 'r') as f:
        t = f['time'][:]
        
        # 選擇時間範圍內的快照
        t_mask = (t >= time_range[0]) & (t <= time_range[1])
        t_indices = np.where(t_mask)[0][::2]  # 每2步取一個
        
        print(f"\n📊 DNS 數據:")
        print(f"   時間範圍: [{t[t_indices[0]]:.2f}, {t[t_indices[-1]]:.2f}]")
        print(f"   快照數: {len(t_indices)}")
        
        # 讀取網格信息
        u0 = f['u'][0]
        Ny, Nx = u0.shape
        
        # 構建座標系統 (假設 Kolmogorov Flow 標準域 [0, 2pi])
        domain_x = 2 * np.pi
        domain_y = 2 * np.pi
        
        x = np.linspace(0, domain_x, Nx, endpoint=False)
        y = np.linspace(0, domain_y, Ny, endpoint=False)
        
        xx, yy = np.meshgrid(x, y)
        coords = np.column_stack([xx.flatten(), yy.flatten()])
        
        print(f"   網格大小: {Ny} × {Nx}")
        print(f"   空間點數: {len(coords)}")
        
        # 讀取物理場數據
        snapshots_u = []
        snapshots_v = []
        
        for idx in t_indices:
            u = f['u'][idx]
            v = f['v'][idx]
            snapshots_u.append(u.flatten())
            snapshots_v.append(v.flatten())
            
    # 2. 構建快照矩陣
    # ------------------------------------------------------
    # [n_locations, n_snapshots]
    snapshots_u = np.array(snapshots_u).T
    snapshots_v = np.array(snapshots_v).T
    
    # 拼接特徵: 這裡將 u 和 v 作為不同的特徵列拼接
    # 形狀: [n_locations, 2 * n_snapshots]
    # 標準化: 對每個特徵（快照）進行標準化，避免能量偏差
    data_matrix = np.concatenate([snapshots_u, snapshots_v], axis=1)
    
    # Z-score 標準化 (對特徵維度)
    # 這有助於 QR 關注結構而非數值大小，減少高能區域聚集
    mean_val = data_matrix.mean(axis=0)
    std_val = data_matrix.std(axis=0) + 1e-8
    data_matrix_norm = (data_matrix - mean_val) / std_val
    
    print(f"\n🧮 快照矩陣構建:")
    print(f"   矩陣形狀: {data_matrix.shape}")
    print(f"   標準化: 已應用 Z-Score")
    
    # 3. QR-Pivot 選擇 (含過採樣)
    # ------------------------------------------------------
    # 我們請求更多的點 (例如 3*K)，然後進行空間過濾
    oversample_factor = 3.0
    K_oversample = int(K * oversample_factor)
    K_oversample = min(K_oversample, len(coords))
    
    print(f"\n🔄 初始化 QRPivotSelector (Circular Indexing):")
    print(f"   Wrap Layers: {n_wrap_layers}")
    print(f"   Seam Weight: {seam_weight}")
    print(f"   Oversampling: {K} -> {K_oversample} (候選池)")
    
    selector = QRPivotSelector(
        use_circular_indexing=True,
        n_wrap_layers=n_wrap_layers,
        seam_weight=seam_weight,
        mode='column',
        pivoting=True,
        regularization=1e-12
    )
    
    # 週期軸配置
    periodic_axes = [0, 1] # x, y 都是週期性
    grid_shape = (Ny, Nx)
    domain_lengths = {0: domain_x, 1: domain_y}
    
    print("\n🚀 執行 QR-Pivot 選擇 (候選池)...")
    candidate_indices, metrics = selector.select_sensors(
        data_matrix=data_matrix_norm,
        n_sensors=K_oversample,
        coords=coords,
        grid_shape=grid_shape,
        periodic_axes=periodic_axes,
        domain_lengths=domain_lengths
    )
    
    # 4. 空間過濾 (Minimum Distance Constraint)
    # ------------------------------------------------------
    print("\n🛡️ 執行空間過濾 (最小距離約束)...")
    
    # 計算啟發式最小距離 r_min
    # 假設均勻分佈時的特徵長度 L_char = sqrt(Area / K)
    # r_min = factor * L_char
    area = domain_x * domain_y
    characteristic_dist = np.sqrt(area / K)
    min_dist = min_dist_factor * characteristic_dist
    min_dist_sq = min_dist ** 2
    
    print(f"   區域面積: {area:.2f}")
    print(f"   特徵間距: {characteristic_dist:.4f}")
    print(f"   最小距離限制 (r_min): {min_dist:.4f} (factor={min_dist_factor})")
    
    final_indices = []
    final_coords = []
    
    # 貪婪過濾
    skipped_count = 0
    for idx in candidate_indices:
        if len(final_indices) >= K:
            break
            
        current_coord = coords[idx]
        
        # 檢查與已選點的距離
        if not final_indices:
            # 第一個點直接加入
            final_indices.append(idx)
            final_coords.append(current_coord)
            continue
            
        # 計算到所有已選點的週期性距離平方
        dists_sq = compute_periodic_distance_squared(
            current_coord, 
            np.array(final_coords), 
            domain_size=domain_x # 假設 x, y 域大小相同
        )
        
        if np.all(dists_sq >= min_dist_sq):
            final_indices.append(idx)
            final_coords.append(current_coord)
        else:
            skipped_count += 1
            
    final_indices = np.array(final_indices)
    
    # 如果過濾後點數不足，從剩下的候選點中回補 (放寬限制)
    if len(final_indices) < K:
        print(f"   ⚠️ 警告: 過濾後點數不足 ({len(final_indices)}/{K})，正在回補...")
        needed = K - len(final_indices)
        
        # 找出尚未被選且被過濾掉的點
        existing_set = set(final_indices)
        for idx in candidate_indices:
            if len(final_indices) >= K:
                break
            if idx not in existing_set:
                final_indices = np.append(final_indices, idx)
    
    # 最終排序 (保持 QR 重要性順序)
    # 其實 final_indices 已經是按重要性加入的，所以不需要額外排序
    
    # 提取座標
    selected_coords = coords[final_indices]
    sensor_x = selected_coords[:, 0]
    sensor_y = selected_coords[:, 1]
    
    print(f"\n✅ 選擇完成:")
    print(f"   最終點數: {len(final_indices)}")
    print(f"   因距離過濾跳過: {skipped_count} 點")
    print(f"   QR 條件數 (前 {K} 點): {metrics.get('condition_number', 'N/A')}")
    
    # 5. 保存結果
    # ------------------------------------------------------
    if output_file is None:
        output_file = f"data/kolmogorov_qr_sensors_spatial_K{K}.npz"
        
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_file,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        sensor_indices=final_indices,
        K=K,
        metrics=metrics,
        method='Spatial-QR-Circular-Filtered',
        periodic=True,
        domain_x=domain_x,
        domain_y=domain_y,
        min_dist=min_dist
    )
    
    print(f"\n💾 感測點已保存: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='DNS HDF5 file')
    parser.add_argument('--K', type=int, default=100, help='Number of sensors')
    parser.add_argument('--output', help='Output NPZ file')
    parser.add_argument('--t-start', type=float, default=20.0)
    parser.add_argument('--t-end', type=float, default=40.0)
    parser.add_argument('--wrap-layers', type=int, default=2, help='Number of wrap layers for circular indexing')
    parser.add_argument('--seam-weight', type=float, default=1.0, help='Weight for seam region (1.0 = unbiased)')
    parser.add_argument('--min-dist-factor', type=float, default=0.5, help='Minimum distance factor (relative to avg spacing)')
    
    args = parser.parse_args()
    
    generate_periodic_qr_sensors(
        dns_file=args.input,
        time_range=(args.t_start, args.t_end),
        K=args.K,
        output_file=args.output,
        n_wrap_layers=args.wrap_layers,
        seam_weight=args.seam_weight,
        min_dist_factor=args.min_dist_factor
    )