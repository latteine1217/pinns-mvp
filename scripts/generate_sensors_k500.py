#!/usr/bin/env python3
"""
Kolmogorov Flow QR-Pivot 感測點生成器
使用 POD + QR-Pivot 策略
"""

import h5py
import numpy as np
from pathlib import Path
from scipy.linalg import qr
import argparse

def generate_qr_sensors_kolmogorov(
    dns_file: str,
    snapshot_idx: int = 100,  # t=10.0
    K: int = 100,
    output_file: str = None,
    n_modes: int = 50  # 使用前 50 個 POD 模態
):
    """
    為 Kolmogorov Flow 生成 QR-Pivot 感測點
    使用 POD 模態作為基函數進行 QR-Pivot
    """
    
    print("=" * 70)
    print("🔬 Kolmogorov Flow QR-Pivot 感測點生成 (POD-based)")
    print("=" * 70)
    
    # 讀取多個快照用於 POD
    with h5py.File(dns_file, 'r') as f:
        # 使用 t=5 到 t=15 的快照
        idx_start = 50  # t=5
        idx_end = 150   # t=15
        idx_step = 5
        
        snapshots_u = []
        snapshots_v = []
        
        for idx in range(idx_start, idx_end, idx_step):
            snapshots_u.append(f['u'][idx].flatten())
            snapshots_v.append(f['v'][idx].flatten())
        
        # 目標快照
        u_target = f['u'][snapshot_idx]
        v_target = f['v'][snapshot_idx]
        time_target = f['time'][snapshot_idx]
        
        Ny, Nx = u_target.shape
        N_points = Ny * Nx
        
        print(f"\n📋 數據資訊:")
        print(f"   目標快照時間: t = {time_target:.2f}")
        print(f"   網格大小: {u_target.shape}")
        print(f"   POD 快照數: {len(snapshots_u)}")
    
    # 構建快照矩陣 (N_points*2, N_snapshots)
    snapshots_u = np.array(snapshots_u).T  # (N_points, N_snapshots)
    snapshots_v = np.array(snapshots_v).T
    
    # 堆疊 u 和 v
    snapshots = np.vstack([snapshots_u, snapshots_v])  # (2*N_points, N_snapshots)
    
    print(f"\n🧮 POD 分解:")
    print(f"   快照矩陣: {snapshots.shape}")
    
    # 減去均值
    mean_field = snapshots.mean(axis=1, keepdims=True)
    snapshots_centered = snapshots - mean_field
    
    # SVD 分解
    U, S, Vt = np.linalg.svd(snapshots_centered, full_matrices=False)
    
    # 選擇前 n_modes 個模態
    Phi = U[:, :n_modes]  # (2*N_points, n_modes)
    
    print(f"   POD 模態數: {n_modes}")
    print(f"   模態矩陣 Φ: {Phi.shape}")
    print(f"   累積能量: {np.sum(S[:n_modes]**2) / np.sum(S**2):.4f}")
    
    # QR 分解與列主元選擇
    # Φ^T = Q @ R @ P^T
    Q, R, P = qr(Phi.T, pivoting=True, mode='economic')
    
    # 選擇前 K 個主元
    sensor_indices = P[:K]
    
    print(f"\n📍 感測點選擇:")
    print(f"   選中索引數: {len(sensor_indices)}")
    print(f"   索引範圍: [{sensor_indices.min()}, {sensor_indices.max()}]")
    
    # 分離 u 和 v 的感測點
    # 索引 < N_points 是 u，>= N_points 是 v
    u_sensor_mask = sensor_indices < N_points
    v_sensor_mask = sensor_indices >= N_points
    
    u_sensor_indices = sensor_indices[u_sensor_mask]
    v_sensor_indices = sensor_indices[v_sensor_mask] - N_points
    
    print(f"   u 感測點: {len(u_sensor_indices)}")
    print(f"   v 感測點: {len(v_sensor_indices)}")
    
    # 合併所有唯一位置
    all_positions = np.unique(np.concatenate([u_sensor_indices, v_sensor_indices]))
    
    # 轉換為 (y, x) 坐標
    sensor_y = all_positions // Nx
    sensor_x = all_positions % Nx
    
    # 物理坐標
    L = 2 * np.pi
    dx = L / Nx
    dy = L / Ny
    
    physical_x = sensor_x * dx
    physical_y = sensor_y * dy
    
    # 提取感測點的速度值
    u_flat = u_target.flatten()
    v_flat = v_target.flatten()
    
    sensor_u = u_flat[all_positions]
    sensor_v = v_flat[all_positions]
    
    K_actual = len(all_positions)
    
    # 計算品質指標
    print(f"\n📊 品質指標:")
    
    # 1. 條件數（基於 POD 模態的子集）
    Phi_sensors = Phi[sensor_indices, :]
    cond_num = np.linalg.cond(Phi_sensors)
    print(f"   條件數: {cond_num:.2f}")
    
    if cond_num < 50:
        print(f"      評估: ✅ 優秀 (< 50)")
    elif cond_num < 100:
        print(f"      評估: ⚠️  可接受 (50-100)")
    else:
        print(f"      評估: ❌ 偏高 (> 100)")
    
    # 2. POD 能量比例
    energy_ratio = np.sum(S[:n_modes]**2) / np.sum(S**2)
    print(f"   POD 能量比例: {energy_ratio:.4f}")
    
    # 3. 空間分佈
    if K_actual > 1:
        distances = []
        for i in range(min(K_actual, 50)):  # 避免計算太多
            for j in range(i+1, min(K_actual, 50)):
                dx_ij = min(abs(physical_x[i] - physical_x[j]), L - abs(physical_x[i] - physical_x[j]))
                dy_ij = min(abs(physical_y[i] - physical_y[j]), L - abs(physical_y[i] - physical_y[j]))
                dist = np.sqrt(dx_ij**2 + dy_ij**2)
                distances.append(dist)
        
        min_dist = np.min(distances)
        mean_dist = np.mean(distances)
    else:
        min_dist = 0
        mean_dist = 0
    
    print(f"   實際感測點數: {K_actual}")
    print(f"   最小距離: {min_dist:.4f}")
    print(f"   平均距離: {mean_dist:.4f}")
    
    # 保存結果
    if output_file is None:
        output_file = f"data/kolmogorov_qr_sensors_K{K_actual}_v2.npz"
    
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_file,
        sensor_indices=all_positions,
        sensor_x=physical_x,
        sensor_y=physical_y,
        sensor_u=sensor_u,
        sensor_v=sensor_v,
        grid_x=sensor_x,
        grid_y=sensor_y,
        snapshot_time=time_target,
        K=K_actual,
        K_requested=K,
        condition_number=cond_num,
        energy_ratio=energy_ratio,
        min_distance=min_dist,
        mean_distance=mean_dist,
        grid_shape=u_target.shape,
        n_modes_used=n_modes,
    )
    
    print(f"\n💾 感測點已保存: {output_file}")
    
    # 總結
    print("\n" + "=" * 70)
    print("✅ QR-Pivot 感測點生成完成")
    print("=" * 70)
    print(f"   請求感測點數: {K}")
    print(f"   實際感測點數: {K_actual}")
    print(f"   條件數: {cond_num:.2f} {'✅' if cond_num < 100 else '⚠️'}")
    print(f"   POD 能量比: {energy_ratio:.4f}")
    print("=" * 70)
    
    return all_positions, cond_num, energy_ratio

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成 Kolmogorov Flow QR-Pivot 感測點")
    parser.add_argument("--input", type=str, required=True, help="DNS HDF5 文件")
    parser.add_argument("--snapshot-idx", type=int, default=100, help="快照索引（默認 100 = t=10）")
    parser.add_argument("--K", type=int, default=100, help="感測點數量")
    parser.add_argument("--output", type=str, default=None, help="輸出 npz 文件")
    parser.add_argument("--n-modes", type=int, default=50, help="POD 模態數")
    
    args = parser.parse_args()
    
    generate_qr_sensors_kolmogorov(
        dns_file=args.input,
        snapshot_idx=args.snapshot_idx,
        K=args.K,
        output_file=args.output,
        n_modes=args.n_modes
    )
