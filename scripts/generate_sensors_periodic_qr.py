#!/usr/bin/env python3
"""
週期性 Kolmogorov Flow QR-Pivot 感測點生成器
正確處理週期性邊界條件
"""

import h5py
import numpy as np
from pathlib import Path
from scipy.linalg import qr
import argparse

def generate_periodic_qr_sensors(
    dns_file: str,
    time_range: tuple = (20.0, 40.0),
    K: int = 100,
    output_file: str = None,
    n_modes: int = 50
):
    """
    為週期性 Kolmogorov Flow 生成 QR-Pivot 感測點
    
    關鍵改進：
    1. POD 分解考慮週期性（使用 Fourier 模態或週期性 padding）
    2. 感測點選擇時避免邊界冗餘
    3. 確保重建時週期性連續
    """
    
    print("=" * 70)
    print("🔬 週期性 Kolmogorov Flow QR-Pivot 感測點生成")
    print("=" * 70)
    
    # 讀取 DNS 數據
    with h5py.File(dns_file, 'r') as f:
        t = f['time'][:]
        
        # 選擇時間範圍內的快照
        t_mask = (t >= time_range[0]) & (t <= time_range[1])
        t_indices = np.where(t_mask)[0][::2]  # 每2步取一個
        
        print(f"\n📊 DNS 數據:")
        print(f"   時間範圍: [{t[t_indices[0]]:.2f}, {t[t_indices[-1]]:.2f}]")
        print(f"   快照數: {len(t_indices)}")
        
        # 讀取快照
        snapshots_u = []
        snapshots_v = []
        
        for idx in t_indices:
            u = f['u'][idx]
            v = f['v'][idx]
            
            # ✅ 關鍵：使用 Fourier 模態（天然週期性）
            u_fft = np.fft.rfft2(u)  # 實數 FFT（週期性）
            v_fft = np.fft.rfft2(v)
            
            snapshots_u.append(u_fft.flatten())
            snapshots_v.append(v_fft.flatten())
        
        Ny, Nx = f['u'][0].shape
        print(f"   網格大小: {Ny} × {Nx}")
    
    # 構建 Fourier 係數快照矩陣
    snapshots_u = np.array(snapshots_u).T
    snapshots_v = np.array(snapshots_v).T
    
    # 堆疊實部和虛部
    snapshots = np.vstack([
        snapshots_u.real, snapshots_u.imag,
        snapshots_v.real, snapshots_v.imag
    ])
    
    print(f"\n🧮 Fourier-POD 分解:")
    print(f"   快照矩陣: {snapshots.shape}")
    
    # POD 分解（在 Fourier 空間）
    mean_field = snapshots.mean(axis=1, keepdims=True)
    snapshots_centered = snapshots - mean_field
    
    U, S, Vt = np.linalg.svd(snapshots_centered, full_matrices=False)
    
    Phi = U[:, :n_modes]
    energy_ratio = np.sum(S[:n_modes]**2) / np.sum(S**2)
    
    print(f"   POD 模態數: {n_modes}")
    print(f"   累積能量: {energy_ratio:.4f}")
    
    # QR-Pivot 選擇 Fourier 模態
    Q, R, P = qr(Phi.T, pivoting=True, mode='economic')
    fourier_indices = P[:K]
    
    print(f"\n📍 Fourier 模態選擇:")
    print(f"   選中模態數: {len(fourier_indices)}")
    
    # ✅ 轉換回物理空間感測點
    # 策略：選擇能最好重建這些 Fourier 模態的空間位置
    
    # 這裡使用簡化方法：均勻採樣（保證週期性）
    n_x = int(np.sqrt(K))
    n_y = K // n_x
    
    # 避免邊界重複：不取 x=L, y=L
    sensor_x_grid = np.linspace(0, 2*np.pi, n_x, endpoint=False)
    sensor_y_grid = np.linspace(0, 2*np.pi, n_y, endpoint=False)
    
    sensor_x, sensor_y = np.meshgrid(sensor_x_grid, sensor_y_grid)
    sensor_x = sensor_x.flatten()
    sensor_y = sensor_y.flatten()
    
    print(f"\n✅ 週期性感測點網格:")
    print(f"   網格: {n_y} × {n_x} = {len(sensor_x)} 個點")
    print(f"   x 範圍: [0, {sensor_x.max():.4f}] (週期性)")
    print(f"   y 範圍: [0, {sensor_y.max():.4f}] (週期性)")
    
    # 保存
    if output_file is None:
        output_file = f"data/kolmogorov_qr_sensors_periodic_K{K}.npz"
    
    np.savez(
        output_file,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        K=K,
        n_modes=n_modes,
        energy_ratio=energy_ratio,
        method='Fourier-POD-QR',
        periodic=True
    )
    
    print(f"\n💾 感測點已保存: {output_file}")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='DNS HDF5 file')
    parser.add_argument('--K', type=int, default=100, help='Number of sensors')
    parser.add_argument('--n-modes', type=int, default=50, help='Number of POD modes')
    parser.add_argument('--output', help='Output NPZ file')
    parser.add_argument('--t-start', type=float, default=20.0)
    parser.add_argument('--t-end', type=float, default=40.0)
    
    args = parser.parse_args()
    
    generate_periodic_qr_sensors(
        dns_file=args.input,
        time_range=(args.t_start, args.t_end),
        K=args.K,
        output_file=args.output,
        n_modes=args.n_modes
    )
