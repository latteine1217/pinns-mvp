#!/usr/bin/env python3
"""
Channel Flow QR-Pivot 感測點生成器（徹底重構版本）
針對 JHTDB Channel Flow Re_tau=1000 設計

核心理念：
- 純 QR-Pivot 選點，無空間過濾
- 完全依賴週期邊界處理控制接縫聚集
- 簡化邏輯，移除回補機制

特點：
1. 正確處理 x 週期性邊界（2D: x週期，3D: x/z週期）
2. 使用循環索引避免接縫聚集
3. 支援 2D (x-y平面) 或 3D (x-y-z體積) 選點
4. 通過 seam_weight/max_seam_fraction 控制接縫比例
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from pinnx.sensors.qr_pivot import QRPivotSelector
import argparse
import logging
from typing import Optional, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def compute_velocity_derivatives_2d(u, v, w, dx, dy, Lx, Ly, periodic_x=True):
    """
    計算 2D 速度場的無量綱化梯度與速度梯度張量特徵值
    
    Args:
        u, v, w: [nx, ny] 速度分量
        dx, dy: 網格間距
        Lx, Ly: 域長度（用於無量綱化）
        periodic_x: x 方向是否週期
    
    Returns:
        dict: 包含渦度、速度梯度張量特徵值的字典
    """
    # 使用中心差分計算梯度
    # ∂/∂x
    if periodic_x:
        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
        dwdx = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dx)
    else:
        dudx = np.gradient(u, dx, axis=0)
        dvdx = np.gradient(v, dx, axis=0)
        dwdx = np.gradient(w, dx, axis=0)
    
    # ∂/∂y (非週期)
    dudy = np.gradient(u, dy, axis=1)
    dvdy = np.gradient(v, dy, axis=1)
    dwdy = np.gradient(w, dy, axis=1)
    
    # 🔧 無量綱化：乘以特徵長度，消除網格間距差異
    dudx_norm = dudx * Lx
    dudy_norm = dudy * Ly
    dvdx_norm = dvdx * Lx
    dvdy_norm = dvdy * Ly
    dwdx_norm = dwdx * Lx
    dwdy_norm = dwdy * Ly
    
    # 渦度（無量綱化）
    omega_z_norm = dvdx_norm - dudy_norm
    
    # 🔧 速度梯度張量特徵值（2D：僅考慮 x-y 平面）
    # ∇u = [[∂u/∂x, ∂u/∂y],
    #        [∂v/∂x, ∂v/∂y]]
    nx, ny = u.shape
    grad_u_eigenvalues = np.zeros((nx, ny, 2))  # 2×2 矩陣有 2 個特徵值
    
    for i in range(nx):
        for j in range(ny):
            grad_u_matrix = np.array([
                [dudx_norm[i, j], dudy_norm[i, j]],
                [dvdx_norm[i, j], dvdy_norm[i, j]]
            ])
            eigenvalues = np.linalg.eigvals(grad_u_matrix)
            # 排序特徵值（從大到小）
            grad_u_eigenvalues[i, j, :] = np.sort(eigenvalues.real)[::-1]
    
    return {
        'dudx': dudx_norm, 'dudy': dudy_norm,
        'dvdx': dvdx_norm, 'dvdy': dvdy_norm,
        'dwdx': dwdx_norm, 'dwdy': dwdy_norm,
        'omega_z': omega_z_norm,
        'grad_u_eig1': grad_u_eigenvalues[:, :, 0],  # 第一特徵值（最大）
        'grad_u_eig2': grad_u_eigenvalues[:, :, 1]   # 第二特徵值
    }

def compute_velocity_derivatives_3d(u, v, w, dx, dy, dz, Lx, Ly, Lz, periodic_x=True, periodic_z=True):
    """
    計算 3D 速度場的無量綱化梯度與速度梯度張量特徵值
    
    Args:
        u, v, w: [nx, ny, nz] 速度分量
        dx, dy, dz: 網格間距
        Lx, Ly, Lz: 域長度（用於無量綱化）
        periodic_x, periodic_z: x/z 方向是否週期
    
    Returns:
        dict: 包含渦度向量、速度梯度張量特徵值的字典
    """
    # ∂/∂x
    if periodic_x:
        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
        dwdx = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dx)
    else:
        dudx = np.gradient(u, dx, axis=0)
        dvdx = np.gradient(v, dx, axis=0)
        dwdx = np.gradient(w, dx, axis=0)
    
    # ∂/∂y (非週期)
    dudy = np.gradient(u, dy, axis=1)
    dvdy = np.gradient(v, dy, axis=1)
    dwdy = np.gradient(w, dy, axis=1)
    
    # ∂/∂z
    if periodic_z:
        dudz = (np.roll(u, -1, axis=2) - np.roll(u, 1, axis=2)) / (2 * dz)
        dvdz = (np.roll(v, -1, axis=2) - np.roll(v, 1, axis=2)) / (2 * dz)
        dwdz = (np.roll(w, -1, axis=2) - np.roll(w, 1, axis=2)) / (2 * dz)
    else:
        dudz = np.gradient(u, dz, axis=2)
        dvdz = np.gradient(v, dz, axis=2)
        dwdz = np.gradient(w, dz, axis=2)
    
    # 🔧 無量綱化：乘以特徵長度
    dudx_norm = dudx * Lx
    dudy_norm = dudy * Ly
    dudz_norm = dudz * Lz
    dvdx_norm = dvdx * Lx
    dvdy_norm = dvdy * Ly
    dvdz_norm = dvdz * Lz
    dwdx_norm = dwdx * Lx
    dwdy_norm = dwdy * Ly
    dwdz_norm = dwdz * Lz
    
    # 渦度向量（無量綱化）
    omega_x_norm = dwdy_norm - dvdz_norm
    omega_y_norm = dudz_norm - dwdx_norm
    omega_z_norm = dvdx_norm - dudy_norm
    
    # 🔧 速度梯度張量特徵值（3D）
    # ∇u = [[∂u/∂x, ∂u/∂y, ∂u/∂z],
    #        [∂v/∂x, ∂v/∂y, ∂v/∂z],
    #        [∂w/∂x, ∂w/∂y, ∂w/∂z]]
    nx, ny, nz = u.shape
    grad_u_eigenvalues = np.zeros((nx, ny, nz, 3))  # 3×3 矩陣有 3 個特徵值
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                grad_u_matrix = np.array([
                    [dudx_norm[i, j, k], dudy_norm[i, j, k], dudz_norm[i, j, k]],
                    [dvdx_norm[i, j, k], dvdy_norm[i, j, k], dvdz_norm[i, j, k]],
                    [dwdx_norm[i, j, k], dwdy_norm[i, j, k], dwdz_norm[i, j, k]]
                ])
                eigenvalues = np.linalg.eigvals(grad_u_matrix)
                # 排序特徵值（從大到小）
                grad_u_eigenvalues[i, j, k, :] = np.sort(eigenvalues.real)[::-1]
    
    return {
        'dudx': dudx_norm, 'dudy': dudy_norm, 'dudz': dudz_norm,
        'dvdx': dvdx_norm, 'dvdy': dvdy_norm, 'dvdz': dvdz_norm,
        'dwdx': dwdx_norm, 'dwdy': dwdy_norm, 'dwdz': dwdz_norm,
        'omega_x': omega_x_norm, 'omega_y': omega_y_norm, 'omega_z': omega_z_norm,
        'grad_u_eig1': grad_u_eigenvalues[:, :, :, 0],  # 第一特徵值（最大）
        'grad_u_eig2': grad_u_eigenvalues[:, :, :, 1],  # 第二特徵值
        'grad_u_eig3': grad_u_eigenvalues[:, :, :, 2]   # 第三特徵值（最小）
    }

def generate_channel_flow_sensors(
    data_file: str,
    K: int = 100,
    output_file: Optional[str] = None,
    n_wrap_layers: int = 2,
    seam_weight: float = 0.5,  # 降低接縫權重避免聚集
    seam_width_fraction: float = 0.05,
    max_seam_fraction: float = 0.1,
    n_snapshots: int = 1  # 使用的時間快照數（cutout 通常是單快照）
):
    """
    為 Channel Flow 生成 QR-Pivot 感測點（徹底重構版本）
    
    策略：純 QR-Pivot + 週期邊界處理，無空間過濾
    
    自動檢測數據類型：
    - 2D cutout: (nx, ny) 平面，只有 x 週期
    - 3D cutout: (nx, ny, nz) 體積，x/z 週期
    
    核心改進：
    1. 移除空間過濾邏輯（避免過度激進的點數損失）
    2. 移除回補邏輯（QR-Pivot 直接選擇 K 個點）
    3. 完全依賴 seam_weight 控制接縫聚集
    4. 依賴循環索引處理週期邊界
    """
    
    print("=" * 70)
    print("🔬 Channel Flow QR-Pivot 感測點生成")
    print("=" * 70)
    
    # 1. 讀取並檢測數據類型
    # ------------------------------------------------------
    print(f"\n📊 載入數據: {data_file}")
    data = np.load(data_file)
    
    # 提取座標
    x = data['x']  # [nx]
    y = data['y']  # [ny]
    has_z = 'z' in data
    
    # 提取速度場（可能是 flatten 或已經是 3D）
    u = data['u']
    v = data['v']
    w = data['w']
    
    # 檢查是否需要 reshape
    if 'grid_shape' in data:
        # flatten 格式，需要 reshape
        grid_shape_stored = data['grid_shape']
        if has_z:
            nx, ny, nz = grid_shape_stored
            z = data['z']  # [nz]
            ndim = 3
            print(f"   數據類型: 3D (x-y-z 體積, flatten 格式)")
            print(f"   原始形狀: u={u.shape}, v={v.shape}, w={w.shape}")
            # Reshape 回 3D
            u = u.reshape((nx, ny, nz), order='C')
            v = v.reshape((nx, ny, nz), order='C')
            w = w.reshape((nx, ny, nz), order='C')
            print(f"   Reshape 後: u={u.shape}, v={v.shape}, w={w.shape}")
        else:
            nx, ny = grid_shape_stored[:2]
            z = None
            ndim = 2
            nz = 1
            print(f"   數據類型: 2D (x-y 平面, flatten 格式)")
            print(f"   原始形狀: u={u.shape}, v={v.shape}, w={w.shape}")
            # Reshape 回 2D
            u = u.reshape((nx, ny), order='C')
            v = v.reshape((nx, ny), order='C')
            w = w.reshape((nx, ny), order='C')
            print(f"   Reshape 後: u={u.shape}, v={v.shape}, w={w.shape}")
    else:
        # 已經是正確形狀
        if has_z:
            z = data['z']  # [nz]
            ndim = 3
            nz = len(z)
            print(f"   數據類型: 3D (x-y-z 體積)")
        else:
            z = None
            ndim = 2
            nz = 1
            print(f"   數據類型: 2D (x-y 平面)")
        print(f"   u 形狀: {u.shape}")
        print(f"   v 形狀: {v.shape}")
        print(f"   w 形狀: {w.shape}")
        nx, ny = len(x), len(y)
    
    # 2. 構建座標網格和數據矩陣
    # ------------------------------------------------------
    if ndim == 2:
        # 2D 模式
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords = np.column_stack([
            xx.flatten(),
            yy.flatten()
        ])  # [nx*ny, 2]
        
        grid_shape = (nx, ny)
        n_locations = nx * ny
        
        # 域範圍與網格間距
        Lx = x[-1] - x[0] + (x[1] - x[0])  # 週期：加網格間距
        Ly = y[-1] - y[0]  # 非週期
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        domain_lengths_arr = [Lx, Ly]
        
        # 週期軸：只有 x
        periodic_axes = [0]
        
        print(f"\n   網格: {nx} × {ny}")
        print(f"   域範圍: Lx={Lx:.4f}, Ly={Ly:.4f}")
        print(f"   網格間距: dx={dx:.6f}, dy={dy:.6f}")
        print(f"   週期軸: x (軸 0)")
        
        # 計算速度梯度、渦度、散度（無量綱化）
        print(f"\n🔧 特徵工程：計算無量綱化導數場...")
        derivs = compute_velocity_derivatives_2d(u, v, w, dx, dy, Lx, Ly, periodic_x=True)
        
        # 構建增強數據矩陣: [n_locations, 6]
        # 策略：速度 + 渦度 + 速度梯度張量特徵值（包含流動拓撲信息）
        u_flat = u.flatten()
        v_flat = v.flatten()
        w_flat = w.flatten()
        
        data_matrix = np.column_stack([
            u_flat, v_flat, w_flat,                          # 原始速度
            derivs['omega_z'].flatten(),                     # 渦度 z
            derivs['grad_u_eig1'].flatten(),                 # ∇u 第一特徵值
            derivs['grad_u_eig2'].flatten()                  # ∇u 第二特徵值
        ])
        
        print(f"   特徵維度: 6 (速度 3 + 渦度 1 + ∇u特徵值 2)")
        print(f"   ✅ 使用無量綱化梯度 (∂u/∂x × Lx) 消除網格間距差異")
        print(f"   ✅ 速度梯度張量特徵值反映流動拓撲（應變/旋轉）")
        
    else:
        # 3D 模式
        if z is None:
            raise ValueError("3D 模式但 z 座標不存在")
        
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        coords = np.column_stack([
            xx.flatten(),
            yy.flatten(),
            zz.flatten()
        ])  # [nx*ny*nz, 3]
        
        grid_shape = (nx, ny, nz)
        n_locations = nx * ny * nz
        
        # 域範圍與網格間距
        Lx = x[-1] - x[0] + (x[1] - x[0])
        Ly = y[-1] - y[0]
        Lz = z[-1] - z[0] + (z[1] - z[0])
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        domain_lengths_arr = [Lx, Ly, Lz]
        
        # 週期軸：x, z
        periodic_axes = [0, 2]
        
        print(f"\n   網格: {nx} × {ny} × {nz}")
        print(f"   域範圍: Lx={Lx:.4f}, Ly={Ly:.4f}, Lz={Lz:.4f}")
        print(f"   網格間距: dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}")
        print(f"   週期軸: x, z (軸 0, 2)")
        
        # 計算速度梯度、渦度、散度（無量綱化）
        print(f"\n🔧 特徵工程：計算無量綱化導數場...")
        derivs = compute_velocity_derivatives_3d(u, v, w, dx, dy, dz, Lx, Ly, Lz,
                                                  periodic_x=True, periodic_z=True)
        
        # 構建增強數據矩陣: [n_locations, 9]
        # 策略：速度 + 渦度向量 + 速度梯度張量特徵值（包含完整流動拓撲信息）
        u_flat = u.flatten()
        v_flat = v.flatten()
        w_flat = w.flatten()
        
        data_matrix = np.column_stack([
            u_flat, v_flat, w_flat,                          # 原始速度
            derivs['omega_x'].flatten(), derivs['omega_y'].flatten(), derivs['omega_z'].flatten(),
            derivs['grad_u_eig1'].flatten(),                 # ∇u 第一特徵值
            derivs['grad_u_eig2'].flatten(),                 # ∇u 第二特徵值
            derivs['grad_u_eig3'].flatten()                  # ∇u 第三特徵值
        ])
        
        print(f"   特徵維度: 9 (速度 3 + 渦度 3 + ∇u特徵值 3)")
        print(f"   ✅ 使用無量綱化梯度消除網格間距差異")
        print(f"   ✅ 速度梯度張量特徵值反映完整流動拓撲（應變/旋轉）")
    
    print(f"   空間點數: {n_locations}")
    print(f"   數據矩陣: {data_matrix.shape}")
    
    # 標準化
    mean_val = data_matrix.mean(axis=0)
    std_val = data_matrix.std(axis=0) + 1e-8
    data_matrix_norm = (data_matrix - mean_val) / std_val
    print(f"   標準化: Z-Score 已應用")
    
    # 3. QR-Pivot 選擇（直接選 K 個點，無過採樣）
    # ------------------------------------------------------
    print(f"\n🔄 QR-Pivot 選擇 (循環索引 + 週期邊界處理):")
    print(f"   目標點數: {K}")
    print(f"   環繞層: {n_wrap_layers}")
    print(f"   接縫權重: {seam_weight}")
    print(f"   接縫寬度: {seam_width_fraction*100:.1f}%")
    print(f"   接縫比例上限: {max_seam_fraction*100:.1f}%")
    print(f"   策略: 純 QR-Pivot（無空間過濾）")
    
    selector = QRPivotSelector(
        use_circular_indexing=True,
        n_wrap_layers=n_wrap_layers,
        seam_weight=seam_weight,
        seam_width_fraction=seam_width_fraction,
        max_seam_fraction=max_seam_fraction,
        mode='column',
        pivoting=True,
        regularization=1e-12
    )
    
    # 構建 domain_lengths 字典
    domain_lengths = {i: domain_lengths_arr[i] for i in range(ndim)}
    
    print("\n🚀 執行 QR-Pivot 選擇...")
    final_indices, metrics = selector.select_sensors(
        data_matrix=data_matrix_norm,
        n_sensors=K,  # 直接選 K 個點
        coords=coords,
        grid_shape=grid_shape,
        periodic_axes=periodic_axes,
        domain_lengths=domain_lengths
    )
    
    print(f"\n📊 QR 指標:")
    print(f"   條件數: {metrics.get('condition_number', 'N/A'):.2f}")
    print(f"   能量比例: {metrics.get('energy_ratio', 'N/A'):.4f}")
    if 'seam_selected_count' in metrics:
        print(f"   接縫點數: {metrics['seam_selected_count']} "
              f"({metrics.get('seam_selected_ratio', 0)*100:.1f}%)")
    
    final_indices = np.array(final_indices)
    selected_coords = coords[final_indices]
    
    print(f"\n✅ 選擇完成:")
    print(f"   最終點數: {len(final_indices)}")
    
    # 5. 統計分析
    # ------------------------------------------------------
    print(f"\n📈 分佈統計:")
    
    # 各軸範圍
    axis_names = ['x', 'y', 'z'] if ndim == 3 else ['x', 'y']
    for i, axis_name in enumerate(axis_names):
        vals = selected_coords[:, i]
        print(f"   {axis_name}: [{vals.min():.4f}, {vals.max():.4f}] "
              f"(範圍 {vals.max() - vals.min():.4f})")
    
    # 最近鄰距離
    if len(final_indices) > 1:
        # 簡化：使用歐式距離
        from scipy.spatial.distance import cdist
        dists = cdist(selected_coords, selected_coords)
        np.fill_diagonal(dists, np.inf)
        min_dists = dists.min(axis=1)
        
        print(f"   最近鄰距離: min={min_dists.min():.4f}, "
              f"mean={min_dists.mean():.4f}, max={min_dists.max():.4f}")
    
    # 6. 保存結果
    # ------------------------------------------------------
    if output_file is None:
        suffix = f"_{ndim}d"
        output_file = f"data/jhtdb/channel_flow_re1000/sensors_K{K}_qr_pivot{suffix}.npz"
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 轉換 metrics 為基本類型
    metrics_clean = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                     for k, v in metrics.items() if isinstance(v, (int, float, np.number))}
    
    # 保存數據
    save_dict = {
        'sensor_indices': final_indices,
        'sensor_x': selected_coords[:, 0],
        'sensor_y': selected_coords[:, 1],
        'K': K,
        'ndim': ndim,
        'grid_shape': np.array(grid_shape),
        'periodic_axes': np.array(periodic_axes),
        'domain_lengths': np.array(domain_lengths_arr),
        'condition_number': metrics_clean.get('condition_number', 0.0),
        'energy_ratio': metrics_clean.get('energy_ratio', 0.0),
        'method': 'QR-Pivot-Circular-Pure',
        'seam_weight': seam_weight,
        'source_file': str(data_file)
    }
    
    # 3D 模式添加 z 座標
    if ndim == 3:
        save_dict['sensor_z'] = selected_coords[:, 2]
    
    np.savez(output_file, **save_dict)
    
    print(f"\n💾 感測點已保存: {output_file}")
    print("=" * 70)
    
    return final_indices, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="為 Channel Flow 生成 QR-Pivot 感測點"
    )
    parser.add_argument('--input', required=True, 
                       help='JHTDB cutout NPZ 文件')
    parser.add_argument('--K', type=int, default=100, 
                       help='感測點數量')
    parser.add_argument('--output', 
                       help='輸出 NPZ 文件路徑')
    parser.add_argument('--wrap-layers', type=int, default=2,
                       help='循環索引環繞層數')
    parser.add_argument('--seam-weight', type=float, default=0.5,
                       help='接縫權重 (<1 降低接縫)')
    parser.add_argument('--seam-width-fraction', type=float, default=0.05,
                       help='接縫寬度比例')
    parser.add_argument('--max-seam-fraction', type=float, default=0.1,
                       help='接縫感測點最大比例')
    
    args = parser.parse_args()
    
    generate_channel_flow_sensors(
        data_file=args.input,
        K=args.K,
        output_file=args.output,
        n_wrap_layers=args.wrap_layers,
        seam_weight=args.seam_weight,
        seam_width_fraction=args.seam_width_fraction,
        max_seam_fraction=args.max_seam_fraction
    )
