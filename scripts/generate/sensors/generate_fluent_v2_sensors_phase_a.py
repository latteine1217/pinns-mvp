#!/usr/bin/env python3
"""
從 Fluent V2 HDF5 數據生成 Phase A QR-Pivot Sensors
使用 18 個物理特徵進行增強的感測點選擇

特點：
1. 直接讀取 Fluent HDF5 格式 (FFF-Setup-Output.dat_2.h5)
2. Phase A 特徵集: 18 個 (minimal 10 + advanced 8)
3. 自動計算湍流特徵 (P_k, y+, b_ij, Re_t, epsilon, enstrophy)
4. 可視化感測點分佈與特徵重要性

作者: PINNs Channel Flow Team
日期: 2025-12-18
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import h5py
import numpy as np
from pinnx.sensors.qr_pivot import QRPivotSelector
import argparse
import logging
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非互動式後端

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_fluent_hdf5(filepath: str, slice_type: str = '2d_center', case_file: str = None):
    """
    從 Fluent HDF5 文件讀取數據（3D → 2D slice）
    
    Args:
        filepath: Fluent .dat.h5 文件路徑
        slice_type: '2d_center' (取 Z 方向中心切面)
        case_file: Fluent .cas.h5 文件路徑（用於獲取網格座標）
    
    Returns:
        coords: (x, y) 座標
        fields: 包含 u, v, w, p, k, omega, mu_t 的字典
        metadata: 網格與物理參數
    """
    logger.info(f"📂 Loading Fluent HDF5: {filepath}")
    logger.info(f"   Slice type: {slice_type}")
    
    # 自動查找 case file
    if case_file is None:
        case_file = str(filepath).replace('.dat_2.h5', '.cas_2.h5').replace('.dat.h5', '.cas.h5')
    
    # Step 1: 讀取網格座標
    with h5py.File(case_file, 'r') as f:
        coords_3d = np.array(f['meshes/1/nodes/coords/3'])  # (N_nodes, 3)
        logger.info(f"   Total nodes: {coords_3d.shape[0]:,}")
        
        # 找出網格結構
        unique_x = np.unique(np.round(coords_3d[:,0], 8))
        unique_y = np.unique(np.round(coords_3d[:,1], 8))
        unique_z = np.unique(np.round(coords_3d[:,2], 8))
        
        nx_node, ny_node, nz_node = len(unique_x), len(unique_y), len(unique_z)
        logger.info(f"   Node grid: {nx_node} × {ny_node} × {nz_node} = {nx_node*ny_node*nz_node:,}")
    
    # Step 2: 讀取 cell-centered 數據
    with h5py.File(filepath, 'r') as f:
        cells = f['results/1/phase-1/cells']
        
        # 讀取所有變數 (1D array)
        u = np.array(cells['SV_U/1'])
        v = np.array(cells['SV_V/1'])
        w = np.array(cells['SV_W/1'])
        p = np.array(cells['SV_P/1'])
        k = np.array(cells['SV_K/1'])
        omega = np.array(cells['SV_O/1'])
        mu_t = np.array(cells['SV_MU_T/1'])
        mu_lam = np.array(cells['SV_MU_LAM/1'])
        rho = np.array(cells['SV_DENSITY/1'])
        wall_dist = np.array(cells['SV_WALL_DIST/1'])
        
        n_cells = len(u)
        logger.info(f"   Total cells: {n_cells:,}")
        
        # Cell grid 比 node grid 少一層
        nx, ny, nz = nx_node - 1, ny_node - 1, nz_node - 1
        logger.info(f"   Inferred cell grid: {nx} × {ny} × {nz} = {nx*ny*nz:,}")
        
        if nx * ny * nz != n_cells:
            logger.error(f"   ❌ Grid mismatch: {nx*ny*nz} ≠ {n_cells}")
            raise ValueError("Cannot reconstruct 3D grid structure")
        
        # Step 3: Reshape 1D → 3D (假設 Fortran order: Z varies fastest)
        try:
            u_3d = u.reshape((nx, ny, nz), order='F')
            v_3d = v.reshape((nx, ny, nz), order='F')
            w_3d = w.reshape((nx, ny, nz), order='F')
            p_3d = p.reshape((nx, ny, nz), order='F')
            k_3d = k.reshape((nx, ny, nz), order='F')
            omega_3d = omega.reshape((nx, ny, nz), order='F')
            mu_t_3d = mu_t.reshape((nx, ny, nz), order='F')
            wall_dist_3d = wall_dist.reshape((nx, ny, nz), order='F')
            logger.info(f"   ✅ Reshaped to 3D successfully")
        except Exception as e:
            logger.error(f"   ❌ Reshape failed: {e}")
            raise
        
        # Step 4: 取 Z 方向中心切面 (X-Y plane)
        iz_center = nz // 2
        logger.info(f"   Taking Z-slice at index {iz_center}/{nz}")
        
        u_2d = u_3d[:, :, iz_center]
        v_2d = v_3d[:, :, iz_center]
        w_2d = w_3d[:, :, iz_center]
        p_2d = p_3d[:, :, iz_center]
        k_2d = k_3d[:, :, iz_center]
        omega_2d = omega_3d[:, :, iz_center]
        mu_t_2d = mu_t_3d[:, :, iz_center]
        wall_dist_2d = wall_dist_3d[:, :, iz_center]
        
        # Step 5: 生成 2D 座標
        x = unique_x[:nx]  # Use actual X coordinates from grid
        y = unique_y[:ny]  # Use actual Y coordinates from grid
        
        Lx = float(x[-1] - x[0])  # 實際流向長度
        Ly = float(y[-1] - y[0])  # 實際壁向長度
        
        logger.info(f"   2D slice: {nx} × {ny}")
        logger.info(f"   X: [{x[0]:.4f}, {x[-1]:.4f}], nx={nx}")
        logger.info(f"   Y: [{y[0]:.4f}, {y[-1]:.4f}], ny={ny}")
        
        fields = {
            'u': u_2d,
            'v': v_2d,
            'w': w_2d,
            'p': p_2d,
            'k': k_2d,
            'omega': omega_2d,
            'mu_t': mu_t_2d,
            'wall_dist': wall_dist_2d,
            'mu_lam': float(mu_lam[0]),  # 常數
            'rho': float(rho[0])          # 常數
        }
        
        metadata = {
            'nx': nx,
            'ny': ny,
            'Lx': Lx,
            'Ly': Ly,
            'dx': Lx / nx if nx > 1 else 0.0,
            'dy': float(np.mean(np.diff(y))) if len(y) > 1 else 0.0,
            'nu': float(mu_lam[0] / rho[0]),  # ν = μ / ρ
            'Re_tau_estimate': 1000.0  # 假設 Re_τ ≈ 1000 (從配置/文獻)
        }
        
        logger.info(f"   Domain: Lx={metadata['Lx']:.2f}, Ly={metadata['Ly']:.2f}")
        logger.info(f"   Grid spacing: dx={metadata['dx']:.6f}, dy={metadata['dy']:.6f}")
        logger.info(f"   Viscosity: ν={metadata['nu']:.2e}")
        
    return (x, y), fields, metadata


def compute_gradients_2d(
    field: np.ndarray,
    dx: float,
    dy: float,
    periodic_x: bool = True
) -> tuple:
    """
    計算 2D 場的梯度
    
    Returns:
        (dfield_dx, dfield_dy)
    """
    if periodic_x:
        dfield_dx = (np.roll(field, -1, axis=0) - np.roll(field, 1, axis=0)) / (2 * dx)
    else:
        dfield_dx = np.gradient(field, dx, axis=0)
    
    dfield_dy = np.gradient(field, dy, axis=1)
    
    return dfield_dx, dfield_dy


def compute_phase_a_features_fluent(
    fields: dict,
    metadata: dict,
    y_coords: np.ndarray
) -> dict:
    """
    計算 Phase A 的 18 個特徵
    
    Phase A 特徵 (18 total):
    Baseline (10):
      1-4. u, v, w, p
      5. dudy (壁面剪切)
      6. omega_z (渦度)
      7. k (TKE)
      8. tau_uv (Reynolds stress)
      9-10. grad_u_eig1, grad_u_eig2 (velocity gradient eigenvalues)
    
    Advanced (8):
      11. P_k (TKE production)
      12. y_plus (wall distance+)
      13-15. b_11, b_22, b_12 (anisotropy tensor)
      16. Re_t (turbulent Reynolds number)
      17. epsilon (dissipation rate)
      18. enstrophy
    """
    logger.info("🔧 Computing Phase A features (18 total)...")
    
    u, v, w, p = fields['u'], fields['v'], fields['w'], fields['p']
    k, omega, mu_t = fields['k'], fields['omega'], fields['mu_t']
    
    nx, ny = u.shape
    dx, dy = metadata['dx'], metadata['dy']
    nu = metadata['nu']
    Re_tau = metadata['Re_tau_estimate']
    rho = fields['rho']
    
    # ===== 速度梯度 =====
    dudx, dudy = compute_gradients_2d(u, dx, dy, periodic_x=True)
    dvdx, dvdy = compute_gradients_2d(v, dx, dy, periodic_x=True)
    dwdx, dwdy = compute_gradients_2d(w, dx, dy, periodic_x=True)
    
    # ===== Baseline Features (10) =====
    
    # 1. 渦度 (omega_z)
    omega_z = dvdx - dudy
    
    # 2. Reynolds stress (Boussinesq approximation)
    # tau_uv = μ_t * (∂u/∂y + ∂v/∂x)
    tau_uv = mu_t * (dudy + dvdx)
    
    # 也計算 normal stresses (用於 anisotropy tensor)
    iso_term = (2.0 / 3.0) * rho * k
    tau_uu = 2.0 * mu_t * dudx - iso_term
    tau_vv = 2.0 * mu_t * dvdy - iso_term
    tau_ww = -2.0 * mu_t * (dudx + dvdy) - iso_term
    
    # 3. Velocity gradient eigenvalues
    grad_u_eigenvalues = np.zeros((nx, ny, 2))
    for i in range(nx):
        for j in range(ny):
            grad_u_matrix = np.array([
                [dudx[i, j], dudy[i, j]],
                [dvdx[i, j], dvdy[i, j]]
            ])
            eigenvalues = np.linalg.eigvals(grad_u_matrix)
            grad_u_eigenvalues[i, j, :] = np.sort(eigenvalues.real)[::-1]
    
    grad_u_eig1 = grad_u_eigenvalues[:, :, 0]
    grad_u_eig2 = grad_u_eigenvalues[:, :, 1]
    
    # ===== Advanced Features (8) =====
    
    # 1. y_plus (wall distance in wall units)
    u_tau = Re_tau * nu / 1.0  # 通道半高 = 1.0
    y_plus = np.tile(y_coords[np.newaxis, :], (nx, 1)) * u_tau / nu
    
    # 2. TKE Production: P_k = -tau_ij * S_ij
    S11 = dudx
    S22 = dvdy
    S12 = 0.5 * (dudy + dvdx)
    
    P_k = -(tau_uu * S11 + tau_vv * S22 + 2 * tau_uv * S12)
    
    # 3. Anisotropy tensor: b_ij = tau_ij / (2k) - (1/3) * δ_ij
    k_safe = np.maximum(k, 1e-10)
    b_11 = tau_uu / (2 * k_safe) - 1.0/3.0
    b_22 = tau_vv / (2 * k_safe) - 1.0/3.0
    b_12 = tau_uv / (2 * k_safe)
    
    # 4. Dissipation rate: ε ≈ 2ν * S_ij * S_ij (加上 k-omega 模型項)
    # 對於 k-omega SST: ε = β* k ω (更穩定的估算)
    beta_star = 0.09  # k-omega SST 模型常數
    epsilon = beta_star * k * omega
    epsilon = np.maximum(epsilon, 1e-8)  # 更強的下界
    
    # 5. Turbulent Reynolds number: Re_t = k^2 / (ν * ε)
    # 加上上界限制避免極端值
    Re_t = k_safe**2 / (nu * epsilon)
    Re_t = np.clip(Re_t, 1.0, 1e6)  # 限制在合理範圍 [1, 10^6]
    
    # 6. Enstrophy: 渦度"動能" = 0.5 * ω^2
    enstrophy = 0.5 * omega_z**2
    
    logger.info("   ✅ Baseline features (9): u, v, p, dudy, omega_z, k, tau_uv, eig1, eig2")
    logger.info("   ✅ Advanced features (8): P_k, y+, b_11, b_22, b_12, Re_t, epsilon, enstrophy")
    logger.info("   ⚠️  Excluded 'w' (spanwise velocity = 0 in 2D slice)")
    
    # 統計摘要
    logger.info(f"\n   📊 Feature statistics:")
    logger.info(f"      u: [{u.min():.3f}, {u.max():.3f}], mean={u.mean():.3f}")
    logger.info(f"      k: [{k.min():.2e}, {k.max():.2e}], mean={k.mean():.2e}")
    logger.info(f"      y+: [{y_plus.min():.1f}, {y_plus.max():.1f}], mean={y_plus.mean():.1f}")
    logger.info(f"      P_k: [{P_k.min():.2e}, {P_k.max():.2e}], mean={P_k.mean():.2e}")
    logger.info(f"      Re_t: [{Re_t.min():.1f}, {Re_t.max():.1f}], mean={Re_t.mean():.1f}")
    
    return {
        # Baseline (9) - EXCLUDED 'w' for 2D slice
        'u': u, 'v': v, 'p': p,
        'dudy': dudy,
        'omega_z': omega_z,
        'k': k,
        'tau_uv': tau_uv,
        'grad_u_eig1': grad_u_eig1,
        'grad_u_eig2': grad_u_eig2,
        
        # Advanced (8)
        'P_k': P_k,
        'y_plus': y_plus,
        'b_11': b_11,
        'b_22': b_22,
        'b_12': b_12,
        'Re_t': Re_t,
        'epsilon': epsilon,
        'enstrophy': enstrophy
    }


def build_phase_a_data_matrix(feature_dict: dict) -> tuple:
    """
    建立 Phase A 資料矩陣 (17 features for 2D slice)
    
    Note: 'w' excluded as it's zero in Z-center slice
    
    Returns:
        data_matrix: [N_points, 17] 標準化矩陣
        feature_names: 特徵名稱列表
    """
    feature_names = [
        # Baseline (9) - EXCLUDED 'w'
        'u', 'v', 'p',
        'dudy',
        'omega_z',
        'k',
        'tau_uv',
        'grad_u_eig1', 'grad_u_eig2',
        
        # Advanced (8)
        'P_k',
        'y_plus',
        'b_11', 'b_22', 'b_12',
        'Re_t',
        'epsilon',
        'enstrophy'
    ]
    
    features = []
    for name in feature_names:
        if name not in feature_dict:
            raise KeyError(f"Feature '{name}' not found in feature_dict")
        features.append(feature_dict[name].flatten())
    
    data_matrix = np.column_stack(features)
    
    # 標準化
    mean = data_matrix.mean(axis=0)
    std = data_matrix.std(axis=0) + 1e-10
    data_matrix = (data_matrix - mean) / std
    
    # 檢查矩陣品質
    rank = np.linalg.matrix_rank(data_matrix)
    cond = np.linalg.cond(data_matrix)
    
    logger.info(f"\n📈 Data matrix quality:")
    logger.info(f"   Shape: {data_matrix.shape}")
    logger.info(f"   Rank: {rank} / {min(data_matrix.shape)}")
    logger.info(f"   Condition number: {cond:.2e}")
    
    return data_matrix, feature_names


def visualize_sensors(
    coords: np.ndarray,
    sensor_indices: np.ndarray,
    fields: dict,
    output_dir: str,
    grid_shape: tuple
):
    """
    可視化感測點分佈與特徵場
    """
    logger.info(f"\n📊 Generating visualizations...")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    nx, ny = grid_shape
    sensor_coords = coords[sensor_indices]
    
    # 1. 感測點分佈 + TKE 場
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (a) TKE + sensors
    ax = axes[0, 0]
    k_2d = fields['k']
    im = ax.contourf(k_2d.T, levels=20, cmap='hot')
    ax.scatter(sensor_coords[:, 0] / (k_2d.shape[0] / nx) * nx, 
               sensor_coords[:, 1] / (k_2d.shape[1] / ny) * ny,
               c='cyan', s=20, edgecolors='blue', linewidths=0.5, label='Sensors')
    ax.set_title('(a) TKE + Sensor Locations')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('y (wall-normal)')
    plt.colorbar(im, ax=ax, label='k')
    ax.legend()
    
    # (b) y+ + sensors
    ax = axes[0, 1]
    y_plus = fields['y_plus'] if 'y_plus' in fields else np.zeros_like(k_2d)
    im = ax.contourf(y_plus.T, levels=20, cmap='viridis')
    ax.scatter(sensor_coords[:, 0] / (k_2d.shape[0] / nx) * nx,
               sensor_coords[:, 1] / (k_2d.shape[1] / ny) * ny,
               c='red', s=20, edgecolors='darkred', linewidths=0.5)
    ax.set_title('(b) y+ (Wall Distance)')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('y (wall-normal)')
    plt.colorbar(im, ax=ax, label='y+')
    
    # (c) TKE Production + sensors
    ax = axes[1, 0]
    P_k = fields['P_k'] if 'P_k' in fields else np.zeros_like(k_2d)
    im = ax.contourf(P_k.T, levels=20, cmap='RdBu_r')
    ax.scatter(sensor_coords[:, 0] / (k_2d.shape[0] / nx) * nx,
               sensor_coords[:, 1] / (k_2d.shape[1] / ny) * ny,
               c='lime', s=20, edgecolors='green', linewidths=0.5)
    ax.set_title('(c) TKE Production (P_k)')
    ax.set_xlabel('x (streamwise)')
    ax.set_ylabel('y (wall-normal)')
    plt.colorbar(im, ax=ax, label='P_k')
    
    # (d) Histogram: sensor distribution in y
    ax = axes[1, 1]
    ax.hist(sensor_coords[:, 1], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('(d) Sensor Distribution in Wall-Normal Direction')
    ax.set_xlabel('y (wall-normal)')
    ax.set_ylabel('Count')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'fluent_v2_sensors_phase_a_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"   ✅ Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Phase A QR-Pivot sensors from Fluent V2 HDF5 data"
    )
    
    parser.add_argument(
        '--fluent-h5',
        type=str,
        default='data/lowfi/channel_fluent_raw/FFF-Setup-Output.dat_2.h5',
        help='Fluent HDF5 data file (.dat.h5 or .dat_2.h5)'
    )
    
    parser.add_argument(
        '-K', '--n-sensors',
        type=int,
        default=100,
        help='Number of sensors (default: 100)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output NPZ file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/fluent_v2_sensors',
        help='Output directory for visualizations'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 Fluent V2 → Phase A QR-Pivot Sensor Generation")
    print("=" * 70)
    
    # 1. 載入 Fluent 數據
    (x, y), fields, metadata = load_fluent_hdf5(args.fluent_h5)
    nx, ny = metadata['nx'], metadata['ny']
    
    # 2. 計算 Phase A 特徵
    feature_dict = compute_phase_a_features_fluent(fields, metadata, y)
    
    # 3. 建立資料矩陣
    data_matrix, feature_names = build_phase_a_data_matrix(feature_dict)
    
    # 4. QR-Pivot 選點
    logger.info(f"\n🔄 QR-Pivot Selection (K={args.n_sensors}):")
    
    coords = np.column_stack([
        np.repeat(x, ny),
        np.tile(y, nx)
    ])
    
    selector = QRPivotSelector(
        use_circular_indexing=True,
        n_wrap_layers=2,
        seam_weight=0.5,
        mode='column',
        pivoting=True
    )
    
    final_indices, metrics = selector.select_sensors(
        data_matrix=data_matrix,
        n_sensors=args.n_sensors,
        coords=coords,
        grid_shape=(nx, ny),
        periodic_axes=[0],
        domain_lengths={0: metadata['Lx'], 1: metadata['Ly']}
    )
    
    selected_coords = coords[final_indices]
    
    logger.info(f"\n✅ Selection Complete:")
    logger.info(f"   Selected sensors: {len(final_indices)}")
    logger.info(f"   Condition number: {metrics.get('condition_number', 0):.2e}")
    logger.info(f"   Energy ratio: {metrics.get('energy_ratio', 0):.6f}")
    
    # 5. 儲存結果
    if args.output is None:
        args.output = f"data/jhtdb/channel_flow_re1000/sensors_K{args.n_sensors}_fluent_v2_phase_a.npz"
    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'sensor_points': selected_coords.astype(np.float32),
        'sensor_indices': final_indices,
        'sensor_x': selected_coords[:, 0].astype(np.float32),
        'sensor_y': selected_coords[:, 1].astype(np.float32),
        'K': args.n_sensors,
        'n_features': len(feature_names),
        'feature_names': np.array(feature_names, dtype=object),
        'feature_selection': 'phase_a',
        'condition_number': float(metrics.get('condition_number', 0)),
        'energy_ratio': float(metrics.get('energy_ratio', 0)),
        'matrix_rank': np.linalg.matrix_rank(data_matrix),
        'method': 'QR-Pivot-Fluent-V2-Phase-A',
        'source_file': args.fluent_h5,
        'domain_Lx': metadata['Lx'],
        'domain_Ly': metadata['Ly'],
        'Re_tau_estimate': metadata['Re_tau_estimate'],
        'nu': metadata['nu']
    }
    
    np.savez(args.output, **save_dict)
    logger.info(f"\n💾 Saved to: {args.output}")
    
    # 6. 可視化
    if args.visualize:
        visualize_sensors(
            coords=coords,
            sensor_indices=final_indices,
            fields=feature_dict,
            output_dir=args.output_dir,
            grid_shape=(nx, ny)
        )
    
    print("\n" + "=" * 70)
    print("✅ Fluent V2 Phase A sensor generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
