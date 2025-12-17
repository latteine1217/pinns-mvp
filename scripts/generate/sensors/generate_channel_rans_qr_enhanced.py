#!/usr/bin/env python3
"""
Enhanced Channel Flow QR-Pivot Sensor Generator with Turbulence Features
從 RANS 資料生成具備豐富湍流特徵的感測點

特點：
1. 使用 RANS 模擬結果（k-omega SST）
2. 包含湍流動能 (TKE)、Reynolds stresses、壓力梯度等
3. 特徵數從 6 增加到 20+ 以改善矩陣 rank
4. 自動提取 2D 切片或處理 3D 體積

作者：PINNs Channel Flow Team
日期：2025-12-17
"""

import sys
from pathlib import Path
# Add project root to path (3 levels up: sensors/ -> generate/ -> scripts/ -> root/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
from pinnx.sensors.qr_pivot import QRPivotSelector
import argparse
import logging
from typing import Optional, Tuple, Dict

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_rans_data_2d_slice(
    rans_npz_file: str,
    slice_axis: str = 'z',
    slice_location: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    從 3D RANS NPZ 檔案提取 2D 切片
    
    Args:
        rans_npz_file: RANS 資料檔案路徑
        slice_axis: 切片軸 ('x', 'y', 'z')
        slice_location: 切片位置（None = 中心平面）
    
    Returns:
        x, y: 2D 網格座標 [nx] [ny]
        fields: 包含 u, v, w, p, k, mu_t 的字典 [nx, ny]
    """
    data = np.load(rans_npz_file)
    
    logger.info(f"📂 Loading RANS data: {rans_npz_file}")
    logger.info(f"   Original shape: {data['u'].shape}")
    
    x_full, y_full, z_full = data['x'], data['y'], data['z']
    u_full = data['u']  # [nx, ny, nz]
    v_full = data['v']
    w_full = data['w']
    p_full = data['p']
    k_full = data['k']
    mu_t_full = data['mu_t']
    
    # 提取切片
    if slice_axis == 'z':
        if slice_location is None:
            slice_idx = len(z_full) // 2
        else:
            slice_idx = np.argmin(np.abs(z_full - slice_location))
        
        logger.info(f"   Extracting z-slice at index {slice_idx}/{len(z_full)-1} (z={z_full[slice_idx]:.4f})")
        
        x, y = x_full, y_full
        u = u_full[:, :, slice_idx]
        v = v_full[:, :, slice_idx]
        w = w_full[:, :, slice_idx]
        p = p_full[:, :, slice_idx]
        k = k_full[:, :, slice_idx]
        mu_t = mu_t_full[:, :, slice_idx]
        
    elif slice_axis == 'y':
        if slice_location is None:
            slice_idx = len(y_full) // 2
        else:
            slice_idx = np.argmin(np.abs(y_full - slice_location))
        
        logger.info(f"   Extracting y-slice at index {slice_idx}/{len(y_full)-1} (y={y_full[slice_idx]:.4f})")
        
        x, y = x_full, z_full
        u = u_full[:, slice_idx, :]
        v = v_full[:, slice_idx, :]
        w = w_full[:, slice_idx, :]
        p = p_full[:, slice_idx, :]
        k = k_full[:, slice_idx, :]
        mu_t = mu_t_full[:, slice_idx, :]
        
    elif slice_axis == 'x':
        if slice_location is None:
            slice_idx = len(x_full) // 2
        else:
            slice_idx = np.argmin(np.abs(x_full - slice_location))
        
        logger.info(f"   Extracting x-slice at index {slice_idx}/{len(x_full)-1} (x={x_full[slice_idx]:.4f})")
        
        x, y = y_full, z_full
        u = u_full[slice_idx, :, :]
        v = v_full[slice_idx, :, :]
        w = w_full[slice_idx, :, :]
        p = p_full[slice_idx, :, :]
        k = k_full[slice_idx, :, :]
        mu_t = mu_t_full[slice_idx, :, :]
    else:
        raise ValueError(f"Invalid slice_axis: {slice_axis}")
    
    logger.info(f"   2D slice shape: {u.shape}")
    
    fields = {
        'u': u, 'v': v, 'w': w,
        'p': p, 'k': k, 'mu_t': mu_t
    }
    
    return x, y, fields


def compute_reynolds_stresses_boussinesq(
    k: np.ndarray,
    mu_t: np.ndarray,
    dudx: np.ndarray,
    dudy: np.ndarray,
    dvdx: np.ndarray,
    dvdy: np.ndarray,
    rho: float = 1.0
) -> Dict[str, np.ndarray]:
    """
    使用 Boussinesq 假設計算 Reynolds stresses
    
    Boussinesq 假設：
        τ_ij = -ρ⟨u'_i u'_j⟩ = μ_t (∂U_i/∂x_j + ∂U_j/∂x_i) - (2/3)ρk δ_ij
    
    對於不可壓流（∇·U = 0）：
        τ_xx = 2μ_t ∂U/∂x - (2/3)ρk
        τ_yy = 2μ_t ∂V/∂y - (2/3)ρk
        τ_zz = -2μ_t (∂U/∂x + ∂V/∂y) - (2/3)ρk  [from continuity]
        τ_xy = μ_t (∂U/∂y + ∂V/∂x)
    
    Args:
        k: 湍流動能 [nx, ny]
        mu_t: 渦黏度 [nx, ny]
        dudx, dudy, dvdx, dvdy: 速度梯度
        rho: 密度（預設 1.0）
    
    Returns:
        dict: tau_uu, tau_vv, tau_ww, tau_uv
    """
    # 各向同性部分
    iso_term = (2.0 / 3.0) * rho * k
    
    # Normal stresses
    tau_uu = 2.0 * mu_t * dudx - iso_term
    tau_vv = 2.0 * mu_t * dvdy - iso_term
    tau_ww = -2.0 * mu_t * (dudx + dvdy) - iso_term  # 從連續方程
    
    # Shear stress
    tau_uv = mu_t * (dudy + dvdx)
    
    return {
        'tau_uu': tau_uu,
        'tau_vv': tau_vv,
        'tau_ww': tau_ww,
        'tau_uv': tau_uv
    }


def compute_phase_a_features(
    fields: Dict[str, np.ndarray],
    tau_dict: Dict[str, np.ndarray],
    S_dict: Dict[str, np.ndarray],
    omega_z: np.ndarray,
    nu: float,
    Re_tau: float,
    delta: float = 1.0,
    y_coords: Optional[np.ndarray] = None
) -> Dict[str, np.ndarray]:
    """
    計算 Phase A 的進階湍流特徵
    
    新增特徵：
    1. P_k: TKE 生成率
    2. y_plus: 壁面距離（無量綱）
    3. b_11, b_22, b_12: 各向異性張量
    4. Re_t: 湍流雷諾數
    5. epsilon: 耗散率（從梯度估算）
    6. enstrophy: 渦度"動能"
    
    Args:
        fields: 包含 u, v, w, p, k, mu_t
        tau_dict: Reynolds stresses (tau_uu, tau_vv, tau_ww, tau_uv)
        S_dict: 應變率張量 (S_11, S_22, S_12)
        omega_z: 渦度
        nu: 運動黏度
        Re_tau: 摩擦雷諾數
        delta: 通道半高（預設 1.0）
        y_coords: Y 座標 [ny]（用於計算 y_plus）
    
    Returns:
        phase_a_features: 包含 7 個新特徵的字典
    """
    k = fields['k']
    
    # 1. TKE Production: P_k = -tau_ij * S_ij
    # 主導項：P_k ≈ -tau_uv * (du/dy)（壁面剪切生成）
    P_k = -(
        tau_dict['tau_uu'] * S_dict['S_11'] +
        tau_dict['tau_vv'] * S_dict['S_22'] +
        2 * tau_dict['tau_uv'] * S_dict['S_12']  # factor 2 for off-diagonal
    )
    
    # 2. Wall Distance (y+)
    if y_coords is not None:
        u_tau = Re_tau * nu / delta
        # 廣播到 2D: y_coords [ny] → [nx, ny]
        nx = k.shape[0]
        y_plus = np.tile(y_coords[np.newaxis, :], (nx, 1)) * u_tau / nu
    else:
        # Fallback: 使用 0-1 規範化的假設
        logger.warning("   No y_coords provided, using normalized y ∈ [0, 1]")
        ny = k.shape[1]
        y_normalized = np.linspace(0, 1, ny)
        u_tau = Re_tau * nu / delta
        y_plus = np.tile(y_normalized[np.newaxis, :], (k.shape[0], 1)) * delta * u_tau / nu
    
    # 3. Anisotropy Tensor: b_ij = tau_ij / (2k) - (1/3) * delta_ij
    # 確保 k > 0 避免除零
    k_safe = np.maximum(k, 1e-10)
    b_11 = tau_dict['tau_uu'] / (2 * k_safe) - 1.0/3.0
    b_22 = tau_dict['tau_vv'] / (2 * k_safe) - 1.0/3.0
    b_12 = tau_dict['tau_uv'] / (2 * k_safe)  # 非對角項無 -1/3
    
    # 4. Dissipation Rate: epsilon ≈ 2 * nu * <s_ij * s_ij>
    # 使用 Frobenius norm of strain rate tensor
    epsilon = 2 * nu * (
        S_dict['S_11']**2 + 
        S_dict['S_22']**2 + 
        2 * S_dict['S_12']**2  # off-diagonal 貢獻兩次
    )
    
    # 5. Turbulent Reynolds Number: Re_t = k^2 / (nu * epsilon)
    epsilon_safe = np.maximum(epsilon, 1e-10)
    Re_t = k_safe**2 / (nu * epsilon_safe)
    
    # 6. Enstrophy: 渦度場的"動能" = 0.5 * omega^2
    enstrophy = 0.5 * omega_z**2
    
    return {
        'P_k': P_k,
        'y_plus': y_plus,
        'b_11': b_11,
        'b_22': b_22,
        'b_12': b_12,
        'Re_t': Re_t,
        'epsilon': epsilon,
        'enstrophy': enstrophy
    }


def compute_enhanced_turbulence_features_2d(
    fields: Dict[str, np.ndarray],
    dx: float,
    dy: float,
    Lx: float,
    Ly: float,
    periodic_x: bool = True,
    rho: float = 1.0,
    nu: Optional[float] = None,
    Re_tau: Optional[float] = None,
    y_coords: Optional[np.ndarray] = None,
    compute_phase_a: bool = False
) -> Dict[str, np.ndarray]:
    """
    計算增強的湍流特徵（用於 QR-Pivot）
    
    特徵集合：
    1. 基本流場：u, v, w, p  (4)
    2. 速度梯度：∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y, ∂w/∂x, ∂w/∂y  (6)
    3. 壓力梯度：∂p/∂x, ∂p/∂y  (2)
    4. 渦度：ω_z  (1)
    5. 湍流動能：k  (1)
    6. Reynolds stresses：τ_uu, τ_vv, τ_ww, τ_uv  (4)
    7. 速度梯度張量特徵值：λ_1, λ_2  (2)
    8. Phase A (可選)：P_k, y_plus, b_ij, Re_t, epsilon, enstrophy  (+8)
    
    總計：20 特徵（基礎）+ 8（Phase A）= 28
    
    Args:
        fields: 包含 u, v, w, p, k, mu_t 的字典
        dx, dy: 網格間距
        Lx, Ly: 域長度（用於無量綱化）
        periodic_x: x 方向是否週期
        rho: 密度
        nu: 運動黏度（Phase A 需要）
        Re_tau: 摩擦雷諾數（Phase A 需要）
        y_coords: Y 座標陣列（Phase A 需要）
        compute_phase_a: 是否計算 Phase A 特徵
    
    Returns:
        feature_dict: 包含所有特徵的字典
    """
    u, v, w = fields['u'], fields['v'], fields['w']
    p, k, mu_t = fields['p'], fields['k'], fields['mu_t']
    
    # ========== 1. 速度梯度 ==========
    if periodic_x:
        dudx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        dvdx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx)
        dwdx = (np.roll(w, -1, axis=0) - np.roll(w, 1, axis=0)) / (2 * dx)
    else:
        dudx = np.gradient(u, dx, axis=0)
        dvdx = np.gradient(v, dx, axis=0)
        dwdx = np.gradient(w, dx, axis=0)
    
    dudy = np.gradient(u, dy, axis=1)
    dvdy = np.gradient(v, dy, axis=1)
    dwdy = np.gradient(w, dy, axis=1)
    
    # 無量綱化
    dudx_norm = dudx * Lx
    dudy_norm = dudy * Ly
    dvdx_norm = dvdx * Lx
    dvdy_norm = dvdy * Ly
    dwdx_norm = dwdx * Lx
    dwdy_norm = dwdy * Ly
    
    # ========== 2. 壓力梯度 ==========
    if periodic_x:
        dpdx = (np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)) / (2 * dx)
    else:
        dpdx = np.gradient(p, dx, axis=0)
    
    dpdy = np.gradient(p, dy, axis=1)
    
    dpdx_norm = dpdx * Lx
    dpdy_norm = dpdy * Ly
    
    # ========== 3. 渦度 ==========
    omega_z_norm = dvdx_norm - dudy_norm
    
    # ========== 4. Reynolds stresses (Boussinesq) ==========
    reynolds_stresses = compute_reynolds_stresses_boussinesq(
        k, mu_t, dudx, dudy, dvdx, dvdy, rho
    )
    
    # ========== 5. 速度梯度張量特徵值 ==========
    nx, ny = u.shape
    grad_u_eigenvalues = np.zeros((nx, ny, 2))
    
    for i in range(nx):
        for j in range(ny):
            grad_u_matrix = np.array([
                [dudx_norm[i, j], dudy_norm[i, j]],
                [dvdx_norm[i, j], dvdy_norm[i, j]]
            ])
            eigenvalues = np.linalg.eigvals(grad_u_matrix)
            grad_u_eigenvalues[i, j, :] = np.sort(eigenvalues.real)[::-1]
    
    # ========== 6. (Optional) Q-criterion ==========
    # Q = -0.5 * (S:S + Ω:Ω)，可辨識渦流結構
    S11 = dudx_norm
    S22 = dvdy_norm
    S12 = 0.5 * (dudy_norm + dvdx_norm)
    Omega12 = 0.5 * (dvdx_norm - dudy_norm)
    
    Q = -(S11**2 + 2*S12**2 + S22**2 + 2*Omega12**2)
    
    # ========== 7. Phase A Advanced Features (Optional) ==========
    phase_a_features = {}
    if compute_phase_a and nu is not None and Re_tau is not None:
        logger.info("   Computing Phase A features...")
        
        # 準備應變率張量字典
        S_dict = {
            'S_11': S11,
            'S_22': S22,
            'S_12': S12
        }
        
        phase_a_features = compute_phase_a_features(
            fields=fields,
            tau_dict=reynolds_stresses,
            S_dict=S_dict,
            omega_z=omega_z_norm,
            nu=nu,
            Re_tau=Re_tau,
            delta=1.0,  # 通道半高
            y_coords=y_coords
        )
    
    # 合併所有特徵
    result = {
        # Primary fields (4)
        'u': u, 'v': v, 'w': w, 'p': p,
        
        # Velocity gradients (6)
        'dudx': dudx_norm, 'dudy': dudy_norm,
        'dvdx': dvdx_norm, 'dvdy': dvdy_norm,
        'dwdx': dwdx_norm, 'dwdy': dwdy_norm,
        
        # Pressure gradients (2)
        'dpdx': dpdx_norm, 'dpdy': dpdy_norm,
        
        # Vorticity (1)
        'omega_z': omega_z_norm,
        
        # Turbulence (1)
        'k': k,
        
        # Reynolds stresses (4)
        'tau_uu': reynolds_stresses['tau_uu'],
        'tau_vv': reynolds_stresses['tau_vv'],
        'tau_ww': reynolds_stresses['tau_ww'],
        'tau_uv': reynolds_stresses['tau_uv'],
        
        # Eigenvalues (2)
        'grad_u_eig1': grad_u_eigenvalues[:, :, 0],
        'grad_u_eig2': grad_u_eigenvalues[:, :, 1],
        
        # Optional (1)
        'Q': Q
    }
    
    # 添加 Phase A 特徵（如果計算了）
    result.update(phase_a_features)
    
    return result


def build_enhanced_data_matrix(
    feature_dict: Dict[str, np.ndarray],
    feature_selection: str = 'full'
) -> Tuple[np.ndarray, list]:
    """
    建立增強的資料矩陣用於 QR-Pivot
    
    Args:
        feature_dict: 特徵字典
        feature_selection: 'full' (20+), 'minimal' (~12), 'original' (6)
    
    Returns:
        data_matrix: [N_points, N_features] 標準化後的矩陣
        feature_names: 特徵名稱列表
    """
    
    if feature_selection == 'full':
        # 完整特徵集：20 個特徵
        feature_names = [
            'u', 'v', 'w', 'p',                                      # 4
            'dudx', 'dudy', 'dvdx', 'dvdy', 'dwdx', 'dwdy',          # 6
            'dpdx', 'dpdy',                                          # 2
            'omega_z',                                               # 1
            'k',                                                     # 1
            'tau_uu', 'tau_vv', 'tau_ww', 'tau_uv',                  # 4
            'grad_u_eig1', 'grad_u_eig2'                             # 2
        ]
        
    elif feature_selection == 'minimal':
        # 最小物理特徵集：12 個
        feature_names = [
            'u', 'v', 'w', 'p',                                      # 4
            'dudy',                                                  # 1 (壁面剪切)
            'omega_z',                                               # 1
            'k',                                                     # 1
            'tau_uv',                                                # 1 (主要 Reynolds stress)
            'grad_u_eig1', 'grad_u_eig2'                             # 2
        ]
        
    elif feature_selection == 'original':
        # 原始特徵集（與舊版一致）：6 個
        feature_names = [
            'u', 'v', 'w',
            'omega_z',
            'grad_u_eig1', 'grad_u_eig2'
        ]
        
    elif feature_selection == 'physics_guided':
        # 物理導向特徵集：15 個（平衡豐富性與效率）
        feature_names = [
            'u', 'v', 'w', 'p',                                      # 4
            'dudx', 'dudy', 'dvdx', 'dvdy',                          # 4 (關鍵梯度)
            'dpdx', 'dpdy',                                          # 2
            'omega_z',                                               # 1
            'k',                                                     # 1
            'tau_uv',                                                # 1
            'grad_u_eig1', 'grad_u_eig2'                             # 2
        ]
        
    elif feature_selection == 'phase_a':
        # Phase A：minimal (10) + 進階湍流特徵 (8) = 18 個
        feature_names = [
            # Minimal baseline (10)
            'u', 'v', 'w', 'p',                                      # 4
            'dudy',                                                  # 1 (壁面剪切)
            'omega_z',                                               # 1
            'k',                                                     # 1
            'tau_uv',                                                # 1 (主要 Reynolds stress)
            'grad_u_eig1', 'grad_u_eig2',                            # 2
            
            # Phase A 新增特徵 (8)
            'P_k',                                                   # 1 (TKE 生成率)
            'y_plus',                                                # 1 (壁面距離)
            'b_11', 'b_22', 'b_12',                                  # 3 (各向異性張量)
            'Re_t',                                                  # 1 (湍流雷諾數)
            'epsilon',                                               # 1 (耗散率)
            'enstrophy'                                              # 1 (渦度動能)
        ]
        
    else:
        raise ValueError(f"Unknown feature_selection: {feature_selection}")
    
    # 建立特徵矩陣
    features = []
    for name in feature_names:
        if name not in feature_dict:
            raise KeyError(f"Feature '{name}' not found in feature_dict")
        features.append(feature_dict[name].flatten())
    
    data_matrix = np.column_stack(features)
    
    # 標準化（critical for QR rank）
    mean = data_matrix.mean(axis=0)
    std = data_matrix.std(axis=0) + 1e-10  # 避免除零
    data_matrix = (data_matrix - mean) / std
    
    logger.info(f"   Feature selection: '{feature_selection}' → {len(feature_names)} features")
    logger.info(f"   Data matrix shape: {data_matrix.shape}")
    
    return data_matrix, feature_names


def generate_channel_rans_qr_enhanced(
    rans_npz_file: str,
    K: int = 100,
    output_file: Optional[str] = None,
    feature_selection: str = 'full',
    slice_axis: str = 'z',
    slice_location: Optional[float] = None,
    n_wrap_layers: int = 2,
    seam_weight: float = 0.5,
    seam_width_fraction: float = 0.05,
    max_seam_fraction: float = 0.1
):
    """
    從 RANS 資料生成增強的 QR-Pivot 感測點
    
    Args:
        rans_npz_file: RANS 資料檔案（NPZ 格式）
        K: 感測點數量
        output_file: 輸出檔案路徑
        feature_selection: 'full', 'minimal', 'original', 'physics_guided'
        slice_axis: 切片軸（'x', 'y', 'z'）
        slice_location: 切片位置（None = 中心）
        n_wrap_layers: 週期邊界包裹層數
        seam_weight: 接縫權重
        seam_width_fraction: 接縫寬度比例
        max_seam_fraction: 最大接縫感測點比例
    
    Returns:
        final_indices: 選中的感測點索引
        metrics: QR-Pivot 指標（condition number, energy ratio）
    """
    
    print("=" * 70)
    print("🔬 Enhanced RANS QR-Pivot Sensor Generation")
    print("=" * 70)
    
    # ========== 1. 載入 RANS 資料 ==========
    x, y, fields = load_rans_data_2d_slice(rans_npz_file, slice_axis, slice_location)
    nx, ny = len(x), len(y)
    
    # 載入 Phase A 所需的額外參數
    rans_data = np.load(rans_npz_file)
    nu = float(rans_data['nu']) if 'nu' in rans_data else None
    Re_tau = float(rans_data['Re_tau_estimate']) if 'Re_tau_estimate' in rans_data else None
    
    # 域資訊
    Lx = x[-1] - x[0] + (x[1] - x[0])  # 假設 x 週期
    Ly = y[-1] - y[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    
    logger.info(f"\n📐 Domain Information:")
    logger.info(f"   Grid: {nx} × {ny} = {nx*ny:,} points")
    logger.info(f"   X: [{x[0]:.4f}, {x[-1]:.4f}], dx={dx:.6f}, Lx≈{Lx:.4f}")
    logger.info(f"   Y: [{y[0]:.4f}, {y[-1]:.4f}], dy={dy:.6f}, Ly≈{Ly:.4f}")
    
    if nu is not None and Re_tau is not None:
        logger.info(f"   Re_tau = {Re_tau:.1f}, nu = {nu:.2e}")
    
    # ========== 2. 計算增強特徵 ==========
    logger.info(f"\n🔧 Computing enhanced turbulence features...")
    
    # 判斷是否需要計算 Phase A 特徵
    compute_phase_a = (feature_selection == 'phase_a')
    
    feature_dict = compute_enhanced_turbulence_features_2d(
        fields, dx, dy, Lx, Ly, 
        periodic_x=True,
        nu=nu,
        Re_tau=Re_tau,
        y_coords=y,
        compute_phase_a=compute_phase_a
    )
    
    # ========== 3. 建立資料矩陣 ==========
    logger.info(f"\n📈 Building data matrix...")
    data_matrix, feature_names = build_enhanced_data_matrix(
        feature_dict, feature_selection
    )
    
    logger.info(f"   Features ({len(feature_names)}): {', '.join(feature_names[:5])}...")
    
    # 檢查矩陣品質
    rank = np.linalg.matrix_rank(data_matrix)
    logger.info(f"   Matrix rank: {rank} / {min(data_matrix.shape)}")
    
    # ========== 4. QR-Pivot 選點 ==========
    logger.info(f"\n🔄 QR-Pivot Selection:")
    
    coords = np.column_stack([
        np.repeat(x, ny),
        np.tile(y, nx)
    ])
    
    selector = QRPivotSelector(
        use_circular_indexing=True,
        n_wrap_layers=n_wrap_layers,
        seam_weight=seam_weight,
        seam_width_fraction=seam_width_fraction,
        max_seam_fraction=max_seam_fraction,
        mode='column',
        pivoting=True
    )
    
    final_indices, metrics = selector.select_sensors(
        data_matrix=data_matrix,
        n_sensors=K,
        coords=coords,
        grid_shape=(nx, ny),
        periodic_axes=[0],  # x 週期
        domain_lengths={0: Lx, 1: Ly}
    )
    
    selected_coords = coords[final_indices]
    
    logger.info(f"\n✅ Selection Complete:")
    logger.info(f"   Selected sensors: {len(final_indices)}")
    logger.info(f"   Condition number: {metrics.get('condition_number', 0):.2e}")
    logger.info(f"   Energy ratio: {metrics.get('energy_ratio', 0):.6f}")
    
    # ========== 5. 儲存結果 ==========
    if output_file is None:
        output_file = f"data/lowfi/channel_rans/sensors_K{K}_rans_enhanced_{feature_selection}.npz"
    
    save_dict = {
        # Sensor locations
        'sensor_points': selected_coords.astype(np.float32),
        'sensor_indices': final_indices,
        'sensor_x': selected_coords[:, 0].astype(np.float32),
        'sensor_y': selected_coords[:, 1].astype(np.float32),
        
        # Metadata
        'K': K,
        'n_features': len(feature_names),
        'feature_names': np.array(feature_names, dtype=object),
        'feature_selection': feature_selection,
        
        # Metrics
        'condition_number': float(metrics.get('condition_number', 0)),
        'energy_ratio': float(metrics.get('energy_ratio', 0)),
        'matrix_rank': rank,
        
        # Configuration
        'method': 'QR-Pivot-RANS-Enhanced',
        'source_file': rans_npz_file,
        'slice_axis': slice_axis,
        'slice_location': slice_location if slice_location is not None else 'center',
        'domain_Lx': Lx,
        'domain_Ly': Ly,
        
        # QR parameters
        'n_wrap_layers': n_wrap_layers,
        'seam_weight': seam_weight,
        'seam_width_fraction': seam_width_fraction,
        'max_seam_fraction': max_seam_fraction,
    }
    
    np.savez(output_file, **save_dict)
    logger.info(f"\n💾 Saved to: {output_file}")
    
    return final_indices, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced QR-Pivot sensor generation from RANS data"
    )
    
    parser.add_argument(
        '--rans-npz',
        type=str,
        default='data/lowfi/channel_rans/rans_k_omega_sst.npz',
        help='RANS data file (NPZ format)'
    )
    
    parser.add_argument(
        '-K', '--n-sensors',
        type=int,
        default=100,
        help='Number of sensors (default: 100)'
    )
    
    parser.add_argument(
        '--feature-selection',
        type=str,
        default='full',
        choices=['full', 'minimal', 'original', 'physics_guided', 'phase_a'],
        help='Feature set selection (default: full). phase_a: minimal + advanced turbulence features (18 total)'
    )
    
    parser.add_argument(
        '--slice-axis',
        type=str,
        default='z',
        choices=['x', 'y', 'z'],
        help='Slice axis for 2D extraction (default: z)'
    )
    
    parser.add_argument(
        '--slice-location',
        type=float,
        default=None,
        help='Slice location (None = center plane)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (auto-generated if not specified)'
    )
    
    parser.add_argument(
        '--n-wrap-layers',
        type=int,
        default=2,
        help='Number of wrap layers for periodic boundary (default: 2)'
    )
    
    parser.add_argument(
        '--seam-weight',
        type=float,
        default=0.5,
        help='Seam weight for periodic boundary (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # 執行生成
    final_indices, metrics = generate_channel_rans_qr_enhanced(
        rans_npz_file=args.rans_npz,
        K=args.n_sensors,
        output_file=args.output,
        feature_selection=args.feature_selection,
        slice_axis=args.slice_axis,
        slice_location=args.slice_location,
        n_wrap_layers=args.n_wrap_layers,
        seam_weight=args.seam_weight
    )
    
    print("\n" + "=" * 70)
    print("✅ Enhanced QR-Pivot sensor generation complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
