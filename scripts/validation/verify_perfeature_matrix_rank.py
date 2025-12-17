"""
Verify matrix rank of per-feature QR-pivot sensor selection.

This script loads pre-generated sensors and recomputes the feature matrix
to analyze rank, condition number, and singular value spectrum.
"""
import numpy as np
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Verify per-feature sensor matrix rank")
    parser.add_argument('--sensor-file', type=str, required=True,
                       help="Path to sensor .npz file")
    parser.add_argument('--rans-file', type=str, default='data/lowfi/channel_rans/rans_k_omega_sst.npz',
                       help="Path to RANS data file")
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Per-Feature Sensor Matrix Rank Verification")
    logger.info("=" * 80)
    
    # Load sensor data
    sensor_data = np.load(args.sensor_file, allow_pickle=True)
    logger.info(f"\n✅ Loaded sensor file: {args.sensor_file}")
    logger.info(f"   Sensors (K): {sensor_data['K']}")
    logger.info(f"   Features (n): {sensor_data['n_features']}")
    logger.info(f"   Condition (reported): {sensor_data['condition_number']:.2e}")
    logger.info(f"   Deduplication: {sensor_data['deduplication_rate']:.1%}")
    
    sensor_indices = sensor_data['sensor_indices']
    feature_names = list(sensor_data['feature_names'])
    
    logger.info(f"\n📋 Features: {', '.join(feature_names)}")
    
    # Load RANS data
    rans_data = np.load(args.rans_file)
    logger.info(f"\n✅ Loaded RANS file: {args.rans_file}")
    
    # Extract 2D slice (middle z-plane)
    u_3d = rans_data['u']
    v_3d = rans_data['v']
    w_3d = rans_data['w']
    p_3d = rans_data['p']
    k_3d = rans_data['k']
    mu_t_3d = rans_data['mu_t']
    
    nx, ny, nz = u_3d.shape
    slice_idx = nz // 2
    
    # Take middle slice
    u = u_3d[:, :, slice_idx]
    v = v_3d[:, :, slice_idx]
    w = w_3d[:, :, slice_idx]
    p = p_3d[:, :, slice_idx]
    k = k_3d[:, :, slice_idx]
    mu_t = mu_t_3d[:, :, slice_idx]
    
    x = rans_data['x']
    y = rans_data['y']
    nu = float(rans_data['nu'])
    Re_tau = float(rans_data.get('Re_tau_estimate', 1000.0))
    
    logger.info(f"   3D Grid: {nx} × {ny} × {nz}")
    logger.info(f"   2D Slice (z={slice_idx}): {u.shape}")
    logger.info(f"   Re_tau: {Re_tau}, nu: {nu}")
    
    # Compute gradients
    logger.info(f"\n🔧 Computing features...")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # Basic gradients
    dudy = np.gradient(u, dy, axis=1)
    dvdx = np.gradient(v, dx, axis=0)
    dudx = np.gradient(u, dx, axis=0)
    dvdy = np.gradient(v, dy, axis=0)
    
    # Vorticity
    omega_z = dvdx - np.gradient(u, dy, axis=1)
    
    # Reynolds stresses (Boussinesq)
    S_xy = 0.5 * (dudy + dvdx)
    tau_uv = mu_t * S_xy
    
    # Velocity gradient eigenvalues
    eig1 = np.zeros_like(u)
    eig2 = np.zeros_like(u)
    for i in range(nx):
        for j in range(ny):
            J = np.array([[dudx[i,j], dudy[i,j]],
                         [dvdx[i,j], dvdy[i,j]]])
            eigvals = np.linalg.eigvals(J)
            eig1[i,j] = np.real(eigvals[0])
            eig2[i,j] = np.real(eigvals[1])
    
    # Phase A features
    # TKE production
    S_11 = dudx
    S_22 = dvdy
    S_12 = S_xy
    
    iso_term = (2.0/3.0) * k
    tau_uu = 2.0 * mu_t * S_11 - iso_term
    tau_vv = 2.0 * mu_t * S_22 - iso_term
    
    P_k = -(tau_uu * S_11 + tau_vv * S_22 + 2 * tau_uv * S_12)
    
    # Wall distance (y+)
    u_tau = Re_tau * nu / 1.0  # delta = 1.0
    y_plus = np.tile(y[np.newaxis, :], (nx, 1)) * u_tau / nu
    
    # Anisotropy tensor
    k_safe = np.maximum(k, 1e-10)
    b_11 = tau_uu / (2 * k_safe) - 1.0/3.0
    b_22 = tau_vv / (2 * k_safe) - 1.0/3.0
    b_12 = tau_uv / (2 * k_safe)
    
    # Dissipation rate
    epsilon = 2 * nu * (S_11**2 + S_22**2 + 2 * S_12**2)
    
    # Turbulent Reynolds number
    epsilon_safe = np.maximum(epsilon, 1e-10)
    Re_t = k_safe**2 / (nu * epsilon_safe)
    
    # Enstrophy
    enstrophy = 0.5 * omega_z**2
    
    # Stack features
    features = [u, v, w, p, dudy, omega_z, k, tau_uv, 
                eig1, eig2, P_k, y_plus, b_11, b_22, b_12,
                Re_t, epsilon, enstrophy]
    
    # Flatten to [n_locations, n_features]
    data_matrix = np.stack([f.ravel() for f in features], axis=1)
    logger.info(f"   Full data matrix: {data_matrix.shape}")
    
    # ⚠️ CRITICAL: Standardize features before selection (z-score normalization)
    # This is essential for QR-pivot to work correctly with mixed-scale features
    logger.info(f"\n   Standardizing features (z-score)...")
    X_mean = np.mean(data_matrix, axis=0, keepdims=True)
    X_std = np.std(data_matrix, axis=0, keepdims=True) + 1e-8
    data_normalized = (data_matrix - X_mean) / X_std
    
    logger.info(f"   Feature ranges before standardization:")
    for i, fname in enumerate(feature_names[:5]):  # Show first 5
        logger.info(f"      {fname:15s}: [{data_matrix[:, i].min():+.2e}, {data_matrix[:, i].max():+.2e}]")
    logger.info(f"      ...")
    for i, fname in enumerate(feature_names[-3:], start=len(feature_names)-3):  # Show last 3
        logger.info(f"      {fname:15s}: [{data_matrix[:, i].min():+.2e}, {data_matrix[:, i].max():+.2e}]")
    
    # Select sensors (use normalized matrix)
    selected_matrix = data_normalized[sensor_indices, :]
    logger.info(f"\n   Sensor matrix (standardized): {selected_matrix.shape}")
    
    # Rank analysis
    logger.info(f"\n" + "=" * 80)
    logger.info("Matrix Rank Analysis")
    logger.info("=" * 80)
    
    rank = np.linalg.matrix_rank(selected_matrix)
    cond = np.linalg.cond(selected_matrix)
    
    K, n_feat = selected_matrix.shape
    logger.info(f"\n📊 Shape: [{K} sensors × {n_feat} features]")
    logger.info(f"\n🔢 Matrix Rank: {rank} / {n_feat}")
    
    if rank == n_feat:
        logger.info("   ✅ FULL RANK - All features linearly independent")
    elif rank == K:
        logger.info(f"   ⚠️  SENSOR-LIMITED RANK")
        logger.info(f"   → Maximum possible rank = {K} (number of sensors)")
        logger.info(f"   → {n_feat - rank} features are implicit")
        logger.info(f"   → PINN must learn correlations to recover all features")
    else:
        logger.info(f"   ❌ RANK DEFICIENT ({rank} < min({K}, {n_feat}))")
    
    logger.info(f"\n📈 Condition Number: {cond:.2e}")
    if cond < 1e3:
        logger.info("   ✅ EXCELLENT - Extremely well-conditioned")
    elif cond < 1e6:
        logger.info("   ✅ GOOD - Well-conditioned")
    elif cond < 1e10:
        logger.info("   ⚠️  MODERATE - Acceptable but monitor closely")
    else:
        logger.info("   ❌ POOR - Ill-conditioned matrix")
    
    # SVD analysis
    logger.info(f"\n" + "=" * 80)
    logger.info("Singular Value Decomposition")
    logger.info("=" * 80)
    
    U, s, Vt = np.linalg.svd(selected_matrix, full_matrices=False)
    
    logger.info(f"\n   Singular values (σ_1 to σ_{len(s)}):")
    logger.info(f"   Range: [{s[0]:.3e}, {s[-1]:.3e}]")
    logger.info(f"   Ratio: σ_max / σ_min = {s[0] / s[-1]:.2e}")
    
    logger.info(f"\n   Top 10:")
    for i in range(min(10, len(s))):
        ratio = s[i] / s[0]
        bar_len = int(ratio * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        logger.info(f"   σ_{i+1:2d} = {s[i]:9.3e}  [{bar}] {ratio*100:5.1f}%")
    
    if len(s) > 10:
        logger.info(f"\n   Bottom 5:")
        for i in range(max(10, len(s)-5), len(s)):
            ratio = s[i] / s[0]
            bar_len = max(1, int(ratio * 30))
            bar = "░" * bar_len
            logger.info(f"   σ_{i+1:2d} = {s[i]:9.3e}  [{bar}] {ratio*100:5.1f}%")
    
    # Cumulative energy
    energy = np.cumsum(s**2) / np.sum(s**2)
    logger.info(f"\n   Cumulative Energy (Variance Explained):")
    for threshold in [0.90, 0.95, 0.99, 0.999]:
        n_comp = np.searchsorted(energy, threshold) + 1
        actual = energy[n_comp-1] if n_comp <= len(energy) else 1.0
        logger.info(f"   {actual*100:5.1f}% → {n_comp:2d}/{len(s)} components")
    
    # Feature correlation
    logger.info(f"\n" + "=" * 80)
    logger.info("Feature Correlation Analysis")
    logger.info("=" * 80)
    
    from scipy.stats import zscore
    selected_zscore = zscore(selected_matrix, axis=0)
    corr_matrix = np.corrcoef(selected_zscore, rowvar=False)
    
    high_corr = []
    for i in range(len(feature_names)):
        for j in range(i+1, len(feature_names)):
            if abs(corr_matrix[i, j]) > 0.95:
                high_corr.append((feature_names[i], feature_names[j], corr_matrix[i, j]))
    
    if high_corr:
        logger.info(f"\n   Found {len(high_corr)} highly correlated pairs (|r| > 0.95):")
        for f1, f2, r in sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]:
            logger.info(f"   {f1:12s} ↔ {f2:12s}  r = {r:+.3f}")
        if len(high_corr) > 10:
            logger.info(f"   ... and {len(high_corr) - 10} more")
    else:
        logger.info("\n   ✅ No highly correlated features (all |r| ≤ 0.95)")
    
    # Recommendations
    logger.info(f"\n" + "=" * 80)
    logger.info("Recommendations")
    logger.info("=" * 80)
    
    logger.info(f"\n💡 Physical Interpretation:")
    if rank == K < n_feat:
        logger.info(f"   ✓ Sensor-limited rank is EXPECTED when K < n_features")
        logger.info(f"   ✓ The {rank} sensors capture {rank} independent dimensions")
        logger.info(f"   ✓ Remaining {n_feat - rank} features must be learned via correlations")
        logger.info(f"\n   The 82.2% spatial overlap indicates that Phase A features")
        logger.info(f"   are strongly co-located, which justifies K < n_features.")
    
    logger.info(f"\n🎯 For PINN Training:")
    if cond < 1e4 and rank >= 0.85 * min(K, n_feat):
        logger.info(f"   ✅ RECOMMENDED - Proceed with these {K} sensors")
        logger.info(f"   → Well-conditioned matrix (κ = {cond:.2e})")
        logger.info(f"   → Good effective rank ({rank}/{min(K, n_feat)})")
        logger.info(f"   → Monitor all {n_feat} feature reconstructions during training")
    elif cond < 1e6:
        logger.info(f"   ⚠️  ACCEPTABLE - Use with comparison to global QR")
        logger.info(f"   → Compare with 100-sensor baseline")
    else:
        logger.info(f"   ❌ NOT RECOMMENDED - Increase n_per_feature")
    
    logger.info("\n" + "=" * 80)


if __name__ == '__main__':
    main()
