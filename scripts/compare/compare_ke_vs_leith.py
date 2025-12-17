"""
Leith vs k-ε RANS 模型對比驗證
============================

比較兩種湍流模型在 2D Kolmogorov flow 的表現

目標：
- 評估 Leith 模型是否改善 k-ε 的層流化問題
- 比較 V-velocity 恢復程度
- 評估動能與 DNS 的匹配度
- 決定論文中使用哪個模型
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from scipy.interpolate import RectBivariateSpline


def load_dns_data(re: int):
    """載入 DNS 數據（時間平均）"""
    dns_file = f'data/kolmogorov_dns/dns_re{re}_t100.h5'
    
    with h5py.File(dns_file, 'r') as f:
        u = f['u'][-100:].mean(axis=0)  # 最後 100 snapshots
        v = f['v'][-100:].mean(axis=0)
        
        # 網格
        if 'x' in f:
            x = f['x'][:]
            y = f['y'][:]
        else:
            N = u.shape[0]
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            y = np.linspace(0, 2*np.pi, N, endpoint=False)
    
    return u, v, x, y


def load_rans_data(file_path: str):
    """載入 RANS 數據"""
    with h5py.File(file_path, 'r') as f:
        u = f['mean_field/u'][:]
        v = f['mean_field/v'][:]
        
        # 嘗試讀取 x, y（Leith 格式）
        if 'mean_field/x' in f:
            x = f['mean_field/x'][:]
            y = f['mean_field/y'][:]
        # 舊格式：2D meshgrid X, Y（k-ε 或 Leith 舊版）
        elif 'mean_field/X' in f:
            X = f['mean_field/X'][:]
            Y = f['mean_field/Y'][:]
            # 嘗試提取 1D（假設 indexing='ij'）
            x = X[:, 0]
            y = Y[0, :]
            # 驗證是否正確
            if not (np.allclose(X, X[:, 0:1]) and np.allclose(Y, Y[0:1, :])):
                # 如果不是標準 meshgrid，重建
                N = u.shape[0]
                x = np.linspace(0, 2*np.pi, N, endpoint=False)
                y = np.linspace(0, 2*np.pi, N, endpoint=False)
        else:
            # 無網格信息，重建
            N = u.shape[0]
            x = np.linspace(0, 2*np.pi, N, endpoint=False)
            y = np.linspace(0, 2*np.pi, N, endpoint=False)
        
        # 渦黏滯（如果有）
        nu_t = f['mean_field/nu_t'][:] if 'mean_field/nu_t' in f else None
        
        # TKE（k-ε 模型）
        k = f['mean_field/k'][:] if 'mean_field/k' in f else None
    
    return u, v, x, y, nu_t, k


def interpolate_to_dns_grid(u_rans, v_rans, x_rans, y_rans, x_dns, y_dns):
    """將 RANS 插值到 DNS 網格"""
    interp_u = RectBivariateSpline(x_rans, y_rans, u_rans)
    interp_v = RectBivariateSpline(x_rans, y_rans, v_rans)
    
    X_dns, Y_dns = np.meshgrid(x_dns, y_dns, indexing='ij')
    u_interp = interp_u(x_dns, y_dns)
    v_interp = interp_v(x_dns, y_dns)
    
    return u_interp, v_interp


def compute_errors(pred, true):
    """計算誤差指標"""
    l2_rel = np.linalg.norm(pred - true) / np.linalg.norm(true) * 100
    rmse = np.sqrt(np.mean((pred - true)**2))
    mae = np.mean(np.abs(pred - true))
    max_err = np.abs(pred - true).max()
    
    return {
        'l2_rel': l2_rel,
        'rmse': rmse,
        'mae': mae,
        'max_err': max_err,
    }


def compare_models(re: int, output_dir: Path):
    """對比 k-ε 與 Leith 模型"""
    
    print("=" * 80)
    print(f"Re = {re} Comparison: k-ε vs Leith")
    print("=" * 80)
    print()
    
    # 載入 DNS
    print("Loading DNS data...")
    u_dns, v_dns, x_dns, y_dns = load_dns_data(re)
    
    # 載入 k-ε
    print("Loading k-ε RANS data...")
    ke_file = f'data/lowfi/kolmogorov_rans/rans_re{re}_kf4_corrected.h5'
    u_ke, v_ke, x_ke, y_ke, nu_t_ke, k_ke = load_rans_data(ke_file)
    
    # 載入 Leith
    print("Loading Leith RANS data...")
    leith_file = f'data/lowfi/kolmogorov_rans/rans_re{re}_kf4_leith.h5'
    u_leith, v_leith, x_leith, y_leith, nu_t_leith, _ = load_rans_data(leith_file)
    
    # 插值到 DNS 網格
    print("Interpolating to DNS grid...")
    u_ke_interp, v_ke_interp = interpolate_to_dns_grid(u_ke, v_ke, x_ke, y_ke, x_dns, y_dns)
    u_leith_interp, v_leith_interp = interpolate_to_dns_grid(
        u_leith, v_leith, x_leith, y_leith, x_dns, y_dns
    )
    
    # 計算誤差
    print()
    print("=" * 80)
    print("Error Metrics vs DNS")
    print("=" * 80)
    print()
    
    err_u_ke = compute_errors(u_ke_interp, u_dns)
    err_v_ke = compute_errors(v_ke_interp, v_dns)
    err_u_leith = compute_errors(u_leith_interp, u_dns)
    err_v_leith = compute_errors(v_leith_interp, v_dns)
    
    print(f"{'Metric':<20} {'k-ε':<20} {'Leith':<20} {'Improvement':<20}")
    print("-" * 80)
    print()
    print("U-velocity:")
    for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
        improvement = (err_u_ke[key] - err_u_leith[key]) / err_u_ke[key] * 100
        arrow = "↓" if improvement > 0 else "↑"
        print(f"  {key:<18} {err_u_ke[key]:<20.6f} {err_u_leith[key]:<20.6f} {improvement:>6.1f}% {arrow}")
    
    print()
    print("V-velocity:")
    for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
        improvement = (err_v_ke[key] - err_v_leith[key]) / err_v_ke[key] * 100
        arrow = "↓" if improvement > 0 else "↑"
        print(f"  {key:<18} {err_v_ke[key]:<20.6f} {err_v_leith[key]:<20.6f} {improvement:>6.1f}% {arrow}")
    
    # 物理統計
    print()
    print("=" * 80)
    print("Physical Statistics")
    print("=" * 80)
    print()
    
    nu = {50: 0.039374, 100: 0.019687, 500: 0.003937}[re]
    
    KE_dns = 0.5 * (u_dns**2 + v_dns**2).mean()
    KE_ke = 0.5 * (u_ke**2 + v_ke**2).mean()
    KE_leith = 0.5 * (u_leith**2 + v_leith**2).mean()
    
    print(f"DNS statistics:")
    print(f"  U: mean={u_dns.mean():.6e}, std={u_dns.std():.6f}, max={np.abs(u_dns).max():.6f}")
    print(f"  V: mean={v_dns.mean():.6e}, std={v_dns.std():.6f}, max={np.abs(v_dns).max():.6f}")
    print(f"  KE: {KE_dns:.6f}")
    print()
    
    print(f"k-ε RANS statistics:")
    print(f"  U: mean={u_ke.mean():.6e}, std={u_ke.std():.6f}, max={np.abs(u_ke).max():.6f}")
    print(f"  V: mean={v_ke.mean():.6e}, std={v_ke.std():.6f}, max={np.abs(v_ke).max():.6f}")
    print(f"  KE: {KE_ke:.6f} ({KE_ke/KE_dns*100:.1f}% of DNS)")
    if k_ke is not None:
        print(f"  TKE: mean={k_ke.mean():.6f}, TKE/KE={k_ke.mean()/KE_ke:.3f}")
    print(f"  ν_t: mean={nu_t_ke.mean():.6f}, ν_t/ν={nu_t_ke.mean()/nu:.3f}")
    if v_ke.std() < 1e-3:
        print(f"  ⚠️  V-velocity laminarized (std < 1e-3)")
    print()
    
    print(f"Leith RANS statistics:")
    print(f"  U: mean={u_leith.mean():.6e}, std={u_leith.std():.6f}, max={np.abs(u_leith).max():.6f}")
    print(f"  V: mean={v_leith.mean():.6e}, std={v_leith.std():.6f}, max={np.abs(v_leith).max():.6f}")
    print(f"  KE: {KE_leith:.6f} ({KE_leith/KE_dns*100:.1f}% of DNS)")
    print(f"  ν_t: mean={nu_t_leith.mean():.6f}, ν_t/ν={nu_t_leith.mean()/nu:.3f}")
    if v_leith.std() > 0.01:
        print(f"  ✅ V-velocity active (std > 0.01)")
    elif v_leith.std() > 1e-3:
        print(f"  ⚠️  V-velocity weak but present")
    else:
        print(f"  ❌ V-velocity laminarized")
    
    # V-velocity 改善
    v_improvement = (v_leith.std() / v_ke.std()) if v_ke.std() > 0 else float('inf')
    print()
    print(f"💡 Key Comparison:")
    print(f"   V_std improvement: k-ε ({v_ke.std():.6f}) → Leith ({v_leith.std():.6f})")
    print(f"   Improvement factor: {v_improvement:.1f}× better")
    print()
    
    # 生成對比圖
    generate_comparison_plot(
        u_dns, v_dns, u_ke, v_ke, u_leith, v_leith,
        x_dns, y_dns, x_ke, y_ke, x_leith, y_leith,
        re, output_dir
    )
    
    return {
        'u_l2_ke': err_u_ke['l2_rel'],
        'v_l2_ke': err_v_ke['l2_rel'],
        'u_l2_leith': err_u_leith['l2_rel'],
        'v_l2_leith': err_v_leith['l2_rel'],
        'v_std_dns': v_dns.std(),
        'v_std_ke': v_ke.std(),
        'v_std_leith': v_leith.std(),
    }


def generate_comparison_plot(u_dns, v_dns, u_ke, v_ke, u_leith, v_leith,
                            x_dns, y_dns, x_ke, y_ke, x_leith, y_leith,
                            re, output_dir):
    """生成對比圖"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    X_dns, Y_dns = np.meshgrid(x_dns, y_dns, indexing='ij')
    X_ke, Y_ke = np.meshgrid(x_ke, y_ke, indexing='ij')
    X_leith, Y_leith = np.meshgrid(x_leith, y_leith, indexing='ij')
    
    # Row 1: U-velocity
    vmin_u, vmax_u = u_dns.min(), u_dns.max()
    
    im = axes[0, 0].contourf(X_dns, Y_dns, u_dns, levels=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    axes[0, 0].set_title(f'DNS: U-velocity (Re={re})', fontsize=14)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im, ax=axes[0, 0])
    
    im = axes[0, 1].contourf(X_ke, Y_ke, u_ke, levels=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    axes[0, 1].set_title(f'k-ε RANS: U-velocity', fontsize=14)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    plt.colorbar(im, ax=axes[0, 1])
    
    im = axes[0, 2].contourf(X_leith, Y_leith, u_leith, levels=20, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    axes[0, 2].set_title(f'Leith RANS: U-velocity', fontsize=14)
    axes[0, 2].set_xlabel('x')
    axes[0, 2].set_ylabel('y')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Row 2: V-velocity
    vmin_v, vmax_v = v_dns.min(), v_dns.max()
    
    im = axes[1, 0].contourf(X_dns, Y_dns, v_dns, levels=20, cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    axes[1, 0].set_title(f'DNS: V-velocity (std={v_dns.std():.3f})', fontsize=14)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im, ax=axes[1, 0])
    
    im = axes[1, 1].contourf(X_ke, Y_ke, v_ke, levels=20, cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    axes[1, 1].set_title(f'k-ε: V-velocity (std={v_ke.std():.6f})', fontsize=14)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    plt.colorbar(im, ax=axes[1, 1])
    
    im = axes[1, 2].contourf(X_leith, Y_leith, v_leith, levels=20, cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    axes[1, 2].set_title(f'Leith: V-velocity (std={v_leith.std():.6f})', fontsize=14)
    axes[1, 2].set_xlabel('x')
    axes[1, 2].set_ylabel('y')
    plt.colorbar(im, ax=axes[1, 2])
    
    plt.tight_layout()
    
    output_file = output_dir / f're{re}_rans_comparison_ke_vs_leith.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"  Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare k-ε and Leith RANS models')
    parser.add_argument('--re', type=int, nargs='+', default=[50, 100, 500],
                       help='Reynolds numbers to compare')
    parser.add_argument('--output-dir', type=str, default='results/rans_validation',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("k-ε vs Leith RANS Model Comparison")
    print("=" * 80)
    print()
    print(f"Reynolds numbers: {args.re}")
    print(f"Output directory: {output_dir}")
    print()
    
    results = {}
    for re in args.re:
        results[re] = compare_models(re, output_dir)
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY: k-ε vs Leith")
    print("=" * 80)
    print()
    print(f"{'Re':<10} {'k-ε U L2%':<15} {'Leith U L2%':<15} {'Improvement':<15}")
    print("-" * 80)
    for re in args.re:
        u_ke = results[re]['u_l2_ke']
        u_leith = results[re]['u_l2_leith']
        improvement = (u_ke - u_leith) / u_ke * 100
        arrow = "↓" if improvement > 0 else "↑"
        print(f"{re:<10} {u_ke:<15.2f} {u_leith:<15.2f} {improvement:>6.1f}% {arrow}")
    
    print()
    print(f"{'Re':<10} {'k-ε V std':<15} {'Leith V std':<15} {'Factor':<15}")
    print("-" * 80)
    for re in args.re:
        v_ke = results[re]['v_std_ke']
        v_leith = results[re]['v_std_leith']
        factor = v_leith / v_ke if v_ke > 0 else float('inf')
        print(f"{re:<10} {v_ke:<15.6f} {v_leith:<15.6f} {factor:>6.1f}×")
    
    print()
    print("=" * 80)
    print("✅ Comparison complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
