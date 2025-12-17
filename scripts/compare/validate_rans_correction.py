#!/usr/bin/env python3
"""
驗證 RANS 修正效果
=================

比較修正前後的 RANS vs DNS 誤差，確認參數修正的影響。

作者：PINNs-MVP 團隊
日期：2025-12-17
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def load_dns_snapshot(filepath: str, time_index: int = -1):
    """Load DNS snapshot at given time index"""
    with h5py.File(filepath, 'r') as f:
        u = f['u'][time_index]
        v = f['v'][time_index]
        if 'p' in f:
            p = f['p'][time_index]
        else:
            p = None
        
        # Get grid
        N = u.shape[0]
        L = f['config'].attrs.get('L', 2*np.pi)
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        
        return u, v, p, x, y


def load_rans_mean(filepath: str):
    """Load RANS time-averaged fields"""
    with h5py.File(filepath, 'r') as f:
        u = f['mean_field/u'][:]
        v = f['mean_field/v'][:]
        
        # Get grid
        N = u.shape[0]
        L = f['parameters'].attrs.get('L', 2*np.pi)
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        
        return u, v, x, y


def interpolate_to_grid(field_coarse, grid_coarse, grid_fine):
    """Interpolate coarse field to fine grid using 2D linear interpolation"""
    from scipy.interpolate import RectBivariateSpline
    
    x_coarse, y_coarse = grid_coarse
    x_fine, y_fine = grid_fine
    
    # Create interpolator
    interp = RectBivariateSpline(x_coarse, y_coarse, field_coarse)
    
    # Interpolate to fine grid
    field_fine = interp(x_fine, y_fine)
    
    return field_fine


def compute_error_metrics(field_ref, field_test):
    """Compute error metrics between reference and test fields"""
    
    # L2 relative error
    l2_rel = np.linalg.norm(field_ref - field_test) / np.linalg.norm(field_ref)
    
    # RMSE
    rmse = np.sqrt(np.mean((field_ref - field_test)**2))
    
    # Max absolute error
    max_err = np.abs(field_ref - field_test).max()
    
    # Mean absolute error
    mae = np.mean(np.abs(field_ref - field_test))
    
    return {
        'l2_rel': l2_rel * 100,  # Convert to percentage
        'rmse': rmse,
        'max_err': max_err,
        'mae': mae
    }


def compare_rans_versions(re_value: int, output_dir: str = 'results/rans_validation'):
    """Compare old vs corrected RANS for given Re"""
    
    print(f"\n{'='*80}")
    print(f"Re = {re_value} Comparison")
    print(f"{'='*80}\n")
    
    # File paths
    dns_file = f'data/kolmogorov_dns/dns_re{re_value}_t100.h5'
    rans_old_file = f'data/lowfi/kolmogorov_rans/rans_re{re_value}_kf4.h5'
    rans_new_file = f'data/lowfi/kolmogorov_rans/rans_re{re_value}_kf4_corrected.h5'
    
    # Load DNS (last snapshot)
    print("Loading DNS data...")
    u_dns, v_dns, p_dns, x_dns, y_dns = load_dns_snapshot(dns_file, time_index=-1)
    print(f"  DNS: {u_dns.shape[0]}×{u_dns.shape[1]} grid")
    
    # Load old RANS
    print("Loading old RANS data...")
    try:
        u_rans_old, v_rans_old, x_rans_old, y_rans_old = load_rans_mean(rans_old_file)
        print(f"  RANS (old): {u_rans_old.shape[0]}×{u_rans_old.shape[1]} grid")
        has_old = True
    except Exception as e:
        print(f"  ⚠️  Could not load old RANS: {e}")
        has_old = False
    
    # Load corrected RANS
    print("Loading corrected RANS data...")
    try:
        u_rans_new, v_rans_new, x_rans_new, y_rans_new = load_rans_mean(rans_new_file)
        print(f"  RANS (corrected): {u_rans_new.shape[0]}×{u_rans_new.shape[1]} grid")
        has_new = True
    except Exception as e:
        print(f"  ❌ Could not load corrected RANS: {e}")
        has_new = False
        return
    
    # Interpolate RANS to DNS grid if needed
    print("\nInterpolating to DNS grid...")
    
    if has_old:
        if u_rans_old.shape != u_dns.shape:
            u_rans_old_interp = interpolate_to_grid(u_rans_old, (x_rans_old, y_rans_old), (x_dns, y_dns))
            v_rans_old_interp = interpolate_to_grid(v_rans_old, (x_rans_old, y_rans_old), (x_dns, y_dns))
        else:
            u_rans_old_interp = u_rans_old
            v_rans_old_interp = v_rans_old
    
    if has_new:
        if u_rans_new.shape != u_dns.shape:
            u_rans_new_interp = interpolate_to_grid(u_rans_new, (x_rans_new, y_rans_new), (x_dns, y_dns))
            v_rans_new_interp = interpolate_to_grid(v_rans_new, (x_rans_new, y_rans_new), (x_dns, y_dns))
        else:
            u_rans_new_interp = u_rans_new
            v_rans_new_interp = v_rans_new
    
    # Compute error metrics
    print("\n" + "="*80)
    print("Error Metrics: RANS vs DNS")
    print("="*80)
    
    print(f"\n{'Metric':<20} {'Old RANS':<20} {'Corrected RANS':<20} {'Improvement':<15}")
    print("-"*80)
    
    if has_old:
        err_u_old = compute_error_metrics(u_dns, u_rans_old_interp)
        err_v_old = compute_error_metrics(v_dns, v_rans_old_interp)
    else:
        err_u_old = None
        err_v_old = None
    
    if has_new:
        err_u_new = compute_error_metrics(u_dns, u_rans_new_interp)
        err_v_new = compute_error_metrics(v_dns, v_rans_new_interp)
    else:
        err_u_new = None
        err_v_new = None
    
    # U-velocity errors
    print("\nU-velocity:")
    if err_u_old and err_u_new:
        for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
            old_val = err_u_old[key]
            new_val = err_u_new[key]
            
            if key == 'l2_rel':
                improvement = f"{old_val:.2f}% → {new_val:.2f}%"
                reduction = (old_val - new_val) / old_val * 100
                improvement += f" ({reduction:.1f}% ↓)"
            else:
                improvement = f"{old_val:.6f} → {new_val:.6f}"
                reduction = (old_val - new_val) / old_val * 100
                improvement += f" ({reduction:.1f}% ↓)"
            
            print(f"  {key:<18} {old_val:<20.6f} {new_val:<20.6f} {improvement:<15}")
    elif err_u_new:
        for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
            new_val = err_u_new[key]
            print(f"  {key:<18} {'N/A':<20} {new_val:<20.6f} {'N/A':<15}")
    
    # V-velocity errors
    print("\nV-velocity:")
    if err_v_old and err_v_new:
        for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
            old_val = err_v_old[key]
            new_val = err_v_new[key]
            
            if key == 'l2_rel':
                improvement = f"{old_val:.2f}% → {new_val:.2f}%"
                reduction = (old_val - new_val) / old_val * 100
                improvement += f" ({reduction:.1f}% ↓)"
            else:
                improvement = f"{old_val:.6f} → {new_val:.6f}"
                reduction = (old_val - new_val) / old_val * 100
                improvement += f" ({reduction:.1f}% ↓)"
            
            print(f"  {key:<18} {old_val:<20.6f} {new_val:<20.6f} {improvement:<15}")
    elif err_v_new:
        for key in ['l2_rel', 'rmse', 'mae', 'max_err']:
            new_val = err_v_new[key]
            print(f"  {key:<18} {'N/A':<20} {new_val:<20.6f} {'N/A':<15}")
    
    # Check physical consistency
    print("\n" + "="*80)
    print("Physical Consistency Checks")
    print("="*80)
    
    print(f"\nDNS statistics:")
    print(f"  U_mean = {u_dns.mean():.6e}, U_std = {u_dns.std():.6f}, U_max = {u_dns.max():.6f}")
    print(f"  V_mean = {v_dns.mean():.6e}, V_std = {v_dns.std():.6f}, V_max = {v_dns.max():.6f}")
    
    if has_old:
        print(f"\nRANS (old) statistics:")
        print(f"  U_mean = {u_rans_old_interp.mean():.6e}, U_std = {u_rans_old_interp.std():.6f}, U_max = {u_rans_old_interp.max():.6f}")
        print(f"  V_mean = {v_rans_old_interp.mean():.6e}, V_std = {v_rans_old_interp.std():.6f}, V_max = {v_rans_old_interp.max():.6f}")
        
        if v_rans_old_interp.max() < 1e-4:
            print(f"  ⚠️  V-velocity essentially zero (laminar solution)")
    
    if has_new:
        print(f"\nRANS (corrected) statistics:")
        print(f"  U_mean = {u_rans_new_interp.mean():.6e}, U_std = {u_rans_new_interp.std():.6f}, U_max = {u_rans_new_interp.max():.6f}")
        print(f"  V_mean = {v_rans_new_interp.mean():.6e}, V_std = {v_rans_new_interp.std():.6f}, V_max = {v_rans_new_interp.max():.6f}")
        
        if v_rans_new_interp.max() > 1e-3:
            print(f"  ✅ V-velocity present (expected for turbulent flow)")
        else:
            print(f"  ⚠️  V-velocity still very small")
    
    # Generate comparison plots
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print("="*80)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # DNS
    im0 = axes[0, 0].contourf(x_dns, y_dns, u_dns, levels=20, cmap='RdBu_r')
    axes[0, 0].set_title(f'DNS Re={re_value}: U')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[1, 0].contourf(x_dns, y_dns, v_dns, levels=20, cmap='RdBu_r')
    axes[1, 0].set_title(f'DNS Re={re_value}: V')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[1, 0])
    
    # Old RANS
    if has_old:
        im2 = axes[0, 1].contourf(x_dns, y_dns, u_rans_old_interp, levels=20, cmap='RdBu_r')
        axes[0, 1].set_title(f'RANS (old, ν×1.5): U')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1])
        
        im3 = axes[1, 1].contourf(x_dns, y_dns, v_rans_old_interp, levels=20, cmap='RdBu_r')
        axes[1, 1].set_title(f'RANS (old): V')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im3, ax=axes[1, 1])
    else:
        axes[0, 1].text(0.5, 0.5, 'Old RANS\nNot Available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        axes[1, 1].text(0.5, 0.5, 'Old RANS\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
    
    # Corrected RANS
    if has_new:
        im4 = axes[0, 2].contourf(x_dns, y_dns, u_rans_new_interp, levels=20, cmap='RdBu_r')
        axes[0, 2].set_title(f'RANS (corrected): U')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        plt.colorbar(im4, ax=axes[0, 2])
        
        im5 = axes[1, 2].contourf(x_dns, y_dns, v_rans_new_interp, levels=20, cmap='RdBu_r')
        axes[1, 2].set_title(f'RANS (corrected): V')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('y')
        plt.colorbar(im5, ax=axes[1, 2])
    else:
        axes[0, 2].text(0.5, 0.5, 'Corrected RANS\nNot Available', 
                       ha='center', va='center', transform=axes[0, 2].transAxes)
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        axes[1, 2].text(0.5, 0.5, 'Corrected RANS\nNot Available', 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
    
    plt.tight_layout()
    
    fig_path = output_path / f're{re_value}_rans_comparison.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")
    plt.close()
    
    print(f"\n{'='*80}\n")
    
    return {
        'err_u_old': err_u_old,
        'err_v_old': err_v_old,
        'err_u_new': err_u_new,
        'err_v_new': err_v_new
    }


def main():
    parser = argparse.ArgumentParser(description='驗證 RANS 修正效果')
    parser.add_argument('--re', type=int, nargs='+', default=[50, 100, 500],
                       help='雷諾數列表 (預設: 50 100 500)')
    parser.add_argument('--output-dir', type=str, default='results/rans_validation',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    print("="*80)
    print("RANS 修正驗證腳本")
    print("="*80)
    print(f"\n目標雷諾數: {args.re}")
    print(f"輸出目錄: {args.output_dir}\n")
    
    all_results = {}
    
    for re_val in args.re:
        try:
            results = compare_rans_versions(re_val, args.output_dir)
            all_results[re_val] = results
        except Exception as e:
            print(f"\n❌ Error processing Re={re_val}: {e}\n")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Error Reduction")
    print("="*80)
    
    print(f"\n{'Re':<10} {'Old U L2%':<15} {'New U L2%':<15} {'Improvement':<15}")
    print("-"*60)
    
    for re_val, results in all_results.items():
        if results['err_u_old'] and results['err_u_new']:
            old_err = results['err_u_old']['l2_rel']
            new_err = results['err_u_new']['l2_rel']
            reduction = (old_err - new_err) / old_err * 100
            print(f"{re_val:<10} {old_err:<15.2f} {new_err:<15.2f} {reduction:>13.1f}%")
        elif results['err_u_new']:
            new_err = results['err_u_new']['l2_rel']
            print(f"{re_val:<10} {'N/A':<15} {new_err:<15.2f} {'N/A':<15}")
    
    print("\n" + "="*80)
    print("✅ Validation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
