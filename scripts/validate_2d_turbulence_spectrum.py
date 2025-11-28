#!/usr/bin/env python3
"""
2D 湍流能量譜驗證工具
驗證 Kraichnan (1967) 雙級串理論
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress

def load_final_snapshot(file_path):
    with h5py.File(file_path, 'r') as f:
        u = f['u'][:]
        v = f['v'][:]
        nu = f['config'].attrs.get('nu', 0.01)
        A = f['config'].attrs.get('A', 1.0)
        L = f['config'].attrs.get('L', 2*np.pi)
        # 計算雷諾數: Re = sqrt(A) * L^(3/2) / nu
        Re = np.sqrt(A) * L**(3/2) / nu
    return u[-1], v[-1], float(Re), float(nu)

def compute_energy_spectrum(u, v):
    N = u.shape[0]
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    
    kx = np.fft.fftfreq(N, 1.0/N)
    ky = np.fft.fftfreq(N, 1.0/N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    k_bins = np.arange(0, N//2)
    E_k = np.zeros(len(k_bins))
    
    for i, k_val in enumerate(k_bins):
        mask = (K >= k_val - 0.5) & (K < k_val + 0.5)
        E_k[i] = E_hat[mask].sum()
    
    return k_bins, E_k

def fit_power_law(k, E, k_min, k_max):
    mask = (k >= k_min) & (k <= k_max) & (E > 0)
    k_fit = k[mask]
    E_fit = E[mask]
    
    if len(k_fit) < 3:
        return None, None, None
    
    res = linregress(np.log10(k_fit), np.log10(E_fit))
    return res.slope, res.intercept, res.rvalue

def main():
    files = {
        'Re100': 'data/kolmogorov_dns_re100_kf4_T100_pert10.h5',
        'Re500': 'data/kolmogorov_dns_re500_kf4_T100_pert10.h5'
    }
    
    output_dir = Path('results/spectrum_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    k_f = 4
    
    print("="*70)
    print("🔬 2D 湍流能量譜驗證工具 (Kraichnan 1967)")
    print("="*70)
    
    for name, fpath in files.items():
        print(f"\n{'='*70}")
        print(f"🔍 驗證: {name}")
        print(f"{'='*70}")
        
        u, v, Re, nu = load_final_snapshot(fpath)
        print(f"\n📊 物理參數:")
        print(f"   Re = {Re:.2f}")
        print(f"   ν = {nu:.6f}")
        print(f"   k_f = {k_f}")
        
        k, E = compute_energy_spectrum(u, v)
        
        # 逆級串擬合
        alpha_inv, intercept_inv, r_inv = fit_power_law(k, E, 2, k_f-1)
        if alpha_inv:
            err_inv = abs(alpha_inv + 5/3) / (5/3) * 100
            print(f"\n🔹 逆級串 (k < {k_f}):")
            print(f"   斜率: {alpha_inv:.3f} (理論: -1.667, 誤差: {err_inv:.1f}%)")
            print(f"   R²: {r_inv**2:.4f}")
            print(f"   {'✅ 通過' if err_inv < 15 else '⚠️ 偏差較大'}")
        else:
            print(f"\n🔹 逆級串 (k < {k_f}): ❌ 無法擬合")
        
        # 正向級串擬合
        alpha_fwd, intercept_fwd, r_fwd = fit_power_law(k, E, k_f+2, min(k_f+15, len(k)//3))
        if alpha_fwd:
            err_fwd = abs(alpha_fwd + 3.0) / 3.0 * 100
            print(f"\n🔹 正向級串 (k > {k_f}):")
            print(f"   斜率: {alpha_fwd:.3f} (理論: -3.000, 誤差: {err_fwd:.1f}%)")
            print(f"   R²: {r_fwd**2:.4f}")
            print(f"   {'✅ 通過' if err_fwd < 20 else '⚠️ 偏差較大'}")
        else:
            print(f"\n🔹 正向級串 (k > {k_f}): ❌ 無法擬合")
        
        # 繪圖
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.loglog(k[1:], E[1:], 'b-', lw=2.5, label=f'DNS (Re={Re:.0f})', zorder=3)
        
        if alpha_inv and intercept_inv:
            k_fit = k[(k >= 2) & (k <= k_f-1)]
            E_fit = 10**(alpha_inv * np.log10(k_fit) + intercept_inv)
            ax.loglog(k_fit, E_fit, 'r--', lw=2, 
                     label=f'Fit: $k^{{{alpha_inv:.2f}}}$ (R²={r_inv**2:.3f})')
            ax.axvspan(2, k_f-1, alpha=0.15, color='red')
        
        if alpha_fwd and intercept_fwd:
            k_max_fit = min(k_f+15, len(k)//3)
            k_fit = k[(k >= k_f+2) & (k <= k_max_fit)]
            E_fit = 10**(alpha_fwd * np.log10(k_fit) + intercept_fwd)
            ax.loglog(k_fit, E_fit, 'g--', lw=2, 
                     label=f'Fit: $k^{{{alpha_fwd:.2f}}}$ (R²={r_fwd**2:.3f})')
            ax.axvspan(k_f+2, k_max_fit, alpha=0.15, color='green')
        
        # 理論線
        k_th_inv = k[(k > 1) & (k < k_f)]
        if len(k_th_inv) > 0:
            E_th = E[2] * (k_th_inv / k[2])**(-5/3)
            ax.loglog(k_th_inv, E_th, 'r:', lw=2, label=r'Theory $k^{-5/3}$ (Inverse)')
        
        k_th_fwd = k[(k > k_f) & (k < 30)]
        if len(k_th_fwd) > 0:
            idx = np.argmin(np.abs(k - (k_f+3)))
            E_th = E[idx] * (k_th_fwd / k[idx])**(-3)
            ax.loglog(k_th_fwd, E_th, 'g:', lw=2, label=r'Theory $k^{-3}$ (Forward)')
        
        ax.axvline(k_f, color='orange', ls=':', lw=3, label=f'Forcing $k_f={k_f}$', zorder=4)
        ax.set_xlabel('Wavenumber k', fontsize=13)
        ax.set_ylabel('Energy Spectrum E(k)', fontsize=13)
        ax.set_title(f'2D Turbulence Spectrum Validation (Re={Re:.0f})', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim([1, len(k)//3])
        
        out_file = output_dir / f'validation_re{int(Re)}.png'
        plt.savefig(out_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n✅ 圖表: {out_file}")
    
    print(f"\n{'='*70}")
    print("✅ 驗證完成")
    print("="*70)

if __name__ == '__main__':
    main()
