#!/usr/bin/env python3
"""
生成 Kolmogorov Flow RANS vs DNS 能量譜比較圖

用途：展示 RANS 作為 low-fidelity baseline 無法重現 DNS 的能量級聯特性
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_dns_snapshot(file_path):
    """載入 DNS 最終快照"""
    import re
    with h5py.File(file_path, 'r') as f:
        u = np.array(f['u'][-1])  # 最後一個時間步
        v = np.array(f['v'][-1])
        
        # 從文件名推斷 Re（使用專案的 forcing-scale 定義）
        # 專案使用 Re = U * (L/k_f) / nu ≈ 40-50，而非標準定義 Re = U * L^(3/2) / nu ≈ 400
        match = re.search(r'_re(\d+)_', str(file_path))
        Re = float(match.group(1)) if match else 50.0
        
    return u, v, float(Re)

def load_rans_field(file_path):
    """載入 RANS 穩態場"""
    import re
    with h5py.File(file_path, 'r') as f:
        # RANS 文件結構: mean_field/u, mean_field/v
        u = np.array(f['mean_field/u'])
        v = np.array(f['mean_field/v'])
        
        # 嘗試從參數組讀取 Re（如果沒有則從文件名推斷）
        Re = None
        if 'parameters' in f:
            params = f['parameters']
            Re = params.attrs.get('Re', None)
            if Re is None:
                Re = params.attrs.get('reynolds_number', None)
        
        # 從文件名推斷（備用方案）
        if Re is None:
            match = re.search(r'_re(\d+)_', str(file_path))
            Re = float(match.group(1)) if match else 50.0
        
    return u, v, float(Re)

def compute_energy_spectrum(u, v):
    """
    計算 2D 能量譜 E(k)
    
    Args:
        u, v: 速度場 (Ny, Nx)
    
    Returns:
        k_bins: 波數
        E_k: 能量譜
    """
    N = u.shape[0]
    
    # 2D FFT
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    # 能量密度（已歸一化）
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    
    # 波數網格
    kx = np.fft.fftfreq(N, 1.0/N)
    ky = np.fft.fftfreq(N, 1.0/N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # 等值波數 binning
    k_bins = np.arange(1, N//2)  # 從 k=1 開始
    E_k = np.zeros(len(k_bins))
    
    for i, k_val in enumerate(k_bins):
        mask = (K >= k_val - 0.5) & (K < k_val + 0.5)
        E_k[i] = E_hat[mask].sum()
    
    return k_bins, E_k

def main():
    parser = argparse.ArgumentParser(description='Generate RANS vs DNS spectrum comparison')
    parser.add_argument('--re', type=int, default=50, choices=[50, 100, 500],
                        help='Reynolds number')
    parser.add_argument('--output', type=str, 
                        default='thesis/result_figures/kolmogorov/rans_dns_spectrum_re50.png',
                        help='Output figure path')
    parser.add_argument('--kf', type=int, default=4, help='Forcing wavenumber')
    args = parser.parse_args()
    
    Re = args.re
    k_f = args.kf
    
    # 數據檔案路徑
    dns_file = f'data/kolmogorov_dns/dns_re{Re}_t100.h5'
    rans_file = f'data/lowfi/kolmogorov_rans/rans_re{Re}_kf{k_f}.h5'
    
    print("="*70)
    print(f"📊 生成 RANS vs DNS 能量譜比較圖 (Re={Re})")
    print("="*70)
    
    # 載入數據
    print(f"\n📂 載入數據...")
    print(f"   DNS:  {dns_file}")
    u_dns, v_dns, Re_dns = load_dns_snapshot(dns_file)
    print(f"   ✅ DNS loaded: Re={Re_dns:.1f}, shape={u_dns.shape}")
    
    print(f"   RANS: {rans_file}")
    u_rans, v_rans, Re_rans = load_rans_field(rans_file)
    print(f"   ✅ RANS loaded: Re={Re_rans:.1f}, shape={u_rans.shape}")
    
    # 計算能量譜
    print(f"\n🔬 計算能量譜...")
    k_dns, E_dns = compute_energy_spectrum(u_dns, v_dns)
    k_rans, E_rans = compute_energy_spectrum(u_rans, v_rans)
    
    # 計算 TKE
    tke_dns = 0.5 * (np.mean(u_dns**2) + np.mean(v_dns**2))
    tke_rans = 0.5 * (np.mean(u_rans**2) + np.mean(v_rans**2))
    print(f"   DNS TKE:  {tke_dns:.6f}")
    print(f"   RANS TKE: {tke_rans:.6f}")
    print(f"   差異: {abs(tke_dns - tke_rans)/tke_dns * 100:.1f}%")
    
    # 繪圖
    print(f"\n🎨 生成圖表...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # DNS 能量譜
    ax.loglog(k_dns, E_dns, 'b-', linewidth=2.5, label=f'DNS (Re={Re_dns:.0f})', 
              alpha=0.9, zorder=3)
    
    # RANS 能量譜
    ax.loglog(k_rans, E_rans, 'r--', linewidth=2, label=f'RANS (Re={Re_rans:.0f})', 
              alpha=0.8, zorder=2)
    
    # Kraichnan 理論線
    # 逆級串: k^(-5/3) for k < k_f
    k_inv = k_dns[(k_dns > 2) & (k_dns < k_f)]
    if len(k_inv) > 0:
        E_inv = E_dns[2] * (k_inv / k_dns[2])**(-5/3)
        ax.loglog(k_inv, E_inv, 'k:', linewidth=2, 
                 label=r'Theory: $k^{-5/3}$ (Inverse cascade)', alpha=0.7)
    
    # 正向級串: k^(-3) for k > k_f
    k_fwd = k_dns[(k_dns > k_f+1) & (k_dns < k_f+15)]
    if len(k_fwd) > 0:
        idx = np.argmin(np.abs(k_dns - (k_f+2)))
        E_fwd = E_dns[idx] * (k_fwd / k_dns[idx])**(-3)
        ax.loglog(k_fwd, E_fwd, 'k--', linewidth=2, 
                 label=r'Theory: $k^{-3}$ (Forward cascade)', alpha=0.7)
    
    # Forcing wavenumber
    ax.axvline(k_f, color='orange', linestyle=':', linewidth=3, 
              label=f'Forcing $k_f={k_f}$', zorder=4, alpha=0.8)
    
    # 裝飾
    ax.set_xlabel('Wavenumber $k$', fontsize=14, fontweight='bold')
    ax.set_ylabel('Energy Spectrum $E(k)$', fontsize=14, fontweight='bold')
    ax.set_title(f'RANS vs DNS Energy Spectrum Comparison\n2D Kolmogorov Flow (Re={Re})', 
                fontsize=15, fontweight='bold', pad=15)
    ax.legend(fontsize=12, loc='best', framealpha=0.95, edgecolor='black')
    ax.grid(True, which='both', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xlim([1, len(k_dns)//3])
    
    # 添加文字說明
    textstr = f'DNS TKE: {tke_dns:.4f}\nRANS TKE: {tke_rans:.4f}\nError: {abs(tke_dns-tke_rans)/tke_dns*100:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✅ 圖表已保存: {output_path}")
    print(f"   檔案大小: {output_path.stat().st_size / 1024:.1f} KB")
    
    plt.close()
    
    print("\n" + "="*70)
    print("✅ 完成")
    print("="*70)
    
    # 分析差異
    print(f"\n📈 能量譜差異分析:")
    
    # 計算相對誤差（在重疊的波數範圍）
    k_min = max(k_dns.min(), k_rans.min())
    k_max = min(k_dns.max(), k_rans.max())
    mask_dns = (k_dns >= k_min) & (k_dns <= k_max) & (E_dns > 0)
    mask_rans = (k_rans >= k_min) & (k_rans <= k_max) & (E_rans > 0)
    
    # 插值到相同波數
    from scipy.interpolate import interp1d
    f_rans = interp1d(k_rans[mask_rans], E_rans[mask_rans], 
                      kind='linear', fill_value='extrapolate')
    E_rans_interp = f_rans(k_dns[mask_dns])
    
    rel_error = np.abs(E_rans_interp - E_dns[mask_dns]) / E_dns[mask_dns]
    mean_error = np.mean(rel_error) * 100
    
    print(f"   平均相對誤差: {mean_error:.1f}%")
    print(f"   最大相對誤差: {np.max(rel_error) * 100:.1f}%")
    
    # 檢查級聯斜率
    from scipy.stats import linregress
    
    # DNS 逆級串斜率
    mask_inv_dns = (k_dns >= 2) & (k_dns <= k_f-1) & (E_dns > 0)
    if np.sum(mask_inv_dns) >= 3:
        res = linregress(np.log10(k_dns[mask_inv_dns]), np.log10(E_dns[mask_inv_dns]))
        print(f"\n   DNS 逆級串斜率: {res.slope:.3f} (理論: -1.667)")
    
    # RANS 逆級串斜率
    mask_inv_rans = (k_rans >= 2) & (k_rans <= k_f-1) & (E_rans > 0)
    if np.sum(mask_inv_rans) >= 3:
        res = linregress(np.log10(k_rans[mask_inv_rans]), np.log10(E_rans[mask_inv_rans]))
        print(f"   RANS 逆級串斜率: {res.slope:.3f} (理論: -1.667)")
        print(f"   → RANS 未能正確捕捉逆級串特性" if abs(res.slope + 5/3) > 0.3 else "   → RANS 近似逆級串")

if __name__ == '__main__':
    main()
