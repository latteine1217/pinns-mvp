#!/usr/bin/env python3
"""
生成通道流能谱对比图 (PINN重建 vs DNS参考)
仅使用 log-log scale 作为误差分析
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def compute_2d_energy_spectrum(u, v):
    """
    计算2D能谱 E(k)
    
    Args:
        u: x方向速度 (Ny, Nx)
        v: y方向速度 (Ny, Nx)
    
    Returns:
        k_bins: 波数
        E_k: 能谱
    """
    Ny, Nx = u.shape
    N = min(Ny, Nx)
    
    # FFT变换
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    # 能量密度
    E_hat = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / (Ny * Nx)**2
    
    # 波数网格
    kx = np.fft.fftfreq(Nx, 1.0)
    ky = np.fft.fftfreq(Ny, 1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    # 径向平均
    k_max = N // 2
    k_bins = np.arange(1, k_max)  # 从k=1开始，避免k=0
    E_k = np.zeros(len(k_bins))
    
    for i, k_val in enumerate(k_bins):
        mask = (K >= k_val - 0.5) & (K < k_val + 0.5)
        if mask.sum() > 0:
            E_k[i] = E_hat[mask].sum()
    
    return k_bins, E_k


def generate_channel_spectrum_comparison(
    dns_data_path,
    pinn_data_path,
    output_path,
    re_tau=1000
):
    """
    生成通道流能谱对比图
    
    Args:
        dns_data_path: DNS数据路径 (.npz)
        pinn_data_path: PINN重建数据路径 (.npz)
        output_path: 输出图片路径
        re_tau: 摩擦雷诺数
    """
    # 加载DNS数据
    print(f"Loading DNS data from {dns_data_path}")
    dns = np.load(dns_data_path)
    
    # 检查可用字段
    print(f"DNS fields: {list(dns.keys())}")
    
    # 假设是固定z平面的切片 (y, x)
    if 'u' in dns:
        u_dns = dns['u']
        v_dns = dns['v']
    elif 'u_slice' in dns:
        u_dns = dns['u_slice']
        v_dns = dns['v_slice']
    else:
        raise ValueError(f"Cannot find velocity fields in DNS data. Available keys: {list(dns.keys())}")
    
    print(f"DNS velocity shape: {u_dns.shape}")
    
    # 加载PINN数据
    print(f"Loading PINN data from {pinn_data_path}")
    pinn = np.load(pinn_data_path)
    
    print(f"PINN fields: {list(pinn.keys())}")
    
    if 'u_pred' in pinn:
        u_pinn = pinn['u_pred']
        v_pinn = pinn['v_pred']
    elif 'u' in pinn:
        u_pinn = pinn['u']
        v_pinn = pinn['v']
    else:
        raise ValueError(f"Cannot find velocity fields in PINN data. Available keys: {list(pinn.keys())}")
    
    print(f"PINN velocity shape: {u_pinn.shape}")
    
    # 确保形状匹配
    if u_dns.shape != u_pinn.shape:
        print(f"Warning: Shape mismatch! DNS: {u_dns.shape}, PINN: {u_pinn.shape}")
        # 尝试调整PINN到DNS的形状
        if u_pinn.size == u_dns.size:
            u_pinn = u_pinn.reshape(u_dns.shape)
            v_pinn = v_pinn.reshape(v_dns.shape)
            print(f"Reshaped PINN to {u_pinn.shape}")
        else:
            raise ValueError(f"Cannot match shapes: DNS {u_dns.shape} vs PINN {u_pinn.shape}")
    
    # 计算能谱
    print("Computing DNS energy spectrum...")
    k_dns, E_dns = compute_2d_energy_spectrum(u_dns, v_dns)
    
    print("Computing PINN energy spectrum...")
    k_pinn, E_pinn = compute_2d_energy_spectrum(u_pinn, v_pinn)
    
    # 过滤零值（对数刻度）
    mask_dns = E_dns > 0
    mask_pinn = E_pinn > 0
    mask = mask_dns & mask_pinn
    
    k_plot = k_dns[mask]
    E_dns_plot = E_dns[mask]
    E_pinn_plot = E_pinn[mask]
    
    print(f"Valid spectral points: {mask.sum()} / {len(k_dns)}")
    
    # 计算误差
    relative_error = np.abs(E_pinn_plot - E_dns_plot) / (E_dns_plot + 1e-12)
    mean_error = np.mean(relative_error) * 100
    
    # 绘图
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Log-log scale
    ax.loglog(k_plot, E_dns_plot, 'o-', label='DNS Reference', 
              linewidth=2, markersize=4, alpha=0.8)
    ax.loglog(k_plot, E_pinn_plot, 's-', label='PINN Reconstruction', 
              linewidth=2, markersize=4, alpha=0.8)
    
    # 参考斜率 (Kolmogorov -5/3)
    k_ref = k_plot[len(k_plot)//3:2*len(k_plot)//3]
    if len(k_ref) > 0:
        E_ref = E_dns_plot[len(k_plot)//3] * (k_ref / k_plot[len(k_plot)//3])**(-5/3)
        ax.loglog(k_ref, E_ref, 'k--', label=r'$k^{-5/3}$ (Kolmogorov)', 
                  linewidth=1.5, alpha=0.5)
    
    ax.set_xlabel('Wavenumber $k$', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Spectrum $E(k)$', fontsize=12, fontweight='bold')
    ax.set_title(f'Energy Spectrum Comparison (Channel Flow, $Re_\\tau={re_tau}$)\n'
                 f'Mean Relative Error: {mean_error:.1f}%', 
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right', framealpha=0.9)
    ax.grid(True, which='both', alpha=0.3)
    
    # 文本框：统计信息
    stats_text = (
        f'DNS TKE: {0.5 * (np.mean(u_dns**2) + np.mean(v_dns**2)):.4f}\n'
        f'PINN TKE: {0.5 * (np.mean(u_pinn**2) + np.mean(v_pinn**2)):.4f}'
    )
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 保存
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to {output_path}")
    
    plt.close()
    
    return mean_error


def main():
    parser = argparse.ArgumentParser(
        description='Generate channel flow energy spectrum comparison (log-log scale)'
    )
    parser.add_argument('--dns', type=str, required=True,
                        help='Path to DNS reference data (.npz)')
    parser.add_argument('--pinn', type=str, required=True,
                        help='Path to PINN reconstruction data (.npz)')
    parser.add_argument('--output', type=str, 
                        default='thesis/result_figures/channel_flow/channel_energy_spectrum_comparison.png',
                        help='Output figure path')
    parser.add_argument('--re-tau', type=int, default=1000,
                        help='Friction Reynolds number')
    
    args = parser.parse_args()
    
    mean_error = generate_channel_spectrum_comparison(
        dns_data_path=args.dns,
        pinn_data_path=args.pinn,
        output_path=args.output,
        re_tau=args.re_tau
    )
    
    print(f"\n{'='*60}")
    print(f"Energy Spectrum Analysis Complete")
    print(f"{'='*60}")
    print(f"Mean Relative Error: {mean_error:.2f}%")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
