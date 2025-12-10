#!/usr/bin/env python3
"""
多雷諾數 Kolmogorov Flow 比較視覺化工具
支援比較 2-4 個雷諾數的流場特性
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse


def load_dns_data(file_path):
    """載入 DNS 數據"""
    data = {}
    with h5py.File(file_path, 'r') as f:
        data['u'] = np.array(f['u'])
        data['v'] = np.array(f['v'])
        data['time'] = np.array(f['time'])
        if 'p' in f:
            data['p'] = np.array(f['p'])
        if 'parameters' in f:
            config_dict = {}
            for key in f['parameters'].attrs.keys():
                config_dict[key] = f['parameters'].attrs[key]
            data['config'] = config_dict
        elif 'config' in f:
            config_dict = {}
            for key in f['config'].attrs.keys():
                config_dict[key] = f['config'].attrs[key]
            data['config'] = config_dict
        else:
            data['config'] = {}
    return data


def compute_vorticity_2d(u, v, dx, dy):
    """計算 2D 渦度"""
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    return dv_dx - du_dy


def plot_final_state_comparison(data_list, labels, output_path):
    """比較多個雷諾數的最終狀態"""
    n_cases = len(data_list)
    fig = plt.figure(figsize=(20, 4 * n_cases))
    gs = GridSpec(n_cases, 3, figure=fig, hspace=0.3, wspace=0.25)

    for i, (data, label) in enumerate(zip(data_list, labels)):
        u = data['u'][-1]  # 最後時刻
        v = data['v'][-1]
        config = data.get('config', {})

        N = u.shape[0]
        L = config.get('L', 2 * np.pi)
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        X, Y = np.meshgrid(x, y)

        dx = dy = L / N
        vorticity = compute_vorticity_2d(u, v, dx, dy)
        speed = np.sqrt(u**2 + v**2)

        # V 速度場
        ax1 = fig.add_subplot(gs[i, 0])
        im1 = ax1.contourf(X, Y, v, levels=50, cmap='RdBu_r')
        ax1.set_title(f'{label}: V Velocity', fontsize=12, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='v')

        # 速度大小
        ax2 = fig.add_subplot(gs[i, 1])
        im2 = ax2.contourf(X, Y, speed, levels=50, cmap='viridis')
        skip = N // 20
        ax2.streamplot(X[::skip, ::skip], Y[::skip, ::skip],
                      u[::skip, ::skip], v[::skip, ::skip],
                      color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
        ax2.set_title(f'{label}: Speed', fontsize=12, fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='|u|')

        # 渦度場
        ax3 = fig.add_subplot(gs[i, 2])
        vort_max = np.abs(vorticity).max()
        im3 = ax3.contourf(X, Y, vorticity, levels=50, cmap='RdBu_r',
                          vmin=-vort_max, vmax=vort_max)
        ax3.set_title(f'{label}: Vorticity', fontsize=12, fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='ω')

    plt.suptitle('Kolmogorov Flow Comparison (t=100s)',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_path}')


def plot_temporal_evolution_comparison(data_list, labels, output_path):
    """比較多個雷諾數的時間演化"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    colors = ['purple', 'blue', 'green', 'red']

    for data, label, color in zip(data_list, labels, colors):
        time = data['time']
        u = data['u']
        v = data['v']

        # 計算動能
        ke = 0.5 * (u**2 + v**2)
        ke_mean = ke.mean(axis=(1, 2))

        # 計算渦量 RMS
        N = u.shape[1]
        L = data['config'].get('L', 2 * np.pi)
        dx = dy = L / N

        vorticity_rms = []
        for i in range(len(time)):
            vort = compute_vorticity_2d(u[i], v[i], dx, dy)
            vorticity_rms.append(np.sqrt((vort**2).mean()))
        vorticity_rms = np.array(vorticity_rms)

        # V 速度統計
        v_mean = v.mean(axis=(1, 2))
        v_std = v.std(axis=(1, 2))

        # 繪圖
        axes[0, 0].plot(time, ke_mean, linewidth=2, label=label, color=color)
        axes[0, 1].plot(time, vorticity_rms, linewidth=2, label=label, color=color)
        axes[1, 0].plot(time, v_mean, linewidth=2, label=label, color=color)
        axes[1, 1].plot(time, v_std, linewidth=2, label=label, color=color)

    # 設置標題和標籤
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Mean Kinetic Energy')
    axes[0, 0].set_title('Kinetic Energy Evolution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Vorticity RMS')
    axes[0, 1].set_title('Enstrophy Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('V Mean')
    axes[1, 0].set_title('V Velocity Mean')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel('Time')
    axes[1, 1].set_ylabel('V Std')
    axes[1, 1].set_title('V Velocity Fluctuation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Reynolds Number Comparison: Temporal Evolution',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_path}')


def plot_energy_spectrum_comparison(data_list, labels, output_path):
    """比較多個雷諾數的能量譜"""
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['purple', 'blue', 'green', 'red']

    for data, label, color in zip(data_list, labels, colors):
        u = data['u'][-1]
        v = data['v'][-1]
        config = data.get('config', {})
        k_f = config.get('k_f', 4)

        N = u.shape[0]

        # 計算 2D FFT
        u_fft = np.fft.fft2(u)
        v_fft = np.fft.fft2(v)

        # 能量密度
        E_k = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2) / N**4

        # 徑向平均
        kx = np.fft.fftfreq(N, 1.0/N)
        ky = np.fft.fftfreq(N, 1.0/N)
        KX, KY = np.meshgrid(kx, ky)
        K = np.sqrt(KX**2 + KY**2)

        k_bins = np.arange(1, N//2)
        E_spectrum = np.zeros(len(k_bins))

        for i, k in enumerate(k_bins):
            mask = (K >= k - 0.5) & (K < k + 0.5)
            E_spectrum[i] = E_k[mask].sum()

        # 繪製能量譜
        ax.loglog(k_bins, E_spectrum, linewidth=2.5, label=label,
                 color=color, zorder=3)

    # 強迫波數標註
    k_f = 4
    ax.axvline(x=k_f, color='orange', linestyle=':', linewidth=2.5,
              label=f'Forcing $k_f={k_f}$', zorder=1)

    # 理論參考線
    k_ref = k_bins[(k_bins > 1) & (k_bins < k_f)]
    if len(k_ref) > 0:
        E_ref = E_spectrum[1] * (k_ref / k_bins[1])**(-5/3)
        ax.loglog(k_ref, E_ref, 'k--', linewidth=2,
                 label=r'$k^{-5/3}$ (Inverse Cascade)', zorder=2)

    k_forward = k_bins[(k_bins > k_f) & (k_bins < N//3)]
    if len(k_forward) > 0:
        anchor_idx = np.argmin(np.abs(k_bins - (k_f + 2)))
        E_ref_forward = E_spectrum[anchor_idx] * (k_forward / k_bins[anchor_idx])**(-3)
        ax.loglog(k_forward, E_ref_forward, 'k:', linewidth=2,
                 label=r'$k^{-3}$ (Forward Cascade)', zorder=2)

    ax.set_xlabel('Wavenumber k', fontsize=13)
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=13)
    ax.set_title('Energy Spectrum Comparison (t=100s)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_xlim([1, N//3])

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Compare Reynolds numbers')
    parser.add_argument('--re50', type=str, default=None,
                       help='Re=50 DNS file (optional)')
    parser.add_argument('--re70', type=str, default=None,
                       help='Re=70 DNS file (optional)')
    parser.add_argument('--re100', type=str, default=None,
                       help='Re=100 DNS file (optional)')
    parser.add_argument('--re500', type=str, default=None,
                       help='Re=500 DNS file (optional)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory')

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集有效的輸入
    data_list = []
    labels = []

    if args.re50:
        data_list.append(load_dns_data(args.re50))
        labels.append('Re=50')
    if args.re70:
        data_list.append(load_dns_data(args.re70))
        labels.append('Re=70')
    if args.re100:
        data_list.append(load_dns_data(args.re100))
        labels.append('Re=100')
    if args.re500:
        data_list.append(load_dns_data(args.re500))
        labels.append('Re=500')

    if len(data_list) < 2:
        raise ValueError("至少需要提供兩個雷諾數的 DNS 檔案")

    print(f"{'='*70}")
    print(f"🔬 {len(data_list)} 雷諾數比較視覺化工具")
    print(f"{'='*70}")

    # 載入數據
    print(f"\n📥 載入 DNS 數據: {', '.join(labels)}...")

    print("✅ 數據載入完成")

    # 生成比較圖
    print("\n🎨 生成最終狀態比較...")
    plot_final_state_comparison(data_list, labels,
                                output_dir / 'final_state_comparison.png')

    print("\n📈 生成時間演化比較...")
    plot_temporal_evolution_comparison(data_list, labels,
                                      output_dir / 'temporal_evolution_comparison.png')

    print("\n📊 生成能量譜比較...")
    plot_energy_spectrum_comparison(data_list, labels,
                                    output_dir / 'energy_spectrum_comparison.png')

    print(f"\n{'='*70}")
    print(f"✅ 比較視覺化完成！所有圖表已儲存至: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
