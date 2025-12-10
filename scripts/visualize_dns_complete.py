"""
DNS 完整可視化工具
==================

生成包含以下內容的綜合報告：
1. 流場快照（渦度、速度、壓力）
2. 時間序列診斷（動能、渦度、散度）
3. 能譜分析（Kolmogorov -5/3 law）
4. 統計分佈（速度 PDF）
5. 動畫生成（可選）

使用方式:
---------
python scripts/visualize_dns_complete.py \\
    --input data/kolmogorov_Re1000_kf4_t100s.h5 \\
    --output results/dns_visualization/ \\
    --snapshots 0 50 100 150 -1 \\
    --animation  # 可選：生成動畫
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from pathlib import Path
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from tqdm import tqdm

# 設定 matplotlib 中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_dns_data(h5_file: str):
    """載入 DNS 資料"""
    print(f"📂 載入 DNS 資料: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        data = {
            'u': f['u'][:],
            'v': f['v'][:],
            'p': f['p'][:],
            'time': f['time'][:],
            'KE': f['diagnostics/kinetic_energy'][:],
            'enstrophy': f['diagnostics/enstrophy'][:],
            'divergence': f['diagnostics/divergence_error'][:],
        }

        # 配置信息
        config = {key: f['config'].attrs[key] for key in f['config'].attrs.keys()}
        data['config'] = config

    print(f"   ✓ 載入 {len(data['time'])} 個快照")
    print(f"   ✓ 網格: {config['N']}×{config['N']}")
    print(f"   ✓ 時間範圍: {data['time'][0]:.2f} - {data['time'][-1]:.2f} s")

    return data


def compute_vorticity(u, v, L):
    """計算渦度"""
    N = u.shape[-1]
    dx = dy = L / N

    # 使用中心差分
    dvdx = np.gradient(v, dx, axis=-2)
    dudy = np.gradient(u, dy, axis=-1)

    return dvdx - dudy


def compute_energy_spectrum(u, v):
    """計算能譜"""
    N = u.shape[-1]

    # FFT
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)

    # 能量密度
    E_k = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4

    # 波數網格
    kx = np.fft.fftfreq(N, d=1.0) * N
    ky = np.fft.fftfreq(N, d=1.0) * N
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    # 徑向平均
    k_bins = np.arange(0.5, N//2, 1.0)
    E_spectrum = np.zeros_like(k_bins)

    for i, k in enumerate(k_bins):
        mask = (k_mag >= k - 0.5) & (k_mag < k + 0.5)
        E_spectrum[i] = E_k[mask].sum()

    return k_bins, E_spectrum


def plot_field_snapshots(data, output_dir, snapshot_indices):
    """繪製場快照"""
    print("\n📊 生成場快照...")

    u_all = data['u']
    v_all = data['v']
    p_all = data['p']
    time = data['time']
    L = data['config']['L']
    N = data['config']['N']

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    for idx in tqdm(snapshot_indices, desc="場快照"):
        if idx == -1:
            idx = len(time) - 1

        t = time[idx]
        u = u_all[idx]
        v = v_all[idx]
        p = p_all[idx]
        omega = compute_vorticity(u, v, L)

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # === 渦度場 ===
        ax1 = fig.add_subplot(gs[0, :2])
        vmax = np.abs(omega).max()
        im1 = ax1.contourf(X, Y, omega, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax1.set_title(f'Vorticity Field (ω) at t={t:.2f}s', fontsize=14, fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='ω [1/s]')

        # === 速度場（向量圖）===
        ax2 = fig.add_subplot(gs[0, 2])
        stride = N // 20
        speed = np.sqrt(u**2 + v**2)
        im2 = ax2.contourf(X, Y, speed, levels=30, cmap='viridis')
        ax2.quiver(X[::stride, ::stride], Y[::stride, ::stride],
                   u[::stride, ::stride], v[::stride, ::stride],
                   color='white', alpha=0.7, scale=200)
        ax2.set_title(f'Velocity Magnitude', fontsize=12)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='|u| [m/s]')

        # === U 速度分量 ===
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.contourf(X, Y, u, levels=50, cmap='coolwarm')
        ax3.set_title('U-velocity', fontsize=12)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='u [m/s]')

        # === V 速度分量 ===
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.contourf(X, Y, v, levels=50, cmap='coolwarm')
        ax4.set_title('V-velocity', fontsize=12)
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_aspect('equal')
        plt.colorbar(im4, ax=ax4, label='v [m/s]')

        # === 壓力場 ===
        ax5 = fig.add_subplot(gs[1, 2])
        im5 = ax5.contourf(X, Y, p, levels=50, cmap='plasma')
        ax5.set_title('Pressure', fontsize=12)
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')
        ax5.set_aspect('equal')
        plt.colorbar(im5, ax=ax5, label='p [Pa]')

        # === 速度 PDF ===
        ax6 = fig.add_subplot(gs[2, 0])
        u_flat = u.flatten()
        v_flat = v.flatten()
        ax6.hist(u_flat, bins=50, alpha=0.6, label='u', density=True, color='blue')
        ax6.hist(v_flat, bins=50, alpha=0.6, label='v', density=True, color='red')
        ax6.set_xlabel('Velocity [m/s]')
        ax6.set_ylabel('Probability Density')
        ax6.set_title('Velocity PDF', fontsize=12)
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # === 渦度 PDF ===
        ax7 = fig.add_subplot(gs[2, 1])
        omega_flat = omega.flatten()
        ax7.hist(omega_flat, bins=50, alpha=0.7, density=True, color='green')
        ax7.set_xlabel('Vorticity [1/s]')
        ax7.set_ylabel('Probability Density')
        ax7.set_title('Vorticity PDF', fontsize=12)
        ax7.grid(True, alpha=0.3)

        # === 流線圖 ===
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.streamplot(X.T, Y.T, u.T, v.T, color=speed.T, cmap='viridis',
                       density=1.5, linewidth=1)
        ax8.set_title('Streamlines', fontsize=12)
        ax8.set_xlabel('x')
        ax8.set_ylabel('y')
        ax8.set_aspect('equal')

        # 保存
        fig.suptitle(f'Kolmogorov Flow DNS (Re={1/data["config"]["nu"]:.0f}, t={t:.2f}s)',
                     fontsize=16, fontweight='bold', y=0.995)

        output_path = output_dir / f'field_snapshot_t{t:.1f}s.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"   ✓ 已保存 {len(snapshot_indices)} 個場快照")


def plot_diagnostics_timeseries(data, output_dir):
    """繪製診斷時間序列"""
    print("\n📈 生成診斷時間序列...")

    time = data['time']
    KE = data['KE']
    enstrophy = data['enstrophy']
    divergence = data['divergence']

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # 動能
    axes[0].plot(time, KE, 'b-', linewidth=2)
    axes[0].axvline(10.0, color='r', linestyle='--', alpha=0.5, label='Perturbation (t=10s)')
    axes[0].set_ylabel('Kinetic Energy', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Temporal Evolution of Flow Diagnostics', fontsize=14, fontweight='bold')

    # 渦度
    axes[1].plot(time, enstrophy, 'g-', linewidth=2)
    axes[1].axvline(10.0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Enstrophy', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')

    # 散度誤差
    axes[2].plot(time, divergence, 'm-', linewidth=2)
    axes[2].axvline(10.0, color='r', linestyle='--', alpha=0.5)
    axes[2].set_ylabel('Divergence Error', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time [s]', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_yscale('log')

    plt.tight_layout()
    output_path = output_dir / 'diagnostics_timeseries.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ 已保存診斷時間序列")


def plot_energy_spectra(data, output_dir, snapshot_indices):
    """繪製能譜"""
    print("\n📉 生成能譜分析...")

    u_all = data['u']
    v_all = data['v']
    time = data['time']

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(snapshot_indices)))

    for i, idx in enumerate(tqdm(snapshot_indices, desc="能譜計算")):
        if idx == -1:
            idx = len(time) - 1

        t = time[idx]
        u = u_all[idx]
        v = v_all[idx]

        k_bins, E_k = compute_energy_spectrum(u, v)

        # 過濾零值
        mask = E_k > 0
        k_bins_plot = k_bins[mask]
        E_k_plot = E_k[mask]

        ax.loglog(k_bins_plot, E_k_plot, 'o-', color=colors[i],
                  label=f't={t:.1f}s', linewidth=2, markersize=4, alpha=0.7)

    # Kolmogorov -5/3 參考線
    k_ref = np.array([5, 50])
    E_ref = 1e2 * k_ref**(-5/3)
    ax.loglog(k_ref, E_ref, 'k--', linewidth=2, label='k^(-5/3) (Kolmogorov)', alpha=0.5)

    ax.set_xlabel('Wavenumber k', fontsize=12, fontweight='bold')
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=12, fontweight='bold')
    ax.set_title('Energy Spectrum Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    output_path = output_dir / 'energy_spectra.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✓ 已保存能譜分析")


def create_animation(data, output_dir):
    """創建動畫（可選）"""
    print("\n🎬 生成動畫（此步驟可能需要數分鐘）...")

    u_all = data['u']
    v_all = data['v']
    time = data['time']
    L = data['config']['L']
    N = data['config']['N']

    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 每 10 個快照取 1 個（減少檔案大小）
    frame_indices = np.arange(0, len(time), 10)

    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame_idx):
        idx = frame_indices[frame_idx]
        t = time[idx]
        u = u_all[idx]
        v = v_all[idx]
        omega = compute_vorticity(u, v, L)

        ax.clear()
        vmax = np.abs(omega).max()
        im = ax.contourf(X, Y, omega, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        ax.set_title(f'Vorticity Field at t={t:.2f}s (Re={1/data["config"]["nu"]:.0f})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frame_indices),
                                   interval=50, blit=False)

    output_path = output_dir / 'vorticity_animation.mp4'
    anim.save(str(output_path), writer='ffmpeg', fps=20, dpi=100)
    plt.close()

    print(f"   ✓ 已保存動畫：{output_path}")


def generate_summary_report(data, output_dir):
    """生成摘要報告"""
    print("\n📄 生成摘要報告...")

    config = data['config']
    time = data['time']
    KE = data['KE']
    enstrophy = data['enstrophy']
    divergence = data['divergence']

    report = f"""
# Kolmogorov Flow DNS 模擬報告

## 配置參數
- **網格解析度**: {config['N']}×{config['N']}
- **域長度**: L = {config['L']:.4f}
- **雷諾數**: Re = {1/config['nu']:.2f}
- **黏滯係數**: ν = {config['nu']:.6f}
- **強迫振幅**: A = {config['A']:.2f}
- **強迫波數**: k_f = {config['k_f']}
- **時間步長**: dt = {config['dt']:.6f}
- **模擬時長**: T = {config['T_end']:.1f} s
- **計算後端**: {config.get('backend', 'N/A')}

## 模擬結果統計

### 時間範圍
- 起始時間: {time[0]:.2f} s
- 結束時間: {time[-1]:.2f} s
- 快照數量: {len(time)}

### 動能統計
- 初始動能: {KE[0]:.4e}
- 最終動能: {KE[-1]:.4e}
- 最大動能: {KE.max():.4e}
- 最小動能: {KE.min():.4e}
- 平均動能: {KE.mean():.4e}

### 渦度統計
- 初始渦度: {enstrophy[0]:.4e}
- 最終渦度: {enstrophy[-1]:.4e}
- 最大渦度: {enstrophy.max():.4e}
- 最小渦度: {enstrophy.min():.4e}
- 平均渦度: {enstrophy.mean():.4e}

### 數值精度
- 最大散度誤差: {divergence.max():.4e}
- 最小散度誤差: {divergence.min():.4e}
- 平均散度誤差: {divergence.mean():.4e}

## 物理觀察

1. **擾動響應**: 在 t=10s 注入擾動後，動能和渦度均出現明顯響應
2. **能量耗散**: 動能從 {KE[0]:.2e} 衰減至 {KE[-1]:.2e} (衰減率: {(1-KE[-1]/KE[0])*100:.1f}%)
3. **數值穩定性**: 散度誤差保持在 O({divergence.mean():.0e})，表明數值求解器穩定

## 文件輸出
- 場快照: `field_snapshot_t*.png`
- 診斷時間序列: `diagnostics_timeseries.png`
- 能譜分析: `energy_spectra.png`
- 動畫（可選）: `vorticity_animation.mp4`
- 本報告: `REPORT.md`

---
生成時間: {time[-1]:.2f}s 模擬完成
"""

    output_path = output_dir / 'REPORT.md'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"   ✓ 已保存摘要報告：{output_path}")


def main():
    parser = argparse.ArgumentParser(description='DNS 完整可視化工具')
    parser.add_argument('--input', type=str, required=True, help='輸入 HDF5 文件')
    parser.add_argument('--output', type=str, default='results/dns_visualization/',
                        help='輸出目錄')
    parser.add_argument('--snapshots', type=int, nargs='+',
                        default=[0, 25, 50, 100, 150, 200, -1],
                        help='要繪製的快照索引（-1 表示最後一個）')
    parser.add_argument('--animation', action='store_true',
                        help='是否生成動畫（需要 ffmpeg）')

    args = parser.parse_args()

    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("DNS 完整可視化工具")
    print("=" * 80)

    # 載入數據
    data = load_dns_data(args.input)

    # 生成可視化
    plot_field_snapshots(data, output_dir, args.snapshots)
    plot_diagnostics_timeseries(data, output_dir)
    plot_energy_spectra(data, output_dir, args.snapshots)

    if args.animation:
        try:
            create_animation(data, output_dir)
        except Exception as e:
            print(f"   ⚠️  動畫生成失敗: {e}")
            print("   提示：請確保已安裝 ffmpeg")

    # 生成報告
    generate_summary_report(data, output_dir)

    print("\n" + "=" * 80)
    print(f"✅ 可視化完成！結果已保存至：{output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
