#!/usr/bin/env python3
"""
Kolmogorov Flow DNS 數據視覺化工具

用途: 視覺化生成的 DNS 數據，包括速度場、能量演化、渦度分佈等
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path


def load_dns_data(file_path):
    """載入 DNS 數據"""
    data = {}
    with h5py.File(file_path, 'r') as f:
        data['u'] = np.array(f['u'])
        data['v'] = np.array(f['v'])
        data['time'] = np.array(f['time'])
        
        # 載入壓力場（如果存在）
        if 'p' in f:
            data['p'] = np.array(f['p'])
        
        # 載入配置
        if 'config' in f:
            config_dict = {}
            for key in f['config'].attrs.keys():
                config_dict[key] = f['config'].attrs[key]
            data['config'] = config_dict
        
        # 載入診斷資訊（如果存在）
        if 'diagnostics' in f:
            diag_dict = {}
            diag_group = f['diagnostics']
            try:
                for key in list(diag_group.keys()):  # type: ignore
                    diag_dict[key] = np.array(diag_group[key])  # type: ignore
            except AttributeError:
                pass
            data['diagnostics'] = diag_dict
    
    return data


def compute_vorticity(u, v, dx, dy):
    """計算渦度 ω = ∂v/∂x - ∂u/∂y"""
    # 使用中心差分
    dv_dx = np.gradient(v, dx, axis=2)
    du_dy = np.gradient(u, dy, axis=1)
    return dv_dx - du_dy


def compute_kinetic_energy(u, v):
    """計算動能"""
    return 0.5 * (u**2 + v**2)


def plot_snapshots(data, output_dir, time_indices=None):
    """繪製多個時間點的速度場與渦度場快照"""
    u = data['u']
    v = data['v']
    time = data['time']
    config = data.get('config', {})
    
    N = u.shape[1]
    L = config.get('L', 2 * np.pi)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # 如果未指定時間索引，選擇關鍵時間點
    if time_indices is None:
        # 選擇初始、1/4、1/2、3/4、最終時間
        n_frames = len(time)
        time_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4, -1]
    
    for idx in time_indices:
        t = time[idx]
        u_snap = u[idx]
        v_snap = v[idx]
        
        # 計算渦度
        dx = dy = L / N
        u_slice = u[idx:idx+1] if idx != -1 else u[-1:]
        v_slice = v[idx:idx+1] if idx != -1 else v[-1:]
        vort_3d = compute_vorticity(u_slice, v_slice, dx, dy)
        vorticity = vort_3d[0] if vort_3d.shape[0] > 0 else np.zeros((N, N))
        
        # 計算速度大小
        speed = np.sqrt(u_snap**2 + v_snap**2)
        
        # 創建圖形
        fig = plt.figure(figsize=(18, 5))
        gs = GridSpec(1, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. V 速度場
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.contourf(X, Y, v_snap, levels=50, cmap='RdBu_r')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'V Velocity at t={t:.2f}')
        ax1.set_aspect('equal')
        plt.colorbar(im1, ax=ax1, label='v')
        
        # 2. 速度大小
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.contourf(X, Y, speed, levels=50, cmap='viridis')
        # 疊加流線
        skip = N // 20
        ax2.streamplot(X[::skip, ::skip], Y[::skip, ::skip], 
                      u_snap[::skip, ::skip], v_snap[::skip, ::skip],
                      color='white', linewidth=0.5, density=1.5, arrowsize=0.8)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Speed at t={t:.2f}')
        ax2.set_aspect('equal')
        plt.colorbar(im2, ax=ax2, label='|u|')
        
        # 3. 渦度場
        ax3 = fig.add_subplot(gs[0, 2])
        vort_max = np.abs(vorticity).max()
        im3 = ax3.contourf(X, Y, vorticity, levels=50, 
                          cmap='RdBu_r', vmin=-vort_max, vmax=vort_max)
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        ax3.set_title(f'Vorticity at t={t:.2f}')
        ax3.set_aspect('equal')
        plt.colorbar(im3, ax=ax3, label='ω')
        
        # 儲存圖片
        output_file = output_dir / f'snapshot_t{t:.2f}.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'✅ Saved: {output_file}')


def plot_temporal_evolution(data, output_dir):
    """繪製時間演化"""
    time = data['time']
    u = data['u']
    v = data['v']
    
    # 計算全域動能
    ke = compute_kinetic_energy(u, v)
    ke_mean = ke.mean(axis=(1, 2))
    ke_max = ke.max(axis=(1, 2))
    
    # 計算全域渦度 RMS
    N = u.shape[1]
    L = data['config'].get('L', 2 * np.pi)
    dx = dy = L / N
    
    vorticity_rms = []
    for i in range(len(time)):
        vort = compute_vorticity(u[i:i+1], v[i:i+1], dx, dy)[0]
        vorticity_rms.append(np.sqrt((vort**2).mean()))
    vorticity_rms = np.array(vorticity_rms)
    
    # 計算速度統計
    v_mean = v.mean(axis=(1, 2))
    v_std = v.std(axis=(1, 2))
    
    # 創建圖形
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 動能演化
    ax = axes[0, 0]
    ax.plot(time, ke_mean, 'b-', linewidth=2, label='Mean KE')
    ax.plot(time, ke_max, 'r--', linewidth=1.5, label='Max KE')
    ax.set_xlabel('Time')
    ax.set_ylabel('Kinetic Energy')
    ax.set_title('Kinetic Energy Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 渦度演化
    ax = axes[0, 1]
    ax.plot(time, vorticity_rms, 'g-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Vorticity RMS')
    ax.set_title('Enstrophy Evolution (√⟨ω²⟩)')
    ax.grid(True, alpha=0.3)
    
    # 3. V 速度統計
    ax = axes[1, 0]
    ax.plot(time, v_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(time, v_mean - v_std, v_mean + v_std, 
                     alpha=0.3, label='±1 std')
    ax.set_xlabel('Time')
    ax.set_ylabel('V Velocity')
    ax.set_title('V Velocity Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 診斷資訊（如果有）
    ax = axes[1, 1]
    if 'diagnostics' in data and 'divergence_error' in data['diagnostics']:
        div_err = data['diagnostics']['divergence_error']
        ax.semilogy(time, div_err, 'r-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Divergence Error')
        ax.set_title('Incompressibility Check')
        ax.grid(True, alpha=0.3)
        
        # 顯示平均散度誤差
        avg_div = div_err.mean()
        ax.axhline(avg_div, color='k', linestyle='--', linewidth=1, 
                  label=f'Mean: {avg_div:.2e}')
        ax.legend()
    else:
        # 如果沒有診斷資訊，顯示 KE 的功率譜密度
        from scipy import signal
        dt_avg = float(np.mean(np.diff(time)))
        freq, psd = signal.welch(ke_mean, fs=1.0/dt_avg)
        ax.loglog(freq, psd, 'b-', linewidth=2)
        ax.set_xlabel('Frequency')
        ax.set_ylabel('PSD of KE')
        ax.set_title('Kinetic Energy Spectrum (Temporal)')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = output_dir / 'temporal_evolution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_file}')


def plot_energy_spectrum(data, output_dir, time_idx=-1):
    """繪製能量譜（空間） - 2D 湍流理論"""
    u = data['u'][time_idx]
    v = data['v'][time_idx]
    time = data['time'][time_idx]
    config = data.get('config', {})
    k_f = config.get('k_f', 4)  # 強迫波數
    
    N = u.shape[0]
    
    # 計算 2D FFT
    u_fft = np.fft.fft2(u)
    v_fft = np.fft.fft2(v)
    
    # 計算能量密度
    E_k = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2) / N**4
    
    # 徑向平均（2D 各向同性假設）
    kx = np.fft.fftfreq(N, 1.0/N)
    ky = np.fft.fftfreq(N, 1.0/N)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    
    k_bins = np.arange(1, N//2)
    E_spectrum = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        mask = (K >= k - 0.5) & (K < k + 0.5)
        E_spectrum[i] = E_k[mask].sum()
    
    # 繪圖
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # DNS 能量譜
    ax.loglog(k_bins, E_spectrum, 'b-', linewidth=2.5, label='DNS Spectrum', zorder=3)
    
    # === 2D 湍流理論參考線 ===
    # 逆級串區域 (k < k_f): E(k) ∝ k^(-5/3) (Kraichnan-Batchelor)
    k_inverse = k_bins[(k_bins > 1) & (k_bins < k_f)]
    if len(k_inverse) > 0:
        # 使用 k=2 作為錨點
        anchor_idx = 1 if k_bins[1] == 2 else 0
        E_ref_inverse = E_spectrum[anchor_idx] * (k_inverse / k_bins[anchor_idx])**(-5/3)
        ax.loglog(k_inverse, E_ref_inverse, 'r--', linewidth=2, 
                 label=r'$k^{-5/3}$ (Inverse Energy Cascade)', zorder=2)
    
    # 正向級串區域 (k > k_f): E(k) ∝ k^(-3)
    k_forward = k_bins[(k_bins > k_f) & (k_bins < N//3)]
    if len(k_forward) > 0:
        # 使用 k_f+2 作為錨點
        anchor_idx = np.argmin(np.abs(k_bins - (k_f + 2)))
        E_ref_forward = E_spectrum[anchor_idx] * (k_forward / k_bins[anchor_idx])**(-3)
        ax.loglog(k_forward, E_ref_forward, 'g--', linewidth=2, 
                 label=r'$k^{-3}$ (Forward Enstrophy Cascade)', zorder=2)
    
    # 標註強迫波數
    ax.axvline(x=k_f, color='orange', linestyle=':', linewidth=2.5, 
              label=f'Forcing Wavenumber $k_f={k_f}$', zorder=1)
    
    # 標註區域
    ax.text(k_f * 0.4, E_spectrum.max() * 0.5, 'Inverse\nCascade', 
           fontsize=11, ha='center', color='darkred', fontweight='bold')
    ax.text(k_f * 2.5, E_spectrum.max() * 0.01, 'Forward\nCascade', 
           fontsize=11, ha='center', color='darkgreen', fontweight='bold')
    
    ax.set_xlabel('Wavenumber k', fontsize=13)
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=13)
    ax.set_title(f'2D Turbulence Energy Spectrum at t={time:.2f}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, which='both', linestyle='--')
    ax.set_xlim([1, N//3])
    
    output_file = output_dir / f'energy_spectrum_t{time:.2f}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_file}')


def plot_statistics_summary(data, output_dir):
    """繪製統計摘要"""
    u = data['u']
    v = data['v']
    time = data['time']
    config = data.get('config', {})
    
    # 選擇穩態區間（後半段）
    steady_idx = len(time) // 2
    u_steady = u[steady_idx:]
    v_steady = v[steady_idx:]
    
    # 計算時間平均
    u_mean = u_steady.mean(axis=0)
    v_mean = v_steady.mean(axis=0)
    
    # 計算波動
    u_fluct = u_steady - u_mean
    v_fluct = v_steady - v_mean
    
    # Reynolds 應力 <u'v'>
    reynolds_stress = (u_fluct * v_fluct).mean(axis=0)
    
    # 創建圖形
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    N = u.shape[1]
    L = config.get('L', 2 * np.pi)
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # 1. U 時間平均
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, u_mean, levels=50, cmap='RdBu_r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title('Time-Averaged U')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1)
    
    # 2. V 時間平均
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, v_mean, levels=50, cmap='RdBu_r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Time-Averaged V')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Reynolds Stress
    ax3 = fig.add_subplot(gs[0, 2])
    rs_max = np.abs(reynolds_stress).max()
    im3 = ax3.contourf(X, Y, reynolds_stress, levels=50, 
                      cmap='RdBu_r', vmin=-rs_max, vmax=rs_max)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title("Reynolds Stress <u'v'>")
    ax3.set_aspect('equal')
    plt.colorbar(im3, ax=ax3)
    
    # 4. U RMS
    ax4 = fig.add_subplot(gs[1, 0])
    u_rms = np.sqrt((u_fluct**2).mean(axis=0))
    im4 = ax4.contourf(X, Y, u_rms, levels=50, cmap='viridis')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_title("U' RMS")
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4)
    
    # 5. V RMS
    ax5 = fig.add_subplot(gs[1, 1])
    v_rms = np.sqrt((v_fluct**2).mean(axis=0))
    im5 = ax5.contourf(X, Y, v_rms, levels=50, cmap='viridis')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_title("V' RMS")
    ax5.set_aspect('equal')
    plt.colorbar(im5, ax=ax5)
    
    # 6. 湍流強度
    ax6 = fig.add_subplot(gs[1, 2])
    turbulence_intensity = np.sqrt(u_rms**2 + v_rms**2) / np.sqrt(u_mean**2 + v_mean**2 + 1e-10)
    im6 = ax6.contourf(X, Y, turbulence_intensity, levels=50, cmap='plasma')
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title('Turbulence Intensity')
    ax6.set_aspect('equal')
    plt.colorbar(im6, ax=ax6)
    
    output_file = output_dir / 'statistics_summary.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'✅ Saved: {output_file}')


def generate_report(data, output_dir):
    """生成 Markdown 報告"""
    config = data.get('config', {})
    time = data['time']
    u = data['u']
    v = data['v']
    
    # 計算統計量
    N = config.get('N', u.shape[1])
    L = config.get('L', 2 * np.pi)
    nu = config.get('nu', 0.0)
    k_f = config.get('k_f', 4)
    A = config.get('A', 1.0)
    
    # 計算雷諾數
    Re = np.sqrt(A) * L**(3/2) / nu if nu > 0 else 0
    
    # 速度統計
    v_max = np.abs(v).max()
    v_mean_global = v.mean()
    v_std_global = v.std()
    
    # 動能
    ke = compute_kinetic_energy(u, v)
    ke_mean = ke.mean()
    ke_max = ke.max()
    
    # 生成報告
    report = f"""# Kolmogorov Flow DNS 視覺化報告

**生成時間**: {Path(output_dir).name}

---

## 📊 物理參數

| 參數 | 數值 |
|------|------|
| **雷諾數 (Re)** | {Re:.2f} |
| **黏度 (ν)** | {nu:.6f} |
| **強迫波數 (k_f)** | {k_f} |
| **強迫振幅 (A)** | {A} |
| **域大小 (L)** | {L:.4f} |
| **網格點數 (N)** | {N} |

---

## 🕐 時間資訊

| 項目 | 數值 |
|------|------|
| **時間範圍** | [{time[0]:.2f}, {time[-1]:.2f}] |
| **時間步數** | {len(time)} |
| **平均時間步長 (dt)** | {np.mean(np.diff(time)):.6f} |
| **總模擬時間** | {time[-1] - time[0]:.2f} |

---

## 📈 速度場統計

| 統計量 | U 速度 | V 速度 |
|--------|--------|--------|
| **最大值** | {np.abs(u).max():.6f} | {v_max:.6f} |
| **全域平均** | {u.mean():.6f} | {v_mean_global:.6f} |
| **標準差** | {u.std():.6f} | {v_std_global:.6f} |

---

## ⚡ 能量統計

| 項目 | 數值 |
|------|------|
| **平均動能** | {ke_mean:.6f} |
| **最大動能** | {ke_max:.6f} |

---

## 📁 生成的圖表

### 瞬時場快照
- `snapshot_t0.00.png` - 初始狀態
- `snapshot_t10.00.png` - 1/4 時間
- `snapshot_t20.00.png` - 1/2 時間
- `snapshot_t30.00.png` - 3/4 時間
- `snapshot_t40.00.png` - 最終狀態

### 時間演化分析
- `temporal_evolution.png` - 動能、渦度、速度統計演化

### 譜分析
- `energy_spectrum_t40.00.png` - 能量譜（最終時間）

### 統計摘要
- `statistics_summary.png` - 時間平均場、Reynolds 應力、湍流強度

---

## ✅ 物理驗證

### 散度檢查
"""
    
    if 'diagnostics' in data and 'divergence_error' in data['diagnostics']:
        div_err = data['diagnostics']['divergence_error']
        report += f"""- **平均散度誤差**: {div_err.mean():.2e}
- **最大散度誤差**: {div_err.max():.2e}
- **狀態**: {'✅ 通過 (< 1e-6)' if div_err.mean() < 1e-6 else '⚠️ 需檢查'}
"""
    else:
        report += "- 散度診斷資訊未記錄\n"
    
    report += f"""
### 雷諾數驗證
- **計算 Re**: {Re:.2f}
- **預期 Re**: 100.00
- **誤差**: {abs(Re - 100.0):.2f}
- **狀態**: {'✅ 通過 (< 5%)' if abs(Re - 100.0) < 5.0 else '⚠️ 需檢查'}

---

## 🎯 下一步建議

1. **感測器生成**: 使用此 DNS 數據生成 QR-Pivot 感測點
   ```bash
   python scripts/generate_sensors_k500.py \\
     --input data/kolmogorov_dns_re100_512x512_kf4.h5 \\
     --K 100 \\
     --output data/kolmogorov_qr_sensors_re100_K100.npz
   ```

2. **PINNs 訓練**: 使用生成的感測器啟動訓練
   ```bash
   python scripts/train.py \\
     --cfg configs/kolmogorov_re100_kf4_K100.yml \\
     --device mps
   ```

3. **物理驗證**: 完整驗證 DNS 數據
   ```bash
   python scripts/verify_kolmogorov_reynolds.py \\
     --dns data/kolmogorov_dns_re100_512x512_kf4.h5 \\
     --expected-Re 100
   ```

---

**報告生成完成** ✅
"""
    
    # 儲存報告
    report_file = output_dir / 'visualization_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'✅ Saved report: {report_file}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Kolmogorov Flow DNS data')
    parser.add_argument('--input', type=str, 
                       default='data/kolmogorov_dns_re100_512x512_kf4.h5',
                       help='DNS data file (HDF5)')
    parser.add_argument('--output', type=str,
                       default='results/dns_visualization',
                       help='Output directory for plots')
    parser.add_argument('--snapshots', nargs='+', type=int,
                       help='Time indices for snapshots (default: auto-select)')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*70}")
    print("🎨 Kolmogorov Flow DNS 視覺化工具")
    print(f"{'='*70}")
    print(f"📂 輸入檔案: {args.input}")
    print(f"📂 輸出目錄: {output_dir}")
    print()
    
    # 載入數據
    print("📥 載入 DNS 數據...")
    data = load_dns_data(args.input)
    print(f"✅ 載入完成: {len(data['time'])} time frames")
    print()
    
    # 生成視覺化
    print("🎨 生成快照...")
    plot_snapshots(data, output_dir, time_indices=args.snapshots)
    print()
    
    print("📈 生成時間演化圖...")
    plot_temporal_evolution(data, output_dir)
    print()
    
    print("📊 生成能量譜...")
    plot_energy_spectrum(data, output_dir)
    print()
    
    print("📊 生成統計摘要...")
    plot_statistics_summary(data, output_dir)
    print()
    
    print("📝 生成報告...")
    generate_report(data, output_dir)
    print()
    
    print(f"{'='*70}")
    print(f"✅ 視覺化完成！所有圖表已儲存至: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
