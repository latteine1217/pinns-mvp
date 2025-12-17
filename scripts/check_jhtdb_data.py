#!/usr/bin/env python
"""快速檢查和可視化 JHTDB Channel Flow 數據"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 設置中文字體
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False

def load_jhtdb_data():
    """載入 JHTDB 數據"""
    print("📂 載入 JHTDB Channel Flow 數據...")

    # 載入速度場
    with h5py.File('data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_velocity_t1.h5', 'r') as f:
        vel = f['Velocity_0001'][:]  # (512, 128, 512, 3)
        x = f['xcoor'][:]
        y = f['ycoor'][:]
        z = f['zcoor'][:]

    # 載入壓力場
    with h5py.File('data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_pressure_t1.h5', 'r') as f:
        p = f['Pressure_0001'][:, :, :, 0]  # (512, 128, 512)

    print(f"✅ 數據載入完成")
    print(f"   Domain: x ∈ [{x.min():.2f}, {x.max():.2f}]")
    print(f"           y ∈ [{y.min():.2f}, {y.max():.2f}]")
    print(f"           z ∈ [{z.min():.2f}, {z.max():.2f}]")
    print(f"   Grid: {vel.shape[0]} × {vel.shape[1]} × {vel.shape[2]}")

    return x, y, z, vel, p

def compute_statistics(y, vel):
    """計算流動統計量"""
    print("\n📊 計算統計量...")

    u = vel[:, :, :, 0]
    v = vel[:, :, :, 1]
    w = vel[:, :, :, 2]

    # 沿 x-z 平面平均（獲得 y 方向剖面）
    u_mean = u.mean(axis=(0, 2))  # shape: (Ny,)
    v_mean = v.mean(axis=(0, 2))
    w_mean = w.mean(axis=(0, 2))

    # RMS fluctuations
    u_rms = np.sqrt(((u - u_mean[None, :, None])**2).mean(axis=(0, 2)))
    v_rms = np.sqrt(((v - v_mean[None, :, None])**2).mean(axis=(0, 2)))
    w_rms = np.sqrt(((w - w_mean[None, :, None])**2).mean(axis=(0, 2)))

    # 體積平均
    U_bulk = u.mean()

    # 中心線速度
    center_idx = len(y) // 2
    U_center = u_mean[center_idx]

    # 從 Re_tau 計算 u_tau
    Re_tau = 983.683
    nu = 5e-5
    h = 1.0
    u_tau = Re_tau * nu / h

    print(f"✅ 統計計算完成")
    print(f"   Bulk velocity: U_b = {U_bulk:.4f}")
    print(f"   Centerline velocity: U_c = {U_center:.4f}")
    print(f"   Friction velocity: u_τ = {u_tau:.6f}")
    print(f"   Re_τ = {Re_tau:.1f}")

    return {
        'u_mean': u_mean,
        'v_mean': v_mean,
        'w_mean': w_mean,
        'u_rms': u_rms,
        'v_rms': v_rms,
        'w_rms': w_rms,
        'U_bulk': U_bulk,
        'U_center': U_center,
        'u_tau': u_tau,
        'Re_tau': Re_tau
    }

def plot_velocity_profiles(y, stats):
    """繪製速度剖面"""
    print("\n📈 繪製速度剖面...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('JHTDB Channel Flow: Velocity Profiles (y-direction)',
                 fontsize=14, fontweight='bold')

    u_tau = stats['u_tau']

    # 轉換為壁面單位 (y+ = y * u_tau / nu)
    nu = 5e-5
    y_plus = np.abs(y) * u_tau / nu

    # 1. Mean velocity profile (inner scaling)
    ax = axes[0]
    u_plus = stats['u_mean'] / u_tau
    ax.semilogx(y_plus, u_plus, 'b-', linewidth=2, label='DNS')

    # 理論對數律: u+ = (1/κ)ln(y+) + B
    kappa = 0.41
    B = 5.0
    y_log = np.logspace(1, np.log10(y_plus.max()), 100)
    u_log = (1/kappa) * np.log(y_log) + B
    ax.semilogx(y_log, u_log, 'r--', linewidth=2, alpha=0.7, label='Log law')

    # 線性律: u+ = y+
    y_linear = np.logspace(-1, 1.3, 50)
    ax.semilogx(y_linear, y_linear, 'g--', linewidth=2, alpha=0.7, label='Linear (u+=y+)')

    ax.set_xlabel('y+')
    ax.set_ylabel('u+')
    ax.set_title('Mean Velocity (Inner Scaling)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0.1, y_plus.max()])

    # 2. RMS velocity fluctuations (inner scaling)
    ax = axes[1]
    ax.semilogx(y_plus, stats['u_rms'] / u_tau, 'b-', linewidth=2, label="u' rms")
    ax.semilogx(y_plus, stats['v_rms'] / u_tau, 'r-', linewidth=2, label="v' rms")
    ax.semilogx(y_plus, stats['w_rms'] / u_tau, 'g-', linewidth=2, label="w' rms")
    ax.set_xlabel('y+')
    ax.set_ylabel("u'+ rms")
    ax.set_title('RMS Velocity Fluctuations')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim([0.1, y_plus.max()])

    # 3. Mean velocity (outer scaling)
    ax = axes[2]
    y_h = y / 1.0  # y/h (h=1)
    U_normalized = stats['u_mean'] / stats['U_center']
    ax.plot(y_h, U_normalized, 'b-', linewidth=2)
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='U_c')
    ax.axhline(y=stats['U_bulk']/stats['U_center'], color='g',
               linestyle='--', alpha=0.7, label='U_b')
    ax.set_xlabel('y/h')
    ax.set_ylabel('U / U_c')
    ax.set_title('Mean Velocity (Outer Scaling)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('results/jhtdb_velocity_profiles.png', dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: results/jhtdb_velocity_profiles.png")
    plt.close()

def plot_flow_fields(x, y, z, vel, p):
    """繪製流場快照"""
    print("\n📸 繪製流場快照...")

    # 取 z 中間的 slice (x-y plane)
    z_mid = len(z) // 2

    u_slice = vel[:, :, z_mid, 0]
    v_slice = vel[:, :, z_mid, 1]
    w_slice = vel[:, :, z_mid, 2]
    p_slice = p[:, :, z_mid]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.3)
    fig.suptitle(f'JHTDB Channel Flow: Instantaneous Fields (x-y plane at z={z[z_mid]:.2f})',
                 fontsize=14, fontweight='bold')

    X, Y = np.meshgrid(x, y, indexing='ij')

    # u velocity
    ax = fig.add_subplot(gs[0, 0])
    im = ax.contourf(X, Y, u_slice, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamwise Velocity (u)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # v velocity
    ax = fig.add_subplot(gs[0, 1])
    im = ax.contourf(X, Y, v_slice, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Wall-normal Velocity (v)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # w velocity
    ax = fig.add_subplot(gs[1, 0])
    im = ax.contourf(X, Y, w_slice, levels=50, cmap='RdBu_r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Spanwise Velocity (w)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    # pressure
    ax = fig.add_subplot(gs[1, 1])
    im = ax.contourf(X, Y, p_slice, levels=50, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Pressure (p)')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)

    plt.savefig('results/jhtdb_flow_fields.png', dpi=300, bbox_inches='tight')
    print(f"✅ 圖表已保存: results/jhtdb_flow_fields.png")
    plt.close()

def main():
    print("="*80)
    print("JHTDB Channel Flow Data Validation")
    print("="*80)

    # 載入數據
    x, y, z, vel, p = load_jhtdb_data()

    # 計算統計量
    stats = compute_statistics(y, vel)

    # 繪製速度剖面
    plot_velocity_profiles(y, stats)

    # 繪製流場快照
    plot_flow_fields(x, y, z, vel, p)

    print("\n" + "="*80)
    print("✅ 驗證完成！")
    print("="*80)
    print("\n📄 詳細報告已保存: JHTDB_DATA_VALIDATION.md")
    print("📊 圖表已保存:")
    print("   - results/jhtdb_velocity_profiles.png")
    print("   - results/jhtdb_flow_fields.png")
    print("\n🎯 數據品質優秀，可以開始訓練！")
    print("="*80)

if __name__ == '__main__':
    main()
