"""
2D Kolmogorov Flow 訓練結果視覺化
====================================

用途：視覺化純 PDE 訓練的流場結果
- 速度場 (u, v)
- 壓力場 (p)
- 渦度場 (vorticity)
- 流線圖

使用範例：
    python scripts/visualize_kolmogorov_results.py \
        --results results/kolmogorov_2d_turbulent_pure_pde/final_results.npz \
        --output results/kolmogorov_2d_turbulent_pure_pde/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_field_2d(x, y, field, title, output_path, cmap='RdBu_r', vmin=None, vmax=None):
    """繪製 2D 標量場"""
    fig, ax = plt.subplots(figsize=(8, 6))

    if vmin is None:
        vmin = field.min()
    if vmax is None:
        vmax = field.max()

    im = ax.contourf(x, y, field.T, levels=50, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.set_aspect('equal')

    cbar = plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ {output_path}")


def plot_streamlines(x, y, u, v, output_path):
    """繪製流線圖"""
    fig, ax = plt.subplots(figsize=(8, 6))

    # 速度大小
    speed = np.sqrt(u**2 + v**2)

    # 流線
    ax.streamplot(x, y, u.T, v.T, color=speed.T, cmap='viridis', density=2, linewidth=1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Streamlines (colored by velocity magnitude)')
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ {output_path}")


def plot_velocity_profiles(x, y, u, v, output_path):
    """繪製速度剖面"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # x 方向中心線
    y_center_idx = len(y) // 2
    ax1.plot(x, u[:, y_center_idx], label='u (x-velocity)', linewidth=2)
    ax1.plot(x, v[:, y_center_idx], label='v (y-velocity)', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Velocity')
    ax1.set_title(f'Velocity profiles at y={y[y_center_idx]:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # y 方向中心線
    x_center_idx = len(x) // 2
    ax2.plot(y, u[x_center_idx, :], label='u (x-velocity)', linewidth=2)
    ax2.plot(y, v[x_center_idx, :], label='v (y-velocity)', linewidth=2)
    ax2.set_xlabel('y')
    ax2.set_ylabel('Velocity')
    ax2.set_title(f'Velocity profiles at x={x[x_center_idx]:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ {output_path}")


def main():
    parser = argparse.ArgumentParser(description='視覺化 Kolmogorov flow 訓練結果')
    parser.add_argument('--results', type=str,
                       default='results/kolmogorov_2d_turbulent_pure_pde/final_results.npz',
                       help='結果 .npz 檔案路徑')
    parser.add_argument('--output', type=str,
                       default='results/kolmogorov_2d_turbulent_pure_pde/',
                       help='輸出目錄')
    args = parser.parse_args()

    # 載入結果
    results = np.load(args.results)
    x = results['x']
    y = results['y']
    u = results['u']
    v = results['v']
    p = results['p']
    vorticity = results['vorticity']
    kinetic_energy = results['kinetic_energy']
    enstrophy = results['enstrophy']

    # 建立輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 生成視覺化圖表...")
    print(f"   動能 KE = {kinetic_energy:.6f}")
    print(f"   Enstrophy = {enstrophy:.6f}")
    print()

    # === 速度場 ===
    print("1️⃣ 速度場 (u, v)...")
    plot_field_2d(x, y, u, 'x-velocity (u)', output_dir / 'u_field.png', cmap='RdBu_r')
    plot_field_2d(x, y, v, 'y-velocity (v)', output_dir / 'v_field.png', cmap='RdBu_r')

    # === 壓力場 ===
    print("2️⃣ 壓力場 (p)...")
    plot_field_2d(x, y, p, 'Pressure (p)', output_dir / 'p_field.png', cmap='coolwarm')

    # === 渦度場 ===
    print("3️⃣ 渦度場 (vorticity)...")
    vmax = max(abs(vorticity.min()), abs(vorticity.max()))
    plot_field_2d(x, y, vorticity, 'Vorticity (ω)', output_dir / 'vorticity_field.png',
                 cmap='RdBu_r', vmin=-vmax, vmax=vmax)

    # === 流線圖 ===
    print("4️⃣ 流線圖...")
    plot_streamlines(x, y, u, v, output_dir / 'streamlines.png')

    # === 速度剖面 ===
    print("5️⃣ 速度剖面...")
    plot_velocity_profiles(x, y, u, v, output_dir / 'velocity_profiles.png')

    print()
    print("🎉 視覺化完成！")
    print(f"   輸出目錄: {output_dir}")


if __name__ == '__main__':
    main()
