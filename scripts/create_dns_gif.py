"""
DNS 動畫生成器（GIF）
====================

使用 Pillow 直接生成 GIF 動畫，無需 ffmpeg

使用方式:
---------
python scripts/create_dns_gif.py \\
    --input data/kolmogorov_Re1000_kf4_t100s.h5 \\
    --output results/dns_re1000_visualization/vorticity.gif \\
    --fps 20 \\
    --downsample 10
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
from pathlib import Path
from PIL import Image
import io
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_vorticity(u, v, L):
    """計算渦度"""
    N = u.shape[-1]
    dx = dy = L / N
    dvdx = np.gradient(v, dx, axis=-2)
    dudy = np.gradient(u, dy, axis=-1)
    return dvdx - dudy


def create_gif_animation(h5_file: str, output_file: str, fps: int = 20, downsample: int = 10, time_range: list = None):
    """創建 GIF 動畫"""

    print(f"📂 載入 DNS 資料: {h5_file}")

    with h5py.File(h5_file, 'r') as f:
        u_all = f['u'][:]
        v_all = f['v'][:]
        time = f['time'][:]
        L = f['config'].attrs['L']
        N = f['config'].attrs['N']
        nu = f['config'].attrs['nu']

    Re = 1 / nu

    print(f"   ✓ 載入 {len(time)} 個快照")
    
    # 時間範圍過濾
    if time_range:
        t_start, t_end = time_range
        indices = np.where((time >= t_start) & (time <= t_end))[0]
        if len(indices) == 0:
            print(f"⚠️  No frames found in range [{t_start}, {t_end}]")
            return
        # 對過濾後的索引進行降採樣
        frame_indices = indices[::downsample]
        print(f"   ✓ 時間範圍: {t_start:.1f}s - {t_end:.1f}s")
    else:
        # 原始全範圍降採樣
        frame_indices = np.arange(0, len(time), downsample)

    print(f"   ✓ 網格: {N}×{N}")
    print(f"   ✓ Re = {Re:.0f}")

    # 空間降採樣（限制解析度到 512×512 以加快繪圖）
    spatial_stride = max(1, N // 512)

    print(f"\n🎬 生成動畫：{len(frame_indices)} 幀，空間降採樣 {spatial_stride}×")

    x = np.linspace(0, L, N, endpoint=False)[::spatial_stride]
    y = np.linspace(0, L, N, endpoint=False)[::spatial_stride]
    X, Y = np.meshgrid(x, y, indexing='ij')

    frames = []

    for frame_idx in tqdm(frame_indices, desc="渲染幀"):
        t = time[frame_idx]
        u = u_all[frame_idx][::spatial_stride, ::spatial_stride]
        v = v_all[frame_idx][::spatial_stride, ::spatial_stride]
        omega = compute_vorticity(u, v, L)

        # 創建圖像
        fig, ax = plt.subplots(figsize=(10, 8))

        vmax = np.percentile(np.abs(omega), 99)  # 使用 99% 百分位避免極值
        im = ax.contourf(X, Y, omega, levels=50, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)

        ax.set_title(f'Vorticity Field - Re={Re:.0f}, t={t:.2f}s',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

        cbar = plt.colorbar(im, ax=ax, label='ω [1/s]')

        # 轉換為 PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
        buf.seek(0)
        frame = Image.open(buf).copy()
        frames.append(frame)
        buf.close()

        plt.close(fig)

    # 保存 GIF
    print(f"\n💾 保存 GIF: {output_file}")

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    duration_ms = int(1000 / fps)
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False  # 不優化，保持速度
    )

    file_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"✅ GIF 已保存：{output_file} ({file_size:.1f} MB)")
    print(f"   ✓ 幀數: {len(frames)}")
    print(f"   ✓ FPS: {fps}")
    print(f"   ✓ 總時長: {len(frames)/fps:.1f} 秒")


def main():
    parser = argparse.ArgumentParser(description='DNS GIF 動畫生成器')
    parser.add_argument('--input', type=str, required=True, help='輸入 HDF5 文件')
    parser.add_argument('--output', type=str,
                        default='results/dns_re1000_visualization/vorticity.gif',
                        help='輸出 GIF 文件')
    parser.add_argument('--fps', type=int, default=20, help='幀率（預設 20）')
    parser.add_argument('--downsample', type=int, default=10,
                        help='時間降採樣（每 N 個快照取 1 個，預設 10）')
    parser.add_argument('--time-range', type=float, nargs=2, default=None,
                        help='時間範圍 [t_start, t_end] (預設: 全範圍)')

    args = parser.parse_args()

    print("=" * 80)
    print("DNS GIF 動畫生成器")
    print("=" * 80)

    create_gif_animation(args.input, args.output, args.fps, args.downsample, args.time_range)

    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
