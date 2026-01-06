#!/usr/bin/env python3
"""
Kolmogorov Flow DNS 動畫生成器
生成速度場與渦度場的時間演化 GIF 動畫
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imageio
from pathlib import Path
import argparse
from tqdm import tqdm

def load_dns_data(file_path):
    """載入 DNS 數據"""
    data = {}
    with h5py.File(file_path, 'r') as f:
        data['u'] = np.array(f['u'])
        data['v'] = np.array(f['v'])
        data['time'] = np.array(f['time'])
        
        if 'config' in f:
            config_dict = {}
            for key in f['config'].attrs.keys():
                config_dict[key] = f['config'].attrs[key]
            data['config'] = config_dict
    
    return data

def create_frame(u, v, time, X, Y, vmin_u, vmax_u, vmin_v, vmax_v, vmin_vort, vmax_vort, dx, dy, Re):
    """創建單個時間幀"""
    # 計算渦度
    dv_dx = np.gradient(v, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    vorticity = dv_dx - du_dy
    
    # 計算速度大小
    speed = np.sqrt(u**2 + v**2)
    
    # 創建圖形
    fig = plt.figure(figsize=(20, 5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.35)
    
    # u 速度場
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X, Y, u, levels=256, cmap='RdBu_r', vmin=vmin_u, vmax=vmax_u)
    ax1.set_title(f'u velocity', fontsize=13, fontweight='bold')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # v 速度場
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X, Y, v, levels=256, cmap='RdBu_r', vmin=vmin_v, vmax=vmax_v)
    ax2.set_title(f'v velocity', fontsize=13, fontweight='bold')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # 渦度場
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.contourf(X, Y, vorticity, levels=256, cmap='RdBu_r', vmin=vmin_vort, vmax=vmax_vort)
    ax3.set_title(f'Vorticity', fontsize=13, fontweight='bold')
    ax3.set_xlabel('x', fontsize=11)
    ax3.set_ylabel('y', fontsize=11)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # 速度大小
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.contourf(X, Y, speed, levels=256, cmap='viridis')
    ax4.set_title(f'Speed', fontsize=13, fontweight='bold')
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('y', fontsize=11)
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # 添加時間標籤
    fig.suptitle(f'Kolmogorov Flow Re={Re} | t = {time:.2f} s', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 轉換為圖像（使用臨時檔案，不使用 bbox_inches='tight' 以保持一致尺寸）
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=80)
    buf.seek(0)
    image = imageio.imread(buf)
    buf.close()
    
    plt.close(fig)
    return image

def create_animation(data, output_path, Re, fps=10, stride=1, start_time=None, end_time=None):
    """創建動畫"""
    u = data['u']
    v = data['v']
    time = data['time']
    config = data.get('config', {})
    
    # 時間範圍篩選
    if start_time is not None or end_time is not None:
        start_time = start_time if start_time is not None else time[0]
        end_time = end_time if end_time is not None else time[-1]
        
        idx_start = np.argmin(np.abs(time - start_time))
        idx_end = np.argmin(np.abs(time - end_time))
        
        print(f"⏱️  時間範圍篩選: t={time[idx_start]:.2f} ~ {time[idx_end]:.2f}")
        print(f"   幀索引: {idx_start} ~ {idx_end} (共 {idx_end - idx_start + 1} 幀)")
        
        u = u[idx_start:idx_end+1]
        v = v[idx_start:idx_end+1]
        time = time[idx_start:idx_end+1]
    
    N = u.shape[1]
    L = config.get('L', 2 * np.pi)
    dx = dy = L / N
    
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    
    # 計算全局 colormap 範圍（用於一致性）
    print("📊 計算 colormap 範圍...")
    vmin_u, vmax_u = u.min(), u.max()
    vmin_v, vmax_v = v.min(), v.max()
    
    # 計算所有時間步的渦度範圍
    vort_min, vort_max = [], []
    for i in range(0, len(time), max(1, len(time)//10)):
        dv_dx = np.gradient(v[i], dx, axis=1)
        du_dy = np.gradient(u[i], dy, axis=0)
        vorticity = dv_dx - du_dy
        vort_min.append(vorticity.min())
        vort_max.append(vorticity.max())
    vmin_vort, vmax_vort = min(vort_min), max(vort_max)
    
    print(f"   u: [{vmin_u:.2f}, {vmax_u:.2f}]")
    print(f"   v: [{vmin_v:.2f}, {vmax_v:.2f}]")
    print(f"   vorticity: [{vmin_vort:.2f}, {vmax_vort:.2f}]")
    
    # 生成幀
    print(f"\n🎬 生成動畫幀 (stride={stride}, total frames={len(time)//stride})...")
    frames = []
    
    for i in tqdm(range(0, len(time), stride), desc="Rendering"):
        frame = create_frame(
            u[i], v[i], time[i], X, Y,
            vmin_u, vmax_u, vmin_v, vmax_v, vmin_vort, vmax_vort,
            dx, dy, Re
        )
        frames.append(frame)
    
    # 保存 GIF
    print(f"\n💾 保存 GIF 動畫...")
    imageio.mimsave(output_path, frames, fps=fps)
    
    # 計算檔案大小
    file_size_mb = Path(output_path).stat().st_size / 1024 / 1024
    
    print(f"✅ 動畫已保存: {output_path}")
    print(f"   幀數: {len(frames)}")
    print(f"   FPS: {fps}")
    print(f"   時長: {len(frames)/fps:.1f} 秒")
    print(f"   檔案大小: {file_size_mb:.1f} MB")

def main():
    parser = argparse.ArgumentParser(description='Generate DNS animation GIF')
    parser.add_argument('--input', type=str, required=True, help='DNS data file (HDF5)')
    parser.add_argument('--output', type=str, required=True, help='Output GIF file')
    parser.add_argument('--Re', type=int, default=100, help='Reynolds number (for title)')
    parser.add_argument('--fps', type=int, default=10, help='Frames per second')
    parser.add_argument('--stride', type=int, default=1, help='Frame stride (use every Nth frame)')
    parser.add_argument('--start-time', type=float, default=None, help='Start time for animation')
    parser.add_argument('--end-time', type=float, default=None, help='End time for animation')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🎬 Kolmogorov Flow DNS 動畫生成器")
    print("=" * 70)
    print(f"📂 輸入檔案: {args.input}")
    print(f"📂 輸出檔案: {args.output}")
    
    # 載入數據
    print("\n📥 載入 DNS 數據...")
    data = load_dns_data(args.input)
    print(f"✅ 載入完成: {len(data['time'])} 時間幀")
    
    # 創建動畫
    create_animation(data, args.output, args.Re, args.fps, args.stride, args.start_time, args.end_time)
    
    print("\n" + "=" * 70)
    print("✅ 動畫生成完成！")
    print("=" * 70)

if __name__ == '__main__':
    main()
