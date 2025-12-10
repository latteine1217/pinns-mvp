#!/usr/bin/env python3
"""
JHTDB High-Quality Flow Field Slices for Demonstration
Generates individual, high-resolution slice plots for showcase purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable

# === Font Configuration ===
plt.style.use('default')
# Priority: MacOS (Arial Unicode MS, PingFang HK) -> Windows (Microsoft JhengHei, SimHei) -> Linux (WenQuanYi)
font_list = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'PingFang TC', 'Microsoft JhengHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK TC', 'sans-serif']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

def load_data():
    velocity_path = 'data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_velocity_t1.h5'
    pressure_path = 'data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_pressure_t1.h5'
    
    print(f"Loading data...")
    with h5py.File(velocity_path, 'r') as f:
        # Shape (512, 128, 512, 3) -> (x, y, z, component)
        vel = f['Velocity_0001'][:]
        u = vel[..., 0]
        v = vel[..., 1]
        w = vel[..., 2]
        
    with h5py.File(pressure_path, 'r') as f:
        p = f['Pressure_0001'][..., 0]
        
    return u, v, w, p

def plot_slice(data, title, filename, cmap='viridis', label=''):
    """Generates a single slice plot"""
    # Slice at mid-z (spanwise)
    mid_idx = data.shape[2] // 2
    slice_data = data[:, :, mid_idx].T  # Transpose to match (y, x) for plotting
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot
    im = ax.contourf(slice_data, levels=100, cmap=cmap, extend='both')
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Streamwise (x)', fontsize=12)
    ax.set_ylabel('Wall-normal (y)', fontsize=12)
    
    # Remove ticks for cleaner look (optional, keep for now but simpler)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.5) # Increased size, adjusted pad
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(label, fontsize=12)
    
    plt.tight_layout()
    
    save_path = f"results/demo_slices/{filename}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {save_path}")
    plt.close()

def main():
    u, v, w, p = load_data()
    
    # Calculate derived fields
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    vorticity_z = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    
    print("Generating plots...")
    
    # 1. Streamwise Velocity
    plot_slice(u, 'Streamwise Velocity (u)\n流向速度分布', 'slice_velocity_u.png', cmap='RdBu_r', label='u [m/s]')
    
    # 2. Wall-normal Velocity
    plot_slice(v, 'Wall-normal Velocity (v)\n法向速度分布', 'slice_velocity_v.png', cmap='RdBu_r', label='v [m/s]')
    
    # 3. Velocity Magnitude
    plot_slice(vel_mag, 'Velocity Magnitude |V|\n速度幅值分布', 'slice_velocity_mag.png', cmap='inferno', label='|V| [m/s]')
    
    # 4. Pressure
    plot_slice(p, 'Pressure Field (p)\n壓力場分布', 'slice_pressure.png', cmap='coolwarm', label='p [Pa]')
    
    # 5. Vorticity Z
    plot_slice(vorticity_z, 'Spanwise Vorticity (ωz)\n展向渦量分布', 'slice_vorticity_z.png', cmap='seismic', label='ωz [1/s]')

if __name__ == "__main__":
    main()
