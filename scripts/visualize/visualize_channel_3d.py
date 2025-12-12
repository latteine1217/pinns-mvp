#!/usr/bin/env python3
"""
3D Channel Flow Visualization
Generates a 3D visualization of the channel flow field using JHTDB data.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
import os

def load_data(velocity_path):
    print(f"Loading data from {velocity_path}...")
    with h5py.File(velocity_path, 'r') as f:
        # Shape is (512, 128, 512, 3) likely (x, y, z, 3) or (z, y, x, 3)
        # Assuming (Z, Y, X, 3) based on typical JHTDB formats, or (X, Y, Z, 3)
        # Let's assume standard layout. We will transpose if necessary.
        # Usually JHTDB is (time, z, y, x, component) but here it seems to be spatial only.
        # Let's read a subset to check.
        
        # Downsample factor to fit in memory and plot reasonably
        # 512/16 = 32 points
        # 128/8 = 16 points
        skip = 16
        
        # Read coordinate arrays if available
        if 'xcoor' in f:
            x = f['xcoor'][:]
            y = f['ycoor'][:]
            z = f['zcoor'][:]
        else:
            # Fallback if coords not present
            x = np.linspace(0, 8*np.pi, 512)
            y = np.linspace(-1, 1, 128)
            z = np.linspace(0, 3*np.pi, 512)

        # Read velocity
        # Taking a sub-volume for clearer visualization
        # Full domain might be too crowded.
        
        # Let's take a slice in Z (spanwise) to show a block
        z_start = 0
        z_end = 128 # Quarter of the domain
        
        u_data = f['Velocity_0001'][z_start:z_end:skip, ::skip, ::skip, 0]
        v_data = f['Velocity_0001'][z_start:z_end:skip, ::skip, ::skip, 1]
        w_data = f['Velocity_0001'][z_start:z_end:skip, ::skip, ::skip, 2]
        
        # Create meshgrid for the downsampled data
        # Note: We need to match the slicing
        X_sub = x[::skip]
        Y_sub = y[::skip]
        Z_sub = z[z_start:z_end:skip]
        
        # Adjust meshgrid to match data shape (Z, Y, X)
        # Meshgrid 'ij' indexing: Z, Y, X
        Z, Y, X = np.meshgrid(Z_sub, Y_sub, X_sub, indexing='ij')
        
        return X, Y, Z, u_data, v_data, w_data

def plot_3d_channel_flow(X, Y, Z, u, v, w, output_file):
    print("Generating 3D plot...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate velocity magnitude
    vel_mag = np.sqrt(u**2 + v**2 + w**2)
    
    # Normalize for color mapping
    norm = plt.Normalize(vel_mag.min(), vel_mag.max())
    colors = plt.cm.jet(norm(vel_mag))
    
    # Quiver plot (3D vectors)
    # Length of arrows
    length = 0.5
    
    # We can use 'quiver'
    # Arguments: X, Y, Z, U, V, W
    q = ax.quiver(X, Y, Z, u, v, w, color=colors.reshape(-1, 4), length=0.05, normalize=True, alpha=0.6)
    
    # Set labels
    ax.set_xlabel('Streamwise (X)')
    ax.set_ylabel('Wall-normal (Y)')
    ax.set_zlabel('Spanwise (Z)')
    ax.set_title('3D Channel Flow Structure (Subset)\nColored by Velocity Magnitude')
    
    # Set limits
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    
    # Add a colorbar
    m = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
    m.set_array([])
    plt.colorbar(m, ax=ax, label='Velocity Magnitude |U|', shrink=0.6)
    
    # Viewpoint
    ax.view_init(elev=30, azim=-60)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Saved 3D plot to {output_file}")

def main():
    data_path = 'data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_velocity_t1.h5'
    output_path = 'results/channel_flow_3d_visualization.png'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    X, Y, Z, u, v, w = load_data(data_path)
    plot_3d_channel_flow(X, Y, Z, u, v, w, output_path)

if __name__ == "__main__":
    main()
