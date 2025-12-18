#!/usr/bin/env python3
"""
比較兩個 FLUENT RANS 輸出版本

目的：
- 比較原始版本 (FFF-Setup-Output) vs 第二版 (FFF-Setup-Output_2)
- 評估第二版是否有改進（收斂性、物理準確度）
- 決定使用哪個版本作為 lowfi prior
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


def load_fluent_h5(cas_file, dat_file):
    """
    從 FLUENT .cas.h5 和 .dat.h5 載入數據
    
    Returns
    -------
    data : dict
        包含座標、速度、壓力、湍流變量
    """
    print(f"\nLoading FLUENT files:")
    print(f"  CAS: {cas_file}")
    print(f"  DAT: {dat_file}")
    
    with h5py.File(cas_file, 'r') as f_cas:
        # 讀取網格座標 (nodes)
        coords_nodes = np.array(f_cas['meshes']['1']['nodes']['coords']['3'])
        x_nodes = coords_nodes[:, 0]
        y_nodes = coords_nodes[:, 1]
        z_nodes = coords_nodes[:, 2]
        
        n_nodes = len(x_nodes)
        print(f"  Nodes: {n_nodes}")
        
        # 計算 cell center 座標（用於與 cell data 對應）
        x_unique = np.unique(np.round(x_nodes, 6))
        y_unique = np.unique(np.round(y_nodes, 6))
        z_unique = np.unique(np.round(z_nodes, 6))
        
        x_cell_centers = 0.5 * (x_unique[:-1] + x_unique[1:])
        y_cell_centers = 0.5 * (y_unique[:-1] + y_unique[1:])
        z_cell_centers = 0.5 * (z_unique[:-1] + z_unique[1:])
        
        # 創建 cell center 網格（用於匹配 cell data）
        Nx, Ny, Nz = len(x_cell_centers), len(y_cell_centers), len(z_cell_centers)
        X_cell, Y_cell, Z_cell = np.meshgrid(x_cell_centers, y_cell_centers, z_cell_centers, indexing='ij')
        
        # 展平為 1D 陣列
        x_cell_flat = X_cell.ravel(order='F')
        y_cell_flat = Y_cell.ravel(order='F')
        z_cell_flat = Z_cell.ravel(order='F')
        
        print(f"  X range: [{x_nodes.min():.4f}, {x_nodes.max():.4f}]")
        print(f"  Y range: [{y_nodes.min():.4f}, {y_nodes.max():.4f}]")
        print(f"  Z range: [{z_nodes.min():.4f}, {z_nodes.max():.4f}]")
    
    with h5py.File(dat_file, 'r') as f_dat:
        # 讀取 cell-centered 數據
        cells = f_dat['results']['1']['phase-1']['cells']
        
        u = np.array(cells['SV_U']['1'])
        v = np.array(cells['SV_V']['1'])
        w = np.array(cells['SV_W']['1'])
        p = np.array(cells['SV_P']['1'])
        k = np.array(cells['SV_K']['1'])
        mu_t = np.array(cells['SV_MU_T']['1'])
        
        n_cells = len(u)
        print(f"  Cells: {n_cells}")
        
        # 使用計算的 cell center 座標
        x = x_cell_flat[:n_cells]
        y = y_cell_flat[:n_cells]
        z = z_cell_flat[:n_cells]
    
    return {
        'x': x, 'y': y, 'z': z,
        'u': u, 'v': v, 'w': w,
        'p': p, 'k': k, 'mu_t': mu_t,
        'n_cells': n_cells
    }


def compute_statistics(data, name=""):
    """計算流場統計量"""
    
    stats = {}
    
    # 速度統計
    for var in ['u', 'v', 'w']:
        field = data[var]
        stats[var] = {
            'mean': field.mean(),
            'std': field.std(),
            'min': field.min(),
            'max': field.max()
        }
    
    # 壓力統計
    stats['p'] = {
        'mean': data['p'].mean(),
        'std': data['p'].std(),
        'min': data['p'].min(),
        'max': data['p'].max()
    }
    
    # 湍流動能
    stats['k'] = {
        'mean': data['k'].mean(),
        'std': data['k'].std(),
        'min': data['k'].min(),
        'max': data['k'].max()
    }
    
    # 速度大小
    vel_mag = np.sqrt(data['u']**2 + data['v']**2 + data['w']**2)
    stats['vel_mag'] = {
        'mean': vel_mag.mean(),
        'std': vel_mag.std(),
        'min': vel_mag.min(),
        'max': vel_mag.max()
    }
    
    # 估計 u_tau 和 Re_tau (from TKE)
    u_tau_estimate = np.sqrt(data['k'].mean())
    nu = 5e-5  # 假設（需從配置確認）
    h = 1.0
    Re_tau_estimate = u_tau_estimate * h / nu
    
    stats['u_tau'] = u_tau_estimate
    stats['Re_tau'] = Re_tau_estimate
    
    # 體積平均速度
    U_bulk = data['u'].mean()
    stats['U_bulk'] = U_bulk
    
    return stats


def print_comparison_table(stats1, stats2, name1="Version 1", name2="Version 2"):
    """打印對比表格"""
    
    print("\n" + "="*80)
    print(f"COMPARISON: {name1} vs {name2}")
    print("="*80)
    
    # 關鍵參數對比
    print("\n📊 Key Flow Parameters:")
    print(f"{'Parameter':<20} {name1:>15} {name2:>15} {'Diff %':>12}")
    print("-"*80)
    
    params = [
        ('U_bulk', 'Bulk velocity'),
        ('u_tau', 'Friction velocity'),
        ('Re_tau', 'Friction Reynolds')
    ]
    
    for key, desc in params:
        v1 = stats1[key]
        v2 = stats2[key]
        diff_pct = (v2 - v1) / v1 * 100 if v1 != 0 else 0
        print(f"{desc:<20} {v1:>15.6f} {v2:>15.6f} {diff_pct:>11.2f}%")
    
    # 速度場對比
    print("\n🌊 Velocity Field Statistics:")
    print(f"{'Variable':<10} {'Stat':<8} {name1:>15} {name2:>15} {'Diff %':>12}")
    print("-"*80)
    
    for var in ['u', 'v', 'w', 'vel_mag']:
        for stat_name in ['mean', 'std', 'min', 'max']:
            v1 = stats1[var][stat_name]
            v2 = stats2[var][stat_name]
            diff_pct = (v2 - v1) / abs(v1) * 100 if abs(v1) > 1e-10 else 0
            print(f"{var:<10} {stat_name:<8} {v1:>15.6e} {v2:>15.6e} {diff_pct:>11.2f}%")
    
    # 湍流變量對比
    print("\n💨 Turbulence Statistics:")
    print(f"{'Variable':<10} {'Stat':<8} {name1:>15} {name2:>15} {'Diff %':>12}")
    print("-"*80)
    
    for var in ['k', 'p']:
        for stat_name in ['mean', 'std', 'min', 'max']:
            v1 = stats1[var][stat_name]
            v2 = stats2[var][stat_name]
            diff_pct = (v2 - v1) / abs(v1) * 100 if abs(v1) > 1e-10 else 0
            print(f"{var:<10} {stat_name:<8} {v1:>15.6e} {v2:>15.6e} {diff_pct:>11.2f}%")


def plot_comparison(data1, data2, name1="V1", name2="V2", output_dir="results"):
    """
    視覺化對比兩個版本
    """
    
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 速度剖面對比 (Y方向平均)
    ax1 = plt.subplot(2, 3, 1)
    
    # 計算 Y方向平均速度剖面
    # 簡化：使用整體平均（實際應該按Y分層）
    y_bins = np.linspace(data1['y'].min(), data1['y'].max(), 50)
    u_profile1 = []
    u_profile2 = []
    y_centers = []
    
    for i in range(len(y_bins) - 1):
        y_min, y_max = y_bins[i], y_bins[i+1]
        mask1 = (data1['y'] >= y_min) & (data1['y'] < y_max)
        mask2 = (data2['y'] >= y_min) & (data2['y'] < y_max)
        
        if mask1.sum() > 0:
            u_profile1.append(data1['u'][mask1].mean())
            y_centers.append((y_min + y_max) / 2)
        if mask2.sum() > 0:
            u_profile2.append(data2['u'][mask2].mean())
    
    ax1.plot(u_profile1, y_centers, 'b-o', label=name1, alpha=0.7, markersize=4)
    ax1.plot(u_profile2, y_centers, 'r-s', label=name2, alpha=0.7, markersize=4)
    ax1.set_xlabel('U (streamwise velocity)')
    ax1.set_ylabel('Y (wall-normal)')
    ax1.set_title('Mean Velocity Profile')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. TKE 剖面對比
    ax2 = plt.subplot(2, 3, 2)
    
    k_profile1 = []
    k_profile2 = []
    
    for i in range(len(y_bins) - 1):
        y_min, y_max = y_bins[i], y_bins[i+1]
        mask1 = (data1['y'] >= y_min) & (data1['y'] < y_max)
        mask2 = (data2['y'] >= y_min) & (data2['y'] < y_max)
        
        if mask1.sum() > 0:
            k_profile1.append(data1['k'][mask1].mean())
        if mask2.sum() > 0:
            k_profile2.append(data2['k'][mask2].mean())
    
    ax2.plot(k_profile1, y_centers, 'b-o', label=name1, alpha=0.7, markersize=4)
    ax2.plot(k_profile2, y_centers, 'r-s', label=name2, alpha=0.7, markersize=4)
    ax2.set_xlabel('k (TKE)')
    ax2.set_ylabel('Y (wall-normal)')
    ax2.set_title('Turbulent Kinetic Energy Profile')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 速度差異直方圖
    ax3 = plt.subplot(2, 3, 3)
    
    # 需要插值到相同網格（簡化：假設座標相同）
    if len(data1['u']) == len(data2['u']):
        u_diff = data2['u'] - data1['u']
        ax3.hist(u_diff, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax3.set_xlabel('Δu (V2 - V1)')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'Velocity Difference Distribution\nMean: {u_diff.mean():.6f}, Std: {u_diff.std():.6f}')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Grid sizes differ\nCannot compute point-wise difference',
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    
    # 4. TKE 差異直方圖
    ax4 = plt.subplot(2, 3, 4)
    
    if len(data1['k']) == len(data2['k']):
        k_diff = data2['k'] - data1['k']
        ax4.hist(k_diff, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax4.set_xlabel('Δk (V2 - V1)')
        ax4.set_ylabel('Frequency')
        ax4.set_title(f'TKE Difference Distribution\nMean: {k_diff.mean():.6e}, Std: {k_diff.std():.6e}')
        ax4.axvline(0, color='red', linestyle='--', linewidth=2)
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Grid sizes differ\nCannot compute point-wise difference',
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    
    # 5. 速度大小散點圖
    ax5 = plt.subplot(2, 3, 5)
    
    vel_mag1 = np.sqrt(data1['u']**2 + data1['v']**2 + data1['w']**2)
    vel_mag2 = np.sqrt(data2['u']**2 + data2['v']**2 + data2['w']**2)
    
    if len(vel_mag1) == len(vel_mag2):
        # 下採樣以便繪圖
        n_sample = min(5000, len(vel_mag1))
        indices = np.random.choice(len(vel_mag1), n_sample, replace=False)
        
        ax5.scatter(vel_mag1[indices], vel_mag2[indices], 
                   alpha=0.3, s=10, c='blue')
        
        # 對角線 (perfect match)
        lim = [min(vel_mag1.min(), vel_mag2.min()), 
               max(vel_mag1.max(), vel_mag2.max())]
        ax5.plot(lim, lim, 'r--', linewidth=2, label='Perfect match')
        
        ax5.set_xlabel(f'|U| {name1}')
        ax5.set_ylabel(f'|U| {name2}')
        ax5.set_title('Velocity Magnitude Correlation')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_aspect('equal')
    
    # 6. 關鍵指標雷達圖
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # 選擇關鍵指標（標準化到 [0, 1]）
    from scipy.stats import zscore
    
    metrics_names = ['U_bulk', 'u_tau', 'TKE_mean', 'u_std', 'k_std']
    
    # 計算統計值
    stats1 = compute_statistics(data1)
    stats2 = compute_statistics(data2)
    
    values1 = [
        stats1['U_bulk'],
        stats1['u_tau'],
        stats1['k']['mean'],
        stats1['u']['std'],
        stats1['k']['std']
    ]
    
    values2 = [
        stats2['U_bulk'],
        stats2['u_tau'],
        stats2['k']['mean'],
        stats2['u']['std'],
        stats2['k']['std']
    ]
    
    # 標準化（相對於兩者最大值）
    max_vals = [max(v1, v2) for v1, v2 in zip(values1, values2)]
    norm_values1 = [v / m if m > 0 else 0 for v, m in zip(values1, max_vals)]
    norm_values2 = [v / m if m > 0 else 0 for v, m in zip(values2, max_vals)]
    
    # 閉合圓形
    norm_values1 += [norm_values1[0]]
    norm_values2 += [norm_values2[0]]
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_names), endpoint=False).tolist()
    angles += [angles[0]]
    
    ax6.plot(angles, norm_values1, 'o-', linewidth=2, label=name1, color='blue')
    ax6.fill(angles, norm_values1, alpha=0.15, color='blue')
    
    ax6.plot(angles, norm_values2, 's-', linewidth=2, label=name2, color='red')
    ax6.fill(angles, norm_values2, alpha=0.15, color='red')
    
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics_names, fontsize=9)
    ax6.set_ylim(0, 1)
    ax6.set_title('Key Metrics Comparison (Normalized)', pad=20)
    ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax6.grid(True)
    
    plt.tight_layout()
    
    # 保存圖形
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    output_file = output_path / 'fluent_rans_version_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    print(f"\n✅ Figure saved: {output_file}")
    
    plt.close()


def main():
    """主程序"""
    
    # 檔案路徑
    v1_cas = 'data/channel_fluent/FFF-Setup-Output.cas.h5'
    v1_dat = 'data/channel_fluent/FFF-Setup-Output.dat.h5'
    
    v2_cas = 'data/channel_fluent/FFF-Setup-Output.cas_2.h5'
    v2_dat = 'data/channel_fluent/FFF-Setup-Output.dat_2.h5'
    
    # 載入數據
    print("="*80)
    print("LOADING VERSION 1 (Original)")
    print("="*80)
    data1 = load_fluent_h5(v1_cas, v1_dat)
    
    print("\n" + "="*80)
    print("LOADING VERSION 2 (New)")
    print("="*80)
    data2 = load_fluent_h5(v2_cas, v2_dat)
    
    # 計算統計量
    stats1 = compute_statistics(data1, name="V1")
    stats2 = compute_statistics(data2, name="V2")
    
    # 打印對比表
    print_comparison_table(stats1, stats2, name1="Version 1", name2="Version 2")
    
    # 視覺化對比
    print("\nGenerating comparison plots...")
    plot_comparison(data1, data2, name1="V1 (Original)", name2="V2 (New)")
    
    # 評估與建議
    print("\n" + "="*80)
    print("EVALUATION & RECOMMENDATION")
    print("="*80)
    
    # Re_tau 比較
    re_tau_diff = abs(stats2['Re_tau'] - stats1['Re_tau']) / stats1['Re_tau'] * 100
    u_bulk_diff = abs(stats2['U_bulk'] - stats1['U_bulk']) / stats1['U_bulk'] * 100
    
    target_re_tau = 983.7  # JHTDB DNS target
    
    error1 = abs(stats1['Re_tau'] - target_re_tau) / target_re_tau * 100
    error2 = abs(stats2['Re_tau'] - target_re_tau) / target_re_tau * 100
    
    print(f"\n📊 Re_tau Comparison:")
    print(f"  Version 1: {stats1['Re_tau']:.1f} (error: {error1:.1f}%)")
    print(f"  Version 2: {stats2['Re_tau']:.1f} (error: {error2:.1f}%)")
    print(f"  DNS Target: {target_re_tau:.1f}")
    print(f"  → Version 2 is {'BETTER' if error2 < error1 else 'WORSE'} (Δ error: {error2 - error1:+.1f}%)")
    
    print(f"\n📊 U_bulk Comparison:")
    print(f"  Version 1: {stats1['U_bulk']:.6f}")
    print(f"  Version 2: {stats2['U_bulk']:.6f}")
    print(f"  Difference: {u_bulk_diff:.2f}%")
    
    # 決策邏輯
    print(f"\n🎯 RECOMMENDATION:")
    
    if error2 < error1 * 0.9:  # V2 明顯更好 (>10% improvement)
        print(f"  ✅ USE VERSION 2")
        print(f"     - Re_tau error reduced by {error1 - error2:.1f}%")
        print(f"     - Closer to DNS target")
    elif error2 < error1:  # V2 略好
        print(f"  ✅ USE VERSION 2 (marginal improvement)")
        print(f"     - Re_tau error reduced by {error1 - error2:.1f}%")
    elif abs(error2 - error1) < 5:  # 差不多
        print(f"  ⚖️  BOTH VERSIONS SIMILAR")
        print(f"     - Use Version 1 (already processed)")
        print(f"     - Re_tau error difference: {abs(error2 - error1):.1f}%")
    else:  # V1 更好
        print(f"  ⚠️  VERSION 1 IS BETTER")
        print(f"     - Version 2 has {error2 - error1:.1f}% higher error")
        print(f"     - Keep using Version 1")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE")
    print("="*80)
    print(f"  Summary saved: results/fluent_rans_version_comparison.png")
    print("="*80)


if __name__ == '__main__':
    main()
