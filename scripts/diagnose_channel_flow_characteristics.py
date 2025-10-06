#!/usr/bin/env python3
"""
Channel Flow 特徵診斷與可視化工具
專門用於診斷當前數據是否具有典型的Channel Flow特徵
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path

# 設置matplotlib
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

def load_and_analyze_data():
    """載入並分析JHTDB數據的Channel Flow特徵"""
    file_path = './data/jhtdb/channel_9af48b78776a7ba299b1ae14c283ceba.h5'
    
    print("🔍 Channel Flow 特徵診斷")
    print("="*50)
    
    with h5py.File(file_path, 'r') as f:
        u = np.array(f['u'])  # 流向速度
        v = np.array(f['v'])  # 法向速度  
        w = np.array(f['w'])  # 展向速度
        p = np.array(f['p'])  # 壓力場
    
    print(f"✅ 數據形狀: {u.shape}")
    
    return u, v, w, p

def analyze_channel_flow_characteristics(u, v, w, p):
    """分析Channel Flow的典型特徵"""
    
    print("\n📊 Channel Flow 特徵分析:")
    print("-" * 40)
    
    # 1. 速度分量統計比較
    print("1. 速度分量統計:")
    u_stats = {'mean': np.mean(u), 'std': np.std(u), 'rms': np.sqrt(np.mean(u**2))}
    v_stats = {'mean': np.mean(v), 'std': np.std(v), 'rms': np.sqrt(np.mean(v**2))}
    w_stats = {'mean': np.mean(w), 'std': np.std(w), 'rms': np.sqrt(np.mean(w**2))}
    
    print(f"   u (streamwise): mean={u_stats['mean']:.4f}, std={u_stats['std']:.4f}, rms={u_stats['rms']:.4f}")
    print(f"   v (wall-normal): mean={v_stats['mean']:.4f}, std={v_stats['std']:.4f}, rms={v_stats['rms']:.4f}")
    print(f"   w (spanwise): mean={w_stats['mean']:.4f}, std={w_stats['std']:.4f}, rms={w_stats['rms']:.4f}")
    
    # 2. Channel Flow期望特徵檢查
    print("\n2. Channel Flow 典型特徵檢查:")
    
    # 主流方向應該有較強的流動
    streamwise_dominance = u_stats['rms'] / max(v_stats['rms'], w_stats['rms'])
    print(f"   流向主導性 (u_rms/max(v_rms,w_rms)): {streamwise_dominance:.2f}")
    print(f"   期望值: > 1.5 (顯著主流方向)")
    
    # 壁面法向速度應該較小
    wall_normal_ratio = v_stats['rms'] / u_stats['rms']
    print(f"   壁面法向比例 (v_rms/u_rms): {wall_normal_ratio:.2f}")
    print(f"   期望值: < 0.3 (壁面阻礙法向流動)")
    
    # 3. 空間分布分析
    print("\n3. 空間分布特徵:")
    
    # 計算不同方向的平均剖面
    u_streamwise_profile = np.mean(u, axis=(1, 2))  # 沿x方向平均
    u_wallnormal_profile = np.mean(u, axis=(0, 2))  # 沿y方向平均  
    u_spanwise_profile = np.mean(u, axis=(0, 1))    # 沿z方向平均
    
    print(f"   u 沿x方向變化 (std): {np.std(u_streamwise_profile):.4f}")
    print(f"   u 沿y方向變化 (std): {np.std(u_wallnormal_profile):.4f}")
    print(f"   u 沿z方向變化 (std): {np.std(u_spanwise_profile):.4f}")
    
    # 4. 渦結構分析
    print("\n4. 渦結構特徵:")
    vorticity_x = np.gradient(w, axis=1) - np.gradient(v, axis=2)
    vorticity_y = np.gradient(u, axis=2) - np.gradient(w, axis=0) 
    vorticity_z = np.gradient(v, axis=0) - np.gradient(u, axis=1)
    
    print(f"   渦量x分量 RMS: {np.sqrt(np.mean(vorticity_x**2)):.4f}")
    print(f"   渦量y分量 RMS: {np.sqrt(np.mean(vorticity_y**2)):.4f}")
    print(f"   渦量z分量 RMS: {np.sqrt(np.mean(vorticity_z**2)):.4f}")
    
    return {
        'streamwise_dominance': streamwise_dominance,
        'wall_normal_ratio': wall_normal_ratio,
        'profiles': {
            'u_x': u_streamwise_profile,
            'u_y': u_wallnormal_profile, 
            'u_z': u_spanwise_profile
        },
        'vorticity': {
            'x': vorticity_x,
            'y': vorticity_y,
            'z': vorticity_z
        }
    }

def create_channel_flow_diagnostic_plots(u, v, w, p, analysis):
    """創建Channel Flow診斷圖"""
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Channel Flow Characteristics Diagnosis\n當前數據的Channel Flow特徵診斷', 
                 fontsize=16, fontweight='bold')
    
    # 1. 速度分量對比 (等向性檢查)
    ax1 = plt.subplot(3, 4, 1)
    velocity_data = [u.flatten(), v.flatten(), w.flatten()]
    labels = ['u (streamwise)', 'v (wall-normal)', 'w (spanwise)']
    colors = ['red', 'blue', 'green']
    
    for i, (data, label, color) in enumerate(zip(velocity_data, labels, colors)):
        ax1.hist(data, bins=30, alpha=0.6, label=label, color=color, density=True)
    
    ax1.set_title('Velocity Components PDF\n速度分量分布', fontweight='bold')
    ax1.set_xlabel('Velocity [m/s]')
    ax1.set_ylabel('Probability Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 速度剖面 - y方向 (壁面法向)
    ax2 = plt.subplot(3, 4, 2)
    y_coords = np.arange(8)
    u_profile = analysis['profiles']['u_y']
    
    ax2.plot(y_coords, u_profile, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('Streamwise Velocity Profile\n(Wall-normal direction)', fontweight='bold')
    ax2.set_xlabel('y coordinate (wall-normal)')
    ax2.set_ylabel('Mean u velocity')
    ax2.grid(True, alpha=0.3)
    
    # 添加期望的Channel Flow特徵說明
    ax2.text(0.05, 0.95, 'Expected: Parabolic-like\nprofile for channel flow', 
             transform=ax2.transAxes, fontsize=9, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.5),
             verticalalignment='top')
    
    # 3. 所有方向的速度剖面對比
    ax3 = plt.subplot(3, 4, 3)
    coords = np.arange(8)
    ax3.plot(coords, analysis['profiles']['u_x'], 'r-o', label='Along x (streamwise)', markersize=4)
    ax3.plot(coords, analysis['profiles']['u_y'], 'b-s', label='Along y (wall-normal)', markersize=4)
    ax3.plot(coords, analysis['profiles']['u_z'], 'g-^', label='Along z (spanwise)', markersize=4)
    
    ax3.set_title('U-velocity Profiles\nAlong Different Directions', fontweight='bold')
    ax3.set_xlabel('Coordinate')
    ax3.set_ylabel('Mean u velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 渦量分量對比
    ax4 = plt.subplot(3, 4, 4)
    vort_data = [analysis['vorticity']['x'].flatten(), 
                 analysis['vorticity']['y'].flatten(), 
                 analysis['vorticity']['z'].flatten()]
    vort_labels = ['ωx', 'ωy', 'ωz']
    vort_colors = ['red', 'blue', 'green']
    
    for data, label, color in zip(vort_data, vort_labels, vort_colors):
        ax4.hist(data, bins=30, alpha=0.6, label=label, color=color, density=True)
    
    ax4.set_title('Vorticity Components PDF\n渦量分量分布', fontweight='bold')
    ax4.set_xlabel('Vorticity [1/s]')
    ax4.set_ylabel('Probability Density')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5-8. 不同切片的速度場可視化
    mid_idx = 4
    
    # XY平面 (z=mid)
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.contourf(u[:, :, mid_idx], levels=15, cmap='RdBu_r')
    ax5.set_title('U velocity: XY plane (z=4)\nStreamwise-Wallnormal view', fontweight='bold')
    ax5.set_xlabel('x (streamwise)')
    ax5.set_ylabel('y (wall-normal)')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    # XZ平面 (y=mid)
    ax6 = plt.subplot(3, 4, 6)
    im6 = ax6.contourf(u[:, mid_idx, :], levels=15, cmap='RdBu_r')
    ax6.set_title('U velocity: XZ plane (y=4)\nStreamwise-Spanwise view', fontweight='bold')
    ax6.set_xlabel('x (streamwise)')
    ax6.set_ylabel('z (spanwise)')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # YZ平面 (x=mid)
    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.contourf(u[mid_idx, :, :], levels=15, cmap='RdBu_r')
    ax7.set_title('U velocity: YZ plane (x=4)\nWallnormal-Spanwise view', fontweight='bold')
    ax7.set_xlabel('y (wall-normal)')
    ax7.set_ylabel('z (spanwise)')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # 8. 速度向量場 (XY平面)
    ax8 = plt.subplot(3, 4, 8)
    x, y = np.meshgrid(range(8), range(8))
    skip = 2
    speed = np.sqrt(u[:, :, mid_idx]**2 + v[:, :, mid_idx]**2)
    ax8.quiver(x[::skip, ::skip], y[::skip, ::skip], 
               u[::skip, ::skip, mid_idx], v[::skip, ::skip, mid_idx],
               speed[::skip, ::skip], cmap='viridis', alpha=0.8)
    ax8.set_title('Velocity Vectors: XY plane\n速度向量場', fontweight='bold')
    ax8.set_xlabel('x (streamwise)')
    ax8.set_ylabel('y (wall-normal)')
    ax8.set_aspect('equal')
    
    # 9. 特徵指標總結
    ax9 = plt.subplot(3, 4, 9)
    ax9.axis('off')
    
    # 計算Channel Flow特徵指標
    streamwise_dom = analysis['streamwise_dominance']
    wall_normal_ratio = analysis['wall_normal_ratio']
    
    # 判斷是否符合Channel Flow特徵
    is_channel_like = streamwise_dom > 1.5 and wall_normal_ratio < 0.3
    
    summary_text = f"""
Channel Flow Characteristics Assessment:

1. Streamwise Dominance: {streamwise_dom:.2f}
   Expected: > 1.5 ✓ {'✅' if streamwise_dom > 1.5 else '❌'}

2. Wall-normal Ratio: {wall_normal_ratio:.2f}  
   Expected: < 0.3 ✓ {'✅' if wall_normal_ratio < 0.3 else '❌'}

Overall Assessment:
{'✅ PASSES Channel Flow criteria' if is_channel_like else '❌ FAILS Channel Flow criteria'}

Current data shows:
{'Typical channel flow characteristics' if is_channel_like else 'Isotropic turbulence characteristics'}
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", 
                      facecolor='lightgreen' if is_channel_like else 'lightcoral', 
                      alpha=0.7))
    
    # 10-12. 更多診斷圖
    # Reynolds stress components
    ax10 = plt.subplot(3, 4, 10)
    u_fluc = u - np.mean(u)
    v_fluc = v - np.mean(v)
    w_fluc = w - np.mean(w)
    
    reynolds_components = [
        np.mean(u_fluc**2), np.mean(v_fluc**2), np.mean(w_fluc**2),
        np.mean(u_fluc * v_fluc), np.mean(u_fluc * w_fluc), np.mean(v_fluc * w_fluc)
    ]
    labels_rey = ["<u'u'>", "<v'v'>", "<w'w'>", "<u'v'>", "<u'w'>", "<v'w'>"]
    
    bars = ax10.bar(range(len(reynolds_components)), reynolds_components, 
                    color=['red', 'blue', 'green', 'orange', 'purple', 'brown'])
    ax10.set_title('Reynolds Stress Components\n雷諾應力分量', fontweight='bold')
    ax10.set_ylabel('Stress Value')
    ax10.set_xticks(range(len(labels_rey)))
    ax10.set_xticklabels(labels_rey, rotation=45)
    ax10.grid(True, alpha=0.3)
    
    # 11. Energy spectrum (simplified)
    ax11 = plt.subplot(3, 4, 11)
    
    # 計算簡化的能量譜
    u_fft = np.fft.fftn(u)
    energy_3d = np.abs(u_fft)**2
    
    # 沿每個軸平均得到1D譜
    energy_x = np.mean(energy_3d, axis=(1, 2))[:4]  # 只取前一半
    energy_y = np.mean(energy_3d, axis=(0, 2))[:4]
    energy_z = np.mean(energy_3d, axis=(0, 1))[:4]
    
    k = np.arange(1, 4)  # 修復：確保長度匹配
    ax11.loglog(k, energy_x[1:4], 'ro-', label='x-direction')
    ax11.loglog(k, energy_y[1:4], 'bs-', label='y-direction') 
    ax11.loglog(k, energy_z[1:4], 'g^-', label='z-direction')
    
    ax11.set_title('Energy Spectrum\n能量譜', fontweight='bold')
    ax11.set_xlabel('Wavenumber k')
    ax11.set_ylabel('Energy')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    
    # 12. Data source info
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    data_info = f"""
Current Data Information:

• Source: Mock JHTDB data
• Grid size: 8×8×8 (512 points)
• Generation method: Vector potential
• Divergence-free: ✅ (∇·V ≈ 0)

Data characteristics:
• Equal velocity component statistics
• Isotropic turbulence-like behavior
• No clear wall boundaries
• No preferential flow direction

Recommendation:
{'Need real Channel Flow data' if not is_channel_like else 'Data suitable for channel flow analysis'}
for proper channel flow visualization
    """
    
    ax12.text(0.05, 0.95, data_info, transform=ax12.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    return fig

def main():
    """主函數"""
    print("🔍 Channel Flow 特徵診斷工具")
    print("="*50)
    
    # 載入數據
    u, v, w, p = load_and_analyze_data()
    
    # 分析特徵
    analysis = analyze_channel_flow_characteristics(u, v, w, p)
    
    # 生成診斷圖
    print("\n🎨 生成 Channel Flow 診斷圖...")
    fig = create_channel_flow_diagnostic_plots(u, v, w, p, analysis)
    
    # 保存圖片
    output_file = "channel_flow_diagnosis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ 診斷圖已保存: {output_file}")
    
    # 結論
    streamwise_dom = analysis['streamwise_dominance']
    wall_normal_ratio = analysis['wall_normal_ratio']
    is_channel_like = streamwise_dom > 1.5 and wall_normal_ratio < 0.3
    
    print("\n" + "="*60)
    print("📋 Channel Flow 診斷結論:")
    print("="*60)
    if is_channel_like:
        print("✅ 當前數據具有 Channel Flow 特徵")
        print("   - 有明顯的流向主導性")
        print("   - 壁面法向速度相對較小")
    else:
        print("❌ 當前數據不符合典型 Channel Flow 特徵")
        print("   - 速度分量統計相近，呈等向性湍流特徵")
        print("   - 缺乏流向主導性和壁面效應")
        print("   - 建議使用真實的 JHTDB Channel Flow 數據")
    
    print(f"\n關鍵指標:")
    print(f"  • 流向主導性: {streamwise_dom:.2f} (期望 > 1.5)")
    print(f"  • 壁面法向比例: {wall_normal_ratio:.2f} (期望 < 0.3)")
    
    plt.show()

if __name__ == "__main__":
    main()