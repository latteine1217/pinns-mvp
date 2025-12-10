#!/usr/bin/env python3
"""
JHTDB 流場與壓力場專業分布圖生成器
生成高質量的流場和壓力場分布圖，包含多種視角和分析
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import h5py
from pathlib import Path

# 使用內建風格替代seaborn
plt.style.use('default')

# 設置matplotlib中文字體和風格
# 優先順序：MacOS (Arial Unicode MS, PingFang HK) -> Windows (Microsoft JhengHei, SimHei) -> Linux (WenQuanYi) -> Fallback
# Note: 'PingFang TC' might not be available, 'PingFang HK' was found on this system.
font_list = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'PingFang TC', 'Microsoft JhengHei', 'SimHei', 'WenQuanYi Micro Hei', 'Noto Sans CJK TC', 'sans-serif']
plt.rcParams['font.sans-serif'] = font_list
plt.rcParams['axes.unicode_minus'] = False

print(f"ℹ️ Matplotlib font configuration set to: {plt.rcParams['font.sans-serif']}")
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def load_jhtdb_data():
    """載入JHTDB流場數據"""
    velocity_path = 'data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_velocity_t1.h5'
    pressure_path = 'data/jhtdb/channel_flow_re1000/raw/JHU Turbulence Channel_pressure_t1.h5'
    
    print(f"🔄 載入JHTDB數據: {velocity_path} & {pressure_path}")
    
    with h5py.File(velocity_path, 'r') as f:
        # Shape (512, 128, 512, 3) -> (x, y, z, component)
        # Loading full data might be heavy, but let's try.
        vel = f['Velocity_0001'][:]
        u = vel[..., 0]
        v = vel[..., 1]
        w = vel[..., 2]
        
    with h5py.File(pressure_path, 'r') as f:
        p = f['Pressure_0001'][..., 0]
    
    print(f"✅ 數據載入成功: u{u.shape}, v{v.shape}, w{w.shape}, p{p.shape}")
    return u, v, w, p

def create_comprehensive_field_plots(u, v, w, p):
    """創建綜合流場分布圖"""
    
    # 計算衍生場
    velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
    vorticity_z = np.gradient(v, axis=0) - np.gradient(u, axis=1)  # 簡化渦量
    
    # 選擇中間切片進行可視化
    mid_idx = u.shape[2] // 2
    
    # 創建大圖布局
    fig = plt.figure(figsize=(20, 16))
    
    # 主標題
    fig.suptitle('JHTDB Channel Flow: Velocity & Pressure Field Distribution\n' + 
                 '湍流通道流：速度場與壓力場分布分析', 
                 fontsize=18, fontweight='bold', y=0.95)
    
    # === 第一行：速度分量 ===
    
    # u 分量 (流向速度)
    ax1 = plt.subplot(3, 4, 1)
    im1 = ax1.contourf(u[:, :, mid_idx].T, levels=20, cmap='RdBu_r', extend='both')
    ax1.set_title('Streamwise Velocity (u)\n流向速度', fontsize=12, fontweight='bold')
    ax1.set_xlabel('x direction')
    ax1.set_ylabel('y direction')
    plt.colorbar(im1, ax=ax1, shrink=0.8, label='u [m/s]')
    
    # v 分量 (法向速度)
    ax2 = plt.subplot(3, 4, 2)
    im2 = ax2.contourf(v[:, :, mid_idx].T, levels=20, cmap='RdBu_r', extend='both')
    ax2.set_title('Wall-normal Velocity (v)\n壁面法向速度', fontsize=12, fontweight='bold')
    ax2.set_xlabel('x direction')
    ax2.set_ylabel('y direction')
    plt.colorbar(im2, ax=ax2, shrink=0.8, label='v [m/s]')
    
    # w 分量 (展向速度)
    ax3 = plt.subplot(3, 4, 3)
    im3 = ax3.contourf(w[:, :, mid_idx].T, levels=20, cmap='RdBu_r', extend='both')
    ax3.set_title('Spanwise Velocity (w)\n展向速度', fontsize=12, fontweight='bold')
    ax3.set_xlabel('x direction')
    ax3.set_ylabel('y direction')
    plt.colorbar(im3, ax=ax3, shrink=0.8, label='w [m/s]')
    
    # 速度幅值
    ax4 = plt.subplot(3, 4, 4)
    im4 = ax4.contourf(velocity_magnitude[:, :, mid_idx].T, levels=20, cmap='viridis', extend='both')
    ax4.set_title('Velocity Magnitude\n速度幅值', fontsize=12, fontweight='bold')
    ax4.set_xlabel('x direction')
    ax4.set_ylabel('y direction')
    plt.colorbar(im4, ax=ax4, shrink=0.8, label='|V| [m/s]')
    
    # === 第二行：壓力場與向量場 ===
    
    # 壓力場
    ax5 = plt.subplot(3, 4, 5)
    im5 = ax5.contourf(p[:, :, mid_idx].T, levels=20, cmap='coolwarm', extend='both')
    ax5.set_title('Pressure Field\n壓力場', fontsize=12, fontweight='bold')
    ax5.set_xlabel('x direction')
    ax5.set_ylabel('y direction')
    plt.colorbar(im5, ax=ax5, shrink=0.8, label='p [Pa]')
    
    # 速度向量場
    ax6 = plt.subplot(3, 4, 6)
    # 降採樣以避免向量過於密集
    skip = 32
    # 注意：因為使用了 .T (轉置)，這裡的 meshgrid 也要對應
    ny, nx = u[:, :, mid_idx].T.shape
    x, y = np.meshgrid(range(nx), range(ny))
    
    ax6.quiver(x[::skip, ::skip], y[::skip, ::skip], 
               u[::skip, ::skip, mid_idx].T, v[::skip, ::skip, mid_idx].T,
               velocity_magnitude[::skip, ::skip, mid_idx].T, 
               cmap='viridis', alpha=0.8)
    ax6.set_title('Velocity Vector Field\n速度向量場', fontsize=12, fontweight='bold')
    ax6.set_xlabel('x direction')
    ax6.set_ylabel('y direction')
    ax6.set_aspect('equal')
    
    # 簡化渦量 (z方向)
    ax7 = plt.subplot(3, 4, 7)
    im7 = ax7.contourf(vorticity_z[:, :, mid_idx].T, levels=20, cmap='RdBu_r', extend='both')
    ax7.set_title('Vorticity (ωz)\n渦量', fontsize=12, fontweight='bold')
    ax7.set_xlabel('x direction')
    ax7.set_ylabel('y direction')
    plt.colorbar(im7, ax=ax7, shrink=0.8, label='ωz [1/s]')
    
    # 散度檢查
    ax8 = plt.subplot(3, 4, 8)
    div_u = np.gradient(u, axis=0) + np.gradient(v, axis=1) + np.gradient(w, axis=2)
    im8 = ax8.contourf(div_u[:, :, mid_idx].T, levels=20, cmap='RdBu_r', extend='both')
    ax8.set_title('Velocity Divergence\n速度散度', fontsize=12, fontweight='bold')
    ax8.set_xlabel('x direction')
    ax8.set_ylabel('y direction')
    plt.colorbar(im8, ax=ax8, shrink=0.8, label='∇·V [1/s]')
    
    # === 第三行：統計分析 ===
    
    # 速度分量統計分布
    ax9 = plt.subplot(3, 4, 9)
    ax9.hist(u.flatten(), bins=30, alpha=0.6, label='u', density=True)
    ax9.hist(v.flatten(), bins=30, alpha=0.6, label='v', density=True)
    ax9.hist(w.flatten(), bins=30, alpha=0.6, label='w', density=True)
    ax9.set_title('Velocity Components PDF\n速度分量概率密度', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Velocity [m/s]')
    ax9.set_ylabel('Probability Density')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 壓力統計分布
    ax10 = plt.subplot(3, 4, 10)
    ax10.hist(p.flatten(), bins=30, alpha=0.8, color='red', density=True)
    ax10.set_title('Pressure PDF\n壓力概率密度', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Pressure [Pa]')
    ax10.set_ylabel('Probability Density')
    ax10.grid(True, alpha=0.3)
    
    # 雷諾應力分析
    ax11 = plt.subplot(3, 4, 11)
    u_fluc = u - np.mean(u)
    v_fluc = v - np.mean(v)
    reynolds_stress = u_fluc * v_fluc
    im11 = ax11.contourf(reynolds_stress[:, :, mid_idx].T, levels=20, cmap='coolwarm', extend='both')
    ax11.set_title("Reynolds Stress (u'v')\n雷諾應力", fontsize=12, fontweight='bold')
    ax11.set_xlabel('x direction')
    ax11.set_ylabel('y direction')
    plt.colorbar(im11, ax=ax11, shrink=0.8, label="u'v' [m²/s²]")
    
    # 湍流動能
    ax12 = plt.subplot(3, 4, 12)
    tke = 0.5 * (u_fluc**2 + v_fluc**2 + (w - np.mean(w))**2)
    im12 = ax12.contourf(tke[:, :, mid_idx].T, levels=20, cmap='viridis', extend='both')
    ax12.set_title('Turbulent Kinetic Energy\n湍流動能', fontsize=12, fontweight='bold')
    ax12.set_xlabel('x direction')
    ax12.set_ylabel('y direction')
    plt.colorbar(im12, ax=ax12, shrink=0.8, label='TKE [m²/s²]')
    
    plt.tight_layout()
    return fig

def print_field_statistics(u, v, w, p):
    """打印詳細的場統計信息"""
    print("\n" + "="*60)
    print("📊 JHTDB Channel Flow Field Statistics")
    print("="*60)
    
    fields = {'u (Streamwise)': u, 'v (Wall-normal)': v, 'w (Spanwise)': w, 'p (Pressure)': p}
    
    for name, field in fields.items():
        print(f"\n{name}:")
        print(f"  Mean: {np.mean(field):.6f}")
        print(f"  Std:  {np.std(field):.6f}")
        print(f"  Min:  {np.min(field):.6f}")
        print(f"  Max:  {np.max(field):.6f}")
        print(f"  RMS:  {np.sqrt(np.mean(field**2)):.6f}")
    
    # 物理量
    velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
    print(f"\nVelocity Magnitude:")
    print(f"  Mean: {np.mean(velocity_magnitude):.6f}")
    print(f"  Max:  {np.max(velocity_magnitude):.6f}")
    
    # 散度檢查
    div_u = np.gradient(u, axis=0) + np.gradient(v, axis=1) + np.gradient(w, axis=2)
    print(f"\nDivergence Check:")
    print(f"  ∇·V RMS: {np.sqrt(np.mean(div_u**2)):.6f}")
    print(f"  Relative: {np.sqrt(np.mean(div_u**2))/np.sqrt(np.mean(velocity_magnitude**2)):.4f}")
    
    # 雷諾應力
    u_fluc = u - np.mean(u)
    v_fluc = v - np.mean(v)
    w_fluc = w - np.mean(w)
    
    print(f"\nReynolds Stresses:")
    print(f"  <u'u'>: {np.mean(u_fluc**2):.6f}")
    print(f"  <v'v'>: {np.mean(v_fluc**2):.6f}")
    print(f"  <w'w'>: {np.mean(w_fluc**2):.6f}")
    print(f"  <u'v'>: {np.mean(u_fluc * v_fluc):.6f}")
    
    print(f"\nTurbulent Kinetic Energy:")
    tke = 0.5 * (np.mean(u_fluc**2) + np.mean(v_fluc**2) + np.mean(w_fluc**2))
    print(f"  TKE: {tke:.6f}")

def main():
    """主函數"""
    print("🌊 生成 JHTDB 流場與壓力場分布圖")
    print("="*50)
    
    # 載入數據
    u, v, w, p = load_jhtdb_data()
    
    # 打印統計信息
    print_field_statistics(u, v, w, p)
    
    # 生成綜合分布圖
    print("\n🎨 生成綜合流場分布圖...")
    fig = create_comprehensive_field_plots(u, v, w, p)
    
    # 保存圖片
    output_file = "jhtdb_comprehensive_field_analysis.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"✅ 圖片已保存: {output_file}")
    print(f"✅ 圖片尺寸: {fig.get_size_inches()}")
    print(f"✅ 分辨率: 300 DPI")
    
    # 顯示圖片信息
    print("\n📋 圖片內容:")
    print("  第一行: u, v, w 速度分量 + 速度幅值")
    print("  第二行: 壓力場 + 向量場 + 渦量 + 散度")
    print("  第三行: 統計分布 + 雷諾應力 + 湍流動能")
    
    plt.show()

if __name__ == "__main__":
    main()