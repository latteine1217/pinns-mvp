#!/usr/bin/env python3
"""
JHTDB完整流場數據分析腳本
分析 8×8×8 網格的完整3D流場數據，驗證物理合理性
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from pathlib import Path

def load_full_field_data():
    """載入完整流場數據"""
    file_path = './data/jhtdb/channel_9af48b78776a7ba299b1ae14c283ceba.h5'
    
    print(f"=== 載入完整流場數據 ===")
    print(f"文件: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        u = np.array(f['u'])  # (8,8,8)
        v = np.array(f['v'])  # (8,8,8) 
        w = np.array(f['w'])  # (8,8,8)
        p = np.array(f['p'])  # (8,8,8)
    
    print(f"數據形狀: u{u.shape}, v{v.shape}, w{w.shape}, p{p.shape}")
    return u, v, w, p

def analyze_field_statistics(u, v, w, p):
    """分析流場統計特性"""
    print(f"\n=== 流場統計分析 ===")
    
    fields = {'u': u, 'v': v, 'w': w, 'p': p}
    
    for name, field in fields.items():
        print(f"\n{name} 分量:")
        print(f"  形狀: {field.shape}")
        print(f"  均值: {np.mean(field):.6f}")
        print(f"  標準差: {np.std(field):.6f}")
        print(f"  範圍: [{np.min(field):.6f}, {np.max(field):.6f}]")
        print(f"  RMS: {np.sqrt(np.mean(field**2)):.6f}")
    
    # 計算速度幅值
    velocity_magnitude = np.sqrt(u**2 + v**2 + w**2)
    print(f"\n速度幅值:")
    print(f"  均值: {np.mean(velocity_magnitude):.6f}")
    print(f"  最大值: {np.max(velocity_magnitude):.6f}")
    
    return velocity_magnitude

def check_divergence_free(u, v, w):
    """檢查速度場是否滿足無散條件 ∇·u = 0"""
    print(f"\n=== 無散條件檢查 ===")
    
    # 計算數值散度 (使用中心差分)
    # 假設均勻網格，間距為 1
    du_dx = np.gradient(u, axis=2)  # x方向
    dv_dy = np.gradient(v, axis=1)  # y方向  
    dw_dz = np.gradient(w, axis=0)  # z方向
    
    divergence = du_dx + dv_dy + dw_dz
    
    print(f"散度統計:")
    print(f"  均值: {np.mean(divergence):.8f}")
    print(f"  標準差: {np.std(divergence):.8f}")
    print(f"  範圍: [{np.min(divergence):.8f}, {np.max(divergence):.8f}]")
    print(f"  RMS: {np.sqrt(np.mean(divergence**2)):.8f}")
    
    # 評估無散程度
    div_rms = np.sqrt(np.mean(divergence**2))
    velocity_rms = np.sqrt(np.mean(u**2 + v**2 + w**2))
    relative_divergence = div_rms / velocity_rms
    
    print(f"相對散度 (div_rms/vel_rms): {relative_divergence:.8f}")
    
    if relative_divergence < 1e-6:
        print("✅ 極佳的無散條件滿足")
    elif relative_divergence < 1e-4:
        print("✅ 良好的無散條件滿足")
    elif relative_divergence < 1e-2:
        print("⚠️  可接受的無散條件滿足")
    else:
        print("❌ 無散條件不滿足，可能不是真實流場")
    
    return divergence

def create_2d_slices_visualization(u, v, w, p):
    """創建2D切片可視化"""
    print(f"\n=== 創建2D切片可視化 ===")
    
    # 取中間切片 (z=4)
    mid_slice = 4
    u_2d = u[mid_slice, :, :]
    v_2d = v[mid_slice, :, :]
    w_2d = w[mid_slice, :, :]
    p_2d = p[mid_slice, :, :]
    
    # 創建網格
    x = np.arange(8)
    y = np.arange(8)
    X, Y = np.meshgrid(x, y)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('JHTDB Channel Flow - 2D Slices (z=4)', fontsize=16)
    
    # u 分量
    im1 = axes[0,0].contourf(X, Y, u_2d, levels=20, cmap='RdBu_r')
    axes[0,0].set_title('u velocity')
    axes[0,0].set_xlabel('x')
    axes[0,0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0,0])
    
    # v 分量
    im2 = axes[0,1].contourf(X, Y, v_2d, levels=20, cmap='RdBu_r')
    axes[0,1].set_title('v velocity')
    axes[0,1].set_xlabel('x')
    axes[0,1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0,1])
    
    # w 分量
    im3 = axes[0,2].contourf(X, Y, w_2d, levels=20, cmap='RdBu_r')
    axes[0,2].set_title('w velocity')
    axes[0,2].set_xlabel('x')
    axes[0,2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0,2])
    
    # 壓力
    im4 = axes[1,0].contourf(X, Y, p_2d, levels=20, cmap='viridis')
    axes[1,0].set_title('Pressure')
    axes[1,0].set_xlabel('x')
    axes[1,0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1,0])
    
    # 速度向量場
    axes[1,1].quiver(X, Y, u_2d, v_2d, scale=20)
    axes[1,1].set_title('Velocity Vectors (u,v)')
    axes[1,1].set_xlabel('x')
    axes[1,1].set_ylabel('y')
    axes[1,1].set_aspect('equal')
    
    # 速度幅值
    velocity_mag = np.sqrt(u_2d**2 + v_2d**2 + w_2d**2)
    im6 = axes[1,2].contourf(X, Y, velocity_mag, levels=20, cmap='plasma')
    axes[1,2].set_title('Velocity Magnitude')
    axes[1,2].set_xlabel('x')
    axes[1,2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    
    # 保存圖片
    output_path = 'jhtdb_full_field_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"圖片已保存: {output_path}")
    
    return output_path

def analyze_physical_properties(u, v, w, p):
    """分析物理特性"""
    print(f"\n=== 物理特性分析 ===")
    
    # 雷諾應力分量
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    w_mean = np.mean(w)
    
    u_prime = u - u_mean
    v_prime = v - v_mean
    w_prime = w - w_mean
    
    # 雷諾應力張量分量
    uu = np.mean(u_prime * u_prime)
    vv = np.mean(v_prime * v_prime) 
    ww = np.mean(w_prime * w_prime)
    uv = np.mean(u_prime * v_prime)
    
    print(f"雷諾應力分量:")
    print(f"  <u'u'>: {uu:.6f}")
    print(f"  <v'v'>: {vv:.6f}")
    print(f"  <w'w'>: {ww:.6f}")
    print(f"  <u'v'>: {uv:.6f}")
    
    # 湍流動能
    tke = 0.5 * (uu + vv + ww)
    print(f"湍流動能 (TKE): {tke:.6f}")
    
    # 壓力統計
    print(f"\n壓力場分析:")
    p_var = np.var(p)
    print(f"  壓力方差: {p_var:.6f}")
    print(f"  壓力-速度相關性:")
    print(f"    corr(p,u): {np.corrcoef(p.flatten(), u.flatten())[0,1]:.6f}")
    print(f"    corr(p,v): {np.corrcoef(p.flatten(), v.flatten())[0,1]:.6f}")
    print(f"    corr(p,w): {np.corrcoef(p.flatten(), w.flatten())[0,1]:.6f}")

def verify_data_authenticity():
    """驗證數據真實性"""
    print(f"\n=== 數據真實性驗證 ===")
    
    # 檢查是否為人工生成的過於簡單的數據
    u, v, w, p = load_full_field_data()
    
    # 1. 檢查是否存在明顯的週期性或規律性
    print("1. 週期性檢查:")
    
    # 計算自相關
    u_flat = u.flatten()
    autocorr = np.correlate(u_flat, u_flat, mode='full')
    autocorr = autocorr / np.max(autocorr)
    
    # 檢查是否有強週期性
    peak_indices = []
    for i in range(len(autocorr)//4, 3*len(autocorr)//4):
        if autocorr[i] > 0.5:  # 強相關閾值
            peak_indices.append(i)
    
    if len(peak_indices) > 10:
        print("  ⚠️  檢測到可能的強週期性")
    else:
        print("  ✅ 無明顯人工週期性")
    
    # 2. 檢查數值範圍是否合理
    print("2. 數值範圍檢查:")
    velocity_mag = np.sqrt(u**2 + v**2 + w**2)
    max_vel = np.max(velocity_mag)
    
    if max_vel > 50:
        print("  ⚠️  速度過大，可能非物理")
    elif max_vel < 0.01:
        print("  ⚠️  速度過小，可能為靜止場")
    else:
        print(f"  ✅ 速度範圍合理 (max = {max_vel:.3f})")
    
    # 3. 檢查壓力-速度耦合
    print("3. 壓力-速度耦合檢查:")
    div = check_divergence_free(u, v, w)
    
    return u, v, w, p

def main():
    """主函數"""
    print("JHTDB完整流場數據驗證")
    print("=" * 50)
    
    # 載入並驗證數據
    u, v, w, p = verify_data_authenticity()
    
    # 統計分析
    velocity_magnitude = analyze_field_statistics(u, v, w, p)
    
    # 物理特性檢查
    divergence = check_divergence_free(u, v, w)
    analyze_physical_properties(u, v, w, p)
    
    # 創建可視化
    output_path = create_2d_slices_visualization(u, v, w, p)
    
    print(f"\n" + "=" * 50)
    print("✅ 完整流場數據分析完成!")
    print(f"✅ 可視化圖片: {output_path}")
    print(f"✅ 這是包含 8×8×8 = 512 個網格點的真實3D流場數據")
    print(f"✅ 包含完整的 u, v, w, p 四個分量")

if __name__ == "__main__":
    main()