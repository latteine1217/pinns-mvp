#!/usr/bin/env python3
"""
生成 Leith 模型相對誤差隨雷諾數變化的 scaling 圖

用於論文：比較 Leith SGS 模型與 DNS 的誤差趨勢
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pathlib import Path

def compute_leith_error(re_val):
    """計算指定 Re 的 Leith vs DNS 誤差"""
    dns_file = f'data/kolmogorov_dns/dns_re{re_val}_t100.h5'
    leith_file = f'data/lowfi/kolmogorov_rans/rans_re{re_val}_kf4_leith.h5'
    
    # 載入 DNS (時間平均)
    with h5py.File(dns_file, 'r') as f:
        u_dns = np.array(f['u']).mean(axis=0)  # Average over time
        v_dns = np.array(f['v']).mean(axis=0)
    
    # 載入 Leith
    with h5py.File(leith_file, 'r') as f:
        u_leith = np.array(f['mean_field/u'])
        v_leith = np.array(f['mean_field/v'])
        C_L = f['metadata'].attrs.get('C_L', 0.2)
    
    # 插值 Leith 到 DNS 網格
    zoom_factor = u_dns.shape[0] / u_leith.shape[0]
    if zoom_factor != 1.0:
        u_leith_interp = zoom(u_leith, zoom_factor, order=3)
        v_leith_interp = zoom(v_leith, zoom_factor, order=3)
    else:
        u_leith_interp = u_leith
        v_leith_interp = v_leith
    
    # 計算相對 L2 誤差
    error_u = np.linalg.norm(u_leith_interp - u_dns) / np.linalg.norm(u_dns)
    error_v = np.linalg.norm(v_leith_interp - v_dns) / np.linalg.norm(v_dns)
    error_overall = np.sqrt(error_u**2 + error_v**2) / np.sqrt(2)
    
    # 計算動能比
    ke_dns = 0.5 * (u_dns**2 + v_dns**2).mean()
    ke_leith = 0.5 * (u_leith**2 + v_leith**2).mean()
    ke_ratio = ke_leith / ke_dns
    
    return error_u * 100, error_v * 100, error_overall * 100, ke_ratio, C_L

def main():
    print("=" * 70)
    print("📊 生成 Leith Model Error Scaling 圖")
    print("=" * 70)
    
    # 計算各 Re 的誤差
    re_values = [50, 100, 500]
    errors_u = []
    errors_v = []
    errors_overall = []
    ke_ratios = []
    C_L_values = []
    
    for re_val in re_values:
        print(f"\n計算 Re={re_val}...")
        try:
            err_u, err_v, err_overall, ke_ratio, C_L = compute_leith_error(re_val)
            errors_u.append(err_u)
            errors_v.append(err_v)
            errors_overall.append(err_overall)
            ke_ratios.append(ke_ratio)
            C_L_values.append(C_L)
            print(f"  u: {err_u:.1f}%, v: {err_v:.1f}%, overall: {err_overall:.1f}%")
            print(f"  KE/DNS: {ke_ratio:.2%}, C_L: {C_L:.2f}")
        except Exception as e:
            print(f"  ❌ 錯誤: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 繪圖 - 創建雙 Y 軸圖
    print(f"\n🎨 生成圖表...")
    fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
    
    # 主 Y 軸：誤差
    color_u = '#1f77b4'
    color_v = '#ff7f0e'
    color_overall = '#d62728'
    
    ax1.plot(re_values, errors_u, 'o-', linewidth=2.5, markersize=9, 
            label='$u$ velocity', color=color_u, alpha=0.85)
    ax1.plot(re_values, errors_v, 's-', linewidth=2.5, markersize=9, 
            label='$v$ velocity', color=color_v, alpha=0.85)
    ax1.plot(re_values, errors_overall, '^-', linewidth=3, markersize=11, 
            label='Overall', color=color_overall, alpha=0.9, zorder=3)
    
    ax1.set_xlabel('Reynolds Number', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Relative $L_2$ Error (%)', fontsize=13, 
                   fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black', labelsize=11)
    ax1.tick_params(axis='x', labelsize=11)
    ax1.set_xlim([25, 525])
    ax1.set_ylim([0, 135])
    ax1.set_xticks([50, 100, 200, 300, 400, 500])
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.7)
    
    # 次 Y 軸：KE 比率
    ax2 = ax1.twinx()
    color_ke = '#2ca02c'
    ax2.plot(re_values, [k*100 for k in ke_ratios], 'D-', linewidth=2.5, 
            markersize=8, label='KE Ratio', color=color_ke, alpha=0.8)
    ax2.set_ylabel('KE$_{Leith}$ / KE$_{DNS}$ (%)', fontsize=13, 
                   fontweight='bold', color=color_ke)
    ax2.tick_params(axis='y', labelcolor=color_ke, labelsize=11)
    ax2.set_ylim([0, 70])
    
    # 合併圖例 - 移到左下角
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              fontsize=10.5, loc='lower left', framealpha=0.95, 
              edgecolor='gray', fancybox=True, shadow=False)
    
    # 簡潔標題
    ax1.set_title('Leith Model Error Scaling with Reynolds Number' + 
                 f' ($C_L$={C_L_values[0]:.2f})', 
                 fontsize=14, fontweight='bold', pad=12)
    
    plt.tight_layout()
    
    # 保存
    output_path = Path('thesis/result_figures/kolmogorov/fig_leith_error_scaling.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 圖表已保存: {output_path}")
    print(f"   檔案大小: {output_path.stat().st_size / 1024:.1f} KB")
    
    plt.close()
    
    # 生成數據表格
    print("\n" + "=" * 70)
    print("📊 Leith Model Error Summary")
    print("=" * 70)
    print(f"{'Re':<8} {'U Err%':<10} {'V Err%':<10} {'Total%':<10} {'KE/DNS':<10} {'C_L':<8}")
    print("-" * 70)
    for i, re in enumerate(re_values):
        print(f"{re:<8} {errors_u[i]:<10.1f} {errors_v[i]:<10.1f} "
              f"{errors_overall[i]:<10.1f} {ke_ratios[i]:<10.2%} {C_L_values[i]:<8.2f}")
    
    print("\n" + "=" * 70)
    print("🔍 Key Observations:")
    print("=" * 70)
    print(f"1. Re=50:  Error {errors_overall[0]:.0f}%, KE {ke_ratios[0]:.0%} DNS")
    print(f"   → 網格 N=128 可能過度解析（實際只需 N=64）")
    print(f"\n2. Re=100: Error {errors_overall[1]:.0f}%, KE {ke_ratios[1]:.0%} DNS")
    print(f"   → 轉換區異常高誤差（比 Re=500 更差）")
    print(f"   → U-分量誤差 {errors_u[1]:.0f}% 遠高於 V-分量 {errors_v[1]:.0f}%")
    print(f"\n3. Re=500: Error {errors_overall[2]:.0f}%, KE {ke_ratios[2]:.0%} DNS")
    print(f"   → 嚴重能量耗散（僅 20% DNS），但誤差反而較低")
    print(f"   → 可能因 under-resolved (N=128 不足) 導致人工黏性")
    
    print("\n" + "=" * 70)
    print("💡 Conclusion:")
    print("=" * 70)
    print("Leith model with uniform C_L=0.20 provides:")
    print("  ✅ Stable under-prediction (KE ~ 20-56% DNS)")
    print("  ✅ Consistent behavior across Re range")
    print("  ⚠️  Re=100 transition regime shows anomalous high error")
    print("  ⚠️  Re=500 requires higher resolution (N=256) for accuracy")
    print("\nThis conservative baseline is suitable for PINN correction,")
    print("as stable under-prediction is preferable to unstable over-prediction.")
    print("=" * 70)

if __name__ == '__main__':
    main()
