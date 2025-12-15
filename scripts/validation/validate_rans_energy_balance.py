#!/usr/bin/env python3
"""
RANS-k-ε 能量平衡驗證腳本
========================

驗證 RANS 模擬的能量平衡：
1. ⟨P_k⟩ ≈ ⟨ε⟩ （湍動能生產與耗散平衡）
2. y 方向剖面分析
3. 時間演化穩定性

使用範例：
--------
python scripts/validate_rans_energy_balance.py \
    --input data/kolmogorov_rans/rans_re100_kf4.h5 \
    --output results/rans_validation/

作者：PINNs-MVP 團隊
日期：2025-12-12
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_rans_data(file_path: Path):
    """載入 RANS 數據"""
    logging.info(f"載入 RANS 數據: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        # 時間平均場
        u_mean = f['mean_field/u'][:]
        v_mean = f['mean_field/v'][:]
        k_mean = f['mean_field/k'][:]
        eps_mean = f['mean_field/epsilon'][:]
        nu_t_mean = f['mean_field/nu_t'][:]
        X = f['mean_field/X'][:]
        Y = f['mean_field/Y'][:]
        
        # 時間序列
        time = f['statistics/time'][:]
        KE_t = f['statistics/kinetic_energy'][:]
        k_t = f['statistics/turbulent_kinetic_energy'][:]
        eps_t = f['statistics/dissipation_rate'][:]
        nu_t_t = f['statistics/nu_t_mean'][:]
        
        # 參數
        params = {}
        for key in f['parameters'].attrs.keys():
            params[key] = f['parameters'].attrs[key]
    
    logging.info(f"  網格: {params['N']}×{params['N']}")
    logging.info(f"  雷諾數估計: Re ≈ {np.sqrt(params['A']) * (2*np.pi/params['k_f'])**(3/2) / params['nu']:.1f}")
    
    return {
        'mean_field': {
            'u': u_mean,
            'v': v_mean,
            'k': k_mean,
            'epsilon': eps_mean,
            'nu_t': nu_t_mean,
            'X': X,
            'Y': Y,
        },
        'time_series': {
            'time': time,
            'KE': KE_t,
            'k': k_t,
            'epsilon': eps_t,
            'nu_t': nu_t_t,
        },
        'params': params,
    }


def compute_production_field(u_mean, v_mean, nu_t_mean, L):
    """
    從時間平均場重新計算 P_k = ν_t · |S|²
    
    使用有限差分計算梯度
    """
    N = u_mean.shape[0]
    dx = L / N
    
    # 梯度（週期邊界，中心差分）
    u_x = np.gradient(u_mean, dx, axis=0, edge_order=2)
    u_y = np.gradient(u_mean, dx, axis=1, edge_order=2)
    v_x = np.gradient(v_mean, dx, axis=0, edge_order=2)
    v_y = np.gradient(v_mean, dx, axis=1, edge_order=2)
    
    # 應變率張量
    S_xx = u_x
    S_yy = v_y
    S_xy = 0.5 * (u_y + v_x)
    
    # |S|² = 2·S_ij·S_ij
    S_mag_sq = 2 * (S_xx**2 + S_yy**2 + 2 * S_xy**2)
    
    # 生產項
    P_k = nu_t_mean * S_mag_sq
    
    return P_k


def validate_energy_balance(data, output_dir):
    """驗證能量平衡"""
    logging.info("\n=== 能量平衡驗證 ===")
    
    # 提取數據
    k_mean = data['mean_field']['k']
    eps_mean = data['mean_field']['epsilon']
    nu_t_mean = data['mean_field']['nu_t']
    u_mean = data['mean_field']['u']
    v_mean = data['mean_field']['v']
    
    L = data['params']['L']
    nu = data['params']['nu']
    
    # 重新計算 P_k 場
    P_k_field = compute_production_field(u_mean, v_mean, nu_t_mean, L)
    
    # 全域平均
    P_k_avg = np.mean(P_k_field)
    eps_avg = np.mean(eps_mean)
    
    # 能量平衡誤差
    balance_error = np.abs(P_k_avg - eps_avg) / eps_avg
    
    logging.info(f"  ⟨P_k⟩ = {P_k_avg:.6f}")
    logging.info(f"  ⟨ε⟩   = {eps_avg:.6f}")
    logging.info(f"  相對誤差 = {balance_error:.2%}")
    
    if balance_error < 0.05:
        logging.info("  ✅ 能量平衡良好 (誤差 < 5%)")
    elif balance_error < 0.10:
        logging.info("  ⚠️  能量平衡可接受 (誤差 5-10%)")
    else:
        logging.info("  ❌ 能量平衡較差 (誤差 > 10%)")
    
    # y 方向剖面
    P_k_profile = np.mean(P_k_field, axis=0)  # 沿 x 平均
    eps_profile = np.mean(eps_mean, axis=0)
    y = np.linspace(0, L, len(P_k_profile), endpoint=False)
    
    # 繪圖
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 空間分佈對比
    ax = axes[0, 0]
    X = data['mean_field']['X']
    Y = data['mean_field']['Y']
    im = ax.contourf(X, Y, P_k_field, levels=20, cmap='viridis')
    ax.set_title(r'Production $P_k = \nu_t |S|^2$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    ax = axes[0, 1]
    im = ax.contourf(X, Y, eps_mean, levels=20, cmap='viridis')
    ax.set_title(r'Dissipation $\varepsilon$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    # (2) y 方向剖面
    ax = axes[1, 0]
    ax.plot(y, P_k_profile, 'o-', label=r'$\langle P_k \rangle_x$', linewidth=2)
    ax.plot(y, eps_profile, 's-', label=r'$\langle \varepsilon \rangle_x$', linewidth=2, alpha=0.7)
    ax.axhline(P_k_avg, color='blue', linestyle='--', alpha=0.5, label=f'Mean $P_k$ = {P_k_avg:.4f}')
    ax.axhline(eps_avg, color='orange', linestyle='--', alpha=0.5, label=f'Mean $\\varepsilon$ = {eps_avg:.4f}')
    ax.set_xlabel('y')
    ax.set_ylabel('Value')
    ax.set_title('y-direction Profile (x-averaged)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (3) 散點圖：逐點比較
    ax = axes[1, 1]
    ax.scatter(P_k_field.flatten(), eps_mean.flatten(), alpha=0.3, s=1)
    
    # 理想線
    max_val = max(P_k_field.max(), eps_mean.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect Balance', linewidth=2)
    
    ax.set_xlabel(r'$P_k$')
    ax.set_ylabel(r'$\varepsilon$')
    ax.set_title('Point-wise Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.suptitle(f'RANS Energy Balance Validation\n'
                 f'⟨P_k⟩/⟨ε⟩ = {P_k_avg/eps_avg:.3f}, '
                 f'Error = {balance_error:.2%}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'energy_balance_spatial.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return {
        'P_k_avg': P_k_avg,
        'eps_avg': eps_avg,
        'balance_error': balance_error,
    }


def plot_time_evolution(data, output_dir):
    """繪製時間演化"""
    logging.info("繪製時間演化...")
    
    time = data['time_series']['time']
    k_t = data['time_series']['k']
    eps_t = data['time_series']['epsilon']
    nu_t_t = data['time_series']['nu_t']
    nu = data['params']['nu']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # (1) 湍動能
    ax = axes[0, 0]
    ax.plot(time, k_t, linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle k \rangle$')
    ax.set_title('Turbulent Kinetic Energy')
    ax.grid(True, alpha=0.3)
    
    # (2) 耗散率
    ax = axes[0, 1]
    ax.plot(time, eps_t, linewidth=2, color='orange')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle \varepsilon \rangle$')
    ax.set_title('Dissipation Rate')
    ax.grid(True, alpha=0.3)
    
    # (3) 渦黏滯比
    ax = axes[1, 0]
    ax.plot(time, nu_t_t / nu, linewidth=2, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$\langle \nu_t \rangle / \nu$')
    ax.set_title('Eddy Viscosity Ratio')
    ax.grid(True, alpha=0.3)
    
    # (4) k/ε 比值（湍流時間尺度）
    ax = axes[1, 1]
    turb_timescale = k_t / eps_t
    ax.plot(time, turb_timescale, linewidth=2, color='purple')
    ax.set_xlabel('Time')
    ax.set_ylabel(r'$k / \varepsilon$ (Turbulent Timescale)')
    ax.set_title('Turbulent Timescale')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('RANS Time Evolution', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(output_dir / 'time_evolution.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 時間演化圖已保存")


def main():
    parser = argparse.ArgumentParser(description='RANS-k-ε 能量平衡驗證')
    parser.add_argument('--input', type=str, required=True,
                       help='RANS 數據檔案路徑 (HDF5)')
    parser.add_argument('--output', type=str, default='results/rans_validation/',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_path.exists():
        raise FileNotFoundError(f"找不到文件: {input_path}")
    
    logging.info("=" * 70)
    logging.info("RANS-k-ε 能量平衡驗證")
    logging.info("=" * 70)
    
    # 載入數據
    data = load_rans_data(input_path)
    
    # 能量平衡驗證
    balance_result = validate_energy_balance(data, output_dir)
    
    # 時間演化
    plot_time_evolution(data, output_dir)
    
    logging.info("\n" + "=" * 70)
    logging.info("✅ 驗證完成！")
    logging.info(f"   輸出目錄: {output_dir}")
    logging.info(f"   能量平衡誤差: {balance_result['balance_error']:.2%}")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
