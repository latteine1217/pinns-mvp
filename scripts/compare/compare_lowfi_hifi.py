#!/usr/bin/env python3
"""
Low-Fidelity vs High-Fidelity Kolmogorov Flow 比較工具
======================================================

功能：
1. 速度場直接比較（空間分佈）
2. 能譜比較（湍流特性）
3. 統計量比較（KE, enstrophy, dissipation）
4. 誤差分析（L2 norm, 相對誤差）
5. 雷諾數與解析度資訊

使用範例：
----------
# 基本比較
python scripts/compare_lowfi_hifi.py \
    --hifi data/kolmogorov_dns/re100_N256.h5 \
    --lowfi data/kolmogorov_lowfi/lowfi_N32_nu0.02.h5 \
    --output results/lowfi_vs_hifi/

# 指定時間範圍（比較時間平均場）
python scripts/compare_lowfi_hifi.py \
    --hifi data/kolmogorov_dns/re100_N256.h5 \
    --lowfi data/kolmogorov_lowfi/lowfi_N64_nu0.03.h5 \
    --output results/lowfi_vs_hifi/ \
    --time_avg_range 50.0 100.0

作者：PINNs-MVP 團隊
日期：2025-12-10
"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_hifi_data(file_path: Path, time_avg_range: Optional[Tuple[float, float]] = None) -> Dict:
    """
    載入 Hi-Fi DNS 資料
    
    Args:
        file_path: HDF5 檔案路徑
        time_avg_range: 時間平均範圍 (t_start, t_end)，None 表示使用最後一幀
    
    Returns:
        data: 包含 u, v, X, Y, parameters
    """
    logging.info(f"載入 Hi-Fi 資料: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        u_all = np.array(f['u'])
        v_all = np.array(f['v'])
        time = np.array(f['time'])
        
        # 網格資訊
        N = u_all.shape[1]
        L = f['config'].attrs.get('L', 2 * np.pi)
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y)
        
        # 參數資訊
        params = {
            'N': N,
            'L': L,
            'nu': f['config'].attrs.get('nu', 0.01),
            'A': f['config'].attrs.get('A', 1.0),
            'k_f': f['config'].attrs.get('k_f', 4),
        }
        
        # 時間平均或單幀
        if time_avg_range is not None:
            t_start, t_end = time_avg_range
            mask = (time >= t_start) & (time <= t_end)
            if np.sum(mask) == 0:
                logging.warning(f"時間範圍 [{t_start}, {t_end}] 無資料，使用全部時間平均")
                mask = np.ones_like(time, dtype=bool)
            
            u = np.mean(u_all[mask], axis=0)
            v = np.mean(v_all[mask], axis=0)
            logging.info(f"  時間平均: {np.sum(mask)} 幀, t ∈ [{time[mask][0]:.2f}, {time[mask][-1]:.2f}]")
        else:
            u = u_all[-1]
            v = v_all[-1]
            logging.info(f"  使用最後一幀: t = {time[-1]:.2f}")
    
    # 計算雷諾數
    Re = np.sqrt(params['A']) * (2*np.pi/params['k_f'])**(3/2) / params['nu']
    params['Re'] = Re
    
    logging.info(f"  Hi-Fi 參數: N={N}, ν={params['nu']:.6f}, Re={Re:.2f}")
    
    return {
        'u': u,
        'v': v,
        'X': X,
        'Y': Y,
        'params': params,
    }


def load_lowfi_data(file_path: Path) -> Dict:
    """
    載入 Low-Fi 資料
    
    Returns:
        data: 包含 u_mean, v_mean, X, Y, parameters
    """
    logging.info(f"載入 Low-Fi 資料: {file_path}")
    
    with h5py.File(file_path, 'r') as f:
        u = np.array(f['mean_field/u'])
        v = np.array(f['mean_field/v'])
        X = np.array(f['mean_field/X'])
        Y = np.array(f['mean_field/Y'])
        
        # 參數資訊
        params = {}
        for key in f['parameters'].attrs.keys():
            params[key] = f['parameters'].attrs[key]
    
    # 計算雷諾數
    Re = np.sqrt(params['A']) * (2*np.pi/params['k_f'])**(3/2) / params['nu']
    params['Re'] = Re
    
    logging.info(f"  Low-Fi 參數: N={params['N']}, ν={params['nu']:.6f}, Re={Re:.2f}")
    
    return {
        'u': u,
        'v': v,
        'X': X,
        'Y': Y,
        'params': params,
    }


def interpolate_to_hifi_grid(u_lowfi: np.ndarray, v_lowfi: np.ndarray, 
                             X_lowfi: np.ndarray, Y_lowfi: np.ndarray,
                             X_hifi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    將 Low-Fi 場插值到 Hi-Fi 網格
    
    使用雙線性插值（簡單且快速）
    """
    from scipy.interpolate import RegularGridInterpolator
    
    # 從 meshgrid 提取 1D 座標
    # X[i,:] 應該是常數行（對於 indexing='ij'）
    # Y[:,j] 應該是常數列
    N_lowfi = X_lowfi.shape[0]
    N_hifi = X_hifi.shape[0]
    
    # 推算域大小（從第一行最後元素）
    L_x = X_lowfi[-1, 0]  + (X_lowfi[-1, 0] - X_lowfi[-2, 0])
    L_y = Y_lowfi[0, -1] + (Y_lowfi[0, -1] - Y_lowfi[0, -2])
    L = max(L_x, L_y)  # 應該相同
    
    # 重新構建單調座標
    x_lowfi = np.linspace(0, L, N_lowfi, endpoint=False)
    y_lowfi = np.linspace(0, L, N_lowfi, endpoint=False)
    x_hifi = np.linspace(0, L, N_hifi, endpoint=False)
    y_hifi = np.linspace(0, L, N_hifi, endpoint=False)
    
    # 插值器（x 為第一維，y 為第二維，對應 indexing='ij'）
    interp_u = RegularGridInterpolator((x_lowfi, y_lowfi), u_lowfi, 
                                       method='linear', bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((x_lowfi, y_lowfi), v_lowfi,
                                       method='linear', bounds_error=False, fill_value=0.0)
    
    # 插值到 Hi-Fi 網格（使用 meshgrid 構建點陣列）
    X_hifi_grid, Y_hifi_grid = np.meshgrid(x_hifi, y_hifi, indexing='ij')
    points_hifi = np.column_stack([X_hifi_grid.flatten(), Y_hifi_grid.flatten()])
    
    u_interp = interp_u(points_hifi).reshape(X_hifi.shape)
    v_interp = interp_v(points_hifi).reshape(X_hifi.shape)
    
    return u_interp, v_interp


def compute_energy_spectrum(u: np.ndarray, v: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算能譜 E(k)
    
    Args:
        u, v: 速度場
        L: 域大小
    
    Returns:
        k_bins: 波數（徑向）
        E_k: 能譜
    """
    N = u.shape[0]
    
    # FFT
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    # 能量密度（頻譜空間）
    energy_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2) / N**4
    
    # 波數網格
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky = np.meshgrid(k, k)
    k_mag = np.sqrt(kx**2 + ky**2)
    
    # 徑向平均
    k_max = int(np.floor(N / 2))
    k_bins = np.arange(1, k_max)
    E_k = np.zeros_like(k_bins, dtype=float)
    
    for i, k_val in enumerate(k_bins):
        mask = (k_mag >= k_val - 0.5) & (k_mag < k_val + 0.5)
        E_k[i] = np.sum(energy_density[mask])
    
    return k_bins, E_k


def compute_statistics(u: np.ndarray, v: np.ndarray, nu: float, L: float) -> Dict[str, float]:
    """計算流場統計量"""
    N = u.shape[0]
    
    # 動能
    KE = float(0.5 * np.mean(u**2 + v**2))
    
    # 渦度（使用頻譜導數）
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    k = 2 * np.pi * np.fft.fftfreq(N, d=L/N)
    kx, ky = np.meshgrid(k, k)
    
    omega_hat = 1j * (kx * v_hat - ky * u_hat)
    omega = np.real(np.fft.ifft2(omega_hat))
    
    enstrophy = float(0.5 * np.mean(omega**2))
    dissipation = float(nu * np.mean(omega**2))
    
    # RMS 速度
    u_rms = float(np.sqrt(np.mean(u**2)))
    v_rms = float(np.sqrt(np.mean(v**2)))
    
    return {
        'KE': KE,
        'enstrophy': enstrophy,
        'dissipation': dissipation,
        'u_rms': u_rms,
        'v_rms': v_rms,
    }


def compute_errors(u_ref: np.ndarray, v_ref: np.ndarray,
                  u_test: np.ndarray, v_test: np.ndarray) -> Dict[str, float]:
    """計算誤差指標"""
    # L2 相對誤差
    l2_u = float(np.linalg.norm(u_ref - u_test) / (np.linalg.norm(u_ref) + 1e-12))
    l2_v = float(np.linalg.norm(v_ref - v_test) / (np.linalg.norm(v_ref) + 1e-12))
    l2_total = float(np.sqrt(np.linalg.norm(u_ref - u_test)**2 + np.linalg.norm(v_ref - v_test)**2) / \
               (np.sqrt(np.linalg.norm(u_ref)**2 + np.linalg.norm(v_ref)**2) + 1e-12))
    
    # RMSE
    rmse_u = float(np.sqrt(np.mean((u_ref - u_test)**2)))
    rmse_v = float(np.sqrt(np.mean((v_ref - v_test)**2)))
    
    # 最大絕對誤差
    max_err_u = float(np.max(np.abs(u_ref - u_test)))
    max_err_v = float(np.max(np.abs(v_ref - v_test)))
    
    return {
        'l2_u': l2_u,
        'l2_v': l2_v,
        'l2_total': l2_total,
        'rmse_u': rmse_u,
        'rmse_v': rmse_v,
        'max_err_u': max_err_u,
        'max_err_v': max_err_v,
    }


def plot_field_comparison(hifi: Dict, lowfi: Dict, lowfi_interp: Dict,
                         output_dir: Path, dpi: int = 150):
    """繪製速度場比較圖"""
    logging.info("繪製速度場比較...")
    
    u_hifi, v_hifi = hifi['u'], hifi['v']
    u_lowfi_interp, v_lowfi_interp = lowfi_interp['u'], lowfi_interp['v']
    X_hifi, Y_hifi = hifi['X'], hifi['Y']
    
    # 計算誤差
    u_error = u_hifi - u_lowfi_interp
    v_error = v_hifi - v_lowfi_interp
    
    # === 圖 1: V 速度場比較（主要特徵）===
    fig1, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig1.suptitle('V Velocity Field Comparison', fontsize=14, fontweight='bold')
    
    # Hi-Fi
    im1 = axes[0, 0].contourf(X_hifi, Y_hifi, v_hifi, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title(f"Hi-Fi (N={hifi['params']['N']}, Re={hifi['params']['Re']:.1f})")
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='v')
    
    # Low-Fi (插值後)
    im2 = axes[0, 1].contourf(X_hifi, Y_hifi, v_lowfi_interp, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title(f"Low-Fi Interp (N={lowfi['params']['N']}, Re={lowfi['params']['Re']:.1f})")
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='v')
    
    # 誤差場
    v_err_max = np.abs(v_error).max()
    im3 = axes[1, 0].contourf(X_hifi, Y_hifi, v_error, levels=50, 
                              cmap='RdBu_r', vmin=-v_err_max, vmax=v_err_max)
    axes[1, 0].set_title(f"Error (Hi-Fi - Low-Fi)")
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0], label='Δv')
    
    # 相對誤差
    v_rel_error = np.abs(v_error) / (np.abs(v_hifi) + 1e-8)
    im4 = axes[1, 1].contourf(X_hifi, Y_hifi, v_rel_error, levels=50, cmap='viridis')
    axes[1, 1].set_title(f"Relative Error |Δv| / |v_hifi|")
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1], label='Rel. Error')
    
    plt.tight_layout()
    fig1.savefig(output_dir / 'field_comparison_v.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig1)
    
    # === 圖 2: U 速度場比較 ===
    fig2, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig2.suptitle('U Velocity Field Comparison', fontsize=14, fontweight='bold')
    
    im1 = axes[0, 0].contourf(X_hifi, Y_hifi, u_hifi, levels=50, cmap='RdBu_r')
    axes[0, 0].set_title(f"Hi-Fi (N={hifi['params']['N']})")
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('y')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0], label='u')
    
    im2 = axes[0, 1].contourf(X_hifi, Y_hifi, u_lowfi_interp, levels=50, cmap='RdBu_r')
    axes[0, 1].set_title(f"Low-Fi Interp (N={lowfi['params']['N']})")
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('y')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1], label='u')
    
    u_err_max = np.abs(u_error).max()
    im3 = axes[1, 0].contourf(X_hifi, Y_hifi, u_error, levels=50,
                              cmap='RdBu_r', vmin=-u_err_max, vmax=u_err_max)
    axes[1, 0].set_title(f"Error (Hi-Fi - Low-Fi)")
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('y')
    axes[1, 0].set_aspect('equal')
    plt.colorbar(im3, ax=axes[1, 0], label='Δu')
    
    u_rel_error = np.abs(u_error) / (np.abs(u_hifi) + 1e-8)
    im4 = axes[1, 1].contourf(X_hifi, Y_hifi, u_rel_error, levels=50, cmap='viridis')
    axes[1, 1].set_title(f"Relative Error |Δu| / |u_hifi|")
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('y')
    axes[1, 1].set_aspect('equal')
    plt.colorbar(im4, ax=axes[1, 1], label='Rel. Error')
    
    plt.tight_layout()
    fig2.savefig(output_dir / 'field_comparison_u.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig2)
    
    logging.info(f"  ✅ 速度場比較圖已儲存")


def plot_spectrum_comparison(hifi: Dict, lowfi: Dict, lowfi_interp: Dict,
                            output_dir: Path, dpi: int = 150):
    """繪製能譜比較"""
    logging.info("繪製能譜比較...")
    
    # 計算能譜
    k_hifi, E_hifi = compute_energy_spectrum(hifi['u'], hifi['v'], hifi['params']['L'])
    k_lowfi_interp, E_lowfi_interp = compute_energy_spectrum(
        lowfi_interp['u'], lowfi_interp['v'], hifi['params']['L']
    )
    
    # 強迫波數
    k_f = hifi['params']['k_f']
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Hi-Fi 能譜
    ax.loglog(k_hifi, E_hifi, 'o-', label=f"Hi-Fi (N={hifi['params']['N']}, Re={hifi['params']['Re']:.1f})",
              linewidth=2, markersize=4)
    
    # Low-Fi 能譜（插值後）
    ax.loglog(k_lowfi_interp, E_lowfi_interp, 's-', 
              label=f"Low-Fi Interp (N={lowfi['params']['N']}, Re={lowfi['params']['Re']:.1f})",
              linewidth=2, markersize=4, alpha=0.7)
    
    # 強迫波數標記
    ax.axvline(k_f, color='red', linestyle='--', linewidth=1.5, label=f'Forcing k_f={k_f}')
    
    # 參考斜率（2D 湍流理論）
    k_ref = np.array([k_f/2, k_f])
    E_ref_inverse = E_hifi[np.argmin(np.abs(k_hifi - k_f))] * (k_ref / k_f)**(-5/3)
    ax.loglog(k_ref, E_ref_inverse, 'k--', linewidth=1, alpha=0.5, label=r'$k^{-5/3}$ (inverse cascade)')
    
    k_ref_forward = np.array([k_f, 2*k_f])
    E_ref_forward = E_hifi[np.argmin(np.abs(k_hifi - k_f))] * (k_ref_forward / k_f)**(-3)
    ax.loglog(k_ref_forward, E_ref_forward, 'k:', linewidth=1, alpha=0.5, label=r'$k^{-3}$ (enstrophy cascade)')
    
    ax.set_xlabel('Wavenumber k', fontsize=12)
    ax.set_ylabel('Energy Spectrum E(k)', fontsize=12)
    ax.set_title('Energy Spectrum Comparison (2D Kolmogorov Flow)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / 'spectrum_comparison.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 能譜比較圖已儲存")


def plot_statistics_comparison(hifi: Dict, lowfi: Dict, lowfi_interp: Dict,
                               errors: Dict, output_dir: Path, dpi: int = 150):
    """繪製統計量比較"""
    logging.info("繪製統計量比較...")
    
    # 計算統計量
    stats_hifi = compute_statistics(hifi['u'], hifi['v'], hifi['params']['nu'], hifi['params']['L'])
    stats_lowfi_interp = compute_statistics(lowfi_interp['u'], lowfi_interp['v'], 
                                            lowfi['params']['nu'], hifi['params']['L'])
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # === 1. 動能比較 ===
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['KE', 'enstrophy', 'dissipation']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    hifi_values = [stats_hifi[m] for m in metrics]
    lowfi_values = [stats_lowfi_interp[m] for m in metrics]
    
    ax1.bar(x_pos - width/2, hifi_values, width, label='Hi-Fi', alpha=0.8)
    ax1.bar(x_pos + width/2, lowfi_values, width, label='Low-Fi', alpha=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['KE', 'Enstrophy', 'Dissipation'])
    ax1.set_ylabel('Value')
    ax1.set_title('Energy Statistics')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === 2. RMS 速度比較 ===
    ax2 = fig.add_subplot(gs[0, 1])
    rms_metrics = ['u_rms', 'v_rms']
    x_pos = np.arange(len(rms_metrics))
    
    hifi_rms = [stats_hifi[m] for m in rms_metrics]
    lowfi_rms = [stats_lowfi_interp[m] for m in rms_metrics]
    
    ax2.bar(x_pos - width/2, hifi_rms, width, label='Hi-Fi', alpha=0.8)
    ax2.bar(x_pos + width/2, lowfi_rms, width, label='Low-Fi', alpha=0.8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(['u_rms', 'v_rms'])
    ax2.set_ylabel('RMS Velocity')
    ax2.set_title('RMS Velocities')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # === 3. 誤差指標 ===
    ax3 = fig.add_subplot(gs[0, 2])
    error_metrics = ['l2_u', 'l2_v', 'l2_total']
    error_labels = ['L2(u)', 'L2(v)', 'L2(total)']
    error_values = [errors[m] for m in error_metrics]
    colors = ['#ff7f0e', '#2ca02c', '#d62728']
    
    x_pos = np.arange(len(error_metrics))
    bars = ax3.bar(x_pos, error_values, color=colors, alpha=0.8)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(error_labels)
    ax3.set_ylabel('Relative L2 Error')
    ax3.set_title('L2 Errors (Low-Fi vs Hi-Fi)')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 在柱狀圖上標註數值
    for bar, val in zip(bars, error_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # === 4. RMSE ===
    ax4 = fig.add_subplot(gs[1, 0])
    rmse_metrics = ['rmse_u', 'rmse_v']
    rmse_labels = ['RMSE(u)', 'RMSE(v)']
    rmse_values = [errors[m] for m in rmse_metrics]
    
    x_pos = np.arange(len(rmse_metrics))
    ax4.bar(x_pos, rmse_values, color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(rmse_labels)
    ax4.set_ylabel('RMSE')
    ax4.set_title('Root Mean Square Errors')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # === 5. 最大誤差 ===
    ax5 = fig.add_subplot(gs[1, 1])
    max_err_metrics = ['max_err_u', 'max_err_v']
    max_err_labels = ['Max|Δu|', 'Max|Δv|']
    max_err_values = [errors[m] for m in max_err_metrics]
    
    x_pos = np.arange(len(max_err_metrics))
    ax5.bar(x_pos, max_err_values, color=['#2ca02c', '#d62728'], alpha=0.8)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(max_err_labels)
    ax5.set_ylabel('Max Absolute Error')
    ax5.set_title('Maximum Errors')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # === 6. 參數資訊 ===
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    
    info_text = f"""
    ====== Simulation Parameters ======
    
    Hi-Fi:
      Grid: {hifi['params']['N']} × {hifi['params']['N']}
      ν = {hifi['params']['nu']:.6f}
      Re = {hifi['params']['Re']:.2f}
    
    Low-Fi:
      Grid: {lowfi['params']['N']} × {lowfi['params']['N']}
      ν = {lowfi['params']['nu']:.6f}
      Re = {lowfi['params']['Re']:.2f}
    
    Forcing:
      A = {hifi['params']['A']:.4f}
      k_f = {hifi['params']['k_f']}
    
    ====== Error Summary ======
      L2 (u): {errors['l2_u']:.4f}
      L2 (v): {errors['l2_v']:.4f}
      L2 (total): {errors['l2_total']:.4f}
    
    Viscosity Ratio: {lowfi['params']['nu'] / hifi['params']['nu']:.2f}×
    Resolution Ratio: {hifi['params']['N'] / lowfi['params']['N']:.1f}×
    """
    
    ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Low-Fi vs Hi-Fi Statistical Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    fig.savefig(output_dir / 'statistics_comparison.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 統計量比較圖已儲存")


def main():
    parser = argparse.ArgumentParser(
        description='Low-Fi vs Hi-Fi Kolmogorov Flow 比較工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 輸入檔案
    parser.add_argument('--hifi', type=str, required=True,
                       help='Hi-Fi DNS 資料檔案路徑 (HDF5)')
    parser.add_argument('--lowfi', type=str, required=True,
                       help='Low-Fi 資料檔案路徑 (HDF5)')
    
    # 時間範圍
    parser.add_argument('--time_avg_range', type=float, nargs=2, metavar=('T_START', 'T_END'),
                       help='Hi-Fi 時間平均範圍（例如：50.0 100.0）。不指定則使用最後一幀')
    
    # 輸出設定
    parser.add_argument('--output', type=str, default='results/lowfi_vs_hifi/',
                       help='輸出目錄（預設：results/lowfi_vs_hifi/）')
    parser.add_argument('--dpi', type=int, default=150,
                       help='圖片解析度（預設：150）')
    
    args = parser.parse_args()
    
    # === 載入資料 ===
    hifi_path = Path(args.hifi)
    lowfi_path = Path(args.lowfi)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not hifi_path.exists():
        raise FileNotFoundError(f"Hi-Fi 檔案不存在: {hifi_path}")
    if not lowfi_path.exists():
        raise FileNotFoundError(f"Low-Fi 檔案不存在: {lowfi_path}")
    
    logging.info("=" * 70)
    logging.info("Low-Fi vs Hi-Fi Kolmogorov Flow 比較工具")
    logging.info("=" * 70)
    
    hifi_data = load_hifi_data(hifi_path, args.time_avg_range)
    lowfi_data = load_lowfi_data(lowfi_path)
    
    # === 插值 Low-Fi 到 Hi-Fi 網格 ===
    logging.info("插值 Low-Fi 場到 Hi-Fi 網格...")
    u_interp, v_interp = interpolate_to_hifi_grid(
        lowfi_data['u'], lowfi_data['v'],
        lowfi_data['X'], lowfi_data['Y'],
        hifi_data['X']
    )
    
    lowfi_interp = {
        'u': u_interp,
        'v': v_interp,
    }
    
    # === 計算誤差 ===
    logging.info("計算誤差指標...")
    errors = compute_errors(hifi_data['u'], hifi_data['v'], u_interp, v_interp)
    
    logging.info("  誤差總結:")
    logging.info(f"    L2 (u): {errors['l2_u']:.6f}")
    logging.info(f"    L2 (v): {errors['l2_v']:.6f}")
    logging.info(f"    L2 (total): {errors['l2_total']:.6f}")
    logging.info(f"    RMSE (u): {errors['rmse_u']:.6f}")
    logging.info(f"    RMSE (v): {errors['rmse_v']:.6f}")
    
    # === 繪圖 ===
    plot_field_comparison(hifi_data, lowfi_data, lowfi_interp, output_dir, args.dpi)
    plot_spectrum_comparison(hifi_data, lowfi_data, lowfi_interp, output_dir, args.dpi)
    plot_statistics_comparison(hifi_data, lowfi_data, lowfi_interp, errors, output_dir, args.dpi)
    
    logging.info("\n" + "=" * 70)
    logging.info("✅ 比較完成！")
    logging.info(f"   輸出目錄: {output_dir}")
    logging.info(f"   生成檔案:")
    logging.info(f"     - field_comparison_u.png")
    logging.info(f"     - field_comparison_v.png")
    logging.info(f"     - spectrum_comparison.png")
    logging.info(f"     - statistics_comparison.png")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
