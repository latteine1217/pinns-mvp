#!/usr/bin/env python3
"""
RANS vs DNS 速度剖面對比圖生成工具（Channel Flow）
========================================================

功能：
1. 繪製壁面法向速度剖面 (u vs y)
2. 對數律驗證 (U+ vs y+)
3. 湍流動能剖面對比
4. 雷諾應力剖面（如有）
5. 誤差分析

使用範例：
----------
# 基本使用（論文用圖）
python scripts/visualize/plot_rans_dns_velocity_profiles.py \
    --rans data/lowfi/channel_rans/rans_k_omega_sst.npz \
    --dns "data/jhtdb/channel_flow_re1000/slice_z4.71_*.npz" \
    --output thesis/result_figures/rans_dns_velocity_profiles.png \
    --style paper

# 詳細分析用圖
python scripts/visualize/plot_rans_dns_velocity_profiles.py \
    --rans data/lowfi/channel_rans/rans_k_omega_sst.npz \
    --dns "data/jhtdb/channel_flow_re1000/slice_z4.71_*.npz" \
    --output results/rans_vs_dns_detailed.png \
    --style detailed

作者：PINNs-MVP 團隊
日期：2025-12-16
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional
import logging

# 設定 LaTeX 風格（論文用）
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['text.usetex'] = False  # 若有 LaTeX 環境可設為 True
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def load_rans_data(file_path: Path) -> Dict:
    """載入 FLUENT RANS 數據"""
    logging.info(f"載入 RANS 數據: {file_path}")
    
    data = np.load(file_path)
    
    # 提取數據
    x = data['x']  # (251,)
    y = data['y']  # (20,) - 原始座標 [0, 2]
    z = data['z']  # (94,)
    
    u = data['u']  # (251, 20, 94)
    v = data['v']
    w = data['w']
    p = data['p']
    k = data['k']  # Turbulent kinetic energy
    mu_t = data['mu_t']  # Eddy viscosity
    
    # 元數據
    model_type = str(data['model_type'])
    Re_tau_est = float(data['Re_tau_estimate'])
    nu = float(data['nu'])
    
    # 轉換座標系統：RANS [0, 2] → DNS [-1, 1]
    y_dns = y - 1.0
    
    # 計算 x-z 平均剖面（統計平均）
    u_mean = np.mean(u, axis=(0, 2))  # 平均 x, z 方向
    v_mean = np.mean(v, axis=(0, 2))
    w_mean = np.mean(w, axis=(0, 2))
    k_mean = np.mean(k, axis=(0, 2))
    
    logging.info(f"  模型: {model_type}")
    logging.info(f"  Re_τ (估計): {Re_tau_est:.1f}")
    logging.info(f"  黏度: ν = {nu:.2e}")
    logging.info(f"  網格: {x.shape[0]} × {y.shape[0]} × {z.shape[0]}")
    logging.info(f"  y 範圍: [{y.min():.3f}, {y.max():.3f}] (原始)")
    logging.info(f"  y 範圍: [{y_dns.min():.3f}, {y_dns.max():.3f}] (DNS 座標)")
    
    return {
        'y': y_dns,  # DNS 座標系統
        'y_original': y,  # 原始 RANS 座標
        'u_mean': u_mean,
        'v_mean': v_mean,
        'w_mean': w_mean,
        'k_mean': k_mean,
        'Re_tau': Re_tau_est,
        'nu': nu,
        'model': model_type,
        'grid_shape': (x.shape[0], y.shape[0], z.shape[0]),
    }


def load_dns_data(file_pattern: str) -> Dict:
    """載入 JHTDB DNS 數據（支援 cutout 或 slice）"""
    logging.info(f"載入 DNS 數據: {file_pattern}")
    
    # 搜尋匹配的檔案
    from glob import glob
    files = sorted(glob(file_pattern))
    
    if len(files) == 0:
        raise FileNotFoundError(f"找不到匹配的 DNS 檔案: {file_pattern}")
    
    logging.info(f"  找到 {len(files)} 個檔案")
    
    # 載入第一個檔案
    data = np.load(files[0])
    
    # 檢查可用的 keys
    available_keys = list(data.keys())
    logging.info(f"  可用欄位: {available_keys}")
    
    # 提取速度場
    if 'u' not in data:
        raise KeyError("DNS 數據缺少速度場 'u'")
    
    # 判斷資料格式
    if 'coords' in data:
        # Cutout 格式：1D flattened arrays
        logging.info("  檢測到 cutout 格式 (flattened)")
        
        u_flat = data['u']
        v_flat = data['v']
        w_flat = data['w']
        
        # 重建網格
        if 'grid_shape' in data:
            grid_shape = tuple(data['grid_shape'])
            logging.info(f"  網格形狀: {grid_shape}")
        else:
            # 從座標推斷
            nx = len(data['x'])
            ny = len(data['y'])
            nz = len(data['z'])
            grid_shape = (nx, ny, nz)
            logging.info(f"  推斷網格形狀: {grid_shape}")
        
        # Reshape 成 3D
        u = u_flat.reshape(grid_shape)
        v = v_flat.reshape(grid_shape)
        w = w_flat.reshape(grid_shape)
        
        y = data['y']
        
        # 計算 x-z 平均剖面（統計平均）
        u_mean = np.mean(u, axis=(0, 2))  # 平均 x, z
        v_mean = np.mean(v, axis=(0, 2))
        w_mean = np.mean(w, axis=(0, 2))
        
    else:
        # Slice 或結構化格式
        logging.info("  檢測到結構化格式")
        
        u = data['u']
        v = data['v']
        w = data['w']
        
        # 如果是 4D，取最後一個時間步
        if u.ndim == 4:
            u = u[:, :, :, -1]
            v = v[:, :, :, -1]
            w = w[:, :, :, -1]
        
        # 如果是 3D，平均 x-z 方向
        if u.ndim == 3:
            u_mean = np.mean(u, axis=(0, 2))
            v_mean = np.mean(v, axis=(0, 2))
            w_mean = np.mean(w, axis=(0, 2))
        elif u.ndim == 2:
            # 2D slice，平均 x 方向
            u_mean = np.mean(u, axis=0)
            v_mean = np.mean(v, axis=0)
            w_mean = np.mean(w, axis=0)
        else:
            raise ValueError(f"不支援的速度場維度: {u.ndim}D")
        
        # 座標
        if 'y' in data:
            y = data['y']
        elif 'y_coords' in data:
            y = data['y_coords']
        else:
            # 假設 JHTDB 標準座標 [-1, 1]
            Ny = len(u_mean)
            y = np.linspace(-1.0, 1.0, Ny)
            logging.warning("  未找到 y 座標，使用預設 [-1, 1]")
    
    # JHTDB Channel Flow 參數（Re_τ = 1000）
    nu = 5e-5  # 運動黏度
    u_tau = 0.0499  # 摩擦速度（從 JHTDB 文檔）
    Re_tau_est = u_tau * 1.0 / nu
    
    logging.info(f"  DNS 剖面長度: {len(u_mean)}")
    logging.info(f"  y 範圍: [{y.min():.3f}, {y.max():.3f}]")
    logging.info(f"  u_mean 範圍: [{u_mean.min():.4f}, {u_mean.max():.4f}]")
    logging.info(f"  Re_τ (標稱): {Re_tau_est:.1f}")
    
    return {
        'y': y,
        'u_mean': u_mean,
        'v_mean': v_mean,
        'w_mean': w_mean,
        'Re_tau': Re_tau_est,
        'nu': nu,
    }


def compute_wall_units(y: np.ndarray, u: np.ndarray, nu: float, u_tau: float) -> Tuple[np.ndarray, np.ndarray]:
    """計算壁面單位 (y+, U+)
    
    對於 channel flow (y ∈ [-1, 1], h=1):
    - 下壁面在 y = -1
    - 上壁面在 y = +1
    - 中心線在 y = 0
    - 距離最近壁面的距離: y_wall = h - |y| = 1 - |y|
    """
    # 距離最近壁面的距離
    y_wall = 1.0 - np.abs(y)  # 範圍 [0, 1]，中心線距離壁面為 1
    
    # 壁面單位
    y_plus = y_wall * u_tau / nu
    u_plus = u / u_tau
    
    return y_plus, u_plus


def plot_velocity_profiles_paper_style(rans: Dict, dns: Dict, output_path: Path, dpi: int = 300):
    """
    繪製論文用速度剖面對比圖（2x2 布局）
    
    圖表：
    1. 外層單位 (u vs y)
    2. 壁面單位 (U+ vs y+)
    3. 湍流動能剖面
    4. 誤差分析
    """
    logging.info("繪製論文風格圖表...")
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== 1. 外層單位速度剖面 (u vs y) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    
    # DNS (參考)
    ax1.plot(dns['u_mean'], dns['y'], 'k-', linewidth=2, label='DNS (JHTDB)', zorder=3)
    
    # RANS (插值到 DNS y 網格以便對比)
    rans_u_interp = np.interp(dns['y'], rans['y'], rans['u_mean'])
    ax1.plot(rans_u_interp, dns['y'], 'r--', linewidth=2, label=r'RANS ($k$-$\omega$ SST)', zorder=2)
    
    # 體積平均速度線
    U_b_dns = np.mean(dns['u_mean'])
    U_b_rans = np.mean(rans['u_mean'])
    ax1.axvline(U_b_dns, color='gray', linestyle=':', linewidth=1, alpha=0.7, label=f'$U_b$ (DNS) = {U_b_dns:.3f}')
    
    ax1.set_xlabel('Streamwise Velocity $u$')
    ax1.set_ylabel('Wall-Normal Coordinate $y/h$')
    ax1.set_title('(a) Mean Velocity Profile')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(dns['u_mean'].max(), rans['u_mean'].max()) * 1.1)
    ax1.set_ylim(-1, 1)
    
    # ========== 2. 壁面單位 (U+ vs y+) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 計算壁面單位
    u_tau_dns = dns['Re_tau'] * dns['nu'] / 1.0  # h = 1
    u_tau_rans = rans['Re_tau'] * rans['nu'] / 1.0
    
    y_plus_dns, u_plus_dns = compute_wall_units(dns['y'], dns['u_mean'], dns['nu'], u_tau_dns)
    y_plus_rans, u_plus_rans = compute_wall_units(rans['y'], rans['u_mean'], rans['nu'], u_tau_rans)
    
    # 過濾有效範圍（避免 y_plus = 0 導致 log 錯誤）
    mask_dns = y_plus_dns > 0.1  # 最小 y+ = 0.1
    mask_rans = y_plus_rans > 0.1
    
    ax2.semilogx(y_plus_dns[mask_dns], u_plus_dns[mask_dns], 'k-', linewidth=2, label='DNS', zorder=3)
    ax2.semilogx(y_plus_rans[mask_rans], u_plus_rans[mask_rans], 'r--', linewidth=2, label='RANS', zorder=2)
    
    # 對數律參考線 U+ = (1/κ) ln(y+) + B
    kappa = 0.41
    B = 5.2
    y_plus_ref = np.logspace(0, 3, 100)
    u_plus_ref = (1/kappa) * np.log(y_plus_ref) + B
    ax2.semilogx(y_plus_ref, u_plus_ref, 'b:', linewidth=1.5, label=f'Log law: $U^+ = (1/{kappa:.2f})\\ln(y^+) + {B:.1f}$', alpha=0.7)
    
    # 線性底層 U+ = y+
    y_plus_linear = np.linspace(0.1, 10, 50)
    ax2.semilogx(y_plus_linear, y_plus_linear, 'g:', linewidth=1.5, label='Viscous sublayer: $U^+ = y^+$', alpha=0.7)
    
    ax2.set_xlabel('$y^+$')
    ax2.set_ylabel('$U^+$')
    ax2.set_title('(b) Law-of-the-Wall Coordinates')
    ax2.legend(loc='upper left', framealpha=0.9, fontsize=9)
    ax2.grid(True, alpha=0.3, linestyle='--', which='both')
    ax2.set_xlim(1, 1000)
    ax2.set_ylim(0, 25)
    
    # ========== 3. 湍流動能剖面 ==========
    ax3 = fig.add_subplot(gs[1, 0])
    
    # RANS TKE
    k_rans_interp = np.interp(dns['y'], rans['y'], rans['k_mean'])
    ax3.plot(k_rans_interp, dns['y'], 'r-', linewidth=2, label='RANS TKE ($k$)')
    
    # DNS 沒有直接的 TKE，但可以用 RMS 估計（如果有多個快照）
    # 這裡暫時不繪製 DNS TKE
    ax3.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    ax3.set_xlabel('Turbulent Kinetic Energy $k$')
    ax3.set_ylabel('$y/h$')
    ax3.set_title('(c) Turbulent Kinetic Energy Profile')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_ylim(-1, 1)
    
    # ========== 4. 相對誤差分析 ==========
    ax4 = fig.add_subplot(gs[1, 1])
    
    # 計算誤差
    u_error = rans_u_interp - dns['u_mean']
    u_rel_error = np.abs(u_error) / (np.abs(dns['u_mean']) + 1e-8) * 100  # 百分比
    
    ax4.plot(u_rel_error, dns['y'], 'purple', linewidth=2, label='Relative Error')
    ax4.axvline(0, color='gray', linestyle='-', linewidth=0.5)
    ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    
    # 誤差統計
    l2_error = np.linalg.norm(u_error) / (np.linalg.norm(dns['u_mean']) + 1e-12)
    rmse = np.sqrt(np.mean(u_error**2))
    max_error = np.max(np.abs(u_error))
    
    info_text = f"$L_2$ error: {l2_error*100:.1f}%\\nRMSE: {rmse:.4f}\\nMax |$\\Delta u$|: {max_error:.4f}"
    ax4.text(0.95, 0.05, info_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    ax4.set_xlabel('Relative Error (%)')
    ax4.set_ylabel('$y/h$')
    ax4.set_title('(d) RANS Error vs DNS')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_ylim(-1, 1)
    
    # ========== 總標題與說明 ==========
    fig.suptitle(r'RANS vs DNS Velocity Profile Comparison (Channel Flow, $Re_\tau \approx 1000$)',
                fontsize=14, fontweight='bold', y=0.98)
    
    # 參數資訊（右下角）
    param_text = (f"DNS: $Re_\\tau$ = {dns['Re_tau']:.0f}, $\\nu$ = {dns['nu']:.2e}\\n"
                  f"RANS: $Re_\\tau$ = {rans['Re_tau']:.0f}, $\\nu$ = {rans['nu']:.2e}\\n"
                  f"Grid: RANS {rans['grid_shape'][0]}×{rans['grid_shape'][1]}×{rans['grid_shape'][2]}, "
                  f"DNS $\\sim$2048×512×1536")
    
    fig.text(0.99, 0.01, param_text, fontsize=8, verticalalignment='bottom', 
            horizontalalignment='right', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    
    # 儲存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    logging.info(f"  ✅ 論文圖表已儲存: {output_path}")
    logging.info(f"  L2 誤差: {l2_error*100:.2f}%")
    logging.info(f"  RMSE: {rmse:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='RANS vs DNS 速度剖面對比圖生成工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--rans', type=str, required=True,
                       help='RANS 數據路徑 (NPZ 格式)')
    parser.add_argument('--dns', type=str, required=True,
                       help='DNS 數據路徑（可用萬用字元）')
    parser.add_argument('--output', type=str, required=True,
                       help='輸出圖片路徑 (PNG)')
    parser.add_argument('--style', type=str, default='paper', choices=['paper'],
                       help='圖表風格：paper (論文用 2x2)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='圖片解析度（論文建議 300）')
    
    args = parser.parse_args()
    
    logging.info("=" * 70)
    logging.info("RANS vs DNS 速度剖面對比圖生成工具")
    logging.info("=" * 70)
    
    # 載入數據
    rans_data = load_rans_data(Path(args.rans))
    dns_data = load_dns_data(args.dns)
    
    # 繪圖
    output_path = Path(args.output)
    plot_velocity_profiles_paper_style(rans_data, dns_data, output_path, args.dpi)
    
    logging.info("\\n" + "=" * 70)
    logging.info("✅ 完成！")
    logging.info(f"   輸出: {output_path}")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
