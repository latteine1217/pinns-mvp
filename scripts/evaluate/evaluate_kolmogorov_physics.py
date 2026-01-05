"""
Kolmogorov Flow 關鍵物理量統計對比評估

實現參考文獻中的四種物理量對比：
1. 相對 L2 誤差（時間演化）
2. 動能（Kinetic Energy）時間演化
3. 擾動度（Enstrophy）時間演化
4. 能量譜（Energy Spectrum）

使用方式：
    python scripts/evaluate/evaluate_kolmogorov_physics.py \
        --checkpoint checkpoints/kolmogorov_model.pth \
        --reference data/kolmogorov/dns_reference.npz \
        --output results/physics_comparison
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
import json
import argparse
from typing import Dict, Tuple, Optional
from datetime import datetime

# 設置樣式（論文級圖表）
sns.set_context("paper", font_scale=1.3)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from pinnx.utils.evaluation_utils import load_model_for_evaluation, predict_with_denormalization
from pinnx.evals.metrics import relative_L2


# ============================================================
# 物理量計算函數
# ============================================================

def compute_kinetic_energy(u: np.ndarray, v: np.ndarray) -> float:
    """
    計算動能（Kinetic Energy）
    
    KE = 0.5 * ∫∫ (u² + v²) dx dy
    
    Args:
        u: x 方向速度場 [N, 1] 或 [Nx, Ny]
        v: y 方向速度場 [N, 1] 或 [Nx, Ny]
        
    Returns:
        標量動能值
    """
    u = u.squeeze()
    v = v.squeeze()
    
    # 計算速度平方和
    kinetic_energy = 0.5 * np.mean(u**2 + v**2)
    
    return kinetic_energy


def compute_enstrophy(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> float:
    """
    計算擾動度（Enstrophy）
    
    Enstrophy = 0.5 * ∫∫ ω² dx dy
    其中 ω = ∂v/∂x - ∂u/∂y（渦度）
    
    Args:
        u: x 方向速度場 [Nx, Ny]
        v: y 方向速度場 [Nx, Ny]
        dx: x 方向網格間距
        dy: y 方向網格間距
        
    Returns:
        標量擾動度值
    """
    u = u.squeeze()
    v = v.squeeze()
    
    if u.ndim == 1:
        logger.warning("⚠️  Enstrophy 計算需要 2D 網格數據，當前為 1D")
        return np.nan
    
    # 計算渦度 ω = ∂v/∂x - ∂u/∂y
    # 使用中心差分
    dvdx = np.gradient(v, dx, axis=1)
    dudy = np.gradient(u, dy, axis=0)
    vorticity = dvdx - dudy
    
    # 計算擾動度
    enstrophy = 0.5 * np.mean(vorticity**2)
    
    return enstrophy


def compute_energy_spectrum(u: np.ndarray, v: np.ndarray, Lx: float, Ly: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算能量譜（Energy Spectrum）
    
    E(k) = 0.5 * |û(k)|² + |v̂(k)|²
    其中 û, v̂ 是速度場的傅立葉變換
    
    Args:
        u: x 方向速度場 [Nx, Ny]
        v: y 方向速度場 [Nx, Ny]
        Lx: x 方向域長度
        Ly: y 方向域長度
        
    Returns:
        k: 波數數組
        E_k: 對應的能量譜
    """
    u = u.squeeze()
    v = v.squeeze()
    
    if u.ndim == 1:
        logger.warning("⚠️  能量譜計算需要 2D 網格數據，當前為 1D")
        return np.array([]), np.array([])
    
    Nx, Ny = u.shape
    
    # 傅立葉變換
    u_hat = np.fft.fft2(u)
    v_hat = np.fft.fft2(v)
    
    # 計算能量密度
    energy_density = 0.5 * (np.abs(u_hat)**2 + np.abs(v_hat)**2)
    
    # 計算波數
    kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
    
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_magnitude = np.sqrt(kx_grid**2 + ky_grid**2)
    
    # 將能量譜按波數大小分箱
    k_bins = np.arange(0.5, np.max(k_magnitude), 1.0)
    E_k = np.zeros(len(k_bins) - 1)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    
    for i in range(len(k_bins) - 1):
        mask = (k_magnitude >= k_bins[i]) & (k_magnitude < k_bins[i+1])
        if np.any(mask):
            E_k[i] = np.sum(energy_density[mask])
    
    # 歸一化
    E_k = E_k / (Nx * Ny)
    
    return k_centers, E_k


# ============================================================
# 時間序列評估（針對 Time Window Training）
# ============================================================

def evaluate_time_series(
    checkpoint_paths: list,
    reference_data: Dict[str, np.ndarray],
    config: Dict,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    評估時間序列的物理量演化
    
    Args:
        checkpoint_paths: 多個時間窗口的 checkpoint 列表
        reference_data: 參考數據（DNS）
        config: 訓練配置
        device: 計算設備
        output_dir: 輸出目錄
        
    Returns:
        包含所有物理量的字典
    """
    logger.info(f"📊 評估 {len(checkpoint_paths)} 個時間窗口")
    
    # 提取參考數據
    coords_ref = np.column_stack([
        reference_data['x'].flatten(),
        reference_data['y'].flatten()
    ])
    u_ref = reference_data['u'].flatten()
    v_ref = reference_data['v'].flatten()
    
    # 域參數（從 reference 或 config 獲取）
    Lx = reference_data['x'].max() - reference_data['x'].min()
    Ly = reference_data['y'].max() - reference_data['y'].min()
    dx = Lx / reference_data['u'].squeeze().shape[0]
    dy = Ly / reference_data['u'].squeeze().shape[1]
    
    # 初始化結果存儲
    results = {
        'time': [],
        'l2_error_u': [],
        'l2_error_v': [],
        'kinetic_energy_ref': [],
        'kinetic_energy_pred': [],
        'enstrophy_ref': [],
        'enstrophy_pred': [],
    }
    
    # 逐個時間窗口評估
    for i, ckpt_path in enumerate(checkpoint_paths):
        logger.info(f"處理 checkpoint {i+1}/{len(checkpoint_paths)}: {ckpt_path}")
        
        try:
            # 載入模型並預測
            model, physics = load_model_for_evaluation(str(ckpt_path), config, device)
            pred_norm = predict_with_denormalization(
                model, coords_ref, str(ckpt_path), config, device, 
                variable_order=['u', 'v', 'p']
            )
            
            u_pred = pred_norm[:, 0]
            v_pred = pred_norm[:, 1]
            
            # 計算 L2 誤差
            l2_u = relative_L2(u_ref, u_pred)
            l2_v = relative_L2(v_ref, v_pred)
            
            # 計算動能
            ke_ref = compute_kinetic_energy(u_ref, v_ref)
            ke_pred = compute_kinetic_energy(u_pred, v_pred)
            
            # 計算擾動度（需要 2D 網格）
            u_ref_2d = u_ref.reshape(reference_data['u'].squeeze().shape)
            v_ref_2d = v_ref.reshape(reference_data['v'].squeeze().shape)
            u_pred_2d = u_pred.reshape(reference_data['u'].squeeze().shape)
            v_pred_2d = v_pred.reshape(reference_data['v'].squeeze().shape)
            
            ens_ref = compute_enstrophy(u_ref_2d, v_ref_2d, dx, dy)
            ens_pred = compute_enstrophy(u_pred_2d, v_pred_2d, dx, dy)
            
            # 記錄結果
            results['time'].append(i)  # 或從 checkpoint 中提取實際時間
            results['l2_error_u'].append(l2_u)
            results['l2_error_v'].append(l2_v)
            results['kinetic_energy_ref'].append(ke_ref)
            results['kinetic_energy_pred'].append(ke_pred)
            results['enstrophy_ref'].append(ens_ref)
            results['enstrophy_pred'].append(ens_pred)
            
            logger.info(f"  L2(u)={l2_u:.4f}, KE={ke_pred:.6f}, Enstrophy={ens_pred:.4f}")
            
        except Exception as e:
            logger.error(f"❌ 處理 {ckpt_path} 失敗: {e}")
            continue
    
    return results


# ============================================================
# 單個快照評估（針對穩態流場）
# ============================================================

def evaluate_single_snapshot(
    checkpoint_path: Path,
    reference_data: Dict[str, np.ndarray],
    config: Dict,
    device: torch.device,
    output_dir: Path
) -> Dict:
    """
    評估單個時刻的物理量（穩態或單個快照）
    
    Args:
        checkpoint_path: 模型 checkpoint
        reference_data: 參考數據（DNS）
        config: 訓練配置
        device: 計算設備
        output_dir: 輸出目錄
        
    Returns:
        包含所有物理量的字典
    """
    logger.info(f"📊 評估單個快照: {checkpoint_path}")
    
    # 提取參考數據
    coords_ref = np.column_stack([
        reference_data['x'].flatten(),
        reference_data['y'].flatten()
    ])
    u_ref = reference_data['u'].flatten()
    v_ref = reference_data['v'].flatten()
    
    # 域參數
    Lx = reference_data['x'].max() - reference_data['x'].min()
    Ly = reference_data['y'].max() - reference_data['y'].min()
    dx = Lx / reference_data['u'].squeeze().shape[0]
    dy = Ly / reference_data['u'].squeeze().shape[1]
    
    # 載入模型並預測
    model, physics = load_model_for_evaluation(str(checkpoint_path), config, device)
    pred_norm = predict_with_denormalization(
        model, coords_ref, str(checkpoint_path), config, device, 
        variable_order=['u', 'v', 'p']
    )
    
    u_pred = pred_norm[:, 0]
    v_pred = pred_norm[:, 1]
    
    # 計算各項物理量
    l2_u = relative_L2(u_ref, u_pred)
    l2_v = relative_L2(v_ref, v_pred)
    
    ke_ref = compute_kinetic_energy(u_ref, v_ref)
    ke_pred = compute_kinetic_energy(u_pred, v_pred)
    
    # 2D 網格形式
    u_ref_2d = u_ref.reshape(reference_data['u'].squeeze().shape)
    v_ref_2d = v_ref.reshape(reference_data['v'].squeeze().shape)
    u_pred_2d = u_pred.reshape(reference_data['u'].squeeze().shape)
    v_pred_2d = v_pred.reshape(reference_data['v'].squeeze().shape)
    
    ens_ref = compute_enstrophy(u_ref_2d, v_ref_2d, dx, dy)
    ens_pred = compute_enstrophy(u_pred_2d, v_pred_2d, dx, dy)
    
    k, E_k_ref = compute_energy_spectrum(u_ref_2d, v_ref_2d, Lx, Ly)
    _, E_k_pred = compute_energy_spectrum(u_pred_2d, v_pred_2d, Lx, Ly)
    
    results = {
        'l2_error_u': l2_u,
        'l2_error_v': l2_v,
        'kinetic_energy_ref': ke_ref,
        'kinetic_energy_pred': ke_pred,
        'enstrophy_ref': ens_ref,
        'enstrophy_pred': ens_pred,
        'energy_spectrum': {
            'k': k.tolist(),
            'E_k_ref': E_k_ref.tolist(),
            'E_k_pred': E_k_pred.tolist()
        }
    }
    
    logger.info(f"✅ 評估完成:")
    logger.info(f"  L2(u): {l2_u:.4f}, L2(v): {l2_v:.4f}")
    logger.info(f"  KE: ref={ke_ref:.6f}, pred={ke_pred:.6f}")
    logger.info(f"  Enstrophy: ref={ens_ref:.4f}, pred={ens_pred:.4f}")
    
    return results


# ============================================================
# 視覺化函數（仿照參考文獻風格）
# ============================================================

def plot_physics_comparison(results: Dict, output_dir: Path, is_time_series: bool = False):
    """
    繪製四合一物理量對比圖（參考文獻風格）
    
    Args:
        results: 包含所有物理量的字典
        output_dir: 輸出目錄
        is_time_series: 是否為時間序列數據
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('關鍵物理量統計對比', fontsize=16, fontweight='bold')
    
    if is_time_series:
        t = np.array(results['time'])
        
        # 1. 相對 L2 誤差
        ax = axes[0, 0]
        ax.plot(t, results['l2_error_u'], 'b-', linewidth=2, label='u')
        ax.plot(t, results['l2_error_v'], 'r-', linewidth=2, label='v')
        ax.set_xlabel('t', fontsize=12)
        ax.set_ylabel('Rel. L² error', fontsize=12)
        ax.set_title('相對 L2 誤差', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # 2. 動能
        ax = axes[0, 1]
        ax.plot(t, results['kinetic_energy_ref'], 'b-', linewidth=2, label='Reference')
        ax.plot(t, results['kinetic_energy_pred'], 'r--', linewidth=2, label='PINN')
        ax.set_xlabel('t', fontsize=12)
        ax.set_ylabel('Kinetic energy', fontsize=12)
        ax.set_title('動能 (Kinetic Energy)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 3. 擾動度
        ax = axes[1, 0]
        ax.plot(t, results['enstrophy_ref'], 'b-', linewidth=2, label='Reference')
        ax.plot(t, results['enstrophy_pred'], 'r--', linewidth=2, label='PINN')
        ax.set_xlabel('t', fontsize=12)
        ax.set_ylabel('Enstrophy', fontsize=12)
        ax.set_title('擾動度 (Enstrophy)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 4. 能量譜（最後一個時刻）
        ax = axes[1, 1]
        if 'energy_spectrum' in results:
            spec = results['energy_spectrum']
            k = np.array(spec['k'])
            E_k_ref = np.array(spec['E_k_ref'])
            E_k_pred = np.array(spec['E_k_pred'])
            
            ax.loglog(k, E_k_ref, 'b-o', linewidth=2, markersize=4, label='Reference')
            ax.loglog(k, E_k_pred, 'r--s', linewidth=2, markersize=4, label='PINN')
            
            # 添加理論斜率參考線
            k_theory = np.logspace(np.log10(k.min()), np.log10(k.max()), 50)
            ax.loglog(k_theory, k_theory**(-5/3) * E_k_ref[0] / k[0]**(-5/3), 
                     'k:', linewidth=1.5, alpha=0.5, label=r'$k^{-5/3}$')
            ax.loglog(k_theory, k_theory**(-3) * E_k_ref[-1] / k[-1]**(-3), 
                     'k:', linewidth=1.5, alpha=0.5, label=r'$k^{-3}$')
            
            ax.set_xlabel('Wavenumber (k)', fontsize=12)
            ax.set_ylabel('Energy spectrum', fontsize=12)
            ax.set_title('能量譜 (Energy Spectrum)', fontsize=14, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3, which='both')
        else:
            ax.text(0.5, 0.5, 'Time series mode:\nNo spectrum available', 
                   ha='center', va='center', fontsize=12)
            ax.axis('off')
    
    else:
        # 單個快照模式
        # 1. L2 誤差柱狀圖
        ax = axes[0, 0]
        variables = ['u', 'v']
        errors = [results['l2_error_u'], results['l2_error_v']]
        ax.bar(variables, errors, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Rel. L² error', fontsize=12)
        ax.set_title('相對 L2 誤差', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. 動能對比
        ax = axes[0, 1]
        categories = ['Reference', 'PINN']
        ke_values = [results['kinetic_energy_ref'], results['kinetic_energy_pred']]
        colors = ['blue', 'red']
        ax.bar(categories, ke_values, color=colors, alpha=0.7)
        ax.set_ylabel('Kinetic energy', fontsize=12)
        ax.set_title('動能 (Kinetic Energy)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 擾動度對比
        ax = axes[1, 0]
        ens_values = [results['enstrophy_ref'], results['enstrophy_pred']]
        ax.bar(categories, ens_values, color=colors, alpha=0.7)
        ax.set_ylabel('Enstrophy', fontsize=12)
        ax.set_title('擾動度 (Enstrophy)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 4. 能量譜
        ax = axes[1, 1]
        spec = results['energy_spectrum']
        k = np.array(spec['k'])
        E_k_ref = np.array(spec['E_k_ref'])
        E_k_pred = np.array(spec['E_k_pred'])
        
        ax.loglog(k, E_k_ref, 'b-o', linewidth=2, markersize=4, label='Reference')
        ax.loglog(k, E_k_pred, 'r--s', linewidth=2, markersize=4, label='PINN')
        
        # 理論斜率
        k_theory = np.logspace(np.log10(k.min()), np.log10(k.max()), 50)
        ax.loglog(k_theory, k_theory**(-5/3) * E_k_ref[0] / k[0]**(-5/3), 
                 'k:', linewidth=1.5, alpha=0.5, label=r'$k^{-5/3}$')
        ax.loglog(k_theory, k_theory**(-3) * E_k_ref[-1] / k[-1]**(-3), 
                 'k:', linewidth=1.5, alpha=0.5, label=r'$k^{-3}$')
        
        ax.set_xlabel('Wavenumber (k)', fontsize=12)
        ax.set_ylabel('Energy spectrum', fontsize=12)
        ax.set_title('能量譜 (Energy Spectrum)', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    # 保存
    output_path = output_dir / 'physics_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 圖表已保存: {output_path}")
    plt.close()


# ============================================================
# 主函數
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow 物理量統計評估')
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='模型 checkpoint 路徑（單個）或目錄（時間序列）')
    parser.add_argument('--reference', type=str, required=True,
                       help='參考數據路徑（.npz 格式）')
    parser.add_argument('--config', type=str, 
                       help='訓練配置文件（可選，會從 checkpoint 自動讀取）')
    parser.add_argument('--output', type=str, default='results/kolmogorov_physics',
                       help='輸出目錄')
    parser.add_argument('--time-series', action='store_true',
                       help='時間序列模式（多個 checkpoint）')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='計算設備')
    
    args = parser.parse_args()
    
    # 設置輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 載入參考數據
    logger.info(f"📥 載入參考數據: {args.reference}")
    reference_data = np.load(args.reference, allow_pickle=True)
    
    # 載入配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # 從 checkpoint 讀取
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        config = ckpt.get('config', {})
        logger.info("📋 從 checkpoint 自動讀取配置")
    
    device = torch.device(args.device)
    
    # 評估
    if args.time_series:
        # 時間序列模式
        checkpoint_dir = Path(args.checkpoint)
        checkpoint_paths = sorted(checkpoint_dir.glob('*.pth'))
        
        if len(checkpoint_paths) == 0:
            logger.error(f"❌ 未找到 checkpoint: {checkpoint_dir}")
            return
        
        logger.info(f"🔍 找到 {len(checkpoint_paths)} 個 checkpoint")
        
        results = evaluate_time_series(
            checkpoint_paths, reference_data, config, device, output_dir
        )
        
        # 繪圖
        plot_physics_comparison(results, output_dir, is_time_series=True)
        
    else:
        # 單個快照模式
        checkpoint_path = Path(args.checkpoint)
        
        results = evaluate_single_snapshot(
            checkpoint_path, reference_data, config, device, output_dir
        )
        
        # 繪圖
        plot_physics_comparison(results, output_dir, is_time_series=False)
    
    # 保存結果
    results_path = output_dir / 'physics_metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"✅ 評估完成！結果已保存至: {output_dir}")
    logger.info(f"  - 圖表: {output_dir}/physics_comparison.png")
    logger.info(f"  - 數據: {output_dir}/physics_metrics.json")


if __name__ == '__main__':
    main()
