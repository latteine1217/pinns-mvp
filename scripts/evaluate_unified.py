#!/usr/bin/env python3
"""
統一評估腳本 - PINNs 模型完整評估工具
Unified Evaluation Script for PINNs Models

功能特性:
1. 🎯 自動檢測物理場景 (Kolmogorov 2D, Channel 3D)
2. 📊 完整評估指標 (誤差、守恆、能譜、壁剪應力)
3. 📈 高品質視覺化 (場對比、誤差分布、統計圖)
4. 📄 多格式輸出 (JSON, Markdown, PNG)
5. 🔄 多模型比較模式
6. ✅ 統一的反標準化處理

使用範例:
    # 單一模型評估
    python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth
    
    # 指定輸出目錄
    python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth --output results/eval
    
    # 多模型比較
    python scripts/evaluate_unified.py \
        --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \
        --labels "RANS Prior" "Vanilla" "Proposed" \
        --output results/comparison
    
    # 自定義評估選項
    python scripts/evaluate_unified.py \
        --checkpoint checkpoints/model.pth \
        --metrics all \
        --visualize all \
        --format json,markdown,png

作者: PINNs-Sparse-Flow Team
日期: 2026-01-05
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 專案模組
from pinnx.utils.evaluation_utils import load_model_for_evaluation, predict_with_denormalization
from pinnx.utils.denormalization import denormalize_output
from pinnx.evals.metrics import (
    relative_L2, rmse_metrics, conservation_error,
    energy_spectrum_1d, energy_spectrum_2d, wall_shear_stress, 
    compute_kinetic_energy_2d, compute_enstrophy_2d,
    comprehensive_evaluation
)

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore', category=UserWarning)


# ============================================================================
#  1. 模型和數據載入
# ============================================================================

# 物理類型映射表（dict dispatch pattern）
PHYSICS_TYPE_MAP = {
    'kolmogorov_flow_2d': 'kolmogorov_2d',
    'kolmogorov': 'kolmogorov_2d',
    'channel_flow_3d': 'channel_3d',
    'jhtdb_channel': 'channel_3d',
    'channel': 'channel_3d',
    'jhtdb': 'channel_3d',
}

def detect_physics_type(config: Dict) -> str:
    """
    自動檢測物理場景類型
    
    使用 dict dispatch pattern 來避免過多的條件分支。
    支援的類型:
        - kolmogorov_2d: Kolmogorov Flow 2D
        - channel_3d: Channel Flow 3D (JHTDB)
    
    Args:
        config: 訓練配置字典
        
    Returns:
        物理場景類型 ('kolmogorov_2d' 或 'channel_3d')
        
    Raises:
        ValueError: 若物理類型不支援
    """
    physics = config.get('physics', {})
    physics_type = physics.get('type', '').lower()
    
    # 檢查物理類型映射表
    for key, value in PHYSICS_TYPE_MAP.items():
        if key in physics_type:
            logger.info(f"✅ 檢測到物理類型: {value} (從 '{physics_type}' 匹配)")
            return value
    
    # 若未匹配，拋出錯誤（Fail Fast 原則）
    supported_types = list(PHYSICS_TYPE_MAP.keys())
    raise ValueError(
        f"不支援的物理類型: '{physics_type}'. "
        f"支援的類型: {supported_types}"
    )


def load_reference_data_kolmogorov(config: Dict, t_eval: float = 25.0) -> Dict:
    """載入 Kolmogorov 2D 參考數據"""
    data_path = config.get('data', {}).get('kolmogorov_config', {}).get('data_path')
    if not data_path:
        raise ValueError("配置中未找到 kolmogorov_config.data_path")
    
    logger.info(f"載入 Kolmogorov DNS 數據: {data_path}")
    
    with h5py.File(data_path, 'r') as f:
        # 讀取配置
        L = f['config'].attrs.get('L', 2*np.pi)
        N = f['config'].attrs.get('N', 256)
        
        # 讀取時間
        t = f['time'][:]
        t_idx = np.argmin(np.abs(t - t_eval))
        actual_t = t[t_idx]
        
        logger.info(f"  評估時間: t={actual_t:.2f} (requested: {t_eval:.2f})")
        
        # 讀取場數據
        u = f['u'][t_idx, :, :]
        v = f['v'][t_idx, :, :]
        p = f['p'][t_idx, :, :]
        
        # 生成座標
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # 準備座標（t, x, y）
        T = np.full_like(X, actual_t)
        coords = np.stack([T.ravel(), X.ravel(), Y.ravel()], axis=1)
    
    return {
        'coords': coords,
        'u': u.ravel(),
        'v': v.ravel(),
        'p': p.ravel(),
        'shape': u.shape,
        'domain': (L, L),
        'time': actual_t
    }


def load_reference_data_channel(config: Dict) -> Dict:
    """載入 Channel 3D 參考數據"""
    logger.info("載入 JHTDB Channel Flow 數據...")
    
    from pinnx.dataio.channel_flow_loader import ChannelFlowLoader
    
    loader = ChannelFlowLoader(config_path=config)
    field_dataset = loader.load_full_field_data()
    
    # 轉換為點數據
    coords, ref_fields = field_dataset.to_points(order=('x', 'y', 'z'))
    
    return {
        'coords': coords,
        'u': ref_fields['u'],
        'v': ref_fields['v'],
        'w': ref_fields.get('w', np.zeros_like(ref_fields['u'])),
        'p': ref_fields['p'],
        'shape': field_dataset.shape
    }


def load_reference_data(config: Dict, physics_type: str) -> Dict:
    """根據物理類型載入參考數據"""
    if physics_type == 'kolmogorov_2d':
        return load_reference_data_kolmogorov(config)
    elif physics_type == 'channel_3d':
        return load_reference_data_channel(config)
    else:
        raise ValueError(f"不支援的物理類型: {physics_type}")


# ============================================================================
#  2. 指標計算
# ============================================================================

def compute_basic_metrics(pred: Dict, ref: Dict, physics_type: str) -> Dict:
    """計算基本誤差指標"""
    metrics = {}
    
    variables = ['u', 'v', 'p'] if physics_type == 'kolmogorov_2d' else ['u', 'v', 'w', 'p']
    
    for var in variables:
        if var not in pred or var not in ref:
            continue
        
        # L2 相對誤差
        l2_error = np.sqrt(np.mean((pred[var] - ref[var])**2))
        l2_norm = np.sqrt(np.mean(ref[var]**2))
        rel_l2 = (l2_error / l2_norm * 100) if l2_norm > 1e-10 else 0.0
        
        # RMSE
        rmse = np.sqrt(np.mean((pred[var] - ref[var])**2))
        
        # 最大誤差
        max_error = np.max(np.abs(pred[var] - ref[var]))
        
        metrics[var] = {
            'rel_l2': rel_l2,
            'rmse': rmse,
            'max_error': max_error,
            'mean_pred': np.mean(pred[var]),
            'mean_ref': np.mean(ref[var])
        }
    
    return metrics


def compute_physics_metrics(pred: Dict, coords: np.ndarray, shape: Tuple) -> Dict:
    """計算物理一致性指標（散度、守恆）"""
    metrics = {}
    
    # 重塑為網格
    if len(shape) == 2:  # 2D
        nx, ny = shape
        u = pred['u'].reshape(nx, ny)
        v = pred['v'].reshape(nx, ny)
        
        # 計算散度 (有限差分)
        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        div = du_dx + dv_dy
        
        metrics['divergence'] = {
            'mean': np.mean(np.abs(div)),
            'max': np.max(np.abs(div)),
            'rmse': np.sqrt(np.mean(div**2))
        }
        
    elif len(shape) == 3:  # 3D
        nx, ny, nz = shape
        u = pred['u'].reshape(nx, ny, nz)
        v = pred['v'].reshape(nx, ny, nz)
        w = pred['w'].reshape(nx, ny, nz)
        
        # 計算散度
        du_dx = np.gradient(u, axis=0)
        dv_dy = np.gradient(v, axis=1)
        dw_dz = np.gradient(w, axis=2)
        div = du_dx + dv_dy + dw_dz
        
        metrics['divergence'] = {
            'mean': np.mean(np.abs(div)),
            'max': np.max(np.abs(div)),
            'rmse': np.sqrt(np.mean(div**2))
        }
    
    return metrics


def compute_kolmogorov_physics(pred: Dict, ref: Dict, shape: Tuple, 
                               domain_size: float = 2*np.pi) -> Dict:
    """
    計算 Kolmogorov Flow 專用物理量
    
    Args:
        pred: 預測數據 {'u': [N], 'v': [N], ...}
        ref: 參考數據 {'u': [N], 'v': [N], ...}
        shape: 網格形狀 (Nx, Ny)
        domain_size: 域大小（預設 2π）
    
    Returns:
        物理量字典
    """
    physics = {}
    
    # 重塑為 2D 網格
    nx, ny = shape
    u_pred = pred['u'].reshape(nx, ny)
    v_pred = pred['v'].reshape(nx, ny)
    u_ref = ref['u'].reshape(nx, ny)
    v_ref = ref['v'].reshape(nx, ny)
    
    # 轉換為 torch tensor（metrics.py 中的函數需要 torch tensor）
    u_pred_t = torch.from_numpy(u_pred).float()
    v_pred_t = torch.from_numpy(v_pred).float()
    u_ref_t = torch.from_numpy(u_ref).float()
    v_ref_t = torch.from_numpy(v_ref).float()
    
    # 1. 動能 (Kinetic Energy)
    ke_pred = compute_kinetic_energy_2d(u_pred_t.flatten(), v_pred_t.flatten())
    ke_ref = compute_kinetic_energy_2d(u_ref_t.flatten(), v_ref_t.flatten())
    
    physics['kinetic_energy'] = {
        'prediction': ke_pred,
        'reference': ke_ref,
        'error': (ke_pred - ke_ref) / ke_ref * 100  # 百分比
    }
    
    # 2. 擾動度 (Enstrophy) - 需要計算渦度
    # 手動計算渦度（因為 compute_enstrophy_2d 需要座標）
    dx = domain_size / (nx - 1)
    dy = domain_size / (ny - 1)
    
    dvdx_pred = np.gradient(v_pred, dx, axis=1)
    dudy_pred = np.gradient(u_pred, dy, axis=0)
    vorticity_pred = dvdx_pred - dudy_pred
    enstrophy_pred = 0.5 * np.mean(vorticity_pred**2)
    
    dvdx_ref = np.gradient(v_ref, dx, axis=1)
    dudy_ref = np.gradient(u_ref, dy, axis=0)
    vorticity_ref = dvdx_ref - dudy_ref
    enstrophy_ref = 0.5 * np.mean(vorticity_ref**2)
    
    physics['enstrophy'] = {
        'prediction': enstrophy_pred,
        'reference': enstrophy_ref,
        'error': (enstrophy_pred - enstrophy_ref) / enstrophy_ref * 100
    }
    
    # 3. 能量譜 (Energy Spectrum)
    k_pred, E_k_pred = energy_spectrum_2d(u_pred_t, v_pred_t, domain_size)
    k_ref, E_k_ref = energy_spectrum_2d(u_ref_t, v_ref_t, domain_size)
    
    physics['energy_spectrum'] = {
        'k_pred': k_pred,
        'E_k_pred': E_k_pred,
        'k_ref': k_ref,
        'E_k_ref': E_k_ref
    }
    
    return physics


def compute_all_metrics(pred: Dict, ref: Dict, coords: np.ndarray, 
                       shape: Tuple, physics_type: str, 
                       compute_advanced_physics: bool = False) -> Dict:
    """計算所有評估指標"""
    logger.info("計算評估指標...")
    
    metrics = {}
    
    # 基本誤差指標
    metrics['basic'] = compute_basic_metrics(pred, ref, physics_type)
    
    # 物理指標
    metrics['physics'] = compute_physics_metrics(pred, coords, shape)
    
    # Kolmogorov Flow 專用物理量
    if compute_advanced_physics and physics_type == 'kolmogorov_2d':
        logger.info("計算 Kolmogorov Flow 物理量...")
        metrics['kolmogorov_physics'] = compute_kolmogorov_physics(pred, ref, shape)
    
    # 總結
    all_rel_l2 = [m['rel_l2'] for m in metrics['basic'].values()]
    metrics['summary'] = {
        'mean_rel_l2': np.mean(all_rel_l2),
        'max_rel_l2': np.max(all_rel_l2),
        'pass_threshold': np.max(all_rel_l2) <= 15.0  # 15% 閾值
    }
    
    logger.info(f"  平均相對 L2 誤差: {metrics['summary']['mean_rel_l2']:.2f}%")
    
    return metrics


# ============================================================================
#  3. 視覺化
# ============================================================================

def plot_field_comparison(pred: Dict, ref: Dict, shape: Tuple, 
                         output_dir: Path, physics_type: str):
    """繪製場對比圖（DNS vs Prediction vs Error）"""
    logger.info("生成場對比圖...")
    
    variables = ['u', 'v', 'p'] if physics_type == 'kolmogorov_2d' else ['u', 'v', 'w', 'p']
    
    if len(shape) == 2:  # 2D
        nx, ny = shape
        fig, axes = plt.subplots(len(variables), 3, figsize=(15, 4*len(variables)))
        
        for i, var in enumerate(variables):
            if var not in pred:
                continue
            
            pred_field = pred[var].reshape(nx, ny)
            ref_field = ref[var].reshape(nx, ny)
            error = np.abs(pred_field - ref_field)
            
            # DNS
            im1 = axes[i, 0].imshow(ref_field.T, origin='lower', cmap='RdBu_r')
            axes[i, 0].set_title(f'{var.upper()} - DNS')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Prediction
            im2 = axes[i, 1].imshow(pred_field.T, origin='lower', cmap='RdBu_r', 
                                   vmin=ref_field.min(), vmax=ref_field.max())
            axes[i, 1].set_title(f'{var.upper()} - Prediction')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Error
            im3 = axes[i, 2].imshow(error.T, origin='lower', cmap='hot_r')
            axes[i, 2].set_title(f'{var.upper()} - |Error|')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'field_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    elif len(shape) == 3:  # 3D - 繪製中間切片
        nx, ny, nz = shape
        slice_idx = nz // 2
        
        fig, axes = plt.subplots(len(variables), 3, figsize=(15, 4*len(variables)))
        
        for i, var in enumerate(variables):
            if var not in pred:
                continue
            
            pred_field = pred[var].reshape(nx, ny, nz)[:, :, slice_idx]
            ref_field = ref[var].reshape(nx, ny, nz)[:, :, slice_idx]
            error = np.abs(pred_field - ref_field)
            
            # DNS
            im1 = axes[i, 0].imshow(ref_field.T, origin='lower', cmap='RdBu_r')
            axes[i, 0].set_title(f'{var.upper()} - DNS (z={slice_idx})')
            plt.colorbar(im1, ax=axes[i, 0])
            
            # Prediction
            im2 = axes[i, 1].imshow(pred_field.T, origin='lower', cmap='RdBu_r',
                                   vmin=ref_field.min(), vmax=ref_field.max())
            axes[i, 1].set_title(f'{var.upper()} - Prediction')
            plt.colorbar(im2, ax=axes[i, 1])
            
            # Error
            im3 = axes[i, 2].imshow(error.T, origin='lower', cmap='hot_r')
            axes[i, 2].set_title(f'{var.upper()} - |Error|')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'field_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"  已保存: {output_dir / 'field_comparison.png'}")


def plot_error_distribution(metrics: Dict, output_dir: Path):
    """繪製誤差分布圖"""
    logger.info("生成誤差分布圖...")
    
    basic_metrics = metrics['basic']
    variables = list(basic_metrics.keys())
    rel_l2_errors = [basic_metrics[v]['rel_l2'] for v in variables]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(variables, rel_l2_errors, color='skyblue', edgecolor='navy', alpha=0.7)
    
    # 添加閾值線
    ax.axhline(y=15.0, color='red', linestyle='--', linewidth=2, label='Target Threshold (15%)')
    ax.axhline(y=10.0, color='green', linestyle='--', linewidth=2, label='Excellent (10%)')
    
    ax.set_xlabel('Variable', fontsize=12, fontweight='bold')
    ax.set_ylabel('Relative L2 Error (%)', fontsize=12, fontweight='bold')
    ax.set_title('Prediction Errors by Variable', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 在柱狀圖上標註數值
    for bar, val in zip(bars, rel_l2_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  已保存: {output_dir / 'error_distribution.png'}")


def plot_kolmogorov_physics_4panel(metrics: Dict, output_dir: Path):
    """
    繪製 Kolmogorov Flow 4-panel 物理量對比圖
    
    (a) 相對 L2 誤差 (u, v)
    (b) 動能對比
    (c) 擾動度對比
    (d) 能量譜（log-log with scaling lines）
    """
    logger.info("生成 Kolmogorov Flow 物理量 4-panel 對比圖...")
    
    if 'kolmogorov_physics' not in metrics:
        logger.warning("未找到 Kolmogorov 物理量數據，跳過 4-panel 圖")
        return
    
    kol_phys = metrics['kolmogorov_physics']
    basic = metrics['basic']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # ========== Panel (a): 相對 L2 誤差 ==========
    ax = axes[0, 0]
    variables = ['u', 'v']
    errors = [basic[var]['rel_l2'] for var in variables if var in basic]
    
    ax.bar(variables, errors, color=['#2E86AB', '#A23B72'], alpha=0.7, width=0.5)
    ax.set_ylabel('Relative L2 Error (%)', fontsize=11)
    ax.set_title('(a) Velocity Field Errors', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(errors) * 1.2])
    
    # 標註數值
    for i, (var, err) in enumerate(zip(variables, errors)):
        ax.text(i, err, f'{err:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # ========== Panel (b): 動能對比 ==========
    ax = axes[0, 1]
    ke_ref = kol_phys['kinetic_energy']['reference']
    ke_pred = kol_phys['kinetic_energy']['prediction']
    ke_error = kol_phys['kinetic_energy']['error']
    
    bars = ax.bar(['Reference', 'Prediction'], [ke_ref, ke_pred], 
                  color=['#06A77D', '#D4AF37'], alpha=0.7, width=0.5)
    ax.axhline(ke_ref, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Reference')
    
    ax.set_ylabel('Kinetic Energy', fontsize=11)
    ax.set_title(f'(b) Kinetic Energy (Error: {ke_error:+.2f}%)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # 標註數值
    for bar, val in zip(bars, [ke_ref, ke_pred]):
        ax.text(bar.get_x() + bar.get_width()/2, val, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ========== Panel (c): 擾動度對比 ==========
    ax = axes[1, 0]
    ens_ref = kol_phys['enstrophy']['reference']
    ens_pred = kol_phys['enstrophy']['prediction']
    ens_error = kol_phys['enstrophy']['error']
    
    bars = ax.bar(['Reference', 'Prediction'], [ens_ref, ens_pred],
                  color=['#265077', '#F5A962'], alpha=0.7, width=0.5)
    ax.axhline(ens_ref, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Reference')
    
    ax.set_ylabel('Enstrophy', fontsize=11)
    ax.set_title(f'(c) Enstrophy (Error: {ens_error:+.2f}%)', 
                 fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    
    # 標註數值
    for bar, val in zip(bars, [ens_ref, ens_pred]):
        ax.text(bar.get_x() + bar.get_width()/2, val, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ========== Panel (d): 能量譜 ==========
    ax = axes[1, 1]
    k_ref = kol_phys['energy_spectrum']['k_ref']
    E_k_ref = kol_phys['energy_spectrum']['E_k_ref']
    k_pred = kol_phys['energy_spectrum']['k_pred']
    E_k_pred = kol_phys['energy_spectrum']['E_k_pred']
    
    ax.loglog(k_ref, E_k_ref, 'o-', label='Reference', markersize=4, alpha=0.7, linewidth=2)
    ax.loglog(k_pred, E_k_pred, 's-', label='Prediction', markersize=3, alpha=0.7, linewidth=2)
    
    # 理論標度線
    k_range = np.linspace(3.0, 30.0, 20)
    E_inertial = 1e3 * k_range**(-5.0/3.0)
    E_dissipation = 1e-2 * k_range**(-3.0)
    
    ax.loglog(k_range, E_inertial, 'k--', alpha=0.5, linewidth=1.5, label=r'$k^{-5/3}$')
    ax.loglog(k_range, E_dissipation, 'k:', alpha=0.5, linewidth=1.5, label=r'$k^{-3}$')
    
    ax.set_xlabel('Wavenumber $k$', fontsize=11)
    ax.set_ylabel('Energy Spectrum $E(k)$', fontsize=11)
    ax.set_title('(d) Energy Spectrum', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'kolmogorov_physics_4panel.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"  已保存: {output_dir / 'kolmogorov_physics_4panel.png'}")



def generate_visualizations(pred: Dict, ref: Dict, metrics: Dict,
                           shape: Tuple, output_dir: Path, physics_type: str):
    """生成所有視覺化"""
    logger.info("生成視覺化圖表...")
    
    # 場對比圖
    plot_field_comparison(pred, ref, shape, output_dir, physics_type)
    
    # 誤差分布圖
    plot_error_distribution(metrics, output_dir)
    
    # Kolmogorov 物理量 4-panel 圖（如果有計算）
    if 'kolmogorov_physics' in metrics:
        plot_kolmogorov_physics_4panel(metrics, output_dir)
    
    logger.info("✅ 視覺化完成")


# ============================================================================
#  4. 報告生成
# ============================================================================

def generate_json_report(metrics: Dict, output_path: Path, model_info: Dict):
    """生成 JSON 格式報告"""
    
    # 轉換 numpy 類型為 Python 原生類型
    def convert_to_native(obj):
        if isinstance(obj, (np.floating, np.complexfloating)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
            return int(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': convert_to_native(model_info),
        'metrics': convert_to_native(metrics)
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"  JSON 報告: {output_path}")


def generate_markdown_report(metrics: Dict, output_path: Path, model_info: Dict):
    """生成 Markdown 格式報告"""
    
    md = f"""# PINNs 模型評估報告

## 模型資訊
- **Checkpoint**: {model_info['checkpoint']}
- **Epoch**: {model_info.get('epoch', 'Unknown')}
- **評估時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 基本誤差指標

| Variable | Rel L2 (%) | RMSE | Max Error |
|----------|------------|------|-----------|
"""
    
    for var, vals in metrics['basic'].items():
        md += f"| {var.upper()} | {vals['rel_l2']:.2f}% | {vals['rmse']:.4f} | {vals['max_error']:.4f} |\n"
    
    md += f"""
---

## 🔬 物理一致性指標

### 散度誤差
- **平均**: {metrics['physics']['divergence']['mean']:.6f}
- **最大**: {metrics['physics']['divergence']['max']:.6f}
- **RMSE**: {metrics['physics']['divergence']['rmse']:.6f}

---

## ✅ 評估總結

- **平均相對 L2 誤差**: {metrics['summary']['mean_rel_l2']:.2f}%
- **最大相對 L2 誤差**: {metrics['summary']['max_rel_l2']:.2f}%
- **是否通過 15% 閾值**: {'✅ 是' if metrics['summary']['pass_threshold'] else '❌ 否'}

---

## 📈 視覺化圖表

- `field_comparison.png`: 場對比圖（DNS vs Prediction vs Error）
- `error_distribution.png`: 誤差分布柱狀圖

"""
    
    with open(output_path, 'w') as f:
        f.write(md)
    
    logger.info(f"  Markdown 報告: {output_path}")


# ============================================================================
#  5. 主要評估流程
# ============================================================================

def evaluate_single_model(checkpoint_path: str, output_dir: Path, 
                         metrics_level: str = 'all',
                         visualize_level: str = 'all',
                         physics_analysis: bool = False) -> Dict:
    """評估單一模型"""
    
    logger.info("="*80)
    logger.info(f"開始評估: {checkpoint_path}")
    logger.info("="*80)
    
    # 1. 載入 checkpoint 和配置
    logger.info("載入 checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'config' not in checkpoint:
        raise ValueError(f"Checkpoint 中未找到 config: {checkpoint_path}")
    
    config = checkpoint['config']
    logger.info(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    
    # 2. 載入模型
    logger.info("載入模型...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model, physics = load_model_for_evaluation(checkpoint_path, config, device)
    model.eval()
    logger.info("✅ 模型載入完成")
    
    # 2. 檢測物理類型
    physics_type = detect_physics_type(config)
    logger.info(f"檢測到物理類型: {physics_type}")
    
    # 3. 載入參考數據
    ref_data = load_reference_data(config, physics_type)
    
    # 4. 模型預測
    logger.info("執行模型預測...")
    coords_tensor = torch.from_numpy(ref_data['coords']).float()
    pred_array = predict_with_denormalization(
        model, coords_tensor, config, checkpoint_path, 
        physics=physics, device=device
    )
    
    # 組織預測結果
    pred_data = {}
    if physics_type == 'kolmogorov_2d':
        pred_data = {
            'u': pred_array[:, 0],
            'v': pred_array[:, 1],
            'p': pred_array[:, 2]
        }
    else:  # channel_3d
        pred_data = {
            'u': pred_array[:, 0],
            'v': pred_array[:, 1],
            'w': pred_array[:, 2],
            'p': pred_array[:, 3]
        }
    
    # 5. 計算指標（包含物理量分析）
    metrics = compute_all_metrics(
        pred_data, ref_data, ref_data['coords'], 
        ref_data['shape'], physics_type,
        compute_advanced_physics=physics_analysis
    )
    
    # 6. 生成視覺化
    if visualize_level in ['all', 'fields', 'errors']:
        generate_visualizations(
            pred_data, ref_data, metrics, 
            ref_data['shape'], output_dir, physics_type
        )
    
    # 7. 生成報告
    model_info = {
        'checkpoint': checkpoint_path,
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'physics_type': physics_type
    }
    
    generate_json_report(metrics, output_dir / 'evaluation_results.json', model_info)
    generate_markdown_report(metrics, output_dir / 'evaluation_report.md', model_info)
    
    logger.info("="*80)
    logger.info("✅ 評估完成！")
    logger.info(f"📁 輸出目錄: {output_dir}")
    logger.info("="*80)
    
    return metrics


def compare_multiple_models(checkpoint_paths: List[str], labels: List[str],
                           output_dir: Path) -> Dict:
    """比較多個模型"""
    
    logger.info("="*80)
    logger.info(f"開始比較 {len(checkpoint_paths)} 個模型")
    logger.info("="*80)
    
    all_results = {}
    
    for ckpt_path, label in zip(checkpoint_paths, labels):
        logger.info(f"\n評估模型: {label}")
        model_dir = output_dir / label.replace(' ', '_')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        results = evaluate_single_model(ckpt_path, model_dir)
        all_results[label] = results
    
    # 生成比較報告
    comparison_md = "# 模型比較報告\n\n"
    comparison_md += f"評估時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    comparison_md += "## 相對 L2 誤差比較 (%)\n\n"
    comparison_md += "| Variable | " + " | ".join(labels) + " |\n"
    comparison_md += "|----------|" + "|".join(["--------"]*len(labels)) + "|\n"
    
    # 獲取所有變量
    first_result = list(all_results.values())[0]
    variables = list(first_result['basic'].keys())
    
    for var in variables:
        row = f"| {var.upper()} |"
        for label in labels:
            rel_l2 = all_results[label]['basic'][var]['rel_l2']
            row += f" {rel_l2:.2f}% |"
        comparison_md += row + "\n"
    
    # 平均誤差
    comparison_md += "\n## 平均誤差\n\n"
    comparison_md += "| Model | Mean Rel L2 (%) | Pass Threshold |\n"
    comparison_md += "|-------|-----------------|----------------|\n"
    
    for label in labels:
        mean_l2 = all_results[label]['summary']['mean_rel_l2']
        passed = all_results[label]['summary']['pass_threshold']
        comparison_md += f"| {label} | {mean_l2:.2f}% | {'✅' if passed else '❌'} |\n"
    
    with open(output_dir / 'comparison_report.md', 'w') as f:
        f.write(comparison_md)
    
    logger.info("="*80)
    logger.info("✅ 比較完成！")
    logger.info(f"📁 輸出目錄: {output_dir}")
    logger.info("="*80)
    
    return all_results


# ============================================================================
#  6. CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='統一評估腳本 - PINNs 模型完整評估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 單一模型評估
  python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth
  
  # Kolmogorov Flow 物理量分析
  python scripts/evaluate_unified.py --checkpoint checkpoints/model.pth --physics-analysis
  
  # 多模型比較
  python scripts/evaluate_unified.py \\
      --checkpoints ckpt1.pth ckpt2.pth ckpt3.pth \\
      --labels "RANS Prior" "Vanilla" "Proposed"
        """
    )
    
    # 基本參數
    parser.add_argument('--checkpoint', type=str, help='單一 checkpoint 路徑')
    parser.add_argument('--checkpoints', type=str, nargs='+', help='多個 checkpoint 路徑（比較模式）')
    parser.add_argument('--labels', type=str, nargs='+', help='模型標籤（比較模式）')
    parser.add_argument('--output', type=str, default='results/evaluation', help='輸出目錄')
    
    # 評估選項
    parser.add_argument('--metrics', type=str, default='all', 
                       choices=['all', 'basic', 'physics'],
                       help='指標計算層級')
    parser.add_argument('--visualize', type=str, default='all',
                       choices=['all', 'fields', 'errors', 'none'],
                       help='視覺化層級')
    parser.add_argument('--physics-analysis', action='store_true',
                       help='計算進階物理量（Kolmogorov: KE, Enstrophy, Energy Spectrum）')
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 比較模式
        if args.checkpoints:
            if not args.labels:
                args.labels = [f"Model_{i+1}" for i in range(len(args.checkpoints))]
            
            if len(args.labels) != len(args.checkpoints):
                raise ValueError("labels 數量必須與 checkpoints 數量相同")
            
            compare_multiple_models(args.checkpoints, args.labels, output_dir)
        
        # 單一模型評估
        elif args.checkpoint:
            evaluate_single_model(
                args.checkpoint, output_dir, 
                args.metrics, args.visualize,
                physics_analysis=args.physics_analysis
            )
        
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger.error(f"評估失敗: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
