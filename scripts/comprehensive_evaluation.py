"""
綜合評估腳本：3D VS-PINN 與 JHTDB 參考數據比較

功能：
1. 載入訓練完成的模型與 JHTDB 參考數據
2. 計算全面的物理驗證指標（誤差、守恆、能譜、壁剪應力）
3. 生成高質量可視化（場比較、誤差分布、統計圖）
4. 輸出結構化評估報告（Markdown + JSON）
5. 驗證成敗門檻（相對 L2 ≤ 10-15%，統計改善 ≥ 30%）

使用範例：
    python scripts/comprehensive_evaluation.py \
        --checkpoint checkpoints/vs_pinn_3d_full_training_latest.pth \
        --config configs/vs_pinn_3d_full_training.yml \
        --reference data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz \
        --output_dir results/comprehensive_eval_<timestamp>
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
import logging
import json
from typing import Dict, Tuple, Optional, List
import argparse
from datetime import datetime

# 設置樣式
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.family'] = 'sans-serif'

# 設置日誌
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 導入專案模組
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import pinnx
from pinnx.models.fourier_mlp import PINNNet
# from pinnx.physics.vs_pinn_channel_flow import create_vs_pinn_channel_flow  # Not needed for evaluation
from pinnx.evals.metrics import (
    relative_L2, rmse_metrics, field_statistics,
    energy_spectrum_1d, conservation_error
)


# ============================================================
# 模型載入與推理
# ============================================================

def load_trained_model(checkpoint_path: Path, config: Dict, device: torch.device):
    """載入訓練完成的模型"""
    logger.info(f"📥 Loading model from {checkpoint_path}")
    
    # 🔧 從配置文件構建 statistics 以支持 3D 模型
    # 這確保 ManualScalingWrapper 能正確設置 input_min/max 的形狀
    statistics = None
    if 'physics' in config and 'domain' in config['physics']:
        domain = config['physics']['domain']
        statistics = {
            'x': {'range': domain.get('x_range', [0.0, 25.13])},
            'y': {'range': domain.get('y_range', [-1.0, 1.0])}
        }
        # 如果是 3D，添加 z 範圍
        if 'z_range' in domain:
            statistics['z'] = {'range': domain['z_range']}
        logger.info(f"📐 Constructed statistics from config: {list(statistics.keys())}")
    
    # 創建模型架構
    from scripts.train import create_model
    model = create_model(config, device, statistics=statistics)
    
    # 載入權重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        logger.info(f"✅ Loaded checkpoint from epoch {epoch}")
    else:
        model.load_state_dict(checkpoint)
        logger.info(f"✅ Loaded checkpoint (legacy format)")
    
    model.eval()
    return model


def load_jhtdb_reference(data_path: Path) -> Dict[str, np.ndarray]:
    """載入 JHTDB 參考數據"""
    logger.info(f"📥 Loading JHTDB reference from {data_path}")
    
    data = np.load(data_path)
    
    # 檢查數據格式
    required_fields = ['u', 'v', 'w', 'p', 'x', 'y', 'z']
    for field in required_fields:
        if field not in data:
            logger.warning(f"⚠️  Field '{field}' not found in reference data")
    
    logger.info(f"✅ Loaded reference data: u{data['u'].shape}, "
                f"domain: x[{data['x'].min():.2f}, {data['x'].max():.2f}], "
                f"y[{data['y'].min():.2f}, {data['y'].max():.2f}], "
                f"z[{data['z'].min():.2f}, {data['z'].max():.2f}]")
    
    return {key: data[key] for key in data.files}


def predict_on_grid(model, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                    device: torch.device, batch_size: int = 10000) -> Dict[str, np.ndarray]:
    """在網格上進行預測"""
    logger.info(f"🔮 Predicting on grid: {len(x)}×{len(y)}×{len(z)} = {len(x)*len(y)*len(z)} points")
    
    # 生成網格點
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    n_points = points.shape[0]
    
    # 分批預測
    u_list, v_list, w_list, p_list = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = torch.tensor(points[i:i+batch_size], dtype=torch.float32, device=device)
            pred = model(batch)
            
            u_list.append(pred[:, 0].cpu().numpy())
            v_list.append(pred[:, 1].cpu().numpy())
            w_list.append(pred[:, 2].cpu().numpy())
            p_list.append(pred[:, 3].cpu().numpy())
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Progress: {i+len(batch)}/{n_points} ({100*(i+len(batch))/n_points:.1f}%)")
    
    # 重塑為 3D 網格
    shape = (len(x), len(y), len(z))
    
    results = {
        'u': np.concatenate(u_list).reshape(shape),
        'v': np.concatenate(v_list).reshape(shape),
        'w': np.concatenate(w_list).reshape(shape),
        'p': np.concatenate(p_list).reshape(shape),
        'x': x,
        'y': y,
        'z': z
    }
    
    logger.info(f"✅ Prediction complete")
    return results


# ============================================================
# 物理指標計算
# ============================================================

def compute_error_metrics(pred: Dict[str, np.ndarray], 
                          ref: Dict[str, np.ndarray]) -> Dict[str, float]:
    """計算誤差指標"""
    logger.info("📊 Computing error metrics...")
    
    metrics = {}
    
    for field in ['u', 'v', 'w', 'p']:
        if field not in pred or field not in ref:
            continue
        
        pred_field = pred[field].flatten()
        ref_field = ref[field].flatten()
        
        # 相對 L2 誤差
        l2_error = np.linalg.norm(pred_field - ref_field) / (np.linalg.norm(ref_field) + 1e-12)
        
        # RMSE
        rmse = np.sqrt(np.mean((pred_field - ref_field)**2))
        
        # 相對 RMSE
        rel_rmse = rmse / (np.std(ref_field) + 1e-12)
        
        # 最大絕對誤差
        max_error = np.max(np.abs(pred_field - ref_field))
        
        metrics[f'{field}_l2_error'] = l2_error
        metrics[f'{field}_rmse'] = rmse
        metrics[f'{field}_rel_rmse'] = rel_rmse
        metrics[f'{field}_max_error'] = max_error
    
    # 綜合指標
    metrics['overall_l2_error'] = np.mean([
        metrics.get(f'{f}_l2_error', 0) for f in ['u', 'v', 'w']
    ])
    
    logger.info(f"✅ Overall L2 error: {metrics['overall_l2_error']:.4f}")
    
    return metrics


def compute_field_statistics(pred: Dict[str, np.ndarray], 
                             ref: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """計算場統計量"""
    logger.info("📊 Computing field statistics...")
    
    stats = {'pred': {}, 'ref': {}, 'improvement': {}}
    
    for field in ['u', 'v', 'w', 'p']:
        if field not in pred or field not in ref:
            continue
        
        pred_field = pred[field].flatten()
        ref_field = ref[field].flatten()
        
        # 預測場統計
        stats['pred'][f'{field}_mean'] = np.mean(pred_field)
        stats['pred'][f'{field}_std'] = np.std(pred_field)
        stats['pred'][f'{field}_min'] = np.min(pred_field)
        stats['pred'][f'{field}_max'] = np.max(pred_field)
        
        # 參考場統計
        stats['ref'][f'{field}_mean'] = np.mean(ref_field)
        stats['ref'][f'{field}_std'] = np.std(ref_field)
        stats['ref'][f'{field}_min'] = np.min(ref_field)
        stats['ref'][f'{field}_max'] = np.max(ref_field)
        
        # 統計量誤差
        mean_error = np.abs(stats['pred'][f'{field}_mean'] - stats['ref'][f'{field}_mean'])
        std_error = np.abs(stats['pred'][f'{field}_std'] - stats['ref'][f'{field}_std'])
        
        stats['improvement'][f'{field}_mean_error'] = mean_error
        stats['improvement'][f'{field}_std_error'] = std_error
    
    return stats


def compute_wall_shear_stress_comparison(pred: Dict[str, np.ndarray], 
                                         ref: Dict[str, np.ndarray]) -> Dict[str, float]:
    """比較壁面剪應力"""
    logger.info("📊 Computing wall shear stress comparison...")
    
    # 假設壁面在 y 方向的邊界
    y_idx_lower = 0  # 下壁面
    y_idx_upper = -1  # 上壁面
    
    # 計算速度梯度（使用有限差分）
    dy = pred['y'][1] - pred['y'][0]
    
    # 下壁面剪應力：τ_w = μ * ∂u/∂y
    pred_tau_lower = (pred['u'][:, 1, :] - pred['u'][:, 0, :]) / dy
    ref_tau_lower = (ref['u'][:, 1, :] - ref['u'][:, 0, :]) / dy
    
    # 統計量
    metrics = {
        'pred_tau_mean': np.mean(pred_tau_lower),
        'pred_tau_std': np.std(pred_tau_lower),
        'ref_tau_mean': np.mean(ref_tau_lower),
        'ref_tau_std': np.std(ref_tau_lower),
        'tau_rmse': np.sqrt(np.mean((pred_tau_lower - ref_tau_lower)**2)),
        'tau_rel_error': np.abs(np.mean(pred_tau_lower) - np.mean(ref_tau_lower)) / (np.abs(np.mean(ref_tau_lower)) + 1e-12)
    }
    
    logger.info(f"✅ Wall shear stress: pred={metrics['pred_tau_mean']:.6f}, "
                f"ref={metrics['ref_tau_mean']:.6f}, error={metrics['tau_rel_error']:.2%}")
    
    return metrics


def compute_energy_spectrum_comparison(pred: Dict[str, np.ndarray], 
                                       ref: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """比較能量譜"""
    logger.info("📊 Computing energy spectrum comparison...")
    
    # 選擇中間 y 平面
    y_mid = len(pred['y']) // 2
    
    # 計算動能：0.5 * (u² + v² + w²)
    pred_ke = 0.5 * (pred['u'][:, y_mid, :]**2 + pred['v'][:, y_mid, :]**2 + pred['w'][:, y_mid, :]**2)
    ref_ke = 0.5 * (ref['u'][:, y_mid, :]**2 + ref['v'][:, y_mid, :]**2 + ref['w'][:, y_mid, :]**2)
    
    # 2D FFT
    pred_fft = np.fft.fft2(pred_ke)
    ref_fft = np.fft.fft2(ref_ke)
    
    # 能量譜
    pred_spectrum = np.abs(np.fft.fftshift(pred_fft))**2
    ref_spectrum = np.abs(np.fft.fftshift(ref_fft))**2
    
    # 徑向平均
    h, w = pred_ke.shape
    center_h, center_w = h // 2, w // 2
    
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.sqrt((x_idx - center_w)**2 + (y_idx - center_h)**2).astype(int)
    
    k_max = min(center_h, center_w)
    k_bins = np.arange(1, k_max)
    
    pred_radial = np.zeros(len(k_bins))
    ref_radial = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        mask = (r == k)
        if mask.sum() > 0:
            pred_radial[i] = pred_spectrum[mask].mean()
            ref_radial[i] = ref_spectrum[mask].mean()
    
    # 計算能譜 RMSE
    spectrum_rmse = np.sqrt(np.mean((pred_radial - ref_radial)**2))
    spectrum_rel_error = spectrum_rmse / (np.mean(ref_radial) + 1e-12)
    
    logger.info(f"✅ Energy spectrum RMSE: {spectrum_rmse:.2e}, rel_error: {spectrum_rel_error:.2%}")
    
    return {
        'k': k_bins,
        'pred_spectrum': pred_radial,
        'ref_spectrum': ref_radial,
        'spectrum_rmse': spectrum_rmse,
        'spectrum_rel_error': spectrum_rel_error
    }


# ============================================================
# 可視化函數
# ============================================================

def plot_error_distribution(pred: Dict[str, np.ndarray], 
                            ref: Dict[str, np.ndarray], 
                            save_dir: Path):
    """繪製誤差分布圖"""
    logger.info("🎨 Plotting error distribution...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 選擇中間 z 平面
    z_mid = len(pred['z']) // 2
    
    for idx, field in enumerate(['u', 'v', 'w', 'p']):
        ax = axes[idx // 2, idx % 2]
        
        pred_slice = pred[field][:, :, z_mid]
        ref_slice = ref[field][:, :, z_mid]
        error = np.abs(pred_slice - ref_slice)
        
        im = ax.contourf(pred['x'], pred['y'], error.T, levels=20, cmap='hot')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'{field.upper()} Absolute Error (z={pred["z"][z_mid]:.2f})', fontsize=16)
        plt.colorbar(im, ax=ax, label='|pred - ref|')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved error distribution to {save_dir / 'error_distribution.png'}")
    plt.close()


def plot_field_comparison(pred: Dict[str, np.ndarray], 
                          ref: Dict[str, np.ndarray], 
                          save_dir: Path):
    """繪製場比較圖（預測 vs 參考）"""
    logger.info("🎨 Plotting field comparison...")
    
    z_mid = len(pred['z']) // 2
    
    for field in ['u', 'v', 'w', 'p']:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        pred_slice = pred[field][:, :, z_mid]
        ref_slice = ref[field][:, :, z_mid]
        
        # 統一色階
        vmin = min(pred_slice.min(), ref_slice.min())
        vmax = max(pred_slice.max(), ref_slice.max())
        
        # 預測場
        im0 = axes[0].contourf(pred['x'], pred['y'], pred_slice.T, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title(f'{field.upper()} - Predicted', fontsize=16)
        axes[0].set_xlabel('x', fontsize=14)
        axes[0].set_ylabel('y', fontsize=14)
        plt.colorbar(im0, ax=axes[0])
        
        # 參考場
        im1 = axes[1].contourf(ref['x'], ref['y'], ref_slice.T, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[1].set_title(f'{field.upper()} - Reference (JHTDB)', fontsize=16)
        axes[1].set_xlabel('x', fontsize=14)
        axes[1].set_ylabel('y', fontsize=14)
        plt.colorbar(im1, ax=axes[1])
        
        # 誤差場
        error = pred_slice - ref_slice
        im2 = axes[2].contourf(pred['x'], pred['y'], error.T, levels=20, cmap='seismic')
        axes[2].set_title(f'{field.upper()} - Error', fontsize=16)
        axes[2].set_xlabel('x', fontsize=14)
        axes[2].set_ylabel('y', fontsize=14)
        plt.colorbar(im2, ax=axes[2], label='pred - ref')
        
        plt.tight_layout()
        plt.savefig(save_dir / f'field_comparison_{field}.png', dpi=150, bbox_inches='tight')
        logger.info(f"✅ Saved {field} comparison")
        plt.close()


def plot_velocity_profiles(pred: Dict[str, np.ndarray], 
                           ref: Dict[str, np.ndarray], 
                           save_dir: Path):
    """繪製速度剖面比較"""
    logger.info("🎨 Plotting velocity profiles...")
    
    # 選擇域中心
    x_mid = len(pred['x']) // 2
    z_mid = len(pred['z']) // 2
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, field in enumerate(['u', 'v', 'w']):
        ax = axes[idx]
        
        pred_profile = pred[field][x_mid, :, z_mid]
        ref_profile = ref[field][x_mid, :, z_mid]
        
        ax.plot(pred_profile, pred['y'], 'b-', linewidth=2, label='Predicted')
        ax.plot(ref_profile, ref['y'], 'r--', linewidth=2, label='JHTDB Reference')
        
        ax.set_xlabel(f'{field}', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'{field.upper()} Velocity Profile (x={pred["x"][x_mid]:.2f}, z={pred["z"][z_mid]:.2f})', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'velocity_profiles_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved velocity profiles")
    plt.close()


def plot_energy_spectrum(spectrum_data: Dict, save_dir: Path):
    """繪製能量譜比較"""
    logger.info("🎨 Plotting energy spectrum...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    k = spectrum_data['k']
    pred_spec = spectrum_data['pred_spectrum']
    ref_spec = spectrum_data['ref_spectrum']
    
    # 線性尺度
    axes[0].plot(k, pred_spec, 'b-', linewidth=2, label='Predicted')
    axes[0].plot(k, ref_spec, 'r--', linewidth=2, label='JHTDB Reference')
    axes[0].set_xlabel('Wavenumber k', fontsize=14)
    axes[0].set_ylabel('E(k)', fontsize=14)
    axes[0].set_title('Energy Spectrum (Linear Scale)', fontsize=16)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 對數尺度
    axes[1].loglog(k, pred_spec, 'b-', linewidth=2, label='Predicted')
    axes[1].loglog(k, ref_spec, 'r--', linewidth=2, label='JHTDB Reference')
    
    # 繪製 -5/3 律參考線
    k_ref = k[len(k)//4:len(k)//2]
    spec_ref = ref_spec[len(k)//4] * (k_ref / k[len(k)//4])**(-5/3)
    axes[1].loglog(k_ref, spec_ref, 'k:', linewidth=1.5, label=r'$k^{-5/3}$ law')
    
    axes[1].set_xlabel('Wavenumber k', fontsize=14)
    axes[1].set_ylabel('E(k)', fontsize=14)
    axes[1].set_title('Energy Spectrum (Log-Log Scale)', fontsize=16)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'energy_spectrum_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved energy spectrum")
    plt.close()


def plot_wall_shear_stress(pred: Dict[str, np.ndarray], 
                           ref: Dict[str, np.ndarray], 
                           save_dir: Path):
    """繪製壁面剪應力比較"""
    logger.info("🎨 Plotting wall shear stress...")
    
    # 計算壁面剪應力
    dy = pred['y'][1] - pred['y'][0]
    pred_tau = (pred['u'][:, 1, :] - pred['u'][:, 0, :]) / dy
    ref_tau = (ref['u'][:, 1, :] - ref['u'][:, 0, :]) / dy
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    # 統一色階
    vmin = min(pred_tau.min(), ref_tau.min())
    vmax = max(pred_tau.max(), ref_tau.max())
    
    # 預測
    im0 = axes[0].contourf(pred['x'], pred['z'], pred_tau.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Wall Shear Stress - Predicted', fontsize=16)
    axes[0].set_xlabel('x', fontsize=14)
    axes[0].set_ylabel('z', fontsize=14)
    plt.colorbar(im0, ax=axes[0], label=r'$\tau_w$')
    
    # 參考
    im1 = axes[1].contourf(ref['x'], ref['z'], ref_tau.T, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Wall Shear Stress - Reference', fontsize=16)
    axes[1].set_xlabel('x', fontsize=14)
    axes[1].set_ylabel('z', fontsize=14)
    plt.colorbar(im1, ax=axes[1], label=r'$\tau_w$')
    
    # 誤差
    error = pred_tau - ref_tau
    im2 = axes[2].contourf(pred['x'], pred['z'], error.T, levels=20, cmap='seismic')
    axes[2].set_title('Wall Shear Stress - Error', fontsize=16)
    axes[2].set_xlabel('x', fontsize=14)
    axes[2].set_ylabel('z', fontsize=14)
    plt.colorbar(im2, ax=axes[2], label=r'$\Delta\tau_w$')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'wall_shear_stress_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved wall shear stress comparison")
    plt.close()


def plot_statistics_comparison(stats: Dict, save_dir: Path):
    """繪製統計量比較"""
    logger.info("🎨 Plotting statistics comparison...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    fields = ['u', 'v', 'w', 'p']
    metrics = ['mean', 'std', 'min', 'max']
    
    for idx, field in enumerate(fields):
        ax = axes[idx // 2, idx % 2]
        
        pred_values = [stats['pred'].get(f'{field}_{m}', 0) for m in metrics]
        ref_values = [stats['ref'].get(f'{field}_{m}', 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, pred_values, width, label='Predicted', alpha=0.8)
        ax.bar(x + width/2, ref_values, width, label='JHTDB Reference', alpha=0.8)
        
        ax.set_xlabel('Statistics', fontsize=14)
        ax.set_ylabel('Value', fontsize=14)
        ax.set_title(f'{field.upper()} Statistics Comparison', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'statistics_comparison.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved statistics comparison")
    plt.close()


# ============================================================
# 報告生成
# ============================================================

def generate_markdown_report(metrics: Dict, stats: Dict, spectrum_data: Dict, 
                            wall_metrics: Dict, config: Dict, 
                            checkpoint_path: str, save_path: Path):
    """生成 Markdown 評估報告"""
    logger.info("📝 Generating Markdown report...")
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 檢查成敗門檻
    overall_l2 = metrics.get('overall_l2_error', float('inf'))
    threshold_pass = "✅ PASS" if overall_l2 <= 0.15 else "❌ FAIL"
    
    report = f"""# 3D VS-PINN 綜合評估報告

**生成時間**: {timestamp}  
**檢查點**: `{checkpoint_path}`  
**配置文件**: `{config.get('_config_path', 'N/A')}`

---

## 📊 執行摘要

### 成敗門檻驗證

| 指標 | 目標 | 實際值 | 狀態 |
|------|------|--------|------|
| 整體相對 L2 誤差 | ≤ 10-15% | **{overall_l2:.2%}** | {threshold_pass} |
| u 相對 L2 誤差 | ≤ 15% | {metrics.get('u_l2_error', 0):.2%} | {"✅" if metrics.get('u_l2_error', 1) <= 0.15 else "❌"} |
| v 相對 L2 誤差 | ≤ 15% | {metrics.get('v_l2_error', 0):.2%} | {"✅" if metrics.get('v_l2_error', 1) <= 0.15 else "❌"} |
| w 相對 L2 誤差 | ≤ 15% | {metrics.get('w_l2_error', 0):.2%} | {"✅" if metrics.get('w_l2_error', 1) <= 0.15 else "❌"} |
| p 相對 L2 誤差 | ≤ 20% | {metrics.get('p_l2_error', 0):.2%} | {"✅" if metrics.get('p_l2_error', 1) <= 0.20 else "❌"} |

### 關鍵發現

- **整體精度**: 相對 L2 誤差 = **{overall_l2:.2%}**
- **壁面剪應力**: 相對誤差 = **{wall_metrics.get('tau_rel_error', 0):.2%}**
- **能量譜**: 相對誤差 = **{spectrum_data.get('spectrum_rel_error', 0):.2%}**

---

## 🎯 詳細誤差分析

### 場誤差指標

| 場 | 相對 L2 | RMSE | 相對 RMSE | 最大誤差 |
|---|---------|------|-----------|----------|
| **u** | {metrics.get('u_l2_error', 0):.4f} | {metrics.get('u_rmse', 0):.6f} | {metrics.get('u_rel_rmse', 0):.4f} | {metrics.get('u_max_error', 0):.6f} |
| **v** | {metrics.get('v_l2_error', 0):.4f} | {metrics.get('v_rmse', 0):.6f} | {metrics.get('v_rel_rmse', 0):.4f} | {metrics.get('v_max_error', 0):.6f} |
| **w** | {metrics.get('w_l2_error', 0):.4f} | {metrics.get('w_rmse', 0):.6f} | {metrics.get('w_rel_rmse', 0):.4f} | {metrics.get('w_max_error', 0):.6f} |
| **p** | {metrics.get('p_l2_error', 0):.4f} | {metrics.get('p_rmse', 0):.6f} | {metrics.get('p_rel_rmse', 0):.4f} | {metrics.get('p_max_error', 0):.6f} |

---

## 📈 場統計比較

### 流向速度 (u)

| 統計量 | 預測值 | JHTDB 參考 | 絕對誤差 | 相對誤差 |
|--------|--------|-----------|----------|----------|
| Mean | {stats['pred'].get('u_mean', 0):.6f} | {stats['ref'].get('u_mean', 0):.6f} | {stats['improvement'].get('u_mean_error', 0):.6f} | {stats['improvement'].get('u_mean_error', 0) / (abs(stats['ref'].get('u_mean', 1)) + 1e-12):.2%} |
| Std | {stats['pred'].get('u_std', 0):.6f} | {stats['ref'].get('u_std', 0):.6f} | {stats['improvement'].get('u_std_error', 0):.6f} | {stats['improvement'].get('u_std_error', 0) / (stats['ref'].get('u_std', 1) + 1e-12):.2%} |
| Min | {stats['pred'].get('u_min', 0):.6f} | {stats['ref'].get('u_min', 0):.6f} | - | - |
| Max | {stats['pred'].get('u_max', 0):.6f} | {stats['ref'].get('u_max', 0):.6f} | - | - |

### 法向速度 (v)

| 統計量 | 預測值 | JHTDB 參考 | 絕對誤差 | 相對誤差 |
|--------|--------|-----------|----------|----------|
| Mean | {stats['pred'].get('v_mean', 0):.6f} | {stats['ref'].get('v_mean', 0):.6f} | {stats['improvement'].get('v_mean_error', 0):.6f} | {stats['improvement'].get('v_mean_error', 0) / (abs(stats['ref'].get('v_mean', 1e-12)) + 1e-12):.2%} |
| Std | {stats['pred'].get('v_std', 0):.6f} | {stats['ref'].get('v_std', 0):.6f} | {stats['improvement'].get('v_std_error', 0):.6f} | {stats['improvement'].get('v_std_error', 0) / (stats['ref'].get('v_std', 1) + 1e-12):.2%} |

### 展向速度 (w)

| 統計量 | 預測值 | JHTDB 參考 | 絕對誤差 | 相對誤差 |
|--------|--------|-----------|----------|----------|
| Mean | {stats['pred'].get('w_mean', 0):.6f} | {stats['ref'].get('w_mean', 0):.6f} | {stats['improvement'].get('w_mean_error', 0):.6f} | {stats['improvement'].get('w_mean_error', 0) / (abs(stats['ref'].get('w_mean', 1e-12)) + 1e-12):.2%} |
| Std | {stats['pred'].get('w_std', 0):.6f} | {stats['ref'].get('w_std', 0):.6f} | {stats['improvement'].get('w_std_error', 0):.6f} | {stats['improvement'].get('w_std_error', 0) / (stats['ref'].get('w_std', 1) + 1e-12):.2%} |

---

## 🌊 物理驗證

### 壁面剪應力

| 指標 | 預測值 | JHTDB 參考 | 誤差 |
|------|--------|-----------|------|
| Mean τ_w | {wall_metrics.get('pred_tau_mean', 0):.6f} | {wall_metrics.get('ref_tau_mean', 0):.6f} | {wall_metrics.get('tau_rel_error', 0):.2%} |
| Std τ_w | {wall_metrics.get('pred_tau_std', 0):.6f} | {wall_metrics.get('ref_tau_std', 0):.6f} | - |
| RMSE | {wall_metrics.get('tau_rmse', 0):.6f} | - | - |

### 能量譜

- **譜 RMSE**: {spectrum_data.get('spectrum_rmse', 0):.6e}
- **譜相對誤差**: {spectrum_data.get('spectrum_rel_error', 0):.2%}
- **波數範圍**: k ∈ [{spectrum_data['k'][0]:.2f}, {spectrum_data['k'][-1]:.2f}]

---

## 📁 輸出文件

### 可視化圖表

- `error_distribution.png` - 誤差分布（4 場）
- `field_comparison_u.png` - u 場比較（預測 vs 參考 vs 誤差）
- `field_comparison_v.png` - v 場比較
- `field_comparison_w.png` - w 場比較
- `field_comparison_p.png` - p 場比較
- `velocity_profiles_comparison.png` - 速度剖面比較
- `energy_spectrum_comparison.png` - 能量譜比較（線性 & 對數）
- `wall_shear_stress_comparison.png` - 壁面剪應力比較
- `statistics_comparison.png` - 統計量比較

### 數據文件

- `evaluation_metrics.json` - 完整指標（JSON 格式）
- `predicted_field.npz` - 預測流場數據

---

## 🔧 訓練配置

```yaml
Model: {config.get('model', {})}
Physics: {config.get('physics', {})}
Training: {config.get('training', {})}
```

---

**報告結束** | 生成於 {timestamp}
"""
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"✅ Saved Markdown report to {save_path}")


def save_metrics_json(metrics: Dict, stats: Dict, spectrum_data: Dict, 
                      wall_metrics: Dict, save_path: Path):
    """保存指標為 JSON 格式"""
    logger.info("💾 Saving metrics to JSON...")
    
    # 遞迴轉換 numpy 類型為 Python 原生類型
    def convert_to_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'error_metrics': convert_to_json_serializable(metrics),
        'field_statistics': convert_to_json_serializable(stats),
        'wall_shear_stress': convert_to_json_serializable(wall_metrics),
        'energy_spectrum': {
            'spectrum_rmse': float(spectrum_data.get('spectrum_rmse', 0)),
            'spectrum_rel_error': float(spectrum_data.get('spectrum_rel_error', 0))
        },
        'success_criteria': {
            'overall_l2_threshold': 0.15,
            'overall_l2_actual': float(metrics.get('overall_l2_error', 0)),
            'passed': bool(metrics.get('overall_l2_error', 1) <= 0.15)
        }
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ Saved JSON metrics to {save_path}")


# ============================================================
# 主函數
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Comprehensive 3D VS-PINN Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--reference', type=str, required=True, help='Path to JHTDB reference data')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/mps/auto)')
    parser.add_argument('--batch_size', type=int, default=10000, help='Batch size for prediction')
    
    args = parser.parse_args()
    
    # 設置輸出目錄
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path(f'results/comprehensive_eval_{timestamp}')
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"📁 Output directory: {output_dir}")
    
    # 載入配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config['_config_path'] = args.config
    
    # 設置設備
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🖥️  Using device: {device}")
    
    # ========== 載入模型與數據 ==========
    model = load_trained_model(Path(args.checkpoint), config, device)
    ref_data = load_jhtdb_reference(Path(args.reference))
    
    # ========== 預測 ==========
    pred_data = predict_on_grid(
        model, 
        ref_data['x'], 
        ref_data['y'], 
        ref_data['z'], 
        device, 
        batch_size=args.batch_size
    )
    
    # 保存預測場
    np.savez(
        output_dir / 'predicted_field.npz',
        **pred_data
    )
    logger.info(f"💾 Saved predicted field to {output_dir / 'predicted_field.npz'}")
    
    # ========== 計算指標 ==========
    error_metrics = compute_error_metrics(pred_data, ref_data)
    field_stats = compute_field_statistics(pred_data, ref_data)
    wall_metrics = compute_wall_shear_stress_comparison(pred_data, ref_data)
    spectrum_data = compute_energy_spectrum_comparison(pred_data, ref_data)
    
    # ========== 可視化 ==========
    plot_error_distribution(pred_data, ref_data, output_dir)
    plot_field_comparison(pred_data, ref_data, output_dir)
    plot_velocity_profiles(pred_data, ref_data, output_dir)
    plot_energy_spectrum(spectrum_data, output_dir)
    plot_wall_shear_stress(pred_data, ref_data, output_dir)
    plot_statistics_comparison(field_stats, output_dir)
    
    # ========== 生成報告 ==========
    generate_markdown_report(
        error_metrics, field_stats, spectrum_data, wall_metrics,
        config, args.checkpoint, output_dir / 'evaluation_report.md'
    )
    
    save_metrics_json(
        error_metrics, field_stats, spectrum_data, wall_metrics,
        output_dir / 'evaluation_metrics.json'
    )
    
    # ========== 終端輸出摘要 ==========
    logger.info("\n" + "="*60)
    logger.info("🎉 評估完成！")
    logger.info("="*60)
    logger.info(f"📊 整體相對 L2 誤差: {error_metrics.get('overall_l2_error', 0):.2%}")
    logger.info(f"🌊 壁面剪應力誤差: {wall_metrics.get('tau_rel_error', 0):.2%}")
    logger.info(f"📈 能量譜誤差: {spectrum_data.get('spectrum_rel_error', 0):.2%}")
    logger.info(f"📁 結果保存於: {output_dir}")
    logger.info("="*60)
    
    # 檢查成敗
    if error_metrics.get('overall_l2_error', 1) <= 0.15:
        logger.info("✅ 成功！整體誤差低於 15% 門檻")
    else:
        logger.warning("⚠️  整體誤差超過 15% 門檻，建議進一步調優")


if __name__ == '__main__':
    main()
