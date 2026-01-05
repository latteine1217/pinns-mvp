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
    energy_spectrum_1d, conservation_error,
    pressure_gradient_from_finite_diff  # 新增壓力梯度計算
)
from pinnx.utils.denormalization import denormalize_output  # TASK-008: 反標準化工具
from pinnx.utils.evaluation_utils import load_model_for_evaluation, predict_with_denormalization  # 統一評估工具


# ============================================================
# 模型載入與推理（使用統一工具）
# ============================================================

def load_trained_model(checkpoint_path: Path, config: Dict, device: torch.device):
    """載入訓練完成的模型（含 physics 狀態恢復）
    
    ⚠️  此函數已遷移至 pinnx.utils.evaluation_utils.load_model_for_evaluation()
    這裡保留作為向後相容的包裝器
    """
    return load_model_for_evaluation(str(checkpoint_path), config, device)


def load_jhtdb_reference(data_path: Path) -> Dict[str, np.ndarray]:
    """載入 JHTDB 參考數據（支援 2D/3D）"""
    logger.info(f"📥 Loading JHTDB reference from {data_path}")
    
    data = np.load(data_path, allow_pickle=True)
    
    # 檢查數據維度（2D 或 3D）
    is_3d = 'z' in data and 'w' in data
    
    # 檢查是否需要 reshape（1D flatten 格式）
    if 'grid_shape' in data and data['u'].ndim == 1:
        grid_shape = data['grid_shape']
        logger.info(f"🔄 Reshaping 1D data to grid: {grid_shape}")
        
        # Reshape 流場變數
        result = {key: data[key] for key in data.files}
        for var in ['u', 'v', 'w', 'p']:
            if var in data:
                result[var] = data[var].reshape(*grid_shape, 1)  # (nx, ny, 1)
        
        domain_info = (f"domain: x[{data['x'].min():.2f}, {data['x'].max():.2f}], "
                      f"y[{data['y'].min():.2f}, {data['y'].max():.2f}], "
                      f"z[{data['z']:.2f}, {data['z']:.2f}]")
        logger.info(f"✅ Loaded reference data (reshaped): u{result['u'].shape}, {domain_info}")
        return result
    
    # 原有邏輯（3D 或 2D）
    if is_3d:
        required_fields = ['u', 'v', 'w', 'p', 'x', 'y', 'z']
        domain_info = (f"domain: x[{data['x'].min():.2f}, {data['x'].max():.2f}], "
                      f"y[{data['y'].min():.2f}, {data['y'].max():.2f}], "
                      f"z[{data['z'].min():.2f}, {data['z'].max():.2f}]")
    else:
        required_fields = ['u', 'v', 'p', 'x', 'y']
        domain_info = (f"domain: x[{data['x'].min():.2f}, {data['x'].max():.2f}], "
                      f"y[{data['y'].min():.2f}, {data['y'].max():.2f}] (2D slice)")
    
    # 檢查數據格式
    missing_fields = [f for f in required_fields if f not in data]
    if missing_fields:
        logger.warning(f"⚠️  Missing fields: {missing_fields}")
    
    logger.info(f"✅ Loaded reference data ({'3D' if is_3d else '2D'}): u{data['u'].shape}, {domain_info}")
    
    return {key: data[key] for key in data.files}


def predict_on_grid(model, x: np.ndarray, y: np.ndarray, z: np.ndarray, 
                    device: torch.device, batch_size: int = 10000, 
                    physics=None, config: Dict = None, checkpoint_path: str = None) -> Dict[str, np.ndarray]:
    """在網格上進行預測
    
    Args:
        model: 訓練好的模型
        physics: VS-PINN physics 模組（用於座標縮放）
        config: 配置字典（用於反標準化，TASK-008）
        checkpoint_path: checkpoint 路徑（用於載入標準化統計量）
        ...
    """
    # 處理標量 z（2D slice）
    z_array = np.atleast_1d(z)
    logger.info(f"🔮 Predicting on grid: {len(x)}×{len(y)}×{len(z_array)} = {len(x)*len(y)*len(z_array)} points")
    
    # 🆕 檢查是否使用 VS-PINN 縮放
    use_vs_pinn = physics is not None and hasattr(physics, 'scale_coordinates')
    if use_vs_pinn:
        logger.info(f"🔧 Using VS-PINN coordinate scaling: N_x={physics.N_x.item():.2f}, N_y={physics.N_y.item():.2f}, N_z={physics.N_z.item():.2f}")
    else:
        logger.info(f"🔧 Using direct model inference (no scaling)")
    
    # 生成網格點
    X, Y, Z = np.meshgrid(x, y, z_array, indexing='ij')
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
    n_points = points.shape[0]
    
    # 分批預測
    u_list, v_list, w_list, p_list = [], [], [], []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = torch.tensor(points[i:i+batch_size], dtype=torch.float32, device=device)
            
            # 🆕 應用 VS-PINN 座標縮放（如果有）
            if use_vs_pinn:
                batch = physics.scale_coordinates(batch)
            
            # 模型推理（輸出為標準化空間）
            pred = model(batch)
            
            # ✅ TASK-008: 反標準化回物理空間
            if config is None:
                raise ValueError("config is required for comprehensive evaluation to ensure proper denormalization")
            pred_physical = denormalize_output(
                pred.cpu().numpy(), 
                config, 
                output_norm_type='training_data_norm',
                verbose=True,  # 🔍 啟用詳細日誌
                checkpoint_path=checkpoint_path
            )
            
            u_list.append(pred_physical[:, 0])
            v_list.append(pred_physical[:, 1])
            w_list.append(pred_physical[:, 2])
            p_list.append(pred_physical[:, 3])
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Progress: {i+len(batch)}/{n_points} ({100*(i+len(batch))/n_points:.1f}%)")
    
    # 重塑為 3D 網格
    shape = (len(x), len(y), len(z_array))
    
    results = {
        'u': np.concatenate(u_list).reshape(shape),
        'v': np.concatenate(v_list).reshape(shape),
        'w': np.concatenate(w_list).reshape(shape),
        'p': np.concatenate(p_list).reshape(shape),
        'x': x,
        'y': y,
        'z': z_array
    }
    
    logger.info(f"✅ Prediction complete")
    return results


# ============================================================
# 物理指標計算
# ============================================================

def compute_error_metrics(pred: Dict[str, np.ndarray],
                          ref: Dict[str, np.ndarray]) -> Dict[str, float]:
    """計算誤差指標（包含壓力梯度評估）"""
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

    # 🆕 壓力梯度評估（更準確的壓力場指標）
    if 'p' in pred and 'p' in ref:
        logger.info("📊 Computing pressure gradient metrics (more accurate than absolute pressure)...")
        try:
            # 使用有限差分計算壓力梯度
            pred_p_grad = pressure_gradient_from_finite_diff(
                pred['p'],
                {'x': pred['x'], 'y': pred['y'], 'z': pred.get('z', np.array([0.0]))}
            )
            ref_p_grad = pressure_gradient_from_finite_diff(
                ref['p'],
                {'x': ref['x'], 'y': ref['y'], 'z': ref.get('z', np.array([0.0]))}
            )

            # 計算各方向梯度誤差
            for direction in ['dpdx', 'dpdy', 'dpdz']:
                if direction in pred_p_grad and direction in ref_p_grad:
                    pred_grad_flat = pred_p_grad[direction].flatten()
                    ref_grad_flat = ref_p_grad[direction].flatten()

                    # 相對 L2 誤差
                    grad_l2_error = np.linalg.norm(pred_grad_flat - ref_grad_flat) / (np.linalg.norm(ref_grad_flat) + 1e-12)
                    metrics[f'{direction}_l2_error'] = grad_l2_error

                    # RMSE
                    grad_rmse = np.sqrt(np.mean((pred_grad_flat - ref_grad_flat)**2))
                    metrics[f'{direction}_rmse'] = grad_rmse

                    # 平均值（驗證通道流驅動梯度）
                    metrics[f'{direction}_pred_mean'] = np.mean(pred_grad_flat)
                    metrics[f'{direction}_ref_mean'] = np.mean(ref_grad_flat)

                    # 標準差（驗證是否為常數梯度）
                    metrics[f'{direction}_pred_std'] = np.std(pred_grad_flat)
                    metrics[f'{direction}_ref_std'] = np.std(ref_grad_flat)

            # 綜合壓力梯度誤差
            grad_errors = [metrics.get(f'{d}_l2_error', 0) for d in ['dpdx', 'dpdy', 'dpdz'] if f'{d}_l2_error' in metrics]
            if grad_errors:
                metrics['pressure_gradient_l2_error'] = np.mean(grad_errors)
                logger.info(f"✅ Pressure gradient L2 error: {metrics['pressure_gradient_l2_error']:.4f}")

        except Exception as e:
            logger.warning(f"⚠️  Pressure gradient calculation failed: {e}")
            metrics['pressure_gradient_l2_error'] = float('inf')

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
                                         ref: Dict[str, np.ndarray],
                                         nu: float = 5e-5,  # 🔴 新增參數
                                         rho: float = 1.0) -> Dict[str, float]:
    """
    比較壁面剪應力（修正版：包含黏性係數）

    理論公式: τ_w = μ * (∂u/∂y)|_wall = ρ*ν * (∂u/∂y)|_wall

    Args:
        pred: 預測場數據
        ref: 參考場數據
        nu: 運動黏性係數 [m²/s] (JHTDB Channel Flow: 5e-5)
        rho: 密度 [kg/m³] (標準化為 1.0)

    Returns:
        壁面剪應力指標字典,包含:
        - pred/ref_tau_{lower,upper}_mean/std: 上下壁面統計量
        - tau_{lower,upper}_rmse/rel_error: 誤差指標
        - symmetry_error: 對稱性檢查
        - theoretical_tau_w: 理論值 0.0025
        - pred_vs_theory_error: 預測與理論的相對誤差
    """
    logger.info(f"📊 Computing wall shear stress (nu={nu}, rho={rho})...")

    # 🔴 動力黏性係數 (關鍵修正)
    mu = rho * nu

    # 檢測 2D vs 3D
    is_2d = len(ref['u'].shape) == 2

    # 計算網格間距 (假設均勻網格)
    dy = pred['y'][1] - pred['y'][0]
    logger.info(f"   Grid spacing dy = {dy:.6f}")

    # 計算壁面速度梯度 (∂u/∂y)
    if is_2d:
        # 2D: shape (nx, ny) - squeeze pred if needed
        pred_u = pred['u'].squeeze() if pred['u'].ndim == 3 else pred['u']

        # 下壁面 (y=y_min, 索引 0)
        pred_dudy_lower = (pred_u[:, 1] - pred_u[:, 0]) / dy
        ref_dudy_lower = (ref['u'][:, 1] - ref['u'][:, 0]) / dy

        # 上壁面 (y=y_max, 索引 -1)
        pred_dudy_upper = (pred_u[:, -1] - pred_u[:, -2]) / dy
        ref_dudy_upper = (ref['u'][:, -1] - ref['u'][:, -2]) / dy

    else:
        # 3D: shape (nx, ny, nz)
        pred_dudy_lower = (pred['u'][:, 1, :] - pred['u'][:, 0, :]) / dy
        ref_dudy_lower = (ref['u'][:, 1, :] - ref['u'][:, 0, :]) / dy

        pred_dudy_upper = (pred['u'][:, -1, :] - pred['u'][:, -2, :]) / dy
        ref_dudy_upper = (ref['u'][:, -1, :] - ref['u'][:, -2, :]) / dy

    # 🔴 關鍵修正: 乘以黏性係數得到剪應力
    pred_tau_lower = mu * pred_dudy_lower
    ref_tau_lower = mu * ref_dudy_lower

    pred_tau_upper = mu * pred_dudy_upper
    ref_tau_upper = mu * ref_dudy_upper

    # 統計量
    metrics = {
        # 下壁面
        'pred_tau_lower_mean': float(np.mean(pred_tau_lower)),
        'pred_tau_lower_std': float(np.std(pred_tau_lower)),
        'ref_tau_lower_mean': float(np.mean(ref_tau_lower)),
        'ref_tau_lower_std': float(np.std(ref_tau_lower)),

        # 上壁面
        'pred_tau_upper_mean': float(np.mean(pred_tau_upper)),
        'pred_tau_upper_std': float(np.std(pred_tau_upper)),
        'ref_tau_upper_mean': float(np.mean(ref_tau_upper)),
        'ref_tau_upper_std': float(np.std(ref_tau_upper)),

        # 誤差指標
        'tau_lower_rmse': float(np.sqrt(np.mean((pred_tau_lower - ref_tau_lower)**2))),
        'tau_upper_rmse': float(np.sqrt(np.mean((pred_tau_upper - ref_tau_upper)**2))),

        'tau_lower_rel_error': float(np.abs(np.mean(pred_tau_lower) - np.mean(ref_tau_lower))
                                     / (np.abs(np.mean(ref_tau_lower)) + 1e-12)),
        'tau_upper_rel_error': float(np.abs(np.mean(pred_tau_upper) - np.mean(ref_tau_upper))
                                     / (np.abs(np.mean(ref_tau_upper)) + 1e-12)),

        # 對稱性檢查 (通道流應上下對稱)
        'symmetry_error_pred': float(np.abs(np.mean(pred_tau_lower) - np.mean(pred_tau_upper))
                                     / (np.mean(pred_tau_lower) + 1e-12)),
        'symmetry_error_ref': float(np.abs(np.mean(ref_tau_lower) - np.mean(ref_tau_upper))
                                    / (np.mean(ref_tau_lower) + 1e-12)),

        # 理論值比較 (JHTDB Channel Flow Re_τ=1000)
        'theoretical_tau_w': 0.0025,
        'pred_vs_theory_error': float(np.abs(np.mean(pred_tau_lower) - 0.0025) / 0.0025),
    }

    logger.info(f"✅ Wall shear stress ({'2D' if is_2d else '3D'}):")
    logger.info(f"   Lower wall: pred={metrics['pred_tau_lower_mean']:.6f}, "
                f"ref={metrics['ref_tau_lower_mean']:.6f}")
    logger.info(f"   Upper wall: pred={metrics['pred_tau_upper_mean']:.6f}, "
                f"ref={metrics['ref_tau_upper_mean']:.6f}")
    logger.info(f"   Theoretical (Re_τ=1000): {metrics['theoretical_tau_w']:.6f}")
    logger.info(f"   Prediction vs. theory error: {metrics['pred_vs_theory_error']:.2%}")

    # 🔴 警告檢查
    if metrics['pred_vs_theory_error'] > 0.5:  # 超過 50% 偏差
        logger.warning(f"⚠️  Wall shear stress deviates significantly from theory!")
        logger.warning(f"   Expected: {metrics['theoretical_tau_w']:.6f}")
        logger.warning(f"   Predicted: {metrics['pred_tau_lower_mean']:.6f}")
        logger.warning(f"   Error: {metrics['pred_vs_theory_error']:.2%}")

    if metrics['symmetry_error_pred'] > 0.15:  # 超過 15% 不對稱
        logger.warning(f"⚠️  Channel flow lacks symmetry (upper/lower walls differ by "
                      f"{metrics['symmetry_error_pred']:.2%})")

    return metrics


def compute_energy_spectrum_comparison(pred: Dict[str, np.ndarray],
                                       ref: Dict[str, np.ndarray],
                                       spectrum_type: str = 'streamwise_1d') -> Dict[str, np.ndarray]:
    """
    比較能量譜（通道流專用方法）

    通道流是非均勻的剪切湍流，不適用徑向平均2D FFT方法（僅適用於均勻各向同性湍流）。
    應使用流向1D能譜（沿x方向）並對y方向分層或平均。

    Args:
        pred: 預測場數據
        ref: 參考場數據
        spectrum_type: 能譜類型
            - 'streamwise_1d': 流向1D能譜（推薦，適用於通道流）
            - 'radial_2d': 徑向平均2D能譜（傳統方法，不適用於通道流）

    Returns:
        能譜數據字典，包含:
        - k: 波數
        - pred_spectrum: 預測能譜
        - ref_spectrum: 參考能譜
        - spectrum_rmse: RMSE
        - spectrum_rel_error: 相對誤差
        - spectrum_type: 使用的能譜類型
    """
    logger.info(f"📊 Computing energy spectrum (type: {spectrum_type})...")

    # 檢測 2D vs 3D
    is_2d = len(ref['u'].shape) == 2

    if spectrum_type == 'streamwise_1d':
        # ✅ 推薦：流向1D能譜（適用於通道流）
        return _compute_streamwise_spectrum(pred, ref, is_2d)

    elif spectrum_type == 'radial_2d':
        # ⚠️ 傳統方法：徑向平均2D能譜（僅適用於均勻各向同性湍流）
        logger.warning("⚠️  Using radial 2D spectrum for channel flow (not physically appropriate)")
        logger.warning("   Consider using 'streamwise_1d' for channel flow")
        return _compute_radial_spectrum(pred, ref, is_2d)

    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum_type}. "
                        f"Choose 'streamwise_1d' or 'radial_2d'")


def _compute_streamwise_spectrum(pred: Dict[str, np.ndarray],
                                  ref: Dict[str, np.ndarray],
                                  is_2d: bool) -> Dict[str, np.ndarray]:
    """
    計算流向1D能譜（通道流專用）

    方法：沿流向（x方向）進行1D FFT，對法向（y方向）和展向（z方向）平均
    """
    if is_2d:
        # 2D: (nx, ny)
        pred_u = pred['u'].squeeze() if pred['u'].ndim == 3 else pred['u']
        pred_v = pred['v'].squeeze() if pred['v'].ndim == 3 else pred['v']

        # 湍動能 (TKE)
        pred_tke = 0.5 * (pred_u**2 + pred_v**2)
        ref_tke = 0.5 * (ref['u']**2 + ref['v']**2)

        # 沿 x 方向 1D FFT
        nx = pred_tke.shape[0]
        pred_fft_x = np.fft.fft(pred_tke, axis=0)
        ref_fft_x = np.fft.fft(ref_tke, axis=0)

        # 能量譜：對 y 方向平均
        pred_spectrum_1d = np.mean(np.abs(pred_fft_x)**2, axis=1)
        ref_spectrum_1d = np.mean(np.abs(ref_fft_x)**2, axis=1)

        # 波數（流向）
        dx = pred['x'][1] - pred['x'][0]
        k_x = np.fft.fftfreq(nx, d=dx)

        # 僅保留正頻率
        positive_freq = k_x >= 0
        k_x = k_x[positive_freq]
        pred_spectrum_1d = pred_spectrum_1d[positive_freq]
        ref_spectrum_1d = ref_spectrum_1d[positive_freq]

    else:
        # 3D: (nx, ny, nz)
        # 湍動能
        pred_tke = 0.5 * (pred['u']**2 + pred['v']**2 + pred['w']**2)
        ref_tke = 0.5 * (ref['u']**2 + ref['v']**2 + ref['w']**2)

        # 沿 x 方向 1D FFT
        nx = pred_tke.shape[0]
        pred_fft_x = np.fft.fft(pred_tke, axis=0)
        ref_fft_x = np.fft.fft(ref_tke, axis=0)

        # 能量譜：對 y, z 方向平均
        pred_spectrum_1d = np.mean(np.abs(pred_fft_x)**2, axis=(1, 2))
        ref_spectrum_1d = np.mean(np.abs(ref_fft_x)**2, axis=(1, 2))

        # 波數（流向）
        dx = pred['x'][1] - pred['x'][0]
        k_x = np.fft.fftfreq(nx, d=dx)

        # 僅保留正頻率
        positive_freq = k_x >= 0
        k_x = k_x[positive_freq]
        pred_spectrum_1d = pred_spectrum_1d[positive_freq]
        ref_spectrum_1d = ref_spectrum_1d[positive_freq]

    # 計算誤差指標
    spectrum_rmse = np.sqrt(np.mean((pred_spectrum_1d - ref_spectrum_1d)**2))
    spectrum_rel_error = spectrum_rmse / (np.mean(ref_spectrum_1d) + 1e-12)

    logger.info(f"✅ Streamwise 1D spectrum: RMSE={spectrum_rmse:.2e}, rel_error={spectrum_rel_error:.2%}")

    return {
        'k': k_x,
        'pred_spectrum': pred_spectrum_1d,
        'ref_spectrum': ref_spectrum_1d,
        'spectrum_rmse': spectrum_rmse,
        'spectrum_rel_error': spectrum_rel_error,
        'spectrum_type': 'streamwise_1d'
    }


def _compute_radial_spectrum(pred: Dict[str, np.ndarray],
                              ref: Dict[str, np.ndarray],
                              is_2d: bool) -> Dict[str, np.ndarray]:
    """
    計算徑向平均2D能譜（傳統方法，僅適用於均勻各向同性湍流）

    ⚠️ 警告：此方法不適用於通道流（非均勻剪切湍流）
    """
    if is_2d:
        pred_u = pred['u'].squeeze() if pred['u'].ndim == 3 else pred['u']
        pred_v = pred['v'].squeeze() if pred['v'].ndim == 3 else pred['v']
        pred_ke = 0.5 * (pred_u**2 + pred_v**2)
        ref_ke = 0.5 * (ref['u']**2 + ref['v']**2)
    else:
        y_mid = len(pred['y']) // 2
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

    logger.info(f"✅ Radial 2D spectrum: RMSE={spectrum_rmse:.2e}, rel_error={spectrum_rel_error:.2%}")

    return {
        'k': k_bins,
        'pred_spectrum': pred_radial,
        'ref_spectrum': ref_radial,
        'spectrum_rmse': spectrum_rmse,
        'spectrum_rel_error': spectrum_rel_error,
        'spectrum_type': 'radial_2d'
    }


# ============================================================
# 可視化函數
# ============================================================

def plot_error_distribution(pred: Dict[str, np.ndarray], 
                            ref: Dict[str, np.ndarray], 
                            save_dir: Path):
    """繪製誤差分布圖（支援 2D/3D）"""
    logger.info("🎨 Plotting error distribution...")
    
    # 檢測 2D vs 3D
    is_2d = len(ref['u'].shape) == 2
    
    # 決定要繪製的場（僅繪製參考資料中存在的場）
    available_fields = [f for f in ['u', 'v', 'w', 'p'] if f in ref]
    n_fields = len(available_fields)
    n_rows = (n_fields + 1) // 2
    n_cols = 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 7*n_rows))
    if n_fields == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # 提取切片
    if is_2d:
        # 2D: squeeze pred
        def get_slice(data_dict, field):
            if field not in data_dict:
                return None
            d = data_dict[field]
            return d.squeeze() if d.ndim == 3 else d
        z_label = "2D slice"
    else:
        # 3D: 選擇中間 z 平面
        z_mid = len(pred['z']) // 2
        def get_slice(data_dict, field):
            if field not in data_dict:
                return None
            return data_dict[field][:, :, z_mid]
        z_label = f"z={pred['z'][z_mid]:.2f}"
    
    for idx, field in enumerate(available_fields):
        ax = axes[idx]
        
        pred_slice = get_slice(pred, field)
        ref_slice = get_slice(ref, field)
        
        if pred_slice is None or ref_slice is None:
            ax.text(0.5, 0.5, f'{field.upper()} not available', 
                   ha='center', va='center', fontsize=16)
            ax.axis('off')
            continue
        
        error = np.abs(pred_slice - ref_slice)
        
        im = ax.contourf(pred['x'], pred['y'], error.T, levels=20, cmap='hot')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'{field.upper()} Absolute Error ({z_label})', fontsize=16)
        plt.colorbar(im, ax=ax, label='|pred - ref|')
    
    # 隱藏多餘的子圖
    for idx in range(n_fields, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'error_distribution.png', dpi=150, bbox_inches='tight')
    logger.info(f"✅ Saved error distribution to {save_dir / 'error_distribution.png'}")
    plt.close()


def plot_field_comparison(pred: Dict[str, np.ndarray], 
                          ref: Dict[str, np.ndarray], 
                          save_dir: Path):
    """繪製場比較圖（預測 vs 參考，支援 2D/3D）"""
    logger.info("🎨 Plotting field comparison...")
    
    # 檢測 2D vs 3D
    is_2d = len(ref['u'].shape) == 2
    
    # 決定要繪製的場
    available_fields = [f for f in ['u', 'v', 'w', 'p'] if f in ref]
    
    # 提取切片
    if is_2d:
        def get_slice(data_dict, field):
            if field not in data_dict:
                return None
            d = data_dict[field]
            return d.squeeze() if d.ndim == 3 else d
    else:
        z_mid = len(pred['z']) // 2
        def get_slice(data_dict, field):
            if field not in data_dict:
                return None
            return data_dict[field][:, :, z_mid]
    
    for field in available_fields:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        
        pred_slice = get_slice(pred, field)
        ref_slice = get_slice(ref, field)
        
        if pred_slice is None or ref_slice is None:
            logger.warning(f"⚠️ Skipping {field} - not available in data")
            plt.close()
            continue
        
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
    """繪製速度剖面比較（支援 2D/3D）"""
    logger.info("🎨 Plotting velocity profiles...")
    
    # 檢測 2D vs 3D
    is_2d = len(ref['u'].shape) == 2
    
    # 決定要繪製的速度分量
    available_fields = [f for f in ['u', 'v', 'w'] if f in ref]
    n_fields = len(available_fields)
    
    # 選擇域中心
    x_mid = len(pred['x']) // 2
    
    fig, axes = plt.subplots(1, n_fields, figsize=(6*n_fields, 5))
    if n_fields == 1:
        axes = [axes]
    
    for idx, field in enumerate(available_fields):
        ax = axes[idx]
        
        if is_2d:
            # 2D: squeeze pred
            pred_field = pred[field].squeeze() if pred[field].ndim == 3 else pred[field]
            pred_profile = pred_field[x_mid, :]
            ref_profile = ref[field][x_mid, :]
            title_suffix = f"x={pred['x'][x_mid]:.2f}"
        else:
            # 3D: 使用中間 z
            z_mid = len(pred['z']) // 2
            pred_profile = pred[field][x_mid, :, z_mid]
            ref_profile = ref[field][x_mid, :, z_mid]
            title_suffix = f"x={pred['x'][x_mid]:.2f}, z={pred['z'][z_mid]:.2f}"
        
        ax.plot(pred_profile, pred['y'], 'b-', linewidth=2, label='Predicted')
        ax.plot(ref_profile, ref['y'], 'r--', linewidth=2, label='JHTDB Reference')
        
        ax.set_xlabel(f'{field}', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.set_title(f'{field.upper()} Velocity Profile ({title_suffix})', fontsize=16)
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
    
    # 檢測維度：2D (nx, ny) 或 3D (nx, ny, nz)
    is_2d = len(ref['u'].shape) == 2
    
    # Squeeze pred data if needed (預測總是3D，但參考可能是2D)
    pred_u = pred['u'].squeeze()
    
    # 計算壁面剪應力
    dy = pred['y'][1] - pred['y'][0]
    
    if is_2d:
        # 2D: shape (nx, ny) -> tau shape (nx,)
        pred_tau = (pred_u[:, 1] - pred_u[:, 0]) / dy
        ref_tau = (ref['u'][:, 1] - ref['u'][:, 0]) / dy
        
        # 2D 繪製：線圖
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 預測 vs 參考
        axes[0].plot(pred['x'], pred_tau, 'b-', linewidth=2, label='Predicted')
        axes[0].plot(ref['x'], ref_tau, 'r--', linewidth=2, label='Reference')
        axes[0].set_title('Wall Shear Stress Comparison', fontsize=16)
        axes[0].set_xlabel('x', fontsize=14)
        axes[0].set_ylabel(r'$\tau_w$', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # 誤差
        error = pred_tau - ref_tau
        axes[1].plot(pred['x'], error, 'k-', linewidth=2)
        axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1].set_title('Wall Shear Stress Error', fontsize=16)
        axes[1].set_xlabel('x', fontsize=14)
        axes[1].set_ylabel(r'$\Delta\tau_w$', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        
    else:
        # 3D: shape (nx, ny, nz) -> tau shape (nx, nz)
        pred_tau = (pred_u[:, 1, :] - pred_u[:, 0, :]) / dy
        ref_tau = (ref['u'][:, 1, :] - ref['u'][:, 0, :]) / dy
        
        # 3D 繪製：等高線圖
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

### 💡 壓力梯度誤差（更準確的壓力場評估）

**說明**: 由於壓力場僅定義到任意常數，壓力梯度 ∇p 的比較比絕對值更具物理意義。

| 梯度分量 | 相對 L2 誤差 | RMSE | 預測均值 | 參考均值 | 標準差（預測/參考） |
|---------|-------------|------|---------|---------|------------------|
| **∂p/∂x** | {metrics.get('dpdx_l2_error', 0):.4f} | {metrics.get('dpdx_rmse', 0):.6f} | {metrics.get('dpdx_pred_mean', 0):.6f} | {metrics.get('dpdx_ref_mean', 0):.6f} | {metrics.get('dpdx_pred_std', 0):.6f} / {metrics.get('dpdx_ref_std', 0):.6f} |
| **∂p/∂y** | {metrics.get('dpdy_l2_error', 0):.4f} | {metrics.get('dpdy_rmse', 0):.6f} | {metrics.get('dpdy_pred_mean', 0):.6f} | {metrics.get('dpdy_ref_mean', 0):.6f} | {metrics.get('dpdy_pred_std', 0):.6f} / {metrics.get('dpdy_ref_std', 0):.6f} |
| **∂p/∂z** | {metrics.get('dpdz_l2_error', 0):.4f} | {metrics.get('dpdz_rmse', 0):.6f} | {metrics.get('dpdz_pred_mean', 0):.6f} | {metrics.get('dpdz_ref_mean', 0):.6f} | {metrics.get('dpdz_pred_std', 0):.6f} / {metrics.get('dpdz_ref_std', 0):.6f} |
| **綜合梯度** | **{metrics.get('pressure_gradient_l2_error', 0):.4f}** | - | - | - | - |

**通道流驗證**: ∂p/∂x 應為常數（驅動流動），標準差應接近零。

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

**理論公式**: τ_w = μ·(∂u/∂y)|_wall，其中 μ = ρ·ν = 5×10⁻⁵

**理論值 (JHTDB Re_τ=1000)**: τ_w ≈ 0.0025，可接受範圍: [0.0020, 0.0030] (±20%)

| 位置 | 預測值 | JHTDB 參考 | 相對誤差 |
|------|--------|-----------|---------|
| **下壁面** (y=-1) | {wall_metrics.get('pred_tau_lower_mean', 0):.6f} | {wall_metrics.get('ref_tau_lower_mean', 0):.6f} | {wall_metrics.get('tau_lower_rel_error', 0):.2%} |
| **上壁面** (y=+1) | {wall_metrics.get('pred_tau_upper_mean', 0):.6f} | {wall_metrics.get('ref_tau_upper_mean', 0):.6f} | {wall_metrics.get('tau_upper_rel_error', 0):.2%} |
| **理論值** | 0.0025 | 0.0025 | **{wall_metrics.get('pred_vs_theory_error', 0):.2%}** |

| 統計量 | 下壁面 Std | 上壁面 Std | 對稱性誤差 |
|--------|-----------|-----------|-----------|
| 預測值 | {wall_metrics.get('pred_tau_lower_std', 0):.6f} | {wall_metrics.get('pred_tau_upper_std', 0):.6f} | {wall_metrics.get('symmetry_error_pred', 0):.2%} |
| 參考值 | {wall_metrics.get('ref_tau_lower_std', 0):.6f} | {wall_metrics.get('ref_tau_upper_std', 0):.6f} | {wall_metrics.get('symmetry_error_ref', 0):.2%} |

**對稱性檢查**: 通道流應上下對稱，對稱性誤差應 < 15%

### 能量譜

**計算方法**: {spectrum_data.get('spectrum_type', 'streamwise_1d')}
- `streamwise_1d`: 流向1D能譜（推薦，適用於通道流）
- `radial_2d`: 徑向平均2D能譜（僅適用於均勻各向同性湍流）

**評估指標**:
- **譜 RMSE**: {spectrum_data.get('spectrum_rmse', 0):.6e}
- **譜相對誤差**: {spectrum_data.get('spectrum_rel_error', 0):.2%}
- **波數範圍**: k ∈ [{spectrum_data['k'][0]:.2f}, {spectrum_data['k'][-1]:.2f}]

**說明**: 通道流是非均勻的剪切湍流，應使用流向1D能譜而非徑向平均2D能譜。

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
    model, physics = load_trained_model(Path(args.checkpoint), config, device)
    ref_data = load_jhtdb_reference(Path(args.reference))
    
    # 檢測 2D 或 3D
    is_3d = 'z' in ref_data and 'w' in ref_data
    
    # ========== 預測 ==========
    # 🆕 傳遞 physics 模組以使用 VS-PINN 縮放
    if is_3d:
        pred_data = predict_on_grid(
            model, 
            ref_data['x'], 
            ref_data['y'], 
            ref_data['z'], 
            device, 
            batch_size=args.batch_size,
            physics=physics,  # 🆕 傳遞 physics
            config=config,    # ✅ TASK-008: 傳遞 config 用於反標準化
            checkpoint_path=args.checkpoint  # ✅ TASK-008: 傳遞 checkpoint 路徑
        )
    else:
        # 2D slice: 使用固定 z 值（從配置或參考資料推斷）
        if isinstance(ref_data.get('slice_position'), np.ndarray):
            z_fixed_val = float(ref_data['slice_position'])
        elif 'slice_position' in ref_data:
            z_fixed_val = float(ref_data['slice_position'])
        else:
            z_fixed_val = 4.71  # 默認 z=π/2
        
        z_fixed = np.array([z_fixed_val])
        logger.info(f"📐 2D slice detected, using fixed z={z_fixed_val:.3f}")
        pred_data = predict_on_grid(
            model, 
            ref_data['x'], 
            ref_data['y'], 
            z_fixed,  # 單一 z 值
            device, 
            batch_size=args.batch_size,
            physics=physics,
            config=config,
            checkpoint_path=args.checkpoint  # ✅ TASK-008: 傳遞 checkpoint 路徑
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

    # 🔴 修正: 從配置中讀取黏性係數
    nu = config.get('physics', {}).get('nu', 5e-5)
    if 'physics' in config and 'channel_flow' in config['physics']:
        # 優先使用 channel_flow 配置中的值
        nu = config['physics']['channel_flow'].get('nu', nu)
    rho = config.get('physics', {}).get('rho', 1.0)

    wall_metrics = compute_wall_shear_stress_comparison(
        pred_data,
        ref_data,
        nu=nu,
        rho=rho
    )
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

    # 🆕 壓力梯度誤差摘要
    if 'pressure_gradient_l2_error' in error_metrics and error_metrics['pressure_gradient_l2_error'] != float('inf'):
        logger.info(f"💡 壓力梯度誤差: {error_metrics['pressure_gradient_l2_error']:.2%} (更準確的壓力評估)")
        if 'dpdx_pred_mean' in error_metrics:
            logger.info(f"   ∂p/∂x: 預測={error_metrics['dpdx_pred_mean']:.6f}, "
                       f"參考={error_metrics['dpdx_ref_mean']:.6f}")

    logger.info(f"📁 結果保存於: {output_dir}")
    logger.info("="*60)

    # 檢查成敗
    if error_metrics.get('overall_l2_error', 1) <= 0.15:
        logger.info("✅ 成功！整體誤差低於 15% 門檻")
    else:
        logger.warning("⚠️  整體誤差超過 15% 門檻，建議進一步調優")


if __name__ == '__main__':
    main()
