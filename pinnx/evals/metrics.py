"""
PINNs 評估指標模組

本模組提供完整的物理資訊神經網路評估指標，包括：
- 數值精度指標 (L2, RMSE)
- 物理一致性指標 (守恆律、邊界條件)  
- 流體動力學專用指標 (能譜、壁剪應力)
- 稀疏重建指標 (K-誤差曲線)
- 不確定性量化指標 (UQ可信度)

使用範例:
    >>> import torch
    >>> from pinnx.evals.metrics import relative_L2, conservation_error
    >>> 
    >>> # 計算相對L2誤差
    >>> pred = torch.randn(100, 3)  # [u, v, p]
    >>> ref = torch.randn(100, 3)
    >>> error = relative_L2(pred, ref)
    >>> 
    >>> # 檢查質量守恆
    >>> u, v = pred[:, 0], pred[:, 1]
    >>> coords = torch.randn(100, 3)  # [t, x, y]
    >>> mass_error = conservation_error(u, v, coords)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import warnings


def relative_L2(pred: torch.Tensor, ref: torch.Tensor, 
                dim: Optional[int] = None, eps: float = 1e-12) -> torch.Tensor:
    """
    計算相對 L2 誤差
    
    Args:
        pred: 預測值 [N, ...] 
        ref: 參考值 [N, ...]
        dim: 計算維度，None表示全域計算
        eps: 數值穩定性常數
        
    Returns:
        相對 L2 誤差張量
    """
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    
    # 計算L2範數
    error_norm = torch.norm(pred - ref, p=2, dim=dim)
    ref_norm = torch.norm(ref, p=2, dim=dim)
    
    # 避免除零
    relative_error = error_norm / (ref_norm + eps)
    
    return relative_error


def rmse_metrics(pred: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    """
    計算 RMSE 及其變體
    
    Args:
        pred: 預測值 [N, D]
        ref: 參考值 [N, D]  
        
    Returns:
        包含各種RMSE指標的字典
    """
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs ref {ref.shape}")
    
    mse = F.mse_loss(pred, ref, reduction='mean')
    rmse = torch.sqrt(mse)
    
    # 逐變數RMSE
    var_rmse = torch.sqrt(F.mse_loss(pred, ref, reduction='none').mean(0))
    
    # 相對RMSE 
    ref_magnitude = torch.norm(ref, dim=0) / ref.shape[0]
    relative_rmse = var_rmse / (ref_magnitude + 1e-12)
    
    metrics = {
        'rmse_total': rmse.item(),
        'rmse_u': var_rmse[0].item() if var_rmse.numel() > 0 and var_rmse.dim() > 0 else (var_rmse.item() if var_rmse.numel() == 1 else 0.0),
        'rmse_v': var_rmse[1].item() if var_rmse.numel() > 1 and var_rmse.dim() > 0 else 0.0,
        'rmse_p': var_rmse[2].item() if var_rmse.numel() > 2 and var_rmse.dim() > 0 else 0.0,
        'relative_rmse_u': relative_rmse[0].item() if relative_rmse.numel() > 0 and relative_rmse.dim() > 0 else (relative_rmse.item() if relative_rmse.numel() == 1 else 0.0),
        'relative_rmse_v': relative_rmse[1].item() if relative_rmse.numel() > 1 and relative_rmse.dim() > 0 else 0.0,
        'relative_rmse_p': relative_rmse[2].item() if relative_rmse.numel() > 2 and relative_rmse.dim() > 0 else 0.0,
    }
    
    return metrics


def field_statistics(field: torch.Tensor) -> Dict[str, float]:
    """
    計算場的統計量
    
    Args:
        field: 場變數 [N, D] 或 [N, H, W, D]
        
    Returns:
        統計量字典
    """
    # 展平到 [N*H*W, D]
    if field.dim() > 2:
        field = field.view(-1, field.shape[-1])
    
    stats = {}
    for i in range(field.shape[1]):
        var = field[:, i]
        prefix = ['u', 'v', 'p', 'S'][i] if i < 4 else f'var_{i}'
        
        stats[f'{prefix}_mean'] = var.mean().item()
        stats[f'{prefix}_std'] = var.std().item()
        stats[f'{prefix}_min'] = var.min().item()
        stats[f'{prefix}_max'] = var.max().item()
        
        # 偏度和峰度 (簡化計算)
        centered = var - var.mean()
        var_variance = centered.pow(2).mean()
        if var_variance > 1e-12:
            stats[f'{prefix}_skewness'] = (centered.pow(3).mean() / var_variance.pow(1.5)).item()
            stats[f'{prefix}_kurtosis'] = (centered.pow(4).mean() / var_variance.pow(2)).item()
        else:
            stats[f'{prefix}_skewness'] = 0.0
            stats[f'{prefix}_kurtosis'] = 3.0
    
    return stats


def conservation_error(u: torch.Tensor, v: torch.Tensor, 
                      coords: torch.Tensor, eps: float = 1e-8) -> float:
    """
    計算質量守恆誤差 (∇·u = ∂u/∂x + ∂v/∂y = 0)
    
    Args:
        u: x方向速度 [N]
        v: y方向速度 [N]
        coords: 座標 [N, 2] (x, y) 或 [N, 3] (t, x, y)
        eps: 數值穩定性常數
        
    Returns:
        平均質量守恆誤差
    """
    # 確保 coords 需要梯度
    if not coords.requires_grad:
        coords.requires_grad_(True)
    
    # 確保 u, v 需要梯度且通過 coords 計算
    # 如果 u, v 沒有 grad_fn，需要重新計算它們與 coords 的關係
    if u.grad_fn is None or v.grad_fn is None:
        warnings.warn("u or v not connected to coords in computation graph, conservation error may be inaccurate")
    
    try:
        # 計算梯度
        u_grad = torch.autograd.grad(u.sum(), coords, create_graph=True, allow_unused=True)[0]
        v_grad = torch.autograd.grad(v.sum(), coords, create_graph=True, allow_unused=True)[0]
        
        if u_grad is None or v_grad is None:
            warnings.warn("Cannot compute gradients: u/v not connected to coords")
            return float('inf')
        
        # 質量守恆: ∂u/∂x + ∂v/∂y = 0
        # 根據座標維度確定空間索引
        if coords.shape[1] == 2:  # [x, y]
            div_u = u_grad[:, 0] + v_grad[:, 1]  # ∂u/∂x + ∂v/∂y
        elif coords.shape[1] == 3:  # [t, x, y] 或 [x, y, z]
            # 假設前導維度是時間，空間從索引1開始
            div_u = u_grad[:, 1] + v_grad[:, 2]  # ∂u/∂x + ∂v/∂y
        else:
            raise ValueError(f"Unsupported coords shape: {coords.shape}")
        
        # 計算RMS誤差
        conservation_error = torch.sqrt(torch.mean(div_u**2))
        
        return conservation_error.item()
        
    except RuntimeError as e:
        warnings.warn(f"Conservation calculation failed: {e}")
        return float('inf')


def boundary_compliance(pred: torch.Tensor, bc_values: torch.Tensor, 
                       bc_coords: torch.Tensor, tolerance: float = 1e-3) -> Dict[str, float]:
    """
    檢查邊界條件符合度
    
    Args:
        pred: 預測值在邊界點 [N_bc, D]
        bc_values: 邊界條件值 [N_bc, D]  
        bc_coords: 邊界座標 [N_bc, 3]
        tolerance: 允許誤差
        
    Returns:
        邊界符合度指標
    """
    if pred.shape != bc_values.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs bc_values {bc_values.shape}")
    
    error = torch.abs(pred - bc_values)
    max_error = error.max().item()
    mean_error = error.mean().item()
    
    # 符合度百分比
    compliance = (error < tolerance).float().mean().item() * 100
    
    return {
        'bc_max_error': max_error,
        'bc_mean_error': mean_error,
        'bc_compliance_pct': compliance,
        'bc_tolerance': tolerance
    }


def energy_spectrum_1d(field: torch.Tensor, domain_size: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算1D能量譜
    
    Args:
        field: 場變數 [H, W] 或 [N, H, W]
        domain_size: 計算域大小
        
    Returns:
        (波數, 能量譜)
    """
    if field.dim() == 3:
        field = field.mean(0)  # 對批次維度平均
    
    # 轉為numpy進行FFT
    field_np = field.detach().cpu().numpy()
    
    # 2D FFT
    fft_field = np.fft.fft2(field_np)
    fft_field = np.fft.fftshift(fft_field)
    
    # 計算能量密度
    energy_density = np.abs(fft_field)**2
    
    # 徑向平均
    h, w = field_np.shape
    center_h, center_w = h // 2, w // 2
    
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_w)**2 + (y - center_h)**2)
    
    # 波數網格
    k_max = min(center_h, center_w)
    k_bins = np.arange(1, k_max)
    k_spectrum = np.zeros(len(k_bins))
    
    for i, k in enumerate(k_bins):
        mask = (r >= k - 0.5) & (r < k + 0.5)
        if mask.sum() > 0:
            k_spectrum[i] = energy_density[mask].mean()
    
    # 正規化波數
    k_physical = k_bins * 2 * np.pi / domain_size
    
    return k_physical, k_spectrum


def energy_spectrum_2d(u: torch.Tensor, v: torch.Tensor, 
                       domain_size: float = 2*np.pi) -> Tuple[np.ndarray, np.ndarray]:
    """
    計算2D動能譜 E(k) = 0.5 * (|u_k|² + |v_k|²)
    
    Args:
        u: x方向速度 [H, W] 或 [N, H, W]
        v: y方向速度 [H, W] 或 [N, H, W] 
        domain_size: 計算域大小
        
    Returns:
        (波數, 動能譜)
    """
    if u.dim() == 3:
        u = u.mean(0)
        v = v.mean(0)
    
    # 計算動能場
    kinetic_energy = 0.5 * (u**2 + v**2)
    
    # 使用1D能譜函數
    k, spectrum = energy_spectrum_1d(kinetic_energy, domain_size)
    
    return k, spectrum


def wall_shear_stress(u: torch.Tensor, v: torch.Tensor, coords: torch.Tensor,
                     viscosity: float = 1.0, wall_normal: str = 'y') -> Dict[str, float]:
    """
    計算壁面剪應力 τ_w = μ * (∂u/∂y)|_wall
    
    Args:
        u: x方向速度 [N]
        v: y方向速度 [N]
        coords: 座標 [N, 3] (t, x, y)
        viscosity: 動黏性係數
        wall_normal: 壁面法向方向 ('x' 或 'y')
        
    Returns:
        壁面剪應力統計
    """
    if not u.requires_grad:
        u.requires_grad_(True)
    
    try:
        # 計算速度梯度
        u_grad = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        
        if wall_normal == 'y':
            # 下壁面: τ_w = μ * ∂u/∂y
            shear_rate = u_grad[:, 2]  # ∂u/∂y
        elif wall_normal == 'x':
            # 側壁面: τ_w = μ * ∂u/∂x  
            shear_rate = u_grad[:, 1]  # ∂u/∂x
        else:
            raise ValueError("wall_normal must be 'x' or 'y'")
        
        wall_shear = viscosity * shear_rate
        
        return {
            'tau_w_mean': wall_shear.mean().item(),
            'tau_w_std': wall_shear.std().item(),
            'tau_w_max': wall_shear.max().item(),
            'tau_w_min': wall_shear.min().item()
        }
        
    except RuntimeError as e:
        warnings.warn(f"Wall shear stress calculation failed: {e}")
        return {
            'tau_w_mean': 0.0,
            'tau_w_std': 0.0, 
            'tau_w_max': 0.0,
            'tau_w_min': 0.0
        }


def vorticity_field(u: torch.Tensor, v: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """
    計算渦量場 ω = ∂v/∂x - ∂u/∂y
    
    Args:
        u: x方向速度 [N]
        v: y方向速度 [N]
        coords: 座標 [N, 3] (t, x, y)
        
    Returns:
        渦量場 [N]
    """
    if not u.requires_grad:
        u.requires_grad_(True)
    if not v.requires_grad:
        v.requires_grad_(True)
    
    try:
        u_grad = torch.autograd.grad(u.sum(), coords, create_graph=True)[0]
        v_grad = torch.autograd.grad(v.sum(), coords, create_graph=True)[0]
        
        # ω = ∂v/∂x - ∂u/∂y
        vorticity = v_grad[:, 1] - u_grad[:, 2]
        
        return vorticity
        
    except RuntimeError as e:
        warnings.warn(f"Vorticity calculation failed: {e}")
        return torch.zeros_like(u)


def reconstruction_quality(pred: torch.Tensor, ref: torch.Tensor, 
                          sensor_coords: torch.Tensor) -> Dict[str, float]:
    """
    稀疏點重建品質評估
    
    Args:
        pred: 預測場 [N, D]
        ref: 參考場 [N, D]
        sensor_coords: 感測器座標 [K, 3]
        
    Returns:
        重建品質指標
    """
    # 計算全場誤差
    global_error = relative_L2(pred, ref).item()
    
    # 計算感測點密度
    total_points = pred.shape[0]
    sensor_points = sensor_coords.shape[0]
    sparsity = sensor_points / total_points
    
    # 計算重建效率 (誤差 vs 稀疏度)
    efficiency = (1 - global_error) / max(sparsity, 1e-6)
    
    return {
        'global_l2_error': global_error,
        'sensor_density': sparsity,
        'reconstruction_efficiency': efficiency,
        'sensor_count': sensor_points,
        'total_points': total_points
    }


def k_error_curve(predictions: List[torch.Tensor], references: List[torch.Tensor],
                  k_values: List[int]) -> Dict[str, Union[List[float], float]]:
    """
    計算K-誤差曲線 (感測器數量 vs 重建誤差)
    
    Args:
        predictions: 不同K值的預測結果列表
        references: 對應的參考解列表  
        k_values: 感測器數量列表
        
    Returns:
        K-誤差曲線資料
    """
    if len(predictions) != len(references) or len(predictions) != len(k_values):
        raise ValueError("All input lists must have same length")
    
    errors = []
    for pred, ref in zip(predictions, references):
        error = relative_L2(pred, ref).item()
        errors.append(error)
    
    return {
        'k_values': [float(k) for k in k_values],
        'l2_errors': errors,
        'best_k': float(k_values[np.argmin(errors)]),
        'min_error': float(min(errors))
    }


def uncertainty_correlation(ensemble_mean: torch.Tensor, ensemble_var: torch.Tensor,
                           true_error: torch.Tensor) -> Dict[str, float]:
    """
    不確定性量化可信度評估
    
    Args:
        ensemble_mean: 集成平均 [N, D]
        ensemble_var: 集成方差 [N, D]
        true_error: 真實誤差 [N, D]
        
    Returns:
        UQ可信度指標
    """
    # 展平為1D
    pred_std = torch.sqrt(ensemble_var).flatten()
    true_err = torch.abs(true_error).flatten()
    
    # 計算皮爾森相關係數
    def pearson_correlation(x, y):
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        numerator = (x_centered * y_centered).sum()
        denominator = torch.sqrt((x_centered**2).sum() * (y_centered**2).sum())
        return (numerator / (denominator + 1e-12)).item()
    
    correlation = pearson_correlation(pred_std, true_err)
    
    # 校準誤差 (預測標準差 vs 實際誤差的RMS差異)
    calibration_error = torch.sqrt(((pred_std - true_err)**2).mean()).item()
    
    return {
        'uq_correlation': correlation,
        'calibration_rmse': calibration_error,
        'mean_predicted_std': pred_std.mean().item(),
        'mean_true_error': true_err.mean().item()
    }


def prediction_interval_coverage(pred_mean: torch.Tensor, pred_std: torch.Tensor,
                               true_values: torch.Tensor, confidence: float = 0.95) -> float:
    """
    預測區間覆蓋率
    
    Args:
        pred_mean: 預測均值 [N, D]
        pred_std: 預測標準差 [N, D]
        true_values: 真實值 [N, D]
        confidence: 置信水準
        
    Returns:
        覆蓋率百分比
    """
    from scipy.stats import norm
    
    z_score = norm.ppf(0.5 + confidence / 2)
    
    lower = pred_mean - z_score * pred_std
    upper = pred_mean + z_score * pred_std
    
    coverage_mask = (true_values >= lower) & (true_values <= upper)
    coverage = torch.tensor(coverage_mask, dtype=torch.float32).mean()
    
    return float(coverage.item() * 100)


def training_efficiency(loss_history: List[float], time_history: List[float]) -> Dict[str, float]:
    """
    訓練效率分析
    
    Args:
        loss_history: 損失歷史 [epochs]
        time_history: 時間歷史 [epochs]
        
    Returns:
        效率指標
    """
    if len(loss_history) != len(time_history):
        raise ValueError("Loss and time histories must have same length")
    
    if len(loss_history) < 2:
        return {'convergence_rate': 0.0, 'time_per_epoch': 0.0, 'efficiency_score': 0.0}
    
    # 收斂速率 (log10(loss) per epoch)
    initial_loss = max(loss_history[0], 1e-12)
    final_loss = max(loss_history[-1], 1e-12)
    epochs = len(loss_history)
    
    convergence_rate = (np.log10(initial_loss) - np.log10(final_loss)) / epochs
    
    # 平均每epoch時間
    time_per_epoch = np.mean(time_history)
    
    # 效率評分 (收斂速率 / 時間)
    efficiency_score = convergence_rate / max(time_per_epoch, 1e-6)
    
    return {
        'convergence_rate': float(convergence_rate),
        'time_per_epoch': float(time_per_epoch),
        'efficiency_score': float(efficiency_score),
        'total_time': float(sum(time_history)),
        'final_loss': float(final_loss)
    }


def convergence_analysis(residuals: List[float], threshold: float = 1e-6) -> Dict[str, Union[int, float, bool]]:
    """
    收斂性分析
    
    Args:
        residuals: 殘差歷史
        threshold: 收斂閾值
        
    Returns:
        收斂分析結果
    """
    if not residuals:
        return {'converged': False, 'convergence_epoch': -1, 'final_residual': float('inf')}
    
    # 檢查是否收斂
    converged = residuals[-1] < threshold
    
    # 找到首次收斂的epoch
    convergence_epoch = -1
    for i, res in enumerate(residuals):
        if res < threshold:
            convergence_epoch = i
            break
    
    # 收斂穩定性 (最後10%epochs的變異係數)
    tail_length = max(1, len(residuals) // 10)
    tail_residuals = residuals[-tail_length:]
    stability = float(np.std(tail_residuals) / (np.mean(tail_residuals) + 1e-12))
    
    return {
        'converged': converged,
        'convergence_epoch': convergence_epoch,
        'final_residual': float(residuals[-1]),
        'stability_coefficient': stability,
        'monotonic_decrease': bool(all(residuals[i] >= residuals[i+1] for i in range(len(residuals)-1)))
    }


# 綜合評估函數
def comprehensive_evaluation(predictions: torch.Tensor, references: torch.Tensor,
                           coords: torch.Tensor, physics_params: Dict) -> Dict[str, float]:
    """
    綜合評估PINNs結果
    
    Args:
        predictions: 預測結果 [N, D] (u, v, p, ...)
        references: 參考解 [N, D]
        coords: 座標 [N, 3] (t, x, y)
        physics_params: 物理參數 {'viscosity': float, ...}
        
    Returns:
        全面評估指標
    """
    results = {}
    
    # 基本精度指標
    results.update(rmse_metrics(predictions, references))
    results['relative_l2'] = relative_L2(predictions, references).item()
    
    # 物理一致性
    if predictions.shape[1] >= 2:  # 至少有u, v
        u_pred, v_pred = predictions[:, 0], predictions[:, 1]
        mass_error = conservation_error(u_pred, v_pred, coords)
        results['mass_conservation_error'] = mass_error
    
    # 場統計
    field_stats = field_statistics(predictions)
    results.update({f'pred_{k}': v for k, v in field_stats.items()})
    
    ref_stats = field_statistics(references) 
    results.update({f'ref_{k}': v for k, v in ref_stats.items()})
    
    return results