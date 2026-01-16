"""
簡化版向量化 PDE 殘差計算

策略：不復用梯度，只是組織計算順序來減少開銷
適用於 2D 和 3D Navier-Stokes

Author: Performance Optimization Team  
Date: 2026-01-16
"""

import torch
from typing import Dict, Optional
from pinnx.losses.residuals import compute_gradients, laplacian


def ns_residual_2d_vectorized_simple(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor,
    source: Optional[torch.Tensor] = None,
    nu: float = 1e-3,
    nu_t: Optional[torch.Tensor] = None,
    density: float = 1.0,
    time_coords: Optional[torch.Tensor] = None,
    merge_momentum: bool = True
) -> Dict[str, torch.Tensor]:
    """
    簡化版 2D N-S 殘差（使用標準梯度計算，但優化計算順序）
    
    優化點：
    1. 一次計算所有需要的一階梯度
    2. 按需計算二階梯度
    3. 避免重複計算
    
    Args:
        與 residuals.py 中的 ns_residual_2d 完全相同
        
    Returns:
        與原始版本相同的格式
    """
    batch_size = coords.shape[0]
    
    u, v = velocity[:, 0], velocity[:, 1]
    p = pressure
    
    # 🚀 優化1: 批量計算所有一階梯度
    u_grad = compute_gradients(u, coords, order=1, create_graph=True)
    v_grad = compute_gradients(v, coords, order=1, create_graph=True)
    p_grad = compute_gradients(p, coords, order=1, create_graph=False)  # 壓力梯度不需要二階
    
    ux, uy = u_grad[:, 0], u_grad[:, 1]
    vx, vy = v_grad[:, 0], v_grad[:, 1]
    px, py = p_grad[:, 0], p_grad[:, 1]
    
    # 🚀 優化2: 只在需要時計算 Laplacian
    u_lap = laplacian(u, coords)
    v_lap = laplacian(v, coords)
    
    # 時間導數（如果是非定常）
    if time_coords is not None:
        time_coords.requires_grad_(True)
        u_t = compute_gradients(u, time_coords, order=1)[:, 0]
        v_t = compute_gradients(v, time_coords, order=1)[:, 0]
    else:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
    
    # 源項
    if source is not None:
        sx, sy = source[:, 0], source[:, 1]
    else:
        sx = torch.zeros_like(u)
        sy = torch.zeros_like(v)
    
    # 黏度項
    if nu_t is not None:
        nu_eff = nu + nu_t
        u_lap_term = nu_eff * u_lap
        v_lap_term = nu_eff * v_lap
    else:
        u_lap_term = nu * u_lap
        v_lap_term = nu * v_lap
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy) + px / density - u_lap_term - sx
    momentum_y = v_t + (u * vx + v * vy) + py / density - v_lap_term - sy
    
    # 連續性方程殘差
    continuity = ux + vy
    
    # 返回格式
    if merge_momentum:
        momentum_vector = torch.stack([momentum_x, momentum_y], dim=-1)
        residuals = {
            'momentum': momentum_vector,
            'continuity': continuity
        }
    else:
        residuals = {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'continuity': continuity
        }
    
    return residuals


def ns_residual_3d_vectorized_simple(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor,
    source: Optional[torch.Tensor] = None,
    nu: float = 1e-3,
    nu_t: Optional[torch.Tensor] = None,
    density: float = 1.0,
    time_coords: Optional[torch.Tensor] = None,
    merge_momentum: bool = False
) -> Dict[str, torch.Tensor]:
    """
    簡化版 3D N-S 殘差
    
    與 2D 版本相同的優化策略
    """
    batch_size = coords.shape[0]
    
    u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    p = pressure
    
    # 批量計算一階梯度
    u_grad = compute_gradients(u, coords, order=1, create_graph=True)
    v_grad = compute_gradients(v, coords, order=1, create_graph=True)
    w_grad = compute_gradients(w, coords, order=1, create_graph=True)
    p_grad = compute_gradients(p, coords, order=1, create_graph=False)
    
    ux, uy, uz = u_grad[:, 0], u_grad[:, 1], u_grad[:, 2]
    vx, vy, vz = v_grad[:, 0], v_grad[:, 1], v_grad[:, 2]
    wx, wy, wz = w_grad[:, 0], w_grad[:, 1], w_grad[:, 2]
    px, py, pz = p_grad[:, 0], p_grad[:, 1], p_grad[:, 2]
    
    # Laplacian
    u_lap = laplacian(u, coords)
    v_lap = laplacian(v, coords)
    w_lap = laplacian(w, coords)
    
    # 時間導數
    if time_coords is not None:
        time_coords.requires_grad_(True)
        u_t = compute_gradients(u, time_coords, order=1)[:, 0]
        v_t = compute_gradients(v, time_coords, order=1)[:, 0]
        w_t = compute_gradients(w, time_coords, order=1)[:, 0]
    else:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
        w_t = torch.zeros_like(w)
    
    # 源項
    if source is not None:
        sx, sy, sz = source[:, 0], source[:, 1], source[:, 2]
    else:
        sx = torch.zeros_like(u)
        sy = torch.zeros_like(v)
        sz = torch.zeros_like(w)
    
    # 黏度項
    if nu_t is not None:
        nu_eff = nu + nu_t
        u_lap_term = nu_eff * u_lap
        v_lap_term = nu_eff * v_lap
        w_lap_term = nu_eff * w_lap
    else:
        u_lap_term = nu * u_lap
        v_lap_term = nu * v_lap
        w_lap_term = nu * w_lap
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy + w * uz) + px / density - u_lap_term - sx
    momentum_y = v_t + (u * vx + v * vy + w * vz) + py / density - v_lap_term - sy
    momentum_z = w_t + (u * wx + v * wy + w * wz) + pz / density - w_lap_term - sz
    
    # 連續性方程殘差
    continuity = ux + vy + wz
    
    # 返回格式
    if merge_momentum:
        momentum_vector = torch.stack([momentum_x, momentum_y, momentum_z], dim=-1)
        residuals = {
            'momentum': momentum_vector,
            'continuity': continuity
        }
    else:
        residuals = {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'momentum_z': momentum_z,
            'continuity': continuity
        }
    
    return residuals
