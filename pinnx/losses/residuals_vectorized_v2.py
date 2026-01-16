"""
向量化 PDE 殘差計算（穩定版）

核心策略：
1. 在單一函數內完成所有梯度計算（避免計算圖釋放問題）
2. 使用 vmap（如果可用）或手動批處理
3. 確保所有中間變量保持計算圖

Author: Performance Optimization Team
Date: 2026-01-16
"""

import torch
from typing import Dict, Optional
from pinnx.losses.residuals import compute_gradients


def compute_all_gradients_2d(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    一次性計算 2D NS 方程所需的所有梯度
    
    優勢：
    - 所有梯度在同一個函數內計算，避免計算圖釋放
    - 減少函數調用開銷
    - 清晰的依賴關係管理
    
    Args:
        coords: [N, 2] (x, y)
        velocity: [N, 2] (u, v)
        pressure: [N]
        
    Returns:
        {
            'u_grad': [N, 2],   # ∂u/∂x, ∂u/∂y
            'v_grad': [N, 2],   # ∂v/∂x, ∂v/∂y
            'p_grad': [N, 2],   # ∂p/∂x, ∂p/∂y
            'u_lap': [N],       # ∇²u
            'v_lap': [N]        # ∇²v
        }
    """
    u = velocity[:, 0]
    v = velocity[:, 1]
    p = pressure
    
    # 步驟 1: 計算一階梯度（保留計算圖）
    u_grad = torch.autograd.grad(
        outputs=u.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]  # [N, 2]
    
    v_grad = torch.autograd.grad(
        outputs=v.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]  # [N, 2]
    
    p_grad = torch.autograd.grad(
        outputs=p.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]  # [N, 2]
    
    # 步驟 2: 計算二階梯度（Laplacian）
    # 對 u_grad 的每個分量再求導
    u_xx = torch.autograd.grad(
        outputs=u_grad[:, 0].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]  # ∂²u/∂x²
    
    u_yy = torch.autograd.grad(
        outputs=u_grad[:, 1].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]  # ∂²u/∂y²
    
    u_lap = u_xx + u_yy  # ∇²u
    
    # 對 v_grad 的每個分量再求導
    v_xx = torch.autograd.grad(
        outputs=v_grad[:, 0].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]  # ∂²v/∂x²
    
    v_yy = torch.autograd.grad(
        outputs=v_grad[:, 1].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True  # ⚠️ 改為 True！外部的 backward 還需要計算圖
    )[0][:, 1]  # ∂²v/∂y²
    
    v_lap = v_xx + v_yy  # ∇²v
    
    return {
        'u_grad': u_grad,
        'v_grad': v_grad,
        'p_grad': p_grad,
        'u_lap': u_lap,
        'v_lap': v_lap
    }


def ns_residual_2d_vectorized(
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
    2D Navier-Stokes 殘差（向量化版本）
    
    與 residuals.py 中的 ns_residual_2d 功能完全相同，但使用優化的梯度計算
    
    Args:
        coords: [N, 2] 空間座標 (x, y)
        velocity: [N, 2] 速度場 (u, v)
        pressure: [N] 壓力場
        source: [N, 2] 源項（可選）
        nu: 運動黏度係數
        nu_t: [N] 湍流黏度（可選，RANS 模式）
        density: 流體密度
        time_coords: [N, 1] 時間座標（非定常情況）
        merge_momentum: 是否合併動量方程（向量形式）
        
    Returns:
        residuals: {
            'momentum': [N, 2] or 'momentum_x', 'momentum_y': [N]
            'continuity': [N]
        }
    """
    batch_size = coords.shape[0]
    
    u, v = velocity[:, 0], velocity[:, 1]
    p = pressure
    
    # 🚀 核心優化：一次性計算所有梯度
    grads = compute_all_gradients_2d(coords, velocity, pressure)
    
    u_grad = grads['u_grad']
    v_grad = grads['v_grad']
    p_grad = grads['p_grad']
    u_lap = grads['u_lap']
    v_lap = grads['v_lap']
    
    # 提取梯度分量
    ux, uy = u_grad[:, 0], u_grad[:, 1]
    vx, vy = v_grad[:, 0], v_grad[:, 1]
    px, py = p_grad[:, 0], p_grad[:, 1]
    
    # 時間導數（非定常情況）
    if time_coords is not None:
        time_coords.requires_grad_(True)
        u_t = torch.autograd.grad(
            outputs=u.sum(),
            inputs=time_coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
        v_t = torch.autograd.grad(
            outputs=v.sum(),
            inputs=time_coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
    else:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
    
    # 源項
    if source is not None:
        sx, sy = source[:, 0], source[:, 1]
    else:
        sx = torch.zeros_like(u)
        sy = torch.zeros_like(v)
    
    # 黏度項（考慮湍流黏度）
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
    
    # 連續性方程殘差（不可壓縮）
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


def compute_all_gradients_3d(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    一次性計算 3D NS 方程所需的所有梯度
    
    Args:
        coords: [N, 3] (x, y, z)
        velocity: [N, 3] (u, v, w)
        pressure: [N]
        
    Returns:
        {
            'u_grad': [N, 3],
            'v_grad': [N, 3],
            'w_grad': [N, 3],
            'p_grad': [N, 3],
            'u_lap': [N],
            'v_lap': [N],
            'w_lap': [N]
        }
    """
    u = velocity[:, 0]
    v = velocity[:, 1]
    w = velocity[:, 2]
    p = pressure
    
    # 一階梯度
    u_grad = torch.autograd.grad(
        outputs=u.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    v_grad = torch.autograd.grad(
        outputs=v.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    w_grad = torch.autograd.grad(
        outputs=w.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    p_grad = torch.autograd.grad(
        outputs=p.sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 二階梯度（Laplacian）
    # u
    u_xx = torch.autograd.grad(
        outputs=u_grad[:, 0].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    u_yy = torch.autograd.grad(
        outputs=u_grad[:, 1].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    u_zz = torch.autograd.grad(
        outputs=u_grad[:, 2].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 2]
    
    u_lap = u_xx + u_yy + u_zz
    
    # v
    v_xx = torch.autograd.grad(
        outputs=v_grad[:, 0].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    v_yy = torch.autograd.grad(
        outputs=v_grad[:, 1].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    v_zz = torch.autograd.grad(
        outputs=v_grad[:, 2].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 2]
    
    v_lap = v_xx + v_yy + v_zz
    
    # w
    w_xx = torch.autograd.grad(
        outputs=w_grad[:, 0].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 0]
    
    w_yy = torch.autograd.grad(
        outputs=w_grad[:, 1].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True
    )[0][:, 1]
    
    w_zz = torch.autograd.grad(
        outputs=w_grad[:, 2].sum(),
        inputs=coords,
        create_graph=True,
        retain_graph=True  # ⚠️ 改為 True
    )[0][:, 2]
    
    w_lap = w_xx + w_yy + w_zz
    
    return {
        'u_grad': u_grad,
        'v_grad': v_grad,
        'w_grad': w_grad,
        'p_grad': p_grad,
        'u_lap': u_lap,
        'v_lap': v_lap,
        'w_lap': w_lap
    }


def ns_residual_3d_vectorized(
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
    3D Navier-Stokes 殘差（向量化版本）
    
    與 residuals.py 中的 ns_residual_3d 功能完全相同
    """
    batch_size = coords.shape[0]
    
    u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    p = pressure
    
    # 一次性計算所有梯度
    grads = compute_all_gradients_3d(coords, velocity, pressure)
    
    u_grad = grads['u_grad']
    v_grad = grads['v_grad']
    w_grad = grads['w_grad']
    p_grad = grads['p_grad']
    u_lap = grads['u_lap']
    v_lap = grads['v_lap']
    w_lap = grads['w_lap']
    
    # 提取梯度分量
    ux, uy, uz = u_grad[:, 0], u_grad[:, 1], u_grad[:, 2]
    vx, vy, vz = v_grad[:, 0], v_grad[:, 1], v_grad[:, 2]
    wx, wy, wz = w_grad[:, 0], w_grad[:, 1], w_grad[:, 2]
    px, py, pz = p_grad[:, 0], p_grad[:, 1], p_grad[:, 2]
    
    # 時間導數
    if time_coords is not None:
        time_coords.requires_grad_(True)
        u_t = torch.autograd.grad(outputs=u.sum(), inputs=time_coords, create_graph=True, retain_graph=True)[0][:, 0]
        v_t = torch.autograd.grad(outputs=v.sum(), inputs=time_coords, create_graph=True, retain_graph=True)[0][:, 0]
        w_t = torch.autograd.grad(outputs=w.sum(), inputs=time_coords, create_graph=True, retain_graph=True)[0][:, 0]
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
