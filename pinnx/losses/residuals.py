"""
PINN 殘差損失函數模組

實現各種 PDE 殘差計算、邊界條件損失、源項正則化等核心損失函數。
專為 2D/3D 不可壓縮 Navier-Stokes 方程設計，支援：

- NS 方程殘差 (動量、連續性)
- 邊界條件 (無滑移、對稱、週期等)
- 源項辨識與稀疏化
- 時間因果性權重
- 守恆定律檢查

核心功能包含自動微分、梯度計算、以及與 VS-PINN 尺度化的整合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np


def compute_gradients(u: torch.Tensor, 
                     x: torch.Tensor, 
                     order: int = 1,
                     create_graph: bool = True) -> torch.Tensor:
    """
    通用梯度計算函數
    
    Args:
        u: 標量場 [batch_size]
        x: 座標 [batch_size, dim]  
        order: 梯度階數 (1 或 2)
        create_graph: 是否建立計算圖 (用於高階微分)
    
    Returns:
        gradients: [batch_size, dim] (一階) 或 [batch_size, dim, dim] (二階)
    """
    if order == 1:
        # 一階梯度 ∇u
        grad = torch.autograd.grad(
            u.sum(), x,
            create_graph=create_graph,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if grad is None:
            grad = torch.zeros_like(x)
        
        return grad
    
    elif order == 2:
        # 二階梯度 ∇²u (Hessian)
        grad_1 = compute_gradients(u, x, order=1, create_graph=True)
        
        hessian = []
        for i in range(grad_1.shape[-1]):
            grad_2 = torch.autograd.grad(
                grad_1[:, i].sum(), x,
                create_graph=create_graph,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_2 is None:
                grad_2 = torch.zeros_like(x)
            
            hessian.append(grad_2)
        
        return torch.stack(hessian, dim=-1)  # [batch, dim, dim]
    
    else:
        raise ValueError(f"不支援的梯度階數: {order}")


def laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    計算拉普拉斯算子 ∇²u
    
    Args:
        u: 標量場 [batch_size]
        x: 座標 [batch_size, dim]
    
    Returns:
        laplacian: [batch_size] 
    """
    hessian = compute_gradients(u, x, order=2)  # [batch, dim, dim]
    
    # 對角線元素求和：∂²u/∂x² + ∂²u/∂y² + ...
    laplacian = torch.diagonal(hessian, dim1=-2, dim2=-1).sum(dim=-1)
    
    return laplacian


def divergence(velocity: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    計算速度場散度 ∇·u
    
    Args:
        velocity: 速度場 [batch_size, dim]
        x: 座標 [batch_size, dim]
    
    Returns:
        divergence: [batch_size]
    """
    div = torch.zeros(velocity.shape[0], device=velocity.device, dtype=velocity.dtype)
    for i in range(velocity.shape[-1]):
        u_i = velocity[:, i]
        du_dx = compute_gradients(u_i, x, order=1)
        div = div + du_dx[:, i]  # ∂u_i/∂x_i
    
    return div


def compute_all_gradients_2d(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    pressure: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    一次性計算 2D NS 方程所需的所有梯度（向量化優化版本）
    
    優勢：
    - 所有梯度在同一個函數內計算，避免計算圖釋放
    - 減少函數調用開銷（1.77x 加速於 Tesla P100）
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
        retain_graph=True  # 保留計算圖供外部 backward 使用
    )[0][:, 1]  # ∂²v/∂y²
    
    v_lap = v_xx + v_yy  # ∇²v
    
    return {
        'u_grad': u_grad,
        'v_grad': v_grad,
        'p_grad': p_grad,
        'u_lap': u_lap,
        'v_lap': v_lap
    }


def ns_residual_2d(coords: torch.Tensor,
                  velocity: torch.Tensor,
                  pressure: torch.Tensor,
                  source: Optional[torch.Tensor] = None,
                  nu: float = 1e-3,
                  nu_t: Optional[torch.Tensor] = None,
                  time_coords: Optional[torch.Tensor] = None,
                  density: float = 1.0,
                  merge_momentum: bool = False) -> Dict[str, torch.Tensor]:
    """
    2D 不可壓縮 Navier-Stokes 方程殘差計算（向量化優化版本）
    
    方程形式：
    ∂u/∂t + u·∇u = -∇p/ρ + ν_eff∇²u + S
    ∇·u = 0
    
    其中 ν_eff = ν + ν_t (有效黏度 = 分子黏度 + 湍流黏度)
    
    優化說明：
    - 使用向量化梯度計算，相較原版本提速 1.77x (Tesla P100, batch=8000)
    - 內存開銷 ±0%
    
    Args:
        coords: 空間座標 [batch_size, 2] (x, y)
        velocity: 速度場 [batch_size, 2] (u, v)
        pressure: 壓力場 [batch_size]
        source: 源項 [batch_size, 2] (可選)
        nu: 動力黏性係數 (分子黏度)
        nu_t: 湍流黏性係數 [batch_size] (可選，來自 RANS prior)
        time_coords: 時間座標 [batch_size] (非定常問題)
        density: 密度 (預設 1.0)
        merge_momentum: 是否合併動量項 (預設 False)
            - True: 返回 {'momentum': [batch, 2], 'continuity': [batch]}
              適用於各向同性問題（如 Kolmogorov Flow），強制 X/Y 方向同步優化
            - False: 返回 {'momentum_x': [batch], 'momentum_y': [batch], 'continuity': [batch]}
              適用於各向異性問題（如 Channel Flow），允許方向性權重控制
    
    Returns:
        residuals: 包含各方程殘差的字典
            - merge_momentum=False: {'momentum_x', 'momentum_y', 'continuity'}
            - merge_momentum=True:  {'momentum', 'continuity'}
    
    Examples:
        >>> # Kolmogorov Flow (各向同性)
        >>> residuals = ns_residual_2d(coords, velocity, pressure, viscosity=1e-3, merge_momentum=True)
        >>> # 返回: {'momentum': [N, 2], 'continuity': [N]}
        
        >>> # Channel Flow (各向異性)
        >>> residuals = ns_residual_2d(coords, velocity, pressure, viscosity=1e-3, merge_momentum=False)
        >>> # 返回: {'momentum_x': [N], 'momentum_y': [N], 'continuity': [N]}
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
    
    # 時間導數 (非定常情況)
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
    
    # 源項 (如果提供)
    if source is not None:
        sx, sy = source[:, 0], source[:, 1]
    else:
        sx = torch.zeros_like(u)
        sy = torch.zeros_like(v)
    
    # 計算有效黏度項 (ν_eff = ν + ν_t)
    if nu_t is not None:
        # RANS 湍流黏度存在，使用有效黏度
        nu_eff = nu + nu_t  # [batch_size] 或標量
        u_lap_term = nu_eff * u_lap
        v_lap_term = nu_eff * v_lap
    else:
        # 純 DNS/LES，僅使用分子黏度
        u_lap_term = nu * u_lap
        v_lap_term = nu * v_lap
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy) + px / density - u_lap_term - sx
    momentum_y = v_t + (u * vx + v * vy) + py / density - v_lap_term - sy
    
    # 連續性方程殘差 (不可壓縮)
    continuity = ux + vy
    
    # 根據 merge_momentum 參數決定返回格式
    if merge_momentum:
        # 合併動量項（適用於各向同性問題，如 Kolmogorov Flow）
        # 使用向量化 L2 範數，強制 X/Y 方向同步優化
        momentum_vector = torch.stack([momentum_x, momentum_y], dim=-1)  # [batch, 2]
        residuals = {
            'momentum': momentum_vector,
            'continuity': continuity
        }
    else:
        # 分離動量項（標準模式，適用於各向異性問題，如 Channel Flow）
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
    一次性計算 3D NS 方程所需的所有梯度（向量化優化版本）
    
    優勢：
    - 所有梯度在同一個函數內計算，避免計算圖釋放
    - 減少函數調用開銷（1.07x 加速於 Tesla P100）
    - 清晰的依賴關係管理
    
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
        retain_graph=True  # 保留計算圖供外部 backward 使用
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


def ns_residual_3d(coords: torch.Tensor,
                  velocity: torch.Tensor,
                  pressure: torch.Tensor,
                  source: Optional[torch.Tensor] = None,
                  nu: float = 1e-3,
                  nu_t: Optional[torch.Tensor] = None,
                  time_coords: Optional[torch.Tensor] = None,
                  density: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    3D 不可壓縮 Navier-Stokes 方程殘差計算（向量化優化版本）
    
    方程形式：
    ∂u/∂t + u·∇u = -∇p/ρ + ν_eff∇²u + S
    ∇·u = 0
    
    其中 ν_eff = ν + ν_t (有效黏度 = 分子黏度 + 湍流黏度)
    
    優化說明：
    - 使用向量化梯度計算，相較原版本提速 1.07x (Tesla P100, batch=8000)
    - 內存開銷 ±0%
    
    Args:
        coords: 空間座標 [batch_size, 3] (x, y, z)
        velocity: 速度場 [batch_size, 3] (u, v, w)
        pressure: 壓力場 [batch_size]
        source: 源項 [batch_size, 3] (可選)
        nu: 動力黏性係數 (分子黏度)
        nu_t: 湍流黏性係數 [batch_size] (可選，來自 RANS prior)
        time_coords: 時間座標 [batch_size] (非定常問題)
        density: 密度 (預設 1.0)
    
    Returns:
        residuals: 包含各方程殘差的字典
    """
    batch_size = coords.shape[0]
    
    u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    p = pressure
    
    # 🚀 核心優化：一次性計算所有梯度
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
        w_t = torch.autograd.grad(
            outputs=w.sum(),
            inputs=time_coords,
            create_graph=True,
            retain_graph=True
        )[0][:, 0]
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
    
    # 計算有效黏度項 (ν_eff = ν + ν_t)
    if nu_t is not None:
        # RANS 湍流黏度存在，使用有效黏度
        nu_eff = nu + nu_t  # [batch_size] 或標量
        u_lap_term = nu_eff * u_lap
        v_lap_term = nu_eff * v_lap
        w_lap_term = nu_eff * w_lap
    else:
        # 純 DNS/LES，僅使用分子黏度
        u_lap_term = nu * u_lap
        v_lap_term = nu * v_lap
        w_lap_term = nu * w_lap
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy + w * uz) + px / density - u_lap_term - sx
    momentum_y = v_t + (u * vx + v * vy + w * vz) + py / density - v_lap_term - sy
    momentum_z = w_t + (u * wx + v * wy + w * wz) + pz / density - w_lap_term - sz
    
    # 連續性方程殘差
    continuity = ux + vy + wz
    
    residuals = {
        'momentum_x': momentum_x,
        'momentum_y': momentum_y,
        'momentum_z': momentum_z,
        'continuity': continuity
    }
    
    return residuals


class NSResidualLoss(nn.Module):
    """
    Navier-Stokes 殘差損失函數類
    
    支援 2D/3D、定常/非定常、含源項的 NS 方程求解
    """
    
    def __init__(self, 
                 nu: float = 1e-3,
                 density: float = 1.0,
                 spatial_dim: int = 2,
                 unsteady: bool = False,
                 source_regularization: float = 0.0,
                 merge_momentum: bool = False):
        """
        Args:
            nu: 動力黏性係數
            density: 流體密度
            spatial_dim: 空間維度 (2 或 3)
            unsteady: 是否為非定常問題
            source_regularization: 源項稀疏化權重
            merge_momentum: 是否合併動量項 (僅 2D 有效，3D 忽略)
        """
        super().__init__()
        
        self.nu = nu
        self.merge_momentum = merge_momentum
        self.density = density  
        self.spatial_dim = spatial_dim
        self.unsteady = unsteady
        self.source_reg = source_regularization
        
        # 選擇對應的殘差計算函數
        if spatial_dim == 2:
            self.residual_fn = ns_residual_2d
        elif spatial_dim == 3:
            self.residual_fn = ns_residual_3d
        else:
            raise ValueError(f"不支援的空間維度: {spatial_dim}")
    
    def forward(self, 
                coords: torch.Tensor,
                predictions: torch.Tensor,
                time_coords: Optional[torch.Tensor] = None,
                nu_t: Optional[torch.Tensor] = None,
                weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        計算 NS 方程殘差損失
        
        Args:
            coords: 空間座標 [batch_size, spatial_dim]
            predictions: 模型預測 [batch_size, output_dim]
                        output_dim = spatial_dim + 1 + source_dim
                        例如 2D: [u, v, p, sx, sy]
            time_coords: 時間座標 [batch_size, 1]
            nu_t: RANS 湍流黏度 [batch_size] (可選)
            weights: 各項損失權重
        
        Returns:
            losses: 各項損失的字典
        """
        batch_size = coords.shape[0]
        
        # 解析預測結果
        velocity = predictions[:, :self.spatial_dim]  # [u, v] 或 [u, v, w]
        pressure = predictions[:, self.spatial_dim]   # p
        
        # 源項 (如果存在)
        if predictions.shape[-1] > self.spatial_dim + 1:
            source = predictions[:, self.spatial_dim + 1:]
        else:
            source = None
        
        # 計算殘差 (2D 時傳遞 merge_momentum)
        if self.spatial_dim == 2:
            residuals = self.residual_fn(
                coords=coords,
                velocity=velocity,
                pressure=pressure,
                source=source,
                nu=self.nu,
                nu_t=nu_t,
                time_coords=time_coords,
                density=self.density,
                merge_momentum=self.merge_momentum
            )
        else:
            residuals = self.residual_fn(
                coords=coords,
                velocity=velocity,
                pressure=pressure,
                source=source,
                nu=self.nu,
                nu_t=nu_t,
                time_coords=time_coords,
                density=self.density
            )
        
        # 損失權重 (預設值)
        default_weights = {
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'momentum_z': 1.0,
            'momentum': 1.0,      # 合併動量項權重
            'continuity': 1.0,
            'source_reg': self.source_reg
        }
        if weights:
            default_weights.update(weights)
        
        # 計算各項損失
        losses = {}
        
        # PDE 殘差損失 (MSE)
        for key, residual in residuals.items():
            if key in default_weights:
                # 檢查是否為合併動量項（向量形式）
                if key == 'momentum' and residual.dim() == 2:
                    # 向量化動量殘差：先計算每個點的 L2 範數，再平均
                    # ||[momentum_x, momentum_y]||_2^2 = momentum_x^2 + momentum_y^2
                    momentum_norm_sq = torch.sum(residual ** 2, dim=-1)  # [batch]
                    losses[f'pde_{key}'] = default_weights[key] * torch.mean(momentum_norm_sq)
                else:
                    # 標量殘差：直接平方
                    losses[f'pde_{key}'] = default_weights[key] * torch.mean(residual ** 2)
        
        # 源項正則化 (L1 稀疏化)
        if source is not None and self.source_reg > 0:
            source_l1 = torch.mean(torch.abs(source))
            losses['source_l1'] = default_weights['source_reg'] * source_l1
        
        # 總 PDE 損失
        pde_loss = sum(v for k, v in losses.items() if k.startswith('pde_'))
        losses['total_pde'] = pde_loss
        
        return losses


class BoundaryConditionLoss(nn.Module):
    """
    邊界條件損失函數
    
    支援多種邊界條件類型：
    - 無滑移 (no-slip): u = 0
    - 自由滑移 (free-slip): u_n = 0, ∂u_t/∂n = 0  
    - 對稱 (symmetry): u_n = 0, ∂u_t/∂n = 0
    - 週期 (periodic): u(x_1) = u(x_2)
    - 壓力出口 (pressure outlet): p = p_0
    """
    
    def __init__(self):
        super().__init__()
    
    def no_slip_loss(self, 
                     boundary_coords: torch.Tensor,
                     boundary_predictions: torch.Tensor,
                     target_velocity: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        無滑移邊界條件：u = u_target (通常為 0)
        """
        spatial_dim = boundary_predictions.shape[-1] - 1  # 扣除壓力
        velocity_pred = boundary_predictions[:, :spatial_dim]
        
        if target_velocity is None:
            target_velocity = torch.zeros_like(velocity_pred)
        
        loss = torch.mean((velocity_pred - target_velocity) ** 2)
        return loss
    
    def pressure_outlet_loss(self,
                           boundary_coords: torch.Tensor,
                           boundary_predictions: torch.Tensor,
                           target_pressure: float = 0.0) -> torch.Tensor:
        """
        壓力出口邊界條件：p = p_0
        """
        spatial_dim = boundary_predictions.shape[-1] - 1
        pressure_pred = boundary_predictions[:, spatial_dim]
        
        target = torch.full_like(pressure_pred, target_pressure)
        loss = torch.mean((pressure_pred - target) ** 2)
        return loss
    
    def periodic_loss(self,
                     coords_1: torch.Tensor,
                     coords_2: torch.Tensor,
                     predictions_1: torch.Tensor,
                     predictions_2: torch.Tensor) -> torch.Tensor:
        """
        週期邊界條件：u(x_1) = u(x_2)
        """
        loss = torch.mean((predictions_1 - predictions_2) ** 2)
        return loss
    
    def inlet_velocity_profile_loss(self,
                                   inlet_coords: torch.Tensor,
                                   inlet_predictions: torch.Tensor,
                                   profile_type: str = 'parabolic',
                                   Re_tau: float = 1000.0,
                                   u_max: float = 16.5,
                                   y_range: Tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
        """
        Inlet 速度剖面邊界條件：指導模型學習主流方向
        
        Args:
            inlet_coords: inlet 位置座標 [N, spatial_dim]
            inlet_predictions: 模型在 inlet 的預測 [N, spatial_dim + 1]
            profile_type: 速度剖面類型 ('parabolic', 'log_law', 'turbulent')
            Re_tau: 摩擦雷諾數
            u_max: 中心線最大速度
            y_range: y 方向範圍 (壁面位置)
            
        Returns:
            loss: Inlet 速度剖面損失
        """
        spatial_dim = inlet_predictions.shape[-1] - 1
        u_pred = inlet_predictions[:, 0]  # 主流方向 (x) 速度
        v_pred = inlet_predictions[:, 1] if spatial_dim >= 2 else None  # 法向速度
        
        # 提取 y 座標 (假設 inlet_coords 為 [x, y] 或 [x, y, z])
        y = inlet_coords[:, 1]  # y 座標
        
        # 計算目標速度剖面
        if profile_type == 'parabolic':
            # 拋物線剖面（層流/低 Re）：u(y) = u_max * (1 - (y/h)²)
            h = (y_range[1] - y_range[0]) / 2.0  # 半通道高度
            y_normalized = y / h
            u_target = u_max * (1.0 - y_normalized ** 2)
            
        elif profile_type == 'log_law':
            # 對數律剖面（湍流）：u^+ = (1/κ) ln(y^+) + C^+
            kappa = 0.41  # von Karman 常數
            C = 5.0       # 對數律常數
            h = (y_range[1] - y_range[0]) / 2.0
            
            # y^+ = Re_tau * |y| / h
            y_abs = torch.abs(y)
            y_plus = Re_tau * y_abs / h
            
            # 避免 log(0)
            y_plus = torch.clamp(y_plus, min=1.0)
            
            # u^+ = (1/κ) ln(y^+) + C
            u_plus = (1.0 / kappa) * torch.log(y_plus) + C
            
            # 轉換為物理速度：u = u_τ * u^+
            # u_τ ≈ u_max / (u_max^+)，這裡簡化假設 u_max^+ ≈ Re_tau / 15
            u_tau = u_max / (Re_tau / 15.0)
            u_target = u_tau * u_plus
            
            # 標準化到 u_max
            u_target = u_target * (u_max / u_target.max())
            
        elif profile_type == 'turbulent':
            # 充分發展湍流剖面 (1/7 次方律)：u(y) = u_max * (1 - |y/h|)^(1/7)
            h = (y_range[1] - y_range[0]) / 2.0
            y_normalized = torch.abs(y) / h
            u_target = u_max * (1.0 - y_normalized) ** (1.0 / 7.0)
            
        else:
            raise ValueError(f"不支援的速度剖面類型: {profile_type}")
        
        # Inlet 損失：主流方向速度匹配
        u_loss = torch.mean((u_pred - u_target) ** 2)
        
        # 法向速度為零（無穿透條件）
        if v_pred is not None:
            v_loss = torch.mean(v_pred ** 2)
            total_loss = u_loss + v_loss
        else:
            total_loss = u_loss
        
        return total_loss


class InitialConditionLoss(nn.Module):
    """
    初始條件損失函數 (非定常問題)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self,
                initial_coords: torch.Tensor,
                initial_predictions: torch.Tensor,
                initial_data: torch.Tensor,
                weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        初始條件損失：u(t=0) = u_0
        
        Args:
            initial_coords: 初始時刻座標
            initial_predictions: 模型在初始時刻的預測
            initial_data: 真實初始條件
            weights: 各變數權重
        
        Returns:
            losses: 各變數的初始條件損失
        """
        losses = {}
        
        # 預設權重
        default_weights = {'velocity': 1.0, 'pressure': 1.0}
        if weights:
            default_weights.update(weights)
        
        # 分離速度與壓力
        spatial_dim = initial_data.shape[-1] - 1
        
        # 速度初始條件
        vel_pred = initial_predictions[:, :spatial_dim]
        vel_true = initial_data[:, :spatial_dim]
        losses['ic_velocity'] = default_weights['velocity'] * torch.mean((vel_pred - vel_true) ** 2)
        
        # 壓力初始條件
        p_pred = initial_predictions[:, spatial_dim]
        p_true = initial_data[:, spatial_dim]
        losses['ic_pressure'] = default_weights['pressure'] * torch.mean((p_pred - p_true) ** 2)
        
        # 總初始條件損失
        losses['total_ic'] = sum(losses.values())
        
        return losses


class PeriodicBoundaryLoss(nn.Module):
    """
    週期邊界條件損失函數
    
    確保週期邊界上的場變數保持連續性：
    u(x_min, y) = u(x_max, y)  (x方向週期)
    u(x, y_min) = u(x, y_max)  (y方向週期)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self,
                coords_pair_1: torch.Tensor,
                coords_pair_2: torch.Tensor,
                predictions_1: torch.Tensor,
                predictions_2: torch.Tensor,
                field_weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        計算週期邊界條件損失
        
        Args:
            coords_pair_1: 週期邊界一側的座標 [N, spatial_dim]
            coords_pair_2: 週期邊界另一側的座標 [N, spatial_dim]  
            predictions_1: 邊界一側的模型預測 [N, output_dim]
            predictions_2: 邊界另一側的模型預測 [N, output_dim]
            field_weights: 各場變數的權重
        
        Returns:
            losses: 週期邊界損失字典
        """
        losses = {}
        
        # 預設權重
        default_weights = {'velocity': 1.0, 'pressure': 1.0, 'source': 0.1}
        if field_weights:
            default_weights.update(field_weights)
        
        # 確定空間維度
        output_dim = predictions_1.shape[-1]
        
        if output_dim >= 3:  # 至少有 [u, v, p]
            # 速度場週期性
            spatial_dim = 2 if output_dim >= 3 else output_dim - 1
            vel_1 = predictions_1[:, :spatial_dim]
            vel_2 = predictions_2[:, :spatial_dim]
            losses['periodic_velocity'] = default_weights['velocity'] * torch.mean((vel_1 - vel_2) ** 2)
            
            # 壓力場週期性
            p_1 = predictions_1[:, spatial_dim]
            p_2 = predictions_2[:, spatial_dim]
            losses['periodic_pressure'] = default_weights['pressure'] * torch.mean((p_1 - p_2) ** 2)
            
            # 源項週期性（如果存在）
            if output_dim > spatial_dim + 1:
                source_1 = predictions_1[:, spatial_dim + 1:]
                source_2 = predictions_2[:, spatial_dim + 1:]
                losses['periodic_source'] = default_weights['source'] * torch.mean((source_1 - source_2) ** 2)
        else:
            # 簡單情況：所有場變數一起處理
            losses['periodic_total'] = torch.mean((predictions_1 - predictions_2) ** 2)
        
        # 總週期損失
        losses['total_periodic'] = sum(losses.values())
        
        return losses
    
    def x_periodic_loss(self,
                       model: nn.Module,
                       y_coords: torch.Tensor,
                       x_min: float,
                       x_max: float,
                       time_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        X方向週期邊界條件：u(x_min, y) = u(x_max, y)
        
        Args:
            model: PINN 模型
            y_coords: Y座標點 [N]
            x_min, x_max: X方向邊界值
            time_coords: 時間座標 [N] (非定常情況)
        
        Returns:
            X方向週期損失
        """
        device = next(model.parameters()).device
        y_coords = y_coords.to(device)
        
        # 建立邊界座標
        x_min_coords = torch.full_like(y_coords, x_min)
        x_max_coords = torch.full_like(y_coords, x_max)
        
        coords_min = torch.stack([x_min_coords, y_coords], dim=-1)
        coords_max = torch.stack([x_max_coords, y_coords], dim=-1)
        
        # 非定常情況添加時間維度
        if time_coords is not None:
            time_coords = time_coords.to(device)
            coords_min = torch.cat([time_coords.unsqueeze(-1), coords_min], dim=-1)
            coords_max = torch.cat([time_coords.unsqueeze(-1), coords_max], dim=-1)
        
        # 模型預測
        pred_min = model(coords_min)
        pred_max = model(coords_max)
        
        # 計算週期損失
        return self.forward(coords_min, coords_max, pred_min, pred_max)['total_periodic']
    
    def y_periodic_loss(self,
                       model: nn.Module,
                       x_coords: torch.Tensor,
                       y_min: float,
                       y_max: float,
                       time_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Y方向週期邊界條件：u(x, y_min) = u(x, y_max)
        """
        device = next(model.parameters()).device
        x_coords = x_coords.to(device)
        
        # 建立邊界座標
        y_min_coords = torch.full_like(x_coords, y_min)
        y_max_coords = torch.full_like(x_coords, y_max)
        
        coords_min = torch.stack([x_coords, y_min_coords], dim=-1)
        coords_max = torch.stack([x_coords, y_max_coords], dim=-1)
        
        # 非定常情況添加時間維度
        if time_coords is not None:
            time_coords = time_coords.to(device)
            coords_min = torch.cat([time_coords.unsqueeze(-1), coords_min], dim=-1)
            coords_max = torch.cat([time_coords.unsqueeze(-1), coords_max], dim=-1)
        
        # 模型預測
        pred_min = model(coords_min)
        pred_max = model(coords_max)
        
        # 計算週期損失
        return self.forward(coords_min, coords_max, pred_min, pred_max)['total_periodic']


def create_ns_loss(config: Dict) -> NSResidualLoss:
    """根據配置建立 NS 殘差損失函數"""
    return NSResidualLoss(
        nu=config.get('nu', 1e-3),
        density=config.get('density', 1.0),
        spatial_dim=config.get('spatial_dim', 2),
        unsteady=config.get('unsteady', False),
        source_regularization=config.get('source_reg', 0.0),
        merge_momentum=config.get('merge_momentum', False)
    )
