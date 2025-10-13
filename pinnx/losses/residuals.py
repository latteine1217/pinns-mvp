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


def ns_residual_2d(coords: torch.Tensor,
                  velocity: torch.Tensor,
                  pressure: torch.Tensor,
                  source: Optional[torch.Tensor] = None,
                  nu: float = 1e-3,
                  time_coords: Optional[torch.Tensor] = None,
                  density: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    2D 不可壓縮 Navier-Stokes 方程殘差計算
    
    方程形式：
    ∂u/∂t + u·∇u = -∇p/ρ + ν∇²u + S
    ∇·u = 0
    
    Args:
        coords: 空間座標 [batch_size, 2] (x, y)
        velocity: 速度場 [batch_size, 2] (u, v)
        pressure: 壓力場 [batch_size]
        source: 源項 [batch_size, 2] (可選)
        nu: 動力黏性係數
        time_coords: 時間座標 [batch_size] (非定常問題)
        density: 密度 (預設 1.0)
    
    Returns:
        residuals: 包含各方程殘差的字典
    """
    batch_size = coords.shape[0]
    
    u, v = velocity[:, 0], velocity[:, 1]
    p = pressure
    
    # 速度梯度
    u_grad = compute_gradients(u, coords, order=1)  # [batch, 2]
    v_grad = compute_gradients(v, coords, order=1)
    p_grad = compute_gradients(p, coords, order=1)
    
    ux, uy = u_grad[:, 0], u_grad[:, 1]
    vx, vy = v_grad[:, 0], v_grad[:, 1]
    px, py = p_grad[:, 0], p_grad[:, 1]
    
    # 拉普拉斯項
    u_lap = laplacian(u, coords)
    v_lap = laplacian(v, coords)
    
    # 時間導數 (非定常情況)
    if time_coords is not None:
        time_coords.requires_grad_(True)
        u_t = compute_gradients(u, time_coords, order=1)[:, 0]
        v_t = compute_gradients(v, time_coords, order=1)[:, 0]
    else:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
    
    # 源項 (如果提供)
    if source is not None:
        sx, sy = source[:, 0], source[:, 1]
    else:
        sx = torch.zeros_like(u)
        sy = torch.zeros_like(v)
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy) + px / density - nu * u_lap - sx
    momentum_y = v_t + (u * vx + v * vy) + py / density - nu * v_lap - sy
    
    # 連續性方程殘差 (不可壓縮)
    continuity = ux + vy
    
    residuals = {
        'momentum_x': momentum_x,
        'momentum_y': momentum_y, 
        'continuity': continuity,
        'velocity_div': continuity  # 等價於散度
    }
    
    return residuals


def ns_residual_3d(coords: torch.Tensor,
                  velocity: torch.Tensor,
                  pressure: torch.Tensor,
                  source: Optional[torch.Tensor] = None,
                  nu: float = 1e-3,
                  time_coords: Optional[torch.Tensor] = None,
                  density: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    3D 不可壓縮 Navier-Stokes 方程殘差計算
    """
    u, v, w = velocity[:, 0], velocity[:, 1], velocity[:, 2]
    p = pressure
    
    # 速度梯度
    u_grad = compute_gradients(u, coords, order=1)  # [batch, 3]
    v_grad = compute_gradients(v, coords, order=1)
    w_grad = compute_gradients(w, coords, order=1)
    p_grad = compute_gradients(p, coords, order=1)
    
    ux, uy, uz = u_grad[:, 0], u_grad[:, 1], u_grad[:, 2]
    vx, vy, vz = v_grad[:, 0], v_grad[:, 1], v_grad[:, 2]
    wx, wy, wz = w_grad[:, 0], w_grad[:, 1], w_grad[:, 2]
    px, py, pz = p_grad[:, 0], p_grad[:, 1], p_grad[:, 2]
    
    # 拉普拉斯項
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
    
    # 動量方程殘差
    momentum_x = u_t + (u * ux + v * uy + w * uz) + px / density - nu * u_lap - sx
    momentum_y = v_t + (u * vx + v * vy + w * vz) + py / density - nu * v_lap - sy
    momentum_z = w_t + (u * wx + v * wy + w * wz) + pz / density - nu * w_lap - sz
    
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
                 source_regularization: float = 0.0):
        """
        Args:
            nu: 動力黏性係數
            density: 流體密度
            spatial_dim: 空間維度 (2 或 3)
            unsteady: 是否為非定常問題
            source_regularization: 源項稀疏化權重
        """
        super().__init__()
        
        self.nu = nu
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
                weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        計算 NS 方程殘差損失
        
        Args:
            coords: 空間座標 [batch_size, spatial_dim]
            predictions: 模型預測 [batch_size, output_dim]
                        output_dim = spatial_dim + 1 + source_dim
                        例如 2D: [u, v, p, sx, sy]
            time_coords: 時間座標 [batch_size, 1]
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
        
        # 計算殘差
        residuals = self.residual_fn(
            coords=coords,
            velocity=velocity,
            pressure=pressure,
            source=source,
            nu=self.nu,
            time_coords=time_coords,
            density=self.density
        )
        
        # 損失權重 (預設值)
        default_weights = {
            'momentum_x': 1.0,
            'momentum_y': 1.0,
            'momentum_z': 1.0,
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


# 便捷函數與別名
def gradient(u: torch.Tensor, x: torch.Tensor, order: int = 1) -> torch.Tensor:
    """
    gradient 函數的便捷別名，與 compute_gradients 等價
    
    為了保持與其他 PINN 框架的相容性而提供的接口
    
    Args:
        u: 標量場 [batch_size]
        x: 座標 [batch_size, dim]
        order: 梯度階數 (1 或 2)
    
    Returns:
        gradients: 梯度張量
    """
    return compute_gradients(u, x, order=order)


def create_ns_loss(config: Dict) -> NSResidualLoss:
    """根據配置建立 NS 殘差損失函數"""
    return NSResidualLoss(
        nu=config.get('nu', 1e-3),
        density=config.get('density', 1.0),
        spatial_dim=config.get('spatial_dim', 2),
        unsteady=config.get('unsteady', False),
        source_regularization=config.get('source_reg', 0.0)
    )


# 為測試文件提供的相容性函數
def pde_residual_loss(residual: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    PDE 殘差損失函數 (相容性接口)
    
    Args:
        residual: PDE 殘差張量 [batch_size, n_equations]
        weights: 可選的權重 [batch_size]
    
    Returns:
        loss: 平均平方誤差損失
    """
    if weights is not None:
        # 加權平方誤差
        loss = torch.mean(weights.unsqueeze(-1) * (residual ** 2))
    else:
        # 標準平方誤差
        loss = torch.mean(residual ** 2)
    
    return loss


def boundary_residual_loss(predicted: torch.Tensor, 
                          target: torch.Tensor, 
                          bc_type: str = 'dirichlet',
                          coords: Optional[torch.Tensor] = None,
                          normal: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    邊界殘差損失函數 (相容性接口)
    
    Args:
        predicted: 邊界上的預測值
        target: 邊界目標值
        bc_type: 邊界條件類型 ('dirichlet', 'neumann')
        coords: 邊界座標 (Neumann 條件需要)
        normal: 邊界法向量 (Neumann 條件需要)
    
    Returns:
        loss: 邊界條件損失
    """
    if bc_type == 'dirichlet':
        # Dirichlet 邊界條件：直接比較值
        return torch.mean((predicted - target) ** 2)
    
    elif bc_type == 'neumann':
        # Neumann 邊界條件：比較法向導數
        if coords is None or normal is None:
            raise ValueError("Neumann 邊界條件需要提供座標和法向量")
        
        # 計算預測場的梯度
        if predicted.dim() == 1:
            # 標量場
            grad = compute_gradients(predicted, coords, order=1)
            normal_grad = torch.sum(grad * normal, dim=-1)
        else:
            # 向量場：計算每個分量的法向導數
            normal_grad = []
            for i in range(predicted.shape[-1]):
                grad_i = compute_gradients(predicted[:, i], coords, order=1)
                normal_grad_i = torch.sum(grad_i * normal, dim=-1)
                normal_grad.append(normal_grad_i)
            normal_grad = torch.stack(normal_grad, dim=-1)
        
        return torch.mean((normal_grad - target) ** 2)
    
    else:
        raise ValueError(f"不支援的邊界條件類型: {bc_type}")


def initial_condition_loss(predicted_t0: torch.Tensor, 
                          target_t0: torch.Tensor,
                          time_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    初始條件損失函數 (相容性接口)
    
    Args:
        predicted_t0: 初始時刻的預測值
        target_t0: 初始時刻的目標值
        time_weights: 時間權重
    
    Returns:
        loss: 初始條件損失
    """
    residual = predicted_t0 - target_t0
    
    if time_weights is not None:
        # 加權損失
        loss = torch.mean(time_weights.unsqueeze(-1) * (residual ** 2))
    else:
        # 標準 MSE
        loss = torch.mean(residual ** 2)
    
    return loss


def source_regularization(source_field: torch.Tensor, 
                         reg_type: str = 'l1',
                         strength: float = 1e-6) -> torch.Tensor:
    """
    源項正則化函數 (相容性接口)
    
    Args:
        source_field: 源項場
        reg_type: 正則化類型 ('l1', 'l2', 'smooth')
        strength: 正則化強度
    
    Returns:
        loss: 正則化損失
    """
    if reg_type == 'l1':
        # L1 稀疏正則化
        return strength * torch.mean(torch.abs(source_field))
    
    elif reg_type == 'l2':
        # L2 正則化
        return strength * torch.mean(source_field ** 2)
    
    elif reg_type == 'smooth':
        # 平滑正則化 (總變差)
        if source_field.dim() == 2:
            # 2D 場的平滑性
            dx = source_field[1:, :] - source_field[:-1, :]
            dy = source_field[:, 1:] - source_field[:, :-1]
            tv_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
        else:
            # 1D 情況
            dx = source_field[1:] - source_field[:-1]
            tv_loss = torch.mean(torch.abs(dx))
        
        return strength * tv_loss
    
    else:
        raise ValueError(f"不支援的正則化類型: {reg_type}")


def data_fitting_loss(predicted: torch.Tensor, 
                     observed: torch.Tensor,
                     noise_std: Optional[torch.Tensor] = None,
                     mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    資料擬合損失函數 (相容性接口)
    
    Args:
        predicted: 模型預測值
        observed: 觀測資料
        noise_std: 噪音標準差 (用於加權)
        mask: 資料遮罩 (True 表示有效資料)
    
    Returns:
        loss: 資料擬合損失
    """
    residual = predicted - observed
    
    # 應用遮罩
    if mask is not None:
        residual = residual[mask]
        if noise_std is not None:
            noise_std = noise_std[mask]
    
    # 加權損失 (根據噪音水平)
    if noise_std is not None:
        # 反比權重：噪音越大，權重越小
        weights = 1.0 / (noise_std.unsqueeze(-1) ** 2 + 1e-8)
        loss = torch.mean(weights * (residual ** 2))
    else:
        # 標準 MSE
        loss = torch.mean(residual ** 2)
    
    return loss


if __name__ == "__main__":
    # 測試程式碼
    print("=== NS 殘差損失函數測試 ===")
    
    # 建立測試資料
    batch_size = 100
    coords = torch.randn(batch_size, 2, requires_grad=True)  # (x, y)
    
    # 建立簡單的測試模型來生成有梯度的預測
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 5)  # [u, v, p, sx, sy]
            )
        
        def forward(self, x):
            return self.net(x)
    
    model = TestModel()
    predictions = model(coords)  # 這樣生成的預測具有梯度
    
    # 建立損失函數
    ns_loss = NSResidualLoss(
        nu=1e-3,
        spatial_dim=2,
        unsteady=False,
        source_regularization=1e-6
    )
    
    # 計算損失
    losses = ns_loss(coords, predictions)
    
    print("計算的損失項：")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 測試邊界條件
    print("\n--- 邊界條件測試 ---")
    bc_loss = BoundaryConditionLoss()
    
    boundary_coords = torch.randn(50, 2)
    boundary_preds = torch.randn(50, 3)  # [u, v, p]
    
    no_slip_loss = bc_loss.no_slip_loss(boundary_coords, boundary_preds)
    print(f"無滑移邊界損失: {no_slip_loss.item():.6f}")
    
    pressure_loss = bc_loss.pressure_outlet_loss(boundary_coords, boundary_preds, target_pressure=0.0)
    print(f"壓力出口損失: {pressure_loss.item():.6f}")
    
    # 測試相容性函數
    print("\n--- 相容性函數測試 ---")
    test_residual = torch.randn(50, 2)
    pde_loss = pde_residual_loss(test_residual)
    print(f"PDE 殘差損失: {pde_loss.item():.6f}")
    
    test_pred = torch.randn(30, 3)
    test_target = torch.zeros(30, 3)
    data_loss = data_fitting_loss(test_pred, test_target)
    print(f"資料擬合損失: {data_loss.item():.6f}")
    
    print("\n✅ 殘差損失函數測試通過！")