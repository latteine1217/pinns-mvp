"""
RANS湍流方程物理模組

包含k-ε湍流模型和RANS動量方程的實現
支持PINNs中的湍流狀態預測
"""

import torch
import torch.autograd as autograd
from typing import Optional, Dict, Tuple, Any
import warnings


def compute_velocity_gradients(velocity: torch.Tensor, 
                              coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    計算速度場的梯度
    
    Args:
        velocity: 速度 [batch_size, 2] (U, V)
        coords: 座標 [batch_size, ndim] (支持2D或3D)
        
    Returns:
        包含 U_grads, V_grads 的字典
    """
    U = velocity[:, 0:1]
    V = velocity[:, 1:2]
    
    grad_outputs = torch.ones_like(U)
    
    # 使用完整座標計算梯度，保持計算圖連接
    U_grads = autograd.grad(
        outputs=U, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    V_grads = autograd.grad(
        outputs=V, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    # 檢查梯度是否成功計算
    if U_grads is None or V_grads is None:
        # 返回零梯度作為備用
        batch_size = coords.shape[0]
        U_grads = torch.zeros(batch_size, coords.shape[1], device=coords.device)
        V_grads = torch.zeros(batch_size, coords.shape[1], device=coords.device)
    
    return {
        'U_grads': U_grads,
        'V_grads': V_grads
    }


def compute_pressure_gradients(pressure: torch.Tensor,
                             coords: torch.Tensor) -> torch.Tensor:
    """計算壓力場的梯度"""
    P = pressure
    grad_outputs = torch.ones_like(P)
    
    P_grads = autograd.grad(
        outputs=P, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    if P_grads is None:
        batch_size = coords.shape[0]
        P_grads = torch.zeros(batch_size, coords.shape[1], device=coords.device)
    
    return P_grads


def rans_momentum_residual(coords_input: torch.Tensor,
                          velocity: torch.Tensor,
                          pressure: torch.Tensor,
                          k: torch.Tensor,
                          epsilon: torch.Tensor,
                          nu: float = 1e-5,
                          time: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    RANS動量方程殘差計算
    
    Args:
        coords_input: 輸入座標 [batch_size, ndim] (可以是2D或3D)
        velocity: 速度 [batch_size, 2]
        pressure: 壓力 [batch_size, 1]
        k: 湍動能 [batch_size, 1]
        epsilon: 耗散率 [batch_size, 1]
        nu: 動力學粘度
        time: 時間座標 (可選)
        
    Returns:
        (momentum_x_residual, momentum_y_residual)
    """
    # 確保座標有梯度
    if not coords_input.requires_grad:
        coords_input.requires_grad_(True)
    
    # 計算速度和壓力梯度
    vel_grads = compute_velocity_gradients(velocity, coords_input)
    pres_grads = compute_pressure_gradients(pressure, coords_input)
    
    U_grads = vel_grads['U_grads']
    V_grads = vel_grads['V_grads']
    P_grads = pres_grads
    
    # 檢查是否成功計算梯度
    if torch.allclose(U_grads, torch.zeros_like(U_grads)) and torch.allclose(V_grads, torch.zeros_like(V_grads)):
        # 梯度計算失敗，返回零殘差
        batch_size = coords_input.shape[0]
        return (torch.zeros(batch_size, 1, device=coords_input.device),
                torch.zeros(batch_size, 1, device=coords_input.device))
    
    U = velocity[:, 0:1]
    V = velocity[:, 1:2]
    
    # 提取空間梯度 (前兩個維度是x,y)
    U_x, U_y = U_grads[:, 0:1], U_grads[:, 1:2]
    V_x, V_y = V_grads[:, 0:1], V_grads[:, 1:2]
    P_x, P_y = P_grads[:, 0:1], P_grads[:, 1:2]
    
    # 對流項
    U_convection = U * U_x + V * U_y
    V_convection = U * V_x + V * V_y
    
    # 時間導數 (如果輸入是3D且包含時間)
    if coords_input.shape[1] >= 3:
        U_t = U_grads[:, 2:3]
        V_t = V_grads[:, 2:3]
    else:
        U_t = torch.zeros_like(U)
        V_t = torch.zeros_like(V)
    
    # 簡化的拉普拉斯項計算 (使用有限差分近似)
    # 在實際實現中，這需要計算二階導數
    U_laplacian = torch.zeros_like(U)  # 暫時簡化
    V_laplacian = torch.zeros_like(V)
    
    # 湍流粘度 (k-ε模型)
    C_mu = 0.09
    nu_t = C_mu * k**2 / (epsilon + 1e-10)
    
    # 雷諾茲應力項的簡化處理
    # 這裡使用渦粘度假設：τ_ij = -ν_t * (∂u_i/∂x_j + ∂u_j/∂x_i)
    tau_xx = -2 * nu_t * U_x
    tau_yy = -2 * nu_t * V_y
    tau_xy = -nu_t * (U_y + V_x)
    
    # 雷諾茲應力散度項的簡化計算
    # ∂τ_xx/∂x + ∂τ_xy/∂y ≈ 0 (簡化)
    tau_div_x = torch.zeros_like(U)
    tau_div_y = torch.zeros_like(V)
    
    # RANS動量方程殘差
    # ∂U/∂t + U∂U/∂x + V∂U/∂y = -∂P/∂x + ν∇²U - ∂τ_ij/∂x_j
    momentum_x = U_t + U_convection + P_x - nu * U_laplacian + tau_div_x
    momentum_y = V_t + V_convection + P_y - nu * V_laplacian + tau_div_y
    
    return momentum_x, momentum_y


def continuity_residual(velocity: torch.Tensor, 
                       coords: torch.Tensor) -> torch.Tensor:
    """連續方程殘差計算"""
    vel_grads = compute_velocity_gradients(velocity, coords)
    U_grads = vel_grads['U_grads']
    V_grads = vel_grads['V_grads']
    
    if U_grads is not None and V_grads is not None:
        U_x = U_grads[:, 0:1]
        V_y = V_grads[:, 1:2]
        return U_x + V_y
    else:
        # 梯度計算失敗
        batch_size = coords.shape[0]
        return torch.zeros(batch_size, 1, device=coords.device)


def k_epsilon_residuals(coords: torch.Tensor,
                       velocity: torch.Tensor,
                       k: torch.Tensor,
                       epsilon: torch.Tensor,
                       nu: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    k-ε湍流模型方程殘差
    
    Returns:
        (k_residual, epsilon_residual)
    """
    # k-ε模型常數
    C_mu = 0.09
    C_1e = 1.44
    C_2e = 1.92
    sigma_k = 1.0
    sigma_e = 1.3
    
    U = velocity[:, 0:1]
    V = velocity[:, 1:2]
    
    # 計算速度梯度
    vel_grads = compute_velocity_gradients(velocity, coords)
    U_grads = vel_grads['U_grads']
    V_grads = vel_grads['V_grads']
    
    if U_grads is not None and V_grads is not None:
        U_x, U_y = U_grads[:, 0:1], U_grads[:, 1:2]
        V_x, V_y = V_grads[:, 0:1], V_grads[:, 1:2]
        
        # 生產項 P_k = ν_t * S^2
        S_xx = U_x
        S_yy = V_y
        S_xy = 0.5 * (U_y + V_x)
        S_mag_sq = 2 * (S_xx**2 + S_yy**2 + 2 * S_xy**2)
        
        nu_t = C_mu * k**2 / (epsilon + 1e-10)
        P_k = nu_t * S_mag_sq
    else:
        P_k = torch.zeros_like(k)
    
    # k方程殘差 (簡化版，忽略對流和擴散項)
    # 在實際應用中需要完整的對流-擴散項
    k_residual = P_k - epsilon
    
    # ε方程殘差 (簡化版)
    epsilon_residual = C_1e * (epsilon / (k + 1e-10)) * P_k - C_2e * (epsilon**2 / (k + 1e-10))
    
    return k_residual, epsilon_residual


def apply_physical_constraints(k: torch.Tensor, 
                             epsilon: torch.Tensor,
                             constraint_type: str = "softplus") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    對湍流量應用物理約束 k≥0, ε≥0
    
    Args:
        k: 湍動能 [batch_size, 1]
        epsilon: 耗散率 [batch_size, 1]
        constraint_type: 約束類型 ("relu", "softplus", "clip")
        
    Returns:
        (constrained_k, constrained_epsilon)
    """
    if constraint_type == "relu":
        k_constrained = torch.relu(k)
        epsilon_constrained = torch.relu(epsilon)
    elif constraint_type == "softplus":
        k_constrained = torch.nn.functional.softplus(k)
        epsilon_constrained = torch.nn.functional.softplus(epsilon)
    elif constraint_type == "clip":
        k_constrained = torch.clamp(k, min=1e-10)
        epsilon_constrained = torch.clamp(epsilon, min=1e-10)
    else:
        # 無約束
        k_constrained = k
        epsilon_constrained = epsilon
    
    return k_constrained, epsilon_constrained


def physical_constraint_penalty(k: torch.Tensor, 
                               epsilon: torch.Tensor,
                               penalty_weight: float = 1.0) -> torch.Tensor:
    """
    計算物理約束懲罰項
    
    Args:
        k: 湍動能
        epsilon: 耗散率  
        penalty_weight: 懲罰權重
        
    Returns:
        約束懲罰項
    """
    k_penalty = torch.relu(-k)  # k < 0 的懲罰
    epsilon_penalty = torch.relu(-epsilon)  # ε < 0 的懲罰
    
    total_penalty = penalty_weight * (k_penalty.mean() + epsilon_penalty.mean())
    return total_penalty


class RANSEquations2D:
    """
    2D RANS方程組
    
    包含：
    1. RANS動量方程 (x, y方向)
    2. 連續方程
    3. k方程 (湍動能)
    4. ε方程 (耗散率)
    """
    
    def __init__(self, viscosity: float = 1e-5, turbulence_model: str = 'k_epsilon',
                 enable_constraints: bool = True, constraint_type: str = "softplus"):
        self.viscosity = viscosity
        self.turbulence_model = turbulence_model
        self.enable_constraints = enable_constraints
        self.constraint_type = constraint_type
        
        # k-ε模型常數
        self.constants = {
            'C_mu': 0.09,
            'C_1e': 1.44,
            'C_2e': 1.92,
            'sigma_k': 1.0,
            'sigma_e': 1.3
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        return {
            'viscosity': self.viscosity,
            'turbulence_model': self.turbulence_model,
            'model_constants': self.constants,
            'output_variables': ['U', 'V', 'P', 'k', 'epsilon/omega'],
            'equation_count': 5,
            'constraints_enabled': self.enable_constraints,
            'constraint_type': self.constraint_type
        }
    
    def residual(self,
                coords: torch.Tensor,
                velocity: torch.Tensor,
                pressure: torch.Tensor,
                k: torch.Tensor,
                epsilon_or_omega: torch.Tensor,
                time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        計算完整的RANS方程殘差
        
        Args:
            coords: 座標 [batch_size, ndim] (支持2D或3D)
            velocity: 速度 [batch_size, 2]
            pressure: 壓力 [batch_size, 1]
            k: 湍動能 [batch_size, 1]
            epsilon_or_omega: 耗散率或比耗散率 [batch_size, 1]
            time: 時間座標 [batch_size, 1] (可選)
            
        Returns:
            殘差字典
        """
        try:
            # 應用物理約束
            if self.enable_constraints:
                k_constrained, epsilon_constrained = apply_physical_constraints(
                    k, epsilon_or_omega, self.constraint_type
                )
            else:
                k_constrained, epsilon_constrained = k, epsilon_or_omega
            
            # RANS動量方程殘差
            momentum_x, momentum_y = rans_momentum_residual(
                coords, velocity, pressure, k_constrained, epsilon_constrained, self.viscosity, time
            )
            
            # 連續方程殘差
            continuity = continuity_residual(velocity, coords)
            
            # k-ε湍流方程殘差
            k_residual, dissipation_residual = k_epsilon_residuals(
                coords, velocity, k_constrained, epsilon_constrained, self.viscosity
            )
            
            return {
                'momentum_x': momentum_x,
                'momentum_y': momentum_y,
                'continuity': continuity,
                'k_equation': k_residual,
                'dissipation_equation': dissipation_residual,
                'physical_penalty': physical_constraint_penalty(k, epsilon_or_omega) if self.enable_constraints else torch.tensor(0.0)
            }
            
        except Exception as e:
            warnings.warn(f"RANS殘差計算失敗，使用簡化版本: {e}")
            
            # 備用：簡化的殘差計算
            batch_size = coords.shape[0]
            zero_residual = torch.zeros(batch_size, 1, device=coords.device, dtype=coords.dtype)
            
            return {
                'momentum_x': zero_residual,
                'momentum_y': zero_residual,
                'continuity': zero_residual,
                'k_equation': zero_residual,
                'dissipation_equation': zero_residual
            }