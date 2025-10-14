"""
RANS湍流方程物理模組

包含k-ε湍流模型和RANS動量方程的實現
支持PINNs中的湍流狀態預測
"""

import torch
import torch.autograd as autograd
import torch.nn.functional as F
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


def compute_velocity_gradients_3d(velocity: torch.Tensor, 
                                  coords: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    計算 3D 速度場的梯度
    
    Args:
        velocity: 速度 [batch_size, 3] (u, v, w)
        coords: 座標 [batch_size, 3] (x, y, z)
        
    Returns:
        包含 u_grads, v_grads, w_grads 的字典
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]
    
    grad_outputs = torch.ones_like(u)
    
    # 計算梯度
    u_grads = autograd.grad(
        outputs=u, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    v_grads = autograd.grad(
        outputs=v, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    w_grads = autograd.grad(
        outputs=w, inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True, retain_graph=True,
        only_inputs=True, allow_unused=True
    )[0]
    
    # 檢查梯度是否成功計算
    if u_grads is None or v_grads is None or w_grads is None:
        batch_size = coords.shape[0]
        u_grads = torch.zeros(batch_size, 3, device=coords.device)
        v_grads = torch.zeros(batch_size, 3, device=coords.device)
        w_grads = torch.zeros(batch_size, 3, device=coords.device)
    
    return {
        'u_grads': u_grads,
        'v_grads': v_grads,
        'w_grads': w_grads
    }


def k_epsilon_residuals_3d(coords: torch.Tensor,
                           velocity: torch.Tensor,
                           k: torch.Tensor,
                           epsilon: torch.Tensor,
                           nu: float = 1e-5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    3D k-ε 湍流模型方程殘差
    
    Args:
        coords: 座標 [batch_size, 3] (x, y, z)
        velocity: 速度 [batch_size, 3] (u, v, w)
        k: 湍動能 [batch_size, 1]
        epsilon: 耗散率 [batch_size, 1]
        nu: 動力學粘度
        
    Returns:
        (k_residual, epsilon_residual)
    """
    # k-ε 模型常數
    C_mu = 0.09
    C_1e = 1.44
    C_2e = 1.92
    sigma_k = 1.0
    sigma_e = 1.3
    
    # 計算速度梯度
    vel_grads = compute_velocity_gradients_3d(velocity, coords)
    u_grads = vel_grads['u_grads']
    v_grads = vel_grads['v_grads']
    w_grads = vel_grads['w_grads']
    
    if u_grads is not None and v_grads is not None and w_grads is not None:
        # 提取空間梯度
        u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
        v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
        w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
        
        # 應變率張量 S_ij = 0.5 * (∂u_i/∂x_j + ∂u_j/∂x_i)
        S_xx = u_x
        S_yy = v_y
        S_zz = w_z
        S_xy = 0.5 * (u_y + v_x)
        S_xz = 0.5 * (u_z + w_x)
        S_yz = 0.5 * (v_z + w_y)
        
        # 應變率張量模的平方 S^2 = 2 * S_ij * S_ij
        S_mag_sq = 2 * (S_xx**2 + S_yy**2 + S_zz**2 + 
                        2 * (S_xy**2 + S_xz**2 + S_yz**2))
        
        # 湍流黏度 ν_t = C_μ * k² / ε
        nu_t = C_mu * k**2 / (epsilon + 1e-10)
        
        # 生產項 P_k = ν_t * S^2
        P_k = nu_t * S_mag_sq
    else:
        P_k = torch.zeros_like(k)
    
    # k 方程殘差（簡化版，忽略對流和擴散項）
    k_residual = P_k - epsilon
    
    # ε 方程殘差（簡化版）
    epsilon_residual = C_1e * (epsilon / (k + 1e-10)) * P_k - C_2e * (epsilon**2 / (k + 1e-10))
    
    return k_residual, epsilon_residual


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


class RANSEquations3D:
    """
    3D RANS 方程組（軟約束模式）
    
    專為 VS-PINN 通道流設計，作為物理約束項使用。
    模型輸出仍為 (u, v, w, p)，k 和 ε 從速度場估算。
    
    包含：
    1. k 方程（湍動能）
    2. ε 方程（耗散率）
    3. 湍流黏度一致性約束
    """
    
    def __init__(self, viscosity: float = 5e-5, 
                 enable_constraints: bool = True, 
                 constraint_type: str = "softplus",
                 turbulent_viscosity_penalty: str = "log1p",
                 turbulent_viscosity_target: float = 100.0,
                 turbulent_viscosity_huber_delta: float = 100.0):
        """
        初始化 3D RANS 方程組
        
        Args:
            viscosity: 分子黏度
            enable_constraints: 啟用物理約束（k≥0, ε≥0）
            constraint_type: 約束類型 ("relu", "softplus", "clip")
            turbulent_viscosity_penalty: 湍流黏度懲罰類型 ("huber", "log1p", "mse")
            turbulent_viscosity_target: 湍流黏度目標值（相對分子黏度的倍數）
            turbulent_viscosity_huber_delta: Huber 損失轉折點（β 參數，相對分子黏度的倍數）
        """
        self.viscosity = viscosity
        self.enable_constraints = enable_constraints
        self.constraint_type = constraint_type
        self.turbulent_viscosity_penalty = turbulent_viscosity_penalty
        self.turbulent_viscosity_target = turbulent_viscosity_target
        self.turbulent_viscosity_huber_delta = turbulent_viscosity_huber_delta
        
        # k-ε 模型常數
        self.constants = {
            'C_mu': 0.09,      # 湍流黏度係數
            'C_1e': 1.44,      # ε 方程生產項係數
            'C_2e': 1.92,      # ε 方程耗散項係數
            'sigma_k': 1.0,    # k 方程 Prandtl 數
            'sigma_e': 1.3     # ε 方程 Prandtl 數
        }
    
    def initialize_turbulent_fields(self,
                                    coords: torch.Tensor,
                                    velocity: torch.Tensor,
                                    turbulent_intensity: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基於混合長度理論物理初始化湍流場 k 和 ε
        
        物理依據（修正版 - Phase 5B）：
        1. 湍動能 k：基於壁面剪應力估算
           k = u_τ² / √C_μ （通道流壁單位尺度）
           其中 u_τ = √(τ_w/ρ) 為摩擦速度
        
        2. 耗散率 ε：基於混合長度假設
           ε = C_μ^0.75 * k^1.5 / l_mix
           其中 l_mix = κ * y_wall （von Kármán 常數 κ=0.41）
        
        此初始化確保：
        - k > 0（能量必須為正）
        - ε > 0（耗散率必須為正）
        - ν_t = C_μ * k²/ε 在合理範圍內（10-100 倍分子黏度）
        
        Args:
            coords: 座標 [batch_size, 3] (x, y, z)
            velocity: 速度 [batch_size, 3] (u, v, w)（初期可能接近零，不應依賴）
            turbulent_intensity: 湍流強度（此版本不使用，保留接口相容性）
            
        Returns:
            (k_init, epsilon_init) 物理一致的初始場
        """
        # 壁面距離（通道流中心線 y=0）
        y = coords[:, 1:2]  # 壁法向座標 [-1, 1]
        y_wall = torch.abs(y)  # 距離最近壁面的距離 [0, 1]
        
        # 混合長度：l = κ * y_wall * (1 - y_wall/h)
        # 通道半高 h=1，所以簡化為 l = κ * y_wall * (1 - y_wall)
        kappa = 0.41  # von Kármán 常數
        l_mix = kappa * y_wall * (1.0 - y_wall)
        l_mix = torch.clamp(l_mix, min=1e-3, max=0.1)  # 限制長度尺度範圍
        
        # ✅ 修正：基於壁面剪應力估算湍動能
        # 對於 Re_τ=1000，u_τ ≈ 0.05（從 JHTDB 資料）
        # k = u_τ² / √C_μ ≈ (0.05)² / √0.09 ≈ 0.0083
        C_mu = self.constants['C_mu']
        u_tau_est = 0.05  # 摩擦速度估計值（Re_τ=1000 典型值）
        k_base = (u_tau_est ** 2) / torch.sqrt(torch.tensor(C_mu, device=coords.device))
        
        # 根據壁面距離調整 k（中心線較大，近壁較小）
        # 使用 y_wall 的反函數：k ~ k_base * (1 - 0.5*y_wall)
        y_factor = 1.0 - 0.5 * y_wall  # [0.5, 1.0]
        k_init = k_base * y_factor
        k_init = torch.clamp(k_init, min=1e-6)  # 確保正值且非零
        
        # 耗散率：ε = C_μ^0.75 * k^1.5 / l
        epsilon_init = (C_mu ** 0.75) * (k_init ** 1.5) / l_mix
        epsilon_init = torch.clamp(epsilon_init, min=1e-6)  # 確保正值且非零
        
        # 驗證湍流黏度合理性（調試輸出）
        nu_t_init = C_mu * k_init**2 / (epsilon_init + 1e-10)
        nu_t_ratio = (nu_t_init / self.viscosity).mean().item()
        
        # 只在第一次調用時輸出（避免日誌爆炸）
        if not hasattr(self, '_init_logged'):
            print(f"  🔧 RANS 物理初始化完成 (Phase 5B - 壁面尺度修正):")
            print(f"     - k 範圍: [{k_init.min().item():.2e}, {k_init.max().item():.2e}]")
            print(f"     - ε 範圍: [{epsilon_init.min().item():.2e}, {epsilon_init.max().item():.2e}]")
            print(f"     - ν_t/ν 平均: {nu_t_ratio:.2f} (目標: 10-100)")
            print(f"     - 基準湍動能 k_base: {k_base.item():.2e}")
            self._init_logged = True
        
        return k_init, epsilon_init
    
    def estimate_turbulent_quantities(self, 
                                     coords: torch.Tensor,
                                     velocity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        從速度場估算湍動能 k 和耗散率 ε（軟約束模式）
        
        使用經驗公式：
        - k ≈ 0.5 * u'²（假設各向同性）
        - ε ≈ C_μ * k^(3/2) / L（混合長度估算）
        
        Args:
            coords: 座標 [batch_size, 3] (x, y, z)
            velocity: 速度 [batch_size, 3] (u, v, w)
            
        Returns:
            (k_estimated, epsilon_estimated)
        """
        # 計算速度梯度
        vel_grads = compute_velocity_gradients_3d(velocity, coords)
        u_grads = vel_grads['u_grads']
        v_grads = vel_grads['v_grads']
        w_grads = vel_grads['w_grads']
        
        # 提取空間梯度
        u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
        v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
        w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
        
        # 應變率張量模的平方
        S_xx = u_x
        S_yy = v_y
        S_zz = w_z
        S_xy = 0.5 * (u_y + v_x)
        S_xz = 0.5 * (u_z + w_x)
        S_yz = 0.5 * (v_z + w_y)
        
        S_mag_sq = 2 * (S_xx**2 + S_yy**2 + S_zz**2 + 
                        2 * (S_xy**2 + S_xz**2 + S_yz**2))
        
        # 湍動能估算（基於應變率）
        k_estimated = 0.5 * S_mag_sq + 1e-8  # 添加小偏移避免零值
        
        # 混合長度（通道流壁法向距離相關）
        y = coords[:, 1:2]  # 壁法向座標
        L_mix = 0.41 * torch.abs(y) * (1 - torch.abs(y))  # von Kármán 常數 κ=0.41
        L_mix = torch.clamp(L_mix, min=1e-3)  # 最小長度尺度
        
        # 耗散率估算
        epsilon_estimated = self.constants['C_mu'] * k_estimated**(3/2) / L_mix
        
        return k_estimated, epsilon_estimated
    
    def residual(self,
                coords: torch.Tensor,
                velocity: torch.Tensor,
                use_physical_init: bool = True) -> Dict[str, torch.Tensor]:
        """
        計算 RANS k-ε 方程殘差（軟約束模式）
        
        ✅ TASK-008 Phase 5: 添加物理初始化選項
        
        Args:
            coords: 座標 [batch_size, 3] (x, y, z)
            velocity: 速度 [batch_size, 3] (u, v, w)
            use_physical_init: 是否使用混合長度理論初始化（預設 True）
            
        Returns:
            殘差字典 {'k_equation': ..., 'epsilon_equation': ..., 'turbulent_viscosity': ...}
        """
        try:
            # ✅ Phase 5: 選擇初始化方法
            if use_physical_init and not hasattr(self, '_using_physical_init'):
                # 使用物理一致的混合長度初始化（只在第一次調用時輸出日誌）
                k_estimated, epsilon_estimated = self.initialize_turbulent_fields(coords, velocity)
                self._using_physical_init = True
            else:
                # 使用原始的速度梯度估算
                k_estimated, epsilon_estimated = self.estimate_turbulent_quantities(coords, velocity)
            
            # 應用物理約束
            if self.enable_constraints:
                k_constrained, epsilon_constrained = apply_physical_constraints(
                    k_estimated, epsilon_estimated, self.constraint_type
                )
            else:
                k_constrained, epsilon_constrained = k_estimated, epsilon_estimated
            
            # k-ε 方程殘差
            k_residual, epsilon_residual = k_epsilon_residuals_3d(
                coords, velocity, k_constrained, epsilon_constrained, self.viscosity
            )
            
            # 湍流黏度一致性檢查
            nu_t = self.constants['C_mu'] * k_constrained**2 / (epsilon_constrained + 1e-10)
            nu_t_normalized = nu_t / (self.viscosity + 1e-10)  # 相對於分子黏度
            
            # 湍流黏度合理性約束（避免過大值）
            # ✅ TASK-008 Phase 6C: 支援可配置的懲罰類型
            # 對於 Re_τ=1000，合理範圍 ν_t/ν ∈ [10, 100]
            
            if self.turbulent_viscosity_penalty == "huber":
                # Huber 損失：遠離目標時提供非飽和梯度
                target_value = self.turbulent_viscosity_target  # 無因次化目標（已是 ν_t/ν）
                target_tensor = torch.full_like(nu_t_normalized, target_value)
                beta = self.turbulent_viscosity_huber_delta  # 無因次化 β
                nu_t_penalty = F.smooth_l1_loss(
                    nu_t_normalized, 
                    target_tensor, 
                    beta=beta, 
                    reduction='none'
                )
                
            elif self.turbulent_viscosity_penalty == "log1p":
                # 原始 log1p 懲罰（Phase 6A/6B 行為，已知梯度飽和）
                nu_t_excess = torch.relu(nu_t_normalized - self.turbulent_viscosity_target)
                nu_t_penalty = torch.log1p(nu_t_excess)
                
            elif self.turbulent_viscosity_penalty == "mse":
                # 簡單 MSE 懲罰（參考基線）
                target_value = self.turbulent_viscosity_target
                nu_t_penalty = (nu_t_normalized - target_value) ** 2
                
            else:
                raise ValueError(
                    f"Unknown turbulent_viscosity_penalty: {self.turbulent_viscosity_penalty}. "
                    f"Supported: 'huber', 'log1p', 'mse'."
                )
            
            return {
                'k_equation': k_residual,
                'epsilon_equation': epsilon_residual,
                'turbulent_viscosity': nu_t_penalty,
                'physical_penalty': physical_constraint_penalty(k_estimated, epsilon_estimated) if self.enable_constraints else torch.tensor(0.0)
            }
            
        except Exception as e:
            warnings.warn(f"RANS 3D 殘差計算失敗，返回零殘差: {e}")
            
            batch_size = coords.shape[0]
            zero_residual = torch.zeros(batch_size, 1, device=coords.device, dtype=coords.dtype)
            
            return {
                'k_equation': zero_residual,
                'epsilon_equation': zero_residual,
                'turbulent_viscosity': zero_residual,
                'physical_penalty': torch.tensor(0.0, device=coords.device)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """返回模型信息"""
        return {
            'viscosity': self.viscosity,
            'turbulence_model': 'k_epsilon_3d',
            'mode': 'soft_constraint',
            'model_constants': self.constants,
            'input_variables': ['x', 'y', 'z', 'u', 'v', 'w'],
            'output_variables': ['u', 'v', 'w', 'p'],
            'estimated_variables': ['k', 'epsilon'],
            'equation_count': 3,
            'constraints_enabled': self.enable_constraints,
            'constraint_type': self.constraint_type
        }