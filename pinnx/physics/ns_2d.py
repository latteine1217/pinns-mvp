"""
Navier-Stokes 2D 方程式模組
===========================

提供2D不可壓縮NS方程的物理定律計算功能：
1. NS方程殘差計算 (動量方程 + 連續方程)
2. 渦量計算與Q準則
3. 自動微分梯度計算
4. 守恆定律檢查
5. 邊界條件處理
"""

import torch
import torch.autograd as autograd
from typing import Tuple, Optional, Dict, Any
import warnings

def compute_derivatives(f: torch.Tensor, x: torch.Tensor, 
                       order: int = 1) -> torch.Tensor:
    """
    使用自動微分計算函數對座標的偏微分
    
    Args:
        f: 待微分的標量場 [batch_size, 1]
        x: 座標變數 [batch_size, spatial_dim] 
        order: 微分階數 (1 或 2)
        
    Returns:
        偏微分結果 [batch_size, spatial_dim] (一階) 或 
                 [batch_size, spatial_dim] (二階對角項)
    """
    if not f.requires_grad:
        f.requires_grad_(True)
    if not x.requires_grad:
        x.requires_grad_(True)
        
    # 計算一階偏微分
    grad_outputs = torch.ones_like(f)
    first_derivs = autograd.grad(
        outputs=f, 
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )[0]
    
    if order == 1:
        return first_derivs
    
    elif order == 2:
        # 計算二階偏微分 (拉普拉斯算子的對角項)
        second_derivs = []
        for i in range(x.shape[1]):
            second_deriv = autograd.grad(
                outputs=first_derivs[:, i].sum(),
                inputs=x,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0][:, i]
            second_derivs.append(second_deriv)
        
        return torch.stack(second_derivs, dim=1)
    
    else:
        raise ValueError(f"不支援的微分階數: {order}")

def compute_laplacian(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    計算拉普拉斯算子 ∇²f = ∂²f/∂x² + ∂²f/∂y²
    
    Args:
        f: 標量場 [batch_size, 1]
        x: 座標 [batch_size, 2] -> [x, y]
        
    Returns:
        拉普拉斯算子結果 [batch_size, 1]
    """
    # 計算二階偏微分
    second_derivs = compute_derivatives(f, x, order=2)
    
    # 對所有空間方向求和 (∇² = ∂²/∂x² + ∂²/∂y² + ...)
    laplacian = torch.sum(second_derivs, dim=1, keepdim=True)
    
    return laplacian

def ns_residual_2d(coords: torch.Tensor, 
                   pred: torch.Tensor,
                   nu: float,
                   time: Optional[torch.Tensor] = None,
                   source_term: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    計算2D不可壓縮Navier-Stokes方程殘差
    
    控制方程：
    ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u + S_x  (x-動量)
    ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v + S_y  (y-動量) 
    ∂u/∂x + ∂v/∂y = 0                                (連續方程)
    
    Args:
        coords: 空間座標 [batch_size, 2] -> [x, y]
        pred: 預測結果 [batch_size, 4] -> [u, v, p, S] 
              其中 S 為源項或等效閉合量
        nu: 動力黏性係數
        time: 時間座標 [batch_size, 1] (非定常流場)
        source_term: 外部源項 [batch_size, 2] (可選)
        
    Returns:
        Tuple of (x_momentum_residual, y_momentum_residual, continuity_residual)
        每個都是 [batch_size, 1]
    """
    # 確保輸入需要梯度計算
    if not coords.requires_grad:
        coords.requires_grad_(True)
    if time is not None and not time.requires_grad:
        time.requires_grad_(True)
    
    # 分解預測變數
    u = pred[:, 0:1]  # x方向速度
    v = pred[:, 1:2]  # y方向速度  
    p = pred[:, 2:3]  # 壓力
    S = pred[:, 3:4]  # 源項/閉合項
    
    # 計算速度的空間偏微分
    u_derivs = compute_derivatives(u, coords, order=1)  # [∂u/∂x, ∂u/∂y]
    v_derivs = compute_derivatives(v, coords, order=1)  # [∂v/∂x, ∂v/∂y]
    p_derivs = compute_derivatives(p, coords, order=1)  # [∂p/∂x, ∂p/∂y]
    
    u_x, u_y = u_derivs[:, 0:1], u_derivs[:, 1:2]
    v_x, v_y = v_derivs[:, 0:1], v_derivs[:, 1:2]  
    p_x, p_y = p_derivs[:, 0:1], p_derivs[:, 1:2]
    
    # 計算拉普拉斯算子 (黏性項)
    u_laplacian = compute_laplacian(u, coords)
    v_laplacian = compute_laplacian(v, coords)
    
    # 時間導數 (非定常情況)
    if time is not None:
        u_t = compute_derivatives(u, time, order=1)
        v_t = compute_derivatives(v, time, order=1) 
    else:
        u_t = torch.zeros_like(u)
        v_t = torch.zeros_like(v)
    
    # 對流項 (非線性項)
    u_convection = u * u_x + v * u_y  # u·∇u 的 x 分量
    v_convection = u * v_x + v * v_y  # u·∇v 的 y 分量
    
    # 源項處理
    if source_term is not None:
        S_x = source_term[:, 0:1]
        S_y = source_term[:, 1:2]
    else:
        # 使用預測的源項 (可以是x和y的混合或只有x方向)
        S_x = S
        S_y = torch.zeros_like(S)  # 假設源項主要在x方向
    
    # NS方程殘差計算
    # x-動量方程: ∂u/∂t + u∂u/∂x + v∂u/∂y + ∂p/∂x - ν∇²u - S_x = 0
    momentum_x = u_t + u_convection + p_x - nu * u_laplacian - S_x
    
    # y-動量方程: ∂v/∂t + u∂v/∂x + v∂v/∂y + ∂p/∂y - ν∇²v - S_y = 0  
    momentum_y = v_t + v_convection + p_y - nu * v_laplacian - S_y
    
    # 連續方程 (不可壓縮): ∂u/∂x + ∂v/∂y = 0
    continuity = u_x + v_y
    
    return momentum_x, momentum_y, continuity

def incompressible_ns_2d(coords: torch.Tensor, 
                         pred: torch.Tensor,
                         nu: float,
                         **kwargs) -> torch.Tensor:
    """
    簡化的不可壓縮NS方程殘差計算 (定常流場)
    
    Returns:
        總殘差 [batch_size, 1] (所有方程殘差的加權和)
    """
    mom_x, mom_y, cont = ns_residual_2d(coords, pred, nu, **kwargs)
    
    # 加權組合所有殘差 (可調整權重)
    total_residual = mom_x**2 + mom_y**2 + cont**2
    
    return total_residual

def compute_vorticity(coords: torch.Tensor, 
                     velocity: torch.Tensor) -> torch.Tensor:
    """
    計算2D渦量 ω = ∂v/∂x - ∂u/∂y
    
    Args:
        coords: 座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2] -> [u, v]
        
    Returns:
        渦量 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    u_derivs = compute_derivatives(u, coords)
    v_derivs = compute_derivatives(v, coords)
    
    u_y = u_derivs[:, 1:2]  # ∂u/∂y
    v_x = v_derivs[:, 0:1]  # ∂v/∂x
    
    vorticity = v_x - u_y
    
    return vorticity

def compute_q_criterion(coords: torch.Tensor,
                       velocity: torch.Tensor) -> torch.Tensor:
    """
    計算Q準則 (用於識別渦結構)
    Q = 0.5 * (Ω² - S²)
    其中 Ω 是渦量張量的模長，S 是應變率張量的模長
    
    Args:
        coords: 座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        
    Returns:
        Q準則值 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    # 計算速度梯度
    u_derivs = compute_derivatives(u, coords)
    v_derivs = compute_derivatives(v, coords)
    
    u_x, u_y = u_derivs[:, 0:1], u_derivs[:, 1:2]
    v_x, v_y = v_derivs[:, 0:1], v_derivs[:, 1:2]
    
    # 渦量張量 Ω
    omega_12 = 0.5 * (v_x - u_y)  # 反對稱部分
    omega_squared = 2 * omega_12**2
    
    # 應變率張量 S 
    s_11 = u_x                      # 正常應變率
    s_22 = v_y                      # 正常應變率
    s_12 = 0.5 * (u_y + v_x)      # 剪切應變率
    
    s_squared = 2 * (s_11**2 + s_22**2 + 2 * s_12**2)
    
    # Q準則
    q_criterion = 0.5 * (omega_squared - s_squared)
    
    return q_criterion

def check_conservation_laws(coords: torch.Tensor,
                           velocity: torch.Tensor,
                           pressure: torch.Tensor,
                           nu: float = 1e-3,
                           **kwargs) -> Dict[str, torch.Tensor]:
    """
    檢查守恆定律是否滿足
    
    Args:
        coords: 座標點 [batch_size, 2]
        velocity: 速度場 [batch_size, 2] -> [u, v]
        pressure: 壓力場 [batch_size, 1]
        nu: 黏性係數
        
    Returns:
        守恆定律檢查結果字典，包含張量值
    """
    results = {}
    
    # 構建完整的預測張量 [u, v, p, 0] (源項設為0)
    batch_size = coords.shape[0]
    pred = torch.cat([
        velocity,  # [u, v]
        pressure,  # [p]
        torch.zeros(batch_size, 1, device=coords.device)  # [S] 假設源項為0
    ], dim=1)
    
    # 質量守恆 (連續方程)
    _, _, continuity = ns_residual_2d(coords, pred, nu)
    mass_conservation_error = torch.mean(torch.abs(continuity))
    results['mass_conservation'] = mass_conservation_error
    
    # 動量守恆 (動量方程殘差)
    mom_x, mom_y, _ = ns_residual_2d(coords, pred, nu)
    momentum_conservation_error = torch.mean(torch.abs(mom_x) + torch.abs(mom_y))
    results['momentum_conservation'] = momentum_conservation_error
    
    # 能量守恆 (簡化：動能梯度的均值作為能量守恆指標)
    u, v = velocity[:, 0:1], velocity[:, 1:2]
    kinetic_energy = 0.5 * (u**2 + v**2)
    energy_gradients = compute_derivatives(kinetic_energy, coords)
    energy_conservation_error = torch.mean(torch.abs(energy_gradients))
    results['energy_conservation'] = energy_conservation_error
    
    return results

def apply_boundary_conditions(coords: torch.Tensor,
                            pred: torch.Tensor,
                            bc_type: str = "dirichlet",
                            bc_values: Optional[Dict] = None) -> torch.Tensor:
    """
    應用邊界條件
    
    Args:
        coords: 邊界座標點
        pred: 預測值
        bc_type: 邊界條件類型 ("dirichlet", "neumann", "periodic")
        bc_values: 邊界條件數值
        
    Returns:
        邊界條件殘差
    """
    if bc_type == "dirichlet":
        # 狄利克雷邊界條件: 指定函數值
        if bc_values is None:
            bc_values = {"u": 0.0, "v": 0.0}  # 預設無滑移條件
        
        u_pred = pred[:, 0:1]
        v_pred = pred[:, 1:2]
        
        u_bc_error = u_pred - bc_values.get("u", 0.0)
        v_bc_error = v_pred - bc_values.get("v", 0.0)
        
        return torch.cat([u_bc_error, v_bc_error], dim=1)
    
    elif bc_type == "neumann":
        # 諾依曼邊界條件: 指定法向梯度
        # TODO: 實現法向梯度計算
        warnings.warn("Neumann邊界條件尚未完全實現")
        return torch.zeros_like(pred[:, :2])
    
    elif bc_type == "periodic":
        # 週期邊界條件
        # TODO: 實現週期性檢查
        warnings.warn("週期邊界條件尚未完全實現")
        return torch.zeros_like(pred[:, :2])
    
    else:
        raise ValueError(f"不支援的邊界條件類型: {bc_type}")

# 物理場計算工具函數
def compute_pressure_poisson(coords: torch.Tensor,
                           velocity: torch.Tensor,
                           nu: float) -> torch.Tensor:
    """
    根據速度場計算壓力泊松方程
    ∇²p = -∇·(u·∇u) - ∇·(v·∇v)
    
    用於壓力場的物理一致性檢查
    """
    # TODO: 實現壓力泊松方程求解
    pass

def compute_streamfunction(coords: torch.Tensor,
                          velocity: torch.Tensor) -> torch.Tensor:
    """
    根據速度場計算流函數 (2D不可壓縮流場)
    u = ∂ψ/∂y, v = -∂ψ/∂x
    """
    # TODO: 實現流函數計算
    pass


class NSEquations2D:
    """
    2D Navier-Stokes 方程式統一接口
    
    提供面向對象的N-S方程式操作，包括殘差計算、守恆律檢查、
    邊界條件處理等核心功能。
    """
    
    def __init__(self, viscosity: float = 1e-3, density: float = 1.0, **kwargs):
        """
        Args:
            viscosity: 運動黏度 ν (m²/s)
            density: 流體密度 ρ (kg/m³)
            **kwargs: 其他參數（用於測試相容性）
        """
        self.viscosity = viscosity
        self.density = density
        
        # 測試相容性參數
        self.kinematic_viscosity = kwargs.get('kinematic_viscosity', viscosity)
        self.nu = kwargs.get('nu', viscosity)  # 別名
        self.rho = kwargs.get('rho', density)  # 別名
    
    def residual(self, 
                coords: torch.Tensor, 
                velocity: torch.Tensor, 
                pressure: torch.Tensor,
                time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        計算 N-S 方程式殘差
        
        Args:
            coords: 空間座標 [batch_size, spatial_dim]
            velocity: 速度場 [batch_size, velocity_dim] 
            pressure: 壓力場 [batch_size, 1]
            time: 時間座標 [batch_size, 1] (可選)
            
        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'continuity'}
        """
        return ns_residual_2d(coords, velocity, pressure, self.viscosity, time)
    
    def check_conservation(self, 
                          coords: torch.Tensor,
                          velocity: torch.Tensor, 
                          pressure: torch.Tensor) -> Dict[str, float]:
        """
        檢查守恆律
        
        Args:
            coords: 空間座標
            velocity: 速度場  
            pressure: 壓力場
            
        Returns:
            守恆律偏差字典
        """
        return check_conservation_laws(coords, velocity, pressure)
    
    def compute_vorticity(self, 
                         coords: torch.Tensor,
                         velocity: torch.Tensor) -> torch.Tensor:
        """
        計算渦量場
        
        Args:
            coords: 空間座標
            velocity: 速度場
            
        Returns:
            渦量 [batch_size, 1]
        """
        return compute_vorticity(coords, velocity, velocity.shape[-1])
    
    def apply_boundary_conditions(self,
                                 coords: torch.Tensor,
                                 velocity: torch.Tensor,
                                 boundary_conditions: Dict[str, Any]) -> torch.Tensor:
        """
        應用邊界條件
        
        Args:
            coords: 空間座標
            velocity: 速度場
            boundary_conditions: 邊界條件設定
            
        Returns:
            邊界條件殘差
        """
        return apply_boundary_conditions(coords, velocity, boundary_conditions)
    
    def get_physical_properties(self) -> Dict[str, float]:
        """
        獲取物理屬性
        
        Returns:
            物理屬性字典
        """
        return {
            'viscosity': self.viscosity,
            'density': self.density,
            'kinematic_viscosity': self.kinematic_viscosity,
            'reynolds_number': 1.0 / self.viscosity  # 特徵Re數 (假設特徵速度和長度為1)
        }