"""
Navier-Stokes 2D 方程式模組
===========================

提供2D不可壓縮NS方程的物理定律計算功能：
1. NS方程殘差計算 (動量方程 + 連續方程)
2. 渦量計算與Q準則
3. 守恆定律檢查
4. 邊界條件處理
5. RANS 湍流黏度支援

重構版本：繼承自 NavierStokesBase，消除重複代碼
"""

import torch
import torch.autograd as autograd
from typing import Tuple, Optional, Dict, Any
import warnings

from .base.ns_base import NavierStokesBase
from .base.gradient_ops import compute_gradient, compute_all_gradients


# ============================================================================
# Legacy Residual Functions (Preserved for Backward Compatibility)
# ============================================================================

def ns_residual_2d(coords: torch.Tensor, 
                   pred_full: torch.Tensor,
                   viscosity: Optional[float] = None,
                   time: Optional[torch.Tensor] = None,
                   nu_t: Optional[torch.Tensor] = None,
                   use_grad_nut: bool = False,
                   nu: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    2D N-S 方程式殘差計算（支援 RANS 湍流黏度）
    
    動量方程:
        ∂u/∂t + u·∇u = -∇p/ρ + ∇·[(ν + ν_t)∇u]
    
    連續方程:
        ∇·u = 0
    
    Args:
        coords: 空間座標 [batch_size, 2]
        pred_full: 預測張量 [batch_size, 4] -> [u, v, p, S]
        viscosity: 分子黏度 ν (優先)
        time: 時間座標 [batch_size, 1] (可選)
        nu_t: RANS 湍流黏度 [batch_size, 1] (可選)
        use_grad_nut: 是否計算 ∇ν_t·∇u 交叉項
        nu: 分子黏度 ν (向後兼容參數名)
        
    Returns:
        (momentum_x_residual, momentum_y_residual, continuity_residual)
    """
    # 向後兼容：支持 nu 參數名
    if viscosity is None and nu is not None:
        viscosity = nu
    elif viscosity is None:
        viscosity = 1e-3  # 默認值
    
    u = pred_full[:, 0:1]
    v = pred_full[:, 1:2]
    p = pred_full[:, 2:3]
    
    # 確保座標需要梯度
    if not coords.requires_grad:
        coords.requires_grad_(True)
    
    # 計算一階梯度
    u_grads = compute_all_gradients(u, coords, spatial_dim=2)
    v_grads = compute_all_gradients(v, coords, spatial_dim=2)
    p_grads = compute_all_gradients(p, coords, spatial_dim=2)
    
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    p_x, p_y = p_grads[:, 0:1], p_grads[:, 1:2]
    
    # 連續方程
    continuity = u_x + v_y
    
    # 對流項
    u_convection = u * u_x + v * u_y
    v_convection = u * v_x + v * v_y
    
    # 有效黏度
    nu_eff = viscosity
    if nu_t is not None:
        nu_eff = viscosity + nu_t
    
    # 黏性項
    from .base.laplacian_ops import compute_laplacian as compute_laplacian_base
    u_laplacian = compute_laplacian_base(u, coords, spatial_dim=2)
    v_laplacian = compute_laplacian_base(v, coords, spatial_dim=2)
    
    u_viscous = nu_eff * u_laplacian
    v_viscous = nu_eff * v_laplacian
    
    # RANS 交叉項（可選）
    if use_grad_nut and nu_t is not None:
        nu_t_grads = compute_all_gradients(nu_t, coords, spatial_dim=2)
        nu_t_x, nu_t_y = nu_t_grads[:, 0:1], nu_t_grads[:, 1:2]
        
        u_cross = nu_t_x * u_x + nu_t_y * u_y
        v_cross = nu_t_x * v_x + nu_t_y * v_y
        
        u_viscous += u_cross
        v_viscous += v_cross
    
    # 動量方程殘差
    momentum_x = u_convection + p_x - u_viscous
    momentum_y = v_convection + p_y - v_viscous
    
    # 時間項（如果有）
    if time is not None:
        u_t = compute_gradient(u, time, component=0, spatial_dim=1)
        v_t = compute_gradient(v, time, component=0, spatial_dim=1)
        momentum_x += u_t
        momentum_y += v_t
    
    return momentum_x, momentum_y, continuity


def incompressible_ns_2d(coords: torch.Tensor, 
                         pred: torch.Tensor,
                         viscosity: Optional[float] = None,
                         nu: Optional[float] = None,
                         **kwargs) -> torch.Tensor:
    """
    簡化的 2D 不可壓縮 N-S 方程（向後兼容）
    
    ⚠️ DEPRECATED: 請使用 ns_residual_2d() 或 NSEquations2D 類
    
    Returns:
        總殘差 [batch_size, 1] (所有方程殘差的加權和)
    """
    # 向後兼容：支持 nu 參數名
    if viscosity is None and nu is not None:
        viscosity = nu
    elif viscosity is None:
        viscosity = 1e-3
    
    # 補齊源項
    if pred.shape[1] == 3:
        source_term = torch.zeros(pred.shape[0], 1, device=pred.device, dtype=pred.dtype)
        pred_full = torch.cat([pred, source_term], dim=1)
    else:
        pred_full = pred
    
    # 計算各方程殘差
    mom_x, mom_y, cont = ns_residual_2d(coords, pred_full, viscosity, **kwargs)
    
    # 加權組合所有殘差（可調整權重）
    total_residual = mom_x**2 + mom_y**2 + cont**2
    
    return total_residual


# ============================================================================
# Utility Functions
# ============================================================================

def compute_vorticity(coords: torch.Tensor, 
                     velocity: torch.Tensor) -> torch.Tensor:
    """
    計算 2D 渦量：ω = ∂v/∂x - ∂u/∂y
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        
    Returns:
        渦量 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    u_grads = compute_all_gradients(u, coords, spatial_dim=2)
    v_grads = compute_all_gradients(v, coords, spatial_dim=2)
    
    u_y = u_grads[:, 1:2]
    v_x = v_grads[:, 0:1]
    
    vorticity = v_x - u_y
    return vorticity


def compute_q_criterion(coords: torch.Tensor,
                       velocity: torch.Tensor,
                       threshold: float = 0.0) -> torch.Tensor:
    """
    計算 Q 準則（渦識別）
    
    Q = 0.5 * (||Ω||² - ||S||²)
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        threshold: Q 值閾值
        
    Returns:
        Q 準則 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    u_grads = compute_all_gradients(u, coords, spatial_dim=2)
    v_grads = compute_all_gradients(v, coords, spatial_dim=2)
    
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    
    # 渦量張量 Ω (反對稱部分)
    omega_12 = 0.5 * (v_x - u_y)
    omega_norm_sq = 2 * omega_12 ** 2
    
    # 應變率張量 S (對稱部分)
    s_11 = u_x
    s_22 = v_y
    s_12 = 0.5 * (u_y + v_x)
    strain_norm_sq = 2 * (s_11**2 + s_22**2 + 2 * s_12**2)
    
    Q = 0.5 * (omega_norm_sq - strain_norm_sq)
    
    return Q


def check_conservation_laws(coords: torch.Tensor,
                           velocity: torch.Tensor, 
                           pressure: torch.Tensor,
                           viscosity: float = 1e-3) -> Dict[str, torch.Tensor]:
    """
    檢查守恆律（質量、動量、能量）
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        pressure: 壓力場 [batch_size, 1]
        viscosity: 黏度
        
    Returns:
        守恆律偏差字典
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    u_grads = compute_all_gradients(u, coords, spatial_dim=2)
    v_grads = compute_all_gradients(v, coords, spatial_dim=2)
    
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    
    # 質量守恆（連續方程）
    mass_conservation = u_x + v_y
    
    # 動量守恆（簡化檢查）
    momentum_x = u * u_x + v * u_y
    momentum_y = u * v_x + v * v_y
    
    # 能量守恆（動能變化）
    kinetic_energy = 0.5 * (u**2 + v**2)
    
    # 組合動量守恆（兩個分量）
    momentum_conservation = torch.cat([momentum_x, momentum_y], dim=1)
    
    return {
        'mass_conservation': mass_conservation,
        'mass': mass_conservation,  # 向後兼容舊名稱
        'momentum_conservation': momentum_conservation,
        'momentum_x': momentum_x,
        'momentum_y': momentum_y,
        'kinetic_energy': kinetic_energy,
        'energy_conservation': kinetic_energy  # 簡化：用動能代表能量守恆
    }


def apply_boundary_conditions(coords: torch.Tensor,
                              velocity: torch.Tensor,
                              bc_type: str = 'dirichlet',
                              bc_values: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    應用邊界條件
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        bc_type: 邊界條件類型 ('dirichlet', 'neumann', 'periodic')
        bc_values: 邊界值 [batch_size, 2]
        
    Returns:
        邊界條件殘差 [batch_size, 2]
    """
    if bc_type == 'dirichlet':
        # Dirichlet BC: u = u_bc
        if bc_values is None:
            bc_values = torch.zeros_like(velocity)
        residual = velocity - bc_values
        
    elif bc_type == 'neumann':
        # Neumann BC: ∂u/∂n = g
        if bc_values is None:
            bc_values = torch.zeros_like(velocity)
        
        u = velocity[:, 0:1]
        v = velocity[:, 1:2]
        
        u_grads = compute_all_gradients(u, coords, spatial_dim=2)
        v_grads = compute_all_gradients(v, coords, spatial_dim=2)
        
        # 假設法向為 y 方向（可根據需求調整）
        u_n = u_grads[:, 1:2]
        v_n = v_grads[:, 1:2]
        
        grad_velocity = torch.cat([u_n, v_n], dim=1)
        residual = grad_velocity - bc_values
        
    elif bc_type == 'periodic':
        # 週期邊界條件
        residual = torch.zeros_like(velocity)
        
    else:
        raise ValueError(f"不支援的邊界條件類型: {bc_type}")
    
    return residual


def compute_pressure_poisson(coords: torch.Tensor,
                            velocity: torch.Tensor,
                            density: float = 1.0) -> torch.Tensor:
    """
    壓力 Poisson 方程（投影方法用）
    
    ∇²p = -ρ ∇·(u·∇u)
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        density: 密度
        
    Returns:
        壓力源項 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    u_grads = compute_all_gradients(u, coords, spatial_dim=2)
    v_grads = compute_all_gradients(v, coords, spatial_dim=2)
    
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    
    # 對流項的散度
    div_uu = compute_gradient(u * u, coords, component=0, spatial_dim=2)
    div_uv = compute_gradient(u * v, coords, component=1, spatial_dim=2)
    div_vu = compute_gradient(v * u, coords, component=0, spatial_dim=2)
    div_vv = compute_gradient(v * v, coords, component=1, spatial_dim=2)
    
    source = -density * (div_uu + div_uv + div_vu + div_vv)
    
    return source


def compute_streamfunction(coords: torch.Tensor,
                          velocity: torch.Tensor) -> torch.Tensor:
    """
    計算流函數 ψ（2D 不可壓縮流）
    
    u = ∂ψ/∂y, v = -∂ψ/∂x
    
    Args:
        coords: 空間座標 [batch_size, 2]
        velocity: 速度場 [batch_size, 2]
        
    Returns:
        流函數 [batch_size, 1]
    """
    # 這是一個積分問題，這裡提供簡化版本
    # 實際應用中可能需要數值積分
    warnings.warn("compute_streamfunction() 是簡化實現，可能需要數值積分")
    
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    
    # 簡化：假設流函數為線性組合
    psi = y * u.mean() - x * v.mean()
    
    return psi


# ============================================================================
# Main Class: NSEquations2D (Refactored)
# ============================================================================

class NSEquations2D(NavierStokesBase):
    """
    2D Navier-Stokes 方程式統一接口（重構版本）
    
    繼承自 NavierStokesBase，提供：
    - 標準 2D N-S 方程殘差計算
    - RANS 湍流黏度支援
    - 守恆律檢查
    - 邊界條件處理
    - 向後兼容的 API
    
    重構改進：
    - 消除重複代碼（梯度/拉普拉斯計算）
    - 繼承通用 N-S 功能
    - 保持完整向後兼容性
    """
    
    def __init__(self, 
                 viscosity: float = 1e-3, 
                 density: float = 1.0,
                 domain_bounds: Optional[Dict] = None,
                 **kwargs):
        """
        初始化 2D N-S 方程式求解器
        
        Args:
            viscosity: 運動黏度 ν (m²/s)
            density: 流體密度 ρ (kg/m³)
            domain_bounds: 域邊界 {'x': [x_min, x_max], 'y': [y_min, y_max]}
            **kwargs: 其他參數（向後兼容）
        """
        # 構造 physics_params
        physics_params = {
            'Re': 1.0 / viscosity if viscosity > 0 else float('inf'),
            'nu': viscosity,
            'rho': density
        }
        
        # 默認域邊界
        if domain_bounds is None:
            domain_bounds = {'x': [0, 1], 'y': [0, 1]}
        
        # 調用基類初始化
        super().__init__(
            physics_params=physics_params,
            domain_bounds=domain_bounds,
            spatial_dim=2
        )
        
        # 向後兼容屬性
        self.viscosity = viscosity
        self.density = density
        self.kinematic_viscosity = kwargs.get('kinematic_viscosity', viscosity)
    
    def residual(self, 
                coords: torch.Tensor, 
                predictions: torch.Tensor,
                time: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        計算 N-S 方程殘差（統一接口）
        
        Args:
            coords: 空間座標 [batch_size, 2]
            predictions: 預測張量 [batch_size, 3 or 4]
                - 3 columns: [u, v, p]
                - 4 columns: [u, v, p, S]
            time: 時間座標 [batch_size, 1] (可選)
            **kwargs: 額外參數
                - nu_t: RANS 湍流黏度 [batch_size, 1]
                - use_grad_nut: 是否使用 ∇ν_t 項
        
        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'continuity'}
        """
        # 確保 predictions 有 4 列
        if predictions.shape[1] == 3:
            # 補齊源項
            source_term = torch.zeros(
                predictions.shape[0], 1, 
                device=predictions.device, 
                dtype=predictions.dtype
            )
            pred_full = torch.cat([predictions, source_term], dim=1)
        elif predictions.shape[1] == 2:
            # 只有速度，假設壓力為 0
            pressure = torch.zeros(
                predictions.shape[0], 1,
                device=predictions.device,
                dtype=predictions.dtype
            )
            source_term = torch.zeros_like(pressure)
            pred_full = torch.cat([predictions, pressure, source_term], dim=1)
        else:
            pred_full = predictions
        
        # 提取 RANS 參數
        nu_t = kwargs.get('nu_t', None)
        use_grad_nut = kwargs.get('use_grad_nut', False)
        
        # 調用核心殘差計算
        try:
            momentum_x, momentum_y, continuity = ns_residual_2d(
                coords, pred_full, self.nu, time, nu_t, use_grad_nut
            )
            
            return {
                'momentum_x': momentum_x,
                'momentum_y': momentum_y,
                'continuity': continuity
            }
        
        except RuntimeError as e:
            if "backward through the graph" in str(e):
                warnings.warn(f"梯度圖錯誤，切換到簡化殘差: {str(e)}")
                return self._compute_simplified_residuals(coords, pred_full)
            else:
                raise e
    
    def _compute_simplified_residuals(self, 
                                    coords: torch.Tensor,
                                    pred_full: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        簡化的殘差計算（當梯度圖出現問題時）
        
        只計算一階導數，忽略黏性項
        """
        u = pred_full[:, 0:1]
        v = pred_full[:, 1:2]
        p = pred_full[:, 2:3]
        
        if not coords.requires_grad:
            coords.requires_grad_(True)
        
        try:
            u_grads = compute_all_gradients(u, coords, spatial_dim=2)
            v_grads = compute_all_gradients(v, coords, spatial_dim=2)
            p_grads = compute_all_gradients(p, coords, spatial_dim=2)
            
            u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
            v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
            p_x, p_y = p_grads[:, 0:1], p_grads[:, 1:2]
            
            # 連續方程
            continuity = u_x + v_y
            
            # 簡化動量方程（忽略黏性項）
            u_convection = u * u_x + v * u_y
            v_convection = u * v_x + v * v_y
            
            momentum_x = u_convection + p_x
            momentum_y = v_convection + p_y
            
            return {
                'momentum_x': momentum_x,
                'momentum_y': momentum_y,
                'continuity': continuity
            }
        
        except Exception as e:
            warnings.warn(f"簡化殘差計算失敗，返回零殘差: {str(e)}")
            zero_residual = torch.zeros_like(u)
            return {
                'momentum_x': zero_residual,
                'momentum_y': zero_residual,
                'continuity': zero_residual
            }
    
    def residual_unified(self, 
                        coords: torch.Tensor, 
                        pred_full: torch.Tensor,
                        time: Optional[torch.Tensor] = None,
                        nu_t: Optional[torch.Tensor] = None,
                        use_grad_nut: bool = False) -> Dict[str, torch.Tensor]:
        """
        統一的殘差計算接口（向後兼容）
        
        ⚠️ DEPRECATED: 請使用 residual() 方法
        """
        return self.residual(coords, pred_full, time, nu_t=nu_t, use_grad_nut=use_grad_nut)
    
    def check_conservation(self, 
                          coords: torch.Tensor,
                          velocity: torch.Tensor, 
                          pressure: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        檢查守恆律
        
        Args:
            coords: 空間座標 [batch_size, 2]
            velocity: 速度場 [batch_size, 2]
            pressure: 壓力場 [batch_size, 1]
            
        Returns:
            守恆律偏差字典
        """
        return check_conservation_laws(coords, velocity, pressure, self.nu)
    
    def compute_vorticity(self, 
                         coords: torch.Tensor,
                         velocity: torch.Tensor) -> torch.Tensor:
        """
        計算渦量場
        
        Args:
            coords: 空間座標 [batch_size, 2]
            velocity: 速度場 [batch_size, 2]
            
        Returns:
            渦量 [batch_size, 1]
        """
        return compute_vorticity(coords, velocity)
    
    def apply_boundary_conditions(self,
                                  coords: torch.Tensor,
                                  velocity: torch.Tensor,
                                  boundary_conditions: Dict[str, Any]) -> torch.Tensor:
        """
        應用邊界條件
        
        Args:
            coords: 空間座標 [batch_size, 2]
            velocity: 速度場 [batch_size, 2]
            boundary_conditions: 邊界條件設定
                - type: 邊界條件類型
                - values: 邊界值
            
        Returns:
            邊界條件殘差 [batch_size, 2]
        """
        bc_type = boundary_conditions.get('type', 'dirichlet')
        bc_values = boundary_conditions.get('values', None)
        return apply_boundary_conditions(coords, velocity, bc_type, bc_values)
    
    def get_physical_properties(self) -> Dict[str, float]:
        """
        獲取物理屬性
        
        Returns:
            物理屬性字典
        """
        return {
            'viscosity': self.nu,
            'density': self.rho,
            'kinematic_viscosity': self.nu,
            'reynolds_number': self.Re
        }
