"""
Physics 模組初始化檔案
=====================

物理定律計算與數值方法模組
包含：
- NS方程殘差計算 (ns_2d.py, ns_3d_temporal.py)
- RANS湍流方程 (turbulence.py)
- VS-PINN變數尺度化 (scaling.py)  
- 物理量守恆檢查
- 數值微分算子
"""

from .ns_2d import (
    ns_residual_2d,
    incompressible_ns_2d,
    compute_vorticity,
    compute_q_criterion,
    compute_derivatives,
    check_conservation_laws
)

try:
    from .ns_3d_temporal import (
        NSEquations3DTemporal,
        compute_derivatives_3d_temporal
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Cannot import ns_3d_temporal module: {e}")
    NSEquations3DTemporal = None
    compute_derivatives_3d_temporal = None

# 3D Thin-Slab NS方程模組 (用於通道流等薄片配置)
try:
    from .ns_3d_thin_slab import (
        NSEquations3DThinSlab,
        ns_residual_3d_thin_slab,
        compute_derivatives_3d,
        apply_periodic_bc_3d,
        apply_wall_bc_3d,
        check_conservation_3d
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Cannot import ns_3d_thin_slab module: {e}")
    NSEquations3DThinSlab = None
    ns_residual_3d_thin_slab = None
    compute_derivatives_3d = None
    apply_periodic_bc_3d = None
    apply_wall_bc_3d = None
    check_conservation_3d = None

from .turbulence import (
    rans_momentum_residual,
    continuity_residual,
    k_epsilon_residuals,
    RANSEquations2D
)

from .scaling import (
    VSScaler,
    create_scaler_from_data,
    denormalize_gradients
)

# VS-PINN通道流模組
try:
    from .vs_pinn_channel_flow import (
        VSPINNChannelFlow,
        create_vs_pinn_channel_flow
    )
except ImportError as e:
    import warnings
    warnings.warn(f"Cannot import vs_pinn_channel_flow module: {e}")
    VSPINNChannelFlow = None
    create_vs_pinn_channel_flow = None

# 物理常數與參數
class PhysicsConstants:
    """常用物理常數"""
    # 流體性質 (水，標準條件)
    WATER_DENSITY = 1000.0      # kg/m³
    WATER_VISCOSITY = 1.0e-6    # m²/s (動力黏度)
    
    # 空氣性質 (標準條件)  
    AIR_DENSITY = 1.225         # kg/m³
    AIR_VISCOSITY = 1.5e-5      # m²/s
    
    # 數值計算參數
    EPSILON = 1e-12             # 數值穩定性參數
    PI = 3.14159265359          # 圓周率

# 通用物理工具函數
def reynolds_number(velocity_scale, length_scale, kinematic_viscosity):
    """計算雷諾數 Re = UL/ν"""
    return velocity_scale * length_scale / kinematic_viscosity

def friction_velocity(wall_shear_stress, density):
    """計算摩擦速度 u_τ = sqrt(τ_w/ρ)"""
    import torch
    return torch.sqrt(wall_shear_stress / density)

def plus_units(y, u_tau, nu):
    """轉換為壁面單位 y+ = y*u_τ/ν"""
    return y * u_tau / nu

def check_cfl_condition(velocity, grid_spacing, time_step):
    """檢查CFL條件 CFL = u*dt/dx < 1"""
    import torch
    cfl = torch.abs(velocity) * time_step / grid_spacing
    return torch.max(cfl)

__all__ = [
    # NS方程相關
    'ns_residual_2d', 'incompressible_ns_2d', 'compute_vorticity', 
    'compute_q_criterion', 'compute_derivatives', 'check_conservation_laws',
    
    # 3D時間依賴NS方程
    'NSEquations3DTemporal', 'compute_derivatives_3d_temporal',
    
    # 3D Thin-Slab NS方程
    'NSEquations3DThinSlab', 'ns_residual_3d_thin_slab', 
    'compute_derivatives_3d', 'apply_periodic_bc_3d', 
    'apply_wall_bc_3d', 'check_conservation_3d',
    
    # RANS湍流方程相關
    'rans_momentum_residual', 'continuity_residual', 'k_epsilon_residuals', 'RANSEquations2D',
    
    # 尺度化相關
    'VSScaler', 
    'create_scaler_from_data', 'denormalize_gradients',
    
    # VS-PINN通道流
    'VSPINNChannelFlow', 'create_vs_pinn_channel_flow',
    
    # 物理工具
    'PhysicsConstants', 'reynolds_number', 'friction_velocity', 
    'plus_units', 'check_cfl_condition'
]