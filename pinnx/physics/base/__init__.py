"""
Physics Base Modules
====================

提供物理方程求解器的基礎類別與共用工具函數。

模組結構：
- gradient_ops: 梯度計算工具 (支援2D/3D)
- laplacian_ops: 拉普拉斯算子計算
- pde_base: PDE求解器抽象基類
- ns_base: Navier-Stokes方程基類

作者：PINNs-MVP 團隊
日期：2025-12-15
"""

from .gradient_ops import (
    compute_gradient,
    compute_all_gradients,
    compute_gradient_safe,
    compute_gradient_checkpointed
)

from .laplacian_ops import (
    compute_laplacian,
    compute_laplacian_anisotropic
)

from .pde_base import PDEBase
from .ns_base import NavierStokesBase

__all__ = [
    # Gradient operations
    'compute_gradient',
    'compute_all_gradients', 
    'compute_gradient_safe',
    'compute_gradient_checkpointed',
    
    # Laplacian operations
    'compute_laplacian',
    'compute_laplacian_anisotropic',
    
    # Base classes
    'PDEBase',
    'NavierStokesBase',
]
