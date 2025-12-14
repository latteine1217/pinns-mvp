"""
Laplacian Operations Module
============================

提供拉普拉斯算子計算工具，支援2D/3D場與各向異性縮放。

主要功能：
1. compute_laplacian() - 統一的拉普拉斯算子計算 (2D/3D)
2. compute_laplacian_anisotropic() - 各向異性拉普拉斯（VS-PINN用）

理論基礎：
- 拉普拉斯算子：∇²f = Σ ∂²f/∂x_i²
- 各向異性縮放：∇²f = Σ N_i² ∂²f/∂X_i² (VS-PINN)

作者：PINNs-MVP 團隊
日期：2025-12-15
"""

import torch
from typing import Optional
from .gradient_ops import compute_gradient, compute_second_derivative


def compute_laplacian(
    field: torch.Tensor,
    coords: torch.Tensor,
    spatial_dim: int = 2,
    stabilize: bool = False,
    max_value: float = 1e4
) -> torch.Tensor:
    """
    統一的拉普拉斯算子計算 (支援2D/3D)
    
    計算：∇²f = ∂²f/∂x² + ∂²f/∂y² (+ ∂²f/∂z² for 3D)
    
    Args:
        field: 標量場 [batch_size, 1]
        coords: 座標 [batch_size, spatial_dim]
        spatial_dim: 空間維度 (2 for 2D, 3 for 3D)
        stabilize: 是否應用數值穩定化 (截斷極值)
        max_value: 穩定化時的截斷閾值
        
    Returns:
        拉普拉斯算子結果 [batch_size, 1]
        
    Example:
        >>> # 2D 拉普拉斯
        >>> coords = torch.randn(100, 2, requires_grad=True)
        >>> field = coords[:, 0:1]**2 + coords[:, 1:2]**2  # f = x² + y²
        >>> laplacian = compute_laplacian(field, coords, spatial_dim=2)
        >>> # 解析解：∇²f = 2 + 2 = 4
        
        >>> # 3D 拉普拉斯
        >>> coords_3d = torch.randn(100, 3, requires_grad=True)
        >>> field_3d = torch.sum(coords_3d**2, dim=1, keepdim=True)
        >>> laplacian_3d = compute_laplacian(field_3d, coords_3d, spatial_dim=3)
        >>> # 解析解：∇²f = 2 + 2 + 2 = 6
    """
    # 計算所有方向的二階偏導數
    second_derivs = []
    for i in range(spatial_dim):
        d2f_dxi2 = compute_second_derivative(field, coords, i, i, spatial_dim)
        second_derivs.append(d2f_dxi2)
    
    # 求和得到拉普拉斯算子（正確處理 Tensor 相加）
    laplacian = second_derivs[0]
    for deriv in second_derivs[1:]:
        laplacian = laplacian + deriv
    
    # 數值穩定化（可選）
    if stabilize:
        laplacian = torch.clamp(laplacian, min=-max_value, max=max_value)
    
    return laplacian


def compute_laplacian_anisotropic(
    field: torch.Tensor,
    coords: torch.Tensor,
    scaling_factors: torch.Tensor
) -> torch.Tensor:
    """
    各向異性拉普拉斯算子 (VS-PINN 專用)
    
    計算：∇²f = N_x² ∂²f/∂X² + N_y² ∂²f/∂Y² + N_z² ∂²f/∂Z²
    
    此函數用於 VS-PINN（Variable Scaling PINN），通過不同方向的縮放因子
    來處理流體力學中的各向異性問題（例如通道流中壁法向與流向的剛性差異）。
    
    Args:
        field: 標量場 [batch_size, 1]
        coords: 縮放後的座標 [batch_size, spatial_dim]
        scaling_factors: 縮放因子 [spatial_dim] 或 [1, spatial_dim]
                        例如：[N_x, N_y, N_z] = [2, 12, 2]
        
    Returns:
        各向異性拉普拉斯算子結果 [batch_size, 1]
        
    Example:
        >>> # VS-PINN 通道流設定
        >>> N_x, N_y, N_z = 2.0, 12.0, 2.0  # 壁法向剛性最大
        >>> scaling_factors = torch.tensor([N_x, N_y, N_z])
        >>> 
        >>> coords = torch.randn(100, 3, requires_grad=True)
        >>> field = model(coords * scaling_factors)  # 模型在縮放空間計算
        >>> 
        >>> # 計算各向異性拉普拉斯
        >>> laplacian = compute_laplacian_anisotropic(
        ...     field, coords, scaling_factors
        ... )
        
    Note:
        VS-PINN 理論參考：
        - arXiv:2308.08468 "Variable Scaling PINN"
        - 適用於高各向異性問題（Re_tau = 1000 通道流）
    """
    spatial_dim = coords.shape[1]
    
    # 確保 scaling_factors 形狀正確
    if scaling_factors.dim() == 1:
        scaling_factors = scaling_factors.unsqueeze(0)  # [spatial_dim] → [1, spatial_dim]
    
    # 計算各方向的加權二階偏導數
    weighted_second_derivs = []
    for i in range(spatial_dim):
        # 計算二階導數
        d2f_dxi2 = compute_second_derivative(field, coords, i, i, spatial_dim)
        
        # 乘以縮放因子的平方 (N_i²)
        N_i = scaling_factors[:, i:i+1]  # [1, 1]
        weighted = (N_i ** 2) * d2f_dxi2
        weighted_second_derivs.append(weighted)
    
    # 求和得到各向異性拉普拉斯（正確處理 Tensor 相加）
    laplacian_aniso = weighted_second_derivs[0]
    for deriv in weighted_second_derivs[1:]:
        laplacian_aniso = laplacian_aniso + deriv
    
    return laplacian_aniso


# ============================================================================
# Backward Compatibility Wrappers (向後兼容包裝函數)
# ============================================================================

def compute_laplacian_2d(f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    向後兼容：2D 拉普拉斯算子
    
    ⚠️ Deprecated: 建議使用 compute_laplacian(f, x, spatial_dim=2)
    """
    return compute_laplacian(f, x, spatial_dim=2)


def compute_laplacian_3d(
    f: torch.Tensor,
    coords: torch.Tensor,
    stabilize: bool = True,
    max_value: float = 1e4
) -> torch.Tensor:
    """
    向後兼容：3D 拉普拉斯算子（帶穩定化）
    
    ⚠️ Deprecated: 建議使用 compute_laplacian(f, coords, spatial_dim=3, stabilize=True)
    """
    return compute_laplacian(f, coords, spatial_dim=3, stabilize=stabilize, max_value=max_value)
