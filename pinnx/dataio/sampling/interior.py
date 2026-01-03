"""
內部點採樣模組
"""

from typing import Dict, Tuple

import torch


def sample_interior_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    exclude_boundary_tol: float = 0.01,
    use_sobol: bool = True
) -> torch.Tensor:
    """
    在內部均勻採樣點，排除邊界區域
    
    Args:
        n_points: 內部點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        exclude_boundary_tol: 邊界排除容差（物理座標）
        use_sobol: 是否使用 Sobol 序列（更均勻）
    
    Returns:
        內部點座標 [n_points, 3] (x, y, z)
    """
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    # 調整內部域範圍（排除邊界容差）
    x_min_inner = x_min + exclude_boundary_tol
    x_max_inner = x_max - exclude_boundary_tol
    y_min_inner = y_min + exclude_boundary_tol
    y_max_inner = y_max - exclude_boundary_tol
    z_min_inner = z_min + exclude_boundary_tol
    z_max_inner = z_max - exclude_boundary_tol
    
    if use_sobol:
        # 使用 Sobol 序列（準均勻分佈）
        sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        samples = sobol.draw(n_points).to(device)
        
        # 縮放到內部域
        x_interior = samples[:, 0:1] * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = samples[:, 1:2] * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = samples[:, 2:3] * (z_max_inner - z_min_inner) + z_min_inner
    else:
        # 使用均勻隨機採樣
        x_interior = torch.rand(n_points, 1, device=device) * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = torch.rand(n_points, 1, device=device) * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = torch.rand(n_points, 1, device=device) * (z_max_inner - z_min_inner) + z_min_inner
    
    interior_points = torch.cat([x_interior, y_interior, z_interior], dim=1)
    
    return interior_points
