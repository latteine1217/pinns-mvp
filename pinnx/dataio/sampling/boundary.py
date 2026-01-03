"""
邊界點採樣模組
"""

import logging
from typing import Dict, Tuple, Optional

import torch


def sample_boundary_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    distribution: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    在邊界上均勻採樣點
    
    Args:
        n_points: 總邊界點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        distribution: 邊界點分佈 {'wall': int, 'periodic': int, 'inlet': int}
                     如果為 None，預設為 {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    Returns:
        邊界點座標 [n_points, 3] (x, y, z)
    """
    if distribution is None:
        distribution = {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    # 驗證總點數
    total_requested = sum(distribution.values())
    if total_requested != n_points:
        logging.warning(f"⚠️ Boundary distribution sum ({total_requested}) != n_points ({n_points}), 自動調整比例")
        # 按比例調整
        scale = n_points / total_requested
        distribution = {k: int(v * scale) for k, v in distribution.items()}
        # 修正舍入誤差
        diff = n_points - sum(distribution.values())
        distribution['wall'] += diff
    
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    boundary_points = []
    
    # 1. 壁面點 (y = y_min 和 y = y_max)
    n_wall = distribution['wall']
    n_wall_bottom = n_wall // 2
    n_wall_top = n_wall - n_wall_bottom
    
    # 下壁面 (y = y_min)
    x_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (x_max - x_min) + x_min
    y_wall_bottom = torch.full((n_wall_bottom, 1), y_min, device=device)
    z_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (z_max - z_min) + z_min
    wall_bottom = torch.cat([x_wall_bottom, y_wall_bottom, z_wall_bottom], dim=1)
    
    # 上壁面 (y = y_max)
    x_wall_top = torch.rand(n_wall_top, 1, device=device) * (x_max - x_min) + x_min
    y_wall_top = torch.full((n_wall_top, 1), y_max, device=device)
    z_wall_top = torch.rand(n_wall_top, 1, device=device) * (z_max - z_min) + z_min
    wall_top = torch.cat([x_wall_top, y_wall_top, z_wall_top], dim=1)
    
    boundary_points.extend([wall_bottom, wall_top])
    
    # 2. 週期性邊界點 (x = x_min/x_max, z = z_min/z_max)
    n_periodic = distribution['periodic']
    n_per_face = n_periodic // 4  # 4 個面：x_min, x_max, z_min, z_max
    
    # x = x_min
    x_left = torch.full((n_per_face, 1), x_min, device=device)
    y_left = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_left = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_left = torch.cat([x_left, y_left, z_left], dim=1)
    
    # x = x_max
    x_right = torch.full((n_per_face, 1), x_max, device=device)
    y_right = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_right = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_right = torch.cat([x_right, y_right, z_right], dim=1)
    
    # z = z_min
    x_front = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_front = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_front = torch.full((n_per_face, 1), z_min, device=device)
    periodic_front = torch.cat([x_front, y_front, z_front], dim=1)
    
    # z = z_max
    x_back = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_back = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_back = torch.full((n_per_face, 1), z_max, device=device)
    periodic_back = torch.cat([x_back, y_back, z_back], dim=1)
    
    boundary_points.extend([periodic_left, periodic_right, periodic_front, periodic_back])
    
    # 3. Inlet 點 (x = x_min，特別處理)
    n_inlet = distribution['inlet']
    x_inlet = torch.full((n_inlet, 1), x_min, device=device)
    y_inlet = torch.rand(n_inlet, 1, device=device) * (y_max - y_min) + y_min
    z_inlet = torch.rand(n_inlet, 1, device=device) * (z_max - z_min) + z_min
    inlet = torch.cat([x_inlet, y_inlet, z_inlet], dim=1)
    
    boundary_points.append(inlet)
    
    # 合併所有邊界點
    all_boundary_points = torch.cat(boundary_points, dim=0)
    
    return all_boundary_points
