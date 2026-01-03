"""
Sampling 模組 - 提供邊界和內部點的採樣策略
"""

from .boundary import sample_boundary_points
from .interior import sample_interior_points

__all__ = [
    'sample_boundary_points',
    'sample_interior_points',
]
