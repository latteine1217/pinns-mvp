"""Kolmogorov Flow specific input normalization (JAX-PI aligned)."""

from __future__ import annotations

from typing import Any, Dict, Union
import logging

import numpy as np
import torch

from .base_normalizer import BaseNormalizer

logger = logging.getLogger(__name__)


class KolmogorovInputTransform(BaseNormalizer):
    """
    Kolmogorov Flow 專用標準化器，與 JAX-PI 行為對齊。
    
    JAX-PI 標準化策略:
    - 時間維度: t / t_max → [0, 1]
    - 空間維度: x, y 保持不變 → [0, 2π]
    
    這種策略的物理意義：
    1. 時間標準化使網路輸入分布穩定
    2. 空間維度保留週期性，與 Fourier features 配合
    
    Args:
        t_max: 時間範圍的最大值，用於標準化
    
    Example:
        >>> # 創建標準化器
        >>> transform = KolmogorovInputTransform(t_max=50.0)
        >>> 
        >>> # 輸入坐標 (N, 3): [t, x, y]
        >>> coords = torch.tensor([[0.0, 0.0, 0.0],
        ...                        [50.0, 6.28, 6.28]])
        >>> 
        >>> # 標準化：時間 → [0,1], 空間不變
        >>> coords_norm = transform.transform(coords)
        >>> print(coords_norm)
        >>> # tensor([[0.0, 0.0, 0.0],
        >>> #         [1.0, 6.28, 6.28]])
    
    Reference:
        JAX-PI implementation:
        https://github.com/PredictiveIntelligenceLab/jaxpi/blob/main/examples/kolmogorov_flow/models.py#L42
    """
    
    def __init__(self, t_max: float = 1.0):
        """
        初始化 Kolmogorov 標準化器。
        
        Args:
            t_max: 時間範圍最大值。預設為 1.0，實際使用時應設為實際的 t_max
                  （例如 Kolmogorov flow 通常為 50.0）
        """
        super().__init__(norm_type='kolmogorov')
        
        if t_max <= 0:
            raise ValueError(f"t_max 必須大於 0，當前值: {t_max}")
        
        self.t_max = float(t_max)
        logger.info(f"初始化 KolmogorovInputTransform: t_max={self.t_max:.2f}")
    
    def fit(self, samples: torch.Tensor) -> "KolmogorovInputTransform":
        """
        此標準化器不需要擬合統計量。
        
        Args:
            samples: 輸入樣本 [N, 3] (t, x, y)
        
        Returns:
            self: 返回自身以支援鏈式調用
        """
        logger.debug(f"KolmogorovInputTransform.fit: 無需擬合，直接返回")
        return self
    
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        對輸入坐標進行標準化。
        
        僅標準化時間維度（第 0 列），空間維度保持不變。
        
        Args:
            data: 輸入坐標 [N, 3] (t, x, y)，支援 numpy.ndarray 或 torch.Tensor
        
        Returns:
            標準化後的坐標 [N, 3] (t/t_max, x, y)，類型與輸入相同
        
        Raises:
            ValueError: 如果輸入張量維度不正確
            TypeError: 如果輸入類型不支援
        """
        # 判斷輸入類型
        is_numpy = isinstance(data, np.ndarray)
        is_torch = isinstance(data, torch.Tensor)
        
        if not (is_numpy or is_torch):
            raise TypeError(
                f"輸入類型必須為 numpy.ndarray 或 torch.Tensor，當前類型: {type(data)}"
            )
        
        if data.ndim != 2:
            raise ValueError(f"輸入必須是 2D 張量 [N, 3]，當前形狀: {data.shape}")
        
        if data.shape[1] != 3:
            raise ValueError(f"輸入必須有 3 個特徵 (t, x, y)，當前形狀: {data.shape}")
        
        # 複製以避免原地修改
        if is_numpy:
            result = data.copy()
            result[:, 0] = result[:, 0] / self.t_max
        else:  # torch.Tensor
            result = data.clone()
            result[:, 0] = result[:, 0] / self.t_max
        
        return result
    
    def inverse_transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        將標準化坐標還原為原始尺度。
        
        Args:
            data: 標準化坐標 [N, 3] (t/t_max, x, y)，支援 numpy.ndarray 或 torch.Tensor
        
        Returns:
            原始坐標 [N, 3] (t, x, y)，類型與輸入相同
        
        Raises:
            ValueError: 如果輸入張量維度不正確
            TypeError: 如果輸入類型不支援
        """
        # 判斷輸入類型
        is_numpy = isinstance(data, np.ndarray)
        is_torch = isinstance(data, torch.Tensor)
        
        if not (is_numpy or is_torch):
            raise TypeError(
                f"輸入類型必須為 numpy.ndarray 或 torch.Tensor，當前類型: {type(data)}"
            )
        
        if data.ndim != 2:
            raise ValueError(f"輸入必須是 2D 張量 [N, 3]，當前形狀: {data.shape}")
        
        if data.shape[1] != 3:
            raise ValueError(f"輸入必須有 3 個特徵 (t, x, y)，當前形狀: {data.shape}")
        
        # 複製以避免原地修改
        if is_numpy:
            result = data.copy()
            result[:, 0] = result[:, 0] * self.t_max
        else:  # torch.Tensor
            result = data.clone()
            result[:, 0] = result[:, 0] * self.t_max
        
        return result
    
    def to(self, device: torch.device) -> "KolmogorovInputTransform":
        """
        將標準化器移至指定設備。
        
        此標準化器不儲存張量，因此無需移動。
        
        Args:
            device: 目標設備
        
        Returns:
            self: 返回自身以支援鏈式調用
        """
        # 此標準化器只有純量參數 t_max，無需移至設備
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """
        返回標準化器的元數據，用於檢查點保存。
        
        Returns:
            包含標準化參數的字典
        """
        return {
            'type': 'kolmogorov',
            't_max': self.t_max,
            'dims_normalized': [0],  # 只標準化第 0 維（時間）
            'dims_unchanged': [1, 2],  # 第 1, 2 維（空間）保持不變
            'description': f'Kolmogorov normalization: t/t_max, spatial dims unchanged'
        }
