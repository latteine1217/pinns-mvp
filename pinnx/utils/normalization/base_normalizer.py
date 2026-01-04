"""基礎標準化器抽象類"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import numpy as np
import torch


class BaseNormalizer(ABC):
    """標準化器基類（定義統一接口）

    所有標準化器必須實現的核心方法：
    - transform: 標準化數據
    - inverse_transform: 反標準化數據
    - get_metadata: 獲取可序列化的 metadata（用於 checkpoint）
    - to: 移動到指定設備
    """

    def __init__(self, norm_type: str = "none"):
        self.norm_type = norm_type

    @abstractmethod
    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """標準化數據

        Args:
            data: 原始數據

        Returns:
            標準化後的數據（保持輸入類型）
        """
        pass

    @abstractmethod
    def inverse_transform(
        self, data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """反標準化數據

        Args:
            data: 標準化後的數據

        Returns:
            原始空間的數據（保持輸入類型）
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """獲取 metadata（用於 checkpoint 保存）

        Returns:
            可序列化的 metadata 字典
        """
        pass

    @abstractmethod
    def to(self, device: torch.device) -> "BaseNormalizer":
        """移動內部參數到指定設備

        Args:
            device: 目標設備

        Returns:
            self（支援鏈式調用）
        """
        pass

    def is_enabled(self) -> bool:
        """檢查標準化是否啟用"""
        return self.norm_type not in ("none", "identity")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(norm_type='{self.norm_type}')"
