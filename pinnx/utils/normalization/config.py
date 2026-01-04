"""Normalization configuration dataclasses."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class BaseNormConfig(ABC):
    """基礎標準化配置（共享 norm_type）"""

    norm_type: str = "none"

    def is_enabled(self) -> bool:
        """檢查標準化是否啟用"""
        return self.norm_type not in ("none", "identity")


@dataclass
class InputNormConfig(BaseNormConfig):
    """輸入（坐標）標準化配置

    支援模式:
        - none/identity: 不標準化
        - standard: Z-score (mean=0, std=1)
        - minmax: 映射到 feature_range
        - channel_flow: 使用預定義 bounds
    """

    norm_type: str = "none"
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    bounds: Optional[torch.Tensor] = None  # shape [dim, 2]


@dataclass
class OutputNormConfig(BaseNormConfig):
    """輸出（變量）標準化配置

    支援模式:
        - none: 不標準化
        - training_data_norm: 從訓練數據計算 Z-score
        - manual: 手動指定 means/stds
        - dns_ground_truth_norm: 從 DNS 數據計算

    Note:
        friction_velocity 模式已移除，請使用 compute_friction_velocity_scales()
        輔助函數配合 manual 模式替代。
    """

    norm_type: str = "none"
    variable_order: Optional[List[str]] = None
    means: Optional[Dict[str, float]] = None
    stds: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None  # 額外參數
