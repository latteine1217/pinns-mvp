"""Normalization configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class InputNormConfig:
    """Input normalization config."""

    norm_type: str = "none"  # none, standard, minmax, channel_flow
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    bounds: Optional[torch.Tensor] = None  # shape [dim, 2]


@dataclass
class OutputNormConfig:
    """Output normalization config."""

    norm_type: str = "none"  # none, training_data_norm, friction_velocity, manual
    variable_order: Optional[List[str]] = None  # variable order
    means: Optional[Dict[str, float]] = None
    stds: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None  # extra parameters
