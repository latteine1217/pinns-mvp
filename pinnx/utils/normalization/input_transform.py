"""Input normalization transform."""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging

import torch

from .base_normalizer import BaseNormalizer
from .config import InputNormConfig

logger = logging.getLogger(__name__)


class InputTransform(BaseNormalizer):
    """
    Coordinate normalization transform.

    Supported:
    - none/identity
    - standard: Z-score
    - minmax: map to feature_range
    - channel_flow: map using predefined bounds
    """

    def __init__(self, config: InputNormConfig):
        norm_type = (config.norm_type or "none").lower()
        super().__init__(norm_type=norm_type)
        self.feature_range = config.feature_range
        self.bounds = config.bounds

        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.data_min: Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None

    def fit(self, samples: torch.Tensor) -> "InputTransform":
        """Fit statistics from samples."""
        if self.norm_type == "standard":
            mean = torch.mean(samples, dim=0, keepdim=True)
            std = torch.std(samples, dim=0, keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            self.mean = mean
            self.std = std

        elif self.norm_type == "minmax":
            data_min = torch.min(samples, dim=0, keepdim=True)[0]
            data_max = torch.max(samples, dim=0, keepdim=True)[0]
            data_range = data_max - data_min
            data_range = torch.where(data_range < 1e-8, torch.ones_like(data_range), data_range)
            self.data_min = data_min
            self.data_range = data_range

        elif self.norm_type in ("channel_flow", "vs_pinn"):
            # Use predefined bounds only.
            pass
        else:
            # none/identity
            pass

        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply normalization."""
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor

        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer 尚未擬合")
            return (tensor - self.mean) / self.std

        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer 尚未擬合")
            norm = (tensor - self.data_min) / self.data_range
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo

        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds 未提供")
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            denom = torch.where((maxs - mins) < 1e-8, torch.ones_like(maxs - mins), maxs - mins)
            norm = (tensor - mins) / denom
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo

        raise ValueError(f"不支援的標準化類型: {self.norm_type}")

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """Inverse normalization."""
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor

        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer 尚未擬合")
            return tensor * self.std + self.mean

        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer 尚未擬合")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            return norm * self.data_range + self.data_min

        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds 未提供")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            return norm * (maxs - mins) + mins

        raise ValueError(f"不支援的標準化類型: {self.norm_type}")

    def to(self, device: torch.device) -> "InputTransform":
        """Move stats to device."""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        if self.data_min is not None:
            self.data_min = self.data_min.to(device)
        if self.data_range is not None:
            self.data_range = self.data_range.to(device)
        if self.bounds is not None:
            self.bounds = self.bounds.to(device)
        return self

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for checkpoints."""
        if self.mean is not None:
            device = self.mean.device
        elif self.data_min is not None:
            device = self.data_min.device
        elif self.bounds is not None:
            device = self.bounds.device
        else:
            device = torch.device("cpu")

        feature_range_tensor = torch.tensor(self.feature_range, dtype=torch.float32, device=device)

        metadata: Dict[str, Any] = {
            "norm_type": self.norm_type,
            "feature_range": feature_range_tensor,
        }

        if self.mean is not None:
            metadata["mean"] = self.mean.clone().detach()
        if self.std is not None:
            metadata["std"] = self.std.clone().detach()
        if self.data_min is not None:
            metadata["data_min"] = self.data_min.clone().detach()
        if self.data_range is not None:
            metadata["data_range"] = self.data_range.clone().detach()
        if self.bounds is not None:
            metadata["bounds"] = self.bounds.clone().detach()

        return metadata
