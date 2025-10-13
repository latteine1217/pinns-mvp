from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch


@dataclass
class NormalizationConfig:
    norm_type: str = "none"
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    bounds: Optional[torch.Tensor] = None  # shape [dim, 2]


class InputNormalizer:
    """
    Lightweight normalizer for spatial coordinates.

    Supported types:
        - none / identity
        - standard  : (x - mean) / std
        - minmax    : map to feature_range using observed min/max
        - channel_flow : map using provided domain bounds to feature_range
    """

    def __init__(self, config: NormalizationConfig):
        self.norm_type = (config.norm_type or "none").lower()
        self.feature_range = config.feature_range
        self.bounds = config.bounds

        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.data_min: Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None

    def fit(self, samples: torch.Tensor) -> "InputNormalizer":
        """
        Fit statistics from samples.

        Args:
            samples: [N, D] tensor on any device
        """
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
            # channel_flow uses predefined bounds; nothing to fit.
            pass
        else:
            # identity / none
            pass
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer is not fitted.")
            return (tensor - self.mean) / self.std
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer is not fitted.")
            norm = (tensor - self.data_min) / self.data_range
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds not provided.")
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            denom = torch.where((maxs - mins) < 1e-8, torch.ones_like(maxs - mins), maxs - mins)
            norm = (tensor - mins) / denom
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer is not fitted.")
            return tensor * self.std + self.mean
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer is not fitted.")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            return norm * self.data_range + self.data_min
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds not provided.")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            return norm * (maxs - mins) + mins
        raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def to(self, device: torch.device) -> "InputNormalizer":
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

