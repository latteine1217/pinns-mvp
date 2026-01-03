"""POD-based selector."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
from scipy.linalg import svd

from .qr_pivot import QRPivotSelector


class PODBasedSelector:
    """POD-based sensor selector."""

    def __init__(
        self,
        n_modes: Optional[int] = None,
        energy_threshold: float = 0.99,
        mode_weighting: str = "energy",
    ):
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.mode_weighting = mode_weighting

    def select_sensors(
        self, data_matrix: np.ndarray, n_sensors: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()

        U, s, Vt = svd(data_matrix, full_matrices=False)

        if self.n_modes is None:
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            n_modes = np.argmax(cumulative_energy >= self.energy_threshold) + 1
            n_modes = min(n_modes, len(s))
        else:
            n_modes = min(self.n_modes, len(s))

        pod_modes = U[:, :n_modes]

        if self.mode_weighting == "energy":
            weights = s[:n_modes]
            weighted_modes = pod_modes * weights[np.newaxis, :]
        elif self.mode_weighting == "uniform":
            weighted_modes = pod_modes
        elif self.mode_weighting == "decay":
            weights = np.exp(-np.arange(n_modes) / max(1, n_modes / 3))
            weighted_modes = pod_modes * weights[np.newaxis, :]
        else:
            weighted_modes = pod_modes

        qr_selector = QRPivotSelector(mode="row", pivoting=True)
        selected_indices, qr_metrics = qr_selector.select_sensors(weighted_modes, n_sensors)

        pod_metrics = {
            "n_pod_modes": n_modes,
            "pod_energy_ratio": float(np.sum(s[:n_modes] ** 2) / np.sum(s**2)),
            "effective_rank": float(np.sum(s**2) ** 2 / np.sum(s**4)),
        }

        metrics = {**qr_metrics, **pod_metrics}

        return selected_indices, metrics
