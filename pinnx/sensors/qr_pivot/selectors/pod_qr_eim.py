"""POD + QR/DEIM selector."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
from scipy.linalg import qr, svd

from ..boundary import PeriodicBoundaryHandler
from .qr_pivot import QRPivotSelector

logger = logging.getLogger(__name__)


class PODQREIMSelector:
    """POD + Q-DEIM selector."""

    def __init__(
        self,
        n_modes: Optional[int] = None,
        energy_threshold: float = 0.99,
        use_qr_pivot: bool = True,
        periodic_axes: Optional[List[int]] = None,
        n_circular_shifts: int = 0,
    ):
        self.n_modes = n_modes
        self.energy_threshold = energy_threshold
        self.use_qr_pivot = use_qr_pivot
        self.periodic_handler = (
            PeriodicBoundaryHandler(periodic_axes) if periodic_axes else None
        )
        self.n_circular_shifts = n_circular_shifts

    def select_sensors(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        coords: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        coords_original = coords.copy() if coords is not None else None

        X = data_matrix.copy()

        if self.n_circular_shifts > 0 and coords is not None and self.periodic_handler is not None:
            shifted_matrices, _ = self.periodic_handler.circular_shift_augmentation(
                X, coords, self.n_circular_shifts
            )

            all_selected = []
            for shifted_X in shifted_matrices:
                indices, _ = self._single_pod_deim(shifted_X, n_sensors)
                all_selected.extend(indices)

            unique, counts = np.unique(all_selected, return_counts=True)
            top_k_idx = np.argsort(counts)[-n_sensors:][::-1]
            selected_indices = unique[top_k_idx]

            logger.info(f"POD-DEIM ensemble: {self.n_circular_shifts} 次循環平移")

        else:
            selected_indices, pod_metrics = self._single_pod_deim(X, n_sensors)

        metrics = self._compute_metrics(X, selected_indices)

        if "pod_metrics" in locals():
            metrics.update(pod_metrics)

        return selected_indices, metrics

    def _single_pod_deim(
        self, data_matrix: np.ndarray, n_sensors: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        U, s, Vt = svd(data_matrix, full_matrices=False)

        if self.n_modes is None:
            cumulative_energy = np.cumsum(s**2) / np.sum(s**2)
            n_modes = np.argmax(cumulative_energy >= self.energy_threshold) + 1
            n_modes = min(n_modes, len(s), n_sensors)
        else:
            n_modes = min(self.n_modes, len(s), n_sensors)

        pod_modes = U[:, :n_modes]

        if self.use_qr_pivot:
            Q, R, piv = qr(pod_modes.T, mode="economic", pivoting=True)
            selected_indices = piv[:n_sensors]
        else:
            selected_indices = []
            for i in range(min(n_modes, n_sensors)):
                mode = pod_modes[:, i]
                max_idx = np.argmax(np.abs(mode))
                if max_idx not in selected_indices:
                    selected_indices.append(max_idx)
            selected_indices = np.array(selected_indices)

        selected_indices = selected_indices[:n_sensors]

        metrics = {
            "n_pod_modes": n_modes,
            "pod_energy_ratio": float(np.sum(s[:n_modes] ** 2) / np.sum(s**2)),
        }

        return selected_indices, metrics

    def _compute_metrics(
        self, data_matrix: np.ndarray, selected_indices: np.ndarray
    ) -> Dict[str, float]:
        qr_selector = QRPivotSelector()
        return qr_selector._compute_metrics(data_matrix, selected_indices)
