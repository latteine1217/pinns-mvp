"""Greedy selector."""

from __future__ import annotations

from typing import Dict, Tuple
import logging

import numpy as np
import torch
from scipy.linalg import svd

from .qr_pivot import QRPivotSelector

logger = logging.getLogger(__name__)


class GreedySelector:
    """Greedy sensor selector."""

    def __init__(self, objective: str = "info_gain", regularization: float = 1e-8):
        self.objective = objective
        self.regularization = regularization

    def select_sensors(
        self, data_matrix: np.ndarray, n_sensors: int
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()

        n_locations, _ = data_matrix.shape
        n_sensors = min(n_sensors, n_locations)

        selected_indices = []
        remaining_indices = list(range(n_locations))
        objective_values = []

        for step in range(n_sensors):
            best_idx = None
            best_objective = -np.inf

            for candidate_idx in remaining_indices:
                test_indices = selected_indices + [candidate_idx]
                test_data = data_matrix[test_indices, :]

                objective_val = self._compute_objective(test_data)

                if objective_val > best_objective:
                    best_objective = objective_val
                    best_idx = candidate_idx

            if best_idx is not None:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
                objective_values.append(best_objective)
            else:
                logger.warning(f"無法在第 {step+1} 步找到有效的感測點")
                break

        selected_indices = np.array(selected_indices)

        final_data = data_matrix[selected_indices, :]
        metrics = {
            "final_objective": float(best_objective),
            "objective_progression": objective_values,
            "greedy_efficiency": float(len(selected_indices) / n_sensors),
        }

        qr_selector = QRPivotSelector()
        basic_metrics = qr_selector._compute_metrics(data_matrix, selected_indices)
        metrics.update(basic_metrics)

        return selected_indices, metrics

    def _compute_objective(self, data_subset: np.ndarray) -> float:
        if data_subset.shape[0] == 0:
            return -np.inf

        try:
            gram_matrix = (
                data_subset @ data_subset.T
                + self.regularization * np.eye(data_subset.shape[0])
            )

            if self.objective == "info_gain":
                sign, logdet = np.linalg.slogdet(gram_matrix)
                return logdet if sign > 0 else -np.inf

            if self.objective == "condition":
                _, s, _ = svd(data_subset, full_matrices=False)
                cond = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
                return -np.log(cond + 1e-16)

            if self.objective == "energy":
                return np.trace(gram_matrix)

            if self.objective == "determinant":
                det = np.linalg.det(gram_matrix)
                return det if det > 0 else -np.inf

            raise ValueError(f"未知的目標函數: {self.objective}")

        except np.linalg.LinAlgError:
            return -np.inf
