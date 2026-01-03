"""Multi-objective selector."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
from scipy.optimize import differential_evolution

from .greedy import GreedySelector
from .pod_based import PODBasedSelector
from .qr_pivot import QRPivotSelector

logger = logging.getLogger(__name__)


class MultiObjectiveSelector:
    """Multi-objective sensor selector."""

    def __init__(
        self,
        objectives: List[str] = ["accuracy", "robustness", "efficiency"],
        weights: Optional[List[float]] = None,
        method: str = "weighted_sum",
        max_iterations: int = 100,
    ):
        self.objectives = objectives
        self.weights = weights or [1.0 / len(objectives)] * len(objectives)
        self.method = method
        self.max_iterations = max_iterations

    def select_sensors(
        self, data_matrix: np.ndarray, n_sensors: int, noise_level: float = 0.01
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()

        if self.method == "weighted_sum":
            return self._weighted_sum_optimization(data_matrix, n_sensors, noise_level)
        if self.method == "pareto":
            return self._pareto_optimization(data_matrix, n_sensors, noise_level)

        logger.warning(f"未實現的多目標方法 {self.method}，使用 QR-pivot")
        qr_selector = QRPivotSelector()
        return qr_selector.select_sensors(data_matrix, n_sensors)

    def _weighted_sum_optimization(
        self, data_matrix: np.ndarray, n_sensors: int, noise_level: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n_locations = data_matrix.shape[0]

        def objective_function(binary_selection):
            indices = np.where(binary_selection > 0.5)[0]
            if len(indices) == 0:
                return 1e10

            if len(indices) > n_sensors:
                importance = np.sum(np.abs(data_matrix[indices, :]), axis=1)
                top_indices = np.argsort(importance)[-n_sensors:]
                indices = indices[top_indices]

            objectives_values = self._compute_multi_objectives(
                data_matrix, indices, noise_level
            )

            weighted_objective = sum(
                w * obj for w, obj in zip(self.weights, objectives_values)
            )

            count_penalty = abs(len(indices) - n_sensors) * 0.1

            return -weighted_objective + count_penalty

        bounds = [(0, 1)] * n_locations

        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=self.max_iterations,
            popsize=min(15, max(10, n_locations // 10)),
            seed=42,
            atol=1e-6,
            tol=1e-6,
        )

        binary_solution = result.x
        selected_indices = np.where(binary_solution > 0.5)[0]

        if len(selected_indices) != n_sensors:
            if len(selected_indices) > n_sensors:
                importance = np.sum(np.abs(data_matrix[selected_indices, :]), axis=1)
                top_k = np.argsort(importance)[-n_sensors:]
                selected_indices = selected_indices[top_k]
            else:
                remaining = np.setdiff1d(np.arange(n_locations), selected_indices)
                importance = np.sum(np.abs(data_matrix[remaining, :]), axis=1)
                n_add = n_sensors - len(selected_indices)
                top_add = np.argsort(importance)[-n_add:]
                selected_indices = np.concatenate([selected_indices, remaining[top_add]])

        final_objectives = self._compute_multi_objectives(
            data_matrix, selected_indices, noise_level
        )

        metrics = {
            "multi_objective_score": float(-result.fun),
            "optimization_success": bool(result.success),
            "n_iterations": int(result.nit),
        }

        for i, obj_name in enumerate(self.objectives):
            metrics[f"objective_{obj_name}"] = float(final_objectives[i])

        return selected_indices, metrics

    def _compute_multi_objectives(
        self, data_matrix: np.ndarray, indices: np.ndarray, noise_level: float
    ) -> List[float]:
        if len(indices) == 0:
            return [0.0] * len(self.objectives)

        selected_data = data_matrix[indices, :]
        objectives_values = []

        for obj_name in self.objectives:
            if obj_name == "accuracy":
                try:
                    s = np.linalg.svd(selected_data, compute_uv=False)
                    if s[-1] > 1e-15:
                        cond = s[0] / s[-1]
                    else:
                        cond = np.inf
                    accuracy = 1.0 / (1.0 + np.log(cond + 1e-16))
                except Exception:
                    accuracy = 0.0
                objectives_values.append(accuracy)

            elif obj_name == "robustness":
                try:
                    noisy_data = selected_data + noise_level * np.random.randn(
                        *selected_data.shape
                    )
                    reconstruction_error = np.linalg.norm(noisy_data - selected_data, "fro")
                    robustness = 1.0 / (1.0 + reconstruction_error)
                except Exception:
                    robustness = 0.0
                objectives_values.append(robustness)

            elif obj_name == "efficiency":
                try:
                    info_content = np.linalg.slogdet(
                        selected_data @ selected_data.T
                        + 1e-12 * np.eye(len(indices))
                    )[1]
                    efficiency = info_content / max(1, len(indices))
                except Exception:
                    efficiency = 0.0
                objectives_values.append(efficiency)

            elif obj_name == "coverage":
                if len(indices) > 1:
                    min_dist = np.min(
                        [
                            np.linalg.norm(data_matrix[i] - data_matrix[j])
                            for i in indices
                            for j in indices
                            if i != j
                        ]
                    )
                    coverage = min_dist / (
                        np.linalg.norm(data_matrix.max(axis=0) - data_matrix.min(axis=0))
                        + 1e-16
                    )
                else:
                    coverage = 0.0
                objectives_values.append(coverage)

            else:
                objectives_values.append(0.0)

        return objectives_values

    def _pareto_optimization(
        self, data_matrix: np.ndarray, n_sensors: int, noise_level: float
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        n_candidates = min(50, data_matrix.shape[0])
        candidates = []

        selectors = [
            QRPivotSelector(mode="column"),
            PODBasedSelector(n_modes=min(10, data_matrix.shape[1] // 2)),
            GreedySelector(objective="info_gain"),
            GreedySelector(objective="condition"),
        ]

        for selector in selectors:
            try:
                indices, _ = selector.select_sensors(data_matrix, n_sensors)
                objectives = self._compute_multi_objectives(data_matrix, indices, noise_level)
                candidates.append((indices, objectives))
            except Exception:
                continue

        for _ in range(n_candidates - len(candidates)):
            random_indices = np.random.choice(data_matrix.shape[0], n_sensors, replace=False)
            objectives = self._compute_multi_objectives(data_matrix, random_indices, noise_level)
            candidates.append((random_indices, objectives))

        pareto_candidates = self._find_pareto_front(candidates)

        if pareto_candidates:
            best_score = -np.inf
            best_solution = None

            for indices, objectives in pareto_candidates:
                weighted_score = sum(w * obj for w, obj in zip(self.weights, objectives))
                if weighted_score > best_score:
                    best_score = weighted_score
                    best_solution = (indices, objectives)

            selected_indices, final_objectives = best_solution
        else:
            selected_indices, final_objectives = candidates[0]

        metrics = {
            "pareto_front_size": len(pareto_candidates),
            "n_candidates_evaluated": len(candidates),
            "pareto_score": float(best_score),
        }

        for i, obj_name in enumerate(self.objectives):
            metrics[f"objective_{obj_name}"] = float(final_objectives[i])

        return selected_indices, metrics

    def _find_pareto_front(self, candidates: List[Tuple]) -> List[Tuple]:
        pareto_front = []

        for candidate in candidates:
            is_dominated = False

            for other in candidates:
                if candidate == other:
                    continue

                candidate_objectives = candidate[1]
                other_objectives = other[1]

                if all(c <= o for c, o in zip(candidate_objectives, other_objectives)) and any(
                    c < o for c, o in zip(candidate_objectives, other_objectives)
                ):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(candidate)

        return pareto_front
