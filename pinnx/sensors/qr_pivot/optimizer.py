"""Sensor optimizer wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np

from .base import BaseSensorSelector
from .factory import create_sensor_selector

logger = logging.getLogger(__name__)


class SensorOptimizer:
    """High-level optimizer for sensor placement."""

    def __init__(self, strategy: str = "auto", config: Optional[Dict] = None):
        self.strategy = strategy
        self.config = config or {}

    def optimize_sensor_placement(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        validation_data: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.strategy == "auto":
            return self._auto_strategy_selection(data_matrix, n_sensors, validation_data)

        selector = self._create_selector(self.strategy)
        selected_indices, metrics = selector.select_sensors(data_matrix, n_sensors)

        if validation_data is not None:
            validation_metrics = self._evaluate_on_validation(
                data_matrix, validation_data, selected_indices
            )
            metrics.update(validation_metrics)

        return selected_indices, metrics

    def _create_selector(self, strategy: str) -> BaseSensorSelector:
        selector_cfg = self.config.get(strategy, {})
        return create_sensor_selector(strategy, **selector_cfg)

    def _auto_strategy_selection(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        validation_data: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        n_locations, n_features = data_matrix.shape
        data_rank = np.linalg.matrix_rank(data_matrix)
        aspect_ratio = n_features / n_locations

        if data_rank < min(n_locations, n_features) * 0.8:
            strategy = "pod_based"
            logger.info("檢測到低秩結構，使用 POD-based 策略")
        elif aspect_ratio > 2.0:
            strategy = "qr_pivot"
            logger.info("檢測到寬矩陣結構，使用 QR-pivot 策略")
        elif n_sensors / n_locations < 0.1:
            strategy = "multi_objective"
            logger.info("檢測到極稀疏感測需求，使用多目標最適化")
        else:
            strategy = "greedy"
            logger.info("使用預設貪心策略")

        selector = self._create_selector(strategy)
        selected_indices, metrics = selector.select_sensors(data_matrix, n_sensors)

        metrics["auto_selected_strategy"] = strategy
        metrics["data_analysis"] = {
            "rank": int(data_rank),
            "aspect_ratio": float(aspect_ratio),
            "sparsity_ratio": float(n_sensors / n_locations),
        }

        if validation_data is not None:
            validation_metrics = self._evaluate_on_validation(
                data_matrix, validation_data, selected_indices
            )
            metrics.update(validation_metrics)

        return selected_indices, metrics

    def _evaluate_on_validation(
        self,
        train_data: np.ndarray,
        validation_data: np.ndarray,
        selected_indices: np.ndarray,
    ) -> Dict[str, float]:
        try:
            sensor_data_train = train_data[selected_indices, :]
            sensor_data_val = validation_data[selected_indices, :]

            if sensor_data_train.shape[0] >= sensor_data_train.shape[1]:
                reconstruction_matrix = np.linalg.pinv(sensor_data_train)
                coefficients = reconstruction_matrix @ validation_data
                reconstructed = sensor_data_train @ coefficients
            else:
                regularization = 1e-6
                gram = (
                    sensor_data_train @ sensor_data_train.T
                    + regularization * np.eye(sensor_data_train.shape[0])
                )
                reconstruction_matrix = sensor_data_train.T @ np.linalg.pinv(gram)
                coefficients = reconstruction_matrix @ sensor_data_val
                reconstructed = train_data @ coefficients

            mse = np.mean((validation_data - reconstructed) ** 2)
            relative_error = np.linalg.norm(validation_data - reconstructed, "fro") / (
                np.linalg.norm(validation_data, "fro") + 1e-16
            )

            return {
                "validation_mse": float(mse),
                "validation_relative_error": float(relative_error),
                "reconstruction_rank": int(np.linalg.matrix_rank(reconstruction_matrix)),
            }

        except Exception as exc:
            logger.warning(f"驗證評估失敗: {exc}")
            return {
                "validation_mse": np.inf,
                "validation_relative_error": np.inf,
                "reconstruction_rank": 0,
            }
