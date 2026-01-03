"""Evaluation helpers for sensor placement."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .selectors.qr_pivot import QRPivotSelector


def evaluate_sensor_placement(
    data_matrix: np.ndarray,
    selected_indices: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    noise_levels: List[float] = [0.01, 0.05, 0.1],
) -> Dict[str, Any]:
    """Evaluate placement quality."""
    metrics = {}

    qr_selector = QRPivotSelector()
    basic_metrics = qr_selector._compute_metrics(data_matrix, selected_indices)
    metrics.update(basic_metrics)

    if test_data is not None:
        robustness_metrics = {}

        for noise_level in noise_levels:
            try:
                noisy_test = test_data + noise_level * np.random.randn(*test_data.shape)

                sensor_train = data_matrix[selected_indices, :]
                sensor_test = noisy_test[selected_indices, :]

                reconstruction_matrix = np.linalg.pinv(sensor_train)
                reconstructed = sensor_train @ (reconstruction_matrix @ test_data)

                reconstruction_error = np.linalg.norm(reconstructed - test_data, "fro") / (
                    np.linalg.norm(test_data, "fro") + 1e-16
                )

                robustness_metrics[f"noise_{noise_level}_error"] = float(reconstruction_error)

            except Exception:
                robustness_metrics[f"noise_{noise_level}_error"] = np.inf

        metrics["robustness"] = robustness_metrics

    if len(selected_indices) > 1:
        if data_matrix.shape[1] >= 2:
            coordinates = data_matrix[selected_indices, :2]
        else:
            coordinates = data_matrix[selected_indices, :]

        min_distance = np.inf
        max_distance = 0.0

        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                min_distance = min(min_distance, dist)
                max_distance = max(max_distance, dist)

        metrics["geometry"] = {
            "min_sensor_distance": float(min_distance),
            "max_sensor_distance": float(max_distance),
            "distance_ratio": float(max_distance / (min_distance + 1e-16)),
        }

    return metrics
