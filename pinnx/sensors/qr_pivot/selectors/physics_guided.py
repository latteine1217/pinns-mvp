"""Physics-guided QR-pivot selector."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import logging

import numpy as np
import torch
from scipy.linalg import qr, svd

from .qr_pivot import QRPivotSelector

logger = logging.getLogger(__name__)


class PhysicsGuidedQRPivotSelector(QRPivotSelector):
    """Physics-guided QR-pivot selector."""

    def __init__(
        self,
        mode: str = "column",
        pivoting: bool = True,
        regularization: float = 1e-12,
        wall_weight: float = 5.0,
        wall_threshold: float = 0.1,
        threshold_type: str = "y_over_h",
    ):
        super().__init__(mode=mode, pivoting=pivoting, regularization=regularization)
        self.wall_weight = wall_weight
        self.wall_threshold = wall_threshold
        self.threshold_type = threshold_type

        self._wall_mask = None
        self._wall_coverage = 0.0

    def select_sensors(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        coords: Optional[np.ndarray] = None,
        re_tau: float = 1000.0,
        return_qr: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        if coords is None:
            raise ValueError(
                "PhysicsGuidedQRPivotSelector 需要提供空間座標 'coords' 用於計算壁面距離。"
                "座標格式：[n_locations, 3] (x, y, z)"
            )

        if coords.shape[0] != data_matrix.shape[0]:
            raise ValueError(
                f"座標數量 ({coords.shape[0]}) 與資料點數量 ({data_matrix.shape[0]}) 不匹配"
            )

        X = data_matrix.copy()
        n_locations, _ = X.shape

        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std

        n_sensors = min(n_sensors, n_locations)

        wall_mask = self._identify_wall_region(coords, re_tau)
        self._wall_mask = wall_mask

        weights = np.ones(n_locations, dtype=np.float64)
        weights[wall_mask] = self.wall_weight
        W = np.diag(weights)

        X_weighted = W @ X

        logger.info(
            "物理引導 QR-Pivot: 壁面點 %d/%d (%.1f%%), 權重 %.1fx",
            wall_mask.sum(),
            n_locations,
            100 * wall_mask.sum() / n_locations,
            self.wall_weight,
        )

        Q = None
        R = None
        try:
            if self.pivoting:
                Q, R, piv = qr(X_weighted.T, mode="economic", pivoting=True)
                selected_indices = piv[:n_sensors]
            else:
                Q, R = qr(
                    X_weighted.T if self.mode == "column" else X_weighted,
                    mode="economic",
                )
                diag_importance = np.abs(np.diag(R))
                selected_indices = np.argsort(diag_importance)[-n_sensors:][::-1]

        except np.linalg.LinAlgError as exc:
            logger.warning(f"QR 分解失敗，使用 SVD 回退: {exc}")
            U, s, Vt = svd(X_weighted, full_matrices=False)
            importance = np.sum(np.abs(Vt.T) * s, axis=1)
            selected_indices = np.argsort(importance)[-n_sensors:][::-1]

        selected_indices = selected_indices[selected_indices < n_locations]
        selected_indices = selected_indices[:n_sensors]

        metrics = self._compute_metrics(X, selected_indices)

        wall_coverage = wall_mask[selected_indices].sum() / len(selected_indices)
        self._wall_coverage = wall_coverage

        physics_metrics = {
            "wall_coverage": float(wall_coverage),
            "wall_weight": float(self.wall_weight),
            "wall_threshold": float(self.wall_threshold),
            "threshold_type": self.threshold_type,
            "total_wall_points": int(wall_mask.sum()),
            "selected_wall_points": int(wall_mask[selected_indices].sum()),
        }
        metrics.update(physics_metrics)

        result = (selected_indices, metrics)
        if return_qr:
            result = (*result, Q, R)

        return result

    def _identify_wall_region(self, coords: np.ndarray, re_tau: float) -> np.ndarray:
        y_coords = coords[:, 1]

        if self.threshold_type == "y_over_h":
            h = 1.0
            y_min, y_max = -h, h

            dist_to_lower_wall = np.abs(y_coords - y_min)
            dist_to_upper_wall = np.abs(y_coords - y_max)
            dist_to_wall = np.minimum(dist_to_lower_wall, dist_to_upper_wall)

            y_over_h = dist_to_wall / h
            wall_mask = y_over_h < self.wall_threshold

        elif self.threshold_type == "y_plus":
            u_tau = 0.04997
            nu = 5.0e-5
            delta_nu = nu / u_tau

            h = 1.0
            y_min, y_max = -h, h
            dist_to_lower_wall = np.abs(y_coords - y_min)
            dist_to_upper_wall = np.abs(y_coords - y_max)
            dist_to_wall = np.minimum(dist_to_lower_wall, dist_to_upper_wall)

            y_plus = dist_to_wall / delta_nu
            wall_mask = y_plus < self.wall_threshold

        else:
            raise ValueError(f"未知的壁面識別類型: {self.threshold_type}")

        return wall_mask

    def get_wall_statistics(self) -> Dict[str, Any]:
        if self._wall_mask is None:
            raise RuntimeError("請先調用 select_sensors() 方法")

        return {
            "wall_coverage": float(self._wall_coverage),
            "total_wall_points": int(self._wall_mask.sum()),
            "wall_ratio": float(self._wall_mask.sum() / len(self._wall_mask)),
            "wall_weight": float(self.wall_weight),
            "threshold": float(self.wall_threshold),
            "threshold_type": self.threshold_type,
        }
