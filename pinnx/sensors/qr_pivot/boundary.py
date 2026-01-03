"""Periodic boundary utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class PeriodicBoundaryHandler:
    """Periodic boundary handling utilities."""

    def __init__(self, periodic_axes: List[int] = [0, 2]):
        self.periodic_axes = periodic_axes

    def circular_shift_augmentation(
        self,
        data_matrix: np.ndarray,
        coords: np.ndarray,
        n_shifts: int = 5,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        if random_seed is not None:
            np.random.seed(random_seed)

        n_locations = data_matrix.shape[0]
        shifted_matrices = []
        shifted_coords = []

        grid_sizes = {}
        for ax in self.periodic_axes:
            unique_vals = np.unique(coords[:, ax])
            grid_sizes[ax] = len(unique_vals)

        for _ in range(n_shifts):
            shift_indices = {}
            for ax in self.periodic_axes:
                if grid_sizes[ax] > 1:
                    shift_indices[ax] = np.random.randint(0, grid_sizes[ax])
                else:
                    shift_indices[ax] = 0

            shifted_data, shifted_coord = self._apply_circular_shift(
                data_matrix, coords, shift_indices, grid_sizes
            )

            shifted_matrices.append(shifted_data)
            shifted_coords.append(shifted_coord)

        return shifted_matrices, shifted_coords

    def _apply_circular_shift(
        self,
        data_matrix: np.ndarray,
        coords: np.ndarray,
        shift_indices: Dict[int, int],
        grid_sizes: Dict[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_locations = data_matrix.shape[0]
        n_dims = coords.shape[1]

        grid_shape = self._infer_grid_shape(coords, grid_sizes)

        if grid_shape is None:
            logger.warning("無法推斷網格結構，使用簡化循環平移")
            perm = np.roll(np.arange(n_locations), shift_indices.get(0, 0))
            return data_matrix[perm, :], coords[perm, :]

        linear_indices = np.arange(n_locations)

        if n_dims == 2:
            multi_indices = np.unravel_index(linear_indices, grid_shape, order="C")

            shifted_multi = list(multi_indices)
            for ax in self.periodic_axes:
                if ax < len(shifted_multi) and ax in shift_indices:
                    shifted_multi[ax] = (multi_indices[ax] + shift_indices[ax]) % grid_shape[ax]

            shifted_linear = np.ravel_multi_index(tuple(shifted_multi), grid_shape, order="C")

        elif n_dims == 3:
            multi_indices = np.unravel_index(linear_indices, grid_shape, order="C")

            shifted_multi = list(multi_indices)
            for ax in self.periodic_axes:
                if ax < len(shifted_multi) and ax in shift_indices:
                    shifted_multi[ax] = (multi_indices[ax] + shift_indices[ax]) % grid_shape[ax]

            shifted_linear = np.ravel_multi_index(tuple(shifted_multi), grid_shape, order="C")

        else:
            logger.warning(f"不支援 {n_dims}D 循環平移，使用簡化版本")
            perm = np.roll(np.arange(n_locations), shift_indices.get(0, 0))
            return data_matrix[perm, :], coords[perm, :]

        shifted_data = data_matrix[shifted_linear, :]
        shifted_coords = coords[shifted_linear, :]

        return shifted_data, shifted_coords

    def _infer_grid_shape(
        self, coords: np.ndarray, grid_sizes: Dict[int, int]
    ) -> Optional[Tuple[int, ...]]:
        n_dims = coords.shape[1]

        if grid_sizes:
            shape_list = []
            for dim in range(n_dims):
                if dim in grid_sizes:
                    shape_list.append(grid_sizes[dim])
                else:
                    unique_vals = len(np.unique(coords[:, dim]))
                    shape_list.append(unique_vals)

            grid_shape = tuple(shape_list)

            expected_total = int(np.prod(grid_shape))
            if expected_total == coords.shape[0]:
                return grid_shape
            logger.warning(
                f"網格形狀驗證失敗：推斷 {grid_shape} (總數 {expected_total}) "
                f"!= 實際 {coords.shape[0]}"
            )

        try:
            shape_list = []
            for dim in range(n_dims):
                unique_vals = len(np.unique(np.round(coords[:, dim], decimals=8)))
                shape_list.append(unique_vals)

            grid_shape = tuple(shape_list)
            expected_total = int(np.prod(grid_shape))

            if expected_total == coords.shape[0]:
                return grid_shape
        except Exception as exc:
            logger.warning(f"網格形狀推斷失敗: {exc}")

        return None

    def compute_periodic_distance(
        self,
        coords1: np.ndarray,
        coords2: np.ndarray,
        domain_bounds: Dict[int, Tuple[float, float]],
    ) -> np.ndarray:
        distances = np.zeros(len(coords1))

        for dim in range(coords1.shape[1]):
            if dim in self.periodic_axes and dim in domain_bounds:
                length = domain_bounds[dim][1] - domain_bounds[dim][0]
                diff = np.abs(coords1[:, dim] - coords2[:, dim])
                periodic_diff = np.minimum(diff, length - diff)
                distances += periodic_diff**2
            else:
                distances += (coords1[:, dim] - coords2[:, dim]) ** 2

        return np.sqrt(distances)
