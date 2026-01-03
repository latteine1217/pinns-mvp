"""Feature and snapshot utilities."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import logging

import numpy as np

logger = logging.getLogger(__name__)


def build_circular_snapshot_matrix(
    snapshots: np.ndarray,
    coords: np.ndarray,
    grid_shape: Tuple[int, ...],
    periodic_axes: List[int],
    n_wrap_layers: int = 1,
    domain_lengths: Optional[Dict[int, float]] = None,
    seam_weight: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build augmented snapshot matrix for periodic grids."""
    n_dims = coords.shape[1]
    n_locations, n_snapshots = snapshots.shape

    expected_total = int(np.prod(grid_shape))
    if expected_total != n_locations:
        raise ValueError(
            f"網格形狀 {grid_shape} 的總點數 {expected_total} "
            f"與快照矩陣行數 {n_locations} 不符"
        )

    if n_dims == 2:
        snap_grid = snapshots.reshape(grid_shape[0], grid_shape[1], n_snapshots)
        coord_grid = coords.reshape(grid_shape[0], grid_shape[1], n_dims)
    elif n_dims == 3:
        snap_grid = snapshots.reshape(
            grid_shape[0], grid_shape[1], grid_shape[2], n_snapshots
        )
        coord_grid = coords.reshape(grid_shape[0], grid_shape[1], grid_shape[2], n_dims)
    else:
        raise ValueError(f"不支援 {n_dims}D 網格（僅支援 2D/3D）")

    for ax in periodic_axes:
        if ax >= len(grid_shape):
            logger.warning(f"週期軸 {ax} 超出網格維度 {len(grid_shape)}，跳過")
            continue

        snap_grid = _wrap_grid_along_axis(snap_grid, ax, n_wrap_layers, seam_weight)
        coord_grid = _wrap_grid_along_axis(coord_grid, ax, n_wrap_layers, seam_weight=1.0)

    augmented_snapshots = snap_grid.reshape(-1, n_snapshots)
    augmented_coords_flat = coord_grid.reshape(-1, n_dims)

    if domain_lengths is not None:
        augmented_coords = _add_angular_embedding(
            augmented_coords_flat, periodic_axes, domain_lengths
        )
    else:
        augmented_coords = augmented_coords_flat

    logger.info(
        "循環快照矩陣建構完成：原始 %d → 增強 %d (+%d 環繞點)",
        n_locations,
        augmented_snapshots.shape[0],
        augmented_snapshots.shape[0] - n_locations,
    )

    return augmented_snapshots, augmented_coords


def _wrap_grid_along_axis(
    grid: np.ndarray, axis: int, n_layers: int, seam_weight: float = 1.0
) -> np.ndarray:
    """Wrap grid slices along an axis."""
    indices_front = np.arange(-n_layers, 0) % grid.shape[axis]
    indices_back = np.arange(0, n_layers) % grid.shape[axis]

    front_slice = np.take(grid, indices_front, axis=axis)
    back_slice = np.take(grid, indices_back, axis=axis)

    if seam_weight < 1.0:
        front_slice = front_slice * seam_weight
        back_slice = back_slice * seam_weight
        logger.debug(
            "環繞層降權：seam_weight=%.2f，軸 %d 的前後各 %d 層數值縮放至原始的 %d%%",
            seam_weight,
            axis,
            n_layers,
            int(100 * seam_weight),
        )

    wrapped_grid = np.concatenate([front_slice, grid, back_slice], axis=axis)

    return wrapped_grid


def _add_angular_embedding(
    coords: np.ndarray, periodic_axes: List[int], domain_lengths: Dict[int, float]
) -> np.ndarray:
    """Add sin/cos embeddings for periodic coordinates."""
    angular_features = []

    for ax in periodic_axes:
        if ax not in domain_lengths:
            logger.warning(f"週期軸 {ax} 缺少域長度資訊，跳過角度嵌入")
            continue

        length = domain_lengths[ax]
        x_periodic = coords[:, ax]
        theta = 2 * np.pi * x_periodic / length

        angular_features.append(np.sin(theta)[:, np.newaxis])
        angular_features.append(np.cos(theta)[:, np.newaxis])

    if angular_features:
        coords_with_angles = np.concatenate([coords] + angular_features, axis=1)
    else:
        coords_with_angles = coords

    return coords_with_angles


def prepare_turbulence_features(
    snapshots: Union[np.ndarray, Dict[str, np.ndarray]],
    method: str = "fluctuation",
    n_time_lags: int = 3,
) -> np.ndarray:
    """Prepare turbulence feature matrix."""
    if isinstance(snapshots, dict):
        variables = ["u", "v", "w"]
        processed_vars = []

        for var in variables:
            if var in snapshots:
                var_data = snapshots[var]

                if method == "fluctuation":
                    mean_field = var_data.mean(axis=0, keepdims=True)
                    fluctuation = var_data - mean_field
                    processed_vars.append(fluctuation)

                elif method == "time_lag":
                    lags = []
                    for lag in range(n_time_lags):
                        if lag < var_data.shape[0]:
                            lags.append(var_data[lag, :])
                    processed_vars.append(np.array(lags))

                elif method == "combined":
                    mean_field = var_data.mean(axis=0, keepdims=True)
                    fluctuation = var_data - mean_field
                    processed_vars.append(fluctuation)

                elif method == "raw":
                    processed_vars.append(var_data)

                else:
                    raise ValueError(f"未知的特徵提取方法: {method}")

        if method == "time_lag":
            feature_matrix = np.concatenate([v.T for v in processed_vars], axis=1)
        else:
            feature_matrix = np.concatenate([v.T for v in processed_vars], axis=1)

    else:
        if method == "fluctuation":
            mean_field = snapshots.mean(axis=0, keepdims=True)
            fluctuation = snapshots - mean_field
            feature_matrix = fluctuation.T

        elif method == "time_lag":
            lags = []
            for lag in range(n_time_lags):
                if lag < snapshots.shape[0]:
                    lags.append(snapshots[lag, :])
            feature_matrix = np.array(lags).T

        elif method == "combined":
            mean_field = snapshots.mean(axis=0, keepdims=True)
            fluctuation = snapshots - mean_field
            feature_matrix = fluctuation.T

        elif method == "raw":
            feature_matrix = snapshots.T

        else:
            raise ValueError(f"未知的特徵提取方法: {method}")

    return feature_matrix


def apply_min_distance_constraint(
    selected_indices: np.ndarray,
    coords: np.ndarray,
    min_distance: float,
    replacement_pool: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply minimum-distance constraint to selected sensors."""
    refined_indices = []
    selected_coords = []

    if replacement_pool is None:
        replacement_pool = np.arange(len(coords))

    for idx in selected_indices:
        coord = coords[idx]

        is_far_enough = True
        for sel_coord in selected_coords:
            dist = np.linalg.norm(coord - sel_coord)
            if dist < min_distance:
                is_far_enough = False
                break

        if is_far_enough:
            refined_indices.append(idx)
            selected_coords.append(coord)
        else:
            for candidate_idx in replacement_pool:
                if candidate_idx in refined_indices:
                    continue

                cand_coord = coords[candidate_idx]

                is_valid = True
                for sel_coord in selected_coords:
                    dist = np.linalg.norm(cand_coord - sel_coord)
                    if dist < min_distance:
                        is_valid = False
                        break

                if is_valid:
                    refined_indices.append(candidate_idx)
                    selected_coords.append(cand_coord)
                    break

    return np.array(refined_indices)
