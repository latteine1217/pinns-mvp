"""QR-pivot selector."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch
from scipy.linalg import qr, svd

from ..features import build_circular_snapshot_matrix

logger = logging.getLogger(__name__)


class QRPivotSelector:
    """QR-pivot sensor selector."""

    def __init__(
        self,
        mode: str = "column",
        pivoting: bool = True,
        regularization: float = 1e-12,
        use_circular_indexing: bool = False,
        n_wrap_layers: int = 1,
        seam_weight: float = 1.0,
        seam_width_fraction: float = 0.05,
        max_seam_fraction: float = 0.1,
    ):
        self.mode = mode
        self.pivoting = pivoting
        self.regularization = regularization
        self.use_circular_indexing = use_circular_indexing
        self.n_wrap_layers = n_wrap_layers
        self.seam_weight = seam_weight
        self.seam_width_fraction = seam_width_fraction
        self.max_seam_fraction = max_seam_fraction

    def select_sensors(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        coords: Optional[np.ndarray] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
        periodic_axes: Optional[List[int]] = None,
        domain_lengths: Optional[Dict[int, float]] = None,
        return_qr: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        coords_original = coords.copy() if coords is not None else None

        X = data_matrix.copy()
        n_locations, _ = X.shape

        original_n_locations = n_locations
        index_mapping = None
        original_seam_mask = None
        if coords_original is not None and periodic_axes is not None:
            original_seam_mask = self._compute_seam_mask(
                coords_original, periodic_axes, domain_lengths, grid_shape
            )

        if self.use_circular_indexing:
            if coords is None or grid_shape is None or periodic_axes is None:
                raise ValueError(
                    "循環索引模式需要提供 coords, grid_shape, periodic_axes 參數"
                )

            X_augmented, coords_augmented = build_circular_snapshot_matrix(
                snapshots=X,
                coords=coords,
                grid_shape=grid_shape,
                periodic_axes=periodic_axes,
                n_wrap_layers=self.n_wrap_layers,
                domain_lengths=domain_lengths,
                seam_weight=self.seam_weight,
            )

            index_mapping = self._build_index_mapping(
                n_original=original_n_locations,
                n_augmented=X_augmented.shape[0],
                grid_shape=grid_shape,
                periodic_axes=periodic_axes,
                n_wrap_layers=self.n_wrap_layers,
            )

            X = X_augmented
            coords = coords_augmented
            n_locations = X.shape[0]

            logger.info(
                "循環索引啟用：原始 %d 點 → 增強 %d 點 (環繞層 %d)",
                original_n_locations,
                n_locations,
                self.n_wrap_layers,
            )
            augmented_seam_mask = self._compute_seam_mask(
                coords, periodic_axes, domain_lengths, grid_shape
            )
        else:
            augmented_seam_mask = None

        X_mean = X.mean(axis=0, keepdims=True)
        X_std = X.std(axis=0, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std

        if self.use_circular_indexing and self.seam_weight < 1.0 and augmented_seam_mask is not None:
            weights = np.ones(n_locations, dtype=X.dtype)
            weights[augmented_seam_mask] = self.seam_weight
            X = (weights[:, None] ** 0.5) * X

        n_sensors = min(n_sensors, n_locations)

        Q = None
        R = None
        try:
            if self.pivoting:
                if self.mode == "column":
                    Q, R, piv = qr(X.T, mode="economic", pivoting=True)
                    selected_indices = piv[:n_sensors]
                else:
                    Q, R, piv = qr(X.T, mode="economic", pivoting=True)
                    selected_indices = piv[:n_sensors]
            else:
                Q, R = qr(X.T if self.mode == "column" else X, mode="economic")
                diag_importance = np.abs(np.diag(R))
                selected_indices = np.argsort(diag_importance)[-n_sensors:][::-1]

        except np.linalg.LinAlgError as exc:
            logger.warning(f"QR 分解失敗，使用 SVD 回退: {exc}")
            _, s, Vt = svd(X, full_matrices=False)
            importance = np.sum(np.abs(Vt.T) * s, axis=1)
            selected_indices = np.argsort(importance)[-n_sensors:][::-1]

        selected_indices = selected_indices[selected_indices < n_locations]
        selected_indices = selected_indices[:n_sensors]

        if self.use_circular_indexing and index_mapping is not None:
            selected_indices_original = index_mapping[selected_indices]

            unique_indices, unique_counts = np.unique(
                selected_indices_original, return_counts=True
            )

            fallback_added = 0

            if len(unique_indices) < n_sensors:
                logger.warning(
                    "循環索引去重後點數不足：%d < %d，保留所有唯一點",
                    len(unique_indices),
                    n_sensors,
                )
                selected_indices_final = unique_indices

                remaining = n_sensors - len(selected_indices_final)
                candidate_mask = np.ones(original_n_locations, dtype=bool)
                candidate_mask[selected_indices_final] = False
                if original_seam_mask is not None:
                    candidate_mask &= ~original_seam_mask

                candidate_indices = np.where(candidate_mask)[0]

                if candidate_indices.size > 0 and remaining > 0:
                    extra_indices = self._select_far_from_seam(
                        candidate_indices,
                        remaining,
                        coords_original,
                        periodic_axes,
                    )

                    selected_indices_final = np.concatenate(
                        [selected_indices_final, extra_indices]
                    )
                    fallback_added = len(extra_indices)
                else:
                    logger.warning("沒有可用的非接縫候選點可供回補")
            else:
                sort_by_importance = np.argsort(unique_counts)[::-1]
                selected_indices_final = unique_indices[sort_by_importance[:n_sensors]]

            seam_cap_applied = 0
            if (
                original_seam_mask is not None
                and self.max_seam_fraction < 1.0
                and len(selected_indices_final) > 0
            ):
                seam_flags = original_seam_mask[selected_indices_final]
                seam_count = int(seam_flags.sum())
                max_allowed = int(
                    np.floor(self.max_seam_fraction * len(selected_indices_final))
                )
                max_allowed = max(max_allowed, 0)
                if seam_count > max_allowed:
                    excess = seam_count - max_allowed
                    seam_indices = selected_indices_final[seam_flags]
                    to_remove = seam_indices[:excess]

                    keep_mask = np.ones(len(selected_indices_final), dtype=bool)
                    remove_set = set(to_remove.tolist())
                    for i, idx in enumerate(selected_indices_final):
                        if idx in remove_set:
                            keep_mask[i] = False
                            remove_set.remove(idx)
                            if not remove_set:
                                break

                    selected_indices_pruned = selected_indices_final[keep_mask]

                    candidate_mask = np.ones(original_n_locations, dtype=bool)
                    candidate_mask[selected_indices_pruned] = False
                    candidate_mask &= ~original_seam_mask
                    candidate_indices = np.where(candidate_mask)[0]

                    replacements = self._select_far_from_seam(
                        candidate_indices,
                        excess,
                        coords_original,
                        periodic_axes,
                    )

                    selected_indices_final = np.concatenate(
                        [selected_indices_pruned, replacements]
                    )
                    seam_cap_applied = excess - len(replacements)
                    fallback_added += len(replacements)
                    if len(replacements) < excess:
                        logger.warning("接縫比例限制：替換候選不足，已縮減接縫點數但未達目標比例")

            logger.info(
                "循環索引映射：增強網格 %d 點 → 原始網格 %d 點（去重 %d）",
                len(selected_indices),
                len(selected_indices_final),
                len(selected_indices) - len(unique_indices),
            )

            selected_indices = selected_indices_final

            metrics = self._compute_metrics(data_matrix, selected_indices)
            metrics["circular_indexing_enabled"] = True
            metrics["n_wrap_layers"] = self.n_wrap_layers
            metrics["n_duplicates_removed"] = int(
                len(selected_indices_original) - len(unique_indices)
            )
            if len(unique_indices) < n_sensors:
                metrics["fallback_interior_added"] = int(fallback_added)
            if seam_cap_applied:
                metrics["seam_cap_residual"] = int(seam_cap_applied)
            if original_seam_mask is not None and len(selected_indices) > 0:
                seam_selected = int(original_seam_mask[selected_indices].sum())
                metrics["seam_selected_count"] = seam_selected
                metrics["seam_selected_ratio"] = seam_selected / len(selected_indices)
                metrics["seam_weight"] = self.seam_weight
                metrics["seam_width_fraction"] = self.seam_width_fraction
        else:
            metrics = self._compute_metrics(data_matrix, selected_indices)
            metrics["circular_indexing_enabled"] = False
            if original_seam_mask is not None and len(selected_indices) > 0:
                seam_selected = int(original_seam_mask[selected_indices].sum())
                metrics["seam_selected_count"] = seam_selected
                metrics["seam_selected_ratio"] = seam_selected / len(selected_indices)

        result = (selected_indices, metrics)
        if return_qr:
            result = (*result, Q, R)

        return result

    def select_sensors_per_feature(
        self,
        data_matrix: np.ndarray,
        n_sensors_per_feature: int,
        coords: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        return_details: bool = False,
    ):
        if isinstance(data_matrix, torch.Tensor):
            data_matrix = data_matrix.detach().cpu().numpy()
        if coords is not None and isinstance(coords, torch.Tensor):
            coords = coords.detach().cpu().numpy()

        n_locations, n_features = data_matrix.shape

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        logger.info("\n🔍 Per-Feature QR-Pivot Selection:")
        logger.info(f"   Features: {n_features}")
        logger.info(f"   Sensors per feature: {n_sensors_per_feature}")
        logger.info(f"   Max total sensors: {n_features * n_sensors_per_feature}")

        X_mean = data_matrix.mean(axis=0, keepdims=True)
        X_std = data_matrix.std(axis=0, keepdims=True) + 1e-8
        data_normalized = (data_matrix - X_mean) / X_std

        per_feature_indices = {}
        per_feature_details = {}
        all_indices_list = []

        for i, fname in enumerate(feature_names):
            feature_col = data_normalized[:, i : i + 1]

            try:
                Q, R, piv = qr(feature_col.T, mode="economic", pivoting=True)

                n_select = min(n_sensors_per_feature, n_locations)
                feature_indices = piv[:n_select]

                r_diag = np.abs(np.diag(R)[:n_select])

                per_feature_indices[fname] = feature_indices
                per_feature_details[fname] = {
                    "indices": feature_indices,
                    "importance": r_diag,
                    "feature_index": i,
                    "n_selected": len(feature_indices),
                }

                all_indices_list.append(feature_indices)

                logger.info(
                    "   ✓ %-12s: selected %2d points, importance range [%.2e, %.2e]",
                    fname,
                    len(feature_indices),
                    r_diag.min(),
                    r_diag.max(),
                )

            except Exception as exc:
                logger.warning(f"   ✗ {fname:12s}: QR failed ({exc}), skipping")
                per_feature_indices[fname] = np.array([], dtype=int)
                per_feature_details[fname] = {
                    "indices": np.array([], dtype=int),
                    "importance": np.array([]),
                    "feature_index": i,
                    "n_selected": 0,
                    "error": str(exc),
                }

        all_indices = (
            np.concatenate(all_indices_list) if all_indices_list else np.array([], dtype=int)
        )
        unique_indices, unique_counts = np.unique(all_indices, return_counts=True)

        sort_by_importance = np.argsort(unique_counts)[::-1]
        selected_indices_final = unique_indices[sort_by_importance]

        logger.info("\n📊 Merging Results:")
        logger.info(f"   Total indices collected: {len(all_indices)}")
        logger.info(f"   Unique sensors after deduplication: {len(unique_indices)}")
        logger.info(
            "   Reduction: %d duplicates removed (%.1f%%)",
            len(all_indices) - len(unique_indices),
            (1 - len(unique_indices) / max(len(all_indices), 1)) * 100,
        )

        multi_feature_sensors = unique_counts > 1
        if multi_feature_sensors.any():
            logger.info(
                "   Multi-feature sensors: %d points selected by ≥2 features",
                multi_feature_sensors.sum(),
            )
            max_count = unique_counts.max()
            logger.info("   Most important sensor: selected by %d features", max_count)

        metrics = self._compute_metrics(data_matrix, selected_indices_final)
        metrics["n_features"] = n_features
        metrics["n_sensors_per_feature"] = n_sensors_per_feature
        metrics["n_total_selected"] = len(selected_indices_final)
        metrics["deduplication_rate"] = float(
            1 - len(unique_indices) / max(len(all_indices), 1)
        )
        metrics["multi_feature_sensors"] = int(multi_feature_sensors.sum())
        metrics["max_feature_count"] = int(unique_counts.max())

        if coords is not None:
            selected_coords = coords[selected_indices_final]
            for dim in range(coords.shape[1]):
                coord_name = ["x", "y", "z"][dim] if dim < 3 else f"dim{dim}"
                metrics[f"{coord_name}_mean"] = float(selected_coords[:, dim].mean())
                metrics[f"{coord_name}_std"] = float(selected_coords[:, dim].std())
                metrics[f"{coord_name}_range"] = float(selected_coords[:, dim].ptp())

        if return_details:
            return selected_indices_final, metrics, per_feature_details
        return selected_indices_final, metrics

    def _select_far_from_seam(
        self,
        candidate_indices: np.ndarray,
        count: int,
        coords_original: Optional[np.ndarray],
        periodic_axes: Optional[List[int]],
    ) -> np.ndarray:
        if count <= 0 or candidate_indices.size == 0:
            return np.array([], dtype=int)

        if coords_original is None or periodic_axes is None:
            return candidate_indices[:count]

        candidate_coords = coords_original[candidate_indices]

        seam_dist = None
        for axis in periodic_axes:
            if axis >= candidate_coords.shape[1]:
                continue
            axis_vals = candidate_coords[:, axis]
            axis_min = coords_original[:, axis].min()
            axis_max = coords_original[:, axis].max()
            axis_dist = np.minimum(axis_vals - axis_min, axis_max - axis_vals)
            seam_dist = axis_dist if seam_dist is None else np.minimum(seam_dist, axis_dist)

        if seam_dist is None:
            seam_dist = np.ones(candidate_indices.size)

        order = np.argsort(seam_dist)
        return candidate_indices[order[-count:]]

    def _compute_seam_mask(
        self,
        coords: np.ndarray,
        periodic_axes: Optional[List[int]],
        domain_lengths: Optional[Dict[int, float]] = None,
        grid_shape: Optional[Tuple[int, ...]] = None,
    ) -> Optional[np.ndarray]:
        if periodic_axes is None:
            return None

        n_points = coords.shape[0]
        seam_mask = np.zeros(n_points, dtype=bool)

        for axis in periodic_axes:
            if axis >= coords.shape[1]:
                continue

            axis_vals = coords[:, axis]
            coord_min = axis_vals.min()
            coord_max = axis_vals.max()

            if domain_lengths and axis in domain_lengths:
                domain_length = float(domain_lengths[axis])
            else:
                domain_length = coord_max - coord_min

            if domain_length <= 0:
                continue

            seam_width = self.seam_width_fraction * domain_length
            if grid_shape is not None and axis < len(grid_shape):
                cell_width = domain_length / grid_shape[axis]
                seam_width = max(seam_width, (self.n_wrap_layers + 1) * cell_width)
            seam_width = max(seam_width, 1e-12)
            lower_bound = coord_min + seam_width
            upper_bound = coord_max - seam_width

            seam_mask |= (axis_vals <= lower_bound) | (axis_vals >= upper_bound)

        return seam_mask

    def _build_index_mapping(
        self,
        n_original: int,
        n_augmented: int,
        grid_shape: Tuple[int, ...],
        periodic_axes: List[int],
        n_wrap_layers: int,
    ) -> np.ndarray:
        n_dims = len(grid_shape)

        augmented_shape = list(grid_shape)
        for ax in periodic_axes:
            if ax < len(augmented_shape):
                augmented_shape[ax] += 2 * n_wrap_layers

        augmented_shape = tuple(augmented_shape)

        augmented_linear = np.arange(n_augmented)
        augmented_multi = np.unravel_index(augmented_linear, augmented_shape, order="C")

        original_multi = list(augmented_multi)
        for ax in periodic_axes:
            if ax < len(original_multi):
                original_multi[ax] = (augmented_multi[ax] - n_wrap_layers) % grid_shape[ax]

        original_linear = np.ravel_multi_index(tuple(original_multi), grid_shape, order="C")

        return original_linear

    def _compute_metrics(
        self, data_matrix: np.ndarray, selected_indices: np.ndarray
    ) -> Dict[str, float]:
        selected_data = data_matrix[selected_indices, :]

        try:
            _, s, _ = svd(selected_data, full_matrices=False)
            cond_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
        except Exception:
            cond_number = np.inf

        try:
            det_value = np.linalg.det(
                selected_data @ selected_data.T
                + self.regularization * np.eye(len(selected_indices))
            )
            log_det = np.log(max(det_value, 1e-16))
        except Exception:
            log_det = -np.inf

        coverage = 0.0
        energy_ratio = 0.0

        try:
            U_full, s_full, Vt_full = svd(data_matrix, full_matrices=False)
            U_selected, s_selected, Vt_selected = svd(selected_data, full_matrices=False)

            if len(s_selected) > 0 and len(s_full) > 0:
                n_compare = min(
                    len(s_selected), len(s_full), min(Vt_full.shape[1], Vt_selected.shape[1])
                )

                overlap = Vt_full[:n_compare, :] @ Vt_selected[:n_compare, :].conj().T
                coverage = float(np.linalg.norm(overlap, "fro") ** 2 / n_compare)

                coverage_energy = float(coverage)

                reconstruction_energy = 0.0
                try:
                    from sklearn.linear_model import Ridge

                    ridge = Ridge(alpha=1e-6, fit_intercept=False)
                    ridge.fit(selected_data.T, data_matrix.T)

                    reconstructed = ridge.predict(selected_data.T).T

                    total_energy = np.linalg.norm(data_matrix, "fro") ** 2
                    residual_energy = np.linalg.norm(data_matrix - reconstructed, "fro") ** 2
                    reconstruction_energy = 1.0 - residual_energy / (total_energy + 1e-16)
                    reconstruction_energy = max(0.0, min(1.0, reconstruction_energy))

                except ImportError:
                    reconstruction_energy = coverage_energy
                except Exception as exc:
                    logger.debug(f"重建能量計算失敗: {exc}")
                    reconstruction_energy = coverage_energy

                if reconstruction_energy > 0.0:
                    energy_ratio = float(reconstruction_energy)
                else:
                    energy_ratio = float(coverage_energy)

        except Exception:
            pass

        return {
            "condition_number": float(cond_number),
            "log_determinant": float(log_det),
            "subspace_coverage": float(coverage),
            "energy_ratio": float(energy_ratio),
            "n_sensors": len(selected_indices),
        }
