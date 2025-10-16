"""
Field-aware sensor selection utilities.

This module builds upon the QR-pivot sensor selection toolkit and provides
an opinionated, type-safe pipeline to operate directly on flow fields.
It supports both 3D volumes and 2D slices, aggregates vector components
(`u`, `v`, `w`, `p`), and exposes a unified interface that returns sensor
indices together with raw field values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from .qr_pivot import QRPivotSelector, SensorOptimizer, create_sensor_selector, evaluate_sensor_placement

ArrayLike = Union[np.ndarray, torch.Tensor]
FieldInput = Union[Mapping[str, ArrayLike], ArrayLike]
CoordInput = Union[
    ArrayLike,
    Sequence[ArrayLike],
    Mapping[str, ArrayLike],
    None,
]


def _unwrap_object_array(value: ArrayLike) -> Any:
    """Extract Python objects from 0-d object ndarrays."""
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype == object:
        return value.item()
    return value


def _to_numpy(array: ArrayLike, copy: bool = False) -> np.ndarray:
    """Convert torch / numpy array to numpy ndarray."""
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    return array.copy() if copy else array


@dataclass(frozen=True)
class FeatureScaling:
    """Simple container for column-wise scaling metadata."""

    mode: str
    mean: Optional[np.ndarray] = None
    scale: Optional[np.ndarray] = None


@dataclass(frozen=True)
class FieldSensorMetadata:
    """Metadata accompanying a field-aware sensor matrix."""

    components: Tuple[str, ...]
    feature_slices: Dict[str, slice]
    component_feature_counts: Dict[str, int]
    spatial_shape: Optional[Tuple[int, ...]]
    flatten_order: str
    n_locations: int
    n_features: int
    flat_coordinates: Optional[np.ndarray]
    scaling: Optional[FeatureScaling]

    def unravel_indices(self, indices: np.ndarray) -> np.ndarray:
        if self.spatial_shape is None:
            raise ValueError("Spatial shape unavailable; provide coordinates or spatial_shape to enable unravel.")
        unraveled = np.array(
            np.unravel_index(indices, self.spatial_shape, order=self.flatten_order)
        ).T
        return unraveled


@dataclass(frozen=True)
class FieldSensorSelection:
    """Output bundle for field-aware sensor selection."""

    indices: np.ndarray
    metrics: Dict[str, Any]
    metadata: FieldSensorMetadata
    component_values: Dict[str, np.ndarray]

    def coordinates(self) -> Optional[np.ndarray]:
        coords = self.metadata.flat_coordinates
        if coords is None:
            return None
        return coords[self.indices]

    def unravel(self) -> np.ndarray:
        return self.metadata.unravel_indices(self.indices)

    def to_npz_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            'sensor_indices': self.indices,
            'sensor_data': self.component_values,
            'components': self.metadata.components,
        }
        coords = self.coordinates()
        if coords is not None:
            payload['sensor_points'] = coords
        payload['metrics'] = self.metrics
        return payload


class FieldSensorSelector:
    """
    Unified helper that maps flow fields to sensor selections.

    The selector accepts raw field dictionaries, e.g. ``{'u': u_field, 'v': v_field}``,
    optional coordinate definitions, and applies a QR-pivot (or alternative strategy)
    after constructing a consistent snapshot matrix.
    """

    def __init__(
        self,
        strategy: str = 'qr_pivot',
        selector_kwargs: Optional[MutableMapping[str, Any]] = None,
        components: Sequence[str] = ('u', 'v', 'w', 'p'),
        flatten_order: str = 'C',
        component_weights: Optional[Mapping[str, float]] = None,
        feature_scaling: str = 'none',
        nan_policy: str = 'raise',
        spatial_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        self.strategy = strategy
        self.selector_kwargs = dict(selector_kwargs or {})
        self.components = tuple(components)
        self.flatten_order = flatten_order
        self.component_weights = dict(component_weights or {})
        self.feature_scaling = feature_scaling
        self.nan_policy = nan_policy
        self.spatial_shape_override = spatial_shape

        if feature_scaling not in {'none', 'standard', 'l2'}:
            raise ValueError(f"Unsupported feature_scaling mode: {feature_scaling}")

        if nan_policy not in {'raise', 'zero'}:
            raise ValueError(f"Unsupported nan_policy: {nan_policy}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def select(
        self,
        field: FieldInput,
        n_sensors: int,
        coords: CoordInput = None,
        validation_field: Optional[FieldInput] = None,
        validation_coords: CoordInput = None,
    ) -> FieldSensorSelection:
        """Execute sensor selection on field data."""
        component_arrays, coord_hint = self._extract_components_and_coords(field, coords)
        data_matrix, metadata = self._build_training_matrix(component_arrays, coord_hint)

        validation_matrix = None
        if validation_field is not None:
            val_arrays, val_coord_hint = self._extract_components_and_coords(
                validation_field, validation_coords
            )
            validation_matrix = self._build_matrix_with_metadata(
                val_arrays, val_coord_hint, metadata
            )

        indices, metrics = self._run_strategy(data_matrix, n_sensors, validation_matrix)

        if validation_matrix is not None and self.strategy != 'auto':
            val_metrics = evaluate_sensor_placement(
                data_matrix, indices, test_data=validation_matrix
            )
            metrics['validation'] = val_metrics

        component_values = self._gather_component_values(
            component_arrays, metadata, indices
        )

        return FieldSensorSelection(
            indices=indices,
            metrics=metrics,
            metadata=metadata,
            component_values=component_values,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _extract_components_and_coords(
        self,
        field: FieldInput,
        coords_override: CoordInput = None,
    ) -> Tuple[Dict[str, np.ndarray], CoordInput]:
        """Separate component arrays and coordinate hints."""
        component_arrays: Dict[str, np.ndarray] = {}
        coord_hint: CoordInput = coords_override

        if isinstance(field, Mapping):
            # handle np.load(..., allow_pickle=True)
            mapping: Mapping[str, Any] = {
                key: _unwrap_object_array(val) for key, val in field.items()
            }

            for comp in self.components:
                if comp in mapping and mapping[comp] is not None:
                    component_arrays[comp] = _to_numpy(mapping[comp])

            if coord_hint is None:
                coord_hint = self._extract_coordinate_hint_from_mapping(mapping)
        else:
            array = _to_numpy(field)
            if array.ndim == 0:
                raise ValueError("Field array must be at least 1-D.")
            n_available = array.shape[-1]
            for idx, comp in enumerate(self.components):
                if idx >= n_available:
                    break
                component_arrays[comp] = array[..., idx]

        if not component_arrays:
            raise ValueError(
                f"No matching components found. Expected one of {self.components}."
            )

        return component_arrays, coord_hint

    def _extract_coordinate_hint_from_mapping(
        self,
        mapping: Mapping[str, Any],
    ) -> CoordInput:
        """Discover coordinate definitions embedded in mapping."""
        if 'coordinates' in mapping:
            coords_raw = _unwrap_object_array(mapping['coordinates'])
            if isinstance(coords_raw, Mapping):
                return {
                    key: _to_numpy(value)
                    for key, value in coords_raw.items()
                    if key in {'x', 'y', 'z'}
                }
            return _to_numpy(coords_raw)

        axes = [axis for axis in ('x', 'y', 'z') if axis in mapping]
        if axes:
            return tuple(_to_numpy(mapping[axis]) for axis in axes)

        return None

    def _build_training_matrix(
        self,
        component_arrays: Mapping[str, np.ndarray],
        coord_hint: CoordInput,
    ) -> Tuple[np.ndarray, FieldSensorMetadata]:
        coords_flat, spatial_shape = self._prepare_coordinates(coord_hint)
        if spatial_shape is None:
            spatial_shape = self.spatial_shape_override

        used_components = tuple(
            comp for comp in self.components if comp in component_arrays
        )
        if not used_components:
            raise ValueError("Component arrays empty after filtering.")

        if spatial_shape is None:
            sample = _to_numpy(component_arrays[used_components[0]])
            sample = np.squeeze(sample)
            if sample.ndim == 1:
                spatial_shape = (sample.shape[0],)
            else:
                raise ValueError(
                    "Unable to infer spatial_shape. Provide `coords` or specify `spatial_shape`."
                )

        n_locations = int(np.prod(spatial_shape))

        feature_slices: Dict[str, slice] = {}
        component_feature_counts: Dict[str, int] = {}
        column_start = 0
        stacked_components: List[np.ndarray] = []

        for comp in used_components:
            flattened, n_features = self._reshape_component(
                component_arrays[comp], spatial_shape
            )
            weight = self.component_weights.get(comp, 1.0)
            if weight != 1.0:
                flattened = flattened * weight

            stacked_components.append(flattened)
            feature_slices[comp] = slice(column_start, column_start + flattened.shape[1])
            component_feature_counts[comp] = n_features
            column_start += flattened.shape[1]

        data_matrix = np.concatenate(stacked_components, axis=1)
        data_matrix = self._handle_nan(data_matrix)
        scaled_matrix, scaling = self._apply_scaling(data_matrix, None)

        metadata = FieldSensorMetadata(
            components=used_components,
            feature_slices=feature_slices,
            component_feature_counts=component_feature_counts,
            spatial_shape=spatial_shape,
            flatten_order=self.flatten_order,
            n_locations=n_locations,
            n_features=scaled_matrix.shape[1],
            flat_coordinates=coords_flat,
            scaling=scaling,
        )

        return scaled_matrix, metadata

    def _build_matrix_with_metadata(
        self,
        component_arrays: Mapping[str, np.ndarray],
        coord_hint: CoordInput,
        metadata: FieldSensorMetadata,
    ) -> np.ndarray:
        if coord_hint is not None:
            coords_flat, spatial_shape = self._prepare_coordinates(coord_hint)
            if spatial_shape is not None and metadata.spatial_shape is not None:
                if tuple(spatial_shape) != tuple(metadata.spatial_shape):
                    raise ValueError(
                        f"Validation spatial shape {spatial_shape} does not match training shape {metadata.spatial_shape}."
                    )
            if coords_flat is not None and metadata.flat_coordinates is not None:
                if coords_flat.shape != metadata.flat_coordinates.shape:
                    raise ValueError("Coordinate hint mismatch between training and validation data.")

        stacked_components: List[np.ndarray] = []
        for comp in metadata.components:
            if comp not in component_arrays:
                raise KeyError(f"Validation field missing component '{comp}'.")
            flattened, _ = self._reshape_component(
                component_arrays[comp],
                metadata.spatial_shape or (metadata.n_locations,),
            )
            weight = self.component_weights.get(comp, 1.0)
            if weight != 1.0:
                flattened = flattened * weight
            stacked_components.append(flattened)

        matrix = np.concatenate(stacked_components, axis=1)
        matrix = self._handle_nan(matrix)
        scaled_matrix, _ = self._apply_scaling(matrix, metadata.scaling)
        return scaled_matrix

    def _prepare_coordinates(
        self, coords: CoordInput
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, ...]]]:
        if coords is None:
            return None, None

        coords = _unwrap_object_array(coords)

        if isinstance(coords, Mapping):
            axes = [axis for axis in ('x', 'y', 'z') if axis in coords]
            if not axes:
                return None, None
            axis_arrays = [np.asarray(_unwrap_object_array(coords[axis])) for axis in axes]
            mesh = np.stack(np.meshgrid(*axis_arrays, indexing='ij'), axis=-1)
            flat_coords = mesh.reshape(-1, len(axis_arrays))
            spatial_shape = tuple(len(axis) for axis in axis_arrays)
            return flat_coords, spatial_shape

        if isinstance(coords, (list, tuple)):
            axis_arrays = [np.asarray(_unwrap_object_array(axis)) for axis in coords]
            mesh = np.stack(np.meshgrid(*axis_arrays, indexing='ij'), axis=-1)
            flat_coords = mesh.reshape(-1, len(axis_arrays))
            spatial_shape = tuple(len(axis) for axis in axis_arrays)
            return flat_coords, spatial_shape

        coords_array = np.asarray(coords)
        if coords_array.ndim == 1:
            coords_array = coords_array.reshape(-1, 1)
        elif coords_array.ndim > 2:
            coords_array = coords_array.reshape(coords_array.shape[0], -1)

        return coords_array, None

    def _reshape_component(
        self,
        component_array: np.ndarray,
        spatial_shape: Tuple[int, ...],
    ) -> Tuple[np.ndarray, int]:
        arr = _to_numpy(component_array)
        arr = np.squeeze(arr)

        n_points = int(np.prod(spatial_shape))
        total_values = arr.size
        if total_values % n_points != 0:
            raise ValueError(
                f"Component array size {total_values} is incompatible with spatial shape {spatial_shape}."
            )

        n_features = total_values // n_points
        reshaped = arr.reshape(*spatial_shape, n_features, order=self.flatten_order)
        flattened = reshaped.reshape(n_points, n_features, order=self.flatten_order)
        return flattened, n_features

    def _handle_nan(self, matrix: np.ndarray) -> np.ndarray:
        if not np.isnan(matrix).any():
            return matrix

        if self.nan_policy == 'raise':
            raise ValueError("NaN encountered in data matrix. Consider setting nan_policy='zero'.")
        return np.nan_to_num(matrix, nan=0.0)

    def _apply_scaling(
        self,
        matrix: np.ndarray,
        scaling: Optional[FeatureScaling],
    ) -> Tuple[np.ndarray, Optional[FeatureScaling]]:
        if self.feature_scaling == 'none':
            return matrix, None if scaling is None else scaling

        if scaling is None:
            if self.feature_scaling == 'standard':
                mean = np.mean(matrix, axis=0, keepdims=True)
                std = np.std(matrix, axis=0, keepdims=True)
                std[std < 1e-12] = 1.0
                scaled = (matrix - mean) / std
                return scaled, FeatureScaling(mode='standard', mean=mean, scale=std)

            if self.feature_scaling == 'l2':
                norm = np.linalg.norm(matrix, axis=0, keepdims=True)
                norm[norm < 1e-12] = 1.0
                scaled = matrix / norm
                return scaled, FeatureScaling(mode='l2', scale=norm)
        else:
            if scaling.mode != self.feature_scaling:
                raise ValueError(
                    f"Scaling mismatch. Expected mode '{self.feature_scaling}', got '{scaling.mode}'."
                )
            if scaling.mode == 'standard':
                if scaling.mean is None or scaling.scale is None:
                    raise ValueError("Standard scaling requires mean and scale.")
                scaled = (matrix - scaling.mean) / scaling.scale
                return scaled, scaling
            if scaling.mode == 'l2':
                if scaling.scale is None:
                    raise ValueError("L2 scaling requires the scale vector.")
                scaled = matrix / scaling.scale
                return scaled, scaling

        raise ValueError(f"Unsupported scaling configuration: mode={self.feature_scaling}")

    def _run_strategy(
        self,
        data_matrix: np.ndarray,
        n_sensors: int,
        validation_matrix: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.strategy == 'auto':
            optimizer = SensorOptimizer(strategy='auto', config=self.selector_kwargs or {})
            indices, metrics = optimizer.optimize_sensor_placement(
                data_matrix, n_sensors, validation_matrix
            )
            return indices, metrics

        selector = create_sensor_selector(self.strategy, **self.selector_kwargs)
        indices, metrics = selector.select_sensors(data_matrix, n_sensors)
        return indices, metrics

    def _gather_component_values(
        self,
        component_arrays: Mapping[str, np.ndarray],
        metadata: FieldSensorMetadata,
        indices: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        values: Dict[str, np.ndarray] = {}
        spatial_shape = metadata.spatial_shape or (metadata.n_locations,)

        for comp in metadata.components:
            flattened, _ = self._reshape_component(
                component_arrays[comp], spatial_shape
            )
            values[comp] = flattened[indices]

        return values


__all__ = [
    'FieldSensorSelector',
    'FieldSensorSelection',
    'FieldSensorMetadata',
    'FeatureScaling',
]
