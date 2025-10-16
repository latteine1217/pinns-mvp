"""
Core data-structure utilities for flow datasets.

Provides unified representations for structured grids, point samples,
and training bundles so that 2D slices and full 3D fields share the
same processing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Sequence, Optional, Any, List, Iterable
from collections import OrderedDict
import numpy as np


DEFAULT_AXIS_ORDER: Tuple[str, ...] = ("x", "y", "z", "t")


def _ensure_ordered_axes(axes: Dict[str, np.ndarray],
                         axis_order: Sequence[str] = DEFAULT_AXIS_ORDER) -> OrderedDict:
    """Normalize axis dictionary into an ordered mapping."""
    ordered = OrderedDict()
    # respect provided order first
    for axis in axis_order:
        if axis in axes:
            ordered[axis] = np.asarray(axes[axis], dtype=np.float64)
    # append any remaining axes with deterministic ordering
    for axis, values in sorted(axes.items()):
        if axis not in ordered:
            ordered[axis] = np.asarray(values, dtype=np.float64)
    return ordered


@dataclass(frozen=True)
class StructuredGrid:
    """Represents a structured (tensor) grid of coordinates."""

    axes: OrderedDict[str, np.ndarray]

    @classmethod
    def from_axes(cls,
                  axes: Dict[str, np.ndarray],
                  axis_order: Sequence[str] = DEFAULT_AXIS_ORDER) -> "StructuredGrid":
        ordered = _ensure_ordered_axes(axes, axis_order)
        for name, values in ordered.items():
            if values.ndim != 1:
                raise ValueError(f"Axis '{name}' must be 1D, got shape {values.shape}")
        return cls(ordered)

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(self.axes.keys())

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(len(values) for values in self.axes.values())

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def axis_values(self, axis: str) -> np.ndarray:
        return self.axes[axis]

    def axis_index(self, axis: str) -> int:
        try:
            return self.axis_names.index(axis)
        except ValueError as exc:
            raise KeyError(f"Axis '{axis}' not present in grid") from exc

    def to_points(self, order: Sequence[str] = ("x", "y", "z")) -> np.ndarray:
        """Return flattened coordinates with the specified axis ordering."""
        available_axes = self.axis_names
        mesh = np.meshgrid(*[self.axes[axis] for axis in available_axes], indexing="ij")
        flat_coords = {}
        for axis, values in zip(available_axes, mesh):
            flat_coords[axis] = values.reshape(-1, 1)
        coords = []
        for axis in order:
            if axis in flat_coords:
                coords.append(flat_coords[axis])
            else:
                coords.append(np.zeros((mesh[0].size, 1), dtype=np.float64))
        return np.concatenate(coords, axis=1)

    def slice(self,
              axis: str,
              *,
              index: Optional[int] = None,
              value: Optional[float] = None,
              atol: float = 1e-8) -> Tuple["StructuredGrid", int, float]:
        """Return a reduced grid with a slice taken along `axis`."""
        axis_idx = self.axis_index(axis)
        axis_values = self.axis_values(axis)
        if index is None:
            if value is None:
                raise ValueError("Either index or value must be provided for slicing.")
            nearest = np.argmin(np.abs(axis_values - value))
            if np.abs(axis_values[nearest] - value) > atol:
                raise ValueError(
                    f"Requested slice at {value} for axis '{axis}' not aligned with grid "
                    f"(nearest value {axis_values[nearest]})"
                )
            index = int(nearest)
        slice_value = float(axis_values[index])
        new_axes = OrderedDict(
            (name, values) for i, (name, values) in enumerate(self.axes.items())
            if i != axis_idx
        )
        return StructuredGrid(new_axes), index, slice_value


@dataclass
class StructuredField:
    """Structured field data defined on a StructuredGrid."""

    grid: StructuredGrid
    fields: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected_shape = self.grid.shape
        for name, values in list(self.fields.items()):
            arr = np.asarray(values)
            if arr.shape != expected_shape:
                raise ValueError(
                    f"Field '{name}' has shape {arr.shape}, expected {expected_shape}"
                )
            self.fields[name] = arr

    def variables(self) -> Tuple[str, ...]:
        return tuple(self.fields.keys())

    def to_points(self,
                  order: Sequence[str] = ("x", "y", "z"),
                  fields: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        coords = self.grid.to_points(order)
        if fields is None:
            field_names = self.variables()
        else:
            field_names = tuple(f for f in fields if f in self.fields)
        flat_fields = {
            name: self.fields[name].reshape(-1, 1)
            for name in field_names
        }
        return coords, flat_fields

    def slice(self,
              axis: str,
              *,
              index: Optional[int] = None,
              value: Optional[float] = None,
              atol: float = 1e-8) -> "StructuredField":
        sliced_grid, axis_index, slice_value = self.grid.slice(axis, index=index, value=value, atol=atol)
        sliced_fields = {}
        for name, array in self.fields.items():
            sliced_fields[name] = np.take(array, axis_index, axis=self.grid.axis_index(axis))
        metadata = {
            **self.metadata,
            "slice_axis": axis,
            "slice_value": slice_value,
            "slice_index": axis_index,
        }
        return StructuredField(grid=sliced_grid, fields=sliced_fields, metadata=metadata)

    def __getitem__(self, item: str) -> np.ndarray:
        return self.fields[item]

    @property
    def sensor_points(self) -> np.ndarray:
        coords, _ = self.to_points(order=('x', 'y', 'z'))
        return coords

    @property
    def sensor_data(self) -> Dict[str, np.ndarray]:
        _, fields = self.to_points(order=('x', 'y', 'z'))
        return {name: values.reshape(-1) for name, values in fields.items()}


@dataclass
class DomainSpec:
    """Physical domain specification (bounds, resolution, parameters)."""

    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    parameters: Dict[str, float] = field(default_factory=dict)
    resolution: Dict[str, int] = field(default_factory=dict)
    time_range: Optional[Tuple[float, float]] = None

    def get_range(self, axis: str, default: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
        return self.bounds.get(axis, default)

    def to_config(self) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        for axis, bounds in self.bounds.items():
            config[f"{axis}_range"] = list(bounds)
        if self.parameters:
            config.update(self.parameters)
        if self.resolution:
            config["resolution"] = dict(self.resolution)
        if self.time_range is not None:
            config["time_range"] = list(self.time_range)
        return config


@dataclass
class PointSamples:
    """Sparse samples represented as coordinate/value pairs."""

    coordinates: np.ndarray  # shape (N, D)
    values: Dict[str, np.ndarray]
    axes: Tuple[str, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        coords = np.asarray(self.coordinates, dtype=np.float64)
        if coords.ndim != 2:
            raise ValueError(f"coordinates must be 2D, got shape {coords.shape}")
        self.coordinates = coords
        cleaned_values: Dict[str, np.ndarray] = {}
        for name, arr in self.values.items():
            arr_np = np.asarray(arr).reshape(-1)
            if arr_np.shape[0] != coords.shape[0]:
                raise ValueError(
                    f"Field '{name}' length {arr_np.shape[0]} does not match coordinates {coords.shape[0]}"
                )
            cleaned_values[name] = arr_np.astype(np.float64)
        self.values = cleaned_values

    @property
    def num_points(self) -> int:
        return self.coordinates.shape[0]

    def to_array(self,
                 order: Sequence[str] = ("x", "y", "z"),
                 fill_value: float = 0.0) -> np.ndarray:
        coords = []
        for axis in order:
            if axis in self.axes:
                idx = self.axes.index(axis)
                coords.append(self.coordinates[:, idx:idx + 1])
            else:
                coords.append(np.full((self.num_points, 1), fill_value, dtype=np.float64))
        return np.concatenate(coords, axis=1)

    def get_field(self, name: str) -> np.ndarray:
        return self.values[name]

    def available_fields(self) -> Tuple[str, ...]:
        return tuple(self.values.keys())

    def subset_fields(self, fields: Iterable[str]) -> Dict[str, np.ndarray]:
        subset = {}
        for name in fields:
            if name in self.values:
                subset[name] = self.values[name]
        return subset


@dataclass
class FlowDataBundle:
    """
    Unified container for training-ready flow data.

    Holds sparse sensor samples, optional low-fidelity priors, and domain metadata.
    """

    samples: PointSamples
    domain: DomainSpec
    statistics: Dict[str, Any] = field(default_factory=dict)
    lowfi_prior: Optional[PointSamples] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def as_training_dict(self,
                         target_fields: Sequence[str],
                         device,
                         include_w: bool = False) -> Dict[str, Any]:
        """Convert the bundle into tensorised training inputs."""
        import torch  # lazy import to avoid torch dependency at module import time

        coords_np = self.samples.to_array(order=("x", "y", "z"))
        coords = torch.from_numpy(coords_np).float().to(device)

        sensor_data: Dict[str, torch.Tensor] = {}
        for name in target_fields:
            if name in self.samples.values:
                values_np = self.samples.values[name].reshape(-1, 1)
            elif name == 'w' and include_w:
                values_np = np.zeros((self.samples.num_points, 1), dtype=np.float64)
            else:
                raise KeyError(f"Target field '{name}' not available in sensor samples")
            sensor_data[name] = torch.from_numpy(values_np).float().to(device)

        lowfi_data_tensors: Dict[str, torch.Tensor] = {}
        has_prior = False
        if self.lowfi_prior is not None:
            for name, arr in self.lowfi_prior.values.items():
                values_np = arr.reshape(-1, 1)
                lowfi_data_tensors[name] = torch.from_numpy(values_np).float().to(device)
            has_prior = bool(lowfi_data_tensors)

        training_dict: Dict[str, Any] = {
            'coordinates': coords,
            'sensor_data': sensor_data,
            'domain_bounds': dict(self.domain.bounds),
            'physical_params': dict(self.domain.parameters),
            'statistics': dict(self.statistics),
            'metadata': dict(self.metadata),
            'has_prior': has_prior
        }

        if lowfi_data_tensors:
            training_dict['lowfi_prior'] = lowfi_data_tensors

        return training_dict
