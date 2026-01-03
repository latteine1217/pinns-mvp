"""Unified normalizer (input + output)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch

from .config import InputNormConfig, OutputNormConfig
from .input_transform import InputTransform
from .output_transform import OutputTransform

logger = logging.getLogger(__name__)


class UnifiedNormalizer:
    """
    Unified normalizer for input coordinates and output variables.
    """

    def __init__(self, input_transform: InputTransform, output_transform: OutputTransform):
        self.input_transform = input_transform
        self.output_transform = output_transform

        logger.info("✅ UnifiedNormalizer 初始化完成")
        logger.info(f"   輸入: {self.input_transform.norm_type}")
        logger.info(f"   輸出: {self.output_transform.norm_type}")
        logger.info(f"   變量順序: {self.output_transform.variable_order}")

    @classmethod
    def from_config(
        cls,
        config: Dict,
        training_data: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = torch.device("cpu"),
    ) -> "UnifiedNormalizer":
        """Create from config."""
        scaling_cfg = config.get("model", {}).get("scaling", {})
        input_norm_type = scaling_cfg.get("input_norm", "none")
        feature_range = tuple(scaling_cfg.get("input_norm_range", [-1.0, 1.0]))

        bounds_tensor = None
        if input_norm_type == "channel_flow":
            domain = config.get("physics", {}).get("domain", {})
            bounds: List[Tuple[float, float]] = []
            for axis in ["x", "y", "z"]:
                rng = domain.get(f"{axis}_range")
                if rng is not None:
                    bounds.append((float(rng[0]), float(rng[1])))
            if bounds:
                bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)

        input_config = InputNormConfig(
            norm_type=input_norm_type,
            feature_range=(float(feature_range[0]), float(feature_range[1])),
            bounds=bounds_tensor,
        )
        input_transform = InputTransform(input_config)

        if training_data is not None:
            coord_tensors = cls._collect_coordinate_tensors(training_data)
            if coord_tensors:
                samples = torch.cat(coord_tensors, dim=0)
                if (
                    input_transform.bounds is not None
                    and input_transform.bounds.shape[0] > samples.shape[1]
                ):
                    input_transform.bounds = input_transform.bounds[: samples.shape[1], :]
                input_transform.fit(samples)

        input_transform.to(device)

        if "normalization" not in config:
            logger.warning("⚠️  配置中未找到 'normalization' 段落，使用默認 (type='none')")
            output_config = OutputNormConfig(norm_type="none")
            output_transform = OutputTransform(output_config)
        else:
            norm_cfg = config["normalization"]
            norm_type = norm_cfg.get("type", "none")
            params = norm_cfg.get("params", {})

            variable_order = norm_cfg.get("variable_order") or norm_cfg.get("variables")
            if variable_order is None and training_data is not None:
                data_vars = []
                for k in training_data.keys():
                    if k in OutputTransform.DEFAULT_VAR_ORDER:
                        val = training_data[k]
                        if isinstance(val, torch.Tensor) and val.numel() == 0:
                            continue
                        if isinstance(val, np.ndarray) and val.size == 0:
                            continue
                        data_vars.append(k)

                if data_vars:
                    variable_order = sorted(
                        data_vars, key=lambda x: OutputTransform.DEFAULT_VAR_ORDER.index(x)
                    )
                    logger.info(f"📋 從資料推斷變量順序（已過濾空張量）: {variable_order}")

            if variable_order:
                expected_order = ["u", "v", "w", "p"]
                expected_filtered = [v for v in expected_order if v in variable_order]
                if variable_order != expected_filtered:
                    logger.warning(
                        "⚠️  檢測到 variable_order 可能不一致：\n"
                        f"    實際順序: {variable_order}\n"
                        f"    預期順序: {expected_filtered}\n"
                        "    這可能導致反標準化錯誤！"
                    )

            if norm_type == "training_data_norm":
                means, stds = OutputTransform._extract_training_data_scales(
                    params, training_data, config
                )
            elif norm_type == "friction_velocity":
                means, stds = OutputTransform._extract_friction_velocity_scales(params, config)
            elif norm_type == "manual":
                means = {k.replace("_mean", ""): v for k, v in params.items() if k.endswith("_mean")}
                stds = {k.replace("_std", ""): v for k, v in params.items() if k.endswith("_std")}
            else:
                means = {}
                stds = {}

            output_config = OutputNormConfig(
                norm_type=norm_type,
                variable_order=variable_order,
                means=means,
                stds=stds,
                params=params,
            )
            output_transform = OutputTransform(output_config)

        return cls(input_transform, output_transform)

    @classmethod
    def from_metadata(cls, metadata: Dict) -> "UnifiedNormalizer":
        """Restore from checkpoint metadata."""
        input_meta = metadata.get("input", {})
        output_meta = metadata.get("output", {})

        input_config = InputNormConfig(
            norm_type=input_meta.get("norm_type", "none"),
            feature_range=tuple(input_meta.get("feature_range", (-1.0, 1.0))),
            bounds=input_meta.get("bounds"),
        )
        input_transform = InputTransform(input_config)

        if "mean" in input_meta:
            input_transform.mean = input_meta["mean"]
        if "std" in input_meta:
            input_transform.std = input_meta["std"]
        if "data_min" in input_meta:
            input_transform.data_min = input_meta["data_min"]
        if "data_range" in input_meta:
            input_transform.data_range = input_meta["data_range"]

        output_config = OutputNormConfig(
            norm_type=output_meta.get("norm_type", "none"),
            variable_order=output_meta.get(
                "variable_order", OutputTransform.DEFAULT_VAR_ORDER.copy()
            ),
            means=output_meta.get("means", {}),
            stds=output_meta.get("stds", {}),
            params=output_meta.get("params", {}),
        )
        output_transform = OutputTransform(output_config)

        logger.info("🔄 從 checkpoint 恢復 UnifiedNormalizer")
        return cls(input_transform, output_transform)

    def transform_input(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize input coordinates."""
        return self.input_transform.transform(coords)

    def inverse_transform_input(self, coords: torch.Tensor) -> torch.Tensor:
        """Denormalize input coordinates."""
        return self.input_transform.inverse_transform(coords)

    def transform_output(
        self, predictions: Union[np.ndarray, torch.Tensor], var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Normalize output batch."""
        return self.output_transform.normalize_batch(predictions, var_order)

    def inverse_transform_output(
        self, predictions: Union[np.ndarray, torch.Tensor], var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize output batch."""
        return self.output_transform.denormalize_batch(predictions, var_order)

    def get_metadata(self) -> Dict[str, Any]:
        """Return checkpoint metadata."""
        return {
            "input": self.input_transform.get_metadata(),
            "output": self.output_transform.get_metadata(),
        }

    @property
    def variable_order(self) -> List[str]:
        """Return variable order."""
        return self.output_transform.variable_order

    def has_valid_stats(self) -> bool:
        """Check stats via OutputTransform."""
        return self.output_transform.has_valid_stats()

    def to(self, device: torch.device) -> "UnifiedNormalizer":
        """Move stats to device."""
        self.input_transform.to(device)
        return self

    @staticmethod
    def _collect_coordinate_tensors(training_data: Dict) -> List[torch.Tensor]:
        """Collect coordinate tensors from training data."""
        coord_tensors = []
        for key in ["coords", "boundary_coords", "pde_coords", "sensor_coords"]:
            if key in training_data:
                val = training_data[key]
                if isinstance(val, torch.Tensor):
                    coord_tensors.append(val)
                elif isinstance(val, np.ndarray):
                    coord_tensors.append(torch.from_numpy(val).float())
        return coord_tensors
