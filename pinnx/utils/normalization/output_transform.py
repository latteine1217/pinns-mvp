"""Output normalization transform."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import numpy as np
import torch

from .base_normalizer import BaseNormalizer
from .config import OutputNormConfig

logger = logging.getLogger(__name__)


def compute_friction_velocity_scales(
    u_tau: float, rho: float = 1.0, variable_order: Optional[List[str]] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """計算基於摩擦速度的標準化係數

    用於替代 friction_velocity 模式。計算公式：
    - 速度尺度: u_τ
    - 壓力尺度: ρ · u_τ²

    Args:
        u_tau: 摩擦速度 (friction velocity)
        rho: 流體密度（默認 1.0）
        variable_order: 變量順序（默認 ['u', 'v', 'w', 'p']）

    Returns:
        (means, stds) 元組，可直接用於 manual 模式

    Example:
        >>> means, stds = compute_friction_velocity_scales(u_tau=0.05, rho=1.0)
        >>> config = OutputNormConfig(
        ...     norm_type='manual',
        ...     variable_order=['u', 'v', 'w', 'p'],
        ...     means=means,
        ...     stds=stds
        ... )
        >>> transform = OutputTransform(config)
    """
    if variable_order is None:
        variable_order = ["u", "v", "w", "p"]

    velocity_scale = u_tau
    pressure_scale = rho * u_tau**2

    means = {}
    stds = {}

    for var in variable_order:
        means[var] = 0.0
        if var in ("u", "v", "w"):
            stds[var] = velocity_scale
        elif var == "p":
            stds[var] = pressure_scale
        else:
            # 未知變量，設為 1.0 避免除零
            stds[var] = 1.0
            logger.warning(f"⚠️ 變量 '{var}' 無摩擦速度尺度定義，使用 std=1.0")

    logger.info(f"📐 計算摩擦速度尺度: u_τ={u_tau}, ρ={rho}")
    logger.info(f"   速度尺度={velocity_scale}, 壓力尺度={pressure_scale}")

    return means, stds


class OutputTransform(BaseNormalizer):
    """
    Variable normalization transform.

    Supported:
    - none
    - training_data_norm
    - manual
    - dns_ground_truth_norm

    Note:
        friction_velocity 模式已移除，請使用 compute_friction_velocity_scales() 輔助函數
        配合 manual 模式替代。
    """

    SUPPORTED_TYPES = [
        "none",
        "training_data_norm",
        "manual",
        "dns_ground_truth_norm",
    ]
    DEFAULT_VAR_ORDER = ["u", "v", "w", "p", "S"]

    def __init__(self, config: OutputNormConfig):
        if config.norm_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支援的標準化類型: {config.norm_type}。支援: {self.SUPPORTED_TYPES}"
            )

        super().__init__(norm_type=config.norm_type)
        self.variable_order = config.variable_order or self.DEFAULT_VAR_ORDER.copy()
        self.means = config.means or {}
        self.stds = config.stds or {}
        self.params = config.params or {}

        logger.info(
            f"✅ OutputTransform 初始化: type={self.norm_type}, variables={self.variable_order}"
        )

    @classmethod
    def from_data(
        cls,
        data: Dict[str, Union[np.ndarray, torch.Tensor]],
        norm_type: str = "training_data_norm",
        variable_order: Optional[List[str]] = None,
    ) -> "OutputTransform":
        """Compute stats from data (training_data_norm only)."""
        if norm_type != "training_data_norm":
            raise ValueError(f"from_data 僅支援 training_data_norm，當前為: {norm_type}")

        means = {}
        stds = {}
        valid_vars = []

        for var_name, values in data.items():
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()

            if values.size == 0:
                logger.info(f"⏭️  {var_name} 為空張量，跳過標準化統計量計算")
                continue

            mean = float(np.mean(values))
            std = float(np.std(values))

            if not np.isfinite(mean) or not np.isfinite(std):
                logger.warning(
                    f"⚠️  {var_name} 的統計量包含 NaN/Inf (mean={mean}, std={std})，跳過"
                )
                continue

            if abs(std) < 1e-10:
                logger.warning(f"⚠️  {var_name} 的標準差接近零，設為 1.0")
                std = 1.0

            means[var_name] = mean
            stds[var_name] = std
            valid_vars.append(var_name)

        if variable_order is None:
            variable_order = valid_vars

        params = {"source": "auto_computed_from_data"}

        config = OutputNormConfig(
            norm_type=norm_type,
            variable_order=variable_order,
            means=means,
            stds=stds,
            params=params,
        )

        logger.info(f"✅ 從資料計算 Z-score 係數: means={means}, stds={stds}")
        return cls(config)

    @classmethod
    def from_metadata(cls, metadata: Dict) -> "OutputTransform":
        """Restore OutputTransform from checkpoint metadata."""
        required_fields = ["norm_type", "variable_order", "means", "stds"]
        missing_fields = [f for f in required_fields if f not in metadata]
        if missing_fields:
            raise KeyError(
                f"metadata 缺少必要欄位: {missing_fields}. "
                f"需要: {required_fields}, 得到: {list(metadata.keys())}"
            )

        config = OutputNormConfig(
            norm_type=metadata["norm_type"],
            variable_order=list(metadata["variable_order"]),
            means=metadata["means"],
            stds=metadata["stds"],
            params=metadata.get("params", {}),
        )

        logger.info(
            f"✅ 從 metadata 重建 OutputTransform: "
            f"type={config.norm_type}, variables={config.variable_order}"
        )
        return cls(config)

    @classmethod
    def from_config(
        cls,
        config: Dict,
        training_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> "OutputTransform":
        """Create OutputTransform from config."""
        norm_config = config.get("normalization", {})
        norm_type = norm_config.get("type", "none")
        params = norm_config.get("params", {})
        variable_order = norm_config.get("variable_order", cls.DEFAULT_VAR_ORDER.copy())

        if norm_type == "training_data_norm":
            means, stds = cls._extract_training_data_scales(params, training_data, config)
            variable_order = list(means.keys())
        elif norm_type == "dns_ground_truth_norm":
            means, stds = cls._extract_dns_ground_truth_scales(params, config)
            variable_order = list(means.keys())
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

        return cls(output_config)

    @staticmethod
    def _extract_training_data_scales(
        params: Dict,
        training_data: Optional[Dict[str, torch.Tensor]],
        config: Dict,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract training-data scales."""
        norm_config = config.get("normalization", {})
        variable_order = norm_config.get("variable_order", ["u", "v", "w", "p"])

        required_keys = [f"{v}_{s}" for v in variable_order for s in ["mean", "std"]]
        if all(k in params for k in required_keys):
            means = {v: params[f"{v}_mean"] for v in variable_order}
            stds = {v: params[f"{v}_std"] for v in variable_order}
            logger.info(f"📐 使用配置中的標準化係數: {list(means.keys())}")
            return means, stds

        if training_data is not None:
            available_vars = []
            for var_name in variable_order:
                key = var_name if var_name in training_data else f"{var_name}_sensors"
                if key in training_data:
                    available_vars.append(var_name)

            missing_vars = set(variable_order) - set(available_vars)
            if missing_vars:
                logger.warning(
                    f"⚠️  variable_order 包含不存在的變量: {missing_vars}，"
                    f"將只使用可用變量: {available_vars}"
                )

            means = {}
            stds = {}
            for var_name in available_vars:
                key = var_name if var_name in training_data else f"{var_name}_sensors"
                values = training_data[key]
                if isinstance(values, torch.Tensor):
                    values = values.detach().cpu().numpy()

                if values.size == 0:
                    logger.info(f"⏭️  {var_name} 為空張量，跳過標準化統計量計算")
                    continue

                mean = float(np.mean(values))
                std = float(np.std(values))

                if not np.isfinite(mean) or not np.isfinite(std):
                    logger.warning(
                        f"⚠️  {var_name} 的統計量包含 NaN/Inf (mean={mean}, std={std})，跳過"
                    )
                    continue

                if abs(std) < 1e-10:
                    logger.warning(f"⚠️  {var_name} 的標準差接近零，設為 1.0")
                    std = 1.0

                means[var_name] = mean
                stds[var_name] = std

            if means:
                logger.info(f"📐 從訓練資料計算標準化係數: {list(means.keys())}")
                return means, stds

        raise ValueError(
            "training_data_norm 模式需要提供標準化係數！\n"
            "請在配置中提供 normalization.params (u_mean, u_std, ...)\n"
            "或傳入 training_data 以自動計算。"
        )

    @staticmethod
    def _extract_dns_ground_truth_scales(
        params: Dict,
        config: Dict,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Extract DNS full-field scales."""
        import h5py
        from pathlib import Path

        dns_file = params.get("dns_file")
        if dns_file is None:
            raise ValueError(
                "dns_ground_truth_norm 需要提供 dns_file 參數！\n"
                "範例：\n"
                "  normalization:\n"
                "    type: dns_ground_truth_norm\n"
                "    params:\n"
                "      dns_file: ./data/kolmogorov_dns/dns_re50_t100.h5\n"
                "      time_range: [15.0, 35.0]  # 可選\n"
            )

        dns_path = Path(dns_file)
        if not dns_path.exists():
            raise FileNotFoundError(
                f"DNS 文件不存在: {dns_file}\n"
                f"請確認路徑正確，或使用相對於專案根目錄的路徑。"
            )

        time_range = params.get("time_range")
        variable_order = params.get("variable_order")
        if variable_order is None:
            norm_config = config.get("normalization", {})
            variable_order = norm_config.get("variable_order", ["u", "v", "p"])

        logger.info(f"📂 載入 DNS 完整場數據: {dns_file}")
        logger.info(f"   變量順序: {variable_order}")
        if time_range:
            logger.info(f"   時間範圍: [{time_range[0]:.2f}, {time_range[1]:.2f}]")

        means = {}
        stds = {}

        with h5py.File(dns_path, "r") as f:
            if "time" in f:
                time = f["time"][:]
                if time_range is not None:
                    t_start, t_end = time_range
                    time_mask = (time >= t_start) & (time <= t_end)
                    time_indices = np.where(time_mask)[0]
                    logger.info(
                        f"   選定時間步: {len(time_indices)}/{len(time)} "
                        f"({len(time_indices)/len(time)*100:.1f}%)"
                    )
                else:
                    time_indices = np.arange(len(time))
                    logger.info(f"   使用全部時間步: {len(time_indices)}")
            else:
                time_indices = None
                logger.warning("⚠️  DNS 文件無 'time' 鍵，假設為單一快照")

            for var_name in variable_order:
                if var_name not in f:
                    logger.warning(f"⚠️  DNS 文件缺少變量 '{var_name}'，跳過")
                    continue

                data = f[var_name]

                if time_indices is not None:
                    values = data[time_indices, :, :]
                else:
                    values = data[:]

                mean = float(np.mean(values))
                std = float(np.std(values))

                if not np.isfinite(mean) or not np.isfinite(std):
                    logger.error(
                        f"❌ {var_name} 統計量包含 NaN/Inf！\n"
                        f"   mean={mean}, std={std}\n"
                        f"   這可能表示 DNS 數據損壞或包含異常值。"
                    )
                    raise ValueError(f"{var_name} 統計量異常: mean={mean}, std={std}")

                if abs(std) < 1e-10:
                    logger.warning(
                        f"⚠️  {var_name} 標準差接近零 (std={std:.2e})，設為 1.0\n"
                        f"   這可能表示場是常數或數值精度不足。"
                    )
                    std = 1.0

                means[var_name] = mean
                stds[var_name] = std

                logger.info(
                    f"   {var_name}: mean={mean:+.6f}, std={std:.6f}, "
                    f"range=[{values.min():.6f}, {values.max():.6f}]"
                )

        if not means:
            raise ValueError(
                f"無法從 DNS 文件計算統計量！\n"
                f"DNS 文件: {dns_file}\n"
                f"要求變量: {variable_order}\n"
                f"可用變量: {list(f.keys())}\n"
            )

        logger.info(f"✅ 從 DNS 完整場計算標準化係數: {list(means.keys())}")
        return means, stds

    def normalize(self, value: Union[np.ndarray, torch.Tensor, float], var_name: str):
        """Normalize a single variable."""
        if self.norm_type == "none":
            return value

        if var_name not in self.stds:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過標準化")
            return value

        mean = self.means.get(var_name, 0.0)
        std = self.stds[var_name]

        return (value - mean) / std

    def denormalize(self, value: Union[np.ndarray, torch.Tensor, float], var_name: str):
        """Denormalize a single variable."""
        if self.norm_type == "none":
            return value

        if var_name not in self.stds:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過反標準化")
            return value

        mean = self.means.get(var_name, 0.0)
        std = self.stds[var_name]

        return value * std + mean

    def normalize_batch(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Batch normalize."""
        if self.norm_type == "none":
            return predictions

        if var_order is None:
            var_order = self.variable_order

        is_torch = isinstance(predictions, torch.Tensor)

        if is_torch:
            if predictions.requires_grad:
                result = []
                for i, var_name in enumerate(var_order):
                    if i >= predictions.shape[-1]:
                        break
                    col = predictions[:, i : i + 1]
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        col = (col - mean) / std
                    result.append(col)
                return torch.cat(result, dim=1)

            result = predictions.clone()
            for i, var_name in enumerate(var_order):
                if i >= result.shape[-1]:
                    break
                if var_name in self.stds:
                    mean = self.means.get(var_name, 0.0)
                    std = self.stds[var_name]
                    result[:, i] = (result[:, i] - mean) / std
            return result

        result = predictions.copy()
        for i, var_name in enumerate(var_order):
            if i >= result.shape[-1]:
                break
            if var_name in self.stds:
                mean = self.means.get(var_name, 0.0)
                std = self.stds[var_name]
                result[:, i] = (result[:, i] - mean) / std
        return result

    def denormalize_batch(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Batch denormalize."""
        if self.norm_type == "none":
            return predictions

        if var_order is None:
            var_order = self.variable_order

        is_torch = isinstance(predictions, torch.Tensor)

        if is_torch:
            if predictions.requires_grad:
                result = []
                for i, var_name in enumerate(var_order):
                    if i >= predictions.shape[-1]:
                        break
                    col = predictions[:, i : i + 1]
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        col = col * std + mean
                    result.append(col)
                return torch.cat(result, dim=1)

            result = predictions.clone()
            for i, var_name in enumerate(var_order):
                if i >= result.shape[-1]:
                    break
                if var_name in self.stds:
                    mean = self.means.get(var_name, 0.0)
                    std = self.stds[var_name]
                    result[:, i] = result[:, i] * std + mean
            return result

        result = predictions.copy()
        for i, var_name in enumerate(var_order):
            if i >= result.shape[-1]:
                break
            if var_name in self.stds:
                mean = self.means.get(var_name, 0.0)
                std = self.stds[var_name]
                result[:, i] = result[:, i] * std + mean
        return result

    def has_valid_stats(self) -> bool:
        """Check if stats are valid."""
        if self.norm_type == "none":
            return True

        for var_name in self.variable_order:
            if var_name not in self.means:
                logger.error(f"❌ 變量 '{var_name}' 缺少均值 (mean)")
                return False

            mean_val = self.means[var_name]
            if not np.isfinite(mean_val):
                logger.error(f"❌ 變量 '{var_name}' 的均值無效: {mean_val} (NaN/Inf)")
                return False

            if var_name not in self.stds:
                logger.error(f"❌ 變量 '{var_name}' 缺少標準差 (std)")
                return False

            std_val = self.stds[var_name]

            if not np.isfinite(std_val):
                logger.error(f"❌ 變量 '{var_name}' 的標準差無效: {std_val} (NaN/Inf)")
                return False

            if abs(std_val) < 1e-12:
                logger.error(
                    f"❌ 變量 '{var_name}' 的標準差過小: {std_val:.2e}\n"
                    f"   這會導致除零或數值不穩定。\n"
                    f"   可能原因:\n"
                    f"   1. 訓練資料中該變量為常數\n"
                    f"   2. 資料範圍過小或單位不當\n"
                    f"   建議: 檢查訓練資料或使用不同的標準化方法"
                )
                return False

        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata for checkpoints."""
        return {
            "norm_type": self.norm_type,
            "variable_order": self.variable_order.copy(),
            "means": self.means.copy(),
            "stds": self.stds.copy(),
            "params": self.params.copy(),
        }

    def transform(self, predictions: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """標準化數據（BaseNormalizer 接口實現）

        Args:
            predictions: 原始預測值

        Returns:
            標準化後的預測值
        """
        return self.normalize_batch(predictions, var_order=None)

    def inverse_transform(
        self, predictions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """反標準化數據（BaseNormalizer 接口實現）

        Args:
            predictions: 標準化後的預測值

        Returns:
            物理空間的預測值
        """
        return self.denormalize_batch(predictions, var_order=None)

    def to(self, device: torch.device) -> "OutputTransform":
        """移動到指定設備（BaseNormalizer 接口實現）

        Note:
            OutputTransform 目前不持有 tensor 參數，因此此方法為空操作
            保留此方法以符合 BaseNormalizer 接口

        Args:
            device: 目標設備

        Returns:
            self
        """
        # OutputTransform 使用 dict[str, float] 存儲統計量，無需移動設備
        return self
