"""Factory for sensor selectors."""

from __future__ import annotations

from .base import BaseSensorSelector
from .selectors.greedy import GreedySelector
from .selectors.multi_objective import MultiObjectiveSelector
from .selectors.pod_based import PODBasedSelector
from .selectors.qr_pivot import QRPivotSelector


def create_sensor_selector(strategy: str = "qr_pivot", **kwargs) -> BaseSensorSelector:
    """Create a selector instance by strategy name."""
    if strategy == "qr_pivot":
        return QRPivotSelector(**kwargs)
    if strategy == "pod_based":
        return PODBasedSelector(**kwargs)
    if strategy == "greedy":
        return GreedySelector(**kwargs)
    if strategy == "multi_objective":
        return MultiObjectiveSelector(**kwargs)

    raise ValueError(f"未知的感測點選擇策略: {strategy}")
