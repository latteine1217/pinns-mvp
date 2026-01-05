"""Factory for sensor selectors."""

from __future__ import annotations

from typing import Callable, Dict, Type

from .base import BaseSensorSelector
from .selectors.greedy import GreedySelector
from .selectors.multi_objective import MultiObjectiveSelector
from .selectors.pod_based import PODBasedSelector
from .selectors.qr_pivot import QRPivotSelector


_SELECTOR_REGISTRY: Dict[str, Type[BaseSensorSelector]] = {}


def register_sensor_selector(name: str) -> Callable[[Type[BaseSensorSelector]], Type[BaseSensorSelector]]:
    """Register a sensor selector by strategy name."""
    def decorator(cls: Type[BaseSensorSelector]) -> Type[BaseSensorSelector]:
        _SELECTOR_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def list_sensor_selectors() -> list[str]:
    """List available selector strategies."""
    return sorted(_SELECTOR_REGISTRY.keys())


register_sensor_selector("qr_pivot")(QRPivotSelector)
register_sensor_selector("pod_based")(PODBasedSelector)
register_sensor_selector("greedy")(GreedySelector)
register_sensor_selector("multi_objective")(MultiObjectiveSelector)


def create_sensor_selector(strategy: str = "qr_pivot", **kwargs) -> BaseSensorSelector:
    """Create a selector instance by strategy name."""
    selector_cls = _SELECTOR_REGISTRY.get(strategy.lower())
    if selector_cls is None:
        raise ValueError(f"未知的感測點選擇策略: {strategy}")
    return selector_cls(**kwargs)
