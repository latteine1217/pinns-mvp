"""QR-pivot sensor selection package."""

from .base import BaseSensorSelector
from .boundary import PeriodicBoundaryHandler
from .evaluation import evaluate_sensor_placement
from .factory import create_sensor_selector
from .features import (
    apply_min_distance_constraint,
    build_circular_snapshot_matrix,
    prepare_turbulence_features,
)
from .optimizer import SensorOptimizer
from .selectors.greedy import GreedySelector
from .selectors.multi_objective import MultiObjectiveSelector
from .selectors.physics_guided import PhysicsGuidedQRPivotSelector
from .selectors.pod_based import PODBasedSelector
from .selectors.pod_qr_eim import PODQREIMSelector
from .selectors.qr_pivot import QRPivotSelector

__all__ = [
    "BaseSensorSelector",
    "PeriodicBoundaryHandler",
    "apply_min_distance_constraint",
    "build_circular_snapshot_matrix",
    "prepare_turbulence_features",
    "QRPivotSelector",
    "PODQREIMSelector",
    "PODBasedSelector",
    "GreedySelector",
    "MultiObjectiveSelector",
    "PhysicsGuidedQRPivotSelector",
    "SensorOptimizer",
    "evaluate_sensor_placement",
    "create_sensor_selector",
]
