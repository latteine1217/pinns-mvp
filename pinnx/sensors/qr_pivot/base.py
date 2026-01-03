"""Base selector interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Tuple

import numpy as np


class BaseSensorSelector(ABC):
    """Base class for sensor selectors."""

    @abstractmethod
    def select_sensors(
        self, data_matrix: np.ndarray, n_sensors: int, **kwargs
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """Select sensors and return metrics."""
        raise NotImplementedError
