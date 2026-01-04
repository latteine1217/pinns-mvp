"""
訓練調度器模組

提供多種訓練調度策略：
- WeightScheduler: 權重調度器抽象基類
- WarmupCosineScheduler: 學習率預熱 + 餘弦退火
- StagedWeightScheduler: 階段式損失權重調度
- CurriculumScheduler: 課程訓練（逐步提升難度）
"""

from .base import WeightScheduler
from .warmup_cosine import WarmupCosineScheduler
from .staged_weights import StagedWeightScheduler
from .curriculum import CurriculumScheduler

__all__ = [
    'WeightScheduler',
    'WarmupCosineScheduler',
    'StagedWeightScheduler',
    'CurriculumScheduler',
]
