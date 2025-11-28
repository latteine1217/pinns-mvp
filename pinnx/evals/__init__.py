"""
評估與分析模組

提供模型評估、指標計算和可視化工具。

主要功能：
- 流場誤差分析（L2, RMSE, 相對誤差）
- 物理守恆檢查（連續性、動量、能量）
- 統計量比較（均值、方差、能譜）
- 可視化工具（場圖、等高線、能譜圖）
"""

from . import metrics
from . import visualizer

__all__ = [
    'metrics',
    'visualizer',
]
