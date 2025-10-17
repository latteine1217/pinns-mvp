"""
感測點選擇模組

實現最適感測點選擇算法，用於在少量資料約束下優化觀測點配置。
支援 QR-pivot、POD-based、貪心算法等多種策略。

主要功能：
- QR-pivot 感測點選擇
- POD 基礎的感測點配置
- 貪心最適化策略
- 感測點品質評估
- 多目標最適化

核心目標：
- 最小化感測點數量 K
- 最大化資訊內容
- 確保重建精度
- 提升雜訊穩健性
"""

from .qr_pivot import (
    QRPivotSelector,
    PhysicsGuidedQRPivotSelector,
    PODBasedSelector,
    GreedySelector,
    MultiObjectiveSelector,
    SensorOptimizer,
    evaluate_sensor_placement,
    create_sensor_selector
)

from .field_selector import (
    FieldSensorSelector,
    FieldSensorSelection,
    FieldSensorMetadata,
    FeatureScaling
)

from .stratified_sampling import (
    StratifiedChannelFlowSelector,
    HybridChannelFlowSelector,
    compare_sensor_strategies
)

__all__ = [
    # QR-Pivot 系列
    'QRPivotSelector',
    'PhysicsGuidedQRPivotSelector',
    'PODBasedSelector', 
    'GreedySelector',
    'MultiObjectiveSelector',
    'SensorOptimizer',
    'evaluate_sensor_placement',
    'create_sensor_selector',
    'FieldSensorSelector',
    'FieldSensorSelection',
    'FieldSensorMetadata',
    'FeatureScaling',
    
    # 分層採樣系列
    'StratifiedChannelFlowSelector',
    'HybridChannelFlowSelector',
    'compare_sensor_strategies'
]
