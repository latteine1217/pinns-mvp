"""
權重調度器基礎接口

定義所有權重調度器的統一接口，確保 Trainer 和 LossManager 可以
使用多態方式調用不同的調度策略。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class WeightScheduler(ABC):
    """
    權重調度器抽象基類
    
    所有權重調度器（Curriculum, Staged, Adaptive）都應該實作此接口。
    
    核心職責：
    1. 根據當前 epoch 返回損失權重字典
    2. 提供階段切換的元數據（如需要）
    
    設計原則：
    - Simplicity：僅兩個核心方法
    - Good Taste：使用多態，避免 Trainer 中的條件分支
    - Never Break Userspace：保持向後兼容（子類可保留舊方法）
    """
    
    @abstractmethod
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        獲取當前 epoch 的損失權重
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            權重字典，例如：
            {
                'data': 100.0,
                'momentum_x': 1.0,
                'momentum_y': 1.0,
                'continuity': 1.0,
                'boundary': 10.0,
                'prior': 1.0
            }
        """
        pass
    
    @abstractmethod
    def get_metadata(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        獲取當前 epoch 的調度元數據（可選）
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            元數據字典，例如：
            {
                'stage_name': 'warmup',
                'is_transition': True,
                'lr': 1e-3,  # 如果 curriculum 控制 LR
                'sampling': {...},  # 如果 curriculum 控制採樣
                'Re_tau': 180.0,  # 如果 curriculum 控制物理參數
            }
            
            如果不需要元數據，返回 None
        """
        pass
