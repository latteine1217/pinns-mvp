"""
階段式權重調度器

根據訓練 epoch 切換不同階段的損失權重配置
"""

import logging
from typing import Optional, Dict, Any

from .base import WeightScheduler


class StagedWeightScheduler(WeightScheduler):
    """階段式權重調度器 - 根據 epoch 切換不同訓練階段的權重"""
    
    def __init__(self, phases: list):
        """
        Args:
            phases: 階段配置列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end] epoch 範圍
                - weights: 該階段的權重字典
        """
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase_name = phases[0]['name'] if phases else "default"
        
        # 按 epoch_range 排序
        self.phases.sort(key=lambda p: p['epoch_range'][0])
        
        logging.info(f"✅ StagedWeightScheduler initialized with {len(phases)} phases:")
        for p in self.phases:
            logging.info(f"   {p['name']}: Epoch {p['epoch_range'][0]}-{p['epoch_range'][1]}")
    
    def _find_phase_and_check_transition(self, epoch: int) -> tuple[Dict[str, float], str, bool]:
        """
        內部方法：找到當前階段並檢測是否為切換點
        
        Returns:
            (weights_dict, phase_name, is_transition)
        """
        # 找到當前 epoch 所屬階段
        for idx, phase in enumerate(self.phases):
            start, end = phase['epoch_range']
            if start <= epoch < end:
                # 檢測是否為階段切換點
                is_transition = (idx != self.current_phase_idx)
                self.current_phase_idx = idx
                self.current_phase_name = phase['name']
                
                return phase['weights'], phase['name'], is_transition
        
        # 如果超出所有階段，返回最後階段
        last_phase = self.phases[-1]
        return last_phase['weights'], last_phase['name'], False
    
    # ========================================================================
    # 統一接口實作（WeightScheduler ABC）
    # ========================================================================
    
    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        獲取當前 epoch 的損失權重（統一接口）
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            權重字典，例如：
            {
                'data': 100.0,
                'momentum_x': 1.0,
                'continuity': 1.0,
                'boundary': 10.0
            }
        """
        weights, _, _ = self._find_phase_and_check_transition(epoch)
        return weights
    
    def get_metadata(self, epoch: int) -> Optional[Dict[str, Any]]:
        """
        獲取當前 epoch 的階段元數據（統一接口）
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            元數據字典，包含：
            - phase_name: 階段名稱
            - is_transition: 是否為階段切換點
        """
        _, phase_name, is_transition = self._find_phase_and_check_transition(epoch)
        return {
            'phase_name': phase_name,
            'is_transition': is_transition
        }
