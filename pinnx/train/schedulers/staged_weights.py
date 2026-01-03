"""
階段式權重調度器

根據訓練 epoch 切換不同階段的損失權重配置
"""

import logging


class StagedWeightScheduler:
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
    
    def get_phase_weights(self, epoch: int) -> tuple:
        """
        獲取當前 epoch 對應的權重
        
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
