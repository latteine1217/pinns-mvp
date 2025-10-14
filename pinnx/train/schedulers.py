"""
學習率調度器模組
提供自定義的學習率調度策略
"""

import torch
import torch.optim as optim
import logging
from typing import Optional


class WarmupCosineScheduler:
    """
    Warmup + CosineAnnealing 學習率調度器
    
    前 warmup_epochs 個 epoch 線性增加學習率從 0 到 base_lr
    之後使用 CosineAnnealing 衰減到 min_lr
    """
    
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, max_epochs: int, 
                 base_lr: float, min_lr: float = 0.0):
        """
        Args:
            optimizer: PyTorch 優化器
            warmup_epochs: Warmup 階段的 epoch 數量
            max_epochs: 總訓練 epoch 數量
            base_lr: 基礎學習率（Warmup 後的峰值）
            min_lr: 最小學習率（CosineAnnealing 的下限）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 建立內部的 CosineAnnealing 調度器（用於 Warmup 後階段）
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        logging.info(f"✅ WarmupCosineScheduler initialized:")
        logging.info(f"   Warmup epochs: {warmup_epochs}")
        logging.info(f"   Max epochs: {max_epochs}")
        logging.info(f"   Base LR: {base_lr:.6f}")
        logging.info(f"   Min LR: {min_lr:.6f}")
    
    def step(self):
        """更新學習率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 階段：線性增加
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # CosineAnnealing 階段
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """返回當前學習率（兼容性接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
