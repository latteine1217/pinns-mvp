"""
學習率調度器模組
提供自定義的學習率調度策略
"""

import torch
import torch.optim as optim
import logging
from typing import Optional


class StepsBasedWarmupScheduler:
    """
    Steps-based Warmup + Exponential Decay 調度器（用於 PirateNet）
    
    - Warmup 階段: 線性增加學習率從 0 → base_lr (over warmup_steps)
    - Decay 階段: 每 decay_steps 衰減 γ 倍
    
    Example:
        warmup_steps=2000, decay_steps=2000, gamma=0.9
        - Step 0-2000: 線性 warmup
        - Step 2000-4000: lr = base_lr
        - Step 4000-6000: lr = base_lr * 0.9
        - Step 6000-8000: lr = base_lr * 0.9^2
    """
    
    def __init__(
        self, 
        optimizer: optim.Optimizer, 
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        decay_steps: int = 2000,
        gamma: float = 0.9,
        min_lr: float = 1e-6
    ):
        """
        Args:
            optimizer: PyTorch 優化器
            warmup_steps: Warmup 階段的步數（論文預設: 2000）
            total_steps: 總訓練步數
            base_lr: 基礎學習率（Warmup 後的峰值）
            decay_steps: 每多少步衰減一次（論文預設: 2000）
            gamma: 衰減率（論文預設: 0.9）
            min_lr: 最小學習率下限
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.decay_steps = decay_steps
        self.gamma = gamma
        self.min_lr = min_lr
        self.current_step = 0
        
        logging.info(f"✅ StepsBasedWarmupScheduler initialized:")
        logging.info(f"   Warmup steps: {warmup_steps}")
        logging.info(f"   Total steps: {total_steps}")
        logging.info(f"   Base LR: {base_lr:.6f}")
        logging.info(f"   Decay steps: {decay_steps}")
        logging.info(f"   Gamma: {gamma}")
    
    def step(self):
        """每次 optimizer.step() 後調用"""
        if self.current_step < self.warmup_steps:
            # Warmup 階段: 線性增加
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
        else:
            # Decay 階段: 指數衰減
            decay_count = (self.current_step - self.warmup_steps) // self.decay_steps
            lr = self.base_lr * (self.gamma ** decay_count)
        
        # 應用最小學習率限制
        lr = max(lr, self.min_lr)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
    
    def get_last_lr(self):
        """返回當前學習率（兼容性接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存調度器狀態"""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'base_lr': self.base_lr,
            'decay_steps': self.decay_steps,
            'gamma': self.gamma,
            'min_lr': self.min_lr
        }
    
    def load_state_dict(self, state_dict):
        """載入調度器狀態"""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.total_steps = state_dict['total_steps']
        self.base_lr = state_dict['base_lr']
        self.decay_steps = state_dict['decay_steps']
        self.gamma = state_dict['gamma']
        self.min_lr = state_dict['min_lr']


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
    
    def state_dict(self):
        """保存調度器狀態"""
        return {
            'current_epoch': self.current_epoch,
            'warmup_epochs': self.warmup_epochs,
            'max_epochs': self.max_epochs,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'cosine_scheduler_state': self.cosine_scheduler.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """載入調度器狀態"""
        self.current_epoch = state_dict['current_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.max_epochs = state_dict['max_epochs']
        self.base_lr = state_dict['base_lr']
        self.min_lr = state_dict['min_lr']
        self.cosine_scheduler.load_state_dict(state_dict['cosine_scheduler_state'])
        
        # 關鍵修復：恢復後立即同步學習率到優化器
        if self.current_epoch < self.warmup_epochs:
            # Warmup 階段：手動計算並設置學習率
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # CosineAnnealing 階段：從 cosine_scheduler 獲取當前學習率
            # 注意：cosine_scheduler.load_state_dict() 已設置 last_epoch，
            # 但未更新優化器，需手動同步
            current_lr = self.cosine_scheduler.get_last_lr()[0]
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = current_lr
