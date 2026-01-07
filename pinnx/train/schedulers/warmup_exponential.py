"""
Warmup + Exponential Decay 學習率調度器（JAXpi 風格）

參考 JAXpi 的 LR scheduling 策略：
- 前期：Linear Warmup (0 → base_lr)
- 後期：Exponential Decay (base_lr × decay_rate^((step - warmup) / decay_steps))

適合長時間訓練，提供穩定且可預測的學習率衰減。
"""

import logging
import math


class WarmupExponentialScheduler:
    """
    Warmup + Exponential Decay 學習率調度器
    
    階段 1: 前 warmup_steps 步線性增加學習率從 0 到 base_lr
    階段 2: 之後使用指數衰減到接近 min_lr
    
    數學公式：
        Warmup:  lr(step) = base_lr × (step / warmup_steps)
        Decay:   lr(step) = base_lr × decay_rate^((step - warmup_steps) / decay_steps)
    
    參考：
        JAXpi Kolmogorov Flow 配置（../jaxpi/examples/kolmogorov_flow/configs/soap.py）
        - base_lr: 1e-3
        - warmup_steps: 2000
        - decay_rate: 0.9
        - decay_steps: 2000
    """
    
    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        base_lr: float,
        decay_rate: float = 0.9,
        decay_steps: int = 2000,
        min_lr: float = 1e-6,
        staircase: bool = False
    ):
        """
        Args:
            optimizer: PyTorch 優化器
            warmup_steps: Warmup 階段的步數（建議：2000，參考 JAXpi）
            total_steps: 總訓練步數
            base_lr: 基礎學習率（Warmup 後的峰值）
            decay_rate: 衰減率（每 decay_steps 步乘以此係數，建議：0.9）
            decay_steps: 衰減週期（步數，建議：2000）
            min_lr: 最小學習率（下界，避免學習率過低）
            staircase: 是否使用階梯式衰減（False=平滑，True=階梯）
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.base_lr = base_lr
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.staircase = staircase
        self.current_step = 0
        
        # 計算預期的最終學習率（用於日誌）
        final_step = total_steps - warmup_steps
        if staircase:
            exponent = math.floor(final_step / decay_steps)
        else:
            exponent = final_step / decay_steps
        expected_final_lr = base_lr * (decay_rate ** exponent)
        expected_final_lr = max(expected_final_lr, min_lr)
        
        logging.info("=" * 80)
        logging.info("✅ WarmupExponentialScheduler initialized (JAXpi-style)")
        logging.info("=" * 80)
        logging.info(f"   Base LR:          {base_lr:.6f}")
        logging.info(f"   Warmup steps:     {warmup_steps} (0.0 → {base_lr:.6f})")
        logging.info(f"   Total steps:      {total_steps}")
        logging.info(f"   Decay rate:       {decay_rate} (per {decay_steps} steps)")
        logging.info(f"   Decay steps:      {decay_steps}")
        logging.info(f"   Min LR:           {min_lr:.6e}")
        logging.info(f"   Staircase:        {staircase}")
        logging.info(f"   Expected final:   {expected_final_lr:.6e} (at step {total_steps})")
        logging.info("=" * 80)
        
        # 計算關鍵時刻的學習率（用於驗證）
        if warmup_steps > 0:
            mid_warmup_lr = base_lr * 0.5
            logging.info(f"   @ step {warmup_steps//2}: {mid_warmup_lr:.6e} (warmup mid)")
        
        logging.info(f"   @ step {warmup_steps}: {base_lr:.6e} (warmup end)")
        
        for milestone in [warmup_steps + decay_steps, warmup_steps + 2*decay_steps]:
            if milestone <= total_steps:
                steps_after_warmup = milestone - warmup_steps
                if staircase:
                    exp = math.floor(steps_after_warmup / decay_steps)
                else:
                    exp = steps_after_warmup / decay_steps
                lr_at_milestone = max(base_lr * (decay_rate ** exp), min_lr)
                logging.info(f"   @ step {milestone}: {lr_at_milestone:.6e}")
        
        logging.info("=" * 80)
    
    def step(self):
        """更新學習率（每步調用一次）"""
        if self.current_step < self.warmup_steps:
            # 階段 1: Warmup（線性增加）
            lr = self.base_lr * (self.current_step + 1) / self.warmup_steps
        else:
            # 階段 2: Exponential Decay
            steps_after_warmup = self.current_step - self.warmup_steps
            
            if self.staircase:
                # 階梯式衰減（每 decay_steps 步跳躍一次）
                exponent = math.floor(steps_after_warmup / self.decay_steps)
            else:
                # 平滑衰減（連續函數）
                exponent = steps_after_warmup / self.decay_steps
            
            lr = self.base_lr * (self.decay_rate ** exponent)
            
            # 應用最小學習率限制
            lr = max(lr, self.min_lr)
        
        # 更新所有參數組的學習率
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_step += 1
    
    def get_last_lr(self):
        """返回當前學習率（兼容性接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存狀態（用於 checkpoint）"""
        return {
            'current_step': self.current_step,
            'warmup_steps': self.warmup_steps,
            'base_lr': self.base_lr,
            'decay_rate': self.decay_rate,
            'decay_steps': self.decay_steps,
            'min_lr': self.min_lr,
            'staircase': self.staircase,
        }
    
    def load_state_dict(self, state_dict):
        """載入狀態（從 checkpoint 恢復）"""
        self.current_step = state_dict['current_step']
        self.warmup_steps = state_dict['warmup_steps']
        self.base_lr = state_dict['base_lr']
        self.decay_rate = state_dict['decay_rate']
        self.decay_steps = state_dict['decay_steps']
        self.min_lr = state_dict['min_lr']
        self.staircase = state_dict['staircase']
