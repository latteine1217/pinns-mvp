"""
訓練計時工具（Training Timer）

功能：
- 記錄總訓練時間
- 記錄每個 epoch 時間
- 記錄達標時間（time to convergence）
- 自動記錄到 WandB
- 提供詳細的時間統計

使用範例：
    >>> from pinnx.utils.timer import Timer
    >>> timer = Timer()
    >>> timer.start()
    >>> for epoch in range(max_epochs):
    ...     # 訓練一個 epoch
    ...     timer.tick(epoch)
    ...     if l2_error < target:
    ...         timer.mark_convergence(epoch, l2_error)
    >>> timer.stop()
    >>> summary = timer.get_summary()
"""

import time
from typing import Optional, Dict, Any
import numpy as np


class Timer:
    """訓練計時工具"""

    def __init__(self, use_wandb: bool = True):
        """
        初始化 Timer

        Args:
            use_wandb: 是否自動記錄到 WandB（預設 True）
        """
        self.use_wandb = use_wandb
        self.start_time: Optional[float] = None
        self.stop_time: Optional[float] = None
        self.epoch_times: list = []
        self.convergence_time: Optional[float] = None
        self.convergence_epoch: Optional[int] = None
        self.convergence_value: Optional[float] = None
        self.last_tick_time: Optional[float] = None

        # 嘗試導入 wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. Timer will not log to WandB.")
                self.use_wandb = False

    def start(self):
        """開始計時"""
        self.start_time = time.time()
        self.last_tick_time = self.start_time
        print(f"⏱️  Timer started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    def tick(self, epoch: Optional[int] = None, log_to_wandb: bool = True):
        """
        記錄一個 epoch 的時間

        Args:
            epoch: 當前 epoch 編號（可選）
            log_to_wandb: 是否記錄到 WandB（預設 True）

        Returns:
            float: 本次 epoch 耗時（秒）
        """
        if self.start_time is None:
            raise RuntimeError("Timer has not been started. Call start() first.")

        current_time = time.time()
        epoch_time = current_time - self.last_tick_time
        cumulative_time = current_time - self.start_time

        self.epoch_times.append(epoch_time)
        self.last_tick_time = current_time

        # 記錄到 WandB
        if self.use_wandb and log_to_wandb and epoch is not None:
            self.wandb.log({
                'timing/epoch_time': epoch_time,
                'timing/cumulative_time': cumulative_time,
                'timing/avg_epoch_time': np.mean(self.epoch_times),
            }, step=epoch)

        return epoch_time

    def mark_convergence(
        self,
        epoch: int,
        metric_value: float,
        log_to_wandb: bool = True
    ):
        """
        標記達標時間（第一次達到目標精度）

        Args:
            epoch: 達標的 epoch
            metric_value: 達標時的 metric 值（例如 L2 error）
            log_to_wandb: 是否記錄到 WandB（預設 True）
        """
        # 只記錄第一次達標
        if self.convergence_time is None:
            self.convergence_time = time.time() - self.start_time
            self.convergence_epoch = epoch
            self.convergence_value = metric_value

            print(f"✅ Convergence reached at epoch {epoch}")
            print(f"   Time to convergence: {self.convergence_time:.2f} seconds")
            print(f"   Metric value: {metric_value:.6f}")

            # 記錄到 WandB
            if self.use_wandb and log_to_wandb:
                self.wandb.log({
                    'timing/time_to_convergence': self.convergence_time,
                    'timing/convergence_epoch': epoch,
                    'timing/convergence_value': metric_value,
                })

    def stop(self):
        """停止計時"""
        if self.start_time is None:
            raise RuntimeError("Timer has not been started.")

        self.stop_time = time.time()
        total_time = self.stop_time - self.start_time

        print(f"\n⏱️  Timer stopped at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Total training time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

        # 記錄到 WandB
        if self.use_wandb:
            self.wandb.log({
                'timing/total_wall_time': total_time,
                'timing/gpu_hours': total_time / 3600,
            })

    def get_summary(self) -> Dict[str, Any]:
        """
        獲取時間統計摘要

        Returns:
            dict: 包含所有時間統計的字典
        """
        if self.start_time is None:
            return {}

        total_time = (self.stop_time or time.time()) - self.start_time
        epoch_times = np.array(self.epoch_times) if self.epoch_times else np.array([0])

        summary = {
            'total_wall_time': total_time,
            'gpu_hours': total_time / 3600,
            'num_epochs': len(self.epoch_times),
            'avg_epoch_time': float(np.mean(epoch_times)),
            'median_epoch_time': float(np.median(epoch_times)),
            'min_epoch_time': float(np.min(epoch_times)),
            'max_epoch_time': float(np.max(epoch_times)),
            'std_epoch_time': float(np.std(epoch_times)),
        }

        # 加入達標時間（如果有）
        if self.convergence_time is not None:
            summary.update({
                'time_to_convergence': self.convergence_time,
                'convergence_epoch': self.convergence_epoch,
                'convergence_value': self.convergence_value,
                'speedup': total_time / self.convergence_time if self.convergence_time > 0 else None,
            })

        return summary

    def print_summary(self):
        """打印時間統計摘要"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("⏱️  Training Time Summary")
        print("="*70)
        print(f"Total training time:    {summary['total_wall_time']:.2f} s ({summary['gpu_hours']:.2f} hours)")
        print(f"Number of epochs:       {summary['num_epochs']}")
        print(f"Average epoch time:     {summary['avg_epoch_time']:.4f} s")
        print(f"Median epoch time:      {summary['median_epoch_time']:.4f} s")
        print(f"Min/Max epoch time:     {summary['min_epoch_time']:.4f} / {summary['max_epoch_time']:.4f} s")
        print(f"Std epoch time:         {summary['std_epoch_time']:.4f} s")

        if 'time_to_convergence' in summary:
            print(f"\n✅ Convergence Metrics:")
            print(f"Time to convergence:    {summary['time_to_convergence']:.2f} s")
            print(f"Convergence epoch:      {summary['convergence_epoch']}")
            print(f"Convergence value:      {summary['convergence_value']:.6f}")
            print(f"Efficiency ratio:       {summary['speedup']:.2f}x (達標後繼續訓練)")

        print("="*70 + "\n")

    def reset(self):
        """重置計時器"""
        self.start_time = None
        self.stop_time = None
        self.epoch_times = []
        self.convergence_time = None
        self.convergence_epoch = None
        self.convergence_value = None
        self.last_tick_time = None


# Context manager 支援
class TimerContext:
    """Timer 的 context manager 包裝"""

    def __init__(self, timer: Timer):
        self.timer = timer

    def __enter__(self):
        self.timer.start()
        return self.timer

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.timer.stop()
        self.timer.print_summary()
        return False  # 不抑制異常


# 便利函數
def create_timer(use_wandb: bool = True) -> Timer:
    """
    建立並返回一個 Timer 實例

    Args:
        use_wandb: 是否使用 WandB 記錄（預設 True）

    Returns:
        Timer: Timer 實例
    """
    return Timer(use_wandb=use_wandb)
