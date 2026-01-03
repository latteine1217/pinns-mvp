"""
記憶體追蹤工具（Memory Tracker）

功能：
- 追蹤 GPU 記憶體使用（current / peak）
- 追蹤模型參數量
- 估算激活值記憶體
- 自動記錄到 WandB
- 提供記憶體使用報告

使用範例：
    >>> from pinnx.utils.memory_tracker import MemoryTracker
    >>> tracker = MemoryTracker(model)
    >>> tracker.reset()
    >>> # 訓練...
    >>> tracker.log_current(epoch)
    >>> summary = tracker.get_summary()
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import gc


class MemoryTracker:
    """GPU 記憶體追蹤工具"""

    def __init__(self, model: Optional[nn.Module] = None, use_wandb: bool = True):
        """
        初始化 MemoryTracker

        Args:
            model: PyTorch 模型（可選，用於計算參數量）
            use_wandb: 是否自動記錄到 WandB（預設 True）
        """
        self.model = model
        self.use_wandb = use_wandb
        self.device_available = torch.cuda.is_available()

        # 模型參數統計
        self.model_params = None
        self.model_size_mb = None

        if self.model is not None:
            self._compute_model_stats()

        # 記憶體追蹤
        self.peak_memory_mb = 0.0
        self.memory_history = []

        # 嘗試導入 wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. MemoryTracker will not log to WandB.")
                self.use_wandb = False

        # 初始化
        if self.device_available:
            self.reset()

    def _compute_model_stats(self):
        """計算模型參數統計"""
        if self.model is None:
            return

        # 總參數量
        self.model_params = sum(p.numel() for p in self.model.parameters())

        # 可訓練參數量
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # 模型大小（MB）- 假設 float32
        self.model_size_mb = self.model_params * 4 / (1024 ** 2)

        print(f"\n📊 Model Statistics:")
        print(f"   Total parameters:      {self.model_params:,}")
        print(f"   Trainable parameters:  {trainable_params:,}")
        print(f"   Model size:            {self.model_size_mb:.2f} MB")

    def reset(self):
        """重置記憶體追蹤"""
        if self.device_available:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            gc.collect()
            self.peak_memory_mb = 0.0
            self.memory_history = []
            print("🔄 GPU memory tracker reset")

    def get_current_memory(self) -> Tuple[float, float]:
        """
        獲取當前 GPU 記憶體使用

        Returns:
            tuple: (current_memory_mb, peak_memory_mb)
        """
        if not self.device_available:
            return 0.0, 0.0

        current = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        # 更新峰值
        if peak > self.peak_memory_mb:
            self.peak_memory_mb = peak

        return current, peak

    def log_current(self, epoch: Optional[int] = None, log_to_wandb: bool = True):
        """
        記錄當前記憶體使用

        Args:
            epoch: 當前 epoch 編號（可選）
            log_to_wandb: 是否記錄到 WandB（預設 True）

        Returns:
            dict: 當前記憶體統計
        """
        current_mb, peak_mb = self.get_current_memory()

        stats = {
            'current_memory_mb': current_mb,
            'peak_memory_mb': peak_mb,
        }

        # 記錄歷史
        self.memory_history.append(stats.copy())

        # 記錄到 WandB
        if self.use_wandb and log_to_wandb and epoch is not None:
            log_dict = {
                'memory/current_gpu_memory_mb': current_mb,
                'memory/peak_gpu_memory_mb': peak_mb,
            }

            if self.model_params is not None:
                log_dict['memory/model_parameters'] = self.model_params
                log_dict['memory/model_size_mb'] = self.model_size_mb

            self.wandb.log(log_dict, step=epoch)

        return stats

    def estimate_activation_memory(
        self,
        batch_size: int,
        input_dim: int,
        output_dim: int,
        depth: int,
        width: int
    ) -> float:
        """
        估算激活值記憶體（粗略估計）

        Args:
            batch_size: Batch size
            input_dim: 輸入維度
            output_dim: 輸出維度
            depth: 網路深度
            width: 網路寬度

        Returns:
            float: 估算的激活值記憶體（MB）
        """
        # 每層激活值大小 = batch_size * width * 4 bytes (float32)
        activation_per_layer = batch_size * width * 4 / (1024 ** 2)

        # 總激活值 ≈ depth * activation_per_layer
        total_activation_mb = depth * activation_per_layer

        return total_activation_mb

    def measure_inference_memory(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10
    ) -> Dict[str, float]:
        """
        測量推理時的記憶體使用

        Args:
            model: 模型
            input_tensor: 輸入張量
            num_runs: 測試次數（預設 10）

        Returns:
            dict: 推理記憶體統計
        """
        if not self.device_available:
            return {'inference_memory_mb': 0.0}

        model.eval()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            # 預熱
            for _ in range(5):
                _ = model(input_tensor)

            # 測量
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated() / (1024 ** 2)

            for _ in range(num_runs):
                _ = model(input_tensor)

            end_mem = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

        inference_memory = peak_mem - start_mem

        return {
            'inference_memory_mb': inference_memory,
            'inference_peak_memory_mb': peak_mem,
        }

    def get_summary(self) -> Dict[str, Any]:
        """
        獲取記憶體統計摘要

        Returns:
            dict: 包含所有記憶體統計的字典
        """
        current_mb, peak_mb = self.get_current_memory()

        summary = {
            'current_memory_mb': current_mb,
            'peak_memory_mb': self.peak_memory_mb,
        }

        if self.model_params is not None:
            summary.update({
                'model_parameters': self.model_params,
                'model_size_mb': self.model_size_mb,
            })

        if self.memory_history:
            import numpy as np
            history_array = [h['current_memory_mb'] for h in self.memory_history]
            summary.update({
                'avg_memory_mb': float(np.mean(history_array)),
                'median_memory_mb': float(np.median(history_array)),
                'std_memory_mb': float(np.std(history_array)),
            })

        # GPU 資訊
        if self.device_available:
            summary.update({
                'gpu_name': torch.cuda.get_device_name(0),
                'gpu_count': torch.cuda.device_count(),
                'cuda_version': torch.version.cuda,
            })

        return summary

    def print_summary(self):
        """打印記憶體統計摘要"""
        summary = self.get_summary()

        print("\n" + "="*70)
        print("💾 GPU Memory Summary")
        print("="*70)

        if self.device_available:
            print(f"GPU Name:               {summary.get('gpu_name', 'N/A')}")
            print(f"CUDA Version:           {summary.get('cuda_version', 'N/A')}")
            print(f"\nCurrent memory:         {summary['current_memory_mb']:.2f} MB")
            print(f"Peak memory:            {summary['peak_memory_mb']:.2f} MB")

            if 'avg_memory_mb' in summary:
                print(f"Average memory:         {summary['avg_memory_mb']:.2f} MB")
                print(f"Median memory:          {summary['median_memory_mb']:.2f} MB")
                print(f"Std memory:             {summary['std_memory_mb']:.2f} MB")

            if 'model_parameters' in summary:
                print(f"\nModel parameters:       {summary['model_parameters']:,}")
                print(f"Model size:             {summary['model_size_mb']:.2f} MB")
        else:
            print("GPU not available")

        print("="*70 + "\n")

    def log_summary_to_wandb(self):
        """將摘要記錄到 WandB"""
        if not self.use_wandb:
            return

        summary = self.get_summary()

        log_dict = {
            'memory/final_current_mb': summary['current_memory_mb'],
            'memory/final_peak_mb': summary['peak_memory_mb'],
        }

        if 'model_parameters' in summary:
            log_dict.update({
                'memory/model_parameters': summary['model_parameters'],
                'memory/model_size_mb': summary['model_size_mb'],
            })

        if 'avg_memory_mb' in summary:
            log_dict.update({
                'memory/avg_memory_mb': summary['avg_memory_mb'],
                'memory/median_memory_mb': summary['median_memory_mb'],
            })

        self.wandb.log(log_dict)


def get_gpu_memory_info() -> Dict[str, Any]:
    """
    獲取 GPU 記憶體資訊（全局函數）

    Returns:
        dict: GPU 記憶體資訊
    """
    if not torch.cuda.is_available():
        return {'available': False}

    return {
        'available': True,
        'current_mb': torch.cuda.memory_allocated() / (1024 ** 2),
        'peak_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
        'free_mb': (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated()) / (1024 ** 2),
        'total_mb': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2),
        'gpu_name': torch.cuda.get_device_name(0),
        'gpu_count': torch.cuda.device_count(),
    }
