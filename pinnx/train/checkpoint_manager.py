"""
Checkpoint 管理模組

提供統一的 checkpoint 保存、加載、驗證接口，將 checkpoint 邏輯從 Trainer 中解耦。

設計模式：
- Manager Pattern: 集中管理 checkpoint 相關邏輯
- Strategy Pattern: 可插拔的保存策略
- Dependency Injection: Trainer 接收 Manager 實例

設計哲學：
1. Single Responsibility: CheckpointManager 只負責 checkpoint I/O
2. Good Taste: 消除條件分支，使用多態
3. Simplicity: 清晰的接口，易於理解和擴展
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import logging
import os
import gzip
import pickle


class CheckpointManager(ABC):
    """
    Checkpoint 管理器抽象基類

    職責：
    - 構建 checkpoint 數據結構
    - 保存 checkpoint 到磁盤
    - 從磁盤加載 checkpoint
    - 驗證 checkpoint 完整性

    設計原則：
    - 接口統一：所有 CheckpointManager 使用相同的 save/load 接口
    - 可測試性：不依賴 Trainer，可獨立測試
    - 可擴展性：支持自定義 checkpoint 格式
    """

    @abstractmethod
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        **kwargs
    ) -> str:
        """
        保存 checkpoint

        Args:
            model: PINN 模型
            optimizer: 優化器
            epoch: 當前訓練輪次
            metrics: 訓練指標字典 (e.g., {'total_loss': 0.5, 'data_loss': 0.1})
            config: 訓練配置
            **kwargs: 額外數據 (scheduler_state, scaler_state, physics_validation, etc.)

        Returns:
            保存的文件路徑

        Raises:
            IOError: 保存失敗
        """
        pass

    @abstractmethod
    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """
        加載 checkpoint

        Args:
            checkpoint_path: checkpoint 文件路徑
            model: PINN 模型（用於加載 state_dict）
            optimizer: 優化器（可選，用於恢復訓練狀態）

        Returns:
            checkpoint 數據字典，包含：
            - epoch (int): 訓練輪次
            - metrics (Dict[str, float]): 訓練指標
            - config (Dict[str, Any]): 訓練配置
            - physics_validation (Dict, optional): 物理驗證結果
            - scheduler_state (Dict, optional): Scheduler 狀態
            - scaler_state (Dict, optional): Scaler 狀態

        Raises:
            FileNotFoundError: checkpoint 文件不存在
            ValueError: checkpoint 格式錯誤
        """
        pass

    @abstractmethod
    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """
        驗證 checkpoint 文件是否完整

        Args:
            checkpoint_path: checkpoint 文件路徑

        Returns:
            True 如果 checkpoint 完整，否則 False
        """
        pass


class StandardCheckpointManager(CheckpointManager):
    """
    標準 Checkpoint 管理器

    特性：
    - 保存完整訓練狀態（model, optimizer, scheduler, config, metrics）
    - 支持 physics 驗證結果保存
    - 自動生成文件名（基於 epoch 和 metrics）
    - 可選壓縮（減少磁盤空間）
    - 支持自定義文件命名規則

    使用示例：
    ```python
    manager = StandardCheckpointManager(
        checkpoint_dir='./checkpoints',
        save_optimizer=True,
        compress=False
    )

    # 保存
    filepath = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=100,
        metrics={'total_loss': 0.5},
        config=config,
        scheduler_state=scheduler.state_dict()
    )

    # 加載
    data = manager.load(filepath, model, optimizer)
    print(f"Resumed from epoch {data['epoch']}")
    ```
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_optimizer: bool = True,
        save_physics_validation: bool = True,
        compress: bool = False,
        filename_pattern: str = "checkpoint_epoch{epoch}_loss{loss:.6f}_{timestamp}.pt"
    ):
        """
        初始化 StandardCheckpointManager

        Args:
            checkpoint_dir: checkpoint 保存目錄
            save_optimizer: 是否保存 optimizer state
            save_physics_validation: 是否保存 physics 驗證結果
            compress: 是否壓縮 checkpoint（使用 gzip）
            filename_pattern: 文件命名模式（支持 {epoch}, {loss}, {timestamp}）
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_optimizer = save_optimizer
        self.save_physics_validation = save_physics_validation
        self.compress = compress
        self.filename_pattern = filename_pattern

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        **kwargs
    ) -> str:
        """保存 checkpoint 到 checkpoint_dir"""
        # 構建 checkpoint 數據
        checkpoint_data = self._build_checkpoint_data(
            model, optimizer, epoch, metrics, config, **kwargs
        )

        # 生成文件路徑
        filepath = self._generate_filepath(epoch, metrics)

        # 保存到磁盤
        try:
            if self.compress:
                self._save_compressed(checkpoint_data, filepath)
            else:
                torch.save(checkpoint_data, filepath)

            logging.info(f"✅ Checkpoint saved: {filepath}")
            return str(filepath)

        except Exception as e:
            logging.error(f"❌ Failed to save checkpoint: {e}")
            raise IOError(f"Checkpoint save failed: {e}") from e

    def load(
        self,
        checkpoint_path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> Dict[str, Any]:
        """從磁盤加載 checkpoint"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            # 加載 checkpoint 數據
            if self.compress or checkpoint_path.suffix == '.gz':
                checkpoint = self._load_compressed(str(checkpoint_path))
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # 驗證 checkpoint 完整性
            if not self._validate_checkpoint_data(checkpoint):
                raise ValueError(f"Invalid checkpoint format: {checkpoint_path}")

            # 加載 model state
            model.load_state_dict(checkpoint['model_state_dict'])
            logging.info(f"✅ Model state loaded from {checkpoint_path}")

            # 加載 optimizer state（如果提供）
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logging.info("✅ Optimizer state loaded")

            # 返回 checkpoint 數據
            return {
                'epoch': checkpoint['epoch'],
                'metrics': checkpoint.get('metrics', {}),
                'config': checkpoint.get('config', {}),
                'physics_validation': checkpoint.get('physics_validation', {}),
                'scheduler_state': checkpoint.get('scheduler_state'),
                'scaler_state': checkpoint.get('scaler_state'),
                'normalizers': checkpoint.get('normalizers')
            }

        except Exception as e:
            logging.error(f"❌ Failed to load checkpoint: {e}")
            raise ValueError(f"Checkpoint load failed: {e}") from e

    def validate_checkpoint(self, checkpoint_path: str) -> bool:
        """驗證 checkpoint 完整性"""
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            return False

        try:
            if self.compress or checkpoint_path.suffix == '.gz':
                checkpoint = self._load_compressed(str(checkpoint_path))
            else:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')

            return self._validate_checkpoint_data(checkpoint)

        except Exception:
            return False

    def _build_checkpoint_data(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """構建 checkpoint 數據結構"""
        data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }

        # 可選：保存 optimizer state
        if self.save_optimizer:
            data['optimizer_state_dict'] = optimizer.state_dict()

        # 可選：保存 physics 驗證結果
        if self.save_physics_validation and 'physics_validation' in kwargs:
            data['physics_validation'] = kwargs['physics_validation']

        # 支持額外參數
        optional_keys = ['scheduler_state', 'scaler_state', 'normalizers', 'gradient_scaler']
        for key in optional_keys:
            if key in kwargs:
                data[key] = kwargs[key]

        return data

    def _generate_filepath(self, epoch: int, metrics: Dict[str, float]) -> Path:
        """生成 checkpoint 文件路徑"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        loss = metrics.get('total_loss', 0.0)

        filename = self.filename_pattern.format(
            epoch=epoch,
            loss=loss,
            timestamp=timestamp
        )

        if self.compress and not filename.endswith('.gz'):
            filename += '.gz'

        return self.checkpoint_dir / filename

    def _validate_checkpoint_data(self, checkpoint: Dict) -> bool:
        """驗證 checkpoint 數據完整性"""
        required_keys = ['model_state_dict', 'epoch']
        return all(key in checkpoint for key in required_keys)

    def _save_compressed(self, data: Dict, filepath: Path):
        """保存壓縮的 checkpoint"""
        with gzip.open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def _load_compressed(self, filepath: str) -> Dict:
        """加載壓縮的 checkpoint"""
        with gzip.open(filepath, 'rb') as f:
            return pickle.load(f)


class PeriodicCheckpointStrategy:
    """
    定期保存策略

    特性：
    - 每 N epoch 保存一次
    - 保存最佳 K 個 checkpoint（基於指定 metric）
    - 自動清理舊 checkpoint
    - 支持 min/max 模式（loss vs accuracy）

    使用示例：
    ```python
    strategy = PeriodicCheckpointStrategy(
        save_interval=100,
        keep_best_k=3,
        metric_name='total_loss',
        mode='min'
    )

    # 判斷是否應該保存
    if strategy.should_save(epoch, metrics):
        filepath = manager.save(...)

        # 更新最佳 checkpoint 記錄
        if strategy.is_best(metrics):
            strategy.update_best_checkpoints(
                metrics['total_loss'],
                filepath
            )
    ```
    """

    def __init__(
        self,
        save_interval: int = 100,
        keep_best_k: int = 3,
        metric_name: str = 'total_loss',
        mode: str = 'min'
    ):
        """
        初始化保存策略

        Args:
            save_interval: 每多少 epoch 保存一次
            keep_best_k: 保留最佳 K 個 checkpoint
            metric_name: 用於判斷「最佳」的指標名稱
            mode: 'min'（越小越好）或 'max'（越大越好）
        """
        self.save_interval = save_interval
        self.keep_best_k = keep_best_k
        self.metric_name = metric_name
        self.mode = mode
        self.best_checkpoints: List[Tuple[float, str]] = []  # [(metric_value, filepath), ...]

    def should_save(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """
        判斷是否應該保存

        Args:
            epoch: 當前訓練輪次
            metrics: 訓練指標字典

        Returns:
            True 如果應該保存，否則 False
        """
        # 定期保存
        if epoch % self.save_interval == 0:
            return True

        # 或者是最佳結果
        if self.is_best(metrics):
            return True

        return False

    def is_best(self, metrics: Dict[str, float]) -> bool:
        """
        判斷當前指標是否為最佳

        Args:
            metrics: 訓練指標字典

        Returns:
            True 如果是最佳結果
        """
        if self.metric_name not in metrics:
            return False

        current_value = metrics[self.metric_name]

        # 如果還沒有足夠的 best checkpoints，直接認為是 best
        if len(self.best_checkpoints) < self.keep_best_k:
            return True

        # 找到當前最差的 best checkpoint
        if self.mode == 'min':
            worst_best = max(self.best_checkpoints, key=lambda x: x[0])
            return current_value < worst_best[0]
        else:  # mode == 'max'
            worst_best = min(self.best_checkpoints, key=lambda x: x[0])
            return current_value > worst_best[0]

    def update_best_checkpoints(
        self,
        metric_value: float,
        filepath: str
    ):
        """
        更新最佳 checkpoint 列表

        Args:
            metric_value: 當前指標值
            filepath: checkpoint 文件路徑
        """
        self.best_checkpoints.append((metric_value, filepath))

        # 排序（min 模式：小的在前；max 模式：大的在前）
        self.best_checkpoints.sort(
            key=lambda x: x[0],
            reverse=(self.mode == 'max')
        )

        # 保留最佳 K 個，刪除多餘的
        if len(self.best_checkpoints) > self.keep_best_k:
            to_remove = self.best_checkpoints[self.keep_best_k:]

            for _, old_filepath in to_remove:
                self._remove_checkpoint(old_filepath)

            self.best_checkpoints = self.best_checkpoints[:self.keep_best_k]

    def get_best_checkpoint(self) -> Optional[str]:
        """
        獲取當前最佳 checkpoint 路徑

        Returns:
            最佳 checkpoint 路徑，如果沒有則返回 None
        """
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0][1]

    def _remove_checkpoint(self, filepath: str):
        """刪除 checkpoint 文件"""
        try:
            filepath_obj = Path(filepath)
            if filepath_obj.exists():
                filepath_obj.unlink()
                logging.info(f"🗑️  Removed old checkpoint: {filepath}")
        except Exception as e:
            logging.warning(f"⚠️  Failed to remove checkpoint {filepath}: {e}")
