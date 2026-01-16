"""
Trainer 組件封裝

提供統一的組件管理接口，將 Trainer 初始化邏輯從 17 個方法簡化為依賴注入。

設計哲學：
1. Good Taste: 消除 Trainer 中的深層初始化邏輯
2. Simplicity: 清晰的組件分類和職責
3. Never Break Userspace: 保持向後兼容

使用方式：
    # 方式 1：由 TrainerBuilder 自動創建
    trainer = TrainerBuilder.from_config(config, device).build()

    # 方式 2：手動組裝組件（高級用戶）
    components = TrainerComponents(
        optimizer=my_optimizer,
        lr_scheduler=my_scheduler,
        ...
    )
    trainer = Trainer(model, physics, losses, config, device, components=components)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


@dataclass
class TrainerComponents:
    """
    Trainer 組件封裝

    將 Trainer 所需的所有可選組件集中管理，避免 Trainer.__init__() 參數過多。

    組件分類：
    - 訓練組件：optimizer, scheduler, weight_scheduler
    - 策略組件：checkpoint, validation
    - 功能組件：prior_loss, fourier, adaptive
    - 監控組件：timer, memory, physics_validator, wandb
    - 數據組件：normalizers, data, cache

    Attributes:
        # 訓練組件
        optimizer: 優化器
        lr_scheduler: 學習率調度器
        weight_scheduler: 權重調度器（暫未實現）

        # 策略組件
        checkpoint_manager: Checkpoint 管理器
        checkpoint_strategy: Checkpoint 保存策略
        validation_manager: Validation 管理器

        # 功能組件
        prior_loss_manager: 先驗損失管理器
        fourier_annealing: Fourier 特徵退火調度器
        adaptive_sampler: 自適應採樣器

        # 監控組件
        timer: 計時器
        memory_tracker: 記憶體追蹤器
        physics_validator: 物理驗證器
        wandb_run: WandB 運行實例

        # 數據組件
        input_normalizer: 輸入標準化器
        data_normalizer: 輸出標準化器
        training_data: 訓練資料
        validation_data: 驗證資料
        channel_data_cache: 通道流資料快取
        weighters: 損失權重器字典
    """

    # ========== 訓練組件 ==========
    optimizer: Optional[torch.optim.Optimizer] = None
    lr_scheduler: Optional[Any] = None
    weight_scheduler: Optional[Any] = None

    # ========== 策略組件 ==========
    checkpoint_manager: Optional[Any] = None  # CheckpointManager
    checkpoint_strategy: Optional[Any] = None  # PeriodicCheckpointStrategy
    validation_manager: Optional[Any] = None  # ValidationManager

    # ========== 功能組件 ==========
    prior_loss_manager: Optional[Any] = None  # PriorLossManager
    fourier_annealing: Optional[Any] = None  # FourierAnnealingScheduler
    adaptive_sampler: Optional[Any] = None  # AdaptiveCollocationSampler
    early_stopping_config: Dict[str, Any] = field(default_factory=dict)

    # ========== 監控組件 ==========
    timer: Optional[Any] = None  # Timer
    memory_tracker: Optional[Any] = None  # MemoryTracker
    physics_validator: Optional[Any] = None  # PhysicsValidator
    wandb_run: Optional[Any] = None  # wandb.Run

    # ========== 數據組件 ==========
    input_normalizer: Optional[Any] = None  # InputTransform
    data_normalizer: Optional[Any] = None  # OutputTransform
    training_data: Optional[Dict[str, torch.Tensor]] = None
    validation_data: Optional[Dict[str, torch.Tensor]] = None
    channel_data_cache: Optional[Dict[str, Any]] = None
    weighters: Optional[Dict[str, Any]] = None
    
    # ========== 邊界條件組件 ==========
    hard_constraint_applicator: Optional[Any] = None  # HardConstraintApplicator
    
    # ========== Time Window 組件 ==========
    step_offset: int = 0  # 訓練步數偏移（用於 time window 訓練，讓各窗口的 step 連續）

    def validate_required_components(self) -> None:
        """
        驗證必需組件是否已設置

        Raises:
            ValueError: 缺少必需組件
        """
        # 必需組件檢查
        if self.optimizer is None:
            raise ValueError("Optimizer is required in TrainerComponents")

        # 可選組件警告
        if self.data_normalizer is None:
            import logging
            logging.warning("⚠️  data_normalizer 未設置，可能影響驗證功能")

    def get_summary(self) -> Dict[str, bool]:
        """
        獲取組件配置摘要

        Returns:
            Dict: 組件名稱到是否配置的映射
        """
        return {
            # 訓練組件
            'optimizer': self.optimizer is not None,
            'lr_scheduler': self.lr_scheduler is not None,
            'weight_scheduler': self.weight_scheduler is not None,

            # 策略組件
            'checkpoint_manager': self.checkpoint_manager is not None,
            'checkpoint_strategy': self.checkpoint_strategy is not None,
            'validation_manager': self.validation_manager is not None,

            # 功能組件
            'prior_loss_manager': self.prior_loss_manager is not None,
            'fourier_annealing': self.fourier_annealing is not None,
            'adaptive_sampler': self.adaptive_sampler is not None,
            'early_stopping': bool(self.early_stopping_config),

            # 監控組件
            'timer': self.timer is not None,
            'memory_tracker': self.memory_tracker is not None,
            'physics_validator': self.physics_validator is not None,
            'wandb_run': self.wandb_run is not None,

            # 數據組件
            'input_normalizer': self.input_normalizer is not None,
            'data_normalizer': self.data_normalizer is not None,
            'training_data': self.training_data is not None,
            'validation_data': self.validation_data is not None,
            'channel_data_cache': bool(self.channel_data_cache),
            'weighters': bool(self.weighters),
            
            # 邊界條件組件
            'hard_constraint_applicator': self.hard_constraint_applicator is not None,
        }

    def __str__(self) -> str:
        """返回組件配置的簡潔描述"""
        summary = self.get_summary()
        configured = [name for name, is_set in summary.items() if is_set]
        total = len(summary)
        return f"TrainerComponents({len(configured)}/{total} configured: {', '.join(configured[:5])}...)"
