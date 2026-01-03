"""
Trainer Builder Pattern

目的：簡化 Trainer 初始化，降低參數複雜度
哲學：Good Taste - 複雜性應該被封裝，而非暴露

使用方式：
    # 舊方式（9 個參數，容易出錯）
    trainer = Trainer(model, physics, losses, config, device,
                      weighters=weighters,
                      input_normalizer=input_normalizer,
                      channel_data_cache=cache,
                      training_data=data)
    
    # 新方式 1：使用 Builder（鏈式呼叫）
    trainer = (TrainerBuilder(config, device)
               .with_model(model)
               .with_physics(physics)
               .with_losses(losses)
               .with_weighters(weighters)
               .with_normalizer(input_normalizer)
               .with_training_data(data)
               .build())
    
    # 新方式 2：使用便捷函數（推薦）
    trainer = create_trainer_with_builder(
        model, physics, losses, config, device,
        weighters=weighters,
        input_normalizer=input_normalizer,
        training_data=data
    )
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from pinnx.train.trainer import Trainer
from pinnx.utils.normalization.input_transform import InputTransform
from pinnx.utils.normalization.output_transform import OutputTransform


class TrainerBuilder:
    """
    Trainer 構建器（Builder Pattern）
    
    優點：
    1. 降低初始化複雜度（不需要記住 9 個參數順序）
    2. 可選參數更清晰（用 .with_xxx() 鏈式呼叫）
    3. 支援自動依賴注入（from_config 工廠方法）
    4. 易於擴展（新增功能不破壞現有代碼）
    
    哲學對齊：
    - Good Taste: 隱藏複雜性，暴露簡潔接口
    - Simplicity: 減少呼叫方的心智負擔
    - Never Break Userspace: 保留舊接口（Trainer.__init__ 仍可用）
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device):
        """
        初始化 Builder（只需要必需參數）
        
        Args:
            config: 完整訓練配置
            device: 計算設備
        """
        self.config = config
        self.device = device
        
        # 必需組件（必須顯式設定）
        self._model: Optional[nn.Module] = None
        self._physics: Optional[Any] = None
        self._losses: Optional[Dict[str, nn.Module]] = None
        
        # 可選組件（有預設值）
        self._weighters: Optional[Dict[str, Any]] = None
        self._input_normalizer: Optional[InputTransform] = None
        self._channel_data_cache: Optional[Dict[str, Any]] = None
        self._training_data: Optional[Dict[str, torch.Tensor]] = None
    
    def with_model(self, model: nn.Module) -> 'TrainerBuilder':
        """設定模型"""
        self._model = model
        return self
    
    def with_physics(self, physics: Any) -> 'TrainerBuilder':
        """設定物理模組"""
        self._physics = physics
        return self
    
    def with_losses(self, losses: Dict[str, nn.Module]) -> 'TrainerBuilder':
        """設定損失函數字典"""
        self._losses = losses
        return self
    
    def with_weighters(self, weighters: Dict[str, Any]) -> 'TrainerBuilder':
        """設定權重調度器"""
        self._weighters = weighters
        return self
    
    def with_normalizer(self, normalizer: InputTransform) -> 'TrainerBuilder':
        """設定輸入標準化器"""
        self._input_normalizer = normalizer
        return self
    
    def with_channel_cache(self, cache: Dict[str, Any]) -> 'TrainerBuilder':
        """設定通道流資料快取"""
        self._channel_data_cache = cache
        return self
    
    def with_training_data(self, data: Dict[str, torch.Tensor]) -> 'TrainerBuilder':
        """設定訓練資料"""
        self._training_data = data
        return self
    
    def build(self) -> Trainer:
        """
        構建 Trainer 實例
        
        Returns:
            Trainer: 完整初始化的訓練器
        
        Raises:
            ValueError: 缺少必需組件
        """
        # Fail Fast: 驗證必需組件
        if self._model is None:
            raise ValueError("Model is required. Call .with_model(model) first.")
        if self._physics is None:
            raise ValueError("Physics module is required. Call .with_physics(physics) first.")
        if self._losses is None:
            raise ValueError("Losses are required. Call .with_losses(losses) first.")
        
        # 構建 Trainer
        trainer = Trainer(
            model=self._model,
            physics=self._physics,
            losses=self._losses,
            config=self.config,
            device=self.device,
            weighters=self._weighters,
            input_normalizer=self._input_normalizer,
            channel_data_cache=self._channel_data_cache,
            training_data=self._training_data,
        )
        
        logging.info("✅ TrainerBuilder: Trainer 構建完成")
        return trainer
    
    @classmethod
    def from_components(
        cls,
        model: nn.Module,
        physics: Any,
        losses: Dict[str, nn.Module],
        config: Dict[str, Any],
        device: torch.device,
        **optional_components
    ) -> Trainer:
        """
        工廠方法：從已創建的組件構建 Trainer（推薦方式）
        
        這個方法比直接呼叫 Trainer.__init__() 更清晰，同時保持靈活性。
        
        Args:
            model: 已創建的模型
            physics: 已創建的物理模組
            losses: 已創建的損失函數字典
            config: 完整訓練配置
            device: 計算設備
            **optional_components: 可選組件（weighters, input_normalizer, training_data, etc.）
        
        Returns:
            Trainer: 完整初始化的訓練器
        
        Example:
            >>> # 在 scripts/train/train.py 中
            >>> model = create_model(config, device)
            >>> physics = create_physics(config, device)
            >>> losses = create_loss_functions(config, device)
            >>> 
            >>> trainer = TrainerBuilder.from_components(
            ...     model, physics, losses, config, device,
            ...     weighters=weighters,
            ...     input_normalizer=normalizer,
            ...     training_data=training_data
            ... )
        """
        builder = cls(config, device)
        builder.with_model(model)
        builder.with_physics(physics)
        builder.with_losses(losses)
        
        # 處理可選組件
        if 'weighters' in optional_components:
            builder.with_weighters(optional_components['weighters'])
        if 'input_normalizer' in optional_components:
            builder.with_normalizer(optional_components['input_normalizer'])
        if 'channel_data_cache' in optional_components:
            builder.with_channel_cache(optional_components['channel_data_cache'])
        if 'training_data' in optional_components:
            builder.with_training_data(optional_components['training_data'])
        
        return builder.build()


# ============================================================================
# 便捷函數（Convenience Functions）
# ============================================================================

def create_trainer_with_builder(
    model: nn.Module,
    physics: Any,
    losses: Dict[str, nn.Module],
    config: Dict[str, Any],
    device: torch.device,
    **optional_components
) -> Trainer:
    """
    便捷函數：使用 Builder Pattern 創建 Trainer
    
    這是推薦的創建方式，比直接呼叫 Trainer.__init__() 更清晰。
    
    Args:
        model: 已創建的模型
        physics: 已創建的物理模組
        losses: 已創建的損失函數字典
        config: 完整訓練配置
        device: 計算設備
        **optional_components: 可選組件（weighters, input_normalizer, training_data, etc.）
    
    Returns:
        Trainer: 完整初始化的訓練器
    
    Example:
        >>> # 在 scripts/train/train.py 中，從這樣：
        >>> # trainer = Trainer(model, physics, losses, config, device,
        >>> #                    weighters=weighters, input_normalizer=normalizer, ...)
        >>> 
        >>> # 改為這樣：
        >>> trainer = create_trainer_with_builder(
        ...     model, physics, losses, config, device,
        ...     weighters=weighters,
        ...     input_normalizer=normalizer,
        ...     training_data=training_data
        ... )
    """
    return TrainerBuilder.from_components(
        model, physics, losses, config, device,
        **optional_components
    )
