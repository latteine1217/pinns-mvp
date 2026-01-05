"""
pytest 全局配置和共享 fixtures

提供測試輔助函數，簡化測試文件的 Trainer 創建
"""

import pytest
import torch
from typing import Dict, Any, Optional

from pinnx.train.trainer_builder import TrainerBuilder


@pytest.fixture
def create_trainer():
    """
    測試輔助 fixture：使用 TrainerBuilder 創建 Trainer

    這個 fixture 簡化了測試文件的遷移，允許測試繼續使用類似的語法，
    但內部使用 TrainerBuilder。

    用法：
        def test_something(create_trainer, device):
            trainer = create_trainer(
                model=my_model,
                physics=my_physics,
                losses=my_losses,
                config=my_config,
                device=device,
                training_data=my_data  # 可選
            )

    Returns:
        創建 Trainer 的函數
    """
    def _create(
        model,
        physics,
        losses,
        config,
        device,
        training_data: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs  # 忽略其他舊參數（向後兼容）
    ):
        """
        使用 TrainerBuilder 創建 Trainer 實例

        Args:
            model: PyTorch 模型
            physics: 物理模組
            losses: 損失函數字典
            config: 配置字典
            device: 訓練設備
            training_data: 訓練數據（可選）
            **kwargs: 忽略的舊參數（向後兼容）

        Returns:
            Trainer 實例
        """
        builder = TrainerBuilder(config, device)
        builder.with_model(model)
        builder.with_physics(physics)
        builder.with_losses(losses)

        if training_data is not None:
            builder.with_training_data(training_data)

        return builder.build()

    return _create
