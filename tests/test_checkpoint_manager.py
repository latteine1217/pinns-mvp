"""
Checkpoint Manager 單元測試

測試覆蓋：
1. StandardCheckpointManager.save() - 保存功能
2. StandardCheckpointManager.load() - 加載功能
3. StandardCheckpointManager.validate_checkpoint() - 驗證功能
4. PeriodicCheckpointStrategy - 保存策略邏輯
5. 壓縮功能測試
6. 完整的 save/load 循環測試
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from pinnx.train.checkpoint_manager import (
    StandardCheckpointManager,
    PeriodicCheckpointStrategy
)


class SimpleModel(nn.Module):
    """測試用簡單模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestStandardCheckpointManager:
    """測試 StandardCheckpointManager"""

    @pytest.fixture
    def temp_dir(self):
        """創建臨時目錄用於測試"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def model(self):
        """創建測試模型"""
        return SimpleModel()

    @pytest.fixture
    def optimizer(self, model):
        """創建測試優化器"""
        return torch.optim.Adam(model.parameters(), lr=0.001)

    @pytest.fixture
    def manager(self, temp_dir):
        """創建 CheckpointManager 實例"""
        return StandardCheckpointManager(
            checkpoint_dir=temp_dir,
            save_optimizer=True,
            compress=False
        )

    def test_save_checkpoint(self, manager, model, optimizer, temp_dir):
        """測試保存 checkpoint"""
        epoch = 100
        metrics = {'total_loss': 0.5, 'data_loss': 0.1}
        config = {'learning_rate': 0.001}

        # 保存 checkpoint
        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config
        )

        # 驗證文件存在
        assert Path(filepath).exists()
        assert Path(filepath).parent == Path(temp_dir)

        # 驗證文件名包含 epoch 和 loss
        assert f"epoch{epoch}" in Path(filepath).name
        assert "loss" in Path(filepath).name

    def test_load_checkpoint(self, manager, model, optimizer, temp_dir):
        """測試加載 checkpoint"""
        epoch = 50
        metrics = {'total_loss': 0.3}
        config = {'test': 'config'}

        # 保存 checkpoint
        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            config=config
        )

        # 創建新模型和優化器
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # 加載 checkpoint
        loaded_data = manager.load(
            checkpoint_path=filepath,
            model=new_model,
            optimizer=new_optimizer
        )

        # 驗證加載的數據
        assert loaded_data['epoch'] == epoch
        assert loaded_data['metrics'] == metrics
        assert loaded_data['config'] == config

        # 驗證模型參數已加載
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_save_load_cycle(self, manager, model, optimizer):
        """測試完整的 save/load 循環"""
        # 保存原始模型狀態
        original_params = [p.clone() for p in model.parameters()]

        # 保存 checkpoint
        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=10,
            metrics={'loss': 0.1},
            config={}
        )

        # 修改模型參數
        for p in model.parameters():
            p.data.fill_(0.0)

        # 加載 checkpoint（應恢復原始參數）
        manager.load(filepath, model, optimizer)

        # 驗證參數已恢復
        for orig, curr in zip(original_params, model.parameters()):
            assert torch.allclose(orig, curr)

    def test_validate_checkpoint(self, manager, model, optimizer, temp_dir):
        """測試 checkpoint 驗證"""
        # 保存有效的 checkpoint
        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={},
            config={}
        )

        # 驗證應該通過
        assert manager.validate_checkpoint(filepath)

        # 不存在的文件應該失敗
        assert not manager.validate_checkpoint(str(Path(temp_dir) / "nonexistent.pt"))

    def test_save_with_extra_data(self, manager, model, optimizer):
        """測試保存額外數據（scheduler, scaler, etc.）"""
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
        physics_validation = {'continuity_error': 0.01}

        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={},
            config={},
            scheduler_state=scheduler.state_dict(),
            physics_validation=physics_validation
        )

        # 加載並驗證額外數據
        loaded_data = manager.load(filepath, model)

        assert 'scheduler_state' in loaded_data
        assert loaded_data['physics_validation'] == physics_validation

    def test_compressed_checkpoint(self, temp_dir, model, optimizer):
        """測試壓縮 checkpoint"""
        manager = StandardCheckpointManager(
            checkpoint_dir=temp_dir,
            compress=True
        )

        filepath = manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={},
            config={}
        )

        # 驗證文件名有 .gz 後綴
        assert filepath.endswith('.gz')

        # 驗證可以正常加載
        new_model = SimpleModel()
        loaded_data = manager.load(filepath, new_model)
        assert loaded_data['epoch'] == 1


class TestPeriodicCheckpointStrategy:
    """測試 PeriodicCheckpointStrategy"""

    @pytest.fixture
    def strategy(self):
        """創建測試策略"""
        return PeriodicCheckpointStrategy(
            save_interval=100,
            keep_best_k=3,
            metric_name='total_loss',
            mode='min'
        )

    def test_should_save_periodic(self, strategy):
        """測試定期保存邏輯"""
        # 先填滿 best_checkpoints 列表（keep_best_k=3）
        strategy.update_best_checkpoints(0.1, '/tmp/ckpt1.pt')
        strategy.update_best_checkpoints(0.2, '/tmp/ckpt2.pt')
        strategy.update_best_checkpoints(0.3, '/tmp/ckpt3.pt')

        # 測試定期保存
        metrics = {'total_loss': 0.5}  # 不是最佳（比 0.3 差）

        # epoch=100 應該保存（定期）
        assert strategy.should_save(100, metrics)

        # epoch=200 應該保存（定期）
        assert strategy.should_save(200, metrics)

        # epoch=50 不應該保存（不是定期，也不是最佳）
        assert not strategy.should_save(50, metrics)

    def test_should_save_best(self, strategy):
        """測試最佳結果保存邏輯"""
        # 第一次總是最佳（列表未滿）
        assert strategy.is_best({'total_loss': 1.0})

        # 填滿列表（keep_best_k=3）
        strategy.update_best_checkpoints(1.0, '/tmp/ckpt1.pt')
        strategy.update_best_checkpoints(0.8, '/tmp/ckpt2.pt')
        strategy.update_best_checkpoints(0.9, '/tmp/ckpt3.pt')

        # 更好的結果應該判斷為最佳
        assert strategy.is_best({'total_loss': 0.5})

        # 更差的結果不應該判斷為最佳（比當前最差的 1.0 還差）
        assert not strategy.is_best({'total_loss': 2.0})

    def test_keep_best_k(self, strategy):
        """測試保留最佳 K 個 checkpoint"""
        # 添加 5 個 checkpoint（keep_best_k=3）
        checkpoints = [
            (1.0, '/tmp/ckpt1.pt'),
            (0.5, '/tmp/ckpt2.pt'),
            (0.8, '/tmp/ckpt3.pt'),
            (0.3, '/tmp/ckpt4.pt'),
            (0.9, '/tmp/ckpt5.pt'),
        ]

        for loss, path in checkpoints:
            strategy.update_best_checkpoints(loss, path)

        # 應該只保留 3 個最佳
        assert len(strategy.best_checkpoints) == 3

        # 驗證保留的是最小的 3 個
        kept_losses = [loss for loss, _ in strategy.best_checkpoints]
        assert kept_losses == [0.3, 0.5, 0.8]

    def test_mode_max(self):
        """測試 max 模式（例如 accuracy）"""
        strategy = PeriodicCheckpointStrategy(
            save_interval=100,
            keep_best_k=2,
            metric_name='accuracy',
            mode='max'
        )

        # 添加 checkpoint
        strategy.update_best_checkpoints(0.8, '/tmp/ckpt1.pt')
        strategy.update_best_checkpoints(0.9, '/tmp/ckpt2.pt')
        strategy.update_best_checkpoints(0.7, '/tmp/ckpt3.pt')

        # 應該保留最大的 2 個
        kept_values = [val for val, _ in strategy.best_checkpoints]
        assert kept_values == [0.9, 0.8]

    def test_get_best_checkpoint(self, strategy):
        """測試獲取最佳 checkpoint"""
        # 空列表應返回 None
        assert strategy.get_best_checkpoint() is None

        # 添加 checkpoint
        strategy.update_best_checkpoints(0.5, '/tmp/best.pt')
        strategy.update_best_checkpoints(0.8, '/tmp/worse.pt')

        # 應返回最佳的
        assert strategy.get_best_checkpoint() == '/tmp/best.pt'

    def test_metric_not_in_dict(self, strategy):
        """測試 metric 不存在的情況"""
        # is_best 應該返回 False
        assert not strategy.is_best({'other_metric': 0.5})

        # should_save 應該只根據定期保存判斷
        assert not strategy.should_save(50, {'other_metric': 0.5})
        assert strategy.should_save(100, {'other_metric': 0.5})


class TestCheckpointManagerIntegration:
    """集成測試"""

    @pytest.fixture
    def temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_full_workflow(self, temp_dir):
        """測試完整的 checkpoint 管理流程"""
        # 創建管理器和策略
        manager = StandardCheckpointManager(checkpoint_dir=temp_dir)
        strategy = PeriodicCheckpointStrategy(
            save_interval=10,
            keep_best_k=2,
            metric_name='loss',
            mode='min'
        )

        # 創建模型和優化器
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        # 模擬訓練循環
        saved_checkpoints = []
        for epoch in range(1, 31):
            metrics = {'loss': 1.0 / epoch}  # 逐漸降低的 loss

            if strategy.should_save(epoch, metrics):
                filepath = manager.save(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    config={}
                )
                saved_checkpoints.append(filepath)

                if strategy.is_best(metrics):
                    strategy.update_best_checkpoints(
                        metrics['loss'],
                        filepath
                    )

        # 驗證：應該保存了定期 checkpoint (10, 20, 30)
        assert len(saved_checkpoints) >= 3

        # 驗證：最佳列表應該只有 2 個
        assert len(strategy.best_checkpoints) == 2

        # 驗證：最佳的應該是 epoch 30 和 29（loss 最小）
        best_checkpoint = strategy.get_best_checkpoint()
        assert best_checkpoint is not None

        # 驗證：可以加載最佳 checkpoint
        new_model = SimpleModel()
        loaded_data = manager.load(best_checkpoint, new_model)
        assert loaded_data['epoch'] in [29, 30]
