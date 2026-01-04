"""
ValidationManager 單元測試

測試覆蓋：
1. DataBasedValidation - 數據驗證策略
2. PhysicsBasedValidation - 物理驗證策略
3. ValidationManager - 驗證管理器
4. 完整的驗證工作流測試
"""

import pytest
import torch
import torch.nn as nn
from pinnx.train.validation_manager import (
    ValidationStrategy,
    DataBasedValidation,
    PhysicsBasedValidation,
    ValidationManager
)


class SimpleModel(nn.Module):
    """測試用簡單模型"""
    def __init__(self, input_dim=2, output_dim=4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


class DummyNormalizer:
    """測試用標準化器"""
    def denormalize_batch(self, x, var_order=None):
        return x * 2.0  # 簡單的反標準化


class DummyPhysicsValidator:
    """測試用物理驗證器"""
    def validate(self, model, test_data, log_to_wandb=False):
        # 返回虛擬的物理驗證結果
        return {
            'physics_continuity': 0.001,
            'physics_boundary': 0.002,
            'physics_momentum': 0.003
        }


class TestDataBasedValidation:
    """測試 DataBasedValidation"""

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    @pytest.fixture
    def model(self):
        return SimpleModel(input_dim=2, output_dim=4)

    @pytest.fixture
    def validation_data(self):
        return {
            'coords': torch.randn(100, 2),
            'targets': torch.randn(100, 4),
            'size': 100
        }

    def test_basic_validation(self, model, validation_data, device):
        """測試基本驗證功能"""
        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse', 'rmse']
        )

        results = strategy.validate(model, device)

        assert 'mse' in results
        assert 'rmse' in results
        assert results['mse'] >= 0
        assert results['rmse'] >= 0

    def test_validation_with_normalizer(self, model, validation_data, device):
        """測試帶標準化器的驗證"""
        normalizer = DummyNormalizer()

        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse'],
            data_normalizer=normalizer
        )

        results = strategy.validate(model, device)

        assert 'mse' in results
        assert results['mse'] >= 0

    def test_dimension_mismatch_handling(self, device):
        """測試維度不匹配處理"""
        model = SimpleModel(input_dim=2, output_dim=3)  # 3個輸出
        validation_data = {
            'coords': torch.randn(50, 2),
            'targets': torch.randn(50, 4),  # 4個目標
            'size': 50
        }

        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse']
        )

        # 應該正常處理，比較前3個分量
        results = strategy.validate(model, device)

        assert 'mse' in results
        assert results['mse'] >= 0

    def test_empty_validation_data(self, model, device):
        """測試空驗證數據"""
        validation_data = {
            'coords': torch.tensor([]),
            'targets': torch.tensor([]),
            'size': 0
        }

        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse']
        )

        results = strategy.validate(model, device)

        # 空數據應返回空結果
        assert results == {}

    def test_all_metrics(self, model, validation_data, device):
        """測試所有支持的指標"""
        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse', 'rmse', 'mae', 'relative_l2']
        )

        results = strategy.validate(model, device)

        assert len(results) == 4
        assert all(metric in results for metric in ['mse', 'rmse', 'mae', 'relative_l2'])
        assert all(value >= 0 for value in results.values())

    def test_invalid_metric(self, model, validation_data, device):
        """測試無效的指標名稱"""
        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['invalid_metric']
        )

        with pytest.raises(ValueError, match="Unknown metric"):
            strategy.validate(model, device)


class TestPhysicsBasedValidation:
    """測試 PhysicsBasedValidation"""

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    @pytest.fixture
    def model(self):
        return SimpleModel(input_dim=2, output_dim=4)

    @pytest.fixture
    def physics_validator(self):
        return DummyPhysicsValidator()

    @pytest.fixture
    def validation_coords(self):
        return torch.randn(100, 2)

    def test_basic_physics_validation(self, model, physics_validator, device):
        """測試基本物理驗證"""
        strategy = PhysicsBasedValidation(
            physics_validator=physics_validator
        )

        validation_data = {
            'coords': torch.randn(100, 2)
        }

        results = strategy.validate(
            model, device,
            validation_data=validation_data
        )

        assert 'physics_continuity' in results
        assert 'physics_boundary' in results
        assert 'physics_momentum' in results

    def test_3d_coords_preparation(self, model, physics_validator, device):
        """測試 3D 坐標準備"""
        strategy = PhysicsBasedValidation(
            physics_validator=physics_validator
        )

        validation_data = {
            'coords': torch.randn(100, 3)  # 3D 坐標
        }

        results = strategy.validate(
            model, device,
            validation_data=validation_data,
            is_3d=True
        )

        assert len(results) > 0

    def test_empty_validation_data(self, model, physics_validator, device):
        """測試空驗證數據"""
        strategy = PhysicsBasedValidation(
            physics_validator=physics_validator
        )

        # 未提供 validation_data
        results = strategy.validate(model, device)

        assert results == {}


class TestValidationManager:
    """測試 ValidationManager"""

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    @pytest.fixture
    def model(self):
        return SimpleModel(input_dim=2, output_dim=4)

    @pytest.fixture
    def validation_data(self):
        return {
            'coords': torch.randn(100, 2),
            'targets': torch.randn(100, 4),
            'size': 100
        }

    def test_empty_manager(self, model, device):
        """測試空的 ValidationManager"""
        manager = ValidationManager()

        results = manager.validate(model, device)

        assert results == {}
        assert not manager.has_strategies()

    def test_add_single_strategy(self, model, validation_data, device):
        """測試添加單個策略"""
        manager = ValidationManager()

        strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse']
        )
        manager.add_strategy(strategy)

        assert manager.has_strategies()

        results = manager.validate(model, device)

        assert 'mse' in results

    def test_multiple_strategies(self, model, validation_data, device):
        """測試多個策略組合"""
        manager = ValidationManager()

        # 添加數據驗證策略
        data_strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse', 'rmse']
        )
        manager.add_strategy(data_strategy)

        # 添加物理驗證策略
        physics_validator = DummyPhysicsValidator()
        physics_strategy = PhysicsBasedValidation(
            physics_validator=physics_validator
        )
        manager.add_strategy(physics_strategy)

        # 執行驗證
        results = manager.validate(
            model, device,
            validation_data=validation_data
        )

        # 應該包含兩種策略的結果
        assert 'mse' in results
        assert 'rmse' in results
        assert 'physics_continuity' in results
        assert 'physics_boundary' in results

    def test_strategy_failure_handling(self, model, device):
        """測試策略失敗處理"""
        manager = ValidationManager()

        # 添加一個會失敗的策略
        class FailingStrategy(ValidationStrategy):
            def validate(self, model, device, **kwargs):
                raise RuntimeError("Intentional failure")

        manager.add_strategy(FailingStrategy())

        # 添加一個正常的策略
        validation_data = {
            'coords': torch.randn(50, 2),
            'targets': torch.randn(50, 4),
            'size': 50
        }
        manager.add_strategy(DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse']
        ))

        # 即使一個策略失敗，其他策略應該正常執行
        results = manager.validate(model, device)

        assert 'mse' in results

    def test_initialization_with_strategies(self, model, validation_data, device):
        """測試使用策略列表初始化"""
        strategy1 = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse']
        )

        strategy2 = DataBasedValidation(
            validation_data=validation_data,
            metrics=['rmse']
        )

        manager = ValidationManager(strategies=[strategy1, strategy2])

        assert manager.has_strategies()

        results = manager.validate(model, device)

        assert 'mse' in results
        assert 'rmse' in results


class TestValidationIntegration:
    """集成測試"""

    def test_full_workflow(self):
        """測試完整驗證工作流"""
        device = torch.device('cpu')
        model = SimpleModel(input_dim=2, output_dim=4)

        # 創建驗證數據
        validation_data = {
            'coords': torch.randn(200, 2),
            'targets': torch.randn(200, 4),
            'size': 200
        }

        # 創建驗證管理器
        manager = ValidationManager()

        # 添加數據驗證
        data_strategy = DataBasedValidation(
            validation_data=validation_data,
            metrics=['mse', 'rmse', 'relative_l2'],
            data_normalizer=DummyNormalizer()
        )
        manager.add_strategy(data_strategy)

        # 添加物理驗證
        physics_strategy = PhysicsBasedValidation(
            physics_validator=DummyPhysicsValidator()
        )
        manager.add_strategy(physics_strategy)

        # 執行驗證
        results = manager.validate(
            model, device,
            validation_data=validation_data
        )

        # 驗證結果
        assert len(results) == 6  # 3 data metrics + 3 physics metrics
        assert all(isinstance(v, float) for v in results.values())

        print("✅ 完整驗證工作流測試通過")
        print(f"   驗證結果: {results}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
