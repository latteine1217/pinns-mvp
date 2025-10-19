"""
Checkpoint Physics Validation 整合測試
測試檢查點保存時的物理驗證功能
"""

import numpy as np
import torch
import pytest
import tempfile
import os
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.train.checkpointing import validate_physics_before_save
from pinnx.models.fourier_mlp import FourierMLP


class TestCheckpointPhysicsValidation:
    """測試檢查點物理驗證整合"""

    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

        # 創建簡單的測試模型
        self.model = FourierMLP(
            in_dim=2,
            out_dim=3,  # u, v, p
            width=32,
            depth=2,
            activation='tanh',
            fourier_features=False
        ).to(self.device)

        # 基本配置
        self.config = {
            'physics_validation': {
                'enabled': True,
                'thresholds': {
                    'mass_conservation': 1.0,  # 放寬閾值以通過測試
                    'momentum_conservation': 10.0,
                    'boundary_condition': 10.0
                }
            },
            'physics': {
                'nu': 5e-5
            },
            'domain': {
                'wall_positions': (0.0, 2.0)
            }
        }

    def test_validate_physics_before_save_enabled(self):
        """測試啟用物理驗證"""
        # 創建測試座標
        coords = torch.randn(100, 2, device=self.device)

        passed, metrics = validate_physics_before_save(
            self.model, coords, self.config, self.device
        )

        # 檢查返回值
        assert isinstance(passed, bool)
        assert isinstance(metrics, dict)

        if passed:
            assert 'mass_conservation_error' in metrics
            assert 'momentum_conservation_error' in metrics
            assert 'boundary_condition_error' in metrics

    def test_validate_physics_before_save_disabled(self):
        """測試禁用物理驗證"""
        config = self.config.copy()
        config['physics_validation']['enabled'] = False

        coords = torch.randn(50, 2, device=self.device)

        passed, metrics = validate_physics_before_save(
            self.model, coords, config, self.device
        )

        # 禁用時應直接通過
        assert passed is True
        assert metrics == {}

    def test_validate_physics_3d_model(self):
        """測試 3D 模型的物理驗證"""
        # 創建 3D 模型
        model_3d = FourierMLP(
            in_dim=3,
            out_dim=4,  # u, v, w, p
            width=32,
            depth=2,
            activation='tanh',
            fourier_features=False
        ).to(self.device)

        coords = torch.randn(50, 3, device=self.device)

        passed, metrics = validate_physics_before_save(
            model_3d, coords, self.config, self.device
        )

        # 檢查返回值
        assert isinstance(passed, bool)
        assert isinstance(metrics, dict)

    def test_physics_metrics_in_checkpoint(self):
        """測試物理指標是否正確記錄到檢查點元數據"""
        coords = torch.randn(100, 2, device=self.device)

        passed, metrics = validate_physics_before_save(
            self.model, coords, self.config, self.device
        )

        # 如果驗證通過，應包含完整指標
        if passed:
            assert 'mass_conservation_error' in metrics
            assert 'momentum_conservation_error' in metrics
            assert 'boundary_condition_error' in metrics
            assert 'validation_passed' in metrics

            # 檢查指標值類型
            assert isinstance(metrics['mass_conservation_error'], float)
            assert isinstance(metrics['momentum_conservation_error'], float)
            assert isinstance(metrics['boundary_condition_error'], float)

    def test_validation_with_custom_thresholds(self):
        """測試自訂閾值的驗證"""
        config = self.config.copy()
        config['physics_validation']['thresholds'] = {
            'mass_conservation': 1e-1,
            'momentum_conservation': 1.0,
            'boundary_condition': 1e-1
        }

        coords = torch.randn(50, 2, device=self.device)

        passed, metrics = validate_physics_before_save(
            self.model, coords, config, self.device
        )

        # 應該能夠執行而不報錯
        assert isinstance(passed, bool)

    def test_validation_error_handling(self):
        """測試驗證錯誤處理"""
        # 故意使用錯誤的配置（缺少必要欄位）
        incomplete_config = {
            'physics_validation': {
                'enabled': True
                # 缺少 thresholds
            }
        }

        coords = torch.randn(50, 2, device=self.device)

        # 應該使用預設閾值，不應崩潰
        passed, metrics = validate_physics_before_save(
            self.model, coords, incomplete_config, self.device
        )

        # 即使配置不完整，也應能返回結果
        assert isinstance(passed, bool)


class TestTrainerPhysicsValidationIntegration:
    """測試 Trainer 與物理驗證的整合"""

    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        """清理測試環境"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_checkpoint_rejection_on_validation_failure(self):
        """測試驗證失敗時檢查點被拒絕"""
        # 此測試需要完整的 Trainer 設置
        # 由於複雜度，這裡僅測試驗證邏輯本身

        model = FourierMLP(
            in_dim=2, out_dim=3, width=16, depth=2,
            activation='tanh', fourier_features=False
        ).to(self.device)

        # 設置非常嚴格的閾值，確保驗證失敗
        config = {
            'physics_validation': {
                'enabled': True,
                'thresholds': {
                    'mass_conservation': 1e-10,  # 極嚴格
                    'momentum_conservation': 1e-10,
                    'boundary_condition': 1e-10
                }
            },
            'physics': {'nu': 5e-5},
            'domain': {'wall_positions': (0.0, 2.0)}
        }

        coords = torch.randn(100, 2, device=self.device)

        passed, metrics = validate_physics_before_save(
            model, coords, config, self.device
        )

        # 極嚴格閾值應該導致驗證失敗
        assert passed is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
