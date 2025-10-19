"""
Physics Validators 單元測試
測試物理驗證功能模組
"""

import numpy as np
import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.physics.validators import (
    compute_divergence,
    validate_mass_conservation,
    validate_momentum_conservation,
    validate_boundary_conditions,
    compute_physics_metrics
)


class TestPhysicsValidators:
    """測試物理驗證器模組"""

    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

    def test_compute_divergence_2d(self):
        """測試 2D 散度計算"""
        # 創建測試場：無散度場 (u = sin(y), v = -sin(x))
        x = torch.linspace(0, 2*np.pi, 20, device=self.device)
        y = torch.linspace(0, 2*np.pi, 20, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')

        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        coords.requires_grad_(True)

        u = torch.sin(coords[:, 1:2])
        v = -torch.sin(coords[:, 0:1])

        # 計算散度（理論上應接近 0）
        div = compute_divergence(coords, u, v)

        assert div.shape == (coords.shape[0], 1)
        assert torch.max(torch.abs(div)).item() < 1e-5  # 數值誤差容忍

    def test_compute_divergence_3d(self):
        """測試 3D 散度計算"""
        # 創建測試場
        coords = torch.randn(50, 3, device=self.device, requires_grad=True)

        # 簡單線性場
        u = coords[:, 0:1]
        v = coords[:, 1:2]
        w = coords[:, 2:3]

        div = compute_divergence(coords, u, v, w)

        assert div.shape == (coords.shape[0], 1)
        assert not torch.isnan(div).any()

    def test_validate_mass_conservation_pass(self):
        """測試質量守恆驗證 - 通過情境"""
        # 創建接近無散度的場
        coords = torch.randn(30, 2, device=self.device, requires_grad=True)
        u = torch.zeros(30, 1, device=self.device)
        v = torch.zeros(30, 1, device=self.device)

        passed, error = validate_mass_conservation(coords, u, v, threshold=1e-2)

        assert passed is True
        assert error < 1e-2

    def test_validate_mass_conservation_fail(self):
        """測試質量守恆驗證 - 失敗情境"""
        # 創建有明顯散度的場
        coords = torch.linspace(0, 1, 30, device=self.device).unsqueeze(1)
        coords = torch.cat([coords, coords], dim=1)
        coords.requires_grad_(True)

        # 發散場: u = x, v = y
        u = coords[:, 0:1] * 10
        v = coords[:, 1:2] * 10

        passed, error = validate_mass_conservation(coords, u, v, threshold=1e-3)

        assert passed is False
        assert error > 1e-3

    def test_validate_momentum_conservation(self):
        """測試動量守恆驗證"""
        coords = torch.randn(40, 2, device=self.device, requires_grad=True)

        # 簡單的線性場
        u = coords[:, 0:1] * 0.1
        v = coords[:, 1:2] * 0.1
        p = torch.zeros(40, 1, device=self.device)

        passed, error = validate_momentum_conservation(
            coords, u, v, p, nu=5e-5, threshold=1.0
        )

        # 檢查返回值類型
        assert isinstance(passed, bool)
        assert isinstance(error, float)
        assert not np.isnan(error)

    def test_validate_boundary_conditions_pass(self):
        """測試邊界條件驗證 - 通過情境"""
        # 創建包含壁面點的場
        n_points = 100
        coords = torch.rand(n_points, 2, device=self.device)

        # 設置壁面點（y = 0 和 y = 2）
        coords[0:10, 1] = 0.0  # 底部壁面
        coords[10:20, 1] = 2.0  # 頂部壁面

        # 壁面速度為零
        u = torch.rand(n_points, 1, device=self.device)
        v = torch.rand(n_points, 1, device=self.device)

        u[0:20] = 0.0  # 壁面無滑移
        v[0:20] = 0.0

        passed, error = validate_boundary_conditions(
            coords, u, v, wall_positions=(0.0, 2.0), threshold=1e-3
        )

        assert passed is True
        assert error < 1e-3

    def test_validate_boundary_conditions_fail(self):
        """測試邊界條件驗證 - 失敗情境"""
        n_points = 50
        coords = torch.rand(n_points, 2, device=self.device)

        # 設置壁面點但速度不為零（違反邊界條件）
        coords[0:10, 1] = 0.0
        u = torch.rand(n_points, 1, device=self.device)  # 壁面速度不為零
        v = torch.rand(n_points, 1, device=self.device)

        passed, error = validate_boundary_conditions(
            coords, u, v, wall_positions=(0.0, 2.0), threshold=1e-3
        )

        assert passed is False
        assert error > 1e-3

    def test_compute_physics_metrics_2d(self):
        """測試整合物理指標計算 - 2D"""
        coords = torch.randn(50, 2, device=self.device, requires_grad=True)

        predictions = {
            'u': torch.randn(50, 1, device=self.device) * 0.1,
            'v': torch.randn(50, 1, device=self.device) * 0.1,
            'p': torch.randn(50, 1, device=self.device) * 0.1
        }

        metrics = compute_physics_metrics(coords, predictions)

        # 檢查返回的指標結構
        assert 'mass_conservation_error' in metrics
        assert 'momentum_conservation_error' in metrics
        assert 'boundary_condition_error' in metrics
        assert 'mass_conservation_passed' in metrics
        assert 'momentum_conservation_passed' in metrics
        assert 'boundary_condition_passed' in metrics
        assert 'validation_passed' in metrics

        # 檢查數值類型
        assert isinstance(metrics['mass_conservation_error'], float)
        assert isinstance(metrics['validation_passed'], bool)

    def test_compute_physics_metrics_3d(self):
        """測試整合物理指標計算 - 3D"""
        coords = torch.randn(50, 3, device=self.device, requires_grad=True)

        predictions = {
            'u': torch.randn(50, 1, device=self.device) * 0.1,
            'v': torch.randn(50, 1, device=self.device) * 0.1,
            'w': torch.randn(50, 1, device=self.device) * 0.1,
            'p': torch.randn(50, 1, device=self.device) * 0.1
        }

        metrics = compute_physics_metrics(coords, predictions)

        assert 'validation_passed' in metrics
        assert isinstance(metrics['validation_passed'], bool)

    def test_compute_physics_metrics_with_custom_thresholds(self):
        """測試自訂閾值的物理指標計算"""
        coords = torch.randn(50, 2, device=self.device, requires_grad=True)

        predictions = {
            'u': torch.zeros(50, 1, device=self.device),
            'v': torch.zeros(50, 1, device=self.device),
            'p': torch.zeros(50, 1, device=self.device)
        }

        custom_thresholds = {
            'mass_conservation': 1e-3,
            'momentum_conservation': 1e-2,
            'boundary_condition': 1e-4
        }

        metrics = compute_physics_metrics(
            coords, predictions, validation_thresholds=custom_thresholds
        )

        # 零場應該通過所有驗證
        assert metrics['mass_conservation_passed'] is True
        assert metrics['validation_passed'] is True


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
