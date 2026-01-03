"""
指標追蹤工具測試

測試 Timer、MemoryTracker 和 PhysicsValidator 的核心功能
"""

import torch
import torch.nn as nn
import time
import numpy as np
import pytest
from typing import Dict

from pinnx.utils.timer import Timer
from pinnx.utils.memory_tracker import MemoryTracker
from pinnx.utils.physics_validator import PhysicsValidator


class TestTimer:
    """Timer 測試"""

    def test_timer_basic(self):
        """測試基本計時功能"""
        timer = Timer(use_wandb=False)

        # 開始計時
        timer.start()
        assert timer.start_time is not None

        # 模擬訓練
        time.sleep(0.1)
        epoch_time = timer.tick(epoch=0, log_to_wandb=False)

        assert epoch_time > 0
        assert len(timer.epoch_times) == 1

        # 停止計時
        timer.stop()
        assert timer.stop_time is not None

        # 檢查摘要
        summary = timer.get_summary()
        assert 'total_wall_time' in summary
        assert 'avg_epoch_time' in summary
        assert summary['total_wall_time'] > 0

    def test_timer_convergence(self):
        """測試達標時間追蹤"""
        timer = Timer(use_wandb=False)
        timer.start()

        # 模擬訓練直到達標
        for epoch in range(5):
            time.sleep(0.05)
            timer.tick(epoch=epoch, log_to_wandb=False)

            if epoch == 3:
                timer.mark_convergence(
                    epoch=epoch,
                    metric_value=0.05,
                    log_to_wandb=False
                )

        timer.stop()

        # 檢查達標記錄
        assert timer.convergence_time is not None
        assert timer.convergence_epoch == 3
        assert timer.convergence_value == 0.05

        summary = timer.get_summary()
        assert 'time_to_convergence' in summary
        assert 'convergence_epoch' in summary

    def test_timer_reset(self):
        """測試重置功能"""
        timer = Timer(use_wandb=False)
        timer.start()
        timer.tick(log_to_wandb=False)
        timer.stop()

        # 重置
        timer.reset()

        assert timer.start_time is None
        assert timer.stop_time is None
        assert len(timer.epoch_times) == 0


class TestMemoryTracker:
    """MemoryTracker 測試"""

    def test_memory_tracker_basic(self):
        """測試基本記憶體追蹤"""
        # 建立簡單模型
        model = nn.Sequential(
            nn.Linear(10, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        tracker = MemoryTracker(model=model, use_wandb=False)

        # 檢查模型統計
        assert tracker.model_params > 0
        assert tracker.model_size_mb > 0

    def test_memory_tracker_get_current(self):
        """測試當前記憶體獲取"""
        tracker = MemoryTracker(use_wandb=False)

        current_mb, peak_mb = tracker.get_current_memory()

        # 即使 CUDA 不可用，也應該返回 0
        assert current_mb >= 0
        assert peak_mb >= 0

    def test_memory_tracker_log_current(self):
        """測試記憶體記錄"""
        model = nn.Linear(100, 100)
        tracker = MemoryTracker(model=model, use_wandb=False)

        stats = tracker.log_current(epoch=0, log_to_wandb=False)

        assert 'current_memory_mb' in stats
        assert 'peak_memory_mb' in stats

    def test_memory_tracker_estimate_activation(self):
        """測試激活值記憶體估算"""
        tracker = MemoryTracker(use_wandb=False)

        activation_mb = tracker.estimate_activation_memory(
            batch_size=1000,
            input_dim=3,
            output_dim=4,
            depth=6,
            width=256
        )

        assert activation_mb > 0

    def test_memory_tracker_summary(self):
        """測試摘要生成"""
        model = nn.Linear(10, 10)
        tracker = MemoryTracker(model=model, use_wandb=False)

        # 記錄一些數據
        tracker.log_current(log_to_wandb=False)
        tracker.log_current(log_to_wandb=False)

        summary = tracker.get_summary()

        assert 'current_memory_mb' in summary
        assert 'peak_memory_mb' in summary
        assert 'model_parameters' in summary
        assert 'avg_memory_mb' in summary


class TestPhysicsValidator:
    """PhysicsValidator 測試"""

    def test_physics_validator_init(self):
        """測試初始化"""
        config = {
            'physics': {'nu': 0.01, 'rho': 1.0},
            'dimension': 2,
        }

        validator = PhysicsValidator(config=config, use_wandb=False)

        assert validator.nu == 0.01
        assert validator.rho == 1.0
        assert validator.dimension == 2

    def test_compute_divergence_2d(self):
        """測試 2D 散度計算"""
        validator = PhysicsValidator(dimension=2, use_wandb=False)

        # 建立測試速度場（無散度場：u = sin(x)cos(y), v = -cos(x)sin(y)）
        n = 20
        x = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)
        y = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)

        X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
        x_flat = X.reshape(-1, 1).requires_grad_(True)
        y_flat = Y.reshape(-1, 1).requires_grad_(True)

        u = torch.sin(x_flat) * torch.cos(y_flat)
        v = -torch.cos(x_flat) * torch.sin(y_flat)

        u = u.requires_grad_(True)
        v = v.requires_grad_(True)

        div_results = validator.compute_divergence(u, v, None, x_flat, y_flat, None)

        # 檢查結果
        assert 'divergence_mean' in div_results
        assert 'divergence_max' in div_results
        assert 'divergence_l2' in div_results
        assert 'divergence_relative' in div_results

        # 無散度場應該散度很小
        assert div_results['divergence_l2'] < 0.1

    def test_compute_momentum_residual_2d(self):
        """測試 2D 動量殘差計算"""
        validator = PhysicsValidator(dimension=2, nu=0.01, rho=1.0, use_wandb=False)

        # 建立簡單的速度場和壓力場（通過計算圖）
        n = 20
        x = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)
        y = torch.linspace(0, 2*np.pi, n).reshape(-1, 1).requires_grad_(True)

        X, Y = torch.meshgrid(x.squeeze(), y.squeeze(), indexing='ij')
        x_flat = X.reshape(-1, 1).requires_grad_(True)
        y_flat = Y.reshape(-1, 1).requires_grad_(True)

        # 通過計算產生 u, v, p（保持計算圖）
        u = torch.sin(x_flat) * torch.cos(y_flat)
        v = torch.cos(x_flat) * torch.sin(y_flat)
        p = torch.sin(x_flat) * torch.sin(y_flat)

        mom_results = validator.compute_momentum_residual(
            u, v, None, p, x_flat, y_flat, None
        )

        # 檢查結果
        assert 'momentum_residual_mean' in mom_results
        assert 'momentum_residual_max' in mom_results
        assert mom_results['momentum_residual_mean'] >= 0

    def test_check_boundary_conditions_periodic(self):
        """測試週期邊界條件檢查"""
        validator = PhysicsValidator(dimension=2, use_wandb=False)

        # 建立週期性場
        n = 20
        x = torch.linspace(0, 2*np.pi, n)
        y = torch.linspace(0, 2*np.pi, n)

        X, Y = torch.meshgrid(x, y, indexing='ij')

        # 完美週期場：sin(x)
        u = torch.sin(X).unsqueeze(0)  # [1, n, n]

        bc_results = validator.check_boundary_conditions(u, bc_type='periodic')

        # 檢查結果
        assert 'bc/periodicity_x_error' in bc_results
        assert 'bc/periodicity_y_error' in bc_results

        # sin(0) = sin(2π) = 0，誤差應該很小
        assert bc_results['bc/periodicity_x_error'] < 1e-5

    def test_check_boundary_conditions_no_slip(self):
        """測試無滑移邊界條件檢查"""
        validator = PhysicsValidator(dimension=2, use_wandb=False)

        # 建立場（壁面應為 0）
        n = 20
        u = torch.randn(1, n, n)

        # 強制壁面為 0
        u[:, :, 0] = 0
        u[:, :, -1] = 0

        bc_results = validator.check_boundary_conditions(u, bc_type='no_slip')

        # 檢查結果
        assert 'bc/wall_velocity_violation_bottom' in bc_results
        assert 'bc/wall_velocity_violation_top' in bc_results

        # 壁面已設為 0，違反應該很小
        assert bc_results['bc/wall_velocity_violation_bottom'] < 1e-5
        assert bc_results['bc/wall_velocity_violation_top'] < 1e-5

    def test_compute_physics_violation_score(self):
        """測試綜合物理違反分數"""
        validator = PhysicsValidator(use_wandb=False)

        div_results = {
            'divergence_l2': 0.01,
        }

        momentum_results = {
            'momentum_residual_mean': 0.05,
        }

        bc_results = {
            'bc/periodicity_max_error': 0.001,
        }

        score = validator.compute_physics_violation_score(
            div_results, momentum_results, bc_results
        )

        assert score > 0
        # 散度貢獻最大（權重 10）
        assert score >= 0.01 * 10

    def test_validate_integration(self):
        """測試綜合驗證功能"""
        # 建立簡單模型
        model = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 3)  # u, v, p
        )

        config = {
            'dimension': 2,
            'physics': {'nu': 0.01, 'rho': 1.0},
            'boundary_type': 'periodic'
        }

        validator = PhysicsValidator(config=config, use_wandb=False)

        # 測試資料
        n = 50
        x = torch.randn(n, 1).requires_grad_(True)
        y = torch.randn(n, 1).requires_grad_(True)

        test_data = {'x': x, 'y': y}

        # 執行驗證
        results = validator.validate(
            model=model,
            test_data=test_data,
            log_to_wandb=False
        )

        # 檢查結果結構
        assert 'mass_conservation' in results
        assert 'momentum_conservation' in results
        assert 'physics_violation_score' in results

        # 檢查質量守恆結果
        assert 'divergence_mean' in results['mass_conservation']
        assert 'divergence_l2' in results['mass_conservation']

        # 檢查動量守恆結果
        assert 'momentum_residual_mean' in results['momentum_conservation']


def test_integration_example():
    """測試完整整合範例"""
    # 建立工具
    timer = Timer(use_wandb=False)

    model = nn.Sequential(
        nn.Linear(3, 128),
        nn.Tanh(),
        nn.Linear(128, 4)
    )

    memory_tracker = MemoryTracker(model=model, use_wandb=False)
    physics_validator = PhysicsValidator(dimension=3, use_wandb=False)

    # 模擬訓練
    timer.start()

    for epoch in range(3):
        time.sleep(0.05)
        timer.tick(epoch=epoch, log_to_wandb=False)
        memory_tracker.log_current(epoch=epoch, log_to_wandb=False)

    timer.stop()

    # 獲取摘要
    timing_summary = timer.get_summary()
    memory_summary = memory_tracker.get_summary()

    # 驗證結果
    assert timing_summary['num_epochs'] == 3
    assert timing_summary['total_wall_time'] > 0
    assert memory_summary['model_parameters'] > 0


if __name__ == '__main__':
    # 執行測試
    pytest.main([__file__, '-v'])
