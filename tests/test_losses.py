"""
損失函數模組測試
測試權重模組
"""

import torch
import pytest
import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from pinnx.losses.weighting import (
    GradNormWeighter,
    NTKWeighter,
    AdaptiveWeightScheduler,
    CausalWeighter,
    MultiWeightManager,
)


class TestWeightingStrategies:
    """測試權重策略"""

    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)

        # 模擬簡單模型
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.Tanh(),
            torch.nn.Linear(10, 3)
        ).to(self.device)

    def test_gradnorm_weighting(self):
        """測試 GradNorm 權重策略"""
        loss_names = ['data', 'pde', 'boundary']
        gradnorm = GradNormWeighter(
            self.model, loss_names,
            alpha=0.9, target_ratios=[1.0, 1.0, 0.5], update_frequency=1
        )

        # 模擬損失
        x = torch.randn(16, 2, device=self.device)
        output = self.model(x)
        losses = {
            'data': (output[:, 0] ** 2).mean(),
            'pde': (output[:, 1] - 1.0).pow(2).mean(),
            'boundary': output[:, 2].abs().mean()
        }

        weights = gradnorm.update_weights(losses, sum(losses.values()))

        assert len(weights) == len(loss_names)
        assert all(w > 0 for w in weights.values())
        assert all(loss_name in weights for loss_name in loss_names)

    def test_gradnorm_weight_sum_and_ratio(self):
        """GradNorm 權重應維持總和恆定且比例受限"""
        loss_names = ['data', 'pde', 'boundary']
        gradnorm = GradNormWeighter(
            self.model,
            loss_names,
            alpha=0.5,
            update_frequency=1,
            max_ratio=20.0
        )

        initial_sum = sum(gradnorm.initial_weight_values.values())

        x = torch.randn(24, 2, device=self.device)
        output = self.model(x)

        losses = {
            'data': (output[:, 0] ** 2).mean(),
            'pde': (output[:, 1] - 2.0).pow(2).mean(),
            'boundary': torch.relu(output[:, 2]).mean()
        }

        weights = gradnorm.update_weights(losses, sum(losses.values()))

        total_weight = sum(weights.values())
        assert pytest.approx(total_weight, rel=1e-3) == initial_sum

        max_w = max(weights.values())
        min_w = min(weights.values())
        assert max_w / max(min_w, 1e-12) <= gradnorm.max_ratio + 1e-6

    def test_ntk_weighting(self):
        """測試 NTK 權重策略"""
        ntk = NTKWeighter(self.model, update_freq=100, reg_param=1e-6)

        # 模擬訓練點
        x_train = torch.randn(20, 2, device=self.device)

        # 模擬損失
        losses = {
            'data': torch.tensor(0.8, device=self.device),
            'pde': torch.tensor(1.5, device=self.device)
        }

        # 使用統一的接口：context 字典包含所有可選參數
        context = {'x_train': x_train, 'step': 100}
        weights = ntk.update_weights(losses, context)

        assert len(weights) == len(losses)
        assert all(w > 0 for w in weights.values())

    def test_adaptive_weighting(self):
        """測試自適應權重策略"""
        adaptive = AdaptiveWeightScheduler(
            loss_names=['residual', 'data', 'prior'],
            adaptation_method='exponential',
            adaptation_rate=0.1
        )

        # 模擬訓練歷史
        for step in range(10):
            losses = {
                'residual': torch.tensor(1.0 - step * 0.05, device=self.device),
                'data': torch.tensor(0.5, device=self.device),
                'prior': torch.tensor(0.3 + step * 0.02, device=self.device)
            }

            # 使用統一的接口：context 字典
            context = {'step': step, 'total_steps': 100}
            weights = adaptive.update_weights(losses, context)

            assert len(weights) == 3
            assert all(w > 0 for w in weights.values())

    def test_causal_weighting(self):
        """測試因果權重策略"""
        causal = CausalWeighter(
            epsilon=1.0,
            num_chunks=5,
            t_min=0.0,
            t_max=4.0,
            device=str(self.device)
        )

        # 模擬時間序列損失（已平方）
        time_losses = torch.tensor(
            [1.0, 0.8, 0.6, 0.9, 0.7], device=self.device
        ).unsqueeze(1)
        time_points = torch.arange(len(time_losses), device=self.device).float().unsqueeze(1)

        # 使用統一的 PointWeighter 接口
        context = {'return_pointwise': True}
        weights = causal.compute_weights(time_losses, time_points, context)

        assert weights.shape == time_losses.shape
        assert torch.all(weights > 0)

        # 早期時間點通常有更高權重
        assert weights[0] >= weights[-1]

    def test_multiobjective_weighting(self):
        """測試多目標權重策略"""
        objectives = ['accuracy', 'smoothness', 'physics']
        moo = MultiWeightManager(
            objectives, method='pareto',
            preference_weights=[0.5, 0.3, 0.2]
        )

        # 模擬目標值
        objective_values = {
            'accuracy': torch.tensor(0.1, device=self.device),
            'smoothness': torch.tensor(0.05, device=self.device),
            'physics': torch.tensor(0.8, device=self.device)
        }

        weights = moo.update_weights(objective_values)

        assert len(weights) == len(objectives)
        assert all(w > 0 for w in weights.values())

        # 權重應該反映偏好
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}

        # 物理項權重較高 (因為其值較大，需要平衡)
        assert normalized_weights['physics'] > normalized_weights['accuracy']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
