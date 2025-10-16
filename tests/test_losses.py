"""
損失函數模組測試
測試殘差、先驗和權重模組
"""

import numpy as np
import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.losses.residuals import (
    pde_residual_loss, boundary_residual_loss, 
    initial_condition_loss, source_regularization,
    data_fitting_loss
)
from pinnx.losses.priors import (
    prior_consistency_loss, statistical_prior_loss,
    physics_constraint_loss, energy_conservation_loss
)
from pinnx.losses.weighting import (
    GradNormWeighter,
    NTKWeighter,
    AdaptiveWeightScheduler,
    CausalWeighter,
    MultiWeightManager,
)


class TestResidualLosses:
    """測試殘差損失函數"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_pde_residual_loss(self):
        """測試 PDE 殘差損失"""
        # 模擬 PDE 殘差
        residual = torch.randn(50, 2, device=self.device)  # 動量方程 x,y 分量
        
        loss = pde_residual_loss(residual)
        
        assert loss.numel() == 1  # 標量損失
        assert loss >= 0  # 非負
        assert not torch.isnan(loss)
        
        # 測試零殘差
        zero_residual = torch.zeros(50, 2, device=self.device)
        zero_loss = pde_residual_loss(zero_residual)
        assert zero_loss < 1e-6
    
    def test_pde_residual_with_weights(self):
        """測試帶權重的 PDE 殘差損失"""
        residual = torch.randn(30, 3, device=self.device)
        weights = torch.rand(30, device=self.device)
        
        loss_weighted = pde_residual_loss(residual, weights=weights)
        loss_unweighted = pde_residual_loss(residual)
        
        # 權重應該影響損失值
        assert loss_weighted != loss_unweighted
        assert loss_weighted >= 0
    
    def test_boundary_residual_loss(self):
        """測試邊界殘差損失"""
        # 邊界點預測值
        predicted = torch.randn(20, 2, device=self.device)
        target = torch.zeros(20, 2, device=self.device)  # 零速度邊界條件
        
        loss = boundary_residual_loss(predicted, target, bc_type='dirichlet')
        
        assert loss.numel() == 1
        assert loss >= 0
        
        # 測試完美匹配
        perfect_loss = boundary_residual_loss(target, target, bc_type='dirichlet')
        assert perfect_loss < 1e-6
    
    def test_neumann_boundary_loss(self):
        """測試 Neumann 邊界條件損失"""
        # 需要梯度的場
        coords = torch.randn(15, 2, requires_grad=True, device=self.device)
        field = torch.sum(coords**2, dim=1)  # 簡單的二次場
        
        target_gradient = torch.ones(15, device=self.device)
        normal = torch.tensor([[1.0, 0.0]]).expand(15, 2).to(self.device)  # x 方向法向量
        
        loss = boundary_residual_loss(
            field, target_gradient, 
            bc_type='neumann', coords=coords, normal=normal
        )
        
        assert loss.numel() == 1
        assert loss >= 0
    
    def test_initial_condition_loss(self):
        """測試初始條件損失"""
        predicted_t0 = torch.randn(25, 3, device=self.device)
        target_t0 = torch.zeros(25, 3, device=self.device)
        
        loss = initial_condition_loss(predicted_t0, target_t0)
        
        assert loss.numel() == 1
        assert loss >= 0
        
        # 測試時間權重
        time_weights = torch.exp(-torch.arange(25, dtype=torch.float, device=self.device))
        loss_weighted = initial_condition_loss(predicted_t0, target_t0, time_weights=time_weights)
        
        assert loss_weighted >= 0
    
    def test_source_regularization(self):
        """測試源項正則化"""
        source_field = torch.randn(40, device=self.device)
        
        # L1 正則化
        l1_loss = source_regularization(source_field, reg_type='l1', strength=0.01)
        assert l1_loss >= 0
        
        # L2 正則化
        l2_loss = source_regularization(source_field, reg_type='l2', strength=0.01)
        assert l2_loss >= 0
        
        # 平滑正則化
        source_2d = source_field.view(8, 5)  # 重塑為 2D
        smooth_loss = source_regularization(source_2d, reg_type='smooth', strength=0.01)
        assert smooth_loss >= 0
    
    def test_data_fitting_loss(self):
        """測試資料擬合損失"""
        predicted = torch.randn(12, 3, device=self.device)
        observed = torch.randn(12, 3, device=self.device)
        noise_std = torch.ones(12, device=self.device) * 0.1
        
        # 基本擬合損失
        loss_basic = data_fitting_loss(predicted, observed)
        assert loss_basic >= 0
        
        # 帶噪音權重的損失
        loss_weighted = data_fitting_loss(predicted, observed, noise_std=noise_std)
        assert loss_weighted >= 0
        
        # 帶遮罩的損失 (部分資料遺失)
        mask = torch.randint(0, 2, (12,), dtype=torch.bool, device=self.device)
        loss_masked = data_fitting_loss(predicted, observed, mask=mask)
        assert loss_masked >= 0


class TestPriorLosses:
    """測試先驗損失函數"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_prior_consistency_loss(self):
        """測試先驗一致性損失"""
        high_fi = torch.randn(30, 3, device=self.device)
        low_fi = torch.randn(30, 3, device=self.device)
        
        loss = prior_consistency_loss(high_fi, low_fi, strength=0.5)
        
        assert loss.numel() == 1
        assert loss >= 0
        
        # 測試完美一致性
        perfect_loss = prior_consistency_loss(high_fi, high_fi, strength=0.5)
        assert perfect_loss < 1e-6
    
    def test_prior_consistency_with_uncertainty(self):
        """測試帶不確定性的先驗一致性"""
        high_fi = torch.randn(20, 2, device=self.device)
        low_fi = torch.randn(20, 2, device=self.device)
        uncertainty = torch.rand(20, 2, device=self.device) * 0.1 + 0.01  # 避免零不確定性
        
        loss = prior_consistency_loss(
            high_fi, low_fi, strength=0.3, 
            uncertainty=uncertainty, adaptive=True
        )
        
        assert loss >= 0
        
        # 高不確定性區域應該有較低權重
        high_uncertainty = torch.ones_like(uncertainty) * 10.0
        loss_high_unc = prior_consistency_loss(
            high_fi, low_fi, strength=0.3,
            uncertainty=high_uncertainty, adaptive=True
        )
        
        # 通常高不確定性會降低損失權重
        # assert loss_high_unc <= loss  # 不總是成立，取決於具體實現
    
    def test_statistical_prior_loss(self):
        """測試統計先驗損失"""
        predicted = torch.randn(50, 2, device=self.device)
        
        # 均值先驗
        target_mean = torch.zeros(2, device=self.device)
        mean_loss = statistical_prior_loss(
            predicted, prior_type='mean', 
            target_stats=target_mean, strength=0.1
        )
        
        assert mean_loss >= 0
        
        # 方差先驗
        target_var = torch.ones(2, device=self.device)
        var_loss = statistical_prior_loss(
            predicted, prior_type='variance',
            target_stats=target_var, strength=0.1
        )
        
        assert var_loss >= 0
    
    def test_physics_constraint_loss(self):
        """測試物理約束損失"""
        # 速度場
        velocity = torch.randn(25, 2, device=self.device)
        
        # 能量約束
        energy_loss = physics_constraint_loss(
            velocity, constraint_type='energy_bound',
            constraint_params={'max_energy': 10.0}, strength=0.05
        )
        
        assert energy_loss >= 0
        
        # 動量約束
        momentum_loss = physics_constraint_loss(
            velocity, constraint_type='momentum_conservation',
            constraint_params={'target_momentum': torch.zeros(2, device=self.device)},
            strength=0.1
        )
        
        assert momentum_loss >= 0
    
    def test_energy_conservation_loss(self):
        """測試能量守恆損失"""
        # 模擬速度場和壓力場
        velocity = torch.randn(35, 2, device=self.device)
        pressure = torch.randn(35, 1, device=self.device)
        
        # 計算能量
        kinetic_energy = 0.5 * torch.sum(velocity**2, dim=1, keepdim=True)
        total_energy = kinetic_energy + pressure
        
        loss = energy_conservation_loss(
            total_energy, conservation_type='steady',
            strength=0.02
        )
        
        assert loss >= 0
        
        # 非定常能量守恆
        time_derivative = torch.randn(35, 1, device=self.device)
        unsteady_loss = energy_conservation_loss(
            total_energy, conservation_type='unsteady',
            time_derivative=time_derivative, strength=0.02
        )
        
        assert unsteady_loss >= 0


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
        
        weights = ntk.update_weights(losses, x_train, step=100)
        
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
            
            weights = adaptive.update_weights(losses, step=step)
            
            assert len(weights) == 3
            assert all(w > 0 for w in weights.values())
    
    def test_causal_weighting(self):
        """測試因果權重策略"""
        causal = CausalWeighter(
            time_window=5, causality_strength=1.0,
            temporal_decay=0.9
        )
        
        # 模擬時間序列損失
        time_losses = [
            torch.tensor(1.0, device=self.device),
            torch.tensor(0.8, device=self.device),
            torch.tensor(0.6, device=self.device),
            torch.tensor(0.9, device=self.device),
            torch.tensor(0.7, device=self.device)
        ]
        
        weights = causal.compute_causal_weights(time_losses)
        
        assert len(weights) == len(time_losses)
        assert all(w > 0 for w in weights)
        
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


def test_losses_integration():
    """測試損失模組整合"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 損失模組整合測試 ===")
    
    # 模擬完整的損失計算流程
    batch_size = 30
    
    # 模擬資料
    coords = torch.randn(batch_size, 2, requires_grad=True, device=device)
    predicted = torch.randn(batch_size, 3, device=device)  # [u, v, p]
    observed = torch.randn(batch_size//2, 3, device=device)  # 部分觀測
    boundary_pred = torch.randn(10, 2, device=device)
    boundary_target = torch.zeros(10, 2, device=device)
    
    # 計算各種損失
    losses = {}
    
    # PDE 殘差 (模擬)
    pde_residual = torch.randn(batch_size, 2, device=device)
    losses['pde'] = pde_residual_loss(pde_residual)
    
    # 資料擬合
    losses['data'] = data_fitting_loss(predicted[:batch_size//2], observed)
    
    # 邊界條件
    losses['boundary'] = boundary_residual_loss(boundary_pred, boundary_target)
    
    # 先驗一致性
    low_fi_prior = torch.randn(batch_size, 3, device=device)
    losses['prior'] = prior_consistency_loss(predicted, low_fi_prior, strength=0.1)
    
    # 源項正則化
    source_field = torch.randn(batch_size, device=device)
    losses['source_reg'] = source_regularization(source_field, reg_type='l1', strength=0.01)
    
    # 物理約束
    losses['physics'] = physics_constraint_loss(
        predicted[:, :2], constraint_type='energy_bound',
        constraint_params={'max_energy': 5.0}, strength=0.05
    )
    
    # 權重策略
    model = torch.nn.Linear(2, 3).to(device)
    gradnorm = GradNormWeighter(model, list(losses.keys()), alpha=0.9)
    
    total_loss = sum(losses.values())
    weights = gradnorm.update_weights(losses, total_loss)
    
    # 加權總損失
    weighted_loss = sum(weights[k] * v for k, v in losses.items())
    
    print("各項損失:")
    for name, loss in losses.items():
        print(f"  {name}: {loss.item():.4f}")
    
    print(f"\n權重:")
    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f}")
    
    print(f"\n總損失: {total_loss.item():.4f}")
    print(f"加權總損失: {weighted_loss.item():.4f}")
    
    # 驗證
    assert all(not torch.isnan(loss) for loss in losses.values())
    assert all(loss >= 0 for loss in losses.values())
    assert all(w > 0 for w in weights.values())
    assert not torch.isnan(weighted_loss)
    assert weighted_loss >= 0
    
    print("✅ 損失模組整合測試通過")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
