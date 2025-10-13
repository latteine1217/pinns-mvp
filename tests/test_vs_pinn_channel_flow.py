"""
VS-PINN Channel Flow 单元测试
============================

测试 VS-PINN 各向异性缩放求解器的正确性

测试内容:
1. 坐标变换可逆性
2. 链式法则梯度正确性
3. Laplacian 计算
4. 物理残差计算
5. 周期性边界条件
6. Loss 权重补偿

作者: PINNs-MVP 团队
"""

import pytest
import torch
import numpy as np
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow, create_vs_pinn_channel_flow


class TestVSPINNChannelFlow:
    """VS-PINN Channel Flow 求解器测试套件"""
    
    @pytest.fixture
    def solver(self):
        """创建标准 VS-PINN 求解器"""
        return create_vs_pinn_channel_flow(N_y=12.0, N_x=2.0, N_z=2.0)
    
    @pytest.fixture
    def sample_coords(self):
        """生成测试坐标"""
        torch.manual_seed(42)
        return torch.randn(50, 3)
    
    @pytest.fixture
    def sample_predictions(self):
        """生成测试预测值 [u, v, w, p]"""
        torch.manual_seed(42)
        return torch.randn(50, 4)
    
    # ============================================================
    # 测试 1: 坐标变换
    # ============================================================
    
    def test_coordinate_scaling_reversibility(self, solver, sample_coords):
        """测试坐标缩放变换的可逆性"""
        scaled = solver.scale_coordinates(sample_coords)
        reconstructed = solver.inverse_scale_coordinates(scaled)
        
        max_error = torch.max(torch.abs(sample_coords - reconstructed)).item()
        assert max_error < 1e-6, f"坐标变换可逆性误差过大: {max_error:.2e}"
    
    def test_coordinate_scaling_factors(self, solver):
        """测试坐标缩放因子是否正确应用"""
        # 创建简单的测试点
        coords = torch.tensor([[1.0, 1.0, 1.0]])
        scaled = solver.scale_coordinates(coords)
        
        # 检查缩放是否正确
        expected_x = 2.0  # N_x * 1.0
        expected_y = 12.0  # N_y * 1.0
        expected_z = 2.0  # N_z * 1.0
        
        assert torch.allclose(scaled[0, 0], torch.tensor(expected_x), atol=1e-6)
        assert torch.allclose(scaled[0, 1], torch.tensor(expected_y), atol=1e-6)
        assert torch.allclose(scaled[0, 2], torch.tensor(expected_z), atol=1e-6)
    
    # ============================================================
    # 测试 2: 梯度计算（链式法则）
    # ============================================================
    
    def test_first_order_gradients(self, solver):
        """测试一阶梯度链式法则的正确性"""
        # 创建简单的测试函数: f(x,y,z) = x² + y² + z²
        # 期望: ∂f/∂x = 2x, ∂f/∂y = 2y, ∂f/∂z = 2z
        
        coords = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        scaled_coords = solver.scale_coordinates(coords)
        scaled_coords.requires_grad_(True)
        
        # 在缩放空间计算函数值
        X, Y, Z = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        field = X**2 + Y**2 + Z**2
        field.requires_grad_(True)
        
        # 计算梯度（提供物理與縮放座標）
        grads = solver.compute_gradients(field, coords, order=1, scaled_coords=scaled_coords)
        
        # 手动计算期望梯度（考虑链式法则）
        # ∂f/∂X = 2X, ∂f/∂x = N_x · ∂f/∂X = N_x · 2X = N_x · 2(N_x·x)
        N_x, N_y, N_z = 2.0, 12.0, 2.0
        x, y, z = 1.0, 2.0, 3.0
        
        expected_grad_x = N_x * 2 * (N_x * x)
        expected_grad_y = N_y * 2 * (N_y * y)
        expected_grad_z = N_z * 2 * (N_z * z)
        
        assert torch.allclose(grads['x'], torch.tensor([[expected_grad_x]]), atol=1e-4)
        assert torch.allclose(grads['y'], torch.tensor([[expected_grad_y]]), atol=1e-4)
        assert torch.allclose(grads['z'], torch.tensor([[expected_grad_z]]), atol=1e-4)
    
    def test_second_order_gradients(self, solver):
        """测试二阶梯度链式法则的正确性"""
        coords = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        scaled_coords = solver.scale_coordinates(coords)
        scaled_coords.requires_grad_(True)
        
        # 函数: f(X,Y,Z) = X² + Y² + Z²
        # ∂²f/∂X² = 2, ∂²f/∂x² = N_x² · 2
        X, Y, Z = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        field = X**2 + Y**2 + Z**2
        field.requires_grad_(True)
        
        second_derivs = solver.compute_gradients(field, coords, order=2, scaled_coords=scaled_coords)
        
        N_x, N_y, N_z = 2.0, 12.0, 2.0
        expected_xx = N_x**2 * 2
        expected_yy = N_y**2 * 2
        expected_zz = N_z**2 * 2
        
        assert torch.allclose(second_derivs['xx'], torch.tensor([[expected_xx]]), atol=1e-3)
        assert torch.allclose(second_derivs['yy'], torch.tensor([[expected_yy]]), atol=1e-3)
        assert torch.allclose(second_derivs['zz'], torch.tensor([[expected_zz]]), atol=1e-3)
    
    def test_laplacian_computation(self, solver):
        """测试 Laplacian 计算"""
        coords = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
        scaled_coords = solver.scale_coordinates(coords)
        scaled_coords.requires_grad_(True)
        
        # 函数: f = X² + Y² + Z²
        # ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z² = (N_x² + N_y² + N_z²) · 2
        X, Y, Z = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        field = X**2 + Y**2 + Z**2
        field.requires_grad_(True)
        
        laplacian = solver.compute_laplacian(field, coords, scaled_coords=scaled_coords)
        
        N_x, N_y, N_z = 2.0, 12.0, 2.0
        expected_laplacian = (N_x**2 + N_y**2 + N_z**2) * 2
        
        assert torch.allclose(laplacian, torch.tensor([[expected_laplacian]]), atol=1e-2)
    
    # ============================================================
    # 测试 3: 物理残差
    # ============================================================
    
    def test_momentum_residuals_shape(self, solver, sample_coords, sample_predictions):
        """测试动量残差的输出形状"""
        coords = sample_coords.clone().detach().requires_grad_(True)
        scaled_coords = solver.scale_coordinates(coords).detach().requires_grad_(True)
        # 構造依賴坐標的預測值，確保梯度存在
        u = scaled_coords[:, 0:1] ** 2
        v = scaled_coords[:, 1:2] ** 2
        w = scaled_coords[:, 2:3] ** 2
        p = scaled_coords.sum(dim=1, keepdim=True)
        predictions = torch.cat([u, v, w, p], dim=1)
        residuals = solver.compute_momentum_residuals(coords, predictions, scaled_coords=scaled_coords)
        
        assert 'momentum_x' in residuals
        assert 'momentum_y' in residuals
        assert 'momentum_z' in residuals
        
        assert residuals['momentum_x'].shape == (sample_coords.shape[0], 1)
        assert residuals['momentum_y'].shape == (sample_coords.shape[0], 1)
        assert residuals['momentum_z'].shape == (sample_coords.shape[0], 1)
    
    def test_continuity_residual_shape(self, solver, sample_coords, sample_predictions):
        """测试连续方程残差的输出形状"""
        coords = sample_coords.clone().detach().requires_grad_(True)
        scaled_coords = solver.scale_coordinates(coords).detach().requires_grad_(True)
        u = scaled_coords[:, 0:1] ** 2
        v = scaled_coords[:, 1:2] ** 2
        w = scaled_coords[:, 2:3] ** 2
        p = scaled_coords.sum(dim=1, keepdim=True)
        predictions = torch.cat([u, v, w, p], dim=1)
        continuity = solver.compute_continuity_residual(coords, predictions, scaled_coords=scaled_coords)
        
        assert continuity.shape == (sample_coords.shape[0], 1)
    
    def test_pressure_gradient_in_x_momentum(self, solver):
        """测试 x 方向动量方程包含压降项"""
        # 创建简单的测试场（零速度，均匀压力）
        coords = torch.tensor([[1.0, 0.0, 1.0]], requires_grad=True)
        scaled_coords = solver.scale_coordinates(coords).detach().requires_grad_(True)
        zeros = scaled_coords[:, 0:1] * 0.0
        predictions = torch.cat([zeros, zeros, zeros, torch.ones_like(zeros)], dim=1)
        
        residuals = solver.compute_momentum_residuals(coords, predictions, scaled_coords=scaled_coords)
        
        # 对于零速度场，x 方向残差应该主要由压降项贡献
        # residual_x ≈ -dP/dx (因为对流项和黏性项都为零)
        # 实际上压力梯度项也可能不为零，这里只检查不是全零
        assert not torch.allclose(residuals['momentum_x'], torch.zeros_like(residuals['momentum_x']))
    
    # ============================================================
    # 测试 4: 周期性边界
    # ============================================================
    
    def test_periodic_loss_computation(self, solver):
        """测试周期性边界损失计算"""
        # 创建成对的边界点（x 方向）
        x_min, x_max = solver.domain_bounds['x']
        coords = torch.tensor([
            [x_min, 0.0, 1.0],
            [x_max, 0.0, 1.0],
        ])
        
        # 相同的场值应该得到零损失
        predictions = torch.tensor([
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ])
        
        periodic_loss = solver.compute_periodic_loss(coords, predictions)
        
        assert 'periodic_x' in periodic_loss
        assert 'periodic_z' in periodic_loss
        assert periodic_loss['periodic_x'] < 1e-5
    
    # ============================================================
    # 测试 5: 壁面剪应力
    # ============================================================
    
    def test_wall_shear_stress_computation(self, solver):
        """测试壁面剪应力计算"""
        # 创建壁面点
        y_lower, y_upper = solver.domain_bounds['y']
        coords = torch.tensor([
            [1.0, y_lower, 1.0],
            [1.0, y_upper, 1.0],
        ])
        
        predictions = torch.randn(2, 4)
        
        shear_stress = solver.compute_wall_shear_stress(coords, predictions)
        
        assert 'tau_w_lower' in shear_stress
        assert 'tau_w_upper' in shear_stress
        assert shear_stress['tau_w_lower'].item() >= 0
        assert shear_stress['tau_w_upper'].item() >= 0
    
    # ============================================================
    # 测试 6: Loss 权重补偿
    # ============================================================
    
    def test_loss_weight_compensation(self, solver):
        """测试 Loss 权重补偿因子"""
        compensation = solver.compute_loss_weight_compensation()
        
        # 补偿因子应该是 1/N_max² = 1/144
        expected = 1.0 / 144.0
        assert abs(compensation - expected) < 1e-6
    
    def test_scaling_info_output(self, solver):
        """测试缩放信息输出"""
        info = solver.get_scaling_info()
        
        assert 'scaling_factors' in info
        assert 'physics_parameters' in info
        assert 'loss_compensation' in info
        assert 'domain_bounds' in info
        
        assert info['scaling_factors']['N_y'] == 12.0
        assert info['scaling_factors']['N_x'] == 2.0
        # 使用近似比較以處理浮點數精度問題
        assert abs(info['physics_parameters']['nu'] - 5e-5) < 1e-9
    
    # ============================================================
    # 测试 7: 配置验证
    # ============================================================
    
    def test_invalid_pressure_gradient(self):
        """测试负压降应该抛出错误"""
        with pytest.raises(ValueError, match="压降项 dP/dx"):
            create_vs_pinn_channel_flow(dP_dx=-0.01)
    
    def test_custom_scaling_factors(self):
        """测试自定义缩放因子"""
        solver = create_vs_pinn_channel_flow(N_y=16.0, N_x=4.0, N_z=3.0)
        
        info = solver.get_scaling_info()
        assert info['scaling_factors']['N_y'] == 16.0
        assert info['scaling_factors']['N_x'] == 4.0
        assert info['scaling_factors']['N_z'] == 3.0
        assert info['scaling_factors']['N_max'] == 16.0
    
    # ============================================================
    # 测试 8: 数值稳定性
    # ============================================================
    
    def test_gradient_computation_stability(self, solver):
        """测试梯度计算的数值稳定性（大批量）"""
        large_batch_coords = torch.randn(1000, 3)
        large_batch_preds = torch.randn(1000, 4)
        
        # 不应该产生 NaN 或 Inf
        residuals = solver.compute_momentum_residuals(large_batch_coords, large_batch_preds)
        
        for key, value in residuals.items():
            assert not torch.isnan(value).any(), f"{key} contains NaN"
            assert not torch.isinf(value).any(), f"{key} contains Inf"
    
    def test_extreme_scaling_factors(self):
        """测试极端缩放因子是否稳定"""
        # 非常大的缩放因子
        solver = create_vs_pinn_channel_flow(N_y=100.0, N_x=1.0, N_z=1.0)
        
        coords = torch.randn(10, 3)
        preds = torch.randn(10, 4)
        
        residuals = solver.compute_momentum_residuals(coords, preds)
        
        # 应该仍然产生有限值
        for key, value in residuals.items():
            assert torch.isfinite(value).all(), f"{key} contains non-finite values"


# ==============================================================================
# 性能测试（可选）
# ==============================================================================

class TestVSPINNPerformance:
    """VS-PINN 性能测试"""
    
    @pytest.mark.skip(reason="需要安装 pytest-benchmark: pip install pytest-benchmark")
    def test_gradient_computation_overhead(self, benchmark):
        """测试梯度计算的时间开销"""
        solver = create_vs_pinn_channel_flow()
        coords = torch.randn(100, 3)
        preds = torch.randn(100, 4)
        
        def compute_residuals():
            return solver.compute_momentum_residuals(coords, preds)
        
        # 使用 pytest-benchmark 测量性能
        result = benchmark(compute_residuals)
        assert result is not None


if __name__ == '__main__':
    # 运行测试
    pytest.main([__file__, '-v', '--tb=short'])
