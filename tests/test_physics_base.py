"""
Unit Tests for Physics Base Modules
====================================

測試 pinnx/physics/base/ 中的所有基礎模組：
1. gradient_ops.py - 梯度計算工具
2. laplacian_ops.py - 拉普拉斯算子
3. pde_base.py - PDE 抽象基類
4. ns_base.py - Navier-Stokes 基類

作者：PINNs-MVP 團隊
日期：2025-12-15
"""

import pytest
import torch
import numpy as np
from pinnx.physics.base import (
    compute_gradient,
    compute_all_gradients,
    compute_gradient_safe,
    compute_laplacian,
    compute_laplacian_anisotropic,
    PDEBase,
    NavierStokesBase
)


# ============================================================================
# Test Gradient Operations
# ============================================================================

class TestGradientOps:
    """測試梯度計算工具"""
    
    def test_compute_gradient_2d_linear(self):
        """測試 2D 線性函數梯度"""
        # f(x, y) = 2x + 3y
        # ∂f/∂x = 2, ∂f/∂y = 3
        x = torch.randn(100, 2, requires_grad=True)
        f = 2.0 * x[:, 0:1] + 3.0 * x[:, 1:2]
        
        df_dx = compute_gradient(f, x, component=0, spatial_dim=2)
        df_dy = compute_gradient(f, x, component=1, spatial_dim=2)
        
        assert torch.allclose(df_dx, torch.ones_like(df_dx) * 2.0, atol=1e-5)
        assert torch.allclose(df_dy, torch.ones_like(df_dy) * 3.0, atol=1e-5)
    
    def test_compute_gradient_2d_quadratic(self):
        """測試 2D 二次函數梯度"""
        # f(x, y) = x² + y²
        # ∂f/∂x = 2x, ∂f/∂y = 2y
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2 + x[:, 1:2]**2
        
        df_dx = compute_gradient(f, x, component=0, spatial_dim=2)
        df_dy = compute_gradient(f, x, component=1, spatial_dim=2)
        
        assert torch.allclose(df_dx, 2.0 * x[:, 0:1], atol=1e-4)
        assert torch.allclose(df_dy, 2.0 * x[:, 1:2], atol=1e-4)
    
    def test_compute_gradient_3d(self):
        """測試 3D 梯度計算"""
        # f(x, y, z) = x² + y² + z²
        # ∂f/∂x = 2x, ∂f/∂y = 2y, ∂f/∂z = 2z
        x = torch.randn(50, 3, requires_grad=True)
        f = torch.sum(x**2, dim=1, keepdim=True)
        
        df_dx = compute_gradient(f, x, component=0, spatial_dim=3)
        df_dy = compute_gradient(f, x, component=1, spatial_dim=3)
        df_dz = compute_gradient(f, x, component=2, spatial_dim=3)
        
        assert torch.allclose(df_dx, 2.0 * x[:, 0:1], atol=1e-4)
        assert torch.allclose(df_dy, 2.0 * x[:, 1:2], atol=1e-4)
        assert torch.allclose(df_dz, 2.0 * x[:, 2:3], atol=1e-4)
    
    def test_compute_all_gradients_2d(self):
        """測試批次梯度計算（2D）"""
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2 + x[:, 1:2]**2
        
        gradients = compute_all_gradients(f, x, spatial_dim=2)
        
        # gradients 返回單一張量 [100, 2]，不是列表
        assert gradients.shape == (100, 2)
        assert torch.allclose(gradients[:, 0:1], 2.0 * x[:, 0:1], atol=1e-4)
        assert torch.allclose(gradients[:, 1:2], 2.0 * x[:, 1:2], atol=1e-4)
    
    def test_compute_gradient_safe(self):
        """測試安全梯度計算（錯誤處理）"""
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2
        
        # 正常情況（compute_gradient_safe 不接受 spatial_dim 參數）
        df_dx = compute_gradient_safe(f, x, component=0)
        assert df_dx is not None
        assert not torch.isnan(df_dx).any()
        
        # 測試無效輸入（compute_gradient_safe 會自動處理，返回零梯度或有效梯度）
        x_no_grad = torch.randn(100, 2, requires_grad=False)
        f_no_grad = x_no_grad[:, 0:1]**2
        df_dx_invalid = compute_gradient_safe(f_no_grad, x_no_grad, component=0)
        # compute_gradient_safe 會自動設定 requires_grad=True，所以會返回有效梯度
        assert df_dx_invalid is not None
        assert df_dx_invalid.shape == (100, 1)


# ============================================================================
# Test Laplacian Operations
# ============================================================================

class TestLaplacianOps:
    """測試拉普拉斯算子"""
    
    def test_laplacian_2d_harmonic(self):
        """測試 2D 調和函數（∇²f = 0）"""
        # f(x, y) = x² - y² 是調和函數
        # ∇²f = 2 - 2 = 0
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2 - x[:, 1:2]**2
        
        laplacian = compute_laplacian(f, x, spatial_dim=2)
        
        assert torch.allclose(laplacian, torch.zeros_like(laplacian), atol=1e-4)
    
    def test_laplacian_2d_quadratic(self):
        """測試 2D 二次函數"""
        # f(x, y) = x² + y²
        # ∇²f = 2 + 2 = 4
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2 + x[:, 1:2]**2
        
        laplacian = compute_laplacian(f, x, spatial_dim=2)
        
        assert torch.allclose(laplacian, torch.ones_like(laplacian) * 4.0, atol=1e-4)
    
    def test_laplacian_3d(self):
        """測試 3D 拉普拉斯算子"""
        # f(x, y, z) = x² + y² + z²
        # ∇²f = 2 + 2 + 2 = 6
        x = torch.randn(50, 3, requires_grad=True)
        f = torch.sum(x**2, dim=1, keepdim=True)
        
        laplacian = compute_laplacian(f, x, spatial_dim=3)
        
        assert torch.allclose(laplacian, torch.ones_like(laplacian) * 6.0, atol=1e-4)
    
    def test_laplacian_anisotropic(self):
        """測試各向異性拉普拉斯（VS-PINN）"""
        # f(X, Y) = X² + Y²（在縮放空間）
        # ∇²f = N_x² * 2 + N_y² * 2
        x = torch.randn(100, 2, requires_grad=True)
        f = x[:, 0:1]**2 + x[:, 1:2]**2
        
        N_x, N_y = 2.0, 12.0
        scaling_factors = torch.tensor([[N_x, N_y]])
        
        laplacian_aniso = compute_laplacian_anisotropic(f, x, scaling_factors)
        
        expected = 2.0 * (N_x**2 + N_y**2)  # 2*(4 + 144) = 296
        assert torch.allclose(laplacian_aniso, torch.ones_like(laplacian_aniso) * expected, atol=1e-3)
    
    def test_laplacian_stabilization(self):
        """測試拉普拉斯數值穩定化"""
        x = torch.randn(100, 2, requires_grad=True)
        # 創建一個可能產生大值的函數
        f = 100.0 * (x[:, 0:1]**4 + x[:, 1:2]**4)
        
        # 不穩定版本
        lap_unstable = compute_laplacian(f, x, spatial_dim=2, stabilize=False)
        
        # 穩定版本
        lap_stable = compute_laplacian(f, x, spatial_dim=2, stabilize=True, max_value=1e4)
        
        # 穩定版本應該截斷極值
        assert torch.all(torch.abs(lap_stable) <= 1e4)


# ============================================================================
# Test PDE Base Class
# ============================================================================

class DummyPDE(PDEBase):
    """測試用的 PDE 子類"""
    
    def __init__(self, domain_bounds, loss_config=None):
        super().__init__(domain_bounds, loss_config)
        self.spatial_dim = 2
    
    def residual(self, coords, predictions):
        # 簡單的 Poisson 方程：∇²u = 0
        u = predictions[:, 0:1]
        return self.compute_laplacian(u, coords)
    
    def get_physics_info(self):
        return {'equation': 'Poisson', 'dim': 2}


class TestPDEBase:
    """測試 PDE 抽象基類"""
    
    def test_pde_base_initialization(self):
        """測試 PDE 基類初始化"""
        domain = {'x': [0, 1], 'y': [0, 1]}
        loss_config = {'pde': 1.0, 'data': 100.0}
        
        pde = DummyPDE(domain, loss_config)
        
        assert pde.domain_bounds == domain
        assert pde.loss_config == loss_config
        assert pde.spatial_dim == 2
    
    def test_pde_gradient_interface(self):
        """測試 PDE 梯度計算接口"""
        domain = {'x': [0, 2], 'y': [0, 1]}
        pde = DummyPDE(domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        field = coords[:, 0:1]**2
        
        df_dx = pde.compute_gradient(field, coords, component=0)
        
        assert df_dx.shape == (50, 1)
        assert torch.allclose(df_dx, 2.0 * coords[:, 0:1], atol=1e-4)
    
    def test_pde_laplacian_interface(self):
        """測試 PDE 拉普拉斯接口"""
        domain = {'x': [0, 1], 'y': [0, 1]}
        pde = DummyPDE(domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        field = coords[:, 0:1]**2 + coords[:, 1:2]**2
        
        laplacian = pde.compute_laplacian(field, coords)
        
        assert laplacian.shape == (50, 1)
        assert torch.allclose(laplacian, torch.ones_like(laplacian) * 4.0, atol=1e-4)
    
    def test_pde_loss_weight_management(self):
        """測試損失權重管理"""
        domain = {'x': [0, 1], 'y': [0, 1]}
        loss_config = {'pde': 1.0, 'data': 100.0}
        pde = DummyPDE(domain, loss_config)
        
        # 獲取權重
        assert pde.get_loss_weight('pde') == 1.0
        assert pde.get_loss_weight('data') == 100.0
        assert pde.get_loss_weight('bc', default=10.0) == 10.0
        
        # 更新權重
        pde.update_loss_weight('pde', 2.0)
        assert pde.get_loss_weight('pde') == 2.0
    
    def test_pde_domain_size(self):
        """測試域尺寸計算"""
        domain = {'x': [0, 2], 'y': [-1, 1], 'z': [0, 1]}
        pde = DummyPDE(domain)
        
        sizes = pde.get_domain_size()
        
        assert sizes['x'] == 2.0
        assert sizes['y'] == 2.0
        assert sizes['z'] == 1.0


# ============================================================================
# Test Navier-Stokes Base Class
# ============================================================================

class Dummy2DNS(NavierStokesBase):
    """測試用的 2D N-S 子類"""
    
    def __init__(self, Re, domain_bounds):
        physics_params = {'Re': Re, 'nu': 1.0/Re}
        super().__init__(physics_params, domain_bounds, spatial_dim=2)
    
    def residual(self, coords, predictions):
        # 簡單返回連續性方程殘差
        u, v, p = self.parse_velocity_pressure(predictions)
        return self.compute_continuity_residual(coords, [u, v])


class TestNavierStokesBase:
    """測試 Navier-Stokes 基類"""
    
    def test_ns_initialization(self):
        """測試 N-S 基類初始化"""
        Re = 100.0
        domain = {'x': [0, 2*np.pi], 'y': [0, 1]}
        
        ns = Dummy2DNS(Re, domain)
        
        assert ns.Re == Re
        assert ns.nu == 1.0 / Re
        assert ns.spatial_dim == 2
    
    def test_parse_velocity_pressure_2d(self):
        """測試 2D 速度/壓力解析"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        predictions = torch.randn(50, 3)  # [u, v, p]
        u, v, p = ns.parse_velocity_pressure(predictions)
        
        assert u.shape == (50, 1)
        assert v.shape == (50, 1)
        assert p.shape == (50, 1)
        assert torch.allclose(u, predictions[:, 0:1])
        assert torch.allclose(v, predictions[:, 1:2])
        assert torch.allclose(p, predictions[:, 2:3])
    
    def test_parse_velocity_pressure_3d(self):
        """測試 3D 速度/壓力解析"""
        # 創建具體的 3D N-S 子類（NavierStokesBase 是抽象類）
        class Dummy3DNS(NavierStokesBase):
            def __init__(self, Re, domain_bounds):
                physics_params = {'Re': Re, 'nu': 1.0/Re}
                super().__init__(physics_params, domain_bounds, spatial_dim=3)
            
            def residual(self, coords, predictions):
                u, v, w, p = self.parse_velocity_pressure(predictions)
                return self.compute_continuity_residual(coords, [u, v, w])
        
        domain = {'x': [0, 1], 'y': [0, 1], 'z': [0, 1]}
        ns = Dummy3DNS(1000, domain)
        
        predictions = torch.randn(30, 4)  # [u, v, w, p]
        u, v, w, p = ns.parse_velocity_pressure(predictions)
        
        assert u.shape == (30, 1)
        assert v.shape == (30, 1)
        assert w.shape == (30, 1)
        assert p.shape == (30, 1)
    
    def test_continuity_residual_2d(self):
        """測試 2D 連續性方程"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        # 無散度場：u = sin(x), v = -sin(y) cos(y)
        # ∂u/∂x = cos(x), ∂v/∂y = -cos(2y)
        coords = torch.randn(50, 2, requires_grad=True)
        x, y = coords[:, 0:1], coords[:, 1:2]
        
        u = torch.sin(x)
        v = -torch.sin(y)
        
        # 計算連續性（應該接近 cos(x) - cos(y)）
        continuity = ns.compute_continuity_residual(coords, [u, v])
        
        assert continuity.shape == (50, 1)
        # 注意：這不是完美的無散度場，只是測試接口
    
    def test_advection_term(self):
        """測試對流項計算"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        u = coords[:, 0:1]**2
        v = coords[:, 1:2]**2
        
        advection_u = ns.compute_advection_term(coords, u, [u, v])
        
        assert advection_u.shape == (50, 1)
    
    def test_viscous_term(self):
        """測試黏性項計算"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        u = coords[:, 0:1]**2 + coords[:, 1:2]**2
        
        viscous = ns.compute_viscous_term(coords, u)
        
        assert viscous.shape == (50, 1)
        # ν∇²u = ν * 4
        expected = ns.nu * 4.0
        assert torch.allclose(viscous, torch.ones_like(viscous) * expected, atol=1e-4)
    
    def test_pressure_gradient(self):
        """測試壓力梯度"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        p = 3.0 * coords[:, 0:1] + 2.0 * coords[:, 1:2]
        
        dp_dx = ns.compute_pressure_gradient(p, coords, component=0)
        dp_dy = ns.compute_pressure_gradient(p, coords, component=1)
        
        assert torch.allclose(dp_dx, torch.ones_like(dp_dx) * 3.0, atol=1e-5)
        assert torch.allclose(dp_dy, torch.ones_like(dp_dy) * 2.0, atol=1e-5)
    
    def test_kinetic_energy(self):
        """測試動能計算"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        u = torch.tensor([[3.0]])
        v = torch.tensor([[4.0]])
        
        ke = ns.compute_kinetic_energy([u, v])
        
        # KE = 0.5 * (3² + 4²) = 0.5 * 25 = 12.5
        assert torch.allclose(ke, torch.tensor([[12.5]]))
    
    def test_enstrophy_2d(self):
        """測試渦量平方（Enstrophy）"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        coords = torch.randn(50, 2, requires_grad=True)
        u = coords[:, 0:1]
        v = coords[:, 1:2]
        
        enstrophy = ns.compute_enstrophy(coords, [u, v])
        
        assert enstrophy.shape == (50, 1)
        assert torch.all(enstrophy >= 0)  # Enstrophy 應該非負
    
    def test_physics_info(self):
        """測試物理參數元數據"""
        Re = 1000.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        info = ns.get_physics_info()
        
        assert info['equation'] == 'Navier-Stokes'
        assert info['spatial_dim'] == 2
        assert info['Re'] == Re
        assert info['nu'] == 1.0 / Re
        assert info['compressible'] is False


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """整合測試：測試模組間協作"""
    
    def test_pde_to_ns_inheritance(self):
        """測試 PDE → N-S 繼承鏈"""
        Re = 100.0
        domain = {'x': [0, 1], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        # 確認繼承關係
        assert isinstance(ns, NavierStokesBase)
        assert isinstance(ns, PDEBase)
        
        # 確認基類方法可用
        assert hasattr(ns, 'compute_gradient')
        assert hasattr(ns, 'compute_laplacian')
        assert hasattr(ns, 'get_loss_weight')
        assert hasattr(ns, 'get_physics_info')
    
    def test_full_ns_residual_workflow(self):
        """測試完整 N-S 殘差計算流程"""
        Re = 100.0
        domain = {'x': [0, 2*np.pi], 'y': [0, 1]}
        ns = Dummy2DNS(Re, domain)
        
        # 模擬網路輸出（確保 predictions 有梯度圖）
        coords = torch.randn(100, 2, requires_grad=True)
        
        # 創建簡單模型生成 predictions，確保有梯度連接
        dummy_model = torch.nn.Linear(2, 3)
        predictions = dummy_model(coords)
        
        # 計算殘差
        residual = ns.residual(coords, predictions)
        
        assert residual.shape == (100, 1)
        assert not torch.isnan(residual).any()
        assert not torch.isinf(residual).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
