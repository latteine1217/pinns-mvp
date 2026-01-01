"""
Kolmogorov Flow 2D 物理模組單元測試
===================================

測試範圍：
1. 物理模型初始化與配置驗證
2. 強迫項計算
3. 梯度與 Laplacian 計算
4. NS 方程殘差計算
5. 週期性邊界條件
6. 渦度與 enstrophy 計算
7. 損失歸一化
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.physics.kolmogorov_flow_2d import (
    KolmogorovFlow2D,
    create_kolmogorov_flow_2d,
)
from pinnx.physics.base.gradient_ops import compute_gradient
from pinnx.physics.base.laplacian_ops import compute_laplacian


@pytest.fixture
def device():
    """測試設備（CPU）"""
    return torch.device('cpu')


@pytest.fixture
def kolmogorov_flow():
    """基礎 Kolmogorov flow 實例"""
    return create_kolmogorov_flow_2d(
        forcing_amplitude=1.0,
        forcing_wavenumber=4,
        nu=0.01,
        rho=1.0,
    )


@pytest.fixture
def sample_data(device):
    """生成測試用的樣本數據（建立計算圖連接）"""
    N = 100
    # 創建葉節點座標
    coords_data = torch.rand(N, 2, device=device) * 2 * np.pi  # [0, 2π]
    coords = coords_data.clone().detach().requires_grad_(True)
    
    # 創建 predictions 作為 coords 的函數（確保計算圖連接）
    x = coords[:, 0:1]
    y = coords[:, 1:2]
    u = torch.sin(x) * torch.cos(y)
    v = torch.cos(x) * torch.sin(y)
    p = torch.sin(x + y)
    
    predictions = torch.cat([u, v, p], dim=1)
    return coords, predictions


class TestKolmogorovFlowInitialization:
    """測試初始化與配置"""

    def test_default_initialization(self):
        """測試預設初始化"""
        model = KolmogorovFlow2D()

        assert torch.isclose(model.amplitude, torch.tensor(1.0), atol=1e-5)
        assert torch.isclose(model.wavenumber, torch.tensor(4.0), atol=1e-5)
        assert model.nu == pytest.approx(0.01)
        assert model.rho == pytest.approx(1.0)
        assert model.normalize_losses is True
        assert model.warmup_epochs == 5

    def test_custom_initialization(self):
        """測試自訂初始化"""
        model = create_kolmogorov_flow_2d(
            forcing_amplitude=2.0,
            forcing_wavenumber=8,
            nu=0.05,
            rho=2.0,
        )

        assert torch.isclose(model.amplitude, torch.tensor(2.0), atol=1e-5)
        assert torch.isclose(model.wavenumber, torch.tensor(8.0), atol=1e-5)
        assert model.nu == pytest.approx(0.05)
        assert model.rho == pytest.approx(2.0)

    def test_domain_bounds(self):
        """測試域邊界設定"""
        model = KolmogorovFlow2D()

        assert model.domain_bounds['x'] == (0.0, 2.0 * np.pi)
        assert model.domain_bounds['y'] == (0.0, 2.0 * np.pi)

    def test_invalid_parameters(self):
        """測試無效參數檢測"""
        # 負強迫振幅
        with pytest.raises(ValueError, match="強迫振幅"):
            KolmogorovFlow2D(forcing_params={'amplitude': -1.0, 'wavenumber': 4})

        # 負黏度
        with pytest.raises(ValueError, match="動力黏度"):
            KolmogorovFlow2D(physics_params={'nu': -0.01, 'rho': 1.0})


class TestForcingTerm:
    """測試強迫項計算"""

    def test_forcing_amplitude(self, kolmogorov_flow, sample_data):
        """測試強迫項振幅"""
        coords, _ = sample_data
        forcing = kolmogorov_flow.compute_forcing_term(coords)

        # 檢查形狀
        assert forcing.shape == (coords.shape[0], 1)

        # 檢查振幅範圍：|A sin(k_f y)| ≤ A
        A = kolmogorov_flow.amplitude.item()
        assert torch.all(torch.abs(forcing) <= A + 1e-6)

    def test_forcing_periodicity(self, kolmogorov_flow):
        """測試強迫項的 y 方向週期性"""
        y = torch.linspace(0, 2 * np.pi, 100, requires_grad=True).unsqueeze(1)
        coords = torch.cat([torch.zeros_like(y), y], dim=1)

        forcing = kolmogorov_flow.compute_forcing_term(coords)

        # 檢查週期性：f(y + 2π/k_f) = f(y)
        k_f = kolmogorov_flow.wavenumber.item()
        period = 2 * np.pi / k_f

        y_shifted = y + period
        coords_shifted = torch.cat([torch.zeros_like(y_shifted), y_shifted], dim=1)
        forcing_shifted = kolmogorov_flow.compute_forcing_term(coords_shifted)

        assert torch.allclose(forcing, forcing_shifted, atol=1e-5)


class TestGradientComputation:
    """測試梯度計算"""

    def test_gradient_2d_linear_function(self, device):
        """測試梯度計算（線性函數）"""
        # 測試函數：f(x, y) = 2x + 3y
        coords = torch.rand(50, 2, device=device, requires_grad=True)
        field = 2.0 * coords[:, 0:1] + 3.0 * coords[:, 1:2]

        grad_x = compute_gradient(field, coords, component=0, spatial_dim=2)
        grad_y = compute_gradient(field, coords, component=1, spatial_dim=2)

        # 解析解：∂f/∂x = 2, ∂f/∂y = 3
        assert torch.allclose(grad_x, torch.full_like(grad_x, 2.0), atol=1e-5)
        assert torch.allclose(grad_y, torch.full_like(grad_y, 3.0), atol=1e-5)

    def test_laplacian_2d_quadratic_function(self, device):
        """測試 Laplacian 計算（二次函數）"""
        # 測試函數：f(x, y) = x² + y²
        coords = torch.rand(50, 2, device=device, requires_grad=True)
        field = coords[:, 0:1] ** 2 + coords[:, 1:2] ** 2

        laplacian = compute_laplacian(field, coords, spatial_dim=2)

        # 解析解：∇²f = 2 + 2 = 4
        assert torch.allclose(laplacian, torch.full_like(laplacian, 4.0), atol=1e-4)


class TestMomentumResiduals:
    """測試動量殘差計算"""

    def test_residual_shape(self, kolmogorov_flow, sample_data):
        """測試殘差形狀"""
        coords, predictions = sample_data

        residuals = kolmogorov_flow.compute_momentum_residuals(coords, predictions)

        assert 'momentum_x' in residuals
        assert 'momentum_y' in residuals
        assert residuals['momentum_x'].shape == (coords.shape[0], 1)
        assert residuals['momentum_y'].shape == (coords.shape[0], 1)

    def test_residual_gradient_flow(self, kolmogorov_flow, sample_data):
        """測試殘差梯度回傳"""
        coords, predictions = sample_data

        residuals = kolmogorov_flow.compute_momentum_residuals(coords, predictions)

        # 檢查殘差是否保留計算圖
        assert residuals['momentum_x'].requires_grad
        assert residuals['momentum_y'].requires_grad

        # 測試梯度計算（檢查 coords 的梯度，因為 predictions 不是葉節點）
        loss = residuals['momentum_x'].sum() + residuals['momentum_y'].sum()
        loss.backward()

        # 驗證梯度能夠回傳到輸入 coords（葉節點）
        assert coords.grad is not None
        assert coords.grad.shape == coords.shape


class TestContinuityResidual:
    """測試連續方程殘差"""

    def test_continuity_shape(self, kolmogorov_flow, sample_data):
        """測試連續方程殘差形狀"""
        coords, predictions = sample_data

        u, v, _ = kolmogorov_flow.parse_velocity_pressure(predictions)
        continuity = kolmogorov_flow.compute_continuity_residual(coords, [u, v])

        assert continuity.shape == (coords.shape[0], 1)
        assert continuity.requires_grad

    def test_continuity_incompressible_field(self, kolmogorov_flow, device):
        """測試不可壓縮場的連續方程（理想情況）"""
        # 構造不可壓縮場：u = sin(x)cos(y), v = -cos(x)sin(y)
        # ∂u/∂x + ∂v/∂y = cos(x)cos(y) - cos(x)cos(y) = 0
        N = 50
        coords = torch.rand(N, 2, device=device, requires_grad=True) * 2 * np.pi

        x = coords[:, 0:1]
        y = coords[:, 1:2]

        u = torch.sin(x) * torch.cos(y)
        v = -torch.cos(x) * torch.sin(y)
        p = torch.zeros_like(u)

        predictions = torch.cat([u, v, p], dim=1)
        predictions.requires_grad_(True)

        continuity = kolmogorov_flow.compute_continuity_residual(coords, [u, v])

        # 不可壓縮場的連續方程殘差應接近零
        assert torch.allclose(continuity, torch.zeros_like(continuity), atol=1e-4)


class TestPeriodicBoundaryConditions:
    """測試週期性邊界條件"""

    def test_periodic_loss_shape(self, kolmogorov_flow, sample_data):
        """測試週期性損失形狀"""
        coords, predictions = sample_data

        periodic_loss = kolmogorov_flow.compute_periodic_loss(coords, predictions)

        assert 'periodic_x' in periodic_loss
        assert 'periodic_y' in periodic_loss
        assert isinstance(periodic_loss['periodic_x'], torch.Tensor)
        assert isinstance(periodic_loss['periodic_y'], torch.Tensor)

    def test_periodic_loss_exact_boundaries(self, kolmogorov_flow, device):
        """測試精確邊界點的週期性損失"""
        # 構造成對的邊界點
        N = 50
        y_random = torch.rand(N, 1, device=device) * 2 * np.pi

        # x 方向邊界對
        coords_x_min = torch.cat([torch.zeros_like(y_random), y_random], dim=1)
        coords_x_max = torch.cat([torch.full_like(y_random, 2 * np.pi), y_random], dim=1)

        # 相同的場值
        field_values = torch.rand(N, 3, device=device)

        # 合併座標
        coords = torch.cat([coords_x_min, coords_x_max], dim=0)
        predictions = torch.cat([field_values, field_values], dim=0)

        coords.requires_grad_(True)
        predictions.requires_grad_(True)

        periodic_loss = kolmogorov_flow.compute_periodic_loss(
            coords,
            predictions,
            boundary_band_width=0.1  # 較大的帶寬以確保捕獲邊界點
        )

        # 相同場值的週期性損失應接近零
        assert periodic_loss['periodic_x'].item() < 1e-10


class TestVorticityAndEnstrophy:
    """測試渦度與 enstrophy 計算"""

    def test_vorticity_shape(self, kolmogorov_flow, sample_data):
        """測試渦度計算形狀"""
        coords, predictions = sample_data

        vorticity = kolmogorov_flow.compute_vorticity(coords, predictions)

        assert vorticity.shape == (coords.shape[0], 1)

    def test_enstrophy_positive(self, kolmogorov_flow, sample_data):
        """測試 enstrophy 為非負值"""
        coords, predictions = sample_data

        enstrophy = kolmogorov_flow.compute_enstrophy(coords, predictions)

        assert enstrophy >= 0.0
        assert isinstance(enstrophy, torch.Tensor)

    def test_kinetic_energy_positive(self, kolmogorov_flow, sample_data):
        """測試動能為非負值"""
        coords, predictions = sample_data

        u, v, _ = kolmogorov_flow.parse_velocity_pressure(predictions)
        kinetic_energy = kolmogorov_flow.compute_kinetic_energy([u, v])

        assert kinetic_energy >= 0.0
        assert isinstance(kinetic_energy, torch.Tensor)


class TestPhysicsInfo:
    """測試物理信息獲取"""

    def test_get_physics_info(self, kolmogorov_flow):
        """測試物理信息獲取"""
        info = kolmogorov_flow.get_physics_info()

        # 檢查必要鍵存在
        assert 'forcing_amplitude' in info
        assert 'forcing_wavenumber' in info
        assert 'physics_parameters' in info
        assert 'domain_bounds' in info
        assert 'boundary_conditions' in info
        assert 'loss_normalization' in info

        # 檢查強迫參數
        assert abs(info['forcing_amplitude'] - 1.0) < 1e-5
        assert abs(info['forcing_wavenumber'] - 4) < 1e-5

        # 檢查物理參數
        assert abs(info['physics_parameters']['nu'] - 0.01) < 1e-5
        assert 'Reynolds_number' in info['physics_parameters']

        # 檢查邊界條件
        assert info['boundary_conditions'] == 'double_periodic'


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
