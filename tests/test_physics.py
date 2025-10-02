"""
物理模組測試
測試 Navier-Stokes 方程式和變數縮放功能
"""

import numpy as np
import torch
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.physics.ns_2d import (
    ns_residual_2d, incompressible_ns_2d, 
    compute_derivatives, compute_laplacian, compute_vorticity,
    check_conservation_laws
)
from pinnx.physics.scaling import (
    StandardScaler, MinMaxScaler, VSScaler,
    create_scaler_from_data, denormalize_gradients
)


class TestNSEquations:
    """測試 Navier-Stokes 方程式模組"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_ns_residual_basic(self):
        """測試 NS 方程殘差基本功能"""
        # 建立簡單的測試場
        coords = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]], 
                             requires_grad=True, device=self.device, dtype=torch.float32)
        
        # 創建簡單網路以生成有梯度的預測值
        import torch.nn as nn
        net = nn.Sequential(
            nn.Linear(2, 10),
            nn.Tanh(),
            nn.Linear(10, 3)  # 輸出 [u, v, p]
        ).to(self.device)
        
        predictions = net(coords)
        
        # 計算殘差
        residuals = ns_residual_2d(coords, predictions, nu=0.01)
        
        # 檢查輸出格式
        assert len(residuals) == 3  # momentum_x, momentum_y, continuity
        assert all(r.shape[0] == coords.shape[0] for r in residuals)
        assert all(not torch.isnan(r).any() for r in residuals)
    
    def test_incompressible_ns(self):
        """測試不可壓縮 NS 方程"""
        coords = torch.randn(10, 2, requires_grad=True, device=self.device)
        
        # 創建簡單網路
        import torch.nn as nn
        net = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 3)).to(self.device)
        predictions = net(coords)  # [u, v, p]
        
        result = incompressible_ns_2d(coords, predictions, nu=0.01)
        
        assert result.shape[0] == coords.shape[0]
        assert not torch.isnan(result).any()
    
    def test_derivatives_calculation(self):
        """測試導數計算"""
        x = torch.linspace(0, 2*np.pi, 20, requires_grad=True, device=self.device)
        y = torch.linspace(0, 2*np.pi, 20, requires_grad=True, device=self.device)
        
        # 已知解析解的場: f = sin(x) * cos(y)
        coords = torch.stack([x, y], dim=1)
        f = torch.sin(coords[:, 0]) * torch.cos(coords[:, 1])
        f = f.unsqueeze(1)  # 變為 [batch, 1]
        
        # 計算導數
        derivs = compute_derivatives(f, coords, order=1)
        
        # 檢查導數維度
        assert derivs.shape == coords.shape
        assert not torch.isnan(derivs).any()
    
    def test_laplacian_calculation(self):
        """測試拉普拉斯算子計算"""
        coords = torch.tensor([[0.5, 0.5], [0.3, 0.7]], 
                             requires_grad=True, device=self.device, dtype=torch.float32)
        
        # 簡單的二次函數: f = x^2 + y^2, laplacian = 4
        f = torch.sum(coords**2, dim=1, keepdim=True)
        
        laplacian = compute_laplacian(f, coords)
        
        assert laplacian.shape == (coords.shape[0], 1)
        assert not torch.isnan(laplacian).any()
        
        # 對於 x^2 + y^2，拉普拉斯算子應該接近 4
        expected = torch.full_like(laplacian, 4.0)
        assert torch.allclose(laplacian, expected, atol=0.1)
    
    def test_vorticity_computation(self):
        """測試渦量計算"""
        coords = torch.randn(8, 2, requires_grad=True, device=self.device)
        
        # 創建簡單網路生成速度場
        import torch.nn as nn
        net = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 2)).to(self.device)
        velocity = net(coords)
        
        vorticity = compute_vorticity(coords, velocity)
        
        assert vorticity.shape == (coords.shape[0], 1)
        assert not torch.isnan(vorticity).any()


# class TestBoundaryConditions:
#     """測試邊界條件模組"""
#     
#     def test_boundary_conditions_application(self):
#         """測試邊界條件應用"""
#         coords = torch.tensor([[0.0, 0.5], [1.0, 0.5], [0.5, 0.0], [0.5, 1.0]], 
#                              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
#         
#         predictions = torch.tensor([[0.1, 0.2, 0.0], [0.15, 0.25, 0.05], 
#                                    [0.05, 0.1, -0.02], [0.2, 0.3, 0.1]], 
#                                   device=coords.device)
#         
#         # Dirichlet 邊界條件 (零速度)
#         bc_config = {
#             'type': 'dirichlet',
#             'value': torch.zeros(4, 2, device=coords.device),  # 零速度
#             'components': [0, 1]  # u, v 分量
#         }
#         
#         bc_loss = apply_boundary_conditions(coords, predictions, bc_config)
#         
#         assert bc_loss.numel() == 1
#         assert bc_loss >= 0
#         assert not torch.isnan(bc_loss)


class TestConservationLaws:
    """測試守恆定律"""
    
    def test_conservation_laws_check(self):
        """測試守恆定律檢查"""
        coords = torch.randn(15, 2, requires_grad=True, 
                           device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 創建網路生成速度和壓力場
        import torch.nn as nn
        net = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 3)).to(coords.device)
        predictions = net(coords)
        velocity = predictions[:, :2]
        pressure = predictions[:, 2:3]
        
        conservation_result = check_conservation_laws(coords, velocity, pressure)
        
        assert 'mass_conservation' in conservation_result
        assert 'momentum_conservation' in conservation_result
        assert 'energy_conservation' in conservation_result
        
        for key, value in conservation_result.items():
            assert not torch.isnan(value).any()


class TestScalers:
    """測試縮放器模組"""
    
    def test_standard_scaler(self):
        """測試標準縮放器"""
        data = torch.randn(50, 3)
        scaler = StandardScaler()
        
        # 擬合和轉換
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        
        # 標準化後應該零均值單位方差
        assert torch.allclose(scaled_data.mean(dim=0), torch.zeros(3), atol=1e-5)
        assert torch.allclose(scaled_data.std(dim=0), torch.ones(3), atol=1e-5)
        
        # 逆轉換
        reconstructed = scaler.inverse_transform(scaled_data)
        assert torch.allclose(data, reconstructed, atol=1e-5)
    
    def test_minmax_scaler(self):
        """測試最大最小值縮放器"""
        data = torch.rand(30, 2) * 10 + 5  # [5, 15] 範圍
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        scaler.fit(data)
        scaled_data = scaler.transform(data)
        
        # 應該在 [0, 1] 範圍內
        assert scaled_data.min() >= 0
        assert scaled_data.max() <= 1
        
        # 逆轉換
        reconstructed = scaler.inverse_transform(scaled_data)
        assert torch.allclose(data, reconstructed, atol=1e-5)
    
    def test_vspinn_scaler(self):
        """測試 VS-PINN 縮放器"""
        input_data = torch.randn(40, 2)
        output_data = torch.randn(40, 3)
        
        scaler = VSScaler(input_dim=2, output_dim=3, learnable=True)
        scaler.fit(input_data, output_data)
        
        # 轉換
        scaled_input = scaler.transform_input(input_data)
        scaled_output = scaler.transform_output(output_data)
        
        assert scaled_input.shape == input_data.shape
        assert scaled_output.shape == output_data.shape
        
        # 逆轉換
        reconstructed_input = scaler.inverse_transform_input(scaled_input)
        reconstructed_output = scaler.inverse_transform_output(scaled_output)
        
        assert torch.allclose(input_data, reconstructed_input, atol=1e-4)
        assert torch.allclose(output_data, reconstructed_output, atol=1e-4)
    
    def test_learnable_scaler_parameters(self):
        """測試可學習縮放器參數"""
        scaler = VSScaler(input_dim=2, output_dim=1, learnable=True)
        
        # 檢查參數是否可學習
        learnable_params = list(scaler.parameters())
        assert len(learnable_params) > 0
        assert all(p.requires_grad for p in learnable_params)


def test_integration_physics_modules():
    """測試物理模組整合"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 物理模組整合測試 ===")
    
    # 建立測試資料
    coords = torch.tensor([[0.2, 0.3], [0.5, 0.7], [0.8, 0.1]], 
                         requires_grad=True, device=device, dtype=torch.float32)
    
    # 創建網路生成預測值
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(2, 10), nn.Tanh(), nn.Linear(10, 3)).to(device)
    predictions = net(coords)
    
    # NS 方程計算
    residuals = ns_residual_2d(coords, predictions, nu=0.01)
    
    # 渦量計算
    vorticity = compute_vorticity(coords, predictions[:, :2])
    
    # 守恆定律檢查
    conservation = check_conservation_laws(coords, predictions[:, :2], predictions[:, 2:3])
    
    # 縮放測試
    scaler = StandardScaler()
    scaler.fit(predictions)
    scaled_predictions = scaler.transform(predictions)
    
    print(f"殘差計算完成，共 {len(residuals)} 項")
    print(f"渦量範圍: [{vorticity.min():.3f}, {vorticity.max():.3f}]")
    print(f"守恆定律項目: {list(conservation.keys())}")
    print(f"縮放後資料範圍: [{scaled_predictions.min():.3f}, {scaled_predictions.max():.3f}]")
    
    # 基本的健全性檢查
    assert all(not torch.isnan(r).any() for r in residuals)
    assert not torch.isnan(vorticity).any()
    assert all(not torch.isnan(conservation[k]).any() for k in conservation)
    assert not torch.isnan(scaled_predictions).any()
    
    print("✅ 物理模組整合測試通過")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])