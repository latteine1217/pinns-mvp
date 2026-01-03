"""
測試 Registry Pattern 重構後的 Model & Physics 工廠

驗證：
1. 所有模型類型都能正確創建
2. 所有物理類型都能正確創建  
3. 錯誤處理正確（未知類型）
4. 列表函數返回正確的註冊類型
"""

import pytest
import torch
import torch.nn as nn

from pinnx.train.model_physics_factory import (
    create_model,
    create_physics,
    list_available_models,
    list_available_physics,
)


class TestModelFactory:
    """測試模型工廠（Registry Pattern）"""
    
    def test_list_available_models(self):
        """測試列出可用模型類型"""
        models = list_available_models()
        
        assert isinstance(models, list)
        assert len(models) == 4
        assert 'fourier_vs_mlp' in models
        assert 'resnet' in models
        assert 'piratenet' in models
        assert 'axis_selective_fourier_mlp' in models
        assert models == sorted(models)  # 應該是排序的
    
    def test_create_fourier_vs_mlp(self):
        """測試創建 Fourier-VS MLP"""
        config = {
            'model': {
                'type': 'fourier_vs_mlp',
                'in_dim': 3,
                'out_dim': 4,
                'width': 128,
                'depth': 4,
                'activation': 'tanh',
                'fourier_features': {'type': 'disabled'}
            },
            'physics': {'type': 'ns_2d'}
        }
        device = torch.device('cpu')
        
        model = create_model(config, device)
        
        assert isinstance(model, nn.Module)
        assert sum(p.numel() for p in model.parameters()) > 0
    
    def test_create_resnet(self):
        """測試創建 ResNet 模型"""
        config = {
            'model': {
                'type': 'resnet',
                'in_dim': 3,
                'out_dim': 4,
                'width': 128,
                'depth': 4,
                'activation': 'tanh',
                'fourier_features': {'type': 'disabled'}
            },
            'physics': {'type': 'ns_2d'}
        }
        device = torch.device('cpu')
        
        model = create_model(config, device)
        
        assert isinstance(model, nn.Module)
    
    def test_create_piratenet(self):
        """測試創建 PirateNet 模型"""
        config = {
            'model': {
                'type': 'piratenet',
                'in_dim': 3,
                'out_dim': 4,
                'width': 128,
                'depth': 4,
                'activation': 'tanh',
                'fourier_features': {'type': 'disabled'}
            },
            'physics': {'type': 'ns_2d'}
        }
        device = torch.device('cpu')
        
        model = create_model(config, device)
        
        assert isinstance(model, nn.Module)
    
    def test_unknown_model_type(self):
        """測試未知模型類型會拋出錯誤"""
        config = {
            'model': {
                'type': 'unknown_model',
                'in_dim': 3,
                'out_dim': 4,
                'width': 128,
                'depth': 4,
                'activation': 'tanh',
                'fourier_features': {'type': 'disabled'}
            },
            'physics': {'type': 'ns_2d'}
        }
        device = torch.device('cpu')
        
        with pytest.raises(ValueError) as exc_info:
            create_model(config, device)
        
        assert 'Unknown type' in str(exc_info.value)
        assert 'unknown_model' in str(exc_info.value)
        assert 'Available types' in str(exc_info.value)


class TestPhysicsFactory:
    """測試物理工廠（Registry Pattern）"""
    
    def test_list_available_physics(self):
        """測試列出可用物理類型"""
        physics_types = list_available_physics()
        
        assert isinstance(physics_types, list)
        assert len(physics_types) == 3
        assert 'ns_2d' in physics_types
        assert 'vs_pinn_channel_flow' in physics_types
        assert 'kolmogorov_flow_2d' in physics_types
        assert physics_types == sorted(physics_types)  # 應該是排序的
    
    def test_create_ns_2d(self):
        """測試創建 NS 2D 物理模組"""
        config = {
            'physics': {
                'type': 'ns_2d',
                'nu': 1e-3,
                'rho': 1.0
            }
        }
        device = torch.device('cpu')
        
        physics = create_physics(config, device)
        
        assert physics is not None
        assert hasattr(physics, 'nu')  # NSEquations2D uses 'nu' not 'viscosity'
        assert hasattr(physics, 'rho')  # NSEquations2D uses 'rho' not 'density'
    
    def test_create_kolmogorov_flow_2d(self):
        """測試創建 Kolmogorov Flow 2D"""
        config = {
            'physics': {
                'type': 'kolmogorov_flow_2d',
                'nu': 0.01,
                'rho': 1.0,
                'forcing': {
                    'amplitude': 1.0,
                    'wavenumber': 4
                },
                'domain': {
                    'x_min': 0.0,
                    'x_max': 6.28,
                    'y_min': 0.0,
                    'y_max': 6.28
                }
            },
            'losses': {}
        }
        device = torch.device('cpu')
        
        physics = create_physics(config, device)
        
        assert physics is not None
    
    def test_create_vs_pinn_channel_flow(self):
        """測試創建 VS-PINN 通道流"""
        config = {
            'physics': {
                'type': 'vs_pinn_channel_flow',
                'nu': 5e-5,
                'rho': 1.0,
                'dP_dx': 0.0025,
                'domain': {
                    'x_range': [0.0, 25.13],
                    'y_range': [-1.0, 1.0],
                    'z_range': [0.0, 9.42]
                },
                'vs_pinn': {
                    'scaling_factors': {
                        'N_x': 2.0,
                        'N_y': 12.0,
                        'N_z': 2.0
                    }
                }
            },
            'losses': {},
            'model': {}
        }
        device = torch.device('cpu')
        
        physics = create_physics(config, device)
        
        assert physics is not None
    
    def test_unknown_physics_type(self):
        """測試未知物理類型會拋出錯誤"""
        config = {
            'physics': {
                'type': 'unknown_physics'
            }
        }
        device = torch.device('cpu')
        
        with pytest.raises(ValueError) as exc_info:
            create_physics(config, device)
        
        assert 'Unknown type' in str(exc_info.value)
        assert 'unknown_physics' in str(exc_info.value)
        assert 'Available types' in str(exc_info.value)


class TestIntegration:
    """整合測試：模型 + 物理"""
    
    def test_create_model_and_physics_together(self):
        """測試同時創建模型和物理模組"""
        config = {
            'model': {
                'type': 'fourier_vs_mlp',
                'in_dim': 3,
                'out_dim': 4,
                'width': 128,
                'depth': 4,
                'activation': 'tanh',
                'fourier_features': {'type': 'disabled'}
            },
            'physics': {
                'type': 'ns_2d',
                'nu': 1e-3,
                'rho': 1.0
            }
        }
        device = torch.device('cpu')
        
        model = create_model(config, device)
        physics = create_physics(config, device)
        
        assert isinstance(model, nn.Module)
        assert physics is not None
        
        # 測試簡單的前向傳播
        x = torch.randn(10, 3)
        output = model(x)
        assert output.shape == (10, 4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
