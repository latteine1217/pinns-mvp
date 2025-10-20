"""
AxisSelectiveFourierFeatures 單元測試

測試覆蓋：
1. 軸向頻率分配正確性
2. 輸出維度計算
3. 空軸處理（不使用 Fourier 的軸）
4. 梯度回傳正常
5. 工廠模式創建
6. 頻率動態更新
7. 設備兼容性
"""

import pytest
import torch
import torch.nn as nn
from pinnx.models.axis_selective_fourier import (
    AxisSelectiveFourierFeatures,
    FourierFeatureFactory
)


class TestAxisSelectiveFourierFeatures:
    """AxisSelectiveFourierFeatures 核心功能測試"""
    
    def test_axis_fourier_basic_initialization(self):
        """測試基本初始化"""
        config = {'x': [1, 2], 'y': [], 'z': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        assert fourier.in_dim == 3
        assert fourier.out_dim == 6  # 2*(1+0+1) = 2*2 = 4？應該是 2*3=6
        assert fourier.axes_names == ['x', 'y', 'z']
    
    def test_output_dimensions(self):
        """測試輸出維度計算正確性"""
        test_cases = [
            ({'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}, 14),  # 2*(4+0+3)
            ({'x': [1], 'y': [1], 'z': [1]}, 6),                   # 2*(1+1+1)
            ({'x': [], 'y': [], 'z': []}, 0),                      # 2*0
            ({'x': [1, 2], 'y': [1]}, 6),                          # 2*(2+1)
        ]
        
        for config, expected_dim in test_cases:
            fourier = AxisSelectiveFourierFeatures(config)
            batch_size = 10
            x = torch.randn(batch_size, len(config))
            output = fourier(x)
            
            assert output.shape == (batch_size, expected_dim), \
                f"配置 {config} 預期輸出維度 {expected_dim}，實際 {output.shape[1]}"
    
    def test_empty_axes_handling(self):
        """測試空軸處理（所有軸都不用 Fourier）"""
        config = {'x': [], 'y': [], 'z': []}
        fourier = AxisSelectiveFourierFeatures(config)
        
        x = torch.randn(5, 3)
        output = fourier(x)
        
        assert output.shape == (5, 0), "所有軸空列表應返回零維輸出"
    
    def test_forward_pass_values(self):
        """測試前向傳播計算正確性"""
        config = {'x': [1], 'y': []}
        domain_lengths = {'x': 2.0 * 3.141592653589793, 'y': 1.0}
        fourier = AxisSelectiveFourierFeatures(config, domain_lengths)
        
        # 測試特定輸入
        x = torch.tensor([[0.0, 0.5], [1.0, 0.5]])
        output = fourier(x)
        
        # 預期：x 軸有 k=1，輸出 [cos(k*x_norm), sin(k*x_norm)]
        # x_norm = x * (2π / L) = x * 1.0 = x（因為 L=2π）
        expected_cos = torch.cos(x[:, 0:1])  # k=1
        expected_sin = torch.sin(x[:, 0:1])
        expected = torch.cat([expected_cos, expected_sin], dim=1)
        
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
    
    def test_axis_fourier_gradient_flow(self):
        """測試梯度回傳正常"""
        config = {'x': [1, 2], 'y': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        x = torch.randn(8, 2, requires_grad=True)
        output = fourier(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None, "輸入梯度為 None"
        assert not torch.isnan(x.grad).any(), "梯度包含 NaN"
        assert not torch.isinf(x.grad).any(), "梯度包含 Inf"
    
    def test_domain_normalization(self):
        """測試物理域歸一化"""
        import math
        config = {'x': [1]}
        # 域長度為 8π，應歸一化到 2π
        domain_lengths = {'x': 8.0 * math.pi}
        fourier = AxisSelectiveFourierFeatures(config, domain_lengths)
        
        # 檢查歸一化因子
        expected_norm = 2.0 * math.pi / (8.0 * math.pi)  # = 0.25
        actual_norm = fourier.normalization_factors[0].item()
        
        assert abs(actual_norm - expected_norm) < 1e-6, \
            f"歸一化因子錯誤：預期 {expected_norm}，實際 {actual_norm}"
    
    def test_trainable_parameters(self):
        """測試可訓練模式"""
        config = {'x': [1, 2]}
        fourier_trainable = AxisSelectiveFourierFeatures(config, trainable=True)
        fourier_frozen = AxisSelectiveFourierFeatures(config, trainable=False)
        
        # 可訓練模式：B 應是 Parameter
        assert isinstance(fourier_trainable.B, nn.Parameter)
        assert fourier_trainable.B.requires_grad
        
        # 凍結模式：B 應是 buffer（不可訓練）
        assert not isinstance(fourier_frozen.B, nn.Parameter)
        assert not fourier_frozen.B.requires_grad
    
    def test_device_compatibility(self):
        """測試設備兼容性（CPU/CUDA）"""
        config = {'x': [1, 2], 'y': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        # CPU 測試
        x_cpu = torch.randn(4, 2)
        output_cpu = fourier(x_cpu)
        assert output_cpu.device.type == 'cpu'
        
        # CUDA 測試（如果可用）
        if torch.cuda.is_available():
            x_cuda = torch.randn(4, 2, device='cuda')
            output_cuda = fourier(x_cuda)
            assert output_cuda.device.type == 'cuda'
    
    def test_get_active_frequencies(self):
        """測試獲取當前頻率配置"""
        config = {'x': [1, 2, 4], 'y': [], 'z': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        active = fourier.get_active_frequencies()
        assert active == config
        
        # 驗證返回的是副本（修改不影響原配置）
        active['x'].append(8)
        assert fourier.axes_config['x'] == [1, 2, 4]
    
    def test_set_active_frequencies(self):
        """測試動態更新頻率配置"""
        config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}
        fourier = AxisSelectiveFourierFeatures(config)
        original_out_dim = fourier.out_dim  # 14
        
        # 減少頻率
        new_config = {'x': [1, 2], 'y': [], 'z': [1]}
        fourier.set_active_frequencies(new_config)
        
        assert fourier.out_dim == 6  # 2*(2+0+1)
        assert fourier.axes_config == new_config
        
        # 測試前向傳播仍正常
        x = torch.randn(5, 3)
        output = fourier(x)
        assert output.shape == (5, 6)
    
    def test_set_active_frequencies_invalid(self):
        """測試無效頻率更新（超出原始範圍）"""
        config = {'x': [1, 2], 'y': []}
        fourier = AxisSelectiveFourierFeatures(config)
        
        # 嘗試添加原配置中沒有的頻率
        invalid_config = {'x': [1, 2, 4], 'y': []}  # 4 不在原始配置
        
        with pytest.raises(ValueError):
            fourier.set_active_frequencies(invalid_config)


class TestFourierFeatureFactory:
    """FourierFeatureFactory 工廠模式測試"""
    
    def test_create_with_dict(self):
        """測試字典配置創建"""
        config = {
            'type': 'axis_selective',
            'axes_config': {'x': [1, 2], 'y': [], 'z': [1]},
            'domain_lengths': {'x': 1.0, 'y': 1.0, 'z': 1.0}
        }
        fourier = FourierFeatureFactory.create(config)
        
        assert isinstance(fourier, AxisSelectiveFourierFeatures)
        assert fourier.in_dim == 3
    
    def test_create_with_none(self):
        """測試 None 配置（無 Fourier）"""
        fourier = FourierFeatureFactory.create(None, in_dim=3)
        
        # 應返回一個輸出零維的模組
        x = torch.randn(4, 3)
        output = fourier(x)
        assert output.shape[1] == 0
    
    def test_fourier_factory_backward_compatibility(self):
        """測試向後兼容性（舊版 FourierFeatures）"""
        # 這裡假設舊版接受整數配置
        # 如果專案中有舊版 FourierFeatures，應測試工廠能正確處理
        pass  # TODO: 根據實際舊版實現補充


class TestIntegrationScenarios:
    """整合場景測試"""
    
    def test_channel_flow_scenario(self):
        """測試通道流場景：x/z 週期，y 非週期"""
        config = {
            'x': [1, 2, 4, 8],  # 流向（週期）
            'y': [],            # 壁法向（非週期）
            'z': [1, 2, 4]      # 展向（週期）
        }
        domain_lengths = {
            'x': 8.0 * 3.141592653589793,  # 8π
            'y': 2.0,                        # 2h
            'z': 3.0 * 3.141592653589793   # 3π
        }
        
        fourier = AxisSelectiveFourierFeatures(config, domain_lengths)
        
        # 批量輸入
        batch_size = 128
        x = torch.randn(batch_size, 3)
        output = fourier(x)
        
        assert output.shape == (batch_size, 14)
        assert not torch.isnan(output).any()
    
    def test_frequency_annealing_simulation(self):
        """模擬頻率退火過程"""
        full_config = {'x': [1, 2, 4, 8], 'y': []}
        fourier = AxisSelectiveFourierFeatures(full_config)
        
        # 階段 1：低頻
        stage1 = {'x': [1, 2], 'y': []}
        fourier.set_active_frequencies(stage1)
        assert fourier.out_dim == 4  # 2*(2+0)
        
        # 階段 2：中頻
        stage2 = {'x': [1, 2, 4], 'y': []}
        fourier.set_active_frequencies(stage2)
        assert fourier.out_dim == 6  # 2*(3+0)
        
        # 階段 3：全頻段
        fourier.set_active_frequencies(full_config)
        assert fourier.out_dim == 8  # 2*(4+0)
    
    def test_axis_fourier_integration_with_mlp(self):
        """測試與 MLP 網路集成"""
        config = {'x': [1, 2], 'y': [], 'z': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        # 構建簡單 MLP
        mlp = nn.Sequential(
            nn.Linear(fourier.out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 4)  # 輸出 u, v, w, p
        )
        
        # 完整前向傳播
        x = torch.randn(10, 3)
        features = fourier(x)
        output = mlp(features)
        
        assert output.shape == (10, 4)
        
        # 測試梯度
        loss = output.sum()
        loss.backward()


class TestEdgeCases:
    """邊界條件測試"""
    
    def test_single_axis(self):
        """測試單軸配置"""
        config = {'x': [1, 2, 4]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        x = torch.randn(8, 1)
        output = fourier(x)
        assert output.shape == (8, 6)
    
    def test_high_frequency_range(self):
        """測試高頻率範圍（K=16, 32）"""
        config = {'x': [1, 2, 4, 8, 16, 32]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        x = torch.randn(4, 1)
        output = fourier(x)
        assert output.shape == (4, 12)
    
    def test_large_batch(self):
        """測試大批量輸入"""
        config = {'x': [1, 2], 'y': [1], 'z': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        x = torch.randn(10000, 3)
        output = fourier(x)
        assert output.shape == (10000, 8)
    
    def test_invalid_config(self):
        """測試無效配置"""
        # 空字典
        with pytest.raises(ValueError):
            AxisSelectiveFourierFeatures({})
        
        # 負頻率
        with pytest.raises(ValueError):
            AxisSelectiveFourierFeatures({'x': [1, -2]})
    
    def test_dimension_mismatch(self):
        """測試維度不匹配"""
        config = {'x': [1], 'y': [1], 'z': [1]}
        fourier = AxisSelectiveFourierFeatures(config)
        
        # 輸入維度錯誤（應該是 3，給了 2）
        x = torch.randn(4, 2)
        
        with pytest.raises(RuntimeError):
            fourier(x)


# ========== 測試運行配置 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
