"""
模型模組測試
測試 Fourier MLP 和模型包裝器
"""

import numpy as np
import torch
import pytest
import sys
import os
from typing import Dict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.models.fourier_mlp import FourierFeatures, PINNNet, MultiScalePINNNet
from pinnx.models.wrappers import ScaledPINNWrapper, EnsemblePINNWrapper, PhysicsConstrainedWrapper, MultiHeadWrapper


class TestFourierFeatures:
    """測試 Fourier 特徵模組"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_fourier_features_basic(self):
        """測試 Fourier 特徵基本功能"""
        ff = FourierFeatures(in_dim=2, m=16, sigma=1.0).to(self.device)
        
        # 測試輸入
        x = torch.randn(10, 2, device=self.device)
        
        # 前向傳播
        features = ff(x)
        
        # 檢查輸出維度
        assert features.shape == (10, 32)  # 2*m
        assert not torch.isnan(features).any()
        assert not torch.isinf(features).any()
    
    def test_fourier_features_parameters(self):
        """測試不同參數的 Fourier 特徵"""
        x = torch.randn(5, 3, device=self.device)
        
        # 測試不同的 m 值
        ff1 = FourierFeatures(in_dim=3, m=8, sigma=1.0).to(self.device)
        ff2 = FourierFeatures(in_dim=3, m=16, sigma=1.0).to(self.device)
        
        feat1 = ff1(x)
        feat2 = ff2(x)
        
        assert feat1.shape == (5, 16)  # 2*8
        assert feat2.shape == (5, 32)  # 2*16
        
        # 測試不同的 sigma 值
        ff3 = FourierFeatures(in_dim=3, m=8, sigma=2.0).to(self.device)
        feat3 = ff3(x)
        
        # 不同 sigma 應該產生不同的特徵
        assert not torch.allclose(feat1, feat3)
    
    def test_fourier_features_reproducibility(self):
        """測試 Fourier 特徵的再現性"""
        torch.manual_seed(123)
        ff1 = FourierFeatures(in_dim=2, m=8, sigma=1.0).to(self.device)
        
        torch.manual_seed(123)
        ff2 = FourierFeatures(in_dim=2, m=8, sigma=1.0).to(self.device)
        
        x = torch.randn(5, 2, device=self.device)
        
        # 相同種子應該產生相同的特徵
        feat1 = ff1(x)
        feat2 = ff2(x)
        
        assert torch.allclose(feat1, feat2)


class TestPINNNet:
    """測試 PINN 網路模組"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_pinn_net_basic(self):
        """測試 PINN 網路基本功能"""
        net = PINNNet(in_dim=3, out_dim=4, width=64, depth=3).to(self.device)
        
        # 測試輸入 (時間 + 空間)
        x = torch.randn(20, 3, device=self.device)
        
        # 前向傳播
        output = net(x)
        
        # 檢查輸出維度
        assert output.shape == (20, 4)  # [u, v, p, S]
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_pinn_net_gradients(self):
        """測試 PINN 網路梯度計算"""
        net = PINNNet(in_dim=2, out_dim=3, width=32, depth=2).to(self.device)
        
        # 需要梯度的輸入
        x = torch.randn(10, 2, requires_grad=True, device=self.device)
        
        output = net(x)
        
        # 計算梯度
        grad_outputs = torch.ones_like(output[:, 0])
        gradients = torch.autograd.grad(
            outputs=output[:, 0], 
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )[0]
        
        assert gradients.shape == x.shape
        assert not torch.isnan(gradients).any()
    
    def test_pinn_net_different_configs(self):
        """測試不同配置的 PINN 網路"""
        configs = [
            {"width": 32, "depth": 2, "fourier_m": 8},
            {"width": 64, "depth": 4, "fourier_m": 16},
            {"width": 128, "depth": 3, "fourier_m": 32}
        ]
        
        x = torch.randn(5, 2, device=self.device)
        
        for config in configs:
            net = PINNNet(in_dim=2, out_dim=3, **config).to(self.device)
            output = net(x)
            
            assert output.shape == (5, 3)
            assert not torch.isnan(output).any()
            
            # 檢查參數數量
            param_count = sum(p.numel() for p in net.parameters())
            assert param_count > 0
            print(f"配置 {config}: 參數數量 = {param_count}")

    def test_fourier_inverse_normalizer_standard(self):
        """標準化座標應在 Fourier 前還原為物理座標"""
        net = PINNNet(
            in_dim=2,
            out_dim=2,
            width=16,
            depth=1,
            use_fourier=True,
            use_input_projection=False
        ).to(self.device)
        net.eval()
        torch.manual_seed(0)
        x_phys = torch.randn(32, 2, device=self.device) * 5.0 + 2.0
        mean = x_phys.mean(dim=0, keepdim=True)
        std = x_phys.std(dim=0, keepdim=True).clamp(min=1e-4)
        x_norm = (x_phys - mean) / std
        metadata = {
            'norm_type': 'standard',
            'mean': mean.detach().cpu(),
            'std': std.detach().cpu(),
            'feature_range': torch.tensor([-1.0, 1.0])
        }
        net.configure_fourier_input(metadata)
        captured: Dict[str, torch.Tensor] = {}
        handle = net.fourier.register_forward_hook(lambda m, inp, out: captured.setdefault('input', inp[0].detach().clone()))
        try:
            net(x_norm)
        finally:
            handle.remove()
        assert 'input' in captured
        assert torch.allclose(captured['input'], x_phys, atol=1e-5)

    def test_fourier_inverse_normalizer_minmax(self):
        """MinMax 正規化應正確逆變換"""
        net = PINNNet(
            in_dim=2,
            out_dim=2,
            width=16,
            depth=1,
            use_fourier=True,
            use_input_projection=False
        ).to(self.device)
        net.eval()
        torch.manual_seed(1)
        x_phys = torch.rand(32, 2, device=self.device) * 4.0 - 1.0
        data_min = x_phys.min(dim=0, keepdim=True).values
        data_max = x_phys.max(dim=0, keepdim=True).values
        data_range = (data_max - data_min).clamp(min=1e-4)
        lo, hi = -1.0, 1.0
        x_norm = (x_phys - data_min) / data_range
        x_norm = x_norm * (hi - lo) + lo
        metadata = {
            'norm_type': 'minmax',
            'data_min': data_min.detach().cpu(),
            'data_range': data_range.detach().cpu(),
            'feature_range': torch.tensor([lo, hi])
        }
        net.configure_fourier_input(metadata)
        captured: Dict[str, torch.Tensor] = {}
        handle = net.fourier.register_forward_hook(lambda m, inp, out: captured.setdefault('input', inp[0].detach().clone()))
        try:
            net(x_norm)
        finally:
            handle.remove()
        assert 'input' in captured
        assert torch.allclose(captured['input'], x_phys, atol=1e-5)


class TestMultiHeadWrapper:
    """測試多頭包裝器"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_multihead_wrapper_basic(self):
        """測試多頭包裝器基本功能"""
        base_net = PINNNet(in_dim=2, out_dim=3, width=32, depth=2).to(self.device)
        
        # 多頭配置: 速度場 (2) + 壓力場 (1) + 源項 (1)
        head_configs = [
            {"name": "velocity", "dim": 2, "activation": "tanh"},
            {"name": "pressure", "dim": 1, "activation": None},
            {"name": "source", "dim": 1, "activation": "relu"}
        ]
        
        wrapper = MultiHeadWrapper(base_net, head_configs).to(self.device)
        
        x = torch.randn(10, 2, device=self.device)
        outputs = wrapper(x)
        
        assert isinstance(outputs, dict)
        assert "velocity" in outputs
        assert "pressure" in outputs
        assert "source" in outputs
        
        assert outputs["velocity"].shape == (10, 2)
        assert outputs["pressure"].shape == (10, 1)
        assert outputs["source"].shape == (10, 1)
    
    def test_multihead_wrapper_gradients(self):
        """測試多頭包裝器梯度"""
        base_net = PINNNet(in_dim=2, out_dim=2, width=32, depth=2).to(self.device)
        head_configs = [{"name": "field", "dim": 2, "activation": None}]
        
        wrapper = MultiHeadWrapper(base_net, head_configs).to(self.device)
        
        x = torch.randn(5, 2, requires_grad=True, device=self.device)
        outputs = wrapper(x)
        
        # 計算場的散度
        u, v = outputs["field"][:, 0], outputs["field"][:, 1]
        
        dudx = torch.autograd.grad(u.sum(), x, create_graph=True)[0][:, 0]
        dvdy = torch.autograd.grad(v.sum(), x, create_graph=True)[0][:, 1]
        
        divergence = dudx + dvdy
        
        assert divergence.shape == (5,)
        assert not torch.isnan(divergence).any()


class TestEnsemblePINNWrapper:
    """測試集成包裝器"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_ensemble_wrapper_basic(self):
        """測試集成包裝器基本功能"""
        # 建立多個模型
        models = [
            PINNNet(in_dim=2, out_dim=3, width=32, depth=2).to(self.device)
            for _ in range(3)
        ]
        
        ensemble = EnsemblePINNWrapper(models).to(self.device)
        
        x = torch.randn(8, 2, device=self.device)
        
        # 測試平均預測
        mean_pred = ensemble(x, mode='mean')
        assert mean_pred.shape == (8, 3)
        
        # 測試所有預測
        all_preds = ensemble(x, mode='all')
        assert len(all_preds) == 3
        assert all(pred.shape == (8, 3) for pred in all_preds)
        
        # 測試統計
        stats = ensemble(x, mode='stats')
        assert 'mean' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        
        assert stats['mean'].shape == (8, 3)
        assert stats['std'].shape == (8, 3)
    
    def test_ensemble_uncertainty(self):
        """測試集成不確定性量化"""
        # 建立具有不同初始化的模型
        models = []
        for i in range(5):
            torch.manual_seed(i)
            model = PINNNet(in_dim=2, out_dim=1, width=16, depth=2).to(self.device)
            models.append(model)
        
        ensemble = EnsemblePINNWrapper(models).to(self.device)
        
        x = torch.randn(10, 2, device=self.device)
        stats = ensemble(x, mode='stats')
        
        # 不確定性應該為正值
        assert (stats['std'] >= 0).all()
        
        # 不同模型應該產生不同預測 (有不確定性)
        assert stats['std'].mean() > 0


class TestScaledPINNWrapper:
    """測試 ScaledPINNWrapper"""
    
    def setup_method(self):
        """設置測試環境"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
    
    def test_residual_wrapper_basic(self):
        """測試殘差包裝器基本功能"""
        base_net = PINNNet(in_dim=2, out_dim=3, width=32, depth=2).to(self.device)
        
        wrapper = ScaledPINNWrapper(
            base_model=base_net,
            variable_names=['u', 'v', 'p']  # 正確的參數
        ).to(self.device)
        
        x = torch.randn(6, 2, device=self.device)
        output = wrapper(x)
        
        assert output.shape == (6, 3)
        assert not torch.isnan(output).any()
    
    def test_residual_vs_base(self):
        """測試 ScaledPINNWrapper 的行為"""
        torch.manual_seed(42)
        base_net = PINNNet(in_dim=2, out_dim=2, width=32, depth=3).to(self.device)
        
        residual_wrapper = ScaledPINNWrapper(
            base_model=base_net,
            variable_names=['u', 'v']
        ).to(self.device)
        
        x = torch.randn(5, 2, device=self.device)
        
        base_output = base_net(x)
        wrapper_output = residual_wrapper(x)
        
        # 無尺度器時，包裝器應該產生相同的輸出
        assert torch.allclose(base_output, wrapper_output, atol=1e-6)
        assert wrapper_output.shape == (5, 2)


def test_models_integration():
    """測試模型模組整合"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=== 模型模組整合測試 ===")
    
    # 建立完整的集成模型 (直接使用基礎模型，不使用 MultiHeadWrapper)
    base_models = [
        PINNNet(in_dim=3, out_dim=4, width=64, depth=3).to(device)
        for _ in range(3)
    ]
    
    # 建立集成 (EnsemblePINNWrapper 期待返回張量的模型，不是字典)
    ensemble = EnsemblePINNWrapper(base_models).to(device)
    
    # 測試資料 (時間 + 2D 空間)
    x = torch.randn(15, 3, device=device)
    
    # 集成統計
    stats = ensemble(x, mode='stats')
    
    print(f"集成統計鍵: {list(stats.keys())}")
    print(f"統計輸出形狀:")
    print(f"  mean: {stats['mean'].shape}")
    print(f"  std: {stats['std'].shape}")
    print(f"  var: {stats['var'].shape}")
    
    # 驗證輸出
    assert 'mean' in stats and 'std' in stats and 'var' in stats
    assert 'uncertainty' in stats and 'min' in stats and 'max' in stats
    
    # 驗證形狀一致性
    assert stats['mean'].shape == stats['std'].shape
    assert stats['mean'].shape == stats['var'].shape
    assert stats['mean'].shape == stats['uncertainty'].shape
    
    # 驗證統計特性
    assert stats['mean'].shape[0] == 15  # 批次大小
    assert (stats['std'] >= 0).all()  # 標準差非負
    assert not torch.isnan(stats['mean']).any()
    assert not torch.isnan(stats['std']).any()
    
    print("✅ 模型整合測試通過")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
