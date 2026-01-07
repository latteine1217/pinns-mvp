"""
CausalWeighterPerComponent 測試與使用範例

測試分量級因果權重計算（對齊 JAX-PI）
"""

import torch
import pytest
import sys
import os

# 添加專案路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinnx.losses.causal_weighter_v2 import (
    CausalWeighterPerComponent,
    create_causal_weighter
)


class TestCausalWeighterPerComponent:
    """測試分量級因果權重器"""
    
    def test_basic_functionality(self):
        """測試基本功能"""
        # 設定
        causal_weighter = CausalWeighterPerComponent(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        
        # 模擬 3 個分量的殘差
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.randn(n_points, 1),
            'momentum_y': torch.randn(n_points, 1),
            'continuity': torch.randn(n_points, 1)
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        # 計算因果權重
        gamma, component_gammas = causal_weighter.compute_component_weights(
            residual_dict, time_coords
        )
        
        # 驗證
        assert gamma.shape == (16,), f"Expected shape (16,), got {gamma.shape}"
        assert len(component_gammas) == 3, f"Expected 3 components, got {len(component_gammas)}"
        
        # 驗證 gamma 是所有分量的最小值
        all_gammas = torch.stack(list(component_gammas.values()))
        min_gamma = torch.min(all_gammas, dim=0)[0]
        assert torch.allclose(gamma, min_gamma, atol=1e-6), "Gamma should be minimum of all components"
        
        print("✅ Basic functionality test passed")
    
    def test_apply_weights_to_losses(self):
        """測試損失加權"""
        causal_weighter = CausalWeighterPerComponent(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        
        # 模擬殘差
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.randn(n_points, 1),
            'momentum_y': torch.randn(n_points, 1),
            'continuity': torch.randn(n_points, 1)
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        # 應用權重
        weighted_losses = causal_weighter.apply_weights_to_losses(
            residual_dict, time_coords, return_details=True
        )
        
        # 驗證
        assert 'momentum_x' in weighted_losses
        assert 'momentum_y' in weighted_losses
        assert 'continuity' in weighted_losses
        assert 'causal_gamma' in weighted_losses
        assert 'component_gammas' in weighted_losses
        
        # 驗證損失是標量
        for key in ['momentum_x', 'momentum_y', 'continuity']:
            assert weighted_losses[key].numel() == 1, f"{key} should be scalar"
        
        print("✅ Apply weights test passed")
    
    def test_causality_property(self):
        """測試因果性質：早期時間的權重應該更大"""
        causal_weighter = CausalWeighterPerComponent(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        
        # 創建均勻殘差（確保因果權重是遞減的）
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.ones(n_points, 1),
            'momentum_y': torch.ones(n_points, 1),
            'continuity': torch.ones(n_points, 1)
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        gamma, _ = causal_weighter.compute_component_weights(
            residual_dict, time_coords
        )
        
        # 驗證因果性：權重應該隨時間遞減
        for i in range(len(gamma) - 1):
            assert gamma[i] >= gamma[i+1] - 1e-6, \
                f"Causal weights should be non-increasing: gamma[{i}]={gamma[i]:.6f}, gamma[{i+1}]={gamma[i+1]:.6f}"
        
        print("✅ Causality property test passed")
    
    def test_min_strategy(self):
        """測試最小值策略"""
        causal_weighter = CausalWeighterPerComponent(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        
        # 創建不同大小的殘差（momentum_x 最大）
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.ones(n_points, 1) * 10.0,  # 大殘差
            'momentum_y': torch.ones(n_points, 1) * 1.0,   # 中殘差
            'continuity': torch.ones(n_points, 1) * 0.1,   # 小殘差
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        gamma, component_gammas = causal_weighter.compute_component_weights(
            residual_dict, time_coords
        )
        
        # 驗證：gamma 應該接近最小分量（continuity）的 gamma
        # 因為 continuity 的殘差最小，其 gamma 應該最大，但因為取 min，最終會受其他分量限制
        gamma_continuity = component_gammas['continuity']
        gamma_momentum_x = component_gammas['momentum_x']
        
        # momentum_x 有大殘差，所以其 gamma 應該更小（權重衰減更快）
        # 最終的 gamma 應該是所有分量的最小值
        for key in component_gammas:
            assert torch.all(gamma <= component_gammas[key] + 1e-6), \
                f"Final gamma should be <= {key} gamma"
        
        print("✅ Min strategy test passed")
    
    def test_diagnostic_info(self):
        """測試診斷信息"""
        causal_weighter = CausalWeighterPerComponent(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.randn(n_points, 1),
            'momentum_y': torch.randn(n_points, 1),
            'continuity': torch.randn(n_points, 1)
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        diagnostics = causal_weighter.get_diagnostic_info(
            residual_dict, time_coords
        )
        
        # 驗證診斷信息
        assert 'causal_gamma' in diagnostics
        assert 'component_gammas' in diagnostics
        assert 'min_weight' in diagnostics
        assert 'max_weight' in diagnostics
        assert 'weight_variance' in diagnostics
        assert 'num_chunks' in diagnostics
        
        # 驗證各分量的最小權重
        for key in residual_dict.keys():
            assert f'{key}_min_weight' in diagnostics
        
        print("✅ Diagnostic info test passed")
    
    def test_factory_function(self):
        """測試工廠函數"""
        # 創建因果權重器（v2，對齊 JAX-PI）
        weighter = create_causal_weighter(
            causal_tol=1.0,
            num_chunks=16,
            device='cpu'
        )
        assert isinstance(weighter, CausalWeighterPerComponent)
        
        # 驗證基本功能
        n_points = 1024
        residual_dict = {
            'momentum_x': torch.randn(n_points, 1),
            'momentum_y': torch.randn(n_points, 1),
        }
        time_coords = torch.linspace(0, 1, n_points).unsqueeze(1)
        
        # 應該能正常計算權重
        gamma, _ = weighter.compute_component_weights(residual_dict, time_coords)
        assert gamma.shape[0] > 0, "Should compute valid causal weights"
        
        print("✅ Factory function test passed")


def example_usage():
    """使用範例：展示如何在訓練中使用分量級因果權重"""
    print("\n" + "="*70)
    print("📚 使用範例：分量級因果權重器")
    print("="*70 + "\n")
    
    # 1. 創建因果權重器
    causal_weighter = CausalWeighterPerComponent(
        causal_tol=1.0,    # JAX-PI 默認值
        num_chunks=16,     # JAX-PI Kolmogorov Flow 默認值
        device='cpu'
    )
    
    # 2. 模擬訓練中的殘差
    n_points = 2048
    residual_dict = {
        'momentum_x': torch.randn(n_points, 1) * 2.0,  # x-momentum equation
        'momentum_y': torch.randn(n_points, 1) * 1.5,  # y-momentum equation
        'continuity': torch.randn(n_points, 1) * 0.5,  # continuity equation
    }
    time_coords = torch.linspace(0, 10, n_points).unsqueeze(1)  # t ∈ [0, 10]
    
    # 3. 計算分量級因果權重
    gamma, component_gammas = causal_weighter.compute_component_weights(
        residual_dict, time_coords
    )
    
    print("📊 因果權重統計:")
    print(f"   最終 gamma 形狀: {gamma.shape}")
    print(f"   gamma 範圍: [{gamma.min():.6f}, {gamma.max():.6f}]")
    print(f"   gamma 均值: {gamma.mean():.6f}")
    print()
    
    print("📊 各分量的因果權重:")
    for key, comp_gamma in component_gammas.items():
        print(f"   {key:15s}: min={comp_gamma.min():.6f}, max={comp_gamma.max():.6f}, mean={comp_gamma.mean():.6f}")
    print()
    
    # 4. 應用權重到損失
    weighted_losses = causal_weighter.apply_weights_to_losses(
        residual_dict, time_coords, return_details=True
    )
    
    print("📊 加權後的損失:")
    for key in ['momentum_x', 'momentum_y', 'continuity']:
        print(f"   {key:15s}: {weighted_losses[key]:.6f}")
    print()
    
    # 5. 獲取診斷信息
    diagnostics = causal_weighter.get_diagnostic_info(
        residual_dict, time_coords
    )
    
    print("🔍 診斷信息:")
    print(f"   最小權重: {diagnostics['min_weight']:.6f}")
    print(f"   最大權重: {diagnostics['max_weight']:.6f}")
    print(f"   權重方差: {diagnostics['weight_variance']:.6f}")
    print(f"   分塊數量: {diagnostics['num_chunks']}")
    print()
    
    # 6. 與 JAX-PI 的對比
    print("🔄 與 JAX-PI 的對比:")
    print("   ✅ 對每個分量獨立計算因果權重")
    print("   ✅ 取所有分量的最小值（最保守策略）")
    print("   ✅ 使用相同的數學形式: gamma = exp(-tol * M @ chunk_means)")
    print("   ✅ 支援相同的參數: causal_tol, num_chunks")
    print()
    
    print("="*70)
    print("✅ 範例完成！")
    print("="*70)


def jaxpi_comparison_demo():
    """演示與 JAX-PI 的對比"""
    print("\n" + "="*70)
    print("🔬 JAX-PI 對比演示")
    print("="*70 + "\n")
    
    # JAX-PI 方式（模擬）
    print("📘 JAX-PI 方式:")
    print("""
    # JAX-PI Code (Line 92-115)
    ru_l = jnp.mean(ru_pred**2, axis=1)  # [num_chunks]
    rv_l = jnp.mean(rv_pred**2, axis=1)
    rc_l = jnp.mean(rc_pred**2, axis=1)
    
    ru_gamma = jnp.exp(-tol * (M @ ru_l))
    rv_gamma = jnp.exp(-tol * (M @ rv_l))
    rc_gamma = jnp.exp(-tol * (M @ rc_l))
    
    gamma = jnp.vstack([ru_gamma, rv_gamma, rc_gamma]).min(0)  # 取最小值
    
    ru_loss = jnp.mean(gamma * ru_l)
    rv_loss = jnp.mean(gamma * rv_l)
    rc_loss = jnp.mean(gamma * rc_l)
    """)
    
    print("\n📗 我們的方式:")
    print("""
    # PyTorch Implementation
    causal_weighter = CausalWeighterPerComponent(causal_tol=1.0, num_chunks=16)
    
    residual_dict = {
        'momentum_x': ru_residual,
        'momentum_y': rv_residual,
        'continuity': rc_residual
    }
    
    weighted_losses = causal_weighter.apply_weights_to_losses(
        residual_dict, time_coords
    )
    
    # weighted_losses = {
    #     'momentum_x': weighted_ru_loss,
    #     'momentum_y': weighted_rv_loss,
    #     'continuity': weighted_rc_loss
    # }
    """)
    
    print("\n✅ 核心對齊:")
    print("   1. 分量級計算: 對 ru, rv, rc 獨立計算 gamma")
    print("   2. 取最小值策略: gamma = min(ru_gamma, rv_gamma, rc_gamma)")
    print("   3. 相同數學形式: exp(-tol * M @ chunk_means)")
    print("   4. 相同參數設置: causal_tol=1.0, num_chunks=16")
    print()
    
    print("="*70)


if __name__ == '__main__':
    # 運行測試
    print("🧪 開始測試...\n")
    
    test_suite = TestCausalWeighterPerComponent()
    test_suite.test_basic_functionality()
    test_suite.test_apply_weights_to_losses()
    test_suite.test_causality_property()
    test_suite.test_min_strategy()
    test_suite.test_diagnostic_info()
    test_suite.test_factory_function()
    
    print("\n✅ 所有測試通過！\n")
    
    # 運行使用範例
    example_usage()
    
    # JAX-PI 對比演示
    jaxpi_comparison_demo()
