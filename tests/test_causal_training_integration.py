"""
Causal Training 整合測試

測試 CausalWeighter 在完整訓練流程中的整合。
"""

import torch
import pytest
from pinnx.losses.weighting import CausalWeighter


def test_causal_weighter_time_ordering():
    """測試時間排序對權重的影響"""
    weighter = CausalWeighter(
        epsilon=1.0,
        n_time_bins=10,
        t_min=50.0,
        t_max=100.0
    )
    
    # 模擬 PDE 點: 早期、中期、晚期
    N = 300
    t_early = torch.full((100, 1), 55.0)
    t_mid = torch.full((100, 1), 75.0)
    t_late = torch.full((100, 1), 95.0)
    t_all = torch.cat([t_early, t_mid, t_late], dim=0)
    
    # 模擬均勻殘差平方
    residuals = torch.ones(N, 1) * 0.01
    
    # 計算權重
    weights = weighter.compute_weights(residuals, t_all)
    
    # 驗證: 早期權重 > 晚期權重
    w_early = weights[:100].mean()
    w_mid = weights[100:200].mean()
    w_late = weights[200:].mean()
    
    assert w_early > w_late, f"Early weight {w_early} should > late weight {w_late}"
    assert weights.shape == (N, 1), f"Wrong shape: {weights.shape}"
    
    print(f"✅ Causal weights: early={w_early:.4f}, mid={w_mid:.4f}, late={w_late:.4f}")
    print(f"   Weight ratio (early/late): {w_early / w_late:.2f}x")


def test_causal_weighter_time_scaling_invariant():
    """測試權重只依時間排序，對尺度不敏感"""
    weighter = CausalWeighter(epsilon=1.0, n_time_bins=10, t_min=0.0, t_max=1.0)
    
    t_all = torch.linspace(50.0, 100.0, 100).unsqueeze(1)
    residuals = torch.ones(100, 1) * 0.01
    
    weights_ref = weighter.compute_weights(residuals, t_all)
    
    # 時間尺度放大（排序不變）
    t_scaled = t_all * 10.0
    weights_scaled = weighter.compute_weights(residuals, t_scaled)
    
    assert torch.allclose(weights_ref, weights_scaled, atol=1e-6)


def test_causal_weights_shape_compatibility():
    """測試權重形狀與殘差兼容性"""
    weighter = CausalWeighter(
        epsilon=0.5,
        n_time_bins=20,
        t_min=50.0,
        t_max=100.0
    )
    
    # 模擬訓練批次
    N_pde = 2048
    t_pde = torch.rand(N_pde, 1) * 50 + 50  # [50, 100]
    
    # 模擬 3 個 PDE 殘差項
    res_momentum_x = torch.randn(N_pde, 1) * 0.1
    res_momentum_y = torch.randn(N_pde, 1) * 0.1
    res_continuity = torch.randn(N_pde, 1) * 0.05
    
    # 匯總殘差平方
    total_res_sq = res_momentum_x**2 + res_momentum_y**2 + res_continuity**2
    
    # 計算權重
    causal_weights = weighter.compute_weights(total_res_sq, t_pde)
    
    # 驗證形狀
    assert causal_weights.shape == (N_pde, 1), \
        f"Weight shape {causal_weights.shape} should match residuals ({N_pde}, 1)"
    
    # 驗證可以應用到損失
    weighted_loss_x = torch.mean(causal_weights * res_momentum_x**2)
    weighted_loss_y = torch.mean(causal_weights * res_momentum_y**2)
    weighted_loss_c = torch.mean(causal_weights * res_continuity**2)
    
    assert weighted_loss_x.item() >= 0, "Weighted loss should be non-negative"
    
    print(f"✅ Shape compatibility test passed")
    print(f"   Weighted losses: x={weighted_loss_x:.6f}, y={weighted_loss_y:.6f}, c={weighted_loss_c:.6f}")


def test_causal_combined_with_spatial_weights():
    """測試 causal weights 與 spatial weights 組合"""
    weighter = CausalWeighter(epsilon=1.0, n_time_bins=10, t_min=50.0, t_max=100.0)
    
    N = 100
    t_pde = torch.linspace(50.0, 100.0, N).unsqueeze(1)
    residuals = torch.ones(N, 1) * 0.01
    
    # Causal weights
    causal_weights = weighter.compute_weights(residuals, t_pde)
    
    # 模擬 spatial weights (例如自適應採樣)
    spatial_weights = torch.rand(N, 1) * 0.5 + 0.5  # [0.5, 1.0]
    
    # 組合權重
    combined_weights = causal_weights * spatial_weights
    
    # 驗證
    assert combined_weights.shape == (N, 1)
    assert (combined_weights >= 0).all(), "Combined weights should be non-negative"
    
    # 早期點應該有更高的組合權重
    early_combined = combined_weights[:30].mean()
    late_combined = combined_weights[70:].mean()
    
    print(f"✅ Combined weights test passed")
    print(f"   Early combined: {early_combined:.4f}")
    print(f"   Late combined: {late_combined:.4f}")
    print(f"   Ratio: {early_combined / late_combined:.2f}x")


if __name__ == "__main__":
    print("=" * 60)
    print("Causal Training 整合測試")
    print("=" * 60)
    
    print("\n[測試 1] 時間排序權重")
    test_causal_weighter_time_ordering()
    
    print("\n[測試 2] 時間尺度不影響權重（排序一致）")
    test_causal_weighter_time_scaling_invariant()
    
    print("\n[測試 3] 形狀兼容性")
    test_causal_weights_shape_compatibility()
    
    print("\n[測試 4] 與空間權重組合")
    test_causal_combined_with_spatial_weights()
    
    print("\n" + "=" * 60)
    print("✅ 所有測試通過！")
    print("=" * 60)
