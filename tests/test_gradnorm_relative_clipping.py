"""
測試 GradNorm 相對權重裁剪機制

驗證重點:
1. 裁剪邊界是相對於 initial_weights 的比例（非絕對值）
2. 不同 initial_weight 值應得到不同的絕對邊界
3. 權重更新時正確使用相對邊界進行裁剪
"""

import torch
import pytest
from torch import nn
from pinnx.losses.weighting import GradNormWeighter


class DummyModel(nn.Module):
    """用於測試的假模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.fc(x)


class TestGradNormRelativeClipping:
    """測試 GradNorm 相對權重裁剪"""

    def test_relative_boundaries_calculation(self):
        """測試相對邊界計算正確性"""
        # 設定: 不同的 initial_weights
        loss_names = ["u_ic", "ru", "momentum"]
        init_weights = {
            "u_ic": 100.0,      # 大值
            "ru": 1.0,          # 單位值
            "momentum": 0.5     # 小值
        }
        
        # 創建 GradNormWeighter (ratio: 0.1x ~ 10x)
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5,
            min_weight=0.1,   # 相對比例
            max_weight=10.0   # 相對比例
        )
        
        # 驗證絕對邊界
        assert weighter.min_weight_abs["u_ic"] == pytest.approx(10.0)    # 100 * 0.1
        assert weighter.max_weight_abs["u_ic"] == pytest.approx(1000.0)  # 100 * 10
        
        assert weighter.min_weight_abs["ru"] == pytest.approx(0.1)       # 1 * 0.1
        assert weighter.max_weight_abs["ru"] == pytest.approx(10.0)      # 1 * 10
        
        assert weighter.min_weight_abs["momentum"] == pytest.approx(0.05)   # 0.5 * 0.1
        assert weighter.max_weight_abs["momentum"] == pytest.approx(5.0)    # 0.5 * 10
        
        # 驗證所有項目都有 10x 的相對範圍
        for name in loss_names:
            ratio = weighter.max_weight_abs[name] / weighter.min_weight_abs[name]
            assert ratio == pytest.approx(100.0), f"{name} 的相對範圍應為 100x (10/0.1)"

    def test_initial_weights_clamped_correctly(self):
        """測試初始權重使用相對邊界裁剪"""
        loss_names = ["term1", "term2"]
        init_weights = {
            "term1": 200.0,  # 超過預設上界 (200 > 10)
            "term2": 0.01    # 低於預設下界 (0.01 < 0.1)
        }
        
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5,
            min_weight=0.1,
            max_weight=10.0
        )
        
        # term1 不應被裁剪到 10.0 (舊 bug)，應保持在 [20.0, 2000.0] 範圍內
        assert weighter.weights["term1"].item() == pytest.approx(200.0)
        
        # term2 不應被裁剪到 0.1 (舊 bug)，應裁剪到 [0.001, 0.1] 範圍內
        assert weighter.weights["term2"].item() == pytest.approx(0.01)

    def test_weight_clipping_respects_relative_boundaries(self):
        """測試權重裁剪時遵守相對邊界"""
        loss_names = ["loss1", "loss2"]
        init_weights = {"loss1": 50.0, "loss2": 2.0}
        
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5,
            min_weight=0.2,   # 20% 下界
            max_weight=5.0    # 500% 上界
        )
        
        # 驗證邊界計算
        # loss1: 50.0 * [0.2, 5.0] = [10.0, 250.0]
        assert weighter.min_weight_abs["loss1"] == pytest.approx(10.0)
        assert weighter.max_weight_abs["loss1"] == pytest.approx(250.0)
        
        # loss2: 2.0 * [0.2, 5.0] = [0.4, 10.0]
        assert weighter.min_weight_abs["loss2"] == pytest.approx(0.4)
        assert weighter.max_weight_abs["loss2"] == pytest.approx(10.0)
        
        # 手動測試裁剪功能
        # 超過上界 → 應裁剪到上界
        weighter.weights["loss1"] = torch.tensor(300.0)
        clamped = torch.clamp(
            weighter.weights["loss1"],
            min=weighter.min_weight_abs["loss1"],
            max=weighter.max_weight_abs["loss1"]
        )
        assert clamped.item() == pytest.approx(250.0), "超過上界應裁剪到 250.0"
        
        # 低於下界 → 應裁剪到下界
        weighter.weights["loss2"] = torch.tensor(0.1)
        clamped = torch.clamp(
            weighter.weights["loss2"],
            min=weighter.min_weight_abs["loss2"],
            max=weighter.max_weight_abs["loss2"]
        )
        assert clamped.item() == pytest.approx(0.4), "低於下界應裁剪到 0.4"

    def test_backward_compatibility_default_values(self):
        """測試向後相容性: 預設值 (0.1, 10.0) 對於 init_weights=1.0 行為一致"""
        loss_names = ["pde", "bc"]
        init_weights = {"pde": 1.0, "bc": 1.0}
        
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5
            # 使用預設 min_weight=0.1, max_weight=10.0
        )
        
        # 對於 init_weights=1.0，相對邊界 = 絕對邊界
        assert weighter.min_weight_abs["pde"] == pytest.approx(0.1)
        assert weighter.max_weight_abs["pde"] == pytest.approx(10.0)
        assert weighter.min_weight_abs["bc"] == pytest.approx(0.1)
        assert weighter.max_weight_abs["bc"] == pytest.approx(10.0)

    def test_reset_weights_uses_relative_boundaries(self):
        """測試 reset_weights() 使用相對邊界"""
        loss_names = ["term"]
        init_weights = {"term": 80.0}
        
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5,
            min_weight=0.1,
            max_weight=10.0
        )
        
        # 手動修改權重到極端值
        weighter.weights["term"] = torch.tensor(5000.0)
        
        # 重置權重
        weighter.reset_weights()
        
        # 驗證重置後的權重在相對邊界內 [8.0, 800.0]
        reset_value = weighter.weights["term"].item()
        assert reset_value >= 8.0, "重置後應 >= min_weight_abs"
        assert reset_value <= 800.0, "重置後應 <= max_weight_abs"
        assert reset_value == pytest.approx(80.0), "應重置到 initial_weight"

    def test_extreme_initial_weights(self):
        """測試極端 initial_weight 值"""
        loss_names = ["tiny", "huge"]
        init_weights = {
            "tiny": 1e-3,   # 極小
            "huge": 1e3     # 極大
        }
        
        model = DummyModel()
        weighter = GradNormWeighter(
            model=model,
            loss_names=loss_names,
            initial_weights=init_weights,
            alpha=1.5,
            min_weight=0.1,
            max_weight=10.0
        )
        
        # 驗證邊界計算
        assert weighter.min_weight_abs["tiny"] == pytest.approx(1e-4)   # 1e-3 * 0.1
        assert weighter.max_weight_abs["tiny"] == pytest.approx(1e-2)   # 1e-3 * 10
        
        assert weighter.min_weight_abs["huge"] == pytest.approx(1e2)    # 1e3 * 0.1
        assert weighter.max_weight_abs["huge"] == pytest.approx(1e4)    # 1e3 * 10
        
        # 驗證相對範圍一致
        tiny_ratio = weighter.max_weight_abs["tiny"] / weighter.min_weight_abs["tiny"]
        huge_ratio = weighter.max_weight_abs["huge"] / weighter.min_weight_abs["huge"]
        assert tiny_ratio == pytest.approx(huge_ratio)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
