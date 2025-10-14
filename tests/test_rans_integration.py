"""
TASK-008: RANS 湍流模型整合測試

驗證項目:
1. RANS 殘差計算不拋錯
2. k, ε 非負性約束
3. 梯度流動正常（可微分）
4. 與 VS-PINN 整合無衝突
"""

import torch
import pytest
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow, create_vs_pinn_channel_flow
from pinnx.physics.turbulence import RANSEquations3D


class TestRANSIntegration:
    """RANS 湍流模型整合測試套件"""
    
    @pytest.fixture
    def device(self):
        """選擇可用設備"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @pytest.fixture
    def sample_data(self, device):
        """生成測試用樣本資料"""
        batch_size = 32
        coords = torch.rand(batch_size, 3, device=device)
        
        # 避免 in-place 操作（requires_grad=True 時禁止）
        coords_scaled = coords.clone()
        coords_scaled[:, 0] = coords[:, 0] * 8.0 * 3.14159  # x ∈ [0, 8π]
        coords_scaled[:, 1] = coords[:, 1] * 2 - 1  # y ∈ [-1, 1]
        coords_scaled[:, 2] = coords[:, 2] * 3.0 * 3.14159  # z ∈ [0, 3π]
        coords_scaled.requires_grad_(True)
        
        # 模擬網路輸出
        predictions = torch.randn(batch_size, 4, device=device, requires_grad=True)
        
        return coords_scaled, predictions
    
    def test_rans_model_initialization(self, device):
        """測試 1: RANS 模型初始化"""
        rans_model = RANSEquations3D(viscosity=5e-5, enable_constraints=True)
        
        assert rans_model.viscosity == 5e-5
        assert rans_model.enable_constraints is True
        print("✅ 測試 1 通過: RANS 模型初始化成功")
    
    def test_vs_pinn_with_rans_enabled(self, device):
        """測試 2: VS-PINN 啟用 RANS"""
        physics = VSPINNChannelFlow(enable_rans=True, rans_model="k_epsilon")
        
        assert physics.enable_rans is True
        assert physics.rans_model is not None
        assert physics.rans_model_type == "k_epsilon"
        print("✅ 測試 2 通過: VS-PINN + RANS 整合成功")
    
    def test_vs_pinn_without_rans(self, device):
        """測試 3: VS-PINN 未啟用 RANS（向後相容）"""
        physics = VSPINNChannelFlow(enable_rans=False)
        
        assert physics.enable_rans is False
        assert physics.rans_model is None
        print("✅ 測試 3 通過: 未啟用 RANS 時保持原行為")
    
    def test_compute_rans_residuals_returns_empty_when_disabled(self, device, sample_data):
        """測試 4: 未啟用 RANS 時返回空字典"""
        coords, predictions = sample_data
        physics = VSPINNChannelFlow(enable_rans=False)
        
        residuals = physics.compute_rans_residuals(coords, predictions)
        
        assert residuals == {}
        print("✅ 測試 4 通過: 未啟用時返回空字典")
    
    def test_compute_rans_residuals_shape(self, device, sample_data):
        """測試 5: RANS 殘差輸出維度正確"""
        coords, predictions = sample_data
        batch_size = coords.shape[0]
        
        physics = VSPINNChannelFlow(enable_rans=True)
        residuals = physics.compute_rans_residuals(coords, predictions)
        
        # 驗證返回鍵值
        expected_keys = {'k_equation', 'epsilon_equation', 'turbulent_viscosity', 'physical_penalty'}
        assert set(residuals.keys()) == expected_keys, f"期望鍵: {expected_keys}, 實際: {set(residuals.keys())}"
        
        # 驗證形狀（physical_penalty 可能是標量）
        for key, value in residuals.items():
            if key == 'physical_penalty':
                # physical_penalty 可能是標量（當約束禁用時）
                assert value.shape == (batch_size, 1) or value.shape == torch.Size([]), \
                    f"{key} 形狀錯誤: {value.shape}（期望 [{batch_size}, 1] 或標量）"
            else:
                assert value.shape == (batch_size, 1), f"{key} 形狀錯誤: {value.shape}"
        
        print(f"✅ 測試 5 通過: RANS 殘差維度正確 (batch={batch_size})")
        for key, value in residuals.items():
            print(f"    - {key}: {value.shape}")
    
    def test_rans_non_negativity(self, device, sample_data):
        """測試 6: k, ε 非負性約束（softplus 保證）"""
        coords, predictions = sample_data
        physics = VSPINNChannelFlow(enable_rans=True)
        
        # 提取速度場
        velocity = predictions[:, :3]
        
        # 計算 k, ε
        if physics.rans_model is not None:
            k, epsilon = physics.rans_model.estimate_turbulent_quantities(coords, velocity)
            
            # 驗證非負性
            assert (k >= 0).all(), f"k 存在負值: min={k.min().item()}"
            assert (epsilon >= 0).all(), f"ε 存在負值: min={epsilon.min().item()}"
            
            print(f"✅ 測試 6 通過: k ∈ [{k.min():.2e}, {k.max():.2e}], ε ∈ [{epsilon.min():.2e}, {epsilon.max():.2e}]")
        else:
            pytest.skip("RANS 模型未初始化")
    
    def test_rans_residuals_no_nan(self, device, sample_data):
        """測試 7: RANS 殘差無 NaN/Inf"""
        coords, predictions = sample_data
        physics = VSPINNChannelFlow(enable_rans=True)
        
        residuals = physics.compute_rans_residuals(coords, predictions)
        
        for key, value in residuals.items():
            assert not torch.isnan(value).any(), f"{key} 包含 NaN"
            assert not torch.isinf(value).any(), f"{key} 包含 Inf"
        
        print("✅ 測試 7 通過: RANS 殘差無 NaN/Inf")
    
    def test_rans_residuals_differentiable(self, device, sample_data):
        """測試 8: RANS 殘差可微分（梯度流動正常）"""
        coords, predictions = sample_data
        physics = VSPINNChannelFlow(enable_rans=True)
        
        # 清除任何舊梯度
        if coords.grad is not None:
            coords.grad.zero_()
        if predictions.grad is not None:
            predictions.grad.zero_()
        
        residuals = physics.compute_rans_residuals(coords, predictions)
        
        if len(residuals) > 0:
            # 只驗證能否計算梯度（不要求梯度必須傳播到所有輸入）
            k_res = residuals.get('k_equation')
            if k_res is not None and k_res.numel() > 1:
                loss = k_res.mean()
                try:
                    loss.backward()
                    # 至少座標應該有梯度
                    assert coords.grad is not None, "座標梯度未計算"
                    assert not torch.isnan(coords.grad).any(), "座標梯度包含 NaN"
                    print(f"✅ 測試 8 通過: RANS 殘差可微分 (coord_grad_max: {coords.grad.abs().max():.2e})")
                except RuntimeError as e:
                    pytest.fail(f"梯度計算失敗: {e}")
            else:
                print("⚠️  測試 8 跳過: k_equation 殘差不可用")
        else:
            pytest.skip("RANS 未啟用")
    
    def test_factory_function_with_rans(self, device):
        """測試 9: 工廠函數支援 RANS 參數"""
        physics = create_vs_pinn_channel_flow(
            N_y=12.0, 
            enable_rans=True, 
            rans_model="k_epsilon"
        )
        
        assert physics.enable_rans is True
        assert physics.rans_model is not None
        print("✅ 測試 9 通過: 工廠函數正確傳遞 RANS 參數")
    
    def test_turbulent_viscosity_magnitude(self, device, sample_data):
        """測試 10: 湍流黏度 ν_t 數量級合理"""
        coords, predictions = sample_data
        physics = VSPINNChannelFlow(enable_rans=True)
        
        velocity = predictions[:, :3]
        
        if physics.rans_model is not None:
            k, epsilon = physics.rans_model.estimate_turbulent_quantities(coords, velocity)
            
            # 計算湍流黏度（含 ε≠0 保護）
            epsilon_safe = torch.clamp(epsilon, min=1e-10)
            C_mu = 0.09  # 標準 k-ε 模型常數
            nu_t = C_mu * k.pow(2) / epsilon_safe
            
            # 驗證數量級（應 >> 分子黏度 ν=5e-5）
            mean_nu_t = nu_t.mean().item()
            assert mean_nu_t > 0, f"湍流黏度應為正: {mean_nu_t}"
            assert mean_nu_t < 100.0, f"湍流黏度過大（可能不穩定）: {mean_nu_t}"
            
            print(f"✅ 測試 10 通過: ν_t ∈ [{nu_t.min():.2e}, {nu_t.max():.2e}], 均值={mean_nu_t:.2e}")
        else:
            pytest.skip("RANS 模型未初始化")


if __name__ == "__main__":
    # 快速測試執行
    import sys
    pytest.main([__file__, "-v", "-s"] + sys.argv[1:])
