"""
Wave 2 Phase 2-2: VS-PINN 梯度快取整合測試

測試 VS-PINN 的 residual 方法是否正確支援預計算梯度快取

Author: Performance Optimization Team
Date: 2025-12-15
"""

import numpy as np
import torch
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
from pinnx.physics.gradient_cache import GradientCache


class TestVSPINNGradientCacheIntegration:
    """測試 VS-PINN 與 GradientCache 的整合"""
    
    def setup_method(self):
        """設置測試環境"""
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        torch.manual_seed(42)
        
        # 創建 VS-PINN 實例（標準 JHTDB 設置）
        self.vs_pinn = VSPINNChannelFlow(
            nu=0.000185,  # Re_tau = 1000 對應的黏度
            rho=1.0,
            dP_dx=0.0025,
            N_x=2.0,  # VS-PINN 縮放係數
            N_y=12.0,
            N_z=2.0,
            enable_rans=False  # 不啟用 RANS
        )
    
    def _create_test_data(self, n_points: int = 100):
        """
        創建測試用的座標和預測值
        
        Args:
            n_points: 測試點數量
        
        Returns:
            coords: [N, 3] 座標張量
            predictions_dict: 包含 u, v, w, p 的字典
            predictions_tensor: [N, 4] 預測張量
        """
        # 創建座標（需要梯度）
        coords = torch.randn(n_points, 3, dtype=torch.float32, device=self.device)
        coords.requires_grad_(True)
        
        # 創建簡單的測試流場
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        u = torch.sin(x) * torch.cos(y) + z**2
        v = torch.cos(x) * torch.sin(z) + y**2
        w = torch.sin(y) * torch.cos(z) + x**2
        p = x * y * z + 1.0
        
        predictions_dict = {'u': u, 'v': v, 'w': w, 'p': p}
        predictions_tensor = torch.cat([u, v, w, p], dim=1)  # [N, 4]
        
        return coords, predictions_dict, predictions_tensor
    
    def test_momentum_residual_correctness_no_scaling(self):
        """測試無縮放時，使用快取梯度與原始方法的 momentum residual 一致性"""
        n_points = 50
        coords, predictions_dict, predictions_tensor = self._create_test_data(n_points)
        
        # 計算梯度快取
        cache = GradientCache(device=str(self.device))
        gradients = cache.compute_all_gradients(predictions_dict, coords)
        
        # 方法 1：原始方法（不提供 gradients）
        residuals_original = self.vs_pinn.compute_momentum_residuals(
            coords, predictions_tensor, scaled_coords=None
        )
        
        # 方法 2：使用快取梯度
        residuals_cached = self.vs_pinn.compute_momentum_residuals(
            coords, predictions_tensor, scaled_coords=None, gradients=gradients
        )
        
        # 驗證三個方向的 residual 一致（容許誤差 1e-5）
        assert torch.allclose(
            residuals_original['momentum_x'], 
            residuals_cached['momentum_x'], 
            atol=1e-5
        ), "Momentum X residual mismatch"
        
        assert torch.allclose(
            residuals_original['momentum_y'], 
            residuals_cached['momentum_y'], 
            atol=1e-5
        ), "Momentum Y residual mismatch"
        
        assert torch.allclose(
            residuals_original['momentum_z'], 
            residuals_cached['momentum_z'], 
            atol=1e-5
        ), "Momentum Z residual mismatch"
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ Momentum residual 測試通過（無縮放）")
    
    def test_continuity_residual_correctness_no_scaling(self):
        """測試無縮放時，使用快取梯度與原始方法的 continuity residual 一致性"""
        n_points = 50
        coords, predictions_dict, predictions_tensor = self._create_test_data(n_points)
        
        # 計算梯度快取
        cache = GradientCache(device=str(self.device))
        gradients = cache.compute_all_gradients(predictions_dict, coords)
        
        # 方法 1：原始方法（不提供 gradients）
        residual_original = self.vs_pinn.compute_continuity_residual(
            coords, predictions_tensor, scaled_coords=None
        )
        
        # 方法 2：使用快取梯度
        residual_cached = self.vs_pinn.compute_continuity_residual(
            coords, predictions_tensor, scaled_coords=None, gradients=gradients
        )
        
        # 驗證 continuity residual 一致（容許誤差 1e-5）
        assert torch.allclose(
            residual_original, 
            residual_cached, 
            atol=1e-5
        ), "Continuity residual mismatch"
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ Continuity residual 測試通過（無縮放）")
    
    def test_momentum_residual_correctness_with_scaling(self):
        """測試有 VS-PINN 縮放時，使用快取梯度與原始方法的 momentum residual 一致性"""
        n_points = 50
        coords, predictions_dict, predictions_tensor = self._create_test_data(n_points)
        
        # 創建縮放座標
        scaled_coords = torch.cat([
            coords[:, 0:1] / self.vs_pinn.N_x,  # type: ignore[operator]
            coords[:, 1:2] / self.vs_pinn.N_y,  # type: ignore[operator]
            coords[:, 2:3] / self.vs_pinn.N_z   # type: ignore[operator]
        ], dim=1)
        scaled_coords.requires_grad_(True)
        
        # 重新計算 predictions（基於 scaled_coords）
        # 注意：這裡為了測試方便，使用相同的函數形式
        x_s, y_s, z_s = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        u_s = torch.sin(x_s) * torch.cos(y_s) + z_s**2
        v_s = torch.cos(x_s) * torch.sin(z_s) + y_s**2
        w_s = torch.sin(y_s) * torch.cos(z_s) + x_s**2
        p_s = x_s * y_s * z_s + 1.0
        
        predictions_dict_scaled = {'u': u_s, 'v': v_s, 'w': w_s, 'p': p_s}
        predictions_tensor_scaled = torch.cat([u_s, v_s, w_s, p_s], dim=1)
        
        # 計算梯度快取（對 scaled_coords）
        cache = GradientCache(device=str(self.device))
        gradients = cache.compute_all_gradients(predictions_dict_scaled, scaled_coords)
        
        # 方法 1：原始方法（不提供 gradients）
        residuals_original = self.vs_pinn.compute_momentum_residuals(
            coords, predictions_tensor_scaled, scaled_coords=scaled_coords
        )
        
        # 方法 2：使用快取梯度
        residuals_cached = self.vs_pinn.compute_momentum_residuals(
            coords, predictions_tensor_scaled, scaled_coords=scaled_coords, gradients=gradients
        )
        
        # 驗證三個方向的 residual 一致（容許誤差 1e-5）
        assert torch.allclose(
            residuals_original['momentum_x'], 
            residuals_cached['momentum_x'], 
            atol=1e-5
        ), f"Momentum X residual mismatch with scaling. Max diff: {(residuals_original['momentum_x'] - residuals_cached['momentum_x']).abs().max()}"
        
        assert torch.allclose(
            residuals_original['momentum_y'], 
            residuals_cached['momentum_y'], 
            atol=1e-5
        ), f"Momentum Y residual mismatch with scaling. Max diff: {(residuals_original['momentum_y'] - residuals_cached['momentum_y']).abs().max()}"
        
        assert torch.allclose(
            residuals_original['momentum_z'], 
            residuals_cached['momentum_z'], 
            atol=1e-5
        ), f"Momentum Z residual mismatch with scaling. Max diff: {(residuals_original['momentum_z'] - residuals_cached['momentum_z']).abs().max()}"
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ Momentum residual 測試通過（有 VS-PINN 縮放）")
    
    def test_continuity_residual_correctness_with_scaling(self):
        """測試有 VS-PINN 縮放時，使用快取梯度與原始方法的 continuity residual 一致性"""
        n_points = 50
        coords, predictions_dict, predictions_tensor = self._create_test_data(n_points)
        
        # 創建縮放座標
        scaled_coords = torch.cat([
            coords[:, 0:1] / self.vs_pinn.N_x,  # type: ignore[operator]
            coords[:, 1:2] / self.vs_pinn.N_y,  # type: ignore[operator]
            coords[:, 2:3] / self.vs_pinn.N_z   # type: ignore[operator]
        ], dim=1)
        scaled_coords.requires_grad_(True)
        
        # 重新計算 predictions（基於 scaled_coords）
        x_s, y_s, z_s = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        u_s = torch.sin(x_s) * torch.cos(y_s) + z_s**2
        v_s = torch.cos(x_s) * torch.sin(z_s) + y_s**2
        w_s = torch.sin(y_s) * torch.cos(z_s) + x_s**2
        p_s = x_s * y_s * z_s + 1.0
        
        predictions_dict_scaled = {'u': u_s, 'v': v_s, 'w': w_s, 'p': p_s}
        predictions_tensor_scaled = torch.cat([u_s, v_s, w_s, p_s], dim=1)
        
        # 計算梯度快取（對 scaled_coords）
        cache = GradientCache(device=str(self.device))
        gradients = cache.compute_all_gradients(predictions_dict_scaled, scaled_coords)
        
        # 方法 1：原始方法（不提供 gradients）
        residual_original = self.vs_pinn.compute_continuity_residual(
            coords, predictions_tensor_scaled, scaled_coords=scaled_coords
        )
        
        # 方法 2：使用快取梯度
        residual_cached = self.vs_pinn.compute_continuity_residual(
            coords, predictions_tensor_scaled, scaled_coords=scaled_coords, gradients=gradients
        )
        
        # 驗證 continuity residual 一致（容許誤差 1e-5）
        max_diff = (residual_original - residual_cached).abs().max()
        assert torch.allclose(
            residual_original, 
            residual_cached, 
            atol=1e-5
        ), f"Continuity residual mismatch with scaling. Max diff: {max_diff}"
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ Continuity residual 測試通過（有 VS-PINN 縮放）")
    
    def test_backward_pass_with_cached_gradients(self):
        """測試使用快取梯度時反向傳播是否正常工作"""
        n_points = 50
        coords, predictions_dict, predictions_tensor = self._create_test_data(n_points)
        
        # 創建簡單的 MLP（模擬模型）
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 32),
            torch.nn.Tanh(),
            torch.nn.Linear(32, 4)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # 前向傳播
        coords_input = torch.randn(n_points, 3, device=self.device, requires_grad=True)
        predictions_model = model(coords_input)
        
        # 分離預測值
        predictions_dict_model = {
            'u': predictions_model[:, 0:1],
            'v': predictions_model[:, 1:2],
            'w': predictions_model[:, 2:3],
            'p': predictions_model[:, 3:4]
        }
        
        # 計算梯度快取
        cache = GradientCache(device=str(self.device))
        gradients = cache.compute_all_gradients(predictions_dict_model, coords_input)
        
        # 計算 residual（使用快取梯度）
        residuals = self.vs_pinn.compute_momentum_residuals(
            coords_input, predictions_model, scaled_coords=None, gradients=gradients
        )
        continuity_residual = self.vs_pinn.compute_continuity_residual(
            coords_input, predictions_model, scaled_coords=None, gradients=gradients
        )
        
        # 構造損失
        loss = (residuals['momentum_x'].pow(2).mean() + 
                residuals['momentum_y'].pow(2).mean() + 
                residuals['momentum_z'].pow(2).mean() +
                continuity_residual.pow(2).mean())
        
        # 反向傳播（不應報錯）
        optimizer.zero_grad()
        try:
            loss.backward()
            success = True
        except Exception as e:
            success = False
            print(f"❌ Backward pass failed: {e}")
        
        assert success, "Backward pass should work with cached gradients"
        
        # 驗證梯度存在
        for param in model.parameters():
            assert param.grad is not None, "Model parameters should have gradients"
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ 反向傳播測試通過（使用快取梯度）")


if __name__ == '__main__':
    # 可以直接執行測試
    pytest.main([__file__, '-v', '-s'])
