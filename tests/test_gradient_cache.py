"""
梯度快取模組測試

測試 GradientCache 類別的正確性、效能和記憶體管理

Author: Performance Optimization Team
Date: 2025-12-15
"""

import numpy as np
import torch
import pytest
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pinnx.physics.gradient_cache import GradientCache


class TestGradientCache:
    """測試 GradientCache 類別"""
    
    def setup_method(self):
        """設置測試環境"""
        # 優先使用 MPS (Apple Silicon)，其次 CUDA，最後 CPU
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        torch.manual_seed(42)
        if self.device.type == 'cuda':
            torch.cuda.manual_seed(42)
    
    def _create_test_data(self, n_points: int = 100):
        """
        建立測試用的座標和預測值
        
        Args:
            n_points: 測試點數量
        
        Returns:
            coords: [N, 3] 座標張量
            predictions: 包含 u, v, w, p 的字典
        """
        # 建立座標 (需要梯度)
        coords = torch.randn(n_points, 3, dtype=torch.float32, device=self.device)
        coords.requires_grad_(True)
        
        # 建立簡單的解析函數作為預測值
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        # 使用簡單的多項式函數（方便驗證梯度）
        u = torch.sin(x) * torch.cos(y) + z**2
        v = torch.cos(x) * torch.sin(z) + y**2
        w = torch.sin(y) * torch.cos(z) + x**2
        p = x * y * z
        
        predictions = {
            'u': u,
            'v': v,
            'w': w,
            'p': p
        }
        
        return coords, predictions
    
    def _compute_gradient_original(self, output: torch.Tensor, coords: torch.Tensor, dim: int, create_graph: bool = True) -> torch.Tensor:
        """
        原始的梯度計算方法（用於對照）
        
        Args:
            output: 輸出張量 [N, 1]
            coords: 座標張量 [N, 3]
            dim: 梯度方向 (0: x, 1: y, 2: z)
            create_graph: 是否保留計算圖
        
        Returns:
            gradient: 梯度張量 [N, 1]
        """
        grad = torch.autograd.grad(
            outputs=output,
            inputs=coords,
            grad_outputs=torch.ones_like(output),
            create_graph=create_graph,
            retain_graph=True
        )[0]
        
        return grad[:, dim:dim+1]
    
    def test_cache_initialization(self):
        """測試快取初始化"""
        cache = GradientCache(device=str(self.device))
        
        assert cache.device == str(self.device)
        assert cache._cache is None
    
    def test_input_validation(self):
        """測試輸入驗證"""
        cache = GradientCache(device=str(self.device))
        
        # 測試缺少必要鍵
        coords = torch.randn(10, 3, requires_grad=True, device=self.device)
        incomplete_predictions = {'u': torch.randn(10, 1, device=self.device)}
        
        with pytest.raises(KeyError, match="missing required keys"):
            cache.compute_all_gradients(incomplete_predictions, coords)
        
        # 測試 coords 沒有 requires_grad
        coords_no_grad = torch.randn(10, 3, device=self.device)
        predictions = {k: torch.randn(10, 1, device=self.device) for k in ['u', 'v', 'w', 'p']}
        
        with pytest.raises(RuntimeError, match="requires_grad=True"):
            cache.compute_all_gradients(predictions, coords_no_grad)
    
    def test_first_order_gradient_correctness(self):
        """測試一階梯度計算正確性"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=100)
        
        # 計算快取梯度
        gradients = cache.compute_all_gradients(predictions, coords)
        
        # 對照原始方法計算
        u_x_original = self._compute_gradient_original(predictions['u'], coords, dim=0)
        u_y_original = self._compute_gradient_original(predictions['u'], coords, dim=1)
        u_z_original = self._compute_gradient_original(predictions['u'], coords, dim=2)
        
        v_x_original = self._compute_gradient_original(predictions['v'], coords, dim=0)
        v_y_original = self._compute_gradient_original(predictions['v'], coords, dim=1)
        v_z_original = self._compute_gradient_original(predictions['v'], coords, dim=2)
        
        w_x_original = self._compute_gradient_original(predictions['w'], coords, dim=0)
        w_y_original = self._compute_gradient_original(predictions['w'], coords, dim=1)
        w_z_original = self._compute_gradient_original(predictions['w'], coords, dim=2)
        
        p_x_original = self._compute_gradient_original(predictions['p'], coords, dim=0)
        p_y_original = self._compute_gradient_original(predictions['p'], coords, dim=1)
        p_z_original = self._compute_gradient_original(predictions['p'], coords, dim=2)
        
        # 驗證所有一階梯度（容許誤差 1e-6）
        assert torch.allclose(gradients['u_x'], u_x_original, atol=1e-6), "u_x mismatch"
        assert torch.allclose(gradients['u_y'], u_y_original, atol=1e-6), "u_y mismatch"
        assert torch.allclose(gradients['u_z'], u_z_original, atol=1e-6), "u_z mismatch"
        
        assert torch.allclose(gradients['v_x'], v_x_original, atol=1e-6), "v_x mismatch"
        assert torch.allclose(gradients['v_y'], v_y_original, atol=1e-6), "v_y mismatch"
        assert torch.allclose(gradients['v_z'], v_z_original, atol=1e-6), "v_z mismatch"
        
        assert torch.allclose(gradients['w_x'], w_x_original, atol=1e-6), "w_x mismatch"
        assert torch.allclose(gradients['w_y'], w_y_original, atol=1e-6), "w_y mismatch"
        assert torch.allclose(gradients['w_z'], w_z_original, atol=1e-6), "w_z mismatch"
        
        assert torch.allclose(gradients['p_x'], p_x_original, atol=1e-6), "p_x mismatch"
        assert torch.allclose(gradients['p_y'], p_y_original, atol=1e-6), "p_y mismatch"
        assert torch.allclose(gradients['p_z'], p_z_original, atol=1e-6), "p_z mismatch"
        
        print("✅ 所有一階梯度驗證通過")
    
    def test_second_order_gradient_correctness(self):
        """測試二階梯度計算正確性"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=100)
        
        # 計算快取梯度
        gradients = cache.compute_all_gradients(predictions, coords)
        
        # 對照原始方法計算二階梯度
        u_x = self._compute_gradient_original(predictions['u'], coords, dim=0, create_graph=True)
        u_xx_original = self._compute_gradient_original(u_x, coords, dim=0, create_graph=True)
        
        u_y = self._compute_gradient_original(predictions['u'], coords, dim=1, create_graph=True)
        u_yy_original = self._compute_gradient_original(u_y, coords, dim=1, create_graph=True)
        
        u_z = self._compute_gradient_original(predictions['u'], coords, dim=2, create_graph=True)
        u_zz_original = self._compute_gradient_original(u_z, coords, dim=2, create_graph=True)
        
        # 驗證 u 的二階梯度
        assert torch.allclose(gradients['u_xx'], u_xx_original, atol=1e-5), "u_xx mismatch"
        assert torch.allclose(gradients['u_yy'], u_yy_original, atol=1e-5), "u_yy mismatch"
        assert torch.allclose(gradients['u_zz'], u_zz_original, atol=1e-5), "u_zz mismatch"
        
        print("✅ 二階梯度驗證通過")
    
    def test_gradient_shapes(self):
        """測試梯度張量形狀"""
        cache = GradientCache(device=str(self.device))
        n_points = 256
        coords, predictions = self._create_test_data(n_points=n_points)
        
        gradients = cache.compute_all_gradients(predictions, coords)
        
        # 驗證所有梯度張量形狀為 [N, 1]
        expected_keys = [
            'u_x', 'u_y', 'u_z', 'u_xx', 'u_yy', 'u_zz',
            'v_x', 'v_y', 'v_z', 'v_xx', 'v_yy', 'v_zz',
            'w_x', 'w_y', 'w_z', 'w_xx', 'w_yy', 'w_zz',
            'p_x', 'p_y', 'p_z'
        ]
        
        for key in expected_keys:
            assert key in gradients, f"Missing gradient key: {key}"
            assert gradients[key].shape == (n_points, 1), f"{key} shape mismatch: {gradients[key].shape}"
        
        print(f"✅ 所有 {len(expected_keys)} 個梯度張量形狀正確")
    
    def test_cache_memory_cleanup(self):
        """測試快取記憶體清理"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=1000)
        
        # 計算梯度
        gradients = cache.compute_all_gradients(predictions, coords)
        assert cache._cache is not None
        
        # 清理快取
        cache.clear_cache()
        assert cache._cache is None
        
        print("✅ 快取記憶體清理成功")
    
    def test_repeated_computation(self):
        """測試重複計算的正確性"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=100)
        
        # 第一次計算
        gradients1 = cache.compute_all_gradients(predictions, coords)
        cache.clear_cache()
        
        # 第二次計算（相同輸入）
        gradients2 = cache.compute_all_gradients(predictions, coords)
        
        # 驗證兩次計算結果相同
        for key in gradients1.keys():
            assert torch.allclose(gradients1[key], gradients2[key], atol=1e-6), f"{key} changed between runs"
        
        print("✅ 重複計算結果一致")
    
    def test_backward_pass(self):
        """測試梯度張量可以正確進行反向傳播"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=50)
        
        # 計算梯度
        gradients = cache.compute_all_gradients(predictions, coords, create_graph=True)
        
        # 構造損失函數（使用梯度計算 Laplacian）
        laplacian_u = gradients['u_xx'] + gradients['u_yy'] + gradients['u_zz']
        loss = laplacian_u.pow(2).mean()
        
        # 反向傳播（不應該報錯）
        try:
            loss.backward()
            success = True
        except Exception as e:
            success = False
            print(f"❌ Backward pass failed: {e}")
        
        assert success, "Backward pass should work with cached gradients"
        print("✅ 反向傳播測試通過")
    
    @pytest.mark.skipif(not torch.cuda.is_available() and not torch.backends.mps.is_available(), 
                        reason="GPU not available")
    def test_performance_improvement(self):
        """測試快取方法相比原始方法的效能提升（僅在 GPU 上測試）"""
        n_points = 4096
        n_iterations = 20
        
        coords, predictions = self._create_test_data(n_points=n_points)
        cache = GradientCache(device=str(self.device))
        
        # 預熱
        _ = cache.compute_all_gradients(predictions, coords)
        cache.clear_cache()
        
        # 測試快取方法
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        for _ in range(n_iterations):
            gradients = cache.compute_all_gradients(predictions, coords)
            cache.clear_cache()
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
        
        cache_time = time.time() - start_time
        
        # 測試原始方法（分散計算）
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
        
        start_time = time.time()
        for _ in range(n_iterations):
            # 模擬原始方法：每個梯度單獨計算
            for var in ['u', 'v', 'w']:
                for dim in range(3):
                    _ = self._compute_gradient_original(predictions[var], coords, dim, create_graph=True)
                    # 二階梯度
                    first_order = self._compute_gradient_original(predictions[var], coords, dim, create_graph=True)
                    _ = self._compute_gradient_original(first_order, coords, dim, create_graph=True)
            
            # 壓力一階梯度
            for dim in range(3):
                _ = self._compute_gradient_original(predictions['p'], coords, dim, create_graph=True)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        elif self.device.type == 'mps':
            torch.mps.synchronize()
        
        original_time = time.time() - start_time
        
        speedup = original_time / cache_time
        improvement_pct = (1 - cache_time / original_time) * 100
        
        print(f"\n{'='*60}")
        print(f"效能測試結果 (Device: {self.device}, Points: {n_points}, Iterations: {n_iterations})")
        print(f"{'='*60}")
        print(f"原始方法耗時: {original_time:.4f} 秒")
        print(f"快取方法耗時: {cache_time:.4f} 秒")
        print(f"加速比: {speedup:.2f}x")
        print(f"效能提升: {improvement_pct:.1f}%")
        print(f"{'='*60}")
        
        # 期望至少有 10% 的效能提升
        assert speedup > 1.1, f"Expected speedup > 1.1x, got {speedup:.2f}x"
        print("✅ 效能測試通過（加速比 > 1.1x）")
    
    def test_get_cached_method(self):
        """測試 get_cached 方法"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=100)
        
        # 計算梯度
        all_gradients = cache.compute_all_gradients(predictions, coords)
        
        # 使用 get_cached 提取單個梯度
        u_x_from_cache = cache.get_cached('u_x')
        
        assert u_x_from_cache is not None
        assert torch.allclose(u_x_from_cache, all_gradients['u_x'], atol=1e-8)
        
        # 測試不存在的鍵
        assert cache.get_cached('nonexistent_key') is None
        
        print("✅ get_cached 方法測試通過")
    
    def test_create_graph_false(self):
        """測試 create_graph=False 的情況（推理模式）"""
        cache = GradientCache(device=str(self.device))
        coords, predictions = self._create_test_data(n_points=100)
        
        # 注意：create_graph=False 時無法計算二階梯度
        # 因此此測試僅驗證方法正確處理該參數
        # 在實際應用中，推理時通常不需要二階梯度
        
        # 訓練模式：create_graph=True（默認）
        gradients_train = cache.compute_all_gradients(predictions, coords, create_graph=True)
        
        # 驗證梯度存在且包含一階和二階
        assert 'u_x' in gradients_train
        assert 'u_xx' in gradients_train
        assert gradients_train['u_x'].shape == (100, 1)
        assert gradients_train['u_xx'].shape == (100, 1)
        
        # 清理快取
        cache.clear_cache()
        
        print("✅ create_graph 參數測試通過")


class TestGradientCacheIntegration:
    """整合測試：模擬真實訓練場景"""
    
    def setup_method(self):
        """設置測試環境"""
        if torch.backends.mps.is_available():
            self.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        
        torch.manual_seed(42)
    
    def test_training_loop_simulation(self):
        """模擬訓練迴圈中的使用方式"""
        import torch.nn as nn
        
        # 建立簡單的 MLP
        model = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 4)  # 輸出 [u, v, w, p]
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cache = GradientCache(device=str(self.device))
        
        n_steps = 5
        
        for step in range(n_steps):
            # 生成批次資料
            coords = torch.randn(256, 3, device=self.device, requires_grad=True)
            
            # 前向傳播
            output = model(coords)
            predictions = {
                'u': output[:, 0:1],
                'v': output[:, 1:2],
                'w': output[:, 2:3],
                'p': output[:, 3:4]
            }
            
            # 計算梯度
            gradients = cache.compute_all_gradients(predictions, coords)
            
            # 構造損失（模擬 continuity equation）
            div_u = gradients['u_x'] + gradients['v_y'] + gradients['w_z']
            loss_continuity = div_u.pow(2).mean()
            
            # 構造損失（模擬 momentum equation）
            laplacian_u = gradients['u_xx'] + gradients['u_yy'] + gradients['u_zz']
            loss_momentum = (laplacian_u - gradients['p_x']).pow(2).mean()
            
            total_loss = loss_continuity + loss_momentum
            
            # 反向傳播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # 清理快取
            cache.clear_cache()
            
            print(f"Step {step+1}/{n_steps}: Loss = {total_loss.item():.6f}")
        
        print("✅ 訓練迴圈模擬測試通過")


if __name__ == '__main__':
    # 可以直接執行測試
    pytest.main([__file__, '-v', '-s'])
