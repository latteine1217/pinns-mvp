"""
性能優化功能測試
================

測試梯度檢查點與其他性能優化功能的正確性與效能。

測試項目：
1. 梯度檢查點數值正確性（與標準方法對比）
2. 梯度檢查點記憶體節省驗證
3. 梯度檢查點速度影響測量
4. 高階導數計算正確性

作者：PINNs-MVP 團隊
日期：2025-10-15
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Tuple
import time
import gc

from pinnx.physics.vs_pinn_channel_flow import (
    VSPINNChannelFlow,
    compute_gradient_3d,
    compute_gradient_3d_checkpointed
)


class TestGradientCheckpointing:
    """梯度檢查點功能測試套件"""
    
    @pytest.fixture
    def device(self):
        """設備配置"""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @pytest.fixture
    def test_data(self, device):
        """生成測試資料"""
        batch_size = 128
        coords = torch.randn(batch_size, 3, device=device, requires_grad=True)
        return coords
    
    @pytest.fixture
    def physics_standard(self, device):
        """標準物理模組（關閉檢查點）"""
        return VSPINNChannelFlow(
            use_gradient_checkpointing=False
        ).to(device)
    
    @pytest.fixture
    def physics_checkpointed(self, device):
        """啟用檢查點的物理模組"""
        return VSPINNChannelFlow(
            use_gradient_checkpointing=True
        ).to(device)
    
    def test_gradient_numerical_correctness(self, test_data, device):
        """
        測試 1：數值正確性驗證
        
        驗證梯度檢查點與標準方法計算結果完全一致（容忍度 1e-6）
        """
        # 創建標量場
        field = torch.sin(test_data[:, 0:1]) * torch.cos(test_data[:, 1:2])
        
        # 標準梯度計算
        grad_standard_x = compute_gradient_3d(field, test_data, component=0)
        grad_standard_y = compute_gradient_3d(field, test_data, component=1)
        grad_standard_z = compute_gradient_3d(field, test_data, component=2)
        
        # 檢查點梯度計算
        grad_checkpoint_x = compute_gradient_3d_checkpointed(field, test_data, component=0)
        grad_checkpoint_y = compute_gradient_3d_checkpointed(field, test_data, component=1)
        grad_checkpoint_z = compute_gradient_3d_checkpointed(field, test_data, component=2)
        
        # 數值一致性檢查（絕對誤差 + 相對誤差）
        atol = 1e-6
        rtol = 1e-5
        
        assert torch.allclose(grad_standard_x, grad_checkpoint_x, atol=atol, rtol=rtol), \
            f"X 梯度不一致：最大誤差 {(grad_standard_x - grad_checkpoint_x).abs().max().item()}"
        
        assert torch.allclose(grad_standard_y, grad_checkpoint_y, atol=atol, rtol=rtol), \
            f"Y 梯度不一致：最大誤差 {(grad_standard_y - grad_checkpoint_y).abs().max().item()}"
        
        assert torch.allclose(grad_standard_z, grad_checkpoint_z, atol=atol, rtol=rtol), \
            f"Z 梯度不一致：最大誤差 {(grad_standard_z - grad_checkpoint_z).abs().max().item()}"
        
        print("✅ 梯度數值正確性驗證通過（誤差 < 1e-6）")
    
    def test_second_order_gradient_correctness(self, test_data, device):
        """
        測試 2：二階導數正確性
        
        驗證 Laplacian 計算的數值一致性
        """
        # 創建標量場（可解析求導）
        field = torch.sin(test_data[:, 0:1]) * torch.cos(test_data[:, 1:2]) * torch.exp(test_data[:, 2:3])
        
        # 標準方法：一階導數
        grad_x = compute_gradient_3d(field, test_data, component=0)
        grad_xx_standard = compute_gradient_3d(grad_x, test_data, component=0)
        
        # 檢查點方法：一階導數
        grad_x_cp = compute_gradient_3d_checkpointed(field, test_data, component=0)
        grad_xx_checkpoint = compute_gradient_3d_checkpointed(grad_x_cp, test_data, component=0)
        
        # 二階導數一致性檢查
        atol = 1e-5  # 二階導數允許稍大誤差
        rtol = 1e-4
        
        assert torch.allclose(grad_xx_standard, grad_xx_checkpoint, atol=atol, rtol=rtol), \
            f"二階導數不一致：最大誤差 {(grad_xx_standard - grad_xx_checkpoint).abs().max().item()}"
        
        print("✅ 二階導數正確性驗證通過（誤差 < 1e-5）")
    
    def test_physics_module_integration(self, physics_standard, physics_checkpointed, test_data):
        """
        測試 3：物理模組整合測試
        
        驗證 VSPINNChannelFlow 的梯度計算路由正確
        """
        # 創建標量場（需要計算圖）
        field = torch.sin(test_data[:, 0:1]) * torch.cos(test_data[:, 1:2])
        
        # 標準模組計算
        grads_standard = physics_standard.compute_gradients(
            field, test_data, order=1
        )
        
        # 檢查點模組計算
        grads_checkpoint = physics_checkpointed.compute_gradients(
            field, test_data, order=1
        )
        
        # 逐分量比較
        for key in ['x', 'y', 'z']:
            assert torch.allclose(
                grads_standard[key], 
                grads_checkpoint[key], 
                atol=1e-6, 
                rtol=1e-5
            ), f"物理模組 {key} 梯度不一致"
        
        print("✅ 物理模組整合測試通過")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU 測量記憶體")
    def test_memory_saving(self, physics_standard, physics_checkpointed, device):
        """
        測試 4：記憶體節省驗證（僅 GPU）
        
        測量標準方法與檢查點方法的記憶體差異
        """
        batch_size = 2048  # 大批次以突顯記憶體差異
        coords = torch.randn(batch_size, 3, device=device, requires_grad=True)
        
        # 模擬複雜計算（多次梯度調用）
        def complex_gradient_computation(physics, coords):
            field = torch.sin(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
            grads = physics.compute_gradients(field, coords, order=1)
            # 計算 Laplacian（逐項加總）
            laplacian_terms = [
                physics.compute_gradients(grads[key], coords, order=1)[key] 
                for key in ['x', 'y', 'z']
            ]
            # 正確的 Tensor 加總方式
            laplacian = torch.stack(laplacian_terms, dim=0).sum(dim=0)
            loss = laplacian.mean()
            loss.backward()
            return loss
        
        # 清空 GPU 快取
        torch.cuda.empty_cache()
        gc.collect()
        
        # 測量標準方法記憶體
        torch.cuda.reset_peak_memory_stats()
        _ = complex_gradient_computation(physics_standard, coords.clone().detach().requires_grad_(True))
        mem_standard = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # 清空 GPU 快取
        torch.cuda.empty_cache()
        gc.collect()
        
        # 測量檢查點方法記憶體
        torch.cuda.reset_peak_memory_stats()
        _ = complex_gradient_computation(physics_checkpointed, coords.clone().detach().requires_grad_(True))
        mem_checkpoint = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        # 計算節省比例
        memory_reduction = (mem_standard - mem_checkpoint) / mem_standard
        
        print(f"📊 記憶體使用：")
        print(f"  標準方法：{mem_standard:.2f} MB")
        print(f"  檢查點方法：{mem_checkpoint:.2f} MB")
        print(f"  節省比例：{memory_reduction * 100:.1f}%")
        
        # 驗收標準：至少節省 20%（目標 30-50%）
        assert memory_reduction >= 0.20, \
            f"記憶體節省不足：僅 {memory_reduction * 100:.1f}%（目標 ≥20%）"
        
        print("✅ 記憶體節省驗證通過（≥20%）")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU 測量速度")
    def test_speed_overhead(self, physics_standard, physics_checkpointed, device):
        """
        測試 5：速度影響測量（僅 GPU）
        
        測量檢查點方法的速度開銷（目標 <15%）
        """
        batch_size = 1024
        coords = torch.randn(batch_size, 3, device=device, requires_grad=True)
        num_iterations = 50
        
        def benchmark_forward_backward(physics, coords, iterations):
            """前向+反向傳播基準測試"""
            timings = []
            for _ in range(iterations):
                coords_copy = coords.clone().detach().requires_grad_(True)
                field = torch.sin(coords_copy[:, 0:1])
                
                # 記錄時間
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                grads = physics.compute_gradients(field, coords_copy, order=1)
                # 正確的 Tensor 加總方式
                loss = torch.stack(list(grads.values()), dim=0).sum(dim=0).mean()
                loss.backward()
                
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                timings.append(end - start)
            
            return sum(timings) / len(timings)
        
        # 預熱
        _ = benchmark_forward_backward(physics_standard, coords, 5)
        _ = benchmark_forward_backward(physics_checkpointed, coords, 5)
        
        # 正式測試
        time_standard = benchmark_forward_backward(physics_standard, coords, num_iterations)
        time_checkpoint = benchmark_forward_backward(physics_checkpointed, coords, num_iterations)
        
        # 計算速度開銷
        speed_overhead = (time_checkpoint - time_standard) / time_standard
        
        print(f"⏱️ 速度測試：")
        print(f"  標準方法：{time_standard * 1000:.2f} ms")
        print(f"  檢查點方法：{time_checkpoint * 1000:.2f} ms")
        print(f"  速度開銷：{speed_overhead * 100:.1f}%")
        
        # 驗收標準：速度慢 ≤20%（目標 10-15%）
        assert speed_overhead <= 0.20, \
            f"速度開銷過大：{speed_overhead * 100:.1f}%（目標 ≤20%）"
        
        print("✅ 速度影響驗證通過（開銷 ≤20%）")
    
    def test_backward_gradient_correctness(self, test_data, device):
        """
        測試 6：反向傳播梯度正確性
        
        驗證檢查點不影響參數梯度計算
        """
        # 創建簡單模型
        model = nn.Linear(3, 1).to(device)
        
        # 標準方法梯度
        coords_1 = test_data.clone().detach().requires_grad_(True)
        field_1 = model(coords_1)
        grad_1 = compute_gradient_3d(field_1, coords_1, component=0)
        loss_1 = grad_1.mean()
        loss_1.backward()
        assert model.weight.grad is not None, "標準方法未產生梯度"
        param_grad_1 = model.weight.grad.clone()
        model.zero_grad()
        
        # 檢查點方法梯度
        coords_2 = test_data.clone().detach().requires_grad_(True)
        field_2 = model(coords_2)
        grad_2 = compute_gradient_3d_checkpointed(field_2, coords_2, component=0)
        loss_2 = grad_2.mean()
        loss_2.backward()
        assert model.weight.grad is not None, "檢查點方法未產生梯度"
        param_grad_2 = model.weight.grad.clone()
        
        # 參數梯度一致性檢查
        assert torch.allclose(param_grad_1, param_grad_2, atol=1e-6, rtol=1e-5), \
            f"參數梯度不一致：最大誤差 {(param_grad_1 - param_grad_2).abs().max().item()}"
        
        print("✅ 反向傳播梯度正確性驗證通過")


class TestConfigurationLoading:
    """配置載入測試"""
    
    def test_gradient_checkpoint_config_parsing(self):
        """
        測試 7：配置檔案解析
        
        驗證 use_gradient_checkpointing 參數正確傳遞
        """
        # 測試啟用檢查點
        physics_enabled = VSPINNChannelFlow(use_gradient_checkpointing=True)
        assert physics_enabled.use_gradient_checkpointing is True
        
        # 測試關閉檢查點
        physics_disabled = VSPINNChannelFlow(use_gradient_checkpointing=False)
        assert physics_disabled.use_gradient_checkpointing is False
        
        # 測試默認值（應啟用）
        physics_default = VSPINNChannelFlow()
        assert physics_default.use_gradient_checkpointing is True
        
        print("✅ 配置參數解析測試通過")


if __name__ == "__main__":
    """
    直接執行測試
    
    使用方式：
        python tests/test_performance_optimizations.py
    """
    pytest.main([__file__, "-v", "-s"])
