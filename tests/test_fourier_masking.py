"""
AxisSelectiveFourierFeatures 掩碼機制單元測試

測試覆蓋：
1. 掩碼機制基本功能（雙配置初始化）
2. 頻率退火時維度穩定性
3. 掩碼更新正確性（置零未啟用頻率）
4. 無效頻率驗證（嘗試啟用未在 full_axes_config 中的頻率）
5. 向後相容性（未提供 full_axes_config 時的行為）
6. 梯度流動正確性
7. 與訓練器整合（模擬退火場景）

背景：
TASK-007 Phase 2 實施了掩碼機制以解決 Fourier 退火時的維度不匹配問題：
- 模型始終使用 full_axes_config 初始化（固定維度）
- 退火通過掩碼控制啟用哪些頻率（不改變維度）
- set_active_frequencies() 僅更新掩碼，不重建 B 矩陣
"""

import pytest
import torch
import torch.nn as nn
from pinnx.models.axis_selective_fourier import AxisSelectiveFourierFeatures


class TestMaskingMechanism:
    """掩碼機制核心功能測試"""
    
    def test_dual_config_initialization(self):
        """測試雙配置初始化"""
        full_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}
        current_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        # 驗證配置保存
        assert fourier._full_axes_config == full_config
        assert fourier.axes_config == current_config
        
        # 驗證輸出維度基於完整配置（固定）
        # full_config: x:[1,2,4,8](4), z:[1,2,4](3) → 7 頻率 → 14 維
        assert fourier.out_dim == 14
        
        # 驗證 B 矩陣形狀
        assert fourier.B.shape == (3, 7)  # [in_dim=3, total_freqs=7]
    
    def test_masking_basic_functionality(self):
        """測試掩碼基本功能（未啟用頻率置零）"""
        full_config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        current_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}  # x:4 未啟用
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        # 驗證掩碼存在
        assert hasattr(fourier, '_frequency_mask')
        
        # 驗證掩碼內容：[x:1, x:2, x:4, z:1, z:2] → [1, 1, 0, 1, 1]
        expected_mask = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
        
        # 測試前向傳播：x:4 對應的相位應被置零
        x = torch.randn(10, 3)
        features = fourier(x)
        
        # 輸出維度固定為 10（5 頻率 * 2）
        assert features.shape == (10, 10)
        
        # 驗證掩碼效果：手動計算被禁用頻率的輸出應為零
        # 注意：由於掩碼在計算 cos/sin 前應用，cos(0) = 1, sin(0) = 0
        # 因此禁用頻率對應的 cos 特徵應為 1，sin 特徵應為 0
        
        # 提取 x:4 對應的 cos/sin 特徵（索引 2 和 2+5=7）
        x4_cos = features[:, 2]
        x4_sin = features[:, 7]
        
        # 由於相位被置零（z=0），cos(0)=1, sin(0)=0
        torch.testing.assert_close(x4_cos, torch.ones(10), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(x4_sin, torch.zeros(10), rtol=1e-5, atol=1e-6)
    
    def test_dimension_stability_during_annealing(self):
        """測試退火時維度穩定性（核心測試）"""
        full_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}
        initial_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=initial_config,
            full_axes_config=full_config,
        )
        
        # 初始狀態：輸出維度 14（基於完整配置）
        initial_out_dim = fourier.out_dim
        assert initial_out_dim == 14
        
        # 測試前向傳播
        x = torch.randn(8, 3, requires_grad=True)
        features_initial = fourier(x)
        assert features_initial.shape == (8, 14)
        
        # 🔧 退火階段 1：解鎖中頻
        stage1_config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        fourier.set_active_frequencies(stage1_config)
        
        # 維度不變 ✅
        assert fourier.out_dim == 14
        assert fourier.axes_config == stage1_config
        
        # 前向傳播仍正常
        features_stage1 = fourier(x)
        assert features_stage1.shape == (8, 14)
        
        # 🔧 退火階段 2：解鎖全頻段
        stage2_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}
        fourier.set_active_frequencies(stage2_config)
        
        # 維度仍不變 ✅
        assert fourier.out_dim == 14
        
        # 前向傳播仍正常
        features_stage2 = fourier(x)
        assert features_stage2.shape == (8, 14)
        
        # 驗證梯度回傳正常
        loss = features_stage2.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
    
    def test_mask_update_correctness(self):
        """測試掩碼更新正確性"""
        full_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config={'x': [1], 'y': [], 'z': [1]},
            full_axes_config=full_config,
        )
        
        # 初始掩碼：只啟用 x:1 和 z:1
        # 順序：[x:1, x:2, x:4, x:8, z:1, z:2] → [1, 0, 0, 0, 1, 0]
        expected_mask_1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        torch.testing.assert_close(fourier._frequency_mask, expected_mask_1)
        
        # 更新：啟用 x:[1,2,4], z:[1,2]
        fourier.set_active_frequencies({'x': [1, 2, 4], 'y': [], 'z': [1, 2]})
        expected_mask_2 = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0])
        torch.testing.assert_close(fourier._frequency_mask, expected_mask_2)
        
        # 更新：全頻段
        fourier.set_active_frequencies(full_config)
        expected_mask_3 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        torch.testing.assert_close(fourier._frequency_mask, expected_mask_3)
    
    def test_B_matrix_immutability(self):
        """測試 B 矩陣在退火時不變（關鍵測試）"""
        full_config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config={'x': [1], 'y': [], 'z': [1]},
            full_axes_config=full_config,
        )
        
        # 記錄初始 B 矩陣
        B_initial = fourier.B.clone()
        B_shape_initial = fourier.B.shape
        
        # 執行多次退火更新
        fourier.set_active_frequencies({'x': [1, 2], 'y': [], 'z': [1, 2]})
        fourier.set_active_frequencies({'x': [1, 2, 4], 'y': [], 'z': [1, 2]})
        fourier.set_active_frequencies(full_config)
        
        # 驗證 B 矩陣未改變 ✅
        torch.testing.assert_close(fourier.B, B_initial)
        assert fourier.B.shape == B_shape_initial
    
    def test_invalid_frequency_validation(self):
        """測試無效頻率驗證（嘗試啟用未在 full_axes_config 中的頻率）"""
        full_config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config={'x': [1, 2], 'y': [], 'z': [1, 2]},
            full_axes_config=full_config,
        )
        
        # 測試 1：嘗試啟用未在完整配置中的頻率
        invalid_config_1 = {'x': [1, 2, 8], 'y': [], 'z': [1, 2]}  # x:8 不存在
        with pytest.raises(ValueError, match="不在.*完整配置"):
            fourier.set_active_frequencies(invalid_config_1)
        
        # 測試 2：嘗試添加新軸
        invalid_config_2 = {'x': [1, 2], 'y': [1], 'z': [1, 2]}  # y 軸不存在
        with pytest.raises(ValueError, match="不在.*完整配置.*中"):
            fourier.set_active_frequencies(invalid_config_2)
        
        # 測試 3：合法子集應成功
        valid_config = {'x': [1, 4], 'y': [], 'z': [2]}  # 跳過某些頻率
        fourier.set_active_frequencies(valid_config)  # 不應拋出異常
        assert fourier.axes_config == valid_config
    
    def test_backward_compatibility(self):
        """測試向後相容性（未提供 full_axes_config）"""
        config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        
        # 未提供 full_axes_config（舊版使用方式）
        fourier = AxisSelectiveFourierFeatures(axes_config=config)
        
        # full_axes_config 應自動設為 axes_config
        assert fourier._full_axes_config == config
        assert fourier.axes_config == config
        
        # 輸出維度應基於配置
        assert fourier.out_dim == 10  # 2 * (3 + 0 + 2) = 10
        
        # 掩碼應全為 1（所有頻率啟用）
        expected_mask = torch.ones(5)  # 5 個頻率
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
        
        # 測試前向傳播
        x = torch.randn(5, 3)
        features = fourier(x)
        assert features.shape == (5, 10)
        
        # 測試退火行為（應允許在原配置範圍內更新）
        reduced_config = {'x': [1, 2], 'y': [], 'z': [1]}
        fourier.set_active_frequencies(reduced_config)
        
        # 維度應改變（舊版行為：重建矩陣）
        # 🔧 等等，這裡有問題：Phase 2 後不應重建矩陣
        # 向後相容模式下，維度應保持不變（基於 full_axes_config）
        assert fourier.out_dim == 10  # 維度不變 ✅


class TestGradientFlow:
    """梯度流動測試"""
    
    def test_gradient_through_masking(self):
        """測試掩碼不阻斷梯度"""
        full_config = {'x': [1, 2, 4], 'y': []}
        current_config = {'x': [1, 2], 'y': []}  # x:4 禁用
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        x = torch.randn(8, 2, requires_grad=True)
        features = fourier(x)
        loss = features.sum()
        loss.backward()
        
        # 驗證梯度正常
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isinf(x.grad).any()
        
        # 驗證已啟用頻率的梯度非零
        assert x.grad.abs().sum() > 0
    
    def test_trainable_B_with_masking(self):
        """測試可訓練 B 矩陣與掩碼共存"""
        full_config = {'x': [1, 2, 4], 'y': []}
        current_config = {'x': [1, 2], 'y': []}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
            trainable=True,  # B 可訓練
        )
        
        # 驗證 B 是 Parameter
        assert isinstance(fourier.B, nn.Parameter)
        assert fourier.B.requires_grad
        
        # 測試梯度回傳到 B
        x = torch.randn(4, 2)
        features = fourier(x)
        loss = features.sum()
        loss.backward()
        
        # B 應有梯度
        assert fourier.B.grad is not None
        assert not torch.isnan(fourier.B.grad).any()


class TestIntegrationScenarios:
    """整合場景測試"""
    
    def test_channel_flow_annealing_simulation(self):
        """模擬通道流頻率退火完整流程"""
        # 完整配置（最終要達到的頻率）
        full_config = {
            'x': [1, 2, 4, 8],  # 流向全頻段
            'y': [],            # 壁法向不用 Fourier
            'z': [1, 2, 4]      # 展向全頻段
        }
        
        # 初始配置（低頻啟動）
        initial_config = {
            'x': [1, 2],
            'y': [],
            'z': [1, 2]
        }
        
        # 創建模型
        fourier = AxisSelectiveFourierFeatures(
            axes_config=initial_config,
            full_axes_config=full_config,
        )
        
        # 驗證初始狀態
        assert fourier.out_dim == 14  # 2 * (4 + 0 + 3) = 14
        assert fourier.axes_config == initial_config
        
        # 模擬訓練批次
        batch_x = torch.randn(128, 3)
        
        # Epoch 0-14：低頻訓練
        features_stage1 = fourier(batch_x)
        assert features_stage1.shape == (128, 14)
        
        # Epoch 15：退火到中頻
        mid_config = {'x': [1, 2, 4], 'y': [], 'z': [1, 2]}
        fourier.set_active_frequencies(mid_config)
        features_stage2 = fourier(batch_x)
        assert features_stage2.shape == (128, 14)  # 維度不變 ✅
        
        # Epoch 30：退火到全頻段
        fourier.set_active_frequencies(full_config)
        features_stage3 = fourier(batch_x)
        assert features_stage3.shape == (128, 14)  # 維度仍不變 ✅
        
        # 驗證最終所有頻率均啟用
        expected_mask = torch.ones(7)  # 7 個頻率全啟用
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
    
    def test_with_mlp_network(self):
        """測試與 MLP 網路集成（維度匹配）"""
        full_config = {'x': [1, 2, 4, 8], 'y': [], 'z': [1, 2, 4]}
        initial_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=initial_config,
            full_axes_config=full_config,
        )
        
        # 基於固定輸出維度構建 MLP
        mlp = nn.Sequential(
            nn.Linear(fourier.out_dim, 128),  # 14 → 128
            nn.Tanh(),
            nn.Linear(128, 4)  # → u, v, w, p
        )
        
        # 初始訓練
        x = torch.randn(64, 3)
        features = fourier(x)
        output = mlp(features)
        assert output.shape == (64, 4)
        
        # 退火後仍能前向傳播 ✅
        fourier.set_active_frequencies(full_config)
        features_annealed = fourier(x)
        output_annealed = mlp(features_annealed)
        assert output_annealed.shape == (64, 4)  # 無維度錯誤


class TestEdgeCases:
    """邊界條件測試"""
    
    def test_empty_initial_config(self):
        """測試空初始配置（所有頻率禁用）"""
        full_config = {'x': [1, 2, 4], 'y': []}
        initial_config = {'x': [], 'y': []}  # 所有頻率初始禁用
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=initial_config,
            full_axes_config=full_config,
        )
        
        # 輸出維度基於完整配置
        assert fourier.out_dim == 6  # 2 * (3 + 0)
        
        # 掩碼應全為 0
        expected_mask = torch.zeros(3)
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
        
        # 前向傳播：所有頻率置零 → cos(0)=1, sin(0)=0
        x = torch.randn(5, 2)
        features = fourier(x)
        
        # cos 部分應全為 1
        cos_part = features[:, :3]
        torch.testing.assert_close(cos_part, torch.ones(5, 3), rtol=1e-5, atol=1e-6)
        
        # sin 部分應全為 0
        sin_part = features[:, 3:]
        torch.testing.assert_close(sin_part, torch.zeros(5, 3), rtol=1e-5, atol=1e-6)
    
    def test_identical_configs(self):
        """測試相同的當前與完整配置（無需掩碼）"""
        config = {'x': [1, 2, 4], 'y': []}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=config,
            full_axes_config=config,  # 相同
        )
        
        # 掩碼應全為 1
        expected_mask = torch.ones(3)
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
        
        # 前向傳播正常
        x = torch.randn(10, 2)
        features = fourier(x)
        assert features.shape == (10, 6)
    
    def test_single_frequency_masking(self):
        """測試單頻率掩碼"""
        full_config = {'x': [1, 2, 4, 8, 16]}
        current_config = {'x': [4]}  # 只啟用中頻
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        # 掩碼：[0, 0, 1, 0, 0]
        expected_mask = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0])
        torch.testing.assert_close(fourier._frequency_mask, expected_mask)
        
        # 前向傳播
        x = torch.randn(3, 1)
        features = fourier(x)
        assert features.shape == (3, 10)  # 2 * 5
    
    def test_device_compatibility_with_masking(self):
        """測試掩碼在不同設備上的兼容性"""
        full_config = {'x': [1, 2, 4]}
        current_config = {'x': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        # CPU 測試
        x_cpu = torch.randn(4, 1)
        features_cpu = fourier(x_cpu)
        assert features_cpu.device.type == 'cpu'
        
        # CUDA 測試（如果可用）
        if torch.cuda.is_available():
            fourier_cuda = fourier.cuda()
            x_cuda = torch.randn(4, 1, device='cuda')
            features_cuda = fourier_cuda(x_cuda)
            assert features_cuda.device.type == 'cuda'


class TestPerformance:
    """效能測試"""
    
    def test_masking_overhead(self):
        """測試掩碼計算開銷（應極低）"""
        full_config = {'x': [1, 2, 4, 8, 16, 32], 'y': [], 'z': [1, 2, 4, 8]}
        current_config = {'x': [1, 2], 'y': [], 'z': [1, 2]}
        
        fourier = AxisSelectiveFourierFeatures(
            axes_config=current_config,
            full_axes_config=full_config,
        )
        
        # 大批量測試
        x = torch.randn(10000, 3)
        
        import time
        start = time.time()
        for _ in range(100):
            features = fourier(x)
        elapsed = time.time() - start
        
        # 平均每次前向傳播應 <10ms（寬鬆閾值）
        avg_time = elapsed / 100
        assert avg_time < 0.01, f"前向傳播過慢：{avg_time:.4f}s"


# ========== 測試運行配置 ==========

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
