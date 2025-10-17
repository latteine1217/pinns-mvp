"""
完整的 Checkpoint Normalization 整合測試

測試完整的訓練→保存→載入→推理循環，確保標準化器能正確保存和恢復。

測試流程：
1. 訓練階段：建立標準化器並模擬訓練
2. 保存階段：將標準化器 metadata 保存到 checkpoint
3. 載入階段：從 checkpoint 恢復標準化器
4. 推理階段：使用恢復的標準化器進行預測

更新日期：2025-10-17 (Post Phase-5 NaN Fix)
"""

import pytest
import torch
import numpy as np
import tempfile
import os
from pathlib import Path

from pinnx.utils.normalization import OutputTransform, OutputNormConfig


class TestCheckpointNormalizationRoundtrip:
    """測試完整的 Checkpoint 循環"""
    
    @pytest.fixture
    def training_data_3d(self):
        """生成 3D 訓練資料（包含所有變量）"""
        np.random.seed(42)
        return {
            'u': torch.randn(1000) * 4.5 + 10.0,
            'v': torch.randn(1000) * 0.33,
            'w': torch.randn(1000) * 3.8 - 1.0,
            'p': torch.randn(1000) * 28.0 - 40.0
        }
    
    @pytest.fixture
    def training_data_2d(self):
        """生成 2D 訓練資料（無 w 分量）"""
        np.random.seed(123)
        return {
            'u': torch.randn(500) * 2.0 + 5.0,
            'v': torch.randn(500) * 0.5,
            'p': torch.randn(500) * 10.0 - 20.0
        }
    
    def test_full_checkpoint_cycle_3d(self, training_data_3d, tmp_path):
        """測試完整的 3D checkpoint 循環"""
        
        # === 階段 1: 訓練階段 ===
        # 建立標準化器
        normalizer_train = OutputTransform.from_data(training_data_3d)
        
        # 模擬標準化訓練資料
        normalized_data = {}
        for var_name in ['u', 'v', 'w', 'p']:
            normalized_data[var_name] = normalizer_train.normalize(
                training_data_3d[var_name], var_name
            )
        
        # 驗證標準化後統計量
        for var_name in ['u', 'v', 'w', 'p']:
            if isinstance(normalized_data[var_name], torch.Tensor):
                mean = normalized_data[var_name].mean().item()
                std = normalized_data[var_name].std(unbiased=True).item()
            else:
                mean = np.mean(normalized_data[var_name])
                std = np.std(normalized_data[var_name], ddof=1)
            
            assert abs(mean) < 0.1, f"{var_name}: mean = {mean:.4f}"
            assert abs(std - 1.0) < 0.1, f"{var_name}: std = {std:.4f}"
        
        # === 階段 2: 保存 Checkpoint ===
        checkpoint_path = tmp_path / "test_checkpoint.pth"
        
        # 模擬訓練器保存的 checkpoint 格式
        checkpoint = {
            'epoch': 100,
            'model_state_dict': {},  # 實際訓練中會有模型權重
            'optimizer_state_dict': {},
            'loss': 0.123,
            'normalization': normalizer_train.get_metadata()
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        # === 階段 3: 載入 Checkpoint ===
        loaded_checkpoint = torch.load(checkpoint_path)
        
        # 方法 1: 使用新的 from_metadata() 便利方法
        normalizer_restored = OutputTransform.from_metadata(
            loaded_checkpoint['normalization']
        )
        
        # === 階段 4: 推理階段 ===
        # 模擬新的預測資料
        test_predictions = {
            'u': torch.tensor([12.0, 8.0, 15.0]),
            'v': torch.tensor([0.1, -0.2, 0.3]),
            'w': torch.tensor([-2.0, 1.0, -3.0]),
            'p': torch.tensor([-35.0, -45.0, -30.0])
        }
        
        # 使用原始和恢復的標準化器進行標準化
        for var_name in ['u', 'v', 'w', 'p']:
            original_norm = normalizer_train.normalize(test_predictions[var_name], var_name)
            restored_norm = normalizer_restored.normalize(test_predictions[var_name], var_name)
            
            # 驗證完全一致
            diff = torch.abs(original_norm - restored_norm)
            assert diff.max().item() < 1e-10, (
                f"{var_name}: 標準化不一致，最大差異 {diff.max().item():.2e}"
            )
        
        # === 驗證反標準化 ===
        for var_name in ['u', 'v', 'w', 'p']:
            normalized = normalizer_restored.normalize(test_predictions[var_name], var_name)
            denormalized = normalizer_restored.denormalize(normalized, var_name)
            
            diff = torch.abs(denormalized - test_predictions[var_name])
            assert diff.max().item() < 1e-6, (
                f"{var_name}: 反標準化不一致，最大差異 {diff.max().item():.2e}"
            )
    
    def test_full_checkpoint_cycle_2d(self, training_data_2d, tmp_path):
        """測試 2D checkpoint 循環（無 w 分量）"""
        
        # === 階段 1: 訓練階段 ===
        normalizer_train = OutputTransform.from_data(training_data_2d)
        
        # 驗證變量順序不包含 w
        assert 'w' not in normalizer_train.variable_order
        assert set(normalizer_train.variable_order) == {'u', 'v', 'p'}
        
        # === 階段 2: 保存 ===
        checkpoint_path = tmp_path / "test_checkpoint_2d.pth"
        checkpoint = {
            'normalization': normalizer_train.get_metadata()
        }
        torch.save(checkpoint, checkpoint_path)
        
        # === 階段 3: 載入 ===
        loaded_checkpoint = torch.load(checkpoint_path)
        normalizer_restored = OutputTransform.from_metadata(
            loaded_checkpoint['normalization']
        )
        
        # 驗證恢復的變量順序正確
        assert normalizer_restored.variable_order == normalizer_train.variable_order
        assert 'w' not in normalizer_restored.variable_order
        
        # === 階段 4: 推理 ===
        test_predictions = {
            'u': torch.tensor([5.0, 7.0, 3.0]),
            'v': torch.tensor([0.0, 0.5, -0.3]),
            'p': torch.tensor([-20.0, -15.0, -25.0])
        }
        
        for var_name in ['u', 'v', 'p']:
            original_norm = normalizer_train.normalize(test_predictions[var_name], var_name)
            restored_norm = normalizer_restored.normalize(test_predictions[var_name], var_name)
            
            diff = torch.abs(original_norm - restored_norm)
            assert diff.max().item() < 1e-10
    
    def test_batch_operations_after_checkpoint_load(self, training_data_3d, tmp_path):
        """測試載入 checkpoint 後的批次操作"""
        
        # 建立並保存
        normalizer_train = OutputTransform.from_data(training_data_3d)
        checkpoint_path = tmp_path / "batch_test.pth"
        torch.save({'normalization': normalizer_train.get_metadata()}, checkpoint_path)
        
        # 載入
        loaded = torch.load(checkpoint_path)
        normalizer = OutputTransform.from_metadata(loaded['normalization'])
        
        # 批次標準化測試
        batch_size = 10
        batch_data = torch.randn(batch_size, 4)  # [N, 4] for u, v, w, p
        
        var_order = ['u', 'v', 'w', 'p']
        normalized_batch = normalizer.normalize_batch(batch_data, var_order=var_order)
        denormalized_batch = normalizer.denormalize_batch(normalized_batch, var_order=var_order)
        
        # 驗證恆等變換
        diff = torch.abs(denormalized_batch - batch_data)
        assert diff.max().item() < 1e-5
    
    def test_checkpoint_with_empty_tensor_handling(self, tmp_path):
        """測試 checkpoint 在處理空張量時的穩健性"""
        
        # 建立包含空張量的資料（模擬 2D 場）
        data_with_empty = {
            'u': torch.randn(100) * 2.0 + 5.0,
            'v': torch.randn(100) * 0.5,
            'w': torch.tensor([]),  # 空張量
            'p': torch.randn(100) * 10.0
        }
        
        # 建立標準化器（應自動跳過 w）
        normalizer_train = OutputTransform.from_data(data_with_empty)
        
        # 驗證 w 被排除
        assert 'w' not in normalizer_train.variable_order
        assert 'w' not in normalizer_train.means
        assert 'w' not in normalizer_train.stds
        
        # 保存並載入
        checkpoint_path = tmp_path / "empty_tensor_test.pth"
        torch.save({'normalization': normalizer_train.get_metadata()}, checkpoint_path)
        
        loaded = torch.load(checkpoint_path)
        normalizer_restored = OutputTransform.from_metadata(loaded['normalization'])
        
        # 驗證恢復後的標準化器也沒有 w
        assert 'w' not in normalizer_restored.variable_order
        assert 'w' not in normalizer_restored.means
        
        # 驗證正常變量功能正常
        test_value = torch.tensor([5.0, 7.0, 3.0])
        for var_name in ['u', 'v', 'p']:
            norm = normalizer_restored.normalize(test_value, var_name)
            denorm = normalizer_restored.denormalize(norm, var_name)
            diff = torch.abs(denorm - test_value)
            assert diff.max().item() < 1e-5
    
    def test_multiple_checkpoint_loads(self, training_data_3d, tmp_path):
        """測試多次載入同一個 checkpoint 的穩定性"""
        
        # 建立並保存
        normalizer_train = OutputTransform.from_data(training_data_3d)
        checkpoint_path = tmp_path / "multi_load_test.pth"
        torch.save({'normalization': normalizer_train.get_metadata()}, checkpoint_path)
        
        # 多次載入
        normalizers = []
        for i in range(5):
            loaded = torch.load(checkpoint_path)
            norm = OutputTransform.from_metadata(loaded['normalization'])
            normalizers.append(norm)
        
        # 驗證所有載入的標準化器完全一致
        test_value = torch.tensor([10.0, 15.0, 8.0])
        for var_name in ['u', 'v', 'w', 'p']:
            reference_norm = normalizers[0].normalize(test_value, var_name)
            
            for i in range(1, 5):
                other_norm = normalizers[i].normalize(test_value, var_name)
                diff = torch.abs(other_norm - reference_norm)
                assert diff.max().item() < 1e-10
    
    def test_checkpoint_metadata_integrity(self, training_data_3d, tmp_path):
        """測試 checkpoint metadata 的完整性"""
        
        normalizer = OutputTransform.from_data(training_data_3d)
        metadata = normalizer.get_metadata()
        
        # 驗證必要欄位存在
        required_fields = ['norm_type', 'variable_order', 'means', 'stds']
        for field in required_fields:
            assert field in metadata, f"缺少必要欄位: {field}"
        
        # 驗證資料類型正確
        assert isinstance(metadata['norm_type'], str)
        assert isinstance(metadata['variable_order'], list)
        assert isinstance(metadata['means'], dict)
        assert isinstance(metadata['stds'], dict)
        
        # 驗證變量一致性
        for var_name in metadata['variable_order']:
            assert var_name in metadata['means'], f"{var_name} 不在 means 中"
            assert var_name in metadata['stds'], f"{var_name} 不在 stds 中"
        
        # 驗證數值有效性
        for var_name, mean_val in metadata['means'].items():
            assert np.isfinite(mean_val), f"{var_name} 的 mean 不是有限值"
        
        for var_name, std_val in metadata['stds'].items():
            assert np.isfinite(std_val), f"{var_name} 的 std 不是有限值"
            assert std_val > 0, f"{var_name} 的 std 必須為正值"
    
    def test_backward_compatibility_with_old_checkpoints(self, tmp_path):
        """測試與舊格式 checkpoint 的向後相容性"""
        
        # 模擬舊格式 checkpoint（可能缺少某些欄位）
        old_checkpoint = {
            'normalization': {
                'norm_type': 'training_data_norm',
                'means': {'u': 10.0, 'v': 0.0, 'p': -40.0},
                'stds': {'u': 4.5, 'v': 0.33, 'p': 28.0}
                # 缺少 'variable_order'（應使用預設順序）
            }
        }
        
        checkpoint_path = tmp_path / "old_format.pth"
        torch.save(old_checkpoint, checkpoint_path)
        
        # 載入應該成功
        loaded = torch.load(checkpoint_path)
        normalizer = OutputTransform.from_metadata(loaded['normalization'])
        
        # 驗證功能正常
        assert normalizer.norm_type == 'training_data_norm'
        assert normalizer.variable_order is not None  # 應有預設順序
        
        test_value = 12.5
        normalized = normalizer.normalize(test_value, 'u')
        recovered = normalizer.denormalize(normalized, 'u')
        assert abs(float(recovered) - test_value) < 1e-6


class TestCheckpointNormalizationEdgeCases:
    """測試邊界情況和錯誤處理"""
    
    def test_corrupted_checkpoint_metadata(self):
        """測試損壞的 checkpoint metadata"""
        
        # 缺少必要欄位
        invalid_metadata = {
            'norm_type': 'training_data_norm'
            # 缺少 'means' 和 'stds'
        }
        
        with pytest.raises(KeyError):
            OutputTransform.from_metadata(invalid_metadata)
    
    def test_empty_checkpoint_metadata(self):
        """測試空的 checkpoint metadata"""
        
        with pytest.raises(KeyError):
            OutputTransform.from_metadata({})
    
    def test_mismatched_variable_orders(self, tmp_path):
        """測試變量順序不匹配的情況"""
        
        # 訓練時使用 3D 資料
        data_3d = {
            'u': torch.randn(100) * 2.0,
            'v': torch.randn(100) * 0.5,
            'w': torch.randn(100) * 1.5,
            'p': torch.randn(100) * 10.0
        }
        
        normalizer = OutputTransform.from_data(data_3d)
        checkpoint_path = tmp_path / "mismatch_test.pth"
        torch.save({'normalization': normalizer.get_metadata()}, checkpoint_path)
        
        # 載入並嘗試用 2D 順序（應該能處理，只用存在的變量）
        loaded = torch.load(checkpoint_path)
        normalizer_restored = OutputTransform.from_metadata(loaded['normalization'])
        
        # 標準化 2D 資料（只有 u, v, p）
        test_2d = torch.tensor([5.0, 0.0, -20.0])
        var_order_2d = ['u', 'v', 'p']
        
        # 應該能正常執行（跳過 w）
        normalized = normalizer_restored.normalize_batch(
            test_2d.unsqueeze(0), var_order=var_order_2d
        )
        assert normalized.shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
