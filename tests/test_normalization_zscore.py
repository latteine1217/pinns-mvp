"""
TASK-010: Z-score 標準化回歸測試

測試完整的 Z-score 標準化實現，確保：
1. 恆等變換：normalize → denormalize = identity
2. 統計特性：標準化後 mean ≈ 0, std ≈ 1
3. 邊界情況處理
"""

import pytest
import torch
import numpy as np
from pinnx.utils.normalization import DataNormalizer


class TestZScoreNormalization:
    """測試完整 Z-score 標準化"""
    
    @pytest.fixture
    def jhtdb_stats(self):
        """JHTDB Channel Re_tau=1000 統計量"""
        return {
            'means': {
                'u': 9.921185,
                'v': -0.000085,
                'w': -0.002202,
                'p': -40.374241
            },
            'stds': {
                'u': 4.593879,
                'v': 0.329614,
                'w': 3.865396,
                'p': 28.619722
            }
        }
    
    @pytest.fixture
    def sample_data(self):
        """生成測試樣本數據"""
        np.random.seed(42)
        return {
            'u': np.random.randn(100) * 4.5 + 10.0,
            'v': np.random.randn(100) * 0.33,
            'w': np.random.randn(100) * 3.8,
            'p': np.random.randn(100) * 28.0 - 40.0
        }
    
    def test_identity_transformation_numpy(self, jhtdb_stats):
        """測試 numpy 恆等變換：normalize → denormalize = identity"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        original_values = {
            'u': 12.5,
            'v': 0.1,
            'w': -2.0,
            'p': -35.0
        }
        
        for var_name, original in original_values.items():
            normalized = normalizer.normalize(original, var_name)
            recovered = normalizer.denormalize(normalized, var_name)
            
            # 容差 < 1e-6
            error = abs(float(recovered) - original)
            assert error < 1e-6, (
                f"{var_name}: 期望 {original}, 得到 {recovered}, 誤差 {error}"
            )
    
    def test_identity_transformation_torch(self, jhtdb_stats):
        """測試 PyTorch 張量恆等變換"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        original_values = {
            'u': torch.tensor([10.0, 15.0, 8.0]),
            'v': torch.tensor([0.0, 0.5, -0.3]),
            'w': torch.tensor([-1.0, 3.0, -5.0]),
            'p': torch.tensor([-40.0, -20.0, -60.0])
        }
        
        for var_name, original_tensor in original_values.items():
            # 標準化
            normalized = normalizer.normalize(original_tensor, var_name)
            
            # 反標準化
            recovered = normalizer.denormalize(normalized, var_name)
            
            # 檢查誤差
            if isinstance(recovered, torch.Tensor):
                diff = torch.abs(recovered - original_tensor)
                max_error = diff.max().item()
            else:
                diff = abs(recovered - original_tensor.numpy())
                max_error = np.max(diff)
            
            assert max_error < 1e-5, (
                f"{var_name}: 最大誤差 {max_error:.2e}"
            )
    
    def test_batch_normalization(self, jhtdb_stats, sample_data):
        """測試批次標準化與反標準化"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        # 轉為 PyTorch 張量
        batch = {
            key: torch.tensor(val, dtype=torch.float32)
            for key, val in sample_data.items()
        }
        
        # 批次標準化（逐個變量）
        normalized_batch = {}
        for var_name in ['u', 'v', 'w', 'p']:
            normalized_batch[var_name] = normalizer.normalize(batch[var_name], var_name)
        
        # 批次反標準化（逐個變量）
        recovered_batch = {}
        for var_name in ['u', 'v', 'w', 'p']:
            recovered_batch[var_name] = normalizer.denormalize(normalized_batch[var_name], var_name)
        
        # 檢查每個變量
        for var_name in ['u', 'v', 'w', 'p']:
            original = batch[var_name]
            recovered = recovered_batch[var_name]
            
            if isinstance(recovered, torch.Tensor):
                diff = torch.abs(recovered - original)
                max_error = diff.max().item()
            else:
                diff = np.abs(recovered - original.numpy())
                max_error = float(np.max(diff))
            
            assert max_error < 1e-5, (
                f"{var_name}: 最大誤差 {max_error:.2e}"
            )
    
    def test_normalized_statistics(self, jhtdb_stats, sample_data):
        """測試標準化後統計特性：mean ≈ 0, std ≈ 1"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        for var_name in ['u', 'v', 'w', 'p']:
            original = sample_data[var_name]
            normalized = normalizer.normalize(original, var_name)
            
            # 計算統計量
            if isinstance(normalized, torch.Tensor):
                mean = float(normalized.mean().item())
                std = float(normalized.std(unbiased=True).item())
            else:
                mean = float(np.mean(normalized))
                std = float(np.std(normalized, ddof=1))
            
            # 容差較寬鬆（因為樣本統計量與總體不完全一致）
            assert abs(mean) < 0.15, (
                f"{var_name}: 均值應接近 0，得到 {mean:.4f}"
            )
            assert abs(std - 1.0) < 0.15, (
                f"{var_name}: 標準差應接近 1，得到 {std:.4f}"
            )
    
    def test_edge_cases(self, jhtdb_stats):
        """測試邊界情況"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        # 測試零值
        normalized_zero = normalizer.normalize(0.0, 'u')
        expected_zero = -jhtdb_stats['means']['u'] / jhtdb_stats['stds']['u']
        assert abs(float(normalized_zero) - expected_zero) < 1e-6
        
        # 測試均值
        normalized_mean = normalizer.normalize(jhtdb_stats['means']['u'], 'u')
        assert abs(float(normalized_mean)) < 1e-6, "均值標準化後應為 0"
        
        # 測試極值
        extreme_value = 100.0
        normalized_extreme = normalizer.normalize(extreme_value, 'u')
        recovered_extreme = normalizer.denormalize(normalized_extreme, 'u')
        assert abs(float(recovered_extreme) - extreme_value) < 1e-5
    
    def test_metadata_roundtrip(self, jhtdb_stats):
        """測試 metadata 序列化與反序列化"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        # 獲取 metadata
        metadata = normalizer.get_metadata()
        
        # 確認包含 means 和 scales
        assert 'means' in metadata, "metadata 應包含 'means'"
        assert 'scales' in metadata, "metadata 應包含 'scales'"
        
        # 從 metadata 重建
        new_normalizer = DataNormalizer.from_metadata(metadata)
        
        # 測試功能一致
        test_value = 12.5
        original_norm = normalizer.normalize(test_value, 'u')
        new_norm = new_normalizer.normalize(test_value, 'u')
        
        assert abs(float(original_norm) - float(new_norm)) < 1e-10
    
    def test_from_data_zscore(self):
        """測試從數據直接計算 Z-score 參數"""
        np.random.seed(123)
        torch_data = {
            'u': torch.randn(1000) * 2.0 + 5.0,
            'v': torch.randn(1000) * 0.5,
            'w': torch.randn(1000) * 3.0 - 1.0,
            'p': torch.randn(1000) * 10.0 - 20.0
        }
        
        normalizer = DataNormalizer.from_data(torch_data)
        
        # 標準化後檢查統計量（逐個變量）
        for var_name in ['u', 'v', 'w', 'p']:
            norm_var = normalizer.normalize(torch_data[var_name], var_name)
            
            if isinstance(norm_var, torch.Tensor):
                mean = float(norm_var.mean().item())
                std = float(norm_var.std(unbiased=True).item())
            else:
                mean = float(np.mean(norm_var))
                std = float(np.std(norm_var, ddof=1))
            
            assert abs(mean) < 0.05, f"{var_name}: mean = {mean:.4f}"
            assert abs(std - 1.0) < 0.05, f"{var_name}: std = {std:.4f}"
    
    def test_backward_compatibility(self):
        """測試向後兼容：舊格式（僅 scales）應該發出警告"""
        # 模擬舊格式 metadata（只有 scales，沒有 means）
        old_metadata = {
            'type': 'training_data_norm',
            'scales': {'u': 4.5, 'v': 0.33, 'w': 3.8, 'p': 28.0}
        }
        
        # 應該能載入（使用零均值）
        normalizer = DataNormalizer.from_metadata(old_metadata)
        
        # 檢查降級為僅縮放模式（均值為 0）
        assert normalizer.means.get('u', 0.0) == 0.0
        assert normalizer.means.get('v', 0.0) == 0.0
    
    def test_correct_zscore_formula(self, jhtdb_stats):
        """明確驗證 Z-score 公式：(x - μ) / σ"""
        normalizer = DataNormalizer(
            norm_type='training_data_norm',
            scales=jhtdb_stats['stds'],
            means=jhtdb_stats['means']
        )
        
        x = 15.0
        var_name = 'u'
        
        # 手動計算
        mean = jhtdb_stats['means'][var_name]
        std = jhtdb_stats['stds'][var_name]
        expected = (x - mean) / std
        
        # 使用 normalizer
        actual = normalizer.normalize(x, var_name)
        
        assert abs(float(actual) - expected) < 1e-10, (
            f"期望 {expected:.6f}, 得到 {actual:.6f}"
        )
        
        # 反向計算
        recovered = normalizer.denormalize(actual, var_name)
        assert abs(float(recovered) - x) < 1e-10
    
    def test_config_based_initialization(self, jhtdb_stats):
        """測試從配置文件格式初始化"""
        config = {
            'normalization': {
                'type': 'training_data_norm',
                'params': {
                    'u_mean': jhtdb_stats['means']['u'],
                    'u_std': jhtdb_stats['stds']['u'],
                    'v_mean': jhtdb_stats['means']['v'],
                    'v_std': jhtdb_stats['stds']['v'],
                    'w_mean': jhtdb_stats['means']['w'],
                    'w_std': jhtdb_stats['stds']['w'],
                    'p_mean': jhtdb_stats['means']['p'],
                    'p_std': jhtdb_stats['stds']['p']
                }
            }
        }
        
        normalizer = DataNormalizer.from_config(config)
        
        # 驗證參數正確載入
        assert normalizer.norm_type == 'training_data_norm'
        assert abs(normalizer.means['u'] - jhtdb_stats['means']['u']) < 1e-6
        assert abs(normalizer.scales['u'] - jhtdb_stats['stds']['u']) < 1e-6
        
        # 驗證功能正常
        test_value = 12.5
        normalized = normalizer.normalize(test_value, 'u')
        recovered = normalizer.denormalize(normalized, 'u')
        assert abs(float(recovered) - test_value) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
