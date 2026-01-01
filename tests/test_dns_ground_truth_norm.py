"""
測試 dns_ground_truth_norm 標準化類型

驗證：
1. 能正確從 DNS HDF5 文件載入數據
2. 統計量計算正確（均值、標準差）
3. 時間範圍篩選功能正常
4. 錯誤處理完善（文件不存在、變量缺失等）
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import h5py

from pinnx.utils.normalization import OutputTransform


class TestDNSGroundTruthNorm:
    """測試 DNS Ground Truth 標準化"""
    
    @pytest.fixture
    def dns_file_path(self):
        """DNS 文件路徑"""
        return "./data/kolmogorov_dns/dns_re50_t100.h5"
    
    @pytest.fixture
    def config_full_time(self, dns_file_path):
        """配置：使用全部時間範圍"""
        return {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v', 'p'],
                'params': {
                    'dns_file': dns_file_path
                }
            }
        }
    
    @pytest.fixture
    def config_partial_time(self, dns_file_path):
        """配置：使用部分時間範圍 [15, 35]"""
        return {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v', 'p'],
                'params': {
                    'dns_file': dns_file_path,
                    'time_range': [15.0, 35.0]
                }
            }
        }
    
    def test_dns_file_exists(self, dns_file_path):
        """測試：DNS 文件存在"""
        assert Path(dns_file_path).exists(), f"DNS 文件不存在: {dns_file_path}"
    
    def test_create_normalizer_full_time(self, config_full_time):
        """測試：創建標準化器（全部時間）"""
        normalizer = OutputTransform.from_config(
            norm_type='dns_ground_truth_norm',
            params=config_full_time['normalization']['params'],
            variable_order=config_full_time['normalization']['variable_order'],
            training_data=None,
            config=config_full_time
        )
        
        # 驗證標準化器屬性
        assert normalizer.norm_type == 'dns_ground_truth_norm'
        assert set(normalizer.variable_order) == {'u', 'v', 'p'}
        assert len(normalizer.means) == 3
        assert len(normalizer.stds) == 3
        
        # 驗證統計量在合理範圍內
        for var in ['u', 'v', 'p']:
            assert var in normalizer.means
            assert var in normalizer.stds
            assert np.isfinite(normalizer.means[var])
            assert np.isfinite(normalizer.stds[var])
            assert normalizer.stds[var] > 0
        
        print(f"\n✅ 全時間範圍統計量:")
        for var in ['u', 'v', 'p']:
            print(f"   {var}: mean={normalizer.means[var]:+.6f}, std={normalizer.stds[var]:.6f}")
    
    def test_create_normalizer_partial_time(self, config_partial_time):
        """測試：創建標準化器（部分時間範圍）"""
        normalizer = OutputTransform.from_config(
            norm_type='dns_ground_truth_norm',
            params=config_partial_time['normalization']['params'],
            variable_order=config_partial_time['normalization']['variable_order'],
            training_data=None,
            config=config_partial_time
        )
        
        # 驗證標準化器屬性
        assert normalizer.norm_type == 'dns_ground_truth_norm'
        assert set(normalizer.variable_order) == {'u', 'v', 'p'}
        
        print(f"\n✅ 部分時間範圍統計量 [15.0, 35.0]:")
        for var in ['u', 'v', 'p']:
            print(f"   {var}: mean={normalizer.means[var]:+.6f}, std={normalizer.stds[var]:.6f}")
    
    def test_compare_with_manual_calculation(self, dns_file_path):
        """測試：與手動計算對比驗證"""
        # 手動計算統計量
        with h5py.File(dns_file_path, 'r') as f:
            time = f['time'][:]
            time_mask = (time >= 15.0) & (time <= 35.0)
            time_indices = np.where(time_mask)[0]
            
            u_data = f['u'][time_indices, :, :]
            v_data = f['v'][time_indices, :, :]
            p_data = f['p'][time_indices, :, :]
            
            u_mean_expected = float(np.mean(u_data))
            u_std_expected = float(np.std(u_data))
            v_mean_expected = float(np.mean(v_data))
            v_std_expected = float(np.std(v_data))
            p_mean_expected = float(np.mean(p_data))
            p_std_expected = float(np.std(p_data))
        
        # 使用 dns_ground_truth_norm 計算
        config = {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v', 'p'],
                'params': {
                    'dns_file': dns_file_path,
                    'time_range': [15.0, 35.0]
                }
            }
        }
        
        normalizer = OutputTransform.from_config(
            norm_type='dns_ground_truth_norm',
            params=config['normalization']['params'],
            variable_order=['u', 'v', 'p'],
            training_data=None,
            config=config
        )
        
        # 驗證一致性（容忍 1e-6 誤差）
        assert abs(normalizer.means['u'] - u_mean_expected) < 1e-6, \
            f"u_mean 不匹配: {normalizer.means['u']} vs {u_mean_expected}"
        assert abs(normalizer.stds['u'] - u_std_expected) < 1e-6, \
            f"u_std 不匹配: {normalizer.stds['u']} vs {u_std_expected}"
        
        assert abs(normalizer.means['v'] - v_mean_expected) < 1e-6
        assert abs(normalizer.stds['v'] - v_std_expected) < 1e-6
        
        assert abs(normalizer.means['p'] - p_mean_expected) < 1e-6
        assert abs(normalizer.stds['p'] - p_std_expected) < 1e-6
        
        print(f"\n✅ 統計量驗證通過（與手動計算一致）:")
        print(f"   u: mean={normalizer.means['u']:+.6f} (expected {u_mean_expected:+.6f})")
        print(f"      std={normalizer.stds['u']:.6f} (expected {u_std_expected:.6f})")
    
    def test_error_file_not_found(self):
        """測試：DNS 文件不存在時的錯誤處理"""
        config = {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v', 'p'],
                'params': {
                    'dns_file': './nonexistent_file.h5'
                }
            }
        }
        
        with pytest.raises(FileNotFoundError):
            OutputTransform.from_config(
                norm_type='dns_ground_truth_norm',
                params=config['normalization']['params'],
                variable_order=['u', 'v', 'p'],
                training_data=None,
                config=config
            )
    
    def test_error_missing_dns_file_param(self):
        """測試：缺少 dns_file 參數時的錯誤處理"""
        config = {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v', 'p'],
                'params': {}  # 缺少 dns_file
            }
        }
        
        with pytest.raises(ValueError, match="dns_ground_truth_norm 需要提供 dns_file 參數"):
            OutputTransform.from_config(
                norm_type='dns_ground_truth_norm',
                params=config['normalization']['params'],
                variable_order=['u', 'v', 'p'],
                training_data=None,
                config=config
            )
    
    def test_normalize_denormalize_roundtrip(self, config_partial_time):
        """測試：標準化 → 反標準化 往返一致性"""
        normalizer = OutputTransform.from_config(
            norm_type='dns_ground_truth_norm',
            params=config_partial_time['normalization']['params'],
            variable_order=['u', 'v', 'p'],
            training_data=None,
            config=config_partial_time
        )
        
        # 創建測試數據
        original = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        
        # 標準化 → 反標準化
        normalized = normalizer.normalize(original, 'u')
        recovered = normalizer.denormalize(normalized, 'u')
        
        # 驗證往返一致性
        assert torch.allclose(original, recovered, atol=1e-6), \
            f"往返不一致: {original} -> {normalized} -> {recovered}"
        
        print(f"\n✅ 標準化往返測試通過:")
        print(f"   原始: {original.tolist()}")
        print(f"   標準化: {normalized.tolist()}")
        print(f"   恢復: {recovered.tolist()}")
    
    def test_variable_order_flexibility(self, dns_file_path):
        """測試：variable_order 靈活性（只計算所需變量）"""
        # 只計算 u 和 v
        config = {
            'normalization': {
                'type': 'dns_ground_truth_norm',
                'variable_order': ['u', 'v'],
                'params': {
                    'dns_file': dns_file_path,
                    'time_range': [15.0, 35.0]
                }
            }
        }
        
        normalizer = OutputTransform.from_config(
            norm_type='dns_ground_truth_norm',
            params=config['normalization']['params'],
            variable_order=['u', 'v'],
            training_data=None,
            config=config
        )
        
        # 驗證只有 u 和 v
        assert set(normalizer.variable_order) == {'u', 'v'}
        assert 'u' in normalizer.means
        assert 'v' in normalizer.means
        assert 'p' not in normalizer.means
        
        print(f"\n✅ 靈活 variable_order 測試通過（只計算 u, v）")


if __name__ == '__main__':
    # 運行測試
    pytest.main([__file__, '-v', '-s'])
