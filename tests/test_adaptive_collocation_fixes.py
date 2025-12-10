#!/usr/bin/env python3
"""
測試 Adaptive Collocation Sampler Bug 修復
驗證兩個主要 bug 已修復：
1. 槓桿分數計算的負步長問題
2. 空間約束中的梯度追蹤錯誤
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pytest

from pinnx.train.adaptive_collocation import AdaptiveCollocationSampler


class TestAdaptiveCollocationFixes:
    """測試 Adaptive Collocation Sampler 的 bug 修復"""
    
    @pytest.fixture
    def sampler_config(self):
        """標準採樣器配置"""
        return {
            'enabled': True,
            'trigger': {
                'method': 'epoch_interval',
                'epoch_interval': 1000,
            },
            'resampling_strategy': 'incremental_replace',
            'incremental_replace': {
                'keep_ratio': 0.7,
                'replace_ratio': 0.3,
                'removal_criterion': 'leverage_score',
            },
            'residual_qr': {
                'enabled': True,
                'candidate_pool_size': 1000,
                'snapshot_config': {'n_snapshots': 10, 'snapshot_method': 'batch'},
                'svd': {'energy_threshold': 0.99, 'max_rank': 20},
                'qr': {'column_pivoting': True},
                'spatial_constraints': {'enabled': True, 'min_distance': 0.02}
            }
        }
    
    @pytest.fixture
    def mock_residual_fn(self):
        """模擬殘差函數"""
        def residual_fn(points):
            N = points.shape[0]
            # 模擬 NS 方程殘差 (u, v, continuity, boundary)
            residuals = torch.randn(N, 4) * 0.1
            return residuals
        return residual_fn
    
    def test_leverage_score_negative_stride_fix(self, sampler_config, mock_residual_fn):
        """測試槓桿分數計算的負步長問題已修復"""
        sampler = AdaptiveCollocationSampler(sampler_config)
        
        # 創建測試點
        points = torch.rand(100, 2)
        
        # 執行槓桿分數選點（之前會因負步長失敗）
        try:
            selected_indices = sampler._select_by_leverage_score(
                points, mock_residual_fn, n_select=70, device='cpu'
            )
            
            # 驗證結果
            assert isinstance(selected_indices, torch.Tensor)
            assert len(selected_indices) == 70
            assert selected_indices.min() >= 0
            assert selected_indices.max() < 100
            
            print("✅ 槓桿分數計算（負步長修復）測試通過")
            
        except RuntimeError as e:
            if "negative stride" in str(e):
                pytest.fail("❌ 負步長問題未修復！")
            else:
                raise
    
    def test_spatial_constraints_gradient_fix(self, sampler_config):
        """測試空間約束中的梯度追蹤錯誤已修復"""
        sampler = AdaptiveCollocationSampler(sampler_config)
        
        # 創建帶梯度的張量（模擬訓練中的情況）
        existing_points = torch.rand(10, 2, requires_grad=True)
        new_points = torch.rand(20, 2, requires_grad=True)
        
        # 執行空間約束過濾（之前會因未 detach 而失敗）
        try:
            filtered_points = sampler._apply_spatial_constraints(
                existing_points, new_points, min_distance=0.05
            )
            
            # 驗證結果
            assert isinstance(filtered_points, torch.Tensor)
            assert len(filtered_points) <= len(new_points)
            
            print("✅ 空間約束（梯度追蹤修復）測試通過")
            
        except RuntimeError as e:
            if "requires grad" in str(e):
                pytest.fail("❌ 梯度追蹤問題未修復！")
            else:
                raise
    
    def test_complete_resampling_workflow(self, sampler_config, mock_residual_fn):
        """測試完整重採樣流程"""
        sampler = AdaptiveCollocationSampler(sampler_config)
        
        # 創建測試點
        current_points = torch.rand(100, 2)
        domain_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        
        # 執行完整重採樣
        new_points, metrics = sampler.resample_collocation_points(
            current_points=current_points,
            domain_bounds=domain_bounds,
            residual_fn=mock_residual_fn,
            device='cpu'
        )
        
        # 驗證結果
        assert isinstance(new_points, torch.Tensor)
        assert new_points.shape[1] == 2  # 2D 點
        assert len(new_points) <= 100  # 總點數可能因空間約束而減少
        assert len(new_points) >= 90  # 但不會減少太多
        
        # 驗證指標
        assert metrics['n_kept'] == 70  # keep_ratio=0.7
        assert metrics['n_replaced'] <= 30  # replace_ratio=0.3 (可能因空間約束而減少)
        assert 'svd_rank' in metrics
        assert 'svd_energy_ratio' in metrics
        
        print(f"✅ 完整重採樣流程測試通過")
        print(f"   保留: {metrics['n_kept']} 點")
        print(f"   替換: {metrics['n_replaced']} 點")
        print(f"   SVD 秩: {metrics['svd_rank']}")
        print(f"   能量比: {metrics['svd_energy_ratio']:.4f}")
    
    def test_trigger_conditions(self, sampler_config):
        """測試觸發條件"""
        sampler = AdaptiveCollocationSampler(sampler_config)
        
        # 測試 epoch 間隔觸發
        assert sampler.should_trigger(1000, 0.1) == True
        assert sampler.should_trigger(500, 0.1) == False
        assert sampler.should_trigger(2000, 0.1) == True
        
        print("✅ 觸發條件測試通過")
    
    def test_candidate_pool_generation(self, sampler_config):
        """測試候選池生成"""
        sampler = AdaptiveCollocationSampler(sampler_config)
        
        domain_bounds = {'x': (0.0, 1.0), 'y': (0.0, 1.0)}
        
        # 測試不同採樣策略
        for method in ['latin_hypercube', 'uniform', 'stratified']:
            sampler.candidate_sampling = method
            pool = sampler._generate_candidate_pool(domain_bounds, pool_size=500, dim=2)
            
            assert pool.shape == (500, 2)
            assert pool.min() >= -0.1  # 允許一些數值誤差
            assert pool.max() <= 1.1
            
        print("✅ 候選池生成測試通過")


if __name__ == "__main__":
    """手動運行測試"""
    print("="*60)
    print("🧪 Adaptive Collocation Sampler Bug 修復測試")
    print("="*60)
    print()
    
    # 創建測試實例
    test = TestAdaptiveCollocationFixes()
    
    # 創建 fixtures
    config = {
        'enabled': True,
        'trigger': {'method': 'epoch_interval', 'epoch_interval': 1000},
        'resampling_strategy': 'incremental_replace',
        'incremental_replace': {'keep_ratio': 0.7, 'replace_ratio': 0.3, 'removal_criterion': 'leverage_score'},
        'residual_qr': {
            'enabled': True,
            'candidate_pool_size': 1000,
            'snapshot_config': {'n_snapshots': 10, 'snapshot_method': 'batch'},
            'svd': {'energy_threshold': 0.99, 'max_rank': 20},
            'qr': {'column_pivoting': True},
            'spatial_constraints': {'enabled': True, 'min_distance': 0.02}
        }
    }
    
    def mock_residual_fn(points):
        N = points.shape[0]
        return torch.randn(N, 4) * 0.1
    
    # 運行測試
    print("測試 1: 槓桿分數計算（負步長修復）")
    print("-" * 60)
    test.test_leverage_score_negative_stride_fix(config, mock_residual_fn)
    print()
    
    print("測試 2: 空間約束（梯度追蹤修復）")
    print("-" * 60)
    test.test_spatial_constraints_gradient_fix(config)
    print()
    
    print("測試 3: 完整重採樣流程")
    print("-" * 60)
    test.test_complete_resampling_workflow(config, mock_residual_fn)
    print()
    
    print("測試 4: 觸發條件")
    print("-" * 60)
    test.test_trigger_conditions(config)
    print()
    
    print("測試 5: 候選池生成")
    print("-" * 60)
    test.test_candidate_pool_generation(config)
    print()
    
    print("="*60)
    print("✅ 所有測試通過！")
    print("="*60)
