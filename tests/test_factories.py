"""
Registry Pattern 工廠模組單元測試

測試覆蓋範圍：
- create_optimizer(): 優化器創建（Registry Pattern）
- create_scheduler(): 學習率調度器創建（Registry Pattern）
- 各種優化器類型（Adam, AdamW, SOAP, LBFGS, SGD）
- 各種調度器類型（Cosine, Step, WarmupCosine, Exponential）
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from pinnx.train.factories import (
    create_optimizer,
    create_scheduler,
    list_available_optimizers,
    list_available_schedulers,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def dummy_model():
    """虛擬模型（用於優化器測試）"""
    return nn.Sequential(
        nn.Linear(4, 64),
        nn.Tanh(),
        nn.Linear(64, 4),
    )


@pytest.fixture
def dummy_optimizer(dummy_model):
    """虛擬優化器（用於調度器測試）"""
    return torch.optim.Adam(dummy_model.parameters(), lr=0.001)


# ============================================================================
# Test: Registry 查詢工具
# ============================================================================

class TestRegistryQuery:
    """測試 Registry 查詢功能"""
    
    def test_list_available_optimizers(self):
        """測試列出所有已註冊的優化器類型"""
        optimizers = list_available_optimizers()
        
        assert isinstance(optimizers, list)
        assert 'adam' in optimizers
        assert 'adamw' in optimizers
        assert 'soap' in optimizers
        assert len(optimizers) >= 4  # 至少有 adam, adamw, soap, lbfgs
    
    def test_list_available_schedulers(self):
        """測試列出所有已註冊的調度器類型"""
        schedulers = list_available_schedulers()
        
        assert isinstance(schedulers, list)
        assert 'cosine' in schedulers
        assert 'step' in schedulers
        assert 'warmup_cosine' in schedulers
        assert len(schedulers) >= 4


# ============================================================================
# Test: create_optimizer() (Registry Pattern)
# ============================================================================

class TestCreateOptimizer:
    """測試優化器創建（Registry Pattern）"""
    
    def test_adam_optimizer(self, dummy_model):
        """測試 Adam 優化器"""
        config = {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-4,
        }
        
        optimizer = create_optimizer(dummy_model, config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['lr'] == 0.001
        assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(1e-4)
    
    def test_adamw_optimizer(self, dummy_model):
        """測試 AdamW 優化器"""
        config = {
            'type': 'adamw',
            'lr': 0.002,
            'weight_decay': 0.01,
        }
        
        optimizer = create_optimizer(dummy_model, config)
        
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]['lr'] == 0.002
        assert optimizer.param_groups[0]['weight_decay'] == pytest.approx(0.01)
    
    @pytest.mark.skipif(True, reason="SOAP 需要專門的安裝")
    def test_soap_optimizer(self, dummy_model):
        """測試 SOAP 優化器（需要 pinnx.optim.SOAP）"""
        config = {
            'type': 'soap',
            'lr': 0.001,
            'shampoo_beta': 0.9,
        }
        
        try:
            optimizer = create_optimizer(dummy_model, config)
            assert optimizer.__class__.__name__ == 'SOAP'
        except ImportError:
            pytest.skip("SOAP 優化器未安裝")
    
    def test_sgd_optimizer(self, dummy_model):
        """測試 SGD 優化器"""
        config = {
            'type': 'sgd',
            'lr': 0.01,
            'momentum': 0.9,
        }
        
        optimizer = create_optimizer(dummy_model, config)
        
        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.param_groups[0]['lr'] == 0.01
        assert optimizer.param_groups[0]['momentum'] == pytest.approx(0.9)
    
    def test_unknown_optimizer_fallback(self, dummy_model):
        """測試未知優化器類型 → 回退到 Adam"""
        config = {
            'type': 'unknown_optimizer_type',
            'lr': 0.001,
        }
        
        optimizer = create_optimizer(dummy_model, config)
        
        # 應該回退到 Adam
        assert isinstance(optimizer, torch.optim.Adam)
    
    def test_string_config(self, dummy_model):
        """測試字串配置（應該回退到 Adam）"""
        config = 'adam'
        
        optimizer = create_optimizer(dummy_model, config)
        
        # 字串配置應該觸發 fallback
        assert isinstance(optimizer, torch.optim.Adam)


# ============================================================================
# Test: create_scheduler() (Registry Pattern)
# ============================================================================

class TestCreateScheduler:
    """測試學習率調度器創建（Registry Pattern）"""
    
    def test_cosine_scheduler(self, dummy_optimizer):
        """測試 Cosine Annealing 調度器"""
        config = {
            'type': 'cosine',
            'max_epochs': 1000,
            'eta_min': 0.0,
        }
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    
    def test_step_scheduler(self, dummy_optimizer):
        """測試 Step 調度器"""
        config = {
            'type': 'step',
            'step_size': 500,
            'gamma': 0.5,
        }
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    
    def test_warmup_cosine_scheduler(self, dummy_optimizer):
        """測試 Warmup + Cosine 調度器"""
        config = {
            'type': 'warmup_cosine',
            'warmup_epochs': 100,
            'max_epochs': 1000,
            'eta_min': 0.0,
        }
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is not None
        # 應該是自定義的 WarmupCosineScheduler
        assert scheduler.__class__.__name__ == 'WarmupCosineScheduler'
    
    def test_exponential_scheduler(self, dummy_optimizer):
        """測試 Exponential 調度器"""
        config = {
            'type': 'exponential',
            'gamma': 0.95,
        }
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is not None
        assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    
    def test_none_scheduler(self, dummy_optimizer):
        """測試 'none' 類型（無調度器）"""
        config = {'type': 'none'}
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is None
    
    def test_constant_scheduler(self, dummy_optimizer):
        """測試 'constant' 類型（無調度器）"""
        config = {'type': 'constant'}
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is None
    
    def test_unknown_scheduler(self, dummy_optimizer):
        """測試未知調度器類型 → 返回 None"""
        config = {'type': 'unknown_scheduler_type'}
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        assert scheduler is None
    
    def test_string_config(self, dummy_optimizer):
        """測試字串配置"""
        config = 'cosine'
        
        scheduler = create_scheduler(dummy_optimizer, config)
        
        # 字串配置應該被解析
        assert scheduler is not None or config in ['none', 'constant']


# ============================================================================
# Test: Integration（整合測試）
# ============================================================================

class TestIntegration:
    """測試優化器 + 調度器整合"""
    
    def test_optimizer_and_scheduler_together(self, dummy_model):
        """測試同時創建優化器和調度器"""
        # 創建優化器
        optimizer_config = {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0,
        }
        optimizer = create_optimizer(dummy_model, optimizer_config)
        
        # 創建調度器
        scheduler_config = {
            'type': 'cosine',
            'max_epochs': 1000,
        }
        scheduler = create_scheduler(optimizer, scheduler_config)
        
        # 驗證
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is not None
        
        # 測試調度器運行
        initial_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        updated_lr = optimizer.param_groups[0]['lr']
        
        # Cosine annealing 應該會稍微降低學習率
        assert updated_lr <= initial_lr
    
    def test_no_scheduler_integration(self, dummy_model):
        """測試只有優化器沒有調度器的情況"""
        optimizer_config = {
            'type': 'adam',
            'lr': 0.001,
        }
        optimizer = create_optimizer(dummy_model, optimizer_config)
        
        scheduler_config = {'type': 'none'}
        scheduler = create_scheduler(optimizer, scheduler_config)
        
        assert isinstance(optimizer, torch.optim.Adam)
        assert scheduler is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
