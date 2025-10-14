"""
學習率調度器測試
測試 Trainer 對所有調度器類型的支援
"""

import pytest
import torch
import torch.nn as nn
from pinnx.train.trainer import Trainer


class DummyModel(nn.Module):
    """測試用的簡單模型"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
    
    def forward(self, x):
        return self.fc(x)


class DummyPhysics(nn.Module):
    """測試用的簡單物理模組"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return torch.zeros(x.shape[0], 1)


def create_config(scheduler_type: str, **scheduler_params):
    """創建測試用配置"""
    return {
        'training': {
            'epochs': 100,
            'optimizer': {'type': 'adam', 'lr': 0.001, 'weight_decay': 0.0},
            'lr_scheduler': {
                'type': scheduler_type,
                **scheduler_params
            }
        },
        'losses': {'weights': {}},
        'data': {}
    }


# ============================================================================
# Trainer 調度器初始化測試
# ============================================================================

@pytest.mark.parametrize("scheduler_type,expected_class", [
    ('warmup_cosine', 'WarmupCosineScheduler'),
    ('cosine_warm_restarts', 'CosineAnnealingWarmRestarts'),
    ('cosine', 'CosineAnnealingLR'),
    ('exponential', 'ExponentialLR'),
    ('step', 'StepLR'),
    ('none', None),
    (None, None),
])
def test_trainer_scheduler_initialization(scheduler_type, expected_class):
    """測試 Trainer 可正確初始化所有調度器類型"""
    config = create_config(
        scheduler_type,
        warmup_epochs=5,
        T_0=20,
        gamma=0.95,
        min_lr=1e-6,
        step_size=30
    )
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    if expected_class is None:
        assert trainer.lr_scheduler is None, f"調度器應為 None，但得到 {type(trainer.lr_scheduler)}"
    else:
        assert trainer.lr_scheduler is not None, f"調度器不應為 None（類型: {scheduler_type}）"
        actual_class = type(trainer.lr_scheduler).__name__
        assert actual_class == expected_class, \
            f"預期調度器類型 {expected_class}，但得到 {actual_class}"


def test_trainer_unknown_scheduler_type():
    """測試未知調度器類型應回退到固定學習率"""
    config = create_config('unknown_scheduler_type')
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    # 應該回退到 None（固定學習率）
    assert trainer.lr_scheduler is None


# ============================================================================
# WarmupCosineScheduler 功能測試
# ============================================================================

def test_warmup_cosine_scheduler_warmup_phase():
    """測試 WarmupCosineScheduler 的 Warmup 階段"""
    from pinnx.train.schedulers import WarmupCosineScheduler
    
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=0.001)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=10,
        max_epochs=100,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    # 第1步（Warmup 階段）
    scheduler.step()
    lr_step1 = optimizer.param_groups[0]['lr']
    assert 0 < lr_step1 < 0.001, f"Warmup 第1步學習率應在 (0, 0.001)，但得到 {lr_step1}"
    
    # 第5步（Warmup 中期）
    for _ in range(4):
        scheduler.step()
    lr_step5 = optimizer.param_groups[0]['lr']
    assert lr_step1 < lr_step5 < 0.001, f"Warmup 學習率應逐步增加"
    
    # 第10步（Warmup 結束）
    for _ in range(5):
        scheduler.step()
    lr_step10 = optimizer.param_groups[0]['lr']
    assert abs(lr_step10 - 0.001) < 1e-8, f"Warmup 結束時學習率應達到 base_lr (0.001)，但得到 {lr_step10}"


def test_warmup_cosine_scheduler_cosine_phase():
    """測試 WarmupCosineScheduler 的 CosineAnnealing 階段"""
    from pinnx.train.schedulers import WarmupCosineScheduler
    
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=0.001)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=50,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    # 跳過 Warmup 階段
    for _ in range(5):
        scheduler.step()
    
    # CosineAnnealing 階段：學習率應逐漸下降
    lr_prev = optimizer.param_groups[0]['lr']
    for _ in range(20):
        scheduler.step()
        lr_current = optimizer.param_groups[0]['lr']
        assert lr_current < lr_prev, f"CosineAnnealing 階段學習率應下降"
        lr_prev = lr_current
    
    # 最終學習率應接近 min_lr
    for _ in range(25):  # 到達 epoch 50
        scheduler.step()
    lr_final = optimizer.param_groups[0]['lr']
    assert lr_final < 0.001 and lr_final >= 1e-6, \
        f"最終學習率應在 [min_lr, base_lr) 範圍內，但得到 {lr_final}"


def test_warmup_cosine_scheduler_get_last_lr():
    """測試 WarmupCosineScheduler.get_last_lr() 接口"""
    from pinnx.train.schedulers import WarmupCosineScheduler
    
    optimizer = torch.optim.Adam([torch.nn.Parameter(torch.randn(2, 2))], lr=0.001)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=5,
        max_epochs=50,
        base_lr=0.001,
        min_lr=1e-6
    )
    
    # 初始狀態
    last_lr = scheduler.get_last_lr()
    assert isinstance(last_lr, list), "get_last_lr() 應返回列表"
    assert len(last_lr) == 1, "應該只有一個參數組"
    
    # 執行幾步後
    scheduler.step()
    last_lr = scheduler.get_last_lr()
    assert last_lr[0] == optimizer.param_groups[0]['lr'], \
        "get_last_lr() 應與優化器的實際學習率一致"


# ============================================================================
# PyTorch 標準調度器基本功能測試
# ============================================================================

def test_cosine_annealing_warm_restarts():
    """測試 CosineAnnealingWarmRestarts 調度器"""
    config = create_config('cosine_warm_restarts', T_0=20, T_mult=1, eta_min=1e-6)
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    # 應正確初始化
    assert trainer.lr_scheduler is not None
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
    
    # 測試學習率變化（基本健全性檢查）
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    for _ in range(10):
        trainer.lr_scheduler.step()
    lr_after_10 = trainer.optimizer.param_groups[0]['lr']
    assert lr_after_10 < initial_lr, "學習率應在衰減"


def test_exponential_lr():
    """測試 ExponentialLR 調度器"""
    config = create_config('exponential', gamma=0.95)
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    # 應正確初始化
    assert trainer.lr_scheduler is not None
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.ExponentialLR)
    
    # 測試指數衰減
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    for _ in range(5):
        trainer.lr_scheduler.step()
    lr_after_5 = trainer.optimizer.param_groups[0]['lr']
    
    # 計算預期學習率：lr * gamma^5
    expected_lr = initial_lr * (0.95 ** 5)
    assert abs(lr_after_5 - expected_lr) < 1e-8, \
        f"指數衰減計算錯誤：預期 {expected_lr}，得到 {lr_after_5}"


def test_step_lr():
    """測試 StepLR 調度器"""
    config = create_config('step', step_size=10, gamma=0.5)
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    # 應正確初始化
    assert trainer.lr_scheduler is not None
    assert isinstance(trainer.lr_scheduler, torch.optim.lr_scheduler.StepLR)
    
    # 測試階梯式衰減
    initial_lr = trainer.optimizer.param_groups[0]['lr']
    
    # 前9步學習率不變
    for _ in range(9):
        trainer.lr_scheduler.step()
    lr_after_9 = trainer.optimizer.param_groups[0]['lr']
    assert abs(lr_after_9 - initial_lr) < 1e-8, "step_size 內學習率應保持不變"
    
    # 第10步後應減半
    trainer.lr_scheduler.step()
    lr_after_10 = trainer.optimizer.param_groups[0]['lr']
    expected_lr = initial_lr * 0.5
    assert abs(lr_after_10 - expected_lr) < 1e-8, \
        f"第10步後學習率應減半：預期 {expected_lr}，得到 {lr_after_10}"


# ============================================================================
# 整合測試：學習率調度器更新
# ============================================================================

def test_scheduler_step_updates_lr():
    """測試調度器 step() 方法正確更新學習率"""
    config = create_config('cosine', min_lr=1e-6)
    config['training']['epochs'] = 50
    
    model = DummyModel()
    physics = DummyPhysics()
    trainer = Trainer(model, physics, {}, config, torch.device('cpu'))
    
    # 記錄學習率變化
    lr_history = [trainer.optimizer.param_groups[0]['lr']]
    
    for _ in range(10):
        # 調度器更新
        if trainer.lr_scheduler is not None:
            trainer.lr_scheduler.step()
        lr_history.append(trainer.optimizer.param_groups[0]['lr'])
    
    # 驗證學習率有變化
    assert len(set(lr_history)) > 1, "學習率應該隨訓練變化"
    assert lr_history[-1] < lr_history[0], "CosineAnnealing 應使學習率下降"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

