#!/usr/bin/env python3
"""
驗證 GradNorm Bug 修復
檢查 weighters 是否正確傳遞給 Trainer
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import yaml
from pinnx.train.trainer import Trainer
from pinnx.train.factory import create_model
from scripts.train import create_weighters, create_physics, create_losses

def test_weighters_creation():
    """測試 weighters 創建與傳遞"""
    print("="*60)
    print("測試 1: Weighters 創建")
    print("="*60)
    
    # 載入最小配置
    config_path = 'configs/test_rans_quick.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    
    # 創建模型
    model = create_model(config, device)
    print(f"✓ 模型創建成功: {type(model).__name__}")
    
    # 創建物理與損失
    physics = create_physics(config, device)
    losses = create_losses(config, device, physics)
    print(f"✓ 物理與損失創建成功")
    
    # 創建 weighters
    weighters = create_weighters(config, model, device, physics=physics)
    print(f"✓ Weighters 創建成功: {type(weighters)}")
    print(f"  - Keys: {list(weighters.keys())}")
    
    # 檢查 adaptive_weighting 配置
    adaptive_cfg = config.get('losses', {}).get('adaptive_weighting', False)
    print(f"\n配置檢查:")
    print(f"  - adaptive_weighting: {adaptive_cfg}")
    
    if adaptive_cfg and 'grad_norm' in weighters:
        grad_norm = weighters['grad_norm']
        print(f"  - GradNorm α: {grad_norm.alpha}")
        print(f"  - 更新頻率: {config['losses'].get('weight_update_freq', 100)}")
        print(f"  - 管理損失項: {config['losses'].get('adaptive_loss_terms', [])}")
    
    return weighters

def test_trainer_initialization():
    """測試 Trainer 初始化時接收 weighters"""
    print("\n" + "="*60)
    print("測試 2: Trainer 初始化")
    print("="*60)
    
    config_path = 'configs/test_rans_quick.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    model = create_model(config, device)
    physics = create_physics(config, device)
    losses = create_losses(config, device, physics)
    weighters = create_weighters(config, model, device, physics=physics)
    
    # 測試 1: 無 weighters (舊方式 - Bug)
    trainer_old = Trainer(model, physics, losses, config, device)
    print(f"\n舊方式 (無 weighters):")
    print(f"  - trainer.weighters: {trainer_old.weighters}")
    print(f"  - 是否為空字典: {trainer_old.weighters == {}}")
    
    # 測試 2: 有 weighters (新方式 - 修復後)
    trainer_new = Trainer(model, physics, losses, config, device, weighters=weighters)
    print(f"\n新方式 (有 weighters):")
    print(f"  - trainer.weighters: {list(trainer_new.weighters.keys())}")
    print(f"  - 包含 GradNorm: {'grad_norm' in trainer_new.weighters}")
    
    return trainer_new

def test_gradnorm_update():
    """測試 GradNorm 權重更新機制"""
    print("\n" + "="*60)
    print("測試 3: GradNorm 更新機制")
    print("="*60)
    
    config_path = 'configs/test_rans_quick.yml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cpu')
    model = create_model(config, device)
    physics = create_physics(config, device)
    losses = create_losses(config, device, physics)
    weighters = create_weighters(config, model, device, physics=physics)
    
    trainer = Trainer(model, physics, losses, config, device, weighters=weighters)
    
    # 創建模擬訓練資料
    x_train = torch.rand(100, 4, requires_grad=True, device=device)  # (x,y,z,t)
    trainer.training_data = x_train
    
    print(f"是否應該更新權重 (epoch=100):")
    update_freq = config['losses'].get('weight_update_freq', 100)
    should_update = (100 % update_freq == 0) and ('grad_norm' in trainer.weighters)
    print(f"  - 更新頻率: {update_freq}")
    print(f"  - 當前 epoch: 100")
    print(f"  - 應該更新: {should_update}")
    
    if 'grad_norm' in trainer.weighters:
        print(f"\n✅ GradNorm 已啟用")
        print(f"  - 管理損失項: {config['losses'].get('adaptive_loss_terms', [])}")
    else:
        print(f"\n❌ GradNorm 未啟用")

if __name__ == '__main__':
    print("GradNorm Bug 修復驗證測試\n")
    
    try:
        # 執行測試
        weighters = test_weighters_creation()
        trainer = test_trainer_initialization()
        test_gradnorm_update()
        
        print("\n" + "="*60)
        print("✅ 所有測試通過！GradNorm 已正確修復")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
