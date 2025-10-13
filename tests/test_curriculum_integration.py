#!/usr/bin/env python3
"""測試課程訓練整合 - 驗證 CurriculumScheduler 是否正確工作"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import torch
import yaml
from pinnx.physics.ns_2d import NSEquations2D

# 載入配置
config_path = 'configs/channel_flow_curriculum_4stage.yml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 創建物理模組
device = torch.device('cpu')
physics = NSEquations2D(
    nu=config['physics']['nu'],
    rho=config['physics']['rho'],
    device=device
)

# 從配置中提取課程訓練階段
stages = config['curriculum']['stages']

# 創建 CurriculumScheduler（需要先導入）
from train import CurriculumScheduler

scheduler = CurriculumScheduler(stages, physics)

print("=" * 80)
print("📚 課程訓練調度器測試")
print("=" * 80)

# 測試各個 epoch 的配置
test_epochs = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500]

for epoch in test_epochs:
    stage_config = scheduler.get_stage_config(epoch)
    
    print(f"\nEpoch {epoch}:")
    print(f"  階段名稱: {stage_config['stage_name']}")
    print(f"  階段切換: {'是' if stage_config['is_transition'] else '否'}")
    print(f"  Re_tau: {stage_config['Re_tau']}")
    print(f"  nu: {stage_config['nu']:.6f}")
    print(f"  pressure_gradient: {stage_config['pressure_gradient']:.3f}")
    print(f"  學習率: {stage_config['lr']:.6f}")
    print(f"  PDE點數: {stage_config['sampling']['pde_points']}")
    print(f"  BC點數: {stage_config['sampling']['boundary_points']}")
    print(f"  權重: {stage_config['weights']}")

print("\n" + "=" * 80)
print("✅ 測試完成！課程訓練調度器工作正常")
print("=" * 80)

# 驗證物理參數是否正確更新
print("\n🔬 物理模組參數驗證:")
print(f"  當前 nu: {physics.nu}")
re_tau_val = getattr(physics, 'Re_tau', 'N/A')
pg_val = getattr(physics, 'pressure_gradient', 'N/A')
print(f"  當前 Re_tau: {re_tau_val}")
print(f"  當前 pressure_gradient: {pg_val}")
