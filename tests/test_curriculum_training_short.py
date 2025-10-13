#!/usr/bin/env python3
"""簡短的課程訓練測試 - 每階段僅訓練10個epoch驗證流程"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import yaml
import torch

# 創建測試配置（基於課程訓練配置，但縮短epoch數）
test_config = {
    'experiment': {
        'name': 'curriculum_test_short',
        'seed': 42,
        'device': 'cpu',
        'precision': 'float32'
    },
    'model': {
        'type': 'fourier_mlp',
        'in_dim': 2,
        'out_dim': 3,
        'width': 64,  # 縮小網路
        'depth': 3,
        'activation': 'sine',
        'fourier_m': 16,
        'fourier_sigma': 1.0,
        'scaling': {
            'learnable': False,
            'input_norm': {'x': [0.0, 25.13], 'y': [-1.0, 1.0]},
            'output_norm': {'u': [0.0, 16.5], 'v': [-0.6, 0.6], 'p': [-85.0, 3.0]}
        }
    },
    'physics': {
        'nu': 1.0e-3,
        'rho': 1.0,
        'channel_flow': {
            'Re_tau': 1000.0,
            'pressure_gradient': -1.0
        },
        'domain': {
            'x_range': [0.0, 25.13],
            'y_range': [-1.0, 1.0]
        },
        'boundary_conditions': {
            'wall_velocity': [0.0, 0.0],
            'wall_location': [-1.0, 1.0],
            'periodic_x': True,
            'pressure_driven': True,
            'mean_pressure_gradient': -1.0
        }
    },
    'sensors': {
        'K': 50,
        'selection_method': 'wall_balanced',
        'sensor_file': 'sensors_test.npz'
    },
    'data': {
        'source': 'mock',
        'normalize': True,
        'noise_sigma': 0.02,
        'dropout_prob': 0.10
    },
    'data': {
        'source': 'mock',
        'normalize': True,
        'noise_sigma': 0.02,
        'dropout_prob': 0.10
    },
    'curriculum': {
        'enable': True,
        'stages': [
            {
                'name': 'Stage1_Test',
                'epoch_range': [0, 10],
                'Re_tau': 100.0,
                'nu': 1.0e-2,
                'pressure_gradient': -0.1,
                'weights': {
                    'data': 1.0,
                    'wall_constraint': 50.0,
                    'periodicity': 20.0,
                    'momentum_x': 1.0,
                    'momentum_y': 1.0,
                    'continuity': 2.0,
                    'prior': 0.0
                },
                'sampling': {'pde_points': 512, 'boundary_points': 200},
                'lr': 1.0e-3
            },
            {
                'name': 'Stage2_Test',
                'epoch_range': [10, 20],
                'Re_tau': 300.0,
                'nu': 3.33e-3,
                'pressure_gradient': -0.333,
                'weights': {
                    'data': 1.0,
                    'wall_constraint': 30.0,
                    'periodicity': 10.0,
                    'momentum_x': 2.0,
                    'momentum_y': 2.0,
                    'continuity': 5.0,
                    'prior': 0.0
                },
                'sampling': {'pde_points': 512, 'boundary_points': 200},
                'lr': 5.0e-4
            }
        ]
    },
    'losses': {
        'data_weight': 10.0,
        'prior_weight': 0.0,
        'wall_constraint_weight': 50.0,
        'periodicity_weight': 20.0,
        'momentum_x_weight': 1.0,
        'momentum_y_weight': 1.0,
        'continuity_weight': 2.0,
        'source_l1': 1.0e-6,
        'gradient_penalty': 1.0e-4,
        'adaptive_weighting': False,
        'causal_weighting': False,
        'weight_update_freq': 50,
        'grad_norm_alpha': 0.15
    },
    'training': {
        'optimizer': 'adam',
        'lr': 1.0e-3,
        'weight_decay': 0.0,
        'lr_scheduler': 'none',
        'max_epochs': 20,
        'batch_size': 256,
        'validation_freq': 100,
        'checkpoint_freq': 1000,
        'early_stopping': False,
        'patience': 100,
        'sampling': {
            'pde_points': 512,
            'boundary_points': 200,
            'wall_clustering': 0.1,
            'center_enhancement': 0.4,
            'high_velocity_sampling': 0.3
        }
    },
    'data': {
        'source': 'mock'
    },
    'logging': {
        'level': 'info',
        'log_freq': 5,
        'save_stage_checkpoints': True,
        'stage_checkpoint_dir': './checkpoints/curriculum_test'
    },
    'reproducibility': {
        'deterministic': True,
        'benchmark': False
    }
}

# 保存測試配置
test_config_path = 'configs/curriculum_test_short.yml'
with open(test_config_path, 'w') as f:
    yaml.dump(test_config, f, default_flow_style=False)

print("=" * 80)
print("🧪 課程訓練短期測試（每階段10 epochs）")
print("=" * 80)
print(f"配置文件已保存: {test_config_path}")
print("\n開始訓練...")

# 導入訓練腳本
from scripts.train import (
    load_config, setup_logging, set_random_seed, get_device,
    create_model, create_physics, create_loss_functions,
    train_model
)

# 執行訓練
config = load_config(test_config_path)
logger = setup_logging(config['logging']['level'])
set_random_seed(config['experiment']['seed'], config['reproducibility']['deterministic'])
device = get_device(config['experiment']['device'])

# 創建模型和物理模組
model = create_model(config, device)
physics = create_physics(config, device)
losses = create_loss_functions(config, device)

print("\n開始訓練...")
result = train_model(model, physics, losses, config, device)

print("\n" + "=" * 80)
print("✅ 測試完成！")
print(f"最終損失: {result['final_loss']:.6f}")
print(f"訓練時間: {result['training_time']:.2f}s")
print(f"完成輪數: {result['epochs_completed']}")
print("=" * 80)
