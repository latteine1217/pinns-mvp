#!/usr/bin/env python3
"""
快速驗證：標準化對訓練穩定性的影響

測試內容：
1. Baseline（無標準化）vs Normalized（Z-Score 標準化）
2. 訓練 200 epochs（快速驗證）
3. 損失收斂曲線對比
4. 檢查點保存/載入一致性
5. 早停機制驗證

預計時間：30-45 分鐘
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.config_loader import load_config
from pinnx.train.factory import create_model, create_physics, create_optimizer
from pinnx.train.trainer import Trainer


def create_test_config(enable_normalization: bool) -> Dict[str, Any]:
    """創建測試配置（基於 normalization_baseline_test_fix_v1.yml）"""
    
    config = {
        'experiment': {
            'name': f'quick_val_{"normalized" if enable_normalization else "baseline"}',
            'version': 'v1.0',
            'seed': 42,
            'device': 'auto',
            'precision': 'float32'
        },
        
        'data': {
            'source': 'synthetic',  # 使用合成資料加速測試
            'dataset': 'channel',
            'domain': {
                'x': [0, 25.13],
                'y': [-1.0, 1.0],
                'z': [0, 9.42]
            }
        },
        
        'normalization': {
            'type': 'training_data_norm' if enable_normalization else 'none',
            'noise_sigma': 0.01,
            'dropout_prob': 0.0  # 關閉 dropout 以減少隨機性
        },
        
        'model': {
            'type': 'fourier_mlp',
            'in_dim': 3,
            'out_dim': 4,
            'width': 128,
            'depth': 4,
            'activation': 'sine',
            'use_fourier': True,
            'fourier_m': 16,
            'fourier_sigma': 1.0
        },
        
        'training': {
            'epochs': 200,  # 快速測試
            'batch_size': 512,
            'optimizer': {
                'type': 'adam',
                'lr': 1.0e-3,
                'betas': [0.9, 0.999]
            },
            'scheduler': {
                'type': 'exponential',
                'decay_rate': 0.95,
                'decay_epochs': 50
            },
            'sampling': {
                'pde_points': 1024,
                'boundary_points': 128,
                'sensor_points': 50
            },
            'validation_freq': 10,
            'checkpoint_freq': 50,
            'log_interval': 5,
            'gradient_clip': 1.0,
            'early_stopping': {
                'enabled': True,
                'patience': 50,
                'min_delta': 1.0e-4
            }
        },
        
        'losses': {
            'data_loss_weight': 100.0,
            'pde_loss_weight': 1.0,
            'wall_loss_weight': 50.0,
            'initial_loss_weight': 0.0,
            'adaptive_weights': {
                'enabled': False
            }
        },
        
        'physics': {
            'type': 'vs_pinn_channel_flow',
            'nu': 5.0e-5,
            're_tau': 1000,
            'u_tau': 0.04997,
            'domain': {
                'x_min': 0.0,
                'x_max': 25.13,
                'y_min': -1.0,
                'y_max': 1.0,
                'z_min': 0.0,
                'z_max': 9.42
            },
            'scaling': {
                'use_scaling': True,
                'N_x': 2.0,
                'N_y': 8.0,
                'N_z': 2.0
            }
        },
        
        'output': {
            'checkpoint_dir': f'./checkpoints/quick_val_{"normalized" if enable_normalization else "baseline"}',
            'results_dir': f'./results/quick_val_{"normalized" if enable_normalization else "baseline"}',
            'visualization_dir': f'./results/quick_val_{"normalized" if enable_normalization else "baseline"}/visualizations'
        }
    }
    
    return config


def generate_synthetic_data(config: Dict[str, Any], n_sensors: int = 50, n_pde: int = 500, n_bc: int = 100) -> Dict[str, torch.Tensor]:
    """
    生成合成訓練資料（通道流解析解近似）
    
    返回格式符合 Trainer 期望：
    - 感測點資料：x_sensors, y_sensors, z_sensors, u_sensors, v_sensors, w_sensors, p_sensors
    - PDE 點：x_pde, y_pde, z_pde
    - 邊界點：x_bc, y_bc, z_bc
    """
    
    domain = config['physics']['domain']
    u_tau = config['physics']['u_tau']
    nu = config['physics']['nu']
    
    # 設定隨機種子
    np.random.seed(42)
    
    # ==================== 1. 生成感測點資料 ====================
    x_sensors = np.random.uniform(domain['x_min'], domain['x_max'], n_sensors)
    y_sensors = np.random.uniform(domain['y_min'], domain['y_max'], n_sensors)
    z_sensors = np.random.uniform(domain['z_min'], domain['z_max'], n_sensors)
    
    # 解析解近似（層流 Poiseuille 流）
    # u = u_max * (1 - (y/h)^2)（拋物線分佈）
    u_sensors = 1.0 * (1 - (y_sensors / 1.0)**2)
    v_sensors = 1e-3 * np.random.randn(n_sensors)  # 微小擾動
    w_sensors = 1e-3 * np.random.randn(n_sensors)
    p_sensors = -2.0 * nu * 1.0 * x_sensors / (1.0**2)  # 線性壓降
    
    # ==================== 2. 生成 PDE 點（無需真實值）====================
    x_pde = np.random.uniform(domain['x_min'], domain['x_max'], n_pde)
    y_pde = np.random.uniform(domain['y_min'], domain['y_max'], n_pde)
    z_pde = np.random.uniform(domain['z_min'], domain['z_max'], n_pde)
    
    # ==================== 3. 生成邊界點 ====================
    # 牆面邊界（y = ±1）
    n_wall = n_bc // 2
    x_bc_wall = np.random.uniform(domain['x_min'], domain['x_max'], n_wall)
    y_bc_wall = np.random.choice([-1.0, 1.0], n_wall)
    z_bc_wall = np.random.uniform(domain['z_min'], domain['z_max'], n_wall)
    
    # 週期邊界（x, z）
    n_periodic = n_bc - n_wall
    x_bc_periodic = np.random.choice([domain['x_min'], domain['x_max']], n_periodic)
    y_bc_periodic = np.random.uniform(domain['y_min'], domain['y_max'], n_periodic)
    z_bc_periodic = np.random.choice([domain['z_min'], domain['z_max']], n_periodic)
    
    x_bc = np.concatenate([x_bc_wall, x_bc_periodic])
    y_bc = np.concatenate([y_bc_wall, y_bc_periodic])
    z_bc = np.concatenate([z_bc_wall, z_bc_periodic])
    
    # ==================== 4. 轉換為 PyTorch 張量 ====================
    data = {
        # 感測點資料
        'x_sensors': torch.tensor(x_sensors, dtype=torch.float32).unsqueeze(1),
        'y_sensors': torch.tensor(y_sensors, dtype=torch.float32).unsqueeze(1),
        'z_sensors': torch.tensor(z_sensors, dtype=torch.float32).unsqueeze(1),
        'u_sensors': torch.tensor(u_sensors, dtype=torch.float32).unsqueeze(1),
        'v_sensors': torch.tensor(v_sensors, dtype=torch.float32).unsqueeze(1),
        'w_sensors': torch.tensor(w_sensors, dtype=torch.float32).unsqueeze(1),
        'p_sensors': torch.tensor(p_sensors, dtype=torch.float32).unsqueeze(1),
        
        # PDE 點
        'x_pde': torch.tensor(x_pde, dtype=torch.float32).unsqueeze(1),
        'y_pde': torch.tensor(y_pde, dtype=torch.float32).unsqueeze(1),
        'z_pde': torch.tensor(z_pde, dtype=torch.float32).unsqueeze(1),
        
        # 邊界點
        'x_bc': torch.tensor(x_bc, dtype=torch.float32).unsqueeze(1),
        'y_bc': torch.tensor(y_bc, dtype=torch.float32).unsqueeze(1),
        'z_bc': torch.tensor(z_bc, dtype=torch.float32).unsqueeze(1),
    }
    
    return data


def run_training(config: Dict[str, Any], training_data: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """執行訓練並返回結果"""
    
    device = torch.device('cuda' if torch.cuda.is_available() and config['experiment']['device'] == 'auto' else 'cpu')
    print(f"\n{'='*60}")
    print(f"訓練配置: {config['experiment']['name']}")
    print(f"標準化: {config['normalization']['type']}")
    print(f"設備: {device}")
    print(f"{'='*60}\n")
    
    # 設定隨機種子
    torch.manual_seed(config['experiment']['seed'])
    np.random.seed(config['experiment']['seed'])
    
    # 創建模型、物理
    model = create_model(config, device=device)
    physics = create_physics(config, device=device)
    
    # 創建損失函數字典（Trainer 需要實例化對象，而非配置）
    # 注意：資料損失（MSE）由 Trainer 內部計算，這裡只需要物理損失類別
    from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
    from pinnx.losses.priors import PriorLossManager
    
    loss_cfg = config.get('losses', {})
    losses_dict = {
        'residual': NSResidualLoss(
            nu=config['physics'].get('nu', 5.0e-5),
            density=1.0
        ),
        'boundary': BoundaryConditionLoss(),
        'prior': PriorLossManager(
            consistency_weight=loss_cfg.get('prior_weight', 0.0)  # 快速測試不使用先驗
        )
    }
    
    # 創建訓練器（內部會從 config 創建優化器）
    # 注意：不需要傳遞 optimizer 參數，Trainer 內部會自動創建
    trainer = Trainer(
        model=model,
        physics=physics,
        losses=losses_dict,
        config=config,
        device=device,
        training_data=training_data  # 傳遞訓練資料以計算標準化統計量
    )
    
    # ⭐ 手動設置訓練資料（Trainer 需要 self.training_data 用於 step() 方法）
    # 注意：初始化時的 training_data 僅用於計算標準化統計量，不會自動設為 self.training_data
    trainer.training_data = training_data
    
    # 開始訓練
    start_time = time.time()
    result = trainer.train()
    training_time = time.time() - start_time
    
    print(f"\n✅ 訓練完成！耗時 {training_time:.2f} 秒")
    
    # 返回結果
    return {
        'config': config,
        'history': result['history'],
        'best_epoch': result['best_epoch'],
        'best_loss': result['best_metric'],  # ⭐ 修正：train() 返回 'best_metric' 而非 'best_loss'
        'training_time': training_time,
        'model_checkpoint': result.get('checkpoint_path', None),
        'final_model_state': model.state_dict()  # ⭐ 添加最終模型狀態
    }


def test_checkpoint_consistency(config: Dict[str, Any], original_state: Dict[str, Any]) -> bool:
    """測試檢查點保存/載入一致性"""
    
    print(f"\n{'='*60}")
    print("測試檢查點一致性")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_dir = Path(config['output']['checkpoint_dir'])
    
    # 尋找最新檢查點
    checkpoints = list(checkpoint_dir.glob('epoch_*.pth'))
    if not checkpoints:
        print("❌ 未找到檢查點文件")
        return False
    
    latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
    print(f"載入檢查點: {latest_checkpoint}")
    
    # 載入檢查點
    checkpoint = torch.load(latest_checkpoint, map_location=device)
    
    # 檢查關鍵欄位
    required_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 'history', 'config']
    missing_keys = [k for k in required_keys if k not in checkpoint]
    
    if missing_keys:
        print(f"❌ 檢查點缺少欄位: {missing_keys}")
        return False
    
    # 從 history 中提取最終損失
    final_loss = checkpoint['history']['total_loss'][-1]
    
    print(f"✅ 檢查點完整性驗證通過")
    print(f"  - Epoch: {checkpoint['epoch']}")
    print(f"  - Final Total Loss: {final_loss:.6e}")
    
    # 比較模型參數（與訓練結束時的狀態）
    loaded_state = checkpoint['model_state_dict']
    param_diff = []
    
    for key in original_state.keys():
        if key in loaded_state:
            diff = torch.abs(original_state[key] - loaded_state[key]).max().item()
            param_diff.append(diff)
    
    max_diff = max(param_diff) if param_diff else float('inf')
    print(f"  - 模型參數最大差異: {max_diff:.6e}")
    
    if max_diff < 1e-5:
        print("✅ 模型參數一致性驗證通過")
        return True
    else:
        print(f"❌ 模型參數不一致（差異: {max_diff:.6e} > 1e-5）")
        return False


def plot_comparison(baseline_history: Dict[str, List], normalized_history: Dict[str, List], output_dir: Path):
    """繪製對比圖"""
    
    print(f"\n{'='*60}")
    print("生成對比圖表")
    print(f"{'='*60}\n")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Total Loss（對數尺度）
    ax = axes[0]
    ax.plot(baseline_history['total_loss'], label='Baseline', linewidth=2, alpha=0.8, color='tab:blue')
    ax.plot(normalized_history['total_loss'], label='Normalized', linewidth=2, alpha=0.8, color='tab:orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (log scale)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Total Loss Convergence', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 2. Total Loss（線性尺度 - 最後 50 epochs）
    ax = axes[1]
    start_epoch = max(0, len(baseline_history['total_loss']) - 50)
    ax.plot(
        range(start_epoch, len(baseline_history['total_loss'])),
        baseline_history['total_loss'][start_epoch:],
        label='Baseline',
        linewidth=2,
        alpha=0.8,
        color='tab:blue'
    )
    ax.plot(
        range(start_epoch, len(normalized_history['total_loss'])),
        normalized_history['total_loss'][start_epoch:],
        label='Normalized',
        linewidth=2,
        alpha=0.8,
        color='tab:orange'
    )
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Total Loss (linear scale)', fontsize=12)
    ax.set_title('Final Convergence (Last 50 Epochs)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = output_dir / 'training_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✅ 對比圖表已保存: {output_file}")
    
    plt.close()


def generate_report(baseline_result: Dict, normalized_result: Dict, 
                   baseline_checkpoint_ok: bool, normalized_checkpoint_ok: bool,
                   output_dir: Path):
    """生成驗證報告"""
    
    print(f"\n{'='*60}")
    print("生成驗證報告")
    print(f"{'='*60}\n")
    
    report = {
        'test_name': 'Quick Validation: Normalization Impact on Training Stability',
        'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'baseline': {
            'config_name': baseline_result['config']['experiment']['name'],
            'normalization_type': baseline_result['config']['normalization']['type'],
            'best_epoch': baseline_result['best_epoch'],
            'best_loss': float(baseline_result['best_loss']),
            'final_total_loss': float(baseline_result['history']['total_loss'][-1]),
            'training_time_sec': baseline_result['training_time'],
            'checkpoint_consistency': baseline_checkpoint_ok
        },
        'normalized': {
            'config_name': normalized_result['config']['experiment']['name'],
            'normalization_type': normalized_result['config']['normalization']['type'],
            'best_epoch': normalized_result['best_epoch'],
            'best_loss': float(normalized_result['best_loss']),
            'final_total_loss': float(normalized_result['history']['total_loss'][-1]),
            'training_time_sec': normalized_result['training_time'],
            'checkpoint_consistency': normalized_checkpoint_ok
        },
        'comparison': {
            'best_loss_ratio': float(normalized_result['best_loss'] / baseline_result['best_loss']),
            'final_total_loss_ratio': float(normalized_result['history']['total_loss'][-1] / baseline_result['history']['total_loss'][-1]),
            'convergence_speed_comparison': f"Normalized reached best at epoch {normalized_result['best_epoch']}, Baseline at epoch {baseline_result['best_epoch']}",
            'training_time_ratio': normalized_result['training_time'] / baseline_result['training_time']
        },
        'test_results': {
            'training_stability': {
                'baseline_has_nan': any(np.isnan(baseline_result['history']['total_loss'])),
                'normalized_has_nan': any(np.isnan(normalized_result['history']['total_loss'])),
                'status': '✅ PASS' if not any(np.isnan(normalized_result['history']['total_loss'])) else '❌ FAIL'
            },
            'checkpoint_consistency': {
                'baseline': '✅ PASS' if baseline_checkpoint_ok else '❌ FAIL',
                'normalized': '✅ PASS' if normalized_checkpoint_ok else '❌ FAIL',
                'status': '✅ PASS' if (baseline_checkpoint_ok and normalized_checkpoint_ok) else '❌ FAIL'
            },
            'performance_impact': {
                'normalized_improves_loss': normalized_result['best_loss'] < baseline_result['best_loss'],
                'normalized_faster_convergence': normalized_result['best_epoch'] < baseline_result['best_epoch'],
                'status': '✅ PASS' if normalized_result['best_loss'] < baseline_result['best_loss'] * 1.5 else '⚠️ WARNING'
            }
        }
    }
    
    # 保存 JSON 報告
    output_file = output_dir / 'quick_validation_report.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"✅ JSON 報告已保存: {output_file}")
    
    # 打印摘要
    print(f"\n{'='*60}")
    print("驗證結果摘要")
    print(f"{'='*60}\n")
    
    print("📊 訓練穩定性:")
    print(f"  Baseline - NaN 損失: {'是' if report['test_results']['training_stability']['baseline_has_nan'] else '否'}")
    print(f"  Normalized - NaN 損失: {'是' if report['test_results']['training_stability']['normalized_has_nan'] else '否'}")
    print(f"  狀態: {report['test_results']['training_stability']['status']}")
    
    print("\n📦 檢查點一致性:")
    print(f"  Baseline: {report['test_results']['checkpoint_consistency']['baseline']}")
    print(f"  Normalized: {report['test_results']['checkpoint_consistency']['normalized']}")
    print(f"  狀態: {report['test_results']['checkpoint_consistency']['status']}")
    
    print("\n🎯 性能影響:")
    print(f"  最佳損失 (Normalized/Baseline): {report['comparison']['best_loss_ratio']:.3f}×")
    print(f"  最終總損失 (Normalized/Baseline): {report['comparison']['final_total_loss_ratio']:.3f}×")
    print(f"  訓練時間 (Normalized/Baseline): {report['comparison']['training_time_ratio']:.3f}×")
    print(f"  狀態: {report['test_results']['performance_impact']['status']}")
    
    print(f"\n{'='*60}\n")
    
    return report


def main():
    """主程式"""
    
    print("\n" + "="*60)
    print("🚀 快速驗證：標準化對訓練穩定性的影響")
    print("="*60 + "\n")
    
    # 創建輸出目錄
    output_dir = Path('./results/quick_validation_normalization')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 生成合成訓練資料
    print("📦 生成合成訓練資料...")
    baseline_config = create_test_config(enable_normalization=False)
    training_data = generate_synthetic_data(
        baseline_config, 
        n_sensors=50,    # 感測點數量
        n_pde=500,       # PDE 點數量
        n_bc=100         # 邊界點數量
    )
    print(f"  - 感測點數量: {len(training_data['x_sensors'])}")
    print(f"  - PDE 點數量: {len(training_data['x_pde'])}")
    print(f"  - 邊界點數量: {len(training_data['x_bc'])}")
    print(f"  - 座標範圍: x ∈ [{training_data['x_sensors'].min():.2f}, {training_data['x_sensors'].max():.2f}]")
    print(f"  - 速度範圍: u ∈ [{training_data['u_sensors'].min():.3f}, {training_data['u_sensors'].max():.3f}]")
    
    # 2. 訓練 Baseline（無標準化）
    print("\n" + "="*60)
    print("🔧 測試 1/2: Baseline（無標準化）")
    print("="*60)
    baseline_result = run_training(baseline_config, training_data)
    
    # 3. 訓練 Normalized（Z-Score 標準化）
    print("\n" + "="*60)
    print("🔧 測試 2/2: Normalized（Z-Score 標準化）")
    print("="*60)
    normalized_config = create_test_config(enable_normalization=True)
    normalized_result = run_training(normalized_config, training_data)
    
    # 4. 測試檢查點一致性
    baseline_checkpoint_ok = test_checkpoint_consistency(
        baseline_config, 
        baseline_result['final_model_state']
    )
    normalized_checkpoint_ok = test_checkpoint_consistency(
        normalized_config, 
        normalized_result['final_model_state']
    )
    
    # 5. 生成對比圖
    plot_comparison(
        baseline_result['history'],
        normalized_result['history'],
        output_dir
    )
    
    # 6. 生成驗證報告
    report = generate_report(
        baseline_result,
        normalized_result,
        baseline_checkpoint_ok,
        normalized_checkpoint_ok,
        output_dir
    )
    
    # 7. 最終總結
    print(f"\n{'='*60}")
    print("✅ 快速驗證完成！")
    print(f"{'='*60}\n")
    print(f"📁 輸出目錄: {output_dir}")
    print(f"📊 報告文件: {output_dir}/quick_validation_report.json")
    print(f"📈 對比圖表: {output_dir}/training_comparison.png")
    print(f"\n{'='*60}\n")
    
    # 返回狀態碼
    all_pass = all([
        report['test_results']['training_stability']['status'] == '✅ PASS',
        report['test_results']['checkpoint_consistency']['status'] == '✅ PASS'
    ])
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    exit(main())
