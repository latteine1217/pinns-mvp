"""
TASK-audit-005 Phase 3: 整合驗證測試
端到端驗證標準化流程在真實訓練場景中的穩定性與性能

測試項目：
- Test 8: 端到端訓練穩定性（50 epochs）
- Test 9: 檢查點可重現性（保存/載入驗證）
- Test 10: 性能基準測試（簡化版：2 種配置對比）

執行時間：約 30-40 分鐘
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import yaml
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.trainer import Trainer
from pinnx.train.config_loader import load_config
from pinnx.utils.normalization import UnifiedNormalizer
from pinnx.models.fourier_mlp import PINNNet
from pinnx.train.factory import create_physics  # ✅ 使用工廠函數創建 physics
from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss


# =============================================================================
# 測試配置與工具函數
# =============================================================================

def create_test_config(
    experiment_name: str,
    input_norm: str = 'standard',
    output_norm: str = 'manual',
    epochs: int = 10
) -> Dict[str, Any]:
    """創建測試配置"""
    config = {
        'experiment': {
            'name': experiment_name,
            'version': 'v1.0',
            'seed': 42,
            'device': 'cpu',  # CPU 測試（避免 GPU 依賴）
            'precision': 'float32'
        },
        'model': {
            'type': 'fourier_mlp',
            'in_dim': 3,
            'out_dim': 4,
            'width': 64,  # 小網路（快速）
            'depth': 3,
            'activation': 'sine',
            'scaling': {
                'learnable': False,
                'input_norm': input_norm,
                'input_norm_range': [-1.0, 1.0]
            },
            'fourier_features': {
                'type': 'basic',
                'fourier_m': 8,
                'fourier_sigma': 1.0
            }
        },
        'normalization': {
            'type': output_norm,
            'variable_order': ['u', 'v', 'w', 'p'],
            'params': {
                'u_mean': 1.0,
                'u_std': 0.5,
                'v_mean': 0.0,
                'v_std': 0.3,
                'w_mean': 0.0,
                'w_std': 0.3,
                'p_mean': 0.0,
                'p_std': 0.1
            }
        },
        'physics': {
            'type': 'vs_pinn_channel_flow',
            'nu': 5.0e-5,
            'rho': 1.0,
            'channel_flow': {
                'Re_tau': 1000.0,
                'u_tau': 0.04997,
                'pressure_gradient': 0.0025
            },
            'domain': {
                'x_range': [0.0, 25.13],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, 9.42]
            },
            'vs_pinn': {
                'use_gradient_checkpointing': False
            }
        },
        'training': {
            'optimizer': 'adam',
            'lr': 1.0e-3,
            'epochs': epochs,
            'batch_size': 256,
            'validation_freq': max(epochs // 4, 1),
            'checkpoint_freq': max(epochs // 2, 1),
            'log_interval': max(epochs // 10, 1),
            'gradient_clip': 1.0,
            'sampling': {
                'pde_points': 512,
                'boundary_points': 256
            }
        },
        'losses': {
            'data_weight': 5.0,
            'boundary_weight': 5.0,
            'momentum_x_weight': 1.0,
            'momentum_y_weight': 1.0,
            'momentum_z_weight': 1.0,
            'continuity_weight': 1.0,
            'adaptive_weighting': False
        },
        'output': {
            'checkpoint_dir': './results/audit_005_phase3/checkpoints',
            'results_dir': './results/audit_005_phase3/results',
            'visualization_dir': './results/audit_005_phase3/visualizations'
        },
        'logging': {
            'level': 'warning',  # 減少日誌輸出
            'log_freq': max(epochs // 5, 1),
            'save_predictions': False,
            'tensorboard': False,
            'wandb': False
        }
    }
    return config


def compare_metadata(meta1: Any, meta2: Any, path: str = "root") -> bool:
    """
    遞迴比較 Metadata（處理 Tensor、Dict、List）
    
    Args:
        meta1, meta2: 待比較的物件
        path: 當前比較路徑（用於錯誤訊息）
    
    Returns:
        bool: 是否完全一致
    
    Raises:
        AssertionError: 若不一致，包含詳細差異路徑
    """
    # 類型檢查
    if type(meta1) != type(meta2):
        raise AssertionError(
            f"❌ Metadata 類型不一致 @ {path}\n"
            f"   meta1: {type(meta1).__name__}\n"
            f"   meta2: {type(meta2).__name__}"
        )
    
    # Tensor 比較
    if isinstance(meta1, torch.Tensor):
        if not torch.equal(meta1, meta2):
            raise AssertionError(
                f"❌ Tensor 值不一致 @ {path}\n"
                f"   meta1: {meta1}\n"
                f"   meta2: {meta2}"
            )
        return True
    
    # Dict 比較
    if isinstance(meta1, dict):
        if set(meta1.keys()) != set(meta2.keys()):
            raise AssertionError(
                f"❌ Dict 鍵不一致 @ {path}\n"
                f"   meta1 keys: {set(meta1.keys())}\n"
                f"   meta2 keys: {set(meta2.keys())}"
            )
        for key in meta1.keys():
            compare_metadata(meta1[key], meta2[key], path=f"{path}.{key}")
        return True
    
    # List/Tuple 比較
    if isinstance(meta1, (list, tuple)):
        if len(meta1) != len(meta2):
            raise AssertionError(
                f"❌ 序列長度不一致 @ {path}\n"
                f"   meta1: {len(meta1)}\n"
                f"   meta2: {len(meta2)}"
            )
        for i, (v1, v2) in enumerate(zip(meta1, meta2)):
            compare_metadata(v1, v2, path=f"{path}[{i}]")
        return True
    
    # 基本類型比較
    if meta1 != meta2:
        raise AssertionError(
            f"❌ 值不一致 @ {path}\n"
            f"   meta1: {meta1}\n"
            f"   meta2: {meta2}"
        )
    return True


def create_mock_training_data(n_points: int = 100, n_sensors: int = 20, n_bc: int = 50) -> Dict[str, torch.Tensor]:
    """
    創建模擬訓練資料（符合 Trainer.step() 期望的格式）
    
    Returns:
        字典包含：
        - PDE 點：x_pde, y_pde, z_pde, t_pde
        - 邊界點：x_bc, y_bc, z_bc, t_bc, u_bc, v_bc, w_bc, p_bc
        - 感測點：x_sensors, y_sensors, z_sensors, t_sensors, u_sensors, v_sensors, w_sensors, p_sensors
    """
    torch.manual_seed(42)
    
    # === PDE 點（用於計算物理殘差）===
    x_pde = torch.rand(n_points, 1) * 25.13
    y_pde = torch.rand(n_points, 1) * 2.0 - 1.0
    z_pde = torch.rand(n_points, 1) * 9.42
    t_pde = torch.zeros(n_points, 1)  # 穩態問題
    
    # === 邊界點（上下壁面 y=±1）===
    x_bc = torch.rand(n_bc, 1) * 25.13
    y_bc = torch.where(torch.rand(n_bc, 1) > 0.5, 
                       torch.ones(n_bc, 1), 
                       -torch.ones(n_bc, 1))  # 隨機分配到上壁或下壁
    z_bc = torch.rand(n_bc, 1) * 9.42
    t_bc = torch.zeros(n_bc, 1)
    
    # 邊界條件（無滑移：u=v=w=0）
    u_bc = torch.zeros(n_bc, 1)
    v_bc = torch.zeros(n_bc, 1)
    w_bc = torch.zeros(n_bc, 1)
    p_bc = torch.zeros(n_bc, 1)  # 壓力可選
    
    # === 感測點（稀疏觀測資料）===
    x_sensors = torch.rand(n_sensors, 1) * 25.13
    y_sensors = torch.rand(n_sensors, 1) * 2.0 - 1.0
    z_sensors = torch.rand(n_sensors, 1) * 9.42
    t_sensors = torch.zeros(n_sensors, 1)
    
    # 模擬速度場（簡單 Poiseuille 流 + 噪聲）
    u_sensors = 1.0 * (1.0 - y_sensors**2) + 0.01 * torch.randn(n_sensors, 1)
    v_sensors = 0.01 * torch.randn(n_sensors, 1)
    w_sensors = 0.01 * torch.randn(n_sensors, 1)
    p_sensors = torch.zeros(n_sensors, 1)
    
    return {
        # PDE 點
        'x_pde': x_pde, 'y_pde': y_pde, 'z_pde': z_pde, 't_pde': t_pde,
        # 邊界點
        'x_bc': x_bc, 'y_bc': y_bc, 'z_bc': z_bc, 't_bc': t_bc,
        'u_bc': u_bc, 'v_bc': v_bc, 'w_bc': w_bc, 'p_bc': p_bc,
        # 感測點
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 'z_sensors': z_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'w_sensors': w_sensors, 'p_sensors': p_sensors
    }


def initialize_trainer(config: Dict[str, Any]) -> Tuple[Trainer, UnifiedNormalizer]:
    """
    初始化 Trainer 與 UnifiedNormalizer（遵循實際 API）
    
    Returns:
        (trainer, normalizer): 訓練器與標準化器
    """
    device = torch.device(config['experiment']['device'])
    
    # 1. 創建模型
    model = PINNNet(
        in_dim=config['model']['in_dim'],
        out_dim=config['model']['out_dim'],
        width=config['model']['width'],
        depth=config['model']['depth'],
        activation=config['model']['activation'],
        use_fourier=True,
        fourier_m=config['model']['fourier_features']['fourier_m'],
        fourier_sigma=config['model']['fourier_features']['fourier_sigma']
    ).to(device)
    
    # 2. 創建物理模組（使用工廠函數）
    physics = create_physics(config, device)
    
    # 3. 創建損失函數字典
    losses = {
        'ns_residual': NSResidualLoss(),
        'boundary': BoundaryConditionLoss()
    }
    
    # 4. 創建 UnifiedNormalizer（獨立管理）
    normalizer = UnifiedNormalizer.from_config(config, device=device)
    
    # 5. 創建 Trainer
    trainer = Trainer(
        model=model,
        physics=physics,
        losses=losses,
        config=config,
        device=device,
        weighters=None,  # 簡化測試不使用動態權重
        input_normalizer=None  # UnifiedNormalizer 獨立管理
    )
    
    return trainer, normalizer


# =============================================================================
# Test 8: 端到端訓練穩定性測試
# =============================================================================

def test_8_end_to_end_training_stability():
    """
    Test 8: 端到端訓練穩定性
    
    驗收標準：
    - ✅ 訓練完成無 NaN/Inf
    - ✅ 損失值持續下降（最終 < 初始 × 0.5）
    - ✅ Normalizer metadata 在訓練過程中保持一致
    """
    print("\n" + "="*80)
    print("Test 8: 端到端訓練穩定性測試")
    print("="*80)
    
    # 創建配置（50 epochs，平衡速度與收斂）
    config = create_test_config(
        experiment_name='test_8_end_to_end',
        input_norm='standard',
        output_norm='manual',
        epochs=50
    )
    
    # 初始化 Trainer 與 Normalizer
    print("\n[1/5] 初始化 Trainer 與 Normalizer...")
    trainer, normalizer = initialize_trainer(config)
    
    # 準備訓練資料
    print("[2/5] 準備訓練資料...")
    training_data = create_mock_training_data(n_points=500, n_sensors=50, n_bc=100)
    trainer.training_data = training_data
    
    # 記錄初始 metadata
    print("[3/5] 記錄初始 metadata...")
    metadata_epoch_0 = normalizer.get_metadata()
    
    # 執行訓練
    print("[4/5] 開始訓練（50 epochs）...")
    print("    預計時間：3-5 分鐘")
    start_time = time.time()
    
    train_result = trainer.train()
    
    elapsed_time = time.time() - start_time
    print(f"    訓練完成，耗時：{elapsed_time:.1f} 秒")
    
    # 驗證結果
    print("[5/5] 驗證訓練結果...")
    
    # 檢查 1: 損失值無 NaN/Inf
    final_loss = train_result['final_loss']
    assert np.isfinite(final_loss), f"❌ 最終損失為 NaN/Inf: {final_loss}"
    print(f"    ✅ 最終損失有效: {final_loss:.6f}")
    
    # 檢查 2: 損失下降
    history = train_result.get('history', {})
    if 'total_loss' in history and len(history['total_loss']) > 0:
        initial_loss = history['total_loss'][0]
        loss_ratio = final_loss / initial_loss
        assert loss_ratio < 0.5, f"❌ 損失下降不足: {loss_ratio:.2%} (期望 < 50%)"
        print(f"    ✅ 損失下降: {initial_loss:.6f} → {final_loss:.6f} ({loss_ratio:.2%})")
    
    # 檢查 3: Metadata 一致性
    metadata_epoch_50 = normalizer.get_metadata()
    
    # 比較 input_transform metadata
    input_meta_0 = metadata_epoch_0['input']  # ✅ 修正鍵名
    input_meta_50 = metadata_epoch_50['input']
    assert input_meta_0['norm_type'] == input_meta_50['norm_type'], "❌ Input norm_type 改變"
    # feature_range 是 tensor，需使用 torch.equal
    if 'feature_range' in input_meta_0 and 'feature_range' in input_meta_50:
        assert torch.equal(input_meta_0['feature_range'], input_meta_50['feature_range']), "❌ Feature range 改變"
    
    # 比較 output_transform metadata
    output_meta_0 = metadata_epoch_0['output']  # ✅ 修正鍵名
    output_meta_50 = metadata_epoch_50['output']
    assert output_meta_0['norm_type'] == output_meta_50['norm_type'], "❌ Output norm_type 改變"
    assert output_meta_0['variable_order'] == output_meta_50['variable_order'], "❌ Variable order 改變"
    
    print("    ✅ Metadata 一致性通過")
    
    print("\n" + "="*80)
    print("✅ Test 8 通過: 端到端訓練穩定")
    print("="*80)
    return True


# =============================================================================
# Test 9: 檢查點可重現性測試
# =============================================================================

def test_9_checkpoint_reproducibility():
    """
    Test 9: 檢查點可重現性
    
    驗收標準：
    - ✅ 保存檢查點成功
    - ✅ 載入檢查點後預測誤差 < 1e-5
    - ✅ Metadata 完全一致
    """
    print("\n" + "="*80)
    print("Test 9: 檢查點可重現性測試")
    print("="*80)
    
    # 創建配置（10 epochs，快速）
    config = create_test_config(
        experiment_name='test_9_checkpoint',
        input_norm='standard',
        output_norm='manual',
        epochs=10
    )
    
    # 初始化第一個 Trainer
    print("\n[1/6] 訓練第一個模型...")
    trainer_1, normalizer_1 = initialize_trainer(config)
    training_data = create_mock_training_data(n_points=100)
    trainer_1.training_data = training_data
    
    # 訓練
    start_time = time.time()
    train_result_1 = trainer_1.train()
    elapsed_time = time.time() - start_time
    print(f"    訓練完成，耗時：{elapsed_time:.1f} 秒")
    
    # 保存檢查點
    print("[2/6] 保存檢查點...")
    # ✅ save_checkpoint() 會自動使用 config['output']['checkpoint_dir']
    trainer_1.save_checkpoint(epoch=10, metrics={'test_loss': 0.1}, is_best=False)
    
    # 檢查點路徑為 checkpoint_dir/epoch_10.pth
    checkpoint_path = Path(config['output']['checkpoint_dir']) / 'epoch_10.pth'
    assert checkpoint_path.exists(), f"❌ 檢查點未保存: {checkpoint_path}"
    print(f"    ✅ 檢查點已保存: {checkpoint_path}")
    
    # 生成測試預測
    print("[3/6] 生成測試預測（模型 1）...")
    test_coords = torch.rand(50, 3) * torch.tensor([25.13, 2.0, 9.42]) + torch.tensor([0.0, -1.0, 0.0])
    
    trainer_1.model.eval()
    with torch.no_grad():
        pred_1 = trainer_1.model(test_coords)
    
    # 初始化第二個 Trainer 並載入檢查點
    print("[4/6] 初始化第二個模型並載入檢查點...")
    trainer_2, normalizer_2 = initialize_trainer(config)
    
    trainer_2.load_checkpoint(str(checkpoint_path))
    print("    ✅ 檢查點已載入")
    
    # 生成測試預測（模型 2）
    print("[5/6] 生成測試預測（模型 2）...")
    trainer_2.model.eval()
    with torch.no_grad():
        pred_2 = trainer_2.model(test_coords)
    
    # 驗證預測一致性
    print("[6/6] 驗證預測一致性...")
    pred_error = torch.abs(pred_1 - pred_2).max().item()
    assert pred_error < 1e-5, f"❌ 預測誤差過大: {pred_error:.2e} (期望 < 1e-5)"
    print(f"    ✅ 預測誤差: {pred_error:.2e} < 1e-5")
    
    # 驗證 Metadata 一致性
    metadata_1 = normalizer_1.get_metadata()
    metadata_2 = normalizer_2.get_metadata()
    
    try:
        compare_metadata(metadata_1, metadata_2)
        print("    ✅ Metadata 完全一致")
    except AssertionError as e:
        pytest.fail(str(e))
    
    print("\n" + "="*80)
    print("✅ Test 9 通過: 檢查點可重現性驗證")
    print("="*80)
    return True


# =============================================================================
# Test 10: 性能基準測試（簡化版）
# =============================================================================

def test_10_performance_benchmark():
    """
    Test 10: 性能基準測試（簡化版）
    
    驗收標準：
    - ✅ Baseline 配置訓練成功
    - ✅ Full 配置訓練成功
    - ✅ 生成性能對比報告
    """
    print("\n" + "="*80)
    print("Test 10: 性能基準測試（簡化版）")
    print("="*80)
    
    results = {}
    
    # 配置 1: Baseline（無標準化）
    print("\n[1/3] Baseline 配置（無標準化）...")
    config_baseline = create_test_config(
        experiment_name='test_10_baseline',
        input_norm='none',
        output_norm='none',
        epochs=20
    )
    
    trainer_baseline, _ = initialize_trainer(config_baseline)
    trainer_baseline.training_data = create_mock_training_data(n_points=200)
    
    start_time = time.time()
    result_baseline = trainer_baseline.train()
    elapsed_baseline = time.time() - start_time
    
    results['baseline'] = {
        'final_loss': result_baseline['final_loss'],
        'time': elapsed_baseline
    }
    print(f"    ✅ Baseline: Loss={result_baseline['final_loss']:.6f}, Time={elapsed_baseline:.1f}s")
    
    # 配置 2: Full（標準化）
    print("\n[2/3] Full 配置（標準化）...")
    config_full = create_test_config(
        experiment_name='test_10_full',
        input_norm='standard',
        output_norm='manual',
        epochs=20
    )
    
    trainer_full, _ = initialize_trainer(config_full)
    trainer_full.training_data = create_mock_training_data(n_points=200)
    
    start_time = time.time()
    result_full = trainer_full.train()
    elapsed_full = time.time() - start_time
    
    results['full'] = {
        'final_loss': result_full['final_loss'],
        'time': elapsed_full
    }
    print(f"    ✅ Full: Loss={result_full['final_loss']:.6f}, Time={elapsed_full:.1f}s")
    
    # 生成對比報告
    print("\n[3/3] 生成性能對比報告...")
    print("\n" + "-"*80)
    print("性能對比結果：")
    print("-"*80)
    print(f"{'配置':<15} {'最終損失':<15} {'訓練時間 (s)':<15}")
    print("-"*80)
    for name, res in results.items():
        print(f"{name:<15} {res['final_loss']:<15.6f} {res['time']:<15.1f}")
    print("-"*80)
    
    # 保存報告
    report_path = Path('./results/audit_005_phase3/benchmarks/performance_report.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w') as f:
        f.write("TASK-audit-005 Phase 3 性能基準測試報告\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'配置':<15} {'最終損失':<15} {'訓練時間 (s)':<15}\n")
        f.write("-"*80 + "\n")
        for name, res in results.items():
            f.write(f"{name:<15} {res['final_loss']:<15.6f} {res['time']:<15.1f}\n")
    
    print(f"\n    ✅ 報告已保存: {report_path}")
    
    print("\n" + "="*80)
    print("✅ Test 10 通過: 性能基準測試完成")
    print("="*80)
    return True


# =============================================================================
# 主函數
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TASK-audit-005 Phase 3: 整合驗證測試")
    print("="*80)
    print("預計總時間：30-40 分鐘")
    print("="*80)
    
    # 創建輸出目錄
    output_dirs = [
        './results/audit_005_phase3/checkpoints',
        './results/audit_005_phase3/logs',
        './results/audit_005_phase3/benchmarks'
    ]
    for dir_path in output_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # 執行測試
    try:
        test_8_end_to_end_training_stability()
        test_9_checkpoint_reproducibility()
        test_10_performance_benchmark()
        
        print("\n" + "="*80)
        print("🎉 Phase 3 所有測試通過！")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 執行錯誤: {e}")
        raise
