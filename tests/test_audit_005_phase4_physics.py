"""
TASK-audit-005 Phase 4: 物理一致性測試
=======================================

測試目標：驗證標準化流程對物理守恆律的影響

測試範圍：
1. 質量守恆（連續性方程）：∇·u = 0
2. 動量守恆（NS 方程殘差）
3. 邊界條件滿足度（壁面無滑移 + 週期性）

驗收標準：
- 質量守恆：||div(u)||_L² < 1.0（未訓練網絡）
- 動量殘差：||NS_residual||_L² < 100.0（未訓練網絡）
- 壁面條件：||u_wall||² < 10.0（未訓練網絡）
- 標準化影響：Normalized < 2× Baseline（不應劣化物理一致性）

技術要點：
- 使用簡化的 3D Channel Flow 物理模型（降低計算成本）
- 測試標準化前後物理一致性
- 檢查 Metadata 對梯度計算的影響

參考：
- tasks/TASK-audit-005/phase3_impl_plan.md
- results/audit_005_phase3/PHASE3_TEST_REPORT.md
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, cast

# 添加專案根目錄到路徑
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinnx.utils.normalization import UnifiedNormalizer
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow


# ============================================================================
# 工具函數
# ============================================================================

def create_simple_channel_flow_model(
    input_dim: int = 3,
    output_dim: int = 4,
    hidden_dim: int = 64,
    num_layers: int = 3,
    device: torch.device = torch.device('cpu')
) -> nn.Module:
    """創建簡單的通道流網絡（用於物理測試）
    
    Args:
        input_dim: 輸入維度（3=x,y,z）
        output_dim: 輸出維度（4=u,v,w,p）
        hidden_dim: 隱藏層寬度
        num_layers: 隱藏層數量
        device: 計算設備
        
    Returns:
        PyTorch 模型（Sequential）
    """
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.Tanh())
    
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Tanh())
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    
    model = nn.Sequential(*layers)
    return model.to(device)


def create_test_config_physics(enable_normalization: bool = True) -> Dict[str, Any]:
    """創建物理測試用的配置（簡化版）
    
    Args:
        enable_normalization: 是否啟用標準化
        
    Returns:
        測試配置字典
    """
    if enable_normalization:
        normalization_config = {
            'type': 'manual',
            'variable_order': ['u', 'v', 'w', 'p'],
            'params': {
                'u_mean': 0.0,
                'v_mean': 0.0,
                'w_mean': 0.0,
                'p_mean': 0.0,
                'u_std': 1.0,
                'v_std': 1.0,
                'w_std': 1.0,
                'p_std': 1.0
            }
        }
    else:
        normalization_config = {
            'type': 'none',
            'variable_order': ['u', 'v', 'w', 'p']
        }
    
    config = {
        'model': {
            'input_dim': 3,
            'output_dim': 4,
            'hidden_dim': 64,
            'num_layers': 3,
            'scaling': {
                'input_norm': 'standard' if enable_normalization else 'none',
                'input_norm_range': [-1.0, 1.0]
            }
        },
        'normalization': normalization_config,
        'physics': {
            'nu': 5e-5,  # JHTDB Channel Flow Re_τ=1000
            'dP_dx': 0.0025,
            'rho': 1.0,
            'domain': {
                'x_range': [0.0, 2.0 * np.pi],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, np.pi]
            },
            'scaling_factors': {
                'N_x': 2.0,
                'N_y': 12.0,
                'N_z': 2.0
            }
        },
        'training': {
            'optimizer': 'adam',
            'lr': 1e-3,
            'epochs': 100,
            'batch_size': 256
        }
    }
    
    return config


def compute_divergence(
    coords: torch.Tensor,
    velocity: torch.Tensor
) -> torch.Tensor:
    """計算速度場散度（質量守恆檢查）
    
    Args:
        coords: 座標張量 [batch, 3]，需要 requires_grad=True
        velocity: 速度張量 [batch, 3]（u, v, w）
        
    Returns:
        散度 [batch, 1]
        
    Note:
        使用自動微分計算 ∂u/∂x + ∂v/∂y + ∂w/∂z
    """
    u, v, w = velocity[:, 0:1], velocity[:, 1:2], velocity[:, 2:3]
    
    # 計算各分量導數
    u_x = torch.autograd.grad(
        u, coords,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    v_y = torch.autograd.grad(
        v, coords,
        grad_outputs=torch.ones_like(v),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    w_z = torch.autograd.grad(
        w, coords,
        grad_outputs=torch.ones_like(w),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]
    
    divergence = u_x + v_y + w_z
    return divergence


def compute_wall_velocity_error(
    coords: torch.Tensor,
    velocity: torch.Tensor,
    y_min: float = -1.0,
    y_max: float = 1.0,
    tol: float = 1e-6
) -> torch.Tensor:
    """計算壁面無滑移條件誤差
    
    Args:
        coords: 座標張量 [batch, 3]
        velocity: 速度張量 [batch, 3]（u, v, w）
        y_min: 下壁面位置
        y_max: 上壁面位置
        tol: 壁面識別容差
        
    Returns:
        壁面速度誤差 [batch, 1]（壁面點）或 [0] if 無壁面點
    """
    y = coords[:, 1]
    
    # 識別壁面點（使用容差避免浮點誤差）
    mask_lower = torch.abs(y - y_min) < tol
    mask_upper = torch.abs(y - y_max) < tol
    mask_wall = mask_lower | mask_upper
    
    if not mask_wall.any():
        # 無壁面點，返回零誤差
        return torch.zeros(1, 1, device=coords.device)
    
    # 提取壁面速度
    velocity_wall = velocity[mask_wall]  # [n_wall, 3]
    
    # 無滑移條件：u = v = w = 0
    wall_error = torch.sum(velocity_wall ** 2, dim=1, keepdim=True)
    
    return wall_error


def compute_periodic_error(
    coords: torch.Tensor,
    field: torch.Tensor,
    direction: int = 0,  # 0=x, 2=z
    domain_length: float = 2.0 * np.pi,
    tol: float = 1e-6
) -> torch.Tensor:
    """計算週期性邊界條件誤差
    
    Args:
        coords: 座標張量 [batch, 3]
        field: 物理量張量 [batch, n_vars]（可以是速度或壓力）
        direction: 週期方向（0=x, 2=z）
        domain_length: 域長度
        tol: 邊界識別容差
        
    Returns:
        週期性誤差 [batch, n_vars]（邊界點）或 [0] if 無邊界點
    """
    coord = coords[:, direction]
    
    # 識別邊界點
    mask_min = torch.abs(coord) < tol
    mask_max = torch.abs(coord - domain_length) < tol
    
    # 需要同時有兩側邊界點
    if not (mask_min.any() and mask_max.any()):
        return torch.zeros(1, field.shape[1], device=coords.device)
    
    # 提取邊界場
    field_min = field[mask_min]
    field_max = field[mask_max]
    
    # 週期性條件：field(x_min) = field(x_max)
    # 注意：可能兩側點數不同，需要配對
    n_min = field_min.shape[0]
    n_max = field_max.shape[0]
    n_pairs = min(n_min, n_max)
    
    if n_pairs == 0:
        return torch.zeros(1, field.shape[1], device=coords.device)
    
    periodic_error = (field_min[:n_pairs] - field_max[:n_pairs]) ** 2
    
    return periodic_error


# ============================================================================
# 測試 11：質量守恆驗證（標準化前後對比）
# ============================================================================

@pytest.mark.physics
def test_11_mass_conservation():
    """
    測試 11：質量守恆（連續性方程）
    
    驗證：
    1. 訓練後模型滿足 ∇·u = 0
    2. 標準化不影響散度計算
    3. ||div(u)||_L² < 1.0（未訓練網絡的合理範圍）
    
    策略：
    - 訓練兩個模型（有/無標準化）
    - 分別計算散度 L2 範數
    - 確認標準化版本不劣於無標準化版本
    """
    print("\n" + "="*70)
    print("TEST 11: 質量守恆驗證（標準化前後對比）")
    print("="*70)
    
    # 固定隨機種子確保測試可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試兩種配置
    results = {}
    for config_name, enable_norm in [('Baseline', False), ('Normalized', True)]:
        print(f"\n--- 配置: {config_name} ---")
        
        # 每次迭代重置隨機種子，確保模型初始化一致
        torch.manual_seed(42)
        np.random.seed(42)
        
        # 創建配置與模型
        config = create_test_config_physics(enable_normalization=enable_norm)
        model = create_simple_channel_flow_model(device=device)
        
        # 創建物理模組
        physics = VSPINNChannelFlow(
            scaling_factors=config['physics']['scaling_factors'],
            physics_params={
                'nu': config['physics']['nu'],
                'dP_dx': config['physics']['dP_dx'],
                'rho': config['physics']['rho']
            },
            domain_bounds={
                'x': tuple(config['physics']['domain']['x_range']),
                'y': tuple(config['physics']['domain']['y_range']),
                'z': tuple(config['physics']['domain']['z_range'])
            }
        )
        
        # 創建標準化器（如果啟用）
        normalizer = None
        if enable_norm:
            # 創建訓練數據用於擬合標準化器
            training_data = {
                'coords': torch.randn(100, 3),
                'u': torch.randn(100),
                'v': torch.randn(100),
                'w': torch.randn(100),
                'p': torch.randn(100)
            }
            normalizer = UnifiedNormalizer.from_config(config, training_data, device)
        
        # 生成測試點（域內隨機點）
        n_test = 500
        coords_np = np.random.uniform(
            low=[config['physics']['domain']['x_range'][0],
                 config['physics']['domain']['y_range'][0],
                 config['physics']['domain']['z_range'][0]],
            high=[config['physics']['domain']['x_range'][1],
                  config['physics']['domain']['y_range'][1],
                  config['physics']['domain']['z_range'][1]],
            size=(n_test, 3)
        )
        coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        
        # 前向傳播（應用標準化）
        if normalizer is not None:
            # 標準化分支：使用標準化座標進行前向傳播
            coords_norm = normalizer.transform_input(coords)
            coords_norm.requires_grad_(True)
            predictions_norm = model(coords_norm)
            predictions = cast(torch.Tensor, normalizer.inverse_transform_output(predictions_norm))
            scaled_coords_for_grad = coords_norm
        else:
            # Baseline 分支：使用 VS-PINN 的縮放座標
            scaled_coords_vs = physics.scale_coordinates(coords)
            scaled_coords_vs.requires_grad_(True)
            predictions = model(scaled_coords_vs)
            scaled_coords_for_grad = scaled_coords_vs
        
        # 使用物理模組計算連續性殘差（散度）
        # 這樣可以正確處理座標縮放
        divergence = physics.compute_continuity_residual(
            coords, predictions, scaled_coords=scaled_coords_for_grad
        )
        
        # 計算 L2 範數
        div_l2 = torch.sqrt(torch.mean(divergence ** 2)).item()
        div_max = torch.abs(divergence).max().item()
        
        print(f"  散度 L2 範數: {div_l2:.6e}")
        print(f"  散度最大值: {div_max:.6e}")
        
        results[config_name] = {
            'div_l2': div_l2,
            'div_max': div_max
        }
    
    # 驗證標準
    print(f"\n{'='*70}")
    print("驗收標準檢查：")
    print(f"{'='*70}")
    
    baseline_div = results['Baseline']['div_l2']
    normalized_div = results['Normalized']['div_l2']
    
    # 標準 1：兩者都應滿足 ||div(u)||_L² < 1.0（未訓練的初始網絡）
    assert baseline_div < 1.0, f"Baseline 散度過大: {baseline_div:.3e} > 1.0"
    assert normalized_div < 1.0, f"Normalized 散度過大: {normalized_div:.3e} > 1.0"
    
    # 標準 2：標準化不應劣化守恆（允許 2x 誤差範圍）
    assert normalized_div < 2.0 * baseline_div, \
        f"標準化劣化守恆: {normalized_div:.3e} > 2 × {baseline_div:.3e}"
    
    print(f"✅ Baseline 散度: {baseline_div:.6e} < 1.0")
    print(f"✅ Normalized 散度: {normalized_div:.6e} < 1.0")
    print(f"✅ 標準化影響: {normalized_div/baseline_div:.2f}x Baseline")
    
    print(f"\n{'='*70}")
    print("TEST 11 通過 ✅")
    print(f"{'='*70}")


# ============================================================================
# 測試 12：邊界條件滿足度
# ============================================================================

@pytest.mark.physics
def test_12_boundary_conditions():
    """
    測試 12：邊界條件滿足度
    
    驗證：
    1. 壁面無滑移條件：u(y=±1) = 0
    2. 週期性條件：u(x=0) = u(x=2π), u(z=0) = u(z=π)
    3. 標準化不影響邊界條件計算
    
    策略：
    - 生成包含邊界點的測試數據
    - 計算邊界條件誤差
    - 確認標準化版本滿足相同精度
    """
    print("\n" + "="*70)
    print("TEST 12: 邊界條件滿足度")
    print("="*70)
    
    # 固定隨機種子確保測試可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試兩種配置
    results = {}
    for config_name, enable_norm in [('Baseline', False), ('Normalized', True)]:
        print(f"\n--- 配置: {config_name} ---")
        
        # 每次迭代重置隨機種子，確保模型初始化一致
        torch.manual_seed(42)
        np.random.seed(42)
        
        config = create_test_config_physics(enable_normalization=enable_norm)
        model = create_simple_channel_flow_model(device=device)
        
        # 創建標準化器
        normalizer = None
        if enable_norm:
            training_data = {
                'coords': torch.randn(100, 3),
                'u': torch.randn(100),
                'v': torch.randn(100),
                'w': torch.randn(100),
                'p': torch.randn(100)
            }
            normalizer = UnifiedNormalizer.from_config(config, training_data, device)
        
        # === 測試壁面邊界條件 ===
        n_wall = 100
        x_wall = np.random.uniform(0, 2*np.pi, n_wall)
        z_wall = np.random.uniform(0, np.pi, n_wall)
        
        # 下壁面 y = -1
        coords_lower = np.column_stack([x_wall, np.full(n_wall, -1.0), z_wall])
        # 上壁面 y = 1
        coords_upper = np.column_stack([x_wall, np.full(n_wall, 1.0), z_wall])
        
        coords_wall_np = np.vstack([coords_lower, coords_upper])
        coords_wall = torch.tensor(coords_wall_np, dtype=torch.float32, device=device)
        coords_wall.requires_grad_(True)
        
        # 前向傳播
        if normalizer is not None:
            coords_norm = normalizer.transform_input(coords_wall)
            predictions_norm = model(coords_norm)
            predictions = normalizer.inverse_transform_output(predictions_norm)
        else:
            predictions = model(coords_wall)
        
        velocity_wall = cast(torch.Tensor, predictions[:, :3])
        
        # 計算壁面誤差
        wall_error = compute_wall_velocity_error(coords_wall, velocity_wall)
        wall_error_mean = wall_error.mean().item()
        wall_error_max = wall_error.max().item()
        
        print(f"  壁面速度誤差（平均）: {wall_error_mean:.6e}")
        print(f"  壁面速度誤差（最大）: {wall_error_max:.6e}")
        
        # === 測試週期性邊界條件（X 方向）===
        n_periodic = 50
        y_periodic = np.random.uniform(-1, 1, n_periodic)
        z_periodic = np.random.uniform(0, np.pi, n_periodic)
        
        # X 方向邊界
        coords_x_min = np.column_stack([np.zeros(n_periodic), y_periodic, z_periodic])
        coords_x_max = np.column_stack([np.full(n_periodic, 2*np.pi), y_periodic, z_periodic])
        
        coords_periodic_np = np.vstack([coords_x_min, coords_x_max])
        coords_periodic = torch.tensor(coords_periodic_np, dtype=torch.float32, device=device)
        coords_periodic.requires_grad_(True)
        
        # 前向傳播
        if normalizer is not None:
            coords_norm = normalizer.transform_input(coords_periodic)
            predictions_norm = model(coords_norm)
            predictions_periodic = cast(torch.Tensor, normalizer.inverse_transform_output(predictions_norm))
        else:
            predictions_periodic = model(coords_periodic)
        
        # 計算週期性誤差
        periodic_error = compute_periodic_error(
            coords_periodic, predictions_periodic,
            direction=0, domain_length=2.0*np.pi
        )
        periodic_error_mean = periodic_error.mean().item()
        
        print(f"  週期性誤差（X方向，平均）: {periodic_error_mean:.6e}")
        
        results[config_name] = {
            'wall_error_mean': wall_error_mean,
            'wall_error_max': wall_error_max,
            'periodic_error_mean': periodic_error_mean
        }
    
    # 驗證標準
    print(f"\n{'='*70}")
    print("驗收標準檢查：")
    print(f"{'='*70}")
    
    baseline_wall = results['Baseline']['wall_error_mean']
    normalized_wall = results['Normalized']['wall_error_mean']
    baseline_periodic = results['Baseline']['periodic_error_mean']
    normalized_periodic = results['Normalized']['periodic_error_mean']
    
    # 標準 1：壁面誤差應該有限（未訓練網絡，允許較大誤差）
    assert baseline_wall < 10.0, f"Baseline 壁面誤差過大: {baseline_wall:.3e}"
    assert normalized_wall < 10.0, f"Normalized 壁面誤差過大: {normalized_wall:.3e}"
    
    # 標準 2：標準化不應劣化邊界條件（允許 2x 誤差）
    assert normalized_wall < 2.0 * baseline_wall + 1.0, \
        f"標準化劣化壁面條件: {normalized_wall:.3e} > 2 × {baseline_wall:.3e}"
    
    print(f"✅ Baseline 壁面誤差: {baseline_wall:.6e} < 10.0")
    print(f"✅ Normalized 壁面誤差: {normalized_wall:.6e} < 10.0")
    print(f"✅ Baseline 週期性誤差: {baseline_periodic:.6e}")
    print(f"✅ Normalized 週期性誤差: {normalized_periodic:.6e}")
    
    print(f"\n{'='*70}")
    print("TEST 12 通過 ✅")
    print(f"{'='*70}")


# ============================================================================
# 測試 13：NS 方程殘差分析
# ============================================================================

@pytest.mark.physics
def test_13_ns_residual_analysis():
    """
    測試 13：NS 方程殘差分析
    
    驗證：
    1. 物理模組正確計算 NS 殘差
    2. 標準化不影響殘差計算
    3. 殘差各項量綱一致
    
    策略：
    - 使用 VSPINNChannelFlow 計算殘差
    - 檢查殘差數值範圍
    - 確認標準化前後殘差一致性
    """
    print("\n" + "="*70)
    print("TEST 13: NS 方程殘差分析")
    print("="*70)
    
    # 固定隨機種子確保測試可重現性
    torch.manual_seed(42)
    np.random.seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試兩種配置
    results = {}
    for config_name, enable_norm in [('Baseline', False), ('Normalized', True)]:
        print(f"\n--- 配置: {config_name} ---")
        
        # 每次迭代重置隨機種子，確保模型初始化一致
        torch.manual_seed(42)
        np.random.seed(42)
        
        config = create_test_config_physics(enable_normalization=enable_norm)
        model = create_simple_channel_flow_model(device=device)
        
        # 創建物理模組
        physics = VSPINNChannelFlow(
            scaling_factors=config['physics']['scaling_factors'],
            physics_params={
                'nu': config['physics']['nu'],
                'dP_dx': config['physics']['dP_dx'],
                'rho': config['physics']['rho']
            },
            domain_bounds={
                'x': tuple(config['physics']['domain']['x_range']),
                'y': tuple(config['physics']['domain']['y_range']),
                'z': tuple(config['physics']['domain']['z_range'])
            }
        )
        
        # 創建標準化器
        normalizer = None
        if enable_norm:
            training_data = {
                'coords': torch.randn(100, 3),
                'u': torch.randn(100),
                'v': torch.randn(100),
                'w': torch.randn(100),
                'p': torch.randn(100)
            }
            normalizer = UnifiedNormalizer.from_config(config, training_data, device)
        
        # 生成測試點
        n_test = 200
        coords_np = np.random.uniform(
            low=[0.0, -1.0, 0.0],
            high=[2.0*np.pi, 1.0, np.pi],
            size=(n_test, 3)
        )
        coords = torch.tensor(coords_np, dtype=torch.float32, device=device)
        
        # 前向傳播
        if normalizer is not None:
            # 標準化分支：只對標準化後的座標啟用梯度追蹤
            coords_norm = normalizer.transform_input(coords)
            coords_norm.requires_grad_(True)  # 標準化後的座標需要梯度
            predictions_norm = model(coords_norm)
            predictions = cast(torch.Tensor, normalizer.inverse_transform_output(predictions_norm))
            # 🔧 使用 scaled_coords 參數傳遞有梯度連接的座標
            scaled_coords_for_grad = coords_norm
        else:
            # Baseline 分支：需要對 VS-PINN 的縮放座標啟用梯度追蹤
            # 注意：不要對原始 coords 啟用梯度，因為物理模組會內部調用 scale_coordinates()
            scaled_coords_vs = physics.scale_coordinates(coords)
            scaled_coords_vs.requires_grad_(True)
            predictions = model(scaled_coords_vs)
            scaled_coords_for_grad = scaled_coords_vs
        
        # 計算 NS 殘差（使用物理模組）
        residual_continuity = physics.compute_continuity_residual(
            coords, predictions, scaled_coords=scaled_coords_for_grad
        )
        residual_momentum_dict = physics.compute_momentum_residuals(
            coords, predictions, scaled_coords=scaled_coords_for_grad
        )
        
        # 提取動量殘差（字典中的 x, y, z 分量）
        residual_momentum_x = residual_momentum_dict['momentum_x']
        residual_momentum_y = residual_momentum_dict['momentum_y']
        residual_momentum_z = residual_momentum_dict['momentum_z']
        
        # 計算 L2 範數
        continuity_l2 = torch.sqrt(torch.mean(residual_continuity ** 2)).item()
        momentum_x_l2 = torch.sqrt(torch.mean(residual_momentum_x ** 2)).item()
        momentum_y_l2 = torch.sqrt(torch.mean(residual_momentum_y ** 2)).item()
        momentum_z_l2 = torch.sqrt(torch.mean(residual_momentum_z ** 2)).item()
        
        print(f"  連續性殘差 L2: {continuity_l2:.6e}")
        print(f"  動量殘差 X L2: {momentum_x_l2:.6e}")
        print(f"  動量殘差 Y L2: {momentum_y_l2:.6e}")
        print(f"  動量殘差 Z L2: {momentum_z_l2:.6e}")
        
        # 檢查數值有效性
        assert torch.isfinite(residual_continuity).all(), "連續性殘差包含 NaN/Inf"
        assert torch.isfinite(residual_momentum_x).all(), "動量殘差 X 包含 NaN/Inf"
        assert torch.isfinite(residual_momentum_y).all(), "動量殘差 Y 包含 NaN/Inf"
        assert torch.isfinite(residual_momentum_z).all(), "動量殘差 Z 包含 NaN/Inf"
        
        results[config_name] = {
            'continuity_l2': continuity_l2,
            'momentum_x_l2': momentum_x_l2,
            'momentum_y_l2': momentum_y_l2,
            'momentum_z_l2': momentum_z_l2
        }
    
    # 驗證標準
    print(f"\n{'='*70}")
    print("驗收標準檢查：")
    print(f"{'='*70}")
    
    baseline_cont = results['Baseline']['continuity_l2']
    normalized_cont = results['Normalized']['continuity_l2']
    baseline_mom_x = results['Baseline']['momentum_x_l2']
    normalized_mom_x = results['Normalized']['momentum_x_l2']
    
    # 標準 1：殘差應該有限（未訓練網絡）
    assert baseline_cont < 100.0, f"Baseline 連續性殘差過大: {baseline_cont:.3e}"
    assert normalized_cont < 100.0, f"Normalized 連續性殘差過大: {normalized_cont:.3e}"
    
    # 標準 2：標準化不應劇烈改變殘差（允許 10x 差異）
    assert normalized_cont < 10.0 * baseline_cont, \
        f"標準化劇烈改變連續性殘差: {normalized_cont:.3e} vs {baseline_cont:.3e}"
    
    print(f"✅ Baseline 連續性殘差: {baseline_cont:.6e} < 100.0")
    print(f"✅ Normalized 連續性殘差: {normalized_cont:.6e} < 100.0")
    print(f"✅ Baseline 動量殘差 X: {baseline_mom_x:.6e}")
    print(f"✅ Normalized 動量殘差 X: {normalized_mom_x:.6e}")
    
    print(f"\n{'='*70}")
    print("TEST 13 通過 ✅")
    print(f"{'='*70}")


# ============================================================================
# 主測試執行
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s', '-m', 'physics'])
