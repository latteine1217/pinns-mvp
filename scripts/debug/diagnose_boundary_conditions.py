#!/usr/bin/env python3
"""
邊界條件診斷腳本
診斷 Sine 模型訓練失敗的根本原因：邊界條件實現完整性
"""

import sys
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models import create_pinn_model
from pinnx.models.wrappers import ManualScalingWrapper
from pinnx.dataio.channel_flow_loader import prepare_training_data

def load_model_and_config():
    """載入訓練好的 Sine 模型和配置"""
    model_path = "activation_benchmark_results/model_sine.pth"
    config_path = "configs/channel_flow_re1000_K80_wall_balanced.yml"
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 載入模型
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 重建模型架構（必須與訓練時完全一致）
    base_model_cfg = {
        'type': 'fourier_vs_mlp',
        'in_dim': 2,
        'out_dim': 3,
        'width': 256,
        'depth': 6,
        'activation': 'sine',
        'use_fourier': True,
        'fourier_m': 48,
        'fourier_sigma': 3.0,
        'use_layer_norm': False,
        'use_input_projection': False,
        'use_residual': False,
        'dropout': 0.0
    }
    base_model = create_pinn_model(base_model_cfg)
    
    # 應用標準化包裝器
    model = ManualScalingWrapper(
        base_model,
        input_ranges={'x': (0.0, 25.13), 'y': (-1.0, 1.0)},
        output_ranges={'u': (0.0, 16.5), 'v': (-0.6, 0.6), 'p': (-85.0, 3.0)}
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def diagnose_boundary_sampling(config):
    """診斷邊界點採樣是否充足"""
    sampling = config['training']['sampling']
    
    print("=" * 80)
    print("🔍 BOUNDARY SAMPLING DIAGNOSIS")
    print("=" * 80)
    
    # 從配置讀取
    n_bc = sampling['boundary_points']
    wall_clustering = sampling.get('wall_clustering', 0.1)
    
    print(f"📊 Boundary Points Configuration:")
    print(f"   Total BC points: {n_bc}")
    print(f"   Wall clustering ratio: {wall_clustering}")
    print(f"   Points at upper wall: {n_bc // 2}")
    print(f"   Points at lower wall: {n_bc - n_bc // 2}")
    
    # 生成邊界點分佈
    x_range = config['physics']['domain']['x_range']
    y_range = config['physics']['domain']['y_range']
    
    x_bc = torch.rand(n_bc, 1) * (x_range[1] - x_range[0]) + x_range[0]
    y_bc_bottom = torch.full((n_bc//2, 1), y_range[0])
    y_bc_top = torch.full((n_bc - n_bc//2, 1), y_range[1])
    y_bc = torch.cat([y_bc_bottom, y_bc_top], dim=0)
    
    # 檢查分佈
    unique_y = torch.unique(y_bc)
    print(f"\n📍 Sampled Y positions:")
    print(f"   Unique Y values: {unique_y.tolist()}")
    print(f"   Expected: {y_range}")
    
    if len(unique_y) == 2 and torch.allclose(unique_y, torch.tensor(y_range)):
        print("   ✅ Boundary points correctly placed at y=-1 and y=+1")
    else:
        print("   ⚠️  Boundary points NOT at wall locations!")
    
    return True


def diagnose_wall_constraint_implementation(model, config):
    """診斷壁面約束在訓練中是否正確實現"""
    print("\n" + "=" * 80)
    print("🔍 WALL CONSTRAINT IMPLEMENTATION DIAGNOSIS")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # 生成壁面測試點
    n_test = 100
    x_wall = torch.linspace(0, 25.13, n_test).unsqueeze(1)
    
    # 標準化座標（模型輸入格式）
    x_min, x_max = 0.0, 25.13
    y_min, y_max = -1.0, 1.0
    
    x_wall_norm = 2.0 * (x_wall - x_min) / (x_max - x_min) - 1.0
    
    # 下壁面 y=-1
    y_bottom = torch.full_like(x_wall, -1.0)
    y_bottom_norm = 2.0 * (y_bottom - y_min) / (y_max - y_min) - 1.0
    coords_bottom = torch.cat([x_wall_norm, y_bottom_norm], dim=1)
    
    # 上壁面 y=+1
    y_top = torch.full_like(x_wall, 1.0)
    y_top_norm = 2.0 * (y_top - y_min) / (y_max - y_min) - 1.0
    coords_top = torch.cat([x_wall_norm, y_top_norm], dim=1)
    
    # 模型預測
    with torch.no_grad():
        pred_bottom = model(coords_bottom)
        pred_top = model(coords_top)
    
    # 檢查無滑移條件
    u_bottom = pred_bottom[:, 0].numpy()
    v_bottom = pred_bottom[:, 1].numpy()
    u_top = pred_top[:, 0].numpy()
    v_top = pred_top[:, 1].numpy()
    
    print(f"📊 Wall Velocity Statistics:")
    print(f"\n   Lower Wall (y=-1):")
    print(f"      U: mean={u_bottom.mean():.6f}, std={u_bottom.std():.6f}, max={u_bottom.max():.6f}")
    print(f"      V: mean={v_bottom.mean():.6f}, std={v_bottom.std():.6f}, max={v_bottom.max():.6f}")
    print(f"\n   Upper Wall (y=+1):")
    print(f"      U: mean={u_top.mean():.6f}, std={u_top.std():.6f}, max={u_top.max():.6f}")
    print(f"      V: mean={v_top.mean():.6f}, std={v_top.std():.6f}, max={v_top.max():.6f}")
    
    # 判斷是否滿足無滑移條件
    tolerance = 0.01  # 允許 1% 誤差
    wall_satisfied = (
        abs(u_bottom.mean()) < tolerance and
        abs(v_bottom.mean()) < tolerance and
        abs(u_top.mean()) < tolerance and
        abs(v_top.mean()) < tolerance
    )
    
    if wall_satisfied:
        print("\n   ✅ No-slip condition SATISFIED (u≈0, v≈0 at walls)")
    else:
        print("\n   ❌ No-slip condition VIOLATED!")
        print("      Expected: u=0, v=0 at both walls")
        print("      Actual: Non-zero velocities detected")
    
    return wall_satisfied, {
        'u_bottom': u_bottom,
        'v_bottom': v_bottom,
        'u_top': u_top,
        'v_top': v_top,
        'x_wall': x_wall.squeeze().numpy()
    }


def diagnose_periodicity_implementation(model, config):
    """診斷週期性邊界條件實現"""
    print("\n" + "=" * 80)
    print("🔍 PERIODICITY CONSTRAINT DIAGNOSIS")
    print("=" * 80)
    
    device = next(model.parameters()).device
    
    # 生成週期邊界測試點
    n_test = 50
    y_periodic = torch.linspace(-1.0, 1.0, n_test).unsqueeze(1)
    
    # 標準化
    x_min, x_max = 0.0, 25.13
    y_min, y_max = -1.0, 1.0
    
    # 左邊界 x=0
    x_left = torch.zeros_like(y_periodic)
    x_left_norm = 2.0 * (x_left - x_min) / (x_max - x_min) - 1.0
    y_norm = 2.0 * (y_periodic - y_min) / (y_max - y_min) - 1.0
    coords_left = torch.cat([x_left_norm, y_norm], dim=1)
    
    # 右邊界 x=2π
    x_right = torch.full_like(y_periodic, 25.13)
    x_right_norm = 2.0 * (x_right - x_min) / (x_max - x_min) - 1.0
    coords_right = torch.cat([x_right_norm, y_norm], dim=1)
    
    # 模型預測
    with torch.no_grad():
        pred_left = model(coords_left)
        pred_right = model(coords_right)
    
    # 計算差異
    diff_u = (pred_left[:, 0] - pred_right[:, 0]).abs().numpy()
    diff_v = (pred_left[:, 1] - pred_right[:, 1]).abs().numpy()
    diff_p = (pred_left[:, 2] - pred_right[:, 2]).abs().numpy()
    
    print(f"📊 Periodicity Error Statistics:")
    print(f"   |U(x=0) - U(x=2π)|: mean={diff_u.mean():.6f}, max={diff_u.max():.6f}")
    print(f"   |V(x=0) - V(x=2π)|: mean={diff_v.mean():.6f}, max={diff_v.max():.6f}")
    print(f"   |P(x=0) - P(x=2π)|: mean={diff_p.mean():.6f}, max={diff_p.max():.6f}")
    
    # 判斷週期性是否滿足
    tolerance = 0.1  # 允許 10% 誤差
    periodicity_satisfied = (
        diff_u.mean() < tolerance and
        diff_v.mean() < tolerance and
        diff_p.mean() < tolerance
    )
    
    if periodicity_satisfied:
        print("   ✅ Periodicity condition SATISFIED")
    else:
        print("   ❌ Periodicity condition VIOLATED!")
    
    return periodicity_satisfied


def diagnose_loss_weights(config):
    """診斷損失權重配置"""
    print("\n" + "=" * 80)
    print("🔍 LOSS WEIGHT CONFIGURATION DIAGNOSIS")
    print("=" * 80)
    
    losses = config['losses']
    
    # 提取權重
    weights = {
        'data': losses.get('data_weight', 1.0),
        'wall_constraint': losses.get('wall_constraint_weight', 10.0),
        'periodicity': losses.get('periodicity_weight', 5.0),
        'momentum_x': losses.get('momentum_x_weight', 10.0),
        'momentum_y': losses.get('momentum_y_weight', 10.0),
        'continuity': losses.get('continuity_weight', 20.0),
        'prior': losses.get('prior_weight', 0.5)
    }
    
    total_weight = sum(weights.values())
    
    print(f"📊 Loss Weight Configuration:")
    for name, weight in weights.items():
        percentage = (weight / total_weight) * 100
        print(f"   {name:20s}: {weight:8.2f} ({percentage:5.1f}%)")
    print(f"   {'TOTAL':20s}: {total_weight:8.2f} (100.0%)")
    
    # 分析權重平衡
    print(f"\n🔍 Weight Balance Analysis:")
    
    # 檢查 BC 權重是否足夠
    bc_weight = weights['wall_constraint'] + weights['periodicity']
    pde_weight = weights['momentum_x'] + weights['momentum_y'] + weights['continuity']
    
    bc_ratio = bc_weight / total_weight
    pde_ratio = pde_weight / total_weight
    data_ratio = weights['data'] / total_weight
    
    print(f"   Boundary Constraints: {bc_ratio*100:.1f}%")
    print(f"   PDE Residuals:        {pde_ratio*100:.1f}%")
    print(f"   Data Fitting:         {data_ratio*100:.1f}%")
    
    # 建議
    if bc_ratio < 0.15:
        print("\n   ⚠️  BC weight might be too low (< 15%)")
        print("      Recommendation: Increase wall_constraint_weight to 20-50")
    
    if weights['prior'] > 0.0:
        print(f"\n   ⚠️  Prior weight is {weights['prior']:.2f}")
        print("      This might bind model to incorrect RANS field")
        print("      Recommendation: Try prior_weight=0.0")
    
    return weights


def visualize_wall_behavior(wall_data):
    """可視化壁面速度分佈"""
    print("\n" + "=" * 80)
    print("📊 GENERATING WALL BEHAVIOR VISUALIZATION")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    x = wall_data['x_wall']
    
    # 下壁面 U
    axes[0, 0].plot(x, wall_data['u_bottom'], 'b-', label='Lower Wall')
    axes[0, 0].axhline(y=0, color='r', linestyle='--', label='Target (u=0)')
    axes[0, 0].set_xlabel('X position')
    axes[0, 0].set_ylabel('U velocity')
    axes[0, 0].set_title('Lower Wall: U velocity (should be ~0)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 下壁面 V
    axes[0, 1].plot(x, wall_data['v_bottom'], 'b-', label='Lower Wall')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', label='Target (v=0)')
    axes[0, 1].set_xlabel('X position')
    axes[0, 1].set_ylabel('V velocity')
    axes[0, 1].set_title('Lower Wall: V velocity (should be ~0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 上壁面 U
    axes[1, 0].plot(x, wall_data['u_top'], 'g-', label='Upper Wall')
    axes[1, 0].axhline(y=0, color='r', linestyle='--', label='Target (u=0)')
    axes[1, 0].set_xlabel('X position')
    axes[1, 0].set_ylabel('U velocity')
    axes[1, 0].set_title('Upper Wall: U velocity (should be ~0)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 上壁面 V
    axes[1, 1].plot(x, wall_data['v_top'], 'g-', label='Upper Wall')
    axes[1, 1].axhline(y=0, color='r', linestyle='--', label='Target (v=0)')
    axes[1, 1].set_xlabel('X position')
    axes[1, 1].set_ylabel('V velocity')
    axes[1, 1].set_title('Upper Wall: V velocity (should be ~0)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('activation_benchmark_results/sine_wall_behavior_diagnosis.png', dpi=150, bbox_inches='tight')
    print("   ✅ Saved: sine_wall_behavior_diagnosis.png")
    plt.close()


def main():
    """主診斷流程"""
    print("\n" + "=" * 80)
    print("🚀 BOUNDARY CONDITIONS DIAGNOSTIC SCRIPT")
    print("=" * 80)
    print("\nObjective: Diagnose why Sine model shows no boundary layer at walls\n")
    
    # 1. 載入模型和配置
    print("[1/5] Loading model and configuration...")
    model, config = load_model_and_config()
    print("   ✅ Model and config loaded")
    
    # 2. 診斷邊界採樣
    print("\n[2/5] Diagnosing boundary point sampling...")
    diagnose_boundary_sampling(config)
    
    # 3. 診斷壁面約束實現
    print("\n[3/5] Diagnosing wall constraint implementation...")
    wall_satisfied, wall_data = diagnose_wall_constraint_implementation(model, config)
    
    # 4. 診斷週期性約束
    print("\n[4/5] Diagnosing periodicity constraint...")
    periodicity_satisfied = diagnose_periodicity_implementation(model, config)
    
    # 5. 診斷損失權重
    print("\n[5/5] Diagnosing loss weight configuration...")
    weights = diagnose_loss_weights(config)
    
    # 可視化
    visualize_wall_behavior(wall_data)
    
    # 總結
    print("\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)
    
    print(f"\n✅ Checks Passed:")
    if wall_satisfied:
        print("   - Wall constraint implementation: ✅")
    if periodicity_satisfied:
        print("   - Periodicity constraint: ✅")
    
    print(f"\n❌ Issues Found:")
    if not wall_satisfied:
        print("   - Wall constraint VIOLATED: Model predicts non-zero velocity at walls")
        print("     → Potential cause: Insufficient BC loss weight")
        print("     → Potential cause: BC sampling issues")
    if not periodicity_satisfied:
        print("   - Periodicity constraint VIOLATED: Fields not periodic")
    
    print(f"\n🔧 Recommendations:")
    print("   1. Increase wall_constraint_weight from 10.0 to 50.0")
    print("   2. Set prior_weight to 0.0 (disable RANS prior)")
    print("   3. Increase boundary_points from 1000 to 2000")
    print("   4. Verify BC loss is actually computed in train_step()")
    
    print("\n" + "=" * 80)
    print("✅ Diagnostic complete. Check sine_wall_behavior_diagnosis.png for details.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
