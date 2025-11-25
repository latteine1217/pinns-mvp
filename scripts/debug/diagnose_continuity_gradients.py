"""
診斷連續方程梯度流

檢查項目：
1. continuity residual 計算是否正確
2. 梯度是否正確傳遞到模型參數
3. 梯度數值是否合理
4. 與其他損失項的梯度比較
"""

import torch
import numpy as np
import yaml
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
import torch.nn as nn


def check_gradient_flow(model, physics, coords_pde, u_pred, loss_name=""):
    """
    檢查特定損失項的梯度流
    
    Returns:
        grad_stats: 梯度統計資訊字典
    """
    # 計算損失
    residual = None
    if loss_name == "continuity":
        residual = physics.compute_continuity_residual(coords_pde, u_pred)
        loss = torch.mean(residual ** 2)
    elif loss_name == "momentum_x":
        residuals = physics.compute_momentum_residuals(coords_pde, u_pred)
        loss = torch.mean(residuals['momentum_x'] ** 2)
    elif loss_name == "data":
        # 簡單的 L2 損失
        target = torch.randn_like(u_pred)
        loss = torch.mean((u_pred - target) ** 2)
    else:
        raise ValueError(f"Unknown loss name: {loss_name}")
    
    # 清空梯度
    model.zero_grad()
    
    # 反向傳播
    loss.backward(retain_graph=True)
    
    # 收集梯度統計
    grad_stats = {
        'loss_value': loss.item(),
        'residual_mean': residual.mean().item() if residual is not None else 0.0,
        'residual_std': residual.std().item() if residual is not None else 0.0,
        'residual_max': residual.abs().max().item() if residual is not None else 0.0,
        'param_grads': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_stats['param_grads'].append({
                'name': name,
                'grad_mean': param.grad.abs().mean().item(),
                'grad_std': param.grad.std().item(),
                'grad_max': param.grad.abs().max().item(),
                'grad_norm': param.grad.norm().item(),
                'param_norm': param.norm().item(),
                'has_nan': torch.isnan(param.grad).any().item(),
                'has_inf': torch.isinf(param.grad).any().item(),
            })
        else:
            grad_stats['param_grads'].append({
                'name': name,
                'grad_mean': 0.0,
                'grad_std': 0.0,
                'grad_max': 0.0,
                'grad_norm': 0.0,
                'param_norm': param.norm().item(),
                'has_nan': False,
                'has_inf': False,
            })
    
    return grad_stats


def check_autograd_graph(coords_pde, u_pred, physics):
    """
    檢查自動微分計算圖連接
    """
    print("\n" + "=" * 80)
    print("🔍 檢查自動微分計算圖")
    print("=" * 80)
    
    # 檢查輸入張量的梯度追蹤
    print(f"\n1️⃣ 輸入張量梯度追蹤狀態:")
    print(f"   coords_pde.requires_grad: {coords_pde.requires_grad}")
    print(f"   u_pred.requires_grad: {u_pred.requires_grad}")
    print(f"   coords_pde.grad_fn: {coords_pde.grad_fn}")
    print(f"   u_pred.grad_fn: {u_pred.grad_fn}")
    
    # 手動計算一階導數
    print(f"\n2️⃣ 手動計算速度場梯度:")
    u = u_pred[:, 0:1]
    v = u_pred[:, 1:2]
    
    # 計算 ∂u/∂x
    du_dx = torch.autograd.grad(
        outputs=u,
        inputs=coords_pde,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True
    )[0][:, 0:1]
    
    # 計算 ∂v/∂y
    dv_dy = torch.autograd.grad(
        outputs=v,
        inputs=coords_pde,
        grad_outputs=torch.ones_like(v),
        create_graph=True,
        retain_graph=True
    )[0][:, 1:2]
    
    # 連續性殘差：∂u/∂x + ∂v/∂y
    div_manual = du_dx + dv_dy
    
    print(f"   ∂u/∂x: mean={du_dx.mean().item():.6f}, std={du_dx.std().item():.6f}")
    print(f"   ∂v/∂y: mean={dv_dy.mean().item():.6f}, std={dv_dy.std().item():.6f}")
    print(f"   散度 (∂u/∂x + ∂v/∂y): mean={div_manual.mean().item():.6f}, std={div_manual.std().item():.6f}")
    
    # 與物理模組計算的結果比較
    print(f"\n3️⃣ 物理模組計算的連續性殘差:")
    continuity_residual = physics.compute_continuity_residual(coords_pde, u_pred)
    print(f"   殘差: mean={continuity_residual.mean().item():.6f}, std={continuity_residual.std().item():.6f}")
    print(f"   requires_grad: {continuity_residual.requires_grad}")
    print(f"   grad_fn: {continuity_residual.grad_fn}")
    
    # 比較一致性
    diff = (div_manual - continuity_residual).abs()
    print(f"\n4️⃣ 手動計算 vs 物理模組:")
    print(f"   絕對差異: mean={diff.mean().item():.6e}, max={diff.max().item():.6e}")
    
    if diff.max().item() < 1e-6:
        print(f"   ✅ 計算一致！")
    else:
        print(f"   ❌ 計算不一致，可能存在問題！")
    
    return div_manual, continuity_residual


def main():
    print("=" * 80)
    print("🔬 Kolmogorov Flow 2D 連續方程梯度診斷")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📍 設備: {device}")
    
    # 1. 載入配置
    config_path = project_root / "configs" / "kolmogorov_2d_baseline.yml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ 配置載入完成: {config_path.name}")
    
    # 2. 初始化模型（使用簡單 MLP）
    print(f"\n📦 初始化模型...")
    hidden_layers = config['model']['hidden_layers']
    
    # 簡單的 MLP 用於診斷
    layers = []
    in_dim = 2
    for h_dim in hidden_layers:
        layers.append(nn.Linear(in_dim, h_dim))
        layers.append(nn.Tanh())  # 使用 tanh 以便梯度穩定
        in_dim = h_dim
    layers.append(nn.Linear(in_dim, 3))  # 輸出 (u, v, p)
    
    model = nn.Sequential(*layers).to(device)
    
    print(f"   模型參數數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 3. 初始化物理模組
    print(f"\n⚙️  初始化物理模組...")
    physics_cfg = config['physics']
    physics = KolmogorovFlow2D(
        forcing_params={
            'amplitude': physics_cfg['forcing']['amplitude'],
            'wavenumber': physics_cfg['forcing']['wavenumber']
        },
        physics_params={
            'nu': physics_cfg['nu'],
            'rho': physics_cfg['rho']
        },
        domain_bounds={
            'x': (physics_cfg['domain']['x_min'], physics_cfg['domain']['x_max']),
            'y': (physics_cfg['domain']['y_min'], physics_cfg['domain']['y_max'])
        }
    ).to(device)
    
    # 4. 生成測試數據
    print(f"\n🎲 生成測試數據...")
    n_points = 100
    x_min, x_max = physics_cfg['domain']['x_min'], physics_cfg['domain']['x_max']
    y_min, y_max = physics_cfg['domain']['y_min'], physics_cfg['domain']['y_max']
    
    coords_pde = torch.rand(n_points, 2, device=device)
    coords_pde[:, 0] = coords_pde[:, 0] * (x_max - x_min) + x_min
    coords_pde[:, 1] = coords_pde[:, 1] * (y_max - y_min) + y_min
    coords_pde.requires_grad_(True)
    
    print(f"   座標範圍: x ∈ [{coords_pde[:, 0].min():.3f}, {coords_pde[:, 0].max():.3f}]")
    print(f"              y ∈ [{coords_pde[:, 1].min():.3f}, {coords_pde[:, 1].max():.3f}]")
    
    # 5. 模型前向傳播
    print(f"\n🔮 模型前向傳播...")
    u_pred = model(coords_pde)
    print(f"   輸出形狀: {u_pred.shape}")
    print(f"   u: mean={u_pred[:, 0].mean().item():.6f}, std={u_pred[:, 0].std().item():.6f}")
    print(f"   v: mean={u_pred[:, 1].mean().item():.6f}, std={u_pred[:, 1].std().item():.6f}")
    print(f"   p: mean={u_pred[:, 2].mean().item():.6f}, std={u_pred[:, 2].std().item():.6f}")
    
    # 6. 檢查計算圖
    div_manual, continuity_residual = check_autograd_graph(coords_pde, u_pred, physics)
    
    # 7. 檢查梯度流
    print("\n" + "=" * 80)
    print("📊 梯度流診斷")
    print("=" * 80)
    
    losses_to_check = ["continuity", "momentum_x", "data"]
    results = {}
    
    for loss_name in losses_to_check:
        print(f"\n{'─' * 80}")
        print(f"🔍 檢查 {loss_name.upper()} 損失的梯度")
        print(f"{'─' * 80}")
        
        stats = check_gradient_flow(model, physics, coords_pde, u_pred, loss_name)
        results[loss_name] = stats
        
        print(f"\n損失值: {stats['loss_value']:.6e}")
        if loss_name == "continuity":
            print(f"殘差統計:")
            print(f"  - mean: {stats['residual_mean']:.6e}")
            print(f"  - std:  {stats['residual_std']:.6e}")
            print(f"  - max:  {stats['residual_max']:.6e}")
        
        print(f"\n前 5 層參數的梯度:")
        for i, pg in enumerate(stats['param_grads'][:5]):
            print(f"  {pg['name']:30s} | grad_norm: {pg['grad_norm']:.6e} | "
                  f"grad_mean: {pg['grad_mean']:.6e} | "
                  f"NaN: {pg['has_nan']} | Inf: {pg['has_inf']}")
    
    # 8. 比較不同損失項的梯度
    print("\n" + "=" * 80)
    print("📈 梯度比較分析")
    print("=" * 80)
    
    print(f"\n{'損失項':<15} {'損失值':>15} {'第1層梯度範數':>20} {'最後1層梯度範數':>20}")
    print("─" * 72)
    
    for loss_name in losses_to_check:
        loss_val = results[loss_name]['loss_value']
        first_layer_grad = results[loss_name]['param_grads'][0]['grad_norm']
        last_layer_grad = results[loss_name]['param_grads'][-1]['grad_norm']
        print(f"{loss_name:<15} {loss_val:>15.6e} {first_layer_grad:>20.6e} {last_layer_grad:>20.6e}")
    
    # 9. 診斷結論
    print("\n" + "=" * 80)
    print("🎯 診斷結論")
    print("=" * 80)
    
    continuity_stats = results['continuity']
    data_stats = results['data']
    
    # 檢查梯度是否過小
    continuity_grad_norm = continuity_stats['param_grads'][0]['grad_norm']
    data_grad_norm = data_stats['param_grads'][0]['grad_norm']
    grad_ratio = continuity_grad_norm / (data_grad_norm + 1e-12)
    
    print(f"\n1️⃣ 梯度範數比較:")
    print(f"   Continuity 第1層梯度: {continuity_grad_norm:.6e}")
    print(f"   Data 第1層梯度:       {data_grad_norm:.6e}")
    print(f"   比值 (Continuity/Data): {grad_ratio:.6f}")
    
    if grad_ratio < 0.01:
        print(f"   ❌ Continuity 梯度過小（< 1% Data 梯度），權重可能不足！")
    elif grad_ratio > 100:
        print(f"   ⚠️  Continuity 梯度過大（> 100× Data 梯度），可能導致訓練不穩定！")
    else:
        print(f"   ✅ 梯度比例合理")
    
    # 檢查 NaN/Inf
    has_nan = any(pg['has_nan'] for pg in continuity_stats['param_grads'])
    has_inf = any(pg['has_inf'] for pg in continuity_stats['param_grads'])
    
    print(f"\n2️⃣ 梯度健康檢查:")
    if has_nan or has_inf:
        print(f"   ❌ 檢測到 NaN 或 Inf 梯度！")
    else:
        print(f"   ✅ 無 NaN/Inf 梯度")
    
    # 檢查殘差量級
    print(f"\n3️⃣ 連續性殘差分析:")
    print(f"   殘差均值: {continuity_stats['residual_mean']:.6e}")
    print(f"   殘差標準差: {continuity_stats['residual_std']:.6e}")
    
    if abs(continuity_stats['residual_mean']) > 10.0:
        print(f"   ⚠️  殘差均值過大，散度嚴重違反！")
    elif abs(continuity_stats['residual_mean']) < 1e-3:
        print(f"   ✅ 殘差均值接近零，連續性良好")
    else:
        print(f"   ℹ️  殘差中等，需進一步訓練")
    
    print("\n" + "=" * 80)
    print("✅ 診斷完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
