"""
快速物理驗證評估腳本
評估檢查點的關鍵物理指標
"""
import torch
import numpy as np
import yaml
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pinnx
from pinnx.models.fourier_mlp import PINNNet, create_enhanced_pinn
from pinnx.models.wrappers import ManualScalingWrapper


def load_checkpoint_and_config(ckpt_path, config_path):
    """載入檢查點與配置"""
    # 載入檢查點
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print(f"✅ 檢查點載入成功: epoch {ckpt['epoch']}")
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 如果檢查點有嵌入配置，使用嵌入的（優先）
    if 'config' in ckpt:
        embedded_config = ckpt['config']
        print(f"📋 使用檢查點嵌入配置")
        return ckpt, embedded_config
    else:
        print(f"📋 使用外部配置文件")
        return ckpt, config


def create_model(config, device):
    """根據配置創建模型（複製自 evaluate_curriculum.py）"""
    model_cfg = config['model']
    
    # 建立基礎模型
    if model_cfg.get('type') == 'enhanced_fourier_mlp':
        base_model = create_enhanced_pinn(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=True,
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
            use_rwf=model_cfg.get('use_rwf', False),
            rwf_scale_std=model_cfg.get('rwf_scale_std', 0.1)
        ).to(device)
    else:
        base_model = PINNNet(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=True,
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0)
        ).to(device)
    
    # 檢查是否使用 scaling wrapper
    scaling_cfg = model_cfg.get('scaling', {})
    scaling_enabled = bool(scaling_cfg)
    
    if scaling_enabled:
        # 從配置中提取輸入輸出範圍
        input_x_range = tuple(scaling_cfg.get('input_norm', {}).get('x', [0.0, 25.13]))
        input_y_range = tuple(scaling_cfg.get('input_norm', {}).get('y', [-1.0, 1.0]))
        input_z_range = tuple(scaling_cfg.get('input_norm', {}).get('z', [0.0, 9.42]))
        
        input_scales = {
            'x': input_x_range,
            'y': input_y_range,
            'z': input_z_range
        }
        
        # 輸出範圍
        output_norm = scaling_cfg.get('output_norm', {})
        output_scales = {
            'u': tuple(output_norm.get('u', [0.0, 16.5])),
            'v': tuple(output_norm.get('v', [-0.6, 0.6])),
            'w': tuple(output_norm.get('w', [-0.6, 0.6])),
            'p': tuple(output_norm.get('p', [-50.0, 50.0]))
        }
        
        model = ManualScalingWrapper(
            base_model=base_model,
            input_scales=input_scales,
            output_scales=output_scales,
            learnable=scaling_cfg.get('learnable', False)
        ).to(device)
        
        print(f"🏗️  模型架構 (with scaling): {model_cfg['width']}×{model_cfg['depth']}, {model_cfg['activation']}")
    else:
        model = base_model
        print(f"🏗️  模型架構: {model_cfg['width']}×{model_cfg['depth']}, {model_cfg['activation']}")
    
    return model


def evaluate_physics(model, physics, device, n_test=2048):
    """評估物理一致性"""
    print("\n" + "="*70)
    print("🔬 物理驗證評估")
    print("="*70)
    
    model.eval()
    
    # 生成測試點（通道流域內）
    domain = physics.config['physics']['domain']
    
    x = torch.rand(n_test, 1, device=device) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
    y = torch.rand(n_test, 1, device=device) * (domain['y_range'][1] - domain['y_range'][0]) + domain['y_range'][0]
    z = torch.rand(n_test, 1, device=device) * (domain['z_range'][1] - domain['z_range'][0]) + domain['z_range'][0]
    
    coords = torch.cat([x, y, z], dim=1)
    coords.requires_grad_(True)
    
    with torch.no_grad():
        # 前向傳播
        output = model(coords)
        u, v, w, p = output[:, 0:1], output[:, 1:2], output[:, 2:3], output[:, 3:4]
        
        print(f"\n📊 預測場統計:")
        print(f"  u: mean={u.mean().item():.6f}, std={u.std().item():.6f}, min={u.min().item():.6f}, max={u.max().item():.6f}")
        print(f"  v: mean={v.mean().item():.6f}, std={v.std().item():.6f}, min={v.min().item():.6f}, max={v.max().item():.6f}")
        print(f"  w: mean={w.mean().item():.6f}, std={w.std().item():.6f}, min={w.min().item():.6f}, max={w.max().item():.6f}")
        print(f"  p: mean={p.mean().item():.6f}, std={p.std().item():.6f}, min={p.min().item():.6f}, max={p.max().item():.6f}")
    
    # 計算物理殘差（需要梯度）
    coords.requires_grad_(True)
    output = model(coords)
    
    residuals = physics.compute_pde_residuals(coords, output)
    
    print(f"\n🧮 PDE 殘差:")
    for key, val in residuals.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key}: mean={val.abs().mean().item():.6e}, max={val.abs().max().item():.6e}")
    
    # 壁面剪應力評估
    print(f"\n🧱 壁面邊界條件:")
    
    # 下壁面 (y=-1)
    x_wall = torch.rand(512, 1, device=device) * (domain['x_range'][1] - domain['x_range'][0]) + domain['x_range'][0]
    y_wall_lower = torch.full((512, 1), domain['y_range'][0], device=device)
    z_wall = torch.rand(512, 1, device=device) * (domain['z_range'][1] - domain['z_range'][0]) + domain['z_range'][0]
    
    coords_wall_lower = torch.cat([x_wall, y_wall_lower, z_wall], dim=1)
    coords_wall_lower.requires_grad_(True)
    
    output_wall = model(coords_wall_lower)
    u_wall = output_wall[:, 0:1]
    v_wall = output_wall[:, 1:2]
    w_wall = output_wall[:, 2:3]
    
    # 計算 du/dy (壁面剪應力)
    du_dy = torch.autograd.grad(u_wall.sum(), coords_wall_lower, create_graph=True)[0][:, 1:2]
    tau_w = physics.nu * du_dy.abs()
    
    print(f"  下壁面 (y=-1):")
    print(f"    u_wall: mean={u_wall.abs().mean().item():.6e}, max={u_wall.abs().max().item():.6e}")
    print(f"    v_wall: mean={v_wall.abs().mean().item():.6e}, max={v_wall.abs().max().item():.6e}")
    print(f"    w_wall: mean={w_wall.abs().mean().item():.6e}, max={w_wall.abs().max().item():.6e}")
    print(f"    τ_w (ν·du/dy): mean={tau_w.mean().item():.6e}, min={tau_w.min().item():.6e}, max={tau_w.max().item():.6e}")
    
    # 理論壁面剪應力
    u_tau = physics.config['physics']['channel_flow']['u_tau']
    tau_w_theory = physics.rho * u_tau**2
    print(f"    τ_w_理論 (ρ·u_τ²): {tau_w_theory:.6e}")
    print(f"    相對誤差: {abs(tau_w.mean().item() - tau_w_theory) / tau_w_theory * 100:.2f}%")
    
    # 上壁面 (y=+1)
    y_wall_upper = torch.full((512, 1), domain['y_range'][1], device=device)
    coords_wall_upper = torch.cat([x_wall, y_wall_upper, z_wall], dim=1)
    coords_wall_upper.requires_grad_(True)
    
    output_wall_upper = model(coords_wall_upper)
    u_wall_upper = output_wall_upper[:, 0:1]
    
    du_dy_upper = torch.autograd.grad(u_wall_upper.sum(), coords_wall_upper, create_graph=True)[0][:, 1:2]
    tau_w_upper = physics.nu * du_dy_upper.abs()
    
    print(f"\n  上壁面 (y=+1):")
    print(f"    τ_w: mean={tau_w_upper.mean().item():.6e}")
    print(f"    相對誤差: {abs(tau_w_upper.mean().item() - tau_w_theory) / tau_w_theory * 100:.2f}%")
    
    # 質量守恆
    if 'continuity' in residuals:
        mass_error = residuals['continuity'].abs().mean().item()
        print(f"\n⚖️  質量守恆:")
        print(f"  連續性殘差: {mass_error:.6e}")
        if mass_error < 1e-3:
            print(f"  ✅ 質量守恆良好 (<1e-3)")
        elif mass_error < 1e-2:
            print(f"  ⚠️  質量守恆可接受 (<1e-2)")
        else:
            print(f"  ❌ 質量守恆較差 (>1e-2)")
    
    print("\n" + "="*70)
    
    return {
        'tau_w_lower': tau_w.mean().item(),
        'tau_w_upper': tau_w_upper.mean().item(),
        'tau_w_theory': tau_w_theory,
        'u_wall_error': u_wall.abs().mean().item(),
        'mass_error': mass_error if 'continuity' in residuals else None,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="快速物理驗證評估")
    parser.add_argument('--checkpoint', type=str, required=True, help="檢查點路徑")
    parser.add_argument('--config', type=str, required=True, help="配置文件路徑")
    parser.add_argument('--n_test', type=int, default=2048, help="測試點數")
    args = parser.parse_args()
    
    # 設定設備
    device = setup_device('auto')
    print(f"🖥️  使用設備: {device}")
    
    # 載入檢查點與配置
    ckpt, config = load_checkpoint_and_config(args.checkpoint, args.config)
    
    # 創建模型
    model = create_model(config, device)
    
    # 載入權重
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print(f"✅ 模型權重載入成功")
    
    # 創建物理模組
    physics = create_physics(config, device)
    
    # 載入物理狀態（如果有）
    if 'physics_state_dict' in ckpt and ckpt['physics_state_dict'] is not None:
        physics.load_state_dict(ckpt['physics_state_dict'])
        print(f"✅ 物理模組狀態載入成功")
    
    # 評估
    results = evaluate_physics(model, physics, device, n_test=args.n_test)
    
    # 總結
    print("\n" + "="*70)
    print("📋 評估總結")
    print("="*70)
    print(f"Epoch: {ckpt['epoch']}")
    print(f"壁面剪應力 (下壁面): {results['tau_w_lower']:.6e}")
    print(f"壁面剪應力 (上壁面): {results['tau_w_upper']:.6e}")
    print(f"理論值: {results['tau_w_theory']:.6e}")
    print(f"相對誤差: {abs(results['tau_w_lower'] - results['tau_w_theory']) / results['tau_w_theory'] * 100:.2f}%")
    print(f"壁面速度誤差: {results['u_wall_error']:.6e}")
    if results['mass_error'] is not None:
        print(f"質量守恆誤差: {results['mass_error']:.6e}")
    print("="*70)


if __name__ == "__main__":
    main()
