#!/usr/bin/env python3
"""
快速物理驗證腳本 - 檢查檢查點的物理一致性

驗證項目：
1. 壁面邊界條件 (u,v,w ≈ 0 at y=-1, y=1)
2. 壁面剪應力 τ_w (目標 >5.0, 理想 >15.0)
3. 質量守恆 ∂u/∂x + ∂v/∂y + ∂w/∂z ≈ 0
4. 速度場統計分佈
"""

import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import create_enhanced_pinn

def load_checkpoint(checkpoint_path):
    """載入檢查點"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

def create_model_from_checkpoint(checkpoint):
    """從檢查點重建模型"""
    config = checkpoint['config']
    model_config = config['model']
    
    # 提取 Fourier 參數
    fourier_cfg = model_config.get('fourier_features', {})
    fourier_m = fourier_cfg.get('fourier_m', model_config.get('fourier_m', 32))
    
    # 從檢查點的 state_dict 中提取尺度因子
    input_scale_factors = None
    if 'input_scale_factors' in checkpoint['model_state_dict']:
        input_scale_factors = checkpoint['model_state_dict']['input_scale_factors']
    elif 'scaling' in config and 'input_scale_factors' in config['scaling']:
        input_scale_factors = config['scaling']['input_scale_factors']
    
    # 使用 factory 函數創建模型
    model = create_enhanced_pinn(
        in_dim=model_config.get('in_dim', 3),
        out_dim=model_config.get('out_dim', 4),
        width=model_config.get('width', 256),
        depth=model_config.get('depth', 6),
        activation=model_config.get('activation', 'sine'),
        use_fourier=model_config.get('use_fourier', True),
        fourier_m=fourier_m,
        fourier_sigma=fourier_cfg.get('fourier_sigma', 1.0),
        use_rwf=model_config.get('use_rwf', False),
        rwf_scale_std=model_config.get('rwf_scale_std', 0.1),
        fourier_normalize_input=fourier_cfg.get('normalize_input', True),
        input_scale_factors=input_scale_factors
    )
    
    # 載入 state_dict（strict=False 避免 input_scale_factors 的問題）
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    # 提取歸一化參數（用於反歸一化）
    normalization = checkpoint.get('normalization', None)
    
    return model, config, normalization

def denormalize_output(output, normalization):
    """反歸一化輸出"""
    if normalization is None:
        return output
    
    means = normalization['means']
    scales = normalization['scales']
    
    # output shape: [N, 4] -> [u, v, w, p]
    u = output[:, 0] * scales['u'] + means['u']
    v = output[:, 1] * scales['v'] + means['v']
    w = output[:, 2] * scales['w'] + means['w']
    p = output[:, 3] * scales['p'] + means['p']
    
    return torch.stack([u, v, w, p], dim=1)

def compute_gradients(model, normalization, x, y, z):
    """計算一階導數 (用於剪應力和質量守恆)，自動處理歸一化"""
    xyz = torch.stack([x, y, z], dim=1)
    xyz.requires_grad_(True)
    
    output_norm = model(xyz)
    output = denormalize_output(output_norm, normalization)
    
    u, v, w, p = output[:, 0], output[:, 1], output[:, 2], output[:, 3]
    
    # 計算梯度
    du_dy = torch.autograd.grad(u.sum(), xyz, create_graph=True)[0][:, 1]
    dv_dy = torch.autograd.grad(v.sum(), xyz, create_graph=True)[0][:, 1]
    dw_dy = torch.autograd.grad(w.sum(), xyz, create_graph=True)[0][:, 1]
    
    # 質量守恆項
    du_dx = torch.autograd.grad(u.sum(), xyz, create_graph=True)[0][:, 0]
    dv_dy_cont = torch.autograd.grad(v.sum(), xyz, create_graph=True)[0][:, 1]
    dw_dz = torch.autograd.grad(w.sum(), xyz, create_graph=True)[0][:, 2]
    
    return {
        'u': u, 'v': v, 'w': w, 'p': p,
        'du_dy': du_dy, 'dv_dy': dv_dy, 'dw_dy': dw_dy,
        'du_dx': du_dx, 'dv_dy_cont': dv_dy_cont, 'dw_dz': dw_dz
    }

def validate_wall_boundary(model, normalization, n_points=100):
    """驗證壁面邊界條件"""
    print("\n" + "="*70)
    print("🧱 壁面邊界條件驗證")
    print("="*70)
    
    # 在壁面生成測試點 (y=-1 和 y=1)
    x = torch.linspace(0, 8*np.pi, n_points)
    z = torch.linspace(0, 3*np.pi, n_points)
    x_wall, z_wall = torch.meshgrid(x, z, indexing='ij')
    x_wall, z_wall = x_wall.flatten(), z_wall.flatten()
    
    results = {}
    
    for wall_name, y_val in [("下壁面 (y=-1)", -1.0), ("上壁面 (y=+1)", 1.0)]:
        y_wall = torch.full_like(x_wall, y_val)
        
        with torch.no_grad():
            xyz_wall = torch.stack([x_wall, y_wall, z_wall], dim=1)
            output_norm = model(xyz_wall)
            output = denormalize_output(output_norm, normalization)
            
            u_wall = output[:, 0]
            v_wall = output[:, 1]
            w_wall = output[:, 2]
        
        results[wall_name] = {
            'u': u_wall.numpy(),
            'v': v_wall.numpy(),
            'w': w_wall.numpy()
        }
        
        print(f"\n{wall_name}:")
        print(f"  u: mean={u_wall.mean():.6f}, std={u_wall.std():.6f}, max_abs={u_wall.abs().max():.6f}")
        print(f"  v: mean={v_wall.mean():.6f}, std={v_wall.std():.6f}, max_abs={v_wall.abs().max():.6f}")
        print(f"  w: mean={w_wall.mean():.6f}, std={w_wall.std():.6f}, max_abs={w_wall.abs().max():.6f}")
        
        # 判斷是否滿足 no-slip
        u_violation = u_wall.abs().max().item()
        v_violation = v_wall.abs().max().item()
        w_violation = w_wall.abs().max().item()
        
        if max(u_violation, v_violation, w_violation) < 0.01:
            print(f"  ✅ No-slip 條件良好 (max violation < 0.01)")
        elif max(u_violation, v_violation, w_violation) < 0.1:
            print(f"  ⚠️  No-slip 條件尚可 (max violation < 0.1)")
        else:
            print(f"  ❌ No-slip 條件不佳 (max violation >= 0.1)")
    
    return results

def validate_wall_shear_stress(model, normalization, physics_config, n_points=100):
    """驗證壁面剪應力"""
    print("\n" + "="*70)
    print("📐 壁面剪應力驗證")
    print("="*70)
    
    nu = physics_config['nu']
    
    # 支援兩種配置格式
    if 'Re_tau' in physics_config:
        Re_tau = physics_config['Re_tau']
    elif 'channel_flow' in physics_config and 'Re_tau' in physics_config['channel_flow']:
        Re_tau = physics_config['channel_flow']['Re_tau']
    else:
        print("⚠️  無法找到 Re_tau，跳過剪應力驗證")
        return
    
    # 理論值: τ_w = ρ u_τ² (ρ=1, u_τ=0.04997)
    u_tau_theoretical = nu * Re_tau  # u_τ = ν Re_τ / δ_ν, δ_ν=1
    tau_w_theoretical = u_tau_theoretical ** 2
    
    print(f"\n理論值:")
    print(f"  ν = {nu}")
    print(f"  Re_τ = {Re_tau}")
    print(f"  u_τ (理論) = {u_tau_theoretical:.6f}")
    print(f"  τ_w (理論) = ρ·u_τ² = {tau_w_theoretical:.6f}")
    
    # 在壁面生成測試點
    x = torch.linspace(0, 8*np.pi, n_points)
    z = torch.linspace(0, 3*np.pi, n_points)
    x_wall, z_wall = torch.meshgrid(x, z, indexing='ij')
    x_wall, z_wall = x_wall.flatten(), z_wall.flatten()
    
    for wall_name, y_val in [("下壁面 (y=-1)", -1.0), ("上壁面 (y=+1)", 1.0)]:
        y_wall = torch.full_like(x_wall, y_val)
        
        grads = compute_gradients(model, normalization, x_wall, y_wall, z_wall)
        du_dy = grads['du_dy'].detach().numpy()
        
        # τ_w = μ (∂u/∂y)|_wall (μ = ρ·ν, ρ=1)
        tau_w = nu * du_dy
        
        print(f"\n{wall_name}:")
        print(f"  ∂u/∂y: mean={du_dy.mean():.3f}, std={du_dy.std():.3f}")
        print(f"  τ_w = ν·∂u/∂y: mean={tau_w.mean():.6f}, std={tau_w.std():.6f}")
        print(f"  τ_w / τ_w(理論): {tau_w.mean()/tau_w_theoretical:.3f}")
        
        # 判斷標準
        tau_w_mean = abs(tau_w.mean())
        if tau_w_mean > 15.0 * tau_w_theoretical:
            print(f"  ✅ 剪應力優秀 (>{15*tau_w_theoretical:.4f})")
        elif tau_w_mean > 5.0 * tau_w_theoretical:
            print(f"  ⚠️  剪應力尚可 (>{5*tau_w_theoretical:.4f})")
        else:
            print(f"  ❌ 剪應力不足 (<{5*tau_w_theoretical:.4f})")

def validate_mass_conservation(model, normalization, n_points=50):
    """驗證質量守恆"""
    print("\n" + "="*70)
    print("🌊 質量守恆驗證 (連續性方程)")
    print("="*70)
    
    # 在內部區域生成測試點
    x_test = torch.rand(n_points**2) * 8 * np.pi
    y_test = torch.rand(n_points**2) * 1.8 - 0.9  # 避開邊界
    z_test = torch.rand(n_points**2) * 3 * np.pi
    
    grads = compute_gradients(model, normalization, x_test, y_test, z_test)
    
    # ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z
    div_u = grads['du_dx'] + grads['dv_dy_cont'] + grads['dw_dz']
    div_u_np = div_u.detach().numpy()
    
    print(f"\n散度 ∇·u (不可壓縮要求: ≈ 0):")
    print(f"  mean = {div_u_np.mean():.6e}")
    print(f"  std  = {div_u_np.std():.6e}")
    print(f"  max  = {div_u_np.max():.6e}")
    print(f"  min  = {div_u_np.min():.6e}")
    print(f"  |∇·u| max = {np.abs(div_u_np).max():.6e}")
    
    # 判斷標準
    max_div = np.abs(div_u_np).max()
    if max_div < 1e-3:
        print(f"  ✅ 質量守恆優秀 (|∇·u| < 1e-3)")
    elif max_div < 1e-2:
        print(f"  ⚠️  質量守恆尚可 (|∇·u| < 1e-2)")
    else:
        print(f"  ❌ 質量守恆不佳 (|∇·u| >= 1e-2)")
    
    return div_u_np

def validate_velocity_statistics(model, normalization, n_points=100):
    """驗證速度場統計特性"""
    print("\n" + "="*70)
    print("📊 速度場統計驗證")
    print("="*70)
    
    # 在整個區域生成測試點
    x = torch.rand(n_points**2) * 8 * np.pi
    y = torch.rand(n_points**2) * 2 - 1  # [-1, 1]
    z = torch.rand(n_points**2) * 3 * np.pi
    
    with torch.no_grad():
        xyz = torch.stack([x, y, z], dim=1)
        output_norm = model(xyz)
        output = denormalize_output(output_norm, normalization)
        
        u = output[:, 0].numpy()
        v = output[:, 1].numpy()
        w = output[:, 2].numpy()
        p = output[:, 3].numpy()
    
    print(f"\n速度分量統計:")
    print(f"  u (流向): mean={u.mean():.4f}, std={u.std():.4f}, range=[{u.min():.4f}, {u.max():.4f}]")
    print(f"  v (壁面): mean={v.mean():.4f}, std={v.std():.4f}, range=[{v.min():.4f}, {v.max():.4f}]")
    print(f"  w (展向): mean={w.mean():.4f}, std={w.std():.4f}, range=[{w.min():.4f}, {w.max():.4f}]")
    print(f"  p (壓力): mean={p.mean():.4f}, std={p.std():.4f}, range=[{p.min():.4f}, {p.max():.4f}]")
    
    # 通道流特性檢查
    print(f"\n通道流特性檢查:")
    if u.mean() > 0.5:
        print(f"  ✅ 流向速度 u 平均值合理 (>{0.5})")
    else:
        print(f"  ⚠️  流向速度 u 平均值偏低 (<{0.5})")
    
    if abs(v.mean()) < 0.1:
        print(f"  ✅ 壁面法向速度 v 平均接近零")
    else:
        print(f"  ⚠️  壁面法向速度 v 平均值異常 (|mean|={abs(v.mean()):.4f})")
    
    if abs(w.mean()) < 0.1:
        print(f"  ✅ 展向速度 w 平均接近零")
    else:
        print(f"  ⚠️  展向速度 w 平均值異常 (|mean|={abs(w.mean()):.4f})")

def main():
    # 載入檢查點
    checkpoint_path = "checkpoints/test_physics_fix_1k_v2/best_model.pth"
    
    print("="*70)
    print("🔬 物理驗證開始")
    print("="*70)
    print(f"檢查點: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    model, config, normalization = create_model_from_checkpoint(checkpoint)
    
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"驗證集損失: {checkpoint['metrics'].get('val_loss', 'N/A'):.6f}")
    
    if normalization:
        print(f"\n⚠️  檢測到輸出歸一化:")
        print(f"  Means: u={normalization['means']['u']:.3f}, v={normalization['means']['v']:.3f}, w={normalization['means']['w']:.3f}, p={normalization['means']['p']:.3f}")
        print(f"  Scales: u={normalization['scales']['u']:.3f}, v={normalization['scales']['v']:.3f}, w={normalization['scales']['w']:.3f}, p={normalization['scales']['p']:.3f}")
    
    # 執行驗證（傳遞歸一化參數）
    validate_wall_boundary(model, normalization, n_points=100)
    validate_wall_shear_stress(model, normalization, config['physics'], n_points=100)
    validate_mass_conservation(model, normalization, n_points=50)
    validate_velocity_statistics(model, normalization, n_points=100)
    
    print("\n" + "="*70)
    print("✅ 物理驗證完成")
    print("="*70)

if __name__ == "__main__":
    main()
