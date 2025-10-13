#!/usr/bin/env python3
"""
NaN 訓練診斷腳本
診斷 VS-PINN Channel Flow 訓練中的 NaN 問題
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import PINNNet

def check_model_initialization():
    """檢查模型初始化是否產生 NaN"""
    print("=" * 60)
    print("步驟 1: 檢查模型初始化")
    print("=" * 60)
    
    model = PINNNet(
        in_dim=3,
        out_dim=4,
        width=200,
        depth=8,
        activation='sine',
        fourier_m=64,
        fourier_sigma=5.0
    )
    
    # 檢查權重
    has_nan = False
    has_inf = False
    weight_stats = []
    
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"❌ NaN detected in {name}")
            has_nan = True
        if torch.isinf(param).any():
            print(f"❌ Inf detected in {name}")
            has_inf = True
        
        weight_stats.append({
            'name': name,
            'shape': tuple(param.shape),
            'mean': param.mean().item(),
            'std': param.std().item(),
            'min': param.min().item(),
            'max': param.max().item()
        })
    
    if not has_nan and not has_inf:
        print("✅ 所有權重初始化正常")
    
    print("\n權重統計 (前5層):")
    for stat in weight_stats[:5]:
        print(f"  {stat['name']:30s} | shape={str(stat['shape']):15s} | "
              f"mean={stat['mean']:8.4f} | std={stat['std']:8.4f} | "
              f"range=[{stat['min']:8.4f}, {stat['max']:8.4f}]")
    
    return model, has_nan or has_inf

def check_forward_pass(model):
    """檢查前向傳播"""
    print("\n" + "=" * 60)
    print("步驟 2: 檢查前向傳播")
    print("=" * 60)
    
    # 測試輸入 - 使用配置檔的物理域
    x = torch.linspace(0, 25.13, 10)
    y = torch.linspace(-1.0, 1.0, 10)
    z = torch.linspace(0, 9.42, 10)
    
    X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    
    print(f"輸入座標: shape={coords.shape}")
    print(f"  x: [{coords[:,0].min():.3f}, {coords[:,0].max():.3f}]")
    print(f"  y: [{coords[:,1].min():.3f}, {coords[:,1].max():.3f}]")
    print(f"  z: [{coords[:,2].min():.3f}, {coords[:,2].max():.3f}]")
    
    # 前向傳播
    model.eval()
    with torch.no_grad():
        output = model(coords)
    
    print(f"\n輸出: shape={output.shape}")
    
    has_nan = torch.isnan(output).any()
    has_inf = torch.isinf(output).any()
    
    if has_nan:
        print(f"❌ 輸出包含 NaN: {torch.isnan(output).sum().item()} / {output.numel()}")
    if has_inf:
        print(f"❌ 輸出包含 Inf: {torch.isinf(output).sum().item()} / {output.numel()}")
    
    if not has_nan and not has_inf:
        print("✅ 前向傳播正常")
        print(f"  u: [{output[:,0].min():.3f}, {output[:,0].max():.3f}]")
        print(f"  v: [{output[:,1].min():.3f}, {output[:,1].max():.3f}]")
        print(f"  w: [{output[:,2].min():.3f}, {output[:,2].max():.3f}]")
        print(f"  p: [{output[:,3].min():.3f}, {output[:,3].max():.3f}]")
    
    return has_nan or has_inf

def check_sensor_data():
    """檢查感測點資料"""
    print("\n" + "=" * 60)
    print("步驟 3: 檢查感測點資料")
    print("=" * 60)
    
    sensor_file = Path("data/jhtdb/channel_flow_re1000/sensors_K50_qr_pivot_3d.npz")
    
    if not sensor_file.exists():
        print(f"❌ 感測點檔案不存在: {sensor_file}")
        return True
    
    data = np.load(sensor_file, allow_pickle=True)
    print(f"檔案: {sensor_file.name}")
    print(f"Keys: {list(data.files)}")
    
    if 'sensor_points' in data:
        pts = data['sensor_points']
        print(f"\nsensor_points: shape={pts.shape}, dtype={pts.dtype}")
        
        if pts.shape[1] != 3:
            print(f"❌ 感測點維度錯誤: 期望 (N, 3)，實際 {pts.shape}")
            return True
        
        has_nan = np.isnan(pts).any()
        has_inf = np.isinf(pts).any()
        
        if has_nan:
            print(f"❌ 感測點包含 NaN")
        if has_inf:
            print(f"❌ 感測點包含 Inf")
        
        if not has_nan and not has_inf:
            print(f"✅ 感測點正常")
            print(f"  x: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
            print(f"  y: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
            print(f"  z: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")
        
        return has_nan or has_inf
    
    return False

def check_gradient_computation(model):
    """檢查梯度計算"""
    print("\n" + "=" * 60)
    print("步驟 4: 檢查梯度計算 (自動微分)")
    print("=" * 60)
    
    # 測試點
    coords = torch.tensor([[12.56, 0.5, 4.71]], requires_grad=True)
    
    model.eval()
    output = model(coords)
    u, v, w, p = output[0, 0], output[0, 1], output[0, 2], output[0, 3]
    
    # 計算一階導數
    grads = []
    for var, name in [(u, 'u'), (v, 'v'), (w, 'w'), (p, 'p')]:
        grad = torch.autograd.grad(var, coords, create_graph=True)[0]
        grads.append((name, grad))
        
        has_nan = torch.isnan(grad).any()
        has_inf = torch.isinf(grad).any()
        
        status = "❌" if (has_nan or has_inf) else "✅"
        print(f"{status} ∂{name}/∂x: {grad[0,0].item():.6f}")
        print(f"{status} ∂{name}/∂y: {grad[0,1].item():.6f}")
        print(f"{status} ∂{name}/∂z: {grad[0,2].item():.6f}")
    
    # 測試二階導數 (u_xx)
    u_x = grads[0][1][0, 0]
    u_xx = torch.autograd.grad(u_x, coords, create_graph=True)[0][0, 0]
    
    has_nan = torch.isnan(u_xx).any()
    has_inf = torch.isinf(u_xx).any()
    status = "❌" if (has_nan or has_inf) else "✅"
    print(f"\n{status} ∂²u/∂x²: {u_xx.item():.6f}")
    
    return has_nan or has_inf

def main():
    print("\n" + "🔍 VS-PINN Channel Flow NaN 診斷" + "\n")
    
    errors = []
    
    # 1. 模型初始化
    model, error = check_model_initialization()
    if error:
        errors.append("模型初始化異常")
    
    # 2. 前向傳播
    error = check_forward_pass(model)
    if error:
        errors.append("前向傳播異常")
    
    # 3. 感測點資料
    error = check_sensor_data()
    if error:
        errors.append("感測點資料異常")
    
    # 4. 梯度計算
    error = check_gradient_computation(model)
    if error:
        errors.append("梯度計算異常")
    
    # 總結
    print("\n" + "=" * 60)
    print("診斷總結")
    print("=" * 60)
    
    if errors:
        print(f"❌ 發現 {len(errors)} 個問題:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        return 1
    else:
        print("✅ 所有檢查通過，NaN 可能來自:")
        print("  1. 損失函數計算邏輯")
        print("  2. VS-PINN 縮放係數設置")
        print("  3. 資料標準化流程")
        print("  4. 自適應權重初始值")
        print("\n建議:")
        print("  - 檢查 vs_pinn_channel_flow.py 的 compute_loss 方法")
        print("  - 驗證輸入資料標準化是否正確")
        print("  - 檢查初始損失權重是否合理")
        return 0

if __name__ == "__main__":
    sys.exit(main())
