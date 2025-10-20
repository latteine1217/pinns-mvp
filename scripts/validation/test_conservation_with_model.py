#!/usr/bin/env python3
"""
測試修復後的 conservation_error 函數與實際模型
Test Fixed conservation_error Function with Real Model
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))

from pinnx.evals.metrics import conservation_error
from pinnx.models.wrappers import ManualScalingWrapper
from pinnx.models.fourier_mlp import PINNNet

def main():
    """主測試"""
    print("="*60)
    print("測試修復後的 conservation_error 函數")
    print("="*60)
    
    # 載入模型
    checkpoint_path = "checkpoints/pinnx_channel_flow_re1000_fix6_k50_phase4b_latest.pth"
    device = torch.device('cpu')
    
    print(f"\n1. 載入 checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 創建模型
    input_ranges = {'x': (0.0, 25.13), 'y': (-1.0, 1.0)}
    output_ranges = {'u': (0.0, 20.0), 'v': (-1.0, 1.0), 'p': (-100.0, 10.0)}
    
    backbone = FourierMLP(
        input_dim=2,
        output_dim=3,
        hidden_dim=256,
        num_layers=6,
        fourier_m=48,
        fourier_std=3.0
    )
    
    model = ManualScalingWrapper(
        model=backbone,
        input_ranges=input_ranges,
        output_ranges=output_ranges
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✅ 模型載入成功")
    
    # 生成測試點（2D 座標）
    print("\n2. 生成測試網格")
    x = torch.linspace(0.5, 24, 64)
    y = torch.linspace(-0.9, 0.9, 32)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack([X.flatten(), Y.flatten()], dim=-1)
    
    print(f"   Coords shape: {coords.shape}")
    print(f"   Coords dtype: {coords.dtype}")
    
    # 確保 coords 需要梯度
    coords = coords.clone().detach().requires_grad_(True)
    
    # 模型預測
    print("\n3. 模型預測")
    with torch.no_grad():
        output = model(coords)
    
    # 重要：需要重新用 requires_grad=True 的 coords 計算 u, v
    # 這樣才能建立梯度連接
    print("\n4. 重新計算預測（建立梯度連接）")
    coords_grad = coords.clone().detach().requires_grad_(True)
    output_grad = model(coords_grad)
    
    u = output_grad[:, 0]
    v = output_grad[:, 1]
    p = output_grad[:, 2]
    
    print(f"   U range: [{u.min().item():.4f}, {u.max().item():.4f}]")
    print(f"   V range: [{v.min().item():.4f}, {v.max().item():.4f}]")
    print(f"   P range: [{p.min().item():.4f}, {p.max().item():.4f}]")
    
    # 測試 conservation_error
    print("\n5. 測試 conservation_error")
    
    # 檢查梯度連接
    print(f"   u.grad_fn: {u.grad_fn}")
    print(f"   v.grad_fn: {v.grad_fn}")
    print(f"   coords_grad.requires_grad: {coords_grad.requires_grad}")
    
    try:
        error = conservation_error(u, v, coords_grad)
        print(f"\n✅ conservation_error = {error:.6e}")
        
        if error < 1e-3:
            print(f"   ✅ 質量守恆達標 (< 1e-3)")
        elif error < 0.1:
            print(f"   ⚠️  質量守恆可接受 (< 0.1)")
        elif not np.isinf(error):
            print(f"   ❌ 質量守恆誤差偏高")
        else:
            print(f"   ❌ 質量守恆計算失敗 (inf)")
            
    except Exception as e:
        print(f"\n❌ conservation_error 調用失敗: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("測試完成")
    print("="*60)

if __name__ == "__main__":
    main()
