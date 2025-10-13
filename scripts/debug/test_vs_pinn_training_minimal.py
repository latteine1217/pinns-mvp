#!/usr/bin/env python3
"""
最小化 VS-PINN 訓練測試腳本
快速診斷訓練 NaN 問題
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 1. 載入感測點資料
    print("\n=== 載入感測點資料 ===")
    sensor_file = Path("data/jhtdb/channel_flow_re1000/sensors_K30_qr_pivot_3d_true.npz")
    data = np.load(sensor_file, allow_pickle=True)
    
    sensor_points = data['sensor_points']  # (30, 3)
    sensor_data = data['sensor_data'].item()  # dict
    
    print(f"感測點: {sensor_points.shape}")
    print(f"資料欄位: {list(sensor_data.keys())}")
    
    # 2. 標準化座標到 [-1, 1]
    print("\n=== 標準化座標 ===")
    x_min, x_max = 0.0, 25.13
    y_min, y_max = -1.0, 1.0
    z_min, z_max = 0.0, 9.42
    
    def normalize(coord, c_min, c_max):
        return 2.0 * (coord - c_min) / (c_max - c_min) - 1.0
    
    coords_norm = np.zeros_like(sensor_points)
    coords_norm[:, 0] = normalize(sensor_points[:, 0], x_min, x_max)
    coords_norm[:, 1] = normalize(sensor_points[:, 1], y_min, y_max)
    coords_norm[:, 2] = normalize(sensor_points[:, 2], z_min, z_max)
    
    coords_tensor = torch.from_numpy(coords_norm).float().to(device)
    
    u_tensor = torch.from_numpy(sensor_data['u']).float().to(device)
    v_tensor = torch.from_numpy(sensor_data['v']).float().to(device)
    w_tensor = torch.from_numpy(sensor_data['w']).float().to(device)
    p_tensor = torch.from_numpy(sensor_data['p']).float().to(device)
    
    print(f"標準化後座標範圍:")
    print(f"  x: [{coords_norm[:,0].min():.3f}, {coords_norm[:,0].max():.3f}]")
    print(f"  y: [{coords_norm[:,1].min():.3f}, {coords_norm[:,1].max():.3f}]")
    print(f"  z: [{coords_norm[:,2].min():.3f}, {coords_norm[:,2].max():.3f}]")
    print(f"資料範圍:")
    print(f"  u: [{u_tensor.min():.3f}, {u_tensor.max():.3f}]")
    print(f"  v: [{v_tensor.min():.3f}, {v_tensor.max():.3f}]")
    print(f"  w: [{w_tensor.min():.3f}, {w_tensor.max():.3f}]")
    print(f"  p: [{p_tensor.min():.3f}, {p_tensor.max():.3f}]")
    
    # 3. 創建模型
    print("\n=== 創建模型 ===")
    model = PINNNet(
        in_dim=3,
        out_dim=4,
        width=200,
        depth=8,
        activation='sine',
        fourier_m=64,
        fourier_sigma=5.0
    ).to(device)
    
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. 創建物理模組
    print("\n=== 創建物理模組 ===")
    physics = VSPINNChannelFlow(
        scaling_factors={'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0},
        physics_params={
            'nu': 5.0e-5,
            'dP_dx': 0.0025,
            'rho': 1.0
        },
        domain_bounds={
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max)
        }
    ).to(device)
    
    print(f"縮放係數: N_x={physics.N_x}, N_y={physics.N_y}, N_z={physics.N_z}")
    print(f"物理參數: nu={physics.nu}, dP_dx={physics.dP_dx}, rho={physics.rho}")
    
    # 5. 前向傳播測試
    print("\n=== 前向傳播測試 ===")
    model.eval()
    with torch.no_grad():
        output = model(coords_tensor)
    
    print(f"輸出 shape: {output.shape}")
    print(f"輸出範圍:")
    print(f"  u_pred: [{output[:,0].min():.3f}, {output[:,0].max():.3f}]")
    print(f"  v_pred: [{output[:,1].min():.3f}, {output[:,1].max():.3f}]")
    print(f"  w_pred: [{output[:,2].min():.3f}, {output[:,2].max():.3f}]")
    print(f"  p_pred: [{output[:,3].min():.3f}, {output[:,3].max():.3f}]")
    
    # 6. 計算資料損失
    print("\n=== 計算資料損失 ===")
    model.train()
    coords_tensor.requires_grad_(True)
    output = model(coords_tensor)
    
    u_pred = output[:, 0:1]
    v_pred = output[:, 1:2]
    w_pred = output[:, 2:3]
    p_pred = output[:, 3:4]
    
    loss_u = torch.mean((u_pred - u_tensor) ** 2)
    loss_v = torch.mean((v_pred - v_tensor) ** 2)
    loss_w = torch.mean((w_pred - w_tensor) ** 2)
    loss_p = torch.mean((p_pred - p_tensor) ** 2)
    
    print(f"MSE loss_u: {loss_u.item():.6f}")
    print(f"MSE loss_v: {loss_v.item():.6f}")
    print(f"MSE loss_w: {loss_w.item():.6f}")
    print(f"MSE loss_p: {loss_p.item():.6f}")
    
    if torch.isnan(loss_u) or torch.isnan(loss_v) or torch.isnan(loss_w) or torch.isnan(loss_p):
        print("❌ 資料損失出現 NaN!")
        return 1
    
    # 7. 計算 PDE 殘差 (使用 VSPINNChannelFlow 的獨立方法)
    print("\n=== 計算 PDE 殘差 ===")
    
    try:
        # 動量方程殘差
        residuals_mom = physics.compute_momentum_residuals(coords_tensor, output)
        print(f"動量殘差結構: {type(residuals_mom)}")
        if isinstance(residuals_mom, dict):
            for key, val in residuals_mom.items():
                if torch.is_tensor(val):
                    print(f"  {key}: [{val.min():.3e}, {val.max():.3e}], mean={val.mean():.3e}")
                    if torch.isnan(val).any():
                        print(f"    ❌ {key} 包含 NaN!")
                        return 1
        
        # 連續性方程殘差
        residual_cont = physics.compute_continuity_residual(coords_tensor, output)
        print(f"連續性殘差: [{residual_cont.min():.3e}, {residual_cont.max():.3e}], mean={residual_cont.mean():.3e}")
        if torch.isnan(residual_cont).any():
            print("❌ 連續性殘差包含 NaN!")
            return 1
        
        # 週期性邊界損失（可選）
        try:
            periodic_loss = physics.compute_periodic_loss(coords_tensor, output)
            print(f"週期性損失: {periodic_loss.item():.3e}")
            if torch.isnan(periodic_loss):
                print("❌ 週期性損失為 NaN!")
                return 1
        except Exception as e:
            print(f"⚠️ 週期性損失計算失敗: {e}")
                
    except Exception as e:
        print(f"⚠️ 物理損失計算失敗: {e}")
        print("  這可能是 NaN 的來源！")
        import traceback
        traceback.print_exc()
        return 1
    
    # 8. 簡單訓練迭代
    print("\n=== 訓練測試 (10 步) ===")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(10):
        optimizer.zero_grad()
        
        coords_tensor.requires_grad_(True)
        output = model(coords_tensor)
        u_pred = output[:, 0:1]
        v_pred = output[:, 1:2]
        w_pred = output[:, 2:3]
        p_pred = output[:, 3:4]
        
        # 只用資料損失（避免物理損失複雜性）
        loss_data = (
            torch.mean((u_pred - u_tensor) ** 2) +
            torch.mean((v_pred - v_tensor) ** 2) +
            torch.mean((w_pred - w_tensor) ** 2) +
            torch.mean((p_pred - p_tensor) ** 2)
        )
        
        loss_data.backward()
        optimizer.step()
        
        if step % 2 == 0:
            print(f"Step {step}: loss={loss_data.item():.6f}")
        
        if torch.isnan(loss_data):
            print(f"❌ 訓練在 step {step} 出現 NaN!")
            return 1
    
    print("\n✅ 所有測試通過！")
    print("\n診斷結論:")
    print("  - 模型本身、資料損失、基礎訓練都正常")
    print("  - NaN 來源應該在 VSPINNChannelFlow 的物理損失計算")
    print("  - 請檢查步驟 7 的輸出，看物理殘差是否正常")
    return 0

if __name__ == "__main__":
    sys.exit(main())
