#!/usr/bin/env python3
"""
逐步測試 VS-PINN 額外約束項以觸發 NaN
Task-10 NaN 診斷 - 步驟 2
"""

import sys
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.models.fourier_mlp import PINNNet
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow

def test_constraint(name, func, *args):
    """測試單一約束項是否產生 NaN"""
    print(f"\n{'='*60}")
    print(f"測試約束: {name}")
    print('='*60)
    
    try:
        result = func(*args)
        
        # 處理字典類型返回值
        if isinstance(result, dict):
            print(f"返回值類型: dict，包含 {len(result)} 個項目")
            all_valid = True
            for key, val in result.items():
                if torch.is_tensor(val):
                    is_nan = torch.isnan(val).any()
                    print(f"  {key}: {val.item():.6e} {'❌ NaN!' if is_nan else '✅ OK'}")
                    if is_nan:
                        all_valid = False
                else:
                    print(f"  {key}: {val} (非張量)")
            return all_valid
        
        # 處理張量返回值
        elif torch.is_tensor(result):
            is_nan = torch.isnan(result).any()
            print(f"返回值: {result.item():.6e} {'❌ NaN!' if is_nan else '✅ OK'}")
            return not is_nan
        
        else:
            print(f"⚠️ 未預期的返回類型: {type(result)}")
            return False
            
    except Exception as e:
        print(f"❌ 約束計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


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
    
    # 2. 標準化座標到 [-1, 1]
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
    coords_tensor.requires_grad_(True)  # 重要：梯度計算需要
    
    # 額外準備物理座標（用於某些需要物理座標的約束）
    coords_physical = torch.from_numpy(sensor_points).float().to(device)
    coords_physical.requires_grad_(True)
    
    print(f"感測點 y 座標範圍: [{sensor_points[:,1].min():.3f}, {sensor_points[:,1].max():.3f}]")
    print(f"是否包含中心線點 (y=0): {np.any(np.abs(sensor_points[:,1]) < 1e-6)}")
    print(f"是否包含壁面點 (y=±1): {np.any(np.abs(sensor_points[:,1] - 1.0) < 1e-6) or np.any(np.abs(sensor_points[:,1] + 1.0) < 1e-6)}")
    
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
        },
        normalize_losses=True,  # 啟用損失歸一化
        warmup_epochs=10
    ).to(device)
    
    print(f"損失歸一化: {physics.normalize_losses}")
    print(f"Warmup epochs: {physics.warmup_epochs}")
    
    # 5. 前向傳播
    print("\n=== 前向傳播 ===")
    model.train()
    output = model(coords_tensor)
    
    print(f"輸出範圍:")
    print(f"  u: [{output[:,0].min():.3f}, {output[:,0].max():.3f}]")
    print(f"  v: [{output[:,1].min():.3f}, {output[:,1].max():.3f}]")
    print(f"  w: [{output[:,2].min():.3f}, {output[:,2].max():.3f}]")
    print(f"  p: [{output[:,3].min():.3f}, {output[:,3].max():.3f}]")
    
    # ========================================
    # 6. 逐項測試額外約束
    # ========================================
    
    results = {}
    
    # 測試 1: Bulk Velocity Constraint
    if hasattr(physics, 'compute_bulk_velocity_constraint'):
        results['bulk_velocity'] = test_constraint(
            "Bulk Velocity Constraint (流量約束)",
            physics.compute_bulk_velocity_constraint,
            coords_physical,  # 需要物理座標
            output,
            1.0  # target_bulk_velocity
        )
    
    # 測試 2: Centerline Symmetry
    if hasattr(physics, 'compute_centerline_symmetry'):
        results['centerline_symmetry'] = test_constraint(
            "Centerline Symmetry (中心線對稱)",
            physics.compute_centerline_symmetry,
            coords_physical,  # 需要物理座標
            output
        )
    
    # 測試 3: Pressure Reference
    if hasattr(physics, 'compute_pressure_reference'):
        results['pressure_reference'] = test_constraint(
            "Pressure Reference (壓力參考點)",
            physics.compute_pressure_reference,
            coords_physical,  # 需要物理座標
            output,
            None  # 使用預設參考點
        )
    
    # 測試 4: 損失歸一化（Epoch 0）
    print(f"\n{'='*60}")
    print("測試損失歸一化 (Epoch 0 - Warmup 階段)")
    print('='*60)
    
    loss_dict_raw = {
        'data': torch.tensor(3200.0, device=device),
        'momentum_x': torch.tensor(2.5, device=device),
        'momentum_y': torch.tensor(1.8, device=device),
        'continuity': torch.tensor(3.1, device=device),
        'wall_constraint': torch.tensor(0.5, device=device),
        'bulk_velocity': torch.tensor(0.8, device=device),
        'centerline_dudy': torch.tensor(1.2, device=device),
        'centerline_v': torch.tensor(0.3, device=device),
    }
    
    print("原始損失:")
    for k, v in loss_dict_raw.items():
        print(f"  {k}: {v.item():.6f}")
    
    # 執行歸一化（模擬 Epoch 0）
    normalized_dict = physics.normalize_loss_dict(loss_dict_raw, epoch=0)
    
    print("\n歸一化後損失 (Epoch 0 - 應該返回原值):")
    all_valid = True
    for k, v in normalized_dict.items():
        is_nan = torch.isnan(v).any()
        print(f"  {k}: {v.item():.6f} {'❌ NaN!' if is_nan else '✅ OK'}")
        if is_nan:
            all_valid = False
    
    results['normalization_epoch0'] = all_valid
    
    # 測試 5: 損失歸一化（Epoch 10 - 訓練階段）
    print(f"\n{'='*60}")
    print("測試損失歸一化 (Epoch 10 - 訓練階段)")
    print('='*60)
    
    # 模擬 warmup 完成，執行真正歸一化
    normalized_dict_ep10 = physics.normalize_loss_dict(loss_dict_raw, epoch=10)
    
    print("歸一化後損失 (Epoch 10):")
    all_valid = True
    for k, v in normalized_dict_ep10.items():
        is_nan = torch.isnan(v).any()
        normalizer = physics.loss_normalizers.get(k, 1.0)
        print(f"  {k}: {v.item():.6f} (normalizer={normalizer:.6f}) {'❌ NaN!' if is_nan else '✅ OK'}")
        if is_nan:
            all_valid = False
    
    results['normalization_epoch10'] = all_valid
    
    # ========================================
    # 7. 總結結果
    # ========================================
    
    print(f"\n{'='*60}")
    print("測試總結")
    print('='*60)
    
    all_passed = True
    for name, passed in results.items():
        status = '✅ PASS' if passed else '❌ FAIL'
        print(f"{name:30s} : {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ 所有約束測試通過，NaN 來源仍未找到")
        print("   建議：檢查完整訓練循環中的其他邏輯")
        return 0
    else:
        print("\n❌ 發現 NaN 來源！請檢查上方失敗的約束項")
        return 1


if __name__ == "__main__":
    sys.exit(main())
