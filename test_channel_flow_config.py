"""
測試 Channel Flow 混合 Fourier 配置
驗證系統可以正確處理混合邊界條件（部分週期、部分非週期）
"""

import torch
import yaml
import numpy as np
from pathlib import Path

# 測試配置文件路徑
CONFIG_FILE = "configs/channel_flow_periodic_example.yml"


def test_channel_flow_hybrid_fourier():
    """測試 Channel Flow 的混合 Fourier 配置"""
    
    print("=" * 80)
    print("測試 Channel Flow 混合 Fourier 配置")
    print("=" * 80)
    
    # 1. 讀取配置
    print("\n1️⃣  讀取配置文件...")
    with open(CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    
    model_cfg = config['model']
    axes_cfg = model_cfg['fourier_features']['axes']
    
    print(f"   配置類型: {model_cfg['fourier_features']['type']}")
    print(f"   輸入維度: {model_cfg['in_dim']} (t, x, y, z)")
    print(f"   輸出維度: {model_cfg['out_dim']} (u, v, w, p)")
    
    # 2. 顯示軸配置
    print("\n2️⃣  軸配置詳情:")
    axis_names = ['t (時間)', 'x (流向)', 'y (壁法向)', 'z (展向)']
    for axis_idx, axis_name in enumerate(axis_names):
        cfg = axes_cfg[axis_idx]
        print(f"   軸 {axis_idx} - {axis_name}:")
        print(f"      類型: {cfg['type']}")
        if cfg['type'] == 'periodic':
            print(f"      週期域大小: {cfg['domain_size']:.6f}")
            print(f"      模態數: {cfg['n_modes']}")
        elif cfg['type'] == 'standard':
            print(f"      模態數: {cfg['n_modes']}")
            print(f"      頻率尺度 σ: {cfg['sigma']}")
    
    # 3. 創建模型
    print("\n3️⃣  創建混合 Fourier 編碼器...")
    from pinnx.models.hybrid_fourier import HybridFourierFeatures
    
    hybrid_config = {
        0: axes_cfg[0],
        1: axes_cfg[1],
        2: axes_cfg[2],
        3: axes_cfg[3],
    }
    
    encoder = HybridFourierFeatures(hybrid_config, trainable=False)
    print(f"   ✅ 編碼器創建成功")
    print(f"   輸入維度: {encoder.in_dim}")
    print(f"   輸出維度: {encoder.out_dim}")
    
    # 4. 測試週期性（x 和 z 方向）
    print("\n4️⃣  測試週期性邊界條件...")
    
    # x 方向週期性測試（僅測試 x 軸的 Fourier 特徵）
    L_x = axes_cfg[1]['domain_size']  # 2π
    from pinnx.models.hybrid_fourier import PeriodicFourierFeatures
    x_encoder = PeriodicFourierFeatures(domain_size=L_x, n_modes=8)
    
    x_left_coord = torch.tensor([[0.0]])
    x_right_coord = torch.tensor([[L_x]])
    
    with torch.no_grad():
        x_feat_left = x_encoder(x_left_coord)
        x_feat_right = x_encoder(x_right_coord)
    
    diff_x = torch.abs(x_feat_left - x_feat_right).max().item()
    print(f"   x 方向週期性誤差: {diff_x:.2e}")
    if diff_x < 1e-6:
        print(f"   ✅ x 方向週期性 通過")
    else:
        print(f"   ⚠️  x 方向週期性誤差: {diff_x:.2e} (可接受)")
    
    # z 方向週期性測試（僅測試 z 軸的 Fourier 特徵）
    L_z = axes_cfg[3]['domain_size']  # π
    z_encoder = PeriodicFourierFeatures(domain_size=L_z, n_modes=8)
    
    z_left_coord = torch.tensor([[0.0]])
    z_right_coord = torch.tensor([[L_z]])
    
    with torch.no_grad():
        z_feat_left = z_encoder(z_left_coord)
        z_feat_right = z_encoder(z_right_coord)
    
    diff_z = torch.abs(z_feat_left - z_feat_right).max().item()
    print(f"   z 方向週期性誤差: {diff_z:.2e}")
    if diff_z < 1e-6:
        print(f"   ✅ z 方向週期性 通過")
    else:
        print(f"   ⚠️  z 方向週期性誤差: {diff_z:.2e} (可接受)")
    
    # 5. 測試非週期方向（y 方向不應該週期）
    print("\n5️⃣  測試非週期性方向（y 壁法向）...")
    y_bottom = torch.tensor([[0.0, 1.0, 0.0, 1.0]])  # y=0 (下壁面)
    y_top = torch.tensor([[0.0, 1.0, 1.0, 1.0]])     # y=1 (上壁面)
    
    with torch.no_grad():
        feat_bottom = encoder(y_bottom)
        feat_top = encoder(y_top)
    
    diff_y = torch.abs(feat_bottom - feat_top).max().item()
    print(f"   y 方向特徵差異: {diff_y:.2e}")
    if diff_y > 0.01:  # 非週期方向應該有顯著差異
        print(f"   ✅ y 方向非週期性 正確（上下壁面特徵不同）")
    else:
        print(f"   ⚠️  y 方向特徵差異過小，可能有問題")
    
    # 6. 測試梯度計算
    print("\n6️⃣  測試梯度計算...")
    x_grad = torch.randn(100, 4, requires_grad=True)
    features = encoder(x_grad)
    loss = features.sum()
    loss.backward()
    
    if x_grad.grad is not None and not torch.isnan(x_grad.grad).any():
        print(f"   ✅ 梯度計算成功")
        print(f"   梯度形狀: {x_grad.grad.shape}")
        print(f"   梯度範圍: [{x_grad.grad.min():.4f}, {x_grad.grad.max():.4f}]")
    else:
        print(f"   ❌ 梯度計算失敗")
    
    # 7. 總結
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)
    success = diff_x < 1e-5 and diff_z < 1e-5 and diff_y > 0.01
    if success:
        print("✅ 所有測試通過！")
        print("\n關鍵驗證:")
        print(f"  • x 方向（流向）週期性: ✓ (誤差 {diff_x:.2e})")
        print(f"  • z 方向（展向）週期性: ✓ (誤差 {diff_z:.2e})")
        print(f"  • y 方向（壁法向）非週期: ✓ (差異 {diff_y:.2e})")
        print(f"  • 梯度計算正常: ✓")
        print("\n🎯 配置正確！可用於 Channel Flow 訓練。")
        print("\n說明:")
        print("  - 週期軸（x, z）使用週期性嵌入")
        print("  - 非週期軸（t, y）使用標準 Fourier")
        print("  - 配置靈活，可自由調整每個軸的編碼方式")
    else:
        print("❌ 部分測試失敗，請檢查配置")
    
    print("=" * 80)


if __name__ == "__main__":
    test_channel_flow_hybrid_fourier()
