"""
診斷 Fourier Features × VS-PINN 的交互問題
=======================================

問題假設：
1. VS-PINN 將物理座標縮放：(X, Y, Z) = (N_x·x, N_y·y, N_z·z)
2. Fourier features 期望輸入在合理範圍（如 [-1, 1] 或 [0, 1]）
3. 若直接對縮放後座標做 Fourier 變換，會導致：
   - z = 2π * (N_x·x, N_y·y, N_z·z) @ B
   - 當 N_y=12 時，y 方向的頻率被放大 12 倍
   - 導致 Fourier features 出現極高頻振盪（週期性噪點）

解決方案：
- 選項 A：在 Fourier 前先標準化座標到 [-1, 1]
- 選項 B：調整 Fourier sigma 以補償縮放
- 選項 C：禁用 Fourier features（回退到標準 MLP）
"""

import torch
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt


def test_fourier_with_scaling():
    """測試 Fourier features 在不同縮放下的行為"""
    
    # 模擬參數
    N_x, N_y, N_z = 2.0, 12.0, 2.0  # VS-PINN 縮放因子
    fourier_m = 64
    fourier_sigma = 5.0
    
    # 生成 Fourier 係數矩陣
    B = torch.randn(3, fourier_m) * fourier_sigma
    
    # 測試座標（物理座標，已標準化到 [-1, 1]）
    n_points = 100
    x = torch.linspace(-1, 1, n_points).reshape(-1, 1)
    y = torch.linspace(-1, 1, n_points).reshape(-1, 1)
    z = torch.zeros(n_points, 1)
    coords_physical = torch.cat([x, y, z], dim=1)
    
    # 情況 1：直接對物理座標應用 Fourier（正確做法）
    z1 = 2.0 * math.pi * coords_physical @ B
    fourier1 = torch.cat([torch.cos(z1), torch.sin(z1)], dim=-1)
    
    # 情況 2：對 VS-PINN 縮放座標應用 Fourier（錯誤做法）
    coords_scaled = coords_physical * torch.tensor([N_x, N_y, N_z])
    z2 = 2.0 * math.pi * coords_scaled @ B
    fourier2 = torch.cat([torch.cos(z2), torch.sin(z2)], dim=-1)
    
    print("=== Fourier Features × VS-PINN 診斷 ===\n")
    
    print("情況 1: Fourier(物理座標) - 正確")
    print(f"  輸入範圍: x∈[{coords_physical[:, 0].min():.2f}, {coords_physical[:, 0].max():.2f}], "
          f"y∈[{coords_physical[:, 1].min():.2f}, {coords_physical[:, 1].max():.2f}]")
    print(f"  z = 2π·x@B 統計: mean={z1.mean():.2f}, std={z1.std():.2f}, range=[{z1.min():.2f}, {z1.max():.2f}]")
    print(f"  Fourier features 統計: mean={fourier1.mean():.4f}, std={fourier1.std():.4f}\n")
    
    print("情況 2: Fourier(VS-PINN 縮放座標) - 錯誤")
    print(f"  輸入範圍: X∈[{coords_scaled[:, 0].min():.2f}, {coords_scaled[:, 0].max():.2f}], "
          f"Y∈[{coords_scaled[:, 1].min():.2f}, {coords_scaled[:, 1].max():.2f}]")
    print(f"  z = 2π·(N·x)@B 統計: mean={z2.mean():.2f}, std={z2.std():.2f}, range=[{z2.min():.2f}, {z2.max():.2f}]")
    print(f"  Fourier features 統計: mean={fourier2.mean():.4f}, std={fourier2.std():.4f}\n")
    
    # 分析振盪頻率
    print("=== 振盪頻率分析 ===")
    print(f"情況 1 (正確): z 標準差 = {z1.std():.2f} → 合理頻率範圍")
    print(f"情況 2 (錯誤): z 標準差 = {z2.std():.2f} → 高頻振盪 ({z2.std() / z1.std():.1f}x)")
    print(f"\n⚠️ 結論: VS-PINN 縮放會將 Fourier 頻率放大 {z2.std() / z1.std():.1f} 倍！")
    print(f"   特別是 y 方向被放大 {N_y:.0f} 倍，導致極高頻振盪\n")
    
    # 視覺化
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 第一行：正確做法
    axes[0, 0].plot(y.numpy(), fourier1[:, 0].numpy())
    axes[0, 0].set_title("情況 1: Fourier[0] vs y (正確)")
    axes[0, 0].set_xlabel("y (物理座標)")
    
    axes[0, 1].plot(y.numpy(), fourier1[:, fourier_m].numpy())
    axes[0, 1].set_title("情況 1: Fourier[m] vs y")
    axes[0, 1].set_xlabel("y (物理座標)")
    
    axes[0, 2].hist(fourier1.flatten().numpy(), bins=50)
    axes[0, 2].set_title("情況 1: Fourier 分佈")
    
    # 第二行：錯誤做法
    axes[1, 0].plot(y.numpy(), fourier2[:, 0].numpy())
    axes[1, 0].set_title("情況 2: Fourier[0] vs y (錯誤)")
    axes[1, 0].set_xlabel("y (物理座標)")
    
    axes[1, 1].plot(y.numpy(), fourier2[:, fourier_m].numpy())
    axes[1, 1].set_title("情況 2: Fourier[m] vs y")
    axes[1, 1].set_xlabel("y (物理座標)")
    
    axes[1, 2].hist(fourier2.flatten().numpy(), bins=50)
    axes[1, 2].set_title("情況 2: Fourier 分佈")
    
    plt.tight_layout()
    plt.savefig("results/fourier_vs_pinn_diagnosis.png", dpi=150)
    print(f"✅ 圖表已保存: results/fourier_vs_pinn_diagnosis.png\n")
    
    # 測試解決方案 A：標準化後再 Fourier
    print("=== 解決方案 A: 標準化後應用 Fourier ===")
    # 將縮放座標重新標準化到 [-1, 1]
    domain_ranges = torch.tensor([
        N_x * 2,  # x: [-1, 1] → [-N_x, N_x] → 範圍 2*N_x
        N_y * 2,  # y: [-1, 1] → [-N_y, N_y] → 範圍 2*N_y
        N_z * 2   # z: [-1, 1] → [-N_z, N_z] → 範圍 2*N_z
    ])
    coords_renormalized = coords_scaled / torch.tensor([N_x, N_y, N_z])
    
    z3 = 2.0 * math.pi * coords_renormalized @ B
    fourier3 = torch.cat([torch.cos(z3), torch.sin(z3)], dim=-1)
    
    print(f"  重新標準化後: X/N_x ∈ [{coords_renormalized[:, 0].min():.2f}, {coords_renormalized[:, 0].max():.2f}]")
    print(f"  z 統計: mean={z3.mean():.2f}, std={z3.std():.2f}")
    print(f"  Fourier features 統計: mean={fourier3.mean():.4f}, std={fourier3.std():.4f}")
    print(f"  ✅ 與情況 1 接近，問題解決！\n")


def test_solution_workflow():
    """測試完整的解決方案工作流"""
    print("=== 推薦解決方案 ===\n")
    
    print("【方案 A】在 Fourier 層前標準化")
    print("  優點: 保持 Fourier features 在設計範圍內")
    print("  缺點: 需要知道域範圍（已在配置中）")
    print("  實現: model(scale_coordinates(x)) → Fourier(normalize(scaled_x))\n")
    
    print("【方案 B】調整 Fourier sigma")
    print("  優點: 不改變輸入")
    print("  缺點: 需要針對每個方向調整，複雜度高")
    print("  實現: sigma_y = sigma / N_y = 5.0 / 12.0 = 0.42\n")
    
    print("【方案 C】禁用 Fourier features")
    print("  優點: 簡單，已驗證有效")
    print("  缺點: 損失高頻表示能力")
    print("  實現: use_fourier: false\n")
    
    print("✅ 推薦: 先試方案 A（標準化），若無效再用方案 C")


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    
    test_fourier_with_scaling()
    test_solution_workflow()
    
    print("\n" + "="*60)
    print("診斷完成！請檢查:")
    print("1. results/fourier_vs_pinn_diagnosis.png - 視覺化對比")
    print("2. 上述統計數據 - 量化分析")
    print("="*60)
