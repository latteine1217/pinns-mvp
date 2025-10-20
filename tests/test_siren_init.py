#!/usr/bin/env python3
"""
測試 SIREN 初始化是否正確應用
"""

import sys
from pathlib import Path
import torch
import numpy as np

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models import PINNNet, init_siren_weights

def test_standalone_siren_initialization():
    """測試 SIREN 權重初始化"""
    print("=" * 80)
    print("🧪 測試 SIREN 權重初始化")
    print("=" * 80)
    
    # 創建 Sine 激活的模型
    model = PINNNet(
        in_dim=2,
        out_dim=3,
        width=256,
        depth=6,
        activation='sine',
        sine_omega_0=1.0,
        use_fourier=True,
        fourier_m=48,
        fourier_sigma=3.0
    )
    
    print(f"\n✅ 模型創建成功")
    print(f"   - 激活函數: sine")
    print(f"   - ω₀: 1.0")
    print(f"   - 網路寬度: 256")
    print(f"   - 網路深度: 6")
    
    # 檢查初始化前的權重分佈
    print("\n📊 應用 SIREN 初始化前的權重分佈:")
    first_layer_weight = model.hidden_layers[0].linear.weight.data  # type: ignore
    w_min: float = first_layer_weight.min().item()  # type: ignore
    w_max: float = first_layer_weight.max().item()  # type: ignore
    w_std: float = first_layer_weight.std().item()  # type: ignore
    print(f"   第一層權重範圍: [{w_min:.6f}, {w_max:.6f}]")
    print(f"   第一層權重標準差: {w_std:.6f}")
    
    # 應用 SIREN 初始化
    print("\n🔧 應用 SIREN 初始化...")
    init_siren_weights(model)
    
    # 檢查初始化後的權重分佈
    print("\n📊 應用 SIREN 初始化後的權重分佈:")
    first_layer_after = model.hidden_layers[0].linear.weight.data  # type: ignore
    n_in = first_layer_after.shape[1]  # type: ignore
    expected_bound = 1.0 / n_in
    
    print(f"   第一層權重範圍: [{first_layer_after.min().item():.6f}, {first_layer_after.max().item():.6f}]")  # type: ignore
    print(f"   第一層權重標準差: {first_layer_after.std().item():.6f}")  # type: ignore
    print(f"   理論邊界: ±{expected_bound:.6f}")
    
    # 驗證權重是否在預期範圍內
    within_bounds = (first_layer_after >= -expected_bound * 1.01) & (first_layer_after <= expected_bound * 1.01)  # type: ignore
    print(f"   ✅ {within_bounds.float().mean().item() * 100:.1f}% 權重在預期範圍內")  # type: ignore
    
    # 測試前向傳播
    print("\n🔬 測試前向傳播:")
    x_test = torch.randn(100, 2)
    with torch.no_grad():
        output = model(x_test)
    
    print(f"   輸入形狀: {x_test.shape}")
    print(f"   輸出形狀: {output.shape}")
    print(f"   輸出範圍: [{output.min():.6f}, {output.max():.6f}]")
    print(f"   輸出是否包含 NaN: {'❌ 是' if torch.isnan(output).any() else '✅ 否'}")
    print(f"   輸出是否包含 Inf: {'❌ 是' if torch.isinf(output).any() else '✅ 否'}")
    
    # 測試梯度計算
    print("\n🔬 測試梯度計算:")
    x_test.requires_grad_(True)
    output = model(x_test)
    loss = output.sum()
    loss.backward()
    
    grad = x_test.grad
    if grad is not None:
        print(f"   梯度是否包含 NaN: {'❌ 是' if torch.isnan(grad).any() else '✅ 否'}")
        print(f"   梯度是否包含 Inf: {'❌ 是' if torch.isinf(grad).any() else '✅ 否'}")
        print(f"   梯度範圍: [{grad.min().item():.6f}, {grad.max().item():.6f}]")
    else:
        print("   ⚠️  梯度為 None")
    
    print("\n" + "=" * 80)
    print("✅ SIREN 初始化測試完成")
    print("=" * 80)

if __name__ == "__main__":
    test_standalone_siren_initialization()
