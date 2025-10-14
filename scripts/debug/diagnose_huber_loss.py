#!/usr/bin/env python3
"""
診斷 Huber 損失在 Phase 6C 中的數值行為

Purpose:
    分析 PyTorch smooth_l1_loss 在實際 ν_t/ν 分布下的數值表現
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

def compute_huber_loss(nu_t_ratio, target, beta):
    """手動計算 Huber 損失"""
    diff = torch.abs(nu_t_ratio - target)
    mask = diff < beta
    
    # 平方區
    loss_quad = 0.5 * (nu_t_ratio - target)**2 / beta
    # 線性區
    loss_linear = diff - 0.5 * beta
    
    loss = torch.where(mask, loss_quad, loss_linear)
    return loss, mask

def main():
    print("=" * 80)
    print("🔍 Huber 損失診斷：Phase 6C-v1 vs Phase 6C-v2")
    print("=" * 80)
    
    # 模擬實際的 ν_t/ν 分布（根據訓練日誌）
    # Phase 6C-v2 @ Epoch 0: ν_t penalty mean=656.5614 (單點)
    # 對應的 ν_t/ν 範圍：需反推
    
    # 測試案例 1：典型值
    target = 100.0
    beta_v1 = 100.0
    beta_v2 = 1000.0
    
    # 模擬 ν_t/ν 範圍 [0, 2000]
    nu_t_ratios = torch.linspace(0, 2000, 1000)
    
    print(f"\n📊 配置：")
    print(f"   Target: {target:.1f}")
    print(f"   Beta (v1): {beta_v1:.1f}")
    print(f"   Beta (v2): {beta_v2:.1f}")
    
    # 計算損失
    loss_v1_pytorch = F.smooth_l1_loss(
        nu_t_ratios, 
        torch.full_like(nu_t_ratios, target), 
        beta=beta_v1, 
        reduction='none'
    )
    
    loss_v2_pytorch = F.smooth_l1_loss(
        nu_t_ratios, 
        torch.full_like(nu_t_ratios, target), 
        beta=beta_v2, 
        reduction='none'
    )
    
    loss_log1p = torch.log1p(torch.relu(nu_t_ratios - target))
    
    # 手動計算驗證
    loss_v2_manual, mask_v2 = compute_huber_loss(nu_t_ratios, target, beta_v2)
    
    print(f"\n✅ 驗證：PyTorch vs 手動計算")
    print(f"   Max diff: {(loss_v2_pytorch - loss_v2_manual).abs().max():.2e}")
    
    # 關鍵點分析
    test_points = [500, 700, 900, 1100, 1200, 1500]
    print(f"\n📊 關鍵點分析（target={target}）：")
    print(f"{'ν_t/ν':>8} | {'β=100 (v1)':>12} | {'β=1000 (v2)':>13} | {'log1p':>10} | {'Ratio v2/v1':>12}")
    print("-" * 80)
    
    for nu_t_val in test_points:
        idx = (nu_t_ratios - nu_t_val).abs().argmin()
        loss_v1 = loss_v1_pytorch[idx].item()
        loss_v2 = loss_v2_pytorch[idx].item()
        loss_log = loss_log1p[idx].item()
        ratio = loss_v2 / loss_v1 if loss_v1 > 0 else float('inf')
        
        print(f"{nu_t_val:8.1f} | {loss_v1:12.2f} | {loss_v2:13.2f} | {loss_log:10.2f} | {ratio:12.2f}")
    
    # 反推：如果單點平均損失 = 656.56，對應的 ν_t/ν 是多少？
    print(f"\n🔍 反推分析：")
    print(f"   目標：找到使 Huber loss ≈ 656.56 的 ν_t/ν 值")
    
    target_loss = 656.56
    # 使用 β=1000
    idx_v2 = (loss_v2_pytorch - target_loss).abs().argmin()
    nu_t_inferred = nu_t_ratios[idx_v2].item()
    
    print(f"   結果：ν_t/ν ≈ {nu_t_inferred:.1f}")
    print(f"   驗證：Huber(β=1000) = {loss_v2_pytorch[idx_v2].item():.2f}")
    print(f"   對應 log1p = {loss_log1p[idx_v2].item():.2f}")
    
    # 繪圖
    plt.figure(figsize=(14, 5))
    
    # 子圖 1：損失曲線
    plt.subplot(1, 2, 1)
    plt.plot(nu_t_ratios.numpy(), loss_v1_pytorch.numpy(), 
             label=f'Huber (β={beta_v1})', linewidth=2, alpha=0.7)
    plt.plot(nu_t_ratios.numpy(), loss_v2_pytorch.numpy(), 
             label=f'Huber (β={beta_v2})', linewidth=2, alpha=0.7)
    plt.plot(nu_t_ratios.numpy(), loss_log1p.numpy(), 
             label='log1p (Phase 6B)', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.axhline(y=656.56, color='red', linestyle=':', label='實際觀測值 (656.56)')
    plt.axvline(x=target, color='green', linestyle=':', alpha=0.5, label='Target (100)')
    
    plt.xlabel('ν_t/ν')
    plt.ylabel('Loss Value')
    plt.title('Penalty Function Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2000)
    plt.ylim(0, 1000)
    
    # 子圖 2：梯度曲線
    plt.subplot(1, 2, 2)
    
    # 數值計算梯度
    spacing = float(nu_t_ratios[1] - nu_t_ratios[0])
    grad_v1 = torch.gradient(loss_v1_pytorch, spacing=(spacing,))[0]
    grad_v2 = torch.gradient(loss_v2_pytorch, spacing=(spacing,))[0]
    grad_log1p = 1.0 / (1.0 + torch.relu(nu_t_ratios - target))
    
    plt.plot(nu_t_ratios.numpy(), grad_v1.numpy(), 
             label=f'∂Huber/∂ν_t (β={beta_v1})', linewidth=2, alpha=0.7)
    plt.plot(nu_t_ratios.numpy(), grad_v2.numpy(), 
             label=f'∂Huber/∂ν_t (β={beta_v2})', linewidth=2, alpha=0.7)
    plt.plot(nu_t_ratios.numpy(), grad_log1p.numpy(), 
             label='∂log1p/∂ν_t', linewidth=2, linestyle='--', alpha=0.7)
    
    plt.axvline(x=target, color='green', linestyle=':', alpha=0.5, label='Target (100)')
    plt.xlabel('ν_t/ν')
    plt.ylabel('Gradient')
    plt.title('Gradient Comparison (Non-saturation Check)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2000)
    plt.ylim(0, 2)
    
    plt.tight_layout()
    output_path = '/Users/latteine/Documents/coding/pinns-mvp/results/debug_huber_loss_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 圖表已保存：{output_path}")
    
    # 總結分析
    print(f"\n" + "=" * 80)
    print("📝 診斷結論：")
    print("=" * 80)
    
    # 計算平均損失（模擬 2048 個採樣點）
    # 假設 ν_t/ν ~ Uniform[500, 1200]（根據訓練日誌推測）
    nu_t_sample = torch.linspace(500, 1200, 2048)
    loss_v2_sample = F.smooth_l1_loss(
        nu_t_sample, 
        torch.full_like(nu_t_sample, target), 
        beta=beta_v2, 
        reduction='mean'
    )
    
    print(f"1. 模擬採樣（ν_t/ν ∈ [500, 1200]，N=2048）：")
    print(f"   - 平均 Huber 損失（β=1000）：{loss_v2_sample.item():.2f}")
    print(f"   - 對應 turbulent_viscosity_loss：{loss_v2_sample.item():.2f}（應與日誌中的 431K 比較）")
    
    print(f"\n2. 問題診斷：")
    if loss_v2_sample.item() > 100:
        print(f"   ⚠️ 即使 β=1000，損失仍過大（>{loss_v2_sample.item():.0f}）")
        print(f"   ❌ 原因：實際 ν_t/ν 遠超 target=100（可能在 700-1100 範圍）")
        print(f"   💡 建議：")
        print(f"      (1) 提升 target: 100 → 500-800（接近實際中心值）")
        print(f"      (2) 或大幅提升 β: 1000 → 5000（使更多區域處於平方區）")
        print(f"      (3) 或降低權重：0.001 → 0.0001（10 倍）")
    else:
        print(f"   ✅ 損失在合理範圍（<100）")
    
    print(f"\n3. 梯度分析：")
    grad_at_1000 = grad_v2[(nu_t_ratios - 1000).abs().argmin()].item()
    grad_log1p_at_1000 = grad_log1p[(nu_t_ratios - 1000).abs().argmin()].item()
    print(f"   - Huber 梯度 @ ν_t/ν=1000（β=1000）：{grad_at_1000:.4f}")
    print(f"   - log1p 梯度 @ ν_t/ν=1000：{grad_log1p_at_1000:.6f}")
    print(f"   - 梯度提升倍數：{grad_at_1000/grad_log1p_at_1000:.1f}×")
    
    if grad_at_1000 > 0.5:
        print(f"   ✅ Huber 梯度未飽和（>{grad_at_1000:.2f}）")
    else:
        print(f"   ⚠️ Huber 梯度可能仍偏小（<{grad_at_1000:.2f}）")

if __name__ == "__main__":
    main()
