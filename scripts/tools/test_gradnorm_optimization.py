#!/usr/bin/env python3
"""
GradNorm 優化驗證測試

功能：
1. 驗證優化後的 GradNorm 梯度計算數值正確性
2. 比較優化前後的梯度範數誤差
3. 測量效能提升

使用方式：
    python scripts/tools/test_gradnorm_optimization.py
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
import time
from typing import Dict

# 添加專案根目錄到 path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pinnx.losses.weighting import GradNormWeighter


class SimpleModel(nn.Module):
    """簡單的測試模型"""
    def __init__(self, input_dim=10, hidden_dim=50, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.fc3(x)


def compute_gradients_original(
    losses: Dict[str, torch.Tensor],
    model: nn.Module,
    loss_names: list,
    device: torch.device,
    eps: float = 1e-8
) -> Dict[str, torch.Tensor]:
    """
    原始版本的 GradNorm 梯度計算（用於對照）
    
    這是優化前的版本，會對每個損失分別調用 autograd.grad
    """
    gradients = {}
    for name, loss in losses.items():
        if name not in loss_names:
            continue
        
        if not loss.requires_grad or abs(float(loss.detach())) < eps:
            gradients[name] = torch.tensor(eps, device=device)
            continue
            
        try:
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=list(model.parameters()),  # 每次都創建新列表
                grad_outputs=torch.ones_like(loss),
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )
            
            grad_norm = torch.tensor(0.0, device=device)
            for grad in grads:
                if grad is not None:
                    grad_norm += (grad.detach() ** 2).sum()  # 使用 ** 2
            
            gradients[name] = torch.sqrt(grad_norm + eps)
            
        except Exception as e:
            gradients[name] = torch.tensor(eps, device=device)
            print(f"Warning: Gradient computation failed for {name}: {e}")
    
    return gradients


def test_numerical_correctness():
    """測試數值正確性"""
    print("=" * 60)
    print("測試 1: 數值正確性驗證")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    
    # 創建測試損失
    loss_names = ['loss_a', 'loss_b', 'loss_c', 'loss_d']
    x = torch.randn(100, 10, device=device, requires_grad=True)
    output = model(x)
    
    losses = {
        'loss_a': output[:, 0].mean(),
        'loss_b': output[:, 1].pow(2).mean(),
        'loss_c': output[:, 2:4].sum(dim=1).mean(),
        'loss_d': torch.nn.functional.mse_loss(output[:, 4], torch.zeros(100, device=device))
    }
    
    # 確保所有損失都需要梯度
    for loss in losses.values():
        assert loss.requires_grad, "Loss must require gradients"
    
    # 計算原始版本的梯度
    grads_original = compute_gradients_original(
        losses, model, loss_names, device
    )
    
    # 計算優化版本的梯度
    weighter = GradNormWeighter(
        model=model,
        loss_names=loss_names,
        device=device
    )
    grads_optimized = weighter.compute_gradients(losses)
    
    # 比較結果
    print(f"\n梯度範數比較:")
    print(f"{'Loss Name':<15} {'Original':<15} {'Optimized':<15} {'Rel Error':<15} {'Status'}")
    print("-" * 80)
    
    max_error = 0.0
    all_passed = True
    
    for name in loss_names:
        orig = float(grads_original[name])
        opt = float(grads_optimized[name])
        rel_error = abs(orig - opt) / (abs(orig) + 1e-8)
        max_error = max(max_error, rel_error)
        
        status = "✓ PASS" if rel_error < 1e-5 else "✗ FAIL"
        if rel_error >= 1e-5:
            all_passed = False
        
        print(f"{name:<15} {orig:<15.6e} {opt:<15.6e} {rel_error:<15.6e} {status}")
    
    print("-" * 80)
    print(f"最大相對誤差: {max_error:.6e}")
    print(f"測試結果: {'✓ 全部通過' if all_passed else '✗ 存在失敗'}\n")
    
    return all_passed


def test_performance():
    """測試效能提升"""
    print("=" * 60)
    print("測試 2: 效能測試")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(input_dim=50, hidden_dim=256, output_dim=10).to(device)
    
    # 創建更多損失項（模擬真實場景）
    loss_names = [
        'data', 'momentum_x', 'momentum_y', 'momentum_z',
        'continuity', 'wall_constraint', 'periodic_x', 'periodic_z'
    ]
    
    x = torch.randn(500, 50, device=device, requires_grad=True)
    output = model(x)
    
    losses = {
        'data': output[:, 0].mean(),
        'momentum_x': output[:, 1].pow(2).mean(),
        'momentum_y': output[:, 2].pow(2).mean(),
        'momentum_z': output[:, 3].pow(2).mean(),
        'continuity': output[:, 4:7].sum(dim=1).mean(),
        'wall_constraint': output[:, 7].abs().mean(),
        'periodic_x': (output[:, 8] - output[:, 9]).pow(2).mean(),
        'periodic_z': output[:, 0:5].std(dim=1).mean()
    }
    
    # 熱身
    for _ in range(3):
        _ = compute_gradients_original(losses, model, loss_names, device)
    
    # 測試原始版本
    n_runs = 20
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    for _ in range(n_runs):
        _ = compute_gradients_original(losses, model, loss_names, device)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    time_original = (time.perf_counter() - start) / n_runs
    
    # 測試優化版本
    weighter = GradNormWeighter(model=model, loss_names=loss_names, device=device)
    
    for _ in range(3):
        _ = weighter.compute_gradients(losses)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    
    for _ in range(n_runs):
        _ = weighter.compute_gradients(losses)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    time_optimized = (time.perf_counter() - start) / n_runs
    
    # 結果
    speedup = time_original / time_optimized
    improvement = (1 - time_optimized / time_original) * 100
    
    print(f"\n效能測試結果 ({n_runs} 次運行平均):")
    print(f"  - 原始版本:   {time_original*1000:.2f} ms")
    print(f"  - 優化版本:   {time_optimized*1000:.2f} ms")
    print(f"  - 加速比:     {speedup:.2f}x")
    print(f"  - 效能提升:   {improvement:.1f}%")
    print(f"  - 節省時間:   {(time_original - time_optimized)*1000:.2f} ms\n")
    
    return speedup > 1.05  # 至少 5% 提升


def test_edge_cases():
    """測試邊界情況"""
    print("=" * 60)
    print("測試 3: 邊界情況測試")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)
    loss_names = ['loss_a', 'loss_b', 'loss_c']
    
    weighter = GradNormWeighter(model=model, loss_names=loss_names, device=device)
    
    # 測試案例
    test_cases = []
    
    # Case 1: 包含零損失
    x = torch.randn(10, 10, device=device, requires_grad=True)
    output = model(x)
    losses_zero = {
        'loss_a': output[:, 0].mean(),
        'loss_b': torch.tensor(0.0, device=device, requires_grad=True),
        'loss_c': output[:, 2].pow(2).mean()
    }
    test_cases.append(("零損失", losses_zero))
    
    # Case 2: 包含極小損失
    losses_tiny = {
        'loss_a': torch.tensor(1e-10, device=device, requires_grad=True),
        'loss_b': output[:, 1].mean(),
        'loss_c': output[:, 2].mean()
    }
    test_cases.append(("極小損失", losses_tiny))
    
    # Case 3: 只有一個損失
    losses_single = {
        'loss_a': output[:, 0].mean()
    }
    test_cases.append(("單一損失", losses_single))
    
    # 執行測試
    print(f"\n邊界情況測試:")
    all_passed = True
    
    for case_name, losses in test_cases:
        try:
            grads = weighter.compute_gradients(losses)
            print(f"  ✓ {case_name}: 通過 (計算了 {len(grads)} 個梯度)")
        except Exception as e:
            print(f"  ✗ {case_name}: 失敗 - {e}")
            all_passed = False
    
    print(f"\n測試結果: {'✓ 全部通過' if all_passed else '✗ 存在失敗'}\n")
    return all_passed


def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "GradNorm 優化驗證測試" + " " * 26 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # 執行所有測試
    results = []
    
    try:
        results.append(("數值正確性", test_numerical_correctness()))
    except Exception as e:
        print(f"❌ 數值正確性測試失敗: {e}\n")
        results.append(("數值正確性", False))
    
    try:
        results.append(("效能提升", test_performance()))
    except Exception as e:
        print(f"❌ 效能測試失敗: {e}\n")
        results.append(("效能提升", False))
    
    try:
        results.append(("邊界情況", test_edge_cases()))
    except Exception as e:
        print(f"❌ 邊界情況測試失敗: {e}\n")
        results.append(("邊界情況", False))
    
    # 總結
    print("=" * 60)
    print("最終結果總結")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20} {status}")
    
    all_passed = all(passed for _, passed in results)
    print("=" * 60)
    print(f"總體結果: {'✓ 全部測試通過' if all_passed else '✗ 部分測試失敗'}")
    print("=" * 60)
    print()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
