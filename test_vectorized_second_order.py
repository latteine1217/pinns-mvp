"""
測試向量化二階梯度計算
======================

比較三種方法：
1. 當前方法（標量化 .sum()）
2. 向量化方法（使用 grad_outputs）
3. 驗證兩者結果是否一致

Author: Performance Optimization Team
Date: 2026-01-13
"""

import torch
import time


def current_method_second_order(first_grad, coords, create_graph=True):
    """
    當前方法：使用 .sum() 標量化
    """
    second_grads = []
    for i in range(2):  # x, y
        full_grad_i = torch.autograd.grad(
            outputs=first_grad[:, i].sum(),  # 標量化
            inputs=coords,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        second_grads.append(full_grad_i[:, i:i+1])
    return torch.cat(second_grads, dim=1)


def vectorized_method_second_order(first_grad, coords, create_graph=True):
    """
    向量化方法：使用 grad_outputs 避免標量化
    """
    second_grads = []
    for i in range(2):  # x, y
        # 使用單位向量作為 grad_outputs
        grad_outputs = torch.zeros_like(first_grad)
        grad_outputs[:, i] = 1.0
        
        full_grad_i = torch.autograd.grad(
            outputs=first_grad,
            inputs=coords,
            grad_outputs=grad_outputs,
            create_graph=create_graph,
            retain_graph=True
        )[0]
        second_grads.append(full_grad_i[:, i:i+1])
    return torch.cat(second_grads, dim=1)


def test_correctness():
    """
    測試正確性：兩種方法是否產生相同結果
    """
    print("=" * 60)
    print("測試 1: 正確性驗證")
    print("=" * 60)
    
    # 設置隨機種子
    torch.manual_seed(42)
    
    # 創建測試數據
    N = 100
    coords = torch.randn(N, 2, requires_grad=True)
    
    # 定義一個簡單的二次函數：f(x, y) = x^2 + 2*y^2
    # 理論上：∂²f/∂x² = 2, ∂²f/∂y² = 4
    f = (coords[:, 0:1] ** 2) + 2 * (coords[:, 1:2] ** 2)  # [N, 1]
    
    # 計算一階梯度
    first_grad = torch.autograd.grad(
        outputs=f,
        inputs=coords,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True
    )[0]  # [N, 2]
    
    print(f"輸入形狀: coords={coords.shape}, f={f.shape}")
    print(f"一階梯度形狀: {first_grad.shape}")
    print(f"一階梯度範例（前3個點）:\n{first_grad[:3]}")
    
    # 方法 1：當前方法
    try:
        second_grad_current = current_method_second_order(first_grad, coords, create_graph=True)
        print(f"\n✅ 當前方法成功")
        print(f"   二階梯度形狀: {second_grad_current.shape}")
        print(f"   ∂²f/∂x² 平均值: {second_grad_current[:, 0].mean().item():.4f} (理論值: 2.0)")
        print(f"   ∂²f/∂y² 平均值: {second_grad_current[:, 1].mean().item():.4f} (理論值: 4.0)")
    except Exception as e:
        print(f"\n❌ 當前方法失敗: {e}")
        return False
    
    # 方法 2：向量化方法
    try:
        # 重新計算一階梯度（因為計算圖已被消耗）
        coords2 = coords.detach().clone().requires_grad_(True)
        f2 = (coords2[:, 0:1] ** 2) + 2 * (coords2[:, 1:2] ** 2)
        first_grad2 = torch.autograd.grad(
            outputs=f2,
            inputs=coords2,
            grad_outputs=torch.ones_like(f2),
            create_graph=True,
            retain_graph=True
        )[0]
        
        second_grad_vectorized = vectorized_method_second_order(first_grad2, coords2, create_graph=True)
        print(f"\n✅ 向量化方法成功")
        print(f"   二階梯度形狀: {second_grad_vectorized.shape}")
        print(f"   ∂²f/∂x² 平均值: {second_grad_vectorized[:, 0].mean().item():.4f} (理論值: 2.0)")
        print(f"   ∂²f/∂y² 平均值: {second_grad_vectorized[:, 1].mean().item():.4f} (理論值: 4.0)")
    except Exception as e:
        print(f"\n❌ 向量化方法失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 比較結果
    diff = (second_grad_current - second_grad_vectorized).abs().max().item()
    print(f"\n📊 結果比較:")
    print(f"   最大差異: {diff:.2e}")
    
    if diff < 1e-5:
        print(f"   ✅ 兩種方法結果一致！")
        return True
    else:
        print(f"   ⚠️  兩種方法結果不同！")
        return False


def test_performance():
    """
    測試性能：比較兩種方法的速度
    """
    print("\n" + "=" * 60)
    print("測試 2: 性能比較")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用設備: {device}")
    
    batch_sizes = [1000, 2000, 4000, 8000]
    
    for N in batch_sizes:
        print(f"\n批次大小: {N}")
        
        # 創建測試數據
        coords = torch.randn(N, 2, device=device, requires_grad=True)
        f = (coords[:, 0:1] ** 2) + 2 * (coords[:, 1:2] ** 2)
        
        # 方法 1：當前方法
        first_grad1 = torch.autograd.grad(
            outputs=f,
            inputs=coords,
            grad_outputs=torch.ones_like(f),
            create_graph=True,
            retain_graph=True
        )[0]
        
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        
        second_grad1 = current_method_second_order(first_grad1, coords, create_graph=True)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        t_current = time.time() - t0
        
        # 方法 2：向量化方法
        coords2 = torch.randn(N, 2, device=device, requires_grad=True)
        f2 = (coords2[:, 0:1] ** 2) + 2 * (coords2[:, 1:2] ** 2)
        first_grad2 = torch.autograd.grad(
            outputs=f2,
            inputs=coords2,
            grad_outputs=torch.ones_like(f2),
            create_graph=True,
            retain_graph=True
        )[0]
        
        torch.cuda.synchronize() if device == 'cuda' else None
        t0 = time.time()
        
        second_grad2 = vectorized_method_second_order(first_grad2, coords2, create_graph=True)
        
        torch.cuda.synchronize() if device == 'cuda' else None
        t_vectorized = time.time() - t0
        
        speedup = t_current / t_vectorized
        print(f"  當前方法:   {t_current*1000:.2f} ms")
        print(f"  向量化方法: {t_vectorized*1000:.2f} ms")
        print(f"  加速比:     {speedup:.2f}×")


def test_backward_pass():
    """
    測試反向傳播：確保向量化方法支持梯度計算
    """
    print("\n" + "=" * 60)
    print("測試 3: 反向傳播支持")
    print("=" * 60)
    
    # 模擬 PINNs loss 計算
    N = 100
    coords = torch.randn(N, 2, requires_grad=True)
    f = (coords[:, 0:1] ** 2) + 2 * (coords[:, 1:2] ** 2)
    
    first_grad = torch.autograd.grad(
        outputs=f,
        inputs=coords,
        grad_outputs=torch.ones_like(f),
        create_graph=True,
        retain_graph=True
    )[0]
    
    # 計算二階梯度
    second_grad = vectorized_method_second_order(first_grad, coords, create_graph=True)
    
    # 模擬 PDE residual: Δf = ∂²f/∂x² + ∂²f/∂y²
    laplacian = second_grad[:, 0] + second_grad[:, 1]  # [N]
    
    # 計算 loss
    loss = (laplacian ** 2).mean()
    
    print(f"Laplacian 形狀: {laplacian.shape}")
    print(f"Loss 值: {loss.item():.4f}")
    
    # 反向傳播
    try:
        loss.backward()
        print(f"✅ 反向傳播成功")
        print(f"   coords.grad 形狀: {coords.grad.shape}")
        print(f"   coords.grad 範數: {coords.grad.norm().item():.4f}")
        return True
    except Exception as e:
        print(f"❌ 反向傳播失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主測試函數
    """
    print("\n" + "=" * 60)
    print("向量化二階梯度計算 - 完整測試")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    # 測試 1: 正確性
    test1_passed = test_correctness()
    
    # 測試 2: 性能
    if test1_passed:
        test_performance()
    
    # 測試 3: 反向傳播
    test3_passed = test_backward_pass()
    
    # 總結
    print("\n" + "=" * 60)
    print("測試總結")
    print("=" * 60)
    print(f"正確性測試:   {'✅ 通過' if test1_passed else '❌ 失敗'}")
    print(f"反向傳播測試: {'✅ 通過' if test3_passed else '❌ 失敗'}")
    
    if test1_passed and test3_passed:
        print("\n🎉 所有測試通過！向量化方法可以安全使用。")
        return True
    else:
        print("\n⚠️  部分測試失敗，需要進一步調查。")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
