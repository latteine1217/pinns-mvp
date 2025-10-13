"""
診斷 .detach() 是否導致梯度斷開

檢查點：
1. compute_derivatives_safe() 是否觸發 .detach() 分支
2. 梯度是否能正確反向傳播
3. 物理殘差損失是否能影響模型參數
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/Users/latteine/Documents/coding/pinns-mvp')

from pinnx.physics.ns_2d import NSEquations2D, compute_derivatives_safe

def test_gradient_flow():
    """測試梯度是否能通過物理殘差傳播"""
    print("=" * 80)
    print("🔍 測試 1: 梯度流動性檢查")
    print("=" * 80)
    
    # 創建簡單模型
    model = nn.Sequential(
        nn.Linear(2, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, 3)
    )
    
    # 創建物理求解器
    physics = NSEquations2D(viscosity=1e-3)
    
    # 創建測試數據
    coords = torch.randn(16, 2, requires_grad=True)
    
    # 前向傳播
    pred = model(coords)  # [batch, 3] -> [u, v, p]
    
    print(f"\n📊 輸入狀態:")
    print(f"  coords.requires_grad: {coords.requires_grad}")
    print(f"  pred.requires_grad: {pred.requires_grad}")
    print(f"  pred.grad_fn: {pred.grad_fn}")
    
    # 提取速度和壓力
    velocity = pred[:, :2]
    pressure = pred[:, 2:3]
    
    print(f"\n📊 分量狀態:")
    print(f"  velocity.requires_grad: {velocity.requires_grad}")
    print(f"  pressure.requires_grad: {pressure.requires_grad}")
    
    # 計算物理殘差
    try:
        residuals = physics.residual(coords, velocity, pressure)
        
        print(f"\n📊 殘差計算成功:")
        for key, val in residuals.items():
            print(f"  {key}: shape={val.shape}, requires_grad={val.requires_grad}, grad_fn={val.grad_fn}")
        
        # 計算損失
        loss_pde = sum(r.pow(2).mean() for r in residuals.values())
        print(f"\n📊 PDE Loss: {loss_pde.item():.6f}")
        print(f"  loss_pde.requires_grad: {loss_pde.requires_grad}")
        print(f"  loss_pde.grad_fn: {loss_pde.grad_fn}")
        
        # 反向傳播
        print(f"\n🔙 開始反向傳播...")
        loss_pde.backward()
        
        # 檢查梯度
        grad_count = 0
        total_params = 0
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_count += 1
                grad_norm = param.grad.norm().item()
                if grad_norm > 1e-10:
                    print(f"  ✅ {name}: grad_norm={grad_norm:.6e}")
                else:
                    print(f"  ⚠️  {name}: grad_norm={grad_norm:.6e} (太小!)")
            else:
                print(f"  ❌ {name}: grad=None")
        
        print(f"\n📈 梯度統計:")
        print(f"  有梯度的參數: {grad_count}/{total_params}")
        
        if grad_count == total_params:
            print(f"  ✅ 所有參數都有梯度！")
            return True
        else:
            print(f"  ❌ 部分參數沒有梯度！")
            return False
            
    except Exception as e:
        print(f"\n❌ 殘差計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_detach_branch_trigger():
    """測試 compute_derivatives_safe 是否觸發 .detach() 分支"""
    print("\n" + "=" * 80)
    print("🔍 測試 2: .detach() 分支觸發檢查")
    print("=" * 80)
    
    # 測試場景 1: f 和 x 都有 requires_grad
    print("\n📊 場景 1: f 和 x 都有 requires_grad=True")
    x = torch.randn(16, 2, requires_grad=True)
    f = torch.randn(16, 1, requires_grad=True)
    
    print(f"  輸入: f.requires_grad={f.requires_grad}, x.requires_grad={x.requires_grad}")
    
    # 檢查是否會進入 .detach() 分支
    will_detach_f = not f.requires_grad
    will_detach_x = not x.requires_grad
    
    print(f"  是否觸發 .detach()? f:{will_detach_f}, x:{will_detach_x}")
    
    grad = compute_derivatives_safe(f, x, order=1, keep_graph=True)
    print(f"  輸出: grad.shape={grad.shape}, requires_grad={grad.requires_grad}")
    
    # 測試場景 2: 從模型輸出的 f
    print("\n📊 場景 2: 從模型輸出的 f（真實訓練場景）")
    model = nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 1)
    )
    
    x = torch.randn(16, 2, requires_grad=True)
    f = model(x)
    
    print(f"  輸入: f.requires_grad={f.requires_grad}, x.requires_grad={x.requires_grad}")
    print(f"  f.grad_fn={f.grad_fn}")
    
    will_detach_f = not f.requires_grad
    will_detach_x = not x.requires_grad
    
    print(f"  是否觸發 .detach()? f:{will_detach_f}, x:{will_detach_x}")
    
    grad = compute_derivatives_safe(f, x, order=1, keep_graph=True)
    print(f"  輸出: grad.shape={grad.shape}, requires_grad={grad.requires_grad}")
    
    # 測試場景 3: 分解後的 pred[:, 0:1]
    print("\n📊 場景 3: 分解後的 pred[:, 0:1]（ns_residual_2d 內部）")
    pred = model(x)  # [16, 1]
    # 模擬 ns_residual_2d 中的操作
    u = pred[:, 0:1].requires_grad_(True)
    
    print(f"  輸入: u.requires_grad={u.requires_grad}, x.requires_grad={x.requires_grad}")
    print(f"  u.grad_fn={u.grad_fn}")
    
    will_detach_u = not u.requires_grad
    will_detach_x = not x.requires_grad
    
    print(f"  是否觸發 .detach()? u:{will_detach_u}, x:{will_detach_x}")
    
    grad = compute_derivatives_safe(u, x, order=1, keep_graph=True)
    print(f"  輸出: grad.shape={grad.shape}, requires_grad={grad.requires_grad}")


def test_gradient_with_vs_without_detach():
    """對比有無 .detach() 的梯度差異"""
    print("\n" + "=" * 80)
    print("🔍 測試 3: 對比有無 .detach() 的梯度")
    print("=" * 80)
    
    def compute_grad_with_detach(f, x):
        """模擬當前有問題的版本"""
        if not f.requires_grad:
            f = f.clone().detach().requires_grad_(True)
        if not x.requires_grad:
            x = x.clone().detach().requires_grad_(True)
        
        grad_outputs = torch.ones_like(f)
        grads = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        return grads
    
    def compute_grad_without_detach(f, x):
        """修復後的版本"""
        if not f.requires_grad:
            f.requires_grad_(True)
        if not x.requires_grad:
            x.requires_grad_(True)
        
        grad_outputs = torch.ones_like(f)
        grads = torch.autograd.grad(
            outputs=f,
            inputs=x,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True
        )[0]
        return grads
    
    # 創建模型
    model = nn.Linear(2, 1)
    
    # 測試數據
    x = torch.randn(8, 2, requires_grad=True)
    
    # 版本 1: 有 .detach()
    print("\n📊 版本 1: 使用 .detach() (當前版本)")
    f1 = model(x)
    grad1 = compute_grad_with_detach(f1, x)
    loss1 = grad1.pow(2).mean()
    
    model.zero_grad()
    loss1.backward()
    
    weight_grad1 = model.weight.grad.clone() if model.weight.grad is not None else None
    print(f"  Loss: {loss1.item():.6f}")
    if weight_grad1 is not None:
        print(f"  Weight grad norm: {weight_grad1.norm().item():.6e}")
    else:
        print(f"  ❌ Weight grad: None")
    
    # 版本 2: 無 .detach()
    print("\n📊 版本 2: 不使用 .detach() (修復版本)")
    model.zero_grad()
    x2 = torch.randn(8, 2, requires_grad=True)
    f2 = model(x2)
    grad2 = compute_grad_without_detach(f2, x2)
    loss2 = grad2.pow(2).mean()
    
    loss2.backward()
    
    weight_grad2 = model.weight.grad.clone() if model.weight.grad is not None else None
    print(f"  Loss: {loss2.item():.6f}")
    if weight_grad2 is not None:
        print(f"  Weight grad norm: {weight_grad2.norm().item():.6e}")
    else:
        print(f"  ❌ Weight grad: None")
    
    # 對比
    print("\n📈 對比結果:")
    if weight_grad1 is None and weight_grad2 is not None:
        print("  ❌ 版本 1 (.detach()) 無法產生梯度！")
        print("  ✅ 版本 2 (無 .detach()) 可以產生梯度！")
        return False
    elif weight_grad1 is not None and weight_grad2 is not None:
        diff = (weight_grad1 - weight_grad2).abs().max().item()
        print(f"  梯度差異: {diff:.6e}")
        if diff > 1e-6:
            print("  ⚠️  兩版本梯度不同！")
        else:
            print("  ✅ 兩版本梯度一致")
        return True
    else:
        print("  ❌ 兩版本都無法產生梯度！")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚨 診斷 .detach() 導致的梯度斷開問題")
    print("=" * 80)
    
    # 運行測試
    test1_pass = test_gradient_flow()
    test_detach_branch_trigger()
    test2_pass = test_gradient_with_vs_without_detach()
    
    print("\n" + "=" * 80)
    print("📊 診斷總結")
    print("=" * 80)
    print(f"  測試 1 (梯度流動): {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  測試 3 (有無 .detach() 對比): {'✅ PASS' if test2_pass else '❌ FAIL'}")
    
    if not test1_pass:
        print("\n⚠️  結論: 當前版本的 compute_derivatives_safe() 可能阻斷梯度！")
        print("  建議: 移除 Line 36-38 的 .detach() 調用")
    else:
        print("\n✅ 結論: 梯度流動正常")
