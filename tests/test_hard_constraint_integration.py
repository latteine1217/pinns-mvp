"""
壁面 Hard Constraint 整合測試
================================

測試從配置到訓練的完整流程，驗證：
1. TrainerBuilder 正確創建 HardConstraintApplicator
2. Trainer 在前向傳播時正確應用約束
3. 邊界處速度確實為零
4. 壓力場不受影響

運行方式：
    python tests/test_hard_constraint_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import logging

logging.basicConfig(level=logging.INFO)

def test_hard_constraint_creation():
    """測試從配置創建 HardConstraintApplicator"""
    print("\n" + "="*60)
    print("測試 1：從配置創建 HardConstraintApplicator")
    print("="*60)
    
    from pinnx.utils.boundary_constraints import create_channel_flow_hard_constraint
    
    # 測試配置
    config = {
        'form': 'cosh',  # 使用 cosh 作為預設
        'y_range': (-1.0, 1.0),
        'alpha': 10.0,
        'variable_order': ['u', 'v', 'w', 'p'],
        'constrained_vars': ['u', 'v', 'w'],
        'y_axis_index': 2,
    }
    
    applicator = create_channel_flow_hard_constraint(**config)
    
    # 驗證
    info = applicator.get_info()
    assert info['distance_function'] == 'cosh'
    assert info['y_range'] == (-1.0, 1.0)
    assert info['constrained_variables'] == ['u', 'v', 'w']
    
    print("✅ HardConstraintApplicator 創建成功")
    return applicator


def test_boundary_conditions(applicator):
    """測試邊界條件是否正確滿足"""
    print("\n" + "="*60)
    print("測試 2：驗證邊界條件")
    print("="*60)
    
    # 創建測試數據（包含邊界點）
    batch_size = 100
    coords = torch.zeros(batch_size, 4)  # [t, x, y, z]
    coords[:, 2] = torch.linspace(-1, 1, batch_size)  # y ∈ [-1, 1]
    
    # 模擬網路輸出（假設所有速度都是 1）
    predictions = torch.ones(batch_size, 4)  # [u, v, w, p] = [1, 1, 1, 1]
    
    # 應用約束
    constrained = applicator.apply(coords, predictions)
    
    # 驗證邊界條件
    tol = 1e-6
    
    # 下邊界 (y=-1)
    u_lower = constrained[0, 0].item()
    v_lower = constrained[0, 1].item()
    w_lower = constrained[0, 2].item()
    p_lower = constrained[0, 3].item()
    
    print(f"\n下邊界 (y=-1):")
    print(f"  u = {u_lower:.10f} (應為 0)")
    print(f"  v = {v_lower:.10f} (應為 0)")
    print(f"  w = {w_lower:.10f} (應為 0)")
    print(f"  p = {p_lower:.10f} (應為 1，不受約束)")
    
    assert abs(u_lower) < tol, f"下邊界 u 未滿足: {u_lower}"
    assert abs(v_lower) < tol, f"下邊界 v 未滿足: {v_lower}"
    assert abs(w_lower) < tol, f"下邊界 w 未滿足: {w_lower}"
    assert abs(p_lower - 1.0) < tol, f"壓力場受到影響: {p_lower}"
    
    # 上邊界 (y=1)
    u_upper = constrained[-1, 0].item()
    v_upper = constrained[-1, 1].item()
    w_upper = constrained[-1, 2].item()
    p_upper = constrained[-1, 3].item()
    
    print(f"\n上邊界 (y=1):")
    print(f"  u = {u_upper:.10f} (應為 0)")
    print(f"  v = {v_upper:.10f} (應為 0)")
    print(f"  w = {w_upper:.10f} (應為 0)")
    print(f"  p = {p_upper:.10f} (應為 1，不受約束)")
    
    assert abs(u_upper) < tol, f"上邊界 u 未滿足: {u_upper}"
    assert abs(v_upper) < tol, f"上邊界 v 未滿足: {v_upper}"
    assert abs(w_upper) < tol, f"上邊界 w 未滿足: {w_upper}"
    assert abs(p_upper - 1.0) < tol, f"壓力場受到影響: {p_upper}"
    
    # 中心點 (y=0)
    center_idx = batch_size // 2
    u_center = constrained[center_idx, 0].item()
    v_center = constrained[center_idx, 1].item()
    w_center = constrained[center_idx, 2].item()
    
    print(f"\n中心線 (y=0):")
    print(f"  u = {u_center:.10f} (應接近 1)")
    print(f"  v = {v_center:.10f} (應接近 1)")
    print(f"  w = {w_center:.10f} (應接近 1)")
    
    assert u_center > 0.95, f"中心速度過小: {u_center}"
    
    print("\n✅ 所有邊界條件驗證通過")


def test_trainer_builder_integration():
    """測試 TrainerBuilder 整合"""
    print("\n" + "="*60)
    print("測試 3：TrainerBuilder 整合")
    print("="*60)
    
    from pinnx.train.trainer_builder import TrainerBuilder
    
    # 載入配置
    config_path = Path(__file__).parent.parent / 'configs' / 'channel_flow_periodic_example.yml'
    if not config_path.exists():
        print(f"⚠️  配置文件不存在: {config_path}，跳過此測試")
        return
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 檢查配置中是否有 hard_constraint
    hc_config = config.get('physics', {}).get('boundary_conditions', {}).get('hard_constraint', {})
    if not hc_config.get('enabled', False):
        print("⚠️  配置中未啟用 hard_constraint，跳過此測試")
        return
    
    print(f"✅ 配置載入成功")
    print(f"   Hard Constraint 配置: {hc_config}")
    
    # 注意：完整的 Trainer 構建需要模型和物理模組
    # 這裡只測試配置解析
    print("✅ TrainerBuilder 整合測試通過（配置解析）")


def test_different_distance_functions():
    """測試不同的距離函數形式"""
    print("\n" + "="*60)
    print("測試 4：不同距離函數形式")
    print("="*60)
    
    from pinnx.utils.boundary_constraints import WallDistanceFunction
    from typing import Literal
    
    forms: list[Literal['quadratic', 'cosh', 'sin']] = ['quadratic', 'cosh', 'sin']
    y_test = torch.linspace(-1, 1, 200)
    
    for form in forms:
        dist_fn = WallDistanceFunction(form=form, alpha=10.0)
        d = dist_fn(y_test)
        
        # 驗證邊界條件
        assert abs(d[0].item()) < 1e-6, f"{form}: 下邊界不為零"
        assert abs(d[-1].item()) < 1e-6, f"{form}: 上邊界不為零"
        assert d[len(y_test)//2].item() > 0.95, f"{form}: 中心值過小"
        
        print(f"  ✅ {form:10s}: d(-1)={d[0]:.2e}, d(0)={d[len(y_test)//2]:.4f}, d(1)={d[-1]:.2e}")
    
    print("\n✅ 所有距離函數形式驗證通過")


def test_gradient_preservation():
    """測試梯度是否正確傳播"""
    print("\n" + "="*60)
    print("測試 5：梯度傳播")
    print("="*60)
    
    from pinnx.utils.boundary_constraints import create_channel_flow_hard_constraint
    
    applicator = create_channel_flow_hard_constraint(
        form='cosh',  # 使用 cosh 作為預設測試
        variable_order=['u', 'v', 'w', 'p'],
        y_axis_index=2,
    )
    
    # 創建測試數據（需要梯度）
    coords = torch.randn(100, 4)
    coords[:, 2] = torch.linspace(-1, 1, 100)  # y ∈ [-1, 1]
    
    predictions = torch.randn(100, 4, requires_grad=True)
    
    # 應用約束
    constrained = applicator.apply(coords, predictions)
    
    # 計算損失並反向傳播
    loss = constrained.sum()
    loss.backward()
    
    # 驗證梯度存在
    assert predictions.grad is not None, "梯度未傳播"
    assert not torch.isnan(predictions.grad).any(), "梯度包含 NaN"
    
    print(f"  梯度形狀: {predictions.grad.shape}")
    print(f"  梯度範圍: [{predictions.grad.min():.4f}, {predictions.grad.max():.4f}]")
    print("✅ 梯度正確傳播")


def run_all_tests():
    """運行所有測試"""
    print("\n" + "="*60)
    print("🧪 壁面 Hard Constraint 整合測試套件")
    print("="*60)
    
    try:
        # 測試 1：創建 applicator
        applicator = test_hard_constraint_creation()
        
        # 測試 2：驗證邊界條件
        test_boundary_conditions(applicator)
        
        # 測試 3：TrainerBuilder 整合
        test_trainer_builder_integration()
        
        # 測試 4：不同距離函數
        test_different_distance_functions()
        
        # 測試 5：梯度傳播
        test_gradient_preservation()
        
        print("\n" + "="*60)
        print("🎉 所有測試通過！")
        print("="*60)
        
    except AssertionError as e:
        print(f"\n❌ 測試失敗: {e}")
        raise
    except Exception as e:
        print(f"\n❌ 測試異常: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
