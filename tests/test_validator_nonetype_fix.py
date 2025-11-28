"""
測試 validators.py 的 NoneType 修復

運行方式:
    pytest tests/test_validator_nonetype_fix.py -v
    或
    python tests/test_validator_nonetype_fix.py
"""

import torch
import pytest
from pinnx.physics.validators import (
    validate_momentum_conservation,
    compute_physics_metrics
)


class TestValidatorNoneTypeFix:
    """測試 w=None 時的物理驗證功能"""
    
    def test_2d_coords_with_none_w(self):
        """測試案例 1: 標準 2D 座標，w=None"""
        coords = torch.tensor([
            [1.0, 0.5],
            [2.0, 1.0],
            [1.5, 1.5]
        ], requires_grad=True)
        
        u = torch.tensor([[0.5], [0.6], [0.55]], requires_grad=True)
        v = torch.tensor([[0.1], [0.12], [0.11]], requires_grad=True)
        p = torch.tensor([[1.0], [1.1], [1.05]], requires_grad=True)
        
        # 不應拋出 NoneType 錯誤
        passed, error = validate_momentum_conservation(coords, u, v, p, w=None)
        
        assert isinstance(passed, bool)
        assert isinstance(error, float)
        assert error >= 0.0
    
    def test_3d_coords_with_none_w(self):
        """測試案例 2: 3D 座標但 w=None（Kolmogorov Flow 場景）"""
        coords = torch.tensor([
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 0.5],
            [1.5, 0.8, 0.3]
        ], requires_grad=True)
        
        u = torch.tensor([[0.5], [0.6], [0.55]], requires_grad=True)
        v = torch.tensor([[0.1], [0.12], [0.11]], requires_grad=True)
        p = torch.tensor([[1.0], [1.1], [1.05]], requires_grad=True)
        
        # 修復後應正常處理（使用 2D 路徑）
        passed, error = validate_momentum_conservation(coords, u, v, p, w=None)
        
        assert isinstance(passed, bool)
        assert isinstance(error, float)
        assert error >= 0.0
    
    def test_3d_coords_with_w(self):
        """測試案例 3: 完整 3D 流場（w 不為 None）"""
        coords = torch.tensor([
            [1.0, 0.5, 0.3],
            [2.0, 1.0, 0.6]
        ], requires_grad=True)
        
        u = torch.tensor([[0.5], [0.6]], requires_grad=True)
        v = torch.tensor([[0.1], [0.12]], requires_grad=True)
        w = torch.tensor([[0.05], [0.06]], requires_grad=True)
        p = torch.tensor([[1.0], [1.1]], requires_grad=True)
        
        # 完整 3D 應正常工作
        passed, error = validate_momentum_conservation(coords, u, v, p, w=w)
        
        assert isinstance(passed, bool)
        assert isinstance(error, float)
        assert error >= 0.0
    
    def test_compute_physics_metrics_with_none_w(self):
        """測試案例 4: compute_physics_metrics 整合驗證（w=None）"""
        coords = torch.tensor([
            [1.0, 0.5, 0.0],
            [2.0, 1.0, 0.5]
        ], requires_grad=True)
        
        predictions = {
            'u': torch.tensor([[0.5], [0.6]], requires_grad=True),
            'v': torch.tensor([[0.1], [0.12]], requires_grad=True),
            'p': torch.tensor([[1.0], [1.1]], requires_grad=True)
            # 注意：沒有 'w' 鍵
        }
        
        # 不應拋出 NoneType 錯誤
        metrics = compute_physics_metrics(coords, predictions)
        
        assert 'mass_conservation_error' in metrics
        assert 'momentum_conservation_error' in metrics
        assert 'validation_passed' in metrics
        assert isinstance(metrics['validation_passed'], bool)


def manual_test():
    """手動測試（不使用 pytest）"""
    print("=" * 70)
    print("物理驗證器 NoneType 修復 - 手動測試")
    print("=" * 70)
    
    # 測試 1: 2D 座標
    print("\n[測試 1] 2D 座標，w=None")
    coords_2d = torch.tensor([[1.0, 0.5], [2.0, 1.0]], requires_grad=True)
    u = torch.tensor([[0.5], [0.6]], requires_grad=True)
    v = torch.tensor([[0.1], [0.12]], requires_grad=True)
    p = torch.tensor([[1.0], [1.1]], requires_grad=True)
    
    try:
        passed, error = validate_momentum_conservation(coords_2d, u, v, p, w=None)
        print(f"  ✓ 通過: passed={passed}, error={error:.6e}")
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
    
    # 測試 2: 3D 座標但 w=None
    print("\n[測試 2] 3D 座標但 w=None（Kolmogorov Flow）")
    coords_3d = torch.tensor([[1.0, 0.5, 0.0], [2.0, 1.0, 0.5]], requires_grad=True)
    
    try:
        passed, error = validate_momentum_conservation(coords_3d, u, v, p, w=None)
        print(f"  ✓ 通過: passed={passed}, error={error:.6e}")
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
    
    # 測試 3: 完整 3D
    print("\n[測試 3] 完整 3D 流場（w 不為 None）")
    w_3d = torch.tensor([[0.05], [0.06]], requires_grad=True)
    
    try:
        passed, error = validate_momentum_conservation(coords_3d, u, v, p, w=w_3d)
        print(f"  ✓ 通過: passed={passed}, error={error:.6e}")
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
    
    # 測試 4: compute_physics_metrics
    print("\n[測試 4] compute_physics_metrics 整合測試（w=None）")
    predictions = {'u': u, 'v': v, 'p': p}
    
    try:
        metrics = compute_physics_metrics(coords_3d, predictions)
        print(f"  ✓ 通過:")
        print(f"    - 質量守恆誤差: {metrics['mass_conservation_error']:.6e}")
        print(f"    - 動量守恆誤差: {metrics['momentum_conservation_error']:.6e}")
        print(f"    - 驗證通過: {metrics['validation_passed']}")
    except Exception as e:
        print(f"  ✗ 失敗: {e}")
    
    print("\n" + "=" * 70)
    print("所有測試完成！")
    print("=" * 70)


if __name__ == "__main__":
    manual_test()
