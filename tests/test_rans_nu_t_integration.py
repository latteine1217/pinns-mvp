"""
測試 RANS 湍流黏度 (nu_t) 整合到 PDE 殘差計算

驗證 nu_t 確實影響 NS 方程殘差值
"""

import torch
import pytest
import numpy as np
from pinnx.physics.ns_2d import ns_residual_2d, NSEquations2D


class TestRANSNuTIntegration:
    """測試 RANS nu_t 整合"""
    
    def test_nu_t_affects_residuals_basic(self):
        """驗證 nu_t 改變 PDE 殘差值（基礎測試）"""
        torch.manual_seed(42)
        
        # 建立具有空間結構的測試資料（才會產生非零 Laplacian）
        x = torch.linspace(0, 2*np.pi, 10, requires_grad=True)
        y = torch.linspace(0, 2*np.pi, 10, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        # 使用正弦函數產生具有非零 Laplacian 的場
        u = torch.sin(coords[:, 0:1]) * torch.cos(coords[:, 1:2])
        v = torch.cos(coords[:, 0:1]) * torch.sin(coords[:, 1:2])
        p = torch.sin(coords[:, 0:1] + coords[:, 1:2])
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        pred.requires_grad_(True)
        
        nu = 1e-3  # 分子黏度
        
        # 計算殘差（無 nu_t）
        mom_x_no_nut, mom_y_no_nut, cont_no_nut = ns_residual_2d(
            coords, pred, nu, nu_t=None
        )
        
        # 計算殘差（有 nu_t，設為較大的湍流黏度）
        nu_t = torch.ones(coords.shape[0], 1) * 0.1  # 湍流黏度 >> 分子黏度
        mom_x_with_nut, mom_y_with_nut, cont_with_nut = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t
        )
        
        # 驗證 nu_t 改變動量殘差（黏性項變化）
        diff_x = torch.abs(mom_x_with_nut - mom_x_no_nut).mean()
        diff_y = torch.abs(mom_y_with_nut - mom_y_no_nut).mean()
        
        # nu_t 應該顯著改變動量殘差（因為 nu_t >> nu）
        assert diff_x > 1e-4, f"nu_t 應該影響 x-動量殘差，但差異僅 {diff_x:.2e}"
        assert diff_y > 1e-4, f"nu_t 應該影響 y-動量殘差，但差異僅 {diff_y:.2e}"
        
        # 連續方程不應受 nu_t 影響
        diff_cont = torch.abs(cont_with_nut - cont_no_nut).mean()
        assert diff_cont < 1e-9, f"nu_t 不應影響連續方程，但差異為 {diff_cont:.2e}"
        
        print(f"✅ nu_t 影響殘差測試通過:")
        print(f"   x-動量差異: {diff_x:.6f}")
        print(f"   y-動量差異: {diff_y:.6f}")
        print(f"   連續方程差異: {diff_cont:.2e}")
    
    def test_nu_t_magnitude_effect(self):
        """驗證 nu_t 大小對殘差的影響方向正確"""
        torch.manual_seed(123)
        
        # 使用結構化數據
        x = torch.linspace(0, np.pi, 7, requires_grad=True)
        y = torch.linspace(0, np.pi, 7, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        u = torch.sin(coords[:, 0:1])
        v = torch.cos(coords[:, 1:2])
        p = torch.sin(coords[:, 0:1] * coords[:, 1:2])
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        pred.requires_grad_(True)
        
        nu = 1e-3
        
        # 計算三種情況的殘差
        mom_x_0, _, _ = ns_residual_2d(coords, pred, nu, nu_t=None)
        
        nu_t_small = torch.ones(coords.shape[0], 1) * 0.01
        mom_x_small, _, _ = ns_residual_2d(coords, pred, nu, nu_t=nu_t_small)
        
        nu_t_large = torch.ones(coords.shape[0], 1) * 0.1
        mom_x_large, _, _ = ns_residual_2d(coords, pred, nu, nu_t=nu_t_large)
        
        # nu_t 越大，有效黏度越大，黏性耗散越強
        diff_small = torch.abs(mom_x_small - mom_x_0).mean()
        diff_large = torch.abs(mom_x_large - mom_x_0).mean()
        
        assert diff_large > diff_small, \
            f"更大的 nu_t 應該產生更大的殘差變化，但 large={diff_large:.2e} <= small={diff_small:.2e}"
        
        print(f"✅ nu_t 量級影響測試通過:")
        print(f"   小 nu_t (0.01) 影響: {diff_small:.6f}")
        print(f"   大 nu_t (0.1) 影響: {diff_large:.6f}")
    
    def test_nsequations2d_nu_t_propagation(self):
        """驗證 NSEquations2D 正確傳遞 nu_t 參數"""
        torch.manual_seed(456)
        
        ns_eq = NSEquations2D(viscosity=1e-3)
        
        # 使用結構化數據
        x = torch.linspace(0, 2*np.pi, 6, requires_grad=True)
        y = torch.linspace(0, 2*np.pi, 5, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        u = torch.sin(coords[:, 0:1])
        v = torch.cos(coords[:, 1:2])
        p = torch.sin(coords[:, 0:1] + coords[:, 1:2])
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        pred.requires_grad_(True)
        
        # 測試無 nu_t
        residuals_no_nut = ns_eq.residual(coords, pred, nu_t=None)
        
        # 測試有 nu_t
        nu_t = torch.ones(coords.shape[0], 1) * 0.05
        residuals_with_nut = ns_eq.residual(coords, pred, nu_t=nu_t)
        
        # 驗證殘差不同
        diff_x = torch.abs(
            residuals_with_nut['momentum_x'] - residuals_no_nut['momentum_x']
        ).mean()
        
        assert diff_x > 1e-4, \
            f"NSEquations2D 應該傳遞 nu_t 到 ns_residual_2d，但殘差差異僅 {diff_x:.2e}"
        
        print(f"✅ NSEquations2D nu_t 傳遞測試通過:")
        print(f"   momentum_x 差異: {diff_x:.6f}")
    
    def test_nu_t_zero_equivalent_to_none(self):
        """驗證 nu_t=0 等價於 nu_t=None"""
        torch.manual_seed(789)
        
        # 使用結構化數據
        x = torch.linspace(0, 2*np.pi, 8, requires_grad=True)
        y = torch.linspace(0, 2*np.pi, 5, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        coords.requires_grad_(True)
        
        u = torch.sin(coords[:, 0:1])
        v = torch.cos(coords[:, 1:2])
        p = torch.sin(coords[:, 0:1] * coords[:, 1:2])
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        pred.requires_grad_(True)
        
        nu = 1e-3
        
        # nu_t=None
        mom_x_none, mom_y_none, cont_none = ns_residual_2d(coords, pred, nu, nu_t=None)
        
        # nu_t=0
        nu_t_zero = torch.zeros(coords.shape[0], 1)
        mom_x_zero, mom_y_zero, cont_zero = ns_residual_2d(coords, pred, nu, nu_t=nu_t_zero)
        
        # 應該完全相同（數值誤差在合理範圍內）
        assert torch.allclose(mom_x_none, mom_x_zero, atol=1e-7), \
            "nu_t=None 和 nu_t=0 應該產生相同的殘差"
        assert torch.allclose(mom_y_none, mom_y_zero, atol=1e-7), \
            "nu_t=None 和 nu_t=0 應該產生相同的殘差"
        
        print(f"✅ nu_t=0 等價測試通過")
    
    def test_nu_t_gradient_flow(self):
        """驗證 nu_t 項保持梯度傳播"""
        torch.manual_seed(999)
        
        # 使用結構化數據
        x = torch.linspace(0, np.pi, 5, requires_grad=True)
        y = torch.linspace(0, np.pi, 4, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        u = torch.sin(coords[:, 0:1])
        v = torch.cos(coords[:, 1:2])
        p = torch.sin(coords[:, 0:1] + coords[:, 1:2])
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        
        nu = 1e-3
        # 創建 leaf tensor（直接賦值，不經過運算）
        nu_t = torch.full((coords.shape[0], 1), 0.05, requires_grad=True)
        
        # 計算殘差
        mom_x, mom_y, cont = ns_residual_2d(coords, pred, nu, nu_t=nu_t)
        
        # 計算損失並反向傳播
        loss = torch.mean(mom_x**2 + mom_y**2 + cont**2)
        loss.backward()
        
        # 驗證 nu_t 梯度存在且非零
        # (nu_t 是 leaf tensor，所以一定有 .grad 屬性)
        assert nu_t.grad is not None, "nu_t 應該有梯度"
        
        # 驗證 nu_t 梯度非零（至少有一些非零值）
        assert torch.any(torch.abs(nu_t.grad) > 1e-9), \
            "nu_t 梯度應該非零（至少部分）"
        
        print(f"✅ nu_t 梯度傳播測試通過:")
        print(f"   nu_t.grad 非零比例: {(torch.abs(nu_t.grad) > 1e-9).float().mean():.2%}")
        print(f"   nu_t.grad 平均值: {nu_t.grad.abs().mean():.2e}")


def test_all():
    """運行所有測試"""
    test_suite = TestRANSNuTIntegration()
    
    print("\n" + "="*60)
    print("測試 RANS nu_t 整合到 PDE 殘差計算")
    print("="*60 + "\n")
    
    try:
        test_suite.test_nu_t_affects_residuals_basic()
        print()
        test_suite.test_nu_t_magnitude_effect()
        print()
        test_suite.test_nsequations2d_nu_t_propagation()
        print()
        test_suite.test_nu_t_zero_equivalent_to_none()
        print()
        test_suite.test_nu_t_gradient_flow()
        
        print("\n" + "="*60)
        print("✅ 所有測試通過！RANS nu_t 整合完成。")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print("\n" + "="*60)
        print(f"❌ 測試失敗: {e}")
        print("="*60 + "\n")
        return False


if __name__ == "__main__":
    success = test_all()
    exit(0 if success else 1)
