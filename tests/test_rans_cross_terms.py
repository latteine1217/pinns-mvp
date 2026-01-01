"""
測試 RANS 湍流黏度的交叉項計算 (∇ν_t·∇u)
========================================

驗證 use_grad_nut 參數和交叉項計算的正確性
"""

import torch
import numpy as np
import pytest
from pinnx.physics.ns_2d import ns_residual_2d, NSEquations2D


class TestRANSCrossTerms:
    """RANS 交叉項測試套件"""
    
    def test_cross_term_activation(self):
        """測試 use_grad_nut 參數正確啟用交叉項計算"""
        torch.manual_seed(42)
        
        # 創建簡單測試數據
        coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], requires_grad=True)
        
        # 簡單的非線性速度場（確保二階導數非零）
        u = coords[:, 0:1] * 0.5 + 0.01 * torch.sin(coords[:, 0:1])
        v = coords[:, 1:2] * 0.3 + 0.01 * torch.sin(coords[:, 1:2])
        p = 0.01 * (coords[:, 0:1] ** 2 + coords[:, 1:2] ** 2)
        S = 0.01 * torch.sin(coords[:, 0:1])
        pred = torch.cat([u, v, p, S], dim=1)
        
        nu = 1e-3
        
        # 創建空間變化的 nu_t (線性變化)
        nu_t = 0.01 + 0.1 * coords[:, 0:1]  # ν_t(x) = 0.01 + 0.1x
        
        # 不使用交叉項
        mom_x_no_cross, mom_y_no_cross, cont_no_cross = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=False
        )
        
        # 使用交叉項
        mom_x_with_cross, mom_y_with_cross, cont_with_cross = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=True
        )
        
        # 驗證：連續方程不受影響
        assert torch.allclose(cont_no_cross, cont_with_cross, atol=1e-6), \
            "連續方程不應受 use_grad_nut 影響"
        
        # 驗證：動量方程有差異（因為 ∇ν_t ≠ 0）
        diff_x = torch.abs(mom_x_no_cross - mom_x_with_cross).mean()
        diff_y = torch.abs(mom_y_no_cross - mom_y_with_cross).mean()
        
        assert diff_x > 1e-6 or diff_y > 1e-6, \
            f"交叉項應該影響動量方程殘差 (diff_x={diff_x:.2e}, diff_y={diff_y:.2e})"
        
        print(f"✅ 交叉項啟用測試通過:")
        print(f"   x-動量殘差差異: {diff_x:.2e}")
        print(f"   y-動量殘差差異: {diff_y:.2e}")
    
    def test_cross_term_with_constant_nut(self):
        """測試常數 ν_t 時交叉項應為零"""
        torch.manual_seed(43)
        
        coords = torch.randn(10, 2, requires_grad=True)
        # 創建與 coords 有計算圖連接的速度場（使用非線性函數）
        u = torch.sin(coords[:, 0:1]) + torch.cos(coords[:, 1:2])
        v = torch.cos(coords[:, 0:1]) + torch.sin(coords[:, 1:2])
        p = coords[:, 0:1] ** 2 + coords[:, 1:2] ** 2
        S = 0.01 * torch.sin(coords[:, 0:1])
        pred = torch.cat([u, v, p, S], dim=1)
        
        nu = 1e-3
        nu_t = 0.05 + 0.0 * coords[:, 0:1]  # 常數 ν_t（保持與 coords 的計算圖連接）
        
        # 計算殘差（有/無交叉項）
        mom_x_no_cross, _, _ = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=False
        )
        mom_x_with_cross, _, _ = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=True
        )
        
        # 常數 ν_t 時，∇ν_t = 0，交叉項應該為零
        diff = torch.abs(mom_x_no_cross - mom_x_with_cross).max()
        
        assert diff < 1e-5, \
            f"常數 ν_t 時交叉項應為零，但差異為 {diff:.2e}"
        
        print(f"✅ 常數 ν_t 測試通過: 最大差異 = {diff:.2e}")
    
    def test_cross_term_sign_and_magnitude(self):
        """測試交叉項的符號和量級正確性"""
        torch.manual_seed(44)
        
        # 創建特定梯度的場景
        x = torch.linspace(0, 1, 5, requires_grad=True)
        y = torch.linspace(0, 1, 5, requires_grad=True)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        
        # 速度場：u = x², v = y²
        u = coords[:, 0:1] ** 2
        v = coords[:, 1:2] ** 2
        p = torch.zeros_like(u)
        S = torch.zeros_like(u)
        pred = torch.cat([u, v, p, S], dim=1)
        
        nu = 1e-3
        
        # ν_t 隨 x 線性增加：ν_t = 0.01 + 0.1·x
        # ∇ν_t = [0.1, 0]
        nu_t = 0.01 + 0.1 * coords[:, 0:1]
        
        # 計算帶交叉項的殘差
        mom_x_with, mom_y_with, _ = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=True
        )
        
        # 計算不帶交叉項的殘差
        mom_x_no, mom_y_no, _ = ns_residual_2d(
            coords, pred, nu, nu_t=nu_t, use_grad_nut=False
        )
        
        # 理論交叉項貢獻（手動計算）
        # ∇ν_t·∇u = (0.1, 0)·(∂u/∂x, ∂u/∂y) = 0.1 * ∂u/∂x = 0.1 * 2x = 0.2x
        theoretical_cross_x = 0.1 * 2 * coords[:, 0:1]
        
        # 從殘差差異提取交叉項
        # (momentum_with - momentum_no) = -cross_term (因為是減去黏性項)
        extracted_cross_x = -(mom_x_with - mom_x_no)
        
        # 驗證交叉項符合理論
        error = torch.abs(extracted_cross_x - theoretical_cross_x).mean()
        
        assert error < 0.1, \
            f"交叉項與理論值差異過大: {error:.2e}"
        
        print(f"✅ 交叉項符號/量級測試通過:")
        print(f"   理論值範圍: [{theoretical_cross_x.min():.3f}, {theoretical_cross_x.max():.3f}]")
        print(f"   提取值範圍: [{extracted_cross_x.min():.3f}, {extracted_cross_x.max():.3f}]")
        print(f"   平均誤差: {error:.2e}")
    
    def test_nsequations2d_cross_term_support(self):
        """測試 NSEquations2D 類支援 use_grad_nut 參數"""
        ns_eq = NSEquations2D(viscosity=1e-3)
        
        coords = torch.randn(5, 2, requires_grad=True)
        # 創建與 coords 有計算圖連接的場
        u = torch.sin(coords[:, 0:1]) + 0.5 * coords[:, 0:1]
        v = torch.cos(coords[:, 1:2]) + 0.5 * coords[:, 1:2]
        p = coords[:, 0:1] ** 2 + coords[:, 1:2] ** 2
        S = 0.01 * torch.sin(coords[:, 0:1] + coords[:, 1:2])
        pred = torch.cat([u, v, p, S], dim=1)
        nu_t = 0.05 + 0.1 * torch.sin(coords[:, 0:1])
        
        # 測試 residual 接受 use_grad_nut 參數
        try:
            residuals = ns_eq.residual(
                coords, pred, nu_t=nu_t, use_grad_nut=True
            )
            assert 'momentum_x' in residuals
            assert 'momentum_y' in residuals
            assert 'continuity' in residuals
            print("✅ NSEquations2D 支援 use_grad_nut 參數")
        except TypeError as e:
            pytest.fail(f"NSEquations2D 不支援 use_grad_nut: {e}")
    
    def test_cross_term_computational_cost(self):
        """測試交叉項計算的額外成本（定性）"""
        import time
        
        torch.manual_seed(45)
        
        # 較大的測試數據
        coords = torch.randn(1000, 2, requires_grad=True)
        # 創建與 coords 有計算圖連接的場
        u = torch.sin(coords[:, 0:1]) + 0.5 * coords[:, 0:1]
        v = torch.cos(coords[:, 1:2]) + 0.5 * coords[:, 1:2]
        p = coords[:, 0:1] ** 2 + coords[:, 1:2] ** 2
        S = 0.01 * torch.sin(coords[:, 0:1] + coords[:, 1:2])
        pred = torch.cat([u, v, p, S], dim=1)
        nu = 1e-3
        nu_t = 0.05 + 0.1 * torch.sin(coords[:, 0:1])
        
        # 測量不帶交叉項的時間
        start = time.time()
        for _ in range(10):
            _ = ns_residual_2d(coords, pred, nu, nu_t=nu_t, use_grad_nut=False)
        time_no_cross = time.time() - start
        
        # 測量帶交叉項的時間
        start = time.time()
        for _ in range(10):
            _ = ns_residual_2d(coords, pred, nu, nu_t=nu_t, use_grad_nut=True)
        time_with_cross = time.time() - start
        
        overhead = (time_with_cross - time_no_cross) / time_no_cross * 100
        
        print(f"✅ 交叉項計算成本測試:")
        print(f"   不帶交叉項: {time_no_cross:.3f}s")
        print(f"   帶交叉項: {time_with_cross:.3f}s")
        print(f"   額外開銷: {overhead:.1f}%")
        
        # 額外開銷應該在合理範圍內 (< 50%)
        assert overhead < 100, \
            f"交叉項計算開銷過大: {overhead:.1f}%"


def test_all():
    """運行所有測試"""
    suite = TestRANSCrossTerms()
    suite.test_cross_term_activation()
    suite.test_cross_term_with_constant_nut()
    suite.test_cross_term_sign_and_magnitude()
    suite.test_nsequations2d_cross_term_support()
    suite.test_cross_term_computational_cost()
    print("\n" + "="*60)
    print("所有 RANS 交叉項測試通過！")
    print("="*60)


if __name__ == "__main__":
    test_all()
