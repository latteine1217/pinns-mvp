"""
測試 turbulence_utils 模組
==========================

測試湍流工具函數，特別是 van Driest damping 功能
"""

import torch
import numpy as np
import pytest
from pinnx.physics.turbulence_utils import (
    compute_wall_distance_channel,
    compute_yplus,
    van_driest_damping,
    apply_van_driest_damping,
    clip_turbulent_viscosity,
    preprocess_rans_prior,
    diagnose_turbulent_viscosity
)


class TestWallDistanceCalculation:
    """壁面距離計算測試"""
    
    def test_2d_channel_symmetric(self):
        """測試 2D 通道的對稱性"""
        # 通道高度 h = 2.0，y ∈ [0, 2]
        coords_bottom = torch.tensor([[1.0, 0.1]])  # 近下壁面
        coords_top = torch.tensor([[1.0, 1.9]])     # 近上壁面
        coords_center = torch.tensor([[1.0, 1.0]])  # 中心線
        
        domain_bounds = (0.0, 2*np.pi, 0.0, 2.0)
        
        d_bottom = compute_wall_distance_channel(coords_bottom, domain_bounds)
        d_top = compute_wall_distance_channel(coords_top, domain_bounds)
        d_center = compute_wall_distance_channel(coords_center, domain_bounds)
        
        assert torch.isclose(d_bottom, torch.tensor([[0.1]]), atol=1e-6)
        assert torch.isclose(d_top, torch.tensor([[0.1]]), atol=1e-6)
        assert torch.isclose(d_center, torch.tensor([[1.0]]), atol=1e-6)
        
        print("✅ 2D 通道對稱性測試通過")
    
    def test_auto_infer_bounds(self):
        """測試自動推斷邊界"""
        coords = torch.tensor([
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 1.0]
        ])
        
        d_wall = compute_wall_distance_channel(coords, domain_bounds=None)
        
        # 自動推斷：y_min=0, y_max=1
        expected = torch.tensor([[0.0], [0.5], [0.0]])
        assert torch.allclose(d_wall, expected, atol=1e-6)
        
        print("✅ 自動推斷邊界測試通過")


class TestYPlus:
    """y+ 計算測試"""
    
    def test_yplus_computation(self):
        """測試 y+ 計算"""
        d_wall = torch.tensor([[0.001], [0.01], [0.1]])
        u_tau = 0.05  # Re_tau=1000 的典型值
        nu = 1e-4
        
        yplus = compute_yplus(d_wall, u_tau, nu)
        
        expected = torch.tensor([[0.5], [5.0], [50.0]])
        assert torch.allclose(yplus, expected, atol=1e-6)
        
        print("✅ y+ 計算測試通過")
    
    def test_yplus_invalid_inputs(self):
        """測試無效輸入"""
        d_wall = torch.tensor([[0.01]])
        
        with pytest.raises(ValueError, match="u_tau 必須為正數"):
            compute_yplus(d_wall, u_tau=-0.1, nu=1e-4)
        
        with pytest.raises(ValueError, match="nu 必須為正數"):
            compute_yplus(d_wall, u_tau=0.05, nu=-1e-4)
        
        print("✅ y+ 無效輸入測試通過")


class TestVanDriestDamping:
    """van Driest 阻尼函數測試"""
    
    def test_damping_asymptotic_behavior(self):
        """測試阻尼函數的漸近行為"""
        yplus = torch.tensor([[0.0], [5.0], [26.0], [100.0], [1000.0]])
        f_damp = van_driest_damping(yplus, A_plus=26.0)
        
        # y+=0: f→0
        assert f_damp[0] < 0.01, "y+=0 時阻尼應接近 0"
        
        # y+=26 (A+): f≈0.632
        assert 0.6 < f_damp[2] < 0.7, "y+=A+ 時阻尼應約為 0.632"
        
        # y+→∞: f→1
        assert f_damp[-1] > 0.99, "y+→∞ 時阻尼應接近 1"
        
        # 單調遞增
        assert torch.all(f_damp[1:] > f_damp[:-1]), "阻尼函數應單調遞增"
        
        print(f"✅ van Driest 阻尼漸近行為測試通過")
        print(f"   y+=0: f={f_damp[0].item():.3f}")
        print(f"   y+=26: f={f_damp[2].item():.3f}")
        print(f"   y+=1000: f={f_damp[-1].item():.3f}")
    
    def test_damping_with_different_Aplus(self):
        """測試不同 A+ 常數"""
        yplus = torch.tensor([[50.0]])
        
        f1 = van_driest_damping(yplus, A_plus=20.0)
        f2 = van_driest_damping(yplus, A_plus=26.0)
        f3 = van_driest_damping(yplus, A_plus=30.0)
        
        # A+ 越小，阻尼恢復越快
        assert f1 > f2 > f3
        
        print(f"✅ 不同 A+ 常數測試通過")
        print(f"   A+=20: f={f1.item():.3f}")
        print(f"   A+=26: f={f2.item():.3f}")
        print(f"   A+=30: f={f3.item():.3f}")


class TestApplyVanDriestDamping:
    """應用 van Driest 阻尼測試"""
    
    def test_damping_near_wall_suppression(self):
        """測試近壁抑制效果"""
        # 創建均勻 ν_t，測試三個代表性位置
        coords = torch.tensor([
            [1.0, 0.001],  # 極近壁 (y+=0.5, d_wall=0.001, f_damp≈1.9%)
            [1.0, 0.01],   # 近壁區 (y+=5, d_wall=0.01, f_damp≈17.5%)
            [1.0, 0.16]    # 遠壁區 (y+=80, d_wall=0.16, f_damp≈95.4%)
        ])
        nu_t_raw = torch.ones(3, 1) * 0.1
        
        u_tau = 0.05
        nu = 1e-4
        domain_bounds = (0.0, 2*np.pi, 0.0, 2.0)
        
        nu_t_damped = apply_van_driest_damping(
            nu_t_raw, coords, u_tau, nu, domain_bounds
        )
        
        # 近壁: ν_t 應該被強烈抑制 (y+=0.5 → f_damp≈1.9%)
        assert nu_t_damped[0] < 0.02 * nu_t_raw[0], "極近壁區應強烈抑制"
        
        # 遠壁: ν_t 應該接近原值 (y+=80 → f_damp≈95%)
        assert nu_t_damped[2] > 0.95 * nu_t_raw[2], "遠壁區應接近原值"
        
        # 單調性：距壁面越遠，ν_t 越大
        assert torch.all(nu_t_damped[1:] > nu_t_damped[:-1]), "ν_t 應隨距離單調遞增"
        
        print(f"✅ 近壁抑制測試通過")
        print(f"   極近壁 (y+=0.5): ν_t={nu_t_damped[0].item():.5f} ({(nu_t_damped[0]/nu_t_raw[0]).item()*100:.1f}% of raw)")
        print(f"   近壁區 (y+=5):   ν_t={nu_t_damped[1].item():.5f} ({(nu_t_damped[1]/nu_t_raw[1]).item()*100:.1f}% of raw)")
        print(f"   遠壁區 (y+=80):  ν_t={nu_t_damped[2].item():.5f} ({(nu_t_damped[2]/nu_t_raw[2]).item()*100:.1f}% of raw)")


class TestClipping:
    """裁剪測試"""
    
    def test_clip_negative_values(self):
        """測試負值裁剪"""
        nu_t = torch.tensor([[-0.1], [0.5], [1.0]])
        nu = 1e-3
        
        with pytest.warns(UserWarning, match="發現 1 個負值"):
            nu_t_clipped = clip_turbulent_viscosity(nu_t, nu, max_ratio=1000)
        
        assert nu_t_clipped[0] == 0.0, "負值應被裁剪為 0"
        assert nu_t_clipped[1] == 0.5, "正值應不變"
        
        print("✅ 負值裁剪測試通過")
    
    def test_clip_exceed_max_ratio(self):
        """測試超限裁剪"""
        nu_t = torch.tensor([[0.1], [0.5], [2.0]])  # 2.0 超過 1.0 (1000*1e-3)
        nu = 1e-3
        
        with pytest.warns(UserWarning, match="超出"):
            nu_t_clipped = clip_turbulent_viscosity(nu_t, nu, max_ratio=1000)
        
        assert nu_t_clipped[2] == 1.0, "超限值應被裁剪到上限"
        
        print("✅ 超限裁剪測試通過")


class TestPreprocessRANSPrior:
    """完整預處理流程測試"""
    
    def test_full_pipeline(self):
        """測試完整預處理流程"""
        # 創建測試數據
        torch.manual_seed(42)
        N = 100
        coords = torch.rand(N, 2)
        coords[:, 1] = coords[:, 1] * 2.0  # y ∈ [0, 2]
        
        # 隨機 ν_t
        nu_t_raw = torch.rand(N, 1) * 0.2
        
        # 參數
        nu = 1e-4
        u_tau = 0.05
        domain_bounds = (0.0, 2*np.pi, 0.0, 2.0)
        
        # 預處理
        nu_t_processed, stats = preprocess_rans_prior(
            nu_t_raw, coords, nu, u_tau, domain_bounds,
            apply_damping=True,
            apply_clipping=True,
            apply_smoothing=False
        )
        
        # 檢查統計
        assert 'raw_mean' in stats
        assert 'processed_mean' in stats
        assert 'damping_factor_mean' in stats
        
        # 處理後應該更小（因為阻尼）
        assert stats['processed_mean'] < stats['raw_mean']
        
        # 阻尼係數應該 < 1
        assert 0 < stats['damping_factor_mean'] < 1
        
        print(f"✅ 完整預處理流程測試通過")
        print(f"   原始平均: {stats['raw_mean']:.5f}")
        print(f"   處理後平均: {stats['processed_mean']:.5f}")
        print(f"   平均阻尼係數: {stats['damping_factor_mean']:.3f}")
        print(f"   裁剪點數: {stats['n_clipped']}")


class TestSmoothing:
    """空間平滑測試"""
    
    def test_smoothing_none(self):
        """測試無平滑模式"""
        from pinnx.physics.turbulence_utils import smooth_turbulent_viscosity
        
        coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        nu_t = torch.tensor([[0.1], [0.2], [0.3]])
        
        nu_t_smoothed = smooth_turbulent_viscosity(
            nu_t, coords, smoothing_radius=0.1, method="none"
        )
        
        # 應該完全相同
        assert torch.allclose(nu_t_smoothed, nu_t)
        print("✅ 無平滑模式測試通過")
    
    def test_gaussian_smoothing(self):
        """測試 Gaussian 平滑"""
        from pinnx.physics.turbulence_utils import smooth_turbulent_viscosity
        
        # 創建含噪聲的數據：中心點異常高
        coords = torch.tensor([
            [0.0, 0.0],  # 鄰近點
            [0.1, 0.0],  # 目標點（異常值）
            [0.2, 0.0],  # 鄰近點
        ])
        nu_t = torch.tensor([[1.0], [10.0], [1.0]])  # 中心異常高
        
        nu_t_smoothed = smooth_turbulent_viscosity(
            nu_t, coords, smoothing_radius=0.15, method="gaussian"
        )
        
        # 平滑後中心點應該降低
        assert nu_t_smoothed[1] < nu_t[1], "異常值應被平滑"
        assert nu_t_smoothed[1] > 1.0, "但不應降到鄰近點值"
        
        # 邊界點變化較小
        assert torch.abs(nu_t_smoothed[0] - nu_t[0]) < 5.0
        
        print("✅ Gaussian 平滑測試通過")
        print(f"   原始: {nu_t.squeeze().tolist()}")
        print(f"   平滑後: {nu_t_smoothed.squeeze().tolist()}")
    
    def test_uniform_smoothing(self):
        """測試 uniform 平滑"""
        from pinnx.physics.turbulence_utils import smooth_turbulent_viscosity
        
        coords = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ])
        nu_t = torch.tensor([[1.0], [10.0], [1.0]])
        
        nu_t_smoothed = smooth_turbulent_viscosity(
            nu_t, coords, smoothing_radius=0.15, method="uniform"
        )
        
        # 平滑後應該降低
        assert nu_t_smoothed[1] < nu_t[1]
        
        print("✅ Uniform 平滑測試通過")


class TestDiagnosis:
    """診斷功能測試"""
    
    def test_diagnosis_warnings(self):
        """測試診斷警告"""
        # 創建有問題的 ν_t
        nu_t = torch.tensor([[-0.1], [0.1], [10.0]])  # 負值 + 過高值
        coords = torch.tensor([[1.0, 0.5], [1.0, 1.0], [1.0, 1.5]])
        nu = 1e-3
        
        diagnosis = diagnose_turbulent_viscosity(nu_t, coords, nu)
        
        # 檢查基本統計
        assert diagnosis['n_negative'] == 1
        assert diagnosis['ratio_max'] == 10000.0  # 10.0 / 1e-3
        
        # 應該有警告
        assert len(diagnosis['warnings']) > 0
        
        print(f"✅ 診斷功能測試通過")
        print(f"   負值點數: {diagnosis['n_negative']}")
        print(f"   ν_t/ν 最大值: {diagnosis['ratio_max']:.0f}")
        print(f"   警告數: {len(diagnosis['warnings'])}")
        for warning in diagnosis['warnings']:
            print(f"   - {warning}")


def test_all():
    """運行所有測試"""
    print("\n" + "="*60)
    print("運行 turbulence_utils 測試套件")
    print("="*60 + "\n")
    
    # 壁面距離
    suite1 = TestWallDistanceCalculation()
    suite1.test_2d_channel_symmetric()
    suite1.test_auto_infer_bounds()
    
    # y+
    suite2 = TestYPlus()
    suite2.test_yplus_computation()
    suite2.test_yplus_invalid_inputs()
    
    # van Driest damping
    suite3 = TestVanDriestDamping()
    suite3.test_damping_asymptotic_behavior()
    suite3.test_damping_with_different_Aplus()
    
    # 應用阻尼
    suite4 = TestApplyVanDriestDamping()
    suite4.test_damping_near_wall_suppression()
    
    # 裁剪
    suite5 = TestClipping()
    suite5.test_clip_negative_values()
    suite5.test_clip_exceed_max_ratio()
    
    # 完整流程
    suite6 = TestPreprocessRANSPrior()
    suite6.test_full_pipeline()
    
    # 診斷
    suite7 = TestDiagnosis()
    suite7.test_diagnosis_warnings()
    
    print("\n" + "="*60)
    print("所有 turbulence_utils 測試通過！")
    print("="*60)


if __name__ == "__main__":
    test_all()
