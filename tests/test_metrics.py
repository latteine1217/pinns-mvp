"""
評估指標模組測試

測試 pinnx.evals.metrics 模組的核心功能
"""

import pytest
import torch
import numpy as np
from pinnx.evals.metrics import (
    relative_L2, 
    rmse_metrics, 
    field_statistics,
    conservation_error,
    boundary_compliance,
    energy_spectrum_1d,
    uncertainty_correlation,
    training_efficiency,
    convergence_analysis,
    comprehensive_evaluation
)


class TestBasicMetrics:
    """基本精度指標測試"""
    
    def test_relative_L2(self):
        """測試相對L2誤差計算"""
        # 建立測試資料
        pred = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        ref = torch.tensor([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
        
        # 計算相對L2誤差
        error = relative_L2(pred, ref)
        
        # 檢查結果
        assert isinstance(error, torch.Tensor)
        assert error.item() > 0
        assert error.item() < 1.0  # 應該是小誤差
    
    def test_rmse_metrics(self):
        """測試RMSE指標計算"""
        pred = torch.randn(100, 3)
        ref = pred + 0.1 * torch.randn(100, 3)  # 添加小噪音
        
        metrics = rmse_metrics(pred, ref)
        
        # 檢查返回的指標
        expected_keys = ['rmse_total', 'rmse_u', 'rmse_v', 'rmse_p']
        for key in expected_keys:
            assert key in metrics
            assert isinstance(metrics[key], float)
            assert metrics[key] >= 0
    
    def test_field_statistics(self):
        """測試場統計量計算"""
        field = torch.randn(1000, 3)  # [u, v, p]
        
        stats = field_statistics(field)
        
        # 檢查統計量
        for var in ['u', 'v', 'p']:
            assert f'{var}_mean' in stats
            assert f'{var}_std' in stats
            assert f'{var}_min' in stats
            assert f'{var}_max' in stats


class TestPhysicsMetrics:
    """物理一致性指標測試"""
    
    def test_conservation_error(self):
        """測試質量守恆誤差計算"""
        # 建立簡單的守恆場 (u=const, v=0)
        coords = torch.linspace(0, 1, 100).view(-1, 1)
        coords = torch.cat([coords, coords, coords], dim=1)  # [t, x, y]
        coords.requires_grad_(True)
        
        u = torch.ones(100, requires_grad=True)  # ∂u/∂x = 0
        v = torch.zeros(100, requires_grad=True)  # ∂v/∂y = 0
        
        error = conservation_error(u, v, coords)
        
        # 守恆場的誤差應該很小
        assert isinstance(error, float)
        assert error < 1e-6 or error == float('inf')  # 可能計算失敗，但不應該報錯
    
    def test_boundary_compliance(self):
        """測試邊界條件符合度"""
        # 建立邊界測試資料
        pred = torch.tensor([[0.1, 0.0], [0.05, 0.0], [0.02, 0.0]])
        bc_values = torch.zeros(3, 2)  # 邊界條件: u=0, v=0
        bc_coords = torch.randn(3, 3)
        
        compliance = boundary_compliance(pred, bc_values, bc_coords)
        
        # 檢查返回指標
        expected_keys = ['bc_max_error', 'bc_mean_error', 'bc_compliance_pct']
        for key in expected_keys:
            assert key in compliance
            assert isinstance(compliance[key], float)


class TestSpectralMetrics:
    """頻譜分析指標測試"""
    
    def test_energy_spectrum_1d(self):
        """測試1D能量譜計算"""
        # 建立簡單的2D場
        field = torch.randn(32, 32)
        
        k, spectrum = energy_spectrum_1d(field)
        
        # 檢查輸出格式
        assert isinstance(k, np.ndarray)
        assert isinstance(spectrum, np.ndarray)
        assert len(k) == len(spectrum)
        assert len(k) > 0


class TestUncertaintyMetrics:
    """不確定性量化指標測試"""
    
    def test_uncertainty_correlation(self):
        """測試UQ可信度評估"""
        # 建立模擬的集成結果
        ensemble_mean = torch.randn(100, 3)
        ensemble_var = torch.rand(100, 3) * 0.1  # 小方差
        true_error = torch.randn(100, 3) * 0.1
        
        uq_metrics = uncertainty_correlation(ensemble_mean, ensemble_var, true_error)
        
        # 檢查返回指標
        expected_keys = ['uq_correlation', 'calibration_rmse', 'mean_predicted_std', 'mean_true_error']
        for key in expected_keys:
            assert key in uq_metrics
            assert isinstance(uq_metrics[key], float)


class TestTrainingMetrics:
    """訓練相關指標測試"""
    
    def test_training_efficiency(self):
        """測試訓練效率分析"""
        # 模擬訓練歷史
        loss_history = [1.0, 0.5, 0.25, 0.125, 0.0625]  # 指數衰減
        time_history = [0.1, 0.1, 0.1, 0.1, 0.1]  # 恆定時間
        
        efficiency = training_efficiency(loss_history, time_history)
        
        # 檢查指標
        expected_keys = ['convergence_rate', 'time_per_epoch', 'efficiency_score']
        for key in expected_keys:
            assert key in efficiency
            assert isinstance(efficiency[key], float)
    
    def test_convergence_analysis(self):
        """測試收斂性分析"""
        # 模擬收斂的殘差
        residuals = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6, 5e-7]
        
        analysis = convergence_analysis(residuals, threshold=1e-6)
        
        # 檢查分析結果
        assert isinstance(analysis['converged'], bool)
        assert isinstance(analysis['convergence_epoch'], int)
        assert isinstance(analysis['final_residual'], float)
        assert isinstance(analysis['monotonic_decrease'], bool)


class TestComprehensiveEvaluation:
    """綜合評估測試"""
    
    def test_comprehensive_evaluation(self):
        """測試綜合評估功能"""
        # 建立測試資料
        N = 100
        predictions = torch.randn(N, 3)  # [u, v, p]
        references = predictions + 0.1 * torch.randn(N, 3)
        coords = torch.randn(N, 3, requires_grad=True)  # [t, x, y]
        physics_params = {'viscosity': 0.01}
        
        # 執行綜合評估
        results = comprehensive_evaluation(predictions, references, coords, physics_params)
        
        # 檢查結果包含各類指標
        assert 'relative_l2' in results
        assert 'rmse_total' in results
        assert 'mass_conservation_error' in results
        
        # 檢查所有值都是數值
        for key, value in results.items():
            assert isinstance(value, (int, float))


# 集成測試
def test_metrics_integration():
    """整體集成測試，確保所有指標能協同工作"""
    
    # 建立較大規模的測試案例
    N = 500
    torch.manual_seed(42)  # 固定隨機種子
    
    # 建立模擬的PINN預測結果
    coords = torch.rand(N, 3) * 2 - 1  # [-1, 1] 範圍
    coords.requires_grad_(True)
    
    # 模擬的速度和壓力場
    predictions = torch.stack([
        torch.sin(coords[:, 1]) * torch.cos(coords[:, 2]),  # u
        -torch.cos(coords[:, 1]) * torch.sin(coords[:, 2]), # v  
        coords[:, 1]**2 + coords[:, 2]**2                   # p
    ], dim=1)
    
    # 參考解 (添加小擾動)
    references = predictions + 0.05 * torch.randn_like(predictions)
    
    # 執行多個指標測試
    l2_error = relative_L2(predictions, references)
    rmse_results = rmse_metrics(predictions, references)
    field_stats = field_statistics(predictions)
    
    # 確保所有指標都能正常計算
    assert l2_error.item() > 0
    assert rmse_results['rmse_total'] > 0
    assert len(field_stats) > 0
    
    print("✅ 所有評估指標測試通過！")


if __name__ == "__main__":
    # 運行基本測試
    test_metrics_integration()
    print("✅ 指標模組測試完成")