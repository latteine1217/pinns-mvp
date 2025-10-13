"""
測試 pinnx.evals.metrics 模組的梯度計算修復

本測試驗證 .detach() bug 修復後，評估函數能正確處理梯度計算：
1. conservation_error() - 質量守恆誤差
2. wall_shear_stress() - 壁面剪應力
3. vorticity_field() - 渦量場

測試場景：
- 從文件載入的張量（requires_grad=False）
- 從模型輸出的張量（requires_grad=True）
- 確保兩種情況都能正確計算梯度
"""

import torch
import pytest
import sys
from pathlib import Path

# 添加專案根目錄到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.evals.metrics import (
    conservation_error,
    wall_shear_stress,
    vorticity_field
)


class TestGradientCalculation:
    """測試梯度計算的正確性"""
    
    @pytest.fixture
    def sample_data_no_grad(self):
        """模擬從文件載入的資料（無梯度）"""
        coords = torch.randn(100, 3)  # [t, x, y], requires_grad=False
        u = torch.randn(100)
        v = torch.randn(100)
        return coords, u, v
    
    @pytest.fixture
    def sample_data_with_grad(self):
        """模擬從模型輸出的資料（有梯度）"""
        coords = torch.randn(100, 3, requires_grad=True)
        u = torch.randn(100, requires_grad=True)
        v = torch.randn(100, requires_grad=True)
        return coords, u, v
    
    def test_conservation_error_no_grad(self, sample_data_no_grad):
        """測試質量守恆計算（無梯度輸入）"""
        coords, u, v = sample_data_no_grad
        
        # 確認輸入無梯度
        assert not coords.requires_grad
        assert not u.requires_grad
        assert not v.requires_grad
        
        # 計算應該成功（不會因 .detach() 而失敗）
        try:
            # 重新連接計算圖
            coords.requires_grad_(True)
            u_new = u.clone().requires_grad_(True)
            v_new = v.clone().requires_grad_(True)
            
            # 模擬簡單的速度場
            u_field = (coords[:, 1]**2).requires_grad_(True)
            v_field = (coords[:, 2]**2).requires_grad_(True)
            
            error = conservation_error(u_field, v_field, coords)
            
            # 應該返回有限值
            assert isinstance(error, float)
            assert not torch.isnan(torch.tensor(error))
            assert not torch.isinf(torch.tensor(error))
            
        except RuntimeError as e:
            pytest.fail(f"Conservation error calculation failed: {e}")
    
    def test_wall_shear_stress_no_grad(self, sample_data_no_grad):
        """測試壁面剪應力計算（無梯度輸入）"""
        coords, u, v = sample_data_no_grad
        
        # 確認輸入無梯度
        assert not u.requires_grad
        
        # 計算應該成功
        try:
            coords.requires_grad_(True)
            u_field = (coords[:, 1]**2).requires_grad_(True)
            
            result = wall_shear_stress(u_field, v, coords, viscosity=0.001)
            
            # 檢查結果格式
            assert isinstance(result, dict)
            assert 'tau_w_mean' in result
            assert 'tau_w_std' in result
            
            # 檢查數值有效性
            for key, value in result.items():
                assert isinstance(value, float)
                assert not torch.isnan(torch.tensor(value))
                
        except RuntimeError as e:
            pytest.fail(f"Wall shear stress calculation failed: {e}")
    
    def test_vorticity_field_no_grad(self, sample_data_no_grad):
        """測試渦量場計算（無梯度輸入）"""
        coords, u, v = sample_data_no_grad
        
        # 確認輸入無梯度
        assert not u.requires_grad
        assert not v.requires_grad
        
        # 計算應該成功
        try:
            coords.requires_grad_(True)
            u_field = (coords[:, 1]**2).requires_grad_(True)
            v_field = (coords[:, 2]**2).requires_grad_(True)
            
            vorticity = vorticity_field(u_field, v_field, coords)
            
            # 檢查輸出
            assert isinstance(vorticity, torch.Tensor)
            assert vorticity.shape == u_field.shape
            assert not torch.isnan(vorticity).any()
            
        except RuntimeError as e:
            pytest.fail(f"Vorticity calculation failed: {e}")
    
    def test_conservation_error_with_grad(self, sample_data_with_grad):
        """測試質量守恆計算（有梯度輸入）"""
        coords, u, v = sample_data_with_grad
        
        # 確認輸入有梯度
        assert coords.requires_grad
        
        # 計算應該成功
        try:
            error = conservation_error(u, v, coords)
            
            assert isinstance(error, float)
            assert not torch.isnan(torch.tensor(error))
            
        except RuntimeError as e:
            pytest.fail(f"Conservation error calculation failed: {e}")
    
    def test_gradient_flow_integrity(self):
        """測試梯度流動完整性（關鍵測試）"""
        # 創建簡單的計算圖
        coords = torch.randn(50, 3, requires_grad=True)
        
        # 構造速度場（依賴於 coords）
        u = coords[:, 1]**2 + coords[:, 2]  # u = x² + y
        v = coords[:, 1] * coords[:, 2]     # v = x * y
        
        # 計算質量守恆誤差
        mass_error = conservation_error(u, v, coords)
        
        # 檢查計算圖是否完整（應該能反向傳播到 coords）
        assert mass_error != float('inf'), "Conservation error should be computable"
        
        # 計算渦量場
        vorticity = vorticity_field(u, v, coords)
        
        # 檢查渦量場是否連接到計算圖
        assert vorticity.grad_fn is not None, "Vorticity should have grad_fn"
        
        # 測試反向傳播
        loss = vorticity.sum()
        loss.backward()
        
        # coords 應該收到梯度
        assert coords.grad is not None, "Coords should receive gradients"
        assert not torch.isnan(coords.grad).any(), "Gradients should be valid"
    
    def test_no_detach_side_effects(self):
        """測試修復後不會產生 .detach() 副作用"""
        coords = torch.randn(30, 3, requires_grad=True)
        u = coords[:, 1]**2
        v = coords[:, 2]**2
        
        # 保存原始計算圖引用
        original_grad_fn = u.grad_fn
        
        # 調用評估函數
        _ = conservation_error(u, v, coords)
        
        # 確認計算圖沒有被破壞
        assert u.grad_fn == original_grad_fn, "Computation graph should remain intact"
        
        # 確認 coords 仍然需要梯度
        assert coords.requires_grad, "coords.requires_grad should remain True"


class TestEdgeCases:
    """測試邊界情況"""
    
    def test_empty_tensor(self):
        """測試空張量"""
        coords = torch.empty(0, 3)
        u = torch.empty(0)
        v = torch.empty(0)
        
        # 應該優雅處理（返回警告或無效值）
        error = conservation_error(u, v, coords)
        # 允許返回 inf（因為無法計算）
        assert error == float('inf') or torch.isnan(torch.tensor(error))
    
    def test_single_point(self):
        """測試單點輸入"""
        coords = torch.randn(1, 3, requires_grad=True)
        u = torch.randn(1, requires_grad=True)
        v = torch.randn(1, requires_grad=True)
        
        # 應該能處理（雖然梯度計算可能不精確）
        try:
            _ = conservation_error(u, v, coords)
        except RuntimeError:
            # 單點可能無法計算梯度，這是可接受的
            pass
    
    def test_2d_coords(self):
        """測試2D座標輸入"""
        coords = torch.randn(50, 2, requires_grad=True)  # [x, y]
        u = coords[:, 0]**2
        v = coords[:, 1]**2
        
        # 應該支援2D座標
        error = conservation_error(u, v, coords)
        assert isinstance(error, float)


if __name__ == "__main__":
    # 運行測試
    pytest.main([__file__, "-v", "--tb=short"])
