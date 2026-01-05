"""測試 Kolmogorov 標準化與 JAX-PI 對齊."""

import math
import pytest
import numpy as np
import torch

from pinnx.utils.normalization import KolmogorovInputTransform


class TestKolmogorovInputTransform:
    """KolmogorovInputTransform 單元測試."""
    
    def test_initialization(self):
        """測試標準化器初始化."""
        transform = KolmogorovInputTransform(t_max=50.0)
        assert transform.t_max == 50.0
        assert transform.norm_type == 'kolmogorov'
    
    def test_initialization_invalid_t_max(self):
        """測試無效 t_max 參數."""
        with pytest.raises(ValueError, match="t_max 必須大於 0"):
            KolmogorovInputTransform(t_max=0.0)
        
        with pytest.raises(ValueError, match="t_max 必須大於 0"):
            KolmogorovInputTransform(t_max=-1.0)
    
    def test_transform_torch_tensor(self):
        """測試 torch.Tensor 標準化."""
        # 創建測試數據
        t = torch.linspace(0, 50, 11).unsqueeze(1)
        x = torch.linspace(0, 2*math.pi, 11).unsqueeze(1)
        y = torch.linspace(0, 2*math.pi, 11).unsqueeze(1)
        coords = torch.cat([t, x, y], dim=1)
        
        # 標準化
        transform = KolmogorovInputTransform(t_max=50.0)
        coords_norm = transform.transform(coords)
        
        # 驗證時間維度標準化到 [0, 1]
        assert coords_norm[:, 0].min().item() == pytest.approx(0.0, abs=1e-6)
        assert coords_norm[:, 0].max().item() == pytest.approx(1.0, abs=1e-6)
        
        # 驗證空間維度保持不變
        assert torch.allclose(coords_norm[:, 1], x.squeeze(), rtol=1e-5)
        assert torch.allclose(coords_norm[:, 2], y.squeeze(), rtol=1e-5)
    
    def test_transform_numpy_array(self):
        """測試 numpy.ndarray 標準化."""
        # 創建測試數據
        t = np.linspace(0, 50, 11).reshape(-1, 1)
        x = np.linspace(0, 2*math.pi, 11).reshape(-1, 1)
        y = np.linspace(0, 2*math.pi, 11).reshape(-1, 1)
        coords = np.concatenate([t, x, y], axis=1)
        
        # 標準化
        transform = KolmogorovInputTransform(t_max=50.0)
        coords_norm = transform.transform(coords)
        
        # 驗證時間維度標準化到 [0, 1]
        assert coords_norm[:, 0].min() == pytest.approx(0.0, abs=1e-6)
        assert coords_norm[:, 0].max() == pytest.approx(1.0, abs=1e-6)
        
        # 驗證空間維度保持不變
        assert np.allclose(coords_norm[:, 1], x.squeeze(), rtol=1e-5)
        assert np.allclose(coords_norm[:, 2], y.squeeze(), rtol=1e-5)
    
    def test_inverse_transform_torch(self):
        """測試 torch.Tensor 反標準化."""
        # 創建測試數據
        coords = torch.tensor([
            [0.0, 0.0, 0.0],
            [25.0, math.pi, math.pi],
            [50.0, 2*math.pi, 2*math.pi]
        ])
        
        # 標準化後再反標準化
        transform = KolmogorovInputTransform(t_max=50.0)
        coords_norm = transform.transform(coords)
        coords_restored = transform.inverse_transform(coords_norm)
        
        # 驗證還原正確
        assert torch.allclose(coords_restored, coords, rtol=1e-5)
    
    def test_inverse_transform_numpy(self):
        """測試 numpy.ndarray 反標準化."""
        # 創建測試數據
        coords = np.array([
            [0.0, 0.0, 0.0],
            [25.0, math.pi, math.pi],
            [50.0, 2*math.pi, 2*math.pi]
        ])
        
        # 標準化後再反標準化
        transform = KolmogorovInputTransform(t_max=50.0)
        coords_norm = transform.transform(coords)
        coords_restored = transform.inverse_transform(coords_norm)
        
        # 驗證還原正確
        assert np.allclose(coords_restored, coords, rtol=1e-5)
    
    def test_transform_invalid_shape(self):
        """測試無效輸入形狀."""
        transform = KolmogorovInputTransform(t_max=50.0)
        
        # 1D 張量
        with pytest.raises(ValueError, match="輸入必須是 2D 張量"):
            transform.transform(torch.randn(10))
        
        # 3D 張量
        with pytest.raises(ValueError, match="輸入必須是 2D 張量"):
            transform.transform(torch.randn(10, 3, 2))
        
        # 錯誤的特徵數
        with pytest.raises(ValueError, match="輸入必須有 3 個特徵"):
            transform.transform(torch.randn(10, 4))
    
    def test_transform_invalid_type(self):
        """測試無效輸入類型."""
        transform = KolmogorovInputTransform(t_max=50.0)
        
        # 列表
        with pytest.raises(TypeError, match="輸入類型必須為"):
            transform.transform([[0, 0, 0], [1, 1, 1]])
        
        # 字符串
        with pytest.raises(TypeError, match="輸入類型必須為"):
            transform.transform("invalid")
    
    def test_fit_no_op(self):
        """測試 fit 方法不執行操作."""
        transform = KolmogorovInputTransform(t_max=50.0)
        coords = torch.randn(10, 3)
        
        # fit 應該直接返回 self
        result = transform.fit(coords)
        assert result is transform
    
    def test_to_device(self):
        """測試 to 方法."""
        transform = KolmogorovInputTransform(t_max=50.0)
        
        # to 應該直接返回 self（因為沒有需要移動的張量）
        result = transform.to(torch.device('cpu'))
        assert result is transform
    
    def test_get_metadata(self):
        """測試獲取元數據."""
        transform = KolmogorovInputTransform(t_max=50.0)
        metadata = transform.get_metadata()
        
        assert metadata['type'] == 'kolmogorov'
        assert metadata['t_max'] == 50.0
        assert metadata['dims_normalized'] == [0]
        assert metadata['dims_unchanged'] == [1, 2]
        assert 'description' in metadata
    
    def test_jaxpi_alignment(self):
        """測試與 JAX-PI 對齊（最重要的測試）."""
        # JAX-PI 標準化策略：
        # - Time: t / t_max → [0, 1]
        # - Space: x, y 保持不變 → [0, 2π]
        
        # 創建 Kolmogorov flow 典型坐標
        t = torch.linspace(0, 50, 101).unsqueeze(1)
        x = torch.linspace(0, 2*math.pi, 64).repeat(101, 1)[:, :1]
        y = torch.linspace(0, 2*math.pi, 64).repeat(101, 1)[:, :1]
        coords = torch.cat([t, x[:101], y[:101]], dim=1)
        
        # 應用標準化
        transform = KolmogorovInputTransform(t_max=50.0)
        coords_norm = transform.transform(coords)
        
        # 驗證與 JAX-PI 一致
        # 1. 時間維度：[0, 50] → [0, 1]
        assert coords_norm[:, 0].min().item() >= 0.0
        assert coords_norm[:, 0].max().item() <= 1.0
        assert coords_norm[0, 0].item() == pytest.approx(0.0, abs=1e-6)
        assert coords_norm[-1, 0].item() == pytest.approx(1.0, abs=1e-6)
        
        # 2. 空間維度：保持 [0, 2π]
        assert coords_norm[:, 1].min().item() >= 0.0
        assert coords_norm[:, 1].max().item() <= 2*math.pi
        assert coords_norm[:, 2].min().item() >= 0.0
        assert coords_norm[:, 2].max().item() <= 2*math.pi
        
        print("\n" + "="*60)
        print("✅ JAX-PI 對齊驗證通過")
        print("="*60)
        print(f"時間範圍: [{coords_norm[:, 0].min():.4f}, {coords_norm[:, 0].max():.4f}]")
        print(f"X 範圍:   [{coords_norm[:, 1].min():.4f}, {coords_norm[:, 1].max():.4f}]")
        print(f"Y 範圍:   [{coords_norm[:, 2].min():.4f}, {coords_norm[:, 2].max():.4f}]")
        print("="*60)


if __name__ == '__main__':
    # 直接運行測試
    import sys
    
    test = TestKolmogorovInputTransform()
    
    print("執行 KolmogorovInputTransform 測試套件...")
    print("="*60)
    
    tests = [
        ('初始化', test.test_initialization),
        ('無效 t_max', test.test_initialization_invalid_t_max),
        ('Torch 標準化', test.test_transform_torch_tensor),
        ('Numpy 標準化', test.test_transform_numpy_array),
        ('Torch 反標準化', test.test_inverse_transform_torch),
        ('Numpy 反標準化', test.test_inverse_transform_numpy),
        ('無效形狀', test.test_transform_invalid_shape),
        ('無效類型', test.test_transform_invalid_type),
        ('Fit 方法', test.test_fit_no_op),
        ('To 方法', test.test_to_device),
        ('元數據', test.test_get_metadata),
        ('JAX-PI 對齊', test.test_jaxpi_alignment),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1
    
    print("="*60)
    print(f"測試結果: {passed} 通過, {failed} 失敗")
    print("="*60)
    
    sys.exit(0 if failed == 0 else 1)
