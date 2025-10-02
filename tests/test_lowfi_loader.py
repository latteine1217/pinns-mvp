"""
低保真資料載入器測試套件

測試 LowFiLoader 及其相關組件的功能，包括：
- RANS/LES/DNS 資料讀取
- 空間插值功能  
- 下採樣處理
- 資料驗證和品質檢查
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from typing import Dict, Any

# 導入要測試的模組
import sys
sys.path.append('/Users/latteine/Documents/coding/pinns-mvp')

from pinnx.dataio import (
    LowFiData, LowFiLoader, SpatialInterpolator, 
    DownsampledDNSProcessor, create_mock_rans_data,
    NetCDFReader, NPZReader, RANSReader, LESReader
)


class TestLowFiData:
    """測試 LowFiData 類別"""
    
    def test_creation(self):
        """測試基本創建"""
        coords = {'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 8)}
        fields = {'u': np.random.rand(10, 8), 'v': np.random.rand(10, 8)}
        metadata = {'case': 'test'}
        
        data = LowFiData(coordinates=coords, fields=fields, metadata=metadata)
        
        assert data.get_spatial_dims() == (10, 8)
        assert data.get_time_range() is None
        
    def test_with_time(self):
        """測試包含時間的資料"""
        coords = {
            'x': np.linspace(0, 1, 10), 
            'y': np.linspace(0, 1, 8),
            't': np.linspace(0, 10, 5)
        }
        fields = {'u': np.random.rand(10, 8, 5)}
        metadata = {'case': 'unsteady'}
        
        data = LowFiData(coordinates=coords, fields=fields, metadata=metadata)
        
        assert data.get_time_range() == (0.0, 10.0)
        
    def test_physics_validation(self):
        """測試物理驗證"""
        coords = {'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 8)}
        
        # 正常資料
        fields = {'u': np.random.rand(10, 8), 'v': np.random.rand(10, 8)}
        data = LowFiData(coordinates=coords, fields=fields, metadata={})
        
        checks = data.validate_physics()
        assert all(checks.values())
        
        # 包含無限值的資料
        fields_bad = {'u': np.full((10, 8), np.inf)}
        data_bad = LowFiData(coordinates=coords, fields=fields_bad, metadata={})
        
        checks_bad = data_bad.validate_physics()
        assert not checks_bad['u_finite']


class TestSpatialInterpolator:
    """測試空間插值器"""
    
    def setup_method(self):
        """設置測試資料"""
        self.coords = {
            'x': np.linspace(0, 4*np.pi, 32),
            'y': np.linspace(-1, 1, 16)
        }
        xx, yy = np.meshgrid(self.coords['x'], self.coords['y'], indexing='ij')
        self.fields = {
            'u': np.sin(xx) * np.cos(yy),
            'v': np.cos(xx) * np.sin(yy),
            'p': np.sin(xx) * np.sin(yy)
        }
        self.lowfi_data = LowFiData(
            coordinates=self.coords,
            fields=self.fields,
            metadata={'test': True}
        )
        
    def test_linear_interpolation(self):
        """測試線性插值"""
        interpolator = SpatialInterpolator(method='linear')
        
        # 目標點
        target_points = np.array([
            [np.pi, 0.0],
            [2*np.pi, 0.5],
            [3*np.pi, -0.5]
        ])
        
        result = interpolator.interpolate_to_points(self.lowfi_data, target_points)
        
        assert 'u' in result
        assert 'v' in result  
        assert 'p' in result
        assert len(result['u']) == len(target_points)
        assert all(np.isfinite(result['u']))
        
    def test_rbf_interpolation(self):
        """測試 RBF 插值"""
        interpolator = SpatialInterpolator(method='rbf')
        
        target_points = np.array([
            [np.pi, 0.0],
            [2*np.pi, 0.5]
        ])
        
        result = interpolator.interpolate_to_points(self.lowfi_data, target_points)
        
        assert 'u' in result
        assert len(result['u']) == len(target_points)
        
    def test_idw_interpolation(self):
        """測試反距離權重插值"""
        interpolator = SpatialInterpolator(method='idw')
        
        target_points = np.array([
            [np.pi, 0.0],
            [2*np.pi, 0.5]
        ])
        
        result = interpolator.interpolate_to_points(self.lowfi_data, target_points)
        
        assert 'u' in result
        assert len(result['u']) == len(target_points)
        
    def test_grid_interpolation(self):
        """測試網格插值"""
        interpolator = SpatialInterpolator(method='linear')
        
        target_grid = {
            'x': np.linspace(0, 4*np.pi, 16),
            'y': np.linspace(-1, 1, 8)
        }
        
        result = interpolator.interpolate_to_grid(self.lowfi_data, target_grid)
        
        assert result['u'].shape == (16, 8)
        assert result['v'].shape == (16, 8)


class TestDownsampledDNSProcessor:
    """測試下採樣 DNS 處理器"""
    
    def setup_method(self):
        """設置高解析度測試資料"""
        nx, ny = 64, 32
        self.coords = {
            'x': np.linspace(0, 4*np.pi, nx),
            'y': np.linspace(-1, 1, ny)
        }
        xx, yy = np.meshgrid(self.coords['x'], self.coords['y'], indexing='ij')
        
        # 高頻成分豐富的場
        self.fields = {
            'u': np.sin(2*xx) * np.cos(3*yy) + 0.5*np.sin(8*xx) * np.cos(yy),
            'v': np.cos(3*xx) * np.sin(2*yy) + 0.3*np.cos(xx) * np.sin(6*yy),
            'p': np.sin(xx) * np.sin(yy) + 0.2*np.sin(4*xx) * np.sin(4*yy)
        }
        
        self.hifi_data = LowFiData(
            coordinates=self.coords,
            fields=self.fields,
            metadata={'resolution': 'high'}
        )
        
    def test_box_filter_downsampling(self):
        """測試箱型濾波下採樣"""
        processor = DownsampledDNSProcessor(
            downsample_factor=4,
            filter_type='box'
        )
        
        lowfi_data = processor.process(self.hifi_data)
        
        # 檢查解析度降低
        assert lowfi_data.get_spatial_dims() == (16, 8)
        assert 'downsample_factor' in lowfi_data.metadata
        assert lowfi_data.metadata['filter_type'] == 'box'
        
    def test_gaussian_filter_downsampling(self):
        """測試高斯濾波下採樣"""
        processor = DownsampledDNSProcessor(
            downsample_factor=2,
            filter_type='gaussian',
            preserve_energy=True
        )
        
        lowfi_data = processor.process(self.hifi_data)
        
        assert lowfi_data.get_spatial_dims() == (32, 16)
        assert 'energy_ratio' in lowfi_data.metadata
        
    def test_spectral_filter_downsampling(self):
        """測試頻譜濾波下採樣"""
        processor = DownsampledDNSProcessor(
            downsample_factor=4,
            filter_type='spectral'
        )
        
        lowfi_data = processor.process(self.hifi_data)
        
        assert lowfi_data.get_spatial_dims() == (16, 8)
        assert lowfi_data.metadata['filter_type'] == 'spectral'


class TestRANSReader:
    """測試 RANS 讀取器"""
    
    def test_rans_field_detection(self):
        """測試 RANS 場檢測"""
        # 創建模擬 NPZ 資料
        coords = {'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 8)}
        
        # 包含 RANS 特定場
        fields = {
            'u_mean': np.random.rand(10, 8),
            'v_mean': np.random.rand(10, 8),
            'uu': np.random.rand(10, 8),  # 雷諾應力
            'uv': np.random.rand(10, 8),
            'vv': np.random.rand(10, 8)
        }
        
        base_data = LowFiData(coordinates=coords, fields=fields, metadata={})
        
        # 模擬 NPZ 讀取器
        class MockNPZReader:
            def supports_format(self, filepath):
                return True
            def read(self, filepath):
                return base_data
        
        rans_reader = RANSReader(MockNPZReader())
        result = rans_reader.read("dummy.npz")
        
        assert result.metadata['data_type'] == 'RANS'
        assert result.metadata['has_reynolds_stress'] == True
        assert 'k' in result.fields  # 應該自動計算湍流動能


class TestLESReader:
    """測試 LES 讀取器"""
    
    def test_les_field_detection(self):
        """測試 LES 場檢測"""
        coords = {'x': np.linspace(0, 1, 10), 'y': np.linspace(0, 1, 8)}
        
        # 包含 LES 特定場
        fields = {
            'u': np.random.rand(10, 8),
            'v': np.random.rand(10, 8),
            'tau11': np.random.rand(10, 8),  # SGS 應力
            'tau12': np.random.rand(10, 8)
        }
        
        base_data = LowFiData(coordinates=coords, fields=fields, metadata={})
        
        class MockNPZReader:
            def supports_format(self, filepath):
                return True
            def read(self, filepath):
                return base_data
        
        les_reader = LESReader(MockNPZReader(), sgs_model='smagorinsky')
        result = les_reader.read("dummy.npz")
        
        assert result.metadata['data_type'] == 'LES'
        assert result.metadata['sgs_model'] == 'smagorinsky'
        assert 'nu_sgs' in result.fields  # 應該計算 SGS 黏度


class TestLowFiLoader:
    """測試主載入器"""
    
    def test_mock_data_creation(self):
        """測試模擬資料創建"""
        mock_data = create_mock_rans_data(nx=32, ny=16, case='channel')
        
        assert mock_data.get_spatial_dims() == (32, 16)
        assert 'u' in mock_data.fields
        assert 'v' in mock_data.fields
        assert 'p' in mock_data.fields
        assert mock_data.metadata['case'] == 'channel'
        
    def test_data_type_inference(self):
        """測試資料類型推斷"""
        loader = LowFiLoader()
        
        # RANS 資料 (需要至少 2 個 RANS 特徵字段)
        rans_data = create_mock_rans_data()
        rans_data.fields['uu'] = np.random.rand(*rans_data.get_spatial_dims())
        rans_data.fields['k'] = np.random.rand(*rans_data.get_spatial_dims())  # 新增第二個特徵
        
        inferred_type = loader._infer_data_type(rans_data)
        assert inferred_type == 'rans'
        
    def test_end_to_end_workflow(self):
        """測試端到端工作流程"""
        # 創建模擬資料
        mock_data = create_mock_rans_data(nx=32, ny=16)
        
        # 目標點
        target_points = np.random.rand(20, 2) * [4*np.pi, 2] + [0, -1]
        
        # 載入器
        loader = LowFiLoader()
        
        # 創建先驗資料
        prior_data = loader.create_prior_data(mock_data, target_points)
        
        assert 'u' in prior_data
        assert 'v' in prior_data
        assert 'p' in prior_data
        assert len(prior_data['u']) == len(target_points)
        
    def test_statistics_computation(self):
        """測試統計量計算"""
        loader = LowFiLoader()
        mock_data = create_mock_rans_data(nx=32, ny=16)
        
        stats = loader.get_statistics(mock_data)
        
        assert 'coord_x' in stats
        assert 'coord_y' in stats
        assert 'field_u' in stats
        assert 'field_v' in stats
        assert 'field_p' in stats
        
        # 檢查統計量欄位
        for field_stats in stats.values():
            assert 'mean' in field_stats
            assert 'std' in field_stats
            assert 'min' in field_stats
            assert 'max' in field_stats


def test_integration_with_file_io():
    """測試檔案 I/O 整合"""
    # 創建臨時 NPZ 檔案
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        
        # 創建測試資料
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 8)
        u = np.random.rand(10, 8)
        v = np.random.rand(10, 8)
        p = np.random.rand(10, 8)
        
        np.savez(tmp_path, x=x, y=y, u=u, v=v, p=p)
    
    try:
        # 載入資料
        loader = LowFiLoader()
        data = loader.load(tmp_path, data_type='auto')
        
        assert data is not None
        assert 'u' in data.fields
        assert 'v' in data.fields
        assert 'p' in data.fields
        assert data.metadata['format'] == 'NPZ'
        
    finally:
        # 清理臨時檔案
        os.unlink(tmp_path)


if __name__ == "__main__":
    """運行測試"""
    print("===== 低保真資料載入器測試開始 =====")
    
    # 基本功能測試
    print("測試 LowFiData 基本功能...")
    test_data = TestLowFiData()
    test_data.test_creation()
    test_data.test_with_time()
    test_data.test_physics_validation()
    print("✓ LowFiData 測試通過")
    
    # 插值器測試
    print("測試空間插值器...")
    test_interp = TestSpatialInterpolator()
    test_interp.setup_method()
    test_interp.test_linear_interpolation()
    test_interp.test_rbf_interpolation()
    test_interp.test_idw_interpolation()
    test_interp.test_grid_interpolation()
    print("✓ 空間插值器測試通過")
    
    # 下採樣處理器測試
    print("測試下採樣 DNS 處理器...")
    test_dns = TestDownsampledDNSProcessor()
    test_dns.setup_method()
    test_dns.test_box_filter_downsampling()
    test_dns.test_gaussian_filter_downsampling()
    test_dns.test_spectral_filter_downsampling()
    print("✓ 下採樣處理器測試通過")
    
    # RANS/LES 讀取器測試
    print("測試 RANS/LES 讀取器...")
    test_rans = TestRANSReader()
    test_rans.test_rans_field_detection()
    test_les = TestLESReader()
    test_les.test_les_field_detection()
    print("✓ RANS/LES 讀取器測試通過")
    
    # 主載入器測試
    print("測試主載入器...")
    test_loader = TestLowFiLoader()
    test_loader.test_mock_data_creation()
    test_loader.test_data_type_inference()
    test_loader.test_end_to_end_workflow()
    test_loader.test_statistics_computation()
    print("✓ 主載入器測試通過")
    
    # 檔案 I/O 測試
    print("測試檔案 I/O 整合...")
    test_integration_with_file_io()
    print("✓ 檔案 I/O 測試通過")
    
    print("===== 所有測試通過！ =====")
    
    # 展示使用範例
    print("\n===== 使用範例 =====")
    
    # 創建模擬 RANS 資料
    print("1. 創建模擬 RANS 資料")
    rans_data = create_mock_rans_data(nx=64, ny=32, case='channel')
    print(f"   資料維度: {rans_data.get_spatial_dims()}")
    print(f"   包含場: {list(rans_data.fields.keys())}")
    
    # 載入和插值
    print("2. 載入和插值到 PINN 訓練點")
    loader = LowFiLoader()
    target_points = np.random.rand(50, 2) * [4*np.pi, 2] + [0, -1]
    prior_data = loader.create_prior_data(rans_data, target_points)
    print(f"   插值點數: {len(target_points)}")
    print(f"   插值場: {list(prior_data.keys())}")
    
    # 統計量計算
    print("3. 統計量計算（用於 VS-PINN 尺度化）")
    stats = loader.get_statistics(rans_data)
    print(f"   速度 u 統計: mean={stats['field_u']['mean']:.3f}, std={stats['field_u']['std']:.3f}")
    print(f"   速度 v 統計: mean={stats['field_v']['mean']:.3f}, std={stats['field_v']['std']:.3f}")
    
    print("\n低保真資料載入器實現完成！")