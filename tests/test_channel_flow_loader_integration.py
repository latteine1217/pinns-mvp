#!/usr/bin/env python3
"""
測試 channel_flow_loader.py 的正確性與訓練整合
驗證新架構能否順利在訓練流程中使用
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from pinnx.dataio.channel_flow_loader import (
    ChannelFlowLoader,
    load_channel_flow_data,
    prepare_training_data
)
from pinnx.dataio.structures import (
    PointSamples,
    FlowDataBundle,
    DomainSpec,
    StructuredField
)


class TestChannelFlowLoaderBasics:
    """測試基本載入功能"""
    
    def test_loader_initialization(self):
        """測試載入器初始化"""
        loader = ChannelFlowLoader()
        assert loader is not None
        assert loader.config_path.exists()
        assert loader.cache_dir.exists()
    
    def test_load_sensor_data_qr_pivot(self):
        """測試載入 QR-pivot 感測點資料"""
        loader = ChannelFlowLoader()
        
        # 假設已有預生成的感測點資料
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 驗證資料結構
            assert data.samples is not None
            assert data.domain is not None
            assert len(data.sensor_points) > 0
            assert len(data.sensor_data) > 0
            
            # 驗證必要欄位
            assert 'u' in data.sensor_data
            assert 'v' in data.sensor_data
            assert 'p' in data.sensor_data
            
            # 驗證資料形狀一致性
            n_points = len(data.sensor_points)
            assert len(data.sensor_data['u']) == n_points
            assert len(data.sensor_data['v']) == n_points
            assert len(data.sensor_data['p']) == n_points
            
            print(f"✅ 成功載入 {n_points} 個 QR-pivot 感測點")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_load_sensor_data_with_w(self):
        """測試載入包含 w 分量的 3D 感測點資料"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 檢查是否有 w 分量（新架構支援）
            if 'w' in data.sensor_data:
                n_points = len(data.sensor_points)
                assert len(data.sensor_data['w']) == n_points
                print(f"✅ 檢測到 w 分量（3D 資料）")
            else:
                print(f"ℹ️  無 w 分量（2D 資料）")
                
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_compute_statistics(self):
        """測試統計資訊計算"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 驗證統計資訊
            assert data.statistics is not None
            assert len(data.statistics) > 0
            
            # 檢查必要的統計欄位
            for field in ['u', 'v', 'p']:
                if field in data.sensor_data:
                    assert field in data.statistics
                    stats = data.statistics[field]
                    assert 'min' in stats
                    assert 'max' in stats
                    assert 'mean' in stats
                    assert 'std' in stats
                    assert 'range' in stats
            
            print(f"✅ 統計資訊計算正確: {list(data.statistics.keys())}")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_domain_config_extraction(self):
        """測試域配置提取"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 驗證域配置
            domain_bounds = data.get_domain_bounds()
            assert 'x' in domain_bounds
            assert 'y' in domain_bounds
            
            # 驗證物理參數
            params = data.get_physical_parameters()
            assert 'Re_tau' in params or 'nu' in params
            
            print(f"✅ 域配置提取正確")
            print(f"   - 域邊界: {domain_bounds}")
            print(f"   - 物理參數: {params}")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")


class TestChannelFlowLoaderAdvanced:
    """測試進階功能"""
    
    def test_add_lowfi_prior_mock(self):
        """測試添加 mock 低保真先驗"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            data = loader.add_lowfi_prior(data, prior_type='mock')
            
            # 驗證先驗資料
            assert data.has_lowfi_prior()
            assert data.lowfi_prior is not None
            
            # 驗證先驗場
            prior_values = data.lowfi_prior.values
            assert 'u' in prior_values
            assert 'v' in prior_values
            assert 'p' in prior_values
            
            # 驗證形狀一致性
            n_points = len(data.sensor_points)
            assert len(prior_values['u']) == n_points
            assert len(prior_values['v']) == n_points
            assert len(prior_values['p']) == n_points
            
            print(f"✅ Mock 低保真先驗添加成功")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_prepare_for_training(self):
        """測試準備訓練資料格式"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 準備訓練資料（自動檢測 2D/3D）
            training_bundle = loader.prepare_for_training(data)
            
            # 驗證 FlowDataBundle 結構
            assert isinstance(training_bundle, FlowDataBundle)
            assert training_bundle.samples is not None
            assert training_bundle.domain is not None
            
            print(f"✅ 訓練資料準備成功")
            print(f"   - 目標場: {training_bundle.metadata.get('target_fields', [])}")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_noise_and_dropout(self):
        """測試噪聲與丟失功能"""
        loader = ChannelFlowLoader()
        
        try:
            # 原始資料
            data_clean = loader.load_sensor_data(strategy='qr_pivot', K=16)
            n_clean = len(data_clean.sensor_points)
            
            # 添加噪聲
            data_noisy = loader.load_sensor_data(
                strategy='qr_pivot', 
                K=16, 
                noise_sigma=0.01
            )
            assert len(data_noisy.sensor_points) == n_clean
            assert 'noise_sigma' in data_noisy.selection_info
            
            # 添加丟失
            data_dropout = loader.load_sensor_data(
                strategy='qr_pivot', 
                K=16, 
                dropout_prob=0.2
            )
            assert len(data_dropout.sensor_points) < n_clean
            assert 'dropout_prob' in data_dropout.selection_info
            
            print(f"✅ 噪聲與丟失功能正常")
            print(f"   - 原始: {n_clean} 點")
            print(f"   - 丟失後: {len(data_dropout.sensor_points)} 點")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")


class TestTrainingIntegration:
    """測試與訓練流程的整合"""
    
    def test_prepare_training_data_convenience(self):
        """測試便利函數"""
        try:
            training_bundle = prepare_training_data(
                strategy='qr_pivot',
                K=8,
                prior_type='none'  # 不添加先驗
            )
            
            # 驗證返回類型
            assert isinstance(training_bundle, FlowDataBundle)
            
            print(f"✅ 便利函數 prepare_training_data 正常工作")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_as_training_dict_conversion(self):
        """測試轉換為訓練字典"""
        try:
            training_bundle = prepare_training_data(
                strategy='qr_pivot',
                K=8,
                prior_type='none'
            )
            
            # 轉換為訓練字典
            device = torch.device('cpu')
            training_dict = training_bundle.as_training_dict(
                target_fields=['u', 'v', 'p'],
                device=device,
                include_w=False
            )
            
            # 驗證必要鍵
            assert 'coordinates' in training_dict
            assert 'sensor_data' in training_dict
            assert 'domain_bounds' in training_dict
            assert 'physical_params' in training_dict
            assert 'statistics' in training_dict
            
            # 驗證 tensor 類型
            assert isinstance(training_dict['coordinates'], torch.Tensor)
            assert isinstance(training_dict['sensor_data']['u'], torch.Tensor)
            assert isinstance(training_dict['sensor_data']['v'], torch.Tensor)
            assert isinstance(training_dict['sensor_data']['p'], torch.Tensor)
            
            # 驗證設備
            assert training_dict['coordinates'].device == device
            
            print(f"✅ as_training_dict 轉換正確")
            print(f"   - 座標形狀: {training_dict['coordinates'].shape}")
            print(f"   - 感測點數: {len(training_dict['sensor_data']['u'])}")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_3d_training_dict_with_w(self):
        """測試 3D 訓練字典（含 w 分量）"""
        try:
            training_bundle = prepare_training_data(
                strategy='qr_pivot',
                K=8,
                target_fields=['u', 'v', 'w', 'p'],
                prior_type='none'
            )
            
            device = torch.device('cpu')
            training_dict = training_bundle.as_training_dict(
                target_fields=['u', 'v', 'w', 'p'],
                device=device,
                include_w=True
            )
            
            # 驗證 w 分量
            assert 'w' in training_dict['sensor_data']
            assert isinstance(training_dict['sensor_data']['w'], torch.Tensor)
            
            print(f"✅ 3D 訓練字典（含 w）轉換正確")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_load_full_field_data(self):
        """測試載入完整場資料"""
        loader = ChannelFlowLoader()
        
        try:
            full_field = loader.load_full_field_data()
            
            # 驗證 StructuredField 類型
            assert isinstance(full_field, StructuredField)
            assert full_field.grid is not None
            assert full_field.fields is not None
            
            # 驗證必要場
            assert 'u' in full_field.fields
            assert 'v' in full_field.fields
            assert 'p' in full_field.fields
            
            print(f"✅ 完整場資料載入成功")
            print(f"   - 網格形狀: {full_field.grid.shape}")
            print(f"   - 可用場: {list(full_field.fields.keys())}")
            
        except FileNotFoundError as e:
            pytest.skip(f"完整場資料未預生成: {e}")


class TestDataValidation:
    """測試資料驗證功能"""
    
    def test_validate_data(self):
        """測試資料完整性驗證"""
        loader = ChannelFlowLoader()
        
        try:
            data = loader.load_sensor_data(strategy='qr_pivot', K=8)
            
            # 執行驗證
            checks = loader.validate_data(data)
            
            # 驗證基本檢查通過
            assert checks['has_sensor_points'] == True
            assert checks['has_sensor_data'] == True
            assert checks['has_domain_config'] == True
            
            # 驗證數值合理性
            for field in ['u', 'v', 'p']:
                if field in data.sensor_data:
                    assert checks[f'{field}_finite'] == True
                    assert checks[f'{field}_dimension_match'] == True
            
            print(f"✅ 資料驗證通過")
            print(f"   - 檢查項目: {len(checks)}")
            print(f"   - 失敗項目: {sum(1 for v in checks.values() if not v)}")
            
        except FileNotFoundError as e:
            pytest.skip(f"感測點資料未預生成: {e}")
    
    def test_get_available_datasets(self):
        """測試獲取可用資料集列表"""
        loader = ChannelFlowLoader()
        
        available = loader.get_available_datasets()
        
        print(f"✅ 可用資料集: {available}")
        
        # 如果沒有資料集，提示用戶
        if len(available) == 0:
            pytest.skip("無可用資料集，請先執行 scripts/fetch_channel_flow.py")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
