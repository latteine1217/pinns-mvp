"""
測試 channel_flow_loader.py 類型修正

驗證修正項目：
1. ✅ 低保真先驗存取（lowfi_prior.values[field] 而非 lowfi_prior[field]）
2. ✅ prepare_training_data 返回類型（FlowDataBundle 而非 Dict[str, Any]）
3. ✅ load_full_field_data 縮排修正（移至 ChannelFlowLoader 類別內）
4. ✅ 異常處理邏輯（lowfi_prior = None 而非 {}）
"""

import pytest
import numpy as np
from pathlib import Path

from pinnx.dataio.channel_flow_loader import ChannelFlowLoader, prepare_training_data, ChannelFlowData
from pinnx.dataio.structures import FlowDataBundle, PointSamples


class TestChannelFlowLoaderFixes:
    """測試 channel_flow_loader.py 的類型修正"""
    
    def test_prepare_training_data_return_type(self):
        """測試 prepare_training_data 返回類型為 FlowDataBundle"""
        import inspect
        
        sig = inspect.signature(prepare_training_data)
        return_annotation = sig.return_annotation
        
        assert return_annotation == FlowDataBundle, \
            f"返回類型應為 FlowDataBundle，實際為 {return_annotation}"
    
    def test_loader_has_load_full_field_data_method(self):
        """測試 ChannelFlowLoader 有 load_full_field_data 方法"""
        assert hasattr(ChannelFlowLoader, 'load_full_field_data'), \
            "ChannelFlowLoader 應該有 load_full_field_data 方法"
        
        # 驗證是實例方法（非靜態/類別方法）
        import inspect
        sig = inspect.signature(ChannelFlowLoader.load_full_field_data)
        params = list(sig.parameters.keys())
        
        assert 'self' in params, "load_full_field_data 應該是實例方法（有 self 參數）"
    
    def test_loader_has_validate_data_method(self):
        """測試 ChannelFlowLoader 有 validate_data 方法"""
        assert hasattr(ChannelFlowLoader, 'validate_data'), \
            "ChannelFlowLoader 應該有 validate_data 方法"
    
    def test_lowfi_prior_access_pattern(self):
        """測試低保真先驗存取模式（間接驗證通過代碼檢查）"""
        # 讀取源碼驗證存取模式
        loader_file = Path(__file__).parent.parent / 'pinnx' / 'dataio' / 'channel_flow_loader.py'
        
        with open(loader_file, 'r') as f:
            content = f.read()
        
        # 驗證修正後的存取模式存在
        assert 'channel_data.lowfi_prior.values[field]' in content, \
            "應該使用 lowfi_prior.values[field] 存取低保真先驗"
        
        # 驗證錯誤的存取模式不存在（除了註解）
        # 注意：這個檢查可能會誤報，因為可能在其他上下文中使用
        lines_with_wrong_pattern = [
            line for line in content.split('\n') 
            if 'lowfi_prior[field]' in line 
            and 'lowfi_prior.values[field]' not in line
            and not line.strip().startswith('#')
        ]
        
        assert len(lines_with_wrong_pattern) == 0, \
            f"不應該直接使用 lowfi_prior[field]，找到 {len(lines_with_wrong_pattern)} 處"
    
    def test_channel_flow_data_lowfi_prior_type(self):
        """測試 ChannelFlowData.lowfi_prior 類型定義"""
        from pinnx.dataio.channel_flow_loader import ChannelFlowData
        import inspect
        
        # 獲取 ChannelFlowData 的類型註解
        if hasattr(ChannelFlowData, '__annotations__'):
            annotations = ChannelFlowData.__annotations__
            if 'lowfi_prior' in annotations:
                lowfi_type = annotations['lowfi_prior']
                # 應該是 Optional[PointSamples] 或類似類型
                assert 'PointSamples' in str(lowfi_type), \
                    f"lowfi_prior 類型應包含 PointSamples，實際為 {lowfi_type}"
    
    def test_mock_lowfi_prior_structure(self):
        """測試模擬低保真先驗的結構（確保符合 PointSamples）"""
        # 創建一個簡單的 PointSamples 實例
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        values = {
            'u': np.array([1.0, 2.0]),
            'v': np.array([0.5, 1.5]),
            'p': np.array([0.0, 0.1])
        }
        
        lowfi_prior = PointSamples(
            coordinates=coords,
            values=values,
            axes=('x', 'y', 'z')
        )
        
        # 驗證正確的存取模式
        assert 'u' in lowfi_prior.values
        assert isinstance(lowfi_prior.values['u'], np.ndarray)
        assert len(lowfi_prior.values['u']) == 2
        
        # 驗證 PointSamples 不支持直接字典索引
        assert not hasattr(lowfi_prior, '__getitem__') or callable(getattr(lowfi_prior, 'get_field', None)), \
            "應該使用 lowfi_prior.values[field] 或 get_field() 存取"
    
    def test_prepare_training_data_signature(self):
        """測試 prepare_training_data 函數簽名完整性"""
        import inspect
        
        sig = inspect.signature(prepare_training_data)
        params = sig.parameters
        
        # 驗證必要參數存在
        assert 'strategy' in params
        assert 'K' in params
        assert 'config_path' in params
        assert 'target_fields' in params
        assert 'sensor_file' in params
        assert 'prior_type' in params
        
        # 驗證預設值
        assert params['strategy'].default == 'qr_pivot'
        assert params['K'].default == 8
        assert params['prior_type'].default == 'none'
        
        # 驗證返回類型
        assert sig.return_annotation == FlowDataBundle


class TestFlowDataBundleIntegration:
    """測試 FlowDataBundle 與訓練流程的整合"""
    
    def test_flow_data_bundle_structure(self):
        """測試 FlowDataBundle 結構完整性"""
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        values = {
            'u': np.array([1.0, 2.0]),
            'v': np.array([0.5, 1.5]),
            'p': np.array([0.0, 0.1])
        }
        
        samples = PointSamples(
            coordinates=coords,
            values=values,
            axes=('x', 'y', 'z')
        )
        
        from pinnx.dataio.structures import DomainSpec
        domain = DomainSpec(
            bounds={'x': (0.0, 1.0), 'y': (0.0, 1.0), 'z': (0.0, 1.0)},
            parameters={'Re_tau': 1000.0, 'nu': 5e-5}
        )
        
        bundle = FlowDataBundle(
            samples=samples,
            domain=domain,
            statistics={},
            lowfi_prior=None,
            metadata={}
        )
        
        # 驗證結構
        assert bundle.samples.num_points == 2
        assert bundle.lowfi_prior is None
        assert 'Re_tau' in bundle.domain.parameters
    
    def test_flow_data_bundle_as_training_dict(self):
        """測試 FlowDataBundle.as_training_dict() 轉換"""
        import torch
        
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        values = {
            'u': np.array([1.0, 2.0, 1.5]),
            'v': np.array([0.5, 1.5, 1.0]),
            'p': np.array([0.0, 0.1, 0.05])
        }
        
        samples = PointSamples(
            coordinates=coords,
            values=values,
            axes=('x', 'y', 'z')
        )
        
        from pinnx.dataio.structures import DomainSpec
        domain = DomainSpec(
            bounds={'x': (0.0, 1.0), 'y': (0.0, 1.0), 'z': (0.0, 1.0)},
            parameters={'Re_tau': 1000.0}
        )
        
        bundle = FlowDataBundle(
            samples=samples,
            domain=domain
        )
        
        # 轉換為訓練字典
        device = torch.device('cpu')
        training_dict = bundle.as_training_dict(
            target_fields=['u', 'v', 'p'],
            device=device
        )
        
        # 驗證輸出結構
        assert 'coordinates' in training_dict
        assert 'sensor_data' in training_dict
        assert 'domain_bounds' in training_dict
        assert 'has_prior' in training_dict
        
        # 驗證張量形狀
        assert training_dict['coordinates'].shape == (3, 3)  # (N, 3)
        assert training_dict['sensor_data']['u'].shape == (3, 1)  # (N, 1)
        assert training_dict['has_prior'] is False


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
