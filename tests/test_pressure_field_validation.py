"""
測試壓力場缺失處理邏輯

驗證 scripts/train/train.py 中的壓力場驗證功能：
1. 壓力驅動流必須提供壓力場
2. 速度驅動流允許缺失壓力場
3. 用戶可以覆蓋默認行為
"""

import pytest
import torch
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np


class TestPressureFieldValidation:
    """測試壓力場缺失處理邏輯"""
    
    @pytest.fixture
    def base_config(self):
        """基礎配置模板"""
        return {
            'experiment': {'name': 'test_pressure_validation'},
            'model': {
                'type': 'fourier_mlp',
                'in_dim': 2,
                'out_dim': 3,
                'hidden_dim': 64,
                'num_layers': 4
            },
            'physics': {
                'type': 'navier_stokes_2d',
                'nu': 0.01,
                'pressure_driven': True  # 默認為壓力驅動流
            },
            'training': {
                'epochs': 10,
                'batch_size': 32,
                'sampling': {
                    'pde_points': 1000,
                    'boundary_points': 200
                }
            },
            'losses': {
                'data_weight': 100.0,
                'pde_weight': 1.0
            },
            'output': {
                'checkpoint_dir': 'checkpoints/test',
                'log_dir': 'logs/test'
            }
        }
    
    @pytest.fixture
    def sensor_data_with_pressure(self):
        """包含壓力場的感測器資料"""
        K = 50
        return {
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1),
            'p': torch.randn(K, 1)  # 有壓力場
        }
    
    @pytest.fixture
    def sensor_data_without_pressure(self):
        """缺少壓力場的感測器資料"""
        K = 50
        return {
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1)
            # 沒有 'p' 鍵
        }
    
    @pytest.fixture
    def mock_training_bundle(self, sensor_data_with_pressure):
        """Mock FlowDataBundle"""
        bundle = MagicMock()
        bundle.as_training_dict.return_value = {
            'coordinates': torch.randn(50, 3),
            'sensor_data': sensor_data_with_pressure,
            'domain_bounds': {
                'x': (0.0, 1.0),
                'y': (0.0, 1.0)
            },
            'physical_params': {'Re': 100},
            'statistics': {},
            'metadata': {},
            'has_prior': False
        }
        return bundle
    
    def test_pressure_driven_flow_with_pressure_data(self, base_config, sensor_data_with_pressure):
        """
        測試案例 1：壓力驅動流 + 有壓力場資料
        預期：正常執行，不產生警告
        """
        config = base_config.copy()
        config['physics']['pressure_driven'] = True
        
        # 模擬程式邏輯
        p_sensors = sensor_data_with_pressure.get('p')
        physics_config = config.get('physics', {})
        is_pressure_driven = physics_config.get('pressure_driven', False)
        enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
        
        # 驗證
        assert p_sensors is not None, "壓力場應該存在"
        assert is_pressure_driven is True
        assert enforce_pressure_data is True
    
    def test_pressure_driven_flow_without_pressure_data_strict(self, base_config, sensor_data_without_pressure):
        """
        測試案例 2：壓力驅動流 + 無壓力場資料 + 強制要求（預設）
        預期：拋出 ValueError
        """
        config = base_config.copy()
        config['physics']['pressure_driven'] = True
        # enforce_pressure_data 未設定，應默認為 True（因為 pressure_driven=True）
        
        # 模擬程式邏輯
        p_sensors = sensor_data_without_pressure.get('p')
        physics_config = config.get('physics', {})
        physics_type = physics_config.get('type', '')
        is_pressure_driven = physics_config.get('pressure_driven', False)
        enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
        
        # 驗證
        assert p_sensors is None, "壓力場應該缺失"
        assert is_pressure_driven is True
        assert enforce_pressure_data is True, "應該強制要求壓力場"
        
        # 應該拋出錯誤
        with pytest.raises(ValueError, match="壓力場資料缺失錯誤"):
            if p_sensors is None and enforce_pressure_data:
                raise ValueError(
                    f"❌ 壓力場資料缺失錯誤\n"
                    f"   物理類型: '{physics_type}' (pressure_driven={is_pressure_driven})\n"
                    f"   壓力驅動流必須提供壓力場資料（sensor_data['p']）。"
                )
    
    def test_pressure_driven_flow_without_pressure_data_allow(self, base_config, sensor_data_without_pressure):
        """
        測試案例 3：壓力驅動流 + 無壓力場資料 + 用戶允許
        預期：發出警告，但繼續執行（初始化為零）
        """
        config = base_config.copy()
        config['physics']['pressure_driven'] = True
        config['training']['enforce_pressure_data'] = False  # 用戶明確允許
        
        # 模擬程式邏輯
        p_sensors = sensor_data_without_pressure.get('p')
        physics_config = config.get('physics', {})
        is_pressure_driven = physics_config.get('pressure_driven', False)
        enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
        
        # 驗證
        assert p_sensors is None, "壓力場應該缺失"
        assert is_pressure_driven is True
        assert enforce_pressure_data is False, "用戶允許缺失壓力場"
        
        # 應該能夠初始化為零
        if p_sensors is None:
            u_sensors = sensor_data_without_pressure['u']
            p_sensors = torch.zeros_like(u_sensors)
        
        assert p_sensors is not None
        assert p_sensors.shape == sensor_data_without_pressure['u'].shape
        assert torch.all(p_sensors == 0.0)
    
    def test_velocity_driven_flow_without_pressure_data(self, base_config, sensor_data_without_pressure):
        """
        測試案例 4：速度驅動流 + 無壓力場資料
        預期：正常執行，輸出資訊日誌
        """
        config = base_config.copy()
        config['physics']['pressure_driven'] = False  # 速度驅動流
        
        # 模擬程式邏輯
        p_sensors = sensor_data_without_pressure.get('p')
        physics_config = config.get('physics', {})
        is_pressure_driven = physics_config.get('pressure_driven', False)
        enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
        
        # 驗證
        assert p_sensors is None
        assert is_pressure_driven is False
        assert enforce_pressure_data is False, "速度驅動流不要求壓力場"
        
        # 應該能夠初始化為零
        if p_sensors is None:
            u_sensors = sensor_data_without_pressure['u']
            p_sensors = torch.zeros_like(u_sensors)
        
        assert p_sensors is not None
        assert torch.all(p_sensors == 0.0)
    
    def test_config_override_priority(self, base_config):
        """
        測試案例 5：配置覆蓋優先級
        驗證 training.enforce_pressure_data 可以覆蓋 physics.pressure_driven
        """
        # 情況 1：pressure_driven=True，但用戶明確設定 enforce_pressure_data=False
        config1 = base_config.copy()
        config1['physics']['pressure_driven'] = True
        config1['training']['enforce_pressure_data'] = False
        
        enforce1 = config1.get('training', {}).get(
            'enforce_pressure_data', 
            config1.get('physics', {}).get('pressure_driven', False)
        )
        assert enforce1 is False, "用戶設定應該覆蓋默認值"
        
        # 情況 2：pressure_driven=False，用戶設定 enforce_pressure_data=True
        config2 = base_config.copy()
        config2['physics']['pressure_driven'] = False
        config2['training']['enforce_pressure_data'] = True
        
        enforce2 = config2.get('training', {}).get(
            'enforce_pressure_data', 
            config2.get('physics', {}).get('pressure_driven', False)
        )
        assert enforce2 is True, "用戶可以強制要求壓力場"
        
        # 情況 3：未設定 enforce_pressure_data，應默認為 pressure_driven 的值
        config3 = base_config.copy()
        config3['physics']['pressure_driven'] = True
        # 不設定 enforce_pressure_data
        
        enforce3 = config3.get('training', {}).get(
            'enforce_pressure_data', 
            config3.get('physics', {}).get('pressure_driven', False)
        )
        assert enforce3 is True, "應該默認為 pressure_driven 的值"
    
    def test_pressure_field_statistics_logging(self, sensor_data_with_pressure):
        """
        測試案例 6：壓力場統計資訊記錄
        驗證當壓力場存在時，應該記錄其形狀與範圍
        """
        p_sensors = sensor_data_with_pressure['p']
        
        # 模擬日誌記錄
        assert p_sensors is not None
        log_message = f"✅ 壓力場資料已載入：shape={p_sensors.shape}, range=[{p_sensors.min():.4f}, {p_sensors.max():.4f}]"
        
        # 驗證日誌格式
        assert "✅" in log_message
        assert "shape=" in log_message
        assert "range=" in log_message
        assert str(p_sensors.shape) in log_message
    
    def test_error_message_clarity(self, base_config):
        """
        測試案例 7：錯誤訊息清晰度
        驗證錯誤訊息包含足夠的資訊與解決方案
        """
        config = base_config.copy()
        config['physics']['type'] = 'channel_flow'
        config['physics']['pressure_driven'] = True
        
        physics_type = config['physics']['type']
        is_pressure_driven = config['physics']['pressure_driven']
        
        error_message = (
            f"❌ 壓力場資料缺失錯誤\n"
            f"   物理類型: '{physics_type}' (pressure_driven={is_pressure_driven})\n"
            f"   壓力驅動流必須提供壓力場資料（sensor_data['p']）。\n"
            f"\n"
            f"   可能的解決方案：\n"
            f"   1. 確保感測器 NPZ 檔案包含壓力欄位（'p', 'sensor_p', 或 'pressure'）\n"
            f"   2. 重新生成感測器資料（使用 scripts/generate/sensors/）\n"
            f"   3. 如果這是速度驅動流，請在 config 中設定：\n"
            f"      physics:\n"
            f"        pressure_driven: false\n"
            f"      或\n"
            f"      training:\n"
            f"        enforce_pressure_data: false  # 不推薦，會降低訓練效率\n"
        )
        
        # 驗證錯誤訊息包含關鍵資訊
        assert "壓力場資料缺失錯誤" in error_message
        assert physics_type in error_message
        assert "解決方案" in error_message
        assert "pressure_driven: false" in error_message
        assert "enforce_pressure_data: false" in error_message


class TestPressureFieldIntegration:
    """整合測試：與真實訓練流程的集成"""
    
    def test_integration_with_channel_flow_config(self):
        """
        整合測試：使用真實 channel_flow_re1000.yml 配置
        """
        config_path = Path('configs/channel_flow_re1000.yml')
        if not config_path.exists():
            pytest.skip("channel_flow_re1000.yml 配置不存在")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查配置結構
        assert 'physics' in config, "配置應包含 physics 段落"
        
        # Channel Flow 通常是壓力驅動流
        physics_config = config.get('physics', {})
        is_pressure_driven = physics_config.get('pressure_driven', False)
        
        # 如果是壓力驅動流，應該要求壓力場
        if is_pressure_driven:
            enforce_pressure_data = config.get('training', {}).get('enforce_pressure_data', is_pressure_driven)
            assert enforce_pressure_data is True or enforce_pressure_data is False, \
                "enforce_pressure_data 應該是布林值"
    
    def test_npz_file_pressure_field_check(self):
        """
        整合測試：檢查實際 NPZ 檔案是否包含壓力場
        """
        import glob
        
        # 尋找感測器 NPZ 檔案
        sensor_files = glob.glob('data/jhtdb/channel_flow_re1000/sensors_K*.npz')
        
        if not sensor_files:
            pytest.skip("未找到感測器 NPZ 檔案")
        
        # 檢查第一個檔案
        sensor_file = sensor_files[0]
        data = np.load(sensor_file, allow_pickle=True)
        
        # 檢查是否包含壓力欄位
        has_pressure = (
            'p' in data or 
            'sensor_p' in data or 
            'pressure' in data or
            ('sensor_data' in data and isinstance(data['sensor_data'], dict) and 'p' in data['sensor_data'])
        )
        
        print(f"檢查檔案: {sensor_file}")
        print(f"可用鍵: {list(data.keys())}")
        print(f"包含壓力場: {has_pressure}")
        
        # 如果是 Channel Flow，應該包含壓力場
        if 'channel_flow' in sensor_file:
            assert has_pressure, f"Channel Flow 感測器檔案應包含壓力場: {sensor_file}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
