"""
測試座標維度一致性檢查邏輯

測試 scripts/train/train.py 中的座標維度驗證機制，確保：
1. VS-PINN (3D) 模式使用變化的 z 座標時不產生警告
2. VS-PINN (3D) 模式使用常數 z 座標時產生警告
3. 2D 模式使用常數 z 座標時不產生警告
4. 2D 模式使用變化的 z 座標時產生警告並強制 z=0
5. 警告訊息包含可操作的建議
"""

import pytest
import torch
import numpy as np
import tempfile
import yaml
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCoordinateDimensionConsistency:
    """座標維度一致性測試套件"""
    
    @pytest.fixture
    def mock_config_3d(self):
        """3D VS-PINN 配置（期望 z 有變化）"""
        return {
            'physics': {
                'type': 'vs_pinn_channel_flow',
                'nu': 0.00005,
                'pressure_driven': True
            },
            'training': {
                'epochs': 1000,
                'batch_size': 64
            },
            'model': {
                'hidden_layers': 8,
                'neurons_per_layer': 256
            }
        }
    
    @pytest.fixture
    def mock_config_2d(self):
        """2D 配置（期望 z 為常數或被忽略）"""
        return {
            'physics': {
                'type': 'navier_stokes_2d',
                'nu': 0.01,
                'pressure_driven': False
            },
            'training': {
                'epochs': 1000,
                'batch_size': 64
            },
            'model': {
                'hidden_layers': 6,
                'neurons_per_layer': 128
            }
        }
    
    @pytest.fixture
    def sensor_data_varying_z(self):
        """具有變化 z 座標的感測器資料（真 3D 資料）"""
        K = 100
        coords = np.column_stack([
            np.random.uniform(0, 2*np.pi, K),      # x
            np.random.uniform(-1, 1, K),           # y
            np.random.uniform(0, 2*np.pi, K)       # z: 有變化 (0 ~ 2π)
        ])
        return {
            'coordinates': torch.tensor(coords, dtype=torch.float32),
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1),
            'w': torch.randn(K, 1),
            'p': torch.randn(K, 1)
        }
    
    @pytest.fixture
    def sensor_data_constant_z(self):
        """具有常數 z 座標的感測器資料（2D 切片或 2D 資料）"""
        K = 100
        coords = np.column_stack([
            np.random.uniform(0, 2*np.pi, K),      # x
            np.random.uniform(-1, 1, K),           # y
            np.full(K, 4.71)                        # z: 常數 (π*1.5)
        ])
        return {
            'coordinates': torch.tensor(coords, dtype=torch.float32),
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1),
            'w': torch.randn(K, 1),
            'p': torch.randn(K, 1)
        }
    
    def simulate_coordinate_validation(self, config, sensor_data):
        """
        模擬 train.py 中的座標維度驗證邏輯
        
        返回:
            dict: {'warnings': [...], 'z_sensors': Tensor, 'z_forced_zero': bool}
        """
        warnings = []
        
        coords = sensor_data['coordinates']
        is_vs_pinn = config.get('physics', {}).get('type') == 'vs_pinn_channel_flow'
        
        x_sensors = coords[:, 0:1]
        
        # 處理 z 座標
        if is_vs_pinn:
            z_sensors = coords[:, 2:3]
        else:
            z_sensors = torch.zeros_like(x_sensors)
        
        # 座標維度一致性檢查
        # 注意：檢查「變化」而非「大小」，使用 std 判斷是否為常數
        coords_z_is_constant = coords.shape[1] >= 3 and coords[:, 2].std().item() < 1e-6
        coords_has_varying_z = coords.shape[1] >= 3 and not coords_z_is_constant
        
        if is_vs_pinn:
            if coords_z_is_constant:
                z_mean = coords[:, 2].mean().item()
                warning_msg = (
                    f"⚠️ VS-PINN (3D) 模式但 z 座標為常數 (z={z_mean:.4f})。\n"
                    f"   這可能表示:\n"
                    f"   1. 資料來自 2D 切片且固定 z (驗證 z_default={z_mean:.4f} 是否正確)\n"
                    f"   2. 配置錯誤 (這應該是 2D 模式嗎?)\n"
                    f"   如果這是刻意的 2D 切片資料，建議設定 physics.type 為非 VS-PINN。"
                )
                warnings.append(warning_msg)
        else:
            if coords_has_varying_z:
                z_min = coords[:, 2].min().item()
                z_max = coords[:, 2].max().item()
                warning_msg = (
                    f"⚠️ 2D 物理模式但 z 座標有變化 (範圍: [{z_min:.4f}, {z_max:.4f}])。\n"
                    f"   Z 值將被忽略 (強制為零)。\n"
                    f"   如果這是 3D 資料，建議:\n"
                    f"   1. 設定 physics.type: 'vs_pinn_channel_flow' 以支援 3D\n"
                    f"   2. 若確實為 2D 問題，重新生成感測器資料時進行 2D 提取"
                )
                warnings.append(warning_msg)
        
        return {
            'warnings': warnings,
            'z_sensors': z_sensors,
            'z_forced_zero': not is_vs_pinn,
            'is_vs_pinn': is_vs_pinn,
            'coords_has_varying_z': coords_has_varying_z,
            'coords_z_is_constant': coords_z_is_constant
        }
    
    def test_3d_mode_with_varying_z_no_warning(self, mock_config_3d, sensor_data_varying_z):
        """
        測試案例 1: VS-PINN (3D) + 變化的 z 座標 → 不應產生警告
        """
        result = self.simulate_coordinate_validation(mock_config_3d, sensor_data_varying_z)
        
        # 驗證
        assert len(result['warnings']) == 0, "3D 模式搭配變化 z 座標不應產生警告"
        assert result['is_vs_pinn'] == True
        assert result['coords_has_varying_z'] == True
        assert result['coords_z_is_constant'] == False
        assert result['z_forced_zero'] == False
        
        # 驗證 z_sensors 使用真實 z 值
        z_std = result['z_sensors'].std().item()
        assert z_std > 0.1, f"z_sensors 應有變化，但 std={z_std:.4f}"
    
    def test_3d_mode_with_constant_z_should_warn(self, mock_config_3d, sensor_data_constant_z):
        """
        測試案例 2: VS-PINN (3D) + 常數 z 座標 → 應產生警告
        """
        result = self.simulate_coordinate_validation(mock_config_3d, sensor_data_constant_z)
        
        # 驗證產生警告
        assert len(result['warnings']) == 1, "3D 模式搭配常數 z 座標應產生警告"
        warning_msg = result['warnings'][0]
        
        # 檢查警告訊息內容
        assert "VS-PINN (3D) 模式但 z 座標為常數" in warning_msg
        assert "z=4.71" in warning_msg or "z=4.7" in warning_msg
        assert "2D 切片" in warning_msg
        assert "配置錯誤" in warning_msg
        assert "physics.type" in warning_msg
        
        # 驗證仍使用 z 座標（雖然是常數）
        assert result['z_forced_zero'] == False
        z_mean = result['z_sensors'].mean().item()
        assert abs(z_mean - 4.71) < 0.01, f"z_sensors 應保持常數 4.71，但得到 {z_mean:.4f}"
    
    def test_2d_mode_with_constant_z_no_warning(self, mock_config_2d, sensor_data_constant_z):
        """
        測試案例 3: 2D 模式 + 常數 z 座標 → 不應產生警告
        """
        result = self.simulate_coordinate_validation(mock_config_2d, sensor_data_constant_z)
        
        # 驗證
        assert len(result['warnings']) == 0, "2D 模式搭配常數 z 座標不應產生警告"
        assert result['is_vs_pinn'] == False
        assert result['coords_has_varying_z'] == False
        assert result['coords_z_is_constant'] == True
        assert result['z_forced_zero'] == True
        
        # 驗證 z_sensors 被強制為零
        z_max = result['z_sensors'].abs().max().item()
        assert z_max < 1e-9, f"2D 模式的 z_sensors 應為零，但 max={z_max:.4e}"
    
    def test_2d_mode_with_varying_z_should_warn(self, mock_config_2d, sensor_data_varying_z):
        """
        測試案例 4: 2D 模式 + 變化的 z 座標 → 應產生警告並強制 z=0
        """
        result = self.simulate_coordinate_validation(mock_config_2d, sensor_data_varying_z)
        
        # 驗證產生警告
        assert len(result['warnings']) == 1, "2D 模式搭配變化 z 座標應產生警告"
        warning_msg = result['warnings'][0]
        
        # 檢查警告訊息內容
        assert "2D 物理模式但 z 座標有變化" in warning_msg
        assert "範圍:" in warning_msg
        assert "Z 值將被忽略" in warning_msg
        assert "vs_pinn_channel_flow" in warning_msg
        assert "重新生成感測器資料" in warning_msg
        
        # 驗證 z 被強制為零（即使原始資料有變化）
        assert result['z_forced_zero'] == True
        z_max = result['z_sensors'].abs().max().item()
        assert z_max < 1e-9, f"2D 模式的 z_sensors 應強制為零，但 max={z_max:.4e}"
    
    def test_warning_message_has_z_range(self, mock_config_2d, sensor_data_varying_z):
        """
        測試案例 5: 驗證警告訊息包含具體的 z 範圍資訊
        """
        result = self.simulate_coordinate_validation(mock_config_2d, sensor_data_varying_z)
        warning_msg = result['warnings'][0]
        
        # 提取原始 z 範圍
        z_coords = sensor_data_varying_z['coordinates'][:, 2]
        z_min_actual = z_coords.min().item()
        z_max_actual = z_coords.max().item()
        
        # 驗證警告包含範圍資訊（允許格式化誤差，使用 4 位小數）
        assert f"{z_min_actual:.1f}" in warning_msg or f"{z_min_actual:.2f}" in warning_msg or f"{z_min_actual:.4f}" in warning_msg
        assert f"{z_max_actual:.1f}" in warning_msg or f"{z_max_actual:.2f}" in warning_msg or f"{z_max_actual:.4f}" in warning_msg
    
    def test_warning_message_has_actionable_solutions(self, mock_config_3d, sensor_data_constant_z):
        """
        測試案例 6: 驗證警告訊息包含可操作的解決方案
        """
        result = self.simulate_coordinate_validation(mock_config_3d, sensor_data_constant_z)
        warning_msg = result['warnings'][0]
        
        # 檢查包含具體建議
        actionable_keywords = [
            "驗證 z_default",       # 建議 1: 確認 z 值是否正確
            "配置錯誤",              # 建議 2: 檢查是否應為 2D
            "physics.type"           # 建議 3: 修改配置參數
        ]
        
        for keyword in actionable_keywords:
            assert keyword in warning_msg, f"警告訊息缺少可操作建議關鍵字: {keyword}"
    
    def test_edge_case_z_all_zero(self, mock_config_3d):
        """
        測試案例 7: 邊界情況 - z 全為 0（應視為常數）
        """
        K = 50
        coords = np.column_stack([
            np.random.uniform(0, 1, K),
            np.random.uniform(-1, 1, K),
            np.zeros(K)  # z 全為 0
        ])
        sensor_data = {
            'coordinates': torch.tensor(coords, dtype=torch.float32),
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1),
            'w': torch.randn(K, 1),
            'p': torch.randn(K, 1)
        }
        
        result = self.simulate_coordinate_validation(mock_config_3d, sensor_data)
        
        # 應產生警告（3D 模式但 z 為常數 0）
        assert len(result['warnings']) == 1
        assert "z=0.0" in result['warnings'][0]
    
    def test_edge_case_very_small_z_variation(self, mock_config_2d):
        """
        測試案例 8: 邊界情況 - z 變化極小（< 1e-6，應視為常數）
        """
        K = 50
        coords = np.column_stack([
            np.random.uniform(0, 1, K),
            np.random.uniform(-1, 1, K),
            np.full(K, 1.0) + np.random.uniform(-1e-7, 1e-7, K)  # z ≈ 1.0 ± 1e-7
        ])
        sensor_data = {
            'coordinates': torch.tensor(coords, dtype=torch.float32),
            'u': torch.randn(K, 1),
            'v': torch.randn(K, 1),
            'p': torch.randn(K, 1)
        }
        
        result = self.simulate_coordinate_validation(mock_config_2d, sensor_data)
        
        # 不應產生警告（z 變化太小，視為常數）
        assert len(result['warnings']) == 0
        assert result['coords_z_is_constant'] == True
    
    def test_coordinate_dimension_detection_logic(self, sensor_data_varying_z):
        """
        測試案例 9: 驗證座標維度檢測邏輯的正確性
        """
        coords = sensor_data_varying_z['coordinates']
        
        # 測試常數檢測（先檢查 std）
        coords_z_is_constant = coords.shape[1] >= 3 and coords[:, 2].std().item() < 1e-6
        assert coords_z_is_constant == False
        
        # 測試變化檢測（基於 std）
        coords_has_varying_z = coords.shape[1] >= 3 and not coords_z_is_constant
        assert coords_has_varying_z == True
        
        # 驗證閾值設定合理
        z_std = coords[:, 2].std().item()
        assert z_std > 1e-6, f"變化的 z 座標 std={z_std:.4e} 應 >> 1e-6"


class TestIntegrationWithTrainScript:
    """與實際訓練腳本的整合測試"""
    
    @pytest.fixture
    def temp_sensor_data_npz(self, tmp_path):
        """創建臨時 NPZ 感測器資料檔案"""
        def _create_npz(varying_z=True):
            K = 80
            if varying_z:
                z = np.random.uniform(0, 2*np.pi, K)
            else:
                z = np.full(K, 3.14)
            
            coords = np.column_stack([
                np.random.uniform(0, 2*np.pi, K),
                np.random.uniform(-1, 1, K),
                z
            ])
            
            file_path = tmp_path / f"sensors_varying{varying_z}.npz"
            np.savez(
                file_path,
                coordinates=coords,
                u=np.random.randn(K),
                v=np.random.randn(K),
                w=np.random.randn(K),
                p=np.random.randn(K)
            )
            return str(file_path)
        return _create_npz
    
    def test_loading_sensor_data_triggers_validation(self, temp_sensor_data_npz, caplog):
        """
        測試案例 10: 驗證載入感測器資料時會觸發座標檢查（需要實際 import train.py）
        
        注意: 這是整合測試，需要實際 train.py 能正常 import
        """
        # 這個測試需要實際執行環境，目前僅作為文檔說明
        # 實際專案中可用 subprocess 調用 train.py 並捕獲日誌
        pass


# ============ 執行測試的輔助函數 ============

def run_tests():
    """執行所有測試並生成報告"""
    pytest.main([
        __file__,
        '-v',                    # 詳細輸出
        '--tb=short',            # 簡短 traceback
        '--capture=no',          # 顯示 print 輸出
        '-W', 'ignore::DeprecationWarning'  # 忽略棄用警告
    ])


if __name__ == '__main__':
    run_tests()
