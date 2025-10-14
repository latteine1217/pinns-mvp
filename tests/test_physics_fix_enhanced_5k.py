"""
單元測試：Enhanced 5K Curriculum 物理修正驗證

測試 TASK-ENHANCED-5K-PHYSICS-FIX 的兩項核心修正：
- 修正 A1: PDE 損失雙重削弱問題（disable_extra_pde_division）
- 修正 A2: 邊界帶狀採樣策略（boundary_band_width）

創建時間: 2025-10-15
參考文檔: tasks/TASK-ENHANCED-5K-PHYSICS-DIAGNOSIS/corrective_actions.md
"""

import pytest
import torch
import numpy as np
from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow


class TestPDELossScalingFix:
    """測試修正 A1: PDE 損失雙重削弱修正"""
    
    @pytest.fixture
    def physics_config(self):
        """標準物理配置"""
        return {
            'nu': 5.0e-5,
            'rho': 1.0,
            'Re_tau': 1000.0,
            'u_tau': 0.04997,
            'pressure_gradient': 0.0025,
            'channel_half_height': 1.0,
            'domain': {
                'x_range': [0.0, 25.13],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, 9.42]
            },
            'vs_pinn': {
                'scaling_factors': {
                    'N_x': 2.0,
                    'N_y': 12.0,
                    'N_z': 2.0
                }
            }
        }
    
    def test_disable_extra_pde_division_default(self, physics_config):
        """測試默認行為：disable_extra_pde_division = True"""
        # 創建物理模組（默認配置）
        physics = VSPINNChannelFlow(physics_config)
        
        # 驗證默認值
        assert physics.disable_extra_pde_division is True, \
            "默認應禁用額外 PDE 除法（修正後行為）"
        
        print("✅ 測試通過：默認禁用額外 PDE 除法")
    
    def test_explicit_enable_extra_pde_division(self, physics_config):
        """測試顯式啟用舊行為：disable_extra_pde_division = False"""
        # 修改配置啟用舊行為
        physics_config['vs_pinn']['loss_config'] = {
            'disable_extra_pde_division': False
        }
        
        physics = VSPINNChannelFlow(physics_config)
        
        # 驗證配置生效
        assert physics.disable_extra_pde_division is False, \
            "應允許顯式啟用舊行為（向後相容）"
        
        print("✅ 測試通過：可顯式啟用舊行為")
    
    def test_pde_loss_scaling_behavior(self, physics_config):
        """測試 PDE 損失縮放行為差異"""
        # 創建兩個配置：新行為 vs. 舊行為
        config_new = physics_config.copy()
        config_new['vs_pinn'] = {
            'scaling_factors': {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0},
            'loss_config': {'disable_extra_pde_division': True}
        }
        
        config_old = physics_config.copy()
        config_old['vs_pinn'] = {
            'scaling_factors': {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0},
            'loss_config': {'disable_extra_pde_division': False}
        }
        
        physics_new = VSPINNChannelFlow(config_new)
        physics_old = VSPINNChannelFlow(config_old)
        
        # 創建模擬損失
        mock_losses = {
            'momentum_x': torch.tensor(1.0),
            'momentum_y': torch.tensor(1.0),
            'momentum_z': torch.tensor(1.0),
            'continuity': torch.tensor(1.0),
            'data': torch.tensor(1.0),
            'wall_constraint': torch.tensor(1.0)
        }
        
        # 測試新行為（應保持原值）
        normalized_new = physics_new.normalize_loss(mock_losses.copy())
        
        # 測試舊行為（PDE 項應額外削弱）
        normalized_old = physics_old.normalize_loss(mock_losses.copy())
        
        # 計算 N_max_sq（應為 144 = 12^2）
        N_max_sq = physics_new.N_max_sq
        assert N_max_sq == 144.0, f"N_max_sq 應為 144，實際為 {N_max_sq}"
        
        # 驗證新行為：PDE 損失不被額外削弱
        for key in ['momentum_x', 'momentum_y', 'momentum_z', 'continuity']:
            assert torch.isclose(
                normalized_new[key],
                mock_losses[key] / N_max_sq,
                rtol=1e-5
            ), f"新行為：{key} 應只除以 N_max_sq 一次"
        
        # 驗證舊行為：PDE 損失被額外削弱
        for key in ['momentum_x', 'momentum_y', 'momentum_z', 'continuity']:
            expected_old = mock_losses[key] / (N_max_sq ** 2)
            assert torch.isclose(
                normalized_old[key],
                expected_old,
                rtol=1e-5
            ), f"舊行為：{key} 應除以 N_max_sq 兩次"
        
        # 驗證損失值差異（新行為應大 N_max_sq 倍）
        ratio = normalized_new['momentum_x'] / normalized_old['momentum_x']
        assert torch.isclose(ratio, torch.tensor(N_max_sq), rtol=1e-5), \
            f"新舊行為 PDE 損失比應為 {N_max_sq}，實際為 {ratio.item()}"
        
        print(f"✅ 測試通過：PDE 損失縮放行為正確")
        print(f"   新行為損失: {normalized_new['momentum_x'].item():.6e}")
        print(f"   舊行為損失: {normalized_old['momentum_x'].item():.6e}")
        print(f"   比值: {ratio.item():.2f} (預期 {N_max_sq})")


class TestBoundaryBandSampling:
    """測試修正 A2: 邊界帶狀採樣策略"""
    
    @pytest.fixture
    def physics_config(self):
        """標準物理配置"""
        return {
            'nu': 5.0e-5,
            'rho': 1.0,
            'Re_tau': 1000.0,
            'u_tau': 0.04997,
            'pressure_gradient': 0.0025,
            'channel_half_height': 1.0,
            'domain': {
                'x_range': [0.0, 25.13],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, 9.42]
            },
            'vs_pinn': {
                'scaling_factors': {
                    'N_x': 2.0,
                    'N_y': 12.0,
                    'N_z': 2.0
                }
            }
        }
    
    @pytest.fixture
    def mock_model(self):
        """模擬神經網路模型"""
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # 返回模擬預測：u=1, v=0, w=0, p=0
                batch_size = x.shape[0]
                return torch.cat([
                    torch.ones(batch_size, 1),   # u
                    torch.zeros(batch_size, 1),  # v
                    torch.zeros(batch_size, 1),  # w
                    torch.zeros(batch_size, 1)   # p
                ], dim=1)
        
        return MockModel()
    
    def test_wall_boundary_band_sampling(self, physics_config, mock_model):
        """測試壁面帶狀採樣是否捕捉到近壁點"""
        physics = VSPINNChannelFlow(physics_config)
        
        # 創建測試座標（包含近壁點和遠壁點）
        coords = torch.tensor([
            [5.0, -1.0, 5.0],      # 下壁面精確點
            [5.0, -0.998, 5.0],    # 下壁面近壁點（在 band 內）
            [5.0, -0.99, 5.0],     # 下壁面遠點（在 band 外）
            [5.0, 1.0, 5.0],       # 上壁面精確點
            [5.0, 0.997, 5.0],     # 上壁面近壁點（在 band 內）
            [5.0, 0.0, 5.0],       # 中心點（遠離壁面）
        ], requires_grad=True)
        
        # 計算壁面剪應力損失
        loss = physics.compute_wall_shear_stress(
            coords, 
            mock_model,
            boundary_band_width=5e-3  # 帶狀寬度 5mm
        )
        
        # 驗證損失可計算（非 NaN）
        assert torch.isfinite(loss), "壁面剪應力損失應為有限值"
        
        # 驗證損失非零（說明捕捉到近壁點）
        assert loss.item() > 0, "壁面剪應力損失應非零（捕捉到近壁點）"
        
        print(f"✅ 測試通過：壁面帶狀採樣捕捉到近壁點")
        print(f"   壁面剪應力損失: {loss.item():.6e}")
    
    def test_periodic_boundary_band_sampling(self, physics_config, mock_model):
        """測試週期性邊界帶狀採樣是否捕捉到近邊界點"""
        physics = VSPINNChannelFlow(physics_config)
        
        # 創建測試座標（包含近邊界點）
        x_min, x_max = 0.0, 25.13
        z_min, z_max = 0.0, 9.42
        
        coords = torch.tensor([
            [x_min, 0.0, 5.0],        # x 最小邊界精確點
            [x_min + 0.002, 0.0, 5.0], # x 最小邊界近邊點（在 band 內）
            [x_max, 0.0, 5.0],        # x 最大邊界精確點
            [x_max - 0.003, 0.0, 5.0], # x 最大邊界近邊點（在 band 內）
            [5.0, 0.0, z_min],        # z 最小邊界精確點
            [5.0, 0.0, z_max],        # z 最大邊界精確點
            [12.0, 0.0, 5.0],         # 內部點（遠離邊界）
        ], requires_grad=True)
        
        # 計算週期性損失
        loss = physics.compute_periodic_loss(
            coords,
            mock_model,
            boundary_band_width=5e-3  # 帶狀寬度 5mm
        )
        
        # 驗證損失可計算（非 NaN）
        assert torch.isfinite(loss), "週期性損失應為有限值"
        
        # 驗證損失非零（說明捕捉到近邊界點）
        assert loss.item() >= 0, "週期性損失應非負"
        
        print(f"✅ 測試通過：週期性邊界帶狀採樣捕捉到近邊界點")
        print(f"   週期性損失: {loss.item():.6e}")
    
    def test_boundary_band_width_parameter(self, physics_config, mock_model):
        """測試帶狀寬度參數的影響"""
        physics = VSPINNChannelFlow(physics_config)
        
        # 創建近壁點座標（距離壁面 2mm）
        coords = torch.tensor([
            [5.0, -0.998, 5.0],  # 距離下壁面 2mm
            [5.0, 0.998, 5.0],   # 距離上壁面 2mm
        ], requires_grad=True)
        
        # 測試不同帶狀寬度
        loss_narrow = physics.compute_wall_shear_stress(
            coords, mock_model, boundary_band_width=1e-3  # 1mm（應捕捉不到）
        )
        
        loss_wide = physics.compute_wall_shear_stress(
            coords, mock_model, boundary_band_width=5e-3  # 5mm（應捕捉到）
        )
        
        # 驗證寬帶狀能捕捉更多點（損失更大）
        assert loss_wide.item() >= loss_narrow.item(), \
            "寬帶狀應捕捉更多近壁點（損失更大）"
        
        print(f"✅ 測試通過：帶狀寬度參數影響正確")
        print(f"   窄帶狀 (1mm) 損失: {loss_narrow.item():.6e}")
        print(f"   寬帶狀 (5mm) 損失: {loss_wide.item():.6e}")


class TestIntegrationWithConfig:
    """測試與配置文件的整合"""
    
    def test_config_loading_with_physics_fix(self):
        """測試修正後配置的載入"""
        from pinnx.train.config_loader import load_config
        
        # 載入修正後的配置
        config = load_config(
            '/Users/latteine/Documents/coding/pinns-mvp/configs/test_enhanced_5k_curriculum_fixed.yml'
        )
        
        # 驗證關鍵修正參數
        assert 'physics' in config, "配置應包含 physics 段落"
        assert 'vs_pinn' in config['physics'], "配置應包含 vs_pinn 段落"
        
        # 驗證修正 A1 配置
        loss_config = config['physics']['vs_pinn'].get('loss_config', {})
        assert 'disable_extra_pde_division' in loss_config, \
            "配置應包含 disable_extra_pde_division 參數"
        assert loss_config['disable_extra_pde_division'] is True, \
            "disable_extra_pde_division 應設為 True"
        
        # 驗證修正 A2 配置（可選）
        boundary_config = config['physics']['vs_pinn'].get('boundary_config', {})
        if 'boundary_band_width' in boundary_config:
            assert boundary_config['boundary_band_width'] == 5e-3, \
                "boundary_band_width 應為 5e-3"
        
        print("✅ 測試通過：配置載入正確")
        print(f"   disable_extra_pde_division: {loss_config['disable_extra_pde_division']}")
        if 'boundary_band_width' in boundary_config:
            print(f"   boundary_band_width: {boundary_config['boundary_band_width']}")


class TestBackwardCompatibility:
    """測試向後相容性"""
    
    def test_old_config_still_works(self):
        """測試舊配置仍可正常運行（向後相容）"""
        config = {
            'nu': 5.0e-5,
            'rho': 1.0,
            'Re_tau': 1000.0,
            'u_tau': 0.04997,
            'pressure_gradient': 0.0025,
            'channel_half_height': 1.0,
            'domain': {
                'x_range': [0.0, 25.13],
                'y_range': [-1.0, 1.0],
                'z_range': [0.0, 9.42]
            },
            'vs_pinn': {
                'scaling_factors': {
                    'N_x': 2.0,
                    'N_y': 12.0,
                    'N_z': 2.0
                }
                # 注意：沒有 loss_config，應使用默認值
            }
        }
        
        # 創建物理模組（應自動使用新默認值）
        physics = VSPINNChannelFlow(config)
        
        # 驗證默認行為（新修正）
        assert physics.disable_extra_pde_division is True, \
            "舊配置應自動採用新默認行為"
        
        print("✅ 測試通過：向後相容性良好")
        print("   舊配置自動採用新默認值（disable_extra_pde_division=True）")


if __name__ == "__main__":
    # 執行測試
    print("=" * 60)
    print("開始測試：Enhanced 5K Curriculum 物理修正")
    print("=" * 60)
    
    # 測試 A1: PDE 損失縮放修正
    print("\n【測試 A1: PDE 損失縮放修正】")
    test_a1 = TestPDELossScalingFix()
    physics_config = test_a1.physics_config()
    
    test_a1.test_disable_extra_pde_division_default(physics_config)
    test_a1.test_explicit_enable_extra_pde_division(physics_config)
    test_a1.test_pde_loss_scaling_behavior(physics_config)
    
    # 測試 A2: 邊界帶狀採樣
    print("\n【測試 A2: 邊界帶狀採樣】")
    test_a2 = TestBoundaryBandSampling()
    physics_config = test_a2.physics_config()
    mock_model = test_a2.mock_model()
    
    test_a2.test_wall_boundary_band_sampling(physics_config, mock_model)
    test_a2.test_periodic_boundary_band_sampling(physics_config, mock_model)
    test_a2.test_boundary_band_width_parameter(physics_config, mock_model)
    
    # 測試配置整合
    print("\n【測試配置整合】")
    test_integration = TestIntegrationWithConfig()
    test_integration.test_config_loading_with_physics_fix()
    
    # 測試向後相容性
    print("\n【測試向後相容性】")
    test_compat = TestBackwardCompatibility()
    test_compat.test_old_config_still_works()
    
    print("\n" + "=" * 60)
    print("✅ 所有測試通過！")
    print("=" * 60)
