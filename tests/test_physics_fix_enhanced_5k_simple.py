"""
單元測試：Enhanced 5K Curriculum 物理修正驗證 (簡化版)

測試 TASK-ENHANCED-5K-PHYSICS-FIX 的兩項核心修正：
- 修正 A1: PDE 損失雙重削弱問題（現已固定採用修正後行為）
- 修正 A2: 邊界帶狀採樣策略（boundary_band_width）

創建時間: 2025-10-15
參考文檔: tasks/TASK-ENHANCED-5K-PHYSICS-DIAGNOSIS/corrective_actions.md
"""

import torch
import sys


def test_pde_loss_fix():
    """測試修正 A1: PDE 損失雙重削弱修正"""
    print("\n【測試 A1: PDE 損失縮放修正】")
    
    from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
    
    physics = VSPINNChannelFlow()  # 使用默認配置
    
    assert not hasattr(physics, 'disable_extra_pde_division'), \
        "legacy disable_extra_pde_division flag should be removed"
    print("✅ 測試 1 通過：模組不再暴露 disable_extra_pde_division")
    
    # === 測試 2: 檢查 N_max_sq 計算 ===
    # 使用自定義縮放因子
    scaling_factors = {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}
    physics_custom = VSPINNChannelFlow(scaling_factors=scaling_factors)
    
    assert hasattr(physics_custom, 'N_max_sq'), "應包含 N_max_sq 屬性"
    expected_N_max_sq = 144.0  # max(2, 12, 2)^2 = 12^2 = 144
    # N_max_sq 是 PyTorch Tensor，使用 .item() 提取標量值
    import torch
    if isinstance(physics_custom.N_max_sq, torch.Tensor):
        actual_N_max_sq = physics_custom.N_max_sq.item()
    else:
        actual_N_max_sq = physics_custom.N_max_sq
    assert abs(actual_N_max_sq - expected_N_max_sq) < 1e-5, \
        f"N_max_sq 應為 {expected_N_max_sq}，實際為 {actual_N_max_sq}"
    print(f"✅ 測試 2 通過：N_max_sq = {actual_N_max_sq}")
    
    # === 測試 3: 驗證修正後縮放行為 ===
    mock_losses = {
        'momentum_x': torch.tensor(1.0),
        'momentum_y': torch.tensor(1.0),
        'momentum_z': torch.tensor(1.0),
        'continuity': torch.tensor(1.0),
        'data': torch.tensor(1.0),
        'wall_constraint': torch.tensor(1.0)
    }
    normalized = physics_custom.normalize_loss(mock_losses)
    expected = torch.tensor(1.0) / physics_custom.N_max_sq
    for key in ['momentum_x', 'momentum_y', 'momentum_z', 'continuity']:
        assert torch.isclose(normalized[key], expected, rtol=1e-5)
    print("✅ 測試 3 通過：PDE 損失僅除以 N_max_sq")
    
    # === 測試 4: 舊 key 立即報錯 ===
    legacy_config = {'disable_extra_pde_division': False}
    try:
        VSPINNChannelFlow(scaling_factors=scaling_factors, loss_config=legacy_config)
    except ValueError:
        print("✅ 測試 4 通過：legacy disable_extra_pde_division 會被拒絕")
    else:
        raise AssertionError("legacy disable_extra_pde_division 應觸發 ValueError")
    
    return True


def test_boundary_band_sampling():
    """測試修正 A2: 邊界帶狀採樣策略"""
    print("\n【測試 A2: 邊界帶狀採樣】")
    
    from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
    
    # 使用平坦參數（參考 test_rans_integration.py）
    scaling_factors = {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}
    physics_params = {
        'nu': 5.0e-5,
        'dP_dx': 0.0025,
        'rho': 1.0
    }
    domain_bounds = {
        'x': (0.0, 25.13),
        'y': (-1.0, 1.0),
        'z': (0.0, 9.42)
    }
    
    physics = VSPINNChannelFlow(
        scaling_factors=scaling_factors,
        physics_params=physics_params,
        domain_bounds=domain_bounds
    )
    
    # 檢查帶狀採樣方法存在
    assert hasattr(physics, 'compute_wall_shear_stress'), \
        "應包含 compute_wall_shear_stress 方法"
    assert hasattr(physics, 'compute_periodic_loss'), \
        "應包含 compute_periodic_loss 方法"
    print("✅ 測試 1 通過：邊界採樣方法存在")
    
    # 檢查方法簽名包含 boundary_band_width 參數
    import inspect
    wall_sig = inspect.signature(physics.compute_wall_shear_stress)
    periodic_sig = inspect.signature(physics.compute_periodic_loss)
    
    assert 'boundary_band_width' in wall_sig.parameters, \
        "compute_wall_shear_stress 應包含 boundary_band_width 參數"
    assert 'boundary_band_width' in periodic_sig.parameters, \
        "compute_periodic_loss 應包含 boundary_band_width 參數"
    print("✅ 測試 2 通過：邊界採樣方法包含 boundary_band_width 參數")
    
    # 檢查默認值
    wall_default = wall_sig.parameters['boundary_band_width'].default
    periodic_default = periodic_sig.parameters['boundary_band_width'].default
    
    assert wall_default == 5e-3, \
        f"compute_wall_shear_stress 默認帶寬應為 5e-3，實際為 {wall_default}"
    assert periodic_default == 5e-3, \
        f"compute_periodic_loss 默認帶寬應為 5e-3，實際為 {periodic_default}"
    print(f"✅ 測試 3 通過：默認帶寬 = {wall_default}")
    
    return True


def test_config_integration():
    """測試配置文件整合"""
    print("\n【測試配置整合】")
    
    import yaml
    
    # 載入修正後的配置
    config_path = '/Users/latteine/Documents/coding/pinns-mvp/configs/test_enhanced_5k_curriculum_fixed.yml'
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 驗證關鍵修正參數
    assert 'physics' in config, "配置應包含 physics 段落"
    assert 'vs_pinn' in config['physics'], "配置應包含 vs_pinn 段落"
    
    # 驗證修正 A1 配置：不應再出現舊旗標
    loss_config = config['physics']['vs_pinn'].get('loss_config', {})
    assert 'disable_extra_pde_division' not in loss_config, \
        "配置不應再包含 disable_extra_pde_division"
    print("✅ 測試 1 通過：配置不包含 legacy disable_extra_pde_division")
    
    # 驗證修正 A2 配置（可選）
    boundary_config = config['physics']['vs_pinn'].get('boundary_config', {})
    if 'boundary_band_width' in boundary_config:
        assert boundary_config['boundary_band_width'] == 5e-3, \
            "boundary_band_width 應為 5e-3"
        print(f"✅ 測試 2 通過：配置包含 boundary_band_width = 5e-3")
    else:
        print("ℹ️  測試 2 跳過：配置使用默認 boundary_band_width")
    
    # 驗證實驗名稱
    assert config['experiment']['name'] == 'test_enhanced_5k_curriculum_fixed', \
        "實驗名稱應為 test_enhanced_5k_curriculum_fixed"
    print(f"✅ 測試 3 通過：實驗名稱正確")
    
    return True


def test_backward_compatibility():
    """此功能已廢除，保留測試以提示更新"""
    print("\n【測試向後相容性】")
    print("✅ legacy 模式已移除，請更新到最新配置格式")
    return True


def main():
    """執行所有測試"""
    print("=" * 60)
    print("Enhanced 5K Curriculum 物理修正驗證測試")
    print("=" * 60)
    
    try:
        # 執行所有測試
        tests = [
            ('PDE 損失縮放修正', test_pde_loss_fix),
            ('邊界帶狀採樣', test_boundary_band_sampling),
            ('配置整合', test_config_integration),
            ('向後相容性', test_backward_compatibility),
        ]
        
        passed = 0
        failed = 0
        
        for name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                else:
                    failed += 1
                    print(f"❌ {name} 測試失敗")
            except Exception as e:
                failed += 1
                print(f"❌ {name} 測試異常：{e}")
                import traceback
                traceback.print_exc()
        
        # 彙總結果
        print("\n" + "=" * 60)
        print(f"測試結果：{passed} 通過，{failed} 失敗")
        print("=" * 60)
        
        if failed == 0:
            print("✅ 所有測試通過！物理修正已正確實施。")
            return 0
        else:
            print("⚠️ 部分測試失敗，請檢查修正實施。")
            return 1
            
    except Exception as e:
        print(f"❌ 測試執行失敗：{e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
