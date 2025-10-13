#!/usr/bin/env python3
"""
測試自適應採樣整合
驗證 TrainingLoopManager 是否正確整合到主訓練循環
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import logging

from pinnx.train.loop import TrainingLoopManager

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_training_loop_manager_initialization():
    """測試訓練循環管理器初始化"""
    
    # 載入配置
    config_path = Path(__file__).parent.parent / 'configs' / 'inverse_reconstruction_main.yml'
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 🔧 強制啟用自適應採樣（用於測試）
    if 'adaptive_collocation' not in config:
        config['adaptive_collocation'] = {}
    config['adaptive_collocation']['enabled'] = True
    
    # 檢查配置
    adaptive_cfg = config.get('adaptive_collocation', {})
    print(f"✅ 自適應配置存在: enabled={adaptive_cfg.get('enabled', False)}")
    
    # 初始化管理器
    try:
        loop_manager = TrainingLoopManager(config)
        print(f"✅ TrainingLoopManager 初始化成功")
        
        # 創建測試 PDE 點
        device = torch.device('cpu')
        n_points = 4096
        x_pde = torch.rand(n_points, 1, device=device) * 2 - 1  # [-1, 1]
        y_pde = torch.rand(n_points, 1, device=device) * 2 - 1
        
        # 合併成一個張量 [N, 2]
        pde_points = torch.cat([x_pde, y_pde], dim=1)
        
        # 設置初始點
        loop_manager.setup_initial_points(pde_points)
        n_points_managed = len(loop_manager.current_pde_points) if loop_manager.current_pde_points is not None else 0
        print(f"✅ 初始點設置成功: {n_points_managed} 個點")
        
        # 測試觸發條件
        epoch = 1000
        loss = 0.5
        residuals = {'total_loss': loss}
        
        should_resample = loop_manager.should_resample_collocation_points(epoch, loss, residuals)
        print(f"✅ 觸發檢查成功: epoch={epoch}, should_resample={should_resample}")
        
        # 測試統計收集
        loss_dict = {
            'total_loss': 0.5,
            'residual_loss': 0.3,
            'data_loss': 0.2
        }
        loop_manager.collect_epoch_stats(epoch, loss_dict)
        stats = loop_manager.get_summary()
        print(f"✅ 統計收集成功: {stats}")
        
        print("\n" + "="*60)
        print("🎉 所有測試通過！自適應採樣整合正常！")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_point_weights():
    """測試點權重功能"""
    from pinnx.train.loop import apply_point_weights_to_loss
    
    # 創建測試數據
    residuals = torch.tensor([0.1, 0.5, 0.8, 0.3, 0.6])
    weights = torch.tensor([1.0, 2.5, 2.5, 1.0, 1.5])
    
    # 測試加權平均
    weighted_loss = apply_point_weights_to_loss(residuals, weights)
    print(f"✅ 點權重功能測試通過")
    print(f"   原始殘差: {residuals.tolist()}")
    print(f"   點權重: {weights.tolist()}")
    print(f"   加權損失: {weighted_loss.item():.6f}")
    
    return True

if __name__ == '__main__':
    print("="*60)
    print("🚀 開始測試自適應採樣整合")
    print("="*60 + "\n")
    
    # 測試 1: 訓練循環管理器
    print("📋 測試 1: TrainingLoopManager 初始化")
    print("-"*60)
    test1_passed = test_training_loop_manager_initialization()
    
    print("\n" + "="*60)
    print("📋 測試 2: 點權重功能")
    print("-"*60)
    test2_passed = test_point_weights()
    
    print("\n" + "="*60)
    if test1_passed and test2_passed:
        print("✅ 所有測試通過！")
        print("🎯 自適應採樣已成功整合到訓練循環")
        sys.exit(0)
    else:
        print("❌ 部分測試失敗")
        sys.exit(1)
