#!/usr/bin/env python3
"""
測試 merge_momentum 功能的完整集成
驗證配置載入 → NSResidualLoss 實例化 → 損失計算的完整流程
"""

import sys
import torch
import yaml
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from pinnx.losses.residuals import NSResidualLoss


def test_config_loading():
    """測試配置文件載入"""
    print("\n" + "=" * 80)
    print("📂 測試 1: 配置文件載入")
    print("=" * 80)
    
    config_file = 'configs/experiments/S2_k_scan/s2_qr_K100_2d_re50.yml'
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    losses_cfg = config.get('losses', {})
    merge_momentum = losses_cfg.get('merge_momentum', False)
    
    print(f"   Config: {config_file}")
    print(f"   merge_momentum: {merge_momentum}")
    print(f"   Type: {type(merge_momentum)}")
    
    assert isinstance(merge_momentum, bool), "merge_momentum 應該是 bool 類型"
    assert merge_momentum == True, "merge_momentum 應該是 True"
    
    print("   ✅ 配置載入成功！")
    return losses_cfg


def test_loss_instantiation(losses_cfg):
    """測試 NSResidualLoss 實例化"""
    print("\n" + "=" * 80)
    print("🔧 測試 2: NSResidualLoss 實例化")
    print("=" * 80)
    
    # 標準模式
    loss_std = NSResidualLoss(
        nu=losses_cfg.get('nu', 1e-3),
        density=losses_cfg.get('rho', 1.0),
        merge_momentum=False
    )
    print(f"   標準模式: merge_momentum={loss_std.merge_momentum}")
    assert loss_std.merge_momentum == False
    
    # 合併模式
    loss_merged = NSResidualLoss(
        nu=losses_cfg.get('nu', 1e-3),
        density=losses_cfg.get('rho', 1.0),
        merge_momentum=True
    )
    print(f"   合併模式: merge_momentum={loss_merged.merge_momentum}")
    assert loss_merged.merge_momentum == True
    
    # 從配置讀取
    loss_from_cfg = NSResidualLoss(
        nu=losses_cfg.get('nu', 1e-3),
        density=losses_cfg.get('rho', 1.0),
        merge_momentum=losses_cfg.get('merge_momentum', False)
    )
    print(f"   配置模式: merge_momentum={loss_from_cfg.merge_momentum}")
    assert loss_from_cfg.merge_momentum == True
    
    print("   ✅ 實例化成功！")
    return loss_std, loss_merged


def test_loss_computation(loss_std, loss_merged):
    """測試損失計算"""
    print("\n" + "=" * 80)
    print("🧮 測試 3: 損失計算")
    print("=" * 80)
    
    # 創建測試模型（用於生成具有 grad_fn 的預測）
    class TestModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(2, 32),
                torch.nn.Tanh(),
                torch.nn.Linear(32, 3)  # [u, v, p]
            )
        
        def forward(self, x):
            return self.net(x)
    
    # 創建模擬數據
    batch_size = 100
    coords = torch.randn(batch_size, 2, requires_grad=True)  # [x, y]
    
    # 使用模型生成預測（這樣會有 grad_fn）
    model = TestModel()
    predictions = model(coords)  # [u, v, p]
    
    # 標準模式計算
    print("\n   標準模式 (merge_momentum=False):")
    residuals_std = loss_std(coords, predictions)
    print(f"   損失項: {list(residuals_std.keys())}")
    assert 'pde_momentum_x' in residuals_std, "應該有 pde_momentum_x"
    assert 'pde_momentum_y' in residuals_std, "應該有 pde_momentum_y"
    assert 'pde_continuity' in residuals_std, "應該有 pde_continuity"
    print(f"   ✅ 標準模式：3 個 PDE 損失項")
    
    # 合併模式計算（需要新的座標以避免梯度累積）
    print("\n   合併模式 (merge_momentum=True):")
    coords_merged = torch.randn(batch_size, 2, requires_grad=True)
    predictions_merged = model(coords_merged)
    residuals_merged = loss_merged(coords_merged, predictions_merged)
    print(f"   損失項: {list(residuals_merged.keys())}")
    assert 'pde_momentum' in residuals_merged, "應該有 pde_momentum (合併)"
    assert 'pde_momentum_x' not in residuals_merged, "不應該有 pde_momentum_x"
    assert 'pde_momentum_y' not in residuals_merged, "不應該有 pde_momentum_y"
    assert 'pde_continuity' in residuals_merged, "應該有 pde_continuity"
    print(f"   ✅ 合併模式：2 個 PDE 損失項")
    
    # 驗證梯度流
    print("\n   驗證梯度流:")
    loss_total = residuals_merged['pde_momentum'] + residuals_merged['pde_continuity']
    loss_total.backward()
    assert coords_merged.grad is not None, "coords 應該有梯度"
    # predictions 的梯度會傳遞到模型參數，不是 predictions 本身
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "模型參數應該有梯度"
    print(f"   ✅ 梯度正常傳播")
    
    print("\n   ✅ 損失計算成功！")


def test_all_configs():
    """測試所有實驗配置是否包含 merge_momentum"""
    print("\n" + "=" * 80)
    print("📋 測試 4: 所有實驗配置")
    print("=" * 80)
    
    from pathlib import Path
    config_files = list(Path('configs/experiments').rglob('*.yml'))
    
    print(f"   找到 {len(config_files)} 個配置文件")
    
    missing = []
    for cfg_file in config_files:
        with open(cfg_file, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'merge_momentum' not in config.get('losses', {}):
            missing.append(cfg_file)
    
    if missing:
        print(f"\n   ❌ {len(missing)} 個配置缺少 merge_momentum:")
        for f in missing:
            print(f"      - {f}")
        raise AssertionError("部分配置文件缺少 merge_momentum 參數")
    
    print(f"   ✅ 所有 {len(config_files)} 個配置都包含 merge_momentum")


def main():
    print("\n" + "🔥" * 40)
    print(" Momentum Merging Integration Test")
    print("🔥" * 40)
    
    try:
        # 測試 1: 配置載入
        losses_cfg = test_config_loading()
        
        # 測試 2: 實例化
        loss_std, loss_merged = test_loss_instantiation(losses_cfg)
        
        # 測試 3: 損失計算
        test_loss_computation(loss_std, loss_merged)
        
        # 測試 4: 所有配置
        test_all_configs()
        
        print("\n" + "=" * 80)
        print("🎉 所有測試通過！")
        print("=" * 80)
        print("\n✅ merge_momentum 功能已成功集成到專案中")
        print("✅ 所有 17 個 Kolmogorov Flow 實驗配置已更新")
        print("✅ 損失計算與梯度流正常工作")
        print("\n建議下一步:")
        print("  1. 運行短時訓練測試 (10 epochs)")
        print("  2. 檢查 TensorBoard 日誌")
        print("  3. 執行完整實驗並對比收斂曲線")
        
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
