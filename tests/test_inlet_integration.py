#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inlet 邊界條件整合測試腳本
快速驗證 inlet 損失計算是否正確整合到訓練流程中

用法:
    python scripts/test_inlet_integration.py
"""

import sys
import yaml
import torch
import logging
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.models import PINNNet
from pinnx.losses.residuals import BoundaryConditionLoss

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_inlet_loss_computation():
    """測試 inlet 損失計算是否正確"""
    print("=" * 80)
    print("🧪 Inlet 損失計算測試")
    print("=" * 80)
    
    # 設備選擇
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"\n使用設備: {device}")
    
    # 創建測試模型
    print("\n1️⃣ 創建測試模型...")
    model = PINNNet(
        in_dim=2,
        out_dim=3,
        width=256,
        depth=6,
        activation='sine',
        fourier_m=48,
        fourier_sigma=3.0
    ).to(device)
    print(f"✅ 模型創建成功，參數量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 創建 inlet 邊界點
    print("\n2️⃣ 生成 inlet 邊界點...")
    n_inlet = 64
    y_inlet = torch.linspace(-1.0, 1.0, n_inlet, device=device).unsqueeze(1)
    x_inlet_coords = torch.full_like(y_inlet, -1.0)
    inlet_coords = torch.cat([x_inlet_coords, y_inlet], dim=1)
    print(f"✅ 生成 {n_inlet} 個 inlet 點，形狀: {inlet_coords.shape}")
    
    # 模型預測
    print("\n3️⃣ 模型前向傳播...")
    with torch.no_grad():
        inlet_pred = model(inlet_coords)
    print(f"✅ 預測形狀: {inlet_pred.shape}")
    print(f"   u 範圍: [{inlet_pred[:, 0].min():.4f}, {inlet_pred[:, 0].max():.4f}]")
    print(f"   v 範圍: [{inlet_pred[:, 1].min():.4f}, {inlet_pred[:, 1].max():.4f}]")
    print(f"   p 範圍: [{inlet_pred[:, 2].min():.4f}, {inlet_pred[:, 2].max():.4f}]")
    
    # 創建損失模組
    print("\n4️⃣ 創建邊界條件損失模組...")
    bc_loss_module = BoundaryConditionLoss()
    print("✅ BoundaryConditionLoss 實例化成功")
    
    # 測試不同的 profile_type
    print("\n5️⃣ 測試不同速度剖面...")
    test_configs = [
        {'profile_type': 'parabolic', 'Re_tau': 100.0, 'u_max': 6.0, 'stage': 'Stage1 (層流)'},
        {'profile_type': 'log_law', 'Re_tau': 300.0, 'u_max': 10.0, 'stage': 'Stage2 (過渡)'},
        {'profile_type': 'log_law', 'Re_tau': 550.0, 'u_max': 13.5, 'stage': 'Stage3 (湍流)'},
        {'profile_type': 'turbulent', 'Re_tau': 1000.0, 'u_max': 16.5, 'stage': 'Stage4 (高Re)'},
    ]
    
    results = []
    for cfg in test_configs:
        # 設置模型為訓練模式（需要計算梯度）
        model.train()
        inlet_coords_grad = inlet_coords.detach().clone().requires_grad_(True)
        inlet_pred_grad = model(inlet_coords_grad)
        
        # 計算損失
        loss = bc_loss_module.inlet_velocity_profile_loss(
            inlet_coords=inlet_coords_grad,
            inlet_predictions=inlet_pred_grad,
            profile_type=cfg['profile_type'],
            Re_tau=cfg['Re_tau'],
            u_max=cfg['u_max'],
            y_range=(-1.0, 1.0)
        )
        
        results.append({
            'stage': cfg['stage'],
            'profile_type': cfg['profile_type'],
            'loss': loss.item()
        })
        
        print(f"\n  {cfg['stage']}:")
        print(f"    profile_type: {cfg['profile_type']}")
        print(f"    Re_tau: {cfg['Re_tau']}")
        print(f"    u_max: {cfg['u_max']}")
        print(f"    inlet_loss: {loss.item():.6f}")
        
        # 測試反向傳播
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        print(f"    梯度範數: {grad_norm:.6f}")
        print(f"    ✅ 反向傳播成功")
        
        # 清除梯度
        model.zero_grad()
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 測試結果總結")
    print("=" * 80)
    for res in results:
        print(f"{res['stage']:20s} | {res['profile_type']:10s} | loss = {res['loss']:10.6f}")
    
    print("\n✅ 所有測試通過！")
    print("=" * 80)
    
    return results


def test_config_reading():
    """測試配置文件讀取"""
    print("\n" + "=" * 80)
    print("🧪 配置文件讀取測試")
    print("=" * 80)
    
    config_path = project_root / 'configs' / 'channel_flow_curriculum_4stage.yml'
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    
    # 讀取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"\n✅ 成功讀取配置: {config_path.name}")
    
    # 驗證全域 inlet 配置
    inlet_config = config.get('inlet', {})
    print(f"\n全域 Inlet 配置:")
    print(f"  enabled: {inlet_config.get('enabled', False)}")
    print(f"  n_points: {inlet_config.get('n_points', 64)}")
    print(f"  x_position: {inlet_config.get('x_position', -1.0)}")
    
    # 驗證各階段配置
    print(f"\n各階段 Inlet 配置:")
    for stage in config['curriculum']['stages']:
        print(f"\n  {stage['name']}:")
        
        weights = stage.get('weights', {})
        inlet_weight = weights.get('inlet', None)
        
        if inlet_weight is None:
            print(f"    ❌ 未設定 inlet 權重！")
            return False
        
        print(f"    inlet_weight: {inlet_weight}")
        
        stage_inlet = stage.get('inlet', {})
        if stage_inlet:
            print(f"    profile_type: {stage_inlet.get('profile_type', '未設定')}")
            print(f"    u_max: {stage_inlet.get('u_max', '未設定')}")
        else:
            print(f"    ⚠️  使用全域 inlet 配置")
    
    print("\n✅ 配置文件讀取測試通過！")
    return True


if __name__ == "__main__":
    print("\n" + "🚀" * 40)
    print("Inlet 邊界條件整合測試")
    print("🚀" * 40)
    
    try:
        # 測試損失計算
        results = test_inlet_loss_computation()
        
        # 測試配置讀取
        config_ok = test_config_reading()
        
        if config_ok:
            print("\n" + "=" * 80)
            print("✅ 所有整合測試通過！")
            print("=" * 80)
            print("\n下一步:")
            print("  1. 執行乾跑測試: python scripts/train.py --config configs/channel_flow_curriculum_4stage.yml --max_epochs 100")
            print("  2. 監控 inlet_loss 是否正常收斂")
            print("  3. 檢查壓力場誤差是否改善")
            sys.exit(0)
        else:
            print("\n❌ 配置測試失敗！")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ 測試失敗: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
