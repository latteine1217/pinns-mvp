#!/usr/bin/env python3
"""
診斷模型初始化 NaN 問題
"""
import sys
import torch
import yaml
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

import pinnx
from pinnx.models.fourier_mlp import FourierMLP

def diagnose_model_initialization():
    """診斷模型初始化是否產生 NaN"""
    
    # 載入配置
    config_path = project_root / "configs" / "vs_pinn_channel_flow.yml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("🔍 診斷模型初始化 NaN 問題")
    print("=" * 60)
    
    # 創建模型
    model_cfg = config['model']
    print(f"\n📐 模型配置:")
    print(f"   類型: {model_cfg['type']}")
    print(f"   輸入維度: {model_cfg['in_dim']}")
    print(f"   輸出維度: {model_cfg['out_dim']}")
    print(f"   寬度: {model_cfg['width']}")
    print(f"   深度: {model_cfg['depth']}")
    print(f"   激活函數: {model_cfg['activation']}")
    print(f"   Fourier M: {model_cfg.get('fourier_m', 0)}")
    print(f"   Fourier Sigma: {model_cfg.get('fourier_sigma', 1.0)}")
    
    # 構建模型
    model = FourierMLP(
        in_dim=model_cfg['in_dim'],
        out_dim=model_cfg['out_dim'],
        width=model_cfg['width'],
        depth=model_cfg['depth'],
        activation=model_cfg['activation'],
        fourier_m=model_cfg.get('fourier_m', 64),
        fourier_sigma=model_cfg.get('fourier_sigma', 5.0)
    )
    
    # 檢查權重初始化
    print(f"\n🔧 檢查權重初始化:")
    has_nan_weights = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"   ❌ {name}: 包含 NaN!")
            has_nan_weights = True
        elif torch.isinf(param).any():
            print(f"   ❌ {name}: 包含 Inf!")
            has_nan_weights = True
        else:
            param_min = param.min().item()
            param_max = param.max().item()
            print(f"   ✅ {name}: min={param_min:.6f}, max={param_max:.6f}")
    
    if has_nan_weights:
        print("\n❌ 權重初始化就包含 NaN/Inf!")
        return False
    
    # 測試前向傳播
    print(f"\n🧪 測試前向傳播:")
    test_coords = torch.tensor([
        [12.565, 0.0, 4.71],    # 中心點
        [0.0, -1.0, 0.0],       # 下壁面
        [0.0, 1.0, 0.0],        # 上壁面
        [25.13, 0.5, 9.42],     # 邊界點
    ])
    
    print(f"   輸入座標形狀: {test_coords.shape}")
    print(f"   輸入範圍: x=[{test_coords[:, 0].min():.2f}, {test_coords[:, 0].max():.2f}]")
    print(f"              y=[{test_coords[:, 1].min():.2f}, {test_coords[:, 1].max():.2f}]")
    print(f"              z=[{test_coords[:, 2].min():.2f}, {test_coords[:, 2].max():.2f}]")
    
    with torch.no_grad():
        predictions = model(test_coords)
    
    print(f"\n   輸出形狀: {predictions.shape}")
    print(f"   輸出統計:")
    for i, var in enumerate(['u', 'v', 'w', 'p']):
        vals = predictions[:, i]
        print(f"      {var}: min={vals.min():.6f}, max={vals.max():.6f}, mean={vals.mean():.6f}")
        if torch.isnan(vals).any():
            print(f"         ❌ 包含 NaN!")
            return False
        if torch.isinf(vals).any():
            print(f"         ❌ 包含 Inf!")
            return False
    
    print("\n✅ 模型初始化正常，前向傳播無 NaN/Inf")
    return True

if __name__ == "__main__":
    success = diagnose_model_initialization()
    sys.exit(0 if success else 1)
