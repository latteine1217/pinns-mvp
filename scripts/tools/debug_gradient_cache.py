#!/usr/bin/env python3
"""
調試腳本：檢查 gradient cache 是否被正確啟用
"""

import sys
from pathlib import Path

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from pinnx.dataio.config_loader import load_config
from pinnx.models.factory import create_model
from pinnx.physics.factory import create_physics


def main():
    config_path = PROJECT_ROOT / "configs/perf_test_wave2.yml"
    
    print("🔍 載入配置...")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    cfg = load_config(str(config_path))
    
    # 檢查 physics 類型
    physics_type = config.get('physics', {}).get('type', 'unknown')
    print(f"   Physics Type: {physics_type}")
    
    # 創建 physics 對象
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    physics = create_physics(cfg.physics, cfg.model.input_dim, device)
    
    # 檢查是否是 VS-PINN
    is_vs_pinn = hasattr(physics, 'compute_momentum_residuals')
    print(f"   Is VS-PINN: {is_vs_pinn}")
    print(f"   Has compute_momentum_residuals: {hasattr(physics, 'compute_momentum_residuals')}")
    print(f"   Has compute_continuity_residual: {hasattr(physics, 'compute_continuity_residual')}")
    
    # 檢查 model 輸出維度
    model = create_model(cfg.model, device)
    
    # 測試 forward pass
    dummy_input = torch.randn(10, cfg.model.input_dim, device=device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"   Model Output Shape: {output.shape}")
    print(f"   Output Dimensions: {output.shape[1]}")
    
    # 檢查是否符合 VS-PINN 的維度要求（3 或 4）
    if output.shape[1] in [3, 4]:
        print(f"   ✅ 輸出維度符合 VS-PINN 要求")
    else:
        print(f"   ❌ 輸出維度不符合 VS-PINN 要求（期望 3 或 4）")
    
    # 總結
    print("\n" + "="*80)
    if is_vs_pinn and output.shape[1] in [3, 4]:
        print("✅ Gradient Cache 應該會被啟用")
    else:
        print("❌ Gradient Cache 不會被啟用")
        if not is_vs_pinn:
            print("   原因：Physics 不是 VS-PINN")
        if output.shape[1] not in [3, 4]:
            print(f"   原因：模型輸出維度為 {output.shape[1]}，不符合要求")
    print("="*80)


if __name__ == "__main__":
    main()
