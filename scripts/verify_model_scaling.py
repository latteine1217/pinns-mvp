#!/usr/bin/env python3
"""
驗證模型縮放參數配置
測試修復後的 train.py 是否正確使用域範圍而非感測點統計
"""

import sys
import torch
import yaml
import logging
from pathlib import Path

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train import create_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_scaling():
    """驗證模型縮放參數"""
    
    config_path = Path("configs/vs_pinn_3d_full_training.yml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "=" * 80)
    print("🔍 模型縮放參數驗證測試")
    print("=" * 80)
    
    # 創建模型（無需感測點統計）
    print("\n1️⃣ 創建模型（不提供感測點統計，應使用配置域範圍）...")
    model = create_model(config, statistics=None, device=device)
    
    # 檢查縮放參數
    if not hasattr(model, 'input_min'):
        print("❌ 模型沒有 input_min 屬性（可能未啟用 scaling）")
        return False
    
    print("\n2️⃣ 檢查縮放參數:")
    input_min = model.input_min.cpu().numpy()
    input_max = model.input_max.cpu().numpy()
    output_min = model.output_min.cpu().numpy()
    output_max = model.output_max.cpu().numpy()
    
    print(f"   Input min:  {input_min}")
    print(f"   Input max:  {input_max}")
    print(f"   Output min: {output_min}")
    print(f"   Output max: {output_max}")
    
    # 驗證與配置文件的一致性
    print("\n3️⃣ 驗證與配置文件的一致性:")
    domain = config['physics']['domain']
    expected_x_range = domain['x_range']
    expected_y_range = domain['y_range']
    expected_z_range = domain['z_range']
    
    print(f"   配置 x_range: {expected_x_range}")
    print(f"   配置 y_range: {expected_y_range}")
    print(f"   配置 z_range: {expected_z_range}")
    
    # 容差檢查
    tolerance = 1e-3
    errors = []
    
    if abs(input_min[0] - expected_x_range[0]) > tolerance or \
       abs(input_max[0] - expected_x_range[1]) > tolerance:
        errors.append(f"❌ X範圍不匹配: 期望 {expected_x_range}, 實際 [{input_min[0]:.4f}, {input_max[0]:.4f}]")
    else:
        print(f"   ✅ X範圍匹配: {expected_x_range}")
    
    if abs(input_min[1] - expected_y_range[0]) > tolerance or \
       abs(input_max[1] - expected_y_range[1]) > tolerance:
        errors.append(f"❌ Y範圍不匹配: 期望 {expected_y_range}, 實際 [{input_min[1]:.4f}, {input_max[1]:.4f}]")
    else:
        print(f"   ✅ Y範圍匹配: {expected_y_range}")
    
    if len(input_min) > 2:
        if abs(input_min[2] - expected_z_range[0]) > tolerance or \
           abs(input_max[2] - expected_z_range[1]) > tolerance:
            errors.append(f"❌ Z範圍不匹配: 期望 {expected_z_range}, 實際 [{input_min[2]:.4f}, {input_max[2]:.4f}]")
        else:
            print(f"   ✅ Z範圍匹配: {expected_z_range}")
    
    # 測試使用錯誤統計（模擬感測點範圍）
    print("\n4️⃣ 測試當提供感測點統計時是否被正確覆蓋:")
    fake_statistics = {
        'x': {'range': [0.0, 19.19]},  # 模擬 K=30 感測點範圍
        'y': {'range': [-1.0, 0.008]},
        'z': {'range': [0.91, 8.81]},
        'u': {'range': [0.0, 18.09]},
        'v': {'range': [-0.24, 0.24]},
        'w': {'range': [0.0, 1.0]},
        'p': {'range': [-100.0, 10.0]}
    }
    
    print(f"   提供錯誤統計: x={fake_statistics['x']['range']}, y={fake_statistics['y']['range']}")
    
    model2 = create_model(config, statistics=fake_statistics, device=device)
    input_min2 = model2.input_min.cpu().numpy()
    input_max2 = model2.input_max.cpu().numpy()
    
    print(f"   模型實際使用: x=[{input_min2[0]:.4f}, {input_max2[0]:.4f}], y=[{input_min2[1]:.4f}, {input_max2[1]:.4f}]")
    
    # 應該仍然使用配置範圍，而非統計範圍
    if abs(input_min2[0] - expected_x_range[0]) > tolerance or \
       abs(input_max2[0] - expected_x_range[1]) > tolerance:
        errors.append(f"❌ 提供統計時仍使用了錯誤範圍（應優先使用配置）")
    else:
        print(f"   ✅ 正確優先使用配置範圍，忽略感測點統計")
    
    # 總結
    print("\n" + "=" * 80)
    if errors:
        print("❌ 驗證失敗:")
        for err in errors:
            print(f"   {err}")
        print("=" * 80)
        return False
    else:
        print("✅ 所有驗證通過！模型將使用完整域範圍訓練，可泛化到全域。")
        print("=" * 80)
        return True

if __name__ == '__main__':
    success = verify_scaling()
    sys.exit(0 if success else 1)
