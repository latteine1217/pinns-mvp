#!/usr/bin/env python3
"""
驗證 Phase 3B 配置文件權重總和
確認所有 8 個獨立權重項的總和為 1.0
"""

import yaml
from pathlib import Path

def verify_weights(config_path):
    """驗證配置文件中的權重總和"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    losses = config.get('losses', {})
    
    # 提取所有主要權重項（排除正則項）
    weight_items = {
        'data': losses.get('data_weight', 0.0),
        'momentum_x': losses.get('momentum_x_weight', 0.0),
        'momentum_y': losses.get('momentum_y_weight', 0.0),
        'continuity': losses.get('continuity_weight', 0.0),
        'wall_constraint': losses.get('wall_constraint_weight', 0.0),
        'periodicity': losses.get('periodicity_weight', 0.0),
        'boundary': losses.get('boundary_weight', 0.0),
        'prior': losses.get('prior_weight', 0.0)
    }
    
    print("=" * 70)
    print(f"📊 權重驗證報告：{config_path.name}")
    print("=" * 70)
    
    # 顯示各項權重
    for key, value in weight_items.items():
        percentage = (value / 1.0) * 100 if value > 0 else 0
        print(f"  {key:20s}: {value:.6f}  ({percentage:6.2f}%)")
    
    # 計算總和
    weight_sum = sum(weight_items.values())
    print("-" * 70)
    print(f"  {'總和 (Σw)':20s}: {weight_sum:.6f}  ({(weight_sum)*100:.2f}%)")
    print("=" * 70)
    
    # 驗證結果
    tolerance = 1e-6
    if abs(weight_sum - 1.0) < tolerance:
        print("✅ 驗證通過：權重總和 = 1.0")
    else:
        print(f"⚠️  警告：權重總和 = {weight_sum:.6f} (預期 1.0)")
        print(f"   偏差：{abs(weight_sum - 1.0):.6f}")
    
    # 顯示相對比例（以 data 為基準）
    data_weight = weight_items['data']
    if data_weight > 0:
        print("\n📐 相對比例（以 data 為基準 = 1.0x）：")
        for key, value in weight_items.items():
            if value > 0:
                ratio = value / data_weight
                print(f"  {key:20s}: {ratio:.2f}x")
    
    print("=" * 70)
    
    return weight_sum


if __name__ == "__main__":
    # 驗證 Phase 3B 配置
    config_path = Path(__file__).parent.parent / "configs" / "channel_flow_re1000_fix6_k50_phase3b.yml"
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在：{config_path}")
        exit(1)
    
    weight_sum = verify_weights(config_path)
    
    # 返回狀態碼
    exit(0 if abs(weight_sum - 1.0) < 1e-6 else 1)
