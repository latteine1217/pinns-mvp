#!/usr/bin/env python3
"""
檢查 S1 實驗配置中的因果權重設定
"""

import yaml
from pathlib import Path

def check_causal_weighting():
    print("=" * 80)
    print("🔍 S1 實驗配置檢查：因果權重 (Causal Weighting)")
    print("=" * 80)
    
    s1_configs = [
        "configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml",
        "configs/experiments/S1_sensor_strategy/s1_random_K100_2d_re50.yml"
    ]
    
    results = []
    
    for config_path in s1_configs:
        if not Path(config_path).exists():
            print(f"⚠️ 檔案不存在: {config_path}")
            continue
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        exp_name = config['experiment']['name']
        causal_enabled = config['losses'].get('causal_weighting', False)
        
        results.append({
            'name': exp_name,
            'path': Path(config_path).name,
            'causal_weighting': causal_enabled
        })
    
    # 顯示結果
    print(f"\n📊 檢查結果:\n")
    print(f"{'實驗名稱':<30} {'配置檔案':<35} {'因果權重':<12}")
    print("-" * 80)
    
    all_disabled = True
    for result in results:
        status = "✅ 啟用" if result['causal_weighting'] else "❌ 未啟用"
        print(f"{result['name']:<30} {result['path']:<35} {status}")
        if result['causal_weighting']:
            all_disabled = False
    
    print("\n" + "=" * 80)
    print("📝 總結")
    print("=" * 80)
    
    if all_disabled:
        print("\n⚠️ 所有 S1 實驗配置皆【未啟用】因果權重 (causal_weighting: false)")
        print("\n💡 說明:")
        print("  - S1 實驗目標：比較感測器策略 (QR-Pivot vs Random)")
        print("  - 未使用因果權重 → 所有時間點的損失權重相同")
        print("  - 這是合理的基線設定，用於後續實驗對比")
        print("\n🔧 如需啟用因果權重，請修改配置檔案中的:")
        print("  losses:")
        print("    causal_weighting: true  # 改為 true")
    else:
        print("\n✅ 部分實驗已啟用因果權重")
    
    print()

if __name__ == "__main__":
    check_causal_weighting()
