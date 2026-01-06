#!/usr/bin/env python3
"""
批次更新配置檔案中的因果權重設定
將所有實驗配置的 causal_weighting 設為 true（除了特殊實驗）
"""

import yaml
from pathlib import Path
from typing import List, Tuple

def update_causal_weighting(config_path: Path, enable: bool = True) -> Tuple[bool, str]:
    """
    更新單個配置檔案的因果權重設定
    
    Returns:
        (changed, old_value): 是否有變更, 原始值
    """
    try:
        with open(config_path, 'r') as f:
            content = f.read()
            config = yaml.safe_load(content)
        
        # 檢查是否有 losses.causal_weighting
        if 'losses' not in config:
            return False, "N/A (no losses section)"
        
        old_value = config['losses'].get('causal_weighting', None)
        
        if old_value == enable:
            return False, str(old_value)
        
        # 使用字串替換保留原始格式和註解
        if old_value is not None:
            # 替換現有值
            old_line = f"causal_weighting: {str(old_value).lower()}"
            new_line = f"causal_weighting: {str(enable).lower()}"
            new_content = content.replace(old_line, new_line)
        else:
            # causal_weighting 不存在，需要添加
            return False, "N/A (not found)"
        
        # 寫回檔案
        with open(config_path, 'w') as f:
            f.write(new_content)
        
        return True, str(old_value)
    
    except Exception as e:
        return False, f"ERROR: {str(e)}"


def main():
    print("=" * 80)
    print("🔧 批次更新配置檔案：啟用因果權重 (Causal Weighting)")
    print("=" * 80)
    
    # 定義需要更新的配置檔案分類
    configs_to_update = {
        "S1_sensor_strategy": [
            "configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml",
            "configs/experiments/S1_sensor_strategy/s1_random_K100_2d_re50.yml",
        ],
        "S2_k_scan": [
            "configs/experiments/S2_k_scan/s2_qr_K30_2d_re50.yml",
            "configs/experiments/S2_k_scan/s2_qr_K50_2d_re50.yml",
            "configs/experiments/S2_k_scan/s2_qr_K80_2d_re50.yml",
            "configs/experiments/S2_k_scan/s2_qr_K100_2d_re50.yml",
        ],
        "M1_model_comparison": [
            "configs/experiments/M1_model_comparison/m1_vanilla_K100_2d_re50.yml",
        ],
        "main_configs": [
            "configs/kolmogorov_re50_kf4_K100_vanilla.yml",
        ],
    }
    
    # 不更新的配置（保持 false 或作為模板）
    skip_configs = {
        "configs/main.yml": "主配置模板",
        "configs/main_quick_validate.yml": "快速驗證模板",
        "configs/config_template_example.yml": "配置範例模板",
        "configs/standard_config_template.yml": "標準配置模板",
    }
    
    # 統計
    total_updated = 0
    total_skipped = 0
    total_already_correct = 0
    total_errors = 0
    
    # 更新各類別配置
    for category, config_list in configs_to_update.items():
        print(f"\n📁 {category}")
        print("-" * 80)
        
        for config_path_str in config_list:
            config_path = Path(config_path_str)
            
            if not config_path.exists():
                print(f"  ⚠️  {config_path.name:<50} [檔案不存在]")
                total_errors += 1
                continue
            
            changed, old_value = update_causal_weighting(config_path, enable=True)
            
            if changed:
                print(f"  ✅ {config_path.name:<50} [{old_value} → true]")
                total_updated += 1
            elif old_value == "True":
                print(f"  ⏭️  {config_path.name:<50} [已是 true]")
                total_already_correct += 1
            else:
                print(f"  ⚠️  {config_path.name:<50} [{old_value}]")
                total_errors += 1
    
    # 顯示跳過的配置
    print(f"\n📋 跳過的配置檔案 (保持原狀)")
    print("-" * 80)
    for config_path, reason in skip_configs.items():
        print(f"  ⏭️  {Path(config_path).name:<50} [{reason}]")
        total_skipped += len(skip_configs)
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 更新總結")
    print("=" * 80)
    print(f"  ✅ 已更新: {total_updated} 個")
    print(f"  ⏭️  已正確: {total_already_correct} 個")
    print(f"  📋 跳過: {total_skipped} 個")
    if total_errors > 0:
        print(f"  ⚠️  錯誤/警告: {total_errors} 個")
    
    print(f"\n總計處理: {total_updated + total_already_correct + total_skipped + total_errors} 個配置檔案")
    
    # 驗證
    print("\n" + "=" * 80)
    print("🔍 驗證更新結果")
    print("=" * 80)
    
    # 重新檢查所有實驗配置
    all_experiment_configs = list(Path("configs/experiments").rglob("*.yml"))
    
    causal_true = []
    causal_false = []
    no_causal = []
    
    for config_path in all_experiment_configs:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if 'losses' in config:
                causal = config['losses'].get('causal_weighting', None)
                if causal is True:
                    causal_true.append(config_path)
                elif causal is False:
                    causal_false.append(config_path)
                else:
                    no_causal.append(config_path)
        except:
            pass
    
    print(f"\n✅ 啟用因果權重 (true): {len(causal_true)} 個")
    for p in sorted(causal_true):
        print(f"    {p.relative_to('configs/experiments')}")
    
    print(f"\n❌ 未啟用因果權重 (false): {len(causal_false)} 個")
    for p in sorted(causal_false):
        print(f"    {p.relative_to('configs/experiments')}")
    
    if no_causal:
        print(f"\n⚠️ 缺少 causal_weighting 設定: {len(no_causal)} 個")
        for p in sorted(no_causal):
            print(f"    {p.relative_to('configs/experiments')}")
    
    print("\n✨ 完成！")


if __name__ == "__main__":
    main()
