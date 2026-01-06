#!/usr/bin/env python3
"""
批量更新實驗配置文件，添加週期性 Fourier 嵌入配置註解

用法:
    python scripts/tools/batch_update_experiment_configs.py
"""

import yaml
import re
from pathlib import Path
from typing import Dict, Any

# 週期性 Fourier 嵌入模板（Kolmogorov Flow 2D）
KOLMOGOROV_2D_PERIODIC_TEMPLATE = """
  # 方式 2: 週期性 Fourier 嵌入（推薦）
  # 取消下方註釋並註釋上方 "方式 1" 以啟用：
  # fourier_features:
  #   type: hybrid
  #   axes:
  #     0: {type: standard, n_modes: 12, sigma: 4.0, use_2pi: true}  # time (非週期)
  #     1: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # x (週期: [0, 2π])
  #     2: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # y (週期: [0, 2π])
  #   trainable_fourier: false"""


def get_fourier_section_type(content: str, config: Dict[str, Any]) -> str:
    """
    判斷 fourier_features 的類型
    
    Returns:
        'standard' | 'disabled' | 'axis_selective' | 'already_updated'
    """
    # 檢查是否已更新
    if '方式 2: 週期性 Fourier 嵌入' in content:
        return 'already_updated'
    
    # 從 YAML 配置中獲取類型
    try:
        ff_type = config.get('model', {}).get('fourier_features', {}).get('type', 'standard')
        return ff_type
    except:
        return 'standard'


def get_description_for_type(ff_type: str) -> str:
    """根據 fourier_features 類型生成描述"""
    if ff_type == 'disabled':
        return '禁用 Fourier（當前使用 - 用於消融實驗對比）'
    elif ff_type == 'axis_selective':
        return 'Axis-Selective Fourier（當前使用）'
    else:  # standard
        return '標準 Fourier（當前使用）'


def update_config_file(filepath: Path) -> bool:
    """
    更新單個配置文件
    
    Returns:
        True if updated, False if already updated or error
    """
    try:
        # 讀取文件內容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 讀取 YAML 配置
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查是否已更新
        ff_type = get_fourier_section_type(content, config)
        if ff_type == 'already_updated':
            print(f"⏭️  已更新: {filepath.name}")
            return False
        
        # 查找 fourier_features 區段
        pattern = r'(  fourier_features:\s*\n(?:    [^\n]+\n)*)'
        match = re.search(pattern, content)
        
        if not match:
            print(f"⚠️  找不到 fourier_features: {filepath.name}")
            return False
        
        original_section = match.group(1)
        description = get_description_for_type(ff_type)
        
        # 構建新的區段
        new_section = f"  # ========== Fourier Features 配置 ==========\n"
        new_section += f"  # 方式 1: {description}\n"
        new_section += original_section
        new_section += KOLMOGOROV_2D_PERIODIC_TEMPLATE
        
        # 替換內容
        new_content = content.replace(original_section, new_section)
        
        # 寫回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 已更新: {filepath.name}")
        return True
    
    except Exception as e:
        print(f"❌ 錯誤 {filepath.name}: {e}")
        return False


def main():
    """主函數：批量更新所有實驗配置文件"""
    
    # 定義實驗配置目錄
    experiments_dir = Path(__file__).parent.parent.parent / 'configs' / 'experiments'
    
    print(f"🔍 掃描目錄: {experiments_dir}\n")
    
    # 查找所有 .yml 文件
    config_files = sorted(experiments_dir.rglob('*.yml'))
    
    print(f"📋 找到 {len(config_files)} 個配置文件\n")
    print("=" * 60)
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for config_file in config_files:
        result = update_config_file(config_file)
        if result is True:
            updated_count += 1
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1
    
    print("=" * 60)
    print(f"\n📊 更新統計:")
    print(f"  ✅ 已更新: {updated_count} 個")
    print(f"  ⏭️  跳過（已更新）: {skipped_count} 個")
    print(f"  ❌ 錯誤: {error_count} 個")
    print(f"  📝 總計: {len(config_files)} 個")
    
    if updated_count > 0:
        print(f"\n🎉 成功更新 {updated_count} 個配置文件！")
    else:
        print(f"\n✨ 所有配置文件已是最新狀態")


if __name__ == '__main__':
    main()
