#!/usr/bin/env python3
"""
批量更新 configs/experiments 中的 Fourier Features 配置

目標配置：
- x, y 方向使用 periodic
- t 方向不使用 fourier
- m=64, sigma=2.0
"""

import yaml
import sys
from pathlib import Path


def update_fourier_config(config_path: Path) -> bool:
    """更新單個配置文件的 Fourier Features 配置"""
    
    try:
        # 讀取配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查是否有 model.fourier_features
        if 'model' not in config or 'fourier_features' not in config['model']:
            print(f"⚠️  跳過 {config_path.name}: 沒有 fourier_features 配置")
            return False
        
        # 獲取 domain_size (從 physics.domain 或 data.kolmogorov_config.domain)
        domain_size = 6.283185307179586  # 默認值 2π
        
        if 'physics' in config and 'domain' in config['physics']:
            if 'x_range' in config['physics']['domain']:
                domain_size = config['physics']['domain']['x_range'][1]
        elif 'data' in config and 'kolmogorov_config' in config['data']:
            kf_config = config['data']['kolmogorov_config']
            if 'domain' in kf_config and 'x' in kf_config['domain']:
                domain_size = kf_config['domain']['x'][1]
        
        # 更新 Fourier Features 配置
        # axis 0 (t): type='none' 直接透傳，不使用 fourier
        # axis 1 (x): type='periodic', n_modes=64
        # axis 2 (y): type='periodic', n_modes=64
        config['model']['fourier_features'] = {
            'type': 'hybrid',
            'axes': {
                0: {
                    'type': 'none'
                },
                1: {
                    'type': 'periodic',
                    'domain_size': domain_size,
                    'n_modes': 64
                },
                2: {
                    'type': 'periodic', 
                    'domain_size': domain_size,
                    'n_modes': 64
                }
            },
            'trainable_fourier': False
        }
        
        # 寫回配置文件
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"✅ 已更新 {config_path.name}")
        return True
        
    except Exception as e:
        print(f"❌ 更新失敗 {config_path.name}: {e}")
        return False


def main():
    # 掃描 configs/experiments 目錄
    experiments_dir = Path(__file__).parent.parent.parent / 'configs' / 'experiments'
    
    if not experiments_dir.exists():
        print(f"❌ 目錄不存在: {experiments_dir}")
        sys.exit(1)
    
    # 遞歸查找所有 .yml 文件
    config_files = list(experiments_dir.rglob('*.yml'))
    config_files.extend(experiments_dir.rglob('*.yaml'))
    
    print(f"🔍 找到 {len(config_files)} 個配置文件\n")
    
    # 更新每個配置文件
    success_count = 0
    for config_file in sorted(config_files):
        if update_fourier_config(config_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"✨ 完成！成功更新 {success_count}/{len(config_files)} 個配置文件")
    print(f"{'='*60}")
    
    print("\n📝 更新內容：")
    print("  - axis 0 (t): type='none' (直接透傳，不使用 Fourier)")
    print("  - axis 1 (x): type='periodic', n_modes=64")
    print("  - axis 2 (y): type='periodic', n_modes=64")
    print("  - domain_size: 自動從配置中讀取")
    print("  - trainable_fourier: False")
    print("\n⚠️  重要說明：")
    print("  - axis 0 必須設置為 'none' 而不是完全移除")
    print("  - 因為 HybridFourierFeatures 需要知道總輸入維度 (3D: t, x, y)")
    print("  - type='none' 表示該軸直接透傳，輸出維度為 1")


if __name__ == '__main__':
    main()
