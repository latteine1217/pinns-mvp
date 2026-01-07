#!/usr/bin/env python3
"""
批量更新所有配置檔案中的 DataLoader 設定

功能：
- 更新 num_workers: 8
- 新增 pin_memory: true
- 新增 persistent_workers: true
- 新增 prefetch_factor: 2

使用方式：
    python scripts/tools/batch_update_dataloader_configs.py
    python scripts/tools/batch_update_dataloader_configs.py --dry-run  # 預覽修改
"""
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any


def update_reproducibility_section(config: Dict[str, Any]) -> bool:
    """
    更新配置檔案中的 reproducibility 部分
    
    Returns:
        bool: 是否有修改
    """
    if 'reproducibility' not in config:
        return False
    
    modified = False
    repro = config['reproducibility']
    
    # 更新 num_workers
    if repro.get('num_workers') != 8:
        repro['num_workers'] = 8
        modified = True
    
    # 新增 pin_memory
    if 'pin_memory' not in repro:
        repro['pin_memory'] = True
        modified = True
    
    # 新增 persistent_workers
    if 'persistent_workers' not in repro:
        repro['persistent_workers'] = True
        modified = True
    
    # 新增 prefetch_factor
    if 'prefetch_factor' not in repro:
        repro['prefetch_factor'] = 2
        modified = True
    
    return modified


def process_config_file(config_path: Path, dry_run: bool = False) -> bool:
    """
    處理單個配置檔案
    
    Returns:
        bool: 是否有修改
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            print(f"⚠️  跳過空檔案: {config_path.name}")
            return False
        
        modified = update_reproducibility_section(config)
        
        if modified:
            if dry_run:
                print(f"✓ 將修改: {config_path.name}")
            else:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                print(f"✓ 已更新: {config_path.name}")
            return True
        else:
            print(f"- 無需修改: {config_path.name}")
            return False
    
    except Exception as e:
        print(f"❌ 錯誤處理 {config_path.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='批量更新 DataLoader 配置')
    parser.add_argument('--dry-run', action='store_true', 
                        help='預覽修改而不實際寫入檔案')
    parser.add_argument('--configs-dir', type=str, default='configs',
                        help='配置檔案目錄 (預設: configs)')
    args = parser.parse_args()
    
    # 獲取所有配置檔案
    configs_dir = Path(args.configs_dir)
    if not configs_dir.exists():
        print(f"❌ 配置目錄不存在: {configs_dir}")
        return
    
    config_files = list(configs_dir.glob('*.yml'))
    
    # 排除已手動更新的檔案
    exclude_files = {
        'standard_config_template.yml',
        'kolmogorov_re50_kf4_K100.yml'
    }
    config_files = [f for f in config_files if f.name not in exclude_files]
    
    print(f"{'=' * 60}")
    print(f"DataLoader 配置批量更新工具")
    print(f"{'=' * 60}")
    print(f"模式: {'預覽 (Dry Run)' if args.dry_run else '實際更新'}")
    print(f"配置目錄: {configs_dir.absolute()}")
    print(f"找到 {len(config_files)} 個待處理檔案")
    print(f"{'=' * 60}\n")
    
    # 處理所有檔案
    modified_count = 0
    for config_file in config_files:
        if process_config_file(config_file, dry_run=args.dry_run):
            modified_count += 1
    
    # 總結
    print(f"\n{'=' * 60}")
    print(f"總結:")
    print(f"  - 處理檔案: {len(config_files)}")
    print(f"  - {'將修改' if args.dry_run else '已修改'}: {modified_count}")
    print(f"  - 無需修改: {len(config_files) - modified_count}")
    if args.dry_run:
        print(f"\n💡 執行實際更新: python {Path(__file__).name}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
