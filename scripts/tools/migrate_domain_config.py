#!/usr/bin/env python3
"""
Domain 配置遷移工具（P0-2）

目的：統一所有配置文件，只使用 physics.domain
     刪除 data.jhtdb_config.domain, data.kolmogorov_config.domain 和頂層 domain

使用方法：
    python scripts/tools/migrate_domain_config.py --config <file>
    python scripts/tools/migrate_domain_config.py --batch configs/
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def normalize_domain_format(domain_data: Dict[str, Any]) -> Dict[str, list]:
    """
    將各種 domain 格式統一為 physics.domain 格式
    
    支援輸入格式：
    1. x: [min, max], y: [min, max]  (data.jhtdb_config.domain)
    2. x_range: [min, max], y_range: [min, max]  (physics.domain)
    3. x_min: float, x_max: float  (頂層 domain)
    
    輸出格式：
        x_range: [min, max]
        y_range: [min, max]
        z_range: [min, max]  (如果存在)
    """
    result = {}
    
    # 格式 1: x: [min, max]
    if 'x' in domain_data:
        result['x_range'] = domain_data['x']
        result['y_range'] = domain_data['y']
        if 'z' in domain_data:
            result['z_range'] = domain_data['z']
    
    # 格式 2: x_range: [min, max] (已經是目標格式)
    elif 'x_range' in domain_data:
        result['x_range'] = domain_data['x_range']
        result['y_range'] = domain_data['y_range']
        if 'z_range' in domain_data:
            result['z_range'] = domain_data['z_range']
    
    # 格式 3: x_min, x_max
    elif 'x_min' in domain_data and 'x_max' in domain_data:
        result['x_range'] = [domain_data['x_min'], domain_data['x_max']]
        result['y_range'] = [domain_data['y_min'], domain_data['y_max']]
        if 'z_min' in domain_data and 'z_max' in domain_data:
            result['z_range'] = [domain_data['z_min'], domain_data['z_max']]
    
    else:
        raise ValueError(f"無法識別的 domain 格式: {domain_data}")
    
    return result


def extract_domain_from_config(config: Dict[str, Any]) -> Optional[Dict[str, list]]:
    """
    從配置中提取 domain（優先順序：jhtdb > kolmogorov > 頂層）
    
    Args:
        config: 完整配置字典
    
    Returns:
        統一格式的 domain，或 None（如果已經只有 physics.domain）
    """
    domain = None
    source = None
    
    # 優先順序 1: data.jhtdb_config.domain
    data_config = config.get('data', {})
    jhtdb_config = data_config.get('jhtdb_config', {})
    if 'domain' in jhtdb_config:
        domain = jhtdb_config['domain']
        source = 'data.jhtdb_config.domain'
    
    # 優先順序 2: data.kolmogorov_config.domain
    if domain is None:
        kflow_config = data_config.get('kolmogorov_config', {})
        if 'domain' in kflow_config:
            domain = kflow_config['domain']
            source = 'data.kolmogorov_config.domain'
    
    # 優先順序 3: 頂層 domain
    if domain is None:
        if 'domain' in config:
            domain = config['domain']
            source = 'top-level domain'
    
    # 優先順序 4: physics.domain (已存在，無需遷移)
    if domain is None:
        physics_domain = config.get('physics', {}).get('domain')
        if physics_domain:
            logging.info("  ✓ 已使用 physics.domain，無需遷移")
            return None
    
    if domain:
        logging.info(f"  發現 domain 來源: {source}")
        return normalize_domain_format(domain)
    
    return None


def migrate_config_file(config_path: Path, dry_run: bool = False) -> bool:
    """
    遷移單個配置文件
    
    Args:
        config_path: 配置文件路徑
        dry_run: 如果為 True，只顯示變更但不寫入
    
    Returns:
        是否進行了修改
    """
    logging.info(f"\n處理: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 提取並統一 domain
        unified_domain = extract_domain_from_config(config)
        
        if unified_domain is None:
            # 檢查是否有冗餘的舊格式需要刪除
            has_old_format = False
            if 'domain' in config:
                has_old_format = True
            if 'jhtdb_config' in config.get('data', {}) and 'domain' in config['data']['jhtdb_config']:
                has_old_format = True
            if 'kolmogorov_config' in config.get('data', {}) and 'domain' in config['data']['kolmogorov_config']:
                has_old_format = True
            
            if not has_old_format:
                return False  # 已經是乾淨的配置
        
        # 確保 physics 區塊存在
        if 'physics' not in config:
            config['physics'] = {}
        
        # 設定統一的 physics.domain
        if unified_domain:
            config['physics']['domain'] = unified_domain
            logging.info(f"  ✓ 設定 physics.domain: {unified_domain}")
        
        # 刪除舊格式的 domain 定義
        modified = False
        
        # 刪除頂層 domain
        if 'domain' in config:
            del config['domain']
            logging.info("  ✗ 刪除 top-level domain")
            modified = True
        
        # 刪除 data.jhtdb_config.domain
        if 'jhtdb_config' in config.get('data', {}) and 'domain' in config['data']['jhtdb_config']:
            del config['data']['jhtdb_config']['domain']
            logging.info("  ✗ 刪除 data.jhtdb_config.domain")
            modified = True
        
        # 刪除 data.kolmogorov_config.domain
        if 'kolmogorov_config' in config.get('data', {}) and 'domain' in config['data']['kolmogorov_config']:
            del config['data']['kolmogorov_config']['domain']
            logging.info("  ✗ 刪除 data.kolmogorov_config.domain")
            modified = True
        
        # 寫入變更
        if modified or unified_domain:
            if not dry_run:
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.safe_dump(config, f, 
                                   default_flow_style=False, 
                                   allow_unicode=True,
                                   sort_keys=False)
                logging.info(f"  ✅ 已更新配置文件")
            else:
                logging.info(f"  [DRY RUN] 將更新配置文件")
            return True
        
        return False
    
    except Exception as e:
        logging.error(f"  ❌ 處理失敗: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Domain 配置遷移工具')
    parser.add_argument('--config', type=str, help='單個配置文件路徑')
    parser.add_argument('--batch', type=str, help='批次處理目錄')
    parser.add_argument('--dry-run', action='store_true', help='模擬執行（不寫入文件）')
    
    args = parser.parse_args()
    
    if not args.config and not args.batch:
        parser.error("必須指定 --config 或 --batch")
    
    files_to_process = []
    
    if args.config:
        files_to_process.append(Path(args.config))
    
    if args.batch:
        batch_dir = Path(args.batch)
        files_to_process.extend(batch_dir.rglob("*.yml"))
    
    logging.info(f"{'='*60}")
    logging.info(f"Domain 配置遷移工具 (P0-2)")
    logging.info(f"{'='*60}")
    logging.info(f"找到 {len(files_to_process)} 個配置文件")
    
    modified_count = 0
    for config_path in files_to_process:
        if migrate_config_file(config_path, dry_run=args.dry_run):
            modified_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"完成！修改了 {modified_count}/{len(files_to_process)} 個文件")
    logging.info(f"{'='*60}")
    
    if args.dry_run:
        logging.info("\n⚠️  這是模擬執行，實際文件未修改")
        logging.info("執行 python scripts/tools/migrate_domain_config.py --batch configs/ 來應用變更")


if __name__ == "__main__":
    main()
