#!/usr/bin/env python3
"""
Output Variables 配置遷移工具（P0-3）

目的：為所有配置文件添加 model.output_variables
     根據 physics.type 和 model.out_dim 推斷變數順序

使用方法：
    python scripts/tools/add_output_variables.py --config <file>
    python scripts/tools/add_output_variables.py --batch configs/
"""

import argparse
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def infer_output_variables(config: Dict[str, Any]) -> List[str]:
    """
    根據配置推斷輸出變數順序
    
    規則：
    1. 2D 流場（Kolmogorov, Channel 2D）: [u, v, p]
    2. 3D 流場（Channel 3D）: [u, v, w, p]
    3. 根據 model.out_dim 決定
    
    Args:
        config: 完整配置字典
    
    Returns:
        輸出變數列表，例如 ['u', 'v', 'p']
    """
    model_cfg = config.get('model', {})
    physics_cfg = config.get('physics', {})
    
    # 獲取輸出維度
    out_dim = model_cfg.get('out_dim')
    
    # 檢查物理類型
    physics_type = physics_cfg.get('type', '')
    
    # 規則 1: 明確的物理類型
    if 'kolmogorov' in physics_type.lower():
        # Kolmogorov flow 是 2D，變數: u, v, p
        return ['u', 'v', 'p']
    
    if 'channel' in physics_type.lower():
        domain = physics_cfg.get('domain', {})
        # 檢查是否有 z_range（3D）
        if 'z_range' in domain:
            z_range = domain['z_range']
            # 如果 z 範圍不是預設值 [0, 1]，則為 3D
            if z_range != [0, 1] and z_range != [0.0, 1.0]:
                return ['u', 'v', 'w', 'p']
        # 否則為 2D
        return ['u', 'v', 'p']
    
    # 規則 2: 根據 out_dim 推斷
    if out_dim == 3:
        return ['u', 'v', 'p']
    elif out_dim == 4:
        return ['u', 'v', 'w', 'p']
    elif out_dim == 2:
        return ['u', 'v']
    elif out_dim == 1:
        return ['u']
    elif out_dim == 5:
        return ['u', 'v', 'w', 'p', 'S']
    
    # 預設：假設為 2D 流場
    logging.warning(f"無法確定輸出維度，使用預設值 ['u', 'v', 'p']")
    return ['u', 'v', 'p']


def add_output_variables_to_config(config_path: Path, dry_run: bool = False) -> bool:
    """
    為配置文件添加 model.output_variables
    
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
        
        # 檢查是否已有 output_variables
        model_cfg = config.get('model', {})
        existing_vars = (
            model_cfg.get('output_variables') or 
            model_cfg.get('variable_names') or 
            model_cfg.get('variables')
        )
        
        if existing_vars:
            logging.info(f"  ✓ 已存在變數定義: {existing_vars}")
            return False
        
        # 推斷輸出變數
        output_vars = infer_output_variables(config)
        
        # 驗證與 out_dim 一致
        out_dim = model_cfg.get('out_dim')
        if out_dim and len(output_vars) != out_dim:
            logging.warning(
                f"  ⚠️  推斷的變數數量 ({len(output_vars)}) 與 out_dim ({out_dim}) 不一致"
            )
        
        # 添加 output_variables
        if 'model' not in config:
            config['model'] = {}
        
        config['model']['output_variables'] = output_vars
        logging.info(f"  ✓ 添加 model.output_variables: {output_vars}")
        
        # 寫入變更
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
    
    except Exception as e:
        logging.error(f"  ❌ 處理失敗: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Output Variables 配置遷移工具')
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
    logging.info(f"Output Variables 配置遷移工具 (P0-3)")
    logging.info(f"{'='*60}")
    logging.info(f"找到 {len(files_to_process)} 個配置文件")
    
    modified_count = 0
    for config_path in files_to_process:
        if add_output_variables_to_config(config_path, dry_run=args.dry_run):
            modified_count += 1
    
    logging.info(f"\n{'='*60}")
    logging.info(f"完成！修改了 {modified_count}/{len(files_to_process)} 個文件")
    logging.info(f"{'='*60}")
    
    if args.dry_run:
        logging.info("\n⚠️  這是模擬執行，實際文件未修改")
        logging.info("執行 python scripts/tools/add_output_variables.py --batch configs/ 來應用變更")


if __name__ == "__main__":
    main()
