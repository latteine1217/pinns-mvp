#!/usr/bin/env python3
"""
批量簡化 Kolmogorov Flow YAML 配置文件

移除可由 DNS NPY 自動提供的冗餘參數，保留必要的配置。

參數優先級：DNS NPY（自動回填）> YAML 配置 > 預設值

使用方法：
    python scripts/tools/simplify_kolmogorov_configs.py --dry-run    # 預覽修改
    python scripts/tools/simplify_kolmogorov_configs.py              # 執行修改
    python scripts/tools/simplify_kolmogorov_configs.py --backup     # 執行修改並備份
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml


# ============================================================================
# 配置簡化規則
# ============================================================================

def simplify_kolmogorov_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """簡化 Kolmogorov Flow 配置文件

    Args:
        config: 原始配置字典

    Returns:
        (簡化後的配置, 修改記錄列表)
    """
    changes = []
    simplified = config.copy()

    # ========================================================
    # 1. 簡化 data.kolmogorov_config
    # ========================================================
    if 'data' in simplified and 'kolmogorov_config' in simplified['data']:
        kol_cfg = simplified['data']['kolmogorov_config']

        # 刪除 physics_params（DNS NPY 會自動回填）
        if 'physics_params' in kol_cfg:
            params = kol_cfg.pop('physics_params')
            changes.append(f"  ✂️  data.kolmogorov_config.physics_params: {params}")

        # 刪除 resolution（DNS NPY 自動推導）
        if 'resolution' in kol_cfg:
            res = kol_cfg.pop('resolution')
            changes.append(f"  ✂️  data.kolmogorov_config.resolution: {res}")

        # 刪除 dt（DNS NPY 提供）
        if 'dt' in kol_cfg:
            dt = kol_cfg.pop('dt')
            changes.append(f"  ✂️  data.kolmogorov_config.dt: {dt}")

        # 刪除 L（DNS NPY 提供）
        if 'L' in kol_cfg:
            L = kol_cfg.pop('L')
            changes.append(f"  ✂️  data.kolmogorov_config.L: {L}")

    # ========================================================
    # 2. 簡化 physics 區段
    # ========================================================
    if 'physics' in simplified:
        physics = simplified['physics']

        # 刪除 kolmogorov_flow（重複配置）
        if 'kolmogorov_flow' in physics:
            kf = physics.pop('kolmogorov_flow')
            changes.append(f"  ✂️  physics.kolmogorov_flow: {kf}")

        # 簡化 forcing 為空字典（DNS NPY 會提供所有參數）
        if 'forcing' in physics and physics['forcing']:
            old_forcing = physics['forcing']
            physics['forcing'] = {}
            changes.append(f"  ✂️  physics.forcing: {old_forcing} → {{}}")

        # 簡化 domain（只保留起點，終點從 DNS NPY 的 L 自動推導）
        if 'domain' in physics:
            domain = physics['domain']

            # 保留邊界條件
            if 'x_range' in domain:
                x_range = domain['x_range']
                if isinstance(x_range, list) and len(x_range) == 2:
                    # 移除 x_max（會從 L 自動推導）
                    changes.append(f"  ✂️  physics.domain.x_range[1]: {x_range[1]} (將從 DNS NPY L 自動推導)")

            if 'y_range' in domain:
                y_range = domain['y_range']
                if isinstance(y_range, list) and len(y_range) == 2:
                    # 移除 y_max（會從 L 自動推導）
                    changes.append(f"  ✂️  physics.domain.y_range[1]: {y_range[1]} (將從 DNS NPY L 自動推導)")

            # 移除舊格式的 x_min, x_max, y_min, y_max
            for key in ['x_min', 'x_max', 'y_min', 'y_max']:
                if key in domain:
                    val = domain.pop(key)
                    changes.append(f"  ✂️  physics.domain.{key}: {val}")

    # ========================================================
    # 3. 添加註解說明（如果是第一次簡化）
    # ========================================================
    if changes and 'data' in simplified and 'kolmogorov_config' in simplified['data']:
        kol_cfg = simplified['data']['kolmogorov_config']
        if 'description' in kol_cfg and 'auto-sync' not in kol_cfg['description']:
            kol_cfg['description'] = (
                f"{kol_cfg['description']} "
                f"[物理參數由 DNS NPY 自動同步]"
            )
            changes.append("  ✨ 添加自動同步說明到 description")

    return simplified, changes


def should_process_config(config_path: Path) -> bool:
    """判斷配置文件是否需要處理

    Args:
        config_path: 配置文件路徑

    Returns:
        True 如果是 Kolmogorov Flow 配置且未被簡化
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # 檢查是否為 Kolmogorov Flow 配置
        physics_type = config.get('physics', {}).get('type', '')
        if physics_type != 'kolmogorov_flow_2d':
            return False

        # 檢查是否已簡化（有 physics_params 表示未簡化）
        has_redundant = (
            'physics_params' in config.get('data', {}).get('kolmogorov_config', {}) or
            'kolmogorov_flow' in config.get('physics', {})
        )

        return has_redundant

    except Exception as e:
        logging.warning(f"無法讀取配置文件 {config_path}: {e}")
        return False


def backup_config(config_path: Path) -> Path:
    """備份配置文件

    Args:
        config_path: 原始配置文件路徑

    Returns:
        備份文件路徑
    """
    backup_path = config_path.with_suffix('.yml.backup')
    shutil.copy2(config_path, backup_path)
    return backup_path


def process_config_file(
    config_path: Path,
    dry_run: bool = False,
    create_backup: bool = False
) -> Tuple[bool, List[str]]:
    """處理單個配置文件

    Args:
        config_path: 配置文件路徑
        dry_run: 是否為試運行（不實際修改文件）
        create_backup: 是否創建備份

    Returns:
        (是否有修改, 修改記錄列表)
    """
    try:
        # 讀取配置
        with open(config_path, 'r', encoding='utf-8') as f:
            original_config = yaml.safe_load(f)

        # 簡化配置
        simplified_config, changes = simplify_kolmogorov_config(original_config)

        if not changes:
            return False, []

        if not dry_run:
            # 備份（如果需要）
            if create_backup:
                backup_path = backup_config(config_path)
                changes.append(f"  💾 已備份至: {backup_path.name}")

            # 寫入簡化後的配置
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    simplified_config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                    indent=2
                )

        return True, changes

    except Exception as e:
        logging.error(f"處理配置文件失敗 {config_path}: {e}")
        return False, []


def find_config_files(search_dirs: List[Path]) -> List[Path]:
    """尋找所有需要處理的配置文件

    Args:
        search_dirs: 搜索目錄列表

    Returns:
        配置文件路徑列表
    """
    config_files = []

    for search_dir in search_dirs:
        if not search_dir.exists():
            logging.warning(f"目錄不存在: {search_dir}")
            continue

        # 遞迴搜索所有 .yml 文件
        for config_path in search_dir.rglob('*.yml'):
            # 排除備份文件
            if config_path.suffix == '.backup' or '.backup' in config_path.name:
                continue

            if should_process_config(config_path):
                config_files.append(config_path)

    return sorted(config_files)


# ============================================================================
# 主程式
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='批量簡化 Kolmogorov Flow YAML 配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 預覽修改（不實際修改文件）
  python scripts/tools/simplify_kolmogorov_configs.py --dry-run

  # 執行修改
  python scripts/tools/simplify_kolmogorov_configs.py

  # 執行修改並備份原始文件
  python scripts/tools/simplify_kolmogorov_configs.py --backup

  # 只處理特定目錄
  python scripts/tools/simplify_kolmogorov_configs.py --dir configs/experiments
        """
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='試運行模式（預覽修改但不實際修改文件）'
    )

    parser.add_argument(
        '--backup',
        action='store_true',
        help='修改前備份原始配置文件（.yml.backup）'
    )

    parser.add_argument(
        '--dir',
        type=str,
        action='append',
        default=None,
        help='指定搜索目錄（可多次指定，預設為 configs/）'
    )

    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='顯示詳細日誌'
    )

    args = parser.parse_args()

    # 設置日誌
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(message)s'
    )

    # 確定搜索目錄
    if args.dir:
        search_dirs = [Path(d) for d in args.dir]
    else:
        search_dirs = [Path('configs')]

    # 尋找配置文件
    logging.info("="*70)
    logging.info("🔍 搜索需要簡化的 Kolmogorov Flow 配置文件...")
    logging.info("="*70)

    config_files = find_config_files(search_dirs)

    if not config_files:
        logging.info("\n✅ 未找到需要簡化的配置文件（所有文件已是最新格式）")
        return

    logging.info(f"\n找到 {len(config_files)} 個配置文件需要簡化:\n")
    for config_path in config_files:
        logging.info(f"  📄 {config_path}")

    # 處理配置文件
    logging.info("\n" + "="*70)
    if args.dry_run:
        logging.info("🔍 試運行模式（預覽修改，不實際修改文件）")
    else:
        logging.info("✂️  開始簡化配置文件...")
    logging.info("="*70)

    total_modified = 0
    total_changes = 0

    for config_path in config_files:
        logging.info(f"\n📄 處理: {config_path}")

        modified, changes = process_config_file(
            config_path,
            dry_run=args.dry_run,
            create_backup=args.backup
        )

        if modified:
            total_modified += 1
            total_changes += len(changes)

            logging.info(f"  修改項目 ({len(changes)}):")
            for change in changes:
                logging.info(change)
        else:
            logging.info("  ⏭️  已是最新格式，跳過")

    # 總結
    logging.info("\n" + "="*70)
    if args.dry_run:
        logging.info(f"🔍 試運行完成")
        logging.info(f"   將修改 {total_modified} 個文件")
        logging.info(f"   總計 {total_changes} 項修改")
        logging.info(f"\n   執行實際修改：")
        logging.info(f"   python scripts/tools/simplify_kolmogorov_configs.py")
    else:
        logging.info(f"✅ 簡化完成")
        logging.info(f"   已修改 {total_modified} 個文件")
        logging.info(f"   總計 {total_changes} 項修改")
        if args.backup:
            logging.info(f"   已創建備份文件（.yml.backup）")
    logging.info("="*70)


if __name__ == '__main__':
    main()
