#!/usr/bin/env python3
"""
批次為所有 config 檔案加入 WandB metadata (group, tags, experiment_id)

使用方式：
    python scripts/tools/add_wandb_metadata_to_configs.py --dry-run  # 預覽
    python scripts/tools/add_wandb_metadata_to_configs.py           # 實際執行
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import yaml


# ============================================================================
# 實驗 ID 到 Group/Tags 的映射規則
# ============================================================================
EXPERIMENT_MAPPING = {
    # S1: Random vs QR sensor strategy
    's1_random': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'random', 'K100', 'vanilla_mlp', 'no_prior'],
        'experiment_id': 'S1',
    },
    's1_qr': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'qr_pivot', 'K100', 'vanilla_mlp', 'no_prior'],
        'experiment_id': 'S1',
    },

    # S2: K-scan
    's2_qr_K30': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'qr_pivot', 'K30', 'vanilla_mlp', 's2_kscan'],
        'experiment_id': 'S2',
    },
    's2_qr_K50': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'qr_pivot', 'K50', 'vanilla_mlp', 's2_kscan'],
        'experiment_id': 'S2',
    },
    's2_qr_K80': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'qr_pivot', 'K80', 'vanilla_mlp', 's2_kscan'],
        'experiment_id': 'S2',
    },
    's2_qr_K100': {
        'group': 'kflow_sensor_strategy',
        'tags': ['kflow_re50', 'qr_pivot', 'K100', 'vanilla_mlp', 's2_kscan'],
        'experiment_id': 'S2',
    },

    # M1: Vanilla vs Full model comparison
    'm1_vanilla': {
        'group': 'kflow_representation',
        'tags': ['kflow_re50', 'vanilla_mlp', 'K100', 'no_prior'],
        'experiment_id': 'M1',
    },
    'm1_full': {
        'group': 'kflow_representation',
        'tags': ['kflow_re50', 'full', 'K100', 'no_prior'],
        'experiment_id': 'M1',
    },

    # A1: Fourier ablation
    'a1_with_fourier': {
        'group': 'kflow_ablation',
        'tags': ['kflow_re50', 'full', 'K100', 'with_fourier'],
        'experiment_id': 'A1',
    },
    'a1_without_fourier': {
        'group': 'kflow_ablation',
        'tags': ['kflow_re50', 'full', 'K100', 'no_fourier'],
        'experiment_id': 'A1',
    },

    # A2: Adaptive weights ablation
    'a2_with_adaptive': {
        'group': 'kflow_ablation',
        'tags': ['kflow_re50', 'full', 'K100', 'with_gradnorm'],
        'experiment_id': 'A2',
    },
    'a2_without_adaptive': {
        'group': 'kflow_ablation',
        'tags': ['kflow_re50', 'full', 'K100', 'no_gradnorm'],
        'experiment_id': 'A2',
    },

    # C1: Prior comparison
    'c1_no_prior': {
        'group': 'kflow_prior',
        'tags': ['kflow_re50', 'K100', 'full', 'no_prior'],
        'experiment_id': 'C1',
    },
    'c1_with_prior': {
        'group': 'kflow_prior',
        'tags': ['kflow_re50', 'K100', 'full', 'leith_prior'],
        'experiment_id': 'C1',
    },

    # C2: Prior weight sweep
    'c2_prior_0.1': {
        'group': 'kflow_prior',
        'tags': ['kflow_re50', 'K100', 'full', 'leith_prior', 'c2_prior_sweep'],
        'experiment_id': 'C2',
    },
    'c2_prior_0.3': {
        'group': 'kflow_prior',
        'tags': ['kflow_re50', 'K100', 'full', 'leith_prior', 'c2_prior_sweep'],
        'experiment_id': 'C2',
    },
    'c2_prior_0.5': {
        'group': 'kflow_prior',
        'tags': ['kflow_re50', 'K100', 'full', 'leith_prior', 'c2_prior_sweep'],
        'experiment_id': 'C2',
    },

    # 通用 Kolmogorov 配置（未歸類到實驗矩陣）
    'kolmogorov': {
        'group': 'kflow_baseline',
        'tags': ['kflow_re50', 'K100'],
        'experiment_id': 'BASELINE_2D',
    },
}


def detect_experiment_pattern(filepath: Path) -> Optional[str]:
    """
    從檔案路徑或檔案名稱推斷實驗類型

    Args:
        filepath: Config 檔案路徑

    Returns:
        實驗 pattern key（例如 's1_random', 'm1_vanilla'）或 None
    """
    filepath_str = str(filepath).lower()
    filename = filepath.name.lower()

    # S2 K-scan 特殊處理（因為檔案名是 s2_qr_K30, s2_qr_K50 等）
    if 's2_' in filename and '_qr_' in filename:
        if '_k30' in filename or 'k30_' in filename:
            return 's2_qr_K30'
        elif '_k50' in filename or 'k50_' in filename:
            return 's2_qr_K50'
        elif '_k80' in filename or 'k80_' in filename:
            return 's2_qr_K80'
        elif '_k100' in filename or 'k100_' in filename:
            return 's2_qr_K100'

    # 優先從檔案名稱精確匹配
    for pattern in EXPERIMENT_MAPPING.keys():
        # 處理帶數字的 pattern（例如 c2_prior_0.1）
        if '.' in pattern:
            pattern_normalized = pattern.replace('.', '')
            filename_normalized = filename.replace('.', '')
            if pattern_normalized in filename_normalized:
                return pattern
        else:
            if pattern in filename:
                return pattern

    # 從路徑匹配（例如 S1_sensor_strategy/s1_random_xxx.yml）
    for pattern in EXPERIMENT_MAPPING.keys():
        if pattern.replace('_', '') in filepath_str.replace('_', '').replace('.', ''):
            return pattern

    # 通用 Kolmogorov 配置
    if 'kolmogorov' in filename:
        return 'kolmogorov'

    return None


def add_wandb_metadata(config: Dict, metadata: Dict) -> Dict:
    """
    在 config['experiment'] 中加入 WandB metadata

    Args:
        config: 原始配置字典
        metadata: {group, tags, experiment_id}

    Returns:
        更新後的配置字典
    """
    if 'experiment' not in config:
        config['experiment'] = {}

    exp_config = config['experiment']

    # 加入 group（如果尚未存在）
    if 'group' not in exp_config:
        exp_config['group'] = metadata['group']

    # 加入 tags（如果尚未存在）
    if 'tags' not in exp_config:
        exp_config['tags'] = metadata['tags']

    # 加入 experiment_id（如果尚未存在）
    if 'experiment_id' not in exp_config:
        exp_config['experiment_id'] = metadata['experiment_id']

    return config


def process_config_file(filepath: Path, dry_run: bool = False) -> Tuple[bool, str]:
    """
    處理單個 config 檔案

    Args:
        filepath: Config 檔案路徑
        dry_run: 如果為 True，只預覽不實際修改

    Returns:
        (是否修改, 訊息)
    """
    # 檢測實驗類型
    pattern = detect_experiment_pattern(filepath)
    if pattern is None:
        return False, f"⏭️  跳過（無法識別實驗類型）: {filepath}"

    metadata = EXPERIMENT_MAPPING[pattern]

    # 載入 config
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return False, f"❌ 載入失敗: {filepath} - {e}"

    # 檢查是否已存在 metadata
    if 'experiment' in config:
        exp = config['experiment']
        if 'group' in exp and 'tags' in exp and 'experiment_id' in exp:
            return False, f"✅ 已存在 metadata: {filepath}"

    # 加入 metadata
    config = add_wandb_metadata(config, metadata)

    # 預覽或實際修改
    if dry_run:
        msg = (
            f"📝 [DRY RUN] {filepath.name}\n"
            f"   → group: {metadata['group']}\n"
            f"   → tags: {metadata['tags']}\n"
            f"   → experiment_id: {metadata['experiment_id']}"
        )
        return True, msg
    else:
        # 保存（保持原有格式）
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            msg = (
                f"✅ 已更新: {filepath.name}\n"
                f"   → group: {metadata['group']}\n"
                f"   → tags: {metadata['tags']}\n"
                f"   → experiment_id: {metadata['experiment_id']}"
            )
            return True, msg
        except Exception as e:
            return False, f"❌ 保存失敗: {filepath} - {e}"


def main():
    parser = argparse.ArgumentParser(description='批次為 config 檔案加入 WandB metadata')
    parser.add_argument('--dry-run', action='store_true', help='預覽模式，不實際修改檔案')
    parser.add_argument('--config-dir', type=str, default='configs',
                        help='Config 目錄路徑（預設：configs）')
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"❌ 目錄不存在: {config_dir}")
        return

    # 找出所有 .yml 檔案（排除 sweeps）
    config_files = []
    for pattern in ['**/*.yml', '**/*.yaml']:
        config_files.extend(config_dir.glob(pattern))

    # 排除 sweeps 和 templates（templates 已手動修改）
    config_files = [
        f for f in config_files
        if 'sweeps' not in str(f) and 'templates' not in str(f)
    ]

    print("=" * 80)
    print("🔧 批次加入 WandB Metadata")
    print("=" * 80)
    print(f"Config 目錄: {config_dir.resolve()}")
    print(f"找到 {len(config_files)} 個配置檔案")
    if args.dry_run:
        print("⚠️  預覽模式（不會實際修改檔案）")
    print("=" * 80)

    # 處理每個檔案
    modified_count = 0
    skipped_count = 0
    error_count = 0

    for filepath in sorted(config_files):
        success, msg = process_config_file(filepath, dry_run=args.dry_run)
        print(msg)

        if success:
            modified_count += 1
        elif 'skip' in msg.lower() or '已存在' in msg:
            skipped_count += 1
        else:
            error_count += 1

    # 總結
    print("\n" + "=" * 80)
    print("📊 執行總結")
    print("=" * 80)
    print(f"✅ 修改/預覽: {modified_count} 個檔案")
    print(f"⏭️  跳過: {skipped_count} 個檔案")
    print(f"❌ 錯誤: {error_count} 個檔案")
    print("=" * 80)

    if args.dry_run:
        print("\n💡 預覽完成。若確認無誤，請執行：")
        print("   python scripts/tools/add_wandb_metadata_to_configs.py")
    else:
        print("\n✅ 所有檔案已更新！")


if __name__ == '__main__':
    main()
