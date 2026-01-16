#!/usr/bin/env python3
"""批次清理配置文件中的冗余參數 - P1 清理

清理項目：
1. normalization.variable_order（自動從 model.output_variables 推斷）
2. physics.channel_flow.Re_bulk（未使用）
3. physics.channel_flow.u_tau（從 Re_tau 自動計算）
4. physics.channel_flow.pressure_gradient（從 Re_tau 自動計算）
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List

PROJECT_ROOT = Path("/Users/latteine/Documents/coding/pinns-sparse-flow")


def remove_nested_key(data: Dict[str, Any], key_path: List[str]) -> bool:
    """
    移除嵌套字典中的鍵

    Args:
        data: 配置字典
        key_path: 鍵路徑列表，如 ['normalization', 'variable_order']

    Returns:
        是否成功移除（True）或鍵不存在（False）
    """
    if len(key_path) == 0:
        return False

    if len(key_path) == 1:
        if key_path[0] in data:
            del data[key_path[0]]
            return True
        return False

    # 遞歸處理嵌套鍵
    if key_path[0] in data and isinstance(data[key_path[0]], dict):
        return remove_nested_key(data[key_path[0]], key_path[1:])

    return False


def cleanup_config_file(filepath: Path) -> Dict[str, int]:
    """
    清理單個配置文件

    Returns:
        清理統計：{'variable_order': 0/1, 'Re_bulk': 0/1, 'u_tau': 0/1, 'pressure_gradient': 0/1}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    stats = {
        'variable_order': 0,
        'Re_bulk': 0,
        'u_tau': 0,
        'pressure_gradient': 0
    }

    # 清理 normalization.variable_order
    if remove_nested_key(config, ['normalization', 'variable_order']):
        stats['variable_order'] = 1

    # 清理 physics.channel_flow.Re_bulk
    if remove_nested_key(config, ['physics', 'channel_flow', 'Re_bulk']):
        stats['Re_bulk'] = 1

    # 清理 physics.channel_flow.u_tau
    if remove_nested_key(config, ['physics', 'channel_flow', 'u_tau']):
        stats['u_tau'] = 1

    # 清理 physics.channel_flow.pressure_gradient
    if remove_nested_key(config, ['physics', 'channel_flow', 'pressure_gradient']):
        stats['pressure_gradient'] = 1

    # 如果有任何清理，寫回文件
    if any(stats.values()):
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    return stats


def main():
    """主函數"""
    configs_dir = PROJECT_ROOT / "configs"
    config_files = list(configs_dir.rglob("*.yml"))

    print("=" * 80)
    print("🧹 批次清理配置文件 - P1 冗余參數清理")
    print("=" * 80)
    print(f"找到 {len(config_files)} 個配置文件\n")

    total_stats = {
        'variable_order': 0,
        'Re_bulk': 0,
        'u_tau': 0,
        'pressure_gradient': 0,
        'total_files': 0
    }

    modified_files = []

    for filepath in sorted(config_files):
        rel_path = filepath.relative_to(PROJECT_ROOT)
        stats = cleanup_config_file(filepath)

        if any(stats.values()):
            total_stats['total_files'] += 1
            modified_files.append(str(rel_path))

            # 累加統計
            for key in ['variable_order', 'Re_bulk', 'u_tau', 'pressure_gradient']:
                total_stats[key] += stats[key]

            # 顯示清理項目
            cleaned_items = [key for key, count in stats.items() if count > 0]
            print(f"✅ {rel_path}")
            print(f"   清理項目: {', '.join(cleaned_items)}")

    print("\n" + "=" * 80)
    print("📊 清理統計")
    print("=" * 80)
    print(f"修改檔案數: {total_stats['total_files']}/{len(config_files)}")
    print(f"  - variable_order 清理: {total_stats['variable_order']} 個檔案")
    print(f"  - Re_bulk 清理: {total_stats['Re_bulk']} 個檔案")
    print(f"  - u_tau 清理: {total_stats['u_tau']} 個檔案")
    print(f"  - pressure_gradient 清理: {total_stats['pressure_gradient']} 個檔案")
    print(f"未修改檔案數: {len(config_files) - total_stats['total_files']}")

    if modified_files:
        print("\n修改的檔案清單:")
        for f in modified_files:
            print(f"  - {f}")

    print("\n" + "=" * 80)
    print("✅ 清理完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
