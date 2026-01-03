"""
配置快照工具

功能：
1. 保存實驗運行時的完整 config（包含所有覆寫與動態計算）
2. 記錄 config 的變更歷史（template → sweep → auto-derived）
3. 支援 diff 比較（與 template 的差異）
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import copy


class ConfigSnapshot:
    """配置快照管理器"""

    def __init__(self, config: Dict[str, Any], source_template: Optional[str] = None):
        """
        初始化配置快照

        Args:
            config: 完整的運行配置
            source_template: 源 template 檔案路徑（可選）
        """
        self.config = copy.deepcopy(config)
        self.source_template = source_template
        self.metadata = {
            'timestamp': datetime.now().isoformat(),
            'source_template': source_template,
            'sweep_overrides': [],
            'auto_derived': [],
        }

    def record_sweep_override(self, key: str, old_value: Any, new_value: Any):
        """記錄 Sweep 覆寫的參數"""
        self.metadata['sweep_overrides'].append({
            'key': key,
            'old': old_value,
            'new': new_value,
        })

    def record_auto_derived(self, key: str, value: Any, reason: str):
        """記錄自動推導的參數"""
        self.metadata['auto_derived'].append({
            'key': key,
            'value': value,
            'reason': reason,
        })

    def save(self, output_dir: Path, filename: str = 'config_snapshot.yml'):
        """
        保存完整配置快照

        Args:
            output_dir: 輸出目錄（通常是 checkpoints/{exp_name}/）
            filename: 檔案名稱
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        snapshot_path = output_dir / filename
        metadata_path = output_dir / 'config_metadata.json'

        # 保存完整 config（YAML 格式，易讀）
        with open(snapshot_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        # 保存 metadata（JSON 格式，易解析）
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)

        return snapshot_path, metadata_path

    def get_diff_summary(self) -> str:
        """
        獲取配置變更摘要（用於 logging）

        Returns:
            格式化的變更摘要字串
        """
        lines = []
        lines.append("=" * 80)
        lines.append("📋 Config Snapshot Summary")
        lines.append("=" * 80)

        # 源 template
        if self.source_template:
            lines.append(f"Source Template: {self.source_template}")
        else:
            lines.append("Source Template: <Direct config>")

        # Sweep 覆寫
        if self.metadata['sweep_overrides']:
            lines.append(f"\n🔄 Sweep Overrides ({len(self.metadata['sweep_overrides'])}):")
            for override in self.metadata['sweep_overrides']:
                lines.append(f"  • {override['key']}: {override['old']} → {override['new']}")
        else:
            lines.append("\n🔄 Sweep Overrides: None")

        # 自動推導
        if self.metadata['auto_derived']:
            lines.append(f"\n🔧 Auto-Derived Parameters ({len(self.metadata['auto_derived'])}):")
            for derived in self.metadata['auto_derived']:
                lines.append(f"  • {derived['key']}: {derived['value']} ({derived['reason']})")
        else:
            lines.append("\n🔧 Auto-Derived Parameters: None")

        lines.append("=" * 80)
        return "\n".join(lines)


def create_snapshot_with_tracking(
    config: Dict[str, Any],
    source_template: Optional[str] = None,
    sweep_overrides: Optional[List[Dict[str, Any]]] = None,
    auto_derived: Optional[List[Dict[str, Any]]] = None,
) -> ConfigSnapshot:
    """
    建立配置快照並記錄變更歷史

    Args:
        config: 完整配置
        source_template: 源 template 路徑
        sweep_overrides: Sweep 覆寫列表（從 wandb.config 追蹤）
        auto_derived: 自動推導參數列表

    Returns:
        ConfigSnapshot 實例
    """
    snapshot = ConfigSnapshot(config, source_template)

    # 記錄 sweep 覆寫
    if sweep_overrides:
        for override in sweep_overrides:
            snapshot.record_sweep_override(
                override['key'],
                override.get('old', '<未設置>'),
                override['new']
            )

    # 記錄自動推導
    if auto_derived:
        for derived in auto_derived:
            snapshot.record_auto_derived(
                derived['key'],
                derived['value'],
                derived.get('reason', 'auto-derived')
            )

    return snapshot


def compare_configs(config_a: Dict[str, Any], config_b: Dict[str, Any]) -> Dict[str, Any]:
    """
    比較兩個配置的差異

    Args:
        config_a: 配置 A（例如 template）
        config_b: 配置 B（例如運行配置）

    Returns:
        差異字典 {key: (value_a, value_b)}
    """
    def _flatten_dict(d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """扁平化嵌套字典"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_a = _flatten_dict(config_a)
    flat_b = _flatten_dict(config_b)

    # 找出差異
    diff = {}
    all_keys = set(flat_a.keys()) | set(flat_b.keys())

    for key in all_keys:
        val_a = flat_a.get(key, '<未設置>')
        val_b = flat_b.get(key, '<未設置>')

        if val_a != val_b:
            diff[key] = {'template': val_a, 'runtime': val_b}

    return diff
