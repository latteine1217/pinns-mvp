#!/usr/bin/env python3
"""
全面遷移配置文件：從標準 Fourier（方式1）遷移到週期性 Fourier（方式2）

用法:
    python scripts/tools/migrate_to_periodic_fourier.py [--backup] [--dry-run]
    
選項:
    --backup    創建備份文件（推薦）
    --dry-run   僅顯示將要修改的文件，不實際修改
"""

import yaml
import re
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import argparse
from datetime import datetime


# ==================== 配置映射規則 ====================

def get_periodic_config_for_kolmogorov_2d(original_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成 Kolmogorov Flow 2D 的週期性 Fourier 配置
    
    原配置範例:
        type: standard
        fourier_m: 64
        fourier_sigma: 2.0
    
    新配置:
        type: hybrid
        axes:
          0: {type: standard, n_modes: 12, sigma: 4.0, use_2pi: true}  # time
          1: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # x
          2: {type: periodic, domain_size: 6.283185307179586, n_modes: 8}  # y
        trainable_fourier: false
    """
    # 從原配置推斷參數
    original_m = original_config.get('fourier_m', 64)
    original_sigma = original_config.get('fourier_sigma', 2.0)
    
    # 計算合理的 n_modes（週期軸使用較少模態）
    time_modes = max(12, original_m // 8)
    spatial_modes = max(8, original_m // 8)
    
    return {
        'type': 'hybrid',
        'axes': {
            0: {
                'type': 'standard',
                'n_modes': time_modes,
                'sigma': original_sigma * 2.0,  # 時間軸使用較大 sigma
                'use_2pi': True
            },
            1: {
                'type': 'periodic',
                'domain_size': 6.283185307179586,  # 2π
                'n_modes': spatial_modes
            },
            2: {
                'type': 'periodic',
                'domain_size': 6.283185307179586,  # 2π
                'n_modes': spatial_modes
            }
        },
        'trainable_fourier': False
    }


def get_periodic_config_for_channel_flow(original_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    生成 Channel Flow 3D 的週期性 Fourier 配置
    
    x (streamwise): 週期性
    y (wall-normal): 非週期性
    z (spanwise): 週期性
    """
    original_m = original_config.get('fourier_m', 64)
    original_sigma = original_config.get('fourier_sigma', 2.0)
    
    time_modes = max(24, original_m // 4)
    periodic_modes = max(16, original_m // 4)
    wall_modes = max(20, original_m // 3)
    
    return {
        'type': 'hybrid',
        'axes': {
            0: {
                'type': 'standard',
                'n_modes': time_modes,
                'sigma': original_sigma * 2.5,
                'use_2pi': True
            },
            1: {
                'type': 'periodic',
                'domain_size': 25.13,  # 8π (streamwise)
                'n_modes': periodic_modes
            },
            2: {
                'type': 'standard',  # wall-normal 非週期
                'n_modes': wall_modes,
                'sigma': original_sigma,
                'use_2pi': True
            },
            3: {
                'type': 'periodic',
                'domain_size': 9.42,  # 3π (spanwise)
                'n_modes': periodic_modes
            }
        },
        'trainable_fourier': False
    }


def detect_problem_type(config: Dict[str, Any], filepath: Path) -> str:
    """
    檢測問題類型（Kolmogorov 2D 或 Channel Flow 3D）
    
    Returns:
        'kolmogorov_2d' | 'channel_flow_3d' | 'unknown'
    """
    # 檢查文件名
    filename = filepath.name.lower()
    if 'kolmogorov' in filename or 'kf' in filename:
        return 'kolmogorov_2d'
    if 'channel' in filename or 'main' in filename:
        return 'channel_flow_3d'
    
    # 檢查配置內容
    physics_type = config.get('physics', {}).get('type', '')
    if 'kolmogorov' in physics_type.lower():
        return 'kolmogorov_2d'
    if 'channel' in physics_type.lower():
        return 'channel_flow_3d'
    
    # 檢查數據源
    data_source = config.get('data', {}).get('source', '')
    if 'kolmogorov' in data_source.lower():
        return 'kolmogorov_2d'
    
    # 檢查 in_dim
    in_dim = config.get('model', {}).get('in_dim', 3)
    if in_dim == 3:
        return 'kolmogorov_2d'  # (t, x, y)
    elif in_dim == 4:
        return 'channel_flow_3d'  # (t, x, y, z)
    
    return 'unknown'


# ==================== 遷移邏輯 ====================

def migrate_config_file(filepath: Path, backup: bool = True, dry_run: bool = False) -> bool:
    """
    遷移單個配置文件
    
    Returns:
        True if migrated, False if skipped or error
    """
    try:
        # 讀取原始內容
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 讀取 YAML 配置來檢查實際活動的配置
        with open(filepath, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 檢查活動的 fourier_features 類型（不是註解中的）
        ff_config = config.get('model', {}).get('fourier_features', {})
        if ff_config.get('type') == 'hybrid':
            print(f"⏭️  已是 hybrid: {filepath.name}")
            return False
        
        # 跳過沒有 fourier_features 的文件
        if not ff_config:
            print(f"⏭️  無 fourier_features: {filepath.name}")
            return False
        
        # 獲取原始 fourier_features 配置
        original_ff = ff_config
        ff_type = ff_config.get('type', 'standard')
        
        # 跳過 disabled 類型（用於消融實驗的基線）
        if ff_type == 'disabled':
            print(f"⏭️  保留 disabled: {filepath.name} (消融實驗基線)")
            return False
        
        # 檢測問題類型
        problem_type = detect_problem_type(config, filepath)
        
        if problem_type == 'unknown':
            print(f"⚠️  無法檢測類型: {filepath.name}")
            return False
        
        # 生成新的週期性配置
        if problem_type == 'kolmogorov_2d':
            new_ff_config = get_periodic_config_for_kolmogorov_2d(original_ff)
        else:  # channel_flow_3d
            new_ff_config = get_periodic_config_for_channel_flow(original_ff)
        
        if dry_run:
            print(f"🔍 [DRY-RUN] {filepath.name} ({problem_type})")
            print(f"   原配置: type={ff_type}, m={original_ff.get('fourier_m', 'N/A')}")
            print(f"   新配置: type=hybrid, axes={len(new_ff_config['axes'])}")
            return True
        
        # 備份原文件
        if backup:
            backup_dir = filepath.parent / '.backup_before_periodic_migration'
            backup_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"{filepath.stem}_{timestamp}.yml"
            shutil.copy2(filepath, backup_path)
        
        # 使用正則替換 fourier_features 區段
        # 匹配模式：從 fourier_features 開始到下一個同級鍵或檔尾
        pattern = r'(  fourier_features:.*?)(?=\n  [a-z_]+:|$)'
        
        # 構建新的 YAML 區段
        new_section = f"  fourier_features:\n"
        new_section += f"    type: hybrid\n"
        new_section += f"    axes:\n"
        
        for axis_idx, axis_config in new_ff_config['axes'].items():
            axis_type = axis_config['type']
            if axis_type == 'periodic':
                domain_size = axis_config['domain_size']
                n_modes = axis_config['n_modes']
                new_section += f"      {axis_idx}: {{type: periodic, domain_size: {domain_size}, n_modes: {n_modes}}}\n"
            else:  # standard
                n_modes = axis_config['n_modes']
                sigma = axis_config['sigma']
                use_2pi = str(axis_config.get('use_2pi', True)).lower()
                new_section += f"      {axis_idx}: {{type: standard, n_modes: {n_modes}, sigma: {sigma}, use_2pi: {use_2pi}}}\n"
        
        new_section += f"    trainable_fourier: false"
        
        # 執行替換
        new_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        
        # 如果替換失敗，嘗試另一種模式（註解版本）
        if new_content == content:
            # 匹配包含註解的完整區段
            pattern2 = r'  # ========== Fourier Features.*?(?=\n  [a-z_]+:|$)'
            new_content = re.sub(pattern2, new_section, content, flags=re.DOTALL)
        
        # 寫回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"✅ 已遷移: {filepath.name} → {problem_type}")
        return True
    
    except Exception as e:
        print(f"❌ 錯誤 {filepath.name}: {e}")
        return False


def main():
    """主函數：批量遷移所有配置文件"""
    parser = argparse.ArgumentParser(description='遷移配置文件到週期性 Fourier 嵌入')
    parser.add_argument('--backup', action='store_true', help='創建備份文件（推薦）')
    parser.add_argument('--dry-run', action='store_true', help='僅顯示將要修改的文件')
    args = parser.parse_args()
    
    # 定義配置目錄
    project_root = Path(__file__).parent.parent.parent
    configs_dir = project_root / 'configs'
    
    print("=" * 70)
    print("🔄 週期性 Fourier 嵌入遷移工具")
    print("=" * 70)
    print(f"\n📂 掃描目錄: {configs_dir}")
    
    if args.dry_run:
        print("🔍 模式: DRY-RUN (不會實際修改文件)")
    else:
        print(f"✏️  模式: 實際修改")
        if args.backup:
            print(f"💾 備份: 啟用 (備份至 .backup_before_periodic_migration/)")
        else:
            print(f"⚠️  備份: 禁用 (建議使用 --backup)")
    
    # 查找所有配置文件
    config_files = sorted(configs_dir.rglob('*.yml'))
    
    # 排除範例文件（它們已經是 hybrid）
    exclude_patterns = ['periodic_example', 'periodic_fourier']
    config_files = [f for f in config_files if not any(p in f.name for p in exclude_patterns)]
    
    print(f"\n📋 找到 {len(config_files)} 個配置文件")
    print("=" * 70)
    print()
    
    migrated_count = 0
    skipped_count = 0
    error_count = 0
    
    for config_file in config_files:
        result = migrate_config_file(config_file, backup=args.backup, dry_run=args.dry_run)
        if result is True:
            migrated_count += 1
        elif result is False:
            skipped_count += 1
        else:
            error_count += 1
    
    print()
    print("=" * 70)
    print(f"\n📊 遷移統計:")
    print(f"  ✅ 已遷移: {migrated_count} 個")
    print(f"  ⏭️  跳過: {skipped_count} 個")
    print(f"  ❌ 錯誤: {error_count} 個")
    print(f"  📝 總計: {len(config_files)} 個")
    
    if args.dry_run and migrated_count > 0:
        print(f"\n💡 提示: 移除 --dry-run 以實際執行遷移")
        print(f"💡 建議: 添加 --backup 以創建備份")
    elif migrated_count > 0:
        print(f"\n🎉 成功遷移 {migrated_count} 個配置文件到週期性 Fourier 嵌入！")
        if args.backup:
            print(f"💾 備份文件已保存至各配置目錄的 .backup_before_periodic_migration/")
    else:
        print(f"\n✨ 所有配置文件已是最新狀態或已跳過")


if __name__ == '__main__':
    main()
