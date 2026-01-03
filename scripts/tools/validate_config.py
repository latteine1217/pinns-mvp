#!/usr/bin/env python3
"""
配置文件驗證工具

用法:
    python scripts/tools/validate_config.py --config <path_to_config.yml>
    python scripts/tools/validate_config.py --config <path> --strict  # 嚴格模式

示例:
    python scripts/tools/validate_config.py --config configs/main.yml
"""

import argparse
import sys
from pathlib import Path
import yaml
import logging

# 添加專案根目錄到 sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.utils.config_validator import validate_config_file


def setup_logging():
    """設置日誌格式"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )


def main():
    """主函數"""
    parser = argparse.ArgumentParser(
        description='驗證 PINNs 配置文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --config configs/main.yml
  %(prog)s --config configs/main.yml --strict
  %(prog)s --config configs/experiments/*.yml  # 批量驗證
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置文件路徑（支持通配符）'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='嚴格模式：警告視為錯誤'
    )
    
    args = parser.parse_args()
    setup_logging()
    
    # 解析配置文件路徑（支持通配符）
    config_path = Path(args.config)
    if '*' in str(config_path):
        # 通配符模式
        config_files = list(config_path.parent.glob(config_path.name))
    else:
        config_files = [config_path]
    
    if not config_files:
        print(f"❌ 未找到配置文件: {args.config}")
        sys.exit(1)
    
    # 驗證每個配置文件
    total = len(config_files)
    passed = 0
    failed = 0
    
    print(f"{'=' * 80}")
    print(f"開始驗證 {total} 個配置文件...")
    print(f"{'=' * 80}\n")
    
    for config_file in sorted(config_files):
        print(f"📄 驗證: {config_file}")
        
        if not config_file.exists():
            print(f"   ❌ 文件不存在\n")
            failed += 1
            continue
        
        try:
            # 讀取配置
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            # 驗證配置
            validate_config_file(config, strict_mode=args.strict)
            
            print(f"   ✅ 驗證通過\n")
            passed += 1
            
        except yaml.YAMLError as e:
            print(f"   ❌ YAML 解析失敗:")
            print(f"      {e}\n")
            failed += 1
            
        except ValueError as e:
            print(f"   ❌ 配置驗證失敗:")
            print(f"{e}\n")
            failed += 1
            
        except Exception as e:
            print(f"   ❌ 未預期的錯誤:")
            print(f"      {type(e).__name__}: {e}\n")
            failed += 1
    
    # 打印摘要
    print(f"{'=' * 80}")
    print(f"驗證完成:")
    print(f"  ✅ 通過: {passed}/{total}")
    print(f"  ❌ 失敗: {failed}/{total}")
    print(f"{'=' * 80}")
    
    # 返回狀態碼
    sys.exit(0 if failed == 0 else 1)


if __name__ == '__main__':
    main()
