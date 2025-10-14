#!/usr/bin/env python3
"""
通用訓練監控腳本

用法:
    # 監控所有活躍訓練
    python scripts/monitor_training.py --all
    
    # 監控特定實驗
    python scripts/monitor_training.py --config test_rans_phase6c_v1
    
    # 持續監控模式（每 5 秒刷新）
    python scripts/monitor_training.py --all --watch
    
    # 詳細模式（顯示所有指標）
    python scripts/monitor_training.py --config phase6c_v1 --verbose
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.utils.training_monitor import TrainingMonitor


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(
        description="通用訓練監控工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 監控所有活躍訓練
  python scripts/monitor_training.py --all
  
  # 監控特定實驗
  python scripts/monitor_training.py --config test_rans_phase6c_v1
  
  # 持續監控模式
  python scripts/monitor_training.py --all --watch --interval 10
  
  # 詳細模式
  python scripts/monitor_training.py --config phase6c_v1 --verbose
        """
    )
    
    # 主要參數
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--all",
        action="store_true",
        help="監控所有活躍訓練"
    )
    group.add_argument(
        "--config",
        type=str,
        help="指定配置名稱（不含 .yml 副檔名）"
    )
    
    # 可選參數
    parser.add_argument(
        "--watch",
        action="store_true",
        help="持續監控模式（自動刷新）"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="刷新間隔（秒），預設 5 秒"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="詳細模式（顯示所有指標）"
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="configs/monitoring.yml",
        help="監控配置文件路徑"
    )
    
    return parser.parse_args()


def display_status(monitor: TrainingMonitor, config_name: Optional[str] = None, verbose: bool = False):
    """
    顯示訓練狀態
    
    Args:
        monitor: 訓練監控器
        config_name: 配置名稱（None 表示顯示所有）
        verbose: 是否顯示詳細資訊
    """
    if config_name:
        # 單一訓練
        status = monitor.get_training_status(config_name)
        if status:
            report = monitor.format_status_report(status)
            print(report)
            
            if verbose and status.current_metrics:
                print("\n" + "=" * 80)
                print("📊 All Metrics:")
                print("=" * 80)
                for key, value in sorted(status.current_metrics.losses.items()):
                    print(f"   {key:30s}: {value:12.6f}")
        else:
            print(f"⚠️  找不到訓練: {config_name}")
    else:
        # 所有活躍訓練
        all_status = monitor.get_all_training_status()
        
        if not all_status:
            print("⚠️  沒有檢測到活躍訓練")
            return
        
        for i, status in enumerate(all_status):
            if i > 0:
                print("\n")
            
            report = monitor.format_status_report(status)
            print(report)
            
            if verbose and status.current_metrics:
                print("\n📊 All Metrics:")
                for key, value in sorted(status.current_metrics.losses.items()):
                    print(f"   {key:30s}: {value:12.6f}")


def watch_mode(monitor: TrainingMonitor, config_name: Optional[str] = None, 
               interval: int = 5, verbose: bool = False):
    """
    持續監控模式
    
    Args:
        monitor: 訓練監控器
        config_name: 配置名稱（None 表示監控所有）
        interval: 刷新間隔（秒）
        verbose: 是否顯示詳細資訊
    """
    print(f"🔄 持續監控模式啟動（每 {interval} 秒刷新）")
    print("按 Ctrl+C 停止監控\n")
    
    try:
        while True:
            # 清除螢幕（macOS/Linux）
            print("\033[2J\033[H", end="")
            
            # 顯示時間戳
            from datetime import datetime
            print(f"⏰ 最後更新: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("")
            
            # 顯示狀態
            display_status(monitor, config_name, verbose)
            
            # 等待下次刷新
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\n✅ 監控已停止")


def main():
    """主函數"""
    args = parse_args()
    
    # 初始化監控器
    try:
        monitor = TrainingMonitor(config_path=args.config_file)
    except FileNotFoundError as e:
        print(f"❌ 錯誤: {e}")
        print(f"💡 提示: 請確保配置文件存在於 {args.config_file}")
        return 1
    except Exception as e:
        print(f"❌ 初始化監控器時發生錯誤: {e}")
        return 1
    
    # 確定要監控的配置
    config_name = args.config if not args.all else None
    
    # 執行監控
    try:
        if args.watch:
            watch_mode(monitor, config_name, args.interval, args.verbose)
        else:
            display_status(monitor, config_name, args.verbose)
    except Exception as e:
        print(f"❌ 監控時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
