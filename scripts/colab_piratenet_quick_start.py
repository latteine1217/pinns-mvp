#!/usr/bin/env python3
"""
PirateNet Colab 快速啟動腳本 - 修正版

用途：
  - 自動檢查環境配置
  - 驗證資料檔案
  - 執行修正版訓練
  - 即時監控訓練狀態

使用方式：
  python scripts/colab_piratenet_quick_start.py --config configs/colab_piratenet_2d_slice_fixed_v2.yml
"""

import subprocess
import sys
import os
from pathlib import Path
import yaml
import time

def print_section(title):
    """打印分隔線"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def check_environment():
    """檢查環境配置"""
    print_section("🔍 環境檢查")
    
    # 1. 檢查 Python 版本
    python_version = sys.version_info
    print(f"✅ Python 版本: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 2. 檢查 PyTorch
    try:
        import torch
        print(f"✅ PyTorch 版本: {torch.__version__}")
        print(f"   CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA 版本: {torch.version.cuda}")
            print(f"   GPU 設備: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch 未安裝")
        return False
    
    # 3. 檢查必要套件
    required_packages = ['numpy', 'scipy', 'matplotlib', 'yaml', 'h5py']
    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} 已安裝")
        except ImportError:
            print(f"❌ {pkg} 未安裝")
            return False
    
    return True

def check_data_files(config):
    """檢查資料檔案"""
    print_section("📂 資料檔案檢查")
    
    # 檢查資料目錄
    data_dir = Path(config.get('data', {}).get('data_dir', './data/jhtdb'))
    
    if not data_dir.exists():
        print(f"⚠️  資料目錄不存在: {data_dir}")
        print("   建議執行: python scripts/fetch_channel_flow.py --K 50 --slice-2d")
        return False
    
    print(f"✅ 資料目錄存在: {data_dir}")
    
    # 檢查感測點檔案
    sensor_file = data_dir / "sensors_K50.npz"
    if sensor_file.exists():
        print(f"✅ 感測點檔案存在: {sensor_file}")
    else:
        print(f"⚠️  感測點檔案不存在: {sensor_file}")
        return False
    
    # 檢查 JHTDB 資料檔案
    jhtdb_files = list(data_dir.glob("*.h5")) + list(data_dir.glob("*.npz"))
    if jhtdb_files:
        print(f"✅ 找到 {len(jhtdb_files)} 個 JHTDB 資料檔案")
    else:
        print(f"⚠️  未找到 JHTDB 資料檔案")
        return False
    
    return True

def validate_config(config_path):
    """驗證配置檔案"""
    print_section("📋 配置檔案驗證")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    issues = []
    
    # 1. 檢查學習率
    lr = config.get('training', {}).get('optimizer', {}).get('lr', None)
    if lr and lr > 1e-3:
        issues.append(f"⚠️  學習率過高: {lr}（建議 ≤ 1e-4）")
    else:
        print(f"✅ 學習率: {lr}")
    
    # 2. 檢查梯度裁剪
    grad_clip = config.get('training', {}).get('gradient_clip', None)
    if grad_clip and grad_clip <= 1.0:
        print(f"✅ 梯度裁剪: {grad_clip}")
    else:
        issues.append(f"⚠️  梯度裁剪未設定或過大: {grad_clip}")
    
    # 3. 檢查壁面位置
    y_min = config.get('physics', {}).get('domain', {}).get('y_min', None)
    y_max = config.get('physics', {}).get('domain', {}).get('y_max', None)
    if y_min == -1.0 and y_max == 1.0:
        print(f"✅ 壁面位置: y ∈ [{y_min}, {y_max}]")
    else:
        issues.append(f"⚠️  壁面位置不對稱: y ∈ [{y_min}, {y_max}]")
    
    # 4. 檢查批次大小
    batch_size = config.get('training', {}).get('batch_size', None)
    if batch_size and batch_size <= 1024:
        print(f"✅ Batch size: {batch_size}")
    else:
        issues.append(f"⚠️  Batch size 過大: {batch_size}（建議 ≤ 1024）")
    
    # 5. 檢查歸一化
    norm_enabled = config.get('data', {}).get('normalization', {}).get('enabled', False)
    if norm_enabled:
        print(f"✅ 資料歸一化: 已啟用")
    else:
        issues.append(f"⚠️  資料歸一化未啟用")
    
    if issues:
        print(f"\n發現 {len(issues)} 個問題:")
        for issue in issues:
            print(f"  {issue}")
        return False, config
    else:
        print("\n✅ 配置檔案驗證通過")
        return True, config

def monitor_training(log_file, check_interval=30, max_checks=20):
    """監控訓練進度"""
    print_section("📊 訓練監控")
    
    print(f"監控日誌: {log_file}")
    print(f"檢查間隔: {check_interval} 秒")
    print(f"最多檢查: {max_checks} 次\n")
    
    for i in range(max_checks):
        if not Path(log_file).exists():
            print(f"[{i+1}/{max_checks}] 等待日誌檔案生成...")
            time.sleep(check_interval)
            continue
        
        # 讀取最後幾行
        try:
            result = subprocess.run(
                ['tail', '-20', log_file],
                capture_output=True,
                text=True
            )
            
            lines = result.stdout.strip().split('\n')
            
            # 提取關鍵資訊
            print(f"\n[檢查 {i+1}/{max_checks}] 訓練狀態:")
            print("-" * 80)
            
            for line in lines[-10:]:  # 只顯示最後 10 行
                # 檢查 NaN
                if 'nan' in line.lower() or 'inf' in line.lower():
                    print(f"❌ {line}")
                elif 'epoch' in line.lower():
                    print(f"📈 {line}")
                elif 'loss' in line.lower():
                    print(f"   {line}")
            
            print("-" * 80)
            
        except Exception as e:
            print(f"⚠️  讀取日誌失敗: {e}")
        
        time.sleep(check_interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='PirateNet Colab 快速啟動腳本')
    parser.add_argument('--config', type=str, required=True,
                        help='配置檔案路徑')
    parser.add_argument('--skip-checks', action='store_true',
                        help='跳過環境檢查')
    parser.add_argument('--monitor', action='store_true',
                        help='訓練後監控進度')
    parser.add_argument('--monitor-interval', type=int, default=30,
                        help='監控間隔（秒）')
    args = parser.parse_args()
    
    print("=" * 80)
    print("  🚀 PirateNet Colab 快速啟動工具")
    print("=" * 80)
    print(f"\n配置檔案: {args.config}")
    
    # 1. 環境檢查
    if not args.skip_checks:
        if not check_environment():
            print("\n❌ 環境檢查失敗，請安裝缺失的套件")
            return 1
    
    # 2. 配置驗證
    config_valid, config = validate_config(args.config)
    if not config_valid:
        print("\n⚠️  配置檔案有問題，建議修正後再訓練")
        response = input("是否繼續訓練？(y/n): ")
        if response.lower() != 'y':
            return 1
    
    # 3. 資料檢查
    if not args.skip_checks:
        if not check_data_files(config):
            print("\n❌ 資料檔案檢查失敗")
            print("   請執行: python scripts/fetch_channel_flow.py --K 50 --slice-2d")
            return 1
    
    # 4. 開始訓練
    print_section("🏋️  開始訓練")
    
    train_cmd = [
        'python', 'scripts/train.py',
        '--cfg', args.config
    ]
    
    print(f"執行指令: {' '.join(train_cmd)}\n")
    
    try:
        # 執行訓練（前台）
        process = subprocess.Popen(
            train_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # 即時輸出
        for line in process.stdout:
            print(line, end='')
            
            # 檢查 NaN
            if 'nan' in line.lower():
                print("\n❌ 檢測到 NaN，訓練可能失敗！")
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ 訓練完成")
        else:
            print(f"\n❌ 訓練失敗，返回碼: {process.returncode}")
            return process.returncode
            
    except KeyboardInterrupt:
        print("\n⚠️  訓練被中斷")
        process.terminate()
        return 1
    except Exception as e:
        print(f"\n❌ 訓練過程出錯: {e}")
        return 1
    
    # 5. 監控訓練（可選）
    if args.monitor:
        exp_name = config.get('experiment', {}).get('name', 'unknown')
        log_file = f"log/{exp_name}/training.log"
        monitor_training(log_file, check_interval=args.monitor_interval)
    
    print_section("✅ 完成")
    print("下一步:")
    print("  1. 檢查訓練日誌: tail -50 log/<exp_name>/training.log")
    print("  2. 評估模型: python scripts/evaluate.py --checkpoint <path>")
    print("  3. 視覺化結果: python scripts/visualize_results.py --checkpoint <path>")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
