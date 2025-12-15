#!/usr/bin/env python3
"""
Wave 3 torch.compile() 基準測試腳本

功能：
1. 執行 baseline (compile.enabled=false) 訓練
2. 執行 compiled (compile.enabled=true) 訓練
3. 比較訓練時間，計算加速比
4. 將結果儲存為 JSON 格式

使用方式：
    python scripts/tools/run_wave3_benchmark.py \
        --baseline configs/perf_test_wave3_baseline.yml \
        --compiled configs/perf_test_wave3_compiled.yml \
        --output tasks/perf-wave3-20251215/wave3_mps_results.json

作者：Wave 3 Optimization Team
日期：2025-12-15
"""

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


def run_training(config_path: str, timeout: int = 600) -> Dict:
    """
    執行單次訓練並記錄時間
    
    Args:
        config_path: 訓練配置檔路徑
        timeout: 超時時限（秒），預設 600 秒（10 分鐘）
    
    Returns:
        結果字典 {
            "success": bool,
            "total_time": float (秒),
            "epochs": int,
            "config": str,
            "error": str (若失敗)
        }
    """
    result = {
        "success": False,
        "total_time": 0.0,
        "epochs": 0,
        "config": config_path,
        "error": None
    }
    
    print(f"\n{'='*60}")
    print(f"執行訓練: {config_path}")
    print(f"{'='*60}")
    
    try:
        # 建立訓練命令
        cmd = [
            "python",
            "scripts/train/train.py",
            "--cfg", config_path
        ]
        
        print(f"命令: {' '.join(cmd)}")
        print(f"超時時限: {timeout} 秒")
        print(f"開始時間: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 計時並執行訓練
        start_time = time.time()
        
        # 設定環境變數，確保 pinnx 可被導入
        project_root = Path(__file__).parent.parent.parent
        env = os.environ.copy()
        env['PYTHONPATH'] = str(project_root) + os.pathsep + env.get('PYTHONPATH', '')
        
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=project_root,  # 專案根目錄
            env=env  # 傳遞修改後的環境變數
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 檢查執行結果
        if process.returncode == 0:
            result["success"] = True
            result["total_time"] = elapsed_time
            
            # 從 stdout 提取 epoch 數量（假設日誌包含 "Epoch X/Y"）
            # 這是簡化版，實際可能需要更複雜的解析
            try:
                # 尋找最後一個 epoch 記錄
                for line in process.stdout.split('\n'):
                    if 'Epoch' in line and '/' in line:
                        # 假設格式：Epoch 50/50 ...
                        epoch_info = line.split()[1]  # "50/50"
                        result["epochs"] = int(epoch_info.split('/')[0])
            except Exception as e:
                print(f"⚠️  無法解析 epoch 數量: {e}")
                result["epochs"] = -1
            
            print(f"\n✅ 訓練成功")
            print(f"總時間: {elapsed_time:.2f} 秒")
            print(f"完成 Epochs: {result['epochs']}")
            
        else:
            result["error"] = f"訓練失敗 (return code: {process.returncode})"
            print(f"\n❌ 訓練失敗")
            print(f"Return code: {process.returncode}")
            print(f"\nStderr:\n{process.stderr}")
            
    except subprocess.TimeoutExpired:
        result["error"] = f"訓練超時 (>{timeout} 秒)"
        print(f"\n⏱️  訓練超時 (>{timeout} 秒)")
        
    except Exception as e:
        result["error"] = str(e)
        print(f"\n❌ 執行錯誤: {e}")
    
    return result


def calculate_speedup(baseline: Dict, compiled: Dict) -> Dict:
    """
    計算加速比與相關指標
    
    Args:
        baseline: Baseline 訓練結果
        compiled: Compiled 訓練結果
    
    Returns:
        統計結果字典
    """
    stats = {
        "baseline_time": baseline["total_time"],
        "compiled_time": compiled["total_time"],
        "speedup_ratio": 0.0,
        "speedup_percent": 0.0,
        "time_saved": 0.0,
        "both_successful": baseline["success"] and compiled["success"]
    }
    
    if stats["both_successful"] and baseline["total_time"] > 0:
        stats["speedup_ratio"] = baseline["total_time"] / compiled["total_time"]
        stats["speedup_percent"] = (baseline["total_time"] - compiled["total_time"]) / baseline["total_time"] * 100
        stats["time_saved"] = baseline["total_time"] - compiled["total_time"]
    
    return stats


def save_results(baseline: Dict, compiled: Dict, stats: Dict, output_path: str):
    """
    儲存基準測試結果到 JSON 檔案
    
    Args:
        baseline: Baseline 結果
        compiled: Compiled 結果
        stats: 統計數據
        output_path: 輸出檔案路徑
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "baseline": baseline,
        "compiled": compiled,
        "statistics": stats,
        "conclusion": generate_conclusion(stats)
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 結果已儲存至: {output_file}")


def generate_conclusion(stats: Dict) -> str:
    """
    根據統計數據生成結論建議
    
    Args:
        stats: 統計數據
    
    Returns:
        結論字串
    """
    if not stats["both_successful"]:
        return "❌ 測試未完全成功，無法得出結論"
    
    speedup = stats["speedup_percent"]
    
    if speedup >= 3.0:
        return f"✅ 加速顯著 ({speedup:.2f}%)，建議進入 Phase 3-2 測試更激進的模式"
    elif speedup >= 1.5:
        return f"⚠️  加速中等 ({speedup:.2f}%)，建議提交 Phase 3-1 並停止進一步優化"
    else:
        return f"❌ 加速不足 ({speedup:.2f}%)，建議回滾 Wave 3 改動"


def print_summary(baseline: Dict, compiled: Dict, stats: Dict):
    """
    在終端輸出結果摘要
    
    Args:
        baseline: Baseline 結果
        compiled: Compiled 結果
        stats: 統計數據
    """
    print(f"\n{'='*60}")
    print(f"Wave 3 基準測試結果摘要")
    print(f"{'='*60}")
    
    print(f"\n📊 訓練時間比較:")
    print(f"  Baseline (無編譯):  {baseline['total_time']:.2f} 秒")
    print(f"  Compiled (編譯):    {compiled['total_time']:.2f} 秒")
    print(f"  節省時間:           {stats['time_saved']:.2f} 秒")
    
    print(f"\n⚡ 加速比:")
    print(f"  Speedup Ratio:    {stats['speedup_ratio']:.4f}x")
    print(f"  Speedup Percent:  {stats['speedup_percent']:.2f}%")
    
    print(f"\n💡 結論:")
    print(f"  {stats.get('conclusion', generate_conclusion(stats))}")
    
    print(f"\n{'='*60}")


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(
        description='Wave 3 torch.compile() 基準測試',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法：
  # MPS 測試
  python scripts/tools/run_wave3_benchmark.py \\
      --baseline configs/perf_test_wave3_baseline.yml \\
      --compiled configs/perf_test_wave3_compiled.yml \\
      --output tasks/perf-wave3-20251215/wave3_mps_results.json
  
  # CUDA 測試（記得修改配置檔中的 device）
  python scripts/tools/run_wave3_benchmark.py \\
      --baseline configs/perf_test_wave3_baseline_cuda.yml \\
      --compiled configs/perf_test_wave3_compiled_cuda.yml \\
      --output tasks/perf-wave3-20251215/wave3_cuda_results.json
        """
    )
    
    parser.add_argument(
        '--baseline',
        type=str,
        required=True,
        help='Baseline 配置檔路徑（compile.enabled=false）'
    )
    
    parser.add_argument(
        '--compiled',
        type=str,
        required=True,
        help='Compiled 配置檔路徑（compile.enabled=true）'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='輸出 JSON 檔案路徑'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=600,
        help='每次訓練超時時限（秒），預設 600 秒'
    )
    
    args = parser.parse_args()
    
    # 驗證輸入檔案存在
    if not Path(args.baseline).exists():
        print(f"❌ Baseline 配置檔不存在: {args.baseline}")
        return 1
    
    if not Path(args.compiled).exists():
        print(f"❌ Compiled 配置檔不存在: {args.compiled}")
        return 1
    
    print(f"\n🚀 開始 Wave 3 基準測試")
    print(f"Baseline 配置: {args.baseline}")
    print(f"Compiled 配置: {args.compiled}")
    print(f"輸出路徑: {args.output}")
    
    # 執行 baseline 訓練
    baseline_results = run_training(args.baseline, timeout=args.timeout)
    
    # 執行 compiled 訓練
    compiled_results = run_training(args.compiled, timeout=args.timeout)
    
    # 計算統計數據
    stats = calculate_speedup(baseline_results, compiled_results)
    
    # 儲存結果
    save_results(baseline_results, compiled_results, stats, args.output)
    
    # 輸出摘要
    print_summary(baseline_results, compiled_results, stats)
    
    # 根據結果決定退出碼
    if stats["both_successful"]:
        if stats["speedup_percent"] >= 1.5:
            return 0  # 成功
        else:
            return 2  # 加速不足
    else:
        return 1  # 執行失敗


if __name__ == "__main__":
    exit(main())
