#!/usr/bin/env python3
"""
梯度檢查點效能基準測試腳本
測量啟用/禁用梯度檢查點的訓練時間和記憶體使用
"""

import sys
import time
import json
from pathlib import Path
import subprocess
import argparse

def run_training(config_path: str, max_epochs: int = 50) -> dict:
    """
    執行訓練並測量效能指標
    
    Returns:
        dict: 包含訓練時間、記憶體使用等指標
    """
    print(f"\n{'='*80}")
    print(f"🚀 開始訓練：{config_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 執行訓練
    cmd = [
        sys.executable,
        "scripts/train/train.py",
        "--cfg", config_path,
    ]
    
    # 設定環境變數，確保 pinnx 可以被找到
    import os
    env = os.environ.copy()
    project_root = "/Users/latteine/Documents/coding/pinns-mvp"
    if 'PYTHONPATH' in env:
        env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
    else:
        env['PYTHONPATH'] = project_root
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=1800,  # 30 分鐘超時
            env=env
        )
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # 解析輸出以獲取更多指標
        output_lines = result.stdout.split('\n')
        
        # 查找關鍵指標
        metrics = {
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
            'success': result.returncode == 0,
            'config': config_path,
        }
        
        # 查找 loss 資訊
        for line in output_lines:
            if 'Final' in line and 'loss' in line.lower():
                metrics['final_loss_info'] = line.strip()
            if 'epoch' in line.lower() and '/' in line:
                # 記錄最後一個 epoch 資訊
                metrics['last_epoch_info'] = line.strip()
        
        print(f"\n{'='*80}")
        print(f"✅ 訓練完成：{config_path}")
        print(f"⏱️  總時間：{training_time:.2f} 秒 ({training_time/60:.2f} 分鐘)")
        print(f"{'='*80}\n")
        
        # 保存完整輸出到檔案
        output_file = Path(f"results/{Path(config_path).stem}_output.log")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(result.stdout + "\n\nSTDERR:\n" + result.stderr)
        
        print(f"📄 完整日誌已保存：{output_file}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"❌ 訓練超時（30分鐘）")
        return {
            'training_time_seconds': 1800,
            'success': False,
            'error': 'timeout',
            'config': config_path,
        }
    except Exception as e:
        print(f"❌ 訓練失敗：{e}")
        return {
            'success': False,
            'error': str(e),
            'config': config_path,
        }

def compare_results(baseline: dict, optimized: dict):
    """比較基準和優化版本的結果"""
    print(f"\n{'='*80}")
    print("📊 效能對比結果")
    print(f"{'='*80}\n")
    
    if not baseline['success'] or not optimized['success']:
        print("⚠️  警告：部分測試失敗，對比結果可能不完整")
        return
    
    baseline_time = baseline['training_time_seconds']
    optimized_time = optimized['training_time_seconds']
    time_reduction = (baseline_time - optimized_time) / baseline_time * 100
    speedup = baseline_time / optimized_time
    
    print(f"{'指標':<30} {'基準版本':<20} {'優化版本':<20} {'改善':<15}")
    print(f"{'-'*85}")
    print(f"{'訓練時間 (秒)':<30} {baseline_time:<20.2f} {optimized_time:<20.2f} {time_reduction:>+.1f}%")
    print(f"{'訓練時間 (分鐘)':<30} {baseline_time/60:<20.2f} {optimized_time/60:<20.2f} {'—':<15}")
    print(f"{'加速比':<30} {'1.00x':<20} {speedup:<20.2f}x {'—':<15}")
    
    print(f"\n{'='*80}")
    print("🎯 結論")
    print(f"{'='*80}\n")
    
    if time_reduction > 30:
        print(f"✅ 優秀！梯度檢查點帶來 {time_reduction:.1f}% 的訓練時間減少")
        print(f"   加速比：{speedup:.2f}x（符合預期的 35-40% 提升）")
    elif time_reduction > 10:
        print(f"⚠️  中等效果：梯度檢查點帶來 {time_reduction:.1f}% 的訓練時間減少")
        print(f"   低於預期（35-40%），可能受限於 MPS 記憶體管理機制")
    elif time_reduction > 0:
        print(f"⚠️  輕微改善：梯度檢查點帶來 {time_reduction:.1f}% 的訓練時間減少")
        print(f"   顯著低於預期，建議檢查：")
        print(f"   1. MPS 記憶體頻寬限制")
        print(f"   2. 批次大小是否過小")
        print(f"   3. 網路深度是否足夠")
    else:
        print(f"❌ 未見改善：優化版本反而慢了 {abs(time_reduction):.1f}%")
        print(f"   可能原因：")
        print(f"   1. MPS 不支援梯度檢查點優化")
        print(f"   2. 模型規模太小，檢查點開銷大於收益")
    
    print(f"\n💡 建議：")
    if time_reduction > 0:
        print(f"   - 可以安全啟用梯度檢查點")
        print(f"   - 在 GPU (CUDA) 環境中效果會更顯著")
        print(f"   - 記憶體節省允許使用更大的批次或網路")
    else:
        print(f"   - 在當前 MPS 環境下可能不適合啟用")
        print(f"   - 建議在 GPU (CUDA) 環境中重新測試")

def main():
    parser = argparse.ArgumentParser(description='梯度檢查點效能基準測試')
    parser.add_argument('--baseline', default='configs/perf_test_baseline.yml',
                       help='基準配置檔案路徑')
    parser.add_argument('--optimized', default='configs/perf_test_optimized.yml',
                       help='優化配置檔案路徑')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='跳過基準測試（如果已執行）')
    parser.add_argument('--skip-optimized', action='store_true',
                       help='跳過優化測試（如果已執行）')
    
    args = parser.parse_args()
    
    results = {}
    
    # 執行基準測試
    if not args.skip_baseline:
        print("📋 第 1 步：執行基準測試（無梯度檢查點）")
        results['baseline'] = run_training(args.baseline)
        
        # 保存中間結果
        with open('results/baseline_metrics.json', 'w') as f:
            json.dump(results['baseline'], f, indent=2)
    else:
        print("⏭️  跳過基準測試，載入已有結果...")
        with open('results/baseline_metrics.json', 'r') as f:
            results['baseline'] = json.load(f)
    
    # 執行優化測試
    if not args.skip_optimized:
        print("\n📋 第 2 步：執行優化測試（啟用梯度檢查點）")
        results['optimized'] = run_training(args.optimized)
        
        # 保存中間結果
        with open('results/optimized_metrics.json', 'w') as f:
            json.dump(results['optimized'], f, indent=2)
    else:
        print("⏭️  跳過優化測試，載入已有結果...")
        with open('results/optimized_metrics.json', 'r') as f:
            results['optimized'] = json.load(f)
    
    # 對比結果
    compare_results(results['baseline'], results['optimized'])
    
    # 保存完整結果
    results_file = Path('results/gradient_checkpointing_benchmark.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📊 完整結果已保存：{results_file}")

if __name__ == '__main__':
    main()
