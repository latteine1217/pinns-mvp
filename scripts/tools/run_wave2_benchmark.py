#!/usr/bin/env python3
"""
Wave 2 Phase 2-4: 完整訓練效能測試

比較 Wave 1-2 (tensor pre-concat) vs Wave 2 (gradient cache)
"""

import sys
import os
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, Any

# 添加專案根目錄到 Python 路徑
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def run_training(config_path: str, name: str) -> Dict[str, Any]:
    """
    運行訓練並測量時間
    
    Returns:
        Dict包含：total_time, epochs, time_per_epoch, final_loss
    """
    print(f"\n{'='*80}")
    print(f"🚀 開始訓練: {name}")
    print(f"   配置: {config_path}")
    print(f"{'='*80}\n")
    
    start_time = time.time()
    
    # 運行訓練
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/train/train.py"),
        "--cfg", config_path
    ]
    
    # 設置環境變數（確保 PYTHONPATH 包含專案根目錄）
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT)
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,  # 10 分鐘超時
            env=env
        )
        
        end_time = time.time()
        total_time = end_time - start_time
        
        if result.returncode != 0:
            print(f"❌ 訓練失敗！")
            print(f"STDERR:\n{result.stderr}")
            return {
                'success': False,
                'error': result.stderr,
                'total_time': total_time
            }
        
        # 解析輸出獲取最終損失
        output_lines = result.stdout.split('\n')
        final_loss = None
        epochs = 50  # 預設
        
        for line in output_lines:
            if 'Epoch' in line and 'total_loss' in line:
                # 嘗試解析最後一個 epoch 的 loss
                try:
                    parts = line.split('total_loss')
                    if len(parts) > 1:
                        loss_str = parts[1].split()[0].strip(':=')
                        final_loss = float(loss_str)
                except:
                    pass
        
        time_per_epoch = total_time / epochs
        
        result_dict = {
            'success': True,
            'total_time': total_time,
            'epochs': epochs,
            'time_per_epoch': time_per_epoch,
            'final_loss': final_loss
        }
        
        print(f"\n✅ 訓練完成！")
        print(f"   總時間: {total_time:.2f} 秒")
        print(f"   每 epoch 時間: {time_per_epoch:.2f} 秒")
        if final_loss:
            print(f"   最終 loss: {final_loss:.6f}")
        
        return result_dict
        
    except subprocess.TimeoutExpired:
        end_time = time.time()
        print(f"❌ 訓練超時（> 10 分鐘）")
        return {
            'success': False,
            'error': 'Timeout after 10 minutes',
            'total_time': end_time - start_time
        }
    except Exception as e:
        end_time = time.time()
        print(f"❌ 發生錯誤: {e}")
        return {
            'success': False,
            'error': str(e),
            'total_time': end_time - start_time
        }


def main():
    """主函數"""
    print("\n" + "="*80)
    print("🎯 Wave 2 Phase 2-4: 完整訓練效能測試")
    print("="*80)
    
    # 配置文件路徑
    wave1_2_config = str(PROJECT_ROOT / "configs/perf_test_wave1_2_fair.yml")
    wave2_config = str(PROJECT_ROOT / "configs/perf_test_wave2.yml")
    
    # 確認配置文件存在
    if not Path(wave1_2_config).exists():
        print(f"❌ Wave 1-2 配置文件不存在: {wave1_2_config}")
        return 1
    
    if not Path(wave2_config).exists():
        print(f"❌ Wave 2 配置文件不存在: {wave2_config}")
        return 1
    
    print(f"\n📋 測試配置:")
    print(f"   Wave 1-2 (Baseline): {wave1_2_config}")
    print(f"   Wave 2 (Optimized):  {wave2_config}")
    print(f"   Epochs: 50")
    print(f"   PDE Points: 4096")
    print(f"   BC Points: 1024")
    print(f"   Device: MPS (Apple Silicon)")
    
    # 運行 Wave 1-2 (baseline)
    wave1_2_result = run_training(wave1_2_config, "Wave 1-2 (Tensor Pre-concat)")
    
    if not wave1_2_result['success']:
        print("\n❌ Wave 1-2 訓練失敗，無法繼續比較")
        return 1
    
    # 運行 Wave 2 (gradient cache)
    wave2_result = run_training(wave2_config, "Wave 2 (Gradient Cache)")
    
    if not wave2_result['success']:
        print("\n❌ Wave 2 訓練失敗")
        # 仍然生成報告（顯示失敗）
    
    # 生成比較報告
    print("\n" + "="*80)
    print("📊 效能比較結果")
    print("="*80)
    
    print(f"\n{'指標':<30} {'Wave 1-2':<20} {'Wave 2':<20} {'改進':<15}")
    print("-" * 85)
    
    # 總訓練時間
    wave1_2_time = wave1_2_result['total_time']
    wave2_time = wave2_result.get('total_time', 0)
    
    # 初始化變數（避免在 Wave 2 失敗時未定義）
    time_speedup = 0.0
    time_improvement = 0.0
    loss_diff = 0.0
    
    if wave2_result['success']:
        time_speedup = wave1_2_time / wave2_time
        time_improvement = (1 - wave2_time / wave1_2_time) * 100
        
        print(f"{'總訓練時間 (秒)':<30} {wave1_2_time:>18.2f}   {wave2_time:>18.2f}   {time_improvement:>6.1f}%")
        
        # 每 epoch 時間
        wave1_2_per_epoch = wave1_2_result['time_per_epoch']
        wave2_per_epoch = wave2_result['time_per_epoch']
        
        print(f"{'每 Epoch 時間 (秒)':<30} {wave1_2_per_epoch:>18.2f}   {wave2_per_epoch:>18.2f}   {time_improvement:>6.1f}%")
        print(f"{'加速比':<30} {'':<20} {time_speedup:>18.2f}x")
        
        # 最終 loss (檢查數值一致性)
        if wave1_2_result['final_loss'] and wave2_result['final_loss']:
            loss_diff = abs(wave1_2_result['final_loss'] - wave2_result['final_loss'])
            loss_rel_diff = loss_diff / wave1_2_result['final_loss'] * 100
            
            print(f"\n{'最終 Loss':<30} {wave1_2_result['final_loss']:>18.6f}   {wave2_result['final_loss']:>18.6f}   差異: {loss_rel_diff:.2f}%")
            
            if loss_diff < 1e-3:
                print("✅ 數值精度一致（差異 < 1e-3）")
            elif loss_rel_diff < 1.0:
                print("✅ 數值精度可接受（相對差異 < 1%）")
            else:
                print("⚠️  數值差異較大，需要進一步檢查")
        
        # 判定結果
        print("\n" + "="*80)
        if time_improvement >= 10:
            print("🎉 Wave 2 優化成功！")
            print(f"   ✅ 效能提升: {time_improvement:.1f}% (加速 {time_speedup:.2f}x)")
            if wave1_2_result.get('final_loss') and wave2_result.get('final_loss') and loss_diff < 1e-3:
                print(f"   ✅ 數值精度保持一致")
            print("\n   Wave 2 (Gradient Cache) 整合成功，準備進入生產環境！")
        elif time_improvement >= 5:
            print("✅ Wave 2 有適度提升")
            print(f"   效能提升: {time_improvement:.1f}%")
            print("   建議：可以合併，但提升幅度低於預期（目標 15-20%）")
        else:
            print("⚠️  Wave 2 提升不明顯")
            print(f"   效能提升: {time_improvement:.1f}%")
            print("   需要進一步調查：")
            print("   - 檢查 gradient cache 是否正確啟用")
            print("   - 分析 profiling 數據")
            print("   - 確認 MPS 後端是否有特殊行為")
        
    else:
        print(f"{'總訓練時間 (秒)':<30} {wave1_2_time:>18.2f}   {'FAILED':<20}")
        print("\n❌ Wave 2 訓練失敗，無法比較效能")
    
    print("="*80)
    
    # 保存結果到 JSON
    results = {
        'wave1_2': wave1_2_result,
        'wave2': wave2_result,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    output_file = PROJECT_ROOT / "tasks/perf-analysis-20251215/wave2_phase2-4_benchmark_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 結果已保存至: {output_file}")
    
    return 0 if wave2_result['success'] and time_improvement >= 10 else 1


if __name__ == "__main__":
    sys.exit(main())
