"""
端到端性能對比：Baseline vs Vectorized
運行相同的訓練配置，比較總時間和每 epoch 時間
"""

import subprocess
import time
import sys
import shutil
from pathlib import Path

# 配置
CONFIG_FILE = "configs/vectorized_test_100ep.yml"
BASELINE_RESIDUAL = "pinnx/losses/residuals_baseline.py"
VECTORIZED_RESIDUAL = "pinnx/losses/residuals.py"
ACTIVE_RESIDUAL = "pinnx/losses/residuals.py"
EPOCHS = 100
N_RUNS = 1  # 每個版本運行1次（100 epochs 已經很長）

def backup_current():
    """備份當前版本"""
    shutil.copy(ACTIVE_RESIDUAL, ACTIVE_RESIDUAL + ".tmp")
    print("✅ Backed up current residuals.py")

def restore_backup():
    """恢復備份"""
    shutil.copy(ACTIVE_RESIDUAL + ".tmp", ACTIVE_RESIDUAL)
    Path(ACTIVE_RESIDUAL + ".tmp").unlink()
    print("✅ Restored original residuals.py")

def switch_to_baseline():
    """切換到 baseline 版本"""
    shutil.copy(BASELINE_RESIDUAL, ACTIVE_RESIDUAL)
    print("🔄 Switched to BASELINE version")

def switch_to_vectorized():
    """切換到 vectorized 版本"""
    shutil.copy(ACTIVE_RESIDUAL + ".tmp", ACTIVE_RESIDUAL)
    print("🔄 Switched to VECTORIZED version")

def run_training(version_name, run_id):
    """運行一次訓練"""
    print(f"\n{'='*60}")
    print(f"Running {version_name} - Run {run_id}/{N_RUNS}")
    print(f"{'='*60}")
    
    cmd = ["python3", "scripts/train/train.py", "--cfg", CONFIG_FILE]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    end_time = time.time()
    
    elapsed = end_time - start_time
    
    # 解析輸出獲取訓練時間
    training_time = None
    for line in result.stdout.split('\n'):
        if 'Total training time:' in line:
            try:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'seconds':
                        training_time = float(parts[i-1])
                        break
            except:
                pass
    
    return {
        'version': version_name,
        'run_id': run_id,
        'wall_time': elapsed,
        'training_time': training_time,
        'success': result.returncode in [0, 1],  # 允許 wandb error
    }

def main():
    print("\n" + "="*70)
    print("🔬 端到端性能對比：Baseline vs Vectorized")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Config: {CONFIG_FILE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Runs per version: {N_RUNS}")
    
    # 備份當前版本
    backup_current()
    
    results = []
    
    try:
        # === Baseline 測試 ===
        print("\n" + "="*70)
        print("📊 BASELINE VERSION (Original Code)")
        print("="*70)
        switch_to_baseline()
        
        for run_id in range(1, N_RUNS + 1):
            result = run_training("Baseline", run_id)
            results.append(result)
            
            if result['success']:
                print(f"✅ Run {run_id} completed")
                if result['training_time']:
                    print(f"   Training time: {result['training_time']:.2f}s")
                print(f"   Wall time: {result['wall_time']:.2f}s")
            else:
                print(f"❌ Run {run_id} failed")
        
        # === Vectorized 測試 ===
        print("\n" + "="*70)
        print("🚀 VECTORIZED VERSION (Optimized Code)")
        print("="*70)
        switch_to_vectorized()
        
        for run_id in range(1, N_RUNS + 1):
            result = run_training("Vectorized", run_id)
            results.append(result)
            
            if result['success']:
                print(f"✅ Run {run_id} completed")
                if result['training_time']:
                    print(f"   Training time: {result['training_time']:.2f}s")
                print(f"   Wall time: {result['wall_time']:.2f}s")
            else:
                print(f"❌ Run {run_id} failed")
        
    finally:
        # 恢復原始版本
        restore_backup()
    
    # === 結果分析 ===
    print("\n" + "="*70)
    print("📈 RESULTS SUMMARY")
    print("="*70)
    
    baseline_times = [r['training_time'] for r in results if r['version'] == 'Baseline' and r['training_time']]
    vectorized_times = [r['training_time'] for r in results if r['version'] == 'Vectorized' and r['training_time']]
    
    if baseline_times and vectorized_times:
        import numpy as np
        
        baseline_mean = np.mean(baseline_times)
        vectorized_mean = np.mean(vectorized_times)
        
        speedup = baseline_mean / vectorized_mean
        time_saved = baseline_mean - vectorized_mean
        percent_saved = (time_saved / baseline_mean) * 100
        
        print(f"\nBaseline Version:")
        print(f"  Time: {baseline_mean:.2f} s")
        print(f"  Per epoch: {baseline_mean/EPOCHS:.3f} s")
        
        print(f"\nVectorized Version:")
        print(f"  Time: {vectorized_mean:.2f} s")
        print(f"  Per epoch: {vectorized_mean/EPOCHS:.3f} s")
        
        print(f"\n{'='*70}")
        print(f"🎯 PERFORMANCE GAIN")
        print(f"{'='*70}")
        print(f"Speedup:      {speedup:.2f}x")
        print(f"Time saved:   {time_saved:.2f} s ({percent_saved:.1f}%)")
        print(f"Per epoch:    {time_saved/EPOCHS:.3f} s faster")
        print(f"{'='*70}")
    else:
        print("❌ Not enough successful runs")

if __name__ == "__main__":
    main()
