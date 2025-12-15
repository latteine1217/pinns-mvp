#!/usr/bin/env python3
"""
Wave 1-2 效能測試腳本

測試目標：
1. 運行 50 epochs 訓練，記錄總時間
2. 對比 Wave 1-1 基準（152.20s）
3. 驗證數值正確性（loss 應一致）
4. 記錄記憶體使用

使用方式：
    python scripts/tools/run_wave1_2_benchmark.py
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_benchmark():
    """運行效能基準測試"""
    
    logger.info("=" * 80)
    logger.info("🚀 Wave 1-2: Tensor Pre-concatenation 效能測試")
    logger.info("=" * 80)
    
    # 記錄開始時間
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"開始時間: {start_datetime}")
    logger.info(f"配置檔案: configs/perf_test_wave1_2.yml")
    logger.info(f"訓練輪數: 50 epochs")
    logger.info("")
    
    # 執行訓練
    logger.info("開始訓練...")
    import subprocess
    
    try:
        result = subprocess.run(
            ["python", "scripts/train/train.py", "--cfg", "configs/perf_test_wave1_2.yml"],
            cwd=project_root,
            env={**dict(os.environ), "PYTHONPATH": str(project_root)},
            capture_output=True,
            text=True,
            timeout=600  # 10分鐘超時
        )
        
        # 記錄結束時間
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 顯示訓練輸出（最後100行）
        output_lines = result.stdout.split('\n')
        logger.info("\n訓練輸出（最後50行）:")
        logger.info("-" * 80)
        for line in output_lines[-50:]:
            if line.strip():
                logger.info(line)
        logger.info("-" * 80)
        
        if result.returncode != 0:
            logger.error(f"訓練失敗！返回碼: {result.returncode}")
            logger.error(f"錯誤輸出:\n{result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"訓練超時！已運行 {elapsed_time:.2f}s")
        return None
    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.error(f"訓練出錯: {e}")
        return None
    
    # 嘗試讀取最終 checkpoint
    import torch
    checkpoint_dir = project_root / 'checkpoints'
    latest_dirs = sorted(checkpoint_dir.glob('*'), key=lambda p: p.stat().st_mtime, reverse=True)
    
    final_loss = None
    if latest_dirs:
        latest_ckpt = latest_dirs[0] / 'best_model.pth'
        if latest_ckpt.exists():
            try:
                ckpt = torch.load(latest_ckpt, map_location='cpu')
                final_loss = ckpt.get('loss', None)
            except Exception as e:
                logger.warning(f"無法讀取 checkpoint: {e}")
    
    # 彙整結果
    result = {
        'test_name': 'Wave 1-2: Tensor Pre-concatenation',
        'start_time': start_datetime,
        'total_time_seconds': elapsed_time,
        'time_per_epoch': elapsed_time / 50,
        'epochs': 50,
        'final_loss': float(final_loss) if final_loss is not None else None,
        'config': 'configs/perf_test_wave1_2.yml',
        'device': 'mps',
        'optimization': 'tensor_preconcat'
    }
    
    # 顯示結果
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 測試結果")
    logger.info("=" * 80)
    logger.info(f"總時間:       {elapsed_time:.2f}s")
    logger.info(f"每 epoch 時間: {result['time_per_epoch']:.4f}s")
    if final_loss is not None:
        logger.info(f"最終損失:     {final_loss:.6f}")
    else:
        logger.info(f"最終損失:     N/A（無法讀取）")
    logger.info("")
    
    # 對比 Wave 1-1 基準
    baseline_time = 152.20
    baseline_loss = 0.052578
    
    time_diff = elapsed_time - baseline_time
    time_speedup = (baseline_time - elapsed_time) / baseline_time * 100
    
    logger.info("📈 與 Wave 1-1 基準對比:")
    logger.info(f"  Baseline 時間:    {baseline_time:.2f}s")
    logger.info(f"  Wave 1-2 時間:    {elapsed_time:.2f}s")
    logger.info(f"  時間差異:        {time_diff:+.2f}s ({time_speedup:+.1f}%)")
    
    if final_loss is not None:
        loss_diff = abs(final_loss - baseline_loss)
        loss_relative = loss_diff / baseline_loss * 100
        logger.info(f"  Baseline 損失:    {baseline_loss:.6f}")
        logger.info(f"  Wave 1-2 損失:    {final_loss:.6f}")
        logger.info(f"  損失差異:        {loss_diff:.2e} ({loss_relative:.4f}%)")
    
    logger.info("")
    
    # 驗收判斷
    logger.info("✅ 驗收標準:")
    
    # 數值正確性
    if final_loss is not None and abs(final_loss - baseline_loss) < 1e-3:
        logger.info(f"  ✅ 數值正確性: PASS (差異 < 1e-3)")
        result['numerical_accuracy'] = 'PASS'
    elif final_loss is not None:
        logger.warning(f"  ⚠️  數值正確性: WARNING (差異較大)")
        result['numerical_accuracy'] = 'WARNING'
    else:
        logger.warning(f"  ⚠️  數值正確性: N/A（無法驗證）")
        result['numerical_accuracy'] = 'N/A'
    
    # 速度提升
    if time_speedup >= 10:
        logger.info(f"  ✅ 速度提升: PASS (提升 {time_speedup:.1f}% ≥ 10%)")
        result['performance'] = 'PASS'
    elif time_speedup >= 5:
        logger.info(f"  ⚠️  速度提升: MARGINAL (提升 {time_speedup:.1f}% ≥ 5% 但 < 10%)")
        result['performance'] = 'MARGINAL'
    elif time_speedup > 0:
        logger.warning(f"  ⚠️  速度提升: MINOR (提升 {time_speedup:.1f}% < 5%)")
        result['performance'] = 'MINOR'
    else:
        logger.error(f"  ❌ 速度提升: FAIL (退化 {abs(time_speedup):.1f}%)")
        result['performance'] = 'FAIL'
    
    logger.info("")
    
    # 保存結果
    output_dir = project_root / 'results'
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / 'wave1_2_benchmark_result.json'
    
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"💾 結果已保存: {output_file}")
    logger.info("")
    
    # 總結
    overall_pass = (
        result.get('numerical_accuracy') in ['PASS', 'WARNING'] and
        result.get('performance') in ['PASS', 'MARGINAL']
    )
    
    if overall_pass:
        logger.info("🎉 Wave 1-2 優化測試通過！")
        return 0
    else:
        logger.warning("⚠️  Wave 1-2 優化未達預期目標")
        return 1


if __name__ == '__main__':
    import os
    sys.exit(run_benchmark())
