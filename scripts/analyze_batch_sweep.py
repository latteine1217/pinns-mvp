#!/usr/bin/env python3
"""
Batch Size Sweep 結果分析工具

功能:
  1. 從 SLURM logs 中提取每個 batch size 的訓練時間
  2. 計算加速比與記憶體使用
  3. 生成比較表格與視覺化圖表

使用方式:
  python3 scripts/analyze_batch_sweep.py
  python3 scripts/analyze_batch_sweep.py --log-dir logs --output results/batch_sweep_analysis.txt
"""

import re
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

@dataclass
class BatchResult:
    """Batch Size 實驗結果"""
    batch_size: int
    n_pde: int
    job_id: str
    total_time: float  # seconds
    avg_epoch_time: float  # seconds
    epochs: int
    memory_peak: Optional[float] = None  # MB
    success: bool = True
    error_msg: Optional[str] = None


def parse_log_file(log_path: Path) -> Optional[BatchResult]:
    """解析 SLURM log 文件"""
    try:
        with open(log_path, 'r') as f:
            content = f.read()
        
        # 提取 Job ID
        job_id_match = re.search(r'Job ID: (\d+)', content)
        job_id = job_id_match.group(1) if job_id_match else log_path.stem.split('_')[-1]
        
        # 提取 batch_size 與 N_pde（從配置輸出）
        batch_match = re.search(r'batch_size:\s*(\d+)', content)
        npde_match = re.search(r'N_pde:\s*(\d+)', content)
        
        batch_size = int(batch_match.group(1)) if batch_match else 0
        n_pde = int(npde_match.group(1)) if npde_match else 0
        
        # 提取 epochs（從配置輸出）
        epoch_match = re.search(r'epochs:\s*(\d+)', content)
        epochs = int(epoch_match.group(1)) if epoch_match else 10
        
        # 提取總訓練時間（從 cProfile 輸出）
        # 尋找 "Epoch xx/yy" 的最後一行來計算時間
        epoch_lines = re.findall(r'Epoch\s+\d+/\d+.*?(\d+\.\d+)s', content)
        
        if not epoch_lines:
            # 嘗試從其他格式提取
            time_match = re.search(r'Total time:\s*(\d+\.\d+)s', content)
            if time_match:
                total_time = float(time_match.group(1))
                avg_epoch_time = total_time / epochs
            else:
                return None
        else:
            # 計算平均 epoch 時間
            epoch_times = [float(t) for t in epoch_lines]
            avg_epoch_time = np.mean(epoch_times)
            total_time = sum(epoch_times)
        
        # 提取記憶體使用（如果有 nvidia-smi 輸出）
        mem_match = re.search(r'(\d+)MiB\s*/\s*\d+MiB', content)
        memory_peak = float(mem_match.group(1)) if mem_match else None
        
        # 檢查是否有 OOM 錯誤
        success = True
        error_msg = None
        
        if 'out of memory' in content.lower() or 'cuda error' in content.lower():
            success = False
            error_msg = "OOM Error"
        elif not epoch_lines and not time_match:
            success = False
            error_msg = "No timing data found"
        
        return BatchResult(
            batch_size=batch_size,
            n_pde=n_pde,
            job_id=job_id,
            total_time=total_time,
            avg_epoch_time=avg_epoch_time,
            epochs=epochs,
            memory_peak=memory_peak,
            success=success,
            error_msg=error_msg
        )
        
    except Exception as e:
        print(f"⚠️  解析失敗 {log_path}: {e}")
        return None


def find_batch_sweep_logs(log_dir: Path) -> List[Path]:
    """尋找 batch sweep 相關的 log 文件"""
    # 匹配 profile_simple_*.log
    pattern = "profile_simple_*.log"
    logs = list(log_dir.glob(pattern))
    
    # 過濾出包含 batch_test 配置的 logs
    batch_logs = []
    for log in logs:
        try:
            with open(log, 'r') as f:
                first_kb = f.read(10000)  # 只讀前 10KB
                if 'batch_test' in first_kb or 'BATCH_SWEEP' in first_kb:
                    batch_logs.append(log)
        except:
            continue
    
    return sorted(batch_logs, key=lambda x: x.stat().st_mtime, reverse=True)


def analyze_results(results: List[BatchResult]) -> Dict:
    """分析結果並計算加速比"""
    # 找到 baseline (8k)
    baseline = next((r for r in results if r.batch_size == 8000 and r.success), None)
    
    if not baseline:
        print("⚠️  找不到 baseline (8k) 結果")
        baseline_time = results[0].avg_epoch_time if results else 1.0
    else:
        baseline_time = baseline.avg_epoch_time
    
    analysis = {
        'baseline_time': baseline_time,
        'results': []
    }
    
    for result in sorted(results, key=lambda x: x.batch_size):
        speedup = baseline_time / result.avg_epoch_time if result.success else 0.0
        
        analysis['results'].append({
            'batch_size': result.batch_size,
            'n_pde': result.n_pde,
            'job_id': result.job_id,
            'avg_epoch_time': result.avg_epoch_time,
            'speedup': speedup,
            'memory_peak': result.memory_peak,
            'success': result.success,
            'error_msg': result.error_msg
        })
    
    return analysis


def print_analysis_report(analysis: Dict, output_path: Optional[Path] = None):
    """打印分析報告"""
    
    report_lines = []
    
    def add_line(line: str = ""):
        report_lines.append(line)
        print(line)
    
    add_line("=" * 100)
    add_line("🔍 Batch Size Sweep 效能分析報告")
    add_line("=" * 100)
    add_line()
    add_line(f"Baseline (8k) 平均 Epoch 時間: {analysis['baseline_time']:.3f}s")
    add_line()
    add_line("-" * 100)
    add_line(f"{'Batch Size':<12} {'N_PDE':<10} {'Avg Epoch (s)':<15} {'Speedup':<10} "
             f"{'Memory (MB)':<15} {'Status':<10} {'Job ID':<10}")
    add_line("-" * 100)
    
    for res in analysis['results']:
        status = "✅ OK" if res['success'] else f"❌ {res['error_msg']}"
        memory_str = f"{res['memory_peak']:.0f}" if res['memory_peak'] else "N/A"
        speedup_str = f"{res['speedup']:.2f}x" if res['success'] else "N/A"
        
        add_line(f"{res['batch_size']:<12} {res['n_pde']:<10} {res['avg_epoch_time']:<15.3f} "
                 f"{speedup_str:<10} {memory_str:<15} {status:<10} {res['job_id']:<10}")
    
    add_line("-" * 100)
    add_line()
    
    # 找到最佳配置
    successful_results = [r for r in analysis['results'] if r['success']]
    if successful_results:
        best = max(successful_results, key=lambda x: x['speedup'])
        add_line("🏆 最佳配置:")
        add_line(f"   Batch Size: {best['batch_size']}")
        add_line(f"   Speedup: {best['speedup']:.2f}x")
        add_line(f"   Avg Epoch Time: {best['avg_epoch_time']:.3f}s")
        add_line()
    
    # 建議
    add_line("💡 建議:")
    
    if len(successful_results) >= 2:
        # 計算效能增益遞減
        speedups = [r['speedup'] for r in successful_results]
        if len(speedups) >= 2:
            gains = [speedups[i] - speedups[i-1] for i in range(1, len(speedups))]
            
            if gains[-1] < 0.1:  # 最後一次增益 < 0.1x
                add_line(f"   - Batch size {successful_results[-2]['batch_size']} "
                         f"為最佳平衡點（效能增益遞減）")
            else:
                add_line(f"   - 可以繼續增加 batch size 以獲得更多加速")
    
    # 檢查 OOM
    failed_results = [r for r in analysis['results'] if not r['success']]
    if failed_results:
        oom_batch = failed_results[0]['batch_size']
        add_line(f"   - Batch size {oom_batch} 及以上可能導致 OOM，避免使用")
    
    add_line("=" * 100)
    
    # 保存報告
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"\n📁 報告已保存至: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='分析 batch size sweep 實驗結果')
    parser.add_argument('--log-dir', type=str, default='logs',
                       help='SLURM logs 目錄（預設: logs）')
    parser.add_argument('--output', type=str, default='results/batch_sweep_analysis.txt',
                       help='輸出報告路徑（預設: results/batch_sweep_analysis.txt）')
    parser.add_argument('--json', action='store_true',
                       help='同時輸出 JSON 格式')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_path = Path(args.output)
    
    if not log_dir.exists():
        print(f"❌ Log 目錄不存在: {log_dir}")
        sys.exit(1)
    
    print(f"🔍 搜尋 batch sweep logs: {log_dir}")
    log_files = find_batch_sweep_logs(log_dir)
    
    if not log_files:
        print(f"❌ 找不到 batch sweep 相關的 log 文件")
        print(f"請確認已執行 run_batch_sweep.sh")
        sys.exit(1)
    
    print(f"✅ 找到 {len(log_files)} 個 log 文件")
    print()
    
    # 解析所有 logs
    results = []
    for log_file in log_files:
        print(f"📄 解析: {log_file.name}")
        result = parse_log_file(log_file)
        if result:
            results.append(result)
            status = "✅" if result.success else "❌"
            print(f"   {status} Batch={result.batch_size}, Time={result.avg_epoch_time:.3f}s/epoch")
        print()
    
    if not results:
        print("❌ 沒有成功解析任何 log 文件")
        sys.exit(1)
    
    # 去重（保留最新的結果）
    unique_results = {}
    for result in results:
        key = result.batch_size
        if key not in unique_results or result.job_id > unique_results[key].job_id:
            unique_results[key] = result
    
    results = list(unique_results.values())
    
    # 分析結果
    analysis = analyze_results(results)
    
    # 打印報告
    print_analysis_report(analysis, output_path)
    
    # 輸出 JSON（如果需要）
    if args.json:
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"📁 JSON 結果已保存至: {json_path}")


if __name__ == '__main__':
    main()
