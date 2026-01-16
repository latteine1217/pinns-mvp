"""
最小化 AMP Profiling 測試

直接測試模型 forward + backward 的效能（不包含完整訓練流程）

用法：
    python scripts/profile_amp_minimal.py --epochs 50
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pinnx.models.fourier_mlp import PINNNet


def create_test_model(device):
    """創建測試模型（與 S2_K200 配置一致）"""
    model = PINNNet(
        in_dim=3,  # (x, y, t)
        out_dim=3,  # (u, v, p)
        width=768,
        depth=2,
        activation='swish',
        block_type='piratenet',
        use_rwf=True,
        rwf_scale_mean=1.0,
        rwf_scale_std=0.1,
        fourier_config={
            'type': 'hybrid',
            'axes': {
                0: {'type': 'none'},
                1: {'type': 'periodic', 'domain_size': 6.283185307179586, 'n_modes': 64},
                2: {'type': 'periodic', 'domain_size': 6.283185307179586, 'n_modes': 64}
            },
            'trainable_fourier': False
        }
    ).to(device)
    
    return model


def generate_test_batch(batch_size, device):
    """生成測試批次數據"""
    # 模擬實際訓練的輸入尺寸
    coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
    targets = torch.rand(batch_size, 3, device=device)
    return coords, targets


def profile_forward_backward(model, coords, targets, use_amp=False, scaler=None):
    """
    測試單次 forward + backward
    
    Returns:
        float: 執行時間（秒）
    """
    start = time.time()
    
    if use_amp:
        with amp.autocast():
            outputs = model(coords)
            loss = ((outputs - targets) ** 2).mean()
        
        scaler.scale(loss).backward()
        scaler.step(model.optimizer if hasattr(model, 'optimizer') else torch.optim.Adam(model.parameters()))
        scaler.update()
    else:
        outputs = model(coords)
        loss = ((outputs - targets) ** 2).mean()
        loss.backward()
    
    torch.cuda.synchronize()  # 確保 CUDA 操作完成
    elapsed = time.time() - start
    
    return elapsed, loss.item()


def run_profiling(model, batch_size, num_iterations, use_amp, device):
    """
    執行 profiling
    
    Args:
        model: 測試模型
        batch_size: 批次大小
        num_iterations: 測試迭代次數
        use_amp: 是否使用 AMP
        device: 計算設備
        
    Returns:
        Dict: profiling 結果
    """
    print(f"\n{'='*60}")
    print(f"模式: {'FP16 (AMP)' if use_amp else 'FP32 (Baseline)'}")
    print(f"批次大小: {batch_size}")
    print(f"測試迭代: {num_iterations}")
    print(f"{'='*60}\n")
    
    # 初始化 AMP scaler
    scaler = amp.GradScaler() if use_amp else None
    
    # 初始化優化器（dummy）
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup（避免第一次執行的初始化開銷）
    print("Warmup...")
    for _ in range(5):
        coords, targets = generate_test_batch(batch_size, device)
        optimizer.zero_grad()
        
        if use_amp:
            with amp.autocast():
                outputs = model(coords)
                loss = ((outputs - targets) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(coords)
            loss = ((outputs - targets) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        del coords, targets, outputs, loss
        torch.cuda.empty_cache()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 實際測試
    print("開始 profiling...")
    iteration_times = []
    losses = []
    memory_usage = []
    
    for i in range(num_iterations):
        coords, targets = generate_test_batch(batch_size, device)
        optimizer.zero_grad()
        
        iter_start = time.time()
        
        if use_amp:
            with amp.autocast():
                outputs = model(coords)
                loss = ((outputs - targets) ** 2).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(coords)
            loss = ((outputs - targets) ** 2).mean()
            loss.backward()
            optimizer.step()
        
        torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        
        iteration_times.append(iter_time)
        losses.append(loss.item())
        memory_usage.append(torch.cuda.memory_allocated() / (1024 ** 2))  # MB
        
        if (i + 1) % 10 == 0:
            print(f"Iteration {i+1}/{num_iterations} | "
                  f"Time: {iter_time:.4f}s | "
                  f"Memory: {memory_usage[-1]:.1f} MB")
        
        del coords, targets, outputs, loss
    
    # 統計結果
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    
    results = {
        'total_time': sum(iteration_times),
        'avg_time': np.mean(iteration_times),
        'std_time': np.std(iteration_times),
        'min_time': np.min(iteration_times),
        'max_time': np.max(iteration_times),
        'peak_memory': peak_memory,
        'avg_memory': np.mean(memory_usage),
        'avg_loss': np.mean(losses),
    }
    
    print(f"\n總時間: {results['total_time']:.2f}s")
    print(f"平均時間: {results['avg_time']:.4f}s/iter")
    print(f"峰值記憶體: {results['peak_memory']:.1f} MB")
    print(f"平均記憶體: {results['avg_memory']:.1f} MB")
    
    return results


def compare_results(fp32_results, fp16_results):
    """比較 FP32 vs FP16 結果"""
    print(f"\n{'='*60}")
    print("FP32 vs FP16 (AMP) 比較結果")
    print(f"{'='*60}\n")
    
    speedup = fp32_results['avg_time'] / fp16_results['avg_time']
    mem_ratio = fp16_results['peak_memory'] / fp32_results['peak_memory']
    
    print(f"⏱️  平均迭代時間:")
    print(f"  FP32: {fp32_results['avg_time']*1000:.2f} ms")
    print(f"  FP16: {fp16_results['avg_time']*1000:.2f} ms")
    print(f"  加速比: {speedup:.2f}x ({((speedup-1)*100):.1f}% faster)\n")
    
    print(f"💾 記憶體使用:")
    print(f"  FP32 Peak: {fp32_results['peak_memory']:.1f} MB")
    print(f"  FP16 Peak: {fp16_results['peak_memory']:.1f} MB")
    print(f"  記憶體比: {mem_ratio:.2f}x ({((1-mem_ratio)*100):.1f}% reduction)\n")
    
    print(f"{'='*60}")
    print("結論:")
    print(f"{'='*60}")
    
    if speedup > 1.3:
        print(f"✅ AMP 顯著加速 ({speedup:.2f}x)")
    elif speedup > 1.1:
        print(f"✅ AMP 適度加速 ({speedup:.2f}x)")
    else:
        print(f"⚠️  AMP 加速有限 ({speedup:.2f}x)")
    
    if mem_ratio < 0.7:
        print(f"✅ 記憶體顯著減少 ({(1-mem_ratio)*100:.1f}%)")
    elif mem_ratio < 0.9:
        print(f"✅ 記憶體適度減少 ({(1-mem_ratio)*100:.1f}%)")
    else:
        print(f"⚠️  記憶體減少有限 ({(1-mem_ratio)*100:.1f}%)")
    
    print(f"{'='*60}\n")
    
    return {'speedup': speedup, 'memory_reduction': (1 - mem_ratio) * 100}


def main():
    parser = argparse.ArgumentParser(description='最小化 AMP 效能測試')
    parser.add_argument('--batch-size', type=int, default=8000,
                        help='批次大小（預設: 8000）')
    parser.add_argument('--iterations', type=int, default=50,
                        help='測試迭代次數（預設: 50）')
    parser.add_argument('--skip-fp32', action='store_true',
                        help='跳過 FP32 測試')
    parser.add_argument('--skip-fp16', action='store_true',
                        help='跳過 FP16 測試')
    
    args = parser.parse_args()
    
    # 檢查 CUDA
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，AMP 需要 GPU 支援")
        return 1
    
    device = torch.device('cuda')
    print(f"\n{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"{'='*60}")
    
    # 創建模型
    print("\n創建測試模型...")
    model = create_test_model(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型參數量: {num_params:,}")
    print(f"模型大小: {num_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # 測試 FP32
    fp32_results = None
    if not args.skip_fp32:
        fp32_results = run_profiling(model, args.batch_size, args.iterations, 
                                     use_amp=False, device=device)
    
    # 重新創建模型（避免狀態影響）
    model = create_test_model(device)
    
    # 測試 FP16
    fp16_results = None
    if not args.skip_fp16:
        fp16_results = run_profiling(model, args.batch_size, args.iterations, 
                                     use_amp=True, device=device)
    
    # 比較結果
    if fp32_results and fp16_results:
        comparison = compare_results(fp32_results, fp16_results)
        
        # 保存結果
        import json
        results_file = project_root / 'results' / 'amp_minimal_profiling.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'batch_size': args.batch_size,
            'iterations': args.iterations,
            'fp32': fp32_results,
            'fp16': fp16_results,
            'comparison': comparison
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ 結果已保存至: {results_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
