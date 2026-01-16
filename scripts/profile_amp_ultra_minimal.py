"""
超簡化 AMP Profiling 測試

使用簡單 MLP 測試 AMP 效能（避免複雜模型初始化）

用法：
    python scripts/profile_amp_ultra_minimal.py --iterations 100
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import numpy as np


class SimplePINNMLP(nn.Module):
    """簡化的 PINN MLP（用於 profiling）"""
    def __init__(self, in_dim=3, out_dim=3, width=768, depth=2):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(in_dim, width))
        layers.append(nn.SiLU())
        
        for _ in range(depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(nn.SiLU())
        
        layers.append(nn.Linear(width, out_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def profile_training(model, optimizer, batch_size, num_iterations, use_amp, device):
    """執行 profiling"""
    print(f"\n{'='*60}")
    print(f"模式: {'FP16 (AMP)' if use_amp else 'FP32 (Baseline)'}")
    print(f"批次大小: {batch_size}")
    print(f"測試迭代: {num_iterations}")
    print(f"{'='*60}\n")
    
    scaler = amp.GradScaler() if use_amp else None
    
    # Warmup
    print("Warmup (10 iterations)...")
    for _ in range(10):
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        targets = torch.rand(batch_size, 3, device=device)
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
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 實際測試
    print("開始 profiling...")
    iteration_times = []
    memory_usage = []
    
    for i in range(num_iterations):
        coords = torch.rand(batch_size, 3, device=device, requires_grad=True)
        targets = torch.rand(batch_size, 3, device=device)
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
        memory_usage.append(torch.cuda.memory_allocated() / (1024 ** 2))
        
        if (i + 1) % 20 == 0:
            print(f"Iteration {i+1}/{num_iterations} | "
                  f"Time: {iter_time*1000:.2f} ms | "
                  f"Memory: {memory_usage[-1]:.1f} MB")
        
        del coords, targets, outputs, loss
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
    
    results = {
        'total_time': sum(iteration_times),
        'avg_time': np.mean(iteration_times),
        'std_time': np.std(iteration_times),
        'min_time': np.min(iteration_times),
        'max_time': np.max(iteration_times),
        'peak_memory': peak_memory,
        'avg_memory': np.mean(memory_usage),
    }
    
    print(f"\n總時間: {results['total_time']:.2f}s")
    print(f"平均時間: {results['avg_time']*1000:.2f} ms/iter")
    print(f"峰值記憶體: {results['peak_memory']:.1f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=8000)
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--depth', type=int, default=2)
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return 1
    
    device = torch.device('cuda')
    print(f"\n{'='*60}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"{'='*60}")
    
    # 創建模型
    print(f"\n創建模型 (width={args.width}, depth={args.depth})...")
    model = SimplePINNMLP(width=args.width, depth=args.depth).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"參數量: {num_params:,}")
    print(f"模型大小: {num_params * 4 / (1024**2):.2f} MB (FP32)")
    
    # FP32 測試
    print(f"\n{'='*60}")
    print("FP32 Baseline 測試")
    print(f"{'='*60}")
    optimizer_fp32 = torch.optim.Adam(model.parameters(), lr=0.001)
    fp32_results = profile_training(model, optimizer_fp32, args.batch_size, 
                                    args.iterations, False, device)
    
    # 重新創建模型（FP16）
    model = SimplePINNMLP(width=args.width, depth=args.depth).to(device)
    optimizer_fp16 = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n{'='*60}")
    print("FP16 (AMP) 測試")
    print(f"{'='*60}")
    fp16_results = profile_training(model, optimizer_fp16, args.batch_size, 
                                    args.iterations, True, device)
    
    # 比較
    speedup = fp32_results['avg_time'] / fp16_results['avg_time']
    mem_ratio = fp16_results['peak_memory'] / fp32_results['peak_memory']
    
    print(f"\n{'='*60}")
    print("FP32 vs FP16 (AMP) 比較")
    print(f"{'='*60}\n")
    print(f"⏱️  平均迭代時間:")
    print(f"  FP32: {fp32_results['avg_time']*1000:.2f} ms")
    print(f"  FP16: {fp16_results['avg_time']*1000:.2f} ms")
    print(f"  加速比: {speedup:.2f}x ({((speedup-1)*100):.1f}% faster)\n")
    print(f"💾 記憶體使用:")
    print(f"  FP32: {fp32_results['peak_memory']:.1f} MB")
    print(f"  FP16: {fp16_results['peak_memory']:.1f} MB")
    print(f"  減少: {((1-mem_ratio)*100):.1f}%\n")
    print(f"{'='*60}")
    
    if speedup > 1.3:
        print(f"✅ AMP 顯著加速 ({speedup:.2f}x)")
    elif speedup > 1.1:
        print(f"✅ AMP 適度加速 ({speedup:.2f}x)")
    else:
        print(f"⚠️  AMP 加速有限 ({speedup:.2f}x)")
    print(f"{'='*60}\n")
    
    # 保存結果
    import json
    from pathlib import Path
    
    results_file = Path('results/amp_ultra_minimal.json')
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump({
            'config': {'batch_size': args.batch_size, 'iterations': args.iterations,
                      'width': args.width, 'depth': args.depth},
            'fp32': fp32_results,
            'fp16': fp16_results,
            'comparison': {'speedup': speedup, 'memory_reduction': (1-mem_ratio)*100}
        }, f, indent=2)
    
    print(f"✅ 結果已保存至: {results_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
