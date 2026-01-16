"""
比較三種 PDE 殘差計算方法的性能

方法：
1. 原始 Navier-Stokes (u, v, p)
2. 向量化 Navier-Stokes (優化後)
3. 渦度-流函數方法 (ψ)

Author: Performance Optimization Team
Date: 2026-01-16
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinnx.losses.residuals import ns_residual_2d
from pinnx.losses.residuals_vectorized import ns_residual_2d_vectorized
from pinnx.losses.residuals_vorticity import stream_vorticity_residual_2d


class SimpleNSNet(nn.Module):
    """標準 NS 求解器：輸出 (u, v, p)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 3)  # u, v, p
        )
    
    def forward(self, x):
        return self.net(x)


class VorticityNet(nn.Module):
    """渦度求解器：只輸出流函數 ψ"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)  # ψ
        )
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def count_autograd_calls(residual_fn, net, coords, nu=1e-3):
    """
    統計 autograd 調用次數（近似）
    """
    # 這個只是估算，實際需要通過 profiler 精確測量
    pass


def benchmark_method(
    net: nn.Module,
    residual_fn,
    batch_size: int = 8000,
    num_iterations: int = 20,
    device: str = 'cuda'
):
    """
    Benchmark 單一方法
    """
    net = net.to(device)
    times = []
    
    # Warmup
    for _ in range(3):
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        output = net(coords)
        
        if isinstance(output, torch.Tensor) and output.shape[-1] == 3:
            # NS method: (u, v, p)
            velocity = output[:, :2]
            pressure = output[:, 2]
            residuals = residual_fn(coords, velocity, pressure, nu=1e-3)
        else:
            # Vorticity method: ψ
            residuals = residual_fn(coords, output, nu=1e-3)
        
        if isinstance(residuals, dict):
            loss = sum(torch.mean(v ** 2) for v in residuals.values())
        else:
            loss = (residuals ** 2).mean()
        
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    
    # Benchmark
    for i in range(num_iterations):
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        
        output = net(coords)
        
        if isinstance(output, torch.Tensor) and output.shape[-1] == 3:
            velocity = output[:, :2]
            pressure = output[:, 2]
            residuals = residual_fn(coords, velocity, pressure, nu=1e-3)
        else:
            residuals = residual_fn(coords, output, nu=1e-3)
        
        if isinstance(residuals, dict):
            loss = sum(torch.mean(v ** 2) for v in residuals.values())
        else:
            loss = (residuals ** 2).mean()
        
        loss.backward()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end = time.perf_counter()
        times.append((end - start) * 1000)  # ms
    
    mem_after = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    mem_peak = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else mem_after
    
    return {
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'memory_peak': mem_peak,
        'memory_increase': mem_after - mem_before
    }


def run_comparison(batch_size=8000, device='cuda'):
    """
    執行三種方法的對比
    """
    if not torch.cuda.is_available():
        print("⚠️  未檢測到 GPU，使用 CPU 測試")
        device = 'cpu'
    
    print("=" * 80)
    print("PDE 殘差計算方法對比")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    print()
    
    # 方法 1: 原始 NS
    print("[1] 原始 Navier-Stokes (u, v, p)")
    print("-" * 80)
    net_ns = SimpleNSNet()
    result_ns = benchmark_method(net_ns, ns_residual_2d, batch_size, device=device)
    print(f"  時間: {result_ns['time_mean']:.2f} ± {result_ns['time_std']:.2f} ms")
    print(f"  記憶體: {result_ns['memory_peak']:.2f} MB (增長: {result_ns['memory_increase']:.2f} MB)")
    print()
    
    # 方法 2: 向量化 NS
    print("[2] 向量化 Navier-Stokes (優化版)")
    print("-" * 80)
    try:
        result_ns_vec = benchmark_method(net_ns, ns_residual_2d_vectorized, batch_size, device=device)
        print(f"  時間: {result_ns_vec['time_mean']:.2f} ± {result_ns_vec['time_std']:.2f} ms")
        print(f"  記憶體: {result_ns_vec['memory_peak']:.2f} MB (增長: {result_ns_vec['memory_increase']:.2f} MB)")
        speedup_vec = (result_ns['time_mean'] / result_ns_vec['time_mean'] - 1) * 100
        print(f"  ⚡ 加速: {speedup_vec:+.1f}%")
    except Exception as e:
        print(f"  ❌ 執行失敗: {e}")
        result_ns_vec = None
    print()
    
    # 方法 3: 渦度方法
    print("[3] 渦度-流函數方法 (ψ)")
    print("-" * 80)
    net_vort = VorticityNet()
    result_vort = benchmark_method(net_vort, stream_vorticity_residual_2d, batch_size, device=device)
    print(f"  時間: {result_vort['time_mean']:.2f} ± {result_vort['time_std']:.2f} ms")
    print(f"  記憶體: {result_vort['memory_peak']:.2f} MB (增長: {result_vort['memory_increase']:.2f} MB)")
    speedup_vort = (result_ns['time_mean'] / result_vort['time_mean'] - 1) * 100
    print(f"  ⚡ 加速: {speedup_vort:+.1f}%")
    print()
    
    # 總結對比
    print("=" * 80)
    print("總結對比")
    print("=" * 80)
    print()
    
    print(f"{'方法':<30} {'時間 (ms)':<15} {'記憶體 (MB)':<15} {'加速比':<10}")
    print("-" * 80)
    print(f"{'原始 NS (u,v,p)':<30} {result_ns['time_mean']:>10.2f}     {result_ns['memory_peak']:>10.2f}     {'Baseline':<10}")
    
    if result_ns_vec:
        speedup_vec_ratio = result_ns['time_mean'] / result_ns_vec['time_mean']
        print(f"{'向量化 NS (優化)':<30} {result_ns_vec['time_mean']:>10.2f}     {result_ns_vec['memory_peak']:>10.2f}     {speedup_vec_ratio:.2f}x")
    
    speedup_vort_ratio = result_ns['time_mean'] / result_vort['time_mean']
    print(f"{'渦度方法 (ψ)':<30} {result_vort['time_mean']:>10.2f}     {result_vort['memory_peak']:>10.2f}     {speedup_vort_ratio:.2f}x")
    print()
    
    # 理論分析
    print("=" * 80)
    print("理論分析")
    print("=" * 80)
    print()
    print("Autograd 調用次數估算 (2D Navier-Stokes):")
    print("  原始 NS:     ~18 次 (u梯度×2 + v梯度×2 + p梯度×1 + u_lap×6 + v_lap×6 + continuity×1)")
    print("  向量化 NS:   ~12 次 (梯度複用，減少重複計算)")
    print("  渦度方法:    ~6 次  (ω梯度×2 + ω_lap×4)")
    print()
    print("變量數量:")
    print("  原始 NS:     3 個 (u, v, p)")
    print("  渦度方法:    1 個 (ψ)")
    print()
    print("方程數量:")
    print("  原始 NS:     3 個 (momentum_x, momentum_y, continuity)")
    print("  渦度方法:    1 個 (vorticity_transport)")
    print()
    
    # 建議
    print("=" * 80)
    print("建議")
    print("=" * 80)
    print()
    
    if speedup_vort > 30:
        print("✅ 渦度方法顯著優於原始 NS，強烈推薦用於 2D Kolmogorov Flow")
    elif speedup_vort > 15:
        print("✅ 渦度方法有明顯優勢，建議用於 2D 問題")
    else:
        print("⚠️  渦度方法優勢不明顯，需進一步優化或考慮其他因素")
    
    print()
    print("注意事項:")
    print("  - 渦度方法僅適用於 2D 問題")
    print("  - 需要修改網絡輸出層（3→1）")
    print("  - 需要修改邊界條件（速度→流函數）")
    print("  - 需要修改感測器數據（速度→流函數或渦度）")
    print()


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_comparison(batch_size=8000, device=device)
