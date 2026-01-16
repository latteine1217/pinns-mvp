"""
測試向量化 PDE 殘差的性能與記憶體使用

對比三種方法：
1. 原始逐個計算（Baseline）
2. Gradient Cache（全局快取，Wave 2 優化）
3. Vectorized Residual（部分向量化，記憶體優化版）

Author: Performance Optimization Team
Date: 2026-01-16
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Callable
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pinnx.losses.residuals import ns_residual_2d, ns_residual_3d
from pinnx.losses.residuals_vectorized_v2 import ns_residual_2d_vectorized, ns_residual_3d_vectorized


class SimpleMLP(nn.Module):
    """簡單的 MLP 用於模擬真實訓練場景"""
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


def get_memory_usage():
    """獲取當前 GPU 記憶體使用（MB）"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    else:
        return 0.0


def benchmark_residual_2d(
    residual_fn: Callable,
    batch_size: int = 8000,
    num_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark 2D 殘差計算
    
    Returns:
        {
            'time_mean': 平均執行時間 (ms),
            'time_std': 時間標準差 (ms),
            'memory_peak': 峰值記憶體 (MB),
            'memory_increase': 記憶體增長 (MB)
        }
    """
    # 建立簡單神經網絡（模擬真實訓練）
    net_uv = SimpleMLP(2, 2).to(device)  # u, v
    net_p = SimpleMLP(2, 1).to(device)   # p
    
    # Warmup (避免首次執行的初始化開銷)
    for _ in range(3):
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        velocity = net_uv(coords)  # [batch, 2]
        pressure = net_p(coords).squeeze(-1)  # [batch]
        _ = residual_fn(coords, velocity, pressure, nu=1e-3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # 清除快取並記錄初始記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = get_memory_usage()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        # 重新生成輸入（避免快取影響）
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        velocity = net_uv(coords)
        pressure = net_p(coords).squeeze(-1)
        
        start = time.perf_counter()
        residuals = residual_fn(coords, velocity, pressure, nu=1e-3)
        
        # 處理字典返回值
        if isinstance(residuals, dict):
            loss = sum(torch.mean(v ** 2) for v in residuals.values())
        else:
            loss = (residuals ** 2).sum()
        
        loss.backward()  # 測試完整的 backward pass
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)  # 轉換為 ms
        
        # 清理梯度（重要！）
        net_uv.zero_grad()
        net_p.zero_grad()
    
    mem_after = get_memory_usage()
    
    return {
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'memory_peak': mem_after,
        'memory_increase': mem_after - mem_before
    }


def benchmark_residual_3d(
    residual_fn: Callable,
    batch_size: int = 8000,
    num_iterations: int = 10,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Benchmark 3D 殘差計算
    """
    # 建立簡單神經網絡
    net_uvw = SimpleMLP(3, 3).to(device)  # u, v, w
    net_p = SimpleMLP(3, 1).to(device)    # p
    
    # Warmup
    for _ in range(3):
        coords = torch.randn(batch_size, 3, device=device, requires_grad=True)
        velocity = net_uvw(coords)
        pressure = net_p(coords).squeeze(-1)
        _ = residual_fn(coords, velocity, pressure, nu=1e-3)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # 清除快取並記錄初始記憶體
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    mem_before = get_memory_usage()
    
    # Benchmark
    times = []
    for _ in range(num_iterations):
        coords = torch.randn(batch_size, 3, device=device, requires_grad=True)
        velocity = net_uvw(coords)
        pressure = net_p(coords).squeeze(-1)
        
        start = time.perf_counter()
        residuals = residual_fn(coords, velocity, pressure, nu=1e-3)
        
        # 處理字典返回值
        if isinstance(residuals, dict):
            loss = sum(torch.mean(v ** 2) for v in residuals.values())
        else:
            loss = (residuals ** 2).sum()
        
        loss.backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        
        times.append((end - start) * 1000)
        
        # 清理梯度
        net_uvw.zero_grad()
        net_p.zero_grad()
    
    mem_after = get_memory_usage()
    
    return {
        'time_mean': float(np.mean(times)),
        'time_std': float(np.std(times)),
        'memory_peak': mem_after,
        'memory_increase': mem_after - mem_before
    }


def run_comparison(batch_size: int = 8000, device: str = 'cuda'):
    """
    執行完整對比測試
    """
    print("=" * 80)
    print(f"向量化 PDE 殘差性能測試")
    print(f"Batch Size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 80)
    
    # ========== 2D 測試 ==========
    print("\n📊 2D Navier-Stokes Residual")
    print("-" * 80)
    
    print("\n[1] Baseline（原始逐個計算）")
    baseline_2d = benchmark_residual_2d(ns_residual_2d, batch_size, device=device)
    print(f"  ⏱️  執行時間: {baseline_2d['time_mean']:.2f} ± {baseline_2d['time_std']:.2f} ms")
    print(f"  💾 峰值記憶體: {baseline_2d['memory_peak']:.2f} MB")
    print(f"  📈 記憶體增長: {baseline_2d['memory_increase']:.2f} MB")
    
    print("\n[2] Vectorized（部分向量化）")
    vectorized_2d = benchmark_residual_2d(ns_residual_2d_vectorized, batch_size, device=device)
    print(f"  ⏱️  執行時間: {vectorized_2d['time_mean']:.2f} ± {vectorized_2d['time_std']:.2f} ms")
    print(f"  💾 峰值記憶體: {vectorized_2d['memory_peak']:.2f} MB")
    print(f"  📈 記憶體增長: {vectorized_2d['memory_increase']:.2f} MB")
    
    # 計算提升
    speedup_2d = baseline_2d['time_mean'] / vectorized_2d['time_mean']
    mem_overhead_2d = (vectorized_2d['memory_peak'] - baseline_2d['memory_peak']) / baseline_2d['memory_peak'] * 100
    
    print("\n📈 2D 優化效果:")
    print(f"  🚀 加速比: {speedup_2d:.2f}x ({(speedup_2d - 1) * 100:+.1f}%)")
    print(f"  💾 記憶體開銷: {mem_overhead_2d:+.1f}%")
    
    # ========== 3D 測試 ==========
    print("\n\n📊 3D Navier-Stokes Residual")
    print("-" * 80)
    
    print("\n[1] Baseline（原始逐個計算）")
    baseline_3d = benchmark_residual_3d(ns_residual_3d, batch_size, device=device)
    print(f"  ⏱️  執行時間: {baseline_3d['time_mean']:.2f} ± {baseline_3d['time_std']:.2f} ms")
    print(f"  💾 峰值記憶體: {baseline_3d['memory_peak']:.2f} MB")
    print(f"  📈 記憶體增長: {baseline_3d['memory_increase']:.2f} MB")
    
    print("\n[2] Vectorized（部分向量化）")
    vectorized_3d = benchmark_residual_3d(ns_residual_3d_vectorized, batch_size, device=device)
    print(f"  ⏱️  執行時間: {vectorized_3d['time_mean']:.2f} ± {vectorized_3d['time_std']:.2f} ms")
    print(f"  💾 峰值記憶體: {vectorized_3d['memory_peak']:.2f} MB")
    print(f"  📈 記憶體增長: {vectorized_3d['memory_increase']:.2f} MB")
    
    # 計算提升
    speedup_3d = baseline_3d['time_mean'] / vectorized_3d['time_mean']
    mem_overhead_3d = (vectorized_3d['memory_peak'] - baseline_3d['memory_peak']) / baseline_3d['memory_peak'] * 100
    
    print("\n📈 3D 優化效果:")
    print(f"  🚀 加速比: {speedup_3d:.2f}x ({(speedup_3d - 1) * 100:+.1f}%)")
    print(f"  💾 記憶體開銷: {mem_overhead_3d:+.1f}%")
    
    # ========== 總結 ==========
    print("\n\n" + "=" * 80)
    print("📝 總結")
    print("=" * 80)
    print(f"\n✅ 2D 加速: {speedup_2d:.2f}x, 記憶體開銷: {mem_overhead_2d:+.1f}%")
    print(f"✅ 3D 加速: {speedup_3d:.2f}x, 記憶體開銷: {mem_overhead_3d:+.1f}%")
    
    # 判斷是否值得使用
    if speedup_2d > 1.1 and mem_overhead_2d < 15:
        print("\n🎯 結論: 向量化優化值得使用！")
        print("   - 加速明顯（>10%）")
        print("   - 記憶體開銷可控（<15%）")
    elif speedup_2d > 1.05:
        print("\n⚠️  結論: 向量化優化效果有限")
        print(f"   - 加速較小（{(speedup_2d - 1) * 100:.1f}%）")
        print("   - 建議在記憶體充足時使用")
    else:
        print("\n❌ 結論: 向量化優化不建議使用")
        print("   - 加速不明顯或反而變慢")
        print("   - 建議保持原始實現")
    
    print("=" * 80)


if __name__ == "__main__":
    # 檢測設備
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"🎮 GPU 檢測到: {torch.cuda.get_device_name(0)}")
        print(f"💾 總記憶體: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB\n")
    else:
        device = 'cpu'
        print("⚠️  未檢測到 GPU，使用 CPU 測試\n")
    
    # 執行測試
    run_comparison(batch_size=8000, device=device)
