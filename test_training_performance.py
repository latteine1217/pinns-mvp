"""
測試實際訓練場景中的梯度計算性能
==============================

模擬真實 PINNs 訓練的梯度計算流程：
1. 模型前向傳播
2. 計算所有梯度（一階 + 二階）
3. 計算 PDE residuals
4. 反向傳播

比較向量化方法與當前方法的實際訓練速度差異

Author: Performance Optimization Team
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import time
import sys
from pathlib import Path

# 添加專案路徑
sys.path.insert(0, str(Path(__file__).parent))

from pinnx.physics.gradient_cache_2d import GradientCache2D


class SimplePINN(nn.Module):
    """簡化版 PINN 模型（用於性能測試）"""
    def __init__(self, hidden_dim=256, depth=2):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))
        layers.append(nn.SiLU())
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 3))  # u, v, p
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)


def compute_pde_residual_current(predictions, coords):
    """
    當前方法：使用標準 GradientCache2D
    """
    cache = GradientCache2D(device=coords.device)
    
    # 計算梯度
    grads = cache.compute_all_gradients(predictions, coords, create_graph=True)
    
    # 連續方程: ∂u/∂x + ∂v/∂y = 0
    continuity = grads['u_x'] + grads['v_y']
    
    # 動量方程（簡化，忽略對流項）: -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    nu = 0.01
    momentum_x = -grads['p_x'] + nu * (grads['u_xx'] + grads['u_yy'])
    momentum_y = -grads['p_y'] + nu * (grads['v_xx'] + grads['v_yy'])
    
    # PDE residual
    residual = continuity**2 + momentum_x**2 + momentum_y**2
    return residual.mean()


def simulate_training_step(model, coords, method='current', device='cuda'):
    """
    模擬一個完整的訓練步驟
    
    Args:
        model: PINN 模型
        coords: 輸入座標
        method: 'current' 或 'vectorized'
        device: 計算設備
    
    Returns:
        step_time: 步驟時間（秒）
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    t0 = time.time()
    
    # 前向傳播
    predictions_raw = model(coords)
    predictions = {
        'u': predictions_raw[:, 0:1],
        'v': predictions_raw[:, 1:2],
        'p': predictions_raw[:, 2:3]
    }
    
    # 計算 PDE residual（這裡會調用梯度計算）
    loss = compute_pde_residual_current(predictions, coords)
    
    # 反向傳播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    torch.cuda.synchronize() if device == 'cuda' else None
    step_time = time.time() - t0
    
    return step_time


def benchmark_training(batch_size=7000, n_steps=10, device='cuda'):
    """
    基準測試：比較實際訓練速度
    
    Args:
        batch_size: 批次大小（模擬 batch_size=8000, N_pde=6000 的一半）
        n_steps: 測試步數
        device: 計算設備
    """
    print(f"\n{'='*60}")
    print(f"實際訓練性能測試")
    print(f"{'='*60}")
    print(f"批次大小: {batch_size}")
    print(f"測試步數: {n_steps}")
    print(f"設備: {device}")
    
    # 創建模型和數據
    model = SimplePINN(hidden_dim=256, depth=2).to(device)
    coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
    
    # 預熱（避免初次運行的開銷）
    print("\n預熱中...")
    for _ in range(3):
        _ = simulate_training_step(model, coords, method='current', device=device)
    
    # 測試當前方法
    print(f"\n測試當前方法（標準 GradientCache2D）...")
    times_current = []
    for i in range(n_steps):
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        t = simulate_training_step(model, coords, method='current', device=device)
        times_current.append(t)
        if (i + 1) % 5 == 0:
            print(f"  步驟 {i+1}/{n_steps}: {t*1000:.1f} ms")
    
    avg_time_current = sum(times_current) / len(times_current)
    std_time_current = (sum((t - avg_time_current)**2 for t in times_current) / len(times_current))**0.5
    
    print(f"\n{'='*60}")
    print(f"結果")
    print(f"{'='*60}")
    print(f"當前方法:")
    print(f"  平均時間: {avg_time_current*1000:.2f} ± {std_time_current*1000:.2f} ms")
    print(f"  最快: {min(times_current)*1000:.2f} ms")
    print(f"  最慢: {max(times_current)*1000:.2f} ms")
    
    # 估算完整訓練時間
    epochs_per_window = 300000
    n_windows = 3
    total_epochs = epochs_per_window * n_windows
    
    estimated_time_current = avg_time_current * total_epochs / 3600 / 24  # 天
    
    print(f"\n{'='*60}")
    print(f"估算完整訓練時間")
    print(f"{'='*60}")
    print(f"總 epochs: {total_epochs:,}")
    print(f"當前方法: {estimated_time_current:.1f} 天")
    
    return avg_time_current


def main():
    """主函數"""
    print("\n" + "="*60)
    print("PINNs 訓練性能測試 - 實際場景模擬")
    print("="*60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 記憶體: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 測試 batch_size=7000（對應 DDP 下 total=14000）
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    avg_time = benchmark_training(batch_size=7000, n_steps=10, device=device)
    
    print(f"\n{'='*60}")
    print("測試完成")
    print("="*60)
    print(f"✅ 當前配置的單步時間: {avg_time*1000:.2f} ms")
    print(f"📊 這個結果可以用來評估向量化優化的實際影響")


if __name__ == "__main__":
    main()
