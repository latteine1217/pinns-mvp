"""
PyTorch Profiler 訓練瓶頸分析
===========================

啟用 PyTorch Profiler 分析訓練過程中的性能瓶頸：
- GPU 時間分布
- CPU 時間分布
- 記憶體使用
- 算子耗時排名

使用方法：
    python scripts/train/train_with_profiler.py --cfg configs/xxx.yml

Author: Performance Optimization Team
Date: 2026-01-13
"""

import torch
from torch.profiler import profile, record_function, ProfilerActivity
import sys
from pathlib import Path

# 添加專案路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.physics.gradient_cache_2d import GradientCache2D
import torch.nn as nn


class SimplePINN(nn.Module):
    """簡化版 PINN 模型（用於快速 profiling）"""
    def __init__(self, hidden_dim=768, depth=2):
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


def training_step_with_profiler(model, optimizer, coords, device='cuda'):
    """
    單次訓練步驟（帶 profiler 標記）
    """
    with record_function("## Forward Pass"):
        predictions_raw = model(coords)
        predictions = {
            'u': predictions_raw[:, 0:1],
            'v': predictions_raw[:, 1:2],
            'p': predictions_raw[:, 2:3]
        }
    
    # 計算梯度
    with record_function("## Gradient Computation"):
        cache = GradientCache2D(device=device)
        
        with record_function("### compute_all_gradients"):
            grads = cache.compute_all_gradients(predictions, coords, create_graph=True)
    
    # 計算 PDE residual
    with record_function("## PDE Residual"):
        with record_function("### Continuity"):
            continuity = grads['u_x'] + grads['v_y']
        
        with record_function("### Momentum"):
            nu = 0.01
            momentum_x = -grads['p_x'] + nu * (grads['u_xx'] + grads['u_yy'])
            momentum_y = -grads['p_y'] + nu * (grads['v_xx'] + grads['v_yy'])
        
        with record_function("### Loss"):
            residual = continuity**2 + momentum_x**2 + momentum_y**2
            loss = residual.mean()
    
    # 反向傳播
    with record_function("## Backward Pass"):
        optimizer.zero_grad()
        loss.backward()
    
    with record_function("## Optimizer Step"):
        optimizer.step()
    
    return loss.item()


def profile_training(batch_size=7000, n_steps=5, device='cuda'):
    """
    使用 Profiler 分析訓練過程
    
    Args:
        batch_size: 批次大小（模擬 DDP 下單 GPU 的批次）
        n_steps: 分析的步數
        device: 計算設備
    """
    print(f"\n{'='*60}")
    print(f"PyTorch Profiler 訓練瓶頸分析")
    print(f"{'='*60}")
    print(f"批次大小: {batch_size}")
    print(f"分析步數: {n_steps}")
    print(f"設備: {device}")
    
    # 創建模型和優化器
    model = SimplePINN(hidden_dim=768, depth=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 預熱
    print(f"\n預熱中...")
    for _ in range(3):
        coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
        _ = training_step_with_profiler(model, optimizer, coords, device)
    
    # 開始 profiling
    print(f"\n開始 profiling...")
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(n_steps):
            coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
            loss = training_step_with_profiler(model, optimizer, coords, device)
            print(f"  步驟 {step+1}/{n_steps}: loss={loss:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Profiler 結果")
    print(f"{'='*60}")
    
    events = prof.key_averages()
    cuda_sort_key = "cuda_time_total" if events and hasattr(events[0], "cuda_time_total") else "cuda_time"

    # 1. 按 CUDA 時間排序（最關鍵）
    print(f"\n## Top 20 算子（按 CUDA 時間）")
    print(events.table(
        sort_by=cuda_sort_key,
        row_limit=20,
        max_src_column_width=50
    ))
    
    # 2. 按 CPU 時間排序
    print(f"\n## Top 20 算子（按 CPU 時間）")
    print(events.table(
        sort_by="cpu_time_total", 
        row_limit=20,
        max_src_column_width=50
    ))
    
    # 3. 按記憶體使用排序
    print(f"\n## Top 10 算子（按記憶體使用）")
    print(events.table(
        sort_by="cuda_memory_usage", 
        row_limit=10,
        max_src_column_width=50
    ))
    
    # 4. 自定義標記的時間統計
    print(f"\n## 自定義標記時間統計")
    for event in events:
        if event.key.startswith('##'):
            cuda_time = getattr(event, "cuda_time_total", getattr(event, "cuda_time", 0.0))
            print(f"{event.key:40s} CPU: {event.cpu_time_total/1000:8.2f} ms  "
                  f"CUDA: {cuda_time/1000:8.2f} ms  "
                  f"呼叫次數: {event.count}")
    
    # 5. 導出 Chrome trace（可選）
    output_dir = Path(project_root) / "profiler_results"
    output_dir.mkdir(exist_ok=True)
    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\n✅ Chrome trace 已導出: {trace_path}")
    print(f"   在 Chrome 瀏覽器中打開 chrome://tracing 並載入此文件查看詳細時間線")
    
    # 6. 導出 stacks（可選）
    stacks_path = output_dir / "stacks.txt"
    with open(stacks_path, 'w') as f:
        f.write(prof.key_averages(group_by_stack_n=5).table(
            sort_by=cuda_sort_key,
            row_limit=50
        ))
    print(f"✅ Stack traces 已導出: {stacks_path}")


def profile_gradient_computation_only(batch_size=7000, device='cuda'):
    """
    僅分析梯度計算部分（更細粒度）
    """
    print(f"\n{'='*60}")
    print(f"細粒度梯度計算 Profiling")
    print(f"{'='*60}")
    
    # 創建測試數據
    coords = torch.randn(batch_size, 2, device=device, requires_grad=True)
    u = (coords[:, 0:1] ** 2) + 2 * (coords[:, 1:2] ** 2)
    v = (coords[:, 0:1] * coords[:, 1:2])
    p = coords[:, 0:1] + coords[:, 1:2]
    
    predictions = {'u': u, 'v': v, 'p': p}
    
    # Profiling
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        with record_function("GradientCache2D.compute_all_gradients"):
            cache = GradientCache2D(device=device)
            
            with record_function("First Order - u"):
                u_grad = torch.autograd.grad(
                    u, coords, 
                    grad_outputs=torch.ones_like(u),
                    create_graph=True, retain_graph=True
                )[0]
            
            with record_function("First Order - v"):
                v_grad = torch.autograd.grad(
                    v, coords,
                    grad_outputs=torch.ones_like(v),
                    create_graph=True, retain_graph=True
                )[0]
            
            with record_function("First Order - p"):
                p_grad = torch.autograd.grad(
                    p, coords,
                    grad_outputs=torch.ones_like(p),
                    create_graph=False, retain_graph=True
                )[0]
            
            with record_function("Second Order - u"):
                # 向量化方法
                for i in range(2):
                    grad_outputs = torch.zeros_like(u_grad)
                    grad_outputs[:, i] = 1.0
                    _ = torch.autograd.grad(
                        u_grad, coords,
                        grad_outputs=grad_outputs,
                        create_graph=True, retain_graph=True
                    )[0]
            
            with record_function("Second Order - v"):
                for i in range(2):
                    grad_outputs = torch.zeros_like(v_grad)
                    grad_outputs[:, i] = 1.0
                    _ = torch.autograd.grad(
                        v_grad, coords,
                        grad_outputs=grad_outputs,
                        create_graph=True, retain_graph=True
                    )[0]
    
    print(f"\n梯度計算細節：")
    grad_events = prof.key_averages()
    grad_cuda_sort_key = "cuda_time_total" if grad_events and hasattr(grad_events[0], "cuda_time_total") else "cuda_time"
    print(grad_events.table(sort_by=grad_cuda_sort_key, row_limit=30))


def main():
    """主函數"""
    print(f"\n{'='*60}")
    print(f"PINNs 訓練性能 Profiling 工具")
    print(f"{'='*60}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print(f"\n⚠️  CUDA 不可用，使用 CPU（profiling 結果可能不準確）")
        device = 'cpu'
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    
    # 1. 完整訓練步驟 profiling
    profile_training(batch_size=7000, n_steps=5, device=device)
    
    # 2. 細粒度梯度計算 profiling
    print(f"\n\n")
    profile_gradient_computation_only(batch_size=7000, device=device)
    
    print(f"\n{'='*60}")
    print(f"Profiling 完成")
    print(f"{'='*60}")
    print(f"\n💡 下一步：")
    print(f"  1. 查看上述報告，找出 CUDA 時間最長的算子")
    print(f"  2. 打開 profiler_results/trace.json 查看時間線")
    print(f"  3. 針對瓶頸算子進行優化")


if __name__ == "__main__":
    main()
