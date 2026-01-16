"""
TorchScript Kernel Fusion 性能測試腳本

測試融合後的激活函數與 Fourier Features 的性能提升。

基於 test_amp_p100_improved.py 的測試框架：
- GPU 充分暖身（5秒）
- 多輪測試（3輪）
- 數值等價性驗證
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from pathlib import Path
import sys

# 添加專案根目錄到 path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ========== 融合函數定義 ==========

@torch.jit.script
def fused_silu(x: torch.Tensor) -> torch.Tensor:
    """
    融合的 SiLU (Swish) 激活函數
    
    原始實現：F.silu(x) → 內部調用 x * sigmoid(x)，需要 2 次 kernel 啟動
    融合實現：直接計算 x * sigmoid(x)，僅 1 次 kernel 啟動
    
    預期加速：5-10%（針對 SiLU 密集的網路）
    """
    return x * torch.sigmoid(x)


@torch.jit.script
def fused_fourier_features(
    x: torch.Tensor, 
    B: torch.Tensor,
    sigma: float = 1.0,
    use_2pi: bool = True
) -> torch.Tensor:
    """
    融合的 Fourier Features 計算
    
    原始實現：
        z = x @ B
        z = 2π * z (optional)
        cos_z = torch.cos(z)
        sin_z = torch.sin(z)
        return torch.cat([cos_z, sin_z], dim=-1)
    
    融合實現：減少中間張量創建，合併 matmul + scaling + trig
    
    預期加速：2-5%
    """
    z = torch.matmul(x, B)
    if use_2pi:
        z = 2.0 * 3.14159265358979 * z
    return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)


# ========== 測試模型 ==========

class BaselineMLP(nn.Module):
    """基準模型：使用標準 F.silu"""
    
    def __init__(self, width: int = 256, depth: int = 8):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else 64  # 假設 Fourier Features 輸出 64
            layers.append(nn.Linear(in_dim, width))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(width, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
            h = F.silu(h)  # 標準 SiLU
        return self.output(h)


class FusedMLP(nn.Module):
    """優化模型：使用融合的 fused_silu"""
    
    def __init__(self, width: int = 256, depth: int = 8):
        super().__init__()
        layers = []
        for i in range(depth):
            in_dim = width if i > 0 else 64
            layers.append(nn.Linear(in_dim, width))
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(width, 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = layer(h)
            h = fused_silu(h)  # 融合 SiLU ⭐
        return self.output(h)


# ========== GPU 暖身函數 ==========

def warmup_gpu(device: torch.device, duration: float = 5.0):
    """
    GPU 暖身函數（基於 test_amp_p100_improved.py）
    
    讓 GPU 升頻到穩定狀態，避免測量誤差
    
    Args:
        device: GPU 設備
        duration: 暖身時長（秒）
    """
    print(f"🔥 GPU 暖身中 ({duration} 秒)...")
    start = time.time()
    x = torch.randn(4096, 256, device=device)
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256)
    ).to(device)
    
    while time.time() - start < duration:
        y = model(x)
        loss = y.sum()
        loss.backward()
    
    torch.cuda.synchronize()
    print("✅ GPU 暖身完成")


# ========== 基準測試函數 ==========

def benchmark_model(
    model: nn.Module,
    input_shape: tuple,
    device: torch.device,
    n_iters: int = 100,
    n_rounds: int = 3
) -> dict:
    """
    基準測試模型性能（基於 test_amp_p100_improved.py）
    
    Args:
        model: 待測試模型
        input_shape: 輸入形狀 (batch_size, in_dim)
        device: 計算設備
        n_iters: 每輪測試的迭代次數
        n_rounds: 測試輪數
    
    Returns:
        包含平均時間、標準差的字典
    """
    model.eval()
    
    # 預熱（10次）
    with torch.no_grad():
        x_warmup = torch.randn(*input_shape, device=device)
        for _ in range(10):
            _ = model(x_warmup)
    torch.cuda.synchronize()
    
    # 多輪測試
    round_times = []
    for round_idx in range(n_rounds):
        times = []
        for _ in range(n_iters):
            x = torch.randn(*input_shape, device=device)
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            with torch.no_grad():
                y = model(x)
            
            torch.cuda.synchronize()
            t1 = time.time()
            
            times.append((t1 - t0) * 1000)  # 轉為毫秒
        
        round_time = np.mean(times)
        round_times.append(round_time)
        print(f"  Round {round_idx+1}: {round_time:.3f} ± {np.std(times):.3f} ms")
    
    avg_time = np.mean(round_times)
    std_time = np.std(round_times)
    
    return {
        'mean': avg_time,
        'std': std_time,
        'round_times': round_times
    }


# ========== 數值等價性測試 ==========

def test_numerical_equivalence():
    """測試融合函數與原始函數的數值等價性"""
    print("\n" + "="*60)
    print("🔬 數值等價性測試")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 測試 1: fused_silu vs F.silu
    print("\n1️⃣  測試 fused_silu 數值等價性")
    x = torch.randn(1000, 256, device=device)
    
    y_baseline = F.silu(x)
    y_fused = fused_silu(x)
    
    abs_error = (y_baseline - y_fused).abs().max().item()
    rel_error = ((y_baseline - y_fused).abs() / (y_baseline.abs() + 1e-8)).max().item()
    
    print(f"   絕對誤差: {abs_error:.2e}")
    print(f"   相對誤差: {rel_error:.2e}")
    
    if abs_error < 1e-5:
        print("   ✅ 數值等價性驗證通過")
    else:
        print(f"   ❌ 數值誤差過大！ ({abs_error:.2e})")
        return False
    
    # 測試 2: fused_fourier_features
    print("\n2️⃣  測試 fused_fourier_features 數值等價性")
    x = torch.randn(1000, 3, device=device)
    B = torch.randn(3, 32, device=device)
    
    # 原始實現
    z_baseline = x @ B
    z_baseline = 2.0 * 3.14159265358979 * z_baseline
    y_baseline = torch.cat([torch.cos(z_baseline), torch.sin(z_baseline)], dim=-1)
    
    # 融合實現
    y_fused = fused_fourier_features(x, B, use_2pi=True)
    
    abs_error = (y_baseline - y_fused).abs().max().item()
    rel_error = ((y_baseline - y_fused).abs() / (y_baseline.abs() + 1e-8)).max().item()
    
    print(f"   絕對誤差: {abs_error:.2e}")
    print(f"   相對誤差: {rel_error:.2e}")
    
    if abs_error < 1e-5:
        print("   ✅ 數值等價性驗證通過")
    else:
        print(f"   ❌ 數值誤差過大！ ({abs_error:.2e})")
        return False
    
    return True


# ========== 性能測試 ==========

def test_performance():
    """測試融合函數的性能提升"""
    print("\n" + "="*60)
    print("⚡ TorchScript Fusion 性能測試")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cpu':
        print("❌ 需要 CUDA 設備進行性能測試")
        return
    
    # 檢測 GPU 型號
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n🖥️  GPU: {gpu_name}")
    
    # GPU 暖身
    warmup_gpu(device, duration=5.0)
    
    # 測試配置
    batch_size = 4096  # 較大 batch 以充分利用 GPU
    width = 256
    depth = 8
    n_iters = 100
    n_rounds = 3
    
    print(f"\n📊 測試配置:")
    print(f"   Batch Size: {batch_size}")
    print(f"   Width: {width}")
    print(f"   Depth: {depth}")
    print(f"   Iterations: {n_iters} × {n_rounds} rounds")
    
    # 創建模型
    print(f"\n🔨 創建模型...")
    model_baseline = BaselineMLP(width=width, depth=depth).to(device)
    model_fused = FusedMLP(width=width, depth=depth).to(device)
    
    # 複製權重（確保公平比較）
    model_fused.load_state_dict(model_baseline.state_dict())
    
    print(f"   模型參數: {sum(p.numel() for p in model_baseline.parameters()):,}")
    
    # 基準測試
    print(f"\n🏃 測試 Baseline (F.silu)...")
    result_baseline = benchmark_model(
        model_baseline,
        (batch_size, 64),
        device,
        n_iters=n_iters,
        n_rounds=n_rounds
    )
    
    print(f"\n🏃 測試 Fused (fused_silu)...")
    result_fused = benchmark_model(
        model_fused,
        (batch_size, 64),
        device,
        n_iters=n_iters,
        n_rounds=n_rounds
    )
    
    # 計算加速比
    print("\n" + "="*60)
    print("📈 結果總結")
    print("="*60)
    
    baseline_time = result_baseline['mean']
    fused_time = result_fused['mean']
    speedup = baseline_time / fused_time
    
    print(f"\n{'模型':<15} {'平均時間 (ms)':<20} {'標準差 (ms)':<15}")
    print("-" * 50)
    print(f"{'Baseline':<15} {baseline_time:>8.3f} ± {result_baseline['std']:>6.3f}")
    print(f"{'Fused':<15} {fused_time:>8.3f} ± {result_fused['std']:>6.3f}")
    print("-" * 50)
    print(f"{'加速比:':<15} {speedup:>8.3f}x")
    
    if speedup >= 1.05:
        print(f"\n✅ 達成目標！加速 {(speedup-1)*100:.1f}%")
    elif speedup >= 1.02:
        print(f"\n⚠️  輕微加速 {(speedup-1)*100:.1f}%（可接受）")
    elif speedup >= 0.98:
        print(f"\n⚠️  性能持平（±2%）")
    else:
        print(f"\n❌ 性能下降 {(1-speedup)*100:.1f}%")
    
    return {
        'baseline': result_baseline,
        'fused': result_fused,
        'speedup': speedup
    }


# ========== 主函數 ==========

def main():
    """主測試流程"""
    print("\n" + "="*60)
    print("🚀 TorchScript Kernel Fusion 測試")
    print("="*60)
    
    # Step 1: 數值等價性測試
    print("\n[Step 1/2] 數值等價性驗證")
    if not test_numerical_equivalence():
        print("\n❌ 數值等價性測試失敗，中止性能測試")
        return
    
    # Step 2: 性能測試
    print("\n[Step 2/2] 性能基準測試")
    results = test_performance()
    
    # 保存結果
    if results:
        output_dir = Path(__file__).parent.parent / "profiler_results"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / "torchscript_fusion_results.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("TorchScript Kernel Fusion Performance Test Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n")
            f.write(f"PyTorch: {torch.__version__}\n\n")
            f.write(f"Baseline (F.silu): {results['baseline']['mean']:.3f} ± {results['baseline']['std']:.3f} ms\n")
            f.write(f"Fused (fused_silu): {results['fused']['mean']:.3f} ± {results['fused']['std']:.3f} ms\n")
            f.write(f"Speedup: {results['speedup']:.3f}x ({(results['speedup']-1)*100:.1f}% faster)\n")
        
        print(f"\n💾 結果已保存至: {output_file}")
    
    print("\n" + "="*60)
    print("✅ 測試完成")
    print("="*60)


if __name__ == "__main__":
    main()
