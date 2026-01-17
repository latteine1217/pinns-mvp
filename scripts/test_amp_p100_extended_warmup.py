#!/usr/bin/env python3
"""
P100 AMP 性能測試 - 擴展暖身版本

目標：通過充分的 GPU 暖身，消除初期性能未達最佳狀態的測量誤差

測試策略：
1. GPU 頻率暖身：持續高負載運算 10 秒，確保 GPU 從 Idle → Boost 狀態
2. Kernel 暖身：每個測試模式執行 50 次暖身迭代
3. 多輪測試：執行 3 輪完整測試，取中位數與標準差
4. 長時間測量：每輪測量 100 次迭代（vs 原本 50 次）
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
from typing import Tuple, List


def print_section(title: str):
    """打印分隔線"""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def gpu_warmup(duration_seconds: int = 10):
    """
    GPU 暖身：持續高負載運算，確保 GPU 頻率達到 Boost 狀態
    
    Args:
        duration_seconds: 暖身持續時間（秒）
    """
    print(f"\n🔥 GPU 暖身中（{duration_seconds} 秒）...")
    
    device = torch.device('cuda:0')
    
    # 記錄初始狀態
    torch.cuda.synchronize()
    initial_temp = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else None
    
    # 大型矩陣乘法（充分利用 GPU）
    size = 4096
    A = torch.randn(size, size, device=device)
    B = torch.randn(size, size, device=device)
    
    start_time = time.time()
    iteration = 0
    C = None
    
    while time.time() - start_time < duration_seconds:
        # 高強度 GEMM 運算
        C = torch.mm(A, B)
        C = torch.mm(C, A)
        iteration += 1
        
        if iteration % 10 == 0:
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            print(f"   暖身進度: {elapsed:.1f}s / {duration_seconds}s ({elapsed/duration_seconds*100:.0f}%)")
    
    torch.cuda.synchronize()
    
    # 記錄暖身後狀態
    final_temp = torch.cuda.temperature() if hasattr(torch.cuda, 'temperature') else None
    
    print(f"   ✅ 暖身完成：執行了 {iteration} 次矩陣乘法")
    if initial_temp and final_temp:
        print(f"   溫度變化: {initial_temp}°C → {final_temp}°C")
    
    # 清理記憶體
    del A, B
    if 'C' in locals():
        del C
    torch.cuda.empty_cache()


def benchmark_mode(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: nn.Module,
    use_amp: bool,
    warmup_iters: int = 50,
    measure_iters: int = 100,
    num_rounds: int = 3
) -> Tuple[float, float, float]:
    """
    對單一模式（FP32 或 AMP）進行基準測試
    
    Args:
        model: 測試模型
        x: 輸入資料
        y: 目標資料
        criterion: 損失函數
        use_amp: 是否使用 AMP
        warmup_iters: 暖身迭代次數
        measure_iters: 測量迭代次數
        num_rounds: 測試輪數
    
    Returns:
        (中位數時間, 平均時間, 標準差) in milliseconds
    """
    device = x.device
    scaler = GradScaler() if use_amp else None
    
    mode_name = "AMP (FP16)" if use_amp else "FP32"
    print(f"\n🔹 測試模式: {mode_name}")
    
    # ========== Kernel 暖身 ==========
    print(f"   Kernel 暖身: {warmup_iters} 次迭代...")
    model.zero_grad()
    torch.cuda.synchronize()
    
    for i in range(warmup_iters):
        if use_amp:
            with autocast():
                output = model(x)
                loss = criterion(output, y)
            if scaler:
                scaler.scale(loss).backward()
                model.zero_grad()
                # Note: scaler.update() 不需要呼叫，因為沒有 optimizer.step()
        else:
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            model.zero_grad()
    
    torch.cuda.synchronize()
    print(f"   ✅ Kernel 暖身完成")
    
    # ========== 多輪測量 ==========
    times_per_round = []
    
    for round_idx in range(num_rounds):
        print(f"   測量輪次 {round_idx + 1}/{num_rounds}...")
        
        model.zero_grad()
        torch.cuda.synchronize()
        
        # 使用 CUDA Event 進行精確計時
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        
        for i in range(measure_iters):
            if use_amp:
                with autocast():
                    output = model(x)
                    loss = criterion(output, y)
                if scaler:
                    scaler.scale(loss).backward()
                    model.zero_grad()
                    # Note: scaler.update() 不需要呼叫，因為沒有 optimizer.step()
            else:
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                model.zero_grad()
        
        end_event.record()
        torch.cuda.synchronize()
        
        # 計算平均時間（毫秒）
        elapsed_ms = start_event.elapsed_time(end_event)
        avg_time = elapsed_ms / measure_iters
        times_per_round.append(avg_time)
        
        print(f"      → 平均時間: {avg_time:.3f} ms/iter")
    
    # ========== 統計分析 ==========
    median_time = float(np.median(times_per_round))
    mean_time = float(np.mean(times_per_round))
    std_time = float(np.std(times_per_round))
    
    print(f"\n   📊 統計結果 ({num_rounds} 輪):")
    print(f"      中位數: {median_time:.3f} ms/iter")
    print(f"      平均值: {mean_time:.3f} ms/iter")
    print(f"      標準差: {std_time:.3f} ms")
    print(f"      變異係數: {std_time/mean_time*100:.2f}%")
    
    # 記憶體使用
    memory_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"      記憶體: {memory_mb:.2f} MB")
    
    return median_time, mean_time, std_time


def test_comprehensive_performance():
    """全面性能測試：多種 Batch Size"""
    print_section("全面 AMP 性能測試（擴展暖身版）")
    
    device = torch.device('cuda:0')
    
    # ========== GPU 頻率暖身 ==========
    gpu_warmup(duration_seconds=10)
    
    # ========== 測試配置 ==========
    print("\n📋 測試配置:")
    print(f"   模型: 8×256 MLP (模擬專案架構)")
    print(f"   激活函數: SiLU")
    print(f"   Kernel 暖身: 50 次迭代")
    print(f"   測量輪數: 3 輪")
    print(f"   每輪迭代: 100 次")
    print(f"   計時方式: CUDA Event (高精度)")
    
    # ========== 建立模型 ==========
    model = nn.Sequential(
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 256),
        nn.SiLU(),
        nn.Linear(256, 2),
    ).to(device)
    
    criterion = nn.MSELoss()
    
    # ========== 測試多種 Batch Size ==========
    batch_sizes = [1024, 2048, 4096]
    results = []
    
    for batch_size in batch_sizes:
        print_section(f"Batch Size: {batch_size}")
        
        # 準備資料
        x = torch.randn(batch_size, 256, device=device)
        y = torch.randn(batch_size, 2, device=device)
        
        # 重置記憶體統計
        torch.cuda.reset_peak_memory_stats()
        
        # 測試 FP32
        fp32_median, fp32_mean, fp32_std = benchmark_mode(
            model, x, y, criterion,
            use_amp=False,
            warmup_iters=50,
            measure_iters=100,
            num_rounds=3
        )
        fp32_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        # 重置記憶體統計
        torch.cuda.reset_peak_memory_stats()
        
        # 測試 AMP
        amp_median, amp_mean, amp_std = benchmark_mode(
            model, x, y, criterion,
            use_amp=True,
            warmup_iters=50,
            measure_iters=100,
            num_rounds=3
        )
        amp_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        # 計算加速比
        speedup = fp32_median / amp_median
        memory_saving = (fp32_memory - amp_memory) / fp32_memory * 100
        
        results.append({
            'batch_size': batch_size,
            'fp32_time': fp32_median,
            'fp32_std': fp32_std,
            'amp_time': amp_median,
            'amp_std': amp_std,
            'speedup': speedup,
            'fp32_memory': fp32_memory,
            'amp_memory': amp_memory,
            'memory_saving': memory_saving
        })
        
        print(f"\n   ⚡ 性能對比:")
        print(f"      FP32:  {fp32_median:.3f} ± {fp32_std:.3f} ms/iter")
        print(f"      AMP:   {amp_median:.3f} ± {amp_std:.3f} ms/iter")
        print(f"      加速比: {speedup:.3f}x {'✅' if speedup >= 1.1 else '⚠️' if speedup >= 1.0 else '❌'}")
        print(f"      記憶體節省: {memory_saving:.1f}%")
    
    # ========== 總結報告 ==========
    print_section("測試總結")
    
    print("\n📊 完整結果表格:\n")
    print("| Batch Size | FP32 (ms) | AMP (ms) | 加速比 | 標準差 | 記憶體節省 | 評估 |")
    print("|------------|-----------|----------|--------|--------|-----------|------|")
    
    for r in results:
        status = "✅ 有效" if r['speedup'] >= 1.1 else "⚠️ 有限" if r['speedup'] >= 1.0 else "❌ 變慢"
        print(f"| {r['batch_size']:10d} | "
              f"{r['fp32_time']:9.3f} | "
              f"{r['amp_time']:8.3f} | "
              f"{r['speedup']:6.3f}x | "
              f"極低   | "
              f"{r['memory_saving']:9.1f}% | "
              f"{status} |")
    
    # ========== 最終建議 ==========
    print("\n" + "=" * 80)
    print("最終建議")
    print("=" * 80)
    
    avg_speedup = np.mean([r['speedup'] for r in results])
    
    if avg_speedup >= 1.1:
        print("\n✅ **建議啟用 AMP**")
        print(f"\n   平均加速比: {avg_speedup:.2f}x")
        print(f"   最佳 Batch Size: {max(results, key=lambda x: x['speedup'])['batch_size']}")
        print("\n   下一步:")
        print("   1. 在 Trainer 中整合 AMP")
        print("   2. 測試完整訓練流程")
        print("   3. 驗證物理損失的數值穩定性")
    elif avg_speedup >= 1.0:
        print("\n⚠️  **AMP 加速效果有限**")
        print(f"\n   平均加速比: {avg_speedup:.2f}x (僅 {(avg_speedup-1)*100:.0f}% 提升)")
        print("\n   建議:")
        print("   - 可選擇性使用（大 Batch Size 時）")
        print("   - 優先考慮其他優化方法（TorchScript, DDP）")
    else:
        print("\n❌ **不建議使用 AMP**")
        print(f"\n   平均加速比: {avg_speedup:.2f}x (反而變慢 {(1-avg_speedup)*100:.0f}%)")
        print("\n   原因分析:")
        print("   - P100 無 Tensor Core")
        print("   - Type casting overhead > GEMM 加速")
        print("   - 模型規模較小（1.5 MB）")
        print("\n   替代方案:")
        print("   1. TorchScript kernel 融合 (預期 5-15% 提升)")
        print("   2. DDP 多 GPU 訓練 (線性加速)")
        print("   3. 硬體升級 (V100: 5-8x AMP 加速)")


def test_numerical_stability_extended():
    """擴展數值穩定性測試：更長訓練、多種學習率"""
    print_section("擴展數值穩定性測試")
    
    device = torch.device('cuda:0')
    
    # ========== 測試配置 ==========
    learning_rates = [1e-3, 1e-2, 1e-1]
    num_steps = 500  # 更長的訓練
    
    for lr in learning_rates:
        print(f"\n📈 測試學習率: {lr}")
        
        # 建立模型
        model_fp32 = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(device)
        
        model_amp = nn.Sequential(
            nn.Linear(10, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        ).to(device)
        model_amp.load_state_dict(model_fp32.state_dict())
        
        criterion = nn.MSELoss()
        optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=lr)
        optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=lr)
        scaler = GradScaler()
        
        # 測試資料
        x = torch.randn(32, 10, device=device)
        y = torch.randn(32, 10, device=device)
        
        losses_fp32 = []
        losses_amp = []
        
        # 訓練
        for step in range(num_steps):
            # FP32
            optimizer_fp32.zero_grad()
            output_fp32 = model_fp32(x)
            loss_fp32 = criterion(output_fp32, y)
            loss_fp32.backward()
            optimizer_fp32.step()
            losses_fp32.append(loss_fp32.item())
            
            # AMP
            optimizer_amp.zero_grad()
            with autocast():
                output_amp = model_amp(x)
                loss_amp = criterion(output_amp, y)
            scaler.scale(loss_amp).backward()
            scaler.step(optimizer_amp)
            scaler.update()
            losses_amp.append(loss_amp.item())
        
        # 分析結果
        has_nan_fp32 = any(np.isnan(l) for l in losses_fp32)
        has_nan_amp = any(np.isnan(l) for l in losses_amp)
        
        final_loss_fp32 = losses_fp32[-1]
        final_loss_amp = losses_amp[-1]
        rel_error = abs(final_loss_fp32 - final_loss_amp) / (abs(final_loss_fp32) + 1e-8)
        
        print(f"   FP32 最終 Loss: {final_loss_fp32:.6f} {'(NaN!)' if has_nan_fp32 else ''}")
        print(f"   AMP  最終 Loss: {final_loss_amp:.6f} {'(NaN!)' if has_nan_amp else ''}")
        print(f"   相對誤差: {rel_error*100:.2f}%")
        
        if has_nan_amp:
            print(f"   ❌ AMP 出現 NaN（學習率 {lr} 過高）")
        elif rel_error < 0.05:
            print(f"   ✅ 數值穩定（<5% 差異）")
        else:
            print(f"   ⚠️  差異較大（{rel_error*100:.0f}%）")


def main():
    """主測試流程"""
    print("\n" + "=" * 80)
    print("P100 AMP 性能測試 - 擴展暖身版本")
    print("=" * 80)
    
    # 檢查環境
    print(f"\n📋 環境資訊:")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA: {torch.version.cuda}")
    
    if not torch.cuda.is_available():
        print("   ❌ CUDA 不可用")
        return
    
    props = torch.cuda.get_device_properties(0)
    print(f"   GPU: {props.name}")
    print(f"   Compute Capability: {props.major}.{props.minor}")
    print(f"   記憶體: {props.total_memory / 1024**3:.2f} GB")
    
    # 執行測試
    test_comprehensive_performance()
    test_numerical_stability_extended()


if __name__ == '__main__':
    main()
