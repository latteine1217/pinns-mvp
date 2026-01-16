#!/usr/bin/env python3
"""
改進版 P100 AMP 性能測試

改進項目:
1. 更充分的 warmup（50 次 → 確保 GPU 時鐘頻率穩定）
2. GPU 時鐘頻率監控
3. 多輪測試取平均（減少隨機誤差）
4. 更大的 batch size（更接近實際訓練）
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import subprocess
import sys

def get_gpu_clocks():
    """獲取 GPU 時鐘頻率"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=clocks.sm,clocks.mem', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        sm_clock, mem_clock = result.stdout.strip().split(', ')
        return int(sm_clock), int(mem_clock)
    except:
        return None, None


def warmup_gpu(duration_sec=5):
    """充分暖身 GPU，確保時鐘頻率達到最大值"""
    print(f"\n🔥 GPU 暖身中（{duration_sec} 秒）...")
    
    device = torch.device('cuda:0')
    
    # 大量計算讓 GPU 升頻
    A = torch.randn(4096, 4096, device=device)
    B = torch.randn(4096, 4096, device=device)
    
    # 獲取初始時鐘頻率
    sm_before, mem_before = get_gpu_clocks()
    if sm_before:
        print(f"   初始時鐘: SM={sm_before} MHz, Memory={mem_before} MHz")
    
    start = time.time()
    iteration = 0
    while time.time() - start < duration_sec:
        C = torch.mm(A, B)
        torch.cuda.synchronize()
        iteration += 1
    
    # 獲取暖身後時鐘頻率
    sm_after, mem_after = get_gpu_clocks()
    if sm_after:
        print(f"   暖身後時鐘: SM={sm_after} MHz, Memory={mem_after} MHz")
        if sm_after > sm_before:
            print(f"   ✅ GPU 已升頻（+{sm_after - sm_before} MHz）")
        else:
            print(f"   ⚠️  GPU 時鐘未變化（可能已在最高頻率）")
    
    print(f"   暖身迭代: {iteration} 次")
    del A, B, C
    torch.cuda.empty_cache()


def test_amp_performance_improved():
    """改進版性能測試"""
    print("\n" + "=" * 80)
    print("改進版 AMP 性能測試")
    print("=" * 80)
    
    device = torch.device('cuda:0')
    
    # 測試配置
    configs = [
        {'batch_size': 1024, 'name': '小 Batch (1024)'},
        {'batch_size': 4096, 'name': '大 Batch (4096)'},
    ]
    
    # 模型（8×256 MLP）
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
    scaler = GradScaler()
    
    # 全域暖身
    warmup_gpu(duration_sec=5)
    
    results = []
    
    for config in configs:
        batch_size = config['batch_size']
        print(f"\n{'=' * 80}")
        print(f"測試配置: {config['name']}")
        print(f"{'=' * 80}")
        
        # 測試資料
        x = torch.randn(batch_size, 256, device=device)
        y = torch.randn(batch_size, 2, device=device)
        
        # 測試參數
        num_rounds = 3  # 多輪測試
        warmup_iters = 20  # 每輪 warmup
        measure_iters = 50  # 測量迭代數
        
        fp32_times = []
        fp16_times = []
        
        for round_idx in range(num_rounds):
            print(f"\n🔄 第 {round_idx + 1}/{num_rounds} 輪測試")
            
            # === FP32 測試 ===
            print(f"   🔹 FP32 暖身中...")
            model.zero_grad()
            torch.cuda.synchronize()
            
            # Warmup
            for i in range(warmup_iters):
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            
            # 測量
            print(f"   🔹 FP32 測量中...")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            
            for i in range(measure_iters):
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            fp32_time = (time.time() - t0) / measure_iters * 1000  # ms
            fp32_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            fp32_times.append(fp32_time)
            
            print(f"      平均時間: {fp32_time:.3f} ms/iter")
            print(f"      記憶體: {fp32_mem:.2f} MB")
            
            # === FP16 (AMP) 測試 ===
            print(f"   🔹 FP16 暖身中...")
            model.zero_grad()
            torch.cuda.synchronize()
            
            # Warmup
            for i in range(warmup_iters):
                with autocast():
                    output = model(x)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
                scaler.update()
                model.zero_grad()
            
            torch.cuda.synchronize()
            
            # 測量
            print(f"   🔹 FP16 測量中...")
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            
            for i in range(measure_iters):
                with autocast():
                    output = model(x)
                    loss = criterion(output, y)
                scaler.scale(loss).backward()
                model.zero_grad()
            
            torch.cuda.synchronize()
            fp16_time = (time.time() - t0) / measure_iters * 1000  # ms
            fp16_mem = torch.cuda.max_memory_allocated() / 1024**2  # MB
            fp16_times.append(fp16_time)
            
            print(f"      平均時間: {fp16_time:.3f} ms/iter")
            print(f"      記憶體: {fp16_mem:.2f} MB")
        
        # 計算統計
        fp32_mean = sum(fp32_times) / len(fp32_times)
        fp32_std = (sum((t - fp32_mean)**2 for t in fp32_times) / len(fp32_times))**0.5
        
        fp16_mean = sum(fp16_times) / len(fp16_times)
        fp16_std = (sum((t - fp16_mean)**2 for t in fp16_times) / len(fp16_times))**0.5
        
        speedup = fp32_mean / fp16_mean
        
        print(f"\n📊 統計結果（{num_rounds} 輪平均）:")
        print(f"   FP32: {fp32_mean:.3f} ± {fp32_std:.3f} ms/iter")
        print(f"   FP16: {fp16_mean:.3f} ± {fp16_std:.3f} ms/iter")
        print(f"   加速比: {speedup:.3f}x")
        print(f"   記憶體節省: {(1 - fp16_mem/fp32_mem)*100:.1f}%")
        
        results.append({
            'config': config['name'],
            'batch_size': batch_size,
            'fp32_mean': fp32_mean,
            'fp32_std': fp32_std,
            'fp16_mean': fp16_mean,
            'fp16_std': fp16_std,
            'speedup': speedup,
            'fp32_mem': fp32_mem,
            'fp16_mem': fp16_mem,
        })
    
    # 最終總結
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)
    
    print(f"\n{'配置':<20} {'FP32 (ms)':<15} {'FP16 (ms)':<15} {'加速比':<10} {'結論'}")
    print("-" * 80)
    for r in results:
        conclusion = "✅ 有效" if r['speedup'] > 1.1 else "❌ 無效" if r['speedup'] < 0.95 else "⚠️  持平"
        print(f"{r['config']:<20} {r['fp32_mean']:>6.3f} ± {r['fp32_std']:>4.3f}   {r['fp16_mean']:>6.3f} ± {r['fp16_std']:>4.3f}   {r['speedup']:>6.3f}x    {conclusion}")
    
    # 最終建議
    print("\n" + "=" * 80)
    print("建議")
    print("=" * 80)
    
    best_speedup = max(r['speedup'] for r in results)
    
    if best_speedup > 1.2:
        print(f"\n✅ P100 在某些配置下可從 AMP 獲益（最高 {best_speedup:.2f}x）")
        best_config = [r for r in results if r['speedup'] == best_speedup][0]
        print(f"   最佳配置: {best_config['config']} (Batch={best_config['batch_size']})")
        print(f"\n   建議:")
        print(f"   - 在訓練中使用較大 batch size")
        print(f"   - 啟用 AMP 以獲得 {(best_speedup-1)*100:.0f}% 加速")
    elif best_speedup > 1.05:
        print(f"\n⚠️  P100 的 AMP 加速有限（最高 {best_speedup:.2f}x）")
        print(f"   建議:")
        print(f"   - AMP 可作為可選優化（非必需）")
        print(f"   - 優先考慮其他優化方法")
    else:
        print(f"\n❌ P100 不適合 AMP（最高加速比 {best_speedup:.2f}x < 1.05）")
        print(f"   建議:")
        print(f"   - 不在訓練中使用 AMP")
        print(f"   - 專注於其他優化方向")


def main():
    """主測試流程"""
    print("\n" + "=" * 80)
    print("P100 AMP 性能測試（改進版）")
    print("=" * 80 + "\n")
    
    # 檢查環境
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        sys.exit(1)
    
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    # 執行改進版測試
    test_amp_performance_improved()


if __name__ == '__main__':
    main()
