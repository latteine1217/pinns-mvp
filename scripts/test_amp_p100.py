#!/usr/bin/env python3
"""
測試 P100 GPU 是否支援混合精度訓練 (AMP)

測試項目:
1. 基礎 AMP 功能
2. FP16 GEMM 性能
3. 梯度縮放 (GradScaler)
4. 數值穩定性
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import sys

def test_amp_availability():
    """測試 1: 檢查 AMP 是否可用"""
    print("=" * 80)
    print("測試 1: AMP 可用性檢查")
    print("=" * 80)
    
    # 檢查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 版本: {torch.version.cuda}")
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(0)
    print(f"\nGPU 資訊:")
    print(f"  名稱: {props.name}")
    print(f"  Compute Capability: {props.major}.{props.minor}")
    print(f"  記憶體: {props.total_memory / 1024**3:.2f} GB")
    
    # 檢查是否支援 FP16
    if props.major >= 6:
        print(f"  ✅ 支援 FP16 (Compute Capability >= 6.0)")
    else:
        print(f"  ❌ 不支援 FP16 (Compute Capability < 6.0)")
        return False
    
    # 檢查 AMP 模組
    try:
        from torch.cuda.amp import autocast, GradScaler
        print(f"  ✅ torch.cuda.amp 可用")
        return True
    except ImportError:
        print(f"  ❌ torch.cuda.amp 不可用 (需要 PyTorch >= 1.6)")
        return False


def test_amp_basic():
    """測試 2: 基礎 AMP 功能"""
    print("\n" + "=" * 80)
    print("測試 2: 基礎 AMP 功能")
    print("=" * 80)
    
    device = torch.device('cuda:0')
    
    # 簡單模型
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # 測試資料
    x = torch.randn(64, 256, device=device)
    y = torch.randn(64, 256, device=device)
    
    try:
        # Forward with autocast
        optimizer.zero_grad()
        with autocast():
            output = model(x)
            loss = criterion(output, y)
        
        # Backward with scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        print(f"✅ AMP forward/backward 成功")
        print(f"   Loss: {loss.item():.6f}")
        print(f"   Loss dtype: {loss.dtype}")
        return True
        
    except Exception as e:
        print(f"❌ AMP 執行失敗: {e}")
        return False


def test_amp_performance():
    """測試 3: AMP 性能對比"""
    print("\n" + "=" * 80)
    print("測試 3: FP32 vs FP16 性能對比")
    print("=" * 80)
    
    device = torch.device('cuda:0')
    
    # 較大的模型（模擬 8×256 MLP）
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
    
    # 測試資料
    x = torch.randn(1024, 256, device=device)
    y = torch.randn(1024, 2, device=device)
    
    num_iters = 50
    warmup = 10
    
    # FP32 測試
    print("\n🔹 FP32 模式:")
    model.zero_grad()
    torch.cuda.synchronize()
    
    for i in range(warmup):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    for i in range(num_iters):
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    fp32_time = (time.time() - t0) / num_iters * 1000  # ms
    
    print(f"   平均時間: {fp32_time:.3f} ms/iter")
    print(f"   記憶體使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # 重置記憶體統計
    torch.cuda.reset_peak_memory_stats()
    
    # FP16 (AMP) 測試
    print("\n🔹 FP16 (AMP) 模式:")
    model.zero_grad()
    torch.cuda.synchronize()
    
    for i in range(warmup):
        with autocast():
            output = model(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()
        model.zero_grad()
    
    torch.cuda.synchronize()
    t0 = time.time()
    
    for i in range(num_iters):
        with autocast():
            output = model(x)
            loss = criterion(output, y)
        scaler.scale(loss).backward()
        model.zero_grad()
    
    torch.cuda.synchronize()
    fp16_time = (time.time() - t0) / num_iters * 1000  # ms
    
    print(f"   平均時間: {fp16_time:.3f} ms/iter")
    print(f"   記憶體使用: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
    
    # 計算加速比
    speedup = fp32_time / fp16_time
    memory_saving = (1 - torch.cuda.max_memory_allocated() / (torch.cuda.max_memory_allocated() + 1e-8)) * 100
    
    print(f"\n📊 性能對比:")
    print(f"   加速比: {speedup:.2f}x")
    if speedup > 1.2:
        print(f"   ✅ AMP 在 P100 上有效（加速 >{speedup-1:.0%}）")
    elif speedup > 1.0:
        print(f"   ⚠️  AMP 加速有限（僅 {speedup-1:.0%}）")
    else:
        print(f"   ❌ AMP 反而變慢")
    
    return speedup > 1.0


def test_numerical_stability():
    """測試 4: 數值穩定性"""
    print("\n" + "=" * 80)
    print("測試 4: 數值穩定性檢查")
    print("=" * 80)
    
    device = torch.device('cuda:0')
    
    # 測試模型
    model_fp32 = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    ).to(device)
    
    # 複製權重
    model_amp = nn.Sequential(
        nn.Linear(10, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    ).to(device)
    model_amp.load_state_dict(model_fp32.state_dict())
    
    criterion = nn.MSELoss()
    optimizer_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=0.001)
    optimizer_amp = torch.optim.Adam(model_amp.parameters(), lr=0.001)
    scaler = GradScaler()
    
    # 測試資料
    x = torch.randn(32, 10, device=device)
    y = torch.randn(32, 10, device=device)
    
    losses_fp32 = []
    losses_amp = []
    
    # 訓練 100 步
    num_steps = 100
    for step in range(num_steps):
        # FP32 訓練
        optimizer_fp32.zero_grad()
        output_fp32 = model_fp32(x)
        loss_fp32 = criterion(output_fp32, y)
        loss_fp32.backward()
        optimizer_fp32.step()
        losses_fp32.append(loss_fp32.item())
        
        # AMP 訓練
        optimizer_amp.zero_grad()
        with autocast():
            output_amp = model_amp(x)
            loss_amp = criterion(output_amp, y)
        scaler.scale(loss_amp).backward()
        scaler.step(optimizer_amp)
        scaler.update()
        losses_amp.append(loss_amp.item())
    
    # 檢查 NaN/Inf
    has_nan_fp32 = any(torch.isnan(torch.tensor(l)) for l in losses_fp32)
    has_nan_amp = any(torch.isnan(torch.tensor(l)) for l in losses_amp)
    has_inf_fp32 = any(torch.isinf(torch.tensor(l)) for l in losses_fp32)
    has_inf_amp = any(torch.isinf(torch.tensor(l)) for l in losses_amp)
    
    print(f"\n📈 訓練 {num_steps} 步:")
    print(f"   FP32 最終 Loss: {losses_fp32[-1]:.6f}")
    print(f"   AMP  最終 Loss: {losses_amp[-1]:.6f}")
    print(f"   Loss 差異: {abs(losses_fp32[-1] - losses_amp[-1]):.6f}")
    
    print(f"\n🔍 數值穩定性:")
    if has_nan_fp32 or has_inf_fp32:
        print(f"   ⚠️  FP32 出現 NaN/Inf")
    else:
        print(f"   ✅ FP32 數值穩定")
    
    if has_nan_amp or has_inf_amp:
        print(f"   ❌ AMP 出現 NaN/Inf")
        return False
    else:
        print(f"   ✅ AMP 數值穩定")
    
    # 檢查相對誤差
    rel_error = abs(losses_fp32[-1] - losses_amp[-1]) / (abs(losses_fp32[-1]) + 1e-8)
    print(f"   相對誤差: {rel_error*100:.2f}%")
    
    if rel_error < 0.05:
        print(f"   ✅ AMP 與 FP32 結果一致（<5% 差異）")
        return True
    else:
        print(f"   ⚠️  AMP 與 FP32 差異較大（>{rel_error*100:.0f}%）")
        return False


def main():
    """主測試流程"""
    print("\n" + "=" * 80)
    print("P100 GPU 混合精度訓練 (AMP) 測試")
    print("=" * 80 + "\n")
    
    results = {}
    
    # 測試 1: AMP 可用性
    results['availability'] = test_amp_availability()
    if not results['availability']:
        print("\n❌ AMP 不可用，終止測試")
        sys.exit(1)
    
    # 測試 2: 基礎功能
    results['basic'] = test_amp_basic()
    
    # 測試 3: 性能
    results['performance'] = test_amp_performance()
    
    # 測試 4: 數值穩定性
    results['stability'] = test_numerical_stability()
    
    # 總結
    print("\n" + "=" * 80)
    print("測試總結")
    print("=" * 80)
    
    print(f"\n✅ AMP 可用性: {'通過' if results['availability'] else '失敗'}")
    print(f"✅ 基礎功能: {'通過' if results['basic'] else '失敗'}")
    print(f"{'✅' if results['performance'] else '❌'} 性能提升: {'有效' if results['performance'] else '無效'}")
    print(f"{'✅' if results['stability'] else '⚠️ '} 數值穩定性: {'穩定' if results['stability'] else '需注意'}")
    
    # 最終建議
    print("\n" + "=" * 80)
    print("建議")
    print("=" * 80)
    
    if all(results.values()):
        print("\n✅ P100 完全支援 AMP，建議在訓練中啟用！")
        print("\n   預期效果:")
        print("   - 訓練速度提升 20-50%")
        print("   - 記憶體使用減少 30-40%")
        print("   - 數值結果與 FP32 一致")
        print("\n   下一步:")
        print("   1. 修改 trainer.py 加入 AMP 支援")
        print("   2. 在完整訓練中測試 100 epochs")
        print("   3. 對比 Loss 曲線和最終精度")
    elif results['availability'] and results['basic']:
        print("\n⚠️  P100 支援 AMP，但性能或穩定性有疑慮")
        print("\n   建議:")
        print("   - 在小規模實驗中測試")
        print("   - 物理損失使用 FP32")
        print("   - 加入 Gradient Clipping")
    else:
        print("\n❌ P100 不支援 AMP 或功能異常")
        print("\n   建議使用其他優化方法")


if __name__ == '__main__':
    main()
