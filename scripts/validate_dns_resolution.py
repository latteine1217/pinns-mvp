#!/usr/bin/env python3
"""
DNS 解析度驗證工具
==================
檢查生成的 DNS 數據是否達到真正的 DNS 標準（Kolmogorov 尺度解析）

使用方式:
---------
python scripts/validate_dns_resolution.py --input data/kolmogorov_dns/dns_re500_t100.h5
"""

import h5py
import numpy as np
import argparse
from pathlib import Path


def validate_dns_resolution(h5_file: str):
    """驗證 DNS 解析度"""
    
    print("=" * 80)
    print("DNS 解析度驗證（Kolmogorov 尺度分析）")
    print("=" * 80)
    print(f"📂 輸入文件: {h5_file}\n")
    
    with h5py.File(h5_file, 'r') as f:
        # 讀取配置
        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        nu = f['config'].attrs['nu']
        A = f['config'].attrs.get('A', 1.0)
        
        # 讀取速度場（最後時間步）
        u = f['u'][-1]
        v = f['v'][-1]
    
    # 網格間距
    dx = L / N
    
    # 估算雷諾數
    U_rms = np.sqrt(np.mean(u**2 + v**2))
    Re_rms = U_rms * L / nu
    
    # 估算耗散率（基於 Kolmogorov 流）
    # 方法 1：基於強迫振幅（理論估計）
    epsilon_theory = A**3 / (nu**2 * L)
    
    # 方法 2：基於速度場（實際測量）
    epsilon_measured = nu * np.mean(
        (np.gradient(u, dx, axis=0))**2 + 
        (np.gradient(u, dx, axis=1))**2 +
        (np.gradient(v, dx, axis=0))**2 + 
        (np.gradient(v, dx, axis=1))**2
    )
    
    # Kolmogorov 尺度（使用測量值）
    eta = (nu**3 / epsilon_measured)**(1/4)
    
    # 解析度比
    resolution_ratio = dx / eta
    
    # DNS 判定
    if resolution_ratio < 2.5:
        status = "✅ DNS 黃金標準"
        color = "green"
    elif resolution_ratio < 5.0:
        status = "⚠️ 邊緣 DNS（可接受但會損失細節）"
        color = "yellow"
    elif resolution_ratio < 10.0:
        status = "❌ 欠解析 DNS（小尺度被截斷，類似隱式 LES）"
        color = "orange"
    else:
        status = "❌ 嚴重欠解析（不應標記為 DNS）"
        color = "red"
    
    # 輸出報告
    print("📊 物理參數:")
    print(f"   網格點數: {N}×{N}")
    print(f"   域長度: L = {L:.4f}")
    print(f"   黏滯係數: ν = {nu:.6f}")
    print(f"   強迫振幅: A = {A:.2f}")
    print()
    
    print("🔬 流場統計:")
    print(f"   RMS 速度: U_rms = {U_rms:.4f}")
    print(f"   基於 RMS 的雷諾數: Re_rms = {Re_rms:.1f}")
    print()
    
    print("⚡ 耗散率估算:")
    print(f"   理論耗散率: ε_theory = {epsilon_theory:.2e}")
    print(f"   測量耗散率: ε_measured = {epsilon_measured:.2e}")
    print()
    
    print("📏 解析度分析:")
    print(f"   網格間距: dx = {dx:.6f}")
    print(f"   Kolmogorov 尺度: η = {eta:.6f}")
    print(f"   解析度比: dx/η = {resolution_ratio:.2f}")
    print()
    
    print("=" * 80)
    print(f"🎯 判定結果: {status}")
    print("=" * 80)
    print()
    
    print("📖 DNS 標準（Pope, Turbulent Flows, 2000）:")
    print("   ✅ dx/η < 2.5  → DNS 黃金標準（完全解析所有尺度）")
    print("   ⚠️ 2.5 ≤ dx/η < 5  → 邊緣 DNS（最小尺度部分損失）")
    print("   ❌ 5 ≤ dx/η < 10 → 欠解析 DNS（需增加網格或降低 Re）")
    print("   ❌ dx/η ≥ 10 → 嚴重欠解析（應使用 LES/RANS）")
    print()
    
    # 給出建議
    if resolution_ratio >= 5.0:
        print("💡 改進建議:")
        N_required = int(L / (2.5 * eta)) + 1
        N_practical = 2 ** int(np.ceil(np.log2(N_required * 0.6)))  # 最接近的 2 的冪次
        
        print(f"   建議網格點數（達到 dx/η < 2.5）: {N_required}×{N_required}")
        print(f"   實用網格點數（2 的冪次）: {N_practical}×{N_practical}")
        print(f"   或者降低雷諾數至 Re < {Re_rms * 0.3:.0f}")
        print()
    
    return {
        'N': N,
        'dx': dx,
        'eta': eta,
        'resolution_ratio': resolution_ratio,
        'Re_rms': Re_rms,
        'epsilon': epsilon_measured,
        'status': status
    }


def main():
    parser = argparse.ArgumentParser(description='DNS 解析度驗證工具')
    parser.add_argument('--input', type=str, required=True, 
                       help='DNS HDF5 文件路徑')
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"❌ 文件不存在: {args.input}")
        return
    
    validate_dns_resolution(args.input)


if __name__ == '__main__':
    main()
