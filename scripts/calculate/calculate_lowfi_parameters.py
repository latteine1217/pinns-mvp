#!/usr/bin/env python3
"""
Low-Fidelity 參數計算工具
========================

根據 Hi-Fi DNS 參數，計算合適的 Low-Fi 參數

策略：
1. 保持強迫參數不變 (A, k_f)
2. 增加黏滯係數 (α × ν_hifi)，降低雷諾數
3. 粗網格 (N_lowfi = N_hifi / ratio)
4. 確保 CFL 穩定性

作者：PINNs-MVP 團隊
日期：2025-12-10
"""

import numpy as np
import argparse


def calculate_lowfi_params(
    # Hi-Fi 參數
    N_hifi: int,
    nu_hifi: float,
    A: float,
    k_f: int,
    dt_hifi: float,
    L: float = 2 * np.pi,
    # Low-Fi 策略
    alpha_nu: float = 2.0,  # 黏滯倍數
    N_ratio: int = 8,       # 網格粗化比例
):
    """
    計算 Low-Fi 參數
    
    Args:
        N_hifi: Hi-Fi 網格點數
        nu_hifi: Hi-Fi 黏滯係數
        A: 強迫振幅
        k_f: 強迫波數
        dt_hifi: Hi-Fi 時間步長
        L: 域大小
        alpha_nu: 黏滯倍數 (nu_lowfi = alpha_nu × nu_hifi)
        N_ratio: 網格粗化比例 (N_hifi / N_lowfi)
    
    Returns:
        lowfi_params: 字典
    """
    # === 1. Low-Fi 黏滯係數（增加以降低 Re）===
    nu_lowfi = alpha_nu * nu_hifi
    
    # === 2. Low-Fi 網格（粗化）===
    N_lowfi = N_hifi // N_ratio
    
    # === 3. 網格間距 ===
    dx_hifi = L / N_hifi
    dx_lowfi = L / N_lowfi
    
    # === 4. 雷諾數 ===
    Re_hifi = np.sqrt(A) * (L / k_f)**(3/2) / nu_hifi
    Re_lowfi = np.sqrt(A) * (L / k_f)**(3/2) / nu_lowfi
    
    # === 5. CFL 條件估算 ===
    # 估計速度上限（層流 Kolmogorov 解）
    U_max_laminar = A / (nu_hifi * k_f**2)
    
    # Hi-Fi CFL
    CFL_hifi = U_max_laminar * dt_hifi / dx_hifi
    
    # Low-Fi 建議時間步（保持相似 CFL 或稍大）
    # 策略 1: 保持 CFL 不變
    dt_lowfi_cfl_match = CFL_hifi * dx_lowfi / U_max_laminar
    
    # 策略 2: 利用粗網格優勢，適度增大 dt
    # CFL_target = 0.5 (保守)
    CFL_target = 0.5
    dt_lowfi_cfl_safe = CFL_target * dx_lowfi / U_max_laminar
    
    # 取較小者（安全優先）
    dt_lowfi = min(dt_lowfi_cfl_match, dt_lowfi_cfl_safe, dt_hifi * 2.0)
    
    # 最終 CFL
    CFL_lowfi = U_max_laminar * dt_lowfi / dx_lowfi
    
    # === 6. 擴散數 (Diffusion number) ===
    # D = ν × dt / dx² < 0.5 (穩定性)
    D_hifi = nu_hifi * dt_hifi / dx_hifi**2
    D_lowfi = nu_lowfi * dt_lowfi / dx_lowfi**2
    
    return {
        'N_lowfi': N_lowfi,
        'nu_lowfi': nu_lowfi,
        'A': A,
        'k_f': k_f,
        'dt_lowfi': dt_lowfi,
        'L': L,
        'dx_lowfi': dx_lowfi,
        'Re_lowfi': Re_lowfi,
        'CFL_lowfi': CFL_lowfi,
        'D_lowfi': D_lowfi,
        # 比較資訊
        'Re_hifi': Re_hifi,
        'CFL_hifi': CFL_hifi,
        'D_hifi': D_hifi,
        'alpha_nu': alpha_nu,
        'N_ratio': N_ratio,
        'dx_hifi': dx_hifi,
    }


def print_params(params: dict):
    """格式化輸出參數"""
    print("\n" + "=" * 70)
    print("Low-Fidelity 參數計算結果")
    print("=" * 70)
    
    print("\n【Hi-Fi DNS 參數】")
    print(f"  網格: {params['N_ratio'] * params['N_lowfi']} × {params['N_ratio'] * params['N_lowfi']}")
    print(f"  網格間距: dx = {params['dx_hifi']:.6f}")
    print(f"  黏滯係數: ν = {params['nu_lowfi'] / params['alpha_nu']:.6f}")
    print(f"  雷諾數: Re = {params['Re_hifi']:.2f}")
    print(f"  時間步長: dt = {params['dt_lowfi'] / min(1, params['dt_lowfi'] / 0.001):.6f}")
    print(f"  CFL 數: {params['CFL_hifi']:.4f}")
    print(f"  擴散數: D = {params['D_hifi']:.4f}")
    
    print("\n【Low-Fi 建議參數】")
    print(f"  網格: {params['N_lowfi']} × {params['N_lowfi']} (粗化 {params['N_ratio']}×)")
    print(f"  網格間距: dx = {params['dx_lowfi']:.6f} ({params['dx_lowfi'] / params['dx_hifi']:.1f}× 粗)")
    print(f"  黏滯係數: ν = {params['nu_lowfi']:.6f} ({params['alpha_nu']:.1f}× 高)")
    print(f"  雷諾數: Re = {params['Re_lowfi']:.2f} ({params['Re_lowfi'] / params['Re_hifi']:.2f}× 低)")
    print(f"  時間步長: dt = {params['dt_lowfi']:.6f}")
    print(f"  CFL 數: {params['CFL_lowfi']:.4f} {'✅' if params['CFL_lowfi'] < 1.0 else '❌ 不穩定！'}")
    print(f"  擴散數: D = {params['D_lowfi']:.4f} {'✅' if params['D_lowfi'] < 0.5 else '⚠️  偏高'}")
    
    print("\n【強迫參數（保持不變）】")
    print(f"  振幅: A = {params['A']:.4f}")
    print(f"  波數: k_f = {params['k_f']}")
    print(f"  域大小: L = {params['L']:.4f}")
    
    print("\n【生成指令】")
    print(f"python scripts/generate_kolmogorov_lowfi.py \\")
    print(f"    --N {params['N_lowfi']} \\")
    print(f"    --nu {params['nu_lowfi']:.6f} \\")
    print(f"    --A {params['A']} \\")
    print(f"    --k_f {params['k_f']} \\")
    print(f"    --dt {params['dt_lowfi']:.6f} \\")
    print(f"    --T_total 200.0 \\")
    print(f"    --T_spinup 50.0 \\")
    print(f"    --output data/kolmogorov_lowfi/lowfi_N{params['N_lowfi']}_nu{params['nu_lowfi']:.6f}_Re{params['Re_lowfi']:.0f}.h5")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='計算 Low-Fi 參數（基於 Hi-Fi DNS）',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Hi-Fi 參數
    parser.add_argument('--N_hifi', type=int, required=True,
                       help='Hi-Fi 網格點數')
    parser.add_argument('--nu_hifi', type=float, required=True,
                       help='Hi-Fi 黏滯係數')
    parser.add_argument('--A', type=float, default=1.0,
                       help='強迫振幅（預設 1.0）')
    parser.add_argument('--k_f', type=int, default=4,
                       help='強迫波數（預設 4）')
    parser.add_argument('--dt_hifi', type=float, default=0.001,
                       help='Hi-Fi 時間步長（預設 0.001）')
    parser.add_argument('--L', type=float, default=2*np.pi,
                       help='域大小（預設 2π）')
    
    # Low-Fi 策略
    parser.add_argument('--alpha_nu', type=float, default=2.0,
                       help='黏滯倍數（預設 2.0，範圍 2-5）')
    parser.add_argument('--N_ratio', type=int, default=8,
                       help='網格粗化比例（預設 8，即 256→32, 512→64）')
    
    args = parser.parse_args()
    
    # 計算參數
    params = calculate_lowfi_params(
        N_hifi=args.N_hifi,
        nu_hifi=args.nu_hifi,
        A=args.A,
        k_f=args.k_f,
        dt_hifi=args.dt_hifi,
        L=args.L,
        alpha_nu=args.alpha_nu,
        N_ratio=args.N_ratio,
    )
    
    # 輸出
    print_params(params)
    
    # 警告
    if params['CFL_lowfi'] >= 1.0:
        print("\n⚠️  警告：CFL 數過大，可能不穩定！建議減小 dt 或 alpha_nu")
    if params['D_lowfi'] >= 0.5:
        print("\n⚠️  警告：擴散數過大，可能過度耗散！建議減小 dt 或 nu")
    if params['Re_lowfi'] > params['Re_hifi']:
        print("\n❌ 錯誤：Low-Fi Re 高於 Hi-Fi，違背設計目標！請增大 alpha_nu")


if __name__ == '__main__':
    main()
