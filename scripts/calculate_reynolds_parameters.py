#!/usr/bin/env python3
"""
Kolmogorov Flow 雷諾數與物理參數計算工具
=======================================

根據 Musacchio & Boffetta (2014) 定義計算雷諾數與相關物理參數：
    Re = √f₀ × L^(3/2) / ν
    其中 L = 2π/k

功能：
1. 給定 Re，求所需的 ν 或 f₀
2. 給定 ν、f₀、k，計算 Re
3. 批量生成參數表格
4. 參數掃描與可視化

作者：PINNs-MVP 團隊
日期：2025-11-25
"""

import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys


# =============================================================================
# 核心計算函數
# =============================================================================

def compute_reynolds(f0: float, nu: float, k_f: float) -> float:
    """
    計算雷諾數（Musacchio & Boffetta 2014 定義）

    Re = √f₀ × L^(3/2) / ν，其中 L = 2π/k

    Args:
        f0: 強迫振幅
        nu: 動力黏度
        k_f: 強迫波數

    Returns:
        Re: 雷諾數
    """
    L = 2.0 * np.pi / k_f
    Re = np.sqrt(f0) * (L ** 1.5) / nu
    return Re


def compute_nu_from_Re(Re: float, f0: float, k_f: float) -> float:
    """
    給定 Re、f₀、k，計算所需的 ν

    由 Re = √f₀ × L^(3/2) / ν 推導：
    ν = √f₀ × L^(3/2) / Re

    Args:
        Re: 目標雷諾數
        f0: 強迫振幅
        k_f: 強迫波數

    Returns:
        nu: 所需的動力黏度
    """
    L = 2.0 * np.pi / k_f
    nu = np.sqrt(f0) * (L ** 1.5) / Re
    return nu


def compute_f0_from_Re(Re: float, nu: float, k_f: float) -> float:
    """
    給定 Re、ν、k，計算所需的 f₀

    由 Re = √f₀ × L^(3/2) / ν 推導：
    f₀ = (Re × ν / L^(3/2))²

    Args:
        Re: 目標雷諾數
        nu: 動力黏度
        k_f: 強迫波數

    Returns:
        f0: 所需的強迫振幅
    """
    L = 2.0 * np.pi / k_f
    f0 = ((Re * nu) / (L ** 1.5)) ** 2
    return f0


def compute_laminar_velocity(f0: float, nu: float, k_f: float) -> float:
    """
    計算層流解的特徵速度

    U₀ = f₀ / (ν k²)

    Args:
        f0: 強迫振幅
        nu: 動力黏度
        k_f: 強迫波數

    Returns:
        U0: 層流速度
    """
    return f0 / (nu * k_f**2)


def compute_forcing_wavelength(k_f: float) -> float:
    """
    計算強迫波長

    L = 2π / k

    Args:
        k_f: 強迫波數

    Returns:
        L: 波長
    """
    return 2.0 * np.pi / k_f


def classify_flow_regime(Re: float) -> Tuple[str, str]:
    """
    根據雷諾數分類流動狀態

    Args:
        Re: 雷諾數

    Returns:
        regime: 流動狀態名稱
        description: 詳細描述
    """
    if Re < 30:
        return "層流/弱不穩定", "流場主要為穩態層流解，少量時間波動"
    elif Re < 100:
        return "過渡/弱湍流", "出現渦結構，時間波動增強，開始出現能量級串"
    elif Re < 200:
        return "湍流", "明顯的逆能量級串，大尺度渦結構形成"
    else:
        return "強湍流", "充分發展的 2D 湍流，複雜的多尺度相互作用"


# =============================================================================
# 交互式計算模式
# =============================================================================

def mode_compute_Re():
    """模式 1：計算雷諾數"""
    print("\n" + "="*80)
    print("模式 1: 計算雷諾數 (給定 f₀, ν, k)")
    print("="*80)

    f0 = float(input("輸入強迫振幅 f₀ (默認 1.0): ") or "1.0")
    nu = float(input("輸入動力黏度 ν (默認 0.0125): ") or "0.0125")
    k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")

    Re = compute_reynolds(f0, nu, k_f)
    U0 = compute_laminar_velocity(f0, nu, k_f)
    L = compute_forcing_wavelength(k_f)
    regime, desc = classify_flow_regime(Re)

    print(f"\n{'='*80}")
    print("計算結果")
    print("="*80)
    print(f"輸入參數:")
    print(f"  強迫振幅 f₀ = {f0}")
    print(f"  動力黏度 ν = {nu}")
    print(f"  強迫波數 k = {k_f}")
    print(f"\n計算結果:")
    print(f"  雷諾數 Re = {Re:.2f}")
    print(f"  層流速度 U₀ = {U0:.4f}")
    print(f"  強迫波長 L = {L:.4f}")
    print(f"\n流動狀態:")
    print(f"  {regime}")
    print(f"  {desc}")
    print("="*80)


def mode_compute_nu():
    """模式 2：計算所需的 ν"""
    print("\n" + "="*80)
    print("模式 2: 計算所需的 ν (給定目標 Re, f₀, k)")
    print("="*80)

    Re_target = float(input("輸入目標雷諾數 Re (默認 100): ") or "100")
    f0 = float(input("輸入強迫振幅 f₀ (默認 1.0): ") or "1.0")
    k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")

    nu = compute_nu_from_Re(Re_target, f0, k_f)
    Re_verify = compute_reynolds(f0, nu, k_f)
    U0 = compute_laminar_velocity(f0, nu, k_f)
    L = compute_forcing_wavelength(k_f)

    print(f"\n{'='*80}")
    print("計算結果")
    print("="*80)
    print(f"輸入參數:")
    print(f"  目標雷諾數 Re = {Re_target:.2f}")
    print(f"  強迫振幅 f₀ = {f0}")
    print(f"  強迫波數 k = {k_f}")
    print(f"\n計算結果:")
    print(f"  所需動力黏度 ν = {nu:.6f}")
    print(f"  驗證雷諾數 Re = {Re_verify:.2f} (應等於 {Re_target:.2f})")
    print(f"  層流速度 U₀ = {U0:.4f}")
    print(f"  強迫波長 L = {L:.4f}")
    print(f"\n配置文件建議:")
    print(f"  physics:")
    print(f"    nu: {nu:.6f}")
    print(f"    forcing:")
    print(f"      amplitude: {f0}")
    print(f"      k_f: {k_f}")
    print("="*80)


def mode_compute_f0():
    """模式 3：計算所需的 f₀"""
    print("\n" + "="*80)
    print("模式 3: 計算所需的 f₀ (給定目標 Re, ν, k)")
    print("="*80)

    Re_target = float(input("輸入目標雷諾數 Re (默認 100): ") or "100")
    nu = float(input("輸入動力黏度 ν (默認 0.0125): ") or "0.0125")
    k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")

    f0 = compute_f0_from_Re(Re_target, nu, k_f)
    Re_verify = compute_reynolds(f0, nu, k_f)
    U0 = compute_laminar_velocity(f0, nu, k_f)
    L = compute_forcing_wavelength(k_f)

    print(f"\n{'='*80}")
    print("計算結果")
    print("="*80)
    print(f"輸入參數:")
    print(f"  目標雷諾數 Re = {Re_target:.2f}")
    print(f"  動力黏度 ν = {nu}")
    print(f"  強迫波數 k = {k_f}")
    print(f"\n計算結果:")
    print(f"  所需強迫振幅 f₀ = {f0:.6f}")
    print(f"  驗證雷諾數 Re = {Re_verify:.2f} (應等於 {Re_target:.2f})")
    print(f"  層流速度 U₀ = {U0:.4f}")
    print(f"  強迫波長 L = {L:.4f}")
    print(f"\n配置文件建議:")
    print(f"  physics:")
    print(f"    nu: {nu}")
    print(f"    forcing:")
    print(f"      amplitude: {f0:.6f}")
    print(f"      k_f: {k_f}")
    print("="*80)


def mode_parameter_table():
    """模式 4：生成參數對照表"""
    print("\n" + "="*80)
    print("模式 4: 生成參數對照表")
    print("="*80)

    print("\n選擇生成模式:")
    print("  1. 固定 f₀, k，掃描 ν → Re")
    print("  2. 固定 ν, k，掃描 f₀ → Re")
    print("  3. 固定 f₀, ν，掃描 k → Re")
    print("  4. 固定 k，掃描 ν 與 f₀ → Re (2D 網格)")

    mode = input("選擇模式 (1-4): ") or "1"

    if mode == "1":
        f0 = float(input("輸入強迫振幅 f₀ (默認 1.0): ") or "1.0")
        k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")
        nu_values = np.array([0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025])

        print(f"\n{'='*80}")
        print(f"參數對照表 (f₀={f0}, k={k_f})")
        print("="*80)
        print(f"{'ν':<12} {'Re':<12} {'U₀':<12} {'流動狀態':<20}")
        print("-"*80)

        for nu in nu_values:
            Re = compute_reynolds(f0, nu, k_f)
            U0 = compute_laminar_velocity(f0, nu, k_f)
            regime, _ = classify_flow_regime(Re)
            print(f"{nu:<12.6f} {Re:<12.2f} {U0:<12.4f} {regime:<20}")

    elif mode == "2":
        nu = float(input("輸入動力黏度 ν (默認 0.0125): ") or "0.0125")
        k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")
        f0_values = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0])

        print(f"\n{'='*80}")
        print(f"參數對照表 (ν={nu}, k={k_f})")
        print("="*80)
        print(f"{'f₀':<12} {'Re':<12} {'U₀':<12} {'流動狀態':<20}")
        print("-"*80)

        for f0 in f0_values:
            Re = compute_reynolds(f0, nu, k_f)
            U0 = compute_laminar_velocity(f0, nu, k_f)
            regime, _ = classify_flow_regime(Re)
            print(f"{f0:<12.2f} {Re:<12.2f} {U0:<12.4f} {regime:<20}")

    elif mode == "3":
        f0 = float(input("輸入強迫振幅 f₀ (默認 1.0): ") or "1.0")
        nu = float(input("輸入動力黏度 ν (默認 0.0125): ") or "0.0125")
        k_values = np.array([2, 4, 6, 8, 10, 12, 16])

        print(f"\n{'='*80}")
        print(f"參數對照表 (f₀={f0}, ν={nu})")
        print("="*80)
        print(f"{'k':<12} {'Re':<12} {'L':<12} {'U₀':<12} {'流動狀態':<20}")
        print("-"*80)

        for k_f in k_values:
            Re = compute_reynolds(f0, nu, k_f)
            L = compute_forcing_wavelength(k_f)
            U0 = compute_laminar_velocity(f0, nu, k_f)
            regime, _ = classify_flow_regime(Re)
            print(f"{k_f:<12.0f} {Re:<12.2f} {L:<12.4f} {U0:<12.4f} {regime:<20}")

    elif mode == "4":
        k_f = float(input("輸入強迫波數 k (默認 8): ") or "8")
        nu_values = np.array([0.005, 0.01, 0.0125, 0.015, 0.02])
        f0_values = np.array([0.5, 1.0, 2.0, 3.0, 4.0])

        print(f"\n{'='*80}")
        print(f"參數對照表 2D (k={k_f})")
        print("="*80)
        header = "ν \\ f₀"
        print(f"{header:<12}", end="")
        for f0 in f0_values:
            print(f"{f0:<12.1f}", end="")
        print()
        print("-"*80)

        for nu in nu_values:
            print(f"{nu:<12.6f}", end="")
            for f0 in f0_values:
                Re = compute_reynolds(f0, nu, k_f)
                print(f"{Re:<12.2f}", end="")
            print()

    print("="*80)


# =============================================================================
# 命令行模式
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kolmogorov Flow 雷諾數與物理參數計算工具 (Musacchio & Boffetta 2014)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例用法:

  # 計算雷諾數
  python %(prog)s --f0 1.0 --nu 0.0125 --k 8

  # 計算所需的 ν（給定目標 Re=100）
  python %(prog)s --target-Re 100 --f0 1.0 --k 8 --solve-nu

  # 計算所需的 f₀（給定目標 Re=100）
  python %(prog)s --target-Re 100 --nu 0.0125 --k 8 --solve-f0

  # 交互式模式
  python %(prog)s --interactive

  # 批量計算（掃描 ν）
  python %(prog)s --f0 1.0 --k 8 --nu-range 0.005 0.025 0.005

公式（Musacchio & Boffetta 2014）:
  Re = √f₀ × L^(3/2) / ν，其中 L = 2π/k

參考文獻:
  - Musacchio & Boffetta (2014), Phys. Rev. E
  - Shebalin (2013), Physics of Fluids
        """
    )

    # 基本參數
    parser.add_argument('--f0', type=float, help='強迫振幅（默認 1.0）')
    parser.add_argument('--nu', type=float, help='動力黏度（默認 0.0125）')
    parser.add_argument('--k', '--k-f', type=float, dest='k_f', help='強迫波數（默認 8）')

    # 求解模式
    parser.add_argument('--target-Re', type=float, help='目標雷諾數（用於反推 ν 或 f₀）')
    parser.add_argument('--solve-nu', action='store_true', help='求解所需的 ν')
    parser.add_argument('--solve-f0', action='store_true', help='求解所需的 f₀')

    # 掃描模式
    parser.add_argument('--nu-range', nargs=3, type=float, metavar=('START', 'END', 'STEP'),
                       help='掃描 ν 範圍（起始 結束 步長）')
    parser.add_argument('--f0-range', nargs=3, type=float, metavar=('START', 'END', 'STEP'),
                       help='掃描 f₀ 範圍（起始 結束 步長）')
    parser.add_argument('--k-range', nargs=3, type=float, metavar=('START', 'END', 'STEP'),
                       help='掃描 k 範圍（起始 結束 步長）')

    # 交互式模式
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='交互式模式（選單驅動）')

    # 輸出選項
    parser.add_argument('--output', '-o', type=str, help='輸出文件路徑（CSV 格式）')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出')

    args = parser.parse_args()

    # 交互式模式
    if args.interactive or len(sys.argv) == 1:
        print("="*80)
        print("Kolmogorov Flow 雷諾數計算工具")
        print("="*80)
        print("\n選擇計算模式:")
        print("  1. 計算雷諾數 (給定 f₀, ν, k)")
        print("  2. 計算所需的 ν (給定目標 Re, f₀, k)")
        print("  3. 計算所需的 f₀ (給定目標 Re, ν, k)")
        print("  4. 生成參數對照表")
        print("  5. 退出")

        choice = input("\n請選擇 (1-5): ") or "5"

        if choice == "1":
            mode_compute_Re()
        elif choice == "2":
            mode_compute_nu()
        elif choice == "3":
            mode_compute_f0()
        elif choice == "4":
            mode_parameter_table()
        else:
            print("退出")
        return

    # 命令行模式
    f0 = args.f0 if args.f0 is not None else 1.0
    nu = args.nu if args.nu is not None else 0.0125
    k_f = args.k_f if args.k_f is not None else 8.0

    # 求解 ν
    if args.solve_nu:
        if args.target_Re is None:
            print("錯誤：需要指定 --target-Re")
            sys.exit(1)
        nu = compute_nu_from_Re(args.target_Re, f0, k_f)
        print(f"所需動力黏度: ν = {nu:.6f}")
        print(f"驗證: Re = {compute_reynolds(f0, nu, k_f):.2f}")
        return

    # 求解 f₀
    if args.solve_f0:
        if args.target_Re is None:
            print("錯誤：需要指定 --target-Re")
            sys.exit(1)
        f0 = compute_f0_from_Re(args.target_Re, nu, k_f)
        print(f"所需強迫振幅: f₀ = {f0:.6f}")
        print(f"驗證: Re = {compute_reynolds(f0, nu, k_f):.2f}")
        return

    # 掃描模式
    if args.nu_range or args.f0_range or args.k_range:
        results = []

        if args.nu_range:
            nu_values = np.arange(args.nu_range[0], args.nu_range[1] + args.nu_range[2]/2, args.nu_range[2])
            print(f"{'ν':<12} {'Re':<12} {'U₀':<12} {'流動狀態':<20}")
            print("-"*60)
            for nu in nu_values:
                Re = compute_reynolds(f0, nu, k_f)
                U0 = compute_laminar_velocity(f0, nu, k_f)
                regime, _ = classify_flow_regime(Re)
                print(f"{nu:<12.6f} {Re:<12.2f} {U0:<12.4f} {regime:<20}")
                results.append({'nu': nu, 'f0': f0, 'k': k_f, 'Re': Re, 'U0': U0, 'regime': regime})

        elif args.f0_range:
            f0_values = np.arange(args.f0_range[0], args.f0_range[1] + args.f0_range[2]/2, args.f0_range[2])
            print(f"{'f₀':<12} {'Re':<12} {'U₀':<12} {'流動狀態':<20}")
            print("-"*60)
            for f0 in f0_values:
                Re = compute_reynolds(f0, nu, k_f)
                U0 = compute_laminar_velocity(f0, nu, k_f)
                regime, _ = classify_flow_regime(Re)
                print(f"{f0:<12.2f} {Re:<12.2f} {U0:<12.4f} {regime:<20}")
                results.append({'nu': nu, 'f0': f0, 'k': k_f, 'Re': Re, 'U0': U0, 'regime': regime})

        elif args.k_range:
            k_values = np.arange(args.k_range[0], args.k_range[1] + args.k_range[2]/2, args.k_range[2])
            print(f"{'k':<12} {'Re':<12} {'L':<12} {'U₀':<12} {'流動狀態':<20}")
            print("-"*70)
            for k_f in k_values:
                Re = compute_reynolds(f0, nu, k_f)
                L = compute_forcing_wavelength(k_f)
                U0 = compute_laminar_velocity(f0, nu, k_f)
                regime, _ = classify_flow_regime(Re)
                print(f"{k_f:<12.0f} {Re:<12.2f} {L:<12.4f} {U0:<12.4f} {regime:<20}")
                results.append({'nu': nu, 'f0': f0, 'k': k_f, 'Re': Re, 'L': L, 'U0': U0, 'regime': regime})

        # 保存到文件
        if args.output and results:
            import csv
            with open(args.output, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"\n結果已保存至: {args.output}")

        return

    # 單點計算
    Re = compute_reynolds(f0, nu, k_f)
    U0 = compute_laminar_velocity(f0, nu, k_f)
    L = compute_forcing_wavelength(k_f)
    regime, desc = classify_flow_regime(Re)

    print("="*80)
    print("計算結果")
    print("="*80)
    print(f"參數:")
    print(f"  f₀ = {f0}")
    print(f"  ν = {nu}")
    print(f"  k = {k_f}")
    print(f"\n結果:")
    print(f"  Re = {Re:.2f}")
    print(f"  U₀ = {U0:.4f}")
    print(f"  L = {L:.4f}")
    print(f"\n流動狀態: {regime}")
    print(f"  {desc}")
    print("="*80)


if __name__ == "__main__":
    main()
