#!/usr/bin/env python3
"""
Kolmogorov Flow 雷諾數驗證腳本
=================================

驗證 DNS 數據的實際物理參數，並計算真實的雷諾數。

檢查項目：
1. DNS 數據檔案中的實際參數（如果有元數據）
2. 根據不同 Re 定義計算實際雷諾數
3. 比對配置文件與 DNS 數據的一致性
4. 提供修正建議

作者：PINNs-MVP 團隊
日期：2025-11-25
"""

import numpy as np
import h5py
from pathlib import Path
import argparse
import sys


def compute_reynolds_standard(A: float, nu: float, k_f: float) -> float:
    """
    計算標準定義的雷諾數：Re = F / (ν² k³)

    Args:
        A: 強迫振幅
        nu: 動力黏度
        k_f: 強迫波數

    Returns:
        Re: 雷諾數
    """
    return A / (nu**2 * k_f**3)


def compute_reynolds_from_velocity(U: float, L: float, nu: float) -> float:
    """
    計算基於速度尺度的雷諾數：Re = UL / ν

    Args:
        U: 特徵速度
        L: 特徵長度
        nu: 動力黏度

    Returns:
        Re: 雷諾數
    """
    return U * L / nu


def compute_laminar_velocity(A: float, nu: float, k_f: float) -> float:
    """
    計算層流解的特徵速度：U = A / (ν k²)

    Args:
        A: 強迫振幅
        nu: 動力黏度
        k_f: 強迫波數

    Returns:
        U: 層流速度
    """
    return A / (nu * k_f**2)


def analyze_dns_data(data_path: Path):
    """
    分析 DNS 數據檔案

    Args:
        data_path: DNS 數據檔案路徑
    """
    print("=" * 80)
    print("DNS 數據檔案分析")
    print("=" * 80)
    print(f"檔案路徑: {data_path}")

    if not data_path.exists():
        print(f"❌ 錯誤：檔案不存在")
        return None

    try:
        with h5py.File(data_path, 'r') as f:
            print(f"\n📦 檔案結構:")
            print(f"   可用鍵值: {list(f.keys())}")

            # 檢查元數據
            if 'metadata' in f:
                print(f"\n📋 元數據:")
                metadata = f['metadata']
                for key in metadata.attrs.keys():
                    print(f"   {key}: {metadata.attrs[key]}")
            else:
                print(f"\n⚠️  無元數據")

            # 檢查物理參數
            physics_keys = ['nu', 'Re', 'k_f', 'forcing_amplitude', 'A']
            found_params = {}

            for key in physics_keys:
                if key in f.attrs:
                    found_params[key] = f.attrs[key]
                    print(f"   {key}: {f.attrs[key]}")

            # 檢查場數據
            field_keys = ['u', 'v', 'p', 'velocity', 'pressure']
            print(f"\n🌊 場數據:")
            for key in field_keys:
                if key in f:
                    data = f[key]
                    print(f"   {key}: shape={data.shape}, dtype={data.dtype}")

                    # 計算統計量
                    if key in ['u', 'velocity']:
                        u_data = data[:]
                        u_mean = np.mean(u_data)
                        u_std = np.std(u_data)
                        u_max = np.max(np.abs(u_data))
                        print(f"      統計: mean={u_mean:.4f}, std={u_std:.4f}, max(|u|)={u_max:.4f}")

            # 計算動能
            if 'u' in f and 'v' in f:
                u = f['u'][:]
                v = f['v'][:]
                KE = np.mean(u**2 + v**2) / 2.0
                U_rms = np.sqrt(2.0 * KE)
                print(f"\n⚡ 動能統計:")
                print(f"   平均動能 (KE): {KE:.6f}")
                print(f"   RMS 速度 (U_rms): {U_rms:.6f}")

                return {
                    'found_params': found_params,
                    'KE': KE,
                    'U_rms': U_rms,
                }

    except Exception as e:
        print(f"❌ 讀取檔案失敗: {e}")
        return None


def verify_reynolds_consistency(
    A: float = 1.0,
    nu: float = 0.01,
    k_f: float = 8.0,
    Re_config: float = 100.0,
    dns_results: dict = None
):
    """
    驗證雷諾數一致性

    Args:
        A: 強迫振幅
        nu: 動力黏度
        k_f: 強迫波數
        Re_config: 配置文件中宣稱的 Re
        dns_results: DNS 數據分析結果
    """
    print("\n" + "=" * 80)
    print("雷諾數一致性驗證")
    print("=" * 80)

    print(f"\n📝 配置參數:")
    print(f"   強迫振幅 (A): {A}")
    print(f"   動力黏度 (ν): {nu}")
    print(f"   強迫波數 (k_f): {k_f}")
    print(f"   宣稱雷諾數 (Re_config): {Re_config}")

    # 計算標準定義的雷諾數
    Re_standard = compute_reynolds_standard(A, nu, k_f)
    print(f"\n🔬 標準定義計算:")
    print(f"   Re = F / (ν² k³)")
    print(f"   Re = {A} / ({nu}² × {k_f}³)")
    print(f"   Re = {A} / ({nu**2:.6f} × {k_f**3})")
    print(f"   Re = {A} / {nu**2 * k_f**3:.6f}")
    print(f"   Re = {Re_standard:.2f}")

    # 比對
    diff_standard = abs(Re_standard - Re_config)
    ratio_standard = Re_config / Re_standard
    print(f"\n   與配置差異: {diff_standard:.2f} ({ratio_standard:.2f}x)")

    if abs(ratio_standard - 1.0) > 0.1:
        print(f"   ⚠️  差異過大！")
    else:
        print(f"   ✅ 一致")

    # 計算層流解的雷諾數
    U_laminar = compute_laminar_velocity(A, nu, k_f)
    L_char = 2.0 * np.pi / k_f
    Re_laminar = compute_reynolds_from_velocity(U_laminar, L_char, nu)

    print(f"\n🔬 層流解定義計算:")
    print(f"   層流速度 U = A / (ν k²) = {A} / ({nu} × {k_f}²) = {U_laminar:.4f}")
    print(f"   特徵長度 L = 2π/k = {L_char:.4f}")
    print(f"   Re = UL / ν = {U_laminar:.4f} × {L_char:.4f} / {nu} = {Re_laminar:.2f}")

    diff_laminar = abs(Re_laminar - Re_config)
    ratio_laminar = Re_config / Re_laminar
    print(f"   與配置差異: {diff_laminar:.2f} ({ratio_laminar:.2f}x)")

    if abs(ratio_laminar - 1.0) > 0.1:
        print(f"   ⚠️  差異過大！")
    else:
        print(f"   ✅ 一致")

    # 如果有 DNS 數據，計算有效雷諾數
    if dns_results is not None and 'U_rms' in dns_results:
        U_rms = dns_results['U_rms']
        L_char = 1.0 / k_f  # 使用 1/k 作為特徵長度
        Re_effective = compute_reynolds_from_velocity(U_rms, L_char, nu)

        print(f"\n🔬 DNS 數據有效雷諾數:")
        print(f"   RMS 速度 (U_rms): {U_rms:.4f}")
        print(f"   特徵長度 L = 1/k = {L_char:.4f}")
        print(f"   Re_eff = U_rms × L / ν = {U_rms:.4f} × {L_char:.4f} / {nu} = {Re_effective:.2f}")

        diff_eff = abs(Re_effective - Re_config)
        ratio_eff = Re_config / Re_effective
        print(f"   與配置差異: {diff_eff:.2f} ({ratio_eff:.2f}x)")

        if abs(ratio_eff - 1.0) > 0.1:
            print(f"   ⚠️  差異過大！")
        else:
            print(f"   ✅ 一致")

    # 提供修正建議
    print("\n" + "=" * 80)
    print("修正建議")
    print("=" * 80)

    if abs(ratio_standard - 1.0) > 0.1:
        print(f"\n⚠️  **檢測到雷諾數定義不一致**")
        print(f"\n📌 方案 A：修正動力黏度 (ν) 以匹配 Re={Re_config} (標準定義)")
        nu_corrected = np.sqrt(A / (Re_config * k_f**3))
        print(f"   建議修正: nu = {nu_corrected:.6f}")
        print(f"   配置文件中修改:")
        print(f"   ```yaml")
        print(f"   physics:")
        print(f"     nu: {nu_corrected:.6f}")
        print(f"   ```")

        print(f"\n📌 方案 B：修正強迫振幅 (A) 以匹配 Re={Re_config} (保持 ν={nu})")
        A_corrected = Re_config * nu**2 * k_f**3
        print(f"   建議修正: A = {A_corrected:.6f}")
        print(f"   配置文件中修改:")
        print(f"   ```yaml")
        print(f"   physics:")
        print(f"     forcing:")
        print(f"       amplitude: {A_corrected:.6f}")
        print(f"   ```")

        print(f"\n📌 方案 C：使用實際計算的雷諾數 Re={Re_standard:.2f}")
        print(f"   配置文件中修改:")
        print(f"   ```yaml")
        print(f"   kolmogorov_config:")
        print(f"     physics_params:")
        print(f"       Re: {Re_standard:.2f}  # 標準定義：F/(ν²k³)")
        print(f"   ```")
        print(f"   並在文檔中明確說明使用標準定義")

    else:
        print(f"\n✅ 雷諾數定義一致，無需修正")

    # 總結表格
    print("\n" + "=" * 80)
    print("雷諾數計算總結")
    print("=" * 80)
    print(f"{'定義':<25} {'計算值':<15} {'與配置比值':<15} {'狀態':<10}")
    print("-" * 80)
    print(f"{'配置文件宣稱':<25} {Re_config:<15.2f} {'1.00':<15} {'參考':<10}")
    print(f"{'標準定義 (F/(ν²k³))':<25} {Re_standard:<15.2f} {ratio_standard:<15.2f} {'⚠️' if abs(ratio_standard-1)>0.1 else '✅':<10}")
    print(f"{'層流解 (UL/ν)':<25} {Re_laminar:<15.2f} {ratio_laminar:<15.2f} {'⚠️' if abs(ratio_laminar-1)>0.1 else '✅':<10}")

    if dns_results is not None and 'U_rms' in dns_results:
        print(f"{'DNS 有效 Re':<25} {Re_effective:<15.2f} {ratio_eff:<15.2f} {'⚠️' if abs(ratio_eff-1)>0.1 else '✅':<10}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="驗證 Kolmogorov flow 的雷諾數定義與參數一致性"
    )
    parser.add_argument(
        '--dns-data',
        type=str,
        default='./data/kolmogorov_dns_re100_512x512_kf8_midway.h5',
        help='DNS 數據檔案路徑'
    )
    parser.add_argument(
        '--A',
        type=float,
        default=1.0,
        help='強迫振幅（默認 1.0）'
    )
    parser.add_argument(
        '--nu',
        type=float,
        default=0.01,
        help='動力黏度（默認 0.01）'
    )
    parser.add_argument(
        '--k-f',
        type=float,
        default=8.0,
        help='強迫波數（默認 8.0）'
    )
    parser.add_argument(
        '--Re-config',
        type=float,
        default=100.0,
        help='配置文件中宣稱的雷諾數（默認 100.0）'
    )

    args = parser.parse_args()

    # 分析 DNS 數據
    dns_path = Path(args.dns_data)
    dns_results = analyze_dns_data(dns_path)

    # 驗證雷諾數一致性
    verify_reynolds_consistency(
        A=args.A,
        nu=args.nu,
        k_f=args.k_f,
        Re_config=args.Re_config,
        dns_results=dns_results
    )


if __name__ == "__main__":
    main()
