"""
DNS 快速驗證工具（僅驗證少數快照）
================================

快速版本，僅驗證 5 個關鍵時間點

使用方式:
---------
python scripts/quick_validate_dns.py \\
    --input data/kolmogorov_Re1000_kf4_t100s.h5
"""

import numpy as np
import h5py
import argparse

def quick_validate(h5_file: str):
    """快速驗證 DNS 結果"""

    print("=" * 60)
    print("DNS 快速驗證工具")
    print("=" * 60)
    print("")

    # 載入數據
    print("📂 載入數據...")
    with h5py.File(h5_file, 'r') as f:
        # 只取 5 個快照
        indices = [0, len(f['time'])//4, len(f['time'])//2, 3*len(f['time'])//4, -1]

        u = f['u'][indices]
        v = f['v'][indices]
        p = f['p'][indices]
        time = f['time'][indices]

        N = f['config'].attrs['N']
        L = f['config'].attrs['L']
        nu = f['config'].attrs['nu']
        A = f['config'].attrs['A']
        k_f = f['config'].attrs['k_f']

    dx = dy = L / N
    Re = 1.0 / nu

    print(f"   ✓ 快照數: {len(indices)}")
    print(f"   ✓ 網格: {N}×{N}")
    print(f"   ✓ Re = {Re:.2f}")
    print("")

    # 1. 散度檢查
    print("🔍 1. 不可壓縮條件")
    div_errors = []
    for i in range(len(indices)):
        dudx = np.gradient(u[i], dx, axis=0)
        dvdy = np.gradient(v[i], dy, axis=1)
        div = dudx + dvdy
        div_errors.append(np.abs(div).max())

    div_max = max(div_errors)
    print(f"   最大散度誤差: {div_max:.4e}")
    print(f"   {'✅ 通過' if div_max < 1.0 else '❌ 失敗'} (閾值 < 1.0)")
    print("")

    # 2. 能量統計
    print("🔍 2. 能量統計")
    KE = 0.5 * np.mean(u**2 + v**2, axis=(1, 2))
    print(f"   初始動能: {KE[0]:.4e}")
    print(f"   最終動能: {KE[-1]:.4e}")
    print(f"   能量衰減: {(1 - KE[-1]/KE[0])*100:.1f}%")
    print(f"   {'✅ 合理' if 0 < (1 - KE[-1]/KE[0]) < 0.5 else '⚠️  異常'}")
    print("")

    # 3. 雷諾數
    print("🔍 3. 雷諾數檢查")
    L_forcing = 2 * np.pi / k_f
    Re_theory = np.sqrt(A) * L_forcing**1.5 / nu
    U_rms = np.sqrt(np.mean(u**2 + v**2))
    Re_actual = U_rms * L / nu

    print(f"   理論雷諾數: {Re_theory:.2f}")
    print(f"   實際雷諾數: {Re_actual:.2f}")
    print(f"   相對誤差: {np.abs(Re_theory - Re_actual)/Re_theory*100:.1f}%")
    print(f"   {'✅ 一致' if np.abs(Re_theory - Re_actual)/Re_theory < 0.5 else '⚠️  差異大'}")
    print("")

    # 4. 速度統計
    print("🔍 4. 速度場統計")
    print(f"   U 範圍: [{u.min():.4f}, {u.max():.4f}]")
    print(f"   V 範圍: [{v.min():.4f}, {v.max():.4f}]")
    print(f"   P 範圍: [{p.min():.4f}, {p.max():.4f}]")
    print(f"   {'✅ 正常' if not (np.isnan(u).any() or np.isinf(u).any()) else '❌ 有 NaN/Inf'}")
    print("")

    # 5. Kolmogorov 尺度（估算）
    print("🔍 5. 解析度檢查")
    # 粗略估計耗散率
    dudx = np.gradient(u[-1], dx, axis=0)
    dudy = np.gradient(u[-1], dy, axis=1)
    dvdx = np.gradient(v[-1], dx, axis=0)
    dvdy = np.gradient(v[-1], dy, axis=1)
    strain = dudx**2 + dvdy**2 + 0.5*(dudy + dvdx)**2
    epsilon = nu * np.mean(strain)

    eta = (nu**3 / epsilon)**0.25
    dx_eta_ratio = dx / eta

    print(f"   Kolmogorov 尺度: η = {eta:.6f}")
    print(f"   網格間距: dx = {dx:.6f}")
    print(f"   解析度比: dx/η = {dx_eta_ratio:.2f}")
    print(f"   {'✅ 充分' if dx_eta_ratio < 10 else '⚠️  粗略 (dx/η > 10)'}")
    print("")

    # 總結
    print("=" * 60)
    print("驗證總結")
    print("=" * 60)

    checks = [
        ('不可壓縮', div_max < 1.0),
        ('能量合理', 0 < (1 - KE[-1]/KE[0]) < 0.5),
        ('雷諾數一致', np.abs(Re_theory - Re_actual)/Re_theory < 0.5),
        ('無 NaN/Inf', not (np.isnan(u).any() or np.isinf(u).any())),
        ('解析度充分', dx_eta_ratio < 10)
    ]

    passed = sum([c[1] for c in checks])
    total = len(checks)

    for name, status in checks:
        print(f"  {'✅' if status else '❌'} {name}")

    print("")
    print(f"通過: {passed}/{total} ({passed/total*100:.0f}%)")
    print("")

    if passed >= 4:
        print("🎉 快速驗證通過！DNS 結果基本合理。")
        print("   建議執行完整驗證以獲得詳細報告。")
    else:
        print("⚠️  部分檢查失敗，建議檢查模擬參數或執行完整驗證。")

    print("=" * 60)

    return passed >= 4


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DNS 快速驗證')
    parser.add_argument('--input', type=str, required=True, help='HDF5 文件')
    args = parser.parse_args()

    success = quick_validate(args.input)
    exit(0 if success else 1)
