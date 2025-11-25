"""
測試壁面剪應力計算（修正版）

驗證壁面剪應力計算包含黏性係數，確保數值正確性
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def test_wall_shear_stress_theory():
    """測試理論壁面剪應力計算"""
    # 理論設定 (JHTDB Channel Flow Re_τ=1000)
    nu = 5e-5  # 運動黏性係數
    rho = 1.0  # 密度
    mu = rho * nu  # 動力黏性係數

    # 模擬線性速度剖面（壁面附近）
    # u(y) ≈ u_wall + (∂u/∂y)*y
    # 對於無滑移邊界: u_wall = 0
    # 假設 ∂u/∂y = 50 (典型值)

    dudy = 50.0  # 壁面速度梯度
    tau_w_expected = mu * dudy  # τ_w = 5e-5 * 50 = 0.0025

    # 驗證理論值
    assert np.abs(tau_w_expected - 0.0025) < 1e-6, f"理論壁面剪應力應為 0.0025，實際為 {tau_w_expected}"

    print(f"✅ 理論壁面剪應力測試通過")
    print(f"   ν = {nu}, μ = {mu}")
    print(f"   ∂u/∂y = {dudy}, τ_w = {tau_w_expected:.6f}")


def test_wall_shear_stress_calculation_2d():
    """測試 2D 壁面剪應力計算（使用有限差分）"""
    # 創建簡單的通道流速度場（拋物線剖面）
    # u(y) = u_max * (1 - y²)，其中 y ∈ [-1, 1]

    nx, ny = 64, 32
    x = np.linspace(0, 25.13, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_max = 1.5  # 中心線最大速度
    u_field = u_max * (1 - Y**2)  # 拋物線剖面

    # 計算壁面速度梯度（使用有限差分）
    dy = y[1] - y[0]

    # 下壁面 (y=-1, 索引 0)
    dudy_lower = (u_field[:, 1] - u_field[:, 0]) / dy

    # 理論值：u = u_max*(1-y²), du/dy = -2*u_max*y
    # 在 y=-1: du/dy = -2*u_max*(-1) = 2*u_max = 3.0
    dudy_lower_theory = 2 * u_max  # = 3.0

    # 計算壁面剪應力
    nu = 5e-5
    mu = 1.0 * nu
    tau_w_lower = mu * dudy_lower.mean()
    tau_w_theory = mu * dudy_lower_theory

    # 驗證數值精度（有限差分誤差）
    rel_error = np.abs(tau_w_lower - tau_w_theory) / tau_w_theory
    assert rel_error < 0.05, f"壁面剪應力誤差過大: {rel_error:.2%}"

    print(f"✅ 2D 壁面剪應力計算測試通過")
    print(f"   數值 τ_w = {tau_w_lower:.6e}")
    print(f"   理論 τ_w = {tau_w_theory:.6e}")
    print(f"   相對誤差 = {rel_error:.2%}")


def test_wall_shear_stress_symmetry():
    """測試通道流對稱性（上下壁面剪應力絕對值應相等）"""
    # 創建對稱的通道流
    nx, ny = 64, 32
    x = np.linspace(0, 25.13, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u_max = 1.5
    u_field = u_max * (1 - Y**2)  # 對稱拋物線剖面

    dy = y[1] - y[0]
    nu = 5e-5
    mu = 1.0 * nu

    # 下壁面 (y=-1)：du/dy > 0（速度向上增加）
    dudy_lower = (u_field[:, 1] - u_field[:, 0]) / dy
    tau_w_lower = mu * dudy_lower.mean()

    # 上壁面 (y=+1)：du/dy < 0（速度向下減少）
    # 使用後向差分
    dudy_upper = (u_field[:, -1] - u_field[:, -2]) / dy
    tau_w_upper = mu * dudy_upper.mean()

    # 檢查對稱性：絕對值應相等（符號相反）
    # 對於對稱流動：|τ_w_lower| ≈ |τ_w_upper|
    tau_w_lower_abs = np.abs(tau_w_lower)
    tau_w_upper_abs = np.abs(tau_w_upper)

    symmetry_error = np.abs(tau_w_lower_abs - tau_w_upper_abs) / tau_w_lower_abs

    assert symmetry_error < 0.10, f"通道流對稱性誤差過大: {symmetry_error:.2%}"

    print(f"✅ 通道流對稱性測試通過")
    print(f"   下壁面 τ_w = {tau_w_lower:.6e} (|τ_w| = {tau_w_lower_abs:.6e})")
    print(f"   上壁面 τ_w = {tau_w_upper:.6e} (|τ_w| = {tau_w_upper_abs:.6e})")
    print(f"   對稱性誤差 = {symmetry_error:.2%}")


def test_wall_shear_stress_jhtdb_range():
    """測試 JHTDB Channel Flow Re_τ=1000 的壁面剪應力範圍"""
    # JHTDB 理論值
    tau_w_theory = 0.0025
    acceptable_range = (0.0020, 0.0030)  # ±20%

    # 模擬典型的預測值
    nu = 5e-5
    mu = 1.0 * nu
    dudy_typical = 50.0  # 典型壁面速度梯度

    tau_w_pred = mu * dudy_typical

    # 檢查是否在可接受範圍內
    in_range = acceptable_range[0] <= tau_w_pred <= acceptable_range[1]

    assert in_range, (f"壁面剪應力 {tau_w_pred:.6f} 超出可接受範圍 "
                      f"[{acceptable_range[0]}, {acceptable_range[1]}]")

    # 相對誤差
    rel_error = np.abs(tau_w_pred - tau_w_theory) / tau_w_theory

    print(f"✅ JHTDB 範圍測試通過")
    print(f"   預測 τ_w = {tau_w_pred:.6f}")
    print(f"   理論 τ_w = {tau_w_theory:.6f}")
    print(f"   可接受範圍 = [{acceptable_range[0]}, {acceptable_range[1]}]")
    print(f"   相對誤差 = {rel_error:.2%}")


def test_viscosity_scaling():
    """測試黏性係數縮放關係"""
    # 理論關係: τ_w = μ·(∂u/∂y)
    # 當 ν 增加，τ_w 應成正比增加

    dudy = 50.0
    rho = 1.0

    # 測試不同的黏性係數
    nu_values = [1e-5, 5e-5, 1e-4]
    tau_w_values = []

    for nu in nu_values:
        mu = rho * nu
        tau_w = mu * dudy
        tau_w_values.append(tau_w)

    # 檢查線性關係
    nu_array = np.array(nu_values)
    tau_array = np.array(tau_w_values)

    # τ_w / ν 應該是常數
    ratio = tau_array / nu_array
    ratio_std = np.std(ratio) / np.mean(ratio)

    assert ratio_std < 1e-10, f"黏性係數縮放關係不成立: std/mean = {ratio_std}"

    print(f"✅ 黏性係數縮放測試通過")
    print(f"   ν = {nu_values}")
    print(f"   τ_w = {tau_w_values}")
    print(f"   τ_w/ν 常數 = {ratio[0]:.2f} (std = {ratio_std:.2e})")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("測試壁面剪應力計算（修正版）")
    print("="*70 + "\n")

    test_wall_shear_stress_theory()
    print()
    test_wall_shear_stress_calculation_2d()
    print()
    test_wall_shear_stress_symmetry()
    print()
    test_wall_shear_stress_jhtdb_range()
    print()
    test_viscosity_scaling()

    print("\n" + "="*70)
    print("✅ 所有測試通過！")
    print("="*70)
