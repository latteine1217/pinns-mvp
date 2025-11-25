"""
測試壓力梯度評估指標
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import torch
from pinnx.evals.metrics import pressure_gradient_from_finite_diff


def test_pressure_gradient_from_finite_diff_2d():
    """測試 2D 壓力梯度計算（有限差分）"""
    # 創建簡單的 2D 壓力場：p = x^2 + y^2
    # 預期梯度：dp/dx = 2x, dp/dy = 2y

    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 11)
    X, Y = np.meshgrid(x, y, indexing='ij')

    p_field = X**2 + Y**2  # shape (11, 11)

    coords = {'x': x, 'y': y}

    # 計算梯度
    grad = pressure_gradient_from_finite_diff(p_field, coords)

    # 檢查輸出
    assert 'dpdx' in grad
    assert 'dpdy' in grad
    assert grad['dpdx'].shape == p_field.shape
    assert grad['dpdy'].shape == p_field.shape

    # 驗證數值梯度（中心差分）
    # dp/dx = 2x, 在 x=0.5 處應約為 1.0
    x_mid = len(x) // 2
    y_mid = len(y) // 2

    expected_dpdx = 2 * x[x_mid]
    expected_dpdy = 2 * y[y_mid]

    assert np.abs(grad['dpdx'][x_mid, y_mid] - expected_dpdx) < 0.1
    assert np.abs(grad['dpdy'][x_mid, y_mid] - expected_dpdy) < 0.1

    print(f"✅ 2D 壓力梯度測試通過")
    print(f"   預測 ∂p/∂x = {grad['dpdx'][x_mid, y_mid]:.4f}, 理論值 = {expected_dpdx:.4f}")
    print(f"   預測 ∂p/∂y = {grad['dpdy'][x_mid, y_mid]:.4f}, 理論值 = {expected_dpdy:.4f}")


def test_pressure_gradient_from_finite_diff_3d():
    """測試 3D 壓力梯度計算（有限差分）"""
    # 創建簡單的 3D 壓力場：p = x^2 + y^2 + z^2
    # 預期梯度：dp/dx = 2x, dp/dy = 2y, dp/dz = 2z

    x = np.linspace(0, 1, 11)
    y = np.linspace(0, 1, 11)
    z = np.linspace(0, 1, 11)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    p_field = X**2 + Y**2 + Z**2  # shape (11, 11, 11)

    coords = {'x': x, 'y': y, 'z': z}

    # 計算梯度
    grad = pressure_gradient_from_finite_diff(p_field, coords)

    # 檢查輸出
    assert 'dpdx' in grad
    assert 'dpdy' in grad
    assert 'dpdz' in grad
    assert grad['dpdx'].shape == p_field.shape
    assert grad['dpdy'].shape == p_field.shape
    assert grad['dpdz'].shape == p_field.shape

    # 驗證數值梯度
    x_mid = len(x) // 2
    y_mid = len(y) // 2
    z_mid = len(z) // 2

    expected_dpdx = 2 * x[x_mid]
    expected_dpdy = 2 * y[y_mid]
    expected_dpdz = 2 * z[z_mid]

    assert np.abs(grad['dpdx'][x_mid, y_mid, z_mid] - expected_dpdx) < 0.1
    assert np.abs(grad['dpdy'][x_mid, y_mid, z_mid] - expected_dpdy) < 0.1
    assert np.abs(grad['dpdz'][x_mid, y_mid, z_mid] - expected_dpdz) < 0.1

    print(f"✅ 3D 壓力梯度測試通過")
    print(f"   預測 ∂p/∂x = {grad['dpdx'][x_mid, y_mid, z_mid]:.4f}, 理論值 = {expected_dpdx:.4f}")
    print(f"   預測 ∂p/∂y = {grad['dpdy'][x_mid, y_mid, z_mid]:.4f}, 理論值 = {expected_dpdy:.4f}")
    print(f"   預測 ∂p/∂z = {grad['dpdz'][x_mid, y_mid, z_mid]:.4f}, 理論值 = {expected_dpdz:.4f}")


def test_constant_pressure_gradient():
    """測試常數壓力梯度（通道流場景）"""
    # 創建線性壓力場：p = -0.0025 * x（模擬通道流驅動）
    # 預期梯度：dp/dx = -0.0025（常數），dp/dy = 0, dp/dz = 0

    x = np.linspace(0, 25.13, 51)
    y = np.linspace(-1, 1, 21)
    z = np.linspace(0, 9.42, 31)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    p_field = -0.0025 * X  # 線性壓力場

    coords = {'x': x, 'y': y, 'z': z}

    # 計算梯度
    grad = pressure_gradient_from_finite_diff(p_field, coords)

    # 驗證 ∂p/∂x 是常數
    dpdx_std = np.std(grad['dpdx'])
    dpdx_mean = np.mean(grad['dpdx'])

    assert np.abs(dpdx_mean - (-0.0025)) < 0.001  # 平均值接近 -0.0025
    assert dpdx_std < 0.0001  # 標準差接近零（表示常數）

    # ∂p/∂y 和 ∂p/∂z 應接近零
    dpdy_mean = np.abs(np.mean(grad['dpdy']))
    dpdz_mean = np.abs(np.mean(grad['dpdz']))

    assert dpdy_mean < 0.001
    assert dpdz_mean < 0.001

    print(f"✅ 常數壓力梯度測試通過（通道流場景）")
    print(f"   ∂p/∂x: mean = {dpdx_mean:.6f}, std = {dpdx_std:.6f} (應為常數 -0.0025)")
    print(f"   ∂p/∂y: mean = {dpdy_mean:.6f} (應接近 0)")
    print(f"   ∂p/∂z: mean = {dpdz_mean:.6f} (應接近 0)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("測試壓力梯度評估指標")
    print("="*60 + "\n")

    test_pressure_gradient_from_finite_diff_2d()
    print()
    test_pressure_gradient_from_finite_diff_3d()
    print()
    test_constant_pressure_gradient()

    print("\n" + "="*60)
    print("✅ 所有測試通過！")
    print("="*60)
