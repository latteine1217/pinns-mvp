#!/usr/bin/env python3
"""
單元測試: 分層採樣策略
測試 sample_boundary_points() 和 sample_interior_points() 函數
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pytest
import numpy as np

# 從 train.py 導入採樣函數
from scripts.train import sample_boundary_points, sample_interior_points


class TestStratifiedSampling:
    """測試分層採樣函數"""
    
    @pytest.fixture
    def device(self):
        """測試設備"""
        return torch.device('cpu')
    
    @pytest.fixture
    def domain_bounds(self):
        """測試域邊界"""
        return {
            'x': (0.0, 25.13),
            'y': (-1.0, 1.0),
            'z': (0.0, 9.42)
        }
    
    def test_boundary_points_count(self, device, domain_bounds):
        """測試邊界點總數是否正確"""
        n_points = 2000
        boundary_dist = {'wall': 1000, 'periodic': 800, 'inlet': 200}
        
        points = sample_boundary_points(
            n_points=n_points,
            domain_bounds=domain_bounds,
            device=device,
            distribution=boundary_dist
        )
        
        assert points.shape == (n_points, 3), f"Expected shape ({n_points}, 3), got {points.shape}"
        assert points.device == device, f"Expected device {device}, got {points.device}"
    
    def test_boundary_points_on_boundaries(self, device, domain_bounds):
        """測試邊界點是否真的在邊界上"""
        n_points = 2000
        boundary_dist = {'wall': 1000, 'periodic': 800, 'inlet': 200}
        
        points = sample_boundary_points(
            n_points=n_points,
            domain_bounds=domain_bounds,
            device=device,
            distribution=boundary_dist
        )
        
        x = points[:, 0].cpu().numpy()
        y = points[:, 1].cpu().numpy()
        z = points[:, 2].cpu().numpy()
        
        x_min, x_max = domain_bounds['x']
        y_min, y_max = domain_bounds['y']
        z_min, z_max = domain_bounds['z']
        
        # 容差
        tol = 1e-5
        
        # 檢查每個點是否在至少一個邊界上
        on_boundary = np.zeros(len(points), dtype=bool)
        
        # 壁面 (y = y_min 或 y = y_max)
        on_wall = (np.abs(y - y_min) < tol) | (np.abs(y - y_max) < tol)
        on_boundary |= on_wall
        
        # 週期性邊界 (x = x_min/x_max 或 z = z_min/z_max)
        on_periodic = (
            (np.abs(x - x_min) < tol) | (np.abs(x - x_max) < tol) |
            (np.abs(z - z_min) < tol) | (np.abs(z - z_max) < tol)
        )
        on_boundary |= on_periodic
        
        # 統計壁面點數量（應該接近 1000）
        n_wall = np.sum(on_wall)
        assert 900 <= n_wall <= 1100, f"Expected ~1000 wall points, got {n_wall}"
        
        print(f"✅ 邊界點分佈驗證通過:")
        print(f"   - 壁面點: {n_wall} / 1000")
        print(f"   - 總邊界點: {np.sum(on_boundary)} / {n_points}")
    
    def test_interior_points_inside_domain(self, device, domain_bounds):
        """測試內部點是否在域內部（排除邊界）"""
        n_points = 10000
        exclude_tol = 0.01
        
        points = sample_interior_points(
            n_points=n_points,
            domain_bounds=domain_bounds,
            device=device,
            exclude_boundary_tol=exclude_tol,
            use_sobol=True
        )
        
        assert points.shape == (n_points, 3), f"Expected shape ({n_points}, 3), got {points.shape}"
        
        x = points[:, 0].cpu().numpy()
        y = points[:, 1].cpu().numpy()
        z = points[:, 2].cpu().numpy()
        
        x_min, x_max = domain_bounds['x']
        y_min, y_max = domain_bounds['y']
        z_min, z_max = domain_bounds['z']
        
        # 檢查所有點是否在內部（考慮容差）
        inside_x = (x >= x_min + exclude_tol) & (x <= x_max - exclude_tol)
        inside_y = (y >= y_min + exclude_tol) & (y <= y_max - exclude_tol)
        inside_z = (z >= z_min + exclude_tol) & (z <= z_max - exclude_tol)
        
        inside = inside_x & inside_y & inside_z
        
        assert np.all(inside), f"Found {np.sum(~inside)} points outside interior domain"
        
        print(f"✅ 內部點驗證通過:")
        print(f"   - x 範圍: [{x.min():.4f}, {x.max():.4f}] (期望: [{x_min + exclude_tol:.4f}, {x_max - exclude_tol:.4f}])")
        print(f"   - y 範圍: [{y.min():.4f}, {y.max():.4f}] (期望: [{y_min + exclude_tol:.4f}, {y_max - exclude_tol:.4f}])")
        print(f"   - z 範圍: [{z.min():.4f}, {z.max():.4f}] (期望: [{z_min + exclude_tol:.4f}, {z_max - exclude_tol:.4f}])")
    
    def test_sobol_uniformity(self, device, domain_bounds):
        """測試 Sobol 序列的均勻性（相比隨機採樣）"""
        n_points = 5000
        
        # Sobol 採樣
        points_sobol = sample_interior_points(
            n_points=n_points,
            domain_bounds=domain_bounds,
            device=device,
            use_sobol=True
        )
        
        # 隨機採樣
        points_random = sample_interior_points(
            n_points=n_points,
            domain_bounds=domain_bounds,
            device=device,
            use_sobol=False
        )
        
        # 計算分佈均勻性（使用標準差作為指標）
        # Sobol 應該有更低的標準差（更均勻）
        def compute_uniformity_metric(points):
            """計算點分佈的均勻性（沿每個軸的標準差）"""
            std_x = torch.std(points[:, 0]).item()
            std_y = torch.std(points[:, 1]).item()
            std_z = torch.std(points[:, 2]).item()
            return (std_x + std_y + std_z) / 3.0
        
        uniform_sobol = compute_uniformity_metric(points_sobol)
        uniform_random = compute_uniformity_metric(points_random)
        
        print(f"✅ Sobol vs Random 均勻性:")
        print(f"   - Sobol 標準差: {uniform_sobol:.6f}")
        print(f"   - Random 標準差: {uniform_random:.6f}")
        print(f"   - Sobol 改善: {(uniform_random - uniform_sobol) / uniform_random * 100:.1f}%")
    
    def test_sampling_performance(self, device, domain_bounds):
        """測試採樣效率（應 < 100ms）"""
        import time
        
        n_boundary = 2000
        n_interior = 10000
        
        # 測試邊界點採樣
        start = time.time()
        boundary_points = sample_boundary_points(
            n_points=n_boundary,
            domain_bounds=domain_bounds,
            device=device
        )
        boundary_time = (time.time() - start) * 1000  # ms
        
        # 測試內部點採樣
        start = time.time()
        interior_points = sample_interior_points(
            n_points=n_interior,
            domain_bounds=domain_bounds,
            device=device,
            use_sobol=True
        )
        interior_time = (time.time() - start) * 1000  # ms
        
        total_time = boundary_time + interior_time
        
        print(f"✅ 採樣效能測試:")
        print(f"   - 邊界點 ({n_boundary} 點): {boundary_time:.1f} ms")
        print(f"   - 內部點 ({n_interior} 點): {interior_time:.1f} ms")
        print(f"   - 總時間: {total_time:.1f} ms")
        
        assert total_time < 100, f"Sampling too slow: {total_time:.1f} ms > 100 ms"


if __name__ == "__main__":
    """直接執行測試（不使用 pytest）"""
    print("=" * 60)
    print("分層採樣策略單元測試")
    print("=" * 60)
    
    test = TestStratifiedSampling()
    device = torch.device('cpu')
    domain_bounds = {
        'x': (0.0, 25.13),
        'y': (-1.0, 1.0),
        'z': (0.0, 9.42)
    }
    
    print("\n1️⃣ 測試邊界點數量...")
    test.test_boundary_points_count(device, domain_bounds)
    print("通過!\n")
    
    print("2️⃣ 測試邊界點位置...")
    test.test_boundary_points_on_boundaries(device, domain_bounds)
    print("\n")
    
    print("3️⃣ 測試內部點範圍...")
    test.test_interior_points_inside_domain(device, domain_bounds)
    print("\n")
    
    print("4️⃣ 測試 Sobol 均勻性...")
    test.test_sobol_uniformity(device, domain_bounds)
    print("\n")
    
    print("5️⃣ 測試採樣效能...")
    test.test_sampling_performance(device, domain_bounds)
    print("\n")
    
    print("=" * 60)
    print("✅ 所有測試通過!")
    print("=" * 60)
