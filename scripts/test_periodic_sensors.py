#!/usr/bin/env python3
"""
測試週期性感測點生成功能

驗證項目：
1. 週期性 QR-Pivot 能否正確執行
2. 接縫區域（x≈0, x≈L）是否有足夠覆蓋
3. 與標準 QR-Pivot 的差異
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_channel_flow import ChannelFlowDataFetcher, ChannelFlowConfig


def create_mock_field_data(nx=128, ny=64):
    """創建模擬的 2D 流場數據"""
    x = np.linspace(0, 8*np.pi, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 模擬通道流速度場
    u = 15.0 * (1 - Y**2) + 2.0 * np.sin(2*np.pi*X/(8*np.pi)) * np.exp(-5*Y**2)
    v = 0.5 * np.sin(4*np.pi*X/(8*np.pi)) * np.sin(np.pi*Y)
    w = 0.3 * np.cos(2*np.pi*X/(8*np.pi)) * (1 - Y**2)
    p = -1.0 * X + 0.1 * np.sin(2*np.pi*X/(8*np.pi))

    return {
        'x': x,
        'y': y,
        'u': u,
        'v': v,
        'w': w,
        'p': p
    }


def test_periodic_sensor_generation():
    """測試週期性感測點生成"""
    print("=" * 70)
    print("🧪 測試週期性感測點生成")
    print("=" * 70)

    # 創建模擬數據
    print("\n📊 創建模擬流場數據...")
    field_data = create_mock_field_data(nx=128, ny=64)
    x = field_data['x']

    # 創建 fetcher
    config = ChannelFlowConfig()
    fetcher = ChannelFlowDataFetcher(config, cache_dir="./test_periodic_output")

    K = 100  # 感測點數量

    # ============ 測試 1: 標準 QR-Pivot ============
    print(f"\n🔹 測試 1: 標準 QR-Pivot（無週期性）")
    result_standard = fetcher.generate_sensor_points_with_periodicity(
        field_data,
        K=K,
        use_periodic=False,
        noise_sigma=0.0,
        dropout_prob=0.0
    )

    # ============ 測試 2: 週期性 QR-Pivot ============
    print(f"\n🔹 測試 2: 週期性 QR-Pivot")
    result_periodic = fetcher.generate_sensor_points_with_periodicity(
        field_data,
        K=K,
        periodic_axes=[0],
        use_periodic=True,
        n_wrap_layers=2,
        noise_sigma=0.0,
        dropout_prob=0.0
    )

    # ============ 統計分析 ============
    print("\n" + "=" * 70)
    print("📈 統計分析")
    print("=" * 70)

    def analyze_seam_coverage(sensor_points, x_domain, label):
        """分析接縫覆蓋率"""
        x_sensors = sensor_points[:, 0]
        x_min, x_max = x_domain[0], x_domain[-1]
        seam_threshold = 0.05 * (x_max - x_min)

        near_x0 = np.sum(x_sensors < (x_min + seam_threshold))
        near_xL = np.sum(x_sensors > (x_max - seam_threshold))
        total_seam = near_x0 + near_xL

        print(f"\n{label}:")
        print(f"  接縫區域（5% 邊界）: {seam_threshold:.4f}")
        print(f"  x≈0 區域點數: {near_x0} / {K} ({100*near_x0/K:.1f}%)")
        print(f"  x≈L 區域點數: {near_xL} / {K} ({100*near_xL/K:.1f}%)")
        print(f"  總接縫覆蓋: {total_seam} / {K} ({100*total_seam/K:.1f}%)")

        return near_x0, near_xL, total_seam

    std_x0, std_xL, std_total = analyze_seam_coverage(
        result_standard['sensor_points'], x, "標準 QR-Pivot"
    )

    per_x0, per_xL, per_total = analyze_seam_coverage(
        result_periodic['sensor_points'], x, "週期性 QR-Pivot"
    )

    # ============ 品質指標比較 ============
    print("\n" + "=" * 70)
    print("🎯 品質指標比較")
    print("=" * 70)

    for key in ['condition_number', 'energy_ratio']:
        if key in result_standard['selection_info'] and key in result_periodic['selection_info']:
            std_val = result_standard['selection_info'][key]
            per_val = result_periodic['selection_info'][key]
            print(f"\n{key}:")
            print(f"  標準: {std_val:.6f}")
            print(f"  週期性: {per_val:.6f}")
            if key == 'condition_number':
                improvement = (std_val - per_val) / std_val * 100
                print(f"  改善: {improvement:+.2f}% (越低越好)")
            else:
                improvement = (per_val - std_val) / std_val * 100
                print(f"  改善: {improvement:+.2f}% (越高越好)")

    # ============ 視覺化比較 ============
    print("\n📊 生成視覺化比較圖...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 繪製速度場 u
    X, Y = np.meshgrid(x, field_data['y'], indexing='ij')

    # 標準 QR-Pivot 分佈
    ax = axes[0, 0]
    im = ax.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.6)
    ax.scatter(result_standard['sensor_points'][:, 0],
               result_standard['sensor_points'][:, 1],
               c='black', s=30, marker='o', edgecolors='white', linewidths=0.5)
    ax.set_title(f'標準 QR-Pivot (K={K})\n接縫覆蓋: {std_total}/{K}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='u')

    # 週期性 QR-Pivot 分佈
    ax = axes[0, 1]
    im = ax.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.6)
    ax.scatter(result_periodic['sensor_points'][:, 0],
               result_periodic['sensor_points'][:, 1],
               c='red', s=30, marker='^', edgecolors='white', linewidths=0.5)
    ax.set_title(f'週期性 QR-Pivot (K={K})\n接縫覆蓋: {per_total}/{K}', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(im, ax=ax, label='u')

    # x 方向分佈直方圖
    ax = axes[1, 0]
    ax.hist(result_standard['sensor_points'][:, 0], bins=30, alpha=0.5, label='標準', color='blue')
    ax.hist(result_periodic['sensor_points'][:, 0], bins=30, alpha=0.5, label='週期性', color='red')
    ax.axvline(x[0], color='green', linestyle='--', linewidth=1, label='接縫 x=0')
    ax.axvline(x[-1], color='green', linestyle='--', linewidth=1, label='接縫 x=L')
    ax.set_xlabel('x 座標')
    ax.set_ylabel('感測點數量')
    ax.set_title('x 方向分佈對比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # y 方向分佈直方圖
    ax = axes[1, 1]
    ax.hist(result_standard['sensor_points'][:, 1], bins=30, alpha=0.5, label='標準', color='blue', orientation='horizontal')
    ax.hist(result_periodic['sensor_points'][:, 1], bins=30, alpha=0.5, label='週期性', color='red', orientation='horizontal')
    ax.set_ylabel('y 座標')
    ax.set_xlabel('感測點數量')
    ax.set_title('y 方向分佈對比')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存圖表
    output_dir = Path("./test_periodic_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "periodic_sensor_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 視覺化圖表已保存: {output_path}")

    # ============ 總結 ============
    print("\n" + "=" * 70)
    print("✅ 測試完成")
    print("=" * 70)

    # 檢查是否滿足週期性覆蓋要求
    min_seam_coverage = 0.15  # 至少 15% 點在接縫區域

    if per_total / K >= min_seam_coverage:
        print(f"✅ 週期性處理有效：接縫覆蓋率 {100*per_total/K:.1f}% ≥ {100*min_seam_coverage:.1f}%")
    else:
        print(f"⚠️ 週期性覆蓋不足：接縫覆蓋率 {100*per_total/K:.1f}% < {100*min_seam_coverage:.1f}%")

    if per_total > std_total:
        improvement = (per_total - std_total) / std_total * 100
        print(f"✅ 相比標準方法提升 {improvement:.1f}% 接縫覆蓋")
    else:
        print(f"⚠️ 週期性方法未增加接縫覆蓋（可能需要調整 n_wrap_layers）")

    print("\n結果文件位置:")
    print(f"  - 標準感測點: {output_dir}/sensors_K{K}_qr_pivot_standard.npz")
    print(f"  - 週期性感測點: {output_dir}/sensors_K{K}_qr_pivot_periodic.npz")
    print(f"  - 視覺化圖表: {output_path}")


if __name__ == "__main__":
    test_periodic_sensor_generation()
