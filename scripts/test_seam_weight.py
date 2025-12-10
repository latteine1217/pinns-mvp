#!/usr/bin/env python3
"""
測試不同 seam_weight 值對週期性感測點分佈的影響
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_channel_flow import ChannelFlowDataFetcher, ChannelFlowConfig


def extract_2d_slice(data_3d_path, z_index=64):
    """從 3D cutout 提取 2D 切片"""
    data = np.load(data_3d_path)
    x, y, z = data['x'], data['y'], data['z']
    nx, ny, nz = len(x), len(y), len(z)

    u_3d = data['u'].reshape(nx, ny, nz)
    v_3d = data['v'].reshape(nx, ny, nz)
    w_3d = data['w'].reshape(nx, ny, nz)
    p_3d = data['p'].reshape(nx, ny, nz)

    return {
        'x': x, 'y': y,
        'u': u_3d[:, :, z_index],
        'v': v_3d[:, :, z_index],
        'w': w_3d[:, :, z_index],
        'p': p_3d[:, :, z_index]
    }


def main():
    print("=" * 70)
    print("🧪 測試 seam_weight 對感測點分佈的影響")
    print("=" * 70)

    # 載入數據
    data_path = Path("data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz")
    if not data_path.exists():
        print(f"❌ 找不到數據: {data_path}")
        return 1

    field_data = extract_2d_slice(data_path)
    x = field_data['x']

    # 創建 fetcher
    config = ChannelFlowConfig()
    fetcher = ChannelFlowDataFetcher(config, cache_dir="./test_seam_weight_output")

    K = 100
    seam_weights = [1.0, 0.5, 0.3, 0.1]  # 測試 4 個不同權重

    results = {}

    print(f"\n📊 測試參數:")
    print(f"  感測點數: K={K}")
    print(f"  環繞層數: n_wrap_layers=2")
    print(f"  測試權重: {seam_weights}")

    # 執行測試
    for weight in seam_weights:
        print(f"\n🔹 測試 seam_weight={weight}")

        result = fetcher.generate_sensor_points_with_periodicity(
            field_data,
            K=K,
            periodic_axes=[0],
            use_periodic=True,
            n_wrap_layers=2,
            seam_weight=weight,  # ✅ 測試不同權重
            noise_sigma=0.0,
            dropout_prob=0.0
        )

        results[weight] = result

        # 統計分佈
        x_sensors = result['sensor_points'][:, 0]
        x_min, x_max = x[0], x[-1]
        seam_threshold = 0.05 * (x_max - x_min)

        near_x0 = np.sum(x_sensors < (x_min + seam_threshold))
        near_xL = np.sum(x_sensors > (x_max - seam_threshold))

        print(f"    接縫覆蓋: x≈0 ({near_x0}點), x≈L ({near_xL}點), "
              f"總計 {near_x0+near_xL}/{K} ({100*(near_x0+near_xL)/K:.1f}%)")
        print(f"    條件數: {result['selection_info'].get('condition_number', 'N/A'):.2f}")

    # 視覺化比較
    print("\n📊 生成視覺化比較...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))

    X, Y = np.meshgrid(x, field_data['y'], indexing='ij')

    for idx, weight in enumerate(seam_weights):
        result = results[weight]
        sensor_points = result['sensor_points']

        # 上排：速度場 + 感測點
        ax = axes[0, idx]
        im = ax.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.7)
        ax.scatter(sensor_points[:, 0], sensor_points[:, 1],
                  c='black', s=15, marker='o', alpha=0.7, edgecolors='white', linewidths=0.3)
        ax.axvline(x[0], color='lime', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x[-1], color='lime', linestyle='--', linewidth=1, alpha=0.5)

        x_sensors = sensor_points[:, 0]
        seam_threshold = 0.05 * (x[-1] - x[0])
        seam_count = np.sum((x_sensors < (x[0] + seam_threshold)) |
                           (x_sensors > (x[-1] - seam_threshold)))

        ax.set_title(f'seam_weight={weight}\nSeam: {seam_count}/{K}', fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 下排：x 方向分佈直方圖
        ax = axes[1, idx]
        ax.hist(x_sensors, bins=40, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x[0], color='lime', linestyle='--', linewidth=2, label='x=0')
        ax.axvline(x[-1], color='lime', linestyle='--', linewidth=2, label='x=L')

        # 標示 5% 接縫區域
        ax.axvspan(x[0], x[0]+seam_threshold, alpha=0.1, color='green')
        ax.axvspan(x[-1]-seam_threshold, x[-1], alpha=0.1, color='green')

        ax.set_xlabel('x coordinate')
        ax.set_ylabel('Count')
        ax.set_title(f'x-distribution (weight={weight})')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Seam Weight Effect on Sensor Distribution (K={K}, n_wrap=2)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_dir = Path("results/seam_weight_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"seam_weight_comparison_K{K}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 視覺化已保存: {output_path}")

    # 統計摘要
    print("\n" + "=" * 70)
    print("📈 統計摘要")
    print("=" * 70)

    print(f"\n{'seam_weight':^12} | {'接縫覆蓋':^12} | {'條件數':^12} | {'能量比':^12}")
    print("-" * 70)

    for weight in seam_weights:
        result = results[weight]
        x_sensors = result['sensor_points'][:, 0]
        seam_threshold = 0.05 * (x[-1] - x[0])
        seam_count = np.sum((x_sensors < (x[0] + seam_threshold)) |
                           (x_sensors > (x[-1] - seam_threshold)))
        seam_pct = 100 * seam_count / K

        cond = result['selection_info'].get('condition_number', float('nan'))
        energy = result['selection_info'].get('energy_ratio', float('nan'))

        print(f"{weight:^12.2f} | {seam_count:3d}/{K} ({seam_pct:4.1f}%) | "
              f"{cond:^12.2f} | {energy:^12.4f}")

    # 總結建議
    print("\n" + "=" * 70)
    print("✅ 測試完成")
    print("=" * 70)

    print("\n💡 建議:")
    print("  - seam_weight=1.0: 無調整（原始問題：96%點集中在接縫）")
    print("  - seam_weight=0.5: 中等降權（平衡接縫與內部）")
    print("  - seam_weight=0.3: 強降權（推薦值，避免接縫過度選擇）")
    print("  - seam_weight=0.1: 極強降權（幾乎忽略接縫）")

    print(f"\n📁 輸出文件:")
    print(f"  視覺化: {output_path}")
    for weight in seam_weights:
        print(f"  seam_weight={weight}: test_seam_weight_output/sensors_K{K}_qr_pivot_periodic.npz")


if __name__ == "__main__":
    sys.exit(main())
