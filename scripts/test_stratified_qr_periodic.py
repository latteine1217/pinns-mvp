#!/usr/bin/env python3
"""
測試改進的週期性感測點選擇策略：Stratified QR-Pivot
解決問題：避免 QR-Pivot 在週期性邊界下過度集中於高能量區域
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.fetch_channel_flow import ChannelFlowDataFetcher, ChannelFlowConfig


def stratified_qr_periodic(
    field_data: dict,
    K: int,
    n_segments_x: int = 4,
    n_segments_z: int = 1,
    balance_segments: bool = True,
    min_distance_ratio: float = 0.05
):
    """
    改進的週期性 QR-Pivot：空間分層 + 距離約束

    參數
    ----
    field_data : dict
        包含 'x', 'y', 'u', 'v', 'w', 'p' 的場數據
    K : int
        總感測點數
    n_segments_x : int
        x 方向的分段數（週期方向）
    n_segments_z : int
        z 方向的分段數（週期方向，2D 時設為 1）
    balance_segments : bool
        是否在各 segment 間均勻分配點數（True），或按能量比例分配（False）
    min_distance_ratio : float
        最小距離比例（相對於域長），用於避免點聚集

    返回
    ----
    result : dict
        包含 'sensor_points', 'sensor_values', 'selection_info'
    """

    # 提取網格
    x, y = field_data['x'], field_data['y']
    u, v = field_data['u'], field_data['v']

    nx, ny = len(x), len(y)
    Lx = x[-1] - x[0]

    # 構建 2D snapshot matrix [n_locations, n_features]
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.column_stack([X.ravel(), Y.ravel()])

    # Stack all fields as features
    snapshots = np.column_stack([
        u.ravel(),
        v.ravel(),
        field_data.get('w', np.zeros_like(u)).ravel(),
        field_data.get('p', np.zeros_like(u)).ravel()
    ])

    # 標準化（避免不同物理量量級差異）
    snapshots = (snapshots - snapshots.mean(axis=0)) / (snapshots.std(axis=0) + 1e-8)

    # 計算各點的能量（用於非均勻分配）
    energy = np.sum(snapshots**2, axis=1)

    # === 空間分層 ===
    x_edges = np.linspace(x[0], x[-1], n_segments_x + 1)

    # 為每個點分配 segment ID
    segment_ids = np.digitize(coords[:, 0], x_edges) - 1
    segment_ids = np.clip(segment_ids, 0, n_segments_x - 1)

    # === 分配各 segment 的目標點數 ===
    if balance_segments:
        # 均勻分配
        K_per_segment = K // n_segments_x
        K_segments = np.full(n_segments_x, K_per_segment)
        # 剩餘點數分配給前幾個 segment
        remainder = K - K_per_segment * n_segments_x
        K_segments[:remainder] += 1
    else:
        # 按能量比例分配
        segment_energies = np.array([
            energy[segment_ids == seg_id].sum()
            for seg_id in range(n_segments_x)
        ])
        K_segments = (K * segment_energies / segment_energies.sum()).astype(int)
        # 確保總和為 K
        K_segments[0] += K - K_segments.sum()

    print(f"\n📊 Segment 點數分配:")
    for seg_id in range(n_segments_x):
        x_range = (x_edges[seg_id], x_edges[seg_id+1])
        print(f"  Segment {seg_id}: x ∈ [{x_range[0]:.2f}, {x_range[1]:.2f}], K={K_segments[seg_id]}")

    # === 在各 segment 內執行 QR-Pivot ===
    selected_indices_all = []

    for seg_id in range(n_segments_x):
        # 提取該 segment 的點
        mask = segment_ids == seg_id
        indices_in_segment = np.where(mask)[0]

        if len(indices_in_segment) == 0:
            print(f"⚠️  Segment {seg_id} 無點，跳過")
            continue

        K_target = K_segments[seg_id]
        if K_target == 0:
            continue

        X_segment = snapshots[indices_in_segment]

        # QR-Pivot 選擇
        from scipy.linalg import qr
        Q, R, piv = qr(X_segment.T, mode='economic', pivoting=True)

        # 選擇前 K_target 個
        K_actual = min(K_target, len(piv))
        selected_local = piv[:K_actual]

        # 映射回全局索引
        selected_global = indices_in_segment[selected_local]
        selected_indices_all.extend(selected_global)

        print(f"  ✅ Segment {seg_id}: 選擇 {K_actual}/{len(indices_in_segment)} 點")

    selected_indices = np.array(selected_indices_all)

    # === 提取結果 ===
    sensor_coords = coords[selected_indices]
    sensor_values = {
        'u': u.ravel()[selected_indices],
        'v': v.ravel()[selected_indices]
    }

    # 計算統計
    x_sensors = sensor_coords[:, 0]
    seam_threshold = 0.05 * Lx
    near_x0 = np.sum(x_sensors < (x[0] + seam_threshold))
    near_xL = np.sum(x_sensors > (x[-1] - seam_threshold))

    # 計算 x 方向的空間覆蓋度（均勻性）
    hist, _ = np.histogram(x_sensors, bins=n_segments_x, range=(x[0], x[-1]))
    uniformity = 1.0 - np.std(hist) / (np.mean(hist) + 1e-8)  # 越接近 1 越均勻

    selection_info = {
        'method': 'stratified_qr_periodic',
        'n_segments_x': n_segments_x,
        'balance_segments': balance_segments,
        'seam_coverage': (near_x0 + near_xL) / len(selected_indices),
        'uniformity': uniformity,
        'K_actual': len(selected_indices)
    }

    return {
        'sensor_points': sensor_coords,
        'sensor_values': sensor_values,
        'selection_info': selection_info,
        'indices': selected_indices
    }


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
    print("🧪 測試 Stratified QR-Pivot 週期性採樣策略")
    print("=" * 70)

    # 載入數據
    data_path = Path("data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz")
    if not data_path.exists():
        print(f"❌ 找不到數據: {data_path}")
        return 1

    field_data = extract_2d_slice(data_path)
    x, y = field_data['x'], field_data['y']

    K = 100

    # === 測試 1: 標準 QR-Pivot (seam_weight=0.3) ===
    print("\n" + "=" * 70)
    print("📌 方法 1: 標準 QR-Pivot (seam_weight=0.3)")
    print("=" * 70)

    config = ChannelFlowConfig()
    fetcher = ChannelFlowDataFetcher(config, cache_dir="./test_stratified_output")

    result_standard = fetcher.generate_sensor_points_with_periodicity(
        field_data, K=K,
        periodic_axes=[0],
        use_periodic=True,
        n_wrap_layers=2,
        seam_weight=0.3,
        noise_sigma=0.0,
        dropout_prob=0.0
    )

    x_std = result_standard['sensor_points'][:, 0]
    seam_threshold = 0.05 * (x[-1] - x[0])
    seam_count_std = np.sum((x_std < (x[0] + seam_threshold)) | (x_std > (x[-1] - seam_threshold)))

    print(f"  接縫覆蓋: {seam_count_std}/{K} ({100*seam_count_std/K:.1f}%)")
    print(f"  條件數: {result_standard['selection_info'].get('condition_number', 'N/A'):.2f}")

    # === 測試 2: Stratified QR (均勻分配) ===
    print("\n" + "=" * 70)
    print("📌 方法 2: Stratified QR (均勻分段, n_segments=4)")
    print("=" * 70)

    result_stratified = stratified_qr_periodic(
        field_data, K=K,
        n_segments_x=4,
        balance_segments=True
    )

    x_strat = result_stratified['sensor_points'][:, 0]
    seam_count_strat = np.sum((x_strat < (x[0] + seam_threshold)) | (x_strat > (x[-1] - seam_threshold)))

    print(f"  接縫覆蓋: {seam_count_strat}/{K} ({100*seam_count_strat/K:.1f}%)")
    print(f"  空間均勻性: {result_stratified['selection_info']['uniformity']:.3f}")

    # === 測試 3: Stratified QR (能量分配) ===
    print("\n" + "=" * 70)
    print("📌 方法 3: Stratified QR (能量加權分段, n_segments=4)")
    print("=" * 70)

    result_energy = stratified_qr_periodic(
        field_data, K=K,
        n_segments_x=4,
        balance_segments=False
    )

    x_energy = result_energy['sensor_points'][:, 0]
    seam_count_energy = np.sum((x_energy < (x[0] + seam_threshold)) | (x_energy > (x[-1] - seam_threshold)))

    print(f"  接縫覆蓋: {seam_count_energy}/{K} ({100*seam_count_energy/K:.1f}%)")
    print(f"  空間均勻性: {result_energy['selection_info']['uniformity']:.3f}")

    # === 視覺化比較 ===
    print("\n📊 生成視覺化比較...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    X, Y = np.meshgrid(x, y, indexing='ij')

    methods = [
        ('Standard QR-Pivot\n(seam_weight=0.3)', result_standard, seam_count_std),
        ('Stratified QR\n(Uniform)', result_stratified, seam_count_strat),
        ('Stratified QR\n(Energy-weighted)', result_energy, seam_count_energy)
    ]

    for idx, (title, result, seam_count) in enumerate(methods):
        sensor_points = result['sensor_points']

        # 上排：速度場 + 感測點
        ax = axes[0, idx]
        im = ax.contourf(X, Y, field_data['u'], levels=50, cmap='RdBu_r', alpha=0.7)
        ax.scatter(sensor_points[:, 0], sensor_points[:, 1],
                  c='black', s=20, marker='o', alpha=0.8, edgecolors='yellow', linewidths=0.5)

        # 標示週期邊界
        ax.axvline(x[0], color='lime', linestyle='--', linewidth=1.5, alpha=0.7, label='Periodic boundary')
        ax.axvline(x[-1], color='lime', linestyle='--', linewidth=1.5, alpha=0.7)

        # 標示 5% 接縫區域
        ax.axvspan(x[0], x[0]+seam_threshold, alpha=0.15, color='green')
        ax.axvspan(x[-1]-seam_threshold, x[-1], alpha=0.15, color='green')

        ax.set_title(f'{title}\nSeam: {seam_count}/{K} ({100*seam_count/K:.1f}%)', fontsize=11)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # 下排：x 方向分佈直方圖
        ax = axes[1, idx]
        ax.hist(sensor_points[:, 0], bins=40, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(x[0], color='lime', linestyle='--', linewidth=2, label='x=0')
        ax.axvline(x[-1], color='lime', linestyle='--', linewidth=2, label='x=L')
        ax.axvspan(x[0], x[0]+seam_threshold, alpha=0.1, color='green')
        ax.axvspan(x[-1]-seam_threshold, x[-1], alpha=0.1, color='green')

        # 計算並顯示均勻性指標
        if 'uniformity' in result['selection_info']:
            uniformity = result['selection_info']['uniformity']
            ax.text(0.98, 0.95, f'Uniformity: {uniformity:.3f}',
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel('x coordinate')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Comparison of Sensor Placement Strategies (K={K})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # 保存
    output_dir = Path("results/stratified_qr_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"stratified_vs_standard_K{K}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 視覺化已保存: {output_path}")

    # === 統計摘要 ===
    print("\n" + "=" * 70)
    print("📈 統計摘要")
    print("=" * 70)

    print(f"\n{'Method':^30} | {'接縫覆蓋':^12} | {'均勻性':^10}")
    print("-" * 70)

    for title, result, seam_count in methods:
        title_clean = title.replace('\n', ' ')
        uniformity = result['selection_info'].get('uniformity', float('nan'))
        print(f"{title_clean:^30} | {seam_count:3d}/{K} ({100*seam_count/K:4.1f}%) | {uniformity:^10.3f}")

    print("\n" + "=" * 70)
    print("✅ 測試完成")
    print("=" * 70)

    print("\n💡 結論:")
    print("  1️⃣  標準 QR-Pivot (seam_weight=0.3): 低接縫覆蓋，但空間分佈不均")
    print("  2️⃣  Stratified QR (均勻): 強制空間均勻性，適用於需要全域覆蓋的場景")
    print("  3️⃣  Stratified QR (能量): 平衡空間覆蓋與物理重要性，推薦用於 PINNs 訓練")

    return 0


if __name__ == "__main__":
    sys.exit(main())
