#!/usr/bin/env python3
"""
診斷週期性 QR-Pivot 選中的實際索引
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.sensors.qr_pivot import QRPivotSelector


def main():
    print("=" * 70)
    print("🔍 診斷週期性 QR-Pivot 索引映射")
    print("=" * 70)

    # 載入真實數據
    data = np.load('data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz')
    x, y, z = data['x'], data['y'], data['z']
    nx, ny, nz = len(x), len(y), len(z)

    # 提取 2D 切片
    z_idx = 64
    u_3d = data['u'].reshape(nx, ny, nz)
    v_3d = data['v'].reshape(nx, ny, nz)
    w_3d = data['w'].reshape(nx, ny, nz)
    p_3d = data['p'].reshape(nx, ny, nz)

    u_2d = u_3d[:, :, z_idx]
    v_2d = v_3d[:, :, z_idx]
    w_2d = w_3d[:, :, z_idx]
    p_2d = p_3d[:, :, z_idx]

    # 建立座標與快照矩陣
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.ravel(), Y.ravel()], axis=-1)

    snapshots = []
    for field in [u_2d, v_2d, w_2d, p_2d]:
        snapshots.append(field.ravel()[:, np.newaxis])
    data_matrix = np.hstack(snapshots)

    print(f"\n📊 數據信息:")
    print(f"  網格: {nx} × {ny} = {nx*ny} 點")
    print(f"  x 範圍: [{x[0]:.4f}, {x[-1]:.4f}]")
    print(f"  快照矩陣: {data_matrix.shape}")

    # 創建週期性選擇器（啟用詳細日誌）
    K = 100
    n_wrap_layers = 2
    periodic_axes = [0]

    selector = QRPivotSelector(
        use_circular_indexing=True,
        n_wrap_layers=n_wrap_layers,
        mode='column',
        pivoting=True
    )

    domain_lengths = {0: float(x[-1] - x[0])}

    print(f"\n🔧 選擇器配置:")
    print(f"  use_circular_indexing: True")
    print(f"  n_wrap_layers: {n_wrap_layers}")
    print(f"  periodic_axes: {periodic_axes}")
    print(f"  domain_lengths: {domain_lengths}")

    # ===== 關鍵：手動執行環繞過程以獲取中間結果 =====
    from pinnx.sensors.qr_pivot import build_circular_snapshot_matrix

    # 建立增強網格
    augmented_snapshots, augmented_coords = build_circular_snapshot_matrix(
        snapshots=data_matrix,
        coords=coords,
        grid_shape=(nx, ny),
        periodic_axes=periodic_axes,
        n_wrap_layers=n_wrap_layers,
        domain_lengths=domain_lengths
    )

    print(f"\n📈 增強網格:")
    print(f"  原始: {data_matrix.shape[0]} 點")
    print(f"  增強: {augmented_snapshots.shape[0]} 點")
    print(f"  增強座標範圍:")
    print(f"    x: [{augmented_coords[:, 0].min():.4f}, {augmented_coords[:, 0].max():.4f}]")
    print(f"    y: [{augmented_coords[:, 1].min():.4f}, {augmented_coords[:, 1].max():.4f}]")

    # 執行 QR-Pivot（在增強網格上，不使用角度嵌入）
    # 注意：這裡我們已經手動建立了增強網格，所以直接傳入
    # 不需要再次觸發 build_circular_snapshot_matrix

    # 使用基礎 QR-Pivot（不啟用循環索引，因為我們已經手動處理了）
    from scipy.linalg import qr

    # 對增強快照執行 QR 分解
    Q, R, P = qr(augmented_snapshots.T, pivoting=True, mode='economic')
    indices_augmented = P[:K]

    metrics = {'condition_number': np.linalg.cond(augmented_snapshots[indices_augmented, :])}

    print(f"\n🎯 QR-Pivot 結果（增強網格）:")
    print(f"  選中索引範圍: [{indices_augmented.min()}, {indices_augmented.max()}]")
    print(f"  選中座標 x 範圍: [{augmented_coords[indices_augmented, 0].min():.4f}, "
          f"{augmented_coords[indices_augmented, 0].max():.4f}]")

    # 分析增強索引的分佈
    nx_aug = nx + 2 * n_wrap_layers
    print(f"\n📊 增強索引分佈分析:")
    print(f"  增強網格 x 維度: {nx_aug} (原始 {nx} + 環繞 {2*n_wrap_layers})")

    # 將 2D 索引拆解為 (i, j)
    indices_i_aug = indices_augmented // ny  # x 方向索引
    indices_j_aug = indices_augmented % ny   # y 方向索引

    print(f"\n  增強 x 索引統計:")
    print(f"    範圍: [{indices_i_aug.min()}, {indices_i_aug.max()}]")
    print(f"    環繞前端 [0, {n_wrap_layers-1}]: {np.sum(indices_i_aug < n_wrap_layers)} 點")
    print(f"    原始區域 [{n_wrap_layers}, {n_wrap_layers+nx-1}]: "
          f"{np.sum((indices_i_aug >= n_wrap_layers) & (indices_i_aug < n_wrap_layers+nx))} 點")
    print(f"    環繞後端 [{n_wrap_layers+nx}, {nx_aug-1}]: "
          f"{np.sum(indices_i_aug >= n_wrap_layers+nx)} 點")

    # 映射回原始索引
    def map_augmented_to_original_x(i_aug, nx, n_wrap):
        if i_aug < n_wrap:
            return nx - n_wrap + i_aug
        elif i_aug < n_wrap + nx:
            return i_aug - n_wrap
        else:
            return i_aug - n_wrap - nx

    indices_i_orig = np.array([map_augmented_to_original_x(i, nx, n_wrap_layers)
                                for i in indices_i_aug])

    print(f"\n  映射回原始 x 索引:")
    print(f"    範圍: [{indices_i_orig.min()}, {indices_i_orig.max()}]")

    # 原始 x 座標
    x_coords_orig = x[indices_i_orig]
    print(f"    對應 x 座標: [{x_coords_orig.min():.4f}, {x_coords_orig.max():.4f}]")

    # 分段統計原始索引
    bins_x = [0, 32, 64, 96, 128]
    hist_x, _ = np.histogram(indices_i_orig, bins=bins_x)
    print(f"\n  原始 x 索引分佈:")
    for i in range(len(bins_x)-1):
        pct = 100 * hist_x[i] / K
        print(f"    [{bins_x[i]:3d}, {bins_x[i+1]:3d}): {hist_x[i]:3d} 點 ({pct:5.1f}%)")

    # 視覺化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 增強索引分佈
    ax = axes[0, 0]
    ax.hist(indices_i_aug, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(n_wrap_layers, color='red', linestyle='--', label=f'原始區域開始 (i={n_wrap_layers})')
    ax.axvline(n_wrap_layers+nx, color='red', linestyle='--', label=f'原始區域結束 (i={n_wrap_layers+nx})')
    ax.set_xlabel('Augmented x-index')
    ax.set_ylabel('Count')
    ax.set_title('QR-Pivot選中的增強索引分佈')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 原始索引分佈
    ax = axes[0, 1]
    ax.hist(indices_i_orig, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax.set_xlabel('Original x-index')
    ax.set_ylabel('Count')
    ax.set_title('映射回原始索引的分佈')
    ax.grid(True, alpha=0.3)

    # x 座標分佈
    ax = axes[1, 0]
    ax.hist(x_coords_orig, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.axvline(x[0], color='lime', linestyle='--', linewidth=2, label='x=0 (seam)')
    ax.axvline(x[-1], color='lime', linestyle='--', linewidth=2, label='x=L (seam)')
    ax.set_xlabel('x coordinate')
    ax.set_ylabel('Count')
    ax.set_title('最終感測點 x 座標分佈')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 索引映射關係
    ax = axes[1, 1]
    ax.scatter(indices_i_aug, indices_i_orig, alpha=0.5, s=20)
    ax.plot([0, nx_aug], [0, nx], 'r--', label='y=x-2 (理想映射)')
    ax.set_xlabel('Augmented x-index')
    ax.set_ylabel('Original x-index')
    ax.set_title('索引映射關係')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('results/periodic_sensor_comparison/index_mapping_debug.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ 診斷圖表已保存: {output_path}")

    # 總結
    print("\n" + "=" * 70)
    print("🔍 診斷結論")
    print("=" * 70)

    original_region_count = np.sum((indices_i_aug >= n_wrap_layers) &
                                    (indices_i_aug < n_wrap_layers+nx))
    seam_count = K - original_region_count

    print(f"\n增強網格選點分佈:")
    print(f"  原始區域內: {original_region_count}/{K} ({100*original_region_count/K:.1f}%)")
    print(f"  環繞層內: {seam_count}/{K} ({100*seam_count/K:.1f}%)")

    if original_region_count > 0.9 * K:
        print(f"\n⚠️ 問題確認：{100*original_region_count/K:.1f}% 的點選在原始區域內")
        print("   這導致映射後感測點集中在中間，失去週期性覆蓋優勢")
        print("\n可能原因:")
        print("  1. 環繞層數太少（n_wrap_layers=2 可能不足）")
        print("  2. QR-Pivot 仍偏好能量集中區域（物理場特性）")
        print("  3. 缺少接縫區域的顯式權重增強")


if __name__ == "__main__":
    main()
