#!/usr/bin/env python3
"""
比較標準 QR-Pivot 與循環索引模式的感測點分佈

生成：
1. 標準模式感測點分佈
2. 循環索引模式感測點分佈
3. 對比圖（接縫處聚集分析）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy.stats import ks_2samp

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector, build_circular_snapshot_matrix

# 設定
np.random.seed(42)
output_dir = Path("results/circular_indexing_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("比較標準 vs 循環索引 QR-Pivot 感測點分佈")
print("=" * 80)

# 建立模擬通道流資料（x/z 週期）
grid_shape = (32, 16, 32)  # (nx, ny, nz) - 較小網格便於視覺化
n_total = np.prod(grid_shape)
n_snapshots = 10
n_sensors = 500

# 建立座標網格
x = np.linspace(0, 2*np.pi, grid_shape[0])
y = np.linspace(-1, 1, grid_shape[1])
z = np.linspace(0, 2*np.pi, grid_shape[2])
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)

# 建立快照矩陣（模擬湍流特徵）
# 基礎隨機場
snapshots = np.random.randn(n_total, n_snapshots) * 0.5

# 添加物理相關的空間結構（模擬湍流渦旋）
for i in range(n_snapshots):
    # 大尺度結構（波數 k=1,2）
    kx, kz = 1, 1
    phase = np.sin(kx * coords[:, 0]) * np.cos(kz * coords[:, 2])
    snapshots[:, i] += phase * 0.3

    # 中尺度結構（波數 k=2,3）
    kx, kz = 2, 3
    phase = np.cos(kx * coords[:, 0]) * np.sin(kz * coords[:, 2])
    snapshots[:, i] += phase * 0.2

# 添加溫和的接縫誤差（模擬數值離散化/插值誤差）
seam_threshold = 0.15  # 接縫區域縮小（約 2.4% 範圍）
seam_x = (coords[:, 0] < seam_threshold) | (coords[:, 0] > 2*np.pi - seam_threshold)
seam_z = (coords[:, 2] < seam_threshold) | (coords[:, 2] > 2*np.pi - seam_threshold)
seam_mask = seam_x | seam_z

# 模擬三種真實誤差：
# 1. 隨機噪聲（測量誤差）
seam_noise = np.random.randn(seam_mask.sum(), n_snapshots) * 0.05
snapshots[seam_mask, :] += seam_noise

# 2. 系統性偏差（插值誤差）
snapshots[seam_mask, :] *= (1.0 + 0.03)  # 3% 系統偏差

# 3. 高頻振盪（數值耗散不匹配）
high_freq = np.sin(10 * coords[seam_mask, 0][:, np.newaxis]) * 0.02
snapshots[seam_mask, :] += high_freq

print(f"\n建立模擬資料：")
print(f"  網格形狀: {grid_shape}")
print(f"  總點數: {n_total}")
print(f"  快照數: {n_snapshots}")
print(f"  目標感測點數: {n_sensors}")
print(f"  接縫處點數: {seam_mask.sum()} ({100*seam_mask.sum()/n_total:.1f}%)")

# ============================================================================
# 1. 標準模式
# ============================================================================
print(f"\n{'='*80}")
print("1. 標準 QR-Pivot 模式")
print("=" * 80)

selector_std = QRPivotSelector(use_circular_indexing=False)
indices_std, metrics_std = selector_std.select_sensors(snapshots, n_sensors)

coords_std = coords[indices_std]
print(f"  選中點數: {len(indices_std)}")
print(f"  條件數: {metrics_std['condition_number']:.2f}")
print(f"  能量比例: {metrics_std['energy_ratio']:.3f}")
seam_selected_std = metrics_std.get('seam_selected_count', seam_mask[indices_std].sum())
seam_ratio_std = metrics_std.get('seam_selected_ratio', seam_selected_std / len(indices_std))
print(f"  接縫處選中點數: {seam_selected_std} ({100*seam_ratio_std:.1f}%)")

# ============================================================================
# 2. 循環索引模式
# ============================================================================
print(f"\n{'='*80}")
print("2. 循環索引模式")
print("=" * 80)

selector_circ = QRPivotSelector(
    use_circular_indexing=True,
    n_wrap_layers=2,
    seam_weight=0.1,
    seam_width_fraction=0.03
)

indices_circ, metrics_circ = selector_circ.select_sensors(
    data_matrix=snapshots,
    n_sensors=n_sensors,
    coords=coords,
    grid_shape=grid_shape,
    periodic_axes=[0, 2],  # x/z 週期
    domain_lengths={0: 2*np.pi, 2: 2*np.pi}
)

coords_circ = coords[indices_circ]
print(f"  選中點數: {len(indices_circ)}")
print(f"  條件數: {metrics_circ['condition_number']:.2f}")
print(f"  能量比例: {metrics_circ['energy_ratio']:.3f}")
print(f"  去重數量: {metrics_circ['n_duplicates_removed']}")
# 分析接縫聚集
seam_selected_circ = metrics_circ.get('seam_selected_count', seam_mask[indices_circ].sum())
seam_ratio_circ = metrics_circ.get('seam_selected_ratio', seam_selected_circ / len(indices_circ))
print(f"  接縫處選中點數: {seam_selected_circ} ({100*seam_ratio_circ:.1f}%)")
if 'fallback_interior_added' in metrics_circ:
    print(f"  內部回補點數: {metrics_circ['fallback_interior_added']}")
if 'seam_cap_residual' in metrics_circ:
    print(f"  接縫限制殘量: {metrics_circ['seam_cap_residual']}")


# ============================================================================
# 3. 視覺化對比
# ============================================================================
print(f"\n{'='*80}")
print("3. 生成視覺化對比圖")
print("=" * 80)

fig = plt.figure(figsize=(18, 12))

# --- 子圖 1: XZ 平面分佈（標準模式）---
ax1 = fig.add_subplot(2, 3, 1)
ax1.scatter(coords[:, 0], coords[:, 2], c='lightgray', s=5, alpha=0.3, label='All Points')
ax1.scatter(coords_std[:, 0], coords_std[:, 2], c='red', s=50, alpha=0.7,
           edgecolors='darkred', linewidths=1, label='Selected')
# 標記接縫區域（更新為溫和範圍）
seam_threshold_display = 0.15
ax1.axvspan(0, seam_threshold_display, alpha=0.1, color='blue', label='Seam Zone')
ax1.axvspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax1.axhspan(0, seam_threshold_display, alpha=0.1, color='blue')
ax1.axhspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('z', fontsize=12)
ax1.set_title(f'Standard Mode (Seam Ratio {100*seam_ratio_std:.1f}%)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# --- 子圖 2: XZ 平面分佈（循環索引）---
ax2 = fig.add_subplot(2, 3, 2)
ax2.scatter(coords[:, 0], coords[:, 2], c='lightgray', s=5, alpha=0.3, label='All Points')
ax2.scatter(coords_circ[:, 0], coords_circ[:, 2], c='green', s=50, alpha=0.7,
           edgecolors='darkgreen', linewidths=1, label='Selected')
ax2.axvspan(0, seam_threshold_display, alpha=0.1, color='blue', label='Seam Zone')
ax2.axvspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax2.axhspan(0, seam_threshold_display, alpha=0.1, color='blue')
ax2.axhspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('z', fontsize=12)
ax2.set_title(f'Circular Indexing (Seam Ratio {100*seam_ratio_circ:.1f}%)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# --- 子圖 3: 改善比例條形圖 ---
ax3 = fig.add_subplot(2, 3, 3)
improvement = (seam_ratio_std - seam_ratio_circ) / seam_ratio_std * 100 if seam_ratio_std > 0 else 0
bars = ax3.bar(['Standard', 'Circular'],
              [seam_ratio_std * 100, seam_ratio_circ * 100],
              color=['red', 'green'], alpha=0.7, edgecolor='black')
ax3.axhline(seam_mask.sum()/n_total * 100, color='blue', linestyle='--',
           linewidth=2, label='Global Seam Ratio')
ax3.set_ylabel('Seam Point Ratio (%)', fontsize=12)
ax3.set_title(f'Seam Clustering Improvement {improvement:.1f}%', fontsize=14, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3, axis='y')
for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- 子圖 4: x 座標直方圖（標準）---
ax4 = fig.add_subplot(2, 3, 4)
ax4.hist(coords_std[:, 0], bins=20, alpha=0.7, color='red', edgecolor='black', label='Selected Points')
ax4.axvline(0.3, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax4.axvline(2*np.pi-0.3, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Seam Boundary')
ax4.set_xlabel('x Coordinate', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Standard Mode x Distribution', fontsize=13, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')

# --- 子圖 5: x 座標直方圖（循環）---
ax5 = fig.add_subplot(2, 3, 5)
ax5.hist(coords_circ[:, 0], bins=20, alpha=0.7, color='green', edgecolor='black', label='Selected Points')
ax5.axvline(0.3, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax5.axvline(2*np.pi-0.3, color='blue', linestyle='--', linewidth=2, alpha=0.5, label='Seam Boundary')
ax5.set_xlabel('x Coordinate', fontsize=12)
ax5.set_ylabel('Count', fontsize=12)
ax5.set_title('Circular Indexing x Distribution', fontsize=13, fontweight='bold')
ax5.legend(fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

# --- 子圖 6: 指標對比雷達圖 ---
ax6 = fig.add_subplot(2, 3, 6, projection='polar')

# 準備數據（歸一化到 [0, 1]）
categories = ['Condition\nNumber', 'Energy\nRatio', 'Spatial\nUniformity']
N = len(categories)

# 歸一化指標
cond_norm_std = 1.0 / (1.0 + metrics_std['condition_number'] / 1000)
cond_norm_circ = 1.0 / (1.0 + metrics_circ['condition_number'] / 1000)

uniformity_std = 1.0 - seam_ratio_std
uniformity_circ = 1.0 - seam_ratio_circ

values_std = [cond_norm_std, metrics_std['energy_ratio'], uniformity_std]
values_circ = [cond_norm_circ, metrics_circ['energy_ratio'], uniformity_circ]

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
values_std += values_std[:1]
values_circ += values_circ[:1]
angles += angles[:1]

ax6.plot(angles, values_std, 'o-', linewidth=2, color='red', label='Standard')
ax6.fill(angles, values_std, alpha=0.25, color='red')
ax6.plot(angles, values_circ, 'o-', linewidth=2, color='green', label='Circular')
ax6.fill(angles, values_circ, alpha=0.25, color='green')

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(categories, fontsize=10)
ax6.set_ylim(0, 1)
ax6.set_title('Quality Metrics Comparison', fontsize=13, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
ax6.grid(True)

plt.suptitle('QR-Pivot: Circular Indexing vs Standard Mode',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

output_file = output_dir / 'circular_indexing_comparison.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"  ✅ 已保存: {output_file}")

# ============================================================================
# 3b. 生成 XY 平面對比圖
# ============================================================================
print(f"\n{'='*80}")
print("3b. 生成 XY 平面視覺化圖")
print("=" * 80)

fig_xy = plt.figure(figsize=(18, 6))

# --- 子圖 1: XY 平面分佈（標準模式）---
ax_xy1 = fig_xy.add_subplot(1, 3, 1)
ax_xy1.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=5, alpha=0.3, label='All Points')
ax_xy1.scatter(coords_std[:, 0], coords_std[:, 1], c='red', s=50, alpha=0.7,
               edgecolors='darkred', linewidths=1, label='Selected')
ax_xy1.axvspan(0, seam_threshold_display, alpha=0.1, color='blue', label='Seam Zone')
ax_xy1.axvspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax_xy1.set_xlabel('x', fontsize=12)
ax_xy1.set_ylabel('y', fontsize=12)
ax_xy1.set_title('Standard Mode (XY Plane)', fontsize=14, fontweight='bold')
ax_xy1.legend(loc='upper right', fontsize=10)
ax_xy1.grid(True, alpha=0.3)

# --- 子圖 2: XY 平面分佈（循環索引）---
ax_xy2 = fig_xy.add_subplot(1, 3, 2)
ax_xy2.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=5, alpha=0.3, label='All Points')
ax_xy2.scatter(coords_circ[:, 0], coords_circ[:, 1], c='green', s=50, alpha=0.7,
               edgecolors='darkgreen', linewidths=1, label='Selected')
ax_xy2.axvspan(0, seam_threshold_display, alpha=0.1, color='blue', label='Seam Zone')
ax_xy2.axvspan(2*np.pi-seam_threshold_display, 2*np.pi, alpha=0.1, color='blue')
ax_xy2.set_xlabel('x', fontsize=12)
ax_xy2.set_ylabel('y', fontsize=12)
ax_xy2.set_title('Circular Indexing (XY Plane)', fontsize=14, fontweight='bold')
ax_xy2.legend(loc='upper right', fontsize=10)
ax_xy2.grid(True, alpha=0.3)

# --- 子圖 3: x 座標分佈對比 ---
ax_xy3 = fig_xy.add_subplot(1, 3, 3)
bins = np.linspace(0, 2*np.pi, 31)
ax_xy3.hist(coords_std[:, 0], bins=bins, alpha=0.6, color='red', edgecolor='black', label='Standard')
ax_xy3.hist(coords_circ[:, 0], bins=bins, alpha=0.6, color='green', edgecolor='black', label='Circular')
ax_xy3.axvline(0, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax_xy3.axvline(2*np.pi, color='blue', linestyle='--', linewidth=1, alpha=0.5)
ax_xy3.set_xlabel('x Coordinate', fontsize=12)
ax_xy3.set_ylabel('Count', fontsize=12)
ax_xy3.set_title('X Distribution Comparison', fontsize=13, fontweight='bold')
ax_xy3.legend(fontsize=10)
ax_xy3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_file_xy = output_dir / 'circular_indexing_xy_plane.png'
plt.savefig(output_file_xy, dpi=150, bbox_inches='tight')
print(f"  ✅ 已保存: {output_file_xy}")

# ============================================================================
# 4. 詳細統計
# ============================================================================
print(f"\n{'='*80}")
print("4. 詳細統計分析")
print("=" * 80)

print(f"\n標準模式統計：")
print(f"  x 座標 - mean: {coords_std[:, 0].mean():.3f}, std: {coords_std[:, 0].std():.3f}")
print(f"  z 座標 - mean: {coords_std[:, 2].mean():.3f}, std: {coords_std[:, 2].std():.3f}")

print(f"\n循環索引統計：")
print(f"  x 座標 - mean: {coords_circ[:, 0].mean():.3f}, std: {coords_circ[:, 0].std():.3f}")
print(f"  z 座標 - mean: {coords_circ[:, 2].mean():.3f}, std: {coords_circ[:, 2].std():.3f}")

# x 分布差異統計
quantiles_target = [0.25, 0.5, 0.75]
quantiles_std = np.quantile(coords_std[:, 0], quantiles_target)
quantiles_circ = np.quantile(coords_circ[:, 0], quantiles_target)
ks_stat, ks_p = ks_2samp(coords_std[:, 0], coords_circ[:, 0])

print(f"\nX 分佈比較：")
print(f"  KS 統計量: {ks_stat:.3f} (p-value {ks_p:.3e})")
print(f"  分位數 25/50/75 (Standard): {quantiles_std[0]:.3f}, {quantiles_std[1]:.3f}, {quantiles_std[2]:.3f}")
print(f"  分位數 25/50/75 (Circular): {quantiles_circ[0]:.3f}, {quantiles_circ[1]:.3f}, {quantiles_circ[2]:.3f}")

print(f"\n改善指標：")
print(f"  接縫聚集減少: {improvement:.1f}%")
print(f"  條件數變化: {metrics_circ['condition_number'] / metrics_std['condition_number']:.2f}x")

print(f"\n{'='*80}")
print(f"✅ 比較完成！結果已保存至 {output_dir}")
print("=" * 80)
