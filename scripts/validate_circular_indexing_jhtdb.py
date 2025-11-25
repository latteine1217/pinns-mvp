#!/usr/bin/env python3
"""
使用真實 JHTDB 通道流數據驗證循環索引效果

測試內容：
1. 載入 JHTDB cutout 數據（128x64x128）
2. 比較標準 vs 循環索引的感測點分佈
3. 分析接縫處聚集改善
4. 評估物理場重建品質
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector

# ============================================================================
# 配置
# ============================================================================

parser = argparse.ArgumentParser(description="驗證循環索引（JHTDB 數據）")
parser.add_argument('--data', type=str,
                   default='data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz',
                   help='JHTDB 數據路徑')
parser.add_argument('--n-sensors', type=int, default=50,
                   help='感測點數量')
parser.add_argument('--n-wrap-layers', type=int, default=2,
                   help='環繞層數')
parser.add_argument('--output', type=str,
                   default='results/jhtdb_circular_validation',
                   help='輸出目錄')
args = parser.parse_args()

output_dir = Path(args.output)
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("JHTDB 通道流數據：循環索引驗證")
print("=" * 80)
print(f"數據路徑: {args.data}")
print(f"感測點數: {args.n_sensors}")
print(f"環繞層數: {args.n_wrap_layers}")

# ============================================================================
# 1. 載入 JHTDB 數據
# ============================================================================
print(f"\n{'='*80}")
print("1. 載入 JHTDB 通道流數據")
print("=" * 80)

data = np.load(args.data)
print(f"  可用鍵: {list(data.keys())}")

# 提取座標與速度場
x = data['x']
y = data['y']
z = data.get('z', None)  # 2D 切片可能沒有 z

# 檢查數據形狀
if 'u' in data:
    u = data['u']
    v = data['v']
    w = data.get('w', None)

    # 判斷數據格式
    if u.ndim == 4:
        # [nx, ny, nz, nt] 格式
        n_time = u.shape[3]
        print(f"  檢測到時間序列數據：{u.shape}")
        print(f"  使用前 10 個時間步作為快照")

        # 取前 10 個時間步
        n_snapshots = min(10, n_time)
        u_snapshots = u[:, :, :, :n_snapshots]
        v_snapshots = v[:, :, :, :n_snapshots]
        w_snapshots = w[:, :, :, :n_snapshots]

        # 展平空間維度 [nx*ny*nz, nt]
        grid_shape = u.shape[:3]
        n_locations = np.prod(grid_shape)

        u_flat = u_snapshots.reshape(n_locations, n_snapshots)
        v_flat = v_snapshots.reshape(n_locations, n_snapshots)
        w_flat = w_snapshots.reshape(n_locations, n_snapshots)

        # 建立座標網格 (3D)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        periodic_axes = [0, 2]  # x, z 週期
        domain_lengths = {0: x.max() - x.min(), 2: z.max() - z.min()}

    elif u.ndim == 3:
        # [nx, ny, nz] 格式（單時間步 3D）
        print(f"  檢測到單時間步 3D 數據：{u.shape}")
        grid_shape = u.shape
        n_locations = np.prod(grid_shape)

        # 建立多快照（添加小擾動）
        n_snapshots = 10
        u_flat = u.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01
        v_flat = v.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01
        w_flat = w.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01

        # 建立座標網格 (3D)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        periodic_axes = [0, 2]  # x, z 週期
        domain_lengths = {0: x.max() - x.min(), 2: z.max() - z.min()}

    elif u.ndim == 2:
        # [nx, ny] 格式（2D 切片）
        print(f"  檢測到 2D 切片數據：{u.shape}")
        grid_shape = u.shape
        n_locations = np.prod(grid_shape)

        # 建立多快照（使用 u, v 作為不同特徵）
        n_snapshots = 5
        u_flat = u.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01
        v_flat = v.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01

        if w is not None and w.ndim == 2:
            w_flat = w.flatten()[:, np.newaxis] + np.random.randn(n_locations, n_snapshots) * 0.01
        else:
            w_flat = np.zeros_like(u_flat)

        # 建立座標網格 (2D)
        X, Y = np.meshgrid(x, y, indexing='ij')
        coords = np.stack([X.flatten(), Y.flatten()], axis=1)
        periodic_axes = [0]  # 僅 x 週期
        domain_lengths = {0: x.max() - x.min()}

    else:
        raise ValueError(f"不支援的數據形狀：{u.shape}")
else:
    raise ValueError("數據中缺少速度場 'u', 'v'")

# 組合快照矩陣 [n_locations, 3*n_snapshots]
snapshots = np.hstack([u_flat, v_flat, w_flat])

print(f"  網格形狀: {grid_shape}")
print(f"  總點數: {n_locations}")
print(f"  座標範圍:")
print(f"    x: [{x.min():.3f}, {x.max():.3f}]")
print(f"    y: [{y.min():.3f}, {y.max():.3f}]")
if coords.shape[1] == 3:
    print(f"    z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")
print(f"  快照矩陣: {snapshots.shape}")

# 計算域長度
Lx = x.max() - x.min()
Ly = y.max() - y.min()

if coords.shape[1] == 3:
    # 3D
    Lz = coords[:, 2].max() - coords[:, 2].min()
    print(f"  域長度: Lx={Lx:.3f}, Ly={Ly:.3f}, Lz={Lz:.3f}")

    # 定義接縫區域（週期軸兩端 10% 範圍）
    seam_threshold_x = 0.1 * Lx
    seam_threshold_z = 0.1 * Lz

    seam_mask = (
        (coords[:, 0] < x.min() + seam_threshold_x) |
        (coords[:, 0] > x.max() - seam_threshold_x) |
        (coords[:, 2] < coords[:, 2].min() + seam_threshold_z) |
        (coords[:, 2] > coords[:, 2].max() - seam_threshold_z)
    )

    print(f"  接縫區域定義: x 兩端各 {seam_threshold_x:.3f}, z 兩端各 {seam_threshold_z:.3f}")
else:
    # 2D
    print(f"  域長度: Lx={Lx:.3f}, Ly={Ly:.3f}")

    # 定義接縫區域（僅 x 軸週期）
    seam_threshold_x = 0.1 * Lx

    seam_mask = (
        (coords[:, 0] < x.min() + seam_threshold_x) |
        (coords[:, 0] > x.max() - seam_threshold_x)
    )

    print(f"  接縫區域定義: x 兩端各 {seam_threshold_x:.3f}")

print(f"  接縫處點數: {seam_mask.sum()} ({100*seam_mask.sum()/n_locations:.1f}%)")

# ============================================================================
# 2. 標準 QR-Pivot
# ============================================================================
print(f"\n{'='*80}")
print("2. 標準 QR-Pivot 模式")
print("=" * 80)

selector_std = QRPivotSelector(use_circular_indexing=False)
indices_std, metrics_std = selector_std.select_sensors(snapshots, args.n_sensors)

coords_std = coords[indices_std]
seam_selected_std = seam_mask[indices_std].sum()
seam_ratio_std = seam_selected_std / len(indices_std)

print(f"  選中點數: {len(indices_std)}")
print(f"  條件數: {metrics_std['condition_number']:.2f}")
print(f"  能量比例: {metrics_std['energy_ratio']:.4f}")
print(f"  子空間覆蓋率: {metrics_std['subspace_coverage']:.4f}")
print(f"  接縫處選中: {seam_selected_std} ({100*seam_ratio_std:.1f}%)")

# ============================================================================
# 3. 循環索引模式
# ============================================================================
print(f"\n{'='*80}")
print("3. 循環索引模式")
print("=" * 80)

selector_circ = QRPivotSelector(
    use_circular_indexing=True,
    n_wrap_layers=args.n_wrap_layers
)

indices_circ, metrics_circ = selector_circ.select_sensors(
    data_matrix=snapshots,
    n_sensors=args.n_sensors,
    coords=coords,
    grid_shape=grid_shape,
    periodic_axes=[0, 2],  # x/z 週期
    domain_lengths=domain_lengths
)

coords_circ = coords[indices_circ]
seam_selected_circ = seam_mask[indices_circ].sum()
seam_ratio_circ = seam_selected_circ / len(indices_circ)

print(f"  選中點數: {len(indices_circ)}")
print(f"  條件數: {metrics_circ['condition_number']:.2f}")
print(f"  能量比例: {metrics_circ['energy_ratio']:.4f}")
print(f"  子空間覆蓋率: {metrics_circ['subspace_coverage']:.4f}")
print(f"  去重數量: {metrics_circ['n_duplicates_removed']}")
print(f"  接縫處選中: {seam_selected_circ} ({100*seam_ratio_circ:.1f}%)")

# ============================================================================
# 4. 視覺化對比
# ============================================================================
print(f"\n{'='*80}")
print("4. 生成視覺化對比")
print("=" * 80)

fig = plt.figure(figsize=(20, 10))

# 計算改善幅度
improvement = (seam_ratio_std - seam_ratio_circ) / seam_ratio_std * 100 if seam_ratio_std > 0 else 0

# 判斷是 2D 還是 3D
is_2d = coords.shape[1] == 2

# --- 主平面分佈（標準）---
ax1 = fig.add_subplot(2, 4, 1)
# 接縫區域背景
ax1.axvspan(x.min(), x.min() + seam_threshold_x, alpha=0.1, color='blue')
ax1.axvspan(x.max() - seam_threshold_x, x.max(), alpha=0.1, color='blue')

if is_2d:
    # 2D: XY 平面
    ax1.scatter(coords[::50, 0], coords[::50, 1], c='lightgray', s=1, alpha=0.5)
    ax1.scatter(coords_std[:, 0], coords_std[:, 1], c='red', s=40, alpha=0.7,
               edgecolors='darkred', linewidths=0.5, label=f'Selected (n={len(indices_std)})')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y (wall-normal)', fontsize=11)
else:
    # 3D: XZ 平面
    ax1.axhspan(coords[:, 2].min(), coords[:, 2].min() + seam_threshold_z, alpha=0.1, color='blue')
    ax1.axhspan(coords[:, 2].max() - seam_threshold_z, coords[:, 2].max(), alpha=0.1, color='blue')
    ax1.scatter(coords[::50, 0], coords[::50, 2], c='lightgray', s=1, alpha=0.5)
    ax1.scatter(coords_std[:, 0], coords_std[:, 2], c='red', s=40, alpha=0.7,
               edgecolors='darkred', linewidths=0.5, label=f'Selected (n={len(indices_std)})')
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('z', fontsize=11)

ax1.set_title(f'Standard Mode\nSeam Ratio: {100*seam_ratio_std:.1f}%',
             fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# --- 主平面分佈（循環）---
ax2 = fig.add_subplot(2, 4, 2)
ax2.axvspan(x.min(), x.min() + seam_threshold_x, alpha=0.1, color='blue')
ax2.axvspan(x.max() - seam_threshold_x, x.max(), alpha=0.1, color='blue')

if is_2d:
    # 2D: XY 平面
    ax2.scatter(coords[::50, 0], coords[::50, 1], c='lightgray', s=1, alpha=0.5)
    ax2.scatter(coords_circ[:, 0], coords_circ[:, 1], c='green', s=40, alpha=0.7,
               edgecolors='darkgreen', linewidths=0.5, label=f'Selected (n={len(indices_circ)})')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y (wall-normal)', fontsize=11)
else:
    # 3D: XZ 平面
    ax2.axhspan(coords[:, 2].min(), coords[:, 2].min() + seam_threshold_z, alpha=0.1, color='blue')
    ax2.axhspan(coords[:, 2].max() - seam_threshold_z, coords[:, 2].max(), alpha=0.1, color='blue')
    ax2.scatter(coords[::50, 0], coords[::50, 2], c='lightgray', s=1, alpha=0.5)
    ax2.scatter(coords_circ[:, 0], coords_circ[:, 2], c='green', s=40, alpha=0.7,
               edgecolors='darkgreen', linewidths=0.5, label=f'Selected (n={len(indices_circ)})')
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('z', fontsize=11)

ax2.set_title(f'Circular Indexing\nSeam Ratio: {100*seam_ratio_circ:.1f}%',
             fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# --- 速度場大小分佈（標準）---
ax3 = fig.add_subplot(2, 4, 3)
vel_mag_std = np.linalg.norm(snapshots[indices_std, :n_snapshots], axis=1).mean()
vel_mag_circ = np.linalg.norm(snapshots[indices_circ, :n_snapshots], axis=1).mean()
ax3.scatter(coords_std[:, 0], coords_std[:, 1],
           c=np.linalg.norm(snapshots[indices_std, :n_snapshots], axis=1),
           s=40, alpha=0.7, cmap='hot', edgecolors='black', linewidths=0.5)
ax3.axvspan(x.min(), x.min() + seam_threshold_x, alpha=0.1, color='blue')
ax3.axvspan(x.max() - seam_threshold_x, x.max(), alpha=0.1, color='blue')
ax3.set_xlabel('x', fontsize=11)
ax3.set_ylabel('y', fontsize=11)
ax3.set_title(f'Standard |V| (mean={vel_mag_std:.3f})', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
plt.colorbar(ax3.collections[0], ax=ax3, label='|V|')

# --- 速度場大小分佈（循環）---
ax4 = fig.add_subplot(2, 4, 4)
ax4.scatter(coords_circ[:, 0], coords_circ[:, 1],
           c=np.linalg.norm(snapshots[indices_circ, :n_snapshots], axis=1),
           s=40, alpha=0.7, cmap='hot', edgecolors='black', linewidths=0.5)
ax4.axvspan(x.min(), x.min() + seam_threshold_x, alpha=0.1, color='blue')
ax4.axvspan(x.max() - seam_threshold_x, x.max(), alpha=0.1, color='blue')
ax4.set_xlabel('x', fontsize=11)
ax4.set_ylabel('y', fontsize=11)
ax4.set_title(f'Circular |V| (mean={vel_mag_circ:.3f})', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
plt.colorbar(ax4.collections[0], ax=ax4, label='|V|')

# --- x 座標直方圖對比 ---
ax5 = fig.add_subplot(2, 4, 5)
bins = np.linspace(x.min(), x.max(), 30)
ax5.hist(coords_std[:, 0], bins=bins, alpha=0.6, color='red',
        edgecolor='darkred', linewidth=1, label='Standard')
ax5.hist(coords_circ[:, 0], bins=bins, alpha=0.6, color='green',
        edgecolor='darkgreen', linewidth=1, label='Circular')
ax5.axvline(x.min() + seam_threshold_x, color='blue', linestyle='--', alpha=0.5)
ax5.axvline(x.max() - seam_threshold_x, color='blue', linestyle='--', alpha=0.5)
ax5.set_xlabel('x Coordinate', fontsize=11)
ax5.set_ylabel('Count', fontsize=11)
ax5.set_title('x Distribution Comparison', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# --- y 座標直方圖對比（壁面分佈）---
ax6 = fig.add_subplot(2, 4, 6)
bins = np.linspace(y.min(), y.max(), 30)
ax6.hist(coords_std[:, 1], bins=bins, alpha=0.6, color='red',
        edgecolor='darkred', linewidth=1, label='Standard')
ax6.hist(coords_circ[:, 1], bins=bins, alpha=0.6, color='green',
        edgecolor='darkgreen', linewidth=1, label='Circular')
ax6.set_xlabel('y Coordinate (wall-normal)', fontsize=11)
ax6.set_ylabel('Count', fontsize=11)
ax6.set_title('y Distribution Comparison', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3, axis='y')

# --- 品質指標對比 ---
ax7 = fig.add_subplot(2, 4, 7)
metrics_names = ['Condition\nNumber', 'Energy\nRatio', 'Subspace\nCoverage', 'Seam\nRatio']
std_values = [
    1.0 / (1.0 + metrics_std['condition_number']/100),
    metrics_std['energy_ratio'],
    metrics_std['subspace_coverage'],
    1.0 - seam_ratio_std
]
circ_values = [
    1.0 / (1.0 + metrics_circ['condition_number']/100),
    metrics_circ['energy_ratio'],
    metrics_circ['subspace_coverage'],
    1.0 - seam_ratio_circ
]

x_pos = np.arange(len(metrics_names))
width = 0.35

ax7.bar(x_pos - width/2, std_values, width, label='Standard',
       color='red', alpha=0.7, edgecolor='darkred')
ax7.bar(x_pos + width/2, circ_values, width, label='Circular',
       color='green', alpha=0.7, edgecolor='darkgreen')

ax7.set_ylabel('Normalized Score', fontsize=11)
ax7.set_title('Quality Metrics Comparison', fontsize=12, fontweight='bold')
ax7.set_xticks(x_pos)
ax7.set_xticklabels(metrics_names, fontsize=9)
ax7.legend(fontsize=9)
ax7.grid(True, alpha=0.3, axis='y')
ax7.set_ylim([0, 1.1])

# --- 統計摘要 ---
ax8 = fig.add_subplot(2, 4, 8)
ax8.axis('off')

if is_2d:
    grid_str = f"{grid_shape[0]}×{grid_shape[1]} (2D slice)"
    domain_str = f"x∈[{x.min():.2f},{x.max():.2f}]\n          y∈[{y.min():.2f},{y.max():.2f}]"
else:
    grid_str = f"{grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]}"
    domain_str = f"x∈[{x.min():.2f},{x.max():.2f}]\n          y∈[{y.min():.2f},{y.max():.2f}]\n          z∈[{coords[:, 2].min():.2f},{coords[:, 2].max():.2f}]"

summary_text = f"""
JHTDB 通道流驗證結果

數據資訊：
  網格: {grid_str}
  總點數: {n_locations:,}
  域範圍: {domain_str}

感測點配置：
  目標數量: {args.n_sensors}
  環繞層數: {args.n_wrap_layers}

標準模式：
  條件數: {metrics_std['condition_number']:.2f}
  能量比: {metrics_std['energy_ratio']:.4f}
  接縫比: {100*seam_ratio_std:.1f}%

循環索引：
  條件數: {metrics_circ['condition_number']:.2f}
  能量比: {metrics_circ['energy_ratio']:.4f}
  接縫比: {100*seam_ratio_circ:.1f}%
  去重數: {metrics_circ['n_duplicates_removed']}

改善幅度：
  接縫聚集: {improvement:+.1f}%
  條件數比: {metrics_circ['condition_number']/metrics_std['condition_number']:.2f}×
"""

ax8.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
        verticalalignment='center', bbox=dict(boxstyle='round',
        facecolor='wheat', alpha=0.3))

plt.suptitle(f'JHTDB Channel Flow: Circular Indexing Validation (K={args.n_sensors})',
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

output_file = output_dir / 'jhtdb_circular_validation.png'
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"  ✅ 已保存: {output_file}")

# ============================================================================
# 5. 統計分析
# ============================================================================
print(f"\n{'='*80}")
print("5. 詳細統計分析")
print("=" * 80)

print(f"\n標準模式空間分佈：")
print(f"  x: mean={coords_std[:, 0].mean():.3f}, std={coords_std[:, 0].std():.3f}")
print(f"  y: mean={coords_std[:, 1].mean():.3f}, std={coords_std[:, 1].std():.3f}")
if not is_2d:
    print(f"  z: mean={coords_std[:, 2].mean():.3f}, std={coords_std[:, 2].std():.3f}")

print(f"\n循環索引空間分佈：")
print(f"  x: mean={coords_circ[:, 0].mean():.3f}, std={coords_circ[:, 0].std():.3f}")
print(f"  y: mean={coords_circ[:, 1].mean():.3f}, std={coords_circ[:, 1].std():.3f}")
if not is_2d:
    print(f"  z: mean={coords_circ[:, 2].mean():.3f}, std={coords_circ[:, 2].std():.3f}")

print(f"\n改善指標：")
print(f"  接縫聚集減少: {improvement:+.1f}%")
print(f"  條件數變化: {metrics_circ['condition_number']/metrics_std['condition_number']:.2f}×")
print(f"  能量比例保持: {metrics_circ['energy_ratio']/metrics_std['energy_ratio']:.4f}×")

print(f"\n{'='*80}")
print(f"✅ 驗證完成！結果已保存至 {output_dir}")
print(f"{'='*80}")
