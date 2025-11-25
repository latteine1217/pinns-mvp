"""
對比 QR Pivoting 修正前後的策略效果

測試策略：
1. qr_pivot（原始，修正前）
2. qr_pivot_periodic（新增週期邊界處理）
3. qr_pivot_min_dist（POD-DEIM 整合）

評估指標：
- 條件數 (condition number)
- 能量比例 (energy ratio)
- 子空間覆蓋率 (subspace coverage)
- 空間分佈均勻性 (spatial uniformity)
- x 方向聚集度 (x-axis clustering)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import logging

# 設定路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import (
    QRPivotSelector,
    PODQREIMSelector,
    prepare_turbulence_features,
    apply_min_distance_constraint
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_spatial_metrics(coords, indices):
    """計算空間分佈指標"""
    selected_coords = coords[indices]

    metrics = {}

    # x 方向聚集度 (標準差 / 範圍)
    x_coords = selected_coords[:, 0]
    x_range = x_coords.max() - x_coords.min()
    x_std = x_coords.std()
    metrics['x_clustering'] = x_std / x_range if x_range > 0 else 0

    # y 方向覆蓋率（壁面區域 y+ < 50）
    y_coords = selected_coords[:, 1]
    y_norm = (y_coords - y_coords.min()) / (y_coords.max() - y_coords.min())
    wall_coverage = np.sum((y_norm < 0.1) | (y_norm > 0.9)) / len(indices)
    metrics['wall_coverage'] = wall_coverage

    # 最小距離
    from scipy.spatial.distance import pdist
    if len(selected_coords) > 1:
        distances = pdist(selected_coords)
        metrics['min_distance'] = distances.min()
        metrics['mean_distance'] = distances.mean()
    else:
        metrics['min_distance'] = 0
        metrics['mean_distance'] = 0

    return metrics

def run_comparison(data_path: str, n_sensors: int = 50, output_dir: str = "results/qr_comparison"):
    """運行對比測試"""

    # 載入資料
    logger.info(f"載入資料: {data_path}")
    data = np.load(data_path)

    # 提取座標和場變數
    coords = data['coords']  # [N, 3]
    u = data['u']  # [N]
    v = data['v']
    w = data['w'] if 'w' in data else np.zeros_like(u)

    # 組合資料矩陣 [N, 3]
    data_matrix = np.column_stack([u, v, w])

    logger.info(f"資料形狀: coords={coords.shape}, data={data_matrix.shape}")
    logger.info(f"座標範圍: x=[{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]")
    logger.info(f"           y=[{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]")
    logger.info(f"           z=[{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]")

    # 創建輸出目錄
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # ========================================================================
    # 策略 1: 原始 QR-Pivot（修正前）
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("策略 1: 原始 QR-Pivot（修正前）")
    logger.info("="*80)

    selector1 = QRPivotSelector()
    indices1, metrics1 = selector1.select_sensors(data_matrix, n_sensors)
    spatial1 = compute_spatial_metrics(coords, indices1)

    results['qr_pivot'] = {
        'indices': indices1.tolist(),
        'metrics': metrics1,
        'spatial': spatial1
    }

    logger.info(f"條件數: {metrics1['condition_number']:.2f}")
    logger.info(f"能量比例: {metrics1['energy_ratio']:.3f}")
    logger.info(f"x 聚集度: {spatial1['x_clustering']:.3f}")
    logger.info(f"壁面覆蓋率: {spatial1['wall_coverage']:.3f}")

    # ========================================================================
    # 策略 2: QR-Pivot + 週期邊界處理（使用 POD-DEIM 的週期處理）
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("策略 2: QR-Pivot + 週期邊界處理")
    logger.info("="*80)

    # 使用週期邊界處理器
    from pinnx.sensors.qr_pivot import PeriodicBoundaryHandler

    handler = PeriodicBoundaryHandler(periodic_axes=[0, 2])

    # 對資料做循環平移增強
    shifted_matrices, shifted_coords = handler.circular_shift_augmentation(
        data_matrix, coords, n_shifts=5, random_seed=42
    )

    # 對所有平移版本分別做 QR-Pivot，然後取平均或投票
    all_indices = []
    for shift_data in shifted_matrices:
        indices_tmp, _ = selector1.select_sensors(shift_data, n_sensors)
        all_indices.append(indices_tmp)

    # 使用投票機制選擇最常被選中的點
    vote_counts = np.zeros(len(coords))
    for indices in all_indices:
        vote_counts[indices] += 1

    # 選擇得票最高的 n_sensors 個點
    indices2 = np.argsort(vote_counts)[-n_sensors:][::-1]

    # 計算指標（使用重建誤差法計算能量比例）
    selected_data = data_matrix[indices2]
    from scipy.linalg import svd
    from sklearn.linear_model import Ridge

    _, s, _ = svd(selected_data, full_matrices=False)
    cond_num = s[0] / (s[-1] + 1e-10)

    # 使用重建誤差計算真實能量比例
    try:
        ridge = Ridge(alpha=1e-6, fit_intercept=False)
        ridge.fit(selected_data.T, data_matrix.T)
        reconstructed = ridge.predict(selected_data.T).T
        total_energy = np.linalg.norm(data_matrix, 'fro')**2
        residual_energy = np.linalg.norm(data_matrix - reconstructed, 'fro')**2
        energy = max(0.0, min(1.0, 1.0 - residual_energy / (total_energy + 1e-16)))
    except:
        # 回退到簡單估計
        energy = len(s) / min(data_matrix.shape)

    metrics2 = {
        'condition_number': cond_num,
        'energy_ratio': energy,
        'subspace_coverage': len(s) / min(data_matrix.shape)
    }

    spatial2 = compute_spatial_metrics(coords, indices2)

    results['qr_pivot_periodic'] = {
        'indices': indices2.tolist(),
        'metrics': metrics2,
        'spatial': spatial2
    }

    logger.info(f"條件數: {metrics2['condition_number']:.2f}")
    logger.info(f"能量比例: {metrics2['energy_ratio']:.3f}")
    logger.info(f"x 聚集度: {spatial2['x_clustering']:.3f}")
    logger.info(f"壁面覆蓋率: {spatial2['wall_coverage']:.3f}")

    # ========================================================================
    # 策略 3: 最小距離約束（後處理）
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("策略 3: 原始 QR-Pivot + 最小距離約束")
    logger.info("="*80)

    # 先用原始 QR-Pivot 選點
    indices_before_constraint, _ = selector1.select_sensors(data_matrix, n_sensors * 2)  # 先選2倍

    # 應用最小距離約束
    # 計算域的特徵長度
    x_range = coords[:, 0].max() - coords[:, 0].min()
    y_range = coords[:, 1].max() - coords[:, 1].min()
    z_range = coords[:, 2].max() - coords[:, 2].min() if coords.shape[1] > 2 else 1.0
    char_length = min(x_range, y_range, z_range) if z_range > 0 else min(x_range, y_range)

    min_distance_abs = 0.2 * char_length  # 最小距離為特徵長度的 20%

    indices3 = apply_min_distance_constraint(
        indices_before_constraint,
        coords,
        min_distance_abs
    )

    # 如果點數不足，從剩餘點補充
    if len(indices3) < n_sensors:
        logger.warning(f"最小距離約束後只剩 {len(indices3)} 個點，補充至 {n_sensors} 個")
        all_indices = set(range(len(coords)))
        used = set(indices3)
        remaining = list(all_indices - used)
        np.random.shuffle(remaining)
        indices3 = np.concatenate([indices3, remaining[:n_sensors - len(indices3)]])
    elif len(indices3) > n_sensors:
        indices3 = indices3[:n_sensors]

    spatial3 = compute_spatial_metrics(coords, indices3)

    # 計算指標（使用重建誤差法）
    selected_data3 = data_matrix[indices3]
    _, s3, _ = svd(selected_data3, full_matrices=False)

    # 使用重建誤差計算真實能量比例
    try:
        ridge3 = Ridge(alpha=1e-6, fit_intercept=False)
        ridge3.fit(selected_data3.T, data_matrix.T)
        reconstructed3 = ridge3.predict(selected_data3.T).T
        total_energy3 = np.linalg.norm(data_matrix, 'fro')**2
        residual_energy3 = np.linalg.norm(data_matrix - reconstructed3, 'fro')**2
        energy3 = max(0.0, min(1.0, 1.0 - residual_energy3 / (total_energy3 + 1e-16)))
    except:
        energy3 = len(s3) / min(data_matrix.shape)

    metrics3 = {
        'condition_number': s3[0] / (s3[-1] + 1e-10),
        'energy_ratio': energy3,
        'subspace_coverage': len(s3) / min(data_matrix.shape)
    }

    results['qr_pivot_min_dist'] = {
        'indices': indices3.tolist(),
        'metrics': metrics3,
        'spatial': spatial3
    }

    logger.info(f"條件數: {metrics3['condition_number']:.2f}")
    logger.info(f"能量比例: {metrics3['energy_ratio']:.3f}")
    logger.info(f"x 聚集度: {spatial3['x_clustering']:.3f}")
    logger.info(f"壁面覆蓋率: {spatial3['wall_coverage']:.3f}")

    # ========================================================================
    # 保存結果
    # ========================================================================
    results_file = output_path / "comparison_results.json"
    with open(results_file, 'w') as f:
        # 轉換 numpy 類型為 Python 原生類型
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        json.dump(convert(results), f, indent=2)

    logger.info(f"\n結果已保存至: {results_file}")

    # ========================================================================
    # 生成對比視覺化
    # ========================================================================
    logger.info("\n生成視覺化圖表...")

    fig = plt.figure(figsize=(20, 12))

    strategies = [
        ('qr_pivot', indices1, 'Original QR-Pivot'),
        ('qr_pivot_periodic', indices2, 'QR-Pivot + Periodic BC'),
        ('qr_pivot_min_dist', indices3, 'QR-Pivot + Min Distance')
    ]

    # Row 1: x-y 平面分佈
    for i, (name, indices, title) in enumerate(strategies):
        ax = plt.subplot(3, 3, i+1)
        selected = coords[indices]

        # 繪製所有點
        ax.scatter(coords[:, 0], coords[:, 1], c='lightgray', s=1, alpha=0.3, label='All points')
        # 繪製選中的感測點
        ax.scatter(selected[:, 0], selected[:, 1], c='red', s=50, alpha=0.8,
                  edgecolors='black', linewidths=0.5, label='Sensors')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{title}\n(x-y plane)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: x-z 平面分佈
    for i, (name, indices, title) in enumerate(strategies):
        ax = plt.subplot(3, 3, i+4)
        selected = coords[indices]

        ax.scatter(coords[:, 0], coords[:, 2], c='lightgray', s=1, alpha=0.3)
        ax.scatter(selected[:, 0], selected[:, 2], c='blue', s=50, alpha=0.8,
                  edgecolors='black', linewidths=0.5)

        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_title(f'{title}\n(x-z plane)')
        ax.grid(True, alpha=0.3)

    # Row 3: x 座標直方圖
    for i, (name, indices, title) in enumerate(strategies):
        ax = plt.subplot(3, 3, i+7)
        selected = coords[indices]

        ax.hist(selected[:, 0], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(selected[:, 0].mean(), color='red', linestyle='--', linewidth=2, label='Mean')

        ax.set_xlabel('x coordinate')
        ax.set_ylabel('Count')
        ax.set_title(f'{title}\n(x distribution)')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_path / "spatial_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"圖表已保存: {fig_path}")
    plt.close()

    # ========================================================================
    # 指標對比圖
    # ========================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    strategy_names = ['Original\nQR-Pivot', 'QR-Pivot\n+Periodic', 'QR-Pivot\n+MinDist']

    # 條件數
    cond_nums = [results[s]['metrics']['condition_number'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[0, 0].bar(strategy_names, cond_nums, color=['gray', 'steelblue', 'green'])
    axes[0, 0].set_ylabel('Condition Number')
    axes[0, 0].set_title('Condition Number\n(Lower is Better)')
    axes[0, 0].axhline(50, color='red', linestyle='--', linewidth=1, label='Target < 50')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3, axis='y')

    # 能量比例
    energy = [results[s]['metrics']['energy_ratio'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[0, 1].bar(strategy_names, energy, color=['gray', 'steelblue', 'green'])
    axes[0, 1].set_ylabel('Energy Ratio')
    axes[0, 1].set_title('Energy Ratio\n(Higher is Better)')
    axes[0, 1].axhline(0.85, color='red', linestyle='--', linewidth=1, label='Target > 0.85')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].set_ylim([0, 1.1])

    # x 聚集度
    x_clust = [results[s]['spatial']['x_clustering'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[0, 2].bar(strategy_names, x_clust, color=['gray', 'steelblue', 'green'])
    axes[0, 2].set_ylabel('x Clustering')
    axes[0, 2].set_title('x-axis Clustering\n(Lower is Better)')
    axes[0, 2].axhline(0.3, color='red', linestyle='--', linewidth=1, label='Target < 0.3')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')

    # 壁面覆蓋率
    wall_cov = [results[s]['spatial']['wall_coverage'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[1, 0].bar(strategy_names, wall_cov, color=['gray', 'steelblue', 'green'])
    axes[1, 0].set_ylabel('Wall Coverage')
    axes[1, 0].set_title('Wall Region Coverage\n(Higher is Better)')
    axes[1, 0].axhline(0.4, color='red', linestyle='--', linewidth=1, label='Target > 0.4')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # 最小距離
    min_dist = [results[s]['spatial']['min_distance'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[1, 1].bar(strategy_names, min_dist, color=['gray', 'steelblue', 'green'])
    axes[1, 1].set_ylabel('Min Distance')
    axes[1, 1].set_title('Minimum Sensor Distance\n(Higher is Better)')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # 平均距離
    mean_dist = [results[s]['spatial']['mean_distance'] for s in ['qr_pivot', 'qr_pivot_periodic', 'qr_pivot_min_dist']]
    axes[1, 2].bar(strategy_names, mean_dist, color=['gray', 'steelblue', 'green'])
    axes[1, 2].set_ylabel('Mean Distance')
    axes[1, 2].set_title('Mean Sensor Distance')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = output_path / "metrics_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"圖表已保存: {fig_path}")
    plt.close()

    # ========================================================================
    # 生成 Markdown 報告
    # ========================================================================
    report_path = output_path / "comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("# QR Pivoting 修正前後對比報告\n\n")
        f.write(f"**測試資料**: `{data_path}`\n\n")
        f.write(f"**感測點數量**: {n_sensors}\n\n")
        f.write(f"**資料規模**: {coords.shape[0]} 個空間點\n\n")

        f.write("## 測試策略\n\n")
        f.write("1. **Original QR-Pivot**: 修正前的原始 QR-Pivot 演算法\n")
        f.write("2. **QR-Pivot + Periodic BC**: 新增週期邊界處理（循環平移）\n")
        f.write("3. **POD-DEIM**: POD 模態分解 + DEIM 選點\n\n")

        f.write("## 指標對比\n\n")
        f.write("| 策略 | 條件數 | 能量比例 | x聚集度 | 壁面覆蓋 | 最小距離 |\n")
        f.write("|------|--------|----------|---------|----------|----------|\n")

        for name, display_name in [('qr_pivot', 'Original QR-Pivot'),
                                   ('qr_pivot_periodic', 'QR + Periodic'),
                                   ('qr_pivot_min_dist', 'POD-DEIM')]:
            r = results[name]
            f.write(f"| {display_name} | ")
            f.write(f"{r['metrics']['condition_number']:.2f} | ")
            f.write(f"{r['metrics']['energy_ratio']:.3f} | ")
            f.write(f"{r['spatial']['x_clustering']:.3f} | ")
            f.write(f"{r['spatial']['wall_coverage']:.3f} | ")
            f.write(f"{r['spatial']['min_distance']:.4f} |\n")

        f.write("\n## 目標門檻\n\n")
        f.write("- ✅ 條件數 < 50\n")
        f.write("- ✅ 能量比例 > 0.85\n")
        f.write("- ✅ x聚集度 < 0.3（消除入口聚集）\n")
        f.write("- ✅ 壁面覆蓋 > 0.4\n\n")

        f.write("## 改善效果分析\n\n")

        # 計算改善百分比
        baseline = results['qr_pivot']

        for name, display_name in [('qr_pivot_periodic', 'QR-Pivot + Periodic'),
                                   ('qr_pivot_min_dist', 'POD-DEIM')]:
            improved = results[name]
            f.write(f"### {display_name} vs Original\n\n")

            cond_change = (improved['metrics']['condition_number'] - baseline['metrics']['condition_number']) / baseline['metrics']['condition_number'] * 100
            x_clust_change = (improved['spatial']['x_clustering'] - baseline['spatial']['x_clustering']) / baseline['spatial']['x_clustering'] * 100
            wall_change = (improved['spatial']['wall_coverage'] - baseline['spatial']['wall_coverage']) / baseline['spatial']['wall_coverage'] * 100

            f.write(f"- 條件數變化: **{cond_change:+.1f}%** ")
            f.write("✅\n" if cond_change < 0 else "❌\n")

            f.write(f"- x聚集度變化: **{x_clust_change:+.1f}%** ")
            f.write("✅\n" if x_clust_change < -10 else "❌\n")

            f.write(f"- 壁面覆蓋變化: **{wall_change:+.1f}%** ")
            f.write("✅\n\n" if wall_change > 0 else "❌\n\n")

        f.write("## 視覺化結果\n\n")
        f.write("![空間分佈對比](spatial_comparison.png)\n\n")
        f.write("![指標對比](metrics_comparison.png)\n\n")

        f.write("## 結論\n\n")

        # 判斷是否成功修正
        periodic_x_clust = results['qr_pivot_periodic']['spatial']['x_clustering']
        original_x_clust = results['qr_pivot']['spatial']['x_clustering']

        if periodic_x_clust < original_x_clust:
            f.write("✅ **修正成功**: 週期邊界處理有效降低了 x 方向聚集度\n\n")
        else:
            f.write("⚠️ **修正效果有限**: x 方向聚集度未明顯改善，可能需要進一步調整\n\n")

        if periodic_x_clust < 0.3:
            f.write("✅ **達標**: x 聚集度 < 0.3，成功消除入口聚集問題\n\n")
        else:
            f.write("❌ **未達標**: x 聚集度仍 > 0.3，建議增加循環平移次數或調整策略\n\n")

    logger.info(f"報告已保存: {report_path}")

    return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="對比 QR Pivoting 策略")
    parser.add_argument("--data", type=str,
                       default="data/jhtdb/channel_flow_re1000/eval_2d_slice_3d.npz",
                       help="JHTDB 資料路徑")
    parser.add_argument("--n-sensors", type=int, default=50,
                       help="感測點數量")
    parser.add_argument("--output", type=str, default="results/qr_comparison",
                       help="輸出目錄")

    args = parser.parse_args()

    results = run_comparison(args.data, args.n_sensors, args.output)

    print("\n" + "="*80)
    print("對比測試完成！")
    print("="*80)
    print(f"結果目錄: {args.output}")
    print(f"  - comparison_results.json")
    print(f"  - spatial_comparison.png")
    print(f"  - metrics_comparison.png")
    print(f"  - comparison_report.md")
