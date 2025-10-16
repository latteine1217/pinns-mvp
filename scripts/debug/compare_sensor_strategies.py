#!/usr/bin/env python3
"""
對比感測點選擇策略：QR-Pivot vs. Stratified vs. Hybrid

目的：
1. 驗證分層採樣是否解決 QR-pivot 的壁面偏差問題
2. 對比中心線覆蓋率與統計偏差
3. 評估混合策略的平衡性能

使用方式：
    python scripts/debug/compare_sensor_strategies.py \
        --data_path data/jhtdb/channel_flow_retau1000_cutout_128x32x128.npz \
        --K 50 \
        --output results/sensor_strategy_comparison.png
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple
import logging

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector
from pinnx.sensors.stratified_sampling import (
    StratifiedChannelFlowSelector,
    HybridChannelFlowSelector
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_jhtdb_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """載入 JHTDB 通道流資料"""
    
    logger.info(f"載入資料: {data_path}")
    
    data = np.load(data_path)
    
    # 提取座標與場資料
    if 'coords' in data:
        coords = data['coords']  # (N, 3) - (x, y, z)
    elif 'x' in data and 'y' in data and 'z' in data:
        x = data['x']
        y = data['y']
        z = data['z']
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    else:
        raise ValueError("資料檔案必須包含 'coords' 或 'x/y/z' 欄位")
    
    # 提取速度與壓力場
    if 'u' in data and 'v' in data and 'w' in data and 'p' in data:
        u = data['u'].ravel() if data['u'].ndim > 1 else data['u']
        v = data['v'].ravel() if data['v'].ndim > 1 else data['v']
        w = data['w'].ravel() if data['w'].ndim > 1 else data['w']
        p = data['p'].ravel() if data['p'].ndim > 1 else data['p']
        field_data = np.stack([u, v, w, p], axis=1)
    else:
        raise ValueError("資料檔案必須包含 'u/v/w/p' 欄位")
    
    logger.info(f"資料形狀: coords={coords.shape}, field={field_data.shape}")
    logger.info(f"Y 範圍: [{coords[:, 1].min():.4f}, {coords[:, 1].max():.4f}]")
    
    return coords, field_data


def analyze_layer_distribution(coords: np.ndarray, 
                               indices: np.ndarray,
                               y_index: int = 1) -> Dict:
    """分析感測點在物理分層的分佈"""
    
    y_coords = coords[indices, y_index]
    
    # 定義物理分層（與 StratifiedChannelFlowSelector 一致）
    wall_mask = np.abs(y_coords) > 0.8
    log_mask = (np.abs(y_coords) > 0.2) & (np.abs(y_coords) <= 0.8)
    core_mask = np.abs(y_coords) <= 0.2
    
    n_total = len(indices)
    n_wall = wall_mask.sum()
    n_log = log_mask.sum()
    n_core = core_mask.sum()
    
    return {
        'n_total': n_total,
        'n_wall': n_wall,
        'n_log': n_log,
        'n_core': n_core,
        'ratio_wall': n_wall / n_total if n_total > 0 else 0,
        'ratio_log': n_log / n_total if n_total > 0 else 0,
        'ratio_core': n_core / n_total if n_total > 0 else 0,
        'y_mean': y_coords.mean(),
        'y_std': y_coords.std(),
        'y_range': [y_coords.min(), y_coords.max()]
    }


def compute_field_statistics(field_data: np.ndarray, 
                             indices: np.ndarray,
                             coords: np.ndarray,
                             y_index: int = 1) -> Dict:
    """計算場統計偏差"""
    
    # 完整場統計
    u_full = field_data[:, 0]
    u_sensors = u_full[indices]
    
    # 中心線統計（|y| < 0.1）
    y_coords = coords[:, y_index]
    centerline_mask = np.abs(y_coords) < 0.1
    u_centerline = u_full[centerline_mask].mean() if centerline_mask.any() else 0
    
    return {
        'u_mean_full': u_full.mean(),
        'u_std_full': u_full.std(),
        'u_mean_sensors': u_sensors.mean(),
        'u_std_sensors': u_sensors.std(),
        'u_mean_centerline': u_centerline,
        'ratio_sensors_full': u_sensors.mean() / (u_full.mean() + 1e-16),
        'ratio_sensors_centerline': u_sensors.mean() / (u_centerline + 1e-16)
    }


def compare_strategies(coords: np.ndarray,
                      field_data: np.ndarray,
                      n_sensors: int) -> Dict:
    """對比不同策略的感測點選擇"""
    
    results = {}
    
    logger.info("\n" + "="*60)
    logger.info("開始對比感測點選擇策略")
    logger.info("="*60)
    
    # ===== 1. QR-Pivot 基線 =====
    logger.info("\n【策略 1】QR-Pivot (基線)")
    try:
        qr_selector = QRPivotSelector(mode='column', pivoting=True)
        qr_indices, qr_metrics = qr_selector.select_sensors(field_data, n_sensors)
        
        results['qr_pivot'] = {
            'indices': qr_indices,
            'layer_dist': analyze_layer_distribution(coords, qr_indices),
            'field_stats': compute_field_statistics(field_data, qr_indices, coords),
            'metrics': qr_metrics
        }
        
        layer = results['qr_pivot']['layer_dist']
        logger.info(f"  分層分佈: 壁面={layer['n_wall']} ({layer['ratio_wall']:.1%}), "
                   f"對數律={layer['n_log']} ({layer['ratio_log']:.1%}), "
                   f"中心={layer['n_core']} ({layer['ratio_core']:.1%})")
        
        stats = results['qr_pivot']['field_stats']
        logger.info(f"  統計偏差: 感測點/完整場 = {stats['ratio_sensors_full']:.2%}, "
                   f"感測點/中心線 = {stats['ratio_sensors_centerline']:.2%}")
        
    except Exception as e:
        logger.error(f"  QR-Pivot 失敗: {e}")
        results['qr_pivot'] = {'error': str(e)}
    
    # ===== 2. 分層採樣 =====
    logger.info("\n【策略 2】Stratified (分層採樣)")
    try:
        strat_selector = StratifiedChannelFlowSelector(
            wall_ratio=0.35,
            log_ratio=0.35,
            core_ratio=0.30,
            use_qr_refinement=True,
            seed=42
        )
        strat_indices, strat_metrics = strat_selector.select_sensors(
            coords, field_data, n_sensors
        )
        
        results['stratified'] = {
            'indices': strat_indices,
            'layer_dist': analyze_layer_distribution(coords, strat_indices),
            'field_stats': compute_field_statistics(field_data, strat_indices, coords),
            'metrics': strat_metrics
        }
        
        layer = results['stratified']['layer_dist']
        logger.info(f"  分層分佈: 壁面={layer['n_wall']} ({layer['ratio_wall']:.1%}), "
                   f"對數律={layer['n_log']} ({layer['ratio_log']:.1%}), "
                   f"中心={layer['n_core']} ({layer['ratio_core']:.1%})")
        
        stats = results['stratified']['field_stats']
        logger.info(f"  統計偏差: 感測點/完整場 = {stats['ratio_sensors_full']:.2%}, "
                   f"感測點/中心線 = {stats['ratio_sensors_centerline']:.2%}")
        
    except Exception as e:
        logger.error(f"  Stratified 失敗: {e}")
        results['stratified'] = {'error': str(e)}
    
    # ===== 3. 混合策略 =====
    logger.info("\n【策略 3】Hybrid (混合策略)")
    try:
        hybrid_selector = HybridChannelFlowSelector(
            stratified_ratio=0.7,
            qr_ratio=0.3,
            wall_ratio=0.35,
            log_ratio=0.35,
            core_ratio=0.30,
            use_qr_refinement=True,
            seed=42
        )
        hybrid_indices, hybrid_metrics = hybrid_selector.select_sensors(
            coords, field_data, n_sensors
        )
        
        results['hybrid'] = {
            'indices': hybrid_indices,
            'layer_dist': analyze_layer_distribution(coords, hybrid_indices),
            'field_stats': compute_field_statistics(field_data, hybrid_indices, coords),
            'metrics': hybrid_metrics
        }
        
        layer = results['hybrid']['layer_dist']
        logger.info(f"  分層分佈: 壁面={layer['n_wall']} ({layer['ratio_wall']:.1%}), "
                   f"對數律={layer['n_log']} ({layer['ratio_log']:.1%}), "
                   f"中心={layer['n_core']} ({layer['ratio_core']:.1%})")
        
        stats = results['hybrid']['field_stats']
        logger.info(f"  統計偏差: 感測點/完整場 = {stats['ratio_sensors_full']:.2%}, "
                   f"感測點/中心線 = {stats['ratio_sensors_centerline']:.2%}")
        
    except Exception as e:
        logger.error(f"  Hybrid 失敗: {e}")
        results['hybrid'] = {'error': str(e)}
    
    return results


def visualize_comparison(results: Dict, 
                        coords: np.ndarray,
                        field_data: np.ndarray,
                        output_path: str):
    """視覺化對比結果"""
    
    fig = plt.figure(figsize=(18, 12))
    
    strategies = ['qr_pivot', 'stratified', 'hybrid']
    strategy_names = ['QR-Pivot (Baseline)', 'Stratified', 'Hybrid']
    
    y_coords_full = coords[:, 1]
    u_full = field_data[:, 0]
    
    # ===== 第一行：感測點空間分佈 =====
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        ax = plt.subplot(3, 3, i + 1)
        
        indices = results[strategy]['indices']
        y_sensors = coords[indices, 1]
        x_sensors = coords[indices, 0]
        
        # 背景：完整場 u
        ax.scatter(coords[:, 0], y_coords_full, c=u_full, 
                  cmap='coolwarm', s=1, alpha=0.3, vmin=0, vmax=20)
        
        # 前景：感測點
        ax.scatter(x_sensors, y_sensors, c='lime', s=50, 
                  edgecolors='black', linewidths=1, marker='o', 
                  label=f'Sensors (K={len(indices)})', zorder=5)
        
        # 物理分層線
        ax.axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Wall (|y|>0.8)')
        ax.axhline(-0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Log (0.2<|y|≤0.8)')
        ax.axhline(-0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7, label='Centerline')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(f'{name}\nSpatial Distribution', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # ===== 第二行：分層統計對比 =====
    ax_bar = plt.subplot(3, 3, 4)
    
    bar_width = 0.25
    x_pos = np.arange(3)
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        layer = results[strategy]['layer_dist']
        ratios = [layer['ratio_wall'], layer['ratio_log'], layer['ratio_core']]
        
        ax_bar.bar(x_pos + i * bar_width, ratios, bar_width, 
                  label=name, alpha=0.8)
    
    ax_bar.set_xticks(x_pos + bar_width)
    ax_bar.set_xticklabels(['Wall\n(|y|>0.8)', 'Log\n(0.2<|y|≤0.8)', 'Core\n(|y|≤0.2)'])
    ax_bar.set_ylabel('Sensor Ratio')
    ax_bar.set_title('Layer Coverage Comparison', fontweight='bold')
    ax_bar.legend()
    ax_bar.grid(True, alpha=0.3, axis='y')
    ax_bar.set_ylim(0, 1)
    
    # ===== 第二行中：場統計對比 =====
    ax_stats = plt.subplot(3, 3, 5)
    
    stats_names = ['u_mean_full', 'u_mean_sensors', 'u_mean_centerline']
    stats_labels = ['Full Field', 'Sensors', 'Centerline']
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        stats = results[strategy]['field_stats']
        values = [stats['u_mean_full'], stats['u_mean_sensors'], stats['u_mean_centerline']]
        
        ax_stats.bar(x_pos + i * bar_width, values, bar_width, 
                    label=name, alpha=0.8)
    
    ax_stats.set_xticks(x_pos + bar_width)
    ax_stats.set_xticklabels(stats_labels)
    ax_stats.set_ylabel('u velocity (mean)')
    ax_stats.set_title('Velocity Statistics Comparison', fontweight='bold')
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3, axis='y')
    
    # ===== 第二行右：偏差比例 =====
    ax_ratio = plt.subplot(3, 3, 6)
    
    ratio_names = ['Sensors/Full', 'Sensors/Centerline']
    
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        stats = results[strategy]['field_stats']
        ratios = [stats['ratio_sensors_full'], stats['ratio_sensors_centerline']]
        
        ax_ratio.bar(np.arange(2) + i * bar_width, ratios, bar_width, 
                    label=name, alpha=0.8)
    
    ax_ratio.set_xticks(np.arange(2) + bar_width)
    ax_ratio.set_xticklabels(ratio_names)
    ax_ratio.set_ylabel('Ratio')
    ax_ratio.set_title('Statistical Bias Ratio', fontweight='bold')
    ax_ratio.axhline(1.0, color='red', linestyle='--', linewidth=1, label='Ideal=1.0')
    ax_ratio.legend()
    ax_ratio.grid(True, alpha=0.3, axis='y')
    
    # ===== 第三行：Y 座標分佈直方圖 =====
    for i, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        ax = plt.subplot(3, 3, 7 + i)
        
        indices = results[strategy]['indices']
        y_sensors = coords[indices, 1]
        
        # 完整場分佈（背景）
        ax.hist(y_coords_full, bins=50, alpha=0.3, color='gray', 
               label='Full Field', density=True)
        
        # 感測點分佈（前景）
        ax.hist(y_sensors, bins=30, alpha=0.7, color='blue', 
               label=f'Sensors (K={len(indices)})', density=True)
        
        # 物理分層線
        ax.axvline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(-0.8, color='red', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(-0.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax.set_xlabel('y coordinate')
        ax.set_ylabel('Density')
        ax.set_title(f'{name}\nY Distribution', fontsize=10, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"\n✅ Visualization saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare sensor selection strategies')
    parser.add_argument('--data_path', type=str, 
                       default='data/jhtdb/channel_flow_retau1000_cutout_128x32x128.npz',
                       help='JHTDB data file path')
    parser.add_argument('--K', type=int, default=50,
                       help='Number of sensors')
    parser.add_argument('--output', type=str, 
                       default='results/sensor_strategy_comparison.png',
                       help='Visualization output path')
    
    args = parser.parse_args()
    
    # 載入資料
    coords, field_data = load_jhtdb_data(args.data_path)
    
    # 對比策略
    results = compare_strategies(coords, field_data, args.K)
    
    # 視覺化
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_comparison(results, coords, field_data, str(output_path))
    
    # 輸出數值比較表
    logger.info("\n" + "="*60)
    logger.info("Numerical Comparison Summary")
    logger.info("="*60)
    
    for strategy in ['qr_pivot', 'stratified', 'hybrid']:
        if strategy not in results or 'error' in results[strategy]:
            continue
        
        logger.info(f"\n【{strategy.upper()}】")
        
        layer = results[strategy]['layer_dist']
        logger.info(f"  Layers: Wall={layer['n_wall']} ({layer['ratio_wall']:.1%}), "
                   f"Log={layer['n_log']} ({layer['ratio_log']:.1%}), "
                   f"Core={layer['n_core']} ({layer['ratio_core']:.1%})")
        
        stats = results[strategy]['field_stats']
        logger.info(f"  Stats: Sensors/Full={stats['ratio_sensors_full']:.2%}, "
                   f"Sensors/Centerline={stats['ratio_sensors_centerline']:.2%}")
    
    logger.info("\n✅ Comparison analysis complete!")


if __name__ == '__main__':
    main()
