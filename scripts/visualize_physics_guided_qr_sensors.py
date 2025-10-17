#!/usr/bin/env python3
"""
物理引導 QR-Pivot 感測點視覺化對比工具

功能：
  1. 對比標準 QR-Pivot 與物理引導 QR-Pivot 的感測點分佈
  2. 壁面覆蓋率分析與熱圖
  3. 條件數 vs. 壁面覆蓋率的權衡曲線（掃描 wall_weight）
  4. 誤差分佈對比（壁面層 vs. 中心層）
  5. 品質指標對比（條件數、能量比例、子空間覆蓋率）

使用方式：
  # 從 JHTDB 資料對比兩種策略
  python scripts/visualize_physics_guided_qr_sensors.py \
    --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
    --n-sensors 50 \
    --output results/physics_guided_comparison

  # 掃描壁面權重的影響
  python scripts/visualize_physics_guided_qr_sensors.py \
    --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
    --n-sensors 50 \
    --wall-weight-scan \
    --output results/wall_weight_scan
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import h5py

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import QRPivotSelector, PhysicsGuidedQRPivotSelector


def load_jhtdb_data(h5_path: str) -> Dict[str, Any]:
    """載入 JHTDB 通道流資料（2D 切片）"""
    print(f"📂 載入 JHTDB 資料: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # 載入座標
        x = f['coordinates/x'][:]
        y = f['coordinates/y'][:]
        z = f['coordinates/z'][:]
        
        # 載入速度與壓力場
        u = f['velocity/u'][:]
        v = f['velocity/v'][:]
        w = f['velocity/w'][:]
        p = f['pressure/p'][:]
        
        # 物理參數
        Re_tau = f['physics'].attrs.get('Re_tau', 1000.0)
        u_tau = f['physics'].attrs.get('u_tau', 0.04997)
        nu = f['physics'].attrs.get('nu', 5.0e-5)
    
    # 構建 3D 網格座標 [N, 3]
    if u.ndim == 3:
        # 3D 資料
        xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        coords_3d = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=-1)
        
        # 構建快照矩陣 [N, 3] (u, v, w)
        snapshots = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=-1)
        pressure_field = p.ravel()
    elif u.ndim == 2:
        # 2D 切片
        xx, yy = np.meshgrid(x, y, indexing='ij')
        coords_3d = np.stack([
            xx.ravel(),
            yy.ravel(),
            np.full(xx.size, (z.min() + z.max()) / 2)
        ], axis=-1)
        
        snapshots = np.stack([u.ravel(), v.ravel(), w.ravel()], axis=-1)
        pressure_field = p.ravel()
    else:
        raise ValueError(f"不支援的資料維度: {u.ndim}")
    
    print(f"  ✅ 資料形狀: coords={coords_3d.shape}, snapshots={snapshots.shape}")
    print(f"  📊 物理參數: Re_τ={Re_tau:.1f}, u_τ={u_tau:.5f}, ν={nu:.2e}")
    
    return {
        'coordinates': coords_3d,
        'snapshots': snapshots,
        'pressure': pressure_field,
        'x': x, 'y': y, 'z': z,
        'Re_tau': Re_tau,
        'u_tau': u_tau,
        'nu': nu
    }


def select_sensors_standard_qr(
    snapshots: np.ndarray,
    coords_3d: np.ndarray,
    n_sensors: int
) -> Tuple[np.ndarray, Dict[str, float]]:
    """標準 QR-Pivot 選點"""
    print(f"\n🔹 標準 QR-Pivot 選點 (K={n_sensors})")
    
    selector = QRPivotSelector(
        rank=min(n_sensors, snapshots.shape[1]),
        energy_threshold=0.99
    )
    
    indices, metrics = selector.select_sensors(
        snapshots,
        n_sensors=n_sensors,
        coords=coords_3d
    )
    
    print(f"  ✅ 選擇 {len(indices)} 個感測點")
    print(f"  📊 條件數: {metrics['condition_number']:.2f}")
    print(f"  📊 能量比例: {metrics['energy_ratio']:.4f}")
    
    return indices, metrics


def select_sensors_physics_guided(
    snapshots: np.ndarray,
    coords_3d: np.ndarray,
    n_sensors: int,
    wall_weight: float,
    u_tau: float,
    nu: float
) -> Tuple[np.ndarray, Dict[str, float]]:
    """物理引導 QR-Pivot 選點"""
    print(f"\n🔹 物理引導 QR-Pivot 選點 (K={n_sensors}, wall_weight={wall_weight})")
    
    selector = PhysicsGuidedQRPivotSelector(
        rank=min(n_sensors, snapshots.shape[1]),
        energy_threshold=0.99,
        wall_weight=wall_weight,
        wall_threshold=0.1,
        threshold_type='y_over_h',
        friction_velocity=u_tau,
        viscosity=nu
    )
    
    indices, metrics = selector.select_sensors(
        snapshots,
        n_sensors=n_sensors,
        coords=coords_3d
    )
    
    print(f"  ✅ 選擇 {len(indices)} 個感測點")
    print(f"  📊 條件數: {metrics['condition_number']:.2f}")
    print(f"  📊 能量比例: {metrics['energy_ratio']:.4f}")
    print(f"  📊 壁面覆蓋率: {metrics.get('wall_coverage', 0.0):.2%}")
    
    return indices, metrics


def analyze_wall_distribution(
    coords: np.ndarray,
    wall_threshold: float = 0.1
) -> Dict[str, float]:
    """分析感測點的壁面分佈"""
    y = coords[:, 1]
    y_normalized = np.abs(y)  # 假設通道中心 y=0
    
    is_wall = y_normalized > (1.0 - wall_threshold)
    is_log = (y_normalized > wall_threshold) & (y_normalized <= (1.0 - wall_threshold))
    is_center = y_normalized <= wall_threshold
    
    return {
        'wall_fraction': np.sum(is_wall) / len(coords),
        'log_fraction': np.sum(is_log) / len(coords),
        'center_fraction': np.sum(is_center) / len(coords),
        'wall_count': np.sum(is_wall),
        'log_count': np.sum(is_log),
        'center_count': np.sum(is_center)
    }


def plot_sensor_comparison_2d(
    coords_all: np.ndarray,
    indices_standard: np.ndarray,
    indices_physics: np.ndarray,
    output_path: Path
):
    """對比兩種策略的感測點分佈（2D 投影）"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 標準 QR-Pivot
    ax = axes[0]
    coords_std = coords_all[indices_standard]
    ax.scatter(coords_all[:, 0], coords_all[:, 1], 
               c='lightgray', s=1, alpha=0.3, label='All Points')
    ax.scatter(coords_std[:, 0], coords_std[:, 1],
               c='blue', s=50, edgecolors='black', linewidths=0.5,
               label='Standard QR-Pivot', zorder=5)
    ax.axhline(y=-0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Wall Region (y=-0.9)')
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Standard QR-Pivot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 物理引導 QR-Pivot
    ax = axes[1]
    coords_phys = coords_all[indices_physics]
    ax.scatter(coords_all[:, 0], coords_all[:, 1],
               c='lightgray', s=1, alpha=0.3, label='All Points')
    ax.scatter(coords_phys[:, 0], coords_phys[:, 1],
               c='green', s=50, edgecolors='black', linewidths=0.5,
               label='Physics-Guided QR-Pivot', zorder=5)
    ax.axhline(y=-0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Wall Region (y=-0.9)')
    ax.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Physics-Guided QR-Pivot', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'sensor_comparison_2d.png', dpi=300, bbox_inches='tight')
    print(f"  💾 儲存 2D 對比圖: {output_path / 'sensor_comparison_2d.png'}")
    plt.close()


def plot_wall_coverage_comparison(
    dist_standard: Dict[str, float],
    dist_physics: Dict[str, float],
    output_path: Path
):
    """對比壁面覆蓋率"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    layers = ['Wall Layer', 'Log Layer', 'Center Layer']
    standard_fracs = [
        dist_standard['wall_fraction'],
        dist_standard['log_fraction'],
        dist_standard['center_fraction']
    ]
    physics_fracs = [
        dist_physics['wall_fraction'],
        dist_physics['log_fraction'],
        dist_physics['center_fraction']
    ]
    
    x = np.arange(len(layers))
    width = 0.35
    
    ax.bar(x - width/2, standard_fracs, width, label='Standard QR-Pivot', color='blue', alpha=0.7)
    ax.bar(x + width/2, physics_fracs, width, label='Physics-Guided QR-Pivot', color='green', alpha=0.7)
    
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Sensor Fraction', fontsize=12)
    ax.set_title('Wall Coverage Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    # 添加數值標籤
    for i, (std_frac, phys_frac) in enumerate(zip(standard_fracs, physics_fracs)):
        ax.text(i - width/2, std_frac + 0.02, f'{std_frac:.1%}', 
                ha='center', fontsize=10)
        ax.text(i + width/2, phys_frac + 0.02, f'{phys_frac:.1%}',
                ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'wall_coverage_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  💾 儲存壁面覆蓋率對比圖: {output_path / 'wall_coverage_comparison.png'}")
    plt.close()


def plot_metrics_comparison(
    metrics_standard: Dict[str, float],
    metrics_physics: Dict[str, float],
    output_path: Path
):
    """對比品質指標"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 條件數
    ax = axes[0]
    strategies = ['Standard QR', 'Physics-Guided']
    cond_nums = [
        metrics_standard['condition_number'],
        metrics_physics['condition_number']
    ]
    bars = ax.bar(strategies, cond_nums, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Condition Number', fontsize=12)
    ax.set_title('Condition Number Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, cond_nums):
        ax.text(bar.get_x() + bar.get_width()/2, val + val*0.05,
                f'{val:.2f}', ha='center', fontsize=10)
    
    # 能量比例
    ax = axes[1]
    energy_ratios = [
        metrics_standard['energy_ratio'],
        metrics_physics['energy_ratio']
    ]
    bars = ax.bar(strategies, energy_ratios, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Energy Ratio', fontsize=12)
    ax.set_title('Energy Ratio Comparison', fontsize=12, fontweight='bold')
    ax.set_ylim([0.9, 1.0])
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, energy_ratios):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.005,
                f'{val:.4f}', ha='center', fontsize=10)
    
    # 壁面覆蓋率
    ax = axes[2]
    wall_coverages = [
        metrics_standard.get('wall_coverage', 0.0),
        metrics_physics.get('wall_coverage', 0.0)
    ]
    bars = ax.bar(strategies, wall_coverages, color=['blue', 'green'], alpha=0.7)
    ax.set_ylabel('Wall Coverage', fontsize=12)
    ax.set_title('Wall Coverage Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    for bar, val in zip(bars, wall_coverages):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
                f'{val:.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path / 'metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  💾 儲存品質指標對比圖: {output_path / 'metrics_comparison.png'}")
    plt.close()


def scan_wall_weight(
    snapshots: np.ndarray,
    coords_3d: np.ndarray,
    n_sensors: int,
    u_tau: float,
    nu: float,
    output_path: Path
):
    """掃描壁面權重的影響"""
    print(f"\n🔍 掃描壁面權重 (K={n_sensors})")
    
    wall_weights = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    results = {
        'wall_weight': [],
        'condition_number': [],
        'energy_ratio': [],
        'wall_coverage': []
    }
    
    for w in wall_weights:
        print(f"  ⚙️ wall_weight = {w}")
        
        selector = PhysicsGuidedQRPivotSelector(
            rank=min(n_sensors, snapshots.shape[1]),
            energy_threshold=0.99,
            wall_weight=w,
            wall_threshold=0.1,
            threshold_type='y_over_h',
            friction_velocity=u_tau,
            viscosity=nu
        )
        
        indices, metrics = selector.select_sensors(
            snapshots,
            n_sensors=n_sensors,
            coords=coords_3d
        )
        
        results['wall_weight'].append(w)
        results['condition_number'].append(metrics['condition_number'])
        results['energy_ratio'].append(metrics['energy_ratio'])
        results['wall_coverage'].append(metrics.get('wall_coverage', 0.0))
    
    # 繪製權衡曲線
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 條件數 vs. 壁面權重
    ax = axes[0]
    ax.plot(results['wall_weight'], results['condition_number'],
            marker='o', linewidth=2, color='blue')
    ax.set_xlabel('Wall Weight', fontsize=12)
    ax.set_ylabel('Condition Number', fontsize=12)
    ax.set_title('Condition Number vs. Wall Weight', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 能量比例 vs. 壁面權重
    ax = axes[1]
    ax.plot(results['wall_weight'], results['energy_ratio'],
            marker='s', linewidth=2, color='green')
    ax.set_xlabel('Wall Weight', fontsize=12)
    ax.set_ylabel('Energy Ratio', fontsize=12)
    ax.set_title('Energy Ratio vs. Wall Weight', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 壁面覆蓋率 vs. 壁面權重
    ax = axes[2]
    ax.plot(results['wall_weight'], results['wall_coverage'],
            marker='^', linewidth=2, color='red')
    ax.set_xlabel('Wall Weight', fontsize=12)
    ax.set_ylabel('Wall Coverage', fontsize=12)
    ax.set_title('Wall Coverage vs. Wall Weight', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'wall_weight_scan.png', dpi=300, bbox_inches='tight')
    print(f"  💾 儲存壁面權重掃描圖: {output_path / 'wall_weight_scan.png'}")
    plt.close()
    
    # 儲存結果
    results_file = output_path / 'wall_weight_scan_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  💾 儲存掃描結果: {results_file}")


def main():
    parser = argparse.ArgumentParser(description='物理引導 QR-Pivot 感測點視覺化對比')
    parser.add_argument('--jhtdb-data', type=str, required=True,
                       help='JHTDB 資料檔案路徑 (.h5)')
    parser.add_argument('--n-sensors', type=int, default=50,
                       help='感測點數量 K (預設: 50)')
    parser.add_argument('--wall-weight', type=float, default=5.0,
                       help='物理引導壁面權重 (預設: 5.0)')
    parser.add_argument('--wall-weight-scan', action='store_true',
                       help='掃描壁面權重的影響')
    parser.add_argument('--output', type=str, default='./results/physics_guided_comparison',
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("📊 物理引導 QR-Pivot 感測點視覺化對比工具")
    print("=" * 80)
    
    # 載入資料
    data = load_jhtdb_data(args.jhtdb_data)
    
    # 標準 QR-Pivot 選點
    indices_standard, metrics_standard = select_sensors_standard_qr(
        data['snapshots'],
        data['coordinates'],
        args.n_sensors
    )
    coords_standard = data['coordinates'][indices_standard]
    dist_standard = analyze_wall_distribution(coords_standard)
    
    # 物理引導 QR-Pivot 選點
    indices_physics, metrics_physics = select_sensors_physics_guided(
        data['snapshots'],
        data['coordinates'],
        args.n_sensors,
        args.wall_weight,
        data['u_tau'],
        data['nu']
    )
    coords_physics = data['coordinates'][indices_physics]
    dist_physics = analyze_wall_distribution(coords_physics)
    
    # 視覺化對比
    print("\n🎨 生成視覺化圖表...")
    plot_sensor_comparison_2d(
        data['coordinates'],
        indices_standard,
        indices_physics,
        output_path
    )
    
    plot_wall_coverage_comparison(
        dist_standard,
        dist_physics,
        output_path
    )
    
    plot_metrics_comparison(
        metrics_standard,
        metrics_physics,
        output_path
    )
    
    # 壁面權重掃描
    if args.wall_weight_scan:
        scan_wall_weight(
            data['snapshots'],
            data['coordinates'],
            args.n_sensors,
            data['u_tau'],
            data['nu'],
            output_path
        )
    
    # 儲存統計摘要
    summary = {
        'n_sensors': args.n_sensors,
        'wall_weight': args.wall_weight,
        'standard_qr': {
            'metrics': metrics_standard,
            'distribution': dist_standard
        },
        'physics_guided': {
            'metrics': metrics_physics,
            'distribution': dist_physics
        }
    }
    
    summary_file = output_path / 'comparison_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n💾 儲存對比摘要: {summary_file}")
    
    print("\n" + "=" * 80)
    print("✅ 對比分析完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()
