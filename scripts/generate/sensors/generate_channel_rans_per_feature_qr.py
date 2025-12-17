#!/usr/bin/env python3
"""
Per-Feature QR-Pivot 感測器生成腳本

策略：對每個特徵（u, v, w, p, k, ...）獨立執行 QR-Pivot，
選擇 N 個最重要的空間點，然後合併去重。

這確保每個物理量都有代表性的感測點，避免被主導特徵淹沒。

用法：
    python generate_channel_rans_per_feature_qr.py \\
        --rans-npz data/lowfi/channel_rans/rans_k_omega_sst.npz \\
        --n-per-feature 5 \\
        --feature-selection phase_a \\
        --output data/lowfi/channel_rans/sensors_per_feature_5_phase_a.npz
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
from scipy.linalg import qr
import argparse
import logging

# 重用現有的載入與特徵計算函數
from generate_channel_rans_qr_enhanced import (
    load_rans_data_2d_slice,
    compute_enhanced_turbulence_features_2d,
    build_enhanced_data_matrix
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def select_sensors_per_feature(
    data_matrix: np.ndarray,
    feature_names: list,
    n_per_feature: int = 5
) -> tuple:
    """
    Per-feature QR-Pivot 選點策略
    
    Args:
        data_matrix: [n_locations, n_features]
        feature_names: 特徵名稱列表
        n_per_feature: 每個特徵選擇的感測點數
    
    Returns:
        (selected_indices, metrics, per_feature_details)
    """
    n_locations, n_features = data_matrix.shape
    
    logger.info(f"\n🔍 Per-Feature QR-Pivot Selection:")
    logger.info(f"   Features: {n_features}")
    logger.info(f"   Sensors per feature: {n_per_feature}")
    logger.info(f"   Max total sensors: {n_features * n_per_feature}")
    
    # Z-score 標準化
    X_mean = data_matrix.mean(axis=0, keepdims=True)
    X_std = data_matrix.std(axis=0, keepdims=True) + 1e-8
    data_normalized = (data_matrix - X_mean) / X_std
    
    per_feature_details = {}
    all_indices_list = []
    
    # 對每個特徵獨立執行 QR-Pivot
    for i, fname in enumerate(feature_names):
        feature_col = data_normalized[:, i:i+1]
        
        try:
            # QR 分解：feature_col.T = [1, n_locations]
            Q, R, piv = qr(feature_col.T, mode='economic', pivoting=True)
            
            n_select = min(n_per_feature, n_locations)
            feature_indices = piv[:n_select]
            
            # R 對角線 = 重要性指標
            r_diag = np.abs(np.diag(R)[:n_select])
            
            per_feature_details[fname] = {
                'indices': feature_indices,
                'importance': r_diag,
                'n_selected': len(feature_indices)
            }
            
            all_indices_list.append(feature_indices)
            
            logger.info(f"   ✓ {fname:15s}: {len(feature_indices):2d} points, "
                       f"importance [{r_diag.min():.2e}, {r_diag.max():.2e}]")
        
        except Exception as e:
            logger.warning(f"   ✗ {fname:15s}: QR failed ({e})")
            per_feature_details[fname] = {
                'indices': np.array([], dtype=int),
                'importance': np.array([]),
                'n_selected': 0,
                'error': str(e)
            }
    
    # 合併去重
    all_indices = np.concatenate(all_indices_list) if all_indices_list else np.array([], dtype=int)
    unique_indices, unique_counts = np.unique(all_indices, return_counts=True)
    
    # 按出現次數排序（被多個特徵選中的點更重要）
    sort_by_importance = np.argsort(unique_counts)[::-1]
    selected_indices_final = unique_indices[sort_by_importance]
    
    logger.info(f"\n📊 Merging Results:")
    logger.info(f"   Total collected: {len(all_indices)}")
    logger.info(f"   Unique sensors: {len(unique_indices)}")
    logger.info(f"   Deduplication: {len(all_indices) - len(unique_indices)} removed "
               f"({(1 - len(unique_indices)/max(len(all_indices), 1))*100:.1f}%)")
    
    # 統計多特徵感測點
    multi_feature_mask = unique_counts > 1
    if multi_feature_mask.any():
        logger.info(f"   Multi-feature sensors: {multi_feature_mask.sum()} (≥2 features)")
        logger.info(f"   Max coverage: {unique_counts.max()} features at one point")
    
    # 計算整體指標
    selected_data = data_matrix[selected_indices_final, :]
    
    from scipy.linalg import svd
    try:
        _, s, _ = svd(selected_data, full_matrices=False)
        cond_number = s[0] / s[-1] if s[-1] > 1e-15 else np.inf
    except:
        cond_number = np.inf
    
    metrics = {
        'n_features': n_features,
        'n_per_feature': n_per_feature,
        'n_total_selected': len(selected_indices_final),
        'condition_number': float(cond_number),
        'deduplication_rate': float(1 - len(unique_indices) / max(len(all_indices), 1)),
        'multi_feature_count': int(multi_feature_mask.sum()),
        'max_feature_coverage': int(unique_counts.max())
    }
    
    return selected_indices_final, metrics, per_feature_details


def main():
    parser = argparse.ArgumentParser(
        description="Per-feature QR-Pivot sensor generation"
    )
    
    parser.add_argument(
        '--rans-npz',
        type=str,
        default='data/lowfi/channel_rans/rans_k_omega_sst.npz',
        help='RANS data file (NPZ format)'
    )
    
    parser.add_argument(
        '--n-per-feature',
        type=int,
        default=5,
        help='Number of sensors per feature (default: 5)'
    )
    
    parser.add_argument(
        '--feature-selection',
        type=str,
        default='phase_a',
        choices=['full', 'minimal', 'original', 'physics_guided', 'phase_a'],
        help='Feature set to use (default: phase_a)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output NPZ file path'
    )
    
    parser.add_argument(
        '--slice-axis',
        type=str,
        default='z',
        choices=['x', 'y', 'z'],
        help='Slice axis (default: z)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🔬 Per-Feature QR-Pivot Sensor Generation")
    print("=" * 70)
    
    # 1. 載入 RANS 資料
    x, y, fields = load_rans_data_2d_slice(args.rans_npz, args.slice_axis)
    nx, ny = len(x), len(y)
    
    # 載入 Phase A 所需參數
    rans_data = np.load(args.rans_npz)
    nu = float(rans_data['nu']) if 'nu' in rans_data else None
    Re_tau = float(rans_data['Re_tau_estimate']) if 'Re_tau_estimate' in rans_data else None
    
    Lx = x[-1] - x[0] + (x[1] - x[0])
    Ly = y[-1] - y[0]
    dx, dy = x[1] - x[0], y[1] - y[0]
    
    logger.info(f"\n📐 Domain:")
    logger.info(f"   Grid: {nx} × {ny} = {nx*ny:,} points")
    logger.info(f"   X: [{x[0]:.4f}, {x[-1]:.4f}], dx={dx:.6f}")
    logger.info(f"   Y: [{y[0]:.4f}, {y[-1]:.4f}], dy={dy:.6f}")
    if nu and Re_tau:
        logger.info(f"   Re_tau = {Re_tau:.1f}, nu = {nu:.2e}")
    
    # 2. 計算特徵
    logger.info(f"\n🔧 Computing features ({args.feature_selection})...")
    
    compute_phase_a = (args.feature_selection == 'phase_a')
    
    feature_dict = compute_enhanced_turbulence_features_2d(
        fields, dx, dy, Lx, Ly,
        periodic_x=True,
        nu=nu,
        Re_tau=Re_tau,
        y_coords=y,
        compute_phase_a=compute_phase_a
    )
    
    # 3. 建立資料矩陣
    data_matrix, feature_names = build_enhanced_data_matrix(
        feature_dict, args.feature_selection
    )
    
    logger.info(f"   Features: {len(feature_names)}")
    logger.info(f"   Data matrix: {data_matrix.shape}")
    
    # 4. Per-feature QR-Pivot 選點
    selected_indices, metrics, per_feature_details = select_sensors_per_feature(
        data_matrix,
        feature_names,
        n_per_feature=args.n_per_feature
    )
    
    # 5. 提取感測點座標與數值
    coords = np.column_stack([
        np.repeat(x, ny),
        np.tile(y, nx)
    ])
    
    sensor_coords = coords[selected_indices]
    sensor_x = sensor_coords[:, 0]
    sensor_y = sensor_coords[:, 1]
    
    # 6. 保存結果
    if args.output is None:
        args.output = (f"data/lowfi/channel_rans/sensors_per_feature_"
                      f"{args.n_per_feature}_{args.feature_selection}.npz")
    
    # 將 per-feature details 轉為可保存格式
    per_feature_arrays = {}
    for fname, details in per_feature_details.items():
        per_feature_arrays[f'pf_{fname}_indices'] = details['indices']
        per_feature_arrays[f'pf_{fname}_importance'] = details['importance']
    
    np.savez_compressed(
        args.output,
        sensor_points=sensor_coords,
        sensor_indices=selected_indices,
        sensor_x=sensor_x,
        sensor_y=sensor_y,
        K=len(selected_indices),
        n_features=len(feature_names),
        n_per_feature=args.n_per_feature,
        feature_names=feature_names,
        feature_selection=args.feature_selection,
        condition_number=metrics['condition_number'],
        deduplication_rate=metrics['deduplication_rate'],
        multi_feature_count=metrics['multi_feature_count'],
        max_feature_coverage=metrics['max_feature_coverage'],
        method='per_feature_qr_pivot',
        source_file=args.rans_npz,
        slice_axis=args.slice_axis,
        domain_Lx=Lx,
        domain_Ly=Ly,
        **per_feature_arrays
    )
    
    logger.info(f"\n💾 Saved: {args.output}")
    
    # 7. 輸出摘要
    print("\n" + "=" * 70)
    print("✅ Per-Feature QR-Pivot Complete!")
    print("=" * 70)
    print(f"Feature set:      {args.feature_selection} ({len(feature_names)} features)")
    print(f"Sensors/feature:  {args.n_per_feature}")
    print(f"Total sensors:    {len(selected_indices)} (after deduplication)")
    print(f"Condition number: {metrics['condition_number']:.2e}")
    print(f"Deduplication:    {metrics['deduplication_rate']*100:.1f}% removed")
    print(f"Multi-feature:    {metrics['multi_feature_count']} sensors (≥2 features)")
    print(f"Max coverage:     {metrics['max_feature_coverage']} features at one point")
    print("=" * 70)


if __name__ == '__main__':
    main()
