#!/usr/bin/env python3
"""
QR Pivoting 修正驗證測試

實作10分鐘驗證清單的自動化測試，驗證以下修正：
1. 特徵標準化 + 脈動量提取
2. 循環平移測試（週期邊界）
3. POD-DEIM vs 原始QR-Pivot
4. 條件數、能量比例、空間分佈對比

使用方式：
    # 快速測試（使用mock資料）
    python tests/validate_qr_pivoting_fix.py --mode mock

    # 完整測試（需要JHTDB資料）
    python tests/validate_qr_pivoting_fix.py --mode full --data-path data/jhtdb/channel_flow_re1000/
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import argparse
import logging
from typing import Dict, Any, List
import json

# 標記所有測試為 skip（這是獨立驗證腳本，不應由 pytest 收集）
pytestmark = pytest.mark.skip(reason="Standalone validation script, run directly with python")

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import (
    QRPivotSelector,
    PODQREIMSelector,
    prepare_turbulence_features,
    PeriodicBoundaryHandler,
    apply_min_distance_constraint,
    evaluate_sensor_placement
)
from pinnx.dataio.jhtdb_cutout_loader import JHTDBCutoutLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 測試1：特徵標準化驗證
# ============================================================================

def test_feature_standardization(data_matrix: np.ndarray, n_sensors: int = 50) -> Dict[str, Any]:
    """
    測試1：驗證特徵標準化對條件數的改善

    測試內容：
    - 對比標準化前後的條件數
    - 驗證標準化能降低條件數
    """
    logger.info("\n" + "="*80)
    logger.info("測試1：特徵標準化驗證")
    logger.info("="*80)

    # 未標準化
    selector_raw = QRPivotSelector(mode='row', pivoting=True)
    indices_raw, metrics_raw = selector_raw.select_sensors(data_matrix.copy(), n_sensors)

    # 標準化
    data_normalized = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-8)
    selector_norm = QRPivotSelector(mode='row', pivoting=True)
    indices_norm, metrics_norm = selector_norm.select_sensors(data_normalized, n_sensors)

    # 結果對比
    cond_raw = metrics_raw['condition_number']
    cond_norm = metrics_norm['condition_number']
    improvement = (cond_raw - cond_norm) / cond_raw * 100

    logger.info(f"  未標準化條件數: {cond_raw:.2f}")
    logger.info(f"  標準化後條件數: {cond_norm:.2f}")
    logger.info(f"  改善幅度: {improvement:.1f}%")

    # 判定是否通過
    passed = cond_norm < cond_raw
    status = "✅ 通過" if passed else "❌ 失敗"
    logger.info(f"  測試結果: {status}")

    return {
        'test_name': 'feature_standardization',
        'cond_raw': float(cond_raw),
        'cond_normalized': float(cond_norm),
        'improvement_percent': float(improvement),
        'passed': bool(passed)
    }


# ============================================================================
# 測試2：循環平移測試（週期邊界）
# ============================================================================

def test_circular_shift(data_matrix: np.ndarray,
                       coords: np.ndarray,
                       n_sensors: int = 50,
                       n_shifts: int = 5) -> Dict[str, Any]:
    """
    測試2：驗證循環平移能消除「入口聚集」

    測試內容：
    - 對資料做多次隨機循環平移
    - 對每次平移執行QR-Pivot
    - 驗證選中點的x座標分佈是否隨平移而移動
    """
    logger.info("\n" + "="*80)
    logger.info("測試2：循環平移測試（週期邊界）")
    logger.info("="*80)

    periodic_handler = PeriodicBoundaryHandler(periodic_axes=[0, 2])  # x, z方向週期

    # 原始選點
    selector_orig = QRPivotSelector(mode='row', pivoting=True)
    indices_orig, _ = selector_orig.select_sensors(data_matrix, n_sensors)
    x_coords_orig = coords[indices_orig, 0]
    x_mean_orig = x_coords_orig.mean()
    x_std_orig = x_coords_orig.std()

    logger.info(f"  原始選點 x 座標分佈: mean={x_mean_orig:.3f}, std={x_std_orig:.3f}")

    # 多次循環平移
    x_means_shifted = []
    x_stds_shifted = []

    for shift_idx in range(n_shifts):
        # 執行循環平移
        shifted_matrices, shifted_coords_list = periodic_handler.circular_shift_augmentation(
            data_matrix, coords, n_shifts=1, random_seed=shift_idx
        )

        # 對平移後的資料執行QR-Pivot
        selector_shift = QRPivotSelector(mode='row', pivoting=True)
        indices_shift, _ = selector_shift.select_sensors(shifted_matrices[0], n_sensors)

        # 記錄x座標分佈
        x_coords_shift = shifted_coords_list[0][indices_shift, 0]
        x_means_shifted.append(x_coords_shift.mean())
        x_stds_shifted.append(x_coords_shift.std())

    x_means_shifted = np.array(x_means_shifted)
    x_stds_shifted = np.array(x_stds_shifted)

    logger.info(f"  平移後 x 座標分佈範圍: mean=[{x_means_shifted.min():.3f}, {x_means_shifted.max():.3f}]")
    logger.info(f"                        std=[{x_stds_shifted.min():.3f}, {x_stds_shifted.max():.3f}]")

    # 判定是否通過：平移後的x均值應該有顯著變化
    x_mean_range = x_means_shifted.max() - x_means_shifted.min()
    x_mean_variation = x_mean_range / (coords[:, 0].max() - coords[:, 0].min())

    passed = x_mean_variation > 0.1  # x均值變化 > 10% 域範圍

    status = "✅ 通過" if passed else "❌ 失敗"
    logger.info(f"  x 均值變化比例: {x_mean_variation:.2%}")
    logger.info(f"  測試結果: {status} (期望 > 10%)")

    return {
        'test_name': 'circular_shift',
        'x_mean_original': float(x_mean_orig),
        'x_mean_range_shifted': float(x_mean_range),
        'x_mean_variation_percent': float(x_mean_variation * 100),
        'passed': bool(passed)
    }


# ============================================================================
# 測試3：POD-DEIM vs 原始QR-Pivot
# ============================================================================

def test_pod_deim_comparison(data_matrix: np.ndarray,
                             n_sensors: int = 50) -> Dict[str, Any]:
    """
    測試3：對比POD-DEIM與原始QR-Pivot

    測試內容：
    - 條件數對比
    - 能量比例對比
    - 子空間覆蓋率對比
    """
    logger.info("\n" + "="*80)
    logger.info("測試3：POD-DEIM vs 原始QR-Pivot")
    logger.info("="*80)

    # 原始QR-Pivot
    selector_qr = QRPivotSelector(mode='row', pivoting=True)
    indices_qr, metrics_qr = selector_qr.select_sensors(data_matrix, n_sensors)

    # POD-DEIM
    selector_pod_deim = PODQREIMSelector(
        n_modes=min(20, data_matrix.shape[1]),
        energy_threshold=0.95,
        use_qr_pivot=True
    )
    indices_pod_deim, metrics_pod_deim = selector_pod_deim.select_sensors(data_matrix, n_sensors)

    # 指標對比
    logger.info(f"\n  QR-Pivot 指標:")
    logger.info(f"    條件數: {metrics_qr['condition_number']:.2f}")
    logger.info(f"    能量比例: {metrics_qr['energy_ratio']:.3f}")
    logger.info(f"    子空間覆蓋率: {metrics_qr['subspace_coverage']:.3f}")

    logger.info(f"\n  POD-DEIM 指標:")
    logger.info(f"    條件數: {metrics_pod_deim['condition_number']:.2f}")
    logger.info(f"    能量比例: {metrics_pod_deim['energy_ratio']:.3f}")
    logger.info(f"    子空間覆蓋率: {metrics_pod_deim['subspace_coverage']:.3f}")
    logger.info(f"    POD模態數: {metrics_pod_deim['n_pod_modes']}")
    logger.info(f"    POD能量比例: {metrics_pod_deim['pod_energy_ratio']:.3f}")

    # 判定是否通過：POD-DEIM應該有更好的能量捕捉
    passed = (
        metrics_pod_deim['energy_ratio'] >= metrics_qr['energy_ratio'] * 0.9 and
        metrics_pod_deim['condition_number'] < metrics_qr['condition_number'] * 1.2
    )

    status = "✅ 通過" if passed else "❌ 失敗"
    logger.info(f"\n  測試結果: {status}")

    return {
        'test_name': 'pod_deim_comparison',
        'qr_metrics': {
            'condition_number': float(metrics_qr['condition_number']),
            'energy_ratio': float(metrics_qr['energy_ratio']),
            'subspace_coverage': float(metrics_qr['subspace_coverage'])
        },
        'pod_deim_metrics': {
            'condition_number': float(metrics_pod_deim['condition_number']),
            'energy_ratio': float(metrics_pod_deim['energy_ratio']),
            'subspace_coverage': float(metrics_pod_deim['subspace_coverage']),
            'n_pod_modes': int(metrics_pod_deim['n_pod_modes']),
            'pod_energy_ratio': float(metrics_pod_deim['pod_energy_ratio'])
        },
        'passed': bool(passed)
    }


# ============================================================================
# 測試4：脈動量特徵提取
# ============================================================================

def test_fluctuation_features(snapshots: np.ndarray, n_sensors: int = 50) -> Dict[str, Any]:
    """
    測試4：驗證脈動量特徵提取能改善選點品質

    測試內容：
    - 對比原始快照 vs 脈動量的選點品質
    - 驗證脈動量能提升能量捕捉
    """
    logger.info("\n" + "="*80)
    logger.info("測試4：脈動量特徵提取")
    logger.info("="*80)

    # 原始快照
    data_raw = snapshots.T  # [n_locations, n_time]
    selector_raw = QRPivotSelector(mode='row', pivoting=True)
    indices_raw, metrics_raw = selector_raw.select_sensors(data_raw, n_sensors)

    # 脈動量特徵
    data_fluctuation = prepare_turbulence_features(snapshots, method='fluctuation')
    selector_fluct = QRPivotSelector(mode='row', pivoting=True)
    indices_fluct, metrics_fluct = selector_fluct.select_sensors(data_fluctuation, n_sensors)

    # 指標對比
    logger.info(f"  原始快照:")
    logger.info(f"    條件數: {metrics_raw['condition_number']:.2f}")
    logger.info(f"    能量比例: {metrics_raw['energy_ratio']:.3f}")

    logger.info(f"\n  脈動量特徵:")
    logger.info(f"    條件數: {metrics_fluct['condition_number']:.2f}")
    logger.info(f"    能量比例: {metrics_fluct['energy_ratio']:.3f}")

    # 判定是否通過
    improvement = (metrics_fluct['energy_ratio'] - metrics_raw['energy_ratio']) / metrics_raw['energy_ratio'] * 100
    passed = improvement > 0  # 脈動量應該改善能量捕捉

    status = "✅ 通過" if passed else "❌ 失敗"
    logger.info(f"\n  能量比例改善: {improvement:.1f}%")
    logger.info(f"  測試結果: {status}")

    return {
        'test_name': 'fluctuation_features',
        'raw_metrics': {
            'condition_number': float(metrics_raw['condition_number']),
            'energy_ratio': float(metrics_raw['energy_ratio'])
        },
        'fluctuation_metrics': {
            'condition_number': float(metrics_fluct['condition_number']),
            'energy_ratio': float(metrics_fluct['energy_ratio'])
        },
        'improvement_percent': float(improvement),
        'passed': bool(passed)
    }


# ============================================================================
# 測試5：最小距離約束
# ============================================================================

def test_min_distance_constraint(data_matrix: np.ndarray,
                                coords: np.ndarray,
                                n_sensors: int = 50) -> Dict[str, Any]:
    """
    測試5：驗證最小距離約束能消除點簇集

    測試內容：
    - 應用最小距離約束前後的空間分佈
    - 計算最小距離改善
    """
    logger.info("\n" + "="*80)
    logger.info("測試5：最小距離約束")
    logger.info("="*80)

    # 原始選點
    selector = QRPivotSelector(mode='row', pivoting=True)
    indices_orig, _ = selector.select_sensors(data_matrix, n_sensors)

    # 計算原始最小距離
    coords_orig = coords[indices_orig]
    min_dist_orig = np.inf
    for i in range(len(coords_orig)):
        for j in range(i+1, len(coords_orig)):
            dist = np.linalg.norm(coords_orig[i] - coords_orig[j])
            min_dist_orig = min(min_dist_orig, dist)

    logger.info(f"  原始選點最小距離: {min_dist_orig:.4f}")

    # 應用最小距離約束
    min_distance_threshold = min_dist_orig * 1.5  # 提升50%
    indices_refined = apply_min_distance_constraint(
        indices_orig, coords, min_distance_threshold
    )

    # 計算約束後最小距離
    coords_refined = coords[indices_refined]
    min_dist_refined = np.inf
    for i in range(len(coords_refined)):
        for j in range(i+1, len(coords_refined)):
            dist = np.linalg.norm(coords_refined[i] - coords_refined[j])
            min_dist_refined = min(min_dist_refined, dist)

    logger.info(f"  約束後選點最小距離: {min_dist_refined:.4f}")
    logger.info(f"  最小距離閾值: {min_distance_threshold:.4f}")

    # 判定是否通過
    passed = min_dist_refined >= min_distance_threshold * 0.95

    status = "✅ 通過" if passed else "❌ 失敗"
    logger.info(f"  測試結果: {status}")

    return {
        'test_name': 'min_distance_constraint',
        'min_dist_original': float(min_dist_orig),
        'min_dist_refined': float(min_dist_refined),
        'min_dist_threshold': float(min_distance_threshold),
        'n_sensors_original': int(len(indices_orig)),
        'n_sensors_refined': int(len(indices_refined)),
        'passed': bool(passed)
    }


# ============================================================================
# Mock 資料生成
# ============================================================================

def generate_mock_data(n_locations: int = 256,
                      n_snapshots: int = 50,
                      n_modes: int = 5) -> tuple:
    """生成測試用的mock湍流資料"""
    logger.info(f"\n生成Mock資料: {n_locations} 個空間點, {n_snapshots} 個時間快照")

    # 生成2D網格座標
    nx, ny = int(np.sqrt(n_locations)), int(np.sqrt(n_locations))
    x = np.linspace(0, 2*np.pi, nx)
    y = np.linspace(-1, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    coords = np.stack([X.flatten(), Y.flatten(), np.zeros(n_locations)], axis=1)

    # 生成低維湍流場（POD模態疊加）
    t = np.linspace(0, 2*np.pi, n_snapshots)
    snapshots = np.zeros((n_snapshots, n_locations))

    for mode_idx in range(n_modes):
        # 空間模態（正弦波）
        spatial_mode = np.sin((mode_idx + 1) * X) * np.cos((mode_idx + 1) * Y)
        # 時間係數（衰減振盪）
        temporal_coeff = np.cos((mode_idx + 1) * t) * np.exp(-0.1 * mode_idx)
        # 疊加
        snapshots += spatial_mode.flatten()[np.newaxis, :] * temporal_coeff[:, np.newaxis]

    # 添加噪聲
    snapshots += 0.05 * np.random.randn(n_snapshots, n_locations)

    data_matrix = snapshots.T  # [n_locations, n_snapshots]

    logger.info(f"  座標形狀: {coords.shape}")
    logger.info(f"  資料矩陣形狀: {data_matrix.shape}")
    logger.info(f"  快照形狀: {snapshots.shape}")

    return data_matrix, coords, snapshots


def _build_snapshots_from_fields(fields: Dict[str, np.ndarray]) -> np.ndarray:
    snapshots = []
    for key in ['u', 'v', 'w', 'p']:
        if key in fields:
            snapshots.append(fields[key].reshape(-1))

    if not snapshots:
        raise ValueError("JHTDB 資料中未找到可用場 (u/v/w/p)")

    return np.stack(snapshots, axis=0)


def _subsample_locations(data_matrix: np.ndarray,
                         coords: np.ndarray,
                         snapshots: np.ndarray,
                         max_points: int,
                         random_seed: int = 0) -> tuple:
    if coords.shape[0] <= max_points:
        return data_matrix, coords, snapshots

    rng = np.random.default_rng(random_seed)
    indices = rng.choice(coords.shape[0], size=max_points, replace=False)
    return data_matrix[indices], coords[indices], snapshots[:, indices]


def load_jhtdb_data(data_path: str, max_points: int = 20000) -> tuple:
    """
    載入 JHTDB 資料並轉成 QR 測試格式

    支援：
    - .npz (含 coords 或 x/y/z + u/v/w/p)
    - 目錄（優先使用 cutout_*.npz）
    """
    data_path_obj = Path(data_path)

    if data_path_obj.is_dir():
        cutout_files = sorted(data_path_obj.glob("cutout_*.npz"))
        if cutout_files:
            data_path_obj = min(cutout_files, key=lambda p: p.stat().st_size)
            logger.info(f"使用 cutout 檔案: {data_path_obj}")
        else:
            logger.info("未找到 cutout_*.npz，嘗試使用 HDF5 loader")
            loader = JHTDBCutoutLoader(data_dir=str(data_path_obj / "raw"))
            coords_dict = loader.load_coordinates()
            u, v, w = loader.load_velocity()
            fields = {'u': u, 'v': v, 'w': w}
            coords = np.stack(
                np.meshgrid(coords_dict['x'], coords_dict['y'], coords_dict['z'], indexing='ij'),
                axis=-1
            ).reshape(-1, 3)
            snapshots = _build_snapshots_from_fields(fields)
            data_matrix = snapshots.T
            return _subsample_locations(data_matrix, coords, snapshots, max_points=max_points)

    if data_path_obj.suffix == ".npz":
        with np.load(data_path_obj) as data:
            if 'coords' in data:
                coords = data['coords']
            else:
                if all(key in data for key in ['x', 'y', 'z']):
                    coords = np.stack(
                        np.meshgrid(data['x'], data['y'], data['z'], indexing='ij'),
                        axis=-1
                    ).reshape(-1, 3)
                elif all(key in data for key in ['x', 'y']):
                    z = np.array([0.0], dtype=data['x'].dtype)
                    coords = np.stack(
                        np.meshgrid(data['x'], data['y'], z, indexing='ij'),
                        axis=-1
                    ).reshape(-1, 3)
                else:
                    raise ValueError("NPZ 缺少 coords 或 x/y(/z)")

            fields = {key: data[key] for key in ['u', 'v', 'w', 'p'] if key in data}

        snapshots = _build_snapshots_from_fields(fields)
        data_matrix = snapshots.T
        return _subsample_locations(data_matrix, coords, snapshots, max_points=max_points)

    raise ValueError(f"不支援的 data_path 格式: {data_path_obj}")


# ============================================================================
# 主測試流程
# ============================================================================

def run_all_tests(mode: str = 'mock',
                 data_path: str = None,
                 n_sensors: int = 50,
                 max_points: int = 20000,
                 output_dir: str = 'results/qr_pivoting_tests') -> Dict[str, Any]:
    """執行所有測試"""

    logger.info("=" * 80)
    logger.info("QR Pivoting 修正驗證測試")
    logger.info("=" * 80)
    logger.info(f"測試模式: {mode}")
    logger.info(f"感測點數量: {n_sensors}")

    # 準備資料
    if mode == 'mock':
        data_matrix, coords, snapshots = generate_mock_data(
            n_locations=256, n_snapshots=50
        )
    elif mode == 'full':
        if data_path is None:
            raise ValueError("完整測試模式需要提供 --data-path")
        data_matrix, coords, snapshots = load_jhtdb_data(data_path, max_points=max_points)
    else:
        raise ValueError(f"未知的測試模式: {mode}")

    # 執行測試套件
    results = {}

    results['test1'] = test_feature_standardization(data_matrix, n_sensors)
    results['test2'] = test_circular_shift(data_matrix, coords, n_sensors)
    results['test3'] = test_pod_deim_comparison(data_matrix, n_sensors)
    results['test4'] = test_fluctuation_features(snapshots, n_sensors)
    results['test5'] = test_min_distance_constraint(data_matrix, coords, n_sensors)

    # 統計結果
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['passed'])

    logger.info("\n" + "=" * 80)
    logger.info("測試結果總結")
    logger.info("=" * 80)
    logger.info(f"  總測試數: {total_tests}")
    logger.info(f"  通過測試: {passed_tests}")
    logger.info(f"  失敗測試: {total_tests - passed_tests}")
    logger.info(f"  通過率: {passed_tests/total_tests*100:.1f}%")

    for test_name, test_result in results.items():
        status = "✅" if test_result['passed'] else "❌"
        logger.info(f"  {status} {test_result['test_name']}")

    # 保存結果
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    results_file = output_dir_path / 'test_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n測試結果已保存至: {results_file}")

    return results


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='QR Pivoting 修正驗證測試',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--mode', type=str, default='mock',
                       choices=['mock', 'full'],
                       help='測試模式 (mock=快速測試, full=完整測試)')
    parser.add_argument('--data-path', type=str, default=None,
                       help='JHTDB資料路徑（完整測試時需要）')
    parser.add_argument('--n-sensors', type=int, default=50,
                       help='感測點數量')
    parser.add_argument('--max-points', type=int, default=20000,
                       help='full 模式最大點數（隨機子樣本）')
    parser.add_argument('--output', type=str, default='results/qr_pivoting_tests',
                       help='輸出目錄')

    args = parser.parse_args()

    # 執行測試
    results = run_all_tests(
        mode=args.mode,
        data_path=args.data_path,
        n_sensors=args.n_sensors,
        max_points=args.max_points,
        output_dir=args.output
    )

    # 檢查是否全部通過
    all_passed = all(r['passed'] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
