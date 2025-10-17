"""
從 JHTDB 3D 數據生成 2D 切片並進行多 K 值 QR-Pivot 感測點選擇

功能：
1. 從 3D JHTDB Cutout 數據中提取 2D 切片（X-Y 平面）
2. 對 2D 切片進行 QR-Pivot 感測點選擇
3. 支援多個 K 值（20, 50, 80, 100, 500）
4. 生成視覺化與統計報告

輸出：
- 2D 切片數據（HDF5）
- 多 K 值感測點（NPZ）
- 視覺化圖表
- 統計報告（JSON）
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import json
from typing import Dict, List, Tuple
from datetime import datetime

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_cutout_loader import JHTDBCutoutLoader
from pinnx.sensors.qr_pivot import QRPivotSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_2d_slice(loader: JHTDBCutoutLoader, 
                     plane: str = 'xy',
                     slice_position: float | None = None) -> Dict:
    """
    從 3D 數據提取 2D 切片
    
    Args:
        loader: JHTDB 數據載入器
        plane: 切片平面 ('xy', 'xz', 'yz')
        slice_position: 切片位置（如果為 None，取中心）
    
    Returns:
        dict: {
            'coords': {'x': [...], 'y': [...], 'z': [...]},  # 2D 座標
            'u': [...],  # 2D 速度場
            'v': [...],
            'w': [...],
            'p': [...],  # 2D 壓力場
            'slice_info': {...}  # 切片資訊
        }
    """
    logger.info(f"=== 提取 2D 切片：{plane} 平面 ===")
    
    # 載入完整 3D 數據
    state = loader.load_full_state()
    coords_3d = state['coords']
    
    # 確定切片位置
    if plane == 'xy':
        axis = 'z'
        axis_idx = 2
        if slice_position is None:
            slice_position = (coords_3d['z'].max() + coords_3d['z'].min()) / 2
    elif plane == 'xz':
        axis = 'y'
        axis_idx = 1
        if slice_position is None:
            slice_position = 0.0  # 通道中心
    elif plane == 'yz':
        axis = 'x'
        axis_idx = 0
        if slice_position is None:
            slice_position = (coords_3d['x'].max() + coords_3d['x'].min()) / 2
    else:
        raise ValueError(f"不支援的切片平面: {plane}")
    
    # 找到最接近的索引
    axis_coords = coords_3d[axis]
    slice_idx = np.argmin(np.abs(axis_coords - slice_position))
    actual_position = axis_coords[slice_idx]
    
    logger.info(f"切片軸: {axis}, 目標位置: {slice_position:.4f}, 實際位置: {actual_position:.4f} (索引 {slice_idx})")
    
    # 提取 2D 切片
    if plane == 'xy':
        # 固定 Z，提取 X-Y 平面
        u_2d = state['u'][:, :, slice_idx]
        v_2d = state['v'][:, :, slice_idx]
        w_2d = state['w'][:, :, slice_idx]
        p_2d = state['p'][:, :, slice_idx] if state['p'] is not None else None
        
        coords_2d = {
            'x': coords_3d['x'],
            'y': coords_3d['y'],
            'z': np.array([actual_position])  # 單一值
        }
        
    elif plane == 'xz':
        # 固定 Y，提取 X-Z 平面
        u_2d = state['u'][:, slice_idx, :]
        v_2d = state['v'][:, slice_idx, :]
        w_2d = state['w'][:, slice_idx, :]
        p_2d = state['p'][:, slice_idx, :] if state['p'] is not None else None
        
        coords_2d = {
            'x': coords_3d['x'],
            'y': np.array([actual_position]),
            'z': coords_3d['z']
        }
        
    elif plane == 'yz':
        # 固定 X，提取 Y-Z 平面
        u_2d = state['u'][slice_idx, :, :]
        v_2d = state['v'][slice_idx, :, :]
        w_2d = state['w'][slice_idx, :, :]
        p_2d = state['p'][slice_idx, :, :] if state['p'] is not None else None
        
        coords_2d = {
            'x': np.array([actual_position]),
            'y': coords_3d['y'],
            'z': coords_3d['z']
        }
    
    logger.info(f"2D 切片形狀: {u_2d.shape}")
    logger.info(f"速度統計 - U: [{u_2d.min():.4f}, {u_2d.max():.4f}], V: [{v_2d.min():.4f}, {v_2d.max():.4f}]")
    
    slice_data = {
        'coords': coords_2d,
        'u': u_2d,
        'v': v_2d,
        'w': w_2d,
        'p': p_2d,
        'slice_info': {
            'plane': plane,
            'axis': axis,
            'slice_idx': int(slice_idx),
            'position': float(actual_position),
            'shape': u_2d.shape
        }
    }
    
    return slice_data


def qr_pivot_on_2d_slice(slice_data: Dict, 
                         K_values: List[int] = [20, 50, 80, 100, 500],
                         use_multifeature: bool = True,
                         normalize: bool = True) -> Dict:
    """
    對 2D 切片進行多 K 值 QR-Pivot 感測點選擇
    
    Args:
        slice_data: 2D 切片數據
        K_values: K 值列表
        use_multifeature: 是否使用多特徵（壓力 + 梯度 + Laplacian）
        normalize: 是否對特徵進行標準化（Z-Score）
    
    Returns:
        dict: {K: {'indices': [...], 'coords': [...], 'metrics': {...}}}
    """
    logger.info(f"=== QR-Pivot 感測點選擇（K 值: {K_values}）===")
    
    # 構建 2D 座標網格
    plane = slice_data['slice_info']['plane']
    coords = slice_data['coords']
    
    if plane == 'xy':
        X, Y = np.meshgrid(coords['x'], coords['y'], indexing='ij')
        coords_2d = np.column_stack([X.ravel(), Y.ravel()])
        z_fixed = coords['z'][0]
        coords_3d = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z_fixed)])
    elif plane == 'xz':
        X, Z = np.meshgrid(coords['x'], coords['z'], indexing='ij')
        coords_2d = np.column_stack([X.ravel(), Z.ravel()])
        y_fixed = coords['y'][0]
        coords_3d = np.column_stack([X.ravel(), np.full(X.size, y_fixed), Z.ravel()])
    elif plane == 'yz':
        Y, Z = np.meshgrid(coords['y'], coords['z'], indexing='ij')
        coords_2d = np.column_stack([Y.ravel(), Z.ravel()])
        x_fixed = coords['x'][0]
        coords_3d = np.column_stack([np.full(Y.size, x_fixed), Y.ravel(), Z.ravel()])
    
    # 構建特徵矩陣
    if use_multifeature and slice_data['p'] is not None:
        logger.info("使用多特徵矩陣（速度 + 壓力 + 梯度）")
        
        p_2d = slice_data['p']
        u_2d = slice_data['u']
        v_2d = slice_data['v']
        
        # 計算壓力梯度與 Laplacian
        d2p_dx2 = np.zeros_like(p_2d)
        d2p_dy2 = np.zeros_like(p_2d)
        d2p_dz2 = np.zeros_like(p_2d)
        
        if plane == 'xy':
            dp_dx, dp_dy = np.gradient(p_2d, coords['x'], coords['y'])
            d2p_dx2, _ = np.gradient(dp_dx, coords['x'], coords['y'])
            _, d2p_dy2 = np.gradient(dp_dy, coords['x'], coords['y'])
        elif plane == 'xz':
            dp_dx, dp_dz = np.gradient(p_2d, coords['x'], coords['z'])
            d2p_dx2, _ = np.gradient(dp_dx, coords['x'], coords['z'])
            _, d2p_dz2 = np.gradient(dp_dz, coords['x'], coords['z'])
        elif plane == 'yz':
            dp_dy, dp_dz = np.gradient(p_2d, coords['y'], coords['z'])
            d2p_dy2, _ = np.gradient(dp_dy, coords['y'], coords['z'])
            _, d2p_dz2 = np.gradient(dp_dz, coords['y'], coords['z'])
        
        laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2
        
        snapshot_matrix = np.column_stack([
            u_2d.ravel(),
            v_2d.ravel(),
            p_2d.ravel(),
            laplacian.ravel()
        ])
    else:
        logger.info("使用速度場作為特徵矩陣")
        snapshot_matrix = np.column_stack([
            slice_data['u'].ravel(),
            slice_data['v'].ravel(),
            slice_data['w'].ravel()
        ])
    
    logger.info(f"Snapshot 矩陣形狀: {snapshot_matrix.shape}")
    
    # 特徵標準化（Z-Score）
    if normalize:
        logger.info("執行特徵標準化（Z-Score）")
        
        # 計算每列的均值與標準差
        means = np.mean(snapshot_matrix, axis=0, keepdims=True)
        stds = np.std(snapshot_matrix, axis=0, keepdims=True)
        
        # 避免除以零（對於常數列，保留原值）
        stds[stds < 1e-12] = 1.0
        
        snapshot_matrix_normalized = (snapshot_matrix - means) / stds
        
        logger.info(f"標準化前範圍: [{snapshot_matrix.min():.2e}, {snapshot_matrix.max():.2e}]")
        logger.info(f"標準化後範圍: [{snapshot_matrix_normalized.min():.2e}, {snapshot_matrix_normalized.max():.2e}]")
        
        # 使用標準化矩陣
        data_matrix = snapshot_matrix_normalized
    else:
        logger.info("不進行標準化，使用原始特徵")
        data_matrix = snapshot_matrix
    
    # 初始化選擇器
    selector = QRPivotSelector(mode='column', pivoting=True)
    
    # 對每個 K 值進行選擇
    results = {}
    
    for K in K_values:
        logger.info(f"\n--- K = {K} ---")
        
        if K > snapshot_matrix.shape[0]:
            logger.warning(f"K={K} 超過可用點數 {snapshot_matrix.shape[0]}，跳過")
            continue
        
        try:
            indices, metrics = selector.select_sensors(
                data_matrix=data_matrix,  # 使用標準化後的矩陣
                n_sensors=K
            )
            
            sensor_coords_3d = coords_3d[indices]
            
            logger.info(f"選擇完成: {len(indices)} 個感測點")
            logger.info(f"條件數: {metrics['condition_number']:.2e}")
            logger.info(f"能量比例: {metrics.get('energy_ratio', 0):.4f}")
            
            # 提取感測點的流場數據
            sensor_data = {
                'u': slice_data['u'].ravel()[indices],
                'v': slice_data['v'].ravel()[indices],
                'w': slice_data['w'].ravel()[indices],
            }
            
            if slice_data['p'] is not None:
                sensor_data['p'] = slice_data['p'].ravel()[indices]
            
            results[K] = {
                'indices': indices,
                'coords': sensor_coords_3d,  # 3D 座標（用於訓練）
                'coords_2d': coords_2d[indices],  # 2D 座標（用於視覺化）
                'metrics': metrics,
                'field_values': sensor_data
            }
            
        except Exception as e:
            logger.error(f"K={K} 選擇失敗: {e}")
            continue
    
    return results


def visualize_2d_sensors(slice_data: Dict, 
                         sensor_results: Dict,
                         save_dir: Path):
    """視覺化 2D 切片上的感測點分佈"""
    logger.info("\n=== 視覺化 2D 感測點 ===")
    
    plane = slice_data['slice_info']['plane']
    coords = slice_data['coords']
    
    # 創建多子圖
    n_k = len(sensor_results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # 背景場（壓力或速度 U）
    if slice_data['p'] is not None:
        field_bg = slice_data['p']
        field_name = 'Pressure'
    else:
        field_bg = slice_data['u']
        field_name = 'U velocity'
    
    for idx, (K, result) in enumerate(sorted(sensor_results.items())):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        
        # 繪製背景場
        if plane == 'xy':
            im = ax.contourf(coords['x'], coords['y'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=30, marker='x', linewidths=1.5, label=f'K={K}')
            ax.set_xlabel('X (streamwise)')
            ax.set_ylabel('Y (wall-normal)')
        elif plane == 'xz':
            im = ax.contourf(coords['x'], coords['z'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=30, marker='x', linewidths=1.5, label=f'K={K}')
            ax.set_xlabel('X (streamwise)')
            ax.set_ylabel('Z (spanwise)')
        elif plane == 'yz':
            im = ax.contourf(coords['y'], coords['z'], field_bg.T, levels=50, cmap='RdBu_r', alpha=0.6)
            ax.scatter(result['coords_2d'][:, 0], result['coords_2d'][:, 1], 
                      c='black', s=30, marker='x', linewidths=1.5, label=f'K={K}')
            ax.set_xlabel('Y (wall-normal)')
            ax.set_ylabel('Z (spanwise)')
        
        ax.set_title(f'K={K} (κ={result["metrics"]["condition_number"]:.1e})')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        plt.colorbar(im, ax=ax, label=field_name)
    
    # 隱藏多餘的子圖
    for idx in range(n_k, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'QR-Pivot Sensors on 2D Slice ({plane.upper()} plane)', fontsize=16, y=0.995)
    plt.tight_layout()
    
    fig_path = save_dir / f'2d_sensors_{plane}_all_K.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"保存視覺化: {fig_path}")
    plt.close()


def save_results(slice_data: Dict,
                 sensor_results: Dict,
                 save_dir: Path):
    """保存所有結果"""
    logger.info("\n=== 保存結果 ===")
    
    # 1. 保存 2D 切片數據（HDF5）
    slice_file = save_dir / '2d_slice_data.h5'
    with h5py.File(slice_file, 'w') as f:
        # 座標
        for key, val in slice_data['coords'].items():
            f.create_dataset(f'coords/{key}', data=val)
        
        # 場數據
        f.create_dataset('u', data=slice_data['u'])
        f.create_dataset('v', data=slice_data['v'])
        f.create_dataset('w', data=slice_data['w'])
        if slice_data['p'] is not None:
            f.create_dataset('p', data=slice_data['p'])
        
        # 元數據
        f.attrs['plane'] = slice_data['slice_info']['plane']
        f.attrs['slice_position'] = slice_data['slice_info']['position']
        f.attrs['shape'] = slice_data['slice_info']['shape']
    
    logger.info(f"保存 2D 切片: {slice_file}")
    
    # 2. 保存每個 K 的感測點（NPZ）
    for K, result in sensor_results.items():
        sensor_file = save_dir / f'sensors_K{K}_qr_pivot_2d.npz'
        
        np.savez(
            sensor_file,
            indices=result['indices'],
            coords=result['coords'],  # 3D 座標
            coords_2d=result['coords_2d'],  # 2D 座標
            u=result['field_values']['u'],
            v=result['field_values']['v'],
            w=result['field_values']['w'],
            p=result['field_values'].get('p', np.array([])),
            condition_number=result['metrics']['condition_number'],
            energy_ratio=result['metrics'].get('energy_ratio', 0.0),
            metadata=np.array([{
                'K': K,
                'method': 'QR-Pivot',
                'plane': slice_data['slice_info']['plane'],
                'slice_position': slice_data['slice_info']['position']
            }], dtype=object)
        )
        
        logger.info(f"保存 K={K} 感測點: {sensor_file}")
    
    # 3. 保存統計報告（JSON）
    report = {
        'timestamp': datetime.now().isoformat(),
        'slice_info': slice_data['slice_info'],
        'K_values': list(sensor_results.keys()),
        'sensor_statistics': {}
    }
    
    for K, result in sensor_results.items():
        report['sensor_statistics'][f'K{K}'] = {
            'n_sensors': int(K),
            'condition_number': float(result['metrics']['condition_number']),
            'energy_ratio': float(result['metrics'].get('energy_ratio', 0.0)),
            'coords_range': {
                'x': [float(result['coords'][:, 0].min()), float(result['coords'][:, 0].max())],
                'y': [float(result['coords'][:, 1].min()), float(result['coords'][:, 1].max())],
                'z': [float(result['coords'][:, 2].min()), float(result['coords'][:, 2].max())]
            },
            'field_stats': {
                'u': {'min': float(result['field_values']['u'].min()), 
                      'max': float(result['field_values']['u'].max()),
                      'mean': float(result['field_values']['u'].mean())},
                'v': {'min': float(result['field_values']['v'].min()), 
                      'max': float(result['field_values']['v'].max()),
                      'mean': float(result['field_values']['v'].mean())}
            }
        }
    
    report_file = save_dir / '2d_sensors_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"保存統計報告: {report_file}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="從 3D JHTDB 數據生成 2D 切片並進行 QR-Pivot 感測點選擇")
    parser.add_argument('--data-dir', type=str, default='data/jhtdb/channel_flow_re1000/raw',
                       help='JHTDB 數據目錄')
    parser.add_argument('--plane', type=str, default='xy', choices=['xy', 'xz', 'yz'],
                       help='切片平面')
    parser.add_argument('--slice-position', type=float, default=None,
                       help='切片位置（None 為中心）')
    parser.add_argument('--K-values', type=int, nargs='+', default=[20, 50, 80, 100, 500],
                       help='K 值列表')
    parser.add_argument('--output', type=str, default='results/2d_slice_qr_sensors',
                       help='輸出目錄')
    parser.add_argument('--no-multifeature', action='store_true',
                       help='不使用多特徵（僅速度場）')
    parser.add_argument('--no-normalize', action='store_true',
                       help='不進行特徵標準化（預設會標準化）')
    
    args = parser.parse_args()
    
    # 創建輸出目錄
    save_dir = Path(args.output)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("2D 切片 QR-Pivot 感測點生成")
    logger.info("=" * 80)
    logger.info(f"數據目錄: {args.data_dir}")
    logger.info(f"切片平面: {args.plane}")
    logger.info(f"切片位置: {args.slice_position if args.slice_position else '自動（中心）'}")
    logger.info(f"K 值: {args.K_values}")
    logger.info(f"輸出目錄: {args.output}")
    logger.info(f"特徵標準化: {'否' if args.no_normalize else '是'}")
    logger.info(f"多特徵模式: {'否' if args.no_multifeature else '是'}")
    logger.info("=" * 80)
    
    # 步驟 1: 載入數據並提取 2D 切片
    loader = JHTDBCutoutLoader(data_dir=args.data_dir)
    slice_data = extract_2d_slice(loader, plane=args.plane, slice_position=args.slice_position)
    
    # 步驟 2: QR-Pivot 感測點選擇
    sensor_results = qr_pivot_on_2d_slice(
        slice_data, 
        K_values=args.K_values,
        use_multifeature=not args.no_multifeature,
        normalize=not args.no_normalize  # 預設啟用標準化
    )
    
    # 步驟 3: 視覺化
    visualize_2d_sensors(slice_data, sensor_results, save_dir)
    
    # 步驟 4: 保存結果
    save_results(slice_data, sensor_results, save_dir)
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ 2D 切片 QR-Pivot 感測點生成完成！")
    logger.info(f"結果保存於: {save_dir}")
    logger.info("=" * 80)
    
    # 輸出摘要
    logger.info("\n📊 生成摘要：")
    logger.info(f"  - 2D 切片: {args.plane} 平面, 位置 {slice_data['slice_info']['position']:.4f}")
    logger.info(f"  - 切片形狀: {slice_data['slice_info']['shape']}")
    logger.info(f"  - K 值數量: {len(sensor_results)}")
    for K in sorted(sensor_results.keys()):
        cond = sensor_results[K]['metrics']['condition_number']
        logger.info(f"    * K={K}: κ={cond:.2e}")


if __name__ == "__main__":
    main()
