#!/usr/bin/env python
"""
生成分層採樣感測點並保存
用於後續訓練實驗對比
"""
import numpy as np
import argparse
import logging
from pathlib import Path
import sys
sys.path.insert(0, '.')
from pinnx.sensors.stratified_sampling import StratifiedChannelFlowSelector, HybridChannelFlowSelector

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='生成分層採樣感測點')
    parser.add_argument('--data_path', type=str, default='data/jhtdb/channel_flow_re1000/cutout3d_128x128x32.npz',
                        help='輸入資料路徑')
    parser.add_argument('--K', type=int, required=True, help='感測點數量')
    parser.add_argument('--method', type=str, choices=['stratified', 'hybrid'], default='stratified',
                        help='採樣方法')
    parser.add_argument('--output', type=str, default=None, help='輸出路徑 (預設: data/jhtdb/.../sensors_K{K}_{method}.npz)')
    parser.add_argument('--wall_ratio', type=float, default=0.35, help='壁面區比例 (僅 stratified)')
    parser.add_argument('--log_ratio', type=float, default=0.35, help='對數律區比例 (僅 stratified)')
    parser.add_argument('--stratified_ratio', type=float, default=0.7, help='分層點比例 (僅 hybrid)')
    args = parser.parse_args()
    
    # 載入資料
    logger.info(f'載入資料: {args.data_path}')
    data = np.load(args.data_path)
    
    # 構建座標與場
    x, y, z = data['x'], data['y'], data['z']
    u, v, w, p = data['u'], data['v'], data['w'], data['p']
    
    nx, ny, nz = len(x), len(y), len(z)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    field = np.stack([u.ravel(), v.ravel(), w.ravel(), p.ravel()], axis=1)
    
    # 選擇感測點
    if args.method == 'stratified':
        selector = StratifiedChannelFlowSelector(
            wall_ratio=args.wall_ratio,
            log_ratio=args.log_ratio
        )
        sensor_indices, metrics = selector.select_sensors(coords, field, args.K)
        method_suffix = 'stratified'
    else:
        selector = HybridChannelFlowSelector(
            stratified_ratio=args.stratified_ratio
        )
        sensor_indices, metrics = selector.select_sensors(coords, field, args.K)
        method_suffix = 'hybrid'
    
    # 提取感測點
    sensor_coords = coords[sensor_indices]
    sensor_values = field[sensor_indices]
    
    # 從 layer_distribution 提取計數（注意鍵名！）
    layer_dist = metrics['layer_distribution']
    wall_count = layer_dist.get('wall', {}).get('n_selected', 0)
    log_count = layer_dist.get('log_layer', {}).get('n_selected', 0)
    core_count = layer_dist.get('core', {}).get('n_selected', 0)
    
    logger.info(f'\n{"="*60}')
    logger.info(f'感測點生成完成')
    logger.info(f'{"="*60}')
    logger.info(f'方法: {args.method}')
    logger.info(f'總點數: {args.K}')
    logger.info(f'分層分佈:')
    logger.info(f'  - 壁面區 (|y|>0.8):     {wall_count:3d} ({wall_count/args.K*100:5.1f}%)')
    logger.info(f'  - 對數律區 (0.2<|y|≤0.8): {log_count:3d} ({log_count/args.K*100:5.1f}%)')
    logger.info(f'  - 中心區 (|y|≤0.2):      {core_count:3d} ({core_count/args.K*100:5.1f}%)')
    
    # 保存
    if args.output is None:
        output_dir = Path(args.data_path).parent
        output_path = output_dir / f'sensors_K{args.K}_{method_suffix}.npz'
    else:
        output_path = Path(args.output)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存簡化版 metrics
    save_metrics = {
        'wall_count': wall_count,
        'log_count': log_count,
        'core_count': core_count,
        'y_mean': float(metrics['y_mean']),
        'y_std': float(metrics['y_std']),
    }
    
    np.savez(
        output_path,
        indices=sensor_indices,
        coords=sensor_coords,
        values=sensor_values,
        K=args.K,
        method=args.method,
        **save_metrics
    )
    
    logger.info(f'\n✅ 感測點已保存: {output_path}')
    logger.info(f'檔案大小: {output_path.stat().st_size / 1024:.1f} KB\n')

if __name__ == '__main__':
    main()
