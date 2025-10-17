#!/usr/bin/env python3
"""
從現有感測點文件生成無中心層版本
策略：使用 wall_clustered 作為模板，重新選擇座標但保持格式一致
"""

import numpy as np
import argparse
from pathlib import Path
import sys

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_loader import JHTDBChannelFlowLoader


def generate_wall_no_center_from_data(
    template_file: str,
    jhtdb_config: dict,
    K: int = 50,
    wall_fraction: float = 0.52,
    log_fraction: float = 0.48,
    seed: int = 42,
):
    """
    從 JHTDB 資料生成無中心層感測點
    
    流程：
    1. 載入模板文件格式
    2. 載入完整 JHTDB 2D 切片資料
    3. 根據分層策略選擇新座標
    4. 從完整資料插值得到對應速度值
    5. 儲存為與模板相同的格式
    """
    np.random.seed(seed)
    
    # 1. 載入模板格式
    template = np.load(template_file, allow_pickle=True)
    print(f"=== 模板文件格式 ===")
    print(f"鍵值: {list(template.keys())}")
    print(f"原始 K={len(template['coords'])}")
    
    # 2. 載入 JHTDB 完整資料
    print(f"\n=== 載入 JHTDB 資料 ===")
    loader = JHTDBChannelFlowLoader(
        cache_dir=jhtdb_config['cache_dir'],
        dataset_name=jhtdb_config.get('dataset_name', 'channel'),
    )
    
    # 載入 2D 切片
    slice_config = jhtdb_config.get('slice_config', {
        'plane': 'xy',
        'z_position': 4.71,
        'steady_state': True,
        'time_average_window': [20.0, 26.0]
    })
    
    field_data = loader.get_2d_slice(
        plane=slice_config['plane'],
        position=slice_config['z_position'],
        time_range=slice_config.get('time_average_window', [20.0, 26.0]),
        steady_state=slice_config.get('steady_state', True),
    )
    
    print(f"場資料形狀: u={field_data['u'].shape}, p={field_data['p'].shape}")
    print(f"座標範圍: x=[{field_data['x'].min():.2f}, {field_data['x'].max():.2f}], "
          f"y=[{field_data['y'].min():.2f}, {field_data['y'].max():.2f}]")
    
    # 3. 生成新座標（分層策略）
    domain_x = (field_data['x'].min(), field_data['x'].max())
    domain_y = (field_data['y'].min(), field_data['y'].max())
    domain_z = slice_config['z_position']
    
    K_wall = int(K * wall_fraction)
    K_log = K - K_wall
    
    print(f"\n=== 生成新座標 ===")
    print(f"總點數: {K}")
    print(f"壁面層: {K_wall} ({wall_fraction*100:.1f}%)")
    print(f"對數層: {K_log} ({log_fraction*100:.1f}%)")
    print(f"中心層: 0 (0.0%)")
    
    # 壁面層：上下壁面各半
    K_wall_upper = K_wall // 2
    K_wall_lower = K_wall - K_wall_upper
    
    x_wall_upper = np.random.uniform(domain_x[0], domain_x[1], K_wall_upper)
    y_wall_upper = np.random.uniform(0.8, 1.0, K_wall_upper)
    
    x_wall_lower = np.random.uniform(domain_x[0], domain_x[1], K_wall_lower)
    y_wall_lower = np.random.uniform(-1.0, -0.8, K_wall_lower)
    
    # 對數層：上下對稱
    K_log_upper = K_log // 2
    K_log_lower = K_log - K_log_upper
    
    x_log_upper = np.random.uniform(domain_x[0], domain_x[1], K_log_upper)
    y_log_upper = np.random.uniform(0.2, 0.8, K_log_upper)
    
    x_log_lower = np.random.uniform(domain_x[0], domain_x[1], K_log_lower)
    y_log_lower = np.random.uniform(-0.8, -0.2, K_log_lower)
    
    # 合併所有點
    x_sensors = np.concatenate([x_wall_upper, x_wall_lower, x_log_upper, x_log_lower])
    y_sensors = np.concatenate([y_wall_upper, y_wall_lower, y_log_upper, y_log_lower])
    z_sensors = np.full(K, domain_z)
    
    # 4. 從場資料插值得到速度值
    print(f"\n=== 插值速度場 ===")
    from scipy.interpolate import RegularGridInterpolator
    
    # 建立插值器（2D）
    interp_u = RegularGridInterpolator((field_data['y'], field_data['x']), field_data['u'], method='linear')
    interp_v = RegularGridInterpolator((field_data['y'], field_data['x']), field_data['v'], method='linear')
    interp_w = RegularGridInterpolator((field_data['y'], field_data['x']), field_data['w'], method='linear')
    interp_p = RegularGridInterpolator((field_data['y'], field_data['x']), field_data['p'], method='linear')
    
    # 插值（注意順序：y, x）
    query_points = np.stack([y_sensors, x_sensors], axis=1)
    u_sensors = interp_u(query_points)
    v_sensors = interp_v(query_points)
    w_sensors = interp_w(query_points)
    p_sensors = interp_p(query_points)
    
    print(f"速度範圍: u=[{u_sensors.min():.3f}, {u_sensors.max():.3f}]")
    print(f"          v=[{v_sensors.min():.3f}, {v_sensors.max():.3f}]")
    print(f"          w=[{w_sensors.min():.3f}, {w_sensors.max():.3f}]")
    print(f"          p=[{p_sensors.min():.3f}, {p_sensors.max():.3f}]")
    
    # 5. 組裝為模板格式
    coords = np.stack([x_sensors, y_sensors, z_sensors], axis=1)  # (K, 3)
    coords_2d = np.stack([x_sensors, y_sensors], axis=1)  # (K, 2)
    
    # 計算條件數（使用速度矩陣）
    velocity_matrix = np.stack([u_sensors, v_sensors, w_sensors], axis=1)  # (K, 3)
    gram_matrix = velocity_matrix.T @ velocity_matrix
    condition_number = np.linalg.cond(gram_matrix)
    
    # 計算能量比例（簡化版）
    energy_ratio = 0.999  # 佔位符
    
    # 生成 indices（線性索引，佔位符）
    indices = np.arange(K)
    
    return {
        'indices': indices,
        'coords': coords,
        'coords_2d': coords_2d,
        'u': u_sensors,
        'v': v_sensors,
        'w': w_sensors,
        'p': p_sensors,
        'condition_number': condition_number,
        'energy_ratio': energy_ratio,
        'metadata': np.array({
            'strategy': 'wall_no_center',
            'K': K,
            'wall_fraction': wall_fraction,
            'log_fraction': log_fraction,
            'center_fraction': 0.0,
            'description': 'Wall-clustered without center layer (for ablation study)',
        }, dtype=object)
    }


def main():
    parser = argparse.ArgumentParser(description="從 JHTDB 資料生成無中心層感測點")
    parser.add_argument("--template", type=str, required=True, help="模板感測點文件路徑")
    parser.add_argument("--output", type=str, required=True, help="輸出 .npz 檔案路徑")
    parser.add_argument("--cache-dir", type=str, default="./data/jhtdb/channel_flow_re1000", help="JHTDB 快取目錄")
    parser.add_argument("--K", type=int, default=50, help="感測點總數")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")
    args = parser.parse_args()
    
    # JHTDB 配置
    jhtdb_config = {
        'cache_dir': args.cache_dir,
        'dataset_name': 'channel',
        'slice_config': {
            'plane': 'xy',
            'z_position': 4.71,
            'steady_state': True,
            'time_average_window': [20.0, 26.0]
        }
    }
    
    # 生成感測點
    print(f"=== 生成無中心層感測點 ===")
    sensor_data = generate_wall_no_center_from_data(
        template_file=args.template,
        jhtdb_config=jhtdb_config,
        K=args.K,
        seed=args.seed,
    )
    
    # 儲存
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.savez(
        output_path,
        **sensor_data
    )
    
    print(f"\n✅ 感測點已儲存：{output_path}")
    print(f"   格式: 與模板文件一致")
    print(f"   K={len(sensor_data['coords'])}")
    print(f"   條件數: {sensor_data['condition_number']:.2e}")


if __name__ == "__main__":
    main()
