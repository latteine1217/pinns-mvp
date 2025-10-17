#!/usr/bin/env python3
"""
QR-Pivot 感測點分佈視覺化工具

功能：
  1. 載入 QR-pivot 選擇的感測點資料
  2. 繪製感測點的空間分佈圖（2D/3D）
  3. 顯示感測點的座標與數值大小
  4. 分析感測點的幾何分佈特性
  5. 比較不同選點策略的差異

使用方式：
  # 從 npz 檔案載入
  python scripts/visualize_qr_sensors.py \
    --input data/jhtdb/sensors_K50.npz \
    --output results/sensor_analysis

  # 從 JHTDB 資料重新計算
  python scripts/visualize_qr_sensors.py \
    --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
    --n-sensors 50 \
    --output results/sensor_analysis

  # 比較多種策略
  python scripts/visualize_qr_sensors.py \
    --jhtdb-data data/jhtdb/channel_flow_re1000.h5 \
    --n-sensors 50 \
    --compare-strategies \
    --output results/sensor_comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import h5py

# 添加專案路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.sensors.qr_pivot import (
    QRPivotSelector, 
    PODBasedSelector, 
    GreedySelector,
    evaluate_sensor_placement
)


def load_sensor_data(input_path: str) -> Dict[str, Any]:
    """載入感測點資料
    
    支援格式：
      1. 標準 QR-Pivot: sensor_indices, sensor_coords, sensor_values
      2. 分層 QR-Pivot: indices, coords, pressure_values, layer_info, metadata
      3. 完整 JHTDB: x, y, z, u, v, w, p + indices
    """
    print(f"📂 載入感測點資料: {input_path}")
    
    input_path_obj = Path(input_path)
    
    if input_path_obj.suffix == '.npz':
        data = np.load(str(input_path_obj), allow_pickle=True)
        
        # 檢查是否為分層 QR-Pivot 格式
        is_stratified = 'layer_info' in data and 'metadata' in data
        
        if is_stratified:
            print("  ✅ 檢測到分層 QR-Pivot 格式")
            result: Dict[str, Any] = {}
            
            # 提取基本資料
            result['indices'] = data['indices']
            result['coordinates'] = data['coords']
            result['values'] = data['pressure_values'].reshape(-1, 1)  # [N, 1] 壓力值
            
            # 提取分層資訊
            layer_info = data['layer_info'].item()
            result['layer_info'] = layer_info
            
            # 提取元數據
            metadata = data['metadata'].item()
            result['metadata'] = metadata
            
            # 計算每個感測點的分層標籤
            layer_labels = []
            for i in range(len(result['indices'])):
                if i < layer_info['wall']['n_selected']:
                    layer_labels.append('wall')
                elif i < layer_info['wall']['n_selected'] + layer_info['log']['n_selected']:
                    layer_labels.append('log')
                else:
                    layer_labels.append('center')
            result['layer_labels'] = np.array(layer_labels)
            
            print(f"     分層資訊: wall={layer_info['wall']['n_selected']}, "
                  f"log={layer_info['log']['n_selected']}, "
                  f"center={layer_info['center']['n_selected']}")
            print(f"     總感測點數: {len(result['indices'])}")
            
            return result
        
        # 標準格式：嘗試不同的鍵名
        possible_keys = {
            'indices': ['sensor_indices', 'indices', 'selected_indices'],
            'coordinates': ['sensor_points', 'sensor_coords', 'coordinates', 'coords', 'positions'],
            'values': ['sensor_data', 'sensor_values', 'values', 'u', 'velocity', 'pressure_values']
        }
        
        result: Dict[str, Any] = {}
        
        # 提取索引
        for key in possible_keys['indices']:
            if key in data:
                result['indices'] = data[key]
                break
        
        # 提取座標
        for key in possible_keys['coordinates']:
            if key in data:
                result['coordinates'] = data[key]
                break
        
        # 提取數值（處理 object array）
        for key in possible_keys['values']:
            if key in data:
                val = data[key]
                # 處理 numpy object array (需要 .item() 提取)
                if val.dtype == np.object_ and val.shape == ():
                    val = val.item()
                
                # 如果是字典，提取速度場
                if isinstance(val, dict):
                    # 假設有 u, v, w, p 鍵
                    if 'u' in val:
                        u = val['u']
                        v = val.get('v', np.zeros_like(u))
                        w = val.get('w', np.zeros_like(u))
                        # 堆疊為 (N, 3) 或 (N, 4) 包含壓力
                        if 'p' in val:
                            result['values'] = np.stack([u, v, w, val['p']], axis=1)
                        else:
                            result['values'] = np.stack([u, v, w], axis=1)
                    result['velocity_magnitude'] = np.linalg.norm(
                        np.stack([val.get('u', 0), val.get('v', 0), val.get('w', 0)], axis=1), axis=1
                    ) if 'u' in val else None
                else:
                    result['values'] = val
                break
        
        # 如果有完整的 JHTDB 資料，提取其他資訊
        if 'x' in data and 'y' in data and 'z' in data:
            # 這是完整的 JHTDB 資料
            x, y, z = data['x'], data['y'], data['z']
            
            if 'indices' in result:
                indices = result['indices']
                # 提取選定點的座標
                if len(x.shape) == 1:
                    # 1D 座標陣列
                    n_total = len(x) * len(y) * len(z)
                    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    X_flat = X.flatten()
                    Y_flat = Y.flatten()
                    Z_flat = Z.flatten()
                    result['coordinates'] = np.stack([
                        X_flat[indices],
                        Y_flat[indices],
                        Z_flat[indices]
                    ], axis=1)
                else:
                    # 已經是網格形式
                    result['coordinates'] = np.stack([
                        x.flatten()[indices],
                        y.flatten()[indices],
                        z.flatten()[indices]
                    ], axis=1)
                
                # 提取速度場
                if 'u' in data:
                    u = data['u'].flatten()[indices]
                    v = data['v'].flatten()[indices] if 'v' in data else np.zeros_like(u)
                    w = data['w'].flatten()[indices] if 'w' in data else np.zeros_like(u)
                    result['values'] = np.stack([u, v, w], axis=1)
        
        print(f"  ✅ 載入成功")
        print(f"     包含鍵: {list(data.keys())}")
        
        return result
    
    elif input_path_obj.suffix in ['.h5', '.hdf5']:
        # HDF5 格式
        with h5py.File(str(input_path_obj), 'r') as f:
            result: Dict[str, Any] = {}
            
            # 列出所有資料集
            print(f"  HDF5 資料集: {list(f.keys())}")
            
            if 'sensor_indices' in f:
                result['indices'] = np.array(f['sensor_indices'])
            if 'sensor_coords' in f:
                result['coordinates'] = np.array(f['sensor_coords'])
            if 'sensor_values' in f:
                result['values'] = np.array(f['sensor_values'])
        
        return result
    
    else:
        raise ValueError(f"不支援的檔案格式: {input_path_obj.suffix}")


def compute_sensors_from_jhtdb(jhtdb_path: str, n_sensors: int, strategy: str = 'qr_pivot', temporal_data_path: Optional[str] = None) -> Dict[str, Any]:
    """從 JHTDB 資料計算感測點
    
    Args:
        jhtdb_path: JHTDB 資料檔案路徑（單時間步或空間資料）
        n_sensors: 感測點數量
        strategy: 選點策略（'qr_pivot', 'pod_based', 'greedy'）
        temporal_data_path: 多時間步快照資料路徑（優先使用，格式：[n_time, nx, ny]）
    
    Returns:
        包含 indices, coordinates, values, metrics 的字典
    
    Notes:
        - 若提供 temporal_data_path，將使用多時間步資料矩陣 [n_locations, n_time]
        - 否則回退到空間切片策略（可能導致秩不足與高條件數）
    """
    print(f"\n🧮 從 JHTDB 資料計算感測點...")
    print(f"   策略: {strategy}")
    print(f"   感測點數量: {n_sensors}")
    
    # ==== 優先使用多時間步資料 ====
    if temporal_data_path is not None:
        print(f"   ⏱️  使用多時間步快照資料: {temporal_data_path}")
        temp_data = np.load(temporal_data_path)
        
        # 提取時間快照 [n_time, nx, ny] 或 [n_time, nx, ny, nz]
        u_snapshots = temp_data['u']
        v_snapshots = temp_data.get('v', np.zeros_like(u_snapshots))
        w_snapshots = temp_data.get('w', np.zeros_like(u_snapshots))
        
        n_time = u_snapshots.shape[0]
        spatial_shape = u_snapshots.shape[1:]
        n_locations = np.prod(spatial_shape)
        
        print(f"   時間步數: {n_time}")
        print(f"   空間形狀: {spatial_shape}")
        print(f"   空間點數: {n_locations}")
        
        # 重組為資料矩陣 [n_locations, n_time]
        # 策略：每個時間步作為一個特徵（snapshot）
        u_matrix = u_snapshots.reshape(n_time, -1).T  # [n_locations, n_time]
        v_matrix = v_snapshots.reshape(n_time, -1).T
        w_matrix = w_snapshots.reshape(n_time, -1).T
        
        # 組合三個物理量的時間演化
        # 選項 1: 僅使用 u 分量（降低內存，保留時間動態）
        data_matrix = u_matrix  # [n_locations, n_time]
        
        # 選項 2: 組合 u, v, w（增加特徵，但可能導致過大矩陣）
        # data_matrix = np.hstack([u_matrix, v_matrix, w_matrix])  # [n_locations, 3*n_time]
        
        print(f"   ✅ 資料矩陣形狀: {data_matrix.shape} [n_locations, n_features]")
        print(f"      矩陣秩（估計）: min({n_locations}, {n_time}) = {min(n_locations, n_time)}")
        
        # 提取座標（從 temporal data）
        x = temp_data['x']
        y = temp_data['y']
        z = temp_data.get('z', None)
        
        # 如果沒有 z，嘗試從 metadata 獲取切片位置
        if z is None and 'metadata' in temp_data:
            import json
            metadata = temp_data['metadata']
            if isinstance(metadata, np.ndarray):
                metadata = metadata.item()
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            
            if 'slice_position' in metadata:
                z_pos = metadata['slice_position']
            else:
                z_pos = 0.0
            z = np.array([z_pos])
        
        # 建立網格
        if len(spatial_shape) == 2:
            # 2D 資料
            X, Y = np.meshgrid(x, y, indexing='ij')
            coords = np.stack([X.flatten(), Y.flatten(), np.zeros(n_locations)], axis=1)
        else:
            # 3D 資料
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
            coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # 速度值：使用第一個時間步
        velocities = np.stack([
            u_snapshots[0].flatten(),
            v_snapshots[0].flatten(),
            w_snapshots[0].flatten()
        ], axis=1)
    
    # ==== 回退到單時間步資料 ====
    else:
        jhtdb_path_obj = Path(jhtdb_path)
        
        if jhtdb_path_obj.suffix == '.npz':
            data = np.load(str(jhtdb_path_obj))
            
            # 提取座標
            # 支援 2D (x, y) 和 3D (x, y, z) 資料
            if 'x' in data and 'y' in data:
                x, y = data['x'], data['y']
                z = data.get('z', None)
                
                # 建立網格
                if z is not None:
                    # 3D 資料
                    if len(x.shape) == 1:
                        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                    else:
                        X, Y, Z = x, y, z
                else:
                    # 2D 資料：從切片位置恢復 Z 座標
                    if len(x.shape) == 1:
                        X, Y = np.meshgrid(x, y, indexing='ij')
                    else:
                        X, Y = x, y
                    
                    # 嘗試從 metadata 獲取切片位置
                    if 'slice_position' in data:
                        z_pos = float(data['slice_position'])
                    else:
                        z_pos = 0.0  # 預設值
                    
                    Z = np.full_like(X, z_pos)
                
                # 提取速度場
                if 'u' in data:
                    u = data['u']
                    v = data.get('v', np.zeros_like(u))
                    w = data.get('w', np.zeros_like(u))
                    
                    # 展平
                    coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
                    velocities = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=1)
                    
                    # 建立資料矩陣 [n_locations, n_features]
                    # 方法：使用空間切片作為「虛擬快照」+ 多物理量組合
                    # ⚠️ 警告：這種方法的秩受限於物理量數量，可能導致高條件數
                    print(f"   ⚠️  使用單時間步資料（回退策略）")
                    
                    # 檢查是否為 3D 資料（有多個層可作為快照）
                    if u.ndim == 3 and min(u.shape) > 3:
                        # 3D 資料：使用多方向切片組合
                        # u.shape = (nx, ny, nz)
                        nx, ny, nz = u.shape
                    
                        # 策略：組合三個物理量 × 三個方向的切片
                        # 1. Z 方向切片（XY 平面）：[nx*ny, nz]
                        u_xy = u.transpose(2, 0, 1).reshape(nz, -1).T  # [nx*ny, nz]
                        v_xy = v.transpose(2, 0, 1).reshape(nz, -1).T
                        w_xy = w.transpose(2, 0, 1).reshape(nz, -1).T
                        
                        # 2. Y 方向切片（XZ 平面）：[nx*nz, ny]
                        u_xz = u.transpose(1, 0, 2).reshape(ny, -1).T  # [nx*nz, ny]
                        v_xz = v.transpose(1, 0, 2).reshape(ny, -1).T
                        w_xz = w.transpose(1, 0, 2).reshape(ny, -1).T
                        
                        # 3. X 方向切片（YZ 平面）：[ny*nz, nx]
                        u_yz = u.transpose(0, 1, 2).reshape(nx, -1).T  # [ny*nz, nx]
                        v_yz = v.transpose(0, 1, 2).reshape(nx, -1).T
                        w_yz = w.transpose(0, 1, 2).reshape(nx, -1).T
                        
                        # 組合策略：使用 XY 平面（通常空間點最多）+ 三個物理量
                        # 這樣得到 [nx*ny, 3*nz] 的資料矩陣
                        data_matrix = np.hstack([u_xy, v_xy, w_xy])  # [16384, 96]
                        
                        # 更新座標（僅使用 XY 平面）
                        coords = np.stack([
                            X[:, :, 0].flatten(),
                            Y[:, :, 0].flatten(),
                            np.full(nx*ny, z[nz//2])  # Z 座標設為中間層的實際值
                        ], axis=1)
                        
                        # 對應的速度值（取中間 Z 層）
                        velocities = np.stack([
                            u[:, :, nz//2].flatten(),
                            v[:, :, nz//2].flatten(),
                            w[:, :, nz//2].flatten()
                        ], axis=1)
                        
                        print(f"  ✅ 使用 3D 多方向切片：")
                        print(f"     - u/v/w × {nz} 個 Z 層 = {3*nz} 個特徵")
                        print(f"     - 空間點數: {nx*ny} (XY 平面)")
                        print(f"  資料矩陣形狀: {data_matrix.shape} [n_locations, n_features]")
                    
                    else:
                        # 2D 資料或 Z 層數太少：回退到原始方法
                        # ⚠️ 警告：僅 3 個特徵，品質指標會偏低
                        data_matrix = velocities  # [n_locations, 3]
                        print(f"  ⚠️  使用 2D 資料或單層 3D：僅 3 個特徵（u,v,w）")
                        print(f"  建議使用 POD-based 選點策略以提升品質")
                    
                else:
                    raise ValueError("JHTDB 資料中未找到速度場 'u'")
            else:
                raise ValueError("JHTDB 資料中未找到座標 'x', 'y'")
        
        elif jhtdb_path_obj.suffix in ['.h5', '.hdf5']:
            with h5py.File(str(jhtdb_path_obj), 'r') as f:
                # 根據實際 HDF5 結構調整
                x = np.array(f['x'])
                y = np.array(f['y'])
                z = np.array(f['z'])
                u = np.array(f['u'])
                v = np.array(f['v'])
                w = np.array(f['w'])
                
                X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
                coords = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
                velocities = np.stack([u.flatten(), v.flatten(), w.flatten()], axis=1)
                data_matrix = velocities
        
        else:
            raise ValueError(f"不支援的檔案格式: {jhtdb_path_obj.suffix}")
    
    print(f"   資料矩陣形狀: {data_matrix.shape}")
    print(f"   座標形狀: {coords.shape}")
    
    # ⭐ 數據標準化（改善條件數）
    # 對每個特徵（列）進行 Z-score 標準化
    data_matrix_normalized = (data_matrix - data_matrix.mean(axis=0)) / (data_matrix.std(axis=0) + 1e-10)
    print(f"   ✅ 已進行數據標準化（Z-score）")
    
    # 選擇感測點
    if strategy == 'qr_pivot':
        selector = QRPivotSelector(mode='row', pivoting=True)
    elif strategy == 'pod_based':
        selector = PODBasedSelector(n_modes=min(10, data_matrix.shape[1]))
    elif strategy == 'greedy':
        selector = GreedySelector(objective='info_gain')
    else:
        raise ValueError(f"未知的策略: {strategy}")
    
    print(f"   執行 {strategy} 選點...")
    selected_indices, metrics = selector.select_sensors(data_matrix_normalized, n_sensors)
    
    print(f"   ✅ 選擇完成: {len(selected_indices)} 個點")
    print(f"      條件數: {metrics.get('condition_number', 'N/A'):.2f}")
    print(f"      能量比例: {metrics.get('energy_ratio', 0.0):.3f}")
    
    result = {
        'indices': selected_indices,
        'coordinates': coords[selected_indices],
        'values': velocities[selected_indices],
        'metrics': metrics,
        'strategy': strategy
    }
    
    return result


def plot_sensor_distribution_2d(sensor_data: Dict[str, Any], output_dir: Path, view: str = 'xy'):
    """繪製感測點的 2D 分佈圖"""
    print(f"\n📊 繪製 2D 分佈圖 ({view} 平面)...")
    
    coords = sensor_data['coordinates']
    values = sensor_data.get('values', None)
    
    # 檢查座標維度
    ndim = coords.shape[1]
    
    # 如果是 2D 資料且試圖繪製 xz/yz 平面，跳過
    if ndim == 2 and view != 'xy':
        print(f"  ⚠️  跳過 {view} 平面（資料為 2D）")
        return
    
    # 確定繪圖軸
    if view == 'xy':
        x_idx, y_idx = 0, 1
        z_idx = 2 if ndim >= 3 else None
        x_label, y_label = 'x', 'y'
    elif view == 'xz':
        x_idx, y_idx = 0, 2
        z_idx = 1
        x_label, y_label = 'x', 'z'
    elif view == 'yz':
        x_idx, y_idx = 1, 2
        z_idx = 0
        x_label, y_label = 'y', 'z'
    else:
        raise ValueError(f"未知的視角: {view}")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 子圖 1: 按索引顏色編碼
    ax = axes[0]
    scatter = ax.scatter(
        coords[:, x_idx], 
        coords[:, y_idx],
        c=np.arange(len(coords)),
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    # 標註感測點編號
    for i, (x, y) in enumerate(zip(coords[:, x_idx], coords[:, y_idx])):
        ax.annotate(
            f'{i}',
            (x, y),
            fontsize=8,
            ha='center',
            va='center',
            color='white',
            weight='bold'
        )
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'Sensor Distribution ({view.upper()} plane) - Indexed', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Sensor Index', fontsize=10)
    
    # 子圖 2: 按數值大小顏色編碼
    ax = axes[1]
    
    if values is not None:
        # 使用速度大小
        if len(values.shape) == 2 and values.shape[1] >= 3:
            magnitude = np.linalg.norm(values[:, :3], axis=1)
        else:
            magnitude = np.abs(values.flatten())
        
        scatter = ax.scatter(
            coords[:, x_idx], 
            coords[:, y_idx],
            c=magnitude,
            cmap='jet',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=1.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Velocity Magnitude', fontsize=10)
        title_suffix = '- Velocity Magnitude'
    else:
        # 無數值資料
        if z_idx is not None:
            # 3D 資料：使用 z 座標顏色編碼
            scatter = ax.scatter(
                coords[:, x_idx], 
                coords[:, y_idx],
                c=coords[:, z_idx],
                cmap='coolwarm',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(f'{["x", "y", "z"][z_idx]} coordinate', fontsize=10)
            title_suffix = f'- {["x", "y", "z"][z_idx].upper()} Position'
        else:
            # 2D 資料：使用索引顏色編碼
            scatter = ax.scatter(
                coords[:, x_idx], 
                coords[:, y_idx],
                c=np.arange(len(coords)),
                cmap='viridis',
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Sensor Index', fontsize=10)
            title_suffix = '- Indexed'
    
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(f'Sensor Distribution ({view.upper()} plane) {title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    output_path = output_dir / f'sensor_distribution_2d_{view}.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output_path}")
    
    plt.close()


def plot_sensor_distribution_3d(sensor_data: Dict[str, Any], output_dir: Path):
    """繪製感測點的 3D 分佈圖"""
    print(f"\n📊 繪製 3D 分佈圖...")
    
    coords = sensor_data['coordinates']
    values = sensor_data.get('values', None)
    
    # 檢查座標維度
    if coords.shape[1] < 3:
        print(f"  ⚠️  跳過 3D 繪圖（資料為 {coords.shape[1]}D）")
        return
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 確定顏色編碼
    if values is not None and len(values.shape) == 2 and values.shape[1] >= 3:
        # 使用速度大小
        magnitude = np.linalg.norm(values[:, :3], axis=1)
        color_data = magnitude
        color_label = 'Velocity Magnitude'
    else:
        # 使用索引
        color_data = np.arange(len(coords))
        color_label = 'Sensor Index'
    
    # 繪製散點
    scatter = ax.scatter(
        coords[:, 0],  # x
        coords[:, 1],  # y
        coords[:, 2],  # z
        c=color_data,
        cmap='viridis',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=1.5
    )
    
    # 標註部分感測點編號（避免過於擁擠）
    n_annotate = min(20, len(coords))
    for i in range(n_annotate):
        ax.text(
            coords[i, 0],
            coords[i, 1],
            coords[i, 2],
            f'{i}',  # 3D text 的第四個位置參數
            fontsize=8,
            color='red',
            fontweight='bold'
        )
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)  # type: ignore
    ax.set_title(f'3D Sensor Distribution\n({len(coords)} sensors)', fontsize=14)
    
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(color_label, fontsize=10)
    
    # 設定視角
    ax.view_init(elev=20, azim=45)  # type: ignore
    
    plt.tight_layout()
    
    output_path = output_dir / 'sensor_distribution_3d.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output_path}")
    
    plt.close()


def plot_sensor_statistics(sensor_data: Dict[str, Any], output_dir: Path):
    """繪製感測點統計資訊"""
    print(f"\n📊 繪製統計資訊...")
    
    coords = sensor_data['coordinates']
    values = sensor_data.get('values', None)
    ndim = coords.shape[1]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子圖 1: 座標分佈直方圖
    ax = axes[0, 0]
    ax.hist(coords[:, 0], bins=20, alpha=0.7, label='x', color='red')
    ax.hist(coords[:, 1], bins=20, alpha=0.7, label='y', color='green')
    if ndim >= 3:
        ax.hist(coords[:, 2], bins=20, alpha=0.7, label='z', color='blue')
    ax.set_xlabel('Coordinate Value', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Coordinate Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子圖 2: 感測點間距離分佈
    ax = axes[0, 1]
    distances = []
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dist = np.linalg.norm(coords[i] - coords[j])
            distances.append(dist)
    
    distances = np.array(distances)
    ax.hist(distances, bins=30, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(np.mean(distances), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(distances):.3f}')
    ax.axvline(np.median(distances), color='green', linestyle='--', 
               linewidth=2, label=f'Median: {np.median(distances):.3f}')
    ax.set_xlabel('Distance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Pairwise Distance Distribution', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 子圖 3: 數值大小分佈（如果有）
    ax = axes[1, 0]
    if values is not None:
        if len(values.shape) == 2 and values.shape[1] >= 3:
            # 速度分量
            ax.hist(values[:, 0], bins=20, alpha=0.7, label='u', color='red')
            ax.hist(values[:, 1], bins=20, alpha=0.7, label='v', color='green')
            ax.hist(values[:, 2], bins=20, alpha=0.7, label='w', color='blue')
            ax.set_xlabel('Velocity Component', fontsize=12)
            ax.set_title('Velocity Component Distribution', fontsize=14)
        else:
            ax.hist(values.flatten(), bins=30, alpha=0.7, color='orange', edgecolor='black')
            ax.set_xlabel('Value', fontsize=12)
            ax.set_title('Value Distribution', fontsize=14)
        
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No value data available', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    
    # 子圖 4: 累積統計資訊文字
    ax = axes[1, 1]
    ax.axis('off')
    
    # 建立統計文字（處理 2D/3D 情況）
    stats_text = f"""
    感測點統計資訊
    {'='*40}
    
    總感測點數量: {len(coords)}
    
    座標範圍:
      x: [{coords[:, 0].min():.3f}, {coords[:, 0].max():.3f}]
      y: [{coords[:, 1].min():.3f}, {coords[:, 1].max():.3f}]"""
    
    if ndim >= 3:
        stats_text += f"""
      z: [{coords[:, 2].min():.3f}, {coords[:, 2].max():.3f}]"""
    
    stats_text += f"""
    
    感測點間距離:
      最小距離: {distances.min():.4f}
      最大距離: {distances.max():.4f}
      平均距離: {np.mean(distances):.4f}
      標準差: {np.std(distances):.4f}
    """
    
    if values is not None and len(values.shape) == 2 and values.shape[1] >= 3:
        magnitude = np.linalg.norm(values[:, :3], axis=1)
        stats_text += f"""
    速度場統計:
      |U| 範圍: [{magnitude.min():.3f}, {magnitude.max():.3f}]
      |U| 平均: {magnitude.mean():.3f}
      u 平均: {values[:, 0].mean():.3f}
      v 平均: {values[:, 1].mean():.3f}
      w 平均: {values[:, 2].mean():.3f}
        """
    
    if 'metrics' in sensor_data:
        metrics = sensor_data['metrics']
        stats_text += f"""
    QR-Pivot 指標:
      條件數: {metrics.get('condition_number', 'N/A'):.2f}
      能量比例: {metrics.get('energy_ratio', 0.0):.3f}
      子空間覆蓋率: {metrics.get('subspace_coverage', 0.0):.3f}
        """
    
    ax.text(0.05, 0.95, stats_text, 
            fontsize=10, 
            verticalalignment='top',
            family='monospace',
            transform=ax.transAxes)
    
    plt.tight_layout()
    
    output_path = output_dir / 'sensor_statistics.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"  ✅ 已保存: {output_path}")
    
    plt.close()


def save_sensor_table(sensor_data: Dict[str, Any], output_dir: Path):
    """保存感測點座標與數值表格"""
    print(f"\n💾 保存感測點資料表格...")
    
    coords = sensor_data['coordinates']
    values = sensor_data.get('values', None)
    indices = sensor_data.get('indices', np.arange(len(coords)))
    ndim = coords.shape[1]
    
    # 建立表格
    output_path = output_dir / 'sensor_table.txt'
    
    with open(str(output_path), 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("QR-Pivot 感測點座標與數值表格\n")
        f.write("=" * 100 + "\n\n")
        
        if values is not None and len(values.shape) == 2 and values.shape[1] >= 3:
            # 速度場資料
            if ndim >= 3:
                f.write(f"{'Index':>6} {'Global_ID':>10} {'x':>12} {'y':>12} {'z':>12} "
                        f"{'u':>12} {'v':>12} {'w':>12} {'|U|':>12}\n")
                f.write("-" * 100 + "\n")
                
                for i, (idx, coord, vel) in enumerate(zip(indices, coords, values)):
                    magnitude = np.linalg.norm(vel[:3])
                    f.write(f"{i:6d} {idx:10d} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f} "
                            f"{vel[0]:12.6f} {vel[1]:12.6f} {vel[2]:12.6f} {magnitude:12.6f}\n")
            else:
                # 2D 資料
                f.write(f"{'Index':>6} {'Global_ID':>10} {'x':>12} {'y':>12} "
                        f"{'u':>12} {'v':>12} {'|U|':>12}\n")
                f.write("-" * 80 + "\n")
                
                for i, (idx, coord, vel) in enumerate(zip(indices, coords, values)):
                    magnitude = np.linalg.norm(vel[:2])
                    f.write(f"{i:6d} {idx:10d} {coord[0]:12.6f} {coord[1]:12.6f} "
                            f"{vel[0]:12.6f} {vel[1]:12.6f} {magnitude:12.6f}\n")
        else:
            # 僅座標
            if ndim >= 3:
                f.write(f"{'Index':>6} {'Global_ID':>10} {'x':>12} {'y':>12} {'z':>12}\n")
                f.write("-" * 70 + "\n")
                
                for i, (idx, coord) in enumerate(zip(indices, coords)):
                    f.write(f"{i:6d} {idx:10d} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
            else:
                # 2D 資料
                f.write(f"{'Index':>6} {'Global_ID':>10} {'x':>12} {'y':>12}\n")
                f.write("-" * 50 + "\n")
                
                for i, (idx, coord) in enumerate(zip(indices, coords)):
                    f.write(f"{i:6d} {idx:10d} {coord[0]:12.6f} {coord[1]:12.6f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        
        # 統計摘要
        f.write("\n統計摘要:\n")
        f.write(f"  總感測點數量: {len(coords)}\n")
        f.write(f"  x 範圍: [{coords[:, 0].min():.6f}, {coords[:, 0].max():.6f}]\n")
        f.write(f"  y 範圍: [{coords[:, 1].min():.6f}, {coords[:, 1].max():.6f}]\n")
        if ndim >= 3:
            f.write(f"  z 範圍: [{coords[:, 2].min():.6f}, {coords[:, 2].max():.6f}]\n")
        
        if values is not None and len(values.shape) == 2 and values.shape[1] >= 3:
            magnitude = np.linalg.norm(values[:, :3], axis=1)
            f.write(f"\n  速度大小範圍: [{magnitude.min():.6f}, {magnitude.max():.6f}]\n")
            f.write(f"  平均速度大小: {magnitude.mean():.6f}\n")
    
    print(f"  ✅ 已保存: {output_path}")
    
    # 同時保存為 JSON
    json_path = output_dir / 'sensor_data.json'
    
    json_data: Dict[str, Any] = {
        'n_sensors': int(len(coords)),
        'indices': indices.tolist(),
        'coordinates': coords.tolist(),
    }
    
    if values is not None:
        json_data['values'] = values.tolist()
    
    # 保存分層資訊（如果存在）
    if 'layer_info' in sensor_data:
        layer_info = sensor_data['layer_info']
        json_data['layer_info'] = {
            layer: {
                'n_selected': int(info['n_selected']),
                'y_range': [float(info['y_range'][0]), float(info['y_range'][1])],
                'indices': info['indices'].tolist()
            }
            for layer, info in layer_info.items()
        }
    
    if 'layer_labels' in sensor_data:
        json_data['layer_labels'] = sensor_data['layer_labels'].tolist()
    
    if 'metadata' in sensor_data:
        metadata = sensor_data['metadata']
        json_data['metadata'] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v)
            for k, v in metadata.items()
        }
    
    if 'metrics' in sensor_data:
        json_data['metrics'] = {
            k: float(v) if isinstance(v, (int, float, np.number)) else v
            for k, v in sensor_data['metrics'].items()
        }
    
    with open(str(json_path), 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"  ✅ 已保存: {json_path}")


def compare_strategies(jhtdb_path: str, n_sensors: int, output_dir: Path):
    """比較不同選點策略"""
    print(f"\n🔍 比較不同選點策略...")
    
    strategies = {
        'QR-Pivot': 'qr_pivot',
        'POD-based': 'pod_based',
        'Greedy': 'greedy'
    }
    
    results: Dict[str, Dict[str, Any]] = {}
    
    for name, strategy in strategies.items():
        print(f"\n  計算 {name} 策略...")
        try:
            sensor_data = compute_sensors_from_jhtdb(jhtdb_path, n_sensors, strategy)
            results[name] = sensor_data
        except Exception as e:
            print(f"    ❌ 失敗: {e}")
    
    # 繪製比較圖
    if len(results) > 0:
        print(f"\n  繪製策略比較圖...")
        
        fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 5))
        
        if len(results) == 1:
            axes = [axes]
        
        for ax, (name, data) in zip(axes, results.items()):
            coords = data['coordinates']
            
            # 2D 投影 (xy 平面)
            scatter = ax.scatter(
                coords[:, 0],
                coords[:, 1],
                c=np.arange(len(coords)),
                cmap='viridis',
                s=80,
                alpha=0.7,
                edgecolors='black',
                linewidth=1.5
            )
            
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title(f'{name}\n({len(coords)} sensors)', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
            # 顯示指標
            metrics = data.get('metrics', {})
            cond = metrics.get('condition_number', np.inf)
            energy = metrics.get('energy_ratio', 0.0)
            
            ax.text(0.05, 0.95, 
                    f'Cond: {cond:.1f}\nEnergy: {energy:.2f}',
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        output_path = output_dir / 'strategy_comparison.png'
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"  ✅ 已保存: {output_path}")
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='QR-Pivot 感測點分佈視覺化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # 輸入選項（二選一）
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input', type=str,
                            help='感測點資料檔案路徑 (.npz, .h5)')
    input_group.add_argument('--jhtdb-data', type=str,
                            help='JHTDB 資料檔案路徑（重新計算感測點）')
    
    # 參數選項
    parser.add_argument('--n-sensors', type=int, default=50,
                        help='感測點數量（使用 --jhtdb-data 時）')
    parser.add_argument('--strategy', type=str, default='qr_pivot',
                        choices=['qr_pivot', 'pod_based', 'greedy'],
                        help='感測點選擇策略')
    parser.add_argument('--compare-strategies', action='store_true',
                        help='比較多種策略')
    
    # 多時間步資料支援（新增）⭐
    parser.add_argument('--temporal-data', type=str, default=None,
                        help='多時間步快照資料檔案路徑（.npz 格式，優先於 --jhtdb-data）')
    
    # 輸出選項
    parser.add_argument('--output', type=str, default='results/sensor_analysis',
                        help='輸出目錄')
    parser.add_argument('--views', type=str, nargs='+', 
                        default=['xy', 'xz', 'yz'],
                        help='2D 視角列表')
    
    args = parser.parse_args()
    
    # 建立輸出目錄
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  🎯 QR-Pivot 感測點分佈視覺化工具")
    print("=" * 80)
    
    # 載入或計算感測點資料
    if args.input:
        sensor_data = load_sensor_data(args.input)
    else:
        sensor_data = compute_sensors_from_jhtdb(
            args.jhtdb_data, 
            args.n_sensors, 
            args.strategy,
            temporal_data_path=args.temporal_data  # ⭐ 傳遞多時間步資料
        )
    
    # 視覺化
    print(f"\n{'='*80}")
    print("  📊 生成視覺化圖表")
    print(f"{'='*80}")
    
    # 2D 分佈圖
    for view in args.views:
        plot_sensor_distribution_2d(sensor_data, output_dir, view)
    
    # 3D 分佈圖
    plot_sensor_distribution_3d(sensor_data, output_dir)
    
    # 統計資訊
    plot_sensor_statistics(sensor_data, output_dir)
    
    # 保存表格
    save_sensor_table(sensor_data, output_dir)
    
    # 策略比較（如果啟用）
    if args.compare_strategies and args.jhtdb_data:
        compare_strategies(args.jhtdb_data, args.n_sensors, output_dir)
    
    print(f"\n{'='*80}")
    print("  ✅ 完成")
    print(f"{'='*80}")
    print(f"\n結果已保存至: {output_dir.absolute()}")
    print(f"\n包含檔案:")
    for file in sorted(output_dir.glob('*')):
        print(f"  - {file.name}")


if __name__ == '__main__':
    main()
