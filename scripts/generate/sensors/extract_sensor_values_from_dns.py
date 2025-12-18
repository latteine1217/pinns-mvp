#!/usr/bin/env python3
"""
從 DNS cutout 提取 sensor 座標對應的 u/v/w/p 值

工程場景：使用 K=100 稀疏測量點的統計來計算標準化（而非完整 DNS 場）

Usage:
    python extract_sensor_values_from_dns.py \
        --sensor-file data/jhtdb/channel_flow_re1000/sensors_K100_qr_pivot_3d_v5_gradu_eig_large.npz \
        --dns-cutout data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz \
        --output data/jhtdb/channel_flow_re1000/sensors_K100_qr_pivot_3d_v5_WITH_VALUES.npz
"""

import argparse
import numpy as np
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator


def find_nearest_grid_indices(coords, grid_coords):
    """
    找到每個座標在網格中最接近的索引
    
    Args:
        coords: [N,] 座標陣列
        grid_coords: [Nx,] 或 [Ny,] 或 [Nz,] 網格座標
    
    Returns:
        indices: [N,] 最接近的網格索引
    """
    indices = np.searchsorted(grid_coords, coords)
    # 處理邊界情況
    indices = np.clip(indices, 0, len(grid_coords) - 1)
    
    # 檢查是否前一個索引更接近
    for i, (coord, idx) in enumerate(zip(coords, indices)):
        if idx > 0:
            dist_curr = abs(grid_coords[idx] - coord)
            dist_prev = abs(grid_coords[idx - 1] - coord)
            if dist_prev < dist_curr:
                indices[i] = idx - 1
    
    return indices


def extract_values_at_sensors(
    sensor_coords: dict,
    dns_cutout_path: str,
    method: str = 'nearest'
) -> dict:
    """
    從 DNS cutout 提取 sensor 位置的值
    
    Args:
        sensor_coords: {'x': [K,], 'y': [K,], 'z': [K,]}
        dns_cutout_path: DNS cutout 文件路徑
        method: 'nearest' (最近鄰) or 'linear' (線性插值)
    
    Returns:
        values: {'u': [K,], 'v': [K,], 'w': [K,], 'p': [K,]}
    """
    print(f"\n{'='*80}")
    print(f"📂 Loading DNS cutout: {dns_cutout_path}")
    dns_data = np.load(dns_cutout_path)
    
    # 檢查必要欄位
    required_fields = ['x', 'y', 'z', 'u', 'v', 'w', 'p']
    for field in required_fields:
        if field not in dns_data:
            raise KeyError(f"DNS cutout missing required field: {field}")
    
    # 取得網格座標
    x_grid = dns_data['x']
    y_grid = dns_data['y']
    z_grid = dns_data['z']
    
    print(f"   DNS grid shape: x={len(x_grid)}, y={len(y_grid)}, z={len(z_grid)}")
    print(f"   DNS domain: x=[{x_grid.min():.3f}, {x_grid.max():.3f}]")
    print(f"              y=[{y_grid.min():.3f}, {y_grid.max():.3f}]")
    print(f"              z=[{z_grid.min():.3f}, {z_grid.max():.3f}]")
    
    # 取得場資料（可能是 flattened）
    u_field = dns_data['u']
    v_field = dns_data['v']
    w_field = dns_data['w']
    p_field = dns_data['p']
    
    print(f"   Field shapes: {u_field.shape}")
    
    # 如果是 flattened，需要 reshape
    if 'grid_shape' in dns_data:
        grid_shape = tuple(dns_data['grid_shape'])
        print(f"   Grid shape metadata: {grid_shape}")
        
        if len(u_field.shape) == 1:
            print(f"   Reshaping fields from 1D to 3D...")
            u_field = u_field.reshape(grid_shape)
            v_field = v_field.reshape(grid_shape)
            w_field = w_field.reshape(grid_shape)
            p_field = p_field.reshape(grid_shape)
            print(f"   New field shape: {u_field.shape}")
    
    # 檢查 sensor 座標是否在域內
    K = len(sensor_coords['x'])
    sensor_x = sensor_coords['x']
    sensor_y = sensor_coords['y']
    sensor_z = sensor_coords['z']
    
    print(f"\n🎯 Extracting values at K={K} sensor locations...")
    print(f"   Sensor domain: x=[{sensor_x.min():.3f}, {sensor_x.max():.3f}]")
    print(f"                 y=[{sensor_y.min():.3f}, {sensor_y.max():.3f}]")
    print(f"                 z=[{sensor_z.min():.3f}, {sensor_z.max():.3f}]")
    
    # 檢查是否有超出範圍的點
    x_out = np.sum((sensor_x < x_grid.min()) | (sensor_x > x_grid.max()))
    y_out = np.sum((sensor_y < y_grid.min()) | (sensor_y > y_grid.max()))
    z_out = np.sum((sensor_z < z_grid.min()) | (sensor_z > z_grid.max()))
    
    if x_out > 0 or y_out > 0 or z_out > 0:
        print(f"   ⚠️  Warning: {x_out + y_out + z_out} sensors outside DNS domain!")
        print(f"      x_out={x_out}, y_out={y_out}, z_out={z_out}")
    
    # 根據方法提取值
    if method == 'nearest':
        print(f"   Method: Nearest neighbor")
        
        # 找到最接近的網格索引
        ix = find_nearest_grid_indices(sensor_x, x_grid)
        iy = find_nearest_grid_indices(sensor_y, y_grid)
        iz = find_nearest_grid_indices(sensor_z, z_grid)
        
        # 提取值（假設 field shape 是 [Nx, Ny, Nz]）
        u_sensors = u_field[ix, iy, iz]
        v_sensors = v_field[ix, iy, iz]
        w_sensors = w_field[ix, iy, iz]
        p_sensors = p_field[ix, iy, iz]
        
    elif method == 'linear':
        print(f"   Method: Linear interpolation")
        
        # 使用 scipy 線性插值
        u_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), u_field, 
                                           method='linear', bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), v_field,
                                           method='linear', bounds_error=False, fill_value=None)
        w_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), w_field,
                                           method='linear', bounds_error=False, fill_value=None)
        p_interp = RegularGridInterpolator((x_grid, y_grid, z_grid), p_field,
                                           method='linear', bounds_error=False, fill_value=None)
        
        sensor_points = np.stack([sensor_x, sensor_y, sensor_z], axis=1)  # [K, 3]
        
        u_sensors = u_interp(sensor_points)
        v_sensors = v_interp(sensor_points)
        w_sensors = w_interp(sensor_points)
        p_sensors = p_interp(sensor_points)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'nearest' or 'linear'")
    
    # 統計資訊
    print(f"\n📊 Extracted Sensor Statistics (K={K}):")
    print(f"   u: mean={u_sensors.mean():.6f}, std={u_sensors.std():.6f}, range=[{u_sensors.min():.3f}, {u_sensors.max():.3f}]")
    print(f"   v: mean={v_sensors.mean():.6f}, std={v_sensors.std():.6f}, range=[{v_sensors.min():.3f}, {v_sensors.max():.3f}]")
    print(f"   w: mean={w_sensors.mean():.6f}, std={w_sensors.std():.6f}, range=[{w_sensors.min():.3f}, {w_sensors.max():.3f}]")
    print(f"   p: mean={p_sensors.mean():.6f}, std={p_sensors.std():.6f}, range=[{p_sensors.min():.3f}, {p_sensors.max():.3f}]")
    
    # 驗證統計合理性
    print(f"\n✅ Quality Check:")
    checks_passed = True
    
    if v_sensors.std() < 1e-3:
        print(f"   ❌ v_std={v_sensors.std():.2e} < 1e-3 (suspicious!)")
        checks_passed = False
    else:
        print(f"   ✅ v_std={v_sensors.std():.6f} >= 1e-3")
    
    if w_sensors.std() < 1e-3:
        print(f"   ❌ w_std={w_sensors.std():.2e} < 1e-3 (suspicious!)")
        checks_passed = False
    else:
        print(f"   ✅ w_std={w_sensors.std():.6f} >= 1e-3")
    
    if p_sensors.std() < 1e-4:
        print(f"   ❌ p_std={p_sensors.std():.2e} < 1e-4 (suspicious!)")
        checks_passed = False
    else:
        print(f"   ✅ p_std={p_sensors.std():.6f} >= 1e-4")
    
    if not checks_passed:
        raise ValueError("Extracted sensor values have suspicious statistics!")
    
    print(f"{'='*80}\n")
    
    return {
        'u': u_sensors,
        'v': v_sensors,
        'w': w_sensors,
        'p': p_sensors
    }


def main():
    parser = argparse.ArgumentParser(description='Extract sensor values from DNS cutout')
    parser.add_argument('--sensor-file', type=str, required=True,
                        help='Sensor coordinate file (e.g., sensors_K100_*.npz)')
    parser.add_argument('--dns-cutout', type=str, required=True,
                        help='DNS cutout file (e.g., cutout_128x64x128.npz)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file with sensor values')
    parser.add_argument('--method', type=str, default='nearest', choices=['nearest', 'linear'],
                        help='Interpolation method (default: nearest)')
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    sensor_path = Path(args.sensor_file)
    dns_path = Path(args.dns_cutout)
    output_path = Path(args.output)
    
    if not sensor_path.exists():
        raise FileNotFoundError(f"Sensor file not found: {sensor_path}")
    
    if not dns_path.exists():
        raise FileNotFoundError(f"DNS cutout not found: {dns_path}")
    
    # 載入 sensor 座標
    print(f"{'='*80}")
    print(f"🎯 Extracting Sensor Values from DNS Cutout")
    print(f"{'='*80}")
    print(f"\n📂 Loading sensor coordinates: {sensor_path}")
    
    sensor_data = np.load(sensor_path)
    
    print(f"   Available keys: {list(sensor_data.keys())}")
    
    # 提取座標
    if 'sensor_x' not in sensor_data or 'sensor_y' not in sensor_data:
        raise KeyError("Sensor file must contain 'sensor_x' and 'sensor_y'")
    
    sensor_coords = {
        'x': sensor_data['sensor_x'],
        'y': sensor_data['sensor_y'],
        'z': sensor_data['sensor_z'] if 'sensor_z' in sensor_data else np.zeros_like(sensor_data['sensor_x'])
    }
    
    K = len(sensor_coords['x'])
    print(f"   K = {K} sensors")
    
    # 從 DNS 提取值
    sensor_values = extract_values_at_sensors(
        sensor_coords,
        str(dns_path),
        method=args.method
    )
    
    # 合併所有資料
    print(f"💾 Saving to: {output_path}")
    
    # 複製原始 sensor 文件的所有 metadata
    output_data = {key: sensor_data[key] for key in sensor_data.keys()}
    
    # 添加提取的值
    output_data['u_sensors'] = sensor_values['u']
    output_data['v_sensors'] = sensor_values['v']
    output_data['w_sensors'] = sensor_values['w']
    output_data['p_sensors'] = sensor_values['p']
    
    # 添加提取方法記錄
    output_data['extraction_method'] = args.method
    output_data['dns_source'] = str(dns_path)
    
    # 確保輸出目錄存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 儲存
    np.savez_compressed(output_path, **output_data)
    
    print(f"✅ Successfully saved sensor file with values!")
    print(f"   New keys added: u_sensors, v_sensors, w_sensors, p_sensors")
    print(f"   Total keys: {len(output_data)}")
    print(f"\n{'='*80}")
    print(f"🎉 DONE! Use this file for training:")
    print(f"   {output_path}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
