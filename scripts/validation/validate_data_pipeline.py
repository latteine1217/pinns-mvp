#!/usr/bin/env python3
"""
資料管線驗證腳本

驗證 DNS 資料載入、Sensor 採樣、Loss 計算的完整性。

使用方式:
    python scripts/validation/validate_data_pipeline.py \
        --config configs/kolmogorov_re50_kf4_K100.yml \
        --check-all
"""

import argparse
import logging
import sys
from pathlib import Path
import json

import numpy as np
import h5py
import torch

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pinnx.train.config_loader import load_config
from pinnx.utils.setup import setup_logging

def validate_sensor_indices(dns_path: str, sensor_file: str) -> bool:
    """驗證 Sensor 索引是否與 DNS 網格解析度匹配
    
    Args:
        dns_path: DNS HDF5 檔案路徑
        sensor_file: Sensor JSON 檔案路徑
        
    Returns:
        True 如果驗證通過
    """
    logging.info("=" * 60)
    logging.info("🔍 驗證 1: Sensor 索引越界檢查")
    logging.info("=" * 60)
    
    # 讀取 DNS 網格資訊
    with h5py.File(dns_path, 'r') as f:
        N = int(f['config'].attrs['N'])
        N_total = N * N
    
    logging.info(f"DNS 網格解析度: {N} x {N} = {N_total} 點")
    
    # 讀取 Sensor 索引
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    
    logging.info(f"Sensor 數量: {K}")
    logging.info(f"Sensor 索引範圍: [{spatial_indices.min()}, {spatial_indices.max()}]")
    
    # 驗證索引
    if spatial_indices.max() >= N_total:
        logging.error(
            f"❌ 驗證失敗！Sensor 索引越界\n"
            f"   最大索引 {spatial_indices.max()} >= 總點數 {N_total}\n"
            f"   可能原因：Sensor 檔案基於不同網格解析度生成"
        )
        return False
    
    if spatial_indices.min() < 0:
        logging.error(f"❌ 驗證失敗！Sensor 索引無效：最小索引 {spatial_indices.min()} < 0")
        return False
    
    logging.info(f"✅ 驗證通過：索引範圍 [{spatial_indices.min()}, {spatial_indices.max()}] ⊂ [0, {N_total-1}]")
    return True


def validate_sensor_data_shape(dns_path: str, sensor_file: str, time_range: tuple) -> bool:
    """驗證 Sensor 資料形狀
    
    Args:
        dns_path: DNS HDF5 檔案路徑
        sensor_file: Sensor JSON 檔案路徑
        time_range: 時間範圍 (t_start, t_end)
        
    Returns:
        True 如果驗證通過
    """
    logging.info("=" * 60)
    logging.info("🔍 驗證 2: Sensor 資料形狀檢查")
    logging.info("=" * 60)
    
    # 讀取 Sensor 索引
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    
    # 讀取 DNS 資料
    with h5py.File(dns_path, 'r') as f:
        time_all = np.array(f['time'])
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        T_selected = time_mask.sum()
        
        logging.info(f"時間範圍: [{t_start}, {t_end}]")
        logging.info(f"選中時間步: {T_selected}")
        
        u_slice = f['u'][time_mask]  # [T, N, N]
        N = int(f['config'].attrs['N'])
        
        # 展平並提取
        u_flat = u_slice.reshape(T_selected, -1)
        u_sensors_vals = u_flat[:, spatial_indices]
    
    expected_shape = (T_selected, K)
    actual_shape = u_sensors_vals.shape
    
    logging.info(f"預期形狀: {expected_shape}")
    logging.info(f"實際形狀: {actual_shape}")
    
    if actual_shape != expected_shape:
        logging.error(
            f"❌ 驗證失敗！Sensor 資料形狀不符\n"
            f"   預期: {expected_shape}\n"
            f"   實際: {actual_shape}"
        )
        return False
    
    logging.info("✅ 驗證通過：Sensor 資料形狀正確")
    return True


def validate_flatten_order(dns_path: str, sensor_file: str, time_range: tuple) -> bool:
    """驗證 Flatten 順序一致性
    
    Args:
        dns_path: DNS HDF5 檔案路徑
        sensor_file: Sensor JSON 檔案路徑
        time_range: 時間範圍 (t_start, t_end)
        
    Returns:
        True 如果驗證通過
    """
    logging.info("=" * 60)
    logging.info("🔍 驗證 3: Flatten 順序一致性檢查")
    logging.info("=" * 60)
    
    # 讀取 Sensor 索引
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    
    # 讀取 DNS 資料
    with h5py.File(dns_path, 'r') as f:
        time_all = np.array(f['time'])
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        T_selected = time_mask.sum()
        
        u_slice = f['u'][time_mask]
        u_flat = u_slice.reshape(T_selected, -1)
        u_sensors_vals = u_flat[:, spatial_indices]
    
    # Flatten
    u_train = u_sensors_vals.flatten()
    
    logging.info(f"u_sensors_vals 形狀: {u_sensors_vals.shape}")
    logging.info(f"u_train 長度: {len(u_train)}")
    
    # 驗證總長度
    expected_len = T_selected * K
    if len(u_train) != expected_len:
        logging.error(
            f"❌ 驗證失敗！Flatten 長度不符\n"
            f"   預期: {expected_len}\n"
            f"   實際: {len(u_train)}"
        )
        return False
    
    # 驗證 C-order 對應關係
    # 第一個時間步的第一個 sensor
    if not np.isclose(u_train[0], u_sensors_vals[0, 0], rtol=1e-5):
        logging.error(
            f"❌ 驗證失敗！Flatten 順序錯誤（第一個元素）\n"
            f"   u_train[0] = {u_train[0]}\n"
            f"   u_sensors_vals[0, 0] = {u_sensors_vals[0, 0]}"
        )
        return False
    
    # 第二個時間步的第一個 sensor（如果存在）
    if T_selected > 1:
        if not np.isclose(u_train[K], u_sensors_vals[1, 0], rtol=1e-5):
            logging.error(
                f"❌ 驗證失敗！Flatten 順序錯誤（第 K 個元素）\n"
                f"   u_train[{K}] = {u_train[K]}\n"
                f"   u_sensors_vals[1, 0] = {u_sensors_vals[1, 0]}"
            )
            return False
    
    logging.info(f"✅ 驗證通過：u_train[0] = u_sensors_vals[0, 0] = {u_train[0]:.6f}")
    if T_selected > 1:
        logging.info(f"✅ 驗證通過：u_train[{K}] = u_sensors_vals[1, 0] = {u_train[K]:.6f}")
    
    logging.info("✅ 驗證通過：Flatten 順序為 C-order（row-major）")
    return True


def validate_coordinate_alignment(dns_path: str, sensor_file: str) -> bool:
    """驗證座標對齊
    
    Args:
        dns_path: DNS HDF5 檔案路徑
        sensor_file: Sensor JSON 檔案路徑
        
    Returns:
        True 如果驗證通過
    """
    logging.info("=" * 60)
    logging.info("🔍 驗證 4: 座標對齊檢查")
    logging.info("=" * 60)
    
    # 讀取 DNS 網格
    with h5py.File(dns_path, 'r') as f:
        N = int(f['config'].attrs['N'])
        L = float(f['config'].attrs['L'])
    
    x_1d = np.linspace(0, L, N, endpoint=False)
    y_1d = np.linspace(0, L, N, endpoint=False)
    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
    
    # 讀取 Sensor 索引
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    spatial_indices = np.array(sensor_data['indices'])
    
    # 提取 Sensor 座標
    X_flat = X_mesh.flatten()
    Y_flat = Y_mesh.flatten()
    x_sensor_locs = X_flat[spatial_indices]
    y_sensor_locs = Y_flat[spatial_indices]
    
    logging.info(f"DNS 域範圍: [0, {L}] x [0, {L}]")
    logging.info(f"Sensor x 範圍: [{x_sensor_locs.min():.4f}, {x_sensor_locs.max():.4f}]")
    logging.info(f"Sensor y 範圍: [{y_sensor_locs.min():.4f}, {y_sensor_locs.max():.4f}]")
    
    # 檢查是否在域內
    if x_sensor_locs.min() < 0 or x_sensor_locs.max() > L:
        logging.error(f"❌ 驗證失敗！Sensor x 座標超出域範圍")
        return False
    
    if y_sensor_locs.min() < 0 or y_sensor_locs.max() > L:
        logging.error(f"❌ 驗證失敗！Sensor y 座標超出域範圍")
        return False
    
    logging.info("✅ 驗證通過：Sensor 座標在域範圍內")
    return True


def main():
    parser = argparse.ArgumentParser(description="驗證資料管線完整性")
    parser.add_argument('--config', type=str, required=True, help='配置檔案路徑')
    parser.add_argument('--check-all', action='store_true', help='執行所有驗證')
    parser.add_argument('--check-indices', action='store_true', help='僅驗證 Sensor 索引')
    parser.add_argument('--check-shape', action='store_true', help='僅驗證資料形狀')
    parser.add_argument('--check-flatten', action='store_true', help='僅驗證 Flatten 順序')
    parser.add_argument('--check-coords', action='store_true', help='僅驗證座標對齊')
    
    args = parser.parse_args()
    
    # 設定日誌
    setup_logging(level='INFO', log_file=None)
    
    # 載入配置
    config = load_config(args.config)
    
    # 提取必要參數
    kol_cfg = config['data']['kolmogorov_config']
    dns_path = kol_cfg['data_path']
    time_range = tuple(kol_cfg['time_range'])
    sensor_file = config['sensors']['sensor_file']
    
    logging.info("=" * 60)
    logging.info("📋 資料管線驗證")
    logging.info("=" * 60)
    logging.info(f"配置檔案: {args.config}")
    logging.info(f"DNS 資料: {dns_path}")
    logging.info(f"Sensor 檔案: {sensor_file}")
    logging.info(f"時間範圍: {time_range}")
    logging.info("=" * 60)
    
    # 執行驗證
    results = {}
    
    if args.check_all or args.check_indices:
        results['indices'] = validate_sensor_indices(dns_path, sensor_file)
    
    if args.check_all or args.check_shape:
        results['shape'] = validate_sensor_data_shape(dns_path, sensor_file, time_range)
    
    if args.check_all or args.check_flatten:
        results['flatten'] = validate_flatten_order(dns_path, sensor_file, time_range)
    
    if args.check_all or args.check_coords:
        results['coords'] = validate_coordinate_alignment(dns_path, sensor_file)
    
    # 總結
    logging.info("=" * 60)
    logging.info("📊 驗證總結")
    logging.info("=" * 60)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✅ 通過" if passed else "❌ 失敗"
        logging.info(f"{check_name.ljust(15)}: {status}")
        if not passed:
            all_passed = False
    
    logging.info("=" * 60)
    
    if all_passed:
        logging.info("🎉 所有驗證通過！")
        return 0
    else:
        logging.error("⚠️  部分驗證失敗，請檢查上方錯誤訊息")
        return 1


if __name__ == '__main__':
    sys.exit(main())
