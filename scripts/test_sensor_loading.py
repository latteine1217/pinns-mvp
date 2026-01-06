#!/usr/bin/env python3
"""
簡單測試：驗證 sensor data 是否完整載入訓練流程
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
import h5py
import json
import numpy as np

def test_sensor_loading(config_path: str):
    """測試 sensor data 載入流程"""
    
    print("=" * 80)
    print("🧪 測試：Sensor Data 載入驗證")
    print("=" * 80)
    
    # 1. 讀取配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    sensor_file = config['sensors']['sensor_file']
    K = config['sensors']['K']
    dns_file = config['data']['kolmogorov_config']['data_path']
    time_range = config['data']['kolmogorov_config']['time_range']
    
    print(f"\n📋 配置資訊:")
    print(f"  Config: {Path(config_path).name}")
    print(f"  Sensor File: {sensor_file}")
    print(f"  K (感測點數): {K}")
    print(f"  Time Range: {time_range}")
    print(f"  DNS File: {dns_file}")
    
    # 2. 載入 sensor JSON
    print(f"\n📂 步驟 1: 載入 Sensor JSON")
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
    
    indices = np.array(sensor_data['indices'])
    print(f"  ✓ Sensor 索引數量: {len(indices)}")
    print(f"  ✓ 索引範圍: [{indices.min()}, {indices.max()}]")
    
    # 檢查是否是 temporal sensor
    is_temporal = 'time_range' in sensor_data
    if is_temporal:
        print(f"  ✓ Temporal Sensor 檢測:")
        print(f"    - Time Range: {sensor_data.get('time_range')}")
        print(f"    - Time Steps: {sensor_data.get('num_time_steps')}")
        print(f"    - Features: {sensor_data.get('num_features')}")
        print(f"    - Condition Number: {sensor_data.get('condition_number', 'N/A'):.2e}")
    
    # 3. 載入 DNS 資料
    print(f"\n📂 步驟 2: 載入 DNS 資料")
    with h5py.File(dns_file, 'r') as f:
        u_data = f['u'][:]
        v_data = f['v'][:]
        p_data = f['p'][:]
        t_data = f['time'][:]  # 修正：時間變數名稱是 'time'
    
    print(f"  ✓ DNS 資料形狀:")
    print(f"    - u: {u_data.shape}")
    print(f"    - v: {v_data.shape}")
    print(f"    - p: {p_data.shape}")
    print(f"    - t: {t_data.shape}")
    
    # 4. 模擬訓練時的資料提取 (來自 kolmogorov.py 的邏輯)
    print(f"\n📂 步驟 3: 模擬訓練時的 Sensor 提取")
    
    # 篩選時間範圍
    t_min, t_max = time_range
    time_mask = (t_data >= t_min) & (t_data <= t_max)
    t_selected = t_data[time_mask]
    u_selected = u_data[time_mask]
    v_selected = v_data[time_mask]
    p_selected = p_data[time_mask]
    
    print(f"  ✓ 時間篩選:")
    print(f"    - 原始 timesteps: {len(t_data)}")
    print(f"    - 選中 timesteps: {len(t_selected)} (range: [{t_selected[0]:.1f}, {t_selected[-1]:.1f}])")
    print(f"    - 時間間隔 dt: {t_selected[1] - t_selected[0]:.4f}")
    
    # Flatten 空間維度
    T, Nx, Ny = u_selected.shape
    u_flat = u_selected.reshape(T, -1)
    v_flat = v_selected.reshape(T, -1)
    p_flat = p_selected.reshape(T, -1)
    
    print(f"  ✓ Flatten 後形狀: {u_flat.shape} (T × N_spatial)")
    
    # 提取 sensor 位置的資料
    u_sensors = u_flat[:, indices]
    v_sensors = v_flat[:, indices]
    p_sensors = p_flat[:, indices]
    
    print(f"  ✓ Sensor 提取形狀:")
    print(f"    - u_sensors: {u_sensors.shape} (T × K)")
    print(f"    - v_sensors: {v_sensors.shape}")
    print(f"    - p_sensors: {p_sensors.shape}")
    
    # 5. 檢查資料完整性
    print(f"\n✅ 步驟 4: 資料完整性檢查")
    
    # 檢查 NaN/Inf
    has_nan = np.any(np.isnan(u_sensors)) or np.any(np.isnan(v_sensors)) or np.any(np.isnan(p_sensors))
    has_inf = np.any(np.isinf(u_sensors)) or np.any(np.isinf(v_sensors)) or np.any(np.isinf(p_sensors))
    
    if has_nan:
        print(f"  ⚠️ 警告: 資料包含 NaN!")
    else:
        print(f"  ✓ 無 NaN")
    
    if has_inf:
        print(f"  ⚠️ 警告: 資料包含 Inf!")
    else:
        print(f"  ✓ 無 Inf")
    
    # 檢查數值範圍
    print(f"  ✓ 數值範圍:")
    print(f"    - u: [{u_sensors.min():.4f}, {u_sensors.max():.4f}]")
    print(f"    - v: [{v_sensors.min():.4f}, {v_sensors.max():.4f}]")
    print(f"    - p: [{p_sensors.min():.4f}, {p_sensors.max():.4f}]")
    
    # 6. 計算訓練用的總資料點數
    print(f"\n📊 步驟 5: 訓練資料統計")
    total_sensor_points = T * K * 3  # T timesteps × K sensors × 3 variables
    print(f"  ✓ 總 sensor 資料點: {total_sensor_points:,}")
    print(f"    = {T} timesteps × {K} sensors × 3 variables")
    
    # 檢查是否匹配預期
    expected_T = int((t_max - t_min) / (t_selected[1] - t_selected[0])) + 1
    if abs(T - expected_T) <= 1:  # 允許 ±1 的誤差
        print(f"  ✓ Timestep 數量符合預期")
    else:
        print(f"  ⚠️ Timestep 數量異常: 預期 ~{expected_T}, 實際 {T}")
    
    # 7. 模擬 flatten 成訓練張量
    print(f"\n📦 步驟 6: 模擬 Flatten 成訓練張量")
    
    # 方法 1: 依照 kolmogorov.py 的邏輯 (先 flatten 時間-空間)
    u_train_flat = u_sensors.flatten()
    v_train_flat = v_sensors.flatten()
    p_train_flat = p_sensors.flatten()
    
    print(f"  ✓ Flatten 後形狀: {u_train_flat.shape} (T*K,)")
    print(f"  ✓ 總訓練點數 (per variable): {len(u_train_flat):,}")
    
    # 檢查 flatten 前後數值一致性
    u_check = u_sensors[0, 0]  # 第一個 timestep, 第一個 sensor
    u_flat_check = u_train_flat[0]  # flatten 後的第一個元素
    
    if abs(u_check - u_flat_check) < 1e-10:
        print(f"  ✓ Flatten 順序驗證通過")
    else:
        print(f"  ⚠️ Flatten 順序可能有誤!")
    
    # 8. 總結
    print("\n" + "=" * 80)
    print("📝 測試總結")
    print("=" * 80)
    
    checks = {
        "Sensor 索引載入": len(indices) == K,
        "時間範圍篩選": abs(T - expected_T) <= 1,
        "資料無 NaN/Inf": not (has_nan or has_inf),
        "Sensor 資料提取": u_sensors.shape == (T, K),
        "總資料點數": len(u_train_flat) == T * K,
    }
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"  {status} {check_name}")
    
    if all_passed:
        print("\n🎉 所有檢查通過！Sensor data 完整傳入訓練流程。")
        return True
    else:
        print("\n⚠️ 部分檢查未通過，請檢查資料載入邏輯。")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="測試 sensor data 載入")
    parser.add_argument("--cfg", type=str, 
                       default="configs/experiments/S1_sensor_strategy/s1_qr_K100_2d_re50.yml",
                       help="配置檔案路徑")
    
    args = parser.parse_args()
    
    success = test_sensor_loading(args.cfg)
    sys.exit(0 if success else 1)
