"""
Kolmogorov Flow 訓練資料載入器
"""

import json
import logging
from typing import Dict, Any

import h5py
import numpy as np
import torch


def prepare_kolmogorov_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備 Kolmogorov Flow 訓練資料 (固定感測器 x 時間序列)
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        
    Returns:
        訓練資料字典
    """
    # 載入配置
    kol_cfg = config['data']['kolmogorov_config']
    data_path = kol_cfg['data_path']
    time_range = kol_cfg['time_range']
    
    # 1. 讀取感測器位置檔案 (QR-Pivot 結果)
    sensor_file = config.get('sensors', {}).get('sensor_file')
    if not sensor_file:
        raise ValueError("必須在 config['sensors']['sensor_file'] 指定感測器位置檔案 (.json)")
        
    logging.info(f"📂 載入感測器位置: {sensor_file}")
    with open(sensor_file, 'r') as f:
        sensor_data = json.load(f)
        
    # sensor_indices 是展平後的空間索引 (0 ~ N*N-1)
    spatial_indices = np.array(sensor_data['indices'])
    K = len(spatial_indices)
    logging.info(f"   已選定 {K} 個固定空間感測點")

    # 2. 載入 DNS 全場數據
    logging.info(f"📂 載入 DNS 數據: {data_path}")
    with h5py.File(data_path, 'r') as f:
        # 讀取時間軸
        time_all = np.array(f['time'])
        
        # 選擇時間範圍
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        time_selected = time_all[time_mask]
        T_selected = len(time_selected)
        
        logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}], 共 {T_selected} 個時間步")
        
        # 讀取空間網格資訊
        N = int(f['config'].attrs['N'])
        L = float(f['config'].attrs['L'])
        
        # 建立空間座標網格
        x_1d = np.linspace(0, L, N, endpoint=False)
        y_1d = np.linspace(0, L, N, endpoint=False)
        X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        # 提取感測點的 (x, y) 座標
        X_flat = X_mesh.flatten()
        Y_flat = Y_mesh.flatten()
        
        x_sensor_locs = X_flat[spatial_indices]  # [K]
        y_sensor_locs = Y_flat[spatial_indices]  # [K]
        
        # 3. 提取感測點的時間序列數據
        # 策略：讀取需要的時間步，然後只取選定的空間點
        # 為了效率，我們先讀取所需的時間切片 [T_selected, N, N]
        # 注意：如果是大檔案，可能需要更精細的讀取策略
        u_slice = f['u'][time_mask]  # [T, N, N]
        v_slice = f['v'][time_mask]
        if 'p' in f:
            p_slice = f['p'][time_mask]
        else:
            p_slice = None
            
        # 展平空間維度 [T, N*N]
        u_flat = u_slice.reshape(T_selected, -1)
        v_flat = v_slice.reshape(T_selected, -1)
        p_flat = p_slice.reshape(T_selected, -1) if p_slice is not None else None
        
        # ========== 驗證 1: Sensor 索引越界檢查 ==========
        N_total = u_flat.shape[1]
        if spatial_indices.max() >= N_total:
            raise IndexError(
                f"❌ Sensor 索引越界！最大索引 {spatial_indices.max()} >= 總點數 {N_total}\n"
                f"   可能原因：Sensor 檔案基於不同網格解析度生成\n"
                f"   DNS 網格: {N}x{N} = {N_total} 點\n"
                f"   Sensor 索引範圍: [{spatial_indices.min()}, {spatial_indices.max()}]"
            )
        if spatial_indices.min() < 0:
            raise IndexError(
                f"❌ Sensor 索引無效！最小索引 {spatial_indices.min()} < 0"
            )
        
        logging.info(
            f"✅ Sensor 索引驗證通過: [{spatial_indices.min()}, {spatial_indices.max()}] "
            f"⊂ [0, {N_total-1}]"
        )
        # ===================================================
        
        # 提取感測點的值 [T, K]
        u_sensors_vals = u_flat[:, spatial_indices]
        v_sensors_vals = v_flat[:, spatial_indices]
        p_sensors_vals = p_flat[:, spatial_indices] if p_flat is not None else None
        
        # ========== 驗證 2: Sensor 資料形狀檢查 ==========
        expected_shape = (T_selected, K)
        if u_sensors_vals.shape != expected_shape:
            raise ValueError(
                f"❌ Sensor 資料形狀錯誤！\n"
                f"   預期: {expected_shape}\n"
                f"   實際: {u_sensors_vals.shape}"
            )
        
        logging.info(f"✅ Sensor 資料形狀驗證: u_sensors_vals.shape = {u_sensors_vals.shape}")
        # ===================================================
        
    # 4. 構建訓練張量 (T * K 樣本)
    # 我們需要將 [T, K] 展平成 [T*K, 1]
    
    # 時間座標: 每個感測點重複 T 次
    # [t0, t1, ..., t0, t1, ...]
    # 為了方便，我們使用 meshgrid 構造 (t, k)
    T_grid, K_grid = np.meshgrid(time_selected, np.arange(K), indexing='ij')
    
    # 展平
    t_train = T_grid.flatten()  # [T*K]
    k_indices = K_grid.flatten() # [T*K] 用於索引空間座標
    
    x_train = x_sensor_locs[k_indices]
    y_train = y_sensor_locs[k_indices]
    
    u_train = u_sensors_vals.flatten()
    v_train = v_sensors_vals.flatten()
    if p_sensors_vals is not None:
        p_train = p_sensors_vals.flatten()
    else:
        p_train = np.zeros_like(u_train)
    
    # ========== 驗證 3: Flatten 順序一致性檢查 ==========
    # 驗證 flatten 後的總長度
    assert len(u_train) == T_selected * K, \
        f"❌ Flatten 長度錯誤！預期 {T_selected * K}，實際 {len(u_train)}"
    
    # 驗證 C-order flatten 的對應關係
    # flatten([T, K]) with C-order -> [u(t0,k0), u(t0,k1), ..., u(t0,kK-1), u(t1,k0), ...]
    # 檢查第一個時間步的第一個 sensor
    if not np.isclose(u_train[0], u_sensors_vals[0, 0], rtol=1e-5):
        raise ValueError(
            f"❌ Flatten 順序錯誤！\n"
            f"   u_train[0] = {u_train[0]}\n"
            f"   u_sensors_vals[0, 0] = {u_sensors_vals[0, 0]}\n"
            f"   兩者應該相等（第一個時間步的第一個 sensor）"
        )
    
    # 檢查第二個時間步的第一個 sensor（如果有第二個時間步）
    if T_selected > 1:
        if not np.isclose(u_train[K], u_sensors_vals[1, 0], rtol=1e-5):
            raise ValueError(
                f"❌ Flatten 順序錯誤！\n"
                f"   u_train[{K}] = {u_train[K]}\n"
                f"   u_sensors_vals[1, 0] = {u_sensors_vals[1, 0]}\n"
                f"   兩者應該相等（第二個時間步的第一個 sensor）"
            )
    
    logging.info(f"✅ Flatten 順序驗證通過（C-order）")
    # =======================================================
        
    # 轉換為 Tensor
    x_sensors = torch.tensor(x_train, dtype=torch.float32, device=device).unsqueeze(1)
    y_sensors = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    t_sensors = torch.tensor(t_train, dtype=torch.float32, device=device).unsqueeze(1)
    u_sensors = torch.tensor(u_train, dtype=torch.float32, device=device).unsqueeze(1)
    v_sensors = torch.tensor(v_train, dtype=torch.float32, device=device).unsqueeze(1)
    p_sensors = torch.tensor(p_train, dtype=torch.float32, device=device).unsqueeze(1)
    
    # 5. PDE 配點採樣 (隨機時空採樣)
    # 從全域 (x, y, t) 中隨機採樣
    N_pde = config.get('sampling', {}).get('N_pde', 10000)
    
    x_pde = torch.rand(N_pde, 1, device=device) * L
    y_pde = torch.rand(N_pde, 1, device=device) * L
    t_pde = torch.rand(N_pde, 1, device=device) * (t_end - t_start) + t_start
    
    # 排序 t_pde 以優化因果訓練效率 (雖然 CausalWeighter 會自己處理，但排序好是個好習慣)
    t_pde, sort_idx = torch.sort(t_pde, dim=0)
    x_pde = x_pde[sort_idx].reshape(-1, 1)
    y_pde = y_pde[sort_idx].reshape(-1, 1)
    
    x_bc = torch.empty(0, 1, device=device)
    y_bc = torch.empty(0, 1, device=device)
    x_ic = torch.empty(0, 1, device=device)
    y_ic = torch.empty(0, 1, device=device)
    t_bc = torch.empty(0, 1, device=device)
    t_ic = torch.empty(0, 1, device=device)

    coords_pde_spatial = torch.cat([x_pde, y_pde], dim=1)
    coords_bc_spatial = torch.empty(0, 2, device=device)
    coords_sensors_spatial = torch.cat([x_sensors, y_sensors], dim=1)
    coords_ic_spatial = torch.empty(0, 2, device=device)

    # 構建訓練資料字典
    training_data = {
        # 預拼接座標
        'coords_pde_spatial': coords_pde_spatial,
        'coords_bc_spatial': coords_bc_spatial,
        'coords_sensors_spatial': coords_sensors_spatial,
        'coords_ic_spatial': coords_ic_spatial,

        # 感測點 (固定位置 x 時間序列)
        'x_sensors': x_sensors,
        'y_sensors': y_sensors,
        't_sensors': t_sensors,
        'u_sensors': u_sensors,
        'v_sensors': v_sensors,
        'p_sensors': p_sensors,
        
        # PDE 配點 (隨機時空)
        'x_pde': x_pde,
        'y_pde': y_pde,
        't_pde': t_pde,
        
        # 邊界條件 (Kolmogorov 是週期性的，這裡留空或設為空集合)
        'x_bc': x_bc,
        'y_bc': y_bc,
        't_bc': t_bc,
        
        # 初始條件 (t=0 全場快照，可選)
        # 若需要強 IC 約束，可在此添加 t=t_start 的全場數據
        'x_ic': x_ic,
        'y_ic': y_ic,
        't_ic': t_ic,
    }
    
    logging.info(f"✅ Kolmogorov 訓練數據準備完成 (Fixed Sensors):")
    logging.info(f"   感測點數 K: {K}")
    logging.info(f"   時間步數 T: {T_selected}")
    logging.info(f"   總監督樣本 (K*T): {len(x_sensors)}")
    logging.info(f"   PDE 配點: {N_pde}")
    logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}]")
    
    return training_data
