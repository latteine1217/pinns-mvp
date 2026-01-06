"""
Kolmogorov Flow 訓練資料載入器（NPY 格式，Memory-Mapped）
"""

import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch


def _load_dns_metadata(data_dir: Path) -> Dict[str, Any]:
    """載入 DNS 元數據（從 JSON）"""
    metadata_file = data_dir / "metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            # 確保 N 是整數（防止 JSON 轉換時變成 float）
            if 'N' in metadata:
                metadata['N'] = int(metadata['N'])
            return metadata
    else:
        # 預設值（2π 週期域，256x256 網格）
        logging.warning(f"未找到 metadata.json，使用預設值")
        return {'L': 2 * np.pi, 'N': 256}


def _extract_dns_initial_condition(
    config: Dict[str, Any],
    t_target: float,
    device: torch.device,
    subsample_factor: int = 2
) -> Optional[Dict[str, torch.Tensor]]:
    """
    從 DNS 數據中提取指定時間的全場作為初始條件（使用 NPY mmap）
    
    Args:
        config: Kolmogorov 配置字典（kolmogorov_config）
        t_target: 目標時間
        device: PyTorch 設備
        subsample_factor: 降采樣因子（2 = 每隔一個點取樣，減少 IC 點數）
    
    Returns:
        IC 數據字典，包含：
        - 'x', 'y', 't': 座標張量 [N_ic, 1]
        - 'u', 'v', 'p': 場變量張量 [N_ic, 1]
        - 't_actual': 實際提取的時間（最接近 t_target 的時間步）
        
        若無法提取則返回 None
    """
    try:
        data_path = Path(config['data_path'])
        
        # 載入時間軸（mmap 模式）
        time_all = np.load(data_path / 'time.npy', mmap_mode='r')
        
        # 找到最接近 t_target 的時間步
        t_ic_idx = np.argmin(np.abs(time_all - t_target))
        t_actual = float(time_all[t_ic_idx])
        
        if abs(t_actual - t_target) > 1.0:
            logging.warning(
                f"   ⚠️  DNS 時間步與目標相差較大: "
                f"t_target={t_target:.2f}, t_actual={t_actual:.2f}"
            )
        
        # 載入該時間步的全場數據（mmap，只讀取需要的切片）
        u_all = np.load(data_path / 'u.npy', mmap_mode='r')
        v_all = np.load(data_path / 'v.npy', mmap_mode='r')
        p_all = np.load(data_path / 'p.npy', mmap_mode='r') if (data_path / 'p.npy').exists() else None
        
        u_full = u_all[t_ic_idx]  # [N, N]
        v_full = v_all[t_ic_idx]
        p_full = p_all[t_ic_idx] if p_all is not None else None
        
        # 讀取網格信息
        metadata = _load_dns_metadata(data_path)
        N = metadata.get('N', u_full.shape[0])
        L = metadata.get('L', 2 * np.pi)
        
        # 生成空間網格（降采樣）
        N_sampled = N // subsample_factor
        x_1d = np.linspace(0, L, N_sampled, endpoint=False)
        y_1d = np.linspace(0, L, N_sampled, endpoint=False)
        X, Y = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        # 降采樣場變量（簡單跳點采樣）
        u_sampled = u_full[::subsample_factor, ::subsample_factor]
        v_sampled = v_full[::subsample_factor, ::subsample_factor]
        p_sampled = p_full[::subsample_factor, ::subsample_factor] if p_full is not None else np.zeros_like(u_sampled)
        
        # 展平並轉換為 PyTorch 張量
        N_ic = N_sampled * N_sampled
        ic_data = {
            'x': torch.tensor(X.flatten(), dtype=torch.float32, device=device).reshape(-1, 1),
            'y': torch.tensor(Y.flatten(), dtype=torch.float32, device=device).reshape(-1, 1),
            't': torch.full((N_ic, 1), t_actual, dtype=torch.float32, device=device),
            'u': torch.tensor(u_sampled.flatten(), dtype=torch.float32, device=device).reshape(-1, 1),
            'v': torch.tensor(v_sampled.flatten(), dtype=torch.float32, device=device).reshape(-1, 1),
            'p': torch.tensor(p_sampled.flatten(), dtype=torch.float32, device=device).reshape(-1, 1),
            't_actual': t_actual,  # 記錄實際時間
        }
        
        logging.info(
            f"   從 DNS 提取 IC: t={t_actual:.4f}, "
            f"原始網格 {N}×{N} → 降采樣 {N_sampled}×{N_sampled} = {N_ic} 點"
        )
        
        return ic_data
        
    except Exception as e:
        logging.error(f"   ❌ 提取 DNS IC 失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def prepare_kolmogorov_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備 Kolmogorov Flow 訓練資料 (固定感測器 x 時間序列)
    
    使用 NPY memory-mapped 檔案，實現零拷貝高效讀取
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        
    Returns:
        訓練資料字典
    """
    # 載入配置
    kol_cfg = config['data']['kolmogorov_config']
    data_path = Path(kol_cfg['data_path'])
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

    # 2. 載入 DNS 數據（memory-mapped，零拷貝）
    logging.info(f"📂 載入 DNS 數據（NPY mmap）: {data_path}")
    
    # 載入時間軸
    time_all = np.load(data_path / 'time.npy', mmap_mode='r')
    
    # 選擇時間範圍
    t_start, t_end = time_range
    time_mask = (time_all >= t_start) & (time_all <= t_end)
    time_selected = time_all[time_mask]
    T_selected = len(time_selected)
    
    logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}], 共 {T_selected} 個時間步")
    
    # 載入元數據
    metadata = _load_dns_metadata(data_path)
    N = metadata.get('N', 256)
    L = metadata.get('L', 2 * np.pi)
    
    # 建立空間座標網格
    x_1d = np.linspace(0, L, N, endpoint=False)
    y_1d = np.linspace(0, L, N, endpoint=False)
    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
    
    # 提取感測點的 (x, y) 座標
    X_flat = X_mesh.flatten()
    Y_flat = Y_mesh.flatten()
    
    x_sensor_locs = X_flat[spatial_indices]  # [K]
    y_sensor_locs = Y_flat[spatial_indices]  # [K]
    
    # 3. 提取感測點的時間序列數據（使用 mmap，只讀取需要的時間切片）
    # Memory-mapped 讀取：OS 自動管理分頁，實現按需載入
    u_all = np.load(data_path / 'u.npy', mmap_mode='r')
    v_all = np.load(data_path / 'v.npy', mmap_mode='r')
    
    p_file = data_path / 'p.npy'
    if p_file.exists():
        p_all = np.load(p_file, mmap_mode='r')
    else:
        p_all = None
    
    # 提取時間切片（零拷貝，只在實際存取時才載入記憶體）
    u_slice = u_all[time_mask]  # [T, N, N]
    v_slice = v_all[time_mask]
    p_slice = p_all[time_mask] if p_all is not None else None
    
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
    N_pde = config.get('training', {}).get('sampling', {}).get('N_pde', 10000)
    
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
    
    logging.info(f"✅ Kolmogorov 訓練數據準備完成 (Fixed Sensors, NPY mmap):")
    logging.info(f"   感測點數 K: {K}")
    logging.info(f"   時間步數 T: {T_selected}")
    logging.info(f"   總監督樣本 (K*T): {len(x_sensors)}")
    logging.info(f"   PDE 配點: {N_pde}")
    logging.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}]")
    
    return training_data


def prepare_time_window_data(
    config: Dict[str, Any],
    device: torch.device,
    window_idx: int,
    num_windows: int,
    prev_window_ic: Dict[str, torch.Tensor] | None = None
) -> Dict[str, torch.Tensor]:
    """
    為時間窗口訓練準備單一窗口的資料（使用 NPY mmap）
    
    核心策略（對齊 JAX-PI）：
    1. 時間切片：將總時間範圍劃分為 num_windows 個無重疊窗口
    2. 資料提取：只載入當前窗口的感測點時間序列
    3. IC 轉移：Window N+1 使用 Window N 的預測作為初始條件（可選）
    4. 配點採樣：在當前窗口的時空域內隨機採樣
    
    Args:
        config: 配置字典（包含完整的時間範圍）
        device: PyTorch 設備
        window_idx: 當前窗口索引（0-based）
        num_windows: 總窗口數
        prev_window_ic: 前一個窗口的初始條件（可選）
            格式：{
                'x': Tensor[N_ic, 1],    # IC 點的 x 座標
                'y': Tensor[N_ic, 1],    # IC 點的 y 座標
                't': Tensor[N_ic, 1],    # IC 時間（應該等於 t_window_start）
                'u': Tensor[N_ic, 1],    # u 速度
                'v': Tensor[N_ic, 1],    # v 速度
                'p': Tensor[N_ic, 1],    # 壓力
            }
    
    Returns:
        當前窗口的訓練資料字典（格式同 prepare_kolmogorov_training_data）
    
    Examples:
        >>> # 訓練 25 個窗口
        >>> for window_idx in range(25):
        >>>     window_data = prepare_time_window_data(
        >>>         config, device, window_idx, num_windows=25
        >>>     )
        >>>     # 訓練當前窗口...
    
    Notes:
        - 窗口劃分為無重疊（overlap=0%）
        - 若需要 IC 轉移，需在訓練完成後調用模型預測 t_window_end 的全場
        - 配點採樣在當前窗口時空域內均勻隨機
    """
    # ========== 1. 計算當前窗口的時間範圍 ==========
    kol_cfg = config['data']['kolmogorov_config']
    t_total_start, t_total_end = kol_cfg['time_range']
    
    # 窗口劃分（無重疊）
    window_duration = (t_total_end - t_total_start) / num_windows
    t_window_start = t_total_start + window_idx * window_duration
    t_window_end = t_window_start + window_duration
    
    # 檢查窗口索引有效性
    if window_idx < 0 or window_idx >= num_windows:
        raise ValueError(
            f"Invalid window_idx={window_idx}. Must be in [0, {num_windows-1}]"
        )
    
    logging.info(f"{'='*70}")
    logging.info(f"🪟 準備時間窗口資料")
    logging.info(f"   窗口索引: {window_idx}/{num_windows-1}")
    logging.info(f"   窗口時間範圍: [{t_window_start:.2f}, {t_window_end:.2f}]")
    logging.info(f"   窗口持續時間: {window_duration:.2f}s")
    logging.info(f"{'='*70}")
    
    # ========== 2. 修改配置為當前窗口的時間範圍 ==========
    # 深拷貝配置，避免修改原始配置
    import copy
    window_config = copy.deepcopy(config)
    window_config['data']['kolmogorov_config']['time_range'] = [
        t_window_start, t_window_end
    ]
    
    # ========== 3. 調用原有函數載入當前窗口資料 ==========
    training_data = prepare_kolmogorov_training_data(window_config, device)

    # ========== 3.5 載入低保真先驗（若啟用） ==========
    from pinnx.dataio.loaders.rans_prior import load_rans_prior_data

    rans_prior = load_rans_prior_data(window_config, training_data, device)
    if rans_prior:
        training_data['lowfi_prior'] = rans_prior
        training_data['has_prior'] = True
    else:
        training_data['has_prior'] = False
    
    # ========== 4. 處理初始條件（IC）==========
    # 策略：
    # - Window 0: 使用 DNS t=t_window_start 的全場資料作為 IC
    # - Window N (N>0): 使用前窗口的預測作為 IC（Transfer Learning）
    
    if window_idx == 0:
        # 第一個窗口：從 DNS 提取 t=t_window_start 的全場作為 IC
        logging.info(f"   Window 0: 從 DNS 提取 t={t_window_start:.2f} 的全場作為初始條件")
        
        # 提取 DNS IC
        ic_data = _extract_dns_initial_condition(
            config=kol_cfg,
            t_target=t_window_start,
            device=device,
            subsample_factor=2  # 降采樣因子（2 = 每隔一個點取樣）
        )
        
        if ic_data is not None:
            # 添加到訓練數據
            training_data['x_ic'] = ic_data['x']
            training_data['y_ic'] = ic_data['y']
            training_data['t_ic'] = ic_data['t']
            training_data['u_ic'] = ic_data['u']
            training_data['v_ic'] = ic_data['v']
            training_data['p_ic'] = ic_data['p']
            
            # 更新預拼接座標
            training_data['coords_ic_spatial'] = torch.cat([
                ic_data['x'], ic_data['y']
            ], dim=1)
            
            N_ic = len(ic_data['x'])
            logging.info(f"   ✅ 已添加 {N_ic} 個初始條件點（來自 DNS t={ic_data['t_actual']:.4f}）")
        else:
            logging.warning(f"   ⚠️  無法從 DNS 提取 IC，將使用空 IC（僅依賴 sensor 約束）")
        
    elif prev_window_ic is not None:
        # 後續窗口：使用前窗口的預測作為 IC
        logging.info("   Window N>0: 使用前窗口預測作為初始條件（IC Transfer）")
        
        # 驗證 prev_window_ic 格式
        required_keys = ['x', 'y', 't', 'u', 'v', 'p']
        missing_keys = [k for k in required_keys if k not in prev_window_ic]
        if missing_keys:
            raise ValueError(
                f"prev_window_ic 缺少必要欄位: {missing_keys}. "
                f"必須包含: {required_keys}"
            )
        
        # 將 IC 資料添加到訓練資料中
        # 注意：這裡的 t 應該等於 t_window_start
        training_data['x_ic'] = prev_window_ic['x']
        training_data['y_ic'] = prev_window_ic['y']
        training_data['t_ic'] = prev_window_ic['t']
        training_data['u_ic'] = prev_window_ic['u']
        training_data['v_ic'] = prev_window_ic['v']
        training_data['p_ic'] = prev_window_ic['p']
        
        # 更新預拼接座標
        training_data['coords_ic_spatial'] = torch.cat([
            prev_window_ic['x'], prev_window_ic['y']
        ], dim=1)
        
        N_ic = len(prev_window_ic['x'])
        logging.info(f"   ✅ 已添加 {N_ic} 個初始條件點（來自前窗口預測）")
        
        # 驗證時間一致性
        expected_t = torch.full_like(prev_window_ic['t'], t_window_start)
        if not torch.allclose(prev_window_ic['t'], expected_t, atol=1e-3):
            logging.warning(
                f"⚠️  IC 時間不匹配！\n"
                f"   預期: t={t_window_start:.4f}\n"
                f"   實際: t 範圍 [{prev_window_ic['t'].min():.4f}, "
                f"{prev_window_ic['t'].max():.4f}]"
            )
    else:
        # Window N>0 但沒有提供 IC：使用默認行為
        logging.warning(
            f"⚠️  Window {window_idx}>0 但未提供 prev_window_ic，"
            f"將缺少 IC 約束（可能影響收斂）"
        )
    
    # ========== 5. 添加窗口元資訊 ==========
    training_data['window_metadata'] = {
        'window_idx': window_idx,
        'num_windows': num_windows,
        't_start': t_window_start,
        't_end': t_window_end,
        'duration': window_duration,
        'has_ic_transfer': (prev_window_ic is not None),
    }
    
    logging.info(f"✅ 時間窗口 {window_idx} 資料準備完成")
    logging.info(f"{'='*70}\n")
    
    return training_data
