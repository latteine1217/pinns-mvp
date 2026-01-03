"""
RANS 先驗資料載入器
"""

import logging
from typing import Dict, Any

import h5py
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator


def load_rans_prior_data(
    config: Dict[str, Any], 
    training_data: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """載入 RANS 低保真先驗資料並插值到訓練點
    
    Args:
        config: 配置字典
        training_data: 訓練資料字典（包含座標）
        device: PyTorch 設備
        
    Returns:
        包含 RANS 先驗場的字典 {'u': ..., 'v': ..., 'p': ...}
    """
    lowfi_cfg = config.get('lowfi_prior', {})
    if not lowfi_cfg.get('enabled', False):
        logging.info("⏭️  RANS 先驗未啟用")
        return {}
    
    rans_path = lowfi_cfg.get('data_path')
    if not rans_path:
        logging.warning("⚠️  lowfi_prior.enabled=True 但未指定 data_path")
        return {}
    
    logging.info(f"📂 載入 Leith 先驗資料: {rans_path}")
    
    # 讀取 Leith 資料
    with h5py.File(rans_path, 'r') as f:
        rans_structure = lowfi_cfg.get('rans_structure', {})
        group_path = rans_structure.get('group_path', '/mean_field')
        group = f[group_path]
        
        # ========================================================
        # ✅ FIX #1: 讀取 Leith 1D 座標格式
        # ========================================================
        # Leith 模型使用 1D 陣列: x[N], y[N]
        if 'x' not in group or 'y' not in group:
            raise KeyError(
                f"Leith 資料格式錯誤！需要 1D 座標 'x' 和 'y'，但找到: {list(group.keys())}"
            )
        
        x_rans_1d = np.array(group['x'])
        y_rans_1d = np.array(group['y'])
        logging.info(f"   ✅ 讀取 Leith 1D 座標: x[{len(x_rans_1d)}], y[{len(y_rans_1d)}]")
        
        # 讀取速度場
        u_rans = np.array(group['u'])  # [N, N]
        v_rans = np.array(group['v'])
        
        # ========================================================
        # ✅ FIX #2: Leith 無壓力場，設置無效標記
        # ========================================================
        # Leith 是診斷模型，不求解壓力場
        # 建立零壓力陣列（用於插值器，但不用於 loss）
        p_rans = np.zeros_like(u_rans)
        p_valid = False
        logging.warning("   ⚠️  Leith 模型無壓力場，將在 loss 計算中跳過壓力項")
        
        # ========================================================
        # ✅ 讀取 Leith 渦流黏度 (nu_t)
        # ========================================================
        if 'nu_t' not in group:
            raise KeyError(
                f"Leith 資料缺少 'nu_t' 場！找到: {list(group.keys())}"
            )
        
        nu_t_rans = np.array(group['nu_t'])
        logging.info(f"   ✅ 讀取 Leith 渦流黏度: nu_t 範圍=[{nu_t_rans.min():.2e}, {nu_t_rans.max():.2e}]")
    
    logging.info(f"   Leith 解析度: {u_rans.shape}")
    logging.info(f"   Leith 座標範圍: x=[{x_rans_1d.min():.3f}, {x_rans_1d.max():.3f}], "
                 f"y=[{y_rans_1d.min():.3f}, {y_rans_1d.max():.3f}]")
    
    # 建立插值器
    interp_method = lowfi_cfg.get('interpolation', {}).get('method', 'linear')
    u_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), u_rans, method=interp_method, bounds_error=False, fill_value=None)
    v_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), v_rans, method=interp_method, bounds_error=False, fill_value=None)
    p_interp = RegularGridInterpolator((x_rans_1d, y_rans_1d), p_rans, method=interp_method, bounds_error=False, fill_value=None)
    
    # 建立 nu_t 插值器
    nu_t_interp = RegularGridInterpolator(
        (x_rans_1d, y_rans_1d), nu_t_rans, 
        method=interp_method, bounds_error=False, fill_value=None
    )
    
    # 提取訓練點座標（只需空間座標，忽略時間）
    # 假設使用 PDE 配點作為插值目標
    x_pde_np = training_data['x_pde'].cpu().numpy().flatten()
    y_pde_np = training_data['y_pde'].cpu().numpy().flatten()
    
    coords_pde = np.column_stack([x_pde_np, y_pde_np])
    
    # ========================================================
    # ✅ FIX #3: 外插偵測與警告
    # ========================================================
    # 檢查訓練點是否在先驗資料範圍內
    x_min, x_max = x_rans_1d.min(), x_rans_1d.max()
    y_min, y_max = y_rans_1d.min(), y_rans_1d.max()
    
    extrap_mask = (
        (coords_pde[:, 0] < x_min) | (coords_pde[:, 0] > x_max) |
        (coords_pde[:, 1] < y_min) | (coords_pde[:, 1] > y_max)
    )
    
    n_extrap = extrap_mask.sum()
    if n_extrap > 0:
        ratio = n_extrap / len(coords_pde)
        logging.warning(
            f"   ⚠️  {n_extrap}/{len(coords_pde)} ({ratio:.1%}) PDE 配點位於外插區域"
        )
        if ratio > 0.05:
            raise ValueError(
                f"過多外插點 ({ratio:.1%} > 5%)！請檢查座標對齊：\n"
                f"  先驗資料範圍: x=[{x_min:.4f}, {x_max:.4f}], y=[{y_min:.4f}, {y_max:.4f}]\n"
                f"  訓練點範圍: x=[{coords_pde[:, 0].min():.4f}, {coords_pde[:, 0].max():.4f}], "
                f"y=[{coords_pde[:, 1].min():.4f}, {coords_pde[:, 1].max():.4f}]"
            )
    
    # 插值到 PDE 配點
    u_prior_pde = u_interp(coords_pde)
    v_prior_pde = v_interp(coords_pde)
    p_prior_pde = p_interp(coords_pde)
    
    # 插值 nu_t 到 PDE 配點
    nu_t_prior_pde = nu_t_interp(coords_pde)
    
    # 同樣插值到感測點（用於驗證）
    x_sensors_np = training_data['x_sensors'].cpu().numpy().flatten()
    y_sensors_np = training_data['y_sensors'].cpu().numpy().flatten()
    coords_sensors = np.column_stack([x_sensors_np, y_sensors_np])
    
    u_prior_sensors = u_interp(coords_sensors)
    v_prior_sensors = v_interp(coords_sensors)
    p_prior_sensors = p_interp(coords_sensors)
    
    # 插值 nu_t 到感測點
    nu_t_prior_sensors = nu_t_interp(coords_sensors)
    
    # ========================================================
    # ✅ 轉換為 Tensor 並添加 metadata
    # ========================================================
    rans_prior = {
        'u_pde': torch.tensor(u_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'v_pde': torch.tensor(v_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'p_pde': torch.tensor(p_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'u_sensors': torch.tensor(u_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        'v_sensors': torch.tensor(v_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        'p_sensors': torch.tensor(p_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        'nu_t_pde': torch.tensor(nu_t_prior_pde, dtype=torch.float32, device=device).unsqueeze(1),
        'nu_t_sensors': torch.tensor(nu_t_prior_sensors, dtype=torch.float32, device=device).unsqueeze(1),
        # ✅ FIX #4: 添加 metadata 標記壓力無效
        'metadata': {
            'pressure_valid': p_valid,
            'model_type': 'leith',
            'n_extrapolated': int(n_extrap),
            'extrapolation_ratio': float(n_extrap / len(coords_pde))
        }
    }
    
    logging.info(f"✅ Leith 先驗插值完成:")
    logging.info(f"   PDE 配點: {len(u_prior_pde)} 個")
    logging.info(f"   感測點: {len(u_prior_sensors)} 個")
    logging.info(f"   u 統計: min={u_prior_pde.min():.4f}, max={u_prior_pde.max():.4f}, mean={u_prior_pde.mean():.4f}")
    logging.info(f"   v 統計: min={v_prior_pde.min():.4f}, max={v_prior_pde.max():.4f}, mean={v_prior_pde.mean():.4f}")
    logging.info(f"   nu_t 統計: min={nu_t_prior_pde.min():.2e}, max={nu_t_prior_pde.max():.2e}")
    logging.info(f"   壓力有效: {p_valid}")
    
    return rans_prior
