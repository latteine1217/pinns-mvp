"""
RANS 先驗資料載入器（NPY 格式）
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator


def _validate_les_config_consistency(
    lowfi_cfg: Dict[str, Any],
    main_config: Dict[str, Any],
    file_config: Dict[str, Any]
) -> None:
    """驗證 LES NPY 文件的 config 與 YAML 配置的一致性

    檢查關鍵物理參數（nu, L, k_f, A）是否一致，如不一致則發出警告。

    Args:
        lowfi_cfg: lowfi_prior 配置區段
        main_config: 完整的訓練配置
        file_config: LES NPY 文件中的 config 字典
    """
    # 提取 YAML 配置中的物理參數（可能在多個位置）
    physics_cfg = main_config.get('physics', {})
    kol_cfg = main_config.get('data', {}).get('kolmogorov_config', {})

    # 檢查項目：(NPY key, YAML paths, 容差)
    checks = [
        ('nu', [
            ('physics', 'nu'),
            ('data.kolmogorov_config.physics_params', 'nu'),
        ], 1e-6),
        ('L', [
            ('data.kolmogorov_config', 'L'),
        ], 1e-6),
        ('k_f', [
            ('physics.forcing', 'k_f'),
            ('physics.kolmogorov_flow', 'k_f'),
            ('data.kolmogorov_config.physics_params', 'k_f'),
        ], 1e-6),
        ('A', [
            ('physics.forcing', 'amplitude'),
            ('physics.kolmogorov_flow', 'forcing_amplitude'),
            ('data.kolmogorov_config.physics_params', 'forcing_amplitude'),
        ], 1e-6),
    ]

    warnings = []

    for npy_key, yaml_paths, tolerance in checks:
        npy_value = file_config.get(npy_key)
        if npy_value is None:
            continue

        # 檢查所有可能的 YAML 路徑
        yaml_value = None
        yaml_location = None

        for location, key in yaml_paths:
            if location == 'physics':
                yaml_value = physics_cfg.get(key)
            elif location == 'physics.forcing':
                yaml_value = physics_cfg.get('forcing', {}).get(key)
            elif location == 'physics.kolmogorov_flow':
                yaml_value = physics_cfg.get('kolmogorov_flow', {}).get(key)
            elif location == 'data.kolmogorov_config':
                yaml_value = kol_cfg.get(key)
            elif location == 'data.kolmogorov_config.physics_params':
                yaml_value = kol_cfg.get('physics_params', {}).get(key)

            if yaml_value is not None:
                yaml_location = f"{location}.{key}"
                break

        # 比較數值
        if yaml_value is not None:
            try:
                npy_float = float(npy_value)
                yaml_float = float(yaml_value)

                if abs(npy_float - yaml_float) > tolerance:
                    warnings.append(
                        f"   {npy_key}: NPY={npy_float} vs YAML({yaml_location})={yaml_float}"
                    )
            except (ValueError, TypeError):
                # 非數值比較（例如字符串）
                if str(npy_value) != str(yaml_value):
                    warnings.append(
                        f"   {npy_key}: NPY={npy_value} vs YAML({yaml_location})={yaml_value}"
                    )

    if warnings:
        logging.warning(
            f"\n⚠️  LES 物理參數與配置文件不一致:\n" + "\n".join(warnings) +
            f"\n   建議：使用 NPY 文件中的參數（更可靠）"
        )


def load_rans_prior_data(
    config: Dict[str, Any], 
    training_data: Dict[str, torch.Tensor],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """載入 RANS 低保真先驗資料並插值到訓練點（使用 NPY memory-mapped）
    
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
    
    rans_path = Path(rans_path)
    logging.info(f"📂 載入低保真先驗資料: {rans_path}")

    # ========================================================
    # ✅ 場景檢測：文件 vs 目錄
    # ========================================================
    # 場景 1: 單一 NPY 文件（Kolmogorov flow 瞬時場）
    # 場景 2: 目錄結構（3D channel flow 時間平均場）

    # 初始化場景共用變量
    file_config: Dict[str, Any] = {}  # 存儲 NPY 文件中的 config

    if rans_path.is_file():
        # ========================================================
        # 場景 1: Kolmogorov Flow 瞬時場（單一 NPY 文件）
        # ========================================================
        logging.info("   檢測到單一 NPY 文件（瞬時場）")

        # 載入 NPY 文件（使用 allow_pickle 讀取字典結構）
        try:
            payload = np.load(rans_path, allow_pickle=True)
            if hasattr(payload, 'item'):
                payload = payload.item()  # 解包字典

            if not isinstance(payload, dict):
                raise ValueError(
                    f"NPY 文件格式錯誤！期望字典結構，但得到 {type(payload)}"
                )
        except Exception as e:
            raise ValueError(
                f"無法載入 NPY 文件: {rans_path}\n"
                f"錯誤: {e}"
            )

        # 提取座標（1D 陣列）
        if 'x' not in payload or 'y' not in payload:
            raise ValueError(
                f"NPY 文件缺少座標資訊！需要 'x' 和 'y'，但只找到: {list(payload.keys())}"
            )

        x_rans_1d = np.asarray(payload['x'])
        y_rans_1d = np.asarray(payload['y'])

        # 驗證座標是 1D 陣列
        if x_rans_1d.ndim != 1 or y_rans_1d.ndim != 1:
            raise ValueError(
                f"座標必須是 1D 陣列！得到 x.shape={x_rans_1d.shape}, y.shape={y_rans_1d.shape}"
            )

        logging.info(f"   ✅ 讀取 1D 座標: x[{len(x_rans_1d)}], y[{len(y_rans_1d)}]")

        # 提取速度場（瞬時場，需要從時間序列中選擇一個快照）
        # 策略：取時間序列的時間平均（作為 prior）
        if 'u' not in payload or 'v' not in payload:
            raise ValueError(
                f"NPY 文件缺少速度場！需要 'u' 和 'v'，但只找到: {list(payload.keys())}"
            )

        u_all = np.asarray(payload['u'])  # 可能是 [T, N, N] 或 [N, N]
        v_all = np.asarray(payload['v'])

        # 處理時間序列：如果是 3D，取時間平均
        if u_all.ndim == 3:
            logging.info(f"   檢測到時間序列 shape={u_all.shape}，取時間平均作為 prior")
            u_rans = u_all.mean(axis=0)  # [N, N]
            v_rans = v_all.mean(axis=0)
        elif u_all.ndim == 2:
            logging.info(f"   檢測到單一快照 shape={u_all.shape}")
            u_rans = u_all
            v_rans = v_all
        else:
            raise ValueError(
                f"速度場維度錯誤！期望 [N, N] 或 [T, N, N]，得到 {u_all.shape}"
            )

        # 壓力場（Kolmogorov 瞬時場可能沒有壓力）
        if 'p' in payload:
            p_all = np.asarray(payload['p'])
            if p_all.ndim == 3:
                p_rans = p_all.mean(axis=0)
            elif p_all.ndim == 2:
                p_rans = p_all
            else:
                p_rans = np.zeros_like(u_rans)
            p_valid = True
            logging.info("   ✅ 讀取壓力場")
        else:
            p_rans = np.zeros_like(u_rans)
            p_valid = False
            logging.warning("   ⚠️  NPY 文件無壓力場，將在 loss 計算中跳過壓力項")

        # 渦流黏度（可選）
        if 'nu_t' in payload:
            nu_t_all = np.asarray(payload['nu_t'])
            if nu_t_all.ndim == 3:
                nu_t_rans = nu_t_all.mean(axis=0)
            elif nu_t_all.ndim == 2:
                nu_t_rans = nu_t_all
            else:
                nu_t_rans = np.zeros_like(u_rans)
            logging.info(f"   ✅ 讀取渦流黏度: nu_t 範圍=[{nu_t_rans.min():.2e}, {nu_t_rans.max():.2e}]")
        else:
            # Kolmogorov flow 通常不需要 nu_t（不使用 RANS/LES 閉合模型）
            nu_t_rans = np.zeros_like(u_rans)
            logging.info("   ⚠️  NPY 文件無 nu_t 場，使用零場")

        logging.info(f"   場解析度: {u_rans.shape}")
        logging.info(f"   座標範圍: x=[{x_rans_1d.min():.3f}, {x_rans_1d.max():.3f}], "
                     f"y=[{y_rans_1d.min():.3f}, {y_rans_1d.max():.3f}]")

        # 提取並記錄物理參數（從 NPY config）
        file_config = payload.get('config', {})
        if file_config:
            logging.info(f"\n📋 LES NPY 文件物理參數:")
            for key in ['N', 'L', 'nu', 'A', 'k_f', 'dt', 'model']:
                if key in file_config:
                    logging.info(f"   {key} = {file_config[key]}")

            # 驗證與配置文件的一致性
            _validate_les_config_consistency(lowfi_cfg, config, file_config)

    elif rans_path.is_dir():
        # ========================================================
        # 場景 2: 3D Channel Flow 時間平均場（目錄結構）
        # ========================================================
        logging.info("   檢測到目錄結構（時間平均場）")

        rans_structure = lowfi_cfg.get('rans_structure', {})
        group_path = rans_structure.get('group_path', 'mean_field')

        # 移除開頭的 '/' (NPY 使用目錄結構，不需要)
        if group_path.startswith('/'):
            group_path = group_path[1:]

        mean_field_dir = rans_path / group_path

        if not mean_field_dir.exists():
            raise FileNotFoundError(
                f"RANS 資料目錄不存在: {mean_field_dir}\n"
                f"請確認資料已轉換為 NPY 格式"
            )

        # 讀取 1D 座標
        x_file = mean_field_dir / 'x.npy'
        y_file = mean_field_dir / 'y.npy'

        if not x_file.exists() or not y_file.exists():
            raise FileNotFoundError(
                f"LES 資料格式錯誤！需要 1D 座標 'x.npy' 和 'y.npy'，但找到: {list(mean_field_dir.glob('*.npy'))}"
            )

        x_rans_1d = np.load(x_file, mmap_mode='r')
        y_rans_1d = np.load(y_file, mmap_mode='r')
        logging.info(f"   ✅ 讀取 1D 座標: x[{len(x_rans_1d)}], y[{len(y_rans_1d)}]")

        # 讀取速度場（memory-mapped，零拷貝）
        u_rans = np.load(mean_field_dir / 'u.npy', mmap_mode='r')  # [N, N]
        v_rans = np.load(mean_field_dir / 'v.npy', mmap_mode='r')

        # 壓力場（LES 可能沒有）
        p_file = mean_field_dir / 'p.npy'
        if p_file.exists():
            p_rans = np.load(p_file, mmap_mode='r')
            p_valid = True
        else:
            p_rans = np.zeros_like(u_rans)
            p_valid = False
            logging.warning("   ⚠️  LES 模型無壓力場，將在 loss 計算中跳過壓力項")

        # 渦流黏度（必須存在）
        nu_t_file = mean_field_dir / 'nu_t.npy'
        if not nu_t_file.exists():
            raise FileNotFoundError(
                f"LES 資料缺少 'nu_t.npy' 場！找到: {list(mean_field_dir.glob('*.npy'))}"
            )

        nu_t_rans = np.load(nu_t_file, mmap_mode='r')
        logging.info(f"   ✅ 讀取渦流黏度: nu_t 範圍=[{nu_t_rans.min():.2e}, {nu_t_rans.max():.2e}]")

        logging.info(f"   LES 解析度: {u_rans.shape}")
        logging.info(f"   座標範圍: x=[{x_rans_1d.min():.3f}, {x_rans_1d.max():.3f}], "
                     f"y=[{y_rans_1d.min():.3f}, {y_rans_1d.max():.3f}]")

    else:
        raise FileNotFoundError(
            f"lowfi_prior.data_path 既非文件也非目錄: {rans_path}\n"
            f"請確認路徑是否正確"
        )
    
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
    # ✅ 外插偵測與警告
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
        # ✅ 添加 metadata 標記壓力無效與物理參數
        'metadata': {
            'pressure_valid': p_valid,
            'model_type': 'les',
            'n_extrapolated': int(n_extrap),
            'extrapolation_ratio': float(n_extrap / len(coords_pde)),
            # ✨ 新增：NPY 文件中的物理參數（用於驗證與診斷）
            'file_config': file_config,
            'data_source': str(rans_path),
        }
    }
    
    logging.info(f"✅ LES 先驗插值完成 (NPY mmap):")
    logging.info(f"   PDE 配點: {len(u_prior_pde)} 個")
    logging.info(f"   感測點: {len(u_prior_sensors)} 個")
    logging.info(f"   u 統計: min={u_prior_pde.min():.4f}, max={u_prior_pde.max():.4f}, mean={u_prior_pde.mean():.4f}")
    logging.info(f"   v 統計: min={v_prior_pde.min():.4f}, max={v_prior_pde.max():.4f}, mean={v_prior_pde.mean():.4f}")
    logging.info(f"   nu_t 統計: min={nu_t_prior_pde.min():.2e}, max={nu_t_prior_pde.max():.2e}")
    logging.info(f"   壓力有效: {p_valid}")
    
    return rans_prior
