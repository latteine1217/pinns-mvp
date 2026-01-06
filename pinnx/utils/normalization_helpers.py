"""標準化輔助函數

提供輸入與輸出標準化的設置與配置功能。
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch

from pinnx.utils.normalization import (
    InputTransform, 
    InputNormConfig, 
    OutputTransform,
    KolmogorovInputTransform
)


def _collect_coordinate_tensors(training_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """從訓練資料中收集所有座標張量
    
    Args:
        training_data: 訓練資料字典
        
    Returns:
        座標張量列表
    """
    prefixes = ['sensors', 'pde', 'bc', 'ic']
    coords: List[torch.Tensor] = []
    for prefix in prefixes:
        spatial_key = f'coords_{prefix}_spatial'
        if spatial_key not in training_data:
            continue
        spatial = training_data[spatial_key]
        if not isinstance(spatial, torch.Tensor) or spatial.numel() == 0:
            continue
        t_key = f't_{prefix}'
        if t_key in training_data and training_data[t_key] is not None and training_data[t_key].numel() > 0:
            coords.append(torch.cat([spatial, training_data[t_key]], dim=1))
        else:
            coords.append(spatial)
    return coords


def setup_output_normalization(
    config: Dict[str, Any],
    training_data: Dict[str, torch.Tensor],
    logger: logging.Logger,
    member_id: Optional[int] = None
) -> Optional[OutputTransform]:
    """從訓練資料計算輸出標準化統計量
    
    Args:
        config: 配置字典
        training_data: 訓練資料字典
        logger: logging.Logger 實例
        member_id: Ensemble 成員 ID（用於日誌，None 表示單一模型）
    
    Returns:
        OutputTransform 實例或 None（若不需要標準化）
    """
    if not config.get('data', {}).get('normalize', False):
        return None
    
    norm_cfg = config.get('normalization', {})
    if norm_cfg.get('type') != 'training_data_norm' or norm_cfg.get('params'):
        return None
    
    # 從配置讀取變量順序（若未指定，嘗試自動推斷 2D/3D）
    variable_order = norm_cfg.get('variable_order')
    variable_source = "config"
    if not variable_order:
        model_vars = config.get('model', {}).get('output_variables')
        data_vars = config.get('data', {}).get('kolmogorov_config', {}).get('variables')
        physics_type = config.get('physics', {}).get('type')
        if model_vars:
            variable_order = model_vars
            variable_source = "model.output_variables"
        elif data_vars:
            variable_order = data_vars
            variable_source = "data.kolmogorov_config.variables"
        elif physics_type == 'kolmogorov_flow_2d':
            variable_order = ['u', 'v', 'p']
            variable_source = "physics.type=kolmogorov_flow_2d"
        else:
            variable_order = ['u', 'v', 'w', 'p']
            variable_source = "default_3d"
    logger.info(f"📐 Variable order from {variable_source}: {variable_order}")
    
    # 檢查 sensor data 是否包含實際值
    has_sensor_values = all(
        f'{var}_sensors' in training_data and 
        training_data[f'{var}_sensors'].numel() > 0
        for var in variable_order
    )
    
    if has_sensor_values:
        # 使用 sensor data 計算標準化
        logger.info(f"🔧 Computing normalization from sensor data")
        normalization_data = {
            var: training_data[f'{var}_sensors'].cpu().numpy()
            for var in variable_order
        }
    else:
        # 錯誤：缺少必要的 sensor 數據
        missing_vars = [var for var in variable_order 
                       if f'{var}_sensors' not in training_data 
                       or training_data[f'{var}_sensors'].numel() == 0]
        raise ValueError(
            f"\n❌ CRITICAL: Cannot compute normalization!\n"
            f"\n"
            f"Required variables: {variable_order}\n"
            f"Missing in training_data: {missing_vars}\n"
            f"\n"
            f"Please ensure your data preparation function creates all required '<var>_sensors' fields.\n"
            f"For 2D problems, set normalization.variable_order: ['u', 'v', 'p']\n"
            f"For 3D problems, set normalization.variable_order: ['u', 'v', 'w', 'p']\n"
        )
    
    # 驗證數據質量
    from pinnx.dataio.validation import validate_sensor_data_quality
    validate_sensor_data_quality(normalization_data, logger)
    
    # 記錄統計量供驗證
    logger.info("📊 Sensor Normalization Statistics:")
    for var, data in normalization_data.items():
        if isinstance(data, np.ndarray):
            logger.info(f"   {var}: mean={data.mean():.6f}, std={data.std():.6f}, N={len(data)}")
    
    data_normalizer = OutputTransform.from_data(
        normalization_data,
        norm_type='training_data_norm',
        variable_order=variable_order
    )
    
    member_suffix = f" (Ensemble member {member_id})" if member_id is not None else ""
    logger.info(f"✅ Normalization computed from sensor data{member_suffix}")
    logger.info(f"   Variable order: {variable_order}")
    logger.info(f"   Normalizer: {data_normalizer}")
    
    return data_normalizer


def create_input_normalizer(
    config: Dict[str, Any],
    training_data: Dict[str, torch.Tensor],
    is_vs_pinn: bool,
    device: torch.device
) -> Optional[InputTransform]:
    """創建輸入標準化器
    
    Args:
        config: 配置字典
        training_data: 訓練資料字典
        is_vs_pinn: 是否為 VS-PINN 模型
        device: PyTorch 設備
        
    Returns:
        InputTransform 或 KolmogorovInputTransform 實例，或 None（若不需要標準化）
    """
    scaling_cfg = config.get('model', {}).get('scaling', {})
    norm_type = scaling_cfg.get('input_norm', 'none')
    if norm_type is None:
        norm_type = 'none'

    norm_type = norm_type.lower()

    if norm_type in ('none', 'identity'):
        return None

    if is_vs_pinn and norm_type in ('vs_pinn', 'channel_flow'):
        # VS-PINN already applies dedicated scaling; avoid double normalization.
        return None
    
    # 🆕 特殊處理：Kolmogorov 標準化（與 JAX-PI 對齊）
    if norm_type == 'kolmogorov':
        # 從配置中獲取 t_max
        kol_cfg = config.get('data', {}).get('kolmogorov_config', {})
        time_range = kol_cfg.get('time_range', [0.0, 1.0])
        t_max = float(time_range[1])
        
        logging.info(f"🌀 使用 Kolmogorov 標準化器 (JAX-PI aligned)")
        logging.info(f"   t_max = {t_max:.2f}")
        logging.info(f"   時間維度: [0, {t_max}] → [0, 1]")
        logging.info(f"   空間維度: 保持不變 [0, 2π]")
        
        normalizer = KolmogorovInputTransform(t_max=t_max)
        normalizer.to(device)
        return normalizer

    bounds_tensor: Optional[torch.Tensor] = None
    if norm_type in ('channel_flow',):
        domain = config.get('physics', {}).get('domain', {})
        bounds: List[Tuple[float, float]] = []
        for axis in ['x', 'y', 'z']:
            rng = domain.get(f'{axis}_range')
            if rng is not None:
                bounds.append((float(rng[0]), float(rng[1])))
        if bounds:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)

    feature_range = tuple(scaling_cfg.get('input_norm_range', [-1.0, 1.0]))
    config_obj = InputNormConfig(
        norm_type=norm_type,
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        bounds=bounds_tensor
    )
    normalizer = InputTransform(config_obj)

    coord_tensors = _collect_coordinate_tensors(training_data)
    if coord_tensors:
        samples = torch.cat(coord_tensors, dim=0)
        if normalizer.bounds is not None and normalizer.bounds.shape[0] > samples.shape[1]:
            normalizer.bounds = normalizer.bounds[:samples.shape[1], :]
        normalizer.fit(samples)
    else:
        return None

    normalizer.to(device)
    return normalizer
