"""標準化輔助函數

提供輸入與輸出標準化的設置與配置功能。
"""

import logging
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch

from pinnx.utils.normalization import InputTransform, InputNormConfig, OutputTransform


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
    
    # 從配置讀取變量順序
    variable_order = norm_cfg.get('variable_order', ['u', 'v', 'w', 'p'])
    logger.info(f"📐 Variable order from config: {variable_order}")
    
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
        InputTransform 實例或 None（若不需要標準化）
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
