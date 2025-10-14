"""
配置載入與標準化模組

提供 YAML 配置檔案的載入、標準化與損失權重推導功能。
確保配置格式一致性，支援嵌套與扁平兩種配置格式。

主要功能:
    - load_config: 載入並標準化 YAML 配置
    - normalize_config_structure: 配置結構標準化
    - derive_loss_weights: 損失權重推導

範例:
    >>> from pinnx.train.config_loader import load_config, derive_loss_weights
    >>> 
    >>> # 載入配置
    >>> config = load_config('configs/channel_flow.yml')
    >>> 
    >>> # 推導損失權重
    >>> base_weights, adaptive_terms = derive_loss_weights(
    ...     loss_cfg=config['loss'],
    ...     prior_weight=config.get('prior_weight', 0.1),
    ...     is_vs_pinn=True
    ... )
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


# ============================================================================
# 損失權重配置常數
# ============================================================================

# 損失項名稱到配置鍵的映射
LOSS_KEY_MAP: Dict[str, str] = {
    'data': 'data_weight',
    'momentum_x': 'momentum_x_weight',
    'momentum_y': 'momentum_y_weight',
    'momentum_z': 'momentum_z_weight',
    'continuity': 'continuity_weight',
    'wall_constraint': 'wall_constraint_weight',
    'periodicity': 'periodicity_weight',
    'inlet': 'inlet_weight',
    'initial_condition': 'initial_condition_weight',
    'bulk_velocity': 'bulk_velocity_weight',
    'centerline_dudy': 'centerline_dudy_weight',
    'centerline_v': 'centerline_v_weight',
    'pressure_reference': 'pressure_reference_weight',
}

# 各損失項的預設權重
DEFAULT_WEIGHTS: Dict[str, float] = {
    'data': 10.0,
    'momentum_x': 2.0,
    'momentum_y': 2.0,
    'momentum_z': 2.0,
    'continuity': 2.0,
    'wall_constraint': 10.0,
    'periodicity': 5.0,
    'inlet': 10.0,
    'initial_condition': 100.0,
    'bulk_velocity': 2.0,
    'centerline_dudy': 1.0,
    'centerline_v': 1.0,
    'pressure_reference': 0.0,
}

# VS-PINN 專屬損失項（非 VS-PINN 模式下不使用）
VS_ONLY_LOSSES = {
    'momentum_z',
    'bulk_velocity',
    'centerline_dudy',
    'centerline_v',
    'pressure_reference',
}


# ============================================================================
# 配置載入函數
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """
    載入並標準化 YAML 配置檔案
    
    Args:
        config_path: YAML 配置檔案路徑
        
    Returns:
        標準化後的配置字典
        
    Raises:
        FileNotFoundError: 配置檔案不存在
        yaml.YAMLError: YAML 格式錯誤
        
    範例:
        >>> config = load_config('configs/channel_flow.yml')
        >>> print(config['model']['use_fourier'])
        True
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"配置檔案不存在: {config_path}")
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 解析失敗: {config_path}\n{e}")
    
    if config is None:
        raise ValueError(f"配置檔案為空: {config_path}")
    
    return normalize_config_structure(config)


def normalize_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    標準化配置結構，支持嵌套和扁平兩種格式
    
    處理兼容性問題：
        1. model.fourier.{enabled, m, sigma, trainable} → 
           model.{use_fourier, fourier_m, fourier_sigma, fourier_trainable}
        2. model.fourier_features.{type, fourier_m, fourier_sigma} → 
           model.{use_fourier, fourier_m, fourier_sigma}
        3. 確保所有必要字段都有默認值
    
    Args:
        config: 原始配置字典
        
    Returns:
        標準化後的配置字典
        
    範例:
        >>> # 嵌套格式（舊版）
        >>> config = {
        ...     'model': {
        ...         'fourier': {
        ...             'enabled': True,
        ...             'm': 64,
        ...             'sigma': 2.0,
        ...             'trainable': False
        ...         }
        ...     }
        ... }
        >>> normalized = normalize_config_structure(config)
        >>> print(normalized['model']['use_fourier'])
        True
        >>> print(normalized['model']['fourier_m'])
        64
        
        >>> # 新版格式（fourier_features.type）
        >>> config = {
        ...     'model': {
        ...         'fourier_features': {
        ...             'type': 'disabled',
        ...             'fourier_m': 0,
        ...             'fourier_sigma': 0.0
        ...         }
        ...     }
        ... }
        >>> normalized = normalize_config_structure(config)
        >>> print(normalized['model']['use_fourier'])
        False
        >>> print(normalized['model']['fourier_m'])
        0
    """
    model_cfg = config.get('model', {})
    
    # 處理 Fourier 配置（嵌套格式 → 扁平格式）
    if 'fourier' in model_cfg and isinstance(model_cfg['fourier'], dict):
        fourier_cfg = model_cfg['fourier']
        
        # 映射 enabled → use_fourier
        if 'enabled' in fourier_cfg and 'use_fourier' not in model_cfg:
            model_cfg['use_fourier'] = fourier_cfg['enabled']
        
        # 映射 m → fourier_m
        if 'm' in fourier_cfg and 'fourier_m' not in model_cfg:
            model_cfg['fourier_m'] = fourier_cfg['m']
        
        # 映射 sigma → fourier_sigma
        if 'sigma' in fourier_cfg and 'fourier_sigma' not in model_cfg:
            model_cfg['fourier_sigma'] = fourier_cfg['sigma']
        
        # 映射 trainable → fourier_trainable
        if 'trainable' in fourier_cfg and 'fourier_trainable' not in model_cfg:
            model_cfg['fourier_trainable'] = fourier_cfg['trainable']
        
        logging.debug("✅ 將嵌套 fourier 配置標準化為扁平結構")
    
    # 處理 fourier_features.type 格式（新版格式）
    if 'fourier_features' in model_cfg and isinstance(model_cfg['fourier_features'], dict):
        ff_cfg = model_cfg['fourier_features']
        ff_type = ff_cfg.get('type', 'standard')
        
        # 處理 type="disabled"
        if ff_type == 'disabled':
            model_cfg['use_fourier'] = False
            model_cfg['fourier_m'] = 0
            model_cfg['fourier_sigma'] = 0.0
            logging.debug("✅ Fourier Features 已禁用 (type='disabled')")
        
        # 處理 type="standard" 或其他啟用類型
        elif ff_type in ['standard', 'enhanced', 'adaptive', 'axis_selective']:
            model_cfg['use_fourier'] = True
            model_cfg['fourier_m'] = ff_cfg.get('fourier_m', 32)
            model_cfg['fourier_sigma'] = ff_cfg.get('fourier_sigma', 5.0)
            logging.debug(f"✅ Fourier Features 已啟用 (type='{ff_type}', m={model_cfg['fourier_m']})")
        
        else:
            logging.warning(f"⚠️ 未知的 fourier_features.type='{ff_type}'，使用預設值")
    
    # 設置默認值（如果未設置）
    model_cfg.setdefault('use_fourier', True)  # 默認啟用 Fourier
    model_cfg.setdefault('fourier_m', 32)
    model_cfg.setdefault('fourier_sigma', 1.0)
    model_cfg.setdefault('fourier_trainable', False)
    
    config['model'] = model_cfg
    return config


# ============================================================================
# 損失權重推導
# ============================================================================

def derive_loss_weights(
    loss_cfg: Dict[str, Any],
    prior_weight: float,
    is_vs_pinn: bool
) -> Tuple[Dict[str, float], List[str]]:
    """
    根據配置推導基礎權重與可調整的損失項列表
    
    Args:
        loss_cfg: 損失配置字典（來自 config['loss']）
        prior_weight: 先驗損失權重（來自 config 頂層或默認值）
        is_vs_pinn: 是否為 VS-PINN 模式
        
    Returns:
        (base_weights, adaptive_terms):
            - base_weights: 各損失項的基礎權重字典
            - adaptive_terms: 可進行自適應調整的損失項名稱列表
            
    處理邏輯:
        1. 從配置讀取各損失項權重，未設置則使用預設值
        2. 非 VS-PINN 模式下過濾 VS_ONLY_LOSSES
        3. 處理特殊情況（如 boundary_weight 合併到 wall_constraint）
        4. 生成自適應調整列表（排除 prior，除非其權重 > 0）
        
    範例:
        >>> loss_cfg = {
        ...     'data_weight': 20.0,
        ...     'continuity_weight': 5.0,
        ... }
        >>> weights, terms = derive_loss_weights(loss_cfg, 0.1, is_vs_pinn=True)
        >>> print(weights['data'])
        20.0
        >>> print(weights['continuity'])
        5.0
        >>> print('prior' in terms)
        True
    """
    base_weights: Dict[str, float] = {}
    
    # 遍歷所有已知損失項
    for name, default_val in DEFAULT_WEIGHTS.items():
        # 非 VS-PINN 模式下跳過 VS 專屬損失項
        if not is_vs_pinn and name in VS_ONLY_LOSSES:
            continue
        
        # 從配置讀取權重
        cfg_key = LOSS_KEY_MAP.get(name)
        if cfg_key is not None:
            # 特殊處理：periodicity 在非 VS-PINN 模式下需明確配置才啟用
            if name == 'periodicity' and not is_vs_pinn and cfg_key not in loss_cfg:
                continue
            val = loss_cfg.get(cfg_key, default_val)
        else:
            val = default_val
        
        base_weights[name] = float(val)
    
    # 設置 prior 權重
    base_weights['prior'] = float(loss_cfg.get('prior_weight', prior_weight))
    
    # 處理 boundary_weight（舊配置兼容）
    if 'boundary_weight' in loss_cfg:
        current_wall = base_weights.get('wall_constraint', 0.0)
        base_weights['wall_constraint'] = current_wall + float(loss_cfg['boundary_weight'])
        logging.debug(f"將 boundary_weight 合併到 wall_constraint: {base_weights['wall_constraint']}")
    
    # 生成自適應調整項列表（排除 prior）
    adaptive_terms = [name for name in base_weights if name != 'prior']
    
    # 若 prior 權重 > 0，也加入自適應調整
    if base_weights.get('prior', 0.0) > 0.0:
        adaptive_terms.append('prior')
    
    return base_weights, adaptive_terms


# ============================================================================
# 模組導出
# ============================================================================

__all__ = [
    'load_config',
    'normalize_config_structure',
    'derive_loss_weights',
    'LOSS_KEY_MAP',
    'DEFAULT_WEIGHTS',
    'VS_ONLY_LOSSES',
]
