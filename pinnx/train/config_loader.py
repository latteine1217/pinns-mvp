"""
配置載入與標準化模組

提供 YAML 配置檔案的載入、標準化與損失權重推導功能。
確保配置格式一致性，並對已移除的舊鍵名採取 fail-fast。

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
    ...     loss_cfg=config['losses'],
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
        >>> print(config['model']['fourier_features']['type'])
        axis_selective
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
    標準化配置結構（僅做結構性/安全性修補，不做向後相容轉換）

    原則：
        - 舊鍵名（如 model.use_fourier / model.fourier_m）已移除：若出現直接拋錯。
        - Fourier 配置只接受 model.fourier_features（含 type/fourier_m/fourier_sigma/...）。
    
    Args:
        config: 原始配置字典
        
    Returns:
        標準化後的配置字典
        
    Note:
        若你在遷移舊配置，請先跑 scripts/tools/validate_config_keys.py 取得明確替換路徑。
    """
    model_cfg = config.get('model', {})
    
    # 禁止使用舊版 model.fourier 配置
    if 'fourier' in model_cfg:
        raise ValueError(
            "`model.fourier` 結構已移除，請改用 `model.fourier_features`"
        )

    # 禁止使用已移除的扁平 Fourier 鍵名（避免隱式相容/靜默行為）
    removed_flat_keys = {
        'use_fourier': "model.fourier_features.type",
        'fourier_m': "model.fourier_features.fourier_m",
        'fourier_sigma': "model.fourier_features.fourier_sigma",
        'trainable_fourier': "model.fourier_features.trainable_fourier",
        'fourier_use_2pi': "model.fourier_features.fourier_use_2pi",
        'fourier_multiscale': "model.fourier_features.type (axis_selective/standard) + config",
    }
    for key, replacement in removed_flat_keys.items():
        if key in model_cfg:
            raise ValueError(
                f"已移除的舊鍵名: model.{key}\n"
                f"請改用: {replacement}"
            )

    ff_cfg = model_cfg.get('fourier_features')
    if not isinstance(ff_cfg, dict):
        raise ValueError("缺少必要配置: model.fourier_features（必須是 dict）")

    ff_type = ff_cfg.get('type')
    if ff_type not in {'standard', 'axis_selective', 'hybrid', 'disabled'}:
        raise ValueError(
            "model.fourier_features.type 必須是 'standard' / 'axis_selective' / 'hybrid' / 'disabled'"
        )

    if ff_type == 'hybrid':
        # hybrid 類型需要 axes 配置
        if 'axes' not in ff_cfg:
            raise ValueError(
                "model.fourier_features.type='hybrid' 時必須提供 'axes' 配置"
            )
        axes_cfg = ff_cfg['axes']
        if not isinstance(axes_cfg, dict) or len(axes_cfg) == 0:
            raise ValueError(
                "model.fourier_features.axes 必須是非空字典"
            )
    elif ff_type != 'disabled':
        # standard 和 axis_selective 類型需要 fourier_m 和 fourier_sigma
        if 'fourier_m' not in ff_cfg or 'fourier_sigma' not in ff_cfg:
            raise ValueError(
                "Fourier features 啟用時必須提供 "
                "model.fourier_features.fourier_m 與 model.fourier_features.fourier_sigma"
            )

    config['model'] = model_cfg

    # ✅ 設置 physics_validation 預設值
    if 'physics_validation' not in config:
        config['physics_validation'] = {}

    physics_val_cfg = config['physics_validation']
    physics_val_cfg.setdefault('enabled', True)  # 預設啟用物理診斷
    physics_val_cfg.setdefault('strict_mode', False)  # 預設診斷模式（不拒絕保存）

    if 'thresholds' not in physics_val_cfg:
        physics_val_cfg['thresholds'] = {}

    thresholds = physics_val_cfg['thresholds']
    # 預設閾值設定為參考值（用於診斷，非強制）
    thresholds.setdefault('mass_conservation', 1.0e-2)
    thresholds.setdefault('momentum_conservation', 1.0e-1)
    thresholds.setdefault('boundary_condition', 1.0e-3)

    physics_val_cfg.setdefault('save_metrics', True)  # 預設保存診斷指標

    logging.debug(f"✅ Physics validation 配置: enabled={physics_val_cfg['enabled']}, "
                  f"strict_mode={physics_val_cfg['strict_mode']}, "
                  f"thresholds={physics_val_cfg['thresholds']}")

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
        loss_cfg: 損失配置字典（來自 config['losses']）
        prior_weight: 先驗損失權重（來自 config 頂層或默認值）
        is_vs_pinn: 是否為 VS-PINN 模式
        
    Returns:
        (base_weights, adaptive_terms):
            - base_weights: 各損失項的基礎權重字典
            - adaptive_terms: 可進行自適應調整的損失項名稱列表
            
    處理邏輯:
        1. 從配置讀取各損失項權重，未設置則使用預設值
        2. 非 VS-PINN 模式下過濾 VS_ONLY_LOSSES
        3. 處理特殊情況（如 periodicity 在非 VS-PINN 模式下的啟用條件）
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
