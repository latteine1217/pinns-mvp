"""
評估工具模組

提供統一的模型載入、預測與反標準化介面，供所有評估腳本使用。

主要功能：
1. load_model_for_evaluation: 載入 checkpoint 並自動處理架構偵測
2. predict_with_denormalization: 模型預測 + 自動反標準化

設計原則：
- 避免重複實現（Don't Repeat Yourself）
- 自動偵測 Fourier / ManualScalingWrapper / VS-PINN
- 優先使用 checkpoint 中的配置
- 統一反標準化策略

作者：OpenCode Agent
日期：2026-01-05
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import logging

from pinnx.utils.denormalization import denormalize_output

logger = logging.getLogger(__name__)


def load_model_for_evaluation(
    checkpoint_path: str,
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[nn.Module, Optional[Any]]:
    """
    載入訓練完成的模型（用於評估）
    
    自動處理：
    - Fourier 特徵偵測
    - ManualScalingWrapper 偵測與恢復
    - VS-PINN physics 狀態恢復
    - 配置優先級（checkpoint > file config）
    
    Args:
        checkpoint_path: 檢查點檔案路徑
        config: 配置字典（會被 checkpoint 中的配置覆蓋）
        device: 計算設備
    
    Returns:
        (model, physics): 模型與 physics 物件（若無則為 None）
    
    Examples:
        >>> config = yaml.safe_load(open('config.yml'))
        >>> model, physics = load_model_for_evaluation(
        ...     'checkpoints/model.pth',
        ...     config,
        ...     torch.device('cuda')
        ... )
    
    Notes:
        - 此函數來自 comprehensive_evaluation.py:load_checkpoint()
        - 優先使用 checkpoint 中保存的配置以避免架構不匹配
        - 支援 2D/3D 流場模型
    """
    logger.info(f"📥 Loading model from {checkpoint_path}")
    
    # STEP 1: 預先檢查檢查點架構，動態調整配置
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model_state = checkpoint.get('model_state_dict', checkpoint)
    
    # 優先使用 checkpoint 中保存的配置（如果存在）
    if 'config' in checkpoint:
        ckpt_config = checkpoint['config']
        logger.info("✅ Using config from checkpoint (overriding file config)")
        
        # 合併配置：checkpoint 優先，但保留評估相關的設置
        eval_settings = config.get('evaluation', {})
        config = ckpt_config
        config['evaluation'] = eval_settings  # 保留評估設置
    else:
        logger.warning("⚠️ No config in checkpoint, using file config (may cause architecture mismatch!)")
    
    # 檢測 Fourier 特徵是否存在（支持 ManualScalingWrapper）
    has_fourier = 'fourier.B' in model_state or 'base_model.fourier.B' in model_state
    
    # 檢測是否使用 wrapper（通過 base_model. 前綴或 input_min/max）
    is_wrapped = ('base_model.hidden_layers.0.linear.weight' in model_state or
                  'input_min' in model_state)
    
    # 檢測輸入維度（從 Fourier B 矩陣或第一層權重推斷）
    input_proj_shape = None
    if 'base_model.fourier.B' in model_state:
        input_proj_shape = model_state['base_model.fourier.B']
    elif 'fourier.B' in model_state:
        input_proj_shape = model_state['fourier.B']
    elif 'base_model.hidden_layers.0.linear.weight' in model_state:
        input_proj_shape = model_state['base_model.hidden_layers.0.linear.weight']
    elif 'hidden_layers.0.linear.weight' in model_state:
        input_proj_shape = model_state['hidden_layers.0.linear.weight']
    
    if input_proj_shape is not None:
        if has_fourier:
            # Fourier B matrix 形狀: (input_dim, m)
            # 實際輸入維度是 B.shape[0]，輸出是 2*m
            input_dim = input_proj_shape.shape[0]
            fourier_dim = input_proj_shape.shape[1] * 2  # sin + cos
        else:
            # Hidden layer 形狀: (hidden_size, input_dim)
            input_dim = input_proj_shape.shape[1]
            fourier_dim = None
        
        logger.info(f"🔍 Checkpoint architecture detected:")
        logger.info(f"   Input dim: {input_dim}, Has Fourier: {has_fourier}, Wrapped: {is_wrapped}")
        if fourier_dim:
            logger.info(f"   Fourier output dim: {fourier_dim}")
        
        # 動態調整配置以匹配檢查點
        if 'model' not in config:
            config['model'] = {}
        if 'fourier_features' not in config['model']:
            config['model']['fourier_features'] = {}
        
        if has_fourier and fourier_dim:  # Fourier enabled
            config['model']['fourier_features']['type'] = 'standard'
            # 從 B 矩陣推斷 m
            fourier_m = input_proj_shape.shape[1]
            config['model']['fourier_features']['fourier_m'] = int(fourier_m)
            if config['model']['fourier_features'].get('fourier_sigma', 0) == 0:
                config['model']['fourier_features']['fourier_sigma'] = 5.0
            logger.info(f"✅ Config adjusted to Fourier ENABLED (m={fourier_m})")
        else:  # Fourier disabled
            config['model']['fourier_features']['type'] = 'disabled'
            config['model']['fourier_features']['fourier_m'] = 0
            config['model']['fourier_features']['fourier_sigma'] = 0.0
            logger.info("✅ Config adjusted to Fourier DISABLED")
    
    # 從配置文件構建 statistics 以支持 3D 模型
    # 這確保 ManualScalingWrapper 能正確設置 input_min/max 的形狀
    statistics = None
    if 'physics' in config and 'domain' in config['physics']:
        domain = config['physics']['domain']
        statistics = {
            'x': {'range': domain.get('x_range', [0.0, 25.13])},
            'y': {'range': domain.get('y_range', [-1.0, 1.0])}
        }
        # 如果是 3D，添加 z 範圍
        if 'z_range' in domain:
            statistics['z'] = {'range': domain['z_range']}
        logger.info(f"📐 Constructed statistics from config: {list(statistics.keys())}")
    
    # CRITICAL FIX: 若 checkpoint 使用 ManualScalingWrapper，
    # 則必須創建 plain model（非 VS-PINN），因為 checkpoint 的 base_model 不含 input_scale_factors
    has_wrapper = (is_wrapped and 
                   'input_min' in model_state and 
                   'input_max' in model_state)
    
    original_physics_type = config.get('physics', {}).get('type', '')
    if has_wrapper and original_physics_type == 'vs_pinn_channel_flow':
        # 臨時禁用 VS-PINN，避免 create_model() 創建帶 input_scale_factors 的模型
        logger.info("⚠️  Checkpoint uses ManualScalingWrapper → Disabling VS-PINN mode for model creation")
        config['physics']['type'] = 'channel_flow_3d'  # 使用普通物理類型
    
    # 創建模型架構
    from pinnx.train.model_physics_factory import create_model, create_physics
    base_model = create_model(config, device, statistics=statistics)
    
    # 恢復原始 physics type（用於後續 physics 創建）
    if has_wrapper and original_physics_type == 'vs_pinn_channel_flow':
        config['physics']['type'] = original_physics_type
        logger.info("✅ Restored physics type to vs_pinn_channel_flow for physics module creation")
    
    # 檢查 create_model() 是否已經創建了 wrapper（避免雙重包裝）
    model_already_wrapped = hasattr(base_model, 'input_min') and hasattr(base_model, 'input_max')
    
    if has_wrapper and not model_already_wrapped:
        # Checkpoint 使用 wrapper，但 create_model() 沒有創建 → 需要手動包裝
        logger.info("🔧 Checkpoint uses ManualScalingWrapper, manually applying wrapper")
        from pinnx.models.wrappers import ManualScalingWrapper
        
        # 從 checkpoint 提取縮放範圍
        input_min = model_state['input_min'].cpu().numpy()
        input_max = model_state['input_max'].cpu().numpy()
        output_min = model_state.get('output_min', torch.zeros(4)).cpu().numpy()
        output_max = model_state.get('output_max', torch.ones(4)).cpu().numpy()
        
        # 從配置推斷輸入變數名稱（x, y, z）
        domain = config.get('physics', {}).get('domain', {})
        input_keys = ['x', 'y']
        if 'z_range' in domain or len(input_min) >= 3:
            input_keys.append('z')
        
        # 構建 input/output ranges 字典
        input_ranges = {key: (float(input_min[i]), float(input_max[i])) 
                       for i, key in enumerate(input_keys[:len(input_min)])}
        
        output_keys = ['u', 'v', 'w', 'p'] if len(output_min) >= 4 else ['u', 'v', 'p']
        output_ranges = {key: (float(output_min[i]), float(output_max[i])) 
                        for i, key in enumerate(output_keys[:len(output_min)])}
        
        model = ManualScalingWrapper(
            base_model,
            input_ranges=input_ranges,
            output_ranges=output_ranges
        ).to(device)
        logger.info(f"   Input ranges: {input_ranges}")
        logger.info(f"   Output ranges: {list(output_ranges.keys())}")
    elif model_already_wrapped:
        # create_model() 已經創建了 wrapper → 直接使用
        model = base_model
        logger.info("✅ Model already wrapped by create_model(), using directly")
    else:
        # Checkpoint 不使用 wrapper → 直接使用 base model
        model = base_model
        logger.info("ℹ️  Checkpoint uses bare model (no wrapper)")
    
    # 創建 physics 對象（用於恢復 VS-PINN 縮放參數）
    physics = None
    physics_type = config.get('physics', {}).get('type', '')
    if physics_type == 'vs_pinn_channel_flow':
        physics = create_physics(config, device)
        logger.info("✅ Created VS-PINN physics module")
     
    # 載入權重（使用已載入的 checkpoint）
    if 'model_state_dict' not in checkpoint:
        raise KeyError("checkpoint must include 'model_state_dict'")
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    logger.info(f"✅ Loaded model checkpoint from epoch {epoch}")
    
    # 轉移到目標設備
    model = model.to(device)
    
    # 恢復 physics 的 state_dict（VS-PINN 縮放參數等）
    if 'physics_state_dict' in checkpoint and physics is not None:
        physics.load_state_dict(checkpoint['physics_state_dict'])
        logger.info(f"✅ Restored physics state: {list(checkpoint['physics_state_dict'].keys())}")
        
        # 打印恢復的縮放參數（用於驗證）
        if hasattr(physics, 'N_x'):
            logger.info(f"   VS-PINN 縮放參數: N_x={physics.N_x.item():.2f}, "
                       f"N_y={physics.N_y.item():.2f}, N_z={physics.N_z.item():.2f}")
    elif physics is not None and physics_type == 'vs_pinn_channel_flow':
        raise KeyError("checkpoint missing VS-PINN 'physics_state_dict'")
    
    model.eval()
    return model, physics


def predict_with_denormalization(
    model: nn.Module,
    coords: torch.Tensor,
    config: Dict[str, Any],
    checkpoint_path: str,
    physics: Optional[Any] = None,
    batch_size: int = 10000,
    device: torch.device = torch.device('cpu')
) -> np.ndarray:
    """
    模型預測並自動反標準化到物理空間
    
    Args:
        model: 訓練好的模型
        coords: 座標張量 (N, 2) 或 (N, 3)
        config: 配置字典
        checkpoint_path: checkpoint 路徑（用於載入標準化統計量）
        physics: VS-PINN physics 模組（可選，用於座標縮放）
        batch_size: 批次大小
        device: 計算設備
    
    Returns:
        predictions: 反標準化後的預測 (N, output_dim)
    
    Examples:
        >>> coords = torch.tensor([[0.0, 0.5], [1.0, 0.5]])
        >>> pred = predict_with_denormalization(
        ...     model, coords, config, 'checkpoints/model.pth'
        ... )
        >>> print(pred.shape)  # (2, 3) or (2, 4)
    
    Notes:
        - 自動偵測 VS-PINN 座標縮放
        - 使用 denormalize_output() 統一反標準化
        - 支援大規模批次預測
    """
    # 檢查是否使用 VS-PINN 縮放
    use_vs_pinn = physics is not None and hasattr(physics, 'scale_coordinates')
    if use_vs_pinn:
        logger.info(f"🔧 Using VS-PINN coordinate scaling: N_x={physics.N_x.item():.2f}, "
                   f"N_y={physics.N_y.item():.2f}, N_z={physics.N_z.item():.2f}")
    else:
        logger.info(f"🔧 Using direct model inference (no scaling)")
    
    n_points = coords.shape[0]
    predictions_list = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch = coords[i:i+batch_size].to(device)
            
            # 應用 VS-PINN 座標縮放（如果有）
            if use_vs_pinn:
                batch = physics.scale_coordinates(batch)
            
            # 模型推理（輸出為標準化空間）
            pred = model(batch)
            
            # 反標準化回物理空間
            pred_physical = denormalize_output(
                pred.cpu().numpy(), 
                config, 
                output_norm_type='training_data_norm',
                verbose=(i == 0),  # 只在第一批次輸出詳細日誌
                checkpoint_path=checkpoint_path
            )
            
            predictions_list.append(pred_physical)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"  Progress: {i+len(batch)}/{n_points} ({100*(i+len(batch))/n_points:.1f}%)")
    
    # 拼接所有批次
    predictions = np.concatenate(predictions_list, axis=0)
    logger.info(f"✅ Prediction complete: {predictions.shape}")
    
    return predictions
