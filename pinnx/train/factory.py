"""
模型/物理/優化器工廠模組

本模組提供統一的工廠函數，用於創建 PINN 訓練所需的核心組件：
- 設備選擇（CUDA/MPS/CPU）
- 模型架構（PINN/Enhanced/VS-PINN）
- 物理方程模組（NS 2D/3D、VS-PINN）
- 優化器與學習率調度器

所有函數均提供完整的錯誤處理、類型提示與配置驗證。
"""

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from pinnx.models.fourier_mlp import PINNNet, create_enhanced_pinn, init_siren_weights
from pinnx.physics.ns_2d import NSEquations2D
from pinnx.physics.vs_pinn_channel_flow import create_vs_pinn_channel_flow


# ============================================================================
# 設備選擇
# ============================================================================

def get_device(device_name: str) -> torch.device:
    """
    獲取運算設備（支援自動選擇與手動指定）
    
    Args:
        device_name: 設備名稱，支援：
            - "auto": 自動選擇最佳可用設備（CUDA > MPS > CPU）
            - "cuda": NVIDIA GPU（需 CUDA 可用）
            - "mps": Apple Silicon GPU（需 MPS 可用）
            - "cpu": CPU 運算
    
    Returns:
        torch.device: PyTorch 設備物件
    
    Raises:
        ValueError: 若指定設備名稱無效
    
    Examples:
        >>> device = get_device("auto")
        >>> device = get_device("cuda")  # 若 CUDA 不可用會回退到 CPU
    """
    valid_devices = ["auto", "cuda", "mps", "cpu"]
    if device_name not in valid_devices:
        raise ValueError(
            f"Invalid device name '{device_name}'. "
            f"Must be one of: {valid_devices}"
        )
    
    if device_name == "auto":
        # 自動選擇最佳可用設備
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Auto-selected CUDA: {torch.cuda.get_device_name()}")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Auto-selected Apple Metal Performance Shaders")
        else:
            device = torch.device("cpu")
            logging.info("Auto-selected CPU (no GPU available)")
    
    elif device_name == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logging.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        else:
            logging.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
    
    elif device_name == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logging.info("Using Apple Metal Performance Shaders")
        else:
            logging.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
    
    else:  # cpu
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


# ============================================================================
# 模型創建
# ============================================================================

def create_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """
    建立 PINN 模型（支援 Fourier features、VS-PINN 縮放、手動標準化）
    
    Args:
        config: 完整配置字典，需包含：
            - model: 模型配置（type, in_dim, out_dim, width, depth, activation）
            - physics: 物理配置（type, domain, vs_pinn）
            - [可選] scaling: 標準化配置（output_norm）
        device: 計算設備
        statistics: [可選] 資料統計資訊（用於自動設定輸出範圍）
    
    Returns:
        nn.Module: 已初始化並移至目標設備的 PINN 模型
    
    Raises:
        KeyError: 若配置缺少必要欄位
        ValueError: 若配置參數無效（如輸入範圍不匹配域配置）
        ImportError: 若缺少必要的模組（如 ManualScalingWrapper）
    
    Notes:
        - 若 activation='sine'，自動應用 SIREN 權重初始化
        - VS-PINN 模式下會跳過 ManualScalingWrapper（避免雙重標準化）
        - 支援 Fourier features 消融實驗（use_fourier 開關）
    
    Examples:
        >>> config = {
        ...     'model': {'type': 'fourier_mlp', 'in_dim': 3, 'out_dim': 4,
        ...               'width': 200, 'depth': 8, 'activation': 'sine'},
        ...     'physics': {'type': 'vs_pinn_channel_flow', 'domain': {...}}
        ... }
        >>> model = create_model(config, device)
    """
    model_cfg = config.get('model')
    if not model_cfg:
        raise KeyError("Config missing required key: 'model'")
    
    # 驗證必要欄位
    required_fields = ['in_dim', 'out_dim', 'width', 'depth', 'activation']
    missing = [f for f in required_fields if f not in model_cfg]
    if missing:
        raise KeyError(f"Model config missing required fields: {missing}")
    
    # === 1. Fourier Features 配置 ===
    use_fourier = model_cfg.get('use_fourier', True)  # 默認啟用
    
    # === 2. VS-PINN 縮放因子提取 ===
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_pinn = (physics_type == 'vs_pinn_channel_flow')
    
    input_scale_factors = None
    fourier_normalize_input = False
    
    if is_vs_pinn and use_fourier:
        # VS-PINN 模式：提取縮放因子用於 Fourier 標準化修復
        vs_pinn_cfg = config.get('physics', {}).get('vs_pinn', {})
        scaling_cfg = vs_pinn_cfg.get('scaling_factors', {})
        N_x = scaling_cfg.get('N_x', 2.0)
        N_y = scaling_cfg.get('N_y', 12.0)
        N_z = scaling_cfg.get('N_z', 2.0)
        input_scale_factors = torch.tensor([N_x, N_y, N_z], dtype=torch.float32)
        fourier_normalize_input = True
        logging.info(
            f"🔧 VS-PINN + Fourier 修復啟用：縮放因子 N=[{N_x}, {N_y}, {N_z}]"
        )
    
    # === 3. 建立基礎模型 ===
    model_type = model_cfg.get('type', 'fourier_mlp')
    
    if model_type == 'enhanced_fourier_mlp':
        # 增強版 PINN（支援 RWF 等進階特性）
        base_model = create_enhanced_pinn(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=use_fourier,
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
            use_rwf=model_cfg.get('use_rwf', False),
            rwf_scale_std=model_cfg.get('rwf_scale_std', 0.1),
            fourier_normalize_input=fourier_normalize_input,
            input_scale_factors=input_scale_factors
        ).to(device)
        logging.info(f"Created Enhanced PINN (use_fourier={use_fourier})")
    else:
        # 基礎 PINN
        base_model = PINNNet(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=use_fourier,
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
            fourier_normalize_input=fourier_normalize_input,
            input_scale_factors=input_scale_factors
        ).to(device)
        logging.info(f"Created PINNNet (use_fourier={use_fourier})")
    
    # === 4. 應用手動標準化包裝器（若配置啟用且非 VS-PINN）===
    scaling_cfg = model_cfg.get('scaling', {})
    scaling_enabled = bool(scaling_cfg) and not is_vs_pinn
    
    if scaling_enabled:
        # 使用手動標準化包裝器（避免 Fourier feature 飽和）
        try:
            from pinnx.models.wrappers import ManualScalingWrapper
        except ImportError as exc:
            raise ImportError(
                "ManualScalingWrapper not found. "
                "Ensure pinnx.models.wrappers module exists."
            ) from exc
        
        # 提取輸入/輸出範圍
        input_scales, output_scales = _extract_scaling_ranges(
            config, statistics, model_cfg
        )
        
        # 驗證輸入範圍是否匹配域配置
        _validate_input_ranges(config, input_scales)
        
        # 應用包裝器
        model = ManualScalingWrapper(
            base_model,
            input_ranges=input_scales,
            output_ranges=output_scales
        ).to(device)
        logging.info(
            f"✅ Manual scaling wrapper applied:\n"
            f"   Inputs: {input_scales}\n"
            f"   Outputs: {output_scales}"
        )
    else:
        model = base_model
        if is_vs_pinn:
            logging.info("Using base model without scaling (VS-PINN handles scaling)")
        else:
            logging.info("Using base model without scaling")
    
    # === 5. SIREN 權重初始化（若使用 Sine 激活函數）===
    if model_cfg['activation'] == 'sine':
        target_model = base_model
        if hasattr(model, 'model'):  # 若有包裝器
            target_model = model.model  # type: ignore
        
        if isinstance(target_model, PINNNet):
            init_siren_weights(target_model)
            logging.info("✅ Applied SIREN weight initialization for Sine activation")
        else:
            logging.warning(
                f"⚠️  Cannot apply SIREN initialization: "
                f"base model type is {type(target_model)}"
            )
    
    # === 6. 訓練前驗證（若啟用標準化）===
    if scaling_enabled and hasattr(model, 'input_min'):
        _verify_model_scaling(model, config)
    
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Created model with {num_params:,} parameters")
    
    return model


def _extract_scaling_ranges(
    config: Dict[str, Any],
    statistics: Optional[Dict[str, Dict[str, float]]],
    model_cfg: Dict[str, Any]
) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    提取輸入/輸出標準化範圍（優先級：配置 > 統計 > 硬編碼）
    
    Returns:
        (input_scales, output_scales): 輸入與輸出的範圍字典
    """
    domain_cfg = config.get('physics', {}).get('domain', {})
    scaling_cfg = model_cfg.get('scaling', {})
    
    # === 輸入範圍 ===
    if domain_cfg and 'x_range' in domain_cfg:
        # 優先：從配置文件讀取完整域範圍
        input_x_range = tuple(domain_cfg['x_range'])  # type: ignore
        input_y_range = tuple(domain_cfg['y_range'])  # type: ignore
        logging.info(
            f"✅ Using domain ranges from config: "
            f"x={input_x_range}, y={input_y_range}"
        )
    elif statistics and 'x' in statistics and 'range' in statistics['x']:
        # 回退：使用感測點統計（可能導致泛化問題）
        input_x_range = tuple(statistics['x']['range'])  # type: ignore
        input_y_range = tuple(statistics['y']['range'])  # type: ignore
        logging.warning(
            f"⚠️ Falling back to statistics-based ranges: "
            f"x={input_x_range}, y={input_y_range}"
        )
        logging.warning(
            "⚠️ This may cause generalization issues "
            "if sensors don't cover full domain!"
        )
    else:
        # 最終回退：硬編碼（JHTDB Channel Re1000）
        input_x_range = (0.0, 25.13)
        input_y_range = (-1.0, 1.0)
        logging.warning(
            f"⚠️ Using hardcoded domain ranges: "
            f"x={input_x_range}, y={input_y_range}"
        )
    
    input_scales: Dict[str, Tuple[float, float]] = {
        'x': input_x_range,
        'y': input_y_range
    }
    
    # 3D 自動檢測
    if domain_cfg and 'z_range' in domain_cfg:
        input_z_range = tuple(domain_cfg['z_range'])  # type: ignore
        input_scales['z'] = input_z_range
        logging.info(f"✅ 3D mode: z={input_z_range}")
    elif statistics and 'z' in statistics and 'range' in statistics['z']:
        input_z_range = tuple(statistics['z']['range'])  # type: ignore
        input_scales['z'] = input_z_range
        logging.warning(f"⚠️ 3D mode using statistics: z={input_z_range}")
    
    # === 輸出範圍 ===
    output_scales: Dict[str, Tuple[float, float]] | None = None
    
    # 優先：從配置讀取
    if 'output_norm' in scaling_cfg:
        output_norm_raw = scaling_cfg['output_norm']
        
        if isinstance(output_norm_raw, dict):
            # 字典格式：直接使用
            output_scales = {
                'u': tuple(output_norm_raw.get('u', [0.0, 20.0])),  # type: ignore
                'v': tuple(output_norm_raw.get('v', [-1.0, 1.0])),  # type: ignore
                'w': tuple(output_norm_raw.get('w', [-5.0, 5.0])),  # type: ignore
                'p': tuple(output_norm_raw.get('p', [-100.0, 10.0]))  # type: ignore
            }
            logging.info("✅ Using output ranges from config file (dict format):")
            for key, val in output_scales.items():
                logging.info(f"   {key}: {val}")
        elif isinstance(output_norm_raw, str):
            # 字符串格式：回退到統計
            logging.info(
                f"⚠️ output_norm is string '{output_norm_raw}', "
                "falling back to statistics"
            )
    
    # 回退：使用統計資訊
    if output_scales is None:
        if statistics:
            output_u_range = tuple(statistics.get('u', {}).get('range', (0.0, 20.0)))  # type: ignore
            output_v_range = tuple(statistics.get('v', {}).get('range', (-1.0, 1.0)))  # type: ignore
            output_w_range = tuple(statistics.get('w', {}).get('range', (-5.0, 5.0)))  # type: ignore
            output_p_range = tuple(statistics.get('p', {}).get('range', (-100.0, 10.0)))  # type: ignore
            
            output_scales = {
                'u': output_u_range,
                'v': output_v_range,
                'w': output_w_range,
                'p': output_p_range
            }
            logging.info("✅ Using data-driven output ranges from statistics:")
            for key, val in output_scales.items():
                logging.info(f"   {key}: {val}")
        else:
            # 最終回退：硬編碼
            output_scales = {
                'u': (0.0, 20.0),
                'v': (-1.0, 1.0),
                'w': (-5.0, 5.0),
                'p': (-100.0, 10.0)
            }
            logging.warning(
                "⚠️ No statistics or config output_norm provided, "
                "using hardcoded ranges (may cause NaN)"
            )
    
    # 補充源項範圍（若需要）
    expected_out_dim = model_cfg.get('out_dim', 3)
    if expected_out_dim == 5 and len(output_scales) == 4:
        output_scales['S'] = (-1.0, 1.0)
        logging.info("✅ Added source term 'S' range: (-1.0, 1.0)")
    
    return input_scales, output_scales


def _validate_input_ranges(
    config: Dict[str, Any],
    input_scales: Dict[str, Tuple[float, float]]
) -> None:
    """
    驗證輸入範圍是否與域配置一致
    
    Raises:
        ValueError: 若範圍不匹配（會導致泛化失敗）
    """
    domain_cfg = config.get('physics', {}).get('domain', {})
    if not domain_cfg or 'x_range' not in domain_cfg:
        return  # 無配置可比對
    
    expected_x = tuple(domain_cfg['x_range'])
    expected_y = tuple(domain_cfg['y_range'])
    
    actual_x = input_scales.get('x')
    actual_y = input_scales.get('y')
    
    if actual_x is None or actual_y is None:
        raise ValueError("Input scales missing 'x' or 'y' range")
    
    # 容差檢查（1e-3）
    x_match = (abs(actual_x[0] - expected_x[0]) < 1e-3 and
               abs(actual_x[1] - expected_x[1]) < 1e-3)
    y_match = (abs(actual_y[0] - expected_y[0]) < 1e-3 and
               abs(actual_y[1] - expected_y[1]) < 1e-3)
    
    if not (x_match and y_match):
        raise ValueError(
            f"Input scaling range configuration error:\n"
            f"  Expected x: {expected_x}, got: {actual_x}\n"
            f"  Expected y: {expected_y}, got: {actual_y}\n"
            f"  This will cause generalization failure outside sensor coverage!"
        )
    
    # 3D 檢查
    if 'z_range' in domain_cfg:
        expected_z = tuple(domain_cfg['z_range'])
        actual_z = input_scales.get('z')
        if actual_z is None:
            raise ValueError("3D domain configured but 'z' range missing in input_scales")
        
        z_match = (abs(actual_z[0] - expected_z[0]) < 1e-3 and
                   abs(actual_z[1] - expected_z[1]) < 1e-3)
        if not z_match:
            raise ValueError(
                f"Z-axis scaling range configuration error:\n"
                f"  Expected: {expected_z}, got: {actual_z}"
            )


def _verify_model_scaling(model: nn.Module, config: Dict[str, Any]) -> None:
    """
    訓練前驗證：檢查模型縮放參數是否與配置一致
    
    Logs warnings if mismatches detected.
    """
    if not (hasattr(model, 'input_min') and hasattr(model, 'input_max')):
        return
    
    logging.info("=" * 60)
    logging.info("📐 Model Scaling Parameters Verification:")
    logging.info(f"   Input min:  {model.input_min.cpu().numpy()}")  # type: ignore
    logging.info(f"   Input max:  {model.input_max.cpu().numpy()}")  # type: ignore
    logging.info(f"   Output min: {model.output_min.cpu().numpy()}")  # type: ignore
    logging.info(f"   Output max: {model.output_max.cpu().numpy()}")  # type: ignore
    
    domain_cfg = config.get('physics', {}).get('domain', {})
    if domain_cfg and 'x_range' in domain_cfg:
        expected_x_range = domain_cfg['x_range']
        expected_y_range = domain_cfg['y_range']
        
        actual_x_min = model.input_min[0].item()  # type: ignore
        actual_x_max = model.input_max[0].item()  # type: ignore
        actual_y_min = model.input_min[1].item()  # type: ignore
        actual_y_max = model.input_max[1].item()  # type: ignore
        
        x_match = (abs(actual_x_min - expected_x_range[0]) < 1e-3 and
                   abs(actual_x_max - expected_x_range[1]) < 1e-3)
        y_match = (abs(actual_y_min - expected_y_range[0]) < 1e-3 and
                   abs(actual_y_max - expected_y_range[1]) < 1e-3)
        
        if x_match and y_match:
            logging.info(
                f"✅ Input ranges match config: "
                f"x={expected_x_range}, y={expected_y_range}"
            )
        else:
            logging.error("=" * 60)
            logging.error("❌ CRITICAL: Input range mismatch detected!")
            logging.error(
                f"   Expected x: {expected_x_range}, "
                f"got: [{actual_x_min:.4f}, {actual_x_max:.4f}]"
            )
            logging.error(
                f"   Expected y: {expected_y_range}, "
                f"got: [{actual_y_min:.4f}, {actual_y_max:.4f}]"
            )
            logging.error("=" * 60)
        
        # 3D 檢查
        if 'z_range' in domain_cfg and len(model.input_min) > 2:  # type: ignore
            expected_z_range = domain_cfg['z_range']
            actual_z_min = model.input_min[2].item()  # type: ignore
            actual_z_max = model.input_max[2].item()  # type: ignore
            z_match = (abs(actual_z_min - expected_z_range[0]) < 1e-3 and
                       abs(actual_z_max - expected_z_range[1]) < 1e-3)
            if z_match:
                logging.info(f"✅ 3D z-range matches config: {expected_z_range}")
            else:
                logging.error(
                    f"❌ Expected z: {expected_z_range}, "
                    f"got: [{actual_z_min:.4f}, {actual_z_max:.4f}]"
                )
    
    logging.info("=" * 60)


# ============================================================================
# 物理方程創建
# ============================================================================

def create_physics(config: Dict[str, Any], device: torch.device):
    """
    建立物理方程式模組（支援 VS-PINN 與標準 NS）
    
    Args:
        config: 完整配置字典，需包含：
            - physics: 物理配置（type, nu, rho, domain, vs_pinn, channel_flow）
            - [可選] losses: 損失配置（用於 warmup_epochs 等）
        device: 計算設備
    
    Returns:
        物理方程模組（已移至目標設備）
    
    Raises:
        KeyError: 若配置缺少必要欄位
        ValueError: 若物理類型不支援
    
    Supported Types:
        - 'vs_pinn_channel_flow': VS-PINN 通道流求解器（3D）
        - 'ns_2d': 標準 Navier-Stokes 2D 求解器
    
    Examples:
        >>> config = {
        ...     'physics': {
        ...         'type': 'vs_pinn_channel_flow',
        ...         'nu': 5e-5,
        ...         'domain': {'x_range': [0, 25.13], 'y_range': [-1, 1], ...},
        ...         'vs_pinn': {'scaling_factors': {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}}
        ...     }
        ... }
        >>> physics = create_physics(config, device)
    """
    physics_cfg = config.get('physics')
    if not physics_cfg:
        raise KeyError("Config missing required key: 'physics'")
    
    physics_type = physics_cfg.get('type', 'ns_2d')
    
    if physics_type == 'vs_pinn_channel_flow':
        # === VS-PINN 通道流求解器 ===
        vs_pinn_cfg = physics_cfg.get('vs_pinn', {})
        scaling_cfg = vs_pinn_cfg.get('scaling_factors', {})
        
        # 物理參數（兼容多種配置格式）
        channel_flow_cfg = physics_cfg.get('channel_flow', {})
        
        # 域配置
        domain_cfg = physics_cfg.get('domain', {})
        if not domain_cfg:
            raise ValueError(
                "VS-PINN requires 'domain' configuration with x/y/z_range"
            )
        
        domain_bounds = {
            'x': domain_cfg.get('x_range', [0.0, 25.13]),
            'y': domain_cfg.get('y_range', [-1.0, 1.0]),
            'z': domain_cfg.get('z_range', [0.0, 9.42]),
        }
        
        # 提取物理參數
        nu = physics_cfg.get('nu', channel_flow_cfg.get('u_tau', 5e-5))
        dP_dx = channel_flow_cfg.get('pressure_gradient',
                                      physics_cfg.get('dP_dx', 0.0025))
        rho = physics_cfg.get('rho', 1.0)
        
        physics = create_vs_pinn_channel_flow(
            N_x=scaling_cfg.get('N_x', 2.0),
            N_y=scaling_cfg.get('N_y', 12.0),
            N_z=scaling_cfg.get('N_z', 2.0),
            nu=nu,
            dP_dx=dP_dx,
            rho=rho,
            domain_bounds=domain_bounds,
            loss_config=config.get('losses', {}),
        )
        
        logging.info(
            f"✅ 使用 VS-PINN 求解器 ("
            f"N_x={scaling_cfg.get('N_x', 2.0)}, "
            f"N_y={scaling_cfg.get('N_y', 12.0)}, "
            f"N_z={scaling_cfg.get('N_z', 2.0)})"
        )
        logging.info(
            f"   域邊界: x={domain_bounds['x']}, "
            f"y={domain_bounds['y']}, z={domain_bounds['z']}"
        )
    
    elif physics_type == 'ns_2d':
        # === 標準 NS 2D 求解器 ===
        physics = NSEquations2D(
            viscosity=physics_cfg.get('nu', 1e-3),
            density=physics_cfg.get('rho', 1.0)
        )
        logging.info("✅ 使用標準 NS 2D 求解器")
    
    else:
        raise ValueError(
            f"Unsupported physics type: '{physics_type}'. "
            f"Supported types: 'vs_pinn_channel_flow', 'ns_2d'"
        )
    
    # 移至目標設備（僅對 nn.Module 子類有效）
    if isinstance(physics, nn.Module):
        physics = physics.to(device)
        logging.info(f"   Physics module moved to device: {device}")
    else:
        logging.info(f"   Physics object created (device handling in forward pass)")
    
    return physics


# ============================================================================
# 優化器與調度器創建
# ============================================================================

def create_optimizer(
    model: nn.Module,
    config: Dict[str, Any]
) -> Tuple[torch.optim.Optimizer, Optional[object]]:
    """
    建立優化器與學習率調度器
    
    Args:
        model: 待優化的模型
        config: 完整配置字典，需包含：
            - training.optimizer: 優化器配置（type, lr, weight_decay）
            - training.scheduler: [可選] 調度器配置（type, warmup_epochs, min_lr, ...）
            - training.epochs: [可選] 訓練總輪數（用於 Cosine 調度器）
    
    Returns:
        (optimizer, scheduler): 優化器與調度器（若未配置則為 None）
    
    Raises:
        ImportError: 若請求的優化器（如 SOAP）未安裝
        ValueError: 若配置參數無效
    
    Supported Optimizers:
        - 'adam': torch.optim.Adam（默認）
        - 'soap': SOAP（需安裝 torch_optimizer）
    
    Supported Schedulers:
        - 'warmup_cosine': Warmup + CosineAnnealing
        - 'cosine_warm_restarts': CosineAnnealingWarmRestarts（週期性重啟）
        - 'cosine': CosineAnnealingLR
        - 'exponential': ExponentialLR
        - 'none': 無調度器
    
    Examples:
        >>> config = {
        ...     'training': {
        ...         'optimizer': {'type': 'adam', 'lr': 1e-3, 'weight_decay': 0.0},
        ...         'scheduler': {'type': 'warmup_cosine', 'warmup_epochs': 10},
        ...         'epochs': 1000
        ...     }
        ... }
        >>> optimizer, scheduler = create_optimizer(model, config)
    """
    train_cfg = config.get('training', {})
    
    # === 優化器配置 ===
    optimizer_cfg = train_cfg.get('optimizer', {})
    lr = optimizer_cfg.get('lr', train_cfg.get('lr', 1e-3))
    weight_decay = optimizer_cfg.get('weight_decay',
                                      train_cfg.get('weight_decay', 0.0))
    optimizer_name = optimizer_cfg.get('type',
                                        train_cfg.get('optimizer', 'adam')).lower()
    
    if optimizer_name == 'soap':
        # 動態導入 SOAP（避免強制依賴）
        try:
            from torch_optimizer import SOAP  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Requested optimizer 'SOAP' but torch_optimizer is not installed. "
                "Install with: pip install torch-optimizer"
            ) from exc
        
        soap_kwargs = optimizer_cfg.get('soap', train_cfg.get('soap', {}))
        optimizer = SOAP(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **soap_kwargs
        )
        logging.info("✅ Using SOAP optimizer")
    
    elif optimizer_name == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        logging.info("✅ Using Adam optimizer")
    
    else:
        raise ValueError(
            f"Unsupported optimizer type: '{optimizer_name}'. "
            f"Supported types: 'adam', 'soap'"
        )
    
    # === 學習率調度器配置 ===
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type',
                                        train_cfg.get('lr_scheduler', 'none'))
    max_epochs = train_cfg.get('epochs', train_cfg.get('max_epochs', 1000))
    
    scheduler: Optional[object] = None
    
    if scheduler_type == 'warmup_cosine':
        # 動態導入 WarmupCosineScheduler（避免循環依賴）
        try:
            from pinnx.train.lr_scheduler import WarmupCosineScheduler  # type: ignore
        except ImportError:
            # 回退：使用 train.py 中定義的版本
            logging.warning(
                "pinnx.train.lr_scheduler not found, "
                "falling back to inline WarmupCosineScheduler"
            )
            # TODO: 將 WarmupCosineScheduler 提取為獨立模組
            raise ValueError(
                "WarmupCosineScheduler requires pinnx.train.lr_scheduler module. "
                "Please create it or use alternative scheduler types."
            )
        
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 10)
        min_lr = scheduler_cfg.get('min_lr', 1e-6)
        
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            base_lr=lr,
            min_lr=min_lr
        )
        logging.info(
            f"✅ Using WarmupCosineScheduler "
            f"(warmup={warmup_epochs}, max_epochs={max_epochs})"
        )
    
    elif scheduler_type == 'cosine_warm_restarts':
        T_0 = scheduler_cfg.get('T_0', 100)
        T_mult = scheduler_cfg.get('T_mult', 1)
        eta_min = scheduler_cfg.get('eta_min', 1e-6)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
        logging.info(
            f"✅ Using CosineAnnealingWarmRestarts "
            f"(T_0={T_0}, T_mult={T_mult})"
        )
    
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
        logging.info(f"✅ Using CosineAnnealingLR (T_max={max_epochs})")
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_cfg.get('gamma', 0.999)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        logging.info(f"✅ Using ExponentialLR (gamma={gamma})")
    
    elif scheduler_type in ['none', None]:
        logging.info("No learning rate scheduler configured")
    
    else:
        raise ValueError(
            f"Unsupported scheduler type: '{scheduler_type}'. "
            f"Supported types: 'warmup_cosine', 'cosine_warm_restarts', "
            f"'cosine', 'exponential', 'none'"
        )
    
    return optimizer, scheduler
