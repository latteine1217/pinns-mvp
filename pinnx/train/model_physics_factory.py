"""
模型/物理工廠模組（Registry Pattern）

本模組使用 Registry Pattern 提供統一的工廠函數，用於創建 PINN 訓練所需的核心組件：
- 設備選擇（CUDA/MPS/CPU）
- 模型架構（使用 @ModelFactory.register 註冊）
- 物理方程模組（使用 @PhysicsFactory.register 註冊）

注意：優化器與學習率調度器已遷移至 pinnx.train.factories 模組（Registry Pattern）
請使用：from pinnx.train.factories import create_optimizer, create_scheduler

Registry Pattern 優勢：
- 零條件分支（if-elif）
- 易於擴展（添加新類型只需裝飾器）
- 獨立測試（每個工廠函數可單獨測試）
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pinnx.models import PINNNet, create_pinn_model, init_siren_weights
from pinnx.physics.ns_2d import NSEquations2D
from pinnx.physics.vs_pinn_channel_flow import create_vs_pinn_channel_flow


# ============================================================================
# Registry Pattern 實現（含 Schema 驗證）
# ============================================================================

class ConfigSchema:
    """配置 Schema 定義（用於驗證必要欄位）"""
    
    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[Dict[str, Any]] = None,
        field_types: Optional[Dict[str, type]] = None,
        validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ):
        """
        Args:
            required_fields: 必要欄位列表
            optional_fields: 可選欄位及其預設值
            field_types: 欄位類型約束 {field_name: expected_type}
            validators: 自定義驗證函數 {field_name: validator_func}
        """
        self.required_fields = required_fields or []
        self.optional_fields = optional_fields or {}
        self.field_types = field_types or {}
        self.validators = validators or {}
    
    def validate(self, config: Dict[str, Any], context: str = "") -> None:
        """
        驗證配置是否符合 schema
        
        Args:
            config: 待驗證的配置字典
            context: 上下文訊息（用於錯誤報告）
        
        Raises:
            ValueError: 若配置不符合 schema
        """
        prefix = f"{context}: " if context else ""
        
        # 1. 檢查必要欄位
        missing = [f for f in self.required_fields if f not in config]
        if missing:
            raise ValueError(
                f"{prefix}Missing required fields: {missing}. "
                f"Required: {self.required_fields}"
            )
        
        # 2. 檢查類型約束
        for field, expected_type in self.field_types.items():
            if field in config:
                value = config[field]
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"{prefix}Field '{field}' has wrong type. "
                        f"Expected {expected_type.__name__}, got {type(value).__name__}"
                    )
        
        # 3. 執行自定義驗證
        for field, validator in self.validators.items():
            if field in config:
                value = config[field]
                try:
                    if not validator(value):
                        raise ValueError(
                            f"{prefix}Field '{field}' failed validation. "
                            f"Value: {value}"
                        )
                except Exception as e:
                    raise ValueError(
                        f"{prefix}Field '{field}' validation error: {e}"
                    )
    
    def apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """為缺少的可選欄位填充預設值"""
        result = dict(config)
        for field, default in self.optional_fields.items():
            if field not in result:
                result[field] = default
        return result


class _Registry:
    """通用註冊表（支援裝飾器註冊 + Schema 驗證）"""
    
    def __init__(self, name: str):
        self.name = name
        self._registry: Dict[str, Callable] = {}
        self._schemas: Dict[str, ConfigSchema] = {}
    
    def register(
        self,
        type_name: str,
        schema: Optional[ConfigSchema] = None,
        # 便捷參數（內部轉為 ConfigSchema）
        required_fields: Optional[List[str]] = None,
        optional_fields: Optional[Dict[str, Any]] = None,
        field_types: Optional[Dict[str, type]] = None,
        validators: Optional[Dict[str, Callable[[Any], bool]]] = None
    ):
        """
        裝飾器：註冊工廠函數（支援 Schema 驗證）
        
        Args:
            type_name: 類型名稱
            schema: 完整的 ConfigSchema 物件
            required_fields: 快捷參數 - 必要欄位列表
            optional_fields: 快捷參數 - 可選欄位及預設值
            field_types: 快捷參數 - 欄位類型約束
            validators: 快捷參數 - 自定義驗證函數
        
        Examples:
            # 方式 1: 使用完整 schema
            schema = ConfigSchema(required_fields=['in_dim', 'out_dim'])
            @registry.register('my_type', schema=schema)
            def create_my_type(...): ...
            
            # 方式 2: 使用快捷參數
            @registry.register('my_type', required_fields=['in_dim', 'out_dim'])
            def create_my_type(...): ...
        """
        def decorator(func: Callable) -> Callable:
            if type_name in self._registry:
                logging.warning(
                    f"⚠️  {self.name}: '{type_name}' already registered, overwriting"
                )
            
            self._registry[type_name] = func
            
            # 創建或保存 schema
            if schema is not None:
                self._schemas[type_name] = schema
            elif any([required_fields, optional_fields, field_types, validators]):
                self._schemas[type_name] = ConfigSchema(
                    required_fields=required_fields,
                    optional_fields=optional_fields,
                    field_types=field_types,
                    validators=validators
                )
            
            logging.debug(f"✅ {self.name}: registered '{type_name}'")
            return func
        return decorator
    
    def validate_config(self, type_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        驗證並標準化配置
        
        Args:
            type_name: 類型名稱
            config: 待驗證的配置
        
        Returns:
            標準化後的配置（已填充預設值）
        
        Raises:
            ValueError: 若配置不符合 schema
        """
        if type_name in self._schemas:
            schema = self._schemas[type_name]
            context = f"{self.name}[{type_name}]"
            
            # 驗證
            schema.validate(config, context)
            
            # 應用預設值
            return schema.apply_defaults(config)
        
        # 無 schema 則直接返回原配置
        return config
    
    def get(self, type_name: str) -> Callable:
        """獲取工廠函數"""
        if type_name not in self._registry:
            available = sorted(self._registry.keys())
            raise ValueError(
                f"{self.name}: Unknown type '{type_name}'. "
                f"Available types: {available}"
            )
        return self._registry[type_name]
    
    def list_available(self) -> List[str]:
        """列出所有已註冊的類型"""
        return sorted(self._registry.keys())
    
    def has_schema(self, type_name: str) -> bool:
        """檢查類型是否有 schema 定義"""
        return type_name in self._schemas


# 創建兩個獨立的註冊表
_model_factory = _Registry("ModelFactory")
_physics_factory = _Registry("PhysicsFactory")


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
# 模型創建（Registry Pattern）
# ============================================================================

# ===== 模型工廠函數註冊 =====

@_model_factory.register(
    'fourier_vs_mlp',
    required_fields=['in_dim', 'out_dim', 'width', 'depth', 'activation'],
    field_types={'in_dim': int, 'out_dim': int, 'width': int, 'depth': int, 'activation': str},
    validators={
        'in_dim': lambda x: x > 0,
        'out_dim': lambda x: x > 0,
        'width': lambda x: x > 0,
        'depth': lambda x: x > 0,
        'activation': lambda x: x in ['tanh', 'relu', 'sine', 'gelu', 'swish']
    }
)
def _create_fourier_vs_mlp_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """創建 Fourier-VS MLP 統一架構"""
    model_cfg = config['model']
    base_model = create_pinn_model(model_cfg).to(device)
    
    ff_cfg = model_cfg.get('fourier_features', {})
    use_fourier = ff_cfg.get('type') != 'disabled'
    logging.info(f"✅ Created Fourier-VS MLP (use_fourier={use_fourier})")
    
    return base_model


@_model_factory.register(
    'resnet',
    required_fields=['in_dim', 'out_dim', 'width', 'depth', 'activation'],
    field_types={'in_dim': int, 'out_dim': int, 'width': int, 'depth': int},
    validators={
        'in_dim': lambda x: x > 0,
        'out_dim': lambda x: x > 0,
        'width': lambda x: x > 0,
        'depth': lambda x: x >= 2  # ResNet 至少需要 2 層
    }
)
def _create_resnet_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """創建 ResNet 架構（使用 PINNNet block_type=resnet）"""
    model_cfg = config['model']
    
    res_cfg = dict(model_cfg)
    res_cfg['type'] = 'fourier_vs_mlp'
    res_cfg.setdefault('block_type', 'resnet')
    res_cfg.setdefault('res_block_alpha_init', 0.0)
    res_cfg.setdefault('use_input_projection', True)
    
    base_model = create_pinn_model(res_cfg).to(device)
    logging.info(
        f"✅ Created ResNet-style PINN via PINNNet "
        f"(block=resnet, depth={res_cfg.get('depth')})"
    )
    
    return base_model


@_model_factory.register(
    'piratenet',
    required_fields=['in_dim', 'out_dim', 'width', 'depth', 'activation'],
    field_types={'in_dim': int, 'out_dim': int, 'width': int, 'depth': int},
    validators={
        'in_dim': lambda x: x > 0,
        'out_dim': lambda x: x > 0,
        'width': lambda x: x > 0,
        'depth': lambda x: x >= 2  # PirateNet 至少需要 2 層
    }
)
def _create_piratenet_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """創建 PirateNet 架構（使用 PINNNet block_type=piratenet）"""
    model_cfg = config['model']
    
    pirate_cfg = dict(model_cfg)
    pirate_cfg['type'] = 'fourier_vs_mlp'
    pirate_cfg.setdefault('block_type', 'piratenet')
    pirate_cfg.setdefault('res_block_alpha_init', 0.0)
    pirate_cfg.setdefault('use_input_projection', True)
    
    base_model = create_pinn_model(pirate_cfg).to(device)
    logging.info(
        f"✅ Created PirateNet-style PINN via PINNNet "
        f"(block=piratenet, depth={pirate_cfg.get('depth')})"
    )
    
    return base_model


@_model_factory.register(
    'axis_selective_fourier_mlp',
    required_fields=['in_dim', 'out_dim', 'width', 'depth', 'activation', 'fourier_features'],
    field_types={'in_dim': int, 'out_dim': int, 'width': int, 'depth': int, 'fourier_features': dict},
    validators={
        'in_dim': lambda x: x > 0,
        'out_dim': lambda x: x > 0,
        'width': lambda x: x > 0,
        'depth': lambda x: x > 0,
        'fourier_features': lambda x: x.get('type') == 'axis_selective' and 'axes_config' in x
    }
)
def _create_axis_selective_fourier_mlp_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """創建軸向選擇性 Fourier MLP（詳細實現見 _create_axis_selective_model）"""
    base_model = _create_axis_selective_model(
        model_cfg=config['model'],
        config=config,
        device=device
    )
    logging.info("✅ Created Axis-Selective Fourier MLP")
    return base_model


# ===== 輔助函數（保留原有實現細節）=====

def _create_axis_selective_model(
    model_cfg: Dict[str, Any],
    config: Dict[str, Any],
    device: torch.device
) -> nn.Module:
    """
    創建使用軸向選擇性 Fourier Features 的 PINN 模型
    
    Args:
        model_cfg: 模型配置（包含 fourier_features 等）
        config: 完整配置（用於提取域配置）
        device: 計算設備
    
    Returns:
        AxisSelectiveFourierMLP 實例
    
    Raises:
        ImportError: 若軸向選擇性模組未找到
        ValueError: 若配置參數無效
    
    Notes:
        若啟用 Fourier 退火，模型將使用**初始階段頻率**創建，
        避免訓練開始時維度不匹配（TASK-007 修復）
    """
    # 導入軸向選擇性 Fourier 模組
    try:
        from pinnx.models.axis_selective_fourier import AxisSelectiveFourierFeatures
    except ImportError as exc:
        raise ImportError(
            "Cannot create axis_selective_fourier_mlp: "
            "pinnx.models.axis_selective_fourier module not found"
        ) from exc
    
    # 提取 Fourier Features 配置
    fourier_cfg = model_cfg.get('fourier_features', {})
    if not fourier_cfg or fourier_cfg.get('type') != 'axis_selective':
        raise ValueError(
            "axis_selective_fourier_mlp requires 'fourier_features' config "
            "with 'type': 'axis_selective'"
        )
    
    axes_config = fourier_cfg.get('axes_config')
    if not axes_config:
        raise ValueError(
            "axis_selective Fourier requires 'axes_config' in fourier_features"
        )
    
    # 🔧 TASK-007 Phase 2 修復：雙配置機制
    # 策略：
    # 1. 模型始終使用 full_axes_config 初始化（固定最大維度）
    # 2. 若啟用退火，傳遞 initial_axes_config 作為當前配置
    # 3. forward() 時根據當前配置應用掩碼（置零未啟用頻率）
    full_axes_config = axes_config  # 保存完整配置
    current_axes_config = axes_config  # 預設：當前配置 = 完整配置
    
    annealing_cfg = config.get('fourier_annealing', {})
    if annealing_cfg.get('enabled', False):
        try:
            from pinnx.train.fourier_annealing import (  # type: ignore
                create_channel_flow_annealing, 
                create_default_annealing
            )
        except ImportError as exc:
            raise ImportError(
                "Fourier annealing enabled but pinnx.train.fourier_annealing module not found"
            ) from exc
        
        strategy = annealing_cfg.get('strategy', 'channel_flow')
        initial_axes_config: Dict[str, list] = {}
        
        if strategy == 'channel_flow':
            # 通道流專用配置（每軸獨立階段）
            per_axis_stages = create_channel_flow_annealing()
            
            # 提取初始階段頻率（end_ratio 最小的階段）
            for axis, stages in per_axis_stages.items():
                if stages:
                    initial_stage = min(stages, key=lambda s: s.end_ratio)
                    initial_axes_config[axis] = initial_stage.frequencies
            
            logging.info("🔧 Fourier 退火啟用：使用通道流初始階段頻率")
            logging.info(f"   初始頻率（當前啟用）: {initial_axes_config}")
            logging.info(f"   完整頻率（模型容量）: {full_axes_config}")
            
        else:
            # 通用策略（全局配置）
            global_stages = create_default_annealing(strategy)
            if global_stages:
                initial_stage = min(global_stages, key=lambda s: s.end_ratio)
                initial_freqs = initial_stage.frequencies
                
                # 應用於所有軸
                initial_axes_config = {axis: initial_freqs for axis in axes_config.keys()}
                
                logging.info(f"🔧 Fourier 退火啟用：使用 '{strategy}' 策略初始階段頻率")
                logging.info(f"   初始頻率: {initial_freqs}")
        
        # 設定當前配置（訓練開始時的啟用頻率）
        if initial_axes_config:
            current_axes_config = initial_axes_config
            logging.info("✅ 將使用初始階段頻率作為當前配置（通過掩碼控制）")
        else:
            logging.warning("⚠️  無法提取初始階段頻率，使用完整配置")
    else:
        logging.info("ℹ️  Fourier 退火未啟用，使用配置文件中的完整頻率")
    
    # 從物理域配置提取域長度
    domain_cfg = config.get('physics', {}).get('domain', {})
    domain_lengths = None
    if domain_cfg:
        axes_names = list(axes_config.keys())
        domain_lengths = {}
        if 'x_range' in domain_cfg and 'x' in axes_names:
            x_range = domain_cfg['x_range']
            domain_lengths['x'] = x_range[1] - x_range[0]
        if 'y_range' in domain_cfg and 'y' in axes_names:
            y_range = domain_cfg['y_range']
            domain_lengths['y'] = y_range[1] - y_range[0]
        if 'z_range' in domain_cfg and 'z' in axes_names:
            z_range = domain_cfg['z_range']
            domain_lengths['z'] = z_range[1] - z_range[0]
        
        logging.info(f"  域長度: {domain_lengths}")
    
    # 創建軸向選擇性 Fourier Features
    # 🔧 TASK-007 Phase 2：雙配置機制
    # - full_axes_config：完整頻率（用於構建固定大小的 B 矩陣）
    # - current_axes_config：當前啟用頻率（用於掩碼控制）
    trainable = fourier_cfg.get('trainable', False)
    fourier_features = AxisSelectiveFourierFeatures(
        axes_config=current_axes_config,     # 當前啟用頻率
        domain_lengths=domain_lengths,
        trainable=trainable,
        full_axes_config=full_axes_config    # 完整頻率（固定維度）
    )
    
    # 構建完整網路（Fourier + MLP）
    # 使用與 PINNNet 相同的 MLP 架構
    from pinnx.models.fourier_mlp import DenseLayer
    
    width = model_cfg['width']
    depth = model_cfg['depth']
    out_dim = model_cfg['out_dim']
    activation = model_cfg['activation']
    
    use_residual = model_cfg.get('use_residual', False)
    use_layer_norm = model_cfg.get('use_layer_norm', False)
    dropout = model_cfg.get('dropout', 0.0)
    use_rwf = model_cfg.get('use_rwf', False)
    rwf_scale_std = model_cfg.get('rwf_scale_std', 0.1)
    sine_omega_0 = model_cfg.get('sine_omega_0', 1.0)
    
    # 創建一個包裝模型
    class AxisSelectiveFourierMLP(nn.Module):
        """軸向選擇性 Fourier MLP 模型"""
        def __init__(self):
            super().__init__()
            self.fourier_features = fourier_features
            
            # 計算 MLP 輸入維度（來自 Fourier Features）
            input_features = fourier_features.out_dim
            
            # 隱藏層
            layers = []
            for i in range(depth):
                layer_in_dim = input_features if i == 0 else width
                
                layers.append(DenseLayer(
                    layer_in_dim, width,
                    activation=activation,
                    use_residual=use_residual and i > 0,
                    use_layer_norm=use_layer_norm,
                    dropout=dropout,
                    use_rwf=use_rwf,
                    rwf_scale_std=rwf_scale_std,
                    sine_omega_0=sine_omega_0
                ))
            
            self.hidden_layers = nn.ModuleList(layers)
            
            # 輸出層
            self.output_layer = nn.Linear(width, out_dim)
            output_gain = 0.01 if use_residual else 0.1
            nn.init.xavier_normal_(self.output_layer.weight, gain=output_gain)
            nn.init.zeros_(self.output_layer.bias)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Fourier 特徵編碼
            h = self.fourier_features(x)
            
            # 🎯 DEBUG: 印出 Fourier Features 輸出形狀
            if not hasattr(self, '_debug_printed'):
                logging.info(f"🔍 [DEBUG] Fourier Features 輸出形狀: {h.shape}")
                logging.info(f"🔍 [DEBUG] 輸入形狀: {x.shape}")
                logging.info(f"🔍 [DEBUG] Fourier out_dim: {self.fourier_features.out_dim}")
                logging.info(f"🔍 [DEBUG] B 矩陣形狀: {self.fourier_features.B.shape}")
                self._debug_printed = True
            
            # MLP 層
            for layer in self.hidden_layers:
                h = layer(h)
            
            # 輸出
            return self.output_layer(h)
    
    model = AxisSelectiveFourierMLP().to(device)
    
    # 日誌輸出
    num_params = sum(p.numel() for p in model.parameters())
    logging.info(f"  Fourier 輸出維度: {fourier_features.out_dim}")
    logging.info(f"  MLP 結構: {depth}×{width}, 激活: {activation}")
    logging.info(f"  總參數量: {num_params:,}")
    
    return model


def create_model(
    config: Dict[str, Any],
    device: torch.device,
    statistics: Optional[Dict[str, Dict[str, float]]] = None
) -> nn.Module:
    """
    建立 PINN 模型（支援 Fourier features、VS-PINN 縮放、手動標準化、軸向選擇性 Fourier）
    
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
        - Fourier features 由 model.fourier_features.type 控制（'disabled' 代表關閉）
        - 支援軸向選擇性 Fourier Features（type='axis_selective_fourier_mlp'）
    
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
    ff_cfg = model_cfg.get('fourier_features')
    if not isinstance(ff_cfg, dict):
        raise KeyError("Model config missing required dict: 'fourier_features'")
    ff_type = ff_cfg.get('type')
    if ff_type not in {'standard', 'axis_selective', 'disabled'}:
        raise ValueError(
            "model.fourier_features.type must be 'standard' / 'axis_selective' / 'disabled'"
        )
    use_fourier = (ff_type != 'disabled')
    
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

    # 將 VS-PINN 縮放資訊透傳到模型配置（確保 Fourier 正確還原）
    model_cfg = dict(model_cfg)
    if is_vs_pinn and use_fourier:
        model_cfg.setdefault('input_scale_factors', input_scale_factors.tolist())  # type: ignore
        model_cfg.setdefault('fourier_normalize_input', True)
    
    # === 3. 建立基礎模型（Registry Pattern + Schema 驗證）===
    model_type = model_cfg.get('type', 'fourier_mlp')
    
    # 驗證配置（若有 schema）
    validated_model_cfg = _model_factory.validate_config(model_type, model_cfg)
    
    # 使用 Registry 創建模型（零條件分支）
    factory_fn = _model_factory.get(model_type)
    
    # 更新配置以使用驗證後的版本
    validated_config = dict(config)
    validated_config['model'] = validated_model_cfg
    
    base_model = factory_fn(validated_config, device, statistics)
    
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
# 物理方程創建（Registry Pattern）
# ============================================================================

# ===== Physics 工廠函數註冊 =====

@_physics_factory.register(
    'vs_pinn_channel_flow',
    required_fields=['domain'],
    field_types={'domain': dict, 'nu': (int, float), 'rho': (int, float)},
    validators={
        'domain': lambda x: all(k in x for k in ['x_range', 'y_range', 'z_range']),
        'nu': lambda x: x > 0 if isinstance(x, (int, float)) else True,
        'rho': lambda x: x > 0 if isinstance(x, (int, float)) else True
    }
)
def _create_vs_pinn_channel_flow_physics(
    config: Dict[str, Any],
    device: torch.device
):
    """創建 VS-PINN 通道流求解器"""
    physics_cfg = config['physics']
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
    
    # RANS 配置
    enable_rans = vs_pinn_cfg.get('enable_rans', False)
    rans_model = vs_pinn_cfg.get('rans_model', 'k_epsilon')
    
    # 梯度檢查點配置
    model_cfg = config.get('model', {})
    use_gradient_checkpointing = model_cfg.get('use_gradient_checkpointing', False)
    
    physics = create_vs_pinn_channel_flow(
        N_x=scaling_cfg.get('N_x', 2.0),
        N_y=scaling_cfg.get('N_y', 12.0),
        N_z=scaling_cfg.get('N_z', 2.0),
        nu=nu,
        dP_dx=dP_dx,
        rho=rho,
        domain_bounds=domain_bounds,
        loss_config=config.get('losses', {}),
        enable_rans=enable_rans,
        rans_model=rans_model,
        use_gradient_checkpointing=use_gradient_checkpointing,
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
    
    return physics


@_physics_factory.register(
    'ns_2d',
    optional_fields={'nu': 1e-3, 'rho': 1.0},
    field_types={'nu': (int, float), 'rho': (int, float)},
    validators={
        'nu': lambda x: x > 0,
        'rho': lambda x: x > 0
    }
)
def _create_ns_2d_physics(
    config: Dict[str, Any],
    device: torch.device
):
    """創建標準 NS 2D 求解器"""
    physics_cfg = config['physics']
    
    physics = NSEquations2D(
        viscosity=physics_cfg.get('nu', 1e-3),
        density=physics_cfg.get('rho', 1.0)
    )
    logging.info("✅ 使用標準 NS 2D 求解器")
    
    return physics


@_physics_factory.register(
    'kolmogorov_flow_2d',
    optional_fields={'nu': 0.01, 'rho': 1.0},
    field_types={'nu': (int, float), 'rho': (int, float), 'forcing': dict, 'domain': dict},
    validators={
        'nu': lambda x: x > 0,
        'rho': lambda x: x > 0
    }
)
def _create_kolmogorov_flow_2d_physics(
    config: Dict[str, Any],
    device: torch.device
):
    """創建 Kolmogorov Flow 2D 求解器"""
    from pinnx.physics.kolmogorov_flow_2d import KolmogorovFlow2D
    
    physics_cfg = config['physics']
    
    # 強迫參數
    forcing_cfg = physics_cfg.get('forcing', {})
    forcing_params = {
        'amplitude': forcing_cfg.get('amplitude', 1.0),
        'wavenumber': forcing_cfg.get('wavenumber', 4),
    }
    
    # 物理參數
    physics_params = {
        'nu': physics_cfg.get('nu', 0.01),
        'rho': physics_cfg.get('rho', 1.0),
    }
    
    # 域範圍
    domain_cfg = physics_cfg.get('domain', {})
    domain_bounds = {
        'x': (domain_cfg.get('x_min', 0.0), domain_cfg.get('x_max', 2*3.14159)),
        'y': (domain_cfg.get('y_min', 0.0), domain_cfg.get('y_max', 2*3.14159)),
    }
    
    # 損失配置
    loss_cfg = config.get('losses', {})
    
    physics = KolmogorovFlow2D(
        forcing_params=forcing_params,
        physics_params=physics_params,
        domain_bounds=domain_bounds,
        loss_config=loss_cfg
    )
    
    logging.info("✅ 使用 Kolmogorov Flow 2D 求解器")
    logging.info(f"   強迫參數: A={forcing_params['amplitude']:.3f}, k_f={forcing_params['wavenumber']}")
    logging.info(f"   物理參數: ν={physics_params['nu']:.2e}, ρ={physics_params['rho']:.1f}")
    
    return physics


# ===== 主工廠函數 =====

def create_physics(config: Dict[str, Any], device: torch.device):
    """
    建立物理方程式模組（使用 Registry Pattern）
    
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
        - 'kolmogorov_flow_2d': Kolmogorov 流 2D 求解器
    
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
    
    # 使用 Registry 創建物理模組（零條件分支）
    factory_fn = _physics_factory.get(physics_type)
    physics = factory_fn(config, device)
    
    # 移至目標設備（僅對 nn.Module 子類有效）
    if isinstance(physics, nn.Module):
        physics = physics.to(device)
        logging.info(f"   Physics module moved to device: {device}")
    else:
        logging.info(f"   Physics object created (device handling in forward pass)")
    
    return physics


# ============================================================================
# 公共 API：列出可用類型
# ============================================================================

def list_available_models() -> List[str]:
    """
    列出所有已註冊的模型類型
    
    Returns:
        List[str]: 已註冊的模型類型列表（排序）
    
    Examples:
        >>> models = list_available_models()
        >>> print(models)
        ['axis_selective_fourier_mlp', 'fourier_vs_mlp', 'piratenet', 'resnet']
    """
    return _model_factory.list_available()


def list_available_physics() -> List[str]:
    """
    列出所有已註冊的物理模組類型
    
    Returns:
        List[str]: 已註冊的物理類型列表（排序）
    
    Examples:
        >>> physics_types = list_available_physics()
        >>> print(physics_types)
        ['kolmogorov_flow_2d', 'ns_2d', 'vs_pinn_channel_flow']
    """
    return _physics_factory.list_available()


# ============================================================================
# 優化器與調度器創建
# ============================================================================

# ============================================================================
# 注意：create_optimizer 和 create_scheduler 已遷移至 pinnx.train.factories
# ============================================================================
# 
# 請使用以下方式導入：
#     from pinnx.train.factories import create_optimizer, create_scheduler
# 
# 新的 Registry Pattern 提供了更好的擴展性和可維護性。
# ============================================================================
