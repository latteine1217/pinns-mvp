"""
統一標準化系統模組 (Phase 1 Refactor - 2025-10-17)

提供統一的輸入/輸出標準化管理，消除冗餘定義，確保訓練與評估一致性。

核心設計原則：
1. 單一真相來源 (Single Source of Truth)
2. 明確的優先級 (checkpoint > config > computed)
3. Fail-fast 錯誤處理 (缺失必要參數時直接失敗)
4. VS-PINN 物理縮放獨立 (保留在 scaling.py/NonDimensionalizer)

主要類別：
- UnifiedNormalizer: 統一標準化器（輸入 + 輸出）
- InputTransform: 坐標標準化 (內部組件)
- OutputTransform: 變量標準化 (內部組件)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Union, Any
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# 配置數據類
# ===================================================================

@dataclass
class InputNormConfig:
    """輸入標準化配置"""
    norm_type: str = "none"  # none, standard, minmax, channel_flow
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    bounds: Optional[torch.Tensor] = None  # shape [dim, 2]


@dataclass
class OutputNormConfig:
    """輸出標準化配置"""
    norm_type: str = "none"  # none, training_data_norm, friction_velocity, manual
    variable_order: Optional[List[str]] = None  # 變量順序（單一來源）
    means: Optional[Dict[str, float]] = None
    stds: Optional[Dict[str, float]] = None
    params: Optional[Dict[str, Any]] = None  # 額外參數


# ===================================================================
# 輸入標準化 (坐標)
# ===================================================================

class InputTransform:
    """
    坐標標準化器（內部組件）
    
    支援類型：
    - none/identity: 不處理
    - standard: Z-score (x - mean) / std
    - minmax: 線性映射到 feature_range
    - channel_flow: 使用預定義 bounds 映射到 feature_range
    """
    
    def __init__(self, config: InputNormConfig):
        self.norm_type = (config.norm_type or "none").lower()
        self.feature_range = config.feature_range
        self.bounds = config.bounds
        
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.data_min: Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None
    
    def fit(self, samples: torch.Tensor) -> "InputTransform":
        """從樣本數據擬合統計量"""
        if self.norm_type == "standard":
            mean = torch.mean(samples, dim=0, keepdim=True)
            std = torch.std(samples, dim=0, keepdim=True)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            self.mean = mean
            self.std = std
            
        elif self.norm_type == "minmax":
            data_min = torch.min(samples, dim=0, keepdim=True)[0]
            data_max = torch.max(samples, dim=0, keepdim=True)[0]
            data_range = data_max - data_min
            data_range = torch.where(data_range < 1e-8, torch.ones_like(data_range), data_range)
            self.data_min = data_min
            self.data_range = data_range
            
        elif self.norm_type in ("channel_flow", "vs_pinn"):
            # 使用預定義 bounds，無需擬合
            pass
        else:
            # none/identity
            pass
        
        return self
    
    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """標準化轉換"""
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer 尚未擬合")
            return (tensor - self.mean) / self.std
        
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer 尚未擬合")
            norm = (tensor - self.data_min) / self.data_range
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds 未提供")
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            denom = torch.where((maxs - mins) < 1e-8, torch.ones_like(maxs - mins), maxs - mins)
            norm = (tensor - mins) / denom
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        
        raise ValueError(f"不支援的標準化類型: {self.norm_type}")
    
    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """逆標準化"""
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer 尚未擬合")
            return tensor * self.std + self.mean
        
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer 尚未擬合")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            return norm * self.data_range + self.data_min
        
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds 未提供")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            return norm * (maxs - mins) + mins
        
        raise ValueError(f"不支援的標準化類型: {self.norm_type}")
    
    def to(self, device: torch.device) -> "InputTransform":
        """移動到指定設備"""
        if self.mean is not None:
            self.mean = self.mean.to(device)
        if self.std is not None:
            self.std = self.std.to(device)
        if self.data_min is not None:
            self.data_min = self.data_min.to(device)
        if self.data_range is not None:
            self.data_range = self.data_range.to(device)
        if self.bounds is not None:
            self.bounds = self.bounds.to(device)
        return self
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取元數據（用於 checkpoint）"""
        if self.mean is not None:
            device = self.mean.device
        elif self.data_min is not None:
            device = self.data_min.device
        elif self.bounds is not None:
            device = self.bounds.device
        else:
            device = torch.device('cpu')
        
        feature_range_tensor = torch.tensor(
            self.feature_range, dtype=torch.float32, device=device
        )
        
        metadata: Dict[str, Any] = {
            'norm_type': self.norm_type,
            'feature_range': feature_range_tensor
        }
        
        if self.mean is not None:
            metadata['mean'] = self.mean.clone().detach()
        if self.std is not None:
            metadata['std'] = self.std.clone().detach()
        if self.data_min is not None:
            metadata['data_min'] = self.data_min.clone().detach()
        if self.data_range is not None:
            metadata['data_range'] = self.data_range.clone().detach()
        if self.bounds is not None:
            metadata['bounds'] = self.bounds.clone().detach()
        
        return metadata


# ===================================================================
# 輸出標準化 (變量)
# ===================================================================

class OutputTransform:
    """
    變量標準化器（內部組件）
    
    支援類型：
    - none: 不處理
    - training_data_norm: Z-score 標準化（從訓練資料計算）
    - friction_velocity: 基於摩擦速度縮放
    - manual: 手動指定 means/stds
    """
    
    SUPPORTED_TYPES = ['none', 'training_data_norm', 'friction_velocity', 'manual']
    DEFAULT_VAR_ORDER = ['u', 'v', 'w', 'p', 'S']
    
    def __init__(self, config: OutputNormConfig):
        if config.norm_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支援的標準化類型: {config.norm_type}。支援: {self.SUPPORTED_TYPES}"
            )
        
        self.norm_type = config.norm_type
        self.variable_order = config.variable_order or self.DEFAULT_VAR_ORDER.copy()
        self.means = config.means or {}
        self.stds = config.stds or {}
        self.params = config.params or {}
        
        logger.info(f"✅ OutputTransform 初始化: type={self.norm_type}, variables={self.variable_order}")
    
    @classmethod
    def from_data(
        cls,
        data: Dict[str, Union[np.ndarray, torch.Tensor]],
        norm_type: str = 'training_data_norm',
        variable_order: Optional[List[str]] = None
    ) -> 'OutputTransform':
        """
        從資料自動計算標準化係數（僅 training_data_norm 模式）
        
        Args:
            data: 資料字典 {'u': array, 'v': array, ...}
            norm_type: 標準化類型
            variable_order: 變量順序（若為 None 則從 data.keys() 推斷）
        """
        if norm_type != 'training_data_norm':
            raise ValueError(f"from_data 僅支援 training_data_norm，當前為: {norm_type}")
        
        means = {}
        stds = {}
        valid_vars = []  # 🛡️ 追蹤有效變量（排除空張量）
        
        for var_name, values in data.items():
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            
            # ⚠️ 跳過空張量或零長度陣列（防止 NaN）
            if values.size == 0:
                logger.info(f"⏭️  {var_name} 為空張量，跳過標準化統計量計算")
                continue
            
            mean = float(np.mean(values))
            std = float(np.std(values))
            
            # 🛡️ 防禦性檢查：拒絕 NaN 或 Inf
            if not np.isfinite(mean) or not np.isfinite(std):
                logger.warning(f"⚠️  {var_name} 的統計量包含 NaN/Inf (mean={mean}, std={std})，跳過")
                continue
            
            if abs(std) < 1e-10:
                logger.warning(f"⚠️  {var_name} 的標準差接近零，設為 1.0")
                std = 1.0
            
            means[var_name] = mean
            stds[var_name] = std
            valid_vars.append(var_name)
        
        if variable_order is None:
            variable_order = valid_vars  # 🔧 使用有效變量列表而非所有鍵
        
        params = {'source': 'auto_computed_from_data'}
        
        config = OutputNormConfig(
            norm_type=norm_type,
            variable_order=variable_order,
            means=means,
            stds=stds,
            params=params
        )
        
        logger.info(f"✅ 從資料計算 Z-score 係數: means={means}, stds={stds}")
        return cls(config)
    
    @classmethod
    def from_metadata(cls, metadata: Dict) -> 'OutputTransform':
        """
        從 checkpoint metadata 快速重建 OutputTransform
        
        這是一個便利方法，簡化從保存的 checkpoint 中恢復標準化器的流程。
        
        Args:
            metadata: checkpoint 的 'normalization' 字段，應包含：
                - norm_type: 標準化類型 (str)
                - variable_order: 變量順序列表 (List[str])
                - means: 均值字典 (Dict[str, float])
                - stds: 標準差字典 (Dict[str, float])
                - params: 其他參數 (Dict, 可選)
        
        Returns:
            OutputTransform 實例
        
        Example:
            >>> checkpoint = torch.load('model.pth')
            >>> normalizer = OutputTransform.from_metadata(checkpoint['normalization'])
            >>> predictions = model(x)
            >>> denormalized = normalizer.denormalize_batch(predictions, var_order=['u', 'v', 'p'])
        
        Raises:
            KeyError: 若 metadata 缺少必要欄位
            ValueError: 若 metadata 格式不正確
        """
        # 驗證必要欄位
        required_fields = ['norm_type', 'means', 'stds']
        missing_fields = [f for f in required_fields if f not in metadata]
        if missing_fields:
            raise KeyError(
                f"metadata 缺少必要欄位: {missing_fields}. "
                f"需要: {required_fields}, 得到: {list(metadata.keys())}"
            )
        
        # 建立 OutputNormConfig
        config = OutputNormConfig(
            norm_type=metadata['norm_type'],
            variable_order=metadata.get('variable_order', cls.DEFAULT_VAR_ORDER.copy()),
            means=metadata['means'],
            stds=metadata['stds'],
            params=metadata.get('params', {})
        )
        
        logger.info(
            f"✅ 從 metadata 重建 OutputTransform: "
            f"type={config.norm_type}, variables={config.variable_order}"
        )
        return cls(config)
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        training_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> 'OutputTransform':
        """
        從配置創建 OutputTransform（向後兼容 Trainer）
        
        Args:
            config: 完整配置字典
            training_data: 訓練資料（用於 training_data_norm 模式）
        
        Returns:
            OutputTransform 實例
        """
        norm_config = config.get('normalization', {})
        norm_type = norm_config.get('type', 'none')
        params = norm_config.get('params', {})
        variable_order = norm_config.get('variable_order', cls.DEFAULT_VAR_ORDER.copy())
        
        # 根據類型提取標準化係數
        if norm_type == 'training_data_norm':
            means, stds = cls._extract_training_data_scales(params, training_data, config)
        elif norm_type == 'friction_velocity':
            means, stds = cls._extract_friction_velocity_scales(params, config)
        elif norm_type == 'manual':
            means = {k.replace('_mean', ''): v for k, v in params.items() if k.endswith('_mean')}
            stds = {k.replace('_std', ''): v for k, v in params.items() if k.endswith('_std')}
        else:  # 'none'
            means = {}
            stds = {}
        
        output_config = OutputNormConfig(
            norm_type=norm_type,
            variable_order=variable_order,
            means=means,
            stds=stds,
            params=params
        )
        
        return cls(output_config)
    
    @staticmethod
    def _extract_training_data_scales(
        params: Dict,
        training_data: Optional[Dict[str, torch.Tensor]],
        config: Dict
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """從訓練資料或配置中提取標準化係數（內部輔助方法）"""
        # 優先級 1: 從配置中明確提供
        if all(k in params for k in ['u_mean', 'u_std', 'v_mean', 'v_std', 'w_mean', 'w_std', 'p_mean', 'p_std']):
            means = {
                'u': params['u_mean'],
                'v': params['v_mean'],
                'w': params['w_mean'],
                'p': params['p_mean']
            }
            stds = {
                'u': params['u_std'],
                'v': params['v_std'],
                'w': params['w_std'],
                'p': params['p_std']
            }
            logger.info("📐 使用配置中的標準化係數")
            return means, stds
        
        # 優先級 2: 從訓練資料計算
        if training_data is not None:
            means = {}
            stds = {}
            for var_name in ['u', 'v', 'w', 'p']:
                # 支援兩種鍵名格式：'u' 或 'u_sensors'
                key = var_name if var_name in training_data else f'{var_name}_sensors'
                
                if key in training_data:
                    values = training_data[key]
                    if isinstance(values, torch.Tensor):
                        values = values.detach().cpu().numpy()
                    
                    # ⚠️ 跳過空張量或零長度陣列（防止 NaN）
                    if values.size == 0:
                        logger.info(f"⏭️  {var_name} 為空張量，跳過標準化統計量計算")
                        continue
                    
                    mean = float(np.mean(values))
                    std = float(np.std(values))
                    
                    # 🛡️ 防禦性檢查：拒絕 NaN 或 Inf
                    if not np.isfinite(mean) or not np.isfinite(std):
                        logger.warning(f"⚠️  {var_name} 的統計量包含 NaN/Inf (mean={mean}, std={std})，跳過")
                        continue
                    
                    if abs(std) < 1e-10:
                        logger.warning(f"⚠️  {var_name} 的標準差接近零，設為 1.0")
                        std = 1.0
                    
                    means[var_name] = mean
                    stds[var_name] = std
            
            if means:
                logger.info(f"📐 從訓練資料計算標準化係數: {list(means.keys())}")
                return means, stds
        
        # 優先級 3: 失敗
        raise ValueError(
            "training_data_norm 模式需要提供標準化係數！\n"
            "請在配置中提供 normalization.params (u_mean, u_std, ...)\n"
            "或傳入 training_data 以自動計算。"
        )
    
    @staticmethod
    def _extract_friction_velocity_scales(
        params: Dict,
        config: Dict
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """從物理參數計算 friction_velocity 標準化係數（內部輔助方法）"""
        u_tau = params.get('u_tau')
        if u_tau is None and 'physics' in config:
            physics = config['physics']
            if 'channel_flow' in physics:
                u_tau = physics['channel_flow'].get('u_tau', 1.0)
            else:
                u_tau = 1.0
        
        if u_tau is None:
            raise ValueError("friction_velocity 模式需要提供 u_tau 參數")
        
        rho = params.get('rho')
        if rho is None and 'physics' in config:
            rho = config['physics'].get('rho', 1.0)
        if rho is None:
            rho = 1.0
        
        velocity_scale = u_tau
        pressure_scale = rho * u_tau ** 2
        
        means = {'u': 0.0, 'v': 0.0, 'w': 0.0, 'p': 0.0}
        stds = {
            'u': velocity_scale,
            'v': velocity_scale,
            'w': velocity_scale,
            'p': pressure_scale
        }
        
        logger.info(f"📐 Friction velocity scales: u_τ={u_tau}, ρ={rho}")
        return means, stds
    
    def normalize(
        self,
        value: Union[np.ndarray, torch.Tensor, float],
        var_name: str
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """標準化單個變量"""
        if self.norm_type == 'none':
            return value
        
        if var_name not in self.stds:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過標準化")
            return value
        
        mean = self.means.get(var_name, 0.0)
        std = self.stds[var_name]
        
        return (value - mean) / std
    
    def denormalize(
        self,
        value: Union[np.ndarray, torch.Tensor, float],
        var_name: str
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """反標準化單個變量"""
        if self.norm_type == 'none':
            return value
        
        if var_name not in self.stds:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過反標準化")
            return value
        
        mean = self.means.get(var_name, 0.0)
        std = self.stds[var_name]
        
        return value * std + mean
    
    def normalize_batch(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """批次標準化（用於訓練）"""
        if self.norm_type == 'none':
            return predictions
        
        if var_order is None:
            var_order = self.variable_order
        
        is_torch = isinstance(predictions, torch.Tensor)
        
        if is_torch:
            # 🔧 修正：使用 detach().clone() 僅在不需要梯度時才複製
            # 若 predictions.requires_grad=True，則需要保持計算圖連接
            if predictions.requires_grad:
                # 保持梯度追蹤：直接在原張量上操作（創建新視圖）
                result = []
                for i, var_name in enumerate(var_order):
                    if i >= predictions.shape[-1]:
                        break
                    col = predictions[:, i:i+1]  # 保持維度 [batch, 1]
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        col = (col - mean) / std
                    result.append(col)
                return torch.cat(result, dim=1)
            else:
                # 不需要梯度：可以安全 clone
                result = predictions.clone()
                for i, var_name in enumerate(var_order):
                    if i >= result.shape[-1]:
                        break
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        result[:, i] = (result[:, i] - mean) / std
                return result
        else:
            result = predictions.copy()
            for i, var_name in enumerate(var_order):
                if i >= result.shape[-1]:
                    break
                if var_name in self.stds:
                    mean = self.means.get(var_name, 0.0)
                    std = self.stds[var_name]
                    result[:, i] = (result[:, i] - mean) / std
            return result
    
    def denormalize_batch(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """批次反標準化（用於評估）
        
        🔧 修正：保持梯度追蹤能力
        - 若 predictions.requires_grad=True，則使用 torch.cat 保持計算圖
        - 否則使用 clone() 提升效率
        """
        if self.norm_type == 'none':
            return predictions
        
        if var_order is None:
            var_order = self.variable_order
        
        is_torch = isinstance(predictions, torch.Tensor)
        
        if is_torch:
            # 🔧 修正：根據是否需要梯度選擇不同策略
            if predictions.requires_grad:
                # 保持梯度追蹤：使用 torch.cat 構建新張量
                result = []
                for i, var_name in enumerate(var_order):
                    if i >= predictions.shape[-1]:
                        break
                    col = predictions[:, i:i+1]  # 保持維度 [batch, 1]
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        col = col * std + mean
                    result.append(col)
                return torch.cat(result, dim=1)
            else:
                # 不需要梯度：使用 clone() 提升效率
                result = predictions.clone()
                for i, var_name in enumerate(var_order):
                    if i >= result.shape[-1]:
                        break
                    if var_name in self.stds:
                        mean = self.means.get(var_name, 0.0)
                        std = self.stds[var_name]
                        result[:, i] = result[:, i] * std + mean
                return result
        else:
            result = predictions.copy()
            for i, var_name in enumerate(var_order):
                if i >= result.shape[-1]:
                    break
                if var_name in self.stds:
                    mean = self.means.get(var_name, 0.0)
                    std = self.stds[var_name]
                    result[:, i] = result[:, i] * std + mean
            return result
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取元數據（用於 checkpoint）"""
        return {
            'norm_type': self.norm_type,
            'variable_order': self.variable_order.copy(),
            'means': self.means.copy(),
            'stds': self.stds.copy(),
            'params': self.params.copy()
        }


# ===================================================================
# 統一標準化器
# ===================================================================

class UnifiedNormalizer:
    """
    統一標準化器：管理輸入（坐標）與輸出（變量）的標準化
    
    設計原則：
    1. 單一真相來源：variable_order 僅定義一次
    2. 明確優先級：checkpoint > config > computed
    3. Fail-fast：缺失必要參數時直接失敗（除非 training_data_norm 模式允許計算）
    
    使用範例：
        >>> # 從配置創建
        >>> normalizer = UnifiedNormalizer.from_config(config, training_data)
        >>> 
        >>> # 訓練時標準化
        >>> coords_norm = normalizer.transform_input(coords)
        >>> outputs_norm = normalizer.transform_output(outputs)
        >>> 
        >>> # 評估時反標準化
        >>> predictions_phys = normalizer.inverse_transform_output(predictions_norm)
        >>> 
        >>> # 保存到 checkpoint
        >>> metadata = normalizer.get_metadata()
        >>> torch.save({'model': ..., 'normalization': metadata}, 'ckpt.pth')
    """
    
    def __init__(
        self,
        input_transform: InputTransform,
        output_transform: OutputTransform
    ):
        self.input_transform = input_transform
        self.output_transform = output_transform
        
        logger.info(f"✅ UnifiedNormalizer 初始化完成")
        logger.info(f"   輸入: {self.input_transform.norm_type}")
        logger.info(f"   輸出: {self.output_transform.norm_type}")
        logger.info(f"   變量順序: {self.output_transform.variable_order}")
    
    @classmethod
    def from_config(
        cls,
        config: Dict,
        training_data: Optional[Dict[str, torch.Tensor]] = None,
        device: torch.device = torch.device('cpu')
    ) -> 'UnifiedNormalizer':
        """
        從配置創建標準化器
        
        Args:
            config: 配置字典（需包含 'normalization' 段落）
            training_data: 訓練資料（用於計算統計量，僅 training_data_norm 模式）
            device: 設備
        
        Returns:
            UnifiedNormalizer 實例
        """
        # === 輸入標準化配置 ===
        scaling_cfg = config.get('model', {}).get('scaling', {})
        input_norm_type = scaling_cfg.get('input_norm', 'none')
        feature_range = tuple(scaling_cfg.get('input_norm_range', [-1.0, 1.0]))
        
        # 獲取 bounds (channel_flow 模式需要)
        bounds_tensor = None
        if input_norm_type == 'channel_flow':
            domain = config.get('physics', {}).get('domain', {})
            bounds: List[Tuple[float, float]] = []
            for axis in ['x', 'y', 'z']:
                rng = domain.get(f'{axis}_range')
                if rng is not None:
                    bounds.append((float(rng[0]), float(rng[1])))
            if bounds:
                bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)
        
        input_config = InputNormConfig(
            norm_type=input_norm_type,
            feature_range=(float(feature_range[0]), float(feature_range[1])),
            bounds=bounds_tensor
        )
        input_transform = InputTransform(input_config)
        
        # 從訓練資料擬合輸入統計量
        if training_data is not None:
            coord_tensors = cls._collect_coordinate_tensors(training_data)
            if coord_tensors:
                samples = torch.cat(coord_tensors, dim=0)
                if input_transform.bounds is not None and input_transform.bounds.shape[0] > samples.shape[1]:
                    input_transform.bounds = input_transform.bounds[:samples.shape[1], :]
                input_transform.fit(samples)
        
        input_transform.to(device)
        
        # === 輸出標準化配置 ===
        if 'normalization' not in config:
            logger.warning("⚠️  配置中未找到 'normalization' 段落，使用默認 (type='none')")
            output_config = OutputNormConfig(norm_type='none')
            output_transform = OutputTransform(output_config)
        else:
            norm_cfg = config['normalization']
            norm_type = norm_cfg.get('type', 'none')
            params = norm_cfg.get('params', {})
            
            # 變量順序：優先從配置讀取，否則從資料推斷
            variable_order = norm_cfg.get('variable_order')
            if variable_order is None and training_data is not None:
                # 從訓練資料的 keys 推斷（按預設順序排序）
                # 🛡️ 過濾掉空張量（只保留有效資料的變量）
                data_vars = []
                for k in training_data.keys():
                    if k in OutputTransform.DEFAULT_VAR_ORDER:
                        val = training_data[k]
                        # 檢查是否為空張量
                        if isinstance(val, torch.Tensor) and val.numel() == 0:
                            continue
                        elif isinstance(val, np.ndarray) and val.size == 0:
                            continue
                        data_vars.append(k)
                
                if data_vars:
                    variable_order = sorted(data_vars, key=lambda x: OutputTransform.DEFAULT_VAR_ORDER.index(x))
                    logger.info(f"📋 從資料推斷變量順序（已過濾空張量）: {variable_order}")
            
            # 根據類型提取標準化係數
            if norm_type == 'training_data_norm':
                means, stds = OutputTransform._extract_training_data_scales(params, training_data, config)
            elif norm_type == 'friction_velocity':
                means, stds = OutputTransform._extract_friction_velocity_scales(params, config)
            elif norm_type == 'manual':
                means = {k.replace('_mean', ''): v for k, v in params.items() if k.endswith('_mean')}
                stds = {k.replace('_std', ''): v for k, v in params.items() if k.endswith('_std')}
            else:  # 'none'
                means = {}
                stds = {}
            
            output_config = OutputNormConfig(
                norm_type=norm_type,
                variable_order=variable_order,
                means=means,
                stds=stds,
                params=params
            )
            output_transform = OutputTransform(output_config)
        
        return cls(input_transform, output_transform)
    
    @classmethod
    def from_metadata(cls, metadata: Dict) -> 'UnifiedNormalizer':
        """
        從 checkpoint metadata 恢復標準化器
        
        Args:
            metadata: {'input': dict, 'output': dict}
        """
        input_meta = metadata.get('input', {})
        output_meta = metadata.get('output', {})
        
        # 重建 InputTransform
        input_config = InputNormConfig(
            norm_type=input_meta.get('norm_type', 'none'),
            feature_range=tuple(input_meta.get('feature_range', (-1.0, 1.0))),
            bounds=input_meta.get('bounds')
        )
        input_transform = InputTransform(input_config)
        
        # 恢復統計量
        if 'mean' in input_meta:
            input_transform.mean = input_meta['mean']
        if 'std' in input_meta:
            input_transform.std = input_meta['std']
        if 'data_min' in input_meta:
            input_transform.data_min = input_meta['data_min']
        if 'data_range' in input_meta:
            input_transform.data_range = input_meta['data_range']
        
        # 重建 OutputTransform
        output_config = OutputNormConfig(
            norm_type=output_meta.get('norm_type', 'none'),
            variable_order=output_meta.get('variable_order', OutputTransform.DEFAULT_VAR_ORDER.copy()),
            means=output_meta.get('means', {}),
            stds=output_meta.get('stds', {}),
            params=output_meta.get('params', {})
        )
        output_transform = OutputTransform(output_config)
        
        logger.info(f"🔄 從 checkpoint 恢復 UnifiedNormalizer")
        return cls(input_transform, output_transform)
    
    # === 輸入標準化接口 ===
    
    def transform_input(self, coords: torch.Tensor) -> torch.Tensor:
        """標準化輸入坐標"""
        return self.input_transform.transform(coords)
    
    def inverse_transform_input(self, coords: torch.Tensor) -> torch.Tensor:
        """反標準化輸入坐標"""
        return self.input_transform.inverse_transform(coords)
    
    # === 輸出標準化接口 ===
    
    def transform_output(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """標準化輸出變量（批次）"""
        return self.output_transform.normalize_batch(predictions, var_order)
    
    def inverse_transform_output(
        self,
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[List[str]] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """反標準化輸出變量（批次）"""
        return self.output_transform.denormalize_batch(predictions, var_order)
    
    # === 元數據管理 ===
    
    def get_metadata(self) -> Dict[str, Any]:
        """獲取完整元數據（用於 checkpoint）"""
        return {
            'input': self.input_transform.get_metadata(),
            'output': self.output_transform.get_metadata()
        }
    
    @property
    def variable_order(self) -> List[str]:
        """獲取變量順序（單一來源）"""
        return self.output_transform.variable_order
    
    def to(self, device: torch.device) -> 'UnifiedNormalizer':
        """移動到指定設備"""
        self.input_transform.to(device)
        return self
    
    # === 內部工具函數 ===
    
    @staticmethod
    def _collect_coordinate_tensors(training_data: Dict) -> List[torch.Tensor]:
        """從訓練資料收集坐標張量"""
        coord_tensors = []
        for key in ['coords', 'boundary_coords', 'pde_coords', 'sensor_coords']:
            if key in training_data:
                val = training_data[key]
                if isinstance(val, torch.Tensor):
                    coord_tensors.append(val)
                elif isinstance(val, np.ndarray):
                    coord_tensors.append(torch.from_numpy(val).float())
        return coord_tensors


# ===================================================================
# 向後兼容：保留舊接口
# ===================================================================

# 為了不破壞現有代碼，提供別名
InputNormalizer = InputTransform
DataNormalizer = OutputTransform

# 配置兼容
NormalizationConfig = InputNormConfig


def create_normalizer_from_checkpoint(checkpoint_path: str) -> OutputTransform:
    """
    向後兼容：從 checkpoint 創建舊版 DataNormalizer
    
    ⚠️ 已棄用，請使用 UnifiedNormalizer.from_metadata()
    """
    import torch
    
    if not torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    else:
        ckpt = torch.load(checkpoint_path)
    
    if 'normalization' not in ckpt:
        logger.warning("⚠️  Checkpoint 中未找到 'normalization' metadata，使用默認 (type='none')")
        return OutputTransform(OutputNormConfig(norm_type='none'))
    
    # 嘗試新格式（UnifiedNormalizer）
    norm_meta = ckpt['normalization']
    if 'output' in norm_meta:
        output_meta = norm_meta['output']
    else:
        # 舊格式（DataNormalizer）
        output_meta = norm_meta
    
    config = OutputNormConfig(
        norm_type=output_meta.get('type', 'none'),
        variable_order=output_meta.get('variable_order', OutputTransform.DEFAULT_VAR_ORDER.copy()),
        means=output_meta.get('means', {}),
        stds=output_meta.get('stds', {}),
        params=output_meta.get('params', {})
    )
    
    return OutputTransform(config)
