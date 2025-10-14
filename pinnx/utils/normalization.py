"""
統一標準化系統模組

提供兩種標準化器：
1. InputNormalizer: 空間坐標標準化（保持原有實作）
2. DataNormalizer: 輸出變量標準化（新增，用於 u, v, w, p）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict, Union, Any
import logging

import torch
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class NormalizationConfig:
    norm_type: str = "none"
    feature_range: Tuple[float, float] = (-1.0, 1.0)
    bounds: Optional[torch.Tensor] = None  # shape [dim, 2]


class InputNormalizer:
    """
    Lightweight normalizer for spatial coordinates.

    Supported types:
        - none / identity
        - standard  : (x - mean) / std
        - minmax    : map to feature_range using observed min/max
        - channel_flow : map using provided domain bounds to feature_range
    """

    def __init__(self, config: NormalizationConfig):
        self.norm_type = (config.norm_type or "none").lower()
        self.feature_range = config.feature_range
        self.bounds = config.bounds

        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.data_min: Optional[torch.Tensor] = None
        self.data_range: Optional[torch.Tensor] = None

    def fit(self, samples: torch.Tensor) -> "InputNormalizer":
        """
        Fit statistics from samples.

        Args:
            samples: [N, D] tensor on any device
        """
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
            # channel_flow uses predefined bounds; nothing to fit.
            pass
        else:
            # identity / none
            pass
        return self

    def transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer is not fitted.")
            return (tensor - self.mean) / self.std
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer is not fitted.")
            norm = (tensor - self.data_min) / self.data_range
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds not provided.")
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            denom = torch.where((maxs - mins) < 1e-8, torch.ones_like(maxs - mins), maxs - mins)
            norm = (tensor - mins) / denom
            lo, hi = self.feature_range
            return norm * (hi - lo) + lo
        raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.norm_type in ("none", "identity", "vs_pinn"):
            return tensor
        if self.norm_type == "standard":
            if self.mean is None or self.std is None:
                raise RuntimeError("Standard normalizer is not fitted.")
            return tensor * self.std + self.mean
        if self.norm_type == "minmax":
            if self.data_min is None or self.data_range is None:
                raise RuntimeError("MinMax normalizer is not fitted.")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            return norm * self.data_range + self.data_min
        if self.norm_type == "channel_flow":
            if self.bounds is None:
                raise RuntimeError("Channel-flow bounds not provided.")
            lo, hi = self.feature_range
            norm = (tensor - lo) / (hi - lo + 1e-12)
            mins = self.bounds[:, 0].unsqueeze(0)
            maxs = self.bounds[:, 1].unsqueeze(0)
            return norm * (maxs - mins) + mins
        raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def to(self, device: torch.device) -> "InputNormalizer":
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


# ===================================================================
# 輸出變量標準化器（新增於 TASK-008）
# ===================================================================

class DataNormalizer:
    """
    輸出變量標準化管理器（用於 u, v, w, p 等物理量）
    
    解決 Phase 6B/6C 的核心問題：訓練與評估使用一致的標準化策略。
    
    主要功能：
    - 支持多種標準化類型：training_data_norm, friction_velocity, manual, none
    - 從配置或資料自動計算標準化係數
    - 標準化 metadata 可保存至 checkpoint
    - 與 denormalization.py 完全兼容
    
    使用範例：
        >>> from pinnx.utils.normalization import DataNormalizer
        >>> 
        >>> # 從配置創建
        >>> config = {
        ...     'normalization': {
        ...         'type': 'training_data_norm',
        ...         'params': {'u_scale': 9.84, 'v_scale': 0.19, ...}
        ...     }
        ... }
        >>> normalizer = DataNormalizer.from_config(config)
        >>> 
        >>> # 訓練時標準化
        >>> u_true_norm = normalizer.normalize(u_true, 'u')
        >>> 
        >>> # 保存到 checkpoint
        >>> metadata = normalizer.get_metadata()
        >>> torch.save({'model': ..., 'normalization': metadata}, 'ckpt.pth')
    
    Attributes:
        norm_type: 標準化類型
        scales: 標準化係數字典 {'u': float, 'v': float, ...}
        params: 原始參數（用於重建）
    """
    
    SUPPORTED_TYPES = ['training_data_norm', 'friction_velocity', 'manual', 'none']
    DEFAULT_VAR_ORDER = ['u', 'v', 'w', 'p', 'S']  # 預設變量順序
    
    def __init__(
        self, 
        norm_type: str = 'none',
        scales: Optional[Dict[str, float]] = None,
        means: Optional[Dict[str, float]] = None,
        params: Optional[Dict[str, Any]] = None
    ):
        """
        初始化標準化器
        
        Args:
            norm_type: 標準化類型
            scales: 標準化係數字典（標準差 std）
            means: 均值字典（用於 Z-score 標準化）
            params: 額外參數（用於重建或物理參數）
        """
        if norm_type not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的標準化類型: {norm_type}。支持: {self.SUPPORTED_TYPES}"
            )
        
        self.norm_type = norm_type
        self.scales = scales if scales is not None else {}
        self.means = means if means is not None else {}
        self.params = params if params is not None else {}
        
        logger.info(f"✅ DataNormalizer 初始化: type={norm_type}, means={self.means}, stds={self.scales}")
    
    @classmethod
    def from_config(cls, config: Dict) -> 'DataNormalizer':
        """
        從配置字典創建標準化器
        
        配置格式（新版 Z-score）：
            normalization:
              type: 'training_data_norm'
              params:
                u_mean: 9.921185
                u_std: 4.593879
                v_mean: -0.000085
                v_std: 0.329614
                w_mean: -0.002202
                w_std: 3.865396
                p_mean: -40.374241
                p_std: 28.619722
        
        Args:
            config: 配置字典（需包含 'normalization' 段落）
        
        Returns:
            DataNormalizer 實例
        """
        if 'normalization' not in config:
            logger.warning("⚠️  配置中未找到 'normalization' 段落，使用默認 (type='none')")
            return cls(norm_type='none')
        
        norm_cfg = config['normalization']
        norm_type = norm_cfg.get('type', 'none')
        params = norm_cfg.get('params', {})
        
        # 根據類型提取標準化係數
        if norm_type == 'training_data_norm':
            means, scales = cls._extract_training_data_scales(params)
        elif norm_type == 'friction_velocity':
            means, scales = cls._extract_friction_velocity_scales(params, config)
        elif norm_type == 'manual':
            # 手動模式：假設用戶提供 *_mean 和 *_std
            means = {k.replace('_mean', ''): v for k, v in params.items() if k.endswith('_mean')}
            scales = {k.replace('_std', ''): v for k, v in params.items() if k.endswith('_std')}
        else:  # 'none'
            means = {}
            scales = {}
        
        return cls(norm_type=norm_type, scales=scales, means=means, params=params)
    
    @classmethod
    def from_metadata(cls, metadata: Dict) -> 'DataNormalizer':
        """
        從 checkpoint metadata 恢復標準化器
        
        Args:
            metadata: {'type': str, 'means': dict, 'scales': dict, 'params': dict}
        
        Returns:
            DataNormalizer 實例
        """
        norm_type = metadata.get('type', 'none')
        scales = metadata.get('scales', {})
        means = metadata.get('means', {})
        params = metadata.get('params', {})
        
        logger.info(f"🔄 從 checkpoint 恢復 DataNormalizer: type={norm_type}")
        return cls(norm_type=norm_type, scales=scales, means=means, params=params)
    
    @classmethod
    def from_data(
        cls, 
        data: Dict[str, Union[np.ndarray, torch.Tensor]], 
        norm_type: str = 'training_data_norm'
    ) -> 'DataNormalizer':
        """
        從資料自動計算標準化係數（Z-score: mean + std）
        
        Args:
            data: 資料字典 {'u': array, 'v': array, ...}
            norm_type: 標準化類型
        
        Returns:
            DataNormalizer 實例
        """
        means = {}
        stds = {}
        
        for var_name, values in data.items():
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            
            mean = float(np.mean(values))
            std = float(np.std(values))
            
            # 避免除以零
            if abs(std) < 1e-10:
                logger.warning(f"⚠️  {var_name} 的標準差接近零，設為 1.0")
                std = 1.0
            
            means[var_name] = mean
            stds[var_name] = std
        
        params = {'source': 'auto_computed_zscore'}
        
        logger.info(f"✅ 從資料計算 Z-score 係數: means={means}, stds={stds}")
        return cls(norm_type=norm_type, scales=stds, means=means, params=params)
    
    def normalize(
        self, 
        value: Union[np.ndarray, torch.Tensor, float], 
        var_name: str
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        標準化單個變量（Z-score: (x - mean) / std）
        
        Args:
            value: 原始值（物理空間）
            var_name: 變量名稱 ('u', 'v', 'w', 'p', ...)
        
        Returns:
            標準化後的值
        """
        if self.norm_type == 'none':
            return value
        
        if var_name not in self.scales:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過標準化")
            return value
        
        mean = self.means.get(var_name, 0.0)
        std = self.scales[var_name]
        
        return (value - mean) / std
    
    def denormalize(
        self, 
        value: Union[np.ndarray, torch.Tensor, float], 
        var_name: str
    ) -> Union[np.ndarray, torch.Tensor, float]:
        """
        反標準化單個變量（逆 Z-score: x * std + mean）
        
        Args:
            value: 標準化空間的值
            var_name: 變量名稱
        
        Returns:
            物理空間的值
        """
        if self.norm_type == 'none':
            return value
        
        if var_name not in self.scales:
            logger.warning(f"⚠️  變量 {var_name} 無標準化係數，跳過反標準化")
            return value
        
        mean = self.means.get(var_name, 0.0)
        std = self.scales[var_name]
        
        return value * std + mean
    
    def normalize_batch(
        self, 
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[list] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        批次標準化（用於訓練時標準化整個輸出張量，Z-score）
        
        Args:
            predictions: 預測張量 [N, out_dim]
            var_order: 變量順序列表（默認 ['u', 'v', 'w', 'p', 'S']）
        
        Returns:
            標準化後的張量 [N, out_dim]
        """
        if self.norm_type == 'none':
            return predictions
        
        if var_order is None:
            var_order = self.DEFAULT_VAR_ORDER
        
        is_torch = isinstance(predictions, torch.Tensor)
        result = predictions.clone() if is_torch else predictions.copy()  # type: ignore
        
        for i, var_name in enumerate(var_order):
            if i >= result.shape[-1]:
                break
            if var_name in self.scales:
                mean = self.means.get(var_name, 0.0)
                std = self.scales[var_name]
                if is_torch:
                    result[:, i] = (result[:, i] - mean) / std  # type: ignore
                else:
                    result[:, i] = (result[:, i] - mean) / std  # type: ignore
        
        return result
    
    def denormalize_batch(
        self, 
        predictions: Union[np.ndarray, torch.Tensor],
        var_order: Optional[list] = None
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        批次反標準化（用於評估時恢復物理量綱，逆 Z-score）
        
        Args:
            predictions: 標準化空間的預測 [N, out_dim]
            var_order: 變量順序列表
        
        Returns:
            物理空間的預測 [N, out_dim]
        """
        if self.norm_type == 'none':
            return predictions
        
        if var_order is None:
            var_order = self.DEFAULT_VAR_ORDER
        
        is_torch = isinstance(predictions, torch.Tensor)
        result = predictions.clone() if is_torch else predictions.copy()  # type: ignore
        
        for i, var_name in enumerate(var_order):
            if i >= result.shape[-1]:
                break
            if var_name in self.scales:
                mean = self.means.get(var_name, 0.0)
                std = self.scales[var_name]
                if is_torch:
                    result[:, i] = result[:, i] * std + mean  # type: ignore
                else:
                    result[:, i] = result[:, i] * std + mean  # type: ignore
        
        return result
    
    def get_metadata(self) -> Dict:
        """
        獲取可保存的 metadata（用於 checkpoint）
        
        Returns:
            包含 type, means, scales (stds), params 的字典
        """
        return {
            'type': self.norm_type,
            'means': self.means.copy(),
            'scales': self.scales.copy(),
            'params': self.params.copy()
        }
    
    def __repr__(self) -> str:
        return f"DataNormalizer(type={self.norm_type}, means={self.means}, stds={self.scales})"
    
    # ===================================================================
    # 內部工具函數
    # ===================================================================
    
    @staticmethod
    def _extract_training_data_scales(params: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        提取 training_data_norm 標準化係數（Z-score: mean + std）
        
        Returns:
            (means, stds) 元組
        """
        # 正確的 JHTDB Channel Re_τ=1000 統計量 (cutout3d_128x128x32.npz)
        means = {
            'u': params.get('u_mean', 9.921185),
            'v': params.get('v_mean', -0.000085),
            'w': params.get('w_mean', -0.002202),
            'p': params.get('p_mean', -40.374241)
        }
        
        stds = {
            'u': params.get('u_std', 4.593879),
            'v': params.get('v_std', 0.329614),
            'w': params.get('w_std', 3.865396),
            'p': params.get('p_std', 28.619722)
        }
        
        if 'u_mean' not in params or 'u_std' not in params:
            logger.warning(
                "⚠️  配置中未提供完整統計量，使用 JHTDB Channel Re1000 默認值"
            )
        
        return means, stds
    
    @staticmethod
    def _extract_friction_velocity_scales(params: Dict, config: Dict) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        從物理參數計算 friction_velocity 標準化係數
        
        Returns:
            (means, stds) 元組
        """
        u_tau = params.get('u_tau')
        if u_tau is None and 'physics' in config:
            physics = config['physics']
            if 'channel_flow' in physics:
                u_tau = physics['channel_flow'].get('u_tau', 1.0)
            else:
                u_tau = 1.0
        
        if u_tau is None:
            u_tau = 1.0
            logger.warning("⚠️  未找到 u_tau，使用默認值 1.0")
        
        rho = params.get('rho')
        if rho is None and 'physics' in config:
            rho = config['physics'].get('rho', 1.0)
        if rho is None:
            rho = 1.0
        
        velocity_scale = u_tau
        pressure_scale = rho * u_tau ** 2
        
        # Friction velocity 方法假設均值為 0（已去除平均流）
        means = {
            'u': 0.0,
            'v': 0.0,
            'w': 0.0,
            'p': 0.0
        }
        
        stds = {
            'u': velocity_scale,
            'v': velocity_scale,
            'w': velocity_scale,
            'p': pressure_scale
        }
        
        logger.info(f"📐 Friction velocity scales: u_τ={u_tau}, ρ={rho}")
        
        return means, stds


# ===================================================================
# 便捷函數
# ===================================================================

def create_normalizer_from_checkpoint(checkpoint_path: str) -> DataNormalizer:
    """
    從 checkpoint 創建標準化器
    
    Args:
        checkpoint_path: checkpoint 檔案路徑
    
    Returns:
        DataNormalizer 實例
    """
    import torch
    
    if not torch.cuda.is_available():
        ckpt = torch.load(checkpoint_path, map_location='cpu')
    else:
        ckpt = torch.load(checkpoint_path)
    
    if 'normalization' not in ckpt:
        logger.warning(
            "⚠️  Checkpoint 中未找到 'normalization' metadata，使用默認 (type='none')"
        )
        return DataNormalizer(norm_type='none')
    
    return DataNormalizer.from_metadata(ckpt['normalization'])

