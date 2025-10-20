"""
Fourier-VS MLP 網路架構模組

整合 Fourier Features 與 VS-PINN 變數尺度化的統一架構。

核心特色：
- Fourier Random Features (標準/多尺度)
- VS-PINN 座標縮放支援
- Random Weight Factorization (RWF) 可選
- 可配置的網路深度與寬度
- 支援多種激活函數 (tanh, swish, gelu, sine)
- 殘差連接與層歸一化 (可選)
- 針對 PINNs 自動微分優化的權重初始化

Note:
    此模組僅保留 `PINNNet` (核心網路) 和基礎組件 (RWFLinear, FourierFeatures, 等)。
    舊版本的 `MultiScalePINNNet`、`create_standard_pinn`、`create_enhanced_pinn` 已移除。
    所有模型通過統一的 `create_pinn_model()` 工廠函數創建。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Union, List, Callable, Dict, Any
import numpy as np


class RWFLinear(nn.Module):
    """
    Random Weight Factorization 線性層
    
    基於 arXiv 2308.08468 論文的權重分解技術：
    W^(l) = diag(exp(s^(l))) · V^(l)
    
    優勢：
    1. 改善訓練穩定性 - 防止權重爆炸/消失
    2. 更好的梯度流動 - 指數縮放提供自適應學習率
    3. 隱式正則化 - 對數空間的平滑性約束
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 bias: bool = True,
                 scale_mean: float = 0.0,
                 scale_std: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale_mean = scale_mean
        self.scale_std = scale_std
        
        # V^(l): 標準權重矩陣
        self.V = nn.Parameter(torch.empty(out_features, in_features))
        # s^(l): 對數尺度因子
        self.s = nn.Parameter(torch.empty(out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.V)
        nn.init.normal_(self.s, mean=self.scale_mean, std=self.scale_std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def apply_siren_init(self, omega_0: float, is_first: bool) -> None:
        """
        應用 SIREN 初始化規則到 RWF 權重
        
        基於論文: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
        
        Args:
            omega_0: Sine 激活函數的頻率參數
            is_first: 是否為第一層（第一層使用不同的初始化規則）
        """
        n_in = self.V.shape[1]
        with torch.no_grad():
            if is_first:
                # 第一層: U(-1/n_in, +1/n_in)
                bound = 1.0 / n_in
            else:
                # 隱藏層: U(-sqrt(6/n_in)/omega_0, +sqrt(6/n_in)/omega_0)
                bound = math.sqrt(6.0 / n_in) / omega_0
            
            nn.init.uniform_(self.V, -bound, bound)
            nn.init.zeros_(self.s)  # s 初始化為 0 (exp(0) = 1, 無縮放)
            if self.bias is not None:
                nn.init.zeros_(self.bias)
    
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        weight_key = prefix + 'weight'
        v_key = prefix + 'V'
        s_key = prefix + 's'

        if weight_key in state_dict and v_key not in state_dict:
            raise KeyError(
                f"Checkpoint contains legacy parameter '{weight_key}'. "
                "Please migrate checkpoints to the new RWF format."
            )
        if s_key not in state_dict and v_key in state_dict:
            raise KeyError(
                f"Checkpoint missing required RWF parameter '{s_key}'. "
                "Ensure the checkpoint was saved after the RWF migration."
            )

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        scale_factors = torch.exp(self.s).unsqueeze(1)
        W = scale_factors * self.V
        return F.linear(input, W, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, scale_std={self.scale_std}'


class FourierFeatures(nn.Module):
    """
    Fourier Random Features 編碼層
    
    將輸入座標透過隨機 Fourier 特徵映射到高維空間，
    提升神經網路對高頻函數的擬合能力。
    
    Args:
        in_dim: 輸入維度 (例如 3 代表 t,x,y)
        m: Fourier 特徵數量 (輸出維度為 2m)
        sigma: Fourier 頻率尺度參數
        multiscale: 是否使用多尺度頻率
        trainable: 是否讓 Fourier 係數可訓練
    """
    
    def __init__(self, in_dim: int, m: int = 32, sigma: float = 5.0, 
                 multiscale: bool = False, trainable: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.m = m
        self.sigma = sigma
        self.multiscale = multiscale
        self.out_dim = 2 * m
        
        # 生成 Fourier 係數矩陣
        if multiscale:
            # 多尺度：從低頻到高頻的對數間距
            sigmas = torch.logspace(-1, math.log10(sigma), m // 4).repeat(4)[:m]
            B = torch.randn(in_dim, m) * sigmas.unsqueeze(0)
        else:
            # 單一尺度
            B = torch.randn(in_dim, m) * sigma
        
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_dim] 輸入座標
        Returns:
            [batch_size, 2*m] Fourier 特徵 [cos(z), sin(z)]
        """
        z = 2.0 * math.pi * x @ self.B
        return torch.cat([torch.cos(z), torch.sin(z)], dim=-1)
    
    def extra_repr(self) -> str:
        return f'in_dim={self.in_dim}, m={self.m}, sigma={self.sigma:.2f}, ' \
               f'multiscale={self.multiscale}, out_dim={self.out_dim}'


class SineActivation(nn.Module):
    """
    Sine 激活函數 - 適合高頻特徵捕捉
    
    基於 SIREN (Implicit Neural Representations with Periodic Activation Functions)
    使用周期性激活函數可以更好地表示高頻細節與導數。
    
    Args:
        omega_0: 頻率參數，控制 sine 函數的週期性 (預設 1.0，較保守)
    
    注意：與 Fourier 特徵結合時使用較小的 omega_0 以避免訓練不穩定
    """
    def __init__(self, omega_0: float = 1.0):
        super().__init__()
        self.omega_0 = omega_0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * x)


class DenseLayer(nn.Module):
    """
    密集連接層，支援多種激活函數、殘差連接與層歸一化
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 activation: str = 'tanh', 
                 use_residual: bool = False,
                 use_layer_norm: bool = False,
                 dropout: float = 0.0,
                 use_rwf: bool = False,
                 rwf_scale_mean: float = 0.0,
                 rwf_scale_std: float = 0.1,
                 sine_omega_0: float = 30.0):
        super().__init__()
        
        # 選擇線性層類型
        if use_rwf:
            self.linear = RWFLinear(in_features, out_features, bias=True, 
                                   scale_mean=rwf_scale_mean, scale_std=rwf_scale_std)
        else:
            self.linear = nn.Linear(in_features, out_features)
        
        self.use_residual = use_residual and (in_features == out_features)
        self.use_layer_norm = use_layer_norm
        self.use_rwf = use_rwf
        
        # 層歸一化
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        
        # 激活函數選擇
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'sine':
            self.activation = SineActivation(omega_0=sine_omega_0)
        else:
            raise ValueError(f"不支援的激活函數: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 權重初始化 (僅對標準線性層)
        if not use_rwf and isinstance(self.linear, nn.Linear):
            if activation in ['swish', 'relu', 'elu']:
                nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
            elif activation == 'sine':
                # 🎯 SIREN 專用初始化（論文標準）
                # 隱藏層：U(-sqrt(6/n_in)/omega_0, +sqrt(6/n_in)/omega_0)
                # 注意：第一層初始化由 PINNNet 處理
                if isinstance(self.activation, SineActivation):
                    import numpy as np
                    n_in = self.linear.weight.shape[1]
                    omega_0 = self.activation.omega_0
                    bound = np.sqrt(6 / n_in) / omega_0
                    nn.init.uniform_(self.linear.weight, -bound, bound)
                else:
                    nn.init.xavier_normal_(self.linear.weight, gain=0.5)
                nn.init.zeros_(self.linear.bias)
            else:
                nn.init.xavier_normal_(self.linear.weight)
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.use_residual else None
        
        out = self.linear(x)
        
        if self.use_layer_norm:
            out = self.layer_norm(out)
        
        out = self.activation(out)
        
        if self.dropout is not None:
            out = self.dropout(out)
        
        if residual is not None:
            out = out + residual
            
        return out


class PINNNet(nn.Module):
    """
    基於 Fourier 特徵的 PINN 神經網路 (統一版)
    
    整合標準與增強功能，透過參數控制網路複雜度：
    - 標準模式：較淺較窄，tanh 激活，無殘差/歸一化
    - 增強模式：較深較寬，swish 激活，啟用殘差/歸一化/RWF
    
    網路架構：
    輸入 -> Fourier 特徵 -> [投影層] -> 多層 Dense -> 輸出
    """
    
    def __init__(self,
                 in_dim: int = 3,           # 輸入維度 (t,x,y)
                 out_dim: int = 4,          # 輸出維度 (u,v,p,S)
                 width: int = 256,          # 隱藏層寬度
                 depth: int = 5,            # 網路深度
                 fourier_m: int = 32,       # Fourier 特徵數
                 fourier_sigma: float = 5.0, # Fourier 頻率尺度
                 fourier_multiscale: bool = False, # 多尺度 Fourier
                 activation: str = 'tanh',   # 激活函數
                 use_fourier: bool = True,   # 是否使用 Fourier 特徵
                 trainable_fourier: bool = False, # Fourier 係數是否可訓練
                 use_residual: bool = False, # 是否使用殘差連接
                 use_layer_norm: bool = False, # 是否使用層歸一化
                 use_input_projection: bool = False, # 是否使用輸入投影層
                 dropout: float = 0.0,      # Dropout 比率
                 use_rwf: bool = False,     # 是否使用 RWF
                 rwf_scale_mean: float = 0.0, # RWF 尺度均值（PirateNet: 1.0）
                 rwf_scale_std: float = 0.1, # RWF 尺度標準差
                 sine_omega_0: float = 1.0, # Sine 激活函數頻率參數 (較保守以配合 Fourier 特徵)
                 fourier_normalize_input: bool = False, # 🔧 是否在 Fourier 前標準化輸入（修復 VS-PINN 縮放問題）
                 input_scale_factors: Optional[torch.Tensor] = None): # 🔧 輸入縮放因子 [N_x, N_y, N_z]（用於 VS-PINN）
        
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_fourier = use_fourier
        self.use_input_projection = use_input_projection
        self.fourier_normalize_input = fourier_normalize_input
        self._fourier_norm_type: Optional[str] = None
        self._fourier_shift: Optional[torch.Tensor] = None
        self._fourier_scale: Optional[torch.Tensor] = None
        self._fourier_feature_range: Optional[torch.Tensor] = None
        self._fourier_range: Optional[torch.Tensor] = None
        
        # 🔧 註冊輸入縮放因子（用於 VS-PINN 的標準化補償）
        if input_scale_factors is not None:
            self.register_buffer('input_scale_factors', input_scale_factors)
        else:
            self.input_scale_factors = None
        
        # Fourier 特徵編碼
        if use_fourier:
            self.fourier = FourierFeatures(
                in_dim, fourier_m, fourier_sigma, 
                multiscale=fourier_multiscale,
                trainable=trainable_fourier
            )
            input_features = self.fourier.out_dim
        else:
            self.fourier = None
            input_features = in_dim
        
        # 可選的輸入投影層
        if use_input_projection:
            self.input_projection = nn.Linear(input_features, width)
            nn.init.xavier_normal_(self.input_projection.weight)
            nn.init.zeros_(self.input_projection.bias)
            current_dim = width
        else:
            self.input_projection = None
            current_dim = input_features
        
        # 隱藏層
        layers = []
        for i in range(depth):
            # 第一層的輸入維度可能與後續層不同
            layer_in_dim = current_dim if i == 0 else width
            
            layers.append(DenseLayer(
                layer_in_dim, width, 
                activation=activation,
                use_residual=use_residual and i > 0,  # 第一層不用殘差
                use_layer_norm=use_layer_norm,
                dropout=dropout,
                use_rwf=use_rwf,
                rwf_scale_mean=rwf_scale_mean,
                rwf_scale_std=rwf_scale_std,
                sine_omega_0=sine_omega_0
            ))
            current_dim = width
        
        self.hidden_layers = nn.ModuleList(layers)
        
        # 輸出層 (線性，無激活函數)
        self.output_layer = nn.Linear(width, out_dim)
        
        # 輸出層特殊初始化：較小的權重，有助於訓練穩定
        output_gain = 0.01 if use_residual else 0.1
        nn.init.xavier_normal_(self.output_layer.weight, gain=output_gain)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播
        
        Args:
            x: [batch_size, in_dim] 輸入座標（可能是物理座標或 VS-PINN 縮放座標）
        Returns:
            [batch_size, out_dim] 網路輸出
        """
        # 🔧 修復：若啟用標準化且輸入已被 VS-PINN 縮放，先還原到 [-1, 1] 範圍
        if self.use_fourier and self.fourier is not None:
            x_fourier = x
            if self._fourier_norm_type is not None:
                x_fourier = self._apply_fourier_inverse_normalizer(x_fourier)
            
            if self.fourier_normalize_input:
                if self.input_scale_factors is not None:
                    x_fourier = x_fourier / self.input_scale_factors
                else:
                    x_abs_max = x_fourier.abs().max()
                    if x_abs_max > 2.0:
                        x_min = x_fourier.min(dim=0, keepdim=True).values
                        x_max = x_fourier.max(dim=0, keepdim=True).values
                        x_range = x_max - x_min
                        x_fourier = 2.0 * (x_fourier - x_min) / (x_range + 1e-8) - 1.0
            
            h = self.fourier(x_fourier)
        else:
            h = x
        
        # 可選的輸入投影
        if self.use_input_projection and self.input_projection is not None:
            h = self.input_projection(h)
            h = F.silu(h)  # Swish activation
        
        # 隱藏層前向傳播
        for layer in self.hidden_layers:
            h = layer(h)
        
        # 輸出層
        output = self.output_layer(h)
        
        return output
    
    def _apply_fourier_inverse_normalizer(self, x: torch.Tensor) -> torch.Tensor:
        if self._fourier_norm_type is None:
            return x
        
        if self._fourier_norm_type == 'standard':
            if self._fourier_scale is not None:
                x = x * self._fourier_scale
            if self._fourier_shift is not None:
                x = x + self._fourier_shift
            return x
        
        if self._fourier_norm_type in ('minmax', 'channel_flow'):
            if (
                self._fourier_feature_range is not None and
                self._fourier_range is not None and
                self._fourier_shift is not None
            ):
                lo, hi = self._fourier_feature_range[0], self._fourier_feature_range[1]
                scale = hi - lo
                x = (x - lo) / (scale + 1e-8)
                x = x * self._fourier_range + self._fourier_shift
            return x
        
        return x
    
    def configure_fourier_input(self, metadata: Dict[str, Any]) -> None:
        """
        Configure inverse-normalization for Fourier features using input normalizer stats.
        """
        if not self.use_fourier or self.fourier is None:
            return
        
        norm_type = metadata.get('norm_type', 'none')
        if norm_type in ('none', 'identity', None):
            self._fourier_norm_type = None
            self._fourier_shift = None
            self._fourier_scale = None
            self._fourier_feature_range = None
            self._fourier_range = None
            return
        
        device = next(self.parameters()).device if any(p.requires_grad for p in self.parameters()) else torch.device('cpu')
        dtype = next(self.parameters()).dtype if any(p.requires_grad for p in self.parameters()) else torch.float32
        
        def _prepare(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return None
            return t.to(device=device, dtype=dtype)
        
        self._fourier_norm_type = norm_type
        
        if norm_type == 'standard':
            self._fourier_shift = _prepare(metadata.get('mean'))
            self._fourier_scale = _prepare(metadata.get('std'))
            if self._fourier_scale is not None:
                self._fourier_scale = torch.clamp(self._fourier_scale, min=1e-8)
            self._fourier_feature_range = None
            self._fourier_range = None
        
        elif norm_type in ('minmax', 'channel_flow'):
            feature_range = metadata.get('feature_range')
            data_min = metadata.get('data_min')
            data_range = metadata.get('data_range')
            bounds = metadata.get('bounds')
            
            if data_min is None and bounds is not None:
                data_min = bounds[:, 0].unsqueeze(0)
            if data_range is None and bounds is not None:
                data_range = (bounds[:, 1] - bounds[:, 0]).unsqueeze(0)
            
            self._fourier_shift = _prepare(data_min)
            prepared_range = _prepare(data_range)
            if prepared_range is not None:
                self._fourier_range = torch.clamp(prepared_range, min=1e-8)
            else:
                self._fourier_range = None
            
            feature_tensor = feature_range.to(device=device, dtype=dtype) if isinstance(feature_range, torch.Tensor) else torch.tensor(feature_range, device=device, dtype=dtype)
            self._fourier_feature_range = feature_tensor
            self._fourier_scale = None
        
        else:
            # 無法識別的類型：視為無需處理
            self._fourier_norm_type = None
            self._fourier_shift = None
            self._fourier_scale = None
            self._fourier_feature_range = None
            self._fourier_range = None
    
    def get_num_params(self) -> int:
        """返回模型參數總數"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """返回模型結構摘要"""
        width = 0
        if self.hidden_layers:
            layer = self.hidden_layers[0]
            if isinstance(layer.linear, nn.Linear):
                width = layer.linear.out_features
            elif isinstance(layer.linear, RWFLinear):
                width = layer.linear.out_features
        
        fourier_features = 0
        if self.use_fourier and self.fourier is not None:
            fourier_features = self.fourier.m
        
        return {
            'input_dim': self.in_dim,
            'output_dim': self.out_dim,
            'depth': len(self.hidden_layers),
            'width': width,
            'total_params': self.get_num_params(),
            'fourier_features': fourier_features,
            'use_residual': self.hidden_layers[0].use_residual if self.hidden_layers else False,
            'use_layer_norm': self.hidden_layers[0].use_layer_norm if self.hidden_layers else False,
            'use_rwf': self.hidden_layers[0].use_rwf if self.hidden_layers else False
        }
    
    def extra_repr(self) -> str:
        summary = self.get_model_summary()
        fourier_info = f", fourier_m={summary['fourier_features']}" if self.use_fourier else ""
        return (f"in_dim={summary['input_dim']}, out_dim={summary['output_dim']}, "
                f"width={summary['width']}, depth={summary['depth']}{fourier_info}, "
                f"params={summary['total_params']:,}")


# ========== 已移除的類別 ==========
#
# MultiScalePINNNet 已移除（2025-10-20）
# 原因：增加架構複雜度但效果不顯著，單一尺度的 fourier_vs_mlp 配合適當的
#       fourier_sigma 參數（如 2-5）已能有效捕捉多尺度特徵
#
# 如需多尺度特徵，建議：
# 1. 使用 fourier_multiscale=True（對數間距的多頻率採樣）
# 2. 調整 fourier_sigma 至推薦範圍（2-5 for channel flow）
# 3. 增加 fourier_m（Fourier 特徵數量）至 64-256


# ========== 便捷工廠函數 ==========

def create_pinn_model(config: dict) -> nn.Module:
    """
    根據配置字典建立 Fourier-VS PINN 模型

    支援的模型類型：
    - 'fourier_vs_mlp': 統一的 Fourier-VS 架構
    - 'standard': 別名，指向 fourier_vs_mlp（向後兼容）

    Args:
        config: 模型配置字典，必須包含：
            - type: 模型類型
            - in_dim: 輸入維度
            - out_dim: 輸出維度
            - width: 隱藏層寬度
            - depth: 網路深度
            - fourier_m: Fourier 特徵數量
            - fourier_sigma: Fourier 頻率尺度
            - activation: 激活函數 (tanh/swish/sine/gelu)
            - use_fourier: 是否啟用 Fourier Features
            - use_residual: 是否啟用殘差連接
            - use_rwf: 是否啟用 Random Weight Factorization

    Returns:
        PINNNet 模型實例

    Raises:
        ValueError: 如果指定了不支援的模型類型

    Example:
        >>> config = {
        ...     'type': 'fourier_vs_mlp',
        ...     'in_dim': 3,
        ...     'out_dim': 4,
        ...     'width': 256,
        ...     'depth': 6,
        ...     'fourier_m': 64,
        ...     'fourier_sigma': 3.0,
        ...     'activation': 'swish'
        ... }
        >>> model = create_pinn_model(config)
    """
    model_type = config.get('type', 'fourier_vs_mlp')

    # 向後兼容：支援 'standard' 別名
    if model_type in ('fourier_vs_mlp', 'standard'):
        # 處理 VS-PINN 縮放因子
        input_scale_factors = None
        if 'input_scale_factors' in config:
            scale_list = config['input_scale_factors']
            if isinstance(scale_list, (list, tuple)):
                input_scale_factors = torch.tensor(scale_list, dtype=torch.float32)

        return PINNNet(
            in_dim=config.get('in_dim', 3),
            out_dim=config.get('out_dim', 4),
            width=config.get('width', 256),
            depth=config.get('depth', 5),
            fourier_m=config.get('fourier_m', 32),
            fourier_sigma=config.get('fourier_sigma', 5.0),
            fourier_multiscale=config.get('fourier_multiscale', False),
            activation=config.get('activation', 'tanh'),
            use_fourier=config.get('use_fourier', True),
            trainable_fourier=config.get('trainable_fourier', False),
            use_residual=config.get('use_residual', False),
            use_layer_norm=config.get('use_layer_norm', False),
            use_input_projection=config.get('use_input_projection', False),
            dropout=config.get('dropout', 0.0),
            use_rwf=config.get('use_rwf', False),
            rwf_scale_mean=config.get('rwf_scale_mean', 0.0),
            rwf_scale_std=config.get('rwf_scale_std', 0.1),
            sine_omega_0=config.get('sine_omega_0', 1.0),
            fourier_normalize_input=config.get('fourier_normalize_input', False),
            input_scale_factors=input_scale_factors
        )

    else:
        raise ValueError(
            f"不支援的模型類型: {model_type}\n"
            f"支援的類型: 'fourier_vs_mlp', 'standard' (向後兼容)\n"
            f"注意: \n"
            f"  - 'multiscale' 已移除，請使用 fourier_vs_mlp + fourier_multiscale=True\n"
            f"  - 'enhanced_fourier_mlp' 已移除，請使用 'fourier_vs_mlp'"
        )


# ========== 已移除的便捷函數 ==========
#
# create_standard_pinn(), create_enhanced_pinn(), multiscale_pinn() 已移除（2025-10-20）
# 原因：統一使用 create_pinn_model(config) 工廠函數，避免 API 碎片化
#
# 遷移指南：
#
# 舊代碼:
#   model = create_standard_pinn(in_dim=3, out_dim=4, width=128)
#
# 新代碼:
#   config = {'type': 'fourier_vs_mlp', 'in_dim': 3, 'out_dim': 4, 'width': 128}
#   model = create_pinn_model(config)
#
# 或直接使用:
#   model = PINNNet(in_dim=3, out_dim=4, width=128)


def init_siren_weights(model: PINNNet) -> None:
    """
    對使用 Sine 激活的模型應用 SIREN 權重初始化
    
    參考: Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions"
    
    初始化規則：
    - 第一層：U(-1/n_in, +1/n_in)
    - 隱藏層：U(-sqrt(6/n_in)/omega_0, +sqrt(6/n_in)/omega_0)
    - 輸出層：保持原始初始化（小權重）
    
    支援：
    - 標準 nn.Linear 層
    - RWFLinear 層（透過 apply_siren_init() 方法）
    
    Args:
        model: PINNNet 模型實例（必須使用 sine 激活）
    """
    import numpy as np
    
    # 檢查第一層
    if len(model.hidden_layers) > 0:
        first_layer = model.hidden_layers[0]  # type: ignore
        if isinstance(first_layer, DenseLayer) and isinstance(first_layer.activation, SineActivation):
            # 獲取 omega_0 參數
            omega_0 = first_layer.activation.omega_0
            
            # 第一層特殊初始化
            first_linear = first_layer.linear
            
            if isinstance(first_linear, RWFLinear):
                # RWF 路徑：使用專用初始化方法
                first_linear.apply_siren_init(omega_0, is_first=True)
                print(f"✅ SIREN 初始化完成（RWF 模式）：第一層 omega_0={omega_0:.2f}")
                
                # 初始化後續 RWF 層
                for i, layer in enumerate(model.hidden_layers[1:], start=2):
                    if isinstance(layer, DenseLayer) and isinstance(layer.linear, RWFLinear):
                        layer.linear.apply_siren_init(omega_0, is_first=False)
                
            elif isinstance(first_linear, nn.Linear):
                # 標準 nn.Linear 路徑
                n_in = first_linear.weight.shape[1]  # type: ignore
                bound = float(1.0 / n_in)  # type: ignore
                with torch.no_grad():
                    nn.init.uniform_(first_linear.weight, -bound, bound)  # type: ignore
                    nn.init.zeros_(first_linear.bias)  # type: ignore
                
                # 後續層已經在 DenseLayer.__init__ 中處理
                print(f"✅ SIREN 初始化完成（標準模式）：第一層 bound=±{bound:.6f}")
            else:
                print(f"⚠️  未知的線性層類型: {type(first_linear)}")
        else:
            print("⚠️  模型未使用 Sine 激活，跳過 SIREN 初始化")


if __name__ == "__main__":
    # 測試程式碼
    print("=== Fourier-VS MLP 測試 ===\n")

    # 測試 1: 基礎配置
    config_basic = {
        'type': 'fourier_vs_mlp',
        'in_dim': 3,
        'out_dim': 4,
        'width': 128,
        'depth': 4,
        'fourier_m': 32,
        'fourier_sigma': 3.0,
        'activation': 'tanh'
    }
    model_basic = create_pinn_model(config_basic)
    print(f"1️⃣  基礎模型: {model_basic}")
    print(f"   參數總數: {model_basic.get_num_params():,}\n")

    # 測試 2: 增強配置（Residual + RWF）
    config_enhanced = {
        'type': 'fourier_vs_mlp',
        'in_dim': 3,
        'out_dim': 4,
        'width': 256,
        'depth': 6,
        'fourier_m': 64,
        'fourier_sigma': 3.0,
        'activation': 'swish',
        'use_residual': True,
        'use_rwf': False
    }
    model_enhanced = create_pinn_model(config_enhanced)
    print(f"2️⃣  增強模型: {model_enhanced}")
    print(f"   參數總數: {model_enhanced.get_num_params():,}\n")

    # 測試 3: 向後兼容（舊名稱 'standard'）
    config_legacy = config_basic.copy()
    config_legacy['type'] = 'standard'
    model_legacy = create_pinn_model(config_legacy)
    print(f"3️⃣  向後兼容測試 (type='standard'): ✅")
    print(f"   模型類型: {type(model_legacy).__name__}\n")

    # 測試 4: 前向傳播
    x = torch.randn(100, 3)
    with torch.no_grad():
        y_basic = model_basic(x)
        y_enhanced = model_enhanced(x)
    print(f"4️⃣  前向傳播測試:")
    print(f"   輸入形狀: {x.shape}")
    print(f"   基礎模型輸出: {y_basic.shape}")
    print(f"   增強模型輸出: {y_enhanced.shape}\n")

    # 測試 5: 梯度計算（PINNs 關鍵）
    x_grad = torch.randn(50, 3, requires_grad=True)
    y_grad = model_basic(x_grad)
    u = y_grad[:, 0]
    du_dx = torch.autograd.grad(u.sum(), x_grad, create_graph=True)[0][:, 0]
    print(f"5️⃣  梯度計算測試:")
    print(f"   ∂u/∂x 形狀: {du_dx.shape}")
    print(f"   ∂u/∂x 範圍: [{du_dx.min():.4f}, {du_dx.max():.4f}]\n")

    # 測試 6: 模型摘要
    summary = model_enhanced.get_model_summary()
    print(f"6️⃣  模型摘要:")
    for key, value in summary.items():
        print(f"   {key:20s}: {value}")

    print("\n✅ 所有測試通過！")
