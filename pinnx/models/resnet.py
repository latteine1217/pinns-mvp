"""
ResNet-based PINN Architecture

This module implements a ResNet-style architecture for Physics-Informed Neural Networks (PINNs),
designed to improve training stability and convergence for high Reynolds number turbulence problems.

Key Features:
- Residual connections to mitigate gradient vanishing/exploding in deep networks.
- Fourier feature mapping for better high-frequency component capture.
- Flexible depth and width configuration.
- Support for various activation functions (tanh, sine, swish).

References:
- Parfenyev et al.
- He et al., "Deep Residual Learning for Image Recognition"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

from pinnx.models.fourier_mlp import FourierFeatures, SineActivation, RWFLinear

class ResidualBlock(nn.Module):
    """
    A standard Residual Block: y = x + F(x)
    
    Structure:
    Input -> Linear -> Activation -> Linear -> Activation -> + Input
    """
    def __init__(self, width: int, activation: nn.Module, 
                 use_rwf: bool = False, rwf_scale_std: float = 0.1):
        super().__init__()
        
        if use_rwf:
            self.fc1 = RWFLinear(width, width, scale_std=rwf_scale_std)
            self.fc2 = RWFLinear(width, width, scale_std=rwf_scale_std)
        else:
            self.fc1 = nn.Linear(width, width)
            self.fc2 = nn.Linear(width, width)
            
        self.activation = activation
        
        # Learnable scaling factor for the residual branch
        # Initialize to a small value or 1.0. Parfenyev suggests starting small if stability is an issue.
        # Here we start at 1.0 but allow it to be learned.
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        self._init_weights()

    def _init_weights(self):
        # Initialize weights - Xavier/Glorot is generally good for Tanh
        if isinstance(self.fc1, nn.Linear):
            nn.init.xavier_uniform_(self.fc1.weight)
            nn.init.zeros_(self.fc1.bias)
        if isinstance(self.fc2, nn.Linear):
            nn.init.xavier_uniform_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.fc1(x)
        residual = self.activation(residual)
        residual = self.fc2(residual)
        residual = self.activation(residual)
        
        return x + self.alpha * residual


class ResNetPINN(nn.Module):
    """
    ResNet architecture for PINNs.
    
    Pipeline:
    Input (x,y,z,t) -> [Fourier Features] -> Linear Projection -> [Residual Blocks] -> Output Linear -> (u,v,w,p)
    """
    def __init__(self,
                 in_dim: int = 3,
                 out_dim: int = 4,
                 width: int = 256,
                 depth: int = 6, # Number of residual blocks
                 fourier_m: int = 32,
                 fourier_sigma: float = 5.0,
                 fourier_multiscale: bool = False,
                 activation: str = 'tanh',
                 use_fourier: bool = True,
                 trainable_fourier: bool = False,
                 use_rwf: bool = False,
                 rwf_scale_std: float = 0.1,
                 sine_omega_0: float = 1.0,
                 # VS-PINN support
                 fourier_normalize_input: bool = False,
                 input_scale_factors: Optional[torch.Tensor] = None):
        
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.width = width
        self.depth = depth
        self.use_fourier = use_fourier
        
        # VS-PINN scaling support
        self.fourier_normalize_input = fourier_normalize_input
        if input_scale_factors is not None:
            self.register_buffer('input_scale_factors', input_scale_factors)
        else:
            self.input_scale_factors = None
            
        self._fourier_norm_type: Optional[str] = None
        self._fourier_shift: Optional[torch.Tensor] = None
        self._fourier_scale: Optional[torch.Tensor] = None
        self._fourier_feature_range: Optional[torch.Tensor] = None
        self._fourier_range: Optional[torch.Tensor] = None

        # 1. Input Embedding (Fourier or direct)
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

        # 2. Activation Function
        if activation == 'tanh':
            self.activation_fn = nn.Tanh()
        elif activation == 'swish':
            self.activation_fn = nn.SiLU()
        elif activation == 'gelu':
            self.activation_fn = nn.GELU()
        elif activation == 'sine':
            self.activation_fn = SineActivation(omega_0=sine_omega_0)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 3. Initial Projection
        if use_rwf:
            self.fc_in = RWFLinear(input_features, width, scale_std=rwf_scale_std)
        else:
            self.fc_in = nn.Linear(input_features, width)
            nn.init.xavier_uniform_(self.fc_in.weight)
            nn.init.zeros_(self.fc_in.bias)

        # 4. Residual Blocks
        self.blocks = nn.ModuleList([
            ResidualBlock(width, self.activation_fn, use_rwf=use_rwf, rwf_scale_std=rwf_scale_std)
            for _ in range(depth)
        ])

        # 5. Output Projection
        self.fc_out = nn.Linear(width, out_dim)
        
        # Initialize output layer (keep it small to start neutral)
        nn.init.xavier_uniform_(self.fc_out.weight, gain=0.1)
        nn.init.zeros_(self.fc_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Handle VS-PINN input normalization for Fourier Features
        if self.use_fourier and self.fourier is not None:
            x_fourier = x
            # Apply inverse normalization if metadata was configured
            if self._fourier_norm_type is not None:
                x_fourier = self._apply_fourier_inverse_normalizer(x_fourier)
            
            # Apply scaling factors if provided (VS-PINN specific)
            if self.fourier_normalize_input:
                if self.input_scale_factors is not None:
                    x_fourier = x_fourier / self.input_scale_factors
                else:
                    # Dynamic normalization fallback (not recommended for training stability, but safe)
                    x_abs_max = x_fourier.abs().max()
                    if x_abs_max > 2.0:
                        x_min = x_fourier.min(dim=0, keepdim=True).values
                        x_max = x_fourier.max(dim=0, keepdim=True).values
                        x_range = x_max - x_min
                        x_fourier = 2.0 * (x_fourier - x_min) / (x_range + 1e-8) - 1.0
            
            h = self.fourier(x_fourier)
        else:
            h = x

        # 2. Initial Projection
        h = self.fc_in(h)
        h = self.activation_fn(h)

        # 3. Residual Blocks
        for block in self.blocks:
            h = block(h)

        # 4. Output Projection
        out = self.fc_out(h)
        
        return out
    
    def _apply_fourier_inverse_normalizer(self, x: torch.Tensor) -> torch.Tensor:
        """Helper to inverse-normalize inputs before Fourier mapping."""
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
        Reuse logic from PINNNet.
        """
        if not self.use_fourier or self.fourier is None:
            return
        
        norm_type = metadata.get('norm_type', 'none')
        if norm_type in ('none', 'identity', None):
            self._fourier_norm_type = None
            return
        
        # Get device/dtype from parameters
        device = next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
        dtype = next(self.parameters()).dtype if list(self.parameters()) else torch.float32
        
        def _prepare(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None: return None
            if isinstance(t, (float, int, list, tuple)):
                t = torch.tensor(t)
            return t.to(device=device, dtype=dtype)
        
        self._fourier_norm_type = norm_type
        
        if norm_type == 'standard':
            self._fourier_shift = _prepare(metadata.get('mean'))
            self._fourier_scale = _prepare(metadata.get('std'))
            if self._fourier_scale is not None:
                self._fourier_scale = torch.clamp(self._fourier_scale, min=1e-8)
        
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
            
            self._fourier_feature_range = _prepare(feature_range)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        fourier_features = 0
        if self.use_fourier and self.fourier is not None:
            fourier_features = self.fourier.m
            
        return {
            'type': 'ResNetPINN',
            'input_dim': self.in_dim,
            'output_dim': self.out_dim,
            'width': self.width,
            'depth (blocks)': self.depth,
            'activation': str(self.activation_fn),
            'fourier_features': fourier_features,
            'total_params': self.get_num_params()
        }
        
    def extra_repr(self) -> str:
        s = self.get_model_summary()
        return f"ResNetPINN(in={s['input_dim']}, out={s['output_dim']}, width={s['width']}, blocks={s['depth (blocks)']}, fourier={s['fourier_features']})"

def create_resnet_model(config: dict) -> nn.Module:
    """Factory function for creating ResNetPINN from config dict."""
    
    # 嚴格配置：不再支援扁平 Fourier 鍵名（避免隱式相容）
    removed_flat_keys = [
        'use_fourier',
        'fourier_m',
        'fourier_sigma',
        'trainable_fourier',
        'fourier_use_2pi',
        'fourier_multiscale',
    ]
    for key in removed_flat_keys:
        if key in config:
            raise ValueError(
                f"Deprecated/removed config key: '{key}'. Use 'fourier_features' dict instead."
            )

    ff_cfg = config.get('fourier_features')
    if not isinstance(ff_cfg, dict):
        raise ValueError("Model config missing required dict: 'fourier_features'")

    ff_type = ff_cfg.get('type')
    if ff_type not in {'standard', 'axis_selective', 'hybrid', 'disabled'}:
        raise ValueError("fourier_features.type must be 'standard' / 'axis_selective' / 'hybrid' / 'disabled'")

    use_fourier = ff_type != 'disabled'
    if use_fourier:
        fourier_m = int(ff_cfg.get('fourier_m'))
        fourier_sigma = float(ff_cfg.get('fourier_sigma'))
        if fourier_m <= 0:
            raise ValueError("fourier_features.fourier_m must be > 0 when enabled")
    else:
        fourier_m = 0
        fourier_sigma = 0.0

    trainable_fourier = bool(ff_cfg.get('trainable_fourier', False))

    # Handle VS-PINN input scaling
    input_scale_factors = None
    if 'input_scale_factors' in config:
        scale_list = config['input_scale_factors']
        if isinstance(scale_list, (list, tuple)):
            input_scale_factors = torch.tensor(scale_list, dtype=torch.float32)

    return ResNetPINN(
        in_dim=config.get('in_dim', 3),
        out_dim=config.get('out_dim', 4),
        width=config.get('width', 256),
        depth=config.get('depth', 6),
        fourier_m=fourier_m,
        fourier_sigma=fourier_sigma,
        fourier_multiscale=False,
        activation=config.get('activation', 'tanh'),
        use_fourier=use_fourier,
        trainable_fourier=trainable_fourier,
        use_rwf=config.get('use_rwf', False),
        rwf_scale_std=config.get('rwf_scale_std', 0.1),
        sine_omega_0=config.get('sine_omega_0', 1.0),
        fourier_normalize_input=config.get('fourier_normalize_input', False),
        input_scale_factors=input_scale_factors
    )
