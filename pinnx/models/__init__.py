"""
PINN 模型模組

提供各種 PINN 神經網路架構與包裝器：
- Fourier MLP 基礎網路
- VS-PINN 尺度化包裝器  
- 集成與不確定性量化
- 物理約束強制
- 多尺度架構

主要類別：
- PINNNet: 基礎 Fourier MLP 網路
- MultiScalePINNNet: 多尺度網路架構
- ScaledPINNWrapper: VS-PINN 尺度化包裝器
- EnsemblePINNWrapper: 集成不確定性量化
- PhysicsConstrainedWrapper: 物理約束包裝器
"""

from .fourier_mlp import (
    PINNNet,
    MultiScalePINNNet, 
    FourierFeatures,
    DenseLayer,
    create_pinn_model,
    multiscale_pinn,
    init_siren_weights
)

from .wrappers import (
    ScaledPINNWrapper,
    PhysicsConstrainedWrapper,
    EnsemblePINNWrapper,
    AdaptivePINNWrapper,
    create_scaled_pinn,
    create_ensemble_pinn
)

__all__ = [
    # 基礎網路架構
    'PINNNet',
    'MultiScalePINNNet',
    'FourierFeatures', 
    'DenseLayer',
    
    # 包裝器
    'ScaledPINNWrapper',
    'PhysicsConstrainedWrapper', 
    'EnsemblePINNWrapper',
    'AdaptivePINNWrapper',
    
    # 建構函數
    'create_pinn_model',
    'multiscale_pinn',
    'create_scaled_pinn',
    'create_ensemble_pinn',
    
    # 初始化函數
    'init_siren_weights'
]


def get_model_info():
    """返回可用模型架構資訊"""
    return {
        'base_models': {
            'PINNNet': 'Fourier特徵MLP，支援可變深度/寬度',
            'MultiScalePINNNet': '多尺度Fourier網路，適用於多頻問題'
        },
        'wrappers': {
            'ScaledPINNWrapper': 'VS-PINN變數尺度化包裝器',
            'EnsemblePINNWrapper': '集成不確定性量化包裝器',
            'PhysicsConstrainedWrapper': '物理約束強制包裝器',
            'AdaptivePINNWrapper': '自適應訓練包裝器'
        },
        'features': [
            'Fourier Random Features 高頻擬合',
            'VS-PINN 變數尺度化支援',
            '多種激活函數 (tanh, swish, gelu, sine)',
            'SIREN 權重初始化支援',
            '集成不確定性量化',
            '物理約束強制',
            '多尺度架構支援'
        ]
    }
