"""
PINN 模型模組

提供 Fourier-VS PINN 統一架構與包裝器：
- Fourier-VS MLP: 整合 Fourier Features 與 VS-PINN 的核心網路
- VS-PINN 尺度化包裝器
- 集成與不確定性量化
- 物理約束強制

主要類別：
- PINNNet: 核心 Fourier-VS MLP 網路
- ScaledPINNWrapper: VS-PINN 尺度化包裝器
- EnsemblePINNWrapper: 集成不確定性量化
- PhysicsConstrainedWrapper: 物理約束包裝器

Note (2025-10-20):
    - MultiScalePINNNet 已移除，使用 fourier_multiscale=True 替代
    - create_standard_pinn/create_enhanced_pinn 已移除，統一使用 create_pinn_model()
    - 推薦模型類型: 'fourier_vs_mlp'
    - 'enhanced_fourier_mlp' 已移除，請改用 'fourier_vs_mlp'
"""

from .fourier_mlp import (
    PINNNet,
    FourierFeatures,
    DenseLayer,
    RWFLinear,
    SineActivation,
    create_pinn_model,
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
    # 核心網路架構
    'PINNNet',
    'FourierFeatures',
    'DenseLayer',
    'RWFLinear',
    'SineActivation',

    # 包裝器
    'ScaledPINNWrapper',
    'PhysicsConstrainedWrapper',
    'EnsemblePINNWrapper',
    'AdaptivePINNWrapper',

    # 建構函數
    'create_pinn_model',
    'create_scaled_pinn',
    'create_ensemble_pinn',

    # 初始化函數
    'init_siren_weights'
]


def get_model_info():
    """返回可用模型架構資訊"""
    return {
        'core_model': {
            'PINNNet': 'Fourier-VS MLP統一架構，整合Fourier Features與VS-PINN縮放'
        },
        'wrappers': {
            'ScaledPINNWrapper': 'VS-PINN變數尺度化包裝器',
            'EnsemblePINNWrapper': '集成不確定性量化包裝器',
            'PhysicsConstrainedWrapper': '物理約束強制包裝器',
            'AdaptivePINNWrapper': '自適應訓練包裝器'
        },
        'features': [
            'Fourier Random Features 高頻擬合 (σ=2-5推薦)',
            'VS-PINN 座標尺度化支援',
            'Random Weight Factorization (RWF) 可選',
            '多種激活函數 (tanh, swish, gelu, sine)',
            'SIREN 權重初始化支援',
            '殘差連接與層歸一化',
            '集成不確定性量化',
            '物理約束強制'
        ],
        'removed_in_v2': [
            'MultiScalePINNNet - 使用 fourier_multiscale=True 替代',
            'create_standard_pinn - 使用 create_pinn_model() 替代',
            'create_enhanced_pinn - 使用 create_pinn_model() 替代',
            'multiscale_pinn - 功能整合至 PINNNet'
        ]
    }
