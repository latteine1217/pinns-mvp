"""
訓練循環模組
提供自適應採樣、訓練管理、檢查點管理等功能
"""

from .adaptive_collocation import (
    AdaptiveCollocationSampler,
    create_adaptive_collocation_sampler
)

from .loop import (
    TrainingLoopManager,
    create_training_loop_manager,
    apply_point_weights_to_loss,
    SupervisedPointAugmentor
)

from .checkpointing import (
    save_checkpoint,
    load_checkpoint
)

from .config_loader import (
    load_config,
    normalize_config_structure,
    derive_loss_weights,
    LOSS_KEY_MAP,
    DEFAULT_WEIGHTS,
    VS_ONLY_LOSSES
)

from .factory import (
    get_device,
    create_model,
    create_physics,
    create_optimizer
)

# TODO: ensemble 模組尚未完整實現，以下導入會觸發 NotImplementedError
# 取消註解以啟用 ensemble 功能（當實現完成後）
# from .ensemble import (
#     EnsemblePINNTrainer,
#     create_ensemble_trainer,
#     compute_prediction_uncertainty,
#     compute_uncertainty_error_correlation,
# )

__all__ = [
    # 自適應殘差點採樣
    'AdaptiveCollocationSampler',
    'create_adaptive_collocation_sampler',
    
    # 訓練循環管理
    'TrainingLoopManager',
    'create_training_loop_manager',
    'apply_point_weights_to_loss',
    'SupervisedPointAugmentor',
    
    # 檢查點管理
    'save_checkpoint',
    'load_checkpoint',
    
    # 配置載入與標準化
    'load_config',
    'normalize_config_structure',
    'derive_loss_weights',
    'LOSS_KEY_MAP',
    'DEFAULT_WEIGHTS',
    'VS_ONLY_LOSSES',
    
    # 工廠函數
    'get_device',
    'create_model',
    'create_physics',
    'create_optimizer',
    
    # TODO: Ensemble & UQ（待實現）
    # 'EnsemblePINNTrainer',
    # 'create_ensemble_trainer',
    # 'compute_prediction_uncertainty',
    # 'compute_uncertainty_error_correlation',
]
