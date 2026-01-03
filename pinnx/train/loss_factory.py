"""損失函數工廠模組

提供創建各種損失函數實例的工廠函數。
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses import MeanConstraintLoss
from pinnx.train.config_loader import derive_loss_weights


def create_loss_functions(config: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    """建立損失函數實例
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        
    Returns:
        損失函數字典，包含：
        - 'residual': NS 方程殘差損失
        - 'boundary': 邊界條件損失
        - 'prior': 先驗一致性損失
        - 'mean_constraint': 均值約束損失（可選）
    """
    loss_cfg = config.get('losses', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_cfg = physics_type == 'vs_pinn_channel_flow'
    
    base_weight_template, default_adaptive_terms = derive_loss_weights(
        loss_cfg,
        loss_cfg.get('prior_weight', 0.3),
        is_vs_cfg
    )
    
    losses = {
        'residual': NSResidualLoss(
            nu=loss_cfg.get('nu', 1e-3),
            density=loss_cfg.get('rho', 1.0),
            merge_momentum=loss_cfg.get('merge_momentum', False)
        ),
        'boundary': BoundaryConditionLoss(),
        'prior': PriorLossManager(
            consistency_weight=loss_cfg['prior_weight']
        )
    }
    
    # 均值約束損失（可選）
    mean_constraint_cfg = loss_cfg.get('mean_constraint', {})
    if mean_constraint_cfg.get('enabled', False):
        losses['mean_constraint'] = MeanConstraintLoss()
        logging.info(f"✅ MeanConstraintLoss enabled with weight={mean_constraint_cfg.get('weight', 10.0)}")
        logging.info(f"   Target means: {mean_constraint_cfg.get('target_means', {})}")
    
    return losses
