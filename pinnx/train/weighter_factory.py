"""
工廠函數：創建動態權重器（Weighters）

此模組負責根據配置創建各種損失權重調度器，包括：
- CurriculumScheduler: 課程訓練調度器（最高優先級）
- StagedWeightScheduler: 階段式權重調度器
- GradNormWeighter: 自適應梯度範數權重器
- CausalWeighter: 因果權重器（基於時間順序）
- AdaptiveWeightScheduler: 自適應權重調度器

📌 Phase 5: Factory Functions Extraction
Created: 2026-01-03
From: scripts/train/train.py lines 116-220 (~105 lines)
"""

import logging
from typing import Dict, Any

import torch
import torch.nn as nn

from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.train.schedulers import StagedWeightScheduler, CurriculumScheduler
from pinnx.train.config_loader import derive_loss_weights


def create_weighters(config: Dict[str, Any], model: nn.Module, device: torch.device, physics=None) -> Dict[str, Any]:
    """建立動態權重器 (需要模型實例)
    
    Args:
        config: 完整配置字典
        model: PyTorch 模型實例（GradNormWeighter 需要）
        device: 訓練設備
        physics: 物理模組實例（CurriculumScheduler 需要，可選）
    
    Returns:
        包含各種權重器的字典：
        {
            'curriculum': CurriculumScheduler or None,
            'staged': StagedWeightScheduler or None,
            'gradnorm': GradNormWeighter or None,
            'causal': CausalWeighter or None,
            'scheduler': AdaptiveWeightScheduler or None
        }
    
    Notes:
        - 課程訓練啟用時，會禁用其他損失權重調度器（但允許 LR scheduler）
        - GradNorm 與階段式權重互斥
        - 所有權重器根據配置選擇性啟用
    """
    loss_cfg = config.get('losses', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_cfg = physics_type == 'vs_pinn_channel_flow'
    base_weight_template, default_adaptive_terms = derive_loss_weights(
        loss_cfg,
        loss_cfg.get('prior_weight', 0.3),
        is_vs_cfg
    )
    weighters = {}
    
    # 🚀 課程訓練調度器（最高優先級）
    curriculum_cfg = config.get('curriculum', {})
    if curriculum_cfg.get('enable', False):
        stages = curriculum_cfg.get('stages', [])
        if stages and physics is not None:
            weighters['curriculum'] = CurriculumScheduler(stages, physics)
            logging.info(f"✅ Curriculum scheduler enabled with {len(stages)} stages")
            # 課程訓練啟用時，禁用其他「損失權重」調度器（但允許 LR scheduler）
            weighters['staged'] = None
            weighters['gradnorm'] = None
            weighters['causal'] = None
            logging.info("ℹ️  Other loss weight schedulers disabled (curriculum mode active)")
            logging.info("ℹ️  Global LR scheduler is allowed (can coexist with curriculum)")
        else:
            weighters['curriculum'] = None
            if not stages:
                logging.warning("⚠️  curriculum.enable=true but no stages defined")
            if physics is None:
                logging.warning("⚠️  curriculum requires physics module, falling back to staged weights")
    else:
        weighters['curriculum'] = None
    
    # 階段式權重調度器（優先級第二）
    if 'staged_weights' in loss_cfg and loss_cfg['staged_weights'].get('enable', False):
        phases = loss_cfg['staged_weights'].get('phases', [])
        if phases:
            weighters['staged'] = StagedWeightScheduler(phases)
            logging.info(f"✅ Staged weight scheduler enabled with {len(phases)} phases")
        else:
            weighters['staged'] = None
            logging.warning("⚠️  staged_weights.enable=true but no phases defined")
    else:
        weighters['staged'] = None
    
    # GradNorm 權重器（與階段式權重互斥）
    configured_terms = loss_cfg.get('adaptive_loss_terms')
    if configured_terms is not None:
        adaptive_terms = [name for name in configured_terms if name in base_weight_template]
    else:
        adaptive_terms = default_adaptive_terms
    if loss_cfg.get('adaptive_weighting', False) and weighters['staged'] is None and adaptive_terms:
        initial_weights = {name: base_weight_template.get(name, 1.0) for name in adaptive_terms}
        weighters['gradnorm'] = GradNormWeighter(
            model=model,
            loss_names=adaptive_terms,
            alpha=loss_cfg.get('grad_norm_alpha', 0.12),
            update_frequency=loss_cfg.get('weight_update_freq', 100),
            initial_weights=initial_weights,
            device=str(device),
            min_weight=loss_cfg.get('grad_norm_min_weight', 0.1),
            max_weight=loss_cfg.get('grad_norm_max_weight', 10.0)
        )
        logging.info("GradNorm adaptive weighting enabled")
    else:
        weighters['gradnorm'] = None
        if loss_cfg.get('adaptive_weighting', False) and weighters['staged'] is not None:
            logging.info("⚠️  adaptive_weighting disabled (using staged_weights)")
    
    # 因果權重器（jaxpi: chunk-based, 僅依時間排序）
    if loss_cfg.get('causal_weighting', False):
        # 獲取時間範圍（用於預計算因果矩陣）
        kol_cfg = config['data'].get('kolmogorov_config', {})
        time_range = kol_cfg.get('time_range', [0.0, 1.0])
        t_min, t_max = time_range
        
        weighters['causal'] = CausalWeighter(
            epsilon=loss_cfg.get('causal_eps', 1.0),
            num_chunks=loss_cfg.get('causal_n_bins', 10),
            t_min=t_min,
            t_max=t_max,
            device=device  # 預計算因果矩陣到正確設備
        )
        logging.info(
            f"✅ Causal weighting enabled: ε={loss_cfg.get('causal_eps', 1.0):.2f}, "
            f"chunks={loss_cfg.get('causal_n_bins', 10)}, "
            f"time_range=[{t_min}, {t_max}], device={device}"
        )
    else:
        weighters['causal'] = None
    
    # 自適應權重調度器
    # 🔧 修復：僅在明確要求 phase_scheduling 時啟用（與 GradNorm 衝突）
    if loss_cfg.get('phase_scheduling', False) and weighters['staged'] is None and adaptive_terms:
        weighters['scheduler'] = AdaptiveWeightScheduler(
            loss_names=adaptive_terms
        )
        logging.info("Adaptive weight scheduler created")
    else:
        weighters['scheduler'] = None
        if not loss_cfg.get('phase_scheduling', False) and weighters['staged'] is None:
            logging.info("Adaptive weight scheduler disabled (use 'phase_scheduling: true' to enable)")
    
    return weighters
