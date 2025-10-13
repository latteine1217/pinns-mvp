#!/usr/bin/env python3
"""
PINNs 逆重建主訓練腳本
負責協調資料載入、模型建立、訓練迴圈與評估輸出
"""

import argparse
import logging
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import yaml

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.models.fourier_mlp import PINNNet, create_enhanced_pinn, init_siren_weights
from pinnx.models.wrappers import ScaledPINNWrapper
from pinnx.physics.scaling import VSScaler
from pinnx.physics.ns_2d import NSEquations2D
from pinnx.physics import create_vs_pinn_channel_flow, VSPINNChannelFlow  # VS-PINN
from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.train.loop import TrainingLoopManager, apply_point_weights_to_loss  # 自適應採樣管理器
from pinnx.utils.normalization import InputNormalizer, NormalizationConfig
from pinnx.evals.metrics import relative_L2

_LOSS_KEY_MAP: Dict[str, str] = {
    'data': 'data_weight',
    'momentum_x': 'momentum_x_weight',
    'momentum_y': 'momentum_y_weight',
    'momentum_z': 'momentum_z_weight',
    'continuity': 'continuity_weight',
    'wall_constraint': 'wall_constraint_weight',
    'periodicity': 'periodicity_weight',
    'inlet': 'inlet_weight',
    'initial_condition': 'initial_condition_weight',
    'bulk_velocity': 'bulk_velocity_weight',
    'centerline_dudy': 'centerline_dudy_weight',
    'centerline_v': 'centerline_v_weight',
    'pressure_reference': 'pressure_reference_weight',
}

_DEFAULT_WEIGHTS: Dict[str, float] = {
    'data': 10.0,
    'momentum_x': 2.0,
    'momentum_y': 2.0,
    'momentum_z': 2.0,
    'continuity': 2.0,
    'wall_constraint': 10.0,
    'periodicity': 5.0,
    'inlet': 10.0,
    'initial_condition': 100.0,
    'bulk_velocity': 2.0,
    'centerline_dudy': 1.0,
    'centerline_v': 1.0,
    'pressure_reference': 0.0,
}

_VS_ONLY_LOSSES = {
    'momentum_z',
    'bulk_velocity',
    'centerline_dudy',
    'centerline_v',
    'pressure_reference',
}


def derive_loss_weights(
    loss_cfg: Dict[str, Any],
    prior_weight: float,
    is_vs_pinn: bool
) -> Tuple[Dict[str, float], List[str]]:
    """
    根據配置推導基礎權重與可調整的損失項列表。
    """
    base_weights: Dict[str, float] = {}

    for name, default_val in _DEFAULT_WEIGHTS.items():
        if not is_vs_pinn and name in _VS_ONLY_LOSSES:
            continue

        cfg_key = _LOSS_KEY_MAP.get(name)
        if cfg_key is not None:
            if name == 'periodicity' and not is_vs_pinn and cfg_key not in loss_cfg:
                continue
            val = loss_cfg.get(cfg_key, default_val)
        else:
            val = default_val

        base_weights[name] = float(val)

    base_weights['prior'] = float(loss_cfg.get('prior_weight', prior_weight))

    if 'boundary_weight' in loss_cfg:
        base_weights['wall_constraint'] = base_weights.get('wall_constraint', 0.0) + float(loss_cfg['boundary_weight'])

    adaptive_terms = [name for name in base_weights if name != 'prior']
    if base_weights.get('prior', 0.0) > 0.0:
        adaptive_terms.append('prior')

    return base_weights, adaptive_terms


def _collect_coordinate_tensors(training_data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    prefixes = ['sensors', 'pde', 'bc', 'ic']
    coords: List[torch.Tensor] = []
    axes = ['x', 'y', 'z']
    for prefix in prefixes:
        components = []
        for axis in axes:
            key = f'{axis}_{prefix}'
            if key in training_data and training_data[key].numel() > 0:
                components.append(training_data[key])
        if components:
            coords.append(torch.cat(components, dim=1))
    return coords


def create_input_normalizer(
    config: Dict[str, Any],
    training_data: Dict[str, torch.Tensor],
    is_vs_pinn: bool,
    device: torch.device
) -> Optional[InputNormalizer]:
    scaling_cfg = config.get('model', {}).get('scaling', {})
    norm_type = scaling_cfg.get('input_norm', 'none')
    if norm_type is None:
        norm_type = 'none'

    norm_type = norm_type.lower()

    if norm_type in ('none', 'identity'):
        return None

    if is_vs_pinn and norm_type in ('vs_pinn', 'channel_flow'):
        # VS-PINN already applies dedicated scaling; avoid double normalization.
        return None

    bounds_tensor: Optional[torch.Tensor] = None
    if norm_type in ('channel_flow',):
        domain = config.get('physics', {}).get('domain', {})
        bounds: List[Tuple[float, float]] = []
        for axis in ['x', 'y', 'z']:
            rng = domain.get(f'{axis}_range')
            if rng is not None:
                bounds.append((float(rng[0]), float(rng[1])))
        if bounds:
            bounds_tensor = torch.tensor(bounds, dtype=torch.float32, device=device)

    feature_range = tuple(scaling_cfg.get('input_norm_range', [-1.0, 1.0]))
    config_obj = NormalizationConfig(
        norm_type=norm_type,
        feature_range=(float(feature_range[0]), float(feature_range[1])),
        bounds=bounds_tensor
    )
    normalizer = InputNormalizer(config_obj)

    coord_tensors = _collect_coordinate_tensors(training_data)
    if coord_tensors:
        samples = torch.cat(coord_tensors, dim=0)
        if normalizer.bounds is not None and normalizer.bounds.shape[0] > samples.shape[1]:
            normalizer.bounds = normalizer.bounds[:samples.shape[1], :]
        normalizer.fit(samples)
    else:
        return None

    normalizer.to(device)
    return normalizer


# ============================================================================
# Warmup + CosineAnnealing 學習率調度器
# ============================================================================
class WarmupCosineScheduler:
    """
    Warmup + CosineAnnealing 學習率調度器
    
    前 warmup_epochs 個 epoch 線性增加學習率從 0 到 base_lr
    之後使用 CosineAnnealing 衰減到 min_lr
    """
    
    def __init__(self, optimizer, warmup_epochs: int, max_epochs: int, 
                 base_lr: float, min_lr: float = 0.0):
        """
        Args:
            optimizer: PyTorch 優化器
            warmup_epochs: Warmup 階段的 epoch 數量
            max_epochs: 總訓練 epoch 數量
            base_lr: 基礎學習率（Warmup 後的峰值）
            min_lr: 最小學習率（CosineAnnealing 的下限）
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0
        
        # 建立內部的 CosineAnnealing 調度器（用於 Warmup 後階段）
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
        logging.info(f"✅ WarmupCosineScheduler initialized:")
        logging.info(f"   Warmup epochs: {warmup_epochs}")
        logging.info(f"   Max epochs: {max_epochs}")
        logging.info(f"   Base LR: {base_lr:.6f}")
        logging.info(f"   Min LR: {min_lr:.6f}")
    
    def step(self):
        """更新學習率"""
        if self.current_epoch < self.warmup_epochs:
            # Warmup 階段：線性增加
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        else:
            # CosineAnnealing 階段
            self.cosine_scheduler.step()
        
        self.current_epoch += 1
    
    def get_last_lr(self):
        """返回當前學習率（兼容性接口）"""
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# ============================================================================
# 階段式權重調度器
# ============================================================================
class StagedWeightScheduler:
    """階段式權重調度器 - 根據 epoch 切換不同訓練階段的權重"""
    
    def __init__(self, phases: list):
        """
        Args:
            phases: 階段配置列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end] epoch 範圍
                - weights: 該階段的權重字典
        """
        self.phases = phases
        self.current_phase_idx = 0
        self.current_phase_name = phases[0]['name'] if phases else "default"
        
        # 按 epoch_range 排序
        self.phases.sort(key=lambda p: p['epoch_range'][0])
        
        logging.info(f"✅ StagedWeightScheduler initialized with {len(phases)} phases:")
        for p in self.phases:
            logging.info(f"   {p['name']}: Epoch {p['epoch_range'][0]}-{p['epoch_range'][1]}")
    
    def get_phase_weights(self, epoch: int) -> tuple:
        """
        獲取當前 epoch 對應的權重
        
        Returns:
            (weights_dict, phase_name, is_transition)
        """
        # 找到當前 epoch 所屬階段
        for idx, phase in enumerate(self.phases):
            start, end = phase['epoch_range']
            if start <= epoch < end:
                # 檢測是否為階段切換點
                is_transition = (idx != self.current_phase_idx)
                self.current_phase_idx = idx
                self.current_phase_name = phase['name']
                
                return phase['weights'], phase['name'], is_transition
        
        # 如果超出所有階段，返回最後階段
        last_phase = self.phases[-1]
        return last_phase['weights'], last_phase['name'], False


# 課程訓練調度器 - 支援雷諾數動態變化
class CurriculumScheduler:
    """
    課程訓練調度器 - 逐步提升雷諾數，從層流到湍流
    
    特性：
    - 動態調整 Re_tau, nu, pressure_gradient
    - 階段式權重切換
    - 階段式學習率調整
    - 階段式採樣策略調整
    """
    
    def __init__(self, stages: list, physics_module):
        """
        Args:
            stages: 課程階段列表，每個元素包含:
                - name: 階段名稱
                - epoch_range: [start, end]
                - Re_tau: 雷諾數
                - nu: 黏度
                - pressure_gradient: 壓力梯度
                - weights: 損失權重字典
                - sampling: 採樣配置
                - lr: 學習率
            physics_module: 物理方程模組（用於更新參數）
        """
        self.stages = stages
        self.physics = physics_module
        self.current_stage_idx = 0
        self.current_stage = stages[0] if stages else None
        
        # 按 epoch_range 排序
        self.stages.sort(key=lambda s: s['epoch_range'][0])
        
        logging.info("="*80)
        logging.info("🚀 CurriculumScheduler initialized - Progressive Reynolds Number Training")
        logging.info("="*80)
        for i, s in enumerate(self.stages, 1):
            logging.info(f"Stage {i}: {s['name']}")
            logging.info(f"  Epochs: {s['epoch_range'][0]}-{s['epoch_range'][1]}")
            logging.info(f"  Re_tau: {s['Re_tau']:.1f}, nu: {s['nu']:.6f}, dP/dx: {s['pressure_gradient']:.3f}")
            logging.info(f"  PDE points: {s['sampling']['pde_points']}, BC points: {s['sampling']['boundary_points']}")
            logging.info(f"  Learning rate: {s['lr']:.6f}")
        logging.info("="*80)
    
    def get_stage_config(self, epoch: int) -> Dict[str, Any]:
        """
        獲取當前 epoch 對應的階段配置
        
        Returns:
            {
                'stage_name': str,
                'is_transition': bool,
                'weights': dict,
                'Re_tau': float,
                'nu': float,
                'pressure_gradient': float,
                'sampling': dict,
                'lr': float
            }
        """
        # 找到當前階段
        for idx, stage in enumerate(self.stages):
            start, end = stage['epoch_range']
            if start <= epoch < end:
                # 檢測階段切換
                is_transition = (idx != self.current_stage_idx)
                
                if is_transition:
                    self.current_stage_idx = idx
                    self.current_stage = stage
                    
                    # 更新物理參數
                    self._update_physics_parameters(stage)
                    
                    logging.info("="*80)
                    logging.info(f"🎯 CURRICULUM STAGE TRANSITION at Epoch {epoch}")
                    logging.info(f"📚 New Stage: {stage['name']}")
                    logging.info(f"🔬 Re_tau: {stage['Re_tau']:.1f}, nu: {stage['nu']:.6f}")
                    logging.info(f"⚙️  PDE/BC points: {stage['sampling']['pde_points']}/{stage['sampling']['boundary_points']}")
                    logging.info(f"📊 Weights: {stage['weights']}")
                    logging.info(f"📉 Learning rate: {stage['lr']:.6f}")
                    logging.info("="*80)
                
                return {
                    'stage_name': stage['name'],
                    'is_transition': is_transition,
                    'weights': stage['weights'],
                    'Re_tau': stage['Re_tau'],
                    'nu': stage['nu'],
                    'pressure_gradient': stage['pressure_gradient'],
                    'sampling': stage['sampling'],
                    'lr': stage['lr']
                }
        
        # 超出範圍，返回最後階段
        last_stage = self.stages[-1]
        return {
            'stage_name': last_stage['name'],
            'is_transition': False,
            'weights': last_stage['weights'],
            'Re_tau': last_stage['Re_tau'],
            'nu': last_stage['nu'],
            'pressure_gradient': last_stage['pressure_gradient'],
            'sampling': last_stage['sampling'],
            'lr': last_stage['lr']
        }
    
    def _update_physics_parameters(self, stage: Dict[str, Any]):
        """更新物理方程模組的參數"""
        if hasattr(self.physics, 'nu'):
            self.physics.nu = stage['nu']
        if hasattr(self.physics, 'Re_tau'):
            self.physics.Re_tau = stage['Re_tau']
        if hasattr(self.physics, 'pressure_gradient'):
            self.physics.pressure_gradient = stage['pressure_gradient']
        
        logging.debug(f"✅ Physics parameters updated: Re_tau={stage['Re_tau']}, nu={stage['nu']}")

# 全域快取，用於存儲 Channel Flow 資料和統計資訊
_channel_data_cache: Optional[Dict[str, Any]] = None

# 模型檢查點保存與載入功能
def save_checkpoint(model, optimizer, epoch, loss, config, checkpoint_dir="checkpoints"):
    """保存模型檢查點"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config
    }
    
    experiment_name = config['experiment']['name']
    checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    
    # 同時保存最新的檢查點
    latest_path = os.path.join(checkpoint_dir, f"{experiment_name}_latest.pth")
    torch.save(checkpoint, latest_path)
    
    return checkpoint_path

def load_checkpoint(checkpoint_path, model, optimizer=None):
    """載入模型檢查點"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint.get('config')


def setup_logging(level: str = "info") -> logging.Logger:
    """設置日誌系統"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('pinnx.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """設置隨機種子確保重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """載入YAML配置檔案"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return normalize_config_structure(config)


def normalize_config_structure(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    標準化配置結構，支持嵌套和扁平兩種格式
    
    處理以下兼容性問題：
    1. model.fourier.{enabled, m, sigma} → model.{use_fourier, fourier_m, fourier_sigma}
    2. 確保所有必要字段都有默認值
    """
    model_cfg = config.get('model', {})
    
    # 處理 Fourier 配置（嵌套格式 → 扁平格式）
    if 'fourier' in model_cfg and isinstance(model_cfg['fourier'], dict):
        fourier_cfg = model_cfg['fourier']
        
        # 映射 enabled → use_fourier
        if 'enabled' in fourier_cfg and 'use_fourier' not in model_cfg:
            model_cfg['use_fourier'] = fourier_cfg['enabled']
        
        # 映射 m → fourier_m
        if 'm' in fourier_cfg and 'fourier_m' not in model_cfg:
            model_cfg['fourier_m'] = fourier_cfg['m']
        
        # 映射 sigma → fourier_sigma
        if 'sigma' in fourier_cfg and 'fourier_sigma' not in model_cfg:
            model_cfg['fourier_sigma'] = fourier_cfg['sigma']
        
        # 映射 trainable → fourier_trainable
        if 'trainable' in fourier_cfg and 'fourier_trainable' not in model_cfg:
            model_cfg['fourier_trainable'] = fourier_cfg['trainable']
        
        logging.info("✅ Normalized nested fourier config to flat structure")
    
    # 設置默認值（如果未設置）
    model_cfg.setdefault('use_fourier', True)  # 默認啟用 Fourier
    model_cfg.setdefault('fourier_m', 32)
    model_cfg.setdefault('fourier_sigma', 1.0)
    model_cfg.setdefault('fourier_trainable', False)
    
    config['model'] = model_cfg
    return config


def get_device(device_name: str) -> torch.device:
    """獲取運算設備"""
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
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        logging.info(f"Using CUDA: {torch.cuda.get_device_name()}")
    elif device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
        logging.info("Using Apple Metal Performance Shaders")
    else:
        device = torch.device("cpu")
        logging.info("Using CPU")
    
    return device


def create_model(config: Dict[str, Any], device: torch.device, statistics: Optional[Dict[str, Dict[str, float]]] = None) -> nn.Module:
    """建立 PINN 模型
    
    Args:
        config: 配置字典
        device: 計算設備
        statistics: 資料統計資訊（可選），用於自動設定輸出範圍
    """
    model_cfg = config['model']
    
    # 🔧 從配置讀取 Fourier 開關（支持消融實驗）
    use_fourier = model_cfg.get('use_fourier', True)  # 默認啟用
    
    # 🔧 檢查是否為 VS-PINN，若是則準備縮放因子（用於 Fourier 標準化修復）
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_pinn = (physics_type == 'vs_pinn_channel_flow')
    
    # 提取 VS-PINN 縮放因子（如果是 VS-PINN）
    input_scale_factors = None
    fourier_normalize_input = False
    if is_vs_pinn and use_fourier:
        vs_pinn_cfg = config.get('physics', {}).get('vs_pinn', {})
        scaling_cfg = vs_pinn_cfg.get('scaling_factors', {})
        N_x = scaling_cfg.get('N_x', 2.0)
        N_y = scaling_cfg.get('N_y', 12.0)
        N_z = scaling_cfg.get('N_z', 2.0)
        input_scale_factors = torch.tensor([N_x, N_y, N_z], dtype=torch.float32)
        fourier_normalize_input = True  # 🔧 啟用 Fourier 前標準化修復
        logging.info(f"🔧 VS-PINN + Fourier 修復啟用：縮放因子 N=[{N_x}, {N_y}, {N_z}]")
    
    # 檢查是否使用增強版模型
    if model_cfg.get('type') == 'enhanced_fourier_mlp':
        # 使用增強版 PINN 網路 (統一到 PINNNet，透過參數控制)
        base_model = create_enhanced_pinn(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=use_fourier,  # ✅ 遵守配置開關
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
            use_rwf=model_cfg.get('use_rwf', False),
            rwf_scale_std=model_cfg.get('rwf_scale_std', 0.1),
            fourier_normalize_input=fourier_normalize_input,  # 🔧 新參數
            input_scale_factors=input_scale_factors  # 🔧 新參數
        ).to(device)
    else:
        # 使用基礎 PINN 網路
        base_model = PINNNet(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            use_fourier=use_fourier,  # ✅ 遵守配置開關
            fourier_m=model_cfg.get('fourier_m', 32),
            fourier_sigma=model_cfg.get('fourier_sigma', 1.0),
            fourier_normalize_input=fourier_normalize_input,  # 🔧 新參數
            input_scale_factors=input_scale_factors  # 🔧 新參數
        ).to(device)
    
    # 檢查是否啟用 VS-PINN 尺度化或手動標準化
    scaling_cfg = model_cfg.get('scaling', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_pinn = (physics_type == 'vs_pinn_channel_flow')
    
    # 🔧 修復：VS-PINN 使用自己的 scale_coordinates，跳過 ManualScalingWrapper
    #    避免雙重標準化破壞 Fourier features 和 VS-PINN 縮放
    scaling_enabled = bool(scaling_cfg) and not is_vs_pinn
    
    if scaling_enabled:
        # 暫時使用手動標準化包裝器 (避免 Fourier feature 飽和)
        from pinnx.models.wrappers import ManualScalingWrapper
        
        # 輸入範圍：優先使用配置文件的域範圍，確保模型能泛化到完整域
        # 🔥 修復：使用 physics.domain 而非感測點統計，避免訓練範圍過小
        domain_cfg = config.get('physics', {}).get('domain', {})
        
        if domain_cfg and 'x_range' in domain_cfg:
            # 優先：從配置文件讀取完整域範圍
            input_x_range = tuple(domain_cfg['x_range'])  # type: ignore
            input_y_range = tuple(domain_cfg['y_range'])  # type: ignore
            logging.info(f"✅ Using domain ranges from config: x={input_x_range}, y={input_y_range}")
        elif statistics and 'x' in statistics and 'range' in statistics['x']:
            # 回退：使用感測點統計（僅用於無配置的 legacy 情況）
            input_x_range = tuple(statistics['x']['range'])  # type: ignore
            input_y_range = tuple(statistics['y']['range'])  # type: ignore
            logging.warning(f"⚠️ Falling back to statistics-based ranges: x={input_x_range}, y={input_y_range}")
            logging.warning(f"⚠️ This may cause generalization issues if sensors don't cover full domain!")
        else:
            # 最終回退：硬編碼（JHTDB Channel Re1000）
            input_x_range = (0.0, 25.13)
            input_y_range = (-1.0, 1.0)
            logging.warning(f"⚠️ Using hardcoded domain ranges: x={input_x_range}, y={input_y_range}")
        
        input_scales: Dict[str, Tuple[float, float]] = {
            'x': input_x_range,
            'y': input_y_range
        }
        
        # 🔥 3D 自動檢測：優先使用配置，其次統計
        if domain_cfg and 'z_range' in domain_cfg:
            input_z_range = tuple(domain_cfg['z_range'])  # type: ignore
            input_scales['z'] = input_z_range
            logging.info(f"✅ 3D mode: z={input_z_range}")
        elif statistics and 'z' in statistics and 'range' in statistics['z']:
            input_z_range = tuple(statistics['z']['range'])  # type: ignore
            input_scales['z'] = input_z_range
            logging.warning(f"⚠️ 3D mode using statistics: z={input_z_range}")
        
        # 🔥 修復：優先使用配置文件（用戶顯式指定），然後才是統計資訊
        # 兼容性處理：支持字典格式和字符串格式的 output_norm
        output_scales: Dict[str, Tuple[float, float]] | None = None
        
        if 'output_norm' in scaling_cfg:
            output_norm_raw = scaling_cfg['output_norm']
            
            if isinstance(output_norm_raw, dict):
                # 字典格式：直接使用
                output_scales = {
                    'u': tuple(output_norm_raw.get('u', [0.0, 20.0])),   # type: ignore
                    'v': tuple(output_norm_raw.get('v', [-1.0, 1.0])),   # type: ignore
                    'w': tuple(output_norm_raw.get('w', [-5.0, 5.0])),   # type: ignore  # 🔥 修復：添加 w
                    'p': tuple(output_norm_raw.get('p', [-100.0, 10.0])) # type: ignore
                }
                logging.info(f"✅ Using output ranges from config file (dict format):")
                logging.info(f"   u: {output_scales['u']}")
                logging.info(f"   v: {output_scales['v']}")
                logging.info(f"   w: {output_scales['w']}")  # 🔥 添加日誌
                logging.info(f"   p: {output_scales['p']}")
            elif isinstance(output_norm_raw, str):
                # 字符串格式：使用統計信息自動推導（如 "friction_velocity"）
                logging.info(f"⚠️  output_norm is string '{output_norm_raw}', falling back to statistics")
        
        # 如果沒有從配置獲取到有效的輸出範圍，使用統計資訊
        if output_scales is None:
            if statistics:
                # 回退：統計信息（可能受限於傳感器點數）
                if 'u' in statistics and 'range' in statistics['u']:
                    output_u_range = tuple(statistics['u']['range'])  # type: ignore
                else:
                    output_u_range = (0.0, 20.0)
                    
                if 'v' in statistics and 'range' in statistics['v']:
                    output_v_range = tuple(statistics['v']['range'])  # type: ignore
                else:
                    output_v_range = (-1.0, 1.0)
                
                # 🔥 修復：添加 w 的處理
                if 'w' in statistics and 'range' in statistics['w']:
                    output_w_range = tuple(statistics['w']['range'])  # type: ignore
                else:
                    output_w_range = (-5.0, 5.0)
                    
                if 'p' in statistics and 'range' in statistics['p']:
                    output_p_range = tuple(statistics['p']['range'])  # type: ignore
                else:
                    output_p_range = (-100.0, 10.0)
                
                output_scales = {
                    'u': output_u_range,
                    'v': output_v_range,
                    'w': output_w_range,  # 🔥 添加 w
                    'p': output_p_range
                }
                logging.info(f"✅ Using data-driven output ranges from statistics:")
                logging.info(f"   u: {output_scales['u']}")
                logging.info(f"   v: {output_scales['v']}")
                logging.info(f"   w: {output_scales['w']}")  # 🔥 添加日誌
                logging.info(f"   p: {output_scales['p']}")
            else:
                # 最終回退到硬編碼範圍
                output_scales = {
                    'u': (0.0, 20.0),      # 速度範圍 (基於 JHTDB Channel Re1000 實際資料)
                    'v': (-1.0, 1.0),      # 法向速度範圍
                    'w': (-5.0, 5.0),      # 🔥 添加 w：展向速度範圍
                    'p': (-100.0, 10.0)    # 壓力範圍 (JHTDB 實際範圍約 -78 到 0.3)
                }
                logging.warning("⚠️  No statistics or config output_norm provided, using hardcoded ranges (may cause NaN)")
        
        # 🔥 重要：根據模型輸出維度補充缺失的範圍（如源項 S）
        expected_out_dim = model_cfg.get('out_dim', 3)
        # 🔥 修復：只有當確實缺少輸出變量時才補充
        # 3D 模型輸出順序：u, v, w, p (out_dim=4)
        # 如果有源項則為：u, v, w, p, S (out_dim=5)
        if expected_out_dim == 5 and len(output_scales) == 4:
            # 補充源項 S 的範圍（通常用於逆問題）
            output_scales['S'] = (-1.0, 1.0)  # 初始估計，可根據實際調整
            logging.info("✅ Added source term 'S' range: (-1.0, 1.0)")
        
        try:
            model = ManualScalingWrapper(
                base_model, 
                input_ranges=input_scales,
                output_ranges=output_scales
            ).to(device)
            logging.info(f"✅ Manual scaling wrapper applied: inputs {input_scales}, outputs {output_scales}")
        except ImportError:
            logging.warning("ManualScalingWrapper not found, using base model. Consider implementing it.")
            model = base_model
    else:
        # 直接使用基礎模型
        model = base_model
        logging.info("Using base model without scaling")
    
    # 🔥 應用 SIREN 初始化（若使用 Sine 激活函數）
    if model_cfg['activation'] == 'sine':
        # 若有 wrapper，需要找到底層的 PINNNet
        target_model = base_model
        if hasattr(model, 'model'):  # ManualScalingWrapper 或其他包裝器
            target_model = model.model  # type: ignore
        
        if isinstance(target_model, PINNNet):
            init_siren_weights(target_model)
            logging.info("✅ Applied SIREN weight initialization for Sine activation")
        else:
            logging.warning(f"⚠️  Cannot apply SIREN initialization: base model type is {type(target_model)}")
    
    # 🔥 訓練前驗證：檢查縮放參數是否與配置一致
    if hasattr(model, 'input_min') and hasattr(model, 'input_max'):
        logging.info("=" * 60)
        logging.info("📐 Model Scaling Parameters Verification:")
        logging.info(f"   Input min:  {model.input_min.cpu().numpy()}")
        logging.info(f"   Input max:  {model.input_max.cpu().numpy()}")
        logging.info(f"   Output min: {model.output_min.cpu().numpy()}")
        logging.info(f"   Output max: {model.output_max.cpu().numpy()}")
        
        # 驗證輸入範圍是否匹配配置
        domain_cfg = config.get('physics', {}).get('domain', {})
        if domain_cfg and 'x_range' in domain_cfg:
            expected_x_range = domain_cfg['x_range']
            expected_y_range = domain_cfg['y_range']
            
            actual_x_min = model.input_min[0].item()
            actual_x_max = model.input_max[0].item()
            actual_y_min = model.input_min[1].item()
            actual_y_max = model.input_max[1].item()
            
            # 容差檢查（1e-3）
            x_match = abs(actual_x_min - expected_x_range[0]) < 1e-3 and \
                      abs(actual_x_max - expected_x_range[1]) < 1e-3
            y_match = abs(actual_y_min - expected_y_range[0]) < 1e-3 and \
                      abs(actual_y_max - expected_y_range[1]) < 1e-3
            
            if x_match and y_match:
                logging.info(f"✅ Input ranges match config: x={expected_x_range}, y={expected_y_range}")
            else:
                logging.error("=" * 60)
                logging.error("❌ CRITICAL: Input range mismatch detected!")
                logging.error(f"   Expected x: {expected_x_range}, got: [{actual_x_min:.4f}, {actual_x_max:.4f}]")
                logging.error(f"   Expected y: {expected_y_range}, got: [{actual_y_min:.4f}, {actual_y_max:.4f}]")
                logging.error("   This will cause generalization failure outside sensor coverage!")
                logging.error("=" * 60)
                raise ValueError("Input scaling range configuration error - model cannot generalize to full domain")
            
            # 3D 檢查
            if 'z_range' in domain_cfg and len(model.input_min) > 2:
                expected_z_range = domain_cfg['z_range']
                actual_z_min = model.input_min[2].item()
                actual_z_max = model.input_max[2].item()
                z_match = abs(actual_z_min - expected_z_range[0]) < 1e-3 and \
                          abs(actual_z_max - expected_z_range[1]) < 1e-3
                if z_match:
                    logging.info(f"✅ 3D z-range matches config: {expected_z_range}")
                else:
                    logging.error(f"❌ Expected z: {expected_z_range}, got: [{actual_z_min:.4f}, {actual_z_max:.4f}]")
                    raise ValueError("Z-axis scaling range configuration error")
        logging.info("=" * 60)
    
    logging.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def create_physics(config: Dict[str, Any], device: torch.device):
    """建立物理方程式模組（支援 VS-PINN）"""
    physics_cfg = config['physics']
    physics_type = physics_cfg.get('type', 'ns_2d')  # 默認使用 NS 2D
    
    if physics_type == 'vs_pinn_channel_flow':
        # VS-PINN 通道流求解器
        # 🔧 修復：從正確的巢狀路徑讀取配置（physics.vs_pinn.scaling_factors）
        vs_pinn_cfg = physics_cfg.get('vs_pinn', {})
        scaling_cfg = vs_pinn_cfg.get('scaling_factors', {})
        
        # 物理參數可能在多個位置（兼容舊配置）
        channel_flow_cfg = physics_cfg.get('channel_flow', {})
        
        # 域配置：優先使用 physics.domain（新格式），回退到 vs_pinn 內部
        domain_cfg = physics_cfg.get('domain', {})
        
        # 🔧 修復：解析域邊界（使用物理座標範圍，非標準化座標）
        domain_bounds = {
            'x': domain_cfg.get('x_range', [0.0, 25.13]),  # 物理座標
            'y': domain_cfg.get('y_range', [-1.0, 1.0]),
            'z': domain_cfg.get('z_range', [0.0, 9.42]),
        }
        
        physics = create_vs_pinn_channel_flow(
            N_x=scaling_cfg.get('N_x', 2.0),
            N_y=scaling_cfg.get('N_y', 12.0),
            N_z=scaling_cfg.get('N_z', 2.0),
            nu=physics_cfg.get('nu', channel_flow_cfg.get('u_tau', 5e-5)),  # 兼容多種格式
            dP_dx=channel_flow_cfg.get('pressure_gradient', physics_cfg.get('dP_dx', 0.0025)),
            rho=physics_cfg.get('rho', 1.0),
            domain_bounds=domain_bounds,
            loss_config=config.get('losses', {}),  # 🔴 傳遞損失配置（包含 warmup_epochs）
        )
        logging.info(f"✅ 使用 VS-PINN 求解器 (N_x={scaling_cfg.get('N_x', 2.0)}, N_y={scaling_cfg.get('N_y', 12.0)}, N_z={scaling_cfg.get('N_z', 2.0)})")
        logging.info(f"   域邊界: x={domain_bounds['x']}, y={domain_bounds['y']}, z={domain_bounds['z']}")
    else:
        # 標準 NS 2D 求解器
        physics = NSEquations2D(
            viscosity=physics_cfg.get('nu', 1e-3),
            density=physics_cfg.get('rho', 1.0)
        )
        logging.info(f"✅ 使用標準 NS 2D 求解器")
    
    # 🔧 修復：將 physics 模組移動到正確的設備（避免 CPU/CUDA 張量混合）
    physics = physics.to(device)
    logging.info(f"   Physics module moved to device: {device}")
    
    return physics


def create_loss_functions(config: Dict[str, Any], device: torch.device) -> Dict[str, nn.Module]:
    """建立損失函數"""
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
            density=loss_cfg.get('rho', 1.0)
        ),
        'boundary': BoundaryConditionLoss(),  # 🆕 邊界條件損失（含 inlet）
        'prior': PriorLossManager(
            consistency_weight=loss_cfg['prior_weight']
        )
    }
    
    return losses


def create_weighters(config: Dict[str, Any], model: nn.Module, device: torch.device, physics=None) -> Dict[str, Any]:
    """建立動態權重器 (需要模型實例)"""
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
            # 課程訓練啟用時，禁用其他調度器
            weighters['staged'] = None
            weighters['gradnorm'] = None
            weighters['scheduler'] = None
            weighters['causal'] = None
            logging.info("⚠️  Other schedulers disabled (curriculum mode active)")
            return weighters
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
    
    # 因果權重器
    if loss_cfg.get('causal_weighting', False):
        weighters['causal'] = CausalWeighter(
            causality_strength=loss_cfg.get('causal_eps', 1.0)
        )
        logging.info("Causal weighting enabled")
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


def prepare_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """準備訓練資料 - 支援 JHTDB Channel Flow 或 Mock 資料"""
    
    # 檢查是否使用 JHTDB Channel Flow 載入器
    jhtdb_enabled = config.get('data', {}).get('jhtdb_config', {}).get('enabled', False)
    channel_flow_enabled = 'channel_flow' in config and config['channel_flow'].get('enabled', False)
    
    if jhtdb_enabled or channel_flow_enabled:
        return prepare_channel_flow_training_data(config, device)
    else:
        return prepare_mock_training_data(config, device)


def _apply_validation_split(
    training_dict: Dict[str, torch.Tensor],
    validation_split: float,
    is_vs_pinn: bool
) -> Dict[str, torch.Tensor]:
    """
    依據 validation_split 將感測點資料切分成訓練/驗證集合，並在 training_dict 中新增
    'validation' 索引。
    """
    if validation_split is None or validation_split <= 0.0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    x_sensors = training_dict.get('x_sensors')
    if x_sensors is None or x_sensors.shape[0] == 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    n_total = x_sensors.shape[0]
    if n_total < 2:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    n_val = max(1, int(round(n_total * validation_split)))
    if n_val >= n_total:
        # 至少保留一個訓練感測點
        n_val = max(1, n_total - 1)
    if n_val <= 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    device = x_sensors.device
    perm = torch.randperm(n_total, device=device)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    
    if train_idx.numel() == 0:
        # 當 split 幾乎為 1.0 時，確保仍有訓練資料
        train_idx = val_idx[-1:].clone()
        val_idx = val_idx[:-1]
    
    def split_tensor(tensor: Optional[torch.Tensor]) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if tensor is None:
            return None, None
        if tensor.shape[0] == 0:
            return tensor, tensor
        return tensor[train_idx], tensor[val_idx]
    
    x_train, x_val = split_tensor(training_dict.get('x_sensors'))
    y_train, y_val = split_tensor(training_dict.get('y_sensors'))
    z_train, z_val = split_tensor(training_dict.get('z_sensors'))
    t_train, t_val = split_tensor(training_dict.get('t_sensors'))
    u_train, u_val = split_tensor(training_dict.get('u_sensors'))
    v_train, v_val = split_tensor(training_dict.get('v_sensors'))
    w_train, w_val = split_tensor(training_dict.get('w_sensors'))
    p_train, p_val = split_tensor(training_dict.get('p_sensors'))
    
    # 更新訓練感測資料
    if x_train is not None:
        training_dict['x_sensors'] = x_train
    if y_train is not None:
        training_dict['y_sensors'] = y_train
    if z_train is not None:
        training_dict['z_sensors'] = z_train
    if t_train is not None:
        training_dict['t_sensors'] = t_train
    if u_train is not None:
        training_dict['u_sensors'] = u_train
    if v_train is not None:
        training_dict['v_sensors'] = v_train
    if w_train is not None:
        training_dict['w_sensors'] = w_train
    if p_train is not None:
        training_dict['p_sensors'] = p_train
    
    # 建立驗證資料
    if x_val is None or x_val.shape[0] == 0:
        training_dict['validation'] = {'size': 0}
        return training_dict
    
    coord_parts = [x_val, y_val]
    if is_vs_pinn and z_val is not None and z_val.shape[0] > 0:
        coord_parts.append(z_val)
    validation_coords = torch.cat(coord_parts, dim=1).detach()
    
    target_parts = [u_val, v_val]
    component_order = ['u', 'v']
    if is_vs_pinn:
        if w_val is not None and w_val.shape[0] == validation_coords.shape[0]:
            target_parts.append(w_val)
        else:
            target_parts.append(torch.zeros_like(u_val))
        component_order.append('w')
    target_parts.append(p_val)
    component_order.append('p')
    validation_targets = torch.cat(target_parts, dim=1).detach()
    
    if t_val is not None:
        validation_time = t_val.detach()
    else:
        validation_time = None
    
    training_dict['validation'] = {
        'coords': validation_coords,
        'targets': validation_targets,
        'time': validation_time,
        'components': component_order,
        'size': int(validation_coords.shape[0])
    }
    
    return training_dict


def prepare_mock_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """建立 Mock 訓練資料用於測試整合"""
    
    # 從配置中讀取參數
    K = config['sensors']['K']
    sampling = config['training']['sampling']
    physics_cfg = config['physics']
    domain = physics_cfg['domain']
    
    # 定義域範圍
    x_range = domain['x_range']
    y_range = domain['y_range']
    
    # 生成感測器點 (均勻分佈)
    x_sensors = torch.rand(K, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_sensors = torch.rand(K, 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_sensors = torch.zeros_like(x_sensors)  # 假設穩態
    
    # 生成 Mock 速度和壓力資料 (基於解析解或簡單模式)
    # 簡單的通道流模式: u = U_max * (1 - (2y/H - 1)^2), v = 0, p = 線性分佈
    y_norm = (y_sensors - y_range[0]) / (y_range[1] - y_range[0])  # 歸一化到 [0,1]
    y_centered = 2 * y_norm - 1  # 歸一化到 [-1,1]
    
    u_max = 1.0  # 最大速度
    u_sensors = u_max * (1 - y_centered**2)  # 拋物線型速度分佈
    v_sensors = torch.zeros_like(u_sensors)   # 垂直速度為零
    p_sensors = torch.ones_like(u_sensors) * 0.1  # 簡單的壓力場
    
    # 生成 PDE 殘差點
    x_pde = torch.rand(sampling['pde_points'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_pde = torch.rand(sampling['pde_points'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
    t_pde = torch.zeros_like(x_pde)  # 穩態假設
    
    # 生成邊界點 (上下壁面)
    n_bc = sampling['boundary_points']
    x_bc = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
    y_bc_bottom = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
    y_bc_top = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
    y_bc = torch.cat([y_bc_bottom, y_bc_top], dim=0)
    x_bc = torch.cat([x_bc[:n_bc//2], x_bc[n_bc//2:]], dim=0)
    t_bc = torch.zeros_like(x_bc)
    
    logging.info(f"Mock training data generated: K={K} sensors, {sampling['pde_points']} PDE points, {n_bc} BC points")
    
    training_dict = {
        'x_pde': x_pde, 'y_pde': y_pde, 't_pde': t_pde,
        'x_bc': x_bc, 'y_bc': y_bc, 't_bc': t_bc,
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 't_sensors': t_sensors,
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'p_sensors': p_sensors
    }
    
    validation_split = config.get('training', {}).get('validation_split', 0.0)
    training_dict = _apply_validation_split(training_dict, validation_split, is_vs_pinn=False)
    
    return training_dict


def sample_boundary_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    distribution: Optional[Dict[str, int]] = None
) -> torch.Tensor:
    """
    在邊界上均勻採樣點
    
    Args:
        n_points: 總邊界點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        distribution: 邊界點分佈 {'wall': int, 'periodic': int, 'inlet': int}
                     如果為 None，預設為 {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    Returns:
        邊界點座標 [n_points, 3] (x, y, z)
    """
    if distribution is None:
        distribution = {'wall': 1000, 'periodic': 800, 'inlet': 200}
    
    # 驗證總點數
    total_requested = sum(distribution.values())
    if total_requested != n_points:
        logging.warning(f"⚠️ Boundary distribution sum ({total_requested}) != n_points ({n_points}), 自動調整比例")
        # 按比例調整
        scale = n_points / total_requested
        distribution = {k: int(v * scale) for k, v in distribution.items()}
        # 修正舍入誤差
        diff = n_points - sum(distribution.values())
        distribution['wall'] += diff
    
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    boundary_points = []
    
    # 1. 壁面點 (y = y_min 和 y = y_max)
    n_wall = distribution['wall']
    n_wall_bottom = n_wall // 2
    n_wall_top = n_wall - n_wall_bottom
    
    # 下壁面 (y = y_min)
    x_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (x_max - x_min) + x_min
    y_wall_bottom = torch.full((n_wall_bottom, 1), y_min, device=device)
    z_wall_bottom = torch.rand(n_wall_bottom, 1, device=device) * (z_max - z_min) + z_min
    wall_bottom = torch.cat([x_wall_bottom, y_wall_bottom, z_wall_bottom], dim=1)
    
    # 上壁面 (y = y_max)
    x_wall_top = torch.rand(n_wall_top, 1, device=device) * (x_max - x_min) + x_min
    y_wall_top = torch.full((n_wall_top, 1), y_max, device=device)
    z_wall_top = torch.rand(n_wall_top, 1, device=device) * (z_max - z_min) + z_min
    wall_top = torch.cat([x_wall_top, y_wall_top, z_wall_top], dim=1)
    
    boundary_points.extend([wall_bottom, wall_top])
    
    # 2. 週期性邊界點 (x = x_min/x_max, z = z_min/z_max)
    n_periodic = distribution['periodic']
    n_per_face = n_periodic // 4  # 4 個面：x_min, x_max, z_min, z_max
    
    # x = x_min
    x_left = torch.full((n_per_face, 1), x_min, device=device)
    y_left = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_left = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_left = torch.cat([x_left, y_left, z_left], dim=1)
    
    # x = x_max
    x_right = torch.full((n_per_face, 1), x_max, device=device)
    y_right = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_right = torch.rand(n_per_face, 1, device=device) * (z_max - z_min) + z_min
    periodic_right = torch.cat([x_right, y_right, z_right], dim=1)
    
    # z = z_min
    x_front = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_front = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_front = torch.full((n_per_face, 1), z_min, device=device)
    periodic_front = torch.cat([x_front, y_front, z_front], dim=1)
    
    # z = z_max
    x_back = torch.rand(n_per_face, 1, device=device) * (x_max - x_min) + x_min
    y_back = torch.rand(n_per_face, 1, device=device) * (y_max - y_min) + y_min
    z_back = torch.full((n_per_face, 1), z_max, device=device)
    periodic_back = torch.cat([x_back, y_back, z_back], dim=1)
    
    boundary_points.extend([periodic_left, periodic_right, periodic_front, periodic_back])
    
    # 3. Inlet 點 (x = x_min，特別處理)
    n_inlet = distribution['inlet']
    x_inlet = torch.full((n_inlet, 1), x_min, device=device)
    y_inlet = torch.rand(n_inlet, 1, device=device) * (y_max - y_min) + y_min
    z_inlet = torch.rand(n_inlet, 1, device=device) * (z_max - z_min) + z_min
    inlet = torch.cat([x_inlet, y_inlet, z_inlet], dim=1)
    
    boundary_points.append(inlet)
    
    # 合併所有邊界點
    all_boundary_points = torch.cat(boundary_points, dim=0)
    
    return all_boundary_points


def sample_interior_points(
    n_points: int,
    domain_bounds: Dict[str, Tuple[float, float]],
    device: torch.device,
    exclude_boundary_tol: float = 0.01,
    use_sobol: bool = True
) -> torch.Tensor:
    """
    在內部均勻採樣點，排除邊界區域
    
    Args:
        n_points: 內部點數
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
        device: PyTorch device
        exclude_boundary_tol: 邊界排除容差（物理座標）
        use_sobol: 是否使用 Sobol 序列（更均勻）
    
    Returns:
        內部點座標 [n_points, 3] (x, y, z)
    """
    x_min, x_max = domain_bounds['x']
    y_min, y_max = domain_bounds['y']
    z_min, z_max = domain_bounds['z']
    
    # 調整內部域範圍（排除邊界容差）
    x_min_inner = x_min + exclude_boundary_tol
    x_max_inner = x_max - exclude_boundary_tol
    y_min_inner = y_min + exclude_boundary_tol
    y_max_inner = y_max - exclude_boundary_tol
    z_min_inner = z_min + exclude_boundary_tol
    z_max_inner = z_max - exclude_boundary_tol
    
    if use_sobol:
        # 使用 Sobol 序列（準均勻分佈）
        sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        samples = sobol.draw(n_points).to(device)
        
        # 縮放到內部域
        x_interior = samples[:, 0:1] * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = samples[:, 1:2] * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = samples[:, 2:3] * (z_max_inner - z_min_inner) + z_min_inner
    else:
        # 使用均勻隨機採樣
        x_interior = torch.rand(n_points, 1, device=device) * (x_max_inner - x_min_inner) + x_min_inner
        y_interior = torch.rand(n_points, 1, device=device) * (y_max_inner - y_min_inner) + y_min_inner
        z_interior = torch.rand(n_points, 1, device=device) * (z_max_inner - z_min_inner) + z_min_inner
    
    interior_points = torch.cat([x_interior, y_interior, z_interior], dim=1)
    
    return interior_points


def prepare_channel_flow_training_data(config: Dict[str, Any], device: torch.device) -> Dict[str, torch.Tensor]:
    """使用 Channel Flow 載入器準備訓練資料"""
    from pinnx.dataio.channel_flow_loader import prepare_training_data as load_channel_flow
    
    # 載入 Channel Flow 資料 - 支援兩種配置格式
    if 'channel_flow' in config:
        cf_config = config['channel_flow']
        strategy = cf_config.get('strategy', 'qr_pivot')
    else:
        # 使用 JHTDB 配置格式 - 讀取 sensors.selection_method
        sensors_cfg = config.get('sensors', {})
        strategy = sensors_cfg.get('selection_method', 'qr_pivot')  # 支援從配置讀取策略
    
    K = config['sensors']['K']
    
    # 🆕 讀取自定義感測點文件名（如果有）
    sensor_file = config.get('sensors', {}).get('sensor_file', None)
    
    # 🆕 檢查是否為 3D 案例（決定是否請求 w 分量）
    is_3d = config.get('physics', {}).get('type') == 'vs_pinn_channel_flow'
    target_fields = ['u', 'v', 'w', 'p'] if is_3d else ['u', 'v', 'p']
    
    channel_data = load_channel_flow(
        strategy=strategy,
        K=K,
        target_fields=target_fields,
        sensor_file=sensor_file  # 傳遞自定義文件名
    )
    
    # 提取感測器座標和資料
    coords = channel_data['coordinates']  # (K, 2 or 3) numpy array
    sensor_data = channel_data['sensor_data']  # dict with 'u', 'v', ('w',) 'p'
    domain_bounds = channel_data['domain_bounds']
    
    # 計算標準化參數（VS-PINN 風格：映射到 [-1, 1]）
    x_range = domain_bounds['x']
    y_range = domain_bounds['y']
    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = y_range[0], y_range[1]
    
    def normalize_coord(coord, c_min, c_max):
        """將座標標準化到 [-1, 1]"""
        return 2.0 * (coord - c_min) / (c_max - c_min) - 1.0
    
    def denormalize_coord(coord_norm, c_min, c_max):
        """從 [-1, 1] 反標準化"""
        return (coord_norm + 1.0) / 2.0 * (c_max - c_min) + c_min
    
    # 🆕 檢查是否為 VS-PINN（需要 3D 坐標）
    is_vs_pinn = config.get('physics', {}).get('type') == 'vs_pinn_channel_flow'
    
    # 轉換為 PyTorch tensor
    x_sensors_raw = torch.from_numpy(coords[:, 0:1]).float().to(device)  # (K, 1)
    y_sensors_raw = torch.from_numpy(coords[:, 1:2]).float().to(device)  # (K, 1)
    
    # 🆕 如果是 VS-PINN 且有真實 z 座標，使用它；否則為 0
    if is_vs_pinn and coords.shape[1] >= 3:
        z_sensors_raw = torch.from_numpy(coords[:, 2:3]).float().to(device)  # (K, 1)
        # 從配置讀取 z 範圍
        z_domain = config['physics'].get('domain', {})
        z_min = z_domain.get('z_range', [0.0, 9.42])[0]
        z_max = z_domain.get('z_range', [0.0, 9.42])[1]
    else:
        z_sensors_raw = torch.zeros_like(x_sensors_raw)
        z_min = z_max = 0.0  # 2D 情況
    
    # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
    x_sensors = x_sensors_raw
    y_sensors = y_sensors_raw
    z_sensors = z_sensors_raw if is_vs_pinn else torch.zeros_like(x_sensors)
    t_sensors = torch.zeros_like(x_sensors)  # 暫時假設 t=0
    
    u_sensors = torch.from_numpy(sensor_data['u'].reshape(-1, 1)).float().to(device)
    v_sensors = torch.from_numpy(sensor_data['v'].reshape(-1, 1)).float().to(device)
    p_sensors = torch.from_numpy(sensor_data['p'].reshape(-1, 1)).float().to(device)
    
    # 🆕 如果是 VS-PINN，添加 w 分量（假設為 0 或從數據中獲取）
    if is_vs_pinn:
        if 'w' in sensor_data:
            w_sensors = torch.from_numpy(sensor_data['w'].reshape(-1, 1)).float().to(device)
        else:
            # 2D 切片假設 w=0
            w_sensors = torch.zeros_like(u_sensors)
    else:
        w_sensors = None  # 2D 不需要 w
    
    # 生成 PDE 殘差點和邊界點
    sampling = config['training']['sampling']
    
    # 🆕 檢查是否使用分層採樣策略
    use_stratified = sampling.get('strategy', 'uniform') == 'stratified'
    
    if use_stratified:
        # === 分層採樣策略 ===
        logging.info("📊 使用分層採樣策略 (stratified sampling)")
        
        # 構建域邊界字典（物理座標）
        bounds_dict = {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max) if is_vs_pinn else (0.0, 0.0)
        }
        
        # 獲取邊界點分佈配置
        boundary_dist = sampling.get('boundary_distribution', {
            'wall': 1000, 
            'periodic': 800, 
            'inlet': 200
        })
        
        # 生成邊界點（物理座標）
        n_bc = sampling.get('boundary_points', 2000)
        boundary_points_raw = sample_boundary_points(
            n_points=n_bc,
            domain_bounds=bounds_dict,
            device=device,
            distribution=boundary_dist
        )
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_bc = boundary_points_raw[:, 0:1]
        y_bc = boundary_points_raw[:, 1:2]
        z_bc = boundary_points_raw[:, 2:3] if is_vs_pinn else torch.zeros_like(x_bc)
        t_bc = torch.zeros_like(x_bc)
        
        # 生成內部 PDE 點（物理座標）
        n_pde = sampling.get('interior_points', 10000)
        use_sobol = sampling.get('use_sobol', True)
        exclude_tol = sampling.get('boundary_tolerance', 0.01)
        
        interior_points_raw = sample_interior_points(
            n_points=n_pde,
            domain_bounds=bounds_dict,
            device=device,
            exclude_boundary_tol=exclude_tol,
            use_sobol=use_sobol
        )
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_pde = interior_points_raw[:, 0:1]
        y_pde = interior_points_raw[:, 1:2]
        z_pde = interior_points_raw[:, 2:3] if is_vs_pinn else torch.zeros_like(x_pde)
        t_pde = torch.zeros_like(x_pde)
        
        logging.info(f"✅ 分層採樣完成: {n_bc} 邊界點 + {n_pde} 內部點")
        logging.info(f"   - 邊界分佈: {boundary_dist}")
        logging.info(f"   - Sobol 採樣: {use_sobol}, 邊界容差: {exclude_tol}")
        
    else:
        # === 原始均勻隨機採樣 ===
        logging.info("📊 使用均勻隨機採樣策略 (uniform sampling)")
        
        # PDE 殘差點（原始座標）
        x_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        y_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_pde = x_pde_raw
        y_pde = y_pde_raw
        t_pde = torch.zeros_like(x_pde)  # 穩態假設
        
        # 🆕 如果是 VS-PINN，添加 z 座標到 PDE 點
        if is_vs_pinn:
            z_pde_raw = torch.rand(sampling['pde_points'], 1, device=device) * (z_max - z_min) + z_min
            z_pde = z_pde_raw
        else:
            z_pde = torch.zeros_like(x_pde)  # 2D 情況下 z=0
        
        # 邊界點（原始座標）
        n_bc = sampling['boundary_points']
        x_bc_raw = torch.rand(n_bc, 1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
        y_bc_bottom_raw = torch.full((n_bc//2, 1), y_range[0], device=device)  # 下壁面
        y_bc_top_raw = torch.full((n_bc - n_bc//2, 1), y_range[1], device=device)  # 上壁面
        y_bc_raw = torch.cat([y_bc_bottom_raw, y_bc_top_raw], dim=0)
        x_bc_raw = torch.cat([x_bc_raw[:n_bc//2], x_bc_raw[n_bc//2:]], dim=0)
        
        # 🔧 保持物理座標（由 ManualScalingWrapper 負責標準化）
        x_bc = x_bc_raw
        y_bc = y_bc_raw
        t_bc = torch.zeros_like(x_bc)
        
        # 🆕 如果是 VS-PINN，添加 z 座標到邊界點
        if is_vs_pinn:
            z_bc_raw = torch.rand(n_bc, 1, device=device) * (z_max - z_min) + z_min
            z_bc = z_bc_raw
        else:
            z_bc = torch.zeros_like(x_bc)  # 2D 情況下 z=0
    
    # 存儲額外資訊到全局變量（包含標準化參數）
    global _channel_data_cache
    _channel_data_cache = {
        'domain_bounds': domain_bounds,
        'channel_data': channel_data,
        'normalization': {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max,  # 🆕 添加 z 範圍
            'normalize_fn': normalize_coord,
            'denormalize_fn': denormalize_coord
        },
        'is_vs_pinn': is_vs_pinn  # 🆕 標記是否為 VS-PINN
    }
    
    # 🆕 生成初始條件點（t=0）- 使用感測器數據作為 IC
    ic_config = config.get('initial_condition', {})
    if ic_config.get('enabled', False):
        n_ic = ic_config.get('n_points', 256)
        # 從感測器數據中隨機採樣作為 IC（或使用完整感測器數據）
        if n_ic >= len(x_sensors):
            # 使用所有感測器數據
            x_ic = x_sensors.clone()
            y_ic = y_sensors.clone()
            z_ic = z_sensors.clone()  # 🆕 添加 z_ic
            t_ic = torch.zeros_like(x_ic)
            u_ic = u_sensors.clone()
            v_ic = v_sensors.clone()
            p_ic = p_sensors.clone()
            if is_vs_pinn:
                w_ic = w_sensors.clone() if w_sensors is not None else torch.zeros_like(u_ic)
            else:
                w_ic = torch.empty(0, 1, device=device)
        else:
            # 隨機採樣
            ic_indices = torch.randperm(len(x_sensors), device=device)[:n_ic]
            x_ic = x_sensors[ic_indices]
            y_ic = y_sensors[ic_indices]
            z_ic = z_sensors[ic_indices]  # 🆕 添加 z_ic
            t_ic = torch.zeros_like(x_ic)
            u_ic = u_sensors[ic_indices]
            v_ic = v_sensors[ic_indices]
            p_ic = p_sensors[ic_indices]
            if is_vs_pinn:
                w_ic = w_sensors[ic_indices] if w_sensors is not None else torch.zeros_like(u_ic)
            else:
                w_ic = torch.empty(0, 1, device=device)
    else:
        # IC 禁用時，使用空張量
        x_ic = torch.empty(0, 1, device=device)
        y_ic = torch.empty(0, 1, device=device)
        z_ic = torch.empty(0, 1, device=device)  # 🆕 添加 z_ic
        t_ic = torch.empty(0, 1, device=device)
        u_ic = torch.empty(0, 1, device=device)
        v_ic = torch.empty(0, 1, device=device)
        w_ic = torch.empty(0, 1, device=device)  # 🆕 添加 w_ic
        p_ic = torch.empty(0, 1, device=device)
    
    # 提取低保真先驗資料 (如果有)
    training_dict = {
        'x_pde': x_pde, 'y_pde': y_pde, 'z_pde': z_pde, 't_pde': t_pde,  # 🆕 添加 z_pde
        'x_bc': x_bc, 'y_bc': y_bc, 'z_bc': z_bc, 't_bc': t_bc,  # 🆕 添加 z_bc
        'x_sensors': x_sensors, 'y_sensors': y_sensors, 'z_sensors': z_sensors, 't_sensors': t_sensors,  # 🆕 添加 z_sensors
        'u_sensors': u_sensors, 'v_sensors': v_sensors, 'p_sensors': p_sensors,
        'x_ic': x_ic, 'y_ic': y_ic, 'z_ic': z_ic, 't_ic': t_ic,  # 🆕 添加 z_ic
        'u_ic': u_ic, 'v_ic': v_ic, 'p_ic': p_ic
    }
    
    # 🆕 如果是 VS-PINN，添加 w 分量到訓練字典
    if is_vs_pinn:
        training_dict['w_sensors'] = w_sensors if w_sensors is not None else torch.zeros_like(u_sensors)
        training_dict['w_ic'] = w_ic
    else:
        # 2D 情況下不需要 w，但為了統一性可以添加空張量
        training_dict['w_sensors'] = torch.empty(0, 1, device=device)
        training_dict['w_ic'] = torch.empty(0, 1, device=device)
    
    # 添加低保真先驗資料到批次 (如果可用)
    if 'lowfi_prior' in channel_data and channel_data['lowfi_prior']:
        lowfi = channel_data['lowfi_prior']
        if 'u' in lowfi:
            training_dict['u_prior'] = torch.from_numpy(lowfi['u'].reshape(-1, 1)).float().to(device)
        if 'v' in lowfi:
            training_dict['v_prior'] = torch.from_numpy(lowfi['v'].reshape(-1, 1)).float().to(device)
        if 'p' in lowfi:
            training_dict['p_prior'] = torch.from_numpy(lowfi['p'].reshape(-1, 1)).float().to(device)
        training_dict['has_prior'] = True
    else:
        training_dict['has_prior'] = False
    
    validation_split = config.get('training', {}).get('validation_split', 0.0)
    training_dict = _apply_validation_split(training_dict, validation_split, is_vs_pinn=is_vs_pinn)
    
    return training_dict





def train_step(model: nn.Module, 
               physics: NSEquations2D,
               losses: Dict[str, nn.Module],
               data_batch: Dict[str, torch.Tensor],
               optimizer: torch.optim.Optimizer,
               weighters: Dict[str, Any],
               epoch: int,
               device: torch.device,
               config: Optional[Dict[str, Any]] = None,
               input_normalizer: Optional[InputNormalizer] = None) -> Dict[str, Any]:
    """執行一個訓練步驟，支援動態權重調整"""
    optimizer.zero_grad()
    
    # 🆕 檢查是否為 VS-PINN（3D）還是標準 PINN（2D）
    is_vs_pinn = 'z_pde' in data_batch and hasattr(physics, 'compute_momentum_residuals')
    loss_cfg = config['losses'] if config and 'losses' in config else {}
    
    def prepare_model_coords(coord_tensor: torch.Tensor, require_grad: bool = False) -> torch.Tensor:
        """產生模型輸入座標"""
        coords = coord_tensor
        if input_normalizer is not None:
            coords = input_normalizer.transform(coords)
        if is_vs_pinn and hasattr(physics, 'scale_coordinates'):
            coords = physics.scale_coordinates(coord_tensor)
        # 🔧 修復：不要使用 .detach()，否則會斷開梯度鏈條
        # if require_grad:
        #     coords = coords.clone().detach().requires_grad_(True)
        if require_grad and not coords.requires_grad:
            coords.requires_grad_(True)
        return coords
    
    # PDE 殘差計算 - 支持 2D 和 3D
    if is_vs_pinn:
        # VS-PINN 3D: 使用 (x, y, z) 座標
        coords_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde'], data_batch['z_pde']], dim=1)
    else:
        # 標準 PINN 2D: 只使用 (x, y) 座標
        coords_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde']], dim=1)
    
    coords_pde.requires_grad_(True)  # 物理座標保留梯度以支援非縮放情況
    model_coords_pde = prepare_model_coords(coords_pde, require_grad=True)
    
    u_pred = model(model_coords_pde)
    
    # 根據模式提取速度和壓力分量
    if is_vs_pinn:
        # VS-PINN: 預測 [u, v, w, p]
        if u_pred.shape[1] == 3:
            # 模型只輸出 [u, v, p]，添加 w=0
            velocity = u_pred[:, :2]  # u, v
            pressure = u_pred[:, 2:3]  # p
            w_component = torch.zeros_like(pressure)
            predictions = torch.cat([velocity, w_component, pressure], dim=1)
        elif u_pred.shape[1] == 4:
            # 模型輸出完整 [u, v, w, p]
            predictions = u_pred
            velocity = u_pred[:, :2]  # u, v
            pressure = u_pred[:, 3:4]  # p
        else:
            raise ValueError(f"VS-PINN 模型輸出維度錯誤：{u_pred.shape[1]}，期望 3 或 4")
    else:
        # 標準 PINN: 根據輸出維度處理
        if u_pred.shape[1] == 3:
            # 3D 輸出 [u, v, p]
            velocity = u_pred[:, :2]  # u, v
            pressure = u_pred[:, 2:3]  # p
        elif u_pred.shape[1] == 4:
            # 4D 輸出 [u, v, w, p] - 2D 流場 w≈0
            velocity = u_pred[:, :2]  # u, v
            pressure = u_pred[:, 3:4]  # p (第4個輸出)
        else:
            raise ValueError(f"不支援的標準 PINN 輸出維度: {u_pred.shape[1]}，期望 3 或 4")
        predictions = None  # 標準 PINN 不需要 predictions
    
    try:
        # 檢查是否為 VS-PINN（有專用的殘差計算方法）
        if is_vs_pinn and hasattr(physics, 'compute_momentum_residuals'):
            # VS-PINN 3D - 計算動量和連續性殘差
            residuals_mom = physics.compute_momentum_residuals(
                coords_pde,
                predictions,
                scaled_coords=model_coords_pde
            )
            continuity_residual = physics.compute_continuity_residual(
                coords_pde,
                predictions,
                scaled_coords=model_coords_pde
            )
            
            # 提取殘差（VS-PINN 返回字典）
            residuals = {
                'momentum_x': residuals_mom['momentum_x'],
                'momentum_y': residuals_mom['momentum_y'],
                'momentum_z': residuals_mom['momentum_z'],
                'continuity': continuity_residual,
            }
        else:
            # 標準 NS 2D - 使用原有接口
            residuals = physics.residual(coords_pde, velocity, pressure)
        
        # 🔧 獲取點權重（如果存在）
        pde_point_weights = data_batch.get('pde_point_weights', None)
        
        # 🔧 分離各 PDE 項以支援獨立權重（應用點權重）
        if pde_point_weights is not None:
            # 應用點權重到每個殘差項
            momentum_x_loss = apply_point_weights_to_loss(residuals['momentum_x']**2, pde_point_weights)
            momentum_y_loss = apply_point_weights_to_loss(residuals['momentum_y']**2, pde_point_weights)
            momentum_z_loss = apply_point_weights_to_loss(residuals['momentum_z']**2, pde_point_weights) if is_vs_pinn else torch.tensor(0.0, device=device)
            continuity_loss = apply_point_weights_to_loss(residuals['continuity']**2, pde_point_weights)
        else:
            # 標準均值計算
            momentum_x_loss = torch.mean(residuals['momentum_x']**2)
            momentum_y_loss = torch.mean(residuals['momentum_y']**2)
            momentum_z_loss = torch.mean(residuals['momentum_z']**2) if is_vs_pinn else torch.tensor(0.0, device=device)
            continuity_loss = torch.mean(residuals['continuity']**2)
    except Exception as e:
        # 🚨 物理殘差計算失敗 - 詳細錯誤日誌
        logging.error("=" * 60)
        logging.error("🚨 Physics residual computation FAILED!")
        logging.error("=" * 60)
        logging.error(f"Exception: {e}")
        logging.error(f"coords_pde shape: {coords_pde.shape}")
        logging.error(f"u_pred shape: {u_pred.shape}")
        logging.error(f"velocity shape: {velocity.shape}")
        logging.error(f"pressure shape: {pressure.shape}")
        logging.error(f"is_vs_pinn: {is_vs_pinn}")
        logging.error("Traceback:")
        logging.error(traceback.format_exc())
        logging.error("=" * 60)
        
        # 🔴 在開發階段直接拋出異常，避免靜默失敗
        if os.getenv('PINNS_DEV_MODE', 'true').lower() == 'true':
            raise
        
        # 生產環境備用（應該極少觸發）
        logging.warning("⚠️ Using fallback L2 regularization (PHYSICS CONSTRAINTS DISABLED!)")
        momentum_x_loss = torch.mean(u_pred**2) * 0.001
        momentum_y_loss = torch.mean(u_pred**2) * 0.001
        momentum_z_loss = torch.mean(u_pred**2) * 0.001 if is_vs_pinn else torch.tensor(0.0, device=device)
        continuity_loss = torch.mean(u_pred**2) * 0.001
    
    # 邊界條件損失 - 支持 2D/3D
    if is_vs_pinn:
        # VS-PINN 3D: 使用 (x, y, z) 座標
        coords_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc'], data_batch['z_bc']], dim=1)
    else:
        # 標準 PINN 2D: 只使用 (x, y) 座標
        coords_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc']], dim=1)
    model_coords_bc = prepare_model_coords(coords_bc, require_grad=False)
    u_bc_pred = model(model_coords_bc)
    
    # 🔧 分離壁面約束和週期性邊界
    # 壁面約束：u=v=0 at y=±1（標準化座標）
    # 標準化後的 y 座標範圍為 [-1, 1]，壁面也在 y=±1
    y_bc = data_batch['y_bc']  # 形狀 [N_bc, 1]
    
    # 🔥 明確識別壁面點：容差設為 1e-3（標準化座標）
    wall_mask = (torch.abs(y_bc - 1.0) < 1e-3) | (torch.abs(y_bc + 1.0) < 1e-3)
    wall_mask = wall_mask.squeeze()  # 形狀 [N_bc]
    
    if wall_mask.sum() > 0:
        # 只在壁面點應用無滑移條件
        u_wall = u_bc_pred[wall_mask, 0]  # u 分量
        v_wall = u_bc_pred[wall_mask, 1]  # v 分量
        wall_loss = torch.mean(u_wall**2 + v_wall**2)
    else:
        wall_loss = torch.tensor(0.0, device=device)
        if epoch == 0:
            logging.warning(f"⚠️ No wall boundary points detected! y_bc range: [{y_bc.min():.6f}, {y_bc.max():.6f}]")
    
    # 週期性邊界：u(x=0) = u(x=2π)
    # 從 PDE 點中提取左右邊界（標準化座標 x=-1 和 x=+1）
    x_pde_denorm = data_batch['x_pde']  # 標準化座標 [-1, 1]
    left_mask = (x_pde_denorm < -0.95).squeeze()  # 左邊界附近
    right_mask = (x_pde_denorm > 0.95).squeeze()  # 右邊界附近
    
    if left_mask.sum() > 0 and right_mask.sum() > 0:
        # 在邊界點評估速度場
        coords_left = coords_pde[left_mask]
        coords_right = coords_pde[right_mask]
        model_left = prepare_model_coords(coords_left, require_grad=False)
        model_right = prepare_model_coords(coords_right, require_grad=False)
        u_left = model(model_left)
        u_right = model(model_right)
        
        # 週期性損失：取最小數量對齊（修復類型錯誤）
        n_min = min(left_mask.sum().item(), right_mask.sum().item())
        periodicity_loss = torch.mean((u_left[:n_min] - u_right[:n_min])**2)
    else:
        periodicity_loss = torch.tensor(0.0, device=device)
    
    # 🆕 Inlet 速度剖面邊界條件：指導模型學習主流方向
    # 從全域配置中讀取 inlet 設定
    inlet_enabled = False
    if config is not None and 'inlet' in config:
        inlet_config = config['inlet']
        inlet_enabled = inlet_config.get('enabled', False)
        n_inlet = inlet_config.get('n_points', 64)
        x_inlet_pos = inlet_config.get('x_position', -1.0)  # 標準化座標，預設左邊界
        
        # 🚀 從課程訓練調度器獲取當前階段配置（優先）
        stage_config = None
        if weighters.get('curriculum') is not None:
            stage_config = weighters['curriculum'].get_stage_config(epoch)
        
        # 從階段特定配置讀取（優先），否則使用全域配置
        if stage_config is not None and 'inlet' in stage_config:
            stage_inlet = stage_config['inlet']
            profile_type = stage_inlet.get('profile_type', inlet_config.get('profile_type', 'log_law'))
            u_max = stage_inlet.get('u_max', None)  # 階段特定的最大速度
        else:
            profile_type = inlet_config.get('profile_type', 'log_law')
            u_max = None
    else:
        # 預設配置（如果未在 config 中設定）
        n_inlet = 64
        profile_type = 'log_law'
        x_inlet_pos = -1.0
        u_max = None
        stage_config = None
    
    if inlet_enabled:
        # 生成 inlet 邊界點（x=x_inlet_pos, y ∈ [-1, 1]）
        y_inlet = torch.linspace(-1.0, 1.0, n_inlet, device=device).unsqueeze(1)
        x_inlet_coords = torch.full_like(y_inlet, x_inlet_pos)
        
        # 🆕 如果是 VS-PINN 3D，添加 z 座標（中間值）
        if is_vs_pinn:
            z_inlet = torch.zeros_like(y_inlet)  # z=0 (標準化座標中間值)
            inlet_coords = torch.cat([x_inlet_coords, y_inlet, z_inlet], dim=1)
        else:
            inlet_coords = torch.cat([x_inlet_coords, y_inlet], dim=1)
        
        # 模型預測（在縮放空間計算）
        inlet_model_coords = prepare_model_coords(inlet_coords, require_grad=False)
        inlet_pred = model(inlet_model_coords)
        
        # 從課程階段讀取物理參數（優先），否則使用全域配置
        if stage_config is not None:
            Re_tau = stage_config.get('Re_tau', 1000.0)
            # 如果階段沒有設定 u_max，使用預設值
            if u_max is None:
                u_max = 16.5  # 預設值
        elif config is not None and 'physics' in config:
            Re_tau = config['physics'].get('Re_tau', 1000.0)
            if u_max is None:
                u_max = config['physics'].get('u_bulk', 16.5)
        else:
            # 預設值（與課程訓練對齊）
            Re_tau = 1000.0
            if u_max is None:
                u_max = 16.5
        
        # 計算 inlet 損失
        # 🔧 直接實例化邊界條件損失模組（避免類型推斷問題）
        bc_loss_module = BoundaryConditionLoss()
        
        inlet_loss = bc_loss_module.inlet_velocity_profile_loss(
            inlet_coords=inlet_coords,
            inlet_predictions=inlet_pred,
            profile_type=profile_type,
            Re_tau=Re_tau,
            u_max=u_max,
            y_range=(-1.0, 1.0)
        )
    else:
        inlet_loss = torch.tensor(0.0, device=device)
    
    # 🆕 初始條件損失（t=0 流場約束）
    ic_config = config.get('initial_condition', {}) if config is not None else {}
    if ic_config.get('enabled', False) and len(data_batch.get('x_ic', [])) > 0:
        # 組合初始條件座標
        if is_vs_pinn:
            coords_ic = torch.cat([data_batch['x_ic'], data_batch['y_ic'], data_batch['z_ic']], dim=1)
        else:
            coords_ic = torch.cat([data_batch['x_ic'], data_batch['y_ic']], dim=1)
        
        # 模型在 t=0 的預測
        ic_model_coords = prepare_model_coords(coords_ic, require_grad=False)
        ic_pred = model(ic_model_coords)
        
        # 真實初始條件
        u_ic_true = data_batch['u_ic']
        v_ic_true = data_batch['v_ic']
        p_ic_true = data_batch['p_ic']
        ic_true = torch.cat([u_ic_true, v_ic_true, p_ic_true], dim=1)
        
        # 計算 IC 損失（MSE）
        from pinnx.losses.residuals import InitialConditionLoss
        ic_loss_module = InitialConditionLoss()
        ic_loss_result = ic_loss_module(
            initial_coords=coords_ic,
            initial_predictions=ic_pred,
            initial_data=ic_true,
            weights=None
        )
        ic_loss = ic_loss_result['total_ic']  # 修正：使用正確的鍵名
    else:
        ic_loss = torch.tensor(0.0, device=device)
    
    # 資料匹配損失
    # 感測點資料損失 - 支持 2D/3D (包含 u, v, [w,] p 場)
    if is_vs_pinn:
        # VS-PINN 3D: 使用 (x, y, z) 座標
        coords_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors'], data_batch['z_sensors']], dim=1)
    else:
        # 標準 PINN 2D: 只使用 (x, y) 座標
        coords_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors']], dim=1)
    model_coords_sensors = prepare_model_coords(coords_sensors, require_grad=False)
    u_sensors_pred = model(model_coords_sensors)
    
    # 速度場損失 (u, v) - 使用完整場統計標準化以平衡不同場的尺度
    u_true = data_batch['u_sensors']
    v_true = data_batch['v_sensors']
    p_true = data_batch['p_sensors']
    
    # 🔧 修復：使用完整 DNS 場統計作為標準化因子（非感測點統計）
    # 這些值從 128×64 完整場提取，確保正確的尺度平衡
    u_scale = 9.841839   # DNS 完整場 U 均值
    v_scale = 0.188766   # DNS 完整場 V 標準差（v 均值接近 0）
    p_scale = 35.655934  # DNS 完整場 P 均值絕對值
    
    # 標準化後的 MSE（相對誤差）
    u_loss = torch.mean(((u_sensors_pred[:, 0:1] - u_true) / u_scale)**2)
    v_loss = torch.mean(((u_sensors_pred[:, 1:2] - v_true) / v_scale)**2)
    pressure_loss = torch.mean(((u_sensors_pred[:, 2:3] - p_true) / p_scale)**2)
    
    # 組合資料損失 (各場權重平衡)
    velocity_loss = u_loss + v_loss
    data_loss = velocity_loss + pressure_loss  # 現在各場的尺度已正確平衡
    
    # 先驗一致性損失 (與低保真場的軟約束)
    if data_batch.get('has_prior', False):
        # 在感測點位置計算 PINNs 預測（已在上方計算，重複使用）
        # 注意：x_sensors 已經在上方根據 is_vs_pinn 正確拼接
        
        # 計算與低保真先驗的差異
        prior_loss = torch.tensor(0.0, device=device)
        
        if 'u_prior' in data_batch:
            prior_loss += torch.mean((u_sensors_pred[:, 0:1] - data_batch['u_prior'])**2)
        if 'v_prior' in data_batch:
            prior_loss += torch.mean((u_sensors_pred[:, 1:2] - data_batch['v_prior'])**2)
        if 'p_prior' in data_batch:
            prior_loss += torch.mean((u_sensors_pred[:, 2:3] - data_batch['p_prior'])**2)
    else:
        # 沒有先驗資料時，使用零損失（不需要梯度）
        prior_loss = torch.tensor(0.0, device=device)
    
    # 🔧 新增物理約束損失（Task-10）
    # 1. 流量約束（體積流量守恆）
    bulk_velocity_loss = torch.tensor(0.0, device=device)
    if is_vs_pinn and hasattr(physics, 'compute_bulk_velocity_constraint'):
        try:
            bulk_velocity_loss = physics.compute_bulk_velocity_constraint(coords_pde, predictions)
        except Exception as e:
            logging.debug(f"Bulk velocity constraint failed: {e}")
    
    # 2. 中心線對稱約束
    centerline_dudy_loss = torch.tensor(0.0, device=device)
    centerline_v_loss = torch.tensor(0.0, device=device)
    if is_vs_pinn and hasattr(physics, 'compute_centerline_symmetry'):
        try:
            centerline_losses = physics.compute_centerline_symmetry(
                coords_pde,
                predictions,
                scaled_coords=model_coords_pde
            )
            centerline_dudy_loss = centerline_losses['centerline_dudy']
            centerline_v_loss = centerline_losses['centerline_v']
        except Exception as e:
            logging.debug(f"Centerline symmetry constraint failed: {e}")
    
    # 3. 壓力參考點約束（可選）
    pressure_reference_loss = torch.tensor(0.0, device=device)
    if is_vs_pinn and hasattr(physics, 'compute_pressure_reference'):
        try:
            pressure_reference_loss = physics.compute_pressure_reference(coords_pde, predictions)
        except Exception as e:
            logging.debug(f"Pressure reference constraint failed: {e}")
    
    # 🔧 收集所有損失到字典中（12 個獨立項 + prior，包含 Task-10 新增項）
    loss_dict = {
        'data': data_loss,
        'momentum_x': momentum_x_loss,
        'momentum_y': momentum_y_loss,
        'momentum_z': momentum_z_loss,
        'continuity': continuity_loss,
        'wall_constraint': wall_loss,
        'periodicity': periodicity_loss,
        'inlet': inlet_loss,  # 🆕 Inlet 速度剖面損失
        'initial_condition': ic_loss,  # 🔥 Initial condition 損失
        'bulk_velocity': bulk_velocity_loss,  # 🆕 Task-10: 流量約束
        'centerline_dudy': centerline_dudy_loss,  # 🆕 Task-10: 中心線對稱（梯度）
        'centerline_v': centerline_v_loss,  # 🆕 Task-10: 中心線對稱（速度）
        'pressure_reference': pressure_reference_loss,  # 🆕 Task-10: 壓力參考點
        'prior': prior_loss
    }
    
    # 🆕🆕🆕 損失歸一化（VS-PINN 功能）
    # 在應用權重前，先將所有損失歸一化到同一數量級
    if hasattr(physics, 'normalize_loss_dict'):
        # 記錄原始損失（僅在前 5 個 epoch）
        if epoch < 5 and epoch == 0:
            logging.info(f"[NORMALIZATION] Raw losses at epoch {epoch}:")
            for key, loss in loss_dict.items():
                logging.info(f"  {key}: {loss.item():.6f}")
        
        # 執行歸一化
        loss_dict = physics.normalize_loss_dict(loss_dict, epoch)
        
        # 記錄歸一化後的損失（僅在開始歸一化後的前幾個 epoch）
        if epoch >= physics.warmup_epochs and epoch < physics.warmup_epochs + 3:
            logging.info(f"[NORMALIZATION] Normalized losses at epoch {epoch}:")
            for key, loss in loss_dict.items():
                logging.info(f"  {key}: {loss.item():.6f}")
            
            # 顯示歸一化參考值
            norm_info = physics.get_normalization_info()
            logging.info(f"[NORMALIZATION] Reference values (normalizers):")
            for key, val in norm_info['normalizers'].items():
                logging.info(f"  {key}: {val:.6f}")
    
    # 應用動態權重
    # 初始權重（從配置文件讀取或使用預設）
    prior_weight_from_config = getattr(losses['prior'], 'consistency_weight', 0.3)

    base_weights, adaptive_terms = derive_loss_weights(loss_cfg, prior_weight_from_config, is_vs_pinn)
    if epoch == 0 and loss_cfg:
        logging.info("[CONFIG] Using weights from config file:")
        for key, val in base_weights.items():
            logging.info(f"  {key}_weight: {val:.6f}")
    elif epoch == 0 and not loss_cfg:
        logging.warning("[FALLBACK] No loss config provided, using default hardcoded weights")

    configured_terms = loss_cfg.get('adaptive_loss_terms') if isinstance(loss_cfg, dict) else None
    if configured_terms is not None:
        adaptive_terms = [name for name in configured_terms if name in base_weights]
    
    # 🔧 DEBUG: 記錄讀取的 prior_weight（保留原有邏輯）
    if epoch == 0:
        logging.info(f"[DEBUG] Prior weight from config: {prior_weight_from_config}")
    
    weights = base_weights.copy()
    
    # 🚀🚀🚀 課程訓練調度器（最高優先級）
    if weighters.get('curriculum') is not None:
        stage_config = weighters['curriculum'].get_stage_config(epoch)
        
        # 階段切換時記錄（已在 CurriculumScheduler 內部處理）
        # 應用課程權重（完全覆蓋 base_weights）
        weights.update(stage_config['weights'])
        
        # 🔥 返回額外的配置資訊供訓練迴圈使用
        # 注意：這裡只更新權重，物理參數和學習率需要在訓練迴圈中處理
        curriculum_info = {
            'stage_name': stage_config['stage_name'],
            'is_transition': stage_config['is_transition'],
            'sampling': stage_config['sampling'],
            'lr': stage_config['lr']
        }
    else:
        curriculum_info = None
    
    # 🚀🚀🚀 階段式權重調度器（優先級第二，與課程訓練互斥）
    if weighters.get('staged') is not None and weighters.get('curriculum') is None:
        staged_weights, phase_name, is_transition = weighters['staged'].get_phase_weights(epoch)
        
        # 🔧 階段切換時記錄
        if is_transition:
            logging.info(f"=" * 80)
            logging.info(f"🎯 PHASE TRANSITION at Epoch {epoch}: {phase_name}")
            logging.info(f"   New weights: {staged_weights}")
            logging.info(f"=" * 80)
        
        # 應用階段式權重（覆蓋 base_weights）
        weights.update(staged_weights)
    
    # 使用動態權重器 (如果已啟用且沒有階段式調度器)
    if weighters['gradnorm'] is not None and weighters.get('staged') is None:
        # GradNorm 會根據各損失項的梯度動態調整權重
        try:
            updated_weights = weighters['gradnorm'].update_weights(loss_dict)
            if updated_weights is not None:
                adaptive_base_sum = sum(base_weights.get(name, 0.0) for name in adaptive_terms)
                total_updated = sum(updated_weights.values()) if updated_weights else 1.0
                for key in base_weights:
                    if key in adaptive_terms and key in updated_weights:
                        normalized_factor = updated_weights[key] / (total_updated + 1e-12)
                        weights[key] = adaptive_base_sum * normalized_factor
                    elif key not in adaptive_terms:
                        weights[key] = base_weights[key]
        except Exception as e:
            logging.warning(f"GradNorm update failed: {e}, using base weights")
    
    # 使用權重調度器 (階段性調整)
    if weighters['scheduler'] is not None:
        try:
            if hasattr(weighters['scheduler'], 'get_phase_weights'):
                phase_weights = weighters['scheduler'].get_phase_weights(epoch, 15000)
                # 與基礎權重相乘
                for key in weights:
                    if key in phase_weights:
                        weights[key] *= phase_weights[key]
            elif hasattr(weighters['scheduler'], 'get_weights'):
                scheduled_weights = weighters['scheduler'].get_weights()
                for key in weights:
                    if key in scheduled_weights:
                        weights[key] *= scheduled_weights[key]
        except Exception as e:
            logging.warning(f"Scheduler update failed: {e}, using current weights")
    
    # 因果權重器 (時間序列問題才需要，2D 穩態跳過)
    if weighters['causal'] is not None and hasattr(weighters['causal'], 'compute_causal_weights'):
        try:
            time_losses = [loss_dict.get('data', torch.tensor(0.0)), 
                          loss_dict.get('residual', torch.tensor(0.0))]
            causal_weights_list = weighters['causal'].compute_causal_weights(time_losses)
            causal_weights = dict(zip(['data', 'residual'], causal_weights_list[:2]))
            
            for key in weights:
                if key in causal_weights:
                    weights[key] *= causal_weights[key]
        except Exception as e:
            logging.warning(f"Causal weighting failed: {e}, skipping")
    
    # 🔧 強制覆蓋：如果配置中 prior_weight=0，確保權重被完全禁用
    if base_weights.get('prior', 0.0) == 0.0 and 'prior' in weights:
        weights['prior'] = 0.0
        if epoch == 0:
            logging.info("[FIX] Prior weight forced to 0.0 (config: prior_weight=0.0)")
    
    # 計算加權總損失 - 使用第一個損失項初始化（保留計算圖）
    loss_keys = list(loss_dict.keys())
    if len(loss_keys) > 0:
        # 🔍 除錯：檢查每個損失項和權重
        if epoch == 0:
            logging.info(f"[DEBUG] Loss dict at epoch 0:")
            for key in loss_keys:
                loss_val = loss_dict[key]
                weight_val = weights.get(key, 1.0)
                logging.info(f"  {key}: loss={loss_val.item():.6f}, weight={weight_val}, "
                           f"has_nan={torch.isnan(loss_val).any()}, has_inf={torch.isinf(loss_val).any()}")
        
        # 使用第一個損失項初始化（帶計算圖）
        first_key = loss_keys[0]
        weight_0 = weights.get(first_key, 1.0)
        weight_tensor_0 = torch.tensor(weight_0, device=device, dtype=torch.float32) if not torch.is_tensor(weight_0) else weight_0.to(dtype=torch.float32)
        total_loss = weight_tensor_0 * loss_dict[first_key]
        
        if epoch == 0:
            logging.info(f"[DEBUG] Initial total_loss from '{first_key}': {total_loss.item():.6f}, has_nan={torch.isnan(total_loss).any()}")
        
        # 累加其餘損失項
        for key in loss_keys[1:]:
            weight = weights.get(key, 1.0)
            weight_tensor = torch.tensor(weight, device=device, dtype=torch.float32) if not torch.is_tensor(weight) else weight.to(dtype=torch.float32)
            weighted_term = weight_tensor * loss_dict[key]
            
            if epoch == 0:
                logging.info(f"[DEBUG] Adding '{key}': weighted_term={weighted_term.item():.6f}, has_nan={torch.isnan(weighted_term).any()}")
            
            total_loss = total_loss + weighted_term
            
            if epoch == 0:
                logging.info(f"[DEBUG] Cumulative total_loss after '{key}': {total_loss.item() if not torch.isnan(total_loss).any() else 'NaN'}")
    else:
        # 備用：如果沒有損失項，創建零損失
        total_loss = torch.tensor(0.0, device=device, dtype=torch.float32, requires_grad=True)
    
    total_loss.backward()
    optimizer.step()
    
    # 🔧 準備返回值（包含所有獨立損失項）
    result = {
        'total_loss': float(total_loss.item()),
        'momentum_x_loss': float(momentum_x_loss.item()),
        'momentum_y_loss': float(momentum_y_loss.item()),
        'momentum_z_loss': float(momentum_z_loss.item()),
        'continuity_loss': float(continuity_loss.item()),
        'wall_loss': float(wall_loss.item()),
        'periodicity_loss': float(periodicity_loss.item()),
        'inlet_loss': float(inlet_loss.item()),  # 🆕 Inlet 損失
        'data_loss': float(data_loss.item()),
        'prior_loss': float(prior_loss.item()),
        # 向後兼容（聚合損失）
        'residual_loss': float(momentum_x_loss.item() + momentum_y_loss.item() + momentum_z_loss.item() + continuity_loss.item()),
        'bc_loss': float(wall_loss.item() + inlet_loss.item())  # 🔧 包含 inlet 損失
    }
    
    # 添加權重資訊
    for k, v in weights.items():
        result[f'weight_{k}'] = float(v.item() if torch.is_tensor(v) else v)
    
    # 🔧 監控權重總和（驗證標準化）
    adaptive_weight_sum = sum(float(weights.get(name, 0.0)) for name in adaptive_terms if name in weights)
    fixed_weight_sum = sum(float(weights.get(name, 0.0)) for name in weights if name not in adaptive_terms)
    result['weight_sum_adaptive'] = adaptive_weight_sum
    result['weight_sum_fixed'] = fixed_weight_sum
    
    # 🚀 添加課程訓練資訊（如果啟用）- 使用特殊前綴避免類型衝突
    if curriculum_info is not None:
        result['_curriculum_stage'] = curriculum_info['stage_name']  # type: ignore
        result['_curriculum_transition'] = 1.0 if curriculum_info['is_transition'] else 0.0
        result['_curriculum_lr'] = curriculum_info['lr']
    
    return result


def compute_validation_metrics(
    model: nn.Module,
    validation_info: Optional[Dict[str, torch.Tensor]],
    device: torch.device,
    physics: Optional[Any] = None,
    input_normalizer: Optional[InputNormalizer] = None
) -> Optional[Dict[str, float]]:
    """
    計算驗證資料上的誤差指標（相對 L2 與 MSE）。
    """
    if not validation_info or validation_info.get('size', 0) == 0:
        return None
    
    coords = validation_info.get('coords')
    targets = validation_info.get('targets')
    
    if coords is None or targets is None or coords.numel() == 0 or targets.numel() == 0:
        return None
    
    coords = coords.to(device)
    targets = targets.to(device)
    
    training_mode = model.training
    model.eval()
    
    with torch.no_grad():
        coords_for_model = coords
        if input_normalizer is not None:
            coords_for_model = input_normalizer.transform(coords_for_model)
        if physics is not None and hasattr(physics, 'scale_coordinates'):
            coords_for_model = physics.scale_coordinates(coords)
        preds = model(coords_for_model)
        # 僅比較可用的場分量
        n_pred = preds.shape[1]
        n_targets = targets.shape[1]
        n_common = min(n_pred, n_targets)
        if n_pred != n_targets:
            logging.debug(
                f"[Validation] Output dimension mismatch (pred={n_pred}, target={n_targets}); "
                f"comparing first {n_common} components."
            )
        preds = preds[:, :n_common]
        targets = targets[:, :n_common]
        diff = preds - targets
        mse = torch.mean(diff**2).item()
        rel_l2 = relative_L2(preds, targets).mean().item()
    
    if training_mode:
        model.train()
    
    return {
        'mse': mse,
        'relative_l2': rel_l2
    }


def train_model(model: nn.Module,
                physics: NSEquations2D,
                losses: Dict[str, nn.Module],
                config: Dict[str, Any],
                device: torch.device) -> Dict[str, Any]:
    """主要訓練迴圈"""
    train_cfg = config['training']
    loss_cfg = config.get('losses', {})
    physics_type = config.get('physics', {}).get('type', '')
    is_vs_cfg = physics_type == 'vs_pinn_channel_flow'
    
    # 建立優化器
    optimizer_cfg = train_cfg.get('optimizer', {})
    lr = optimizer_cfg.get('lr', train_cfg.get('lr', 1e-3))
    weight_decay = optimizer_cfg.get('weight_decay', train_cfg.get('weight_decay', 0.0))
    
    optimizer_name = optimizer_cfg.get('type', train_cfg.get('optimizer', 'adam')).lower()
    if optimizer_name == 'soap':
        try:
            from torch_optimizer import SOAP
        except ImportError as exc:
            raise ImportError("Requested optimizer 'SOAP' but torch_optimizer is not installed") from exc
        soap_kwargs = optimizer_cfg.get('soap', train_cfg.get('soap', {}))
        optimizer = SOAP(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            **soap_kwargs
        )
        logging.info("✅ Using SOAP optimizer")
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        logging.info("✅ Using Adam optimizer")
    
    # 學習率調度器
    scheduler_cfg = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_cfg.get('type', train_cfg.get('lr_scheduler', 'none'))
    max_epochs = train_cfg.get('epochs', train_cfg.get('max_epochs', 1000))
    
    if scheduler_type == 'warmup_cosine':
        # Warmup + CosineAnnealing 調度器
        warmup_epochs = scheduler_cfg.get('warmup_epochs', 10)
        min_lr = scheduler_cfg.get('min_lr', 1e-6)
        scheduler = WarmupCosineScheduler(
            optimizer, 
            warmup_epochs=warmup_epochs,
            max_epochs=max_epochs,
            base_lr=lr,
            min_lr=min_lr
        )
    elif scheduler_type == 'cosine_warm_restarts':
        # CosineAnnealingWarmRestarts 調度器（週期性重啟）
        T_0 = scheduler_cfg.get('T_0', 100)  # 第一個週期長度
        T_mult = scheduler_cfg.get('T_mult', 1)  # 週期長度倍增因子
        eta_min = scheduler_cfg.get('eta_min', 1e-6)  # 最小學習率
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    elif scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs
        )
    elif scheduler_type == 'exponential':
        gamma = scheduler_cfg.get('gamma', 0.999)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        scheduler = None
    
    
    # 建立權重器（傳入 physics 以支援課程訓練）- 需要在 early stopping 之前初始化
    weighters = create_weighters(config, model, device, physics)

    # 🚀 Early Stopping 設定
    early_stopping_cfg = train_cfg.get('early_stopping', {})
    adaptive_terms = loss_cfg.get('adaptive_loss_terms')
    if adaptive_terms is None:
        adaptive_terms = [
            'data',
            'momentum_x',
            'momentum_y',
            'momentum_z',
            'continuity',
            'wall_constraint',
            'periodicity',
            'inlet',
            'bulk_velocity',
            'centerline_dudy',
            'centerline_v',
            'pressure_reference',
            'initial_condition',
        ]
        if loss_cfg.get('prior_weight', 0.0) > 0.0:
            adaptive_terms.append('prior')

    if loss_cfg.get('adaptive_weighting', False) and weighters['staged'] is None:
        early_stopping_enabled = early_stopping_cfg.get('enabled', False)
        patience = early_stopping_cfg.get('patience', 400)
        min_delta = early_stopping_cfg.get('min_delta', 0.001)
        metric_name = early_stopping_cfg.get('monitor', 'conservation_error')
        restore_best = early_stopping_cfg.get('restore_best_weights', True)
    else:
        # 向后兼容：如果是布尔值
        early_stopping_enabled = bool(early_stopping_cfg)
        patience = train_cfg.get('patience', 400)
        min_delta = train_cfg.get('min_delta', 0.001)
        metric_name = train_cfg.get('early_stopping_metric', 'conservation_error')
        restore_best = train_cfg.get('restore_best_weights', True)
    
    best_metric = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    if early_stopping_enabled:
        logging.info(f"✅ Early stopping enabled: metric={metric_name}, patience={patience}, min_delta={min_delta}")
    else:
        logging.info("⚠️  Early stopping disabled")
    
    # 準備訓練資料
    training_data = prepare_training_data(config, device)
    input_normalizer = create_input_normalizer(config, training_data, is_vs_cfg, device)
    
    validation_freq = train_cfg.get('validation_freq', train_cfg.get('checkpoint_interval', 1000))
    if validation_freq <= 0:
        validation_freq = 1
    last_val_metrics: Optional[Dict[str, float]] = None
    
    # 訓練循環
    logger = logging.getLogger(__name__)
    
    # 🔧 初始化自適應採樣管理器
    loop_manager = None
    adaptive_cfg = config.get('adaptive_collocation', {})
    if adaptive_cfg.get('enabled', False):
        try:
            loop_manager = TrainingLoopManager(config)
            # 初始化 PDE 點到管理器
            if 'x_pde' in training_data and 'y_pde' in training_data:
                # 合併 x, y 成一個張量 [N, 2]
                pde_points = torch.cat([training_data['x_pde'], training_data['y_pde']], dim=1)
                loop_manager.setup_initial_points(pde_points)
                n_pde = len(loop_manager.current_pde_points) if loop_manager.current_pde_points is not None else 0
                logger.info(f"✅ Adaptive collocation enabled: {n_pde} initial PDE points")
            else:
                logger.warning("⚠️ No PDE points found, adaptive collocation disabled")
                loop_manager = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize adaptive collocation: {e}")
            loop_manager = None
    
    start_time = time.time()
    loss_dict = {'total_loss': 0.0, 'residual_loss': 0.0, 'bc_loss': 0.0, 'data_loss': 0.0}
    epoch = -1  # 初始化 epoch 變數
    max_epochs = train_cfg.get('epochs', train_cfg.get('max_epochs', 1000))
    
    for epoch in range(max_epochs):
        # 🔧 自適應殘差點採樣
        if loop_manager is not None:
            # 1. 更新訓練批次（應用新點與權重衰減）
            training_data = loop_manager.update_training_batch(training_data, epoch)
            
            # 2. 檢查是否需要重採樣
            if loop_manager.should_resample_collocation_points(
                epoch, 
                loss_dict.get('total_loss', float('inf')), 
                loss_dict
            ):
                # 提取域邊界
                domain_bounds = {
                    'x': (config['domain']['x_min'], config['domain']['x_max']),
                    'y': (config['domain']['y_min'], config['domain']['y_max'])
                }
                if 'z_min' in config['domain']:
                    domain_bounds['z'] = (config['domain']['z_min'], config['domain']['z_max'])
                if 't_min' in config['domain']:
                    domain_bounds['t'] = (config['domain']['t_min'], config['domain']['t_max'])
                
                try:
                    new_points, metrics = loop_manager.resample_collocation_points(
                        model, physics, domain_bounds, epoch, device
                    )
                    logger.info(f"🔄 Resampled {len(new_points)} collocation points at epoch {epoch}")
                    logger.info(f"   Metrics: {metrics}")
                except Exception as e:
                    logger.warning(f"⚠️ Resampling failed at epoch {epoch}: {e}")
        
        # 執行訓練步驟
        loss_dict = train_step(
            model, physics, losses, training_data, optimizer, weighters, epoch, device, config, input_normalizer
        )
        
        val_info = training_data.get('validation')
        if val_info and val_info.get('size', 0) > 0:
            if epoch % validation_freq == 0:
                validation_metrics = compute_validation_metrics(
                    model,
                    val_info,
                    device,
                    physics,
                    input_normalizer
                )
                if validation_metrics is not None:
                    last_val_metrics = validation_metrics
            if last_val_metrics is not None:
                loss_dict['val_loss'] = last_val_metrics['relative_l2']
                loss_dict['val_mse'] = last_val_metrics['mse']
        
        # 🚀🚀🚀 課程訓練：處理階段切換
        if '_curriculum_transition' in loss_dict and loss_dict['_curriculum_transition'] > 0.5:
            # 階段切換時更新學習率
            new_lr = loss_dict.get('_curriculum_lr', train_cfg['lr'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logger.info(f"📉 Curriculum: Updated learning rate to {new_lr:.6f}")
            
            # 保存階段檢查點（如果配置啟用）
            if config.get('logging', {}).get('save_stage_checkpoints', False):
                stage_name = loss_dict.get('_curriculum_stage', f'stage_{epoch}')
                checkpoint_dir = config.get('logging', {}).get('stage_checkpoint_dir', './checkpoints/curriculum')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"{stage_name}_epoch{epoch}.pth")
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_dict['total_loss'],
                    'config': config,
                    'stage_name': stage_name
                }
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"💾 Stage checkpoint saved: {checkpoint_path}")
        
        # 更新學習率調度器（僅在非課程訓練模式）
        if scheduler and weighters.get('curriculum') is None:
            scheduler.step()
        
        # 日誌輸出
        log_freq = config.get('training', {}).get('log_interval', config.get('logging', {}).get('log_freq', 50))
        if epoch % log_freq == 0:
            elapsed = time.time() - start_time
            log_msg = (
                f"Epoch {epoch:6d} | "
                f"Total: {loss_dict['total_loss']:.6f} | "
                f"Residual: {loss_dict['residual_loss']:.6f} | "
                f"BC: {loss_dict['bc_loss']:.6f} | "
                f"Data: {loss_dict['data_loss']:.6f} | "
                f"Time: {elapsed:.1f}s"
            )
            if 'val_loss' in loss_dict:
                log_msg += f" | Val(relL2): {loss_dict['val_loss']:.6f}"
            if 'val_mse' in loss_dict:
                log_msg += f" | Val(MSE): {loss_dict['val_mse']:.6f}"
            
            # 添加權重資訊到日誌中 (如果有的話)
            if 'weight_data' in loss_dict:
                log_msg += f" | W_data: {loss_dict['weight_data']:.3f}"
            if 'weight_residual' in loss_dict:
                log_msg += f" | W_residual: {loss_dict['weight_residual']:.3f}"
            if 'weight_boundary' in loss_dict:
                log_msg += f" | W_boundary: {loss_dict['weight_boundary']:.3f}"
            
            # 🔧 監控權重總和（驗證標準化）
            if 'weight_sum' in loss_dict:
                log_msg += f" | ΣW: {loss_dict['weight_sum']:.3f}"
            
            # 🔧 自適應採樣統計
            if loop_manager is not None:
                stats = loop_manager.get_summary()
                if stats.get('total_resamples', 0) > 0:
                    log_msg += f" | Resamples: {stats['total_resamples']}"
                    if 'new_points_count' in stats:
                        log_msg += f" | NewPts: {stats['new_points_count']}"
                
            logger.info(log_msg)
        
        # 🔧 收集自適應採樣統計
        if loop_manager is not None:
            loop_manager.collect_epoch_stats(epoch, loss_dict)
        
        # 檢查點保存
        if epoch % train_cfg.get('checkpoint_freq', 2000) == 0 and epoch > 0:
            save_checkpoint(model, optimizer, epoch, loss_dict['total_loss'], config)
            logger.info(f"Checkpoint saved at epoch {epoch}")
        
        # 🚀 Early Stopping 檢查
        if early_stopping_enabled:
            # 🔧 根據配置選擇監控指標
            if metric_name == 'val_loss':
                if 'val_loss' in loss_dict:
                    current_metric = loss_dict['val_loss']
                else:
                    current_metric = loss_dict['total_loss']
                    if epoch == 0:
                        logger.warning("[Early Stopping] 'val_loss' requested but validation data unavailable; falling back to 'total_loss'")
            elif metric_name == 'total_loss':
                current_metric = loss_dict['total_loss']
            elif metric_name == 'conservation_error' or metric_name == 'continuity_loss':
                current_metric = loss_dict['continuity_loss']
            elif metric_name in loss_dict:
                # 嘗試從 loss_dict 直接讀取
                current_metric = loss_dict[metric_name]
            else:
                # 備用：使用 total_loss
                current_metric = loss_dict['total_loss']
                if epoch == 0:
                    logger.warning(f"[Early Stopping] Metric '{metric_name}' not found, using 'total_loss'")
            
            # 檢查是否改善
            if current_metric < best_metric - min_delta:
                best_metric = current_metric
                best_epoch = epoch
                patience_counter = 0
                
                # 保存最佳模型狀態
                if restore_best:
                    best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
                logger.info(f"🎯 New best {metric_name}: {best_metric:.6f} at epoch {epoch}")
            else:
                patience_counter += 1
            
            # 檢查是否觸發 Early Stopping
            if patience_counter >= patience:
                logger.info(f"=" * 80)
                logger.info(f"🛑 Early stopping triggered at epoch {epoch}")
                logger.info(f"   Best {metric_name}: {best_metric:.6f} at epoch {best_epoch}")
                logger.info(f"   Patience counter: {patience_counter}/{patience}")
                logger.info(f"=" * 80)
                
                # 恢復最佳模型
                if restore_best and best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    logger.info(f"✅ Restored best model from epoch {best_epoch}")
                
                break
        
        # 驗證和檢查點
        if epoch % validation_freq == 0 and epoch > 0:
            logger.info("=== Validation checkpoint ===")
            # 這裡可以添加驗證邏輯
        
        # 早停檢查（原有的快速收斂檢查）
        if loss_dict['total_loss'] < 1e-6:
            logger.info(f"Early convergence at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.1f}s")
    
    # 保存最終模型
    final_checkpoint_path = save_checkpoint(model, optimizer, epoch + 1, loss_dict['total_loss'], config)
    logger.info(f"Final model saved: {final_checkpoint_path}")
    
    return {
        'final_loss': loss_dict['total_loss'],
        'training_time': total_time,
        'epochs_completed': epoch + 1,
        'checkpoint_path': final_checkpoint_path
    }


def main():
    """主程式入口"""
    parser = argparse.ArgumentParser(description='PINNs Inverse Training Script')
    parser.add_argument('--cfg', type=str, default='configs/defaults.yml',
                       help='Path to configuration file')
    parser.add_argument('--ensemble', action='store_true',
                       help='Run ensemble training for UQ')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # 載入配置
    config = load_config(args.cfg)
    
    # 🆕 根據物理類型自動設置模型輸入輸出維度
    if 'in_dim' not in config['model'] or 'out_dim' not in config['model']:
        physics_type = config.get('physics', {}).get('type', 'ns_2d')
        
        if physics_type == 'vs_pinn_channel_flow':
            # VS-PINN 3D: 輸入 (x, y, z)，輸出 (u, v, w, p)
            config['model']['in_dim'] = 3
            config['model']['out_dim'] = 4
            logger_msg = "VS-PINN 3D: in_dim=3 (x,y,z), out_dim=4 (u,v,w,p)"
        else:
            # 標準 PINN 2D: 輸入 (x, y)，輸出 (u, v, p)
            config['model']['in_dim'] = 2
            config['model']['out_dim'] = 3
            logger_msg = "Standard PINN 2D: in_dim=2 (x,y), out_dim=3 (u,v,p)"
        
        # 暫時用 print（因為 logger 尚未設置）
        print(f"🔧 Auto-configured model dimensions: {logger_msg}")
    
    # 設置日誌
    logger = setup_logging(config['logging']['level'])
    logger.info("=" * 60)
    logger.info("PINNs Inverse Reconstruction Training")
    logger.info("=" * 60)
    
    # 設置重現性
    set_random_seed(
        config['experiment']['seed'],
        config['reproducibility']['deterministic']
    )
    
    # 設置設備
    device = get_device(config['experiment']['device'])
    
    # 提前準備資料以提取統計資訊（用於自動輸出範圍）
    logger.info("Preparing training data to extract statistics...")
    training_data_sample = prepare_training_data(config, device)
    
    # 從快取中提取統計資訊（如果可用）
    statistics = None
    if '_channel_data_cache' in globals() and _channel_data_cache is not None:
        channel_data = _channel_data_cache.get('channel_data', {})
        if 'statistics' in channel_data:
            statistics = channel_data['statistics']
            logger.info(f"✅ Extracted statistics for auto output ranges:")
            logger.info(f"   u: {statistics.get('u', {}).get('range', 'N/A')}")
            logger.info(f"   v: {statistics.get('v', {}).get('range', 'N/A')}")
            logger.info(f"   p: {statistics.get('p', {}).get('range', 'N/A')}")
        else:
            logger.warning("⚠️  No statistics found in channel_data")
    else:
        logger.warning("⚠️  Channel data cache not available, will use hardcoded ranges")
    
    # 建立模型和物理模組
    model = create_model(config, device, statistics=statistics)
    physics = create_physics(config, device)
    losses = create_loss_functions(config, device)
    
    logger.info(f"Model architecture: {config['model']['type']}")
    logger.info(f"Input dimension: {config['model']['in_dim']}")
    logger.info(f"Output dimension: {config['model']['out_dim']}")
    
    # 安全讀取物理參數
    physics_type = config.get('physics', {}).get('type', 'unknown')
    if physics_type == 'vs_pinn_channel_flow':
        physics_params = config.get('physics', {}).get('physics_params', {})
        logger.info(f"Physics: VS-PINN Channel Flow with nu={physics_params.get('nu', 'N/A')}")
    else:
        nu = config.get('physics', {}).get('nu', config.get('physics', {}).get('physics_params', {}).get('nu', 'N/A'))
        logger.info(f"Physics: NS-2D with nu={nu}")
    
    if args.ensemble:
        logger.info("Running ensemble training...")
        ensemble_cfg = config['ensemble']
        
        models = []
        for i, seed in enumerate(ensemble_cfg['seeds']):
            logger.info(f"Training ensemble member {i+1}/{len(ensemble_cfg['seeds'])} (seed={seed})")
            
            # 重置隨機種子
            set_random_seed(seed, config['reproducibility']['deterministic'])
            
            # 建立新模型（使用相同的統計資訊）
            member_model = create_model(config, device, statistics=statistics)
            
            # 訓練
            train_result = train_model(member_model, physics, losses, config, device)
            models.append(member_model)
            
            logger.info(f"Member {i+1} final loss: {train_result['final_loss']:.6f}")
        
        # 儲存模型列表（暫時不使用 EnsembleWrapper）
        logger.info(f"Ensemble training completed with {len(models)} members")
        logger.info("Note: EnsembleWrapper not implemented yet - models stored as list")
        
    else:
        logger.info("Running single model training...")
        train_result = train_model(model, physics, losses, config, device)
        logger.info(f"Training completed. Final loss: {train_result['final_loss']:.6f}")
    
    logger.info("Training script finished successfully!")


if __name__ == "__main__":
    main()
