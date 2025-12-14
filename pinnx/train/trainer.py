"""
PINNs 訓練器模組

提供完整的訓練循環管理，包含：
- 單步訓練與梯度計算
- 驗證指標計算
- 檢查點保存與早停
- 學習率與權重調度
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.amp.grad_scaler import GradScaler  # 明確導入 GradScaler
from torch.utils.tensorboard import SummaryWriter

from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.train.loop import TrainingLoopManager, apply_point_weights_to_loss
from pinnx.train.loss_manager import LossManager  # type: ignore
from pinnx.utils.normalization import InputNormalizer, NormalizationConfig, DataNormalizer
from pinnx.evals.metrics import relative_L2
from pinnx.physics.turbulence_utils import preprocess_rans_prior, preprocess_rans_prior_from_config  # type: ignore


class Trainer:
    """
    PINNs 訓練器
    
    管理完整的訓練循環，包含：
    - 優化器與學習率調度
    - 損失函數與動態權重調整
    - 檢查點保存與載入
    - 驗證與早停機制
    - 自適應採樣（可選）
    
    Attributes:
        model (nn.Module): PINN 模型
        physics (Any): 物理方程模組（支援 NSEquations2D 或 VS-PINN）
        losses (Dict[str, nn.Module]): 損失函數字典
        config (Dict[str, Any]): 完整訓練配置
        device (torch.device): 計算設備
        
        optimizer (torch.optim.Optimizer): 優化器
        lr_scheduler: 學習率調度器（可選）
        weight_scheduler: 權重調度器（可選）
        input_normalizer (InputNormalizer): 輸入標準化器（可選）
        
        epoch (int): 當前訓練 epoch
        history (Dict[str, List]): 訓練歷史記錄
    """
    
    def __init__(
        self,
        model: nn.Module,
        physics: Any,  # 支援 NSEquations2D 或 VS-PINN
        losses: Dict[str, nn.Module],
        config: Dict[str, Any],
        device: torch.device,
        weighters: Optional[Dict[str, Any]] = None,
        input_normalizer: Optional[InputNormalizer] = None,
        channel_data_cache: Optional[Dict[str, Any]] = None,
        training_data: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        初始化訓練器
        
        Args:
            model: PINN 模型
            physics: 物理方程模組
            losses: 損失函數字典
            config: 完整訓練配置
            device: 計算設備
            weighters: 損失權重器字典（可選）
            input_normalizer: 輸入標準化器（可選）
            channel_data_cache: 通道流資料快取（可選）
            training_data: 訓練資料（用於自動計算標準化統計量，可選）
        """
        self.model = model
        self.physics = physics
        self.losses = losses
        self.config = config
        self.device = device
        self.weighters = weighters or {}
        self.input_normalizer = input_normalizer
        self.channel_data_cache = channel_data_cache or {}
        
        # 訓練配置提取
        self.train_cfg = config['training']
        self.loss_cfg = config.get('losses', {})
        self.log_cfg = config.get('logging', {})
        self.physics_type = config.get('physics', {}).get('type', '')
        self.is_vs_cfg = self.physics_type == 'vs_pinn_channel_flow'
        
        # ⭐ Phase 5: 檢測模型實際輸入維度（支援 2D/3D 混合配置）
        self.model_input_dim = self._detect_model_input_dim(model, config)
        logging.info(f"🔍 檢測到模型輸入維度: {self.model_input_dim}D")
        
        # ✅ TASK-008: 初始化輸出變量標準化器（傳遞 training_data 以支援自動計算統計量）
        self.data_normalizer = DataNormalizer.from_config(config, training_data=training_data)
        logging.info(f"📐 DataNormalizer 初始化: {self.data_normalizer}")
        
        # 訓練狀態
        self.epoch = 0
        self.global_step = 0
        self.history = {
            'train_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'val_loss': [],
            'lr': [],
        }
        
        # 驗證相關
        self.validation_data = None
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.patience_counter = 0
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None
        
        # 檢查點配置
        self.checkpoint_dir = Path(config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_interval = self.train_cfg.get('checkpoint_interval', 100)
        
        # TensorBoard 配置
        self.use_tensorboard = self.log_cfg.get('tensorboard', False)
        self.writer: Optional[SummaryWriter] = None
        if self.use_tensorboard:
            # TensorBoard 日誌目錄：runs/<experiment_name>
            exp_name = config.get('experiment', {}).get('name', 'default_experiment')
            tensorboard_dir = Path(config.get('output', {}).get('tensorboard_dir', f'runs/{exp_name}'))
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logging.info(f"✅ TensorBoard 已啟用，日誌目錄: {tensorboard_dir}")
        
        # 訓練資料（待外部設置）
        self.training_data: Dict[str, torch.Tensor] = {}
        
        # 🆕 初始化 Prior Loss Manager（若配置啟用）
        self.prior_loss_manager: Optional[PriorLossManager] = None
        self._setup_prior_loss_manager()
        
        # 🆕 Phase 1-3: 初始化 LossManager（統一損失計算）
        self.loss_manager = LossManager(
            config=config,
            physics=physics,
            model=model,
            device=device,
            data_normalizer=self.data_normalizer,
            prior_loss_manager=self.prior_loss_manager,
            weighters=self.weighters,
            losses=losses
        )
        logging.info(f"✅ LossManager 初始化完成")
        
        # 初始化訓練組件
        self._setup_optimizer()
        self._setup_amp()  # ⭐ P0.2: AMP 混合精度
        self._setup_schedulers()
        self._setup_early_stopping()
        self._setup_adaptive_sampling()
        self._setup_fourier_annealing()
        # RANS 權重預熱已移除（2025-10-14）
        self._configure_input_transform()
        
        logging.info(f"✅ Trainer 初始化完成（設備: {device}）")
    
    def _detect_model_input_dim(self, model: nn.Module, config: Dict[str, Any]) -> int:
        """
        檢測模型的實際輸入維度
        
        優先級：
        1. 配置文件中的 model.in_dim
        2. 模型 wrapper 的 input_min 長度（ManualScalingWrapper）
        3. 基礎模型的 in_dim 屬性
        4. 回退到物理配置的域維度檢測
        
        Returns:
            int: 模型輸入維度（2 或 3）
        """
        # 優先：從配置文件讀取
        model_cfg = config.get('model', {})
        if 'in_dim' in model_cfg:
            return int(model_cfg['in_dim'])
        
        # 次要：從 ManualScalingWrapper 讀取
        if hasattr(model, 'input_min'):
            return model.input_min.numel()
        
        # 第三：從基礎模型讀取
        base_model = model.base_model if hasattr(model, 'base_model') else model
        if hasattr(base_model, 'in_dim'):
            return int(base_model.in_dim)
        
        # 回退：從物理配置推斷
        domain_cfg = config.get('physics', {}).get('domain', {})
        if 'z_range' in domain_cfg:
            return 3
        return 2
    
    def _configure_input_transform(self) -> None:
        """Propagate input normalization metadata to the model if needed."""
        if self.input_normalizer is None:
            return
        try:
            self.input_normalizer.to(self.device)
        except AttributeError:
            pass
        
        if hasattr(self.model, 'configure_fourier_input'):
            metadata = self.input_normalizer.get_metadata()
            self.model.configure_fourier_input(metadata)
    
    def _infer_variable_order(
        self,
        out_dim: int,
        context: str = "",
        data_batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> List[str]:
        """
        根據輸出維度推斷對應的物理變量順序。
        
        優先級：
        1. 配置 (model.output_variables / model.variable_names / model.variables)
        2. 模型屬性 (variable_names 或 get_variable_names())
        3. 常用啟發式（u,v,w,p,S）
        """
        if out_dim <= 0:
            return []
        
        model_cfg = self.config.get('model', {})
        explicit_order = model_cfg.get('output_variables') or \
            model_cfg.get('variable_names') or \
            model_cfg.get('variables')
        if explicit_order:
            explicit = list(explicit_order)
            if len(explicit) >= out_dim:
                return explicit[:out_dim]
        
        attr_order = getattr(self.model, 'variable_names', None)
        if attr_order is None and hasattr(self.model, 'get_variable_names'):
            try:
                attr_order = self.model.get_variable_names()
            except Exception:
                attr_order = None
        if attr_order:
            attr_list = list(attr_order)
            if len(attr_list) >= out_dim:
                return attr_list[:out_dim]
        
        if out_dim == 1:
            return ['u']
        if out_dim == 2:
            return ['u', 'v']
        if out_dim == 3:
            return ['u', 'v', 'p']
        if out_dim == 4:
            return ['u', 'v', 'w', 'p']
        if out_dim == 5:
            return ['u', 'v', 'w', 'p', 'S']
        
        default_order = ['u', 'v', 'w', 'p', 'S']
        if out_dim <= len(default_order):
            return default_order[:out_dim]
        
        return [f'var_{i}' for i in range(out_dim)]
    
    def _prepare_model_coords(
        self,
        coord_tensor: torch.Tensor,
        require_grad: bool = False,
        is_vs_pinn: Optional[bool] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        準備模型輸入坐標（標準化 + 縮放）
        
        Args:
            coord_tensor: 物理坐標 [N, spatial_dim + time_dim]
            require_grad: 是否啟用梯度追蹤
            is_vs_pinn: 是否套用 VS-PINN 縮放（None 則自動偵測）
        
        Returns:
            (coords_physical, coords_norm, model_coords):
            - coords_physical: 物理坐標（用於 PDE 自動微分）
            - coords_norm: 標準化坐標（若有 InputNormalizer）
            - model_coords: 最終模型輸入（可能包含 VS-PINN 縮放）
        """
        # 1. 物理坐標（可選梯度追蹤）
        coords_physical = coord_tensor
        if require_grad and not coords_physical.requires_grad:
            coords_physical.requires_grad_(True)
        
        # 2. 輸入標準化（可選）
        if self.input_normalizer is not None:
            coords_norm = self.input_normalizer.transform(coords_physical)
        else:
            coords_norm = coords_physical
        
        # 3. VS-PINN 坐標縮放（可選）
        if is_vs_pinn is None:
            is_vs_pinn = hasattr(self.physics, 'scale_coordinates')
        
        if is_vs_pinn and hasattr(self.physics, 'scale_coordinates'):
            # 時間維度需單獨處理（VS-PINN 僅縮放空間坐標）
            if coords_norm.shape[1] > 3:  # 包含時間維度
                coords_spatial = coords_norm[:, :3]
                coords_time = coords_norm[:, 3:]
                scaled_spatial = self.physics.scale_coordinates(coords_spatial)
                model_coords = torch.cat([scaled_spatial, coords_time], dim=1)
            else:
                model_coords = self.physics.scale_coordinates(coords_norm)
        else:
            model_coords = coords_norm
        
        return coords_physical, coords_norm, model_coords
    
    def _setup_optimizer(self):
        """配置優化器"""
        # 處理 optimizer 配置為字串或字典的情況
        optimizer_raw = self.train_cfg.get('optimizer', {})
        if isinstance(optimizer_raw, str):
            # 簡單配置：optimizer: 'adam'
            optimizer_name = optimizer_raw.lower()
            optimizer_cfg = {}
        elif isinstance(optimizer_raw, dict):
            # 複雜配置：optimizer: {type: 'adam', lr: 0.001}
            optimizer_name = optimizer_raw.get('type', 'adam').lower()
            optimizer_cfg = optimizer_raw
        else:
            # 預設為 Adam
            optimizer_name = 'adam'
            optimizer_cfg = {}
        
        # 從配置中提取參數（優先使用 optimizer_cfg，否則從 train_cfg）
        lr = optimizer_cfg.get('lr', self.train_cfg.get('lr', 1e-3))
        weight_decay = optimizer_cfg.get('weight_decay', self.train_cfg.get('weight_decay', 0.0))
        
        if optimizer_name == 'soap':
            try:
                from pinnx.optim.soap import SOAP  # Import from our implementation
            except ImportError as exc:
                raise ImportError("SOAP 優化器未找到，請檢查 pinnx/optim/soap.py") from exc
            
            # 提取 SOAP 專用參數
            precondition_frequency = optimizer_cfg.get('precondition_frequency', 2)
            shampoo_beta = optimizer_cfg.get('shampoo_beta', -1)
            betas = optimizer_cfg.get('betas', (0.9, 0.999))
            
            self.optimizer = SOAP(
                self.model.parameters(),
                lr=lr,
                betas=betas,
                weight_decay=weight_decay,
                precondition_frequency=precondition_frequency,
                shampoo_beta=shampoo_beta
            )
            logging.info(f"✅ 使用 SOAP 優化器 (lr={lr}, betas={betas}, precond_freq={precondition_frequency})")
        
        elif optimizer_name == 'lbfgs':
            self.optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=lr,
                max_iter=optimizer_cfg.get('max_iter', 20),
                history_size=optimizer_cfg.get('history_size', 100),
                line_search_fn=optimizer_cfg.get('line_search_fn', 'strong_wolfe')
            )
            logging.info(f"✅ 使用 L-BFGS 優化器（lr={lr}）")
        
        elif optimizer_name == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=tuple(optimizer_cfg.get('betas', [0.9, 0.999]))
            )
            logging.info(f"✅ 使用 AdamW 優化器（lr={lr}, wd={weight_decay}）")
        
        else:  # 預設 Adam
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
            logging.info(f"✅ 使用 Adam 優化器（lr={lr}, wd={weight_decay}）")
    
    def _setup_amp(self):
        """
        配置自動混合精度訓練（AMP）
        
        策略：
        - Forward Pass: FP32（物理殘差計算數值穩定）
        - Backward Pass: FP16（節省記憶體）
        - 僅在 Adam + CUDA 時啟用（L-BFGS 不兼容）
        """
        amp_cfg = self.train_cfg.get('amp', {})
        self.use_amp = amp_cfg.get('enabled', False)
        
        # AMP 支援檢查：Adam + (CUDA 或 MPS)
        is_adam = isinstance(self.optimizer, torch.optim.Adam)
        is_cuda = self.device.type == 'cuda'
        is_mps = self.device.type == 'mps'
        
        if self.use_amp and not is_adam:
            logging.warning(
                "⚠️ AMP 僅支援 Adam 優化器，當前使用 "
                f"{type(self.optimizer).__name__}，已禁用 AMP"
            )
            self.use_amp = False
        
        # MPS 限制：GradScaler 不支援（float64 問題）
        if self.use_amp and is_mps:
            logging.warning(
                "⚠️ MPS 後端的 GradScaler 存在已知問題（不支援 float64）\n"
                "   建議：(1) 使用 CUDA 設備，或 (2) 關閉 AMP\n"
                "   已自動禁用 AMP"
            )
            self.use_amp = False
        
        if self.use_amp and not is_cuda:
            logging.warning(
                "⚠️ AMP 僅在 CUDA 環境完全支援，當前設備為 "
                f"{self.device}，已禁用 AMP"
            )
            self.use_amp = False
        
        # 初始化 GradScaler（僅 CUDA）
        if self.use_amp:
            self.scaler = GradScaler(
                'cuda',
                init_scale=2.0**16,  # 初始縮放因子
                growth_factor=2.0,   # 成長因子
                backoff_factor=0.5,  # 回退因子
                growth_interval=2000,  # 增長間隔
                enabled=True
            )
            logging.info(
                "✅ AMP 已啟用（Forward: FP32, Backward: FP16）\n"
                f"   - 優化器: {type(self.optimizer).__name__}\n"
                f"   - 設備: {self.device} (CUDA)\n"
                f"   - GradScaler 初始 scale: {self.scaler.get_scale():.0f}"
            )
        else:
            # 創建禁用的 scaler（統一接口）
            device_type = 'cuda' if is_cuda else 'cpu'
            self.scaler = GradScaler(device_type, enabled=False)
            if amp_cfg.get('enabled', False):
                logging.info("ℹ️ AMP 配置已禁用（不符合啟用條件）")
    
    def _setup_schedulers(self):
        """配置學習率與權重調度器"""
        # 學習率調度器
        scheduler_cfg = self.train_cfg.get('lr_scheduler', {})
        scheduler_type = scheduler_cfg.get('type', None)
        
        if scheduler_type == 'warmup_cosine':
            # Warmup + CosineAnnealing 組合調度器
            from pinnx.train.schedulers import WarmupCosineScheduler
            warmup_epochs = scheduler_cfg.get('warmup_epochs', 10)
            max_epochs = self.train_cfg.get('epochs', 1000)
            base_lr = self.optimizer.param_groups[0]['lr']
            min_lr = scheduler_cfg.get('min_lr', 1e-6)
            
            self.lr_scheduler = WarmupCosineScheduler(
                self.optimizer,
                warmup_epochs=warmup_epochs,
                max_epochs=max_epochs,
                base_lr=base_lr,
                min_lr=min_lr
            )
            logging.info(f"✅ 使用 WarmupCosine 調度器 (warmup={warmup_epochs}, max={max_epochs})")
        
        elif scheduler_type == 'cosine_warm_restarts':
            # CosineAnnealing with Warm Restarts
            T_0 = scheduler_cfg.get('T_0', 100)
            T_mult = scheduler_cfg.get('T_mult', 1)
            eta_min = scheduler_cfg.get('eta_min', 1e-6)
            
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=T_0,
                T_mult=T_mult,
                eta_min=eta_min
            )
            logging.info(f"✅ 使用 CosineAnnealingWarmRestarts (T_0={T_0}, T_mult={T_mult})")
        
        elif scheduler_type == 'cosine':
            # 標準 CosineAnnealing 調度器
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.train_cfg.get('epochs', 1000),
                eta_min=scheduler_cfg.get('min_lr', 1e-6)
            )
            logging.info("✅ 使用 Cosine 學習率調度器")
        
        elif scheduler_type == 'exponential':
            # 指數衰減調度器
            gamma = scheduler_cfg.get('gamma', 0.999)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=gamma
            )
            logging.info(f"✅ 使用 Exponential 調度器 (gamma={gamma})")
        
        elif scheduler_type == 'step':
            # StepLR 調度器
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_cfg.get('step_size', 100),
                gamma=scheduler_cfg.get('gamma', 0.5)
            )
            logging.info("✅ 使用 Step 學習率調度器")
        
        elif scheduler_type in ['none', None]:
            # 無調度器，使用固定學習率
            self.lr_scheduler = None
            logging.info("ℹ️ 未配置學習率調度器，使用固定學習率")
        
        else:
            # 不支援的類型
            logging.warning(
                f"⚠️ 未知的調度器類型 '{scheduler_type}'，使用固定學習率。"\
                f"支援的類型：'warmup_cosine', 'cosine_warm_restarts', 'cosine', "\
                f"'exponential', 'step', 'none'"
            )
            self.lr_scheduler = None
        
        # 權重調度器（暫時保留為 None，由外部管理）
        self.weight_scheduler = None
    
    def _setup_early_stopping(self):
        """配置早停機制"""
        self.early_stopping_cfg = self.train_cfg.get('early_stopping', {})
        self.early_stopping_enabled = self.early_stopping_cfg.get('enabled', False)
        self.patience = self.early_stopping_cfg.get('patience', 50)
        self.min_delta = self.early_stopping_cfg.get('min_delta', 1e-6)
        
        # ⭐ 快速收斂閾值（可配置，預設禁用）
        self.convergence_threshold = self.early_stopping_cfg.get('convergence_threshold', None)
        
        if self.early_stopping_enabled:
            logging.info(f"✅ 早停機制啟用（patience={self.patience}, min_delta={self.min_delta}）")
        if self.convergence_threshold is not None:
            logging.info(f"✅ 快速收斂檢查啟用（threshold={self.convergence_threshold:.2e}）")
    
    def _setup_adaptive_sampling(self):
        """配置自適應採樣"""
        adaptive_cfg = self.train_cfg.get('adaptive_sampling', {})
        self.adaptive_sampling_enabled = adaptive_cfg.get('enabled', False)
        
        if self.adaptive_sampling_enabled:
            self.loop_manager = TrainingLoopManager(self.config)
            logging.info("✅ 自適應採樣啟用")
        else:
            self.loop_manager = None
    
    def _setup_fourier_annealing(self):
        """配置 Fourier 特徵退火調度器"""
        from pinnx.train.fourier_annealing import (
            FourierAnnealingScheduler, 
            create_default_annealing,
            create_channel_flow_annealing
        )
        
        # 檢查配置中是否啟用退火
        annealing_cfg = self.config.get('fourier_annealing', {})
        if not annealing_cfg.get('enabled', False):
            self.fourier_annealing = None
            return
        
        # 提取退火策略
        strategy = annealing_cfg.get('strategy', 'conservative')
        
        # 根據策略創建調度器
        if strategy in ['conservative', 'aggressive', 'fine']:
            # 使用預設策略
            stages = create_default_annealing(strategy)
            axes_names = annealing_cfg.get('axes_names', ['x', 'y', 'z'])
            self.fourier_annealing = FourierAnnealingScheduler(stages, axes_names=axes_names)
            logging.info(f"✅ Fourier 退火啟用（策略: {strategy}）")
        
        elif strategy == 'channel_flow':
            # 使用通道流專用配置
            per_axis_config = create_channel_flow_annealing()
            # 將每軸配置轉換為調度器格式
            # 使用 x 軸作為全局階段，y/z 作為每軸覆蓋
            global_stages = per_axis_config['x']
            per_axis_stages = {'y': per_axis_config['y'], 'z': per_axis_config['z']}
            self.fourier_annealing = FourierAnnealingScheduler(
                global_stages, 
                per_axis_stages=per_axis_stages,
                axes_names=['x', 'y', 'z']
            )
            logging.info("✅ Fourier 退火啟用（通道流專用配置）")
        
        elif strategy == 'custom':
            # 自定義配置（從配置文件讀取）
            stages_cfg = annealing_cfg.get('stages', [])
            if not stages_cfg:
                logging.warning("⚠️ 自定義退火策略未提供階段配置，禁用退火")
                self.fourier_annealing = None
                return
            
            from pinnx.train.fourier_annealing import AnnealingStage
            stages = [
                AnnealingStage(s['end_ratio'], s['frequencies'], s.get('description', ''))
                for s in stages_cfg
            ]
            axes_names = annealing_cfg.get('axes_names', ['x', 'y', 'z'])
            self.fourier_annealing = FourierAnnealingScheduler(stages, axes_names=axes_names)
            logging.info(f"✅ Fourier 退火啟用（自定義配置，{len(stages)} 階段）")
        
        else:
            logging.warning(f"⚠️ 未知退火策略 '{strategy}'，禁用退火")
            self.fourier_annealing = None
    
    # RANS 方法已移除（2025-10-14）：
    # - _setup_rans_warmup()
    # - _update_rans_weights()
    
    def _setup_prior_loss_manager(self):
        """初始化 Prior Loss Manager（若配置啟用 RANS/低保真先驗）"""
        lowfi_cfg = self.config.get('lowfi_prior', {})
        
        if not lowfi_cfg.get('enabled', False):
            self.prior_loss_manager = None
            logging.info("ℹ️  低保真先驗未啟用")
            return
        
        # 提取低保真先驗配置
        consistency_weight = lowfi_cfg.get('consistency_weight', 0.3)
        variable_weights = lowfi_cfg.get('variable_weights', {'u': 1.0, 'v': 1.0, 'p': 0.5})
        distance_metric = lowfi_cfg.get('distance_metric', 'mse')
        
        # 創建 PriorLossManager
        self.prior_loss_manager = PriorLossManager(
            consistency_weight=consistency_weight,
            statistical_weight=0.0,  # 暫不使用統計一致性
            conservation_weight=0.0,  # 暫不使用守恆定律
            symmetry_weight=0.0       # 暫不使用對稱性
        )
        
        # 更新低保真一致性損失的配置
        self.prior_loss_manager.low_fidelity_loss.consistency_weight = consistency_weight
        self.prior_loss_manager.low_fidelity_loss.variable_weights = variable_weights
        self.prior_loss_manager.low_fidelity_loss.distance_metric = distance_metric
        
        logging.info(f"✅ Prior Loss Manager 初始化完成")
        logging.info(f"   - Consistency Weight: {consistency_weight}")
        logging.info(f"   - Variable Weights: {variable_weights}")
        logging.info(f"   - Distance Metric: {distance_metric}")
    
    def step(
        self,
        data_batch: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, Any]:
        """
        執行單步訓練（使用 LossManager 重構版）
    
        核心改進：
        - 所有損失計算委派給 LossManager
        - step() 只負責：資料預處理、模型forward、Loss組合、反向傳播
        - 大幅減少代碼重複與嵌套
    
        Args:
            data_batch: 訓練資料批次
            epoch: 當前 epoch
    
        Returns:
            包含損失和指標的字典
        """
        self.optimizer.zero_grad()
    
        # ==================== 0. 前置準備 ====================
        is_vs_pinn = 'z_pde' in data_batch and hasattr(self.physics, 'compute_momentum_residuals')
    
        # ==================== 1. PDE 點前向傳播 ====================
        # 準備 PDE 點坐標
        x_pde, y_pde = data_batch['x_pde'], data_batch['y_pde']
        z_pde = data_batch.get('z_pde')
        t_pde = data_batch.get('t_pde')
    
        if t_pde is not None:
            t_pde = t_pde.to(self.device).requires_grad_(True)
    
        # 構建輸入張量
        spatial_components = [x_pde, y_pde]
        if z_pde is not None:
            spatial_components.append(z_pde)
        coords_spatial = torch.cat(spatial_components, dim=1)
    
        if t_pde is not None and self.model_input_dim > coords_spatial.shape[1]:
            coords_full = torch.cat([coords_spatial, t_pde], dim=1)
        else:
            coords_full = coords_spatial
    
        # 準備模型輸入（使用實例方法）
        coords_full_physical, coords_full_norm, model_coords_pde = self._prepare_model_coords(
            coords_full, require_grad=True, is_vs_pinn=is_vs_pinn
        )
        coords_pde_physical = coords_full_physical
    
        # 模型預測 + 反標準化
        u_pred_norm = self.model(model_coords_pde)
        var_order = self._infer_variable_order(u_pred_norm.shape[1], context='pde')
        u_pred_pde_physical_raw = self.data_normalizer.denormalize_batch(u_pred_norm, var_order=var_order)
        u_pred_pde_physical: torch.Tensor = u_pred_pde_physical_raw if isinstance(u_pred_pde_physical_raw, torch.Tensor) else torch.tensor(u_pred_pde_physical_raw, device=self.device)
    
        # ==================== 2. 邊界條件點前向傳播 ====================
        spatial_bc = [data_batch['x_bc'], data_batch['y_bc']]
        if 'z_bc' in data_batch:
            spatial_bc.append(data_batch['z_bc'])
        coords_bc = torch.cat(spatial_bc, dim=1)
    
        t_bc = data_batch.get('t_bc')
        if t_bc is not None:
            t_bc = t_bc.to(self.device)
    
        final_bc_input = torch.cat([coords_bc, t_bc], dim=1) if t_bc is not None and self.model_input_dim > coords_bc.shape[1] else coords_bc
        coords_bc_physical, coords_bc_norm, model_coords_bc = self._prepare_model_coords(
            final_bc_input, require_grad=False, is_vs_pinn=is_vs_pinn
        )
    
        u_bc_pred_norm = self.model(model_coords_bc)
        var_order_bc = self._infer_variable_order(u_bc_pred_norm.shape[1], context='bc')
        u_bc_pred_phys_raw = self.data_normalizer.denormalize_batch(u_bc_pred_norm, var_order=var_order_bc)
        u_bc_pred_phys: torch.Tensor = u_bc_pred_phys_raw if isinstance(u_bc_pred_phys_raw, torch.Tensor) else torch.tensor(u_bc_pred_phys_raw, device=self.device)
    
        # ==================== 3. 感測器點前向傳播 ====================
        spatial_sensors = [data_batch['x_sensors'], data_batch['y_sensors']]
        if 'z_sensors' in data_batch:
            spatial_sensors.append(data_batch['z_sensors'])
        coords_sensors = torch.cat(spatial_sensors, dim=1)
    
        t_sensors = data_batch.get('t_sensors')
        if t_sensors is not None:
            t_sensors = t_sensors.to(self.device)
    
        final_sensor_input = torch.cat([coords_sensors, t_sensors], dim=1) if t_sensors is not None and self.model_input_dim > coords_sensors.shape[1] else coords_sensors
        coords_sensors_physical, coords_sensors_norm, model_coords_sensors = self._prepare_model_coords(
            final_sensor_input, require_grad=False, is_vs_pinn=is_vs_pinn
        )
    
        u_sensors_pred_norm = self.model(model_coords_sensors)
        var_order_sensors = self._infer_variable_order(u_sensors_pred_norm.shape[1], context='sensors', data_batch=data_batch)
        u_sensors_pred_phys_raw = self.data_normalizer.denormalize_batch(u_sensors_pred_norm, var_order=var_order_sensors)
        u_sensors_pred_phys: torch.Tensor = u_sensors_pred_phys_raw if isinstance(u_sensors_pred_phys_raw, torch.Tensor) else torch.tensor(u_sensors_pred_phys_raw, device=self.device)
    
        # ==================== 4. 使用 LossManager 計算所有損失 ====================
        # 4.1 PDE 損失
        pde_losses = self.loss_manager.compute_pde_loss(
            coords_pde_physical=coords_pde_physical,
            model_coords_pde=model_coords_pde,
            u_pred_pde_physical=u_pred_pde_physical,
            data_batch=data_batch,
            epoch=epoch,
            is_vs_pinn=is_vs_pinn
        )
    
        # 4.2 邊界條件損失
        bc_losses = self.loss_manager.compute_bc_loss(
            u_bc_pred_phys=u_bc_pred_phys,
            data_batch=data_batch,
            epoch=epoch
        )
    
        # 4.3 資料監督損失
        data_losses = self.loss_manager.compute_data_loss(
            u_sensors_pred_phys=u_sensors_pred_phys,
            data_batch=data_batch
        )
    
        # 4.4 低保真先驗損失
        prior_losses = self.loss_manager.compute_lowfi_prior_loss(
            u_pred_pde_physical=u_pred_pde_physical,
            coords_pde_physical=coords_pde_physical,
            data_batch=data_batch,
            epoch=epoch
        )
    
        # 4.5 均值約束損失
        mean_constraint_loss = self.loss_manager.compute_mean_constraint_loss(
            u_pred_pde_physical=u_pred_pde_physical,
            epoch=epoch
        )
    
        # ==================== 5. 動態權重調整與損失組合 ====================
        # 5.1 課程學習權重
        curriculum_config, loss_cfg = self.loss_manager.apply_curriculum_weights(epoch)
    
        # 5.2 構建損失項字典（供 GradNorm 使用）
        loss_terms = {
            'data': data_losses['data_loss'],
            'momentum_x': pde_losses['momentum_x_loss'],
            'momentum_y': pde_losses['momentum_y_loss'],
            'continuity': pde_losses['continuity_loss'],
        }
    
        if hasattr(self.physics, 'compute_periodic_loss'):
            loss_terms['periodic_x'] = bc_losses['periodic_x_loss']
            loss_terms['periodic_y'] = bc_losses['periodic_y_loss']
        else:
            loss_terms['wall_constraint'] = bc_losses['wall_loss']
    
        if is_vs_pinn:
            loss_terms['momentum_z'] = pde_losses['momentum_z_loss']
    
        # 5.3 GradNorm 動態權重
        gradnorm_weights, gradnorm_ratio = self.loss_manager.apply_gradnorm_weights(loss_terms)
    
        # 5.4 組合所有損失
        all_losses = {**pde_losses, **bc_losses, **data_losses, **prior_losses}
        all_losses['mean_constraint_loss'] = mean_constraint_loss
    
        total_loss, result = self.loss_manager.combine_losses(
            loss_dict=all_losses,
            loss_cfg=loss_cfg,
            gradnorm_ratio=gradnorm_ratio,
            is_vs_pinn=is_vs_pinn,
            epoch=epoch
        )
    
        # ==================== 6. 反向傳播與優化 ====================
        # AMP 混合精度
        scaled_loss = self.scaler.scale(total_loss)
        scaled_loss.backward()
    
        # 梯度裁剪
        if self.train_cfg.get('gradient_clip', 0.0) > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_cfg['gradient_clip']
            )
    
        # 優化器步進
        if isinstance(self.optimizer, torch.optim.LBFGS):
            def closure():
                return total_loss
            self.optimizer.step(closure)
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()
    
        # 學習率調度器更新
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'current_step'):
            self.lr_scheduler.step()
    
        # ==================== 7. 附加課程學習與 GradNorm 信息 ====================
        if curriculum_config is not None:
            is_curriculum_transition = curriculum_config.get('is_transition', False)
            result['_curriculum_transition'] = 1.0 if is_curriculum_transition else 0.0
            result['_curriculum_stage'] = curriculum_config.get('stage_name', 'unknown')
            if 'lr' in curriculum_config:
                result['_curriculum_lr'] = curriculum_config['lr']
    
        if gradnorm_weights is not None:
            result['gradnorm_weights'] = {k: float(v) for k, v in gradnorm_weights.items()}
    
        return result

    def validate(self) -> Optional[Dict[str, float]]:
        """
        計算驗證指標（MSE 與 relative L2）
        
        Returns:
            驗證指標字典，若無驗證資料則返回 None
            - 'mse': 均方誤差
            - 'relative_l2': 相對 L2 誤差
        """
        # 檢查驗證資料是否存在
        if self.validation_data is None:
            return None
        
        if self.validation_data.get('size', 0) == 0:
            return None
        
        coords = self.validation_data.get('coords')
        targets = self.validation_data.get('targets')
        
        if coords is None or targets is None or coords.numel() == 0 or targets.numel() == 0:
            return None
        
        # 移動至設備
        coords = coords.to(self.device)
        targets = targets.to(self.device)
        
        # 保存訓練狀態
        training_mode = self.model.training
        self.model.eval()
        
        with torch.no_grad():
            # 使用共享的坐標預處理方法
            _, _, coords_for_model = self._prepare_model_coords(
                coords, require_grad=False, is_vs_pinn=None
            )
            
            # 模型預測（標準化空間輸出）
            preds_norm = self.model(coords_for_model)
            
            # ✅ 反標準化為物理量（與真實物理量比較）
            var_order_val = self._infer_variable_order(preds_norm.shape[1], context='validation')
            preds_phys_raw = self.data_normalizer.denormalize_batch(preds_norm, var_order=var_order_val)
            preds_phys: torch.Tensor = preds_phys_raw if isinstance(preds_phys_raw, torch.Tensor) else torch.tensor(preds_phys_raw, device=self.device)  # type: ignore
            
            # 處理維度不匹配（僅比較可用的場分量）
            n_pred = preds_phys.shape[1]
            n_targets = targets.shape[1]
            n_common = min(n_pred, n_targets)
            
            if n_pred != n_targets:
                logging.debug(
                    f"[Validation] 輸出維度不匹配 (pred={n_pred}, target={n_targets})；"
                    f"比較前 {n_common} 個分量。"
                )
            
            preds_final = preds_phys[:, :n_common]
            targets_final = targets[:, :n_common]
            
            # ✅ 計算誤差指標（物理空間）
            diff = preds_final - targets_final
            mse = torch.mean(diff**2).item()
            rel_l2 = relative_L2(preds_final, targets_final).mean().item()
        
        # 恢復訓練狀態
        if training_mode:
            self.model.train()
        
        return {
            'mse': mse,
            'relative_l2': rel_l2
        }
    
    def train(self) -> Dict[str, Any]:
        """
        執行完整訓練循環
        
        Returns:
            訓練歷史與最終結果字典
            - 'final_loss': 最終訓練損失
            - 'training_time': 訓練總時間（秒）
            - 'epochs_completed': 完成的 epoch 數
            - 'best_epoch': 最佳模型的 epoch（若啟用早停）
            - 'best_metric': 最佳指標值（若啟用早停）
            - 'history': 訓練歷史（每 epoch 的損失）
        """
        logging.info("=" * 80)
        logging.info("🚀 開始訓練")
        logging.info(f"   模型: {self.model.__class__.__name__}")
        logging.info(f"   優化器: {self.optimizer.__class__.__name__}")
        logging.info(f"   最大 Epochs: {self.train_cfg.get('epochs', 'N/A')}")
        logging.info(f"   早停: {'啟用' if self.early_stopping_enabled else '禁用'}")
        logging.info("=" * 80)
        
        # 訓練配置
        max_epochs = self.train_cfg.get('epochs', self.train_cfg.get('max_epochs', 1000))
        log_freq = self.train_cfg.get('log_interval', self.log_cfg.get('log_freq', 50))
        checkpoint_freq = self.train_cfg.get('checkpoint_freq', 500)
        validation_freq = self.train_cfg.get('validation_freq', self.train_cfg.get('checkpoint_interval', 100))
        
        # 訓練歷史記錄
        history = {
            'total_loss': [],
            'val_loss': [],
            'epoch': []
        }
        
        # 時間記錄
        start_time = time.time()
        last_val_metrics: Optional[Dict[str, float]] = None
        
        # 初始化損失字典（防止 epoch=0 時未定義）
        loss_dict = {'total_loss': 0.0, 'residual_loss': 0.0, 'bc_loss': 0.0, 'data_loss': 0.0}
        
        # 確定訓練起始 epoch（支援從 checkpoint 恢復）
        start_epoch = self.epoch  # 若從 checkpoint 恢復，self.epoch 會被 load_checkpoint() 設定
        if start_epoch > 0:
            logging.info(f"🔄 從 epoch {start_epoch} 恢復訓練")
        
        # 訓練循環
        for epoch in range(start_epoch, max_epochs):
            self.epoch = epoch
            
            # 🔧 自適應採樣（如果啟用）
            if self.loop_manager is not None:
                # 更新訓練批次
                self.training_data = self.loop_manager.update_training_batch(self.training_data, epoch)
                
                # 檢查是否需要重採樣（傳遞 loss_dict 而非殘差）
                if epoch > 0 and self.loop_manager.should_resample_collocation_points(
                    epoch, 
                    history['total_loss'][-1] if history['total_loss'] else float('inf'),
                    None  # residuals 參數設為 None
                ):
                    try:
                        # 提取域邊界
                        domain_bounds = {
                            'x': (self.config['domain']['x_min'], self.config['domain']['x_max']),
                            'y': (self.config['domain']['y_min'], self.config['domain']['y_max'])
                        }
                        if 'z_min' in self.config['domain']:
                            domain_bounds['z'] = (self.config['domain']['z_min'], self.config['domain']['z_max'])
                        if 't_min' in self.config['domain']:
                            domain_bounds['t'] = (self.config['domain']['t_min'], self.config['domain']['t_max'])
                        
                        new_points, metrics = self.loop_manager.resample_collocation_points(
                            self.model, self.physics, domain_bounds, epoch, str(self.device)
                        )
                        logging.info(f"🔄 重採樣 {len(new_points)} 個配點（epoch {epoch}）")
                        logging.debug(f"   指標: {metrics}")
                    except Exception as e:
                        logging.warning(f"⚠️ 重採樣失敗（epoch {epoch}）: {e}")
            
            # 🎯 Fourier 退火更新（如果啟用）
            if self.fourier_annealing is not None:
                try:
                    # 檢查模型是否有 Fourier features 模組
                    fourier_module = None
                    
                    # 嘗試從模型中找到 Fourier features
                    if hasattr(self.model, 'fourier_features'):
                        fourier_module = self.model.fourier_features
                    elif hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'fourier_features'):
                        fourier_module = self.model.encoder.fourier_features
                    
                    if fourier_module is not None:
                        # 獲取更新前的狀態（用於日誌）
                        old_info = self.fourier_annealing.get_info()
                        
                        # 執行更新
                        self.fourier_annealing.update_fourier_features(
                            fourier_module, 
                            current_epoch=epoch, 
                            total_epochs=max_epochs
                        )
                        
                        # 獲取更新後的狀態
                        new_info = self.fourier_annealing.get_info()
                        
                        # 檢查是否發生階段切換（比較 stage_index）
                        if old_info['stage_index'] != new_info['stage_index']:
                            logging.info(f"🎯 Fourier 退火階段切換：{new_info['stage_description']}")
                            logging.info(f"   當前頻率: {new_info['active_frequencies']}")
                            logging.info(f"   輸出維度: {fourier_module.out_dim}")
                    
                except AttributeError as e:
                    # 模型不支持 Fourier 退火，警告一次後禁用
                    if epoch == 0:
                        logging.warning(f"⚠️ 模型不支持 Fourier 退火：{e}，已自動禁用")
                    self.fourier_annealing = None
                except Exception as e:
                    logging.error(f"❌ Fourier 退火更新失敗（epoch {epoch}）: {e}")
            
            # ✅ 執行訓練步驟（傳遞 training_data 和 epoch）
            loss_dict = self.step(self.training_data, epoch)
            
            # ✅ 驗證指標計算
            if validation_freq > 0 and epoch % validation_freq == 0:
                val_metrics = self.validate()
                if val_metrics is not None:
                    last_val_metrics = val_metrics
                    loss_dict['val_loss'] = val_metrics['relative_l2']
                    loss_dict['val_mse'] = val_metrics['mse']
            
            # 記錄歷史
            history['total_loss'].append(loss_dict['total_loss'])
            history['epoch'].append(epoch)
            if 'val_loss' in loss_dict:
                history['val_loss'].append(loss_dict['val_loss'])
            
            # 📊 TensorBoard 記錄（每 log_freq 記錄一次）
            if self.writer is not None and epoch % log_freq == 0:
                # ====================================================================
                # 1. 總損失與主要分量
                # ====================================================================
                self.writer.add_scalar('Loss/total', loss_dict.get('total_loss', 0.0), epoch)
                self.writer.add_scalar('Loss/data', loss_dict.get('data_loss', 0.0), epoch)
                self.writer.add_scalar('Loss/pde', loss_dict.get('pde_loss', 0.0), epoch)
                self.writer.add_scalar('Loss/boundary', loss_dict.get('bc_loss', 0.0), epoch)
                
                # ====================================================================
                # 2. PDE 子項（動量方程與連續性方程）
                # ====================================================================
                if 'momentum_x_loss' in loss_dict:
                    self.writer.add_scalar('Loss/PDE/momentum_x', loss_dict['momentum_x_loss'], epoch)
                if 'momentum_y_loss' in loss_dict:
                    self.writer.add_scalar('Loss/PDE/momentum_y', loss_dict['momentum_y_loss'], epoch)
                if 'momentum_z_loss' in loss_dict:
                    self.writer.add_scalar('Loss/PDE/momentum_z', loss_dict['momentum_z_loss'], epoch)
                if 'continuity_loss' in loss_dict:
                    self.writer.add_scalar('Loss/PDE/continuity', loss_dict['continuity_loss'], epoch)
                if 'div_loss' in loss_dict:
                    self.writer.add_scalar('Loss/PDE/divergence', loss_dict['div_loss'], epoch)
                
                # ====================================================================
                # 3. 數據擬合損失（各變量）
                # ====================================================================
                if 'u_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Data/u', loss_dict['u_loss'], epoch)
                if 'v_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Data/v', loss_dict['v_loss'], epoch)
                if 'w_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Data/w', loss_dict['w_loss'], epoch)
                if 'pressure_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Data/pressure', loss_dict['pressure_loss'], epoch)
                
                # ====================================================================
                # 4. Weighted Loss（分析權重平衡）
                # ====================================================================
                if 'weighted_data_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Weighted/data', loss_dict['weighted_data_loss'], epoch)
                if 'weighted_pde_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Weighted/pde', loss_dict['weighted_pde_loss'], epoch)
                if 'weighted_div_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Weighted/continuity', loss_dict['weighted_div_loss'], epoch)
                if 'weighted_bc_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Weighted/boundary', loss_dict['weighted_bc_loss'], epoch)
                
                # ====================================================================
                # 5. RANS Prior Loss（低保真先驗）
                # ====================================================================
                if 'prior_consistency_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Prior/total', loss_dict['prior_consistency_loss'], epoch)
                if 'prior_loss_u' in loss_dict:
                    self.writer.add_scalar('Loss/Prior/u', loss_dict['prior_loss_u'], epoch)
                if 'prior_loss_v' in loss_dict:
                    self.writer.add_scalar('Loss/Prior/v', loss_dict['prior_loss_v'], epoch)
                if 'prior_loss_p' in loss_dict:
                    self.writer.add_scalar('Loss/Prior/p', loss_dict['prior_loss_p'], epoch)
                
                # ====================================================================
                # 6. 邊界條件損失
                # ====================================================================
                if 'periodic_x_loss' in loss_dict:
                    self.writer.add_scalar('Loss/BC/periodic_x', loss_dict['periodic_x_loss'], epoch)
                if 'periodic_y_loss' in loss_dict:
                    self.writer.add_scalar('Loss/BC/periodic_y', loss_dict['periodic_y_loss'], epoch)
                if 'inlet_loss' in loss_dict:
                    self.writer.add_scalar('Loss/BC/inlet', loss_dict['inlet_loss'], epoch)
                if 'outlet_loss' in loss_dict:
                    self.writer.add_scalar('Loss/BC/outlet', loss_dict['outlet_loss'], epoch)
                if 'wall_loss' in loss_dict:
                    self.writer.add_scalar('Loss/BC/wall', loss_dict['wall_loss'], epoch)
                
                # ====================================================================
                # 7. 正則化項
                # ====================================================================
                if 'regularization_loss' in loss_dict:
                    self.writer.add_scalar('Loss/Regularization/total', loss_dict['regularization_loss'], epoch)
                if 'l2_reg' in loss_dict:
                    self.writer.add_scalar('Loss/Regularization/l2', loss_dict['l2_reg'], epoch)
                if 'gradient_penalty' in loss_dict:
                    self.writer.add_scalar('Loss/Regularization/gradient', loss_dict['gradient_penalty'], epoch)
                
                # ====================================================================
                # 8. 驗證指標
                # ====================================================================
                if 'val_loss' in loss_dict:
                    self.writer.add_scalar('Validation/relative_l2', loss_dict['val_loss'], epoch)
                if 'val_mse' in loss_dict:
                    self.writer.add_scalar('Validation/mse', loss_dict['val_mse'], epoch)
                
                # ====================================================================
                # 9. 訓練超參數
                # ====================================================================
                current_lr = self.get_current_lr()
                self.writer.add_scalar('Training/learning_rate', current_lr, epoch)
                
                # ====================================================================
                # 10. 梯度與權重統計（每 log_freq*2 記錄一次，避免過多）
                # ====================================================================
                if epoch % (log_freq * 2) == 0:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            self.writer.add_scalar(f'Gradients/norm/{name}', grad_norm, epoch)
                            self.writer.add_histogram(f'Gradients/hist/{name}', param.grad, epoch)
                            self.writer.add_histogram(f'Weights/{name}', param, epoch)
            
            # 更新全局步數
            self.global_step += 1
            
            # 🚀 課程訓練：LR 控制與階段管理
            # Patch 1: 強制每個 epoch 應用課程 LR（如果定義），確保控制權
            if '_curriculum_lr' in loss_dict:
                target_lr = loss_dict['_curriculum_lr']
                # 強制寫入 optimizer
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = target_lr
                
                # 若有 scheduler，同步 base_lrs 防止漂移
                if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'base_lrs'):
                    self.lr_scheduler.base_lrs = [target_lr] * len(self.lr_scheduler.base_lrs)

            # 處理階段切換（僅日誌與檢查點）
            if '_curriculum_transition' in loss_dict and loss_dict['_curriculum_transition'] > 0.5:
                stage_name = loss_dict.get('_curriculum_stage', f'stage_{epoch}')
                logging.info(f"📉 課程階段切換: {stage_name}")
                if '_curriculum_lr' in loss_dict:
                     logging.info(f"   學習率強制設置為: {loss_dict['_curriculum_lr']:.2e}")
                
                # 保存階段檢查點（如果啟用）
                if self.log_cfg.get('save_stage_checkpoints', False):
                    self.save_checkpoint(epoch, loss_dict, is_best=False)
                    logging.info(f"💾 階段檢查點已保存: {stage_name}")
            
            # 📉 更新學習率調度器（課程訓練與全域 scheduler 並存）
            if self.lr_scheduler is not None:
                # 課程訓練時：只在非切換 epoch 更新 scheduler
                if hasattr(self, 'curriculum_weighter'):
                    is_transition = '_curriculum_transition' in loss_dict and loss_dict['_curriculum_transition'] > 0.5
                    if not is_transition:
                        self.lr_scheduler.step()
                else:
                    # 非課程訓練：正常更新
                    self.lr_scheduler.step()
            
            # 📊 日誌輸出
            if epoch % log_freq == 0:
                self.log_epoch(epoch, loss_dict)
            
            # 💾 檢查點保存
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0 and epoch > 0:
                self.save_checkpoint(epoch, loss_dict)
                logging.info(f"💾 檢查點已保存（epoch {epoch}）")
            
            # 🛑 早停檢查
            if self.early_stopping_enabled:
                # 選擇監控指標
                metric_name = self.early_stopping_cfg.get('monitor', 'total_loss')
                if metric_name == 'val_loss' and 'val_loss' in loss_dict:
                    current_metric = loss_dict['val_loss']
                elif metric_name in loss_dict:
                    current_metric = loss_dict[metric_name]
                else:
                    current_metric = loss_dict['total_loss']
                
                # 檢查是否應該停止
                if self.check_early_stopping(current_metric):
                    logging.info(f"🛑 早停觸發於 epoch {epoch}")
                    logging.info(f"   最佳指標: {self.best_val_loss:.6f}（epoch {self.best_epoch}）")
                    
                    # 恢復最佳模型（如果啟用）
                    if self.early_stopping_cfg.get('restore_best_weights', True) and self.best_model_state is not None:
                        self.model.load_state_dict(self.best_model_state)
                        logging.info(f"✅ 已恢復最佳模型（epoch {self.best_epoch}）")
                    
                    break
            
            # 快速收斂檢查（可配置）
            if self.convergence_threshold is not None and loss_dict['total_loss'] < self.convergence_threshold:
                logging.info(f"✅ 快速收斂於 epoch {epoch}（loss < {self.convergence_threshold:.2e}）")
                break
        
        # 訓練結束（處理 epoch 變數作用域）
        final_epoch = epoch if 'epoch' in locals() else max_epochs - 1
        final_loss = loss_dict['total_loss']
        
        total_time = time.time() - start_time
        logging.info("=" * 80)
        logging.info(f"✅ 訓練完成")
        logging.info(f"   總時間: {total_time:.1f}s")
        logging.info(f"   完成 Epochs: {final_epoch + 1}")
        logging.info(f"   最終損失: {final_loss:.6f}")
        if self.early_stopping_enabled and self.best_epoch >= 0:
            logging.info(f"   最佳 Epoch: {self.best_epoch}")
            logging.info(f"   最佳指標: {self.best_val_loss:.6f}")
        logging.info("=" * 80)
        
        # 保存最終檢查點
        final_checkpoint = self.save_checkpoint(final_epoch + 1, loss_dict, is_best=False)
        logging.info(f"💾 最終模型已保存")
        
        # 關閉 TensorBoard writer
        if self.writer is not None:
            # 記錄最終超參數與指標
            hparams = {
                'lr': self.train_cfg.get('lr', 1e-3),
                'optimizer': self.optimizer.__class__.__name__,
                'model_width': self.config.get('model', {}).get('width', 256),
                'model_depth': self.config.get('model', {}).get('depth', 8),
                'activation': self.config.get('model', {}).get('activation', 'sine'),
                'K_sensors': self.config.get('sensors', {}).get('K', 0),
            }
            metrics = {
                'hparam/final_loss': final_loss,
                'hparam/best_loss': self.best_val_loss if self.early_stopping_enabled else final_loss,
                'hparam/epochs': final_epoch + 1,
            }
            self.writer.add_hparams(hparams, metrics)
            self.writer.flush()
            self.writer.close()
            logging.info("✅ TensorBoard 日誌已保存並關閉")
        
        # 返回訓練結果
        return {
            'final_loss': final_loss,
            'training_time': total_time,
            'epochs_completed': final_epoch + 1,
            'best_epoch': self.best_epoch if self.early_stopping_enabled else final_epoch,
            'best_metric': self.best_val_loss if self.early_stopping_enabled else final_loss,
            'history': history,
            'checkpoint_path': final_checkpoint
        }
    
    def save_checkpoint(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ):
        """
        保存檢查點

        Args:
            epoch: 當前 epoch
            metrics: 評估指標（可選）
            is_best: 是否為最佳模型
        """
        # ✅ 物理驗證：在保存前檢查模型物理一致性
        from pinnx.train.checkpointing import validate_physics_before_save

        # 生成驗證點座標（從域邊界均勻採樣）
        validation_n_points = 1000  # 驗證點數量
        
        # 嘗試從配置中獲取 domain 資訊（支援多種配置格式）
        domain = None
        
        # 優先順序 1: physics.domain
        physics_config = self.config.get('physics', {})
        if 'domain' in physics_config:
            domain_data = physics_config['domain']
            if 'x_range' in domain_data:
                # 格式: x_range: [min, max]
                domain = {
                    'x_min': domain_data['x_range'][0], 'x_max': domain_data['x_range'][1],
                    'y_min': domain_data['y_range'][0], 'y_max': domain_data['y_range'][1],
                    'z_min': domain_data.get('z_range', [0, 1])[0],
                    'z_max': domain_data.get('z_range', [0, 1])[1],
                }
        
        # 優先順序 2: data.jhtdb_config.domain
        if domain is None:
            data_config = self.config.get('data', {})
            jhtdb_config = data_config.get('jhtdb_config', {})
            if 'domain' in jhtdb_config:
                domain_data = jhtdb_config['domain']
                # 格式: x: [min, max]
                domain = {
                    'x_min': domain_data.get('x', [0, 1])[0],
                    'x_max': domain_data.get('x', [0, 1])[1],
                    'y_min': domain_data.get('y', [-1, 1])[0],
                    'y_max': domain_data.get('y', [-1, 1])[1],
                    'z_min': domain_data.get('z', [0, 1])[0] if 'z' in domain_data else 0.0,
                    'z_max': domain_data.get('z', [0, 1])[1] if 'z' in domain_data else 1.0,
                }
        
        # 優先順序 3: 頂層 domain
        if domain is None:
            domain_data = self.config.get('domain', None)
            if domain_data is not None:
                if 'x_range' in domain_data:
                    domain = {
                        'x_min': domain_data['x_range'][0], 'x_max': domain_data['x_range'][1],
                        'y_min': domain_data['y_range'][0], 'y_max': domain_data['y_range'][1],
                        'z_min': domain_data.get('z_range', [0, 1])[0],
                        'z_max': domain_data.get('z_range', [0, 1])[1],
                    }
                elif 'x_min' in domain_data:
                    domain = domain_data
        
        # 預設值（通道流標準域）
        if domain is None:
            logging.warning("配置中未找到 domain 資訊，使用預設值（通道流 Re_tau=1000）")
            domain = {
                'x_min': 0.0, 'x_max': 25.13,
                'y_min': -1.0, 'y_max': 1.0,
                'z_min': 0.0, 'z_max': 9.42
            }

        if self.model_input_dim == 2:
            x = torch.linspace(domain['x_min'], domain['x_max'], 32, device=self.device)
            y = torch.linspace(domain['y_min'], domain['y_max'], 32, device=self.device)
            X, Y = torch.meshgrid(x, y, indexing='ij')
            validation_coords = torch.stack([X.flatten(), Y.flatten()], dim=1)
        elif self.model_input_dim == 3:
            x = torch.linspace(domain['x_min'], domain['x_max'], 10, device=self.device)
            y = torch.linspace(domain['y_min'], domain['y_max'], 10, device=self.device)
            z = torch.linspace(domain['z_min'], domain['z_max'], 10, device=self.device)
            X, Y, Z = torch.meshgrid(x, y, z, indexing='ij')
            validation_coords = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=1)
        else:
            logging.warning(f"未知的模型輸入維度: {self.model_input_dim}，跳過物理驗證")
            validation_coords = None

        # 執行物理驗證
        physics_metrics = {}
        if validation_coords is not None:
            validation_passed, physics_metrics = validate_physics_before_save(
                self.model,
                validation_coords,
                self.config,
                self.device
            )

            # 物理診斷完成（記錄但不拒絕保存）
            # 注意：validate_physics_before_save 已修改為診斷模式
            # 僅在 strict_mode=True 且檢測到 trivial solution 時才返回 False
            if not validation_passed:
                # 檢查是否是因為 trivial solution 被拒絕（strict mode）
                if physics_metrics.get('trivial_solution', {}).get('is_trivial', False):
                    strict_mode = self.config.get('physics_validation', {}).get('strict_mode', False)
                    if strict_mode:
                        logging.error("❌ Strict Mode: 檢測到 Trivial Solution，拒絕保存")
                        return  # 僅在此情況下拒絕

                # 其他情況：物理約束未滿足（訓練初期正常）
                logging.info("ℹ️  物理診斷完成，指標已記錄至檢查點元數據")
                # 繼續保存，讓使用者根據診斷資訊判斷

        checkpoint_path = self.checkpoint_dir / f"epoch_{epoch}.pth"

        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': self.config,
        }

        # 🆕 保存 physics 的 state_dict（VS-PINN 縮放參數等）
        if self.physics is not None and hasattr(self.physics, 'state_dict'):
            checkpoint_data['physics_state_dict'] = self.physics.state_dict()
            logging.debug(f"💾 Physics state saved: {list(self.physics.state_dict().keys())}")

        # ✅ TASK-008: 保存標準化 metadata
        checkpoint_data['normalization'] = self.data_normalizer.get_metadata()
        logging.debug(f"💾 Normalization metadata saved: type={self.data_normalizer.norm_type}")

        # ⭐ P0.2: 保存 GradScaler 狀態（AMP）
        if self.use_amp and hasattr(self, 'scaler'):
            checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()
            logging.debug(f"💾 GradScaler state saved: scale={self.scaler.get_scale():.0f}")

        # ✅ 保存物理驗證指標
        if physics_metrics:
            checkpoint_data['physics_metrics'] = physics_metrics
            logging.debug(f"💾 Physics metrics saved: validation_passed={physics_metrics.get('validation_passed', False)}")

        if metrics:
            checkpoint_data['metrics'] = metrics

        if self.lr_scheduler:
            checkpoint_data['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint_data, checkpoint_path)
        logging.info(f"💾 檢查點已保存: {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint_data, best_path)
            logging.info(f"⭐ 最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        載入檢查點
        
        Args:
            checkpoint_path: 檢查點路徑
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.history = checkpoint.get('history', self.history)
        
        # 恢復 physics 的 state_dict（VS-PINN 縮放參數等）
        if self.physics is not None:
            if 'physics_state_dict' not in checkpoint:
                raise KeyError("checkpoint is missing required 'physics_state_dict'")
            if not hasattr(self.physics, 'load_state_dict'):
                raise TypeError("physics module does not support load_state_dict()")
            self.physics.load_state_dict(checkpoint['physics_state_dict'])
            logging.info(f"✅ Physics state restored: {list(checkpoint['physics_state_dict'].keys())}")
        
        # 恢復標準化器
        if 'normalization' not in checkpoint:
            raise KeyError("checkpoint is missing required 'normalization' metadata")
        self.data_normalizer = DataNormalizer.from_metadata(checkpoint['normalization'])
        logging.info(f"✅ DataNormalizer restored: {self.data_normalizer}")
        
        # 恢復 GradScaler 狀態（AMP）
        if self.use_amp:
            if not hasattr(self, 'scaler'):
                raise AttributeError("AMP enabled but trainer lacks GradScaler instance")
            if 'scaler_state_dict' not in checkpoint:
                raise KeyError("checkpoint is missing required 'scaler_state_dict' for AMP recovery")
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            logging.info(f"✅ GradScaler state restored: scale={self.scaler.get_scale():.0f}")
        
        if self.lr_scheduler and 'lr_scheduler_state_dict' in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        
        logging.info(f"✅ 檢查點已載入: {checkpoint_path}（epoch={self.epoch}）")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        檢查是否應該早停
        
        Args:
            val_loss: 驗證損失
        
        Returns:
            是否應該停止訓練
        """
        if not self.early_stopping_enabled:
            return False
        
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.best_epoch = self.epoch
            self.patience_counter = 0
            
            # 保存最佳模型狀態（如果配置啟用）
            if self.early_stopping_cfg.get('restore_best_weights', True):
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # 🆕 立即保存最佳模型到磁碟（防止訓練中斷導致遺失）
            metrics = {'val_loss': val_loss, 'best_epoch': self.best_epoch}
            self.save_checkpoint(self.epoch, metrics, is_best=True)
            
            logging.info(f"🎯 新最佳指標: {self.best_val_loss:.6f}（epoch {self.best_epoch}）")
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                logging.info(f"🛑 早停觸發（patience={self.patience}）")
                return True
            return False
    
    def get_current_lr(self) -> float:
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float]):
        """
        記錄 epoch 訓練資訊
        
        Args:
            epoch: 當前 epoch
            metrics: 訓練指標
        """
        current_lr = self.get_current_lr()
        log_str = f"Epoch {epoch}/{self.train_cfg.get('epochs', '?')}"
        
        # Patch 2: Explicitly log effective LR first
        log_str += f" | LR: {current_lr:.2e}"

        if '_curriculum_stage' in metrics:
             log_str += f" | Stage: {metrics['_curriculum_stage']}"
        
        for key, value in metrics.items():
            # 跳過字典類型的值（如 gradnorm_weights, applied_weights）
            if isinstance(value, dict):
                continue
            # 跳過非數值類型（如字串、列表等）
            if not isinstance(value, (int, float)):
                continue
            # Skip internal keys starting with _
            if key.startswith('_'):
                continue
            
            log_str += f" | {key}: {value:.6f}"
        
        logging.info(log_str)
        
        # Patch 2: Log effective weights if available (Separate line for clarity)
        if 'applied_weights' in metrics:
             # Format dictionary for nicer logging
             weights_str = ", ".join([f"{k}={v:.2e}" for k,v in metrics['applied_weights'].items()])
             logging.info(f"   Effective Weights: {{{weights_str}}}")
        
        # 記錄到歷史
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        self.history['lr'].append(self.get_current_lr())