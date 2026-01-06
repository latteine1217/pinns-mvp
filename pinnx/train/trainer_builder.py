"""
Trainer Builder Pattern

目的：簡化 Trainer 初始化，降低參數複雜度
哲學：Good Taste - 複雜性應該被封裝，而非暴露

使用方式：
    # 舊方式（9 個參數，容易出錯）
    trainer = Trainer(model, physics, losses, config, device,
                      weighters=weighters,
                      input_normalizer=input_normalizer,
                      channel_data_cache=cache,
                      training_data=data)
    
    # 新方式 1：使用 Builder（鏈式呼叫）
    trainer = (TrainerBuilder(config, device)
               .with_model(model)
               .with_physics(physics)
               .with_losses(losses)
               .with_weighters(weighters)
               .with_normalizer(input_normalizer)
               .with_training_data(data)
               .build())
    
    # 新方式 2：使用便捷函數（推薦）
    trainer = create_trainer_with_builder(
        model, physics, losses, config, device,
        weighters=weighters,
        input_normalizer=input_normalizer,
        training_data=data
    )
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import logging
from typing import Dict, Any, Optional, cast
from pathlib import Path
from torch.amp.grad_scaler import GradScaler
import wandb

import pinnx  # 需要存取 pinnx.Config
from pinnx.train.trainer import Trainer
from pinnx.train.trainer_components import TrainerComponents
from pinnx.utils.normalization.input_transform import InputTransform
from pinnx.utils.normalization.output_transform import OutputTransform


class TrainerBuilder:
    """
    Trainer 構建器（Builder Pattern）
    
    優點：
    1. 降低初始化複雜度（不需要記住 9 個參數順序）
    2. 可選參數更清晰（用 .with_xxx() 鏈式呼叫）
    3. 支援自動依賴注入（from_config 工廠方法）
    4. 易於擴展（新增功能不破壞現有代碼）
    
    哲學對齊：
    - Good Taste: 隱藏複雜性，暴露簡潔接口
    - Simplicity: 減少呼叫方的心智負擔
    - Never Break Userspace: 保留舊接口（Trainer.__init__ 仍可用）
    """
    
    def __init__(self, config: Dict[str, Any], device: torch.device, skip_wandb_init: bool = False):
        """
        初始化 Builder（只需要必需參數）
        
        Args:
            config: 完整訓練配置
            device: 計算設備
            skip_wandb_init: 是否跳過 WandB 初始化（用於 time window 訓練）
        """
        self.config = config
        self.device = device
        self.skip_wandb_init = skip_wandb_init
        
        # 必需組件（必須顯式設定）
        self._model: Optional[nn.Module] = None
        self._physics: Optional[Any] = None
        self._losses: Optional[Dict[str, nn.Module]] = None
        
        # 可選組件（有預設值）
        self._weighters: Optional[Dict[str, Any]] = None
        self._input_normalizer: Optional[InputTransform] = None
        self._channel_data_cache: Optional[Dict[str, Any]] = None
        self._training_data: Optional[Dict[str, torch.Tensor]] = None
        
        # Time window 模式專用（由外層注入）
        self._wandb_run: Optional[Any] = None
        self._step_offset: int = 0
    
    def with_model(self, model: nn.Module) -> 'TrainerBuilder':
        """設定模型"""
        self._model = model
        return self
    
    def with_physics(self, physics: Any) -> 'TrainerBuilder':
        """設定物理模組"""
        self._physics = physics
        return self
    
    def with_losses(self, losses: Dict[str, nn.Module]) -> 'TrainerBuilder':
        """設定損失函數字典"""
        self._losses = losses
        return self
    
    def with_weighters(self, weighters: Dict[str, Any]) -> 'TrainerBuilder':
        """設定權重調度器"""
        self._weighters = weighters
        return self
    
    def with_normalizer(self, normalizer: InputTransform) -> 'TrainerBuilder':
        """設定輸入標準化器"""
        self._input_normalizer = normalizer
        return self
    
    def with_channel_cache(self, cache: Dict[str, Any]) -> 'TrainerBuilder':
        """設定通道流資料快取"""
        self._channel_data_cache = cache
        return self
    
    def with_training_data(self, data: Dict[str, torch.Tensor]) -> 'TrainerBuilder':
        """設定訓練資料"""
        self._training_data = data
        return self
    
    def with_wandb_run(self, wandb_run: Any) -> 'TrainerBuilder':
        """設定現有的 WandB run（用於 time window 訓練）"""
        self._wandb_run = wandb_run
        return self
    
    def with_step_offset(self, step_offset: int) -> 'TrainerBuilder':
        """設定訓練步數偏移（用於 time window 訓練）"""
        self._step_offset = step_offset
        return self

    # ========================================================================
    # 組件創建方法（P2-3：從 Trainer._setup_* 遷移）
    # ========================================================================

    def _create_training_components(self) -> Dict[str, Any]:
        """
        創建訓練組件：optimizer, lr_scheduler, amp_scaler, weight_scheduler

        從 Trainer._setup_optimizer, _setup_amp, _setup_schedulers 遷移

        Returns:
            Dict 包含: optimizer, lr_scheduler, amp_scaler, weight_scheduler
        """
        from pinnx.train.factories import create_optimizer, create_scheduler

        # 類型檢查（build() 已驗證，此處為靜態類型檢查器）
        assert self._model is not None, "Model must be set before creating training components"
        model = cast(nn.Module, self._model)  # 類型斷言

        train_cfg = self.config['training']

        # 1. 創建優化器
        optimizer_config = train_cfg.get('optimizer', 'adam')

        if isinstance(optimizer_config, dict):
            optimizer_config.setdefault('lr', train_cfg.get('lr', 1e-3))
            optimizer_config.setdefault('weight_decay', train_cfg.get('weight_decay', 0.0))
        else:
            optimizer_config = {
                'type': optimizer_config,
                'lr': train_cfg.get('lr', 1e-3),
                'weight_decay': train_cfg.get('weight_decay', 0.0)
            }

        optimizer = create_optimizer(model, optimizer_config)
        logging.info(f"✅ Optimizer 創建: {type(optimizer).__name__}")

        # 2. 創建學習率調度器
        scheduler_cfg = train_cfg.get('lr_scheduler', {})
        if isinstance(scheduler_cfg, dict):
            scheduler_cfg.setdefault('max_epochs', train_cfg.get('epochs', 1000))

        lr_scheduler = create_scheduler(optimizer, scheduler_cfg)

        # 3. 創建 AMP Scaler
        amp_scaler, use_amp = self._create_amp_scaler(optimizer, train_cfg)

        # 4. 權重調度器（暫未實現）
        weight_scheduler = None

        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler,
            'amp_scaler': amp_scaler,
            'weight_scheduler': weight_scheduler,
            'use_amp': use_amp
        }

    def _create_amp_scaler(self, optimizer: torch.optim.Optimizer, train_cfg: Dict[str, Any]):
        """
        創建 AMP Scaler（從 Trainer._setup_amp 遷移）

        Returns:
            (scaler, use_amp): GradScaler 和是否啟用 AMP
        """
        amp_cfg = train_cfg.get('amp', {})
        use_amp = amp_cfg.get('enabled', False)

        # AMP 支援檢查
        is_adam = isinstance(optimizer, torch.optim.Adam)
        is_cuda = self.device.type == 'cuda'
        is_mps = self.device.type == 'mps'

        if use_amp and not is_adam:
            logging.warning(
                f"⚠️ AMP 僅支援 Adam 優化器，當前使用 {type(optimizer).__name__}，已禁用 AMP"
            )
            use_amp = False

        if use_amp and is_mps:
            logging.warning(
                "⚠️ MPS 後端的 GradScaler 存在已知問題（不支援 float64）\n"
                "   建議：(1) 使用 CUDA 設備，或 (2) 關閉 AMP\n"
                "   已自動禁用 AMP"
            )
            use_amp = False

        if use_amp and not is_cuda:
            logging.warning(
                f"⚠️ AMP 僅在 CUDA 環境完全支援，當前設備為 {self.device}，已禁用 AMP"
            )
            use_amp = False

        # 創建 GradScaler
        if use_amp:
            scaler = GradScaler(
                'cuda',
                init_scale=2.0**16,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True
            )
            logging.info(
                f"✅ AMP 已啟用（Forward: FP32, Backward: FP16）\n"
                f"   - 優化器: {type(optimizer).__name__}\n"
                f"   - 設備: {self.device} (CUDA)\n"
                f"   - GradScaler 初始 scale: {scaler.get_scale():.0f}"
            )
        else:
            device_type = 'cuda' if is_cuda else 'cpu'
            scaler = GradScaler(device_type, enabled=False)
            if amp_cfg.get('enabled', False):
                logging.info("ℹ️ AMP 配置已禁用（不符合啟用條件）")

        return scaler, use_amp

    def _create_strategy_components(self) -> Dict[str, Any]:
        """
        創建策略組件：checkpoint_manager, checkpoint_strategy, validation_manager

        Returns:
            Dict 包含: checkpoint_manager, checkpoint_strategy, validation_manager
        """
        from pinnx.train.checkpoint_manager import (
            StandardCheckpointManager,
            PeriodicCheckpointStrategy
        )
        from pinnx.train.validation_manager import ValidationManager

        train_cfg = self.config['training']

        # 1. CheckpointManager
        checkpoint_dir = Path(self.config.get('output', {}).get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_manager = StandardCheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            save_optimizer=True,
            save_physics_validation=True,
            compress=False
        )

        # 2. CheckpointStrategy
        checkpoint_strategy = PeriodicCheckpointStrategy(
            save_interval=train_cfg.get('checkpoint_interval', 100),
            keep_best_k=3,
            metric_name='total_loss',
            mode='min'
        )

        # 3. ValidationManager（在 Trainer._init_validation_components 中配置策略）
        validation_manager = ValidationManager()

        logging.info(f"✅ 策略組件創建完成（CheckpointManager, ValidationManager）")

        return {
            'checkpoint_manager': checkpoint_manager,
            'checkpoint_strategy': checkpoint_strategy,
            'validation_manager': validation_manager
        }

    def _create_feature_components(self) -> Dict[str, Any]:
        """
        創建功能組件：prior_loss_manager, fourier_annealing, adaptive_sampler, early_stopping

        從 Trainer._setup_* 遷移

        Returns:
            Dict 包含: prior_loss_manager, fourier_annealing, adaptive_sampler, early_stopping_config
        """
        # 1. Prior Loss Manager
        prior_loss_manager = self._create_prior_loss_manager()

        # 2. Fourier Annealing
        fourier_annealing = self._create_fourier_annealing()

        # 3. Adaptive Sampling
        adaptive_sampler = self._create_adaptive_sampler()

        # 4. Early Stopping Config
        early_stopping_config = self._create_early_stopping_config()

        return {
            'prior_loss_manager': prior_loss_manager,
            'fourier_annealing': fourier_annealing,
            'adaptive_sampler': adaptive_sampler,
            'early_stopping_config': early_stopping_config
        }

    def _create_prior_loss_manager(self):
        """創建 Prior Loss Manager（從 Trainer._setup_prior_loss_manager 遷移）"""
        from pinnx.losses.priors import PriorLossManager

        lowfi_cfg = self.config.get('lowfi_prior', {})

        if not lowfi_cfg.get('enabled', False):
            logging.info("ℹ️  低保真先驗未啟用")
            return None

        consistency_weight = lowfi_cfg.get('consistency_weight', 0.3)
        variable_weights = lowfi_cfg.get('variable_weights', {'u': 1.0, 'v': 1.0, 'p': 0.5})
        distance_metric = lowfi_cfg.get('distance_metric', 'mse')

        loss_config = {
            'low_fidelity': {
                'consistency_weight': consistency_weight,
                'variable_weights': variable_weights,
                'distance_metric': distance_metric
            }
        }

        prior_loss_manager = PriorLossManager(
            consistency_weight=consistency_weight,
            loss_config=loss_config
        )
        prior_loss_manager.low_fidelity_loss.distance_metric = distance_metric

        logging.info(f"✅ Prior Loss Manager 初始化完成")
        logging.info(f"   - Consistency Weight: {consistency_weight}")
        logging.info(f"   - Variable Weights: {variable_weights}")
        logging.info(f"   - Distance Metric: {distance_metric}")

        return prior_loss_manager

    def _create_fourier_annealing(self):
        """創建 Fourier Annealing（從 Trainer._setup_fourier_annealing 遷移）"""
        from pinnx.train.fourier_annealing import create_fourier_annealing

        annealing_cfg = self.config.get('fourier_annealing', {})
        return create_fourier_annealing(annealing_cfg)

    def _create_adaptive_sampler(self):
        """創建 Adaptive Sampler（從 Trainer._setup_adaptive_sampling 遷移）"""
        train_cfg = self.config['training']

        sampling_cfg = train_cfg.get('sampling', {})
        simple_flag = sampling_cfg.get('adaptive_sampling', False)

        adaptive_cfg = self.config.get('adaptive_collocation', {})
        detailed_flag = adaptive_cfg.get('enabled', False)

        adaptive_sampling_enabled = simple_flag or detailed_flag

        if adaptive_sampling_enabled:
            from pinnx.train.adaptive_collocation import AdaptiveCollocationSampler
            adaptive_sampler = AdaptiveCollocationSampler(adaptive_cfg)
            trigger_method = adaptive_cfg.get('trigger', {}).get('method', 'epoch_interval')
            epoch_interval = adaptive_cfg.get('trigger', {}).get('epoch_interval', 1000)
            logging.info("✅ 自適應採樣啟用")
            logging.info(f"   觸發方法: {trigger_method}")
            if trigger_method in ['epoch_interval', 'hybrid']:
                logging.info(f"   觸發間隔: {epoch_interval} epochs")
            return adaptive_sampler
        else:
            return None

    def _create_early_stopping_config(self) -> Dict[str, Any]:
        """創建 Early Stopping Config（從 Trainer._setup_early_stopping 遷移）"""
        train_cfg = self.config['training']
        early_stopping_cfg = train_cfg.get('early_stopping', {})

        config = {
            'enabled': early_stopping_cfg.get('enabled', False),
            'patience': early_stopping_cfg.get('patience', 50),
            'min_delta': early_stopping_cfg.get('min_delta', 1e-6),
            'convergence_threshold': early_stopping_cfg.get('convergence_threshold', None),
            'monitor': early_stopping_cfg.get('monitor', 'total_loss'),  # 監控指標
            'restore_best_weights': early_stopping_cfg.get('restore_best_weights', True)  # 是否恢復最佳模型
        }

        if config['enabled']:
            logging.info(f"✅ 早停機制啟用（patience={config['patience']}, min_delta={config['min_delta']}, monitor={config['monitor']}）")
        if config['convergence_threshold'] is not None:
            logging.info(f"✅ 快速收斂檢查啟用（threshold={config['convergence_threshold']:.2e}）")

        return config

    def _create_monitoring_components(self) -> Dict[str, Any]:
        """
        創建監控組件：timer, memory_tracker, physics_validator, wandb_run

        從 Trainer._init_metrics_tracking, _init_wandb 遷移

        Returns:
            Dict 包含: timer, memory_tracker, physics_validator, wandb_run
        """
        from pinnx.utils.timer import Timer
        from pinnx.utils.memory_tracker import MemoryTracker
        from pinnx.utils.physics_validator import PhysicsValidator

        # 檢查是否啟用 WandB
        use_wandb = self.config.get('use_wandb', True)

        # 1. Timer（訓練計時）
        timer = Timer()

        # 2. MemoryTracker（內存監控）
        memory_tracker = MemoryTracker(model=self._model, use_wandb=use_wandb)

        # 3. PhysicsValidator（物理一致性檢查）
        physics_config = self.config.get('physics', {})
        physics_validator = PhysicsValidator(
            config=self.config,
            use_wandb=use_wandb,
            nu=physics_config.get('nu', 0.01),
            rho=physics_config.get('rho', 1.0),
            dimension=physics_config.get('dimension', 2)
        )

        # 4. WandB（支援跳過初始化，用於 time window 訓練）
        wandb_run = None
        if self.skip_wandb_init:
            # Time window 模式：使用外層傳入的 wandb_run
            wandb_run = getattr(self, '_wandb_run', None)
            if wandb_run:
                logging.info("⏭️  使用外層 WandB run（time window 模式）")
            else:
                logging.info("⏭️  跳過 WandB 初始化（time window 模式，將由外層初始化）")
        elif use_wandb:
            wandb_run = self._init_wandb()
            logging.info("✅ 指標追蹤工具初始化完成（Timer, MemoryTracker, PhysicsValidator, WandB）")
        else:
            logging.info("✅ 指標追蹤工具初始化完成（Timer, MemoryTracker, PhysicsValidator）")

        return {
            'timer': timer,
            'memory_tracker': memory_tracker,
            'physics_validator': physics_validator,
            'wandb_run': wandb_run
        }

        logging.info("✅ 指標追蹤工具初始化完成（Timer, MemoryTracker, PhysicsValidator）")

        return {
            'timer': timer,
            'memory_tracker': memory_tracker,
            'physics_validator': physics_validator,
            'wandb_run': wandb_run
        }

    def _init_wandb(self):
        """初始化 WandB（從 Trainer._init_wandb 遷移）"""
        exp_config = self.config.get('experiment', {})
        wandb_config = self._load_wandb_config()

        wandb_run = wandb.init(
            project=wandb_config['project'],
            name=exp_config.get('name', 'default_experiment'),
            group=exp_config.get('group', None),
            tags=exp_config.get('tags', []),
            config=self.config,
            reinit=True
        )

        # 上傳 config metadata
        if 'experiment' in self.config and '_config_metadata' in self.config['experiment']:
            metadata = self.config['experiment']['_config_metadata']
            for key, value in metadata.items():
                wandb.config.update({f'_meta_{key}': value}, allow_val_change=True)

        group = exp_config.get('group')
        tags = exp_config.get('tags', [])
        group_info = f", 分組: {group}" if group else ""
        tags_info = f", 標籤: {tags}" if tags else ""
        logging.info(
            f"✅ WandB 已啟用，專案: {wandb_config['project']}, "
            f"運行: {exp_config.get('name')}{group_info}{tags_info}"
        )

        return wandb_run

    def _load_wandb_config(self) -> Dict[str, str]:
        """載入 WandB 配置（從 Trainer._load_wandb_config 遷移）"""
        wandb_config_path = Path('.wandb_config')
        wandb_api_key = None
        wandb_project = 'pinns-sparse-turbulence'

        if wandb_config_path.exists():
            with open(wandb_config_path, 'r') as f:
                for line in f:
                    if line.startswith('WANDB_API_KEY='):
                        wandb_api_key = line.split('=')[1].strip()
                    elif line.startswith('WANDB_PROJECT='):
                        project_value = line.split('=')[1].strip()
                        if project_value:
                            wandb_project = project_value

        if wandb_api_key:
            os.environ['WANDB_API_KEY'] = wandb_api_key

        return {'project': wandb_project}

    def _create_data_components(self) -> Dict[str, Any]:
        """
        創建數據組件：data_normalizer, hard_constraint_applicator

        input_normalizer, training_data, validation_data, channel_data_cache, weighters
        已由用戶通過 .with_*() 方法提供

        Returns:
            Dict 包含: data_normalizer, hard_constraint_applicator
        """
        # 1. data_normalizer 需要在 Trainer 中創建（依賴 training_data）
        data_normalizer = None
        
        # 2. hard_constraint_applicator（壁面邊界條件）
        hard_constraint_applicator = self._create_hard_constraint_applicator()
        
        return {
            'data_normalizer': data_normalizer,
            'hard_constraint_applicator': hard_constraint_applicator,
        }
    
    def _create_hard_constraint_applicator(self) -> Optional[Any]:
        """
        創建 hard constraint applicator（壁面邊界條件）
        
        從配置讀取 physics.boundary_conditions.hard_constraint
        
        Returns:
            HardConstraintApplicator 或 None
        """
        physics_cfg = self.config.get('physics', {})
        bc_cfg = physics_cfg.get('boundary_conditions', {})
        hc_cfg = bc_cfg.get('hard_constraint', {})
        
        # 檢查是否啟用
        if not hc_cfg.get('enabled', False):
            return None
        
        from pinnx.utils.boundary_constraints import create_channel_flow_hard_constraint
        
        # 提取配置參數
        form = hc_cfg.get('form', 'quadratic')
        y_range_tuple = tuple(hc_cfg.get('y_range', [-1.0, 1.0]))
        alpha = hc_cfg.get('alpha', 10.0)
        constrained_vars = hc_cfg.get('constrained_vars', ['u', 'v', 'w'])
        y_axis_index = hc_cfg.get('y_axis_index', 2)
        
        # 從模型配置推斷 variable_order
        model_cfg = self.config.get('model', {})
        variable_order = model_cfg.get('output_variables', ['u', 'v', 'w', 'p'])
        
        # 創建 applicator
        applicator = create_channel_flow_hard_constraint(
            form=form,
            y_range=y_range_tuple,
            alpha=alpha,
            variable_order=variable_order,
            constrained_vars=constrained_vars,
            y_axis_index=y_axis_index,
            device=self.device,
            verify=True,
        )
        
        logging.info(
            f"✅ Hard Constraint Applicator 創建成功\n"
            f"   形式: {form}\n"
            f"   y 範圍: {y_range_tuple}\n"
            f"   約束變量: {constrained_vars}\n"
            f"   y 軸索引: {y_axis_index}"
        )
        
        return applicator
    
    # ========================================================================
    # DDP 輔助方法
    # ========================================================================
    
    def _should_use_ddp(self) -> bool:
        """
        判斷是否應該使用 DDP
        
        條件：
        1. pinnx.Config.use_ddp == True（自動偵測結果）
        2. torch.distributed 已初始化
        3. 當前在分散式環境中（有 RANK 環境變數）
        
        Returns:
            True 如果應該使用 DDP
        """
        # 檢查自動偵測結果
        if not pinnx.Config.use_ddp:
            return False
        
        # 檢查是否在分散式環境中
        if not dist.is_available():
            logging.warning("⚠️  torch.distributed 不可用，跳過 DDP")
            return False
        
        # 檢查是否已初始化
        if not dist.is_initialized():
            logging.warning("⚠️  torch.distributed 未初始化，跳過 DDP")
            return False
        
        return True
    
    def _get_local_rank(self) -> int:
        """
        獲取本地 GPU 排名
        
        Returns:
            本地 GPU rank（預設 0）
        """
        return int(os.environ.get('LOCAL_RANK', 0))
    
    def _wrap_model_ddp(self, model: nn.Module) -> nn.Module:
        """
        包裝模型為 DDP
        
        Args:
            model: 原始模型
        
        Returns:
            DDP 包裝後的模型
        """
        local_rank = self._get_local_rank()
        
        # 確保模型在正確的設備上
        if not next(model.parameters()).is_cuda:
            model = model.to(f'cuda:{local_rank}')
        
        # 包裝為 DDP
        # find_unused_parameters=False 可以提升效能（預設 False）
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False  # 如果確定所有參數都參與梯度計算
        )
        
        return ddp_model

    def build(self) -> Trainer:
        """
        構建 Trainer 實例（P2-3 重構：創建所有組件）

        流程：
        1. 驗證必需組件
        2. 創建訓練組件（optimizer, scheduler, amp, weight_scheduler）
        3. 創建策略組件（checkpoint, validation）
        4. 創建功能組件（prior_loss, fourier, adaptive, early_stopping）
        5. 創建監控組件（timer, memory, physics_validator, wandb）
        6. 封裝為 TrainerComponents
        7. 傳遞給 Trainer

        Returns:
            Trainer: 完整初始化的訓練器

        Raises:
            ValueError: 缺少必需組件
        """
        # Fail Fast: 驗證必需組件
        if self._model is None:
            raise ValueError("Model is required. Call .with_model(model) first.")
        if self._physics is None:
            raise ValueError("Physics module is required. Call .with_physics(physics) first.")
        if self._losses is None:
            raise ValueError("Losses are required. Call .with_losses(losses) first.")

        logging.info("🏗️  TrainerBuilder: 開始創建組件...")

        # 1. 創建訓練組件
        training = self._create_training_components()

        # 2. 創建策略組件
        strategy = self._create_strategy_components()

        # 3. 創建功能組件
        feature = self._create_feature_components()

        # 4. 創建監控組件
        monitoring = self._create_monitoring_components()

        # 5. 創建數據組件
        data = self._create_data_components()

        # ========== 🚀 NEW: DDP 模型包裝 ==========
        # 在創建 Trainer 之前包裝模型
        original_model = self._model
        
        if self._should_use_ddp():
            logging.info("🚀 包裝模型為 DistributedDataParallel...")
            self._model = self._wrap_model_ddp(original_model)
            logging.info(f"✅ DDP 模型包裝完成（device_ids=[{self._get_local_rank()}]）")
        
        # 6. 封裝為 TrainerComponents
        components = TrainerComponents(
            # 訓練組件
            optimizer=training['optimizer'],
            lr_scheduler=training['lr_scheduler'],
            amp_scaler=training['amp_scaler'],
            weight_scheduler=training['weight_scheduler'],

            # 策略組件
            checkpoint_manager=strategy['checkpoint_manager'],
            checkpoint_strategy=strategy['checkpoint_strategy'],
            validation_manager=strategy['validation_manager'],

            # 功能組件
            prior_loss_manager=feature['prior_loss_manager'],
            fourier_annealing=feature['fourier_annealing'],
            adaptive_sampler=feature['adaptive_sampler'],
            early_stopping_config=feature['early_stopping_config'],

            # 監控組件
            timer=monitoring['timer'],
            memory_tracker=monitoring['memory_tracker'],
            physics_validator=monitoring['physics_validator'],
            wandb_run=monitoring['wandb_run'],

            # 數據組件
            input_normalizer=self._input_normalizer,
            data_normalizer=data['data_normalizer'],
            training_data=self._training_data,
            validation_data=None,  # 將在 Trainer 中設置
            channel_data_cache=self._channel_data_cache,
            weighters=self._weighters,
            
            # 邊界條件組件
            hard_constraint_applicator=data['hard_constraint_applicator'],
            
            # Time Window 組件
            step_offset=self._step_offset,
        )

        # 驗證組件完整性
        components.validate_required_components()

        logging.info(f"✅ TrainerBuilder: 組件創建完成")
        logging.info(f"   組件摘要: {components}")

        # 7. 構建 Trainer（傳遞 components）
        trainer = Trainer(
            model=self._model,
            physics=self._physics,
            losses=self._losses,
            config=self.config,
            device=self.device,
            components=components,
        )

        logging.info("✅ TrainerBuilder: Trainer 構建完成")
        return trainer
    
    @classmethod
    def from_components(
        cls,
        model: nn.Module,
        physics: Any,
        losses: Dict[str, nn.Module],
        config: Dict[str, Any],
        device: torch.device,
        **optional_components
    ) -> Trainer:
        """
        工廠方法：從已創建的組件構建 Trainer（推薦方式）
        
        這個方法比直接呼叫 Trainer.__init__() 更清晰，同時保持靈活性。
        
        Args:
            model: 已創建的模型
            physics: 已創建的物理模組
            losses: 已創建的損失函數字典
            config: 完整訓練配置
            device: 計算設備
            **optional_components: 可選組件（weighters, input_normalizer, training_data, etc.）
        
        Returns:
            Trainer: 完整初始化的訓練器
        
        Example:
            >>> # 在 scripts/train/train.py 中
            >>> model = create_model(config, device)
            >>> physics = create_physics(config, device)
            >>> losses = create_loss_functions(config, device)
            >>> 
            >>> trainer = TrainerBuilder.from_components(
            ...     model, physics, losses, config, device,
            ...     weighters=weighters,
            ...     input_normalizer=normalizer,
            ...     training_data=training_data
            ... )
        """
        builder = cls(config, device)
        builder.with_model(model)
        builder.with_physics(physics)
        builder.with_losses(losses)
        
        # 處理可選組件
        if 'weighters' in optional_components:
            builder.with_weighters(optional_components['weighters'])
        if 'input_normalizer' in optional_components:
            builder.with_normalizer(optional_components['input_normalizer'])
        if 'channel_data_cache' in optional_components:
            builder.with_channel_cache(optional_components['channel_data_cache'])
        if 'training_data' in optional_components:
            builder.with_training_data(optional_components['training_data'])
        
        return builder.build()


# ============================================================================
# 便捷函數（Convenience Functions）
# ============================================================================

def create_trainer_with_builder(
    model: nn.Module,
    physics: Any,
    losses: Dict[str, nn.Module],
    config: Dict[str, Any],
    device: torch.device,
    **optional_components
) -> Trainer:
    """
    便捷函數：使用 Builder Pattern 創建 Trainer
    
    這是推薦的創建方式，比直接呼叫 Trainer.__init__() 更清晰。
    
    Args:
        model: 已創建的模型
        physics: 已創建的物理模組
        losses: 已創建的損失函數字典
        config: 完整訓練配置
        device: 計算設備
        **optional_components: 可選組件（weighters, input_normalizer, training_data, etc.）
    
    Returns:
        Trainer: 完整初始化的訓練器
    
    Example:
        >>> # 在 scripts/train/train.py 中，從這樣：
        >>> # trainer = Trainer(model, physics, losses, config, device,
        >>> #                    weighters=weighters, input_normalizer=normalizer, ...)
        >>> 
        >>> # 改為這樣：
        >>> trainer = create_trainer_with_builder(
        ...     model, physics, losses, config, device,
        ...     weighters=weighters,
        ...     input_normalizer=normalizer,
        ...     training_data=training_data
        ... )
    """
    return TrainerBuilder.from_components(
        model, physics, losses, config, device,
        **optional_components
    )
