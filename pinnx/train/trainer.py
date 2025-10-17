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

from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.train.loop import TrainingLoopManager, apply_point_weights_to_loss
from pinnx.utils.normalization import InputNormalizer, NormalizationConfig, DataNormalizer
from pinnx.evals.metrics import relative_L2


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
        
        # 訓練資料（待外部設置）
        self.training_data: Dict[str, torch.Tensor] = {}
        
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
                f"⚠️ 未知的調度器類型 '{scheduler_type}'，使用固定學習率。"
                f"支援的類型：'warmup_cosine', 'cosine_warm_restarts', 'cosine', "
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
        
        if self.early_stopping_enabled:
            logging.info(f"✅ 早停機制啟用（patience={self.patience}, min_delta={self.min_delta}）")
    
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
    
    def step(
        self,
        data_batch: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, Any]:
        """
        執行單步訓練（簡化版）
        
        包含核心損失計算：
        - PDE 殘差（momentum + continuity）
        - 壁面邊界條件（無滑移）
        - 資料監督損失（sensor points）
        
        Args:
            data_batch: 訓練資料批次（包含 PDE、邊界、資料點）
            epoch: 當前 epoch 數
        
        Returns:
            包含損失和指標的字典
        """
        self.optimizer.zero_grad()
        
        # ⭐ Phase 5: 檢查是否為 VS-PINN 物理（用於選擇對應的殘差計算方法）
        # 注意：座標維度已由 self.model_input_dim 控制，此 flag 僅用於 physics API 選擇
        is_vs_pinn = 'z_pde' in data_batch and hasattr(self.physics, 'compute_momentum_residuals')
        
        # ==================== 輔助函數 ====================
        def prepare_model_coords(
            coord_tensor: torch.Tensor, 
            require_grad: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            準備模型輸入座標（處理標準化與縮放）
            
            Returns:
                (coords_physical, coords_norm, model_coords):
                - coords_physical: 物理座標（供 PDE autograd 使用）
                - coords_norm: 標準化後的座標（若無 InputNormalizer 則與 coords_physical 相同）
                - model_coords: 最終模型輸入（可能經過 VS-PINN scaling）
            """
            # 1. 保留物理座標（啟用梯度追蹤）
            coords_physical = coord_tensor.clone()
            if require_grad:
                coords_physical.requires_grad_(True)
            
            # 2. 輸入標準化（可選）
            if self.input_normalizer is not None:
                coords_norm = self.input_normalizer.transform(coords_physical)
            else:
                coords_norm = coords_physical
            
            # 3. VS-PINN 縮放（可選，作用於標準化後的座標）
            if is_vs_pinn and hasattr(self.physics, 'scale_coordinates'):
                model_coords = self.physics.scale_coordinates(coords_norm)
            else:
                model_coords = coords_norm
            
            return coords_physical, coords_norm, model_coords
        
        # ==================== 1. PDE 殘差損失 ====================
        # 組合 PDE 點座標（使用模型實際輸入維度）
        if self.model_input_dim == 3:
            coords_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde'], data_batch['z_pde']], dim=1)
        else:
            coords_pde = torch.cat([data_batch['x_pde'], data_batch['y_pde']], dim=1)
        
        # ✅ 解包三元組：(物理座標, 標準化座標, 模型輸入座標)
        coords_pde_physical, coords_pde_norm, model_coords_pde = prepare_model_coords(coords_pde, require_grad=True)
        
        # 調試：打印維度（僅第一個 epoch）
        if epoch == 0 and not hasattr(self, '_debug_printed'):
            logging.info(f"🔍 調試資訊：coords_pde.shape={coords_pde.shape}, model_coords_pde.shape={model_coords_pde.shape}")
            logging.info(f"  - coords_pde_physical 梯度追蹤: {coords_pde_physical.requires_grad}")
            self._debug_printed = True
        
        # 模型預測（標準化空間輸出）
        u_pred_norm = self.model(model_coords_pde)
        
        # ✅ 立即反標準化為物理量（供 PDE 殘差計算使用）
        var_order = self._infer_variable_order(u_pred_norm.shape[1], context='pde')
        u_pred_pde_physical_raw = self.data_normalizer.denormalize_batch(u_pred_norm, var_order=var_order)
        # 確保是 Tensor 類型（denormalize_batch 保持輸入類型）
        u_pred_pde_physical: torch.Tensor = u_pred_pde_physical_raw if isinstance(u_pred_pde_physical_raw, torch.Tensor) else torch.tensor(u_pred_pde_physical_raw, device=self.device)  # type: ignore
        
        # ✅ 提取速度和壓力分量（物理空間）
        if is_vs_pinn:
            if u_pred_pde_physical.shape[1] == 3:
                # 模型只輸出 [u, v, p]，添加 w=0
                velocity_phys = u_pred_pde_physical[:, :2]
                pressure_phys = u_pred_pde_physical[:, 2:3]
                w_component_phys = torch.zeros_like(pressure_phys)
                predictions_phys = torch.cat([velocity_phys, w_component_phys, pressure_phys], dim=1)
            elif u_pred_pde_physical.shape[1] == 4:
                # 模型輸出完整 [u, v, w, p]
                predictions_phys = u_pred_pde_physical
                velocity_phys = u_pred_pde_physical[:, :2]
                pressure_phys = u_pred_pde_physical[:, 3:4]
            else:
                raise ValueError(f"VS-PINN 模型輸出維度錯誤：{u_pred_pde_physical.shape[1]}，期望 3 或 4")
        else:
            if u_pred_pde_physical.shape[1] == 3:
                velocity_phys = u_pred_pde_physical[:, :2]
                pressure_phys = u_pred_pde_physical[:, 2:3]
            elif u_pred_pde_physical.shape[1] == 4:
                velocity_phys = u_pred_pde_physical[:, :2]
                pressure_phys = u_pred_pde_physical[:, 3:4]
            else:
                raise ValueError(f"標準 PINN 輸出維度錯誤: {u_pred_pde_physical.shape[1]}，期望 3 或 4")
            predictions_phys = None
        
        # ✅ 計算物理殘差（使用物理座標 + 物理量）
        try:
            if is_vs_pinn and hasattr(self.physics, 'compute_momentum_residuals'):
                # VS-PINN 3D
                residuals_mom = self.physics.compute_momentum_residuals(
                    coords_pde_physical,    # ✅ 物理座標（含梯度）
                    predictions_phys,        # ✅ 物理量
                    scaled_coords=model_coords_pde  # VS-PINN 仍需縮放座標
                )
                continuity_residual = self.physics.compute_continuity_residual(
                    coords_pde_physical,    # ✅ 物理座標（含梯度）
                    predictions_phys,        # ✅ 物理量
                    scaled_coords=model_coords_pde
                )
                residuals = {
                    'momentum_x': residuals_mom['momentum_x'],
                    'momentum_y': residuals_mom['momentum_y'],
                    'momentum_z': residuals_mom['momentum_z'],
                    'continuity': continuity_residual,
                }
            else:
                # 標準 NS 2D：需要 2D 座標 [x, y] + 完整 pred 張量 [u, v, p, S]
                # ⚠️ 修復：只傳遞前 2D 座標給 2D 物理模組
                coords_pde_2d = coords_pde_physical[:, :2]  # 取 [x, y]，忽略 z
                
                # 構建完整預測張量 [u, v, p, S]（NS 2D 需要 4 個分量）
                source_term_phys = torch.zeros_like(pressure_phys)  # 假設源項為 0
                pred_full_phys = torch.cat([velocity_phys, pressure_phys, source_term_phys], dim=1)
                
                # 調用 NSEquations2D.residual_unified() 方法
                residuals = self.physics.residual_unified(coords_pde_2d, pred_full_phys)
            
            # 應用點權重（如果存在）
            pde_point_weights = data_batch.get('pde_point_weights', None)
            if pde_point_weights is not None:
                momentum_x_loss = apply_point_weights_to_loss(residuals['momentum_x']**2, pde_point_weights)
                momentum_y_loss = apply_point_weights_to_loss(residuals['momentum_y']**2, pde_point_weights)
                momentum_z_loss = apply_point_weights_to_loss(residuals['momentum_z']**2, pde_point_weights) if is_vs_pinn else torch.tensor(0.0, device=self.device)
                continuity_loss = apply_point_weights_to_loss(residuals['continuity']**2, pde_point_weights)
            else:
                momentum_x_loss = torch.mean(residuals['momentum_x']**2)
                momentum_y_loss = torch.mean(residuals['momentum_y']**2)
                momentum_z_loss = torch.mean(residuals['momentum_z']**2) if is_vs_pinn else torch.tensor(0.0, device=self.device)
                continuity_loss = torch.mean(residuals['continuity']**2)
        
        except Exception as e:
            logging.error(f"🚨 物理殘差計算失敗: {e}")
            logging.error(f"coords_pde shape: {coords_pde.shape}, u_pred_norm shape: {u_pred_norm.shape}, u_pred_pde_physical shape: {u_pred_pde_physical.shape}")
            raise
        
        # ==================== 1B. RANS 已移除（僅保留為 LoFi 場診斷工具）====================
        # ⚠️ RANS 不再作為損失項參與訓練（2025-10-14 變更）
        # compute_rans_residuals() 保留在 physics 模組中用於：
        # 1. 未來 LoFi 場輸入特徵提取
        # 2. 訓練過程中的診斷與監控
        
        # ==================== 2. 壁面邊界條件損失 ====================
        # 組合邊界條件座標（使用模型實際輸入維度）
        if self.model_input_dim == 3:
            coords_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc'], data_batch['z_bc']], dim=1)
        else:
            coords_bc = torch.cat([data_batch['x_bc'], data_batch['y_bc']], dim=1)
        
        # ✅ 解包三元組
        coords_bc_physical, coords_bc_norm, model_coords_bc = prepare_model_coords(coords_bc, require_grad=False)
        u_bc_pred_norm = self.model(model_coords_bc)
        
        # ✅ 反標準化為物理量（壁面邊界條件在物理空間應為 0）
        var_order_bc = self._infer_variable_order(u_bc_pred_norm.shape[1], context='bc')
        u_bc_pred_phys_raw = self.data_normalizer.denormalize_batch(u_bc_pred_norm, var_order=var_order_bc)
        u_bc_pred_phys: torch.Tensor = u_bc_pred_phys_raw if isinstance(u_bc_pred_phys_raw, torch.Tensor) else torch.tensor(u_bc_pred_phys_raw, device=self.device)  # type: ignore
        
        # 識別壁面點（y = ±1）
        y_bc = data_batch['y_bc']
        wall_mask = (torch.abs(y_bc - 1.0) < 1e-3) | (torch.abs(y_bc + 1.0) < 1e-3)
        wall_mask = wall_mask.squeeze()
        
        if wall_mask.sum() > 0:
            # ✅ 使用物理量（壁面速度應為 0）
            u_wall = u_bc_pred_phys[wall_mask, 0]  # u 分量（物理空間）
            v_wall = u_bc_pred_phys[wall_mask, 1]  # v 分量（物理空間）
            wall_loss = torch.mean(u_wall**2 + v_wall**2)
        else:
            wall_loss = torch.tensor(0.0, device=self.device)
            if epoch == 0:
                logging.warning(f"⚠️ 未檢測到壁面邊界點！y_bc 範圍: [{y_bc.min():.6f}, {y_bc.max():.6f}]")
        
        # ==================== 3. 資料監督損失 ====================
        # 組合感測器座標（使用模型實際輸入維度）
        if self.model_input_dim == 3:
            coords_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors'], data_batch['z_sensors']], dim=1)
        else:
            coords_sensors = torch.cat([data_batch['x_sensors'], data_batch['y_sensors']], dim=1)
        
        # ✅ 解包三元組
        coords_sensors_physical, coords_sensors_norm, model_coords_sensors = prepare_model_coords(coords_sensors, require_grad=False)
        u_sensors_pred_norm = self.model(model_coords_sensors)
        
        # ✅ 反標準化為物理量（直接與真實物理量比較）
        var_order_sensors = self._infer_variable_order(
            u_sensors_pred_norm.shape[1],
            context='sensors',
            data_batch=data_batch
        )
        u_sensors_pred_phys_raw = self.data_normalizer.denormalize_batch(u_sensors_pred_norm, var_order=var_order_sensors)
        u_sensors_pred_phys: torch.Tensor = u_sensors_pred_phys_raw if isinstance(u_sensors_pred_phys_raw, torch.Tensor) else torch.tensor(u_sensors_pred_phys_raw, device=self.device)  # type: ignore
        
        # 提取真實資料（物理空間）
        u_true = data_batch['u_sensors']
        v_true = data_batch['v_sensors']
        w_true = data_batch.get('w_sensors', None)  # 3D 通道流才有 w
        p_true = data_batch['p_sensors']
        
        # ✅ 物理空間的 MSE（直接比對，不再標準化真實值）
        u_loss = torch.mean((u_sensors_pred_phys[:, 0:1] - u_true)**2)
        v_loss = torch.mean((u_sensors_pred_phys[:, 1:2] - v_true)**2)
        
        # ⭐ Phase 5: 根據資料與模型維度動態選擇損失計算
        # 檢查 w 分量是否存在且有效（非空張量）
        has_w_data = w_true is not None and w_true.numel() > 0
        model_has_w = u_sensors_pred_phys.shape[1] >= 4  # 模型輸出至少 4 個分量
        
        if has_w_data and model_has_w:
            # 完整 3D 模式：u, v, w, p
            w_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - w_true)**2)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
            velocity_loss = u_loss + v_loss + w_loss
        elif model_has_w and not has_w_data:
            # 混合模式：模型輸出 4D 但資料僅有 3D（忽略 w 預測）
            w_loss = torch.tensor(0.0, device=u_loss.device)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)  # ⚠️ 壓力在 index 3
            velocity_loss = u_loss + v_loss
        else:
            # 標準 2D 模式：u, v, p
            w_loss = torch.tensor(0.0, device=u_loss.device)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - p_true)**2)  # 壓力在 index 2
            velocity_loss = u_loss + v_loss
        
        data_loss = velocity_loss + pressure_loss
        
        # 統計原始 PDE 損失（未加權，便於日志與分析）
        pde_loss = momentum_x_loss + momentum_y_loss + momentum_z_loss + continuity_loss
        
        # ==================== 4. 組合損失（含 GradNorm 動態權重）====================
        loss_cfg = self.loss_cfg
        
        # 收集可供權重調整的損失項
        loss_terms: Dict[str, torch.Tensor] = {
            'data': data_loss,
            'momentum_x': momentum_x_loss,
            'momentum_y': momentum_y_loss,
            'continuity': continuity_loss,
            'wall_constraint': wall_loss,
        }
        if is_vs_pinn:
            loss_terms['momentum_z'] = momentum_z_loss
        
        gradnorm_weighter = self.weighters.get('gradnorm') if hasattr(self, 'weighters') else None
        gradnorm_weights: Optional[Dict[str, float]] = None
        gradnorm_ratio: Dict[str, float] = {}
        if gradnorm_weighter is not None:
            available_losses = {
                name: loss_terms[name]
                for name in gradnorm_weighter.loss_names
                if name in loss_terms
            }
            if len(available_losses) >= 2:
                gradnorm_weights = gradnorm_weighter.update_weights(available_losses)
                initial_values = getattr(gradnorm_weighter, 'initial_weight_values', {})
                eps = float(getattr(gradnorm_weighter, 'eps', 1e-12))
                for name, weight_val in gradnorm_weights.items():
                    base = float(initial_values.get(name, 1.0))
                    if abs(base) < eps:
                        base = 1.0
                    gradnorm_ratio[name] = float(weight_val) / base
                if gradnorm_weighter.step_count % gradnorm_weighter.update_frequency == 0:
                    logging.debug(
                        "GradNorm weights @ step %d: %s",
                        gradnorm_weighter.step_count,
                        {k: round(v, 4) for k, v in gradnorm_weights.items()}
                    )
            else:
                logging.debug("GradNorm requires at least two loss terms; skipping update.")
        
        def scaled_weight(name: str, base: float) -> float:
            return float(base) * gradnorm_ratio.get(name, 1.0)

        # 基礎權重（可從配置讀取或使用預設）
        base_data_weight = loss_cfg.get('data_weight', 100.0)
        base_pde_weight = loss_cfg.get('pde_weight', 1.0)
        base_bc_weight = loss_cfg.get('bc_weight', 10.0)
        
        # ⭐ 支援細粒度權重配置（優先級：細項 > 統一 > 預設）
        # 優先讀取細項權重，若不存在則回退到統一權重
        w_data = scaled_weight('data', base_data_weight)
        w_momentum_x = scaled_weight('momentum_x', loss_cfg.get('momentum_x_weight', base_pde_weight))
        w_momentum_y = scaled_weight('momentum_y', loss_cfg.get('momentum_y_weight', base_pde_weight))
        w_momentum_z = scaled_weight('momentum_z', loss_cfg.get('momentum_z_weight', base_pde_weight)) if is_vs_pinn else 0.0
        w_continuity = scaled_weight('continuity', loss_cfg.get('continuity_weight', base_pde_weight))
        w_bc = scaled_weight('wall_constraint', loss_cfg.get('wall_constraint_weight', base_bc_weight))
        
        # 📊 診斷日誌：在訓練開始時打印實際應用的權重
        if epoch == 0 and not hasattr(self, '_weights_logged'):
            logging.info("=" * 60)
            logging.info("📊 Loss 權重配置摘要")
            logging.info("=" * 60)
            logging.info(f"  Data Loss:        {w_data:.2e}")
            logging.info(f"  Momentum X:       {w_momentum_x:.2e}")
            logging.info(f"  Momentum Y:       {w_momentum_y:.2e}")
            if is_vs_pinn:
                logging.info(f"  Momentum Z:       {w_momentum_z:.2e}")
            logging.info(f"  Continuity:       {w_continuity:.2e}")
            logging.info(f"  Wall Constraint:  {w_bc:.2e}")
            logging.info("=" * 60)
            self._weights_logged = True
        
        # ==================== RANS 損失已移除（2025-10-14）====================
        # RANS 權重預熱功能已停用，相關損失項已從訓練循環中移除

        weighted_pde_loss = (
            w_momentum_x * momentum_x_loss +
            w_momentum_y * momentum_y_loss +
            w_momentum_z * momentum_z_loss +
            w_continuity * continuity_loss
        )
        weighted_data_loss = w_data * data_loss
        weighted_wall_loss = w_bc * wall_loss
        
        # ⭐ Phase 6C: 均值約束損失（若啟用）
        mean_constraint_loss = torch.tensor(0.0, device=data_loss.device)
        mean_constraint_cfg = loss_cfg.get('mean_constraint', {})
        if mean_constraint_cfg.get('enabled', False) and 'mean_constraint' in self.losses:
            # ✅ 使用已反標準化的 PDE 點預測（更全域的統計量）
            # u_pred_pde_physical 已在 Line 515 反標準化完成
            
            target_means = mean_constraint_cfg.get('target_means', {})
            field_indices = {'u': 0, 'v': 1, 'w': 2}  # 不約束壓力場
            
            mean_constraint_loss_fn = self.losses['mean_constraint']
            mean_constraint_loss = mean_constraint_loss_fn(
                predictions=u_pred_pde_physical,  # ✅ 使用物理空間的預測
                target_means=target_means,         # 物理空間的目標均值
                field_indices=field_indices
            )
            
            w_mean_constraint = mean_constraint_cfg.get('weight', 10.0)
            mean_constraint_loss = w_mean_constraint * mean_constraint_loss
            
            # 記錄（低頻率）
            if epoch == 0 or (epoch > 0 and epoch % 100 == 0):
                logging.info(f"📊 均值約束損失 @ Epoch {epoch}: {mean_constraint_loss.item():.6f}")
                # 記錄預測均值以便診斷
                with torch.no_grad():
                    for field_name, idx in field_indices.items():
                        if idx < u_pred_pde_physical.shape[-1]:
                            pred_mean = u_pred_pde_physical[:, idx].mean().item()
                            target_mean = target_means.get(field_name, 0.0)
                            logging.info(f"   {field_name}: pred={pred_mean:.4f}, target={target_mean:.4f}")
        
        # 總損失（不含 RANS 項）
        total_loss = (
            weighted_data_loss +
            weighted_pde_loss +
            weighted_wall_loss +
            mean_constraint_loss  # ⭐ Phase 6C: 均值約束損失
        )
        
        # ==================== 5. 反向傳播與優化 ====================
        # ⭐ P0.2: AMP 混合精度策略
        # - Forward Pass: 已在上面完成（FP32）
        # - Backward Pass: 使用 GradScaler（FP16 梯度累積）
        
        # 縮放損失並反向傳播
        scaled_loss = self.scaler.scale(total_loss)
        scaled_loss.backward()
        
        # 梯度裁剪（需先 unscale）
        if self.train_cfg.get('gradient_clip', 0.0) > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.train_cfg['gradient_clip']
            )
        
        # L-BFGS 需要 closure 函數，其他優化器直接 step()
        if isinstance(self.optimizer, torch.optim.LBFGS):
            # L-BFGS 不支援 AMP（已在 _setup_amp 中禁用）
            def closure():
                return total_loss
            self.optimizer.step(closure)  # type: ignore
        else:
            # Adam: 使用 scaler.step() 自動處理 unscale + update
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        # 📉 Steps-based 調度器更新（每步調用）
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'current_step'):
            # Steps-based scheduler (如 StepsBasedWarmupScheduler)
            self.lr_scheduler.step()
        
        # ==================== 6. 返回結果 ====================
        result = {
            'total_loss': total_loss.item(),
            'data_loss': data_loss.item(),
            'pde_loss': pde_loss.item(),
            'weighted_data_loss': weighted_data_loss.item(),
            'weighted_pde_loss': weighted_pde_loss.item(),
            'weighted_wall_loss': weighted_wall_loss.item(),
            'momentum_x_loss': momentum_x_loss.item(),
            'momentum_y_loss': momentum_y_loss.item(),
            'momentum_z_loss': momentum_z_loss.item(),
            'continuity_loss': continuity_loss.item(),
            'wall_loss': wall_loss.item(),
            'u_loss': u_loss.item(),
            'v_loss': v_loss.item(),
            'w_loss': w_loss.item(),
            'pressure_loss': pressure_loss.item(),
        }
        
        # RANS 損失項已移除（2025-10-14）
        
        if gradnorm_weights is not None:
            result['gradnorm_weights'] = {k: float(v) for k, v in gradnorm_weights.items()}
            result['applied_weights'] = {
                'data': w_data,
                'momentum_x': w_momentum_x,
                'momentum_y': w_momentum_y,
                'momentum_z': w_momentum_z,
                'continuity': w_continuity,
                'wall_constraint': w_bc,
            }
        
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
            # ✅ 使用 prepare_model_coords 輔助函數處理座標（需要在 step() 之外定義或改為實例方法）
            # 簡化處理：直接在此處理座標標準化與縮放
            coords_for_model = coords
            if self.input_normalizer is not None:
                coords_for_model = self.input_normalizer.transform(coords_for_model)
            if self.physics is not None and hasattr(self.physics, 'scale_coordinates'):
                coords_for_model = self.physics.scale_coordinates(coords_for_model)
            
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
            
            # 🚀 課程訓練：處理階段切換
            if '_curriculum_transition' in loss_dict and loss_dict['_curriculum_transition'] > 0.5:
                new_lr = loss_dict.get('_curriculum_lr', self.train_cfg.get('lr', 1e-3))
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                logging.info(f"📉 課程訓練：學習率更新為 {new_lr:.6f}")
                
                # 保存階段檢查點（如果啟用）
                if self.log_cfg.get('save_stage_checkpoints', False):
                    stage_name = loss_dict.get('_curriculum_stage', f'stage_{epoch}')
                    self.save_checkpoint(epoch, loss_dict, is_best=False)
                    logging.info(f"💾 階段檢查點已保存: {stage_name}")
            
            # 📉 更新學習率調度器（非課程訓練模式）
            if self.lr_scheduler is not None and not hasattr(self, 'curriculum_weighter'):
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
            
            # 快速收斂檢查
            if loss_dict['total_loss'] < 1e-6:
                logging.info(f"✅ 快速收斂於 epoch {epoch}（loss < 1e-6）")
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
        log_str = f"Epoch {epoch}/{self.train_cfg.get('epochs', '?')}"
        
        for key, value in metrics.items():
            # 跳過字典類型的值（如 gradnorm_weights, applied_weights）
            if isinstance(value, dict):
                continue
            # 跳過非數值類型（如字串、列表等）
            if not isinstance(value, (int, float)):
                continue
            log_str += f" | {key}: {value:.6f}"
        
        log_str += f" | lr: {self.get_current_lr():.2e}"
        
        logging.info(log_str)
        
        # 記錄到歷史
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        self.history['lr'].append(self.get_current_lr())
