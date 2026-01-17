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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import wandb


# ========== 資料批次解包結構（減少 dict 查詢開銷）==========

@dataclass
class UnpackedDataBatch:
    """
    解包後的訓練數據批次（減少 dict 查詢開銷）

    優勢：
    1. 屬性訪問比 dict.get() 快 ~20-30%
    2. 類型提示支援（IDE 自動補全）
    3. 可選欄位明確標示

    Note:
        此結構用於 step() 內部優化，不影響外部 API
    """
    # 空間座標（必填）
    coords_pde_spatial: torch.Tensor
    coords_bc_spatial: torch.Tensor
    coords_sensors_spatial: torch.Tensor

    # 時間座標（可選）
    t_pde: Optional[torch.Tensor] = None
    t_bc: Optional[torch.Tensor] = None
    t_sensors: Optional[torch.Tensor] = None

    # 感測器真值
    u_sensors: Optional[torch.Tensor] = None
    v_sensors: Optional[torch.Tensor] = None
    p_sensors: Optional[torch.Tensor] = None
    w_sensors: Optional[torch.Tensor] = None

    # 可選數據
    lowfi_prior: Optional[Dict[str, Any]] = None
    pde_point_weights: Optional[torch.Tensor] = None
    y_bc: Optional[torch.Tensor] = None
    has_prior: bool = False

from pinnx.losses.residuals import NSResidualLoss, BoundaryConditionLoss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter, AdaptiveWeightScheduler
from pinnx.train.loop import TrainingLoopManager, apply_point_weights_to_loss
from pinnx.train.loss_manager import LossManager  # type: ignore
from pinnx.utils.normalization import InputTransform, OutputTransform
from pinnx.evals.metrics import relative_L2
from pinnx.physics.turbulence_utils import preprocess_rans_prior, preprocess_rans_prior_from_config  # type: ignore
from pinnx.physics.gradient_cache import GradientCache  # type: ignore


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
        input_normalizer (InputTransform): 輸入標準化器（可選）
        
        epoch (int): 當前訓練 epoch
        history (Dict[str, List]): 訓練歷史記錄
    """
    
    def __init__(
        self,
        model: nn.Module,
        physics: Any,
        losses: Dict[str, nn.Module],
        config: Dict[str, Any],
        device: torch.device,
        components: 'TrainerComponents',
    ):
        """
        初始化訓練器（P2-3 重構：純 TrainerComponents 依賴注入）

        ⚠️  直接實例化 Trainer 已移除，請使用 TrainerBuilder。

        Args:
            model: PINN 模型
            physics: 物理方程模組
            losses: 損失函數字典
            config: 完整訓練配置
            device: 計算設備
            components: TrainerComponents 實例（必需）

        推薦用法：
            ```python
            from pinnx.train.trainer_builder import TrainerBuilder

            builder = TrainerBuilder(config, device)
            builder.with_model(model)
            builder.with_physics(physics)
            builder.with_losses(losses)
            builder.with_training_data(training_data)

            trainer = builder.build()
            ```

        Raises:
            TypeError: 如果 components 為 None
        """
        # ========== 配置驗證（Fail Fast）==========
        self._validate_config_early(config)

        # ========== 強制使用 TrainerBuilder ==========
        if components is None:
            raise TypeError(
                "\n"
                "❌ Trainer 不再支持直接實例化。\n"
                "\n"
                "請使用 TrainerBuilder：\n"
                "\n"
                "    from pinnx.train.trainer_builder import TrainerBuilder\n"
                "\n"
                "    builder = TrainerBuilder(config, device)\n"
                "    builder.with_model(model)\n"
                "    builder.with_physics(physics)\n"
                "    builder.with_losses(losses)\n"
                "    builder.with_training_data(training_data)\n"
                "\n"
                "    trainer = builder.build()\n"
                "\n"
                "優勢：\n"
                "  - 自動創建所有組件（optimizer, schedulers, normalizers, etc.）\n"
                "  - 依賴注入架構，易於測試和擴展\n"
                "  - 減少手動配置錯誤\n"
                "\n"
                "詳見文檔: docs/TRAINERBUILDER_GUIDE.md\n"
            )

        # ========== 使用 TrainerComponents 初始化 ==========
        logging.info("🏗️  Trainer: 使用 TrainerComponents 初始化")
        self._init_from_components(
            model, physics, losses, config, device, components
        )

        logging.info(f"✅ Trainer 初始化完成（設備: {device})")

    # ========================================================================
    # 新路徑：從 TrainerComponents 初始化（P2-3）
    # ========================================================================

    def _init_from_components(
        self,
        model: nn.Module,
        physics: Any,
        losses: Dict[str, nn.Module],
        config: Dict[str, Any],
        device: torch.device,
        components: 'TrainerComponents'
    ):
        """
        從 TrainerComponents 初始化（新路徑，純依賴注入）

        將所有組件從 components 提取並賦值給 self，不執行任何創建邏輯。

        Args:
            model, physics, losses, config, device: 核心組件
            components: TrainerComponents 實例（包含所有可選組件）
        """
        # 1. 基本組件
        self.model = model
        self.physics = physics
        self.losses = losses
        self.config = config
        self.device = device

        # 提取常用配置
        self.train_cfg = config['training']
        self.loss_cfg = config.get('losses', {})
        self.log_cfg = config.get('logging', {})
        self.physics_type = config.get('physics', {}).get('type', '')
        self.is_vs_cfg = self.physics_type == 'vs_pinn_channel_flow'

        # 檢測模型輸入維度
        self.model_input_dim = self._detect_model_input_dim(model, config)
        logging.info(f"🔍 檢測到模型輸入維度: {self.model_input_dim}D")

        # 2. 訓練組件（從 components 提取）
        self.optimizer = components.optimizer
        self.lr_scheduler = components.lr_scheduler
        self.weight_scheduler = components.weight_scheduler

        # 3. 策略組件
        self.checkpoint_manager = components.checkpoint_manager
        self.checkpoint_strategy = components.checkpoint_strategy
        self.validation_manager = components.validation_manager
        self.checkpoint_dir = self.checkpoint_manager.checkpoint_dir if hasattr(
            self.checkpoint_manager, 'checkpoint_dir'
        ) else 'checkpoints'

        # 4. 功能組件
        self.prior_loss_manager = components.prior_loss_manager
        self.fourier_annealing = components.fourier_annealing
        self.adaptive_sampler = components.adaptive_sampler
        self.adaptive_sampling_enabled = (components.adaptive_sampler is not None)

        # Early stopping config
        es_cfg = components.early_stopping_config
        self.early_stopping_cfg = es_cfg  # 保存完整配置字典
        self.early_stopping_enabled = es_cfg.get('enabled', False)
        self.convergence_threshold = es_cfg.get('convergence_threshold', None)

        # 5. 監控組件
        self.timer = components.timer
        self.memory_tracker = components.memory_tracker
        self.physics_validator = components.physics_validator
        self.wandb_run = components.wandb_run
        self.use_wandb = (components.wandb_run is not None)

        # 6. 數據組件
        self.input_normalizer = components.input_normalizer
        self.weighters = components.weighters or {}
        self.channel_data_cache = components.channel_data_cache or {}
        self.training_data = components.training_data or {}

        # data_normalizer 需要特殊處理（依賴 training_data）
        if components.data_normalizer is not None:
            self.data_normalizer = components.data_normalizer
        else:
            # 從 config 創建（需要 training_data）
            self._init_normalizers(config, components.training_data)
        
        # hard_constraint_applicator（可選）
        self.hard_constraint_applicator = components.hard_constraint_applicator
        if self.hard_constraint_applicator is not None:
            logging.info("✅ Hard constraint applicator 已啟用（壁面邊界條件）")

        # 7. 訓練狀態初始化
        self.epoch = 0
        self.global_step = 0
        self.step_offset = components.step_offset  # ✅ Time window 模式：步數偏移
        self.history = {
            'train_loss': [],
            'pde_loss': [],
            'data_loss': [],
            'val_loss': [],
            'lr': [],
        }

        # 驗證相關
        self.validation_data = components.validation_data
        self.best_val_loss = float('inf')
        self.best_epoch = -1
        self.best_model_state: Optional[Dict[str, torch.Tensor]] = None

        # 8. 初始化 LossManager（依賴多個組件）
        self._init_loss_manager_from_components()

        # 9. 配置 input transform
        self._configure_input_transform()

        # 10. 配置 Validation strategies（如果需要）
        self._init_validation_strategies_if_needed()

        logging.info("✅ TrainerComponents 初始化完成（所有組件已注入）")

    def _init_loss_manager_from_components(self):
        """從 components 初始化 LossManager"""
        from pinnx.train.loss_manager import LossManager

        self.loss_manager = LossManager(
            config=self.config,
            physics=self.physics,
            model=self.model,
            device=self.device,
            data_normalizer=self.data_normalizer,
            prior_loss_manager=self.prior_loss_manager,
            weighters=self.weighters,
            losses=self.losses
        )
        logging.info("✅ LossManager 初始化完成（從 components）")

    def _init_validation_strategies_if_needed(self):
        """配置缺失的驗證策略（獨立檢查每種策略類型）"""
        from pinnx.train.validation_manager import DataBasedValidation, PhysicsBasedValidation

        # 檢查現有策略類型
        has_data_strategy = any(isinstance(s, DataBasedValidation) for s in self.validation_manager.strategies)
        has_physics_strategy = any(isinstance(s, PhysicsBasedValidation) for s in self.validation_manager.strategies)

        # 添加數據驗證策略（如果有驗證數據且尚未添加）
        if not has_data_strategy and hasattr(self, 'validation_data') and self.validation_data is not None:
            def preprocess_coords(coords):
                _, _, coords_for_model = self._prepare_model_coords(
                    coords, require_grad=False, is_vs_pinn=None
                )
                return coords_for_model

            def infer_var_order(n_outputs):
                return self._infer_variable_order(n_outputs, context='validation')

            data_strategy = DataBasedValidation(
                validation_data=self.validation_data,
                metrics=['mse', 'relative_l2'],
                data_normalizer=self.data_normalizer,
                model_input_preprocessor=preprocess_coords,
                variable_order_inference=infer_var_order
            )
            self.validation_manager.add_strategy(data_strategy)

        # 添加物理驗證策略（如果配置了且尚未添加）
        if not has_physics_strategy and hasattr(self, 'physics_validator') and self.physics_validator is not None:
            physics_strategy = PhysicsBasedValidation(
                physics_validator=self.physics_validator
            )
            self.validation_manager.add_strategy(physics_strategy)

    # ========================================================================
    # 舊路徑：傳統初始化邏輯（向後兼容）
    # ========================================================================


    def _validate_config_early(self, config: Dict[str, Any]) -> None:
        """
        配置早期驗證（Fail Fast 原則）
        
        在初始化任何組件之前檢查配置，避免 Silent Failure。
        
        Args:
            config: 完整訓練配置
        
        Raises:
            KeyError: 發現常見配置錯誤
        """
        from pinnx.utils.config_validator import quick_check_common_errors
        
        # 快速檢查最常見的錯誤
        error_msg = quick_check_common_errors(config)
        if error_msg:
            raise KeyError(
                f"\n{'=' * 80}\n"
                f"❌ 配置驗證失敗:\n\n{error_msg}\n"
                f"{'=' * 80}\n"
            )


    def _init_normalizers(self, config: Dict[str, Any], training_data: Optional[Dict[str, torch.Tensor]]):
        """初始化標準化器並驗證統計量有效性"""
        self.data_normalizer = OutputTransform.from_config(config, training_data=training_data)
        logging.info(f"📐 OutputTransform 初始化: {self.data_normalizer}")

        # 驗證標準化統計量
        if not self.data_normalizer.has_valid_stats():
            raise RuntimeError(
                "❌ OutputTransform 統計量無效！\n"
                "   可能原因:\n"
                "   1. 標準化類型為 'training_data_norm' 但未提供 training_data\n"
                "   2. training_data 中某些變量為常數（std ≈ 0）\n"
                "   3. config['normalization']['params'] 中的統計量包含 NaN/Inf\n"
                "   解決方案:\n"
                "   1. 檢查 config['normalization']['params'] 是否正確設定\n"
                "   2. 確認 training_data 傳遞給 Trainer.__init__()\n"
                "   3. 檢查 sensor data 的數值範圍是否合理\n"
                "   4. 若某變量確實為常數，考慮使用 normalization.type='none'"
            )



    def _load_wandb_config(self) -> Dict[str, str]:
        """載入 WandB 配置（API key 與 project）"""
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

    def _upload_wandb_metadata(self, config: Dict[str, Any]):
        """上傳 config metadata 到 WandB"""
        if 'experiment' not in config or '_config_metadata' not in config['experiment']:
            return

        metadata = config['experiment']['_config_metadata']
        for key, value in metadata.items():
            wandb.config.update({f'_meta_{key}': value}, allow_val_change=True)






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
        從配置中獲取輸出變量順序（Fail Fast，必須明確配置）
        
        要求：
            model.output_variables 必須在配置中明確定義
        
        Args:
            out_dim: 模型輸出維度
            context: 呼叫上下文（用於錯誤訊息）
            data_batch: 資料批次（未使用，保留介面相容性）
        
        Returns:
            變數名稱列表，例如 ['u', 'v', 'p']
        
        Raises:
            KeyError: 配置中缺少 model.output_variables
            ValueError: output_variables 長度與 out_dim 不一致
        
        Note:
            P0-3 重構：移除所有 fallback 和啟發式猜測
            若配置文件缺少 output_variables，請執行：
                python scripts/tools/add_output_variables.py --config <file>
        """
        if out_dim <= 0:
            return []
        
        # Fail Fast: 只接受明確的 model.output_variables
        model_cfg = self.config.get('model', {})
        output_variables = model_cfg.get('output_variables')
        
        if output_variables is None:
            raise KeyError(
                f"Config missing required key: model.output_variables\n"
                f"Context: {context}\n"
                f"Expected format:\n"
                f"  model:\n"
                f"    output_variables: [u, v, p]  # List of variable names\n"
                f"\n"
                f"To add output_variables to old configs, run:\n"
                f"  python scripts/tools/add_output_variables.py --config <file>"
            )
        
        # 驗證長度
        var_list = list(output_variables)
        if len(var_list) != out_dim:
            raise ValueError(
                f"Mismatch between model.output_variables length ({len(var_list)}) "
                f"and model output dimension ({out_dim})\n"
                f"Context: {context}\n"
                f"output_variables: {var_list}\n"
                f"Please update config to ensure len(output_variables) == out_dim"
            )
        
        return var_list
    
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
            - coords_norm: 標準化坐標（若有 InputTransform）
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
    
    
    
    
    
    
    
    
    def step(
        self,
        data_batch: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, Any]:
        """
        執行單步訓練（激進重構版：Good Taste 原則）

        將 241 行重構為 ~35 行，消除所有重複代碼。

        Args:
            data_batch: 訓練資料批次
            epoch: 當前 epoch

        Returns:
            包含損失和指標的字典
        """
        self.optimizer.zero_grad()

        # 🚀 優化: 批量異步傳輸所有數據到 GPU（Wave 3 優化）
        # 減少多次 .to(device) 調用，改為一次性批量傳輸
        data_batch = self._transfer_batch_to_device(data_batch)

        # 🚀 優化: 一次性解包常用 keys（減少 dict 查詢開銷）
        # 每步約 50+ 次 dict.get() → 1 次批量解包
        unpacked = self._unpack_data_batch(data_batch)

        # 1. 前向傳播（統一處理 3 類點：PDE, BC, Sensors）
        # Note: 內部方法暫時仍使用 data_batch，後續可逐步遷移到 unpacked
        predictions = self._forward_pass_all_points(data_batch)

        # 2. 計算所有損失（委派給 LossManager）
        losses = self._compute_all_losses(predictions, data_batch, epoch)

        # 3. 動態權重調整 + 損失組合
        total_loss, result = self._combine_and_weight_losses(losses, predictions['is_vs_pinn'], epoch)

        # 4. 反向傳播與優化
        self._backward_and_optimize(total_loss, data_batch, epoch)

        # 4.1 DDP 損失同步（僅用於日誌與監控）
        if self._is_ddp_enabled() and self._should_reduce_losses():
            from pinnx.utils.ddp_utils import reduce_loss_dict
            result = reduce_loss_dict(result, average=True)

        # 5. 附加訓練元數據
        self._add_training_metadata(result, losses, epoch)

        return result

    # ========================================================================
    # step() 輔助方法（Phase 2 重構：消除重複代碼）
    # ========================================================================

    def _unpack_data_batch(self, data_batch: Dict[str, torch.Tensor]) -> UnpackedDataBatch:
        """
        一次性解包 data_batch（減少後續 dict 查詢）

        性能提升：每步約 50+ 次 dict.get() → 1 次批量解包

        Args:
            data_batch: 原始訓練資料批次字典

        Returns:
            UnpackedDataBatch 實例
        """
        return UnpackedDataBatch(
            coords_pde_spatial=data_batch['coords_pde_spatial'],
            coords_bc_spatial=data_batch['coords_bc_spatial'],
            coords_sensors_spatial=data_batch['coords_sensors_spatial'],
            t_pde=data_batch.get('t_pde'),
            t_bc=data_batch.get('t_bc'),
            t_sensors=data_batch.get('t_sensors'),
            u_sensors=data_batch.get('u_sensors'),
            v_sensors=data_batch.get('v_sensors'),
            p_sensors=data_batch.get('p_sensors'),
            w_sensors=data_batch.get('w_sensors'),
            lowfi_prior=data_batch.get('lowfi_prior'),
            pde_point_weights=data_batch.get('pde_point_weights'),
            y_bc=data_batch.get('y_bc'),
            has_prior=data_batch.get('has_prior', False),
        )

    def _forward_pass_all_points(self, data_batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        統一處理所有點類型的前向傳播（PDE, BC, Sensors）

        消除重複：3 處相同的座標處理 + 前向傳播邏輯合併為單一方法

        Args:
            data_batch: 訓練資料批次

        Returns:
            predictions: {
                'pde': {coords_physical, predictions, gradients},
                'bc': {coords_physical, predictions},
                'sensors': {coords_physical, predictions},
                'is_vs_pinn': bool
            }
        """
        # 檢測 Physics 類型與維度
        if 'coords_pde_spatial' not in data_batch:
            raise KeyError("data_batch 缺少必要鍵: coords_pde_spatial")
        
        coord_dim = data_batch['coords_pde_spatial'].shape[1]
        has_3d_coords = coord_dim >= 3
        is_vs_pinn = has_3d_coords and hasattr(self.physics, 'compute_momentum_residuals')
        is_2d_flow = coord_dim in (2, 3) and hasattr(self.physics, 'residual')

        # 處理 3 類點（消除重複！）
        pde_batch = self._process_point_batch(
            'coords_pde_spatial', 't_pde', data_batch,
            require_grad=True, context='pde'
        )
        bc_batch = self._process_point_batch(
            'coords_bc_spatial', 't_bc', data_batch,
            require_grad=False, context='bc'
        )
        sensors_batch = self._process_point_batch(
            'coords_sensors_spatial', 't_sensors', data_batch,
            require_grad=False, context='sensors'
        )

        # 梯度緩存（Wave 2 優化）
        gradients = None
        if is_vs_pinn:
            # VS-PINN 3D：使用 3D 梯度快取
            gradients = self._compute_vs_pinn_gradients(
                pde_batch['predictions'], 
                pde_batch['coords_physical'], 
                is_vs_pinn
            )
        elif is_2d_flow:
            # Kolmogorov Flow 2D：使用 2D 梯度快取
            gradients = self._compute_2d_gradients(
                pde_batch['predictions'], 
                pde_batch['coords_physical']
            )

        return {
            'pde': {
                'coords_physical': pde_batch['coords_physical'],
                'model_coords': pde_batch['model_coords'],
                'predictions': pde_batch['predictions'],
                'gradients': gradients
            },
            'bc': {
                'coords_physical': bc_batch['coords_physical'],
                'predictions': bc_batch['predictions']
            },
            'sensors': {
                'coords_physical': sensors_batch['coords_physical'],
                'predictions': sensors_batch['predictions']
            },
            'is_vs_pinn': is_vs_pinn
        }

    def _transfer_batch_to_device(
        self,
        data_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        批量異步傳輸所有數據到 GPU（Wave 3 優化）
        
        優勢：
        1. 一次性處理所有張量，減少函數調用開銷
        2. 使用 non_blocking=True 允許 CPU 繼續執行
        3. CUDA 流可以並行處理多個傳輸
        
        Args:
            data_batch: 包含所有訓練數據的字典
            
        Returns:
            傳輸到 GPU 後的 data_batch
        """
        # 只對 CUDA 設備使用 non_blocking
        non_blocking = str(self.device).startswith('cuda')
        
        # 批量傳輸所有張量
        transferred_batch = {}
        for key, value in data_batch.items():
            if isinstance(value, torch.Tensor):
                if value.device == self.device:
                    transferred_batch[key] = value
                else:
                    transferred_batch[key] = value.to(self.device, non_blocking=non_blocking)
            else:
                transferred_batch[key] = value
        
        return transferred_batch
    
    def _process_point_batch(
        self,
        spatial_key: str,
        time_key: str,
        data_batch: Dict[str, torch.Tensor],
        require_grad: bool,
        context: str
    ) -> Dict[str, torch.Tensor]:
        """
        處理單一類型的點批次（座標 + 前向 + 反標準化）

        統一的處理流程：
        1. 提取空間座標 + 時間座標
        2. 拼接時間維度（如果需要）
        3. 標準化處理
        4. 模型前向傳播
        5. 反標準化到物理空間

        Args:
            spatial_key: 空間座標鍵名（如 'coords_pde_spatial'）
            time_key: 時間座標鍵名（如 't_pde'）
            data_batch: 訓練資料批次
            require_grad: 是否需要梯度
            context: 上下文名稱（'pde', 'bc', 'sensors'）

        Returns:
            {
                'coords_physical': 物理空間座標,
                'model_coords': 模型輸入座標,
                'predictions': 物理空間預測值
            }
        """
        # 檢查必要鍵
        if spatial_key not in data_batch:
            raise KeyError(f"data_batch 缺少必要鍵: {spatial_key}")

        # 🚀 優化: 數據已在 step() 開頭批量傳輸，這裡直接使用
        # 不再需要 .to(device) 調用
        non_blocking = True if str(self.device).startswith('cuda') else False
        
        # 提取空間座標（已在 GPU 上）
        coords_spatial = data_batch[spatial_key]
        if require_grad:
            coords_spatial = coords_spatial.requires_grad_(True)

        # 提取時間座標（如果存在，已在 GPU 上）
        t_coords = data_batch.get(time_key)
        if t_coords is not None:
            if require_grad:
                t_coords = t_coords.requires_grad_(True)

        # 拼接時間維度
        if t_coords is not None and self.model_input_dim > coords_spatial.shape[1]:
            coords_full = torch.cat([coords_spatial, t_coords], dim=1)
        else:
            coords_full = coords_spatial

        # 座標預處理與標準化
        is_vs_pinn = data_batch['coords_pde_spatial'].shape[1] >= 3 and hasattr(self.physics, 'compute_momentum_residuals')
        coords_physical, coords_norm, model_coords = self._prepare_model_coords(
            coords_full, require_grad=require_grad, is_vs_pinn=is_vs_pinn
        )

        # 前向傳播
        predictions_norm = self.model(model_coords)
        
        # 應用 hard constraint（如果配置）
        if hasattr(self, 'hard_constraint_applicator') and self.hard_constraint_applicator is not None:
            # Hard constraint 應該在標準化輸出上應用
            # 這確保距離函數正確作用於網路原始輸出
            predictions_norm = self.hard_constraint_applicator.apply(
                coords_physical,  # 使用物理座標（包含真實的 y）
                predictions_norm
            )

        # 反標準化
        var_order = self._infer_variable_order(predictions_norm.shape[1], context=context, data_batch=data_batch if context == 'sensors' else None)
        predictions_phys_raw = self.data_normalizer.denormalize_batch(predictions_norm, var_order=var_order)

        # 確保張量格式
        predictions_phys = (
            predictions_phys_raw if isinstance(predictions_phys_raw, torch.Tensor)
            else torch.tensor(predictions_phys_raw, device=self.device)
        )

        return {
            'coords_physical': coords_physical,
            'model_coords': model_coords,
            'predictions': predictions_phys
        }

    def _compute_vs_pinn_gradients(
        self,
        predictions: torch.Tensor,
        coords_physical: torch.Tensor,
        is_vs_pinn: bool
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        計算 VS-PINN 梯度緩存（Wave 2 優化）

        Args:
            predictions: 預測值 [N, n_vars]
            coords_physical: 物理空間座標
            is_vs_pinn: 是否為 VS-PINN

        Returns:
            梯度字典（VS-PINN 3D）或 None
        """
        if not is_vs_pinn:
            return None

        from pinnx.physics.gradient_cache import GradientCache

        # 建立梯度緩存
        grad_cache = GradientCache(device=self.device)

        # 構建預測字典
        n_vars = predictions.shape[1]
        if n_vars == 3:
            # [u, v, p] → 添加 w=0
            predictions_dict = {
                'u': predictions[:, 0:1],
                'v': predictions[:, 1:2],
                'w': torch.zeros_like(predictions[:, 2:3]),
                'p': predictions[:, 2:3]
            }
        elif n_vars == 4:
            # [u, v, w, p]
            predictions_dict = {
                'u': predictions[:, 0:1],
                'v': predictions[:, 1:2],
                'w': predictions[:, 2:3],
                'p': predictions[:, 3:4]
            }
        else:
            raise ValueError(f"VS-PINN 預測維度錯誤: {n_vars}，期望 3 或 4")

        # 計算所有梯度（一次計算，多次使用）
        return grad_cache.compute_all_gradients(predictions_dict, coords_physical)
    
    def _compute_2d_gradients(
        self,
        predictions: torch.Tensor,
        coords_physical: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        計算 2D 流場梯度緩存（Kolmogorov Flow）

        Args:
            predictions: 預測值 [N, 3] = [u, v, p]
            coords_physical: 物理空間座標 [N, 2] 或 [N, 3]
                - [N, 2]: 純 2D 穩態場景 (x, y)
                - [N, 3]: Time Window 場景 (x, y, t)

        Returns:
            梯度字典 (包含 u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, p_x, p_y)
            
        Note:
            GradientCache2D 會自動處理 2D/3D 座標，始終返回空間梯度 (∂/∂x, ∂/∂y)
        """
        from pinnx.physics.gradient_cache_2d import GradientCache2D

        # 建立 2D 梯度緩存
        grad_cache = GradientCache2D(device=self.device)

        # 構建預測字典
        n_vars = predictions.shape[1]
        if n_vars == 3:
            # [u, v, p]
            predictions_dict = {
                'u': predictions[:, 0:1],
                'v': predictions[:, 1:2],
                'p': predictions[:, 2:3]
            }
        else:
            raise ValueError(f"2D 流場預測維度錯誤: {n_vars}，期望 3 (u, v, p)")

        # 🚀 直接傳入完整座標，GradientCache2D 會自動處理維度
        # - 如果 coords_physical 是 [N, 2]，直接計算
        # - 如果 coords_physical 是 [N, 3]，自動只計算空間導數 (∂/∂x, ∂/∂y)
        return grad_cache.compute_all_gradients(predictions_dict, coords_physical)

    def _compute_all_losses(
        self,
        predictions: Dict[str, Any],
        data_batch: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        計算所有損失項（委派給 LossManager）

        Args:
            predictions: 前向傳播結果
            data_batch: 訓練資料批次
            epoch: 當前 epoch

        Returns:
            所有損失項的字典
        """
        # PDE 損失
        pde_losses = self.loss_manager.compute_pde_loss(
            coords_pde_physical=predictions['pde']['coords_physical'],
            model_coords_pde=predictions['pde']['model_coords'],
            u_pred_pde_physical=predictions['pde']['predictions'],
            data_batch=data_batch,
            epoch=epoch,
            is_vs_pinn=predictions['is_vs_pinn'],
            gradients=predictions['pde']['gradients']
        )

        # 邊界條件損失
        bc_losses = self.loss_manager.compute_bc_loss(
            u_bc_pred_phys=predictions['bc']['predictions'],
            data_batch=data_batch,
            epoch=epoch
        )

        # 資料監督損失
        data_losses = self.loss_manager.compute_data_loss(
            u_sensors_pred_phys=predictions['sensors']['predictions'],
            data_batch=data_batch
        )

        # 低保真先驗損失
        prior_losses = self.loss_manager.compute_lowfi_prior_loss(
            u_pred_pde_physical=predictions['pde']['predictions'],
            coords_pde_physical=predictions['pde']['coords_physical'],
            data_batch=data_batch,
            epoch=epoch
        )

        # 均值約束損失
        mean_constraint_loss = self.loss_manager.compute_mean_constraint_loss(
            u_pred_pde_physical=predictions['pde']['predictions'],
            epoch=epoch
        )

        # 合併所有損失
        all_losses = {**pde_losses, **bc_losses, **data_losses, **prior_losses}
        all_losses['mean_constraint_loss'] = mean_constraint_loss

        return all_losses

    def _combine_and_weight_losses(
        self,
        losses: Dict[str, torch.Tensor],
        is_vs_pinn: bool,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        應用動態權重並組合損失

        Args:
            losses: 所有損失項
            is_vs_pinn: 是否為 VS-PINN
            epoch: 當前 epoch

        Returns:
            (total_loss, result_dict)
        """
        # 課程學習權重
        curriculum_config, loss_cfg = self.loss_manager.apply_curriculum_weights(epoch)

        # 構建損失項字典（供 GradNorm 使用）
        loss_terms = {
            'data': losses['data_loss'],
            'momentum_x': losses['momentum_x_loss'],
            'momentum_y': losses['momentum_y_loss'],
            'continuity': losses['continuity_loss'],
        }

        # 邊界條件損失項
        if hasattr(self.physics, 'compute_periodic_loss'):
            loss_terms['periodic_x'] = losses['periodic_x_loss']
            loss_terms['periodic_y'] = losses['periodic_y_loss']
        else:
            loss_terms['wall_constraint'] = losses['wall_loss']

        # VS-PINN 額外的 z 方向動量
        if is_vs_pinn:
            loss_terms['momentum_z'] = losses['momentum_z_loss']

        # 🔧 FIX: 添加先驗一致性損失到 GradNorm（避免 .item() 同步）
        if 'prior_consistency_loss' in losses:
            prior_loss_val = losses['prior_consistency_loss']
            if isinstance(prior_loss_val, torch.Tensor):
                loss_terms['prior'] = prior_loss_val
        
        # 🔧 FIX: 添加初始條件損失到 GradNorm（避免 .item() 同步）
        # 註：目前 loss_manager 可能尚未實現 initial_condition_loss，預留接口
        if 'initial_condition_loss' in losses:
            ic_loss_val = losses['initial_condition_loss']
            if isinstance(ic_loss_val, torch.Tensor):
                loss_terms['initial_condition'] = ic_loss_val

        # GradNorm 動態權重
        gradnorm_weights, gradnorm_ratio = self.loss_manager.apply_gradnorm_weights(loss_terms)

        # 組合損失
        total_loss, result = self.loss_manager.combine_losses(
            loss_dict=losses,
            loss_cfg=loss_cfg,
            gradnorm_ratio=gradnorm_ratio,
            is_vs_pinn=is_vs_pinn,
            epoch=epoch
        )

        # 保存權重資訊
        result['_curriculum_config'] = curriculum_config
        result['_gradnorm_weights'] = gradnorm_weights

        return total_loss, result

    def _backward_and_optimize(
        self,
        total_loss: torch.Tensor,
        data_batch: Dict[str, torch.Tensor],
        epoch: int
    ):
        """
        反向傳播、梯度裁剪、優化器步進

        Args:
            total_loss: 總損失
            data_batch: 當前批次數據（L-BFGS closure 需要）
            epoch: 當前 epoch（L-BFGS closure 需要）
        """
        # L-BFGS 特殊處理：需要 closure 重新計算損失
        if isinstance(self.optimizer, torch.optim.LBFGS):
            def closure():
                self.optimizer.zero_grad()

                # 重新前向傳播
                predictions = self._forward_pass_all_points(data_batch)

                # 重新計算損失
                losses = self._compute_all_losses(predictions, data_batch, epoch)

                # 重新組合損失
                total_loss_new, _ = self._combine_and_weight_losses(
                    losses, predictions['is_vs_pinn'], epoch
                )

                # 反向傳播
                total_loss_new.backward()

                return total_loss_new

            # L-BFGS 優化步進
            self.optimizer.step(closure)

        else:
            # 一階優化器（Adam, AdamW, SOAP, etc.）
            total_loss.backward()

            # 梯度裁剪
            if self.train_cfg.get('gradient_clip', 0.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg['gradient_clip']
                )

            # 優化器步進
            self.optimizer.step()

        # 學習率調度器更新
        if self.lr_scheduler is not None and hasattr(self.lr_scheduler, 'current_step'):
            self.lr_scheduler.step()

    def _add_training_metadata(
        self,
        result: Dict[str, Any],
        losses: Dict[str, torch.Tensor],
        epoch: int
    ):
        """
        附加訓練元數據到結果字典

        Args:
            result: 結果字典（會被原地修改）
            losses: 損失字典
            epoch: 當前 epoch
        """
        # 課程學習資訊
        curriculum_config = result.pop('_curriculum_config', None)
        if curriculum_config is not None:
            is_curriculum_transition = curriculum_config.get('is_transition', False)
            result['_curriculum_transition'] = 1.0 if is_curriculum_transition else 0.0
            result['_curriculum_stage'] = curriculum_config.get('stage_name', 'unknown')
            if 'lr' in curriculum_config:
                result['_curriculum_lr'] = curriculum_config['lr']

        # GradNorm 權重資訊
        gradnorm_weights = result.pop('_gradnorm_weights', None)
        if gradnorm_weights is not None:
            result['gradnorm_weights'] = {k: float(v) for k, v in gradnorm_weights.items()}

    # ========================================================================
    # 驗證方法
    # ========================================================================

    def validate(self) -> Optional[Dict[str, float]]:
        """
        計算驗證指標（委託給 ValidationManager）

        Returns:
            驗證指標字典，若無驗證策略則返回 None
        """
        # 檢查是否有驗證策略
        if not self.validation_manager.has_strategies():
            return None

        # DDP 下只在主程序執行驗證，避免 collective mismatch
        if self._is_ddp_enabled() and not self._is_main_process():
            return None

        # 保存訓練狀態
        training_mode = self.model.training
        model_to_validate = self.model.module if hasattr(self.model, 'module') else self.model

        try:
            # 委託給 ValidationManager 執行所有驗證策略
            results = self.validation_manager.validate(
                model=model_to_validate,
                device=self.device,
                validation_data=self.validation_data,
                is_3d=(self.model_input_dim == 3),
                log_to_wandb=self.use_wandb
            )

            return results if results else None

        finally:
            # 恢復訓練狀態
            if training_mode:
                self.model.train()
            else:
                self.model.eval()
    
    # ========================================================================
    # 訓練循環
    # ========================================================================
    
    def train(self) -> Dict[str, Any]:
        """
        執行完整訓練循環（Phase 2 重構版）
        
        Returns:
            訓練歷史與最終結果字典
            - 'final_loss': 最終訓練損失
            - 'training_time': 訓練總時間（秒）
            - 'epochs_completed': 完成的 epoch 數
            - 'best_epoch': 最佳模型的 epoch（若啟用早停）
            - 'best_metric': 最佳指標值（若啟用早停）
            - 'history': 訓練歷史（每 epoch 的損失）
        """
        from pinnx.train.training_loop_manager import TrainingLoopManager
        
        # === 初始化訓練配置 ===
        # 提取訓練配置
        max_epochs = self.train_cfg.get('epochs', 1000)
        log_freq = self.log_cfg.get('log_freq', 10)
        wandb_log_freq = self.log_cfg.get('wandb_log_freq', log_freq)  # WandB 預設跟 log_freq，一般設定為 10
        checkpoint_freq = self.train_cfg.get('checkpoint', {}).get('save_every', 100)
        validation_freq = self.train_cfg.get('validation', {}).get('val_freq', 50)
        loop_helper = TrainingLoopManager(self.config, self.wandb_run, step_offset=self.step_offset)  # ✅ 傳遞 step_offset
        start_time = time.time()
        start_epoch = self.epoch

        logging.info("=" * 80)
        logging.info("🚀 開始訓練")
        logging.info(f"   模型: {self.model.__class__.__name__}")
        logging.info(f"   優化器: {self.optimizer.__class__.__name__}")
        logging.info(f"   起始 Epoch: {start_epoch}")
        logging.info(f"   最大 Epochs: {max_epochs}")
        logging.info(f"   早停: {'啟用' if self.early_stopping_enabled else '禁用'}")
        logging.info("=" * 80)

        # 🆕 啟動 Timer
        self.timer.start()
        
        # 初始化損失字典（防止 epoch=0 時未定義）
        loss_dict = {'total_loss': 0.0, 'residual_loss': 0.0, 'bc_loss': 0.0, 'data_loss': 0.0}
        
        if start_epoch > 0:
            logging.info(f"🔄 從 epoch {start_epoch} 恢復訓練")
        
        # === 訓練循環 ===
        if os.environ.get('PINNX_DETECT_ANOMALY', '0') == '1':
            torch.autograd.set_detect_anomaly(True)
            logging.warning("⚠️  已啟用 Autograd Anomaly Detection（僅供除錯）")

        for epoch in range(start_epoch, max_epochs):

            self.epoch = epoch
            
            # 1. 自適應更新（adaptive sampling + fourier annealing）
            self.training_data = loop_helper.coordinate_adaptive_updates(
                epoch, self.adaptive_sampler, self.fourier_annealing,
                self.model, self.physics, self.training_data, self.config,
                self.device, loop_helper.get_history()
            )
            
            # 2. DDP 數據分割（必要時）
            data_batch = self.training_data
            if self._is_ddp_enabled() and self._should_split_training_data():
                data_batch = self._split_training_data(data_batch)
            
            # 3. 執行訓練步驟
            loss_dict = self.step(data_batch, epoch)
            
            # 3. 驗證指標計算
            if validation_freq > 0 and epoch % validation_freq == 0:
                val_metrics = self.validate()
                if val_metrics is not None:
                    # 使用 .get() 以支持不同格式的驗證結果
                    loss_dict['val_loss'] = val_metrics.get('val_loss', val_metrics.get('relative_l2', 0.0))
                    loss_dict['val_mse'] = val_metrics.get('val_mse', val_metrics.get('mse', 0.0))

                # 🆕 物理驗證（降低頻率：每隔 5 次驗證才做一次物理驗證）
                physics_validation_freq = validation_freq * 5
                if epoch % physics_validation_freq == 0 and self.validation_data is not None:
                    try:
                        physics_results = self._run_physics_validation()
                        if physics_results is not None:
                            # 記錄物理違反分數
                            loss_dict['physics_violation_score'] = physics_results['physics_violation_score']
                            logging.info(f"📊 物理驗證（epoch {epoch}）: 違反分數 = {physics_results['physics_violation_score']:.6e}")
                    except Exception as e:
                        logging.warning(f"⚠️  物理驗證失敗（epoch {epoch}）: {e}")
            
            # 4. 🆕 記錄 epoch 時間與記憶體
            self.timer.tick(epoch=epoch, log_to_wandb=self.use_wandb)
            self.memory_tracker.log_current(epoch=epoch, log_to_wandb=self.use_wandb)

            # 5. 記錄歷史與 WandB 日誌
            loop_helper.update_history(loss_dict, epoch)
            if epoch % wandb_log_freq == 0 and self.use_wandb:
                loop_helper.log_losses_to_wandb(loss_dict, epoch)
                loop_helper.log_hyperparameters(self.get_current_lr(), epoch)
                if self.log_cfg.get('log_nonlinearities', False):
                    loop_helper.log_nonlinearities(self.model, epoch)
            if epoch % log_freq == 0:
                self.log_epoch(epoch, loss_dict)
            
            # 6. 記錄梯度與權重統計（降低頻率）
            # 註：目前不需要追蹤 layer gradient，已停用以減少 WandB 數據量
            # if epoch % (log_freq * 2) == 0:
            #     loop_helper.log_gradients_and_weights(self.model, epoch)

            # 7. 更新全局步數
            self.global_step += 1

            # 8. 課程訓練 LR 控制
            self._handle_curriculum_lr(loss_dict)

            # 9. 學習率調度
            self._update_lr_scheduler(loss_dict)

            # 10. 檢查點保存
            if checkpoint_freq > 0 and epoch % checkpoint_freq == 0 and epoch > 0:
                self.save_checkpoint(epoch, loss_dict)
                logging.info(f"💾 檢查點已保存（epoch {epoch}）")

            # 11. 早停檢查
            if self._check_and_handle_early_stopping(loss_dict, epoch):
                break

            # 12. 快速收斂檢查（達標時記錄）
            if self._check_convergence(loss_dict, epoch):
                # 🆕 標記達標時間
                metric_value = loss_dict.get('val_loss', loss_dict.get('total_loss', 0.0))
                self.timer.mark_convergence(epoch=epoch, metric_value=metric_value, log_to_wandb=self.use_wandb)
                break
        
        # === 訓練結束 ===
        final_epoch = epoch if 'epoch' in locals() else max_epochs - 1
        return self._finalize_training(final_epoch, loss_dict, start_time, loop_helper)
    
    # ========================================================================
    # 訓練循環輔助方法（Phase 2 重構）
    # ========================================================================
    
    
    def _handle_curriculum_lr(self, loss_dict: Dict):
        """
        處理課程訓練學習率控制
        
        Args:
            loss_dict: 損失字典（可能包含 _curriculum_lr 和 _curriculum_transition）
        """
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
            stage_name = loss_dict.get('_curriculum_stage', f'stage_{self.epoch}')
            logging.info(f"📉 課程階段切換: {stage_name}")
            if '_curriculum_lr' in loss_dict:
                logging.info(f"   學習率強制設置為: {loss_dict['_curriculum_lr']:.2e}")
            
            # 保存階段檢查點（如果啟用）
            if self.log_cfg.get('save_stage_checkpoints', False):
                self.save_checkpoint(self.epoch, loss_dict, is_best=False)
                logging.info(f"💾 階段檢查點已保存: {stage_name}")
    
    def _update_lr_scheduler(self, loss_dict: Dict):
        """
        更新學習率調度器
        
        Args:
            loss_dict: 損失字典
        """
        if self.lr_scheduler is None:
            return
        
        # 課程訓練時：只在非切換 epoch 更新 scheduler
        if hasattr(self, 'curriculum_weighter'):
            is_transition = '_curriculum_transition' in loss_dict and loss_dict['_curriculum_transition'] > 0.5
            if not is_transition:
                self.lr_scheduler.step()
        else:
            # 非課程訓練：正常更新
            self.lr_scheduler.step()
    
    def _check_and_handle_early_stopping(self, loss_dict: Dict, epoch: int) -> bool:
        """
        檢查並處理早停邏輯

        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch

        Returns:
            True 表示應該停止訓練
        """
        if not self.early_stopping_enabled:
            return False

        # 選擇監控指標
        metric_name = self.early_stopping_cfg.get('monitor', 'total_loss')
        if metric_name == 'val_loss' and 'val_loss' in loss_dict:
            current_metric = loss_dict['val_loss']
        elif metric_name in loss_dict:
            current_metric = loss_dict[metric_name]
        else:
            current_metric = loss_dict['total_loss']
        
        current_metric = self._to_scalar(current_metric)
        
        # 檢查是否應該停止
        if self.check_early_stopping(current_metric):
            logging.info(f"🛑 早停觸發於 epoch {epoch}")
            logging.info(f"   最佳指標: {self.best_val_loss:.6f}（epoch {self.best_epoch}）")
            
            # 恢復最佳模型（如果啟用）
            if self.early_stopping_cfg.get('restore_best_weights', True) and self.best_model_state is not None:
                self.model.load_state_dict(self.best_model_state)
                logging.info(f"✅ 已恢復最佳模型（epoch {self.best_epoch}）")
            
            return True
        return False
    
    def _check_convergence(self, loss_dict: Dict, epoch: int) -> bool:
        """
        檢查快速收斂條件
        
        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch
        
        Returns:
            True 表示已達收斂條件
        """
        total_loss = self._to_scalar(loss_dict['total_loss'])
        if self.convergence_threshold is not None and total_loss < self.convergence_threshold:
            logging.info(f"✅ 快速收斂於 epoch {epoch}（loss < {self.convergence_threshold:.2e}）")
            return True
        return False
    
    def _finalize_training(
        self,
        final_epoch: int,
        loss_dict: Dict,
        start_time: float,
        loop_helper: 'TrainingLoopManager'
    ) -> Dict[str, Any]:
        """
        訓練結束處理
        
        Args:
            final_epoch: 最終 epoch
            loss_dict: 最終損失字典
            start_time: 訓練開始時間戳
            loop_helper: 訓練循環管理器
        
        Returns:
            訓練結果字典
        """
        total_time = time.time() - start_time
        final_loss = self._to_scalar(loss_dict['total_loss'])

        # 🆕 停止 Timer 並打印摘要
        self.timer.stop()

        logging.info("=" * 80)
        logging.info(f"✅ 訓練完成")
        logging.info(f"   總時間: {total_time:.1f}s")
        logging.info(f"   完成 Epochs: {final_epoch + 1}")
        logging.info(f"   最終損失: {final_loss:.6f}")
        if self.early_stopping_enabled and self.best_epoch >= 0:
            logging.info(f"   最佳 Epoch: {self.best_epoch}")
            logging.info(f"   最佳指標: {self.best_val_loss:.6f}")
        logging.info("=" * 80)

        # 🆕 打印指標追蹤摘要
        self.timer.print_summary()
        self.memory_tracker.print_summary()
        
        # 保存最終檢查點
        final_checkpoint = self.save_checkpoint(final_epoch + 1, loss_dict, is_best=False)
        logging.info(f"💾 最終模型已保存")
        
        # 關閉 WandB
        hparams = self._build_hparams_dict()
        metrics = {
            'hparam/final_loss': final_loss,
            'hparam/best_loss': self.best_val_loss if self.early_stopping_enabled else final_loss,
            'hparam/epochs': final_epoch + 1,
        }
        loop_helper.finalize_wandb(metrics, hparams)
        
        return {
            'final_loss': final_loss,
            'training_time': total_time,
            'epochs_completed': final_epoch + 1,
            'best_epoch': self.best_epoch if self.early_stopping_enabled else final_epoch,
            'best_metric': self.best_val_loss if self.early_stopping_enabled else final_loss,
            'history': loop_helper.get_history(),
            'checkpoint_path': final_checkpoint
        }
    
    def _build_hparams_dict(self) -> Dict:
        """
        構建超參數字典
        
        Returns:
            超參數字典
        """
        return {
            'lr': self.train_cfg.get('lr', 1e-3),
            'optimizer': self.optimizer.__class__.__name__,
            'model_width': self.config.get('model', {}).get('width', 256),
            'model_depth': self.config.get('model', {}).get('depth', 8),
            'activation': self.config.get('model', {}).get('activation', 'sine'),
            'K_sensors': self.config.get('sensors', {}).get('K', 0),
        }
    
    # ========================================================================
    # 檢查點管理
    # ========================================================================
    
    
    # ========================================================================
    # DDP 輔助方法
    # ========================================================================
    
    def _is_main_process(self) -> bool:
        """
        判斷是否為主程序（rank 0）
        
        Returns:
            True 如果是主程序或非分散式環境
        """
        import torch.distributed as dist
        
        # 非分散式環境，視為主程序
        if not dist.is_available() or not dist.is_initialized():
            return True
        
        # 分散式環境，檢查 rank
        return dist.get_rank() == 0

    def _is_ddp_enabled(self) -> bool:
        """判斷是否啟用 DDP"""
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return False

        ddp_cfg = self.train_cfg.get('ddp', {})
        enabled = ddp_cfg.get('enabled')
        if enabled is False:
            return False

        return True

    def _should_split_training_data(self) -> bool:
        """判斷是否需要分割訓練資料"""
        ddp_cfg = self.train_cfg.get('ddp', {})
        return bool(ddp_cfg.get('split_data', True))

    def _should_reduce_losses(self) -> bool:
        """判斷是否需要同步損失"""
        ddp_cfg = self.train_cfg.get('ddp', {})
        return bool(ddp_cfg.get('reduce_losses', True))

    def _split_training_data(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """執行 DDP 訓練資料分割"""
        import torch.distributed as dist
        from pinnx.utils.ddp_utils import split_data_by_rank, verify_data_split

        ddp_cfg = self.train_cfg.get('ddp', {})
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        split_data = split_data_by_rank(training_data, rank=rank, world_size=world_size)

        if ddp_cfg.get('verify_data_split', False):
            verify_data_split(split_data, rank=rank, world_size=world_size)

        return split_data
    
    # ========================================================================
    # 配置解析輔助方法
    # ========================================================================
    
    def _parse_domain_from_config(self) -> Dict[str, float]:
        """
        從配置中解析 domain 參數（Fail Fast，只支援 physics.domain）
        
        要求格式:
            physics:
              domain:
                x_range: [min, max]  # 必填
                y_range: [min, max]  # 必填
                z_range: [min, max]  # 選填（3D 必填）
        
        Returns:
            domain: {'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'}
        
        Raises:
            KeyError: 配置中缺少 physics.domain
            ValueError: domain 格式不正確
        
        Note:
            P0-2 重構：移除所有 fallback 邏輯，強制使用 physics.domain
            若配置文件為舊格式，請執行：
                python scripts/tools/migrate_domain_config.py --config <file>
        """
        # Fail Fast: 只接受 physics.domain
        physics_config = self.config.get('physics', {})
        domain_data = physics_config.get('domain')
        
        if domain_data is None:
            raise KeyError(
                "Config missing required key: physics.domain\n"
                "Expected format:\n"
                "  physics:\n"
                "    domain:\n"
                "      x_range: [min, max]\n"
                "      y_range: [min, max]\n"
                "\n"
                "To migrate old configs, run:\n"
                "  python scripts/tools/migrate_domain_config.py --config <file>"
            )
        
        # 驗證格式
        if 'x_range' not in domain_data or 'y_range' not in domain_data:
            raise ValueError(
                f"Invalid physics.domain format: {domain_data}\n"
                "Required keys: x_range, y_range\n"
                "Optional keys: z_range"
            )
        
        # 解析 domain
        domain = {
            'x_min': domain_data['x_range'][0],
            'x_max': domain_data['x_range'][1],
            'y_min': domain_data['y_range'][0],
            'y_max': domain_data['y_range'][1],
            'z_min': domain_data.get('z_range', [0, 1])[0],
            'z_max': domain_data.get('z_range', [0, 1])[1],
        }
        
        return domain
    
    def _generate_validation_coords(self, domain: Dict[str, float]) -> Optional[torch.Tensor]:
        """
        根據 domain 和維度生成驗證網格座標（Phase 4-1 Helper）
        
        Args:
            domain: 包含 x_min, x_max, y_min, y_max, z_min, z_max 的字典
        
        Returns:
            validation_coords: (N, dim) 的 torch.Tensor，或 None（未知維度）
        """
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
        
        return validation_coords
    
    def _run_physics_validation_before_save(self, validation_coords: Optional[torch.Tensor]) -> Dict[str, Any]:
        """
        執行物理驗證，處理 strict mode 和 trivial solution（Phase 4-1 Helper）
        
        Args:
            validation_coords: 驗證座標（N, dim），或 None
        
        Returns:
            physics_metrics: 物理驗證指標字典
        
        Side Effects:
            - 可能提前返回（early return via exception），當 strict_mode=True 且檢測到 trivial solution
        """
        from pinnx.train.checkpointing import validate_physics_before_save
        
        physics_metrics = {}
        model_to_validate = self.model.module if hasattr(self.model, 'module') else self.model
        
        if validation_coords is not None:
            validation_passed, physics_metrics = validate_physics_before_save(
                model_to_validate,
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
                        # 使用 exception 模擬 early return（避免直接 return，讓調用者處理）
                        raise RuntimeError("Physics validation failed: Trivial solution detected in strict mode")
                
                # 其他情況：物理約束未滿足（訓練初期正常）
                logging.info("ℹ️  物理診斷完成，指標已記錄至檢查點元數據")
                # 繼續保存，讓使用者根據診斷資訊判斷
        
        return physics_metrics

    def save_checkpoint(
        self,
        epoch: int,
        metrics: Optional[Dict[str, float]] = None,
        is_best: bool = False
    ) -> Optional[str]:
        """
        保存檢查點（委託給 CheckpointManager）
        
        ⚠️  在 DDP 模式下，只有 rank 0 會執行保存

        Args:
            epoch: 當前 epoch
            metrics: 評估指標（可選）
            is_best: 是否為最佳模型

        Returns:
            保存的文件路徑，如果未保存則返回 None（非 rank 0 或保存失敗）
        """
        # ========== 🚀 NEW: DDP 檢查 ==========
        if not self._is_main_process():
            # 非主程序跳過保存
            return None
        
        try:
            # 1. 執行物理驗證（如果配置了 physics）
            physics_metrics = {}
            if self.physics is not None and 'physics' in self.config and 'domain' in self.config['physics']:
                domain = self._parse_domain_from_config()
                validation_coords = self._generate_validation_coords(domain)
                physics_metrics = self._run_physics_validation_before_save(validation_coords)

            # 2. 準備額外數據
            extra_data = {
                'physics_validation': physics_metrics,
                'normalizers': self.data_normalizer.get_metadata(),
            }

            # 保存 physics state_dict（VS-PINN 縮放參數等）
            if self.physics is not None and hasattr(self.physics, 'state_dict'):
                extra_data['physics_state_dict'] = self.physics.state_dict()

            # 保存 learning rate scheduler
            if self.lr_scheduler:
                extra_data['scheduler_state'] = self.lr_scheduler.state_dict()

            # 保存訓練歷史
            extra_data['history'] = self.history

            # ========== 🚀 NEW: 處理 DDP 模型的 state_dict ==========
            # DDP 包裝後，模型的參數在 module 屬性中
            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model

            # 3. 委託給 CheckpointManager 保存
            filepath = self.checkpoint_manager.save(
                model=model_to_save,  # 使用處理後的模型
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=metrics or {},
                config=self.config,
                **extra_data
            )

            # 4. 向後兼容：如果是最佳模型，先保存為 best_model.pth
            #    （必須在 update_best_checkpoints 之前，因為後者可能刪除 filepath）
            if is_best:
                best_model_path = self.checkpoint_dir / "best_model.pth"
                import shutil
                shutil.copy2(filepath, best_model_path)
                logging.info(f"⭐ 最佳模型已保存: {best_model_path}")

            # 5. 更新最佳 checkpoint 記錄（可能刪除舊檢查點）
            if is_best and metrics:
                metric_value = metrics.get(self.checkpoint_strategy.metric_name, float('inf'))
                self.checkpoint_strategy.update_best_checkpoints(metric_value, filepath)

            return filepath

        except RuntimeError as e:
            # 處理 strict mode 拒絕保存的情況（來自 _run_physics_validation_before_save）
            if "Physics validation failed" in str(e):
                logging.warning("檢查點保存被中止（physics validation failed）")
                return None
            else:
                raise  # 其他 RuntimeError 繼續拋出
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        載入檢查點（委託給 CheckpointManager）

        Args:
            checkpoint_path: 檢查點路徑
        """
        # 1. 委託給 CheckpointManager 加載基本數據
        checkpoint_data = self.checkpoint_manager.load(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer
        )

        # 2. 恢復訓練狀態
        self.epoch = checkpoint_data['epoch']
        self.history = checkpoint_data.get('history', self.history)

        # 3. 恢復 physics state（VS-PINN 縮放參數等）
        if self.physics is not None and 'physics_state_dict' in checkpoint_data:
            if not hasattr(self.physics, 'load_state_dict'):
                raise AttributeError("Physics module does not support load_state_dict()")
            self.physics.load_state_dict(checkpoint_data['physics_state_dict'])
            logging.info(f"✅ Physics state restored: {list(checkpoint_data['physics_state_dict'].keys())}")

        # 4. 恢復標準化器
        if 'normalizers' in checkpoint_data:
            self.data_normalizer = OutputTransform.from_metadata(checkpoint_data['normalizers'])
            logging.info(f"✅ OutputTransform restored: {self.data_normalizer}")

        # 5. 恢復 learning rate scheduler
        if self.lr_scheduler and 'scheduler_state' in checkpoint_data:
            self.lr_scheduler.load_state_dict(checkpoint_data['scheduler_state'])

        logging.info(f"✅ 檢查點已載入: {checkpoint_path}（epoch={self.epoch}）")
    
    def check_early_stopping(self, val_loss: float) -> bool:
        """
        檢查是否應該早停（基於目標值）
        
        機制：當 loss 達到或低於 convergence_threshold 時停止訓練
        
        Args:
            val_loss: 驗證損失
        
        Returns:
            是否應該停止訓練
        """
        if not self.early_stopping_enabled:
            return False
        
        # 檢查是否達到目標值
        convergence_threshold = self.early_stopping_cfg.get('convergence_threshold', None)
        if convergence_threshold is None:
            # 如果未設定目標值，不觸發早停（訓練至 max_epochs）
            return False
        
        # 更新最佳模型（無論是否達標）
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = self.epoch
            
            # 保存最佳模型狀態
            if self.early_stopping_cfg.get('restore_best_weights', True):
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            # 立即保存最佳模型到磁碟
            metrics = {'val_loss': val_loss, 'best_epoch': self.best_epoch}
            self.save_checkpoint(self.epoch, metrics, is_best=True)
            
            logging.info(f"🎯 新最佳指標: {self.best_val_loss:.6f}（epoch {self.best_epoch}）")
        
        # 檢查是否達到目標值
        if val_loss <= convergence_threshold:
            logging.info(f"🎯 達到目標值停止: {val_loss:.2e} <= {convergence_threshold:.2e}")
            logging.info(f"   訓練完成於 epoch {self.epoch}")
            return True
        
        return False
    
    def get_current_lr(self) -> float:
        """獲取當前學習率"""
        return self.optimizer.param_groups[0]['lr']

    @staticmethod
    def _to_scalar(value: Any):
        """安全轉換為 Python scalar"""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return value
    
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
            if isinstance(value, torch.Tensor):
                value = self._to_scalar(value)
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
            self.history[key].append(self._to_scalar(value))
        
        self.history['lr'].append(self.get_current_lr())
