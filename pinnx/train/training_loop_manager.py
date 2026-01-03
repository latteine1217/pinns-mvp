"""
訓練循環輔助管理器

負責訓練循環中的輔助功能：
- WandB 日誌記錄
- 訓練歷史管理  
- 自適應更新協調（adaptive sampling, fourier annealing）

不包含核心訓練邏輯（step, validate, checkpoint, early stopping），
這些保留在 Trainer 類中。
"""

import logging
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import wandb


class TrainingLoopManager:
    """
    訓練循環輔助管理器
    
    職責：
    1. WandB 日誌記錄（分類管理所有 scalar/histogram）
    2. 訓練歷史記錄與查詢
    3. 自適應更新協調（采樣+退火）
    
    不包含：
    - 訓練步驟執行（step）
    - 驗證邏輯（validate）
    - 檢查點保存（save_checkpoint）
    - 早停決策（check_early_stopping）
    """
    
    def __init__(self, config: Dict, wandb_run: Optional[Any]):
        """
        Args:
            config: 訓練配置字典
            wandb_run: WandB Run 實例（若為 None 則不記錄）
        """
        self.config = config
        self.wandb_run = wandb_run
        
        # 訓練歷史記錄
        self.history = {
            'total_loss': [],
            'val_loss': [],
            'epoch': []
        }
    
    # ========================================================================
    # 歷史記錄管理
    # ========================================================================
    
    def update_history(self, loss_dict: Dict, epoch: int):
        """
        更新訓練歷史
        
        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch
        """
        self.history['total_loss'].append(loss_dict['total_loss'])
        self.history['epoch'].append(epoch)
        if 'val_loss' in loss_dict:
            self.history['val_loss'].append(loss_dict['val_loss'])
    
    def get_history(self) -> Dict:
        """
        獲取訓練歷史
        
        Returns:
            訓練歷史字典
        """
        return self.history
    
    # ========================================================================
    # WandB 日誌記錄（分類管理）
    # ========================================================================
    
    def log_losses_to_wandb(self, loss_dict: Dict, epoch: int):
        """
        記錄所有損失到 WandB
        
        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch
        """
        if self.wandb_run is None:
            return
        
        # 準備日誌字典
        log_dict = {'epoch': epoch}
        
        # 1. 總損失與主要分量
        self._add_main_losses(log_dict, loss_dict)
        
        # 2. PDE 子項（動量方程與連續性方程）
        self._add_pde_losses(log_dict, loss_dict)
        
        # 3. 數據擬合損失（各變量）
        self._add_data_losses(log_dict, loss_dict)
        
        # 4. Weighted Loss（分析權重平衡）
        self._add_weighted_losses(log_dict, loss_dict)
        
        # 5. RANS Prior Loss（低保真先驗）
        self._add_prior_losses(log_dict, loss_dict)
        
        # 6. 邊界條件損失
        self._add_bc_losses(log_dict, loss_dict)
        
        # 7. 正則化項
        self._add_regularization_losses(log_dict, loss_dict)
        
        # 8. 驗證指標
        self._add_validation_metrics(log_dict, loss_dict)

        # 9. 損失權重（GradNorm 與應用權重）
        self._add_loss_weights(log_dict, loss_dict)

        # 一次性記錄所有指標
        wandb.log(log_dict)
    
    def _add_main_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄主要損失（total, data, pde, boundary）"""
        log_dict['Loss/total'] = loss_dict.get('total_loss', 0.0)
        log_dict['Loss/data'] = loss_dict.get('data_loss', 0.0)
        log_dict['Loss/pde'] = loss_dict.get('pde_loss', 0.0)
        log_dict['Loss/boundary'] = loss_dict.get('bc_loss', 0.0)
    
    def _add_pde_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄 PDE 子項（momentum, continuity）"""
        if 'momentum_x_loss' in loss_dict:
            log_dict['Loss/PDE/momentum_x'] = loss_dict['momentum_x_loss']
        if 'momentum_y_loss' in loss_dict:
            log_dict['Loss/PDE/momentum_y'] = loss_dict['momentum_y_loss']
        if 'momentum_z_loss' in loss_dict:
            log_dict['Loss/PDE/momentum_z'] = loss_dict['momentum_z_loss']
        if 'continuity_loss' in loss_dict:
            log_dict['Loss/PDE/continuity'] = loss_dict['continuity_loss']
        if 'div_loss' in loss_dict:
            log_dict['Loss/PDE/divergence'] = loss_dict['div_loss']
    
    def _add_data_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄數據擬合損失（u, v, w, pressure）"""
        if 'u_loss' in loss_dict:
            log_dict['Loss/Data/u'] = loss_dict['u_loss']
        if 'v_loss' in loss_dict:
            log_dict['Loss/Data/v'] = loss_dict['v_loss']
        if 'w_loss' in loss_dict:
            log_dict['Loss/Data/w'] = loss_dict['w_loss']
        if 'pressure_loss' in loss_dict:
            log_dict['Loss/Data/pressure'] = loss_dict['pressure_loss']
    
    def _add_weighted_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄加權後的損失（分析權重平衡）"""
        if 'weighted_data_loss' in loss_dict:
            log_dict['Loss/Weighted/data'] = loss_dict['weighted_data_loss']
        if 'weighted_pde_loss' in loss_dict:
            log_dict['Loss/Weighted/pde'] = loss_dict['weighted_pde_loss']
        if 'weighted_div_loss' in loss_dict:
            log_dict['Loss/Weighted/continuity'] = loss_dict['weighted_div_loss']
        if 'weighted_bc_loss' in loss_dict:
            log_dict['Loss/Weighted/boundary'] = loss_dict['weighted_bc_loss']
    
    def _add_prior_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄 RANS prior 損失"""
        if 'prior_consistency_loss' in loss_dict:
            log_dict['Loss/Prior/total'] = loss_dict['prior_consistency_loss']
        if 'prior_loss_u' in loss_dict:
            log_dict['Loss/Prior/u'] = loss_dict['prior_loss_u']
        if 'prior_loss_v' in loss_dict:
            log_dict['Loss/Prior/v'] = loss_dict['prior_loss_v']
        if 'prior_loss_p' in loss_dict:
            log_dict['Loss/Prior/p'] = loss_dict['prior_loss_p']
    
    def _add_bc_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄邊界條件損失"""
        if 'periodic_x_loss' in loss_dict:
            log_dict['Loss/BC/periodic_x'] = loss_dict['periodic_x_loss']
        if 'periodic_y_loss' in loss_dict:
            log_dict['Loss/BC/periodic_y'] = loss_dict['periodic_y_loss']
        if 'inlet_loss' in loss_dict:
            log_dict['Loss/BC/inlet'] = loss_dict['inlet_loss']
        if 'outlet_loss' in loss_dict:
            log_dict['Loss/BC/outlet'] = loss_dict['outlet_loss']
        if 'wall_loss' in loss_dict:
            log_dict['Loss/BC/wall'] = loss_dict['wall_loss']
    
    def _add_regularization_losses(self, log_dict: Dict, loss_dict: Dict):
        """記錄正則化損失"""
        if 'regularization_loss' in loss_dict:
            log_dict['Loss/Regularization/total'] = loss_dict['regularization_loss']
        if 'l2_reg' in loss_dict:
            log_dict['Loss/Regularization/l2'] = loss_dict['l2_reg']
        if 'gradient_penalty' in loss_dict:
            log_dict['Loss/Regularization/gradient'] = loss_dict['gradient_penalty']
    
    def _add_validation_metrics(self, log_dict: Dict, loss_dict: Dict):
        """記錄驗證指標"""
        if 'val_loss' in loss_dict:
            log_dict['Validation/relative_l2'] = loss_dict['val_loss']
        if 'val_mse' in loss_dict:
            log_dict['Validation/mse'] = loss_dict['val_mse']

    def _add_loss_weights(self, log_dict: Dict, loss_dict: Dict):
        """
        記錄損失權重（GradNorm 與應用權重）

        記錄兩類權重：
        1. GradNorm 動態權重（gradnorm_weights）- 梯度範數平衡計算的權重
        2. 應用權重（applied_weights）- 實際應用到損失的最終權重（組合了配置、GradNorm、課程學習等）

        Args:
            log_dict: WandB 日誌字典
            loss_dict: 訓練步驟返回的損失字典
        """
        # 記錄 GradNorm 動態權重
        if 'gradnorm_weights' in loss_dict:
            gradnorm_weights = loss_dict['gradnorm_weights']
            if isinstance(gradnorm_weights, dict):
                for key, value in gradnorm_weights.items():
                    # 記錄到 Weights/GradNorm/ 路徑下
                    log_dict[f'Weights/GradNorm/{key}'] = float(value)

        # 記錄實際應用的最終權重
        if 'applied_weights' in loss_dict:
            applied_weights = loss_dict['applied_weights']
            if isinstance(applied_weights, dict):
                for key, value in applied_weights.items():
                    # 記錄到 Weights/Applied/ 路徑下
                    log_dict[f'Weights/Applied/{key}'] = float(value)

    def log_hyperparameters(self, current_lr: float, epoch: int):
        """
        記錄訓練超參數
        
        Args:
            current_lr: 當前學習率
            epoch: 當前 epoch
        """
        if self.wandb_run is not None:
            wandb.log({'Training/learning_rate': current_lr, 'epoch': epoch})
    
    def log_gradients_and_weights(self, model: nn.Module, epoch: int):
        """
        記錄梯度與權重統計（較耗時，建議降低頻率）
        
        Args:
            model: 模型
            epoch: 當前 epoch
        """
        if self.wandb_run is None:
            return
        
        log_dict = {'epoch': epoch}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                log_dict[f'Gradients/norm/{name}'] = grad_norm
                # WandB 自動處理直方圖
                log_dict[f'Gradients/hist/{name}'] = wandb.Histogram(param.grad.detach().cpu().numpy())
                log_dict[f'Weights/{name}'] = wandb.Histogram(param.detach().cpu().numpy())
        
        wandb.log(log_dict)

    def log_nonlinearities(self, model: nn.Module, epoch: int):
        """
        記錄 PirateNet/ResNet 的 alpha 非線性係數

        Args:
            model: 模型
            epoch: 當前 epoch
        """
        if self.wandb_run is None:
            return

        log_dict = {'epoch': epoch}
        idx = 0
        for name, module in model.named_modules():
            alpha = getattr(module, "alpha", None)
            if alpha is None:
                continue
            if not isinstance(alpha, torch.Tensor):
                continue
            log_dict[f'Nonlinearity/alpha_{idx}'] = alpha.item()
            idx += 1
        
        if idx > 0:
            wandb.log(log_dict)
    
    def finalize_wandb(self, final_metrics: Dict, hparams: Dict):
        """
        記錄最終超參數並完成 WandB 運行
        
        Args:
            final_metrics: 最終指標（hparam/final_loss, hparam/best_loss, etc.）
            hparams: 超參數字典（lr, optimizer, model_width, etc.）
        """
        if self.wandb_run is None:
            return
        
        # 記錄最終指標
        wandb.log(final_metrics)
        
        # 更新運行摘要
        for key, value in final_metrics.items():
            wandb.run.summary[key] = value
        
        # WandB 會在訓練結束時自動調用 finish()
        logging.info("✅ WandB 日誌已保存")
    
    # ========================================================================
    # 自適應更新協調
    # ========================================================================
    
    def coordinate_adaptive_updates(
        self,
        epoch: int,
        adaptive_sampler,  # Type: Optional[AdaptiveCollocationSampler]
        fourier_annealing,  # Type: Optional[FourierAnnealing]
        model: nn.Module,
        physics,
        training_data: Dict,
        config: Dict,
        device: torch.device,
        history: Dict
    ) -> Dict:
        """
        協調自適應採樣與 Fourier 退火
        
        Args:
            epoch: 當前 epoch
            adaptive_sampler: 自適應採樣器（可選，AdaptiveCollocationSampler 實例）
            fourier_annealing: Fourier 退火管理器（可選）
            model: 訓練模型
            physics: 物理方程
            training_data: 訓練數據
            config: 配置字典
            device: 計算設備
            history: 訓練歷史
        
        Returns:
            更新後的 training_data
        """
        # 1. 自適應採樣
        if adaptive_sampler is not None:
            training_data = self._handle_adaptive_sampling(
                adaptive_sampler, training_data, epoch, model, physics, config, device, history
            )
        
        # 2. Fourier 退火
        if fourier_annealing is not None:
            self._handle_fourier_annealing(fourier_annealing, model, epoch, config)
        
        return training_data
    
    def _handle_adaptive_sampling(
        self,
        adaptive_sampler,  # Type: AdaptiveCollocationSampler
        training_data: Dict,
        epoch: int,
        model: nn.Module,
        physics,
        config: Dict,
        device: torch.device,
        history: Dict
    ) -> Dict:
        """
        處理自適應採樣
        
        Args:
            adaptive_sampler: 自適應採樣器（AdaptiveCollocationSampler 實例）
            training_data: 訓練數據
            epoch: 當前 epoch
            model: 模型
            physics: 物理方程
            config: 配置字典
            device: 計算設備
            history: 訓練歷史
        
        Returns:
            更新後的 training_data
        """
        # 檢查是否需要重採樣
        current_loss = history['total_loss'][-1] if history['total_loss'] else float('inf')
        
        if adaptive_sampler.should_trigger(epoch, current_loss, residuals=None):
            try:
                # 1. 提取當前配點（支持多種數據格式）
                current_points = self._extract_collocation_points(training_data)
                if current_points is None:
                    logging.warning("⚠️ 無法提取配點座標，跳過重採樣")
                    return training_data
                
                # 2. 提取域邊界
                domain_bounds = self._extract_domain_bounds(config)
                if not domain_bounds:
                    logging.warning("⚠️ 無法提取域邊界，跳過重採樣")
                    return training_data
                
                # 3. 定義殘差計算函數
                def residual_fn(points: torch.Tensor) -> torch.Tensor:
                    """計算 PDE 殘差"""
                    points_device = points.to(device).requires_grad_(True)
                    outputs = model(points_device)
                    
                    # 根據 physics 模組計算殘差
                    if hasattr(physics, 'compute_pde_residuals'):
                        residuals = physics.compute_pde_residuals(points_device, outputs)
                    elif hasattr(physics, 'pde_residuals'):
                        residuals = physics.pde_residuals(points_device, outputs)
                    else:
                        # 回退：使用動量+連續性方程
                        residuals = []
                        if hasattr(physics, 'momentum_x_residual'):
                            residuals.append(physics.momentum_x_residual(points_device, outputs))
                        if hasattr(physics, 'momentum_y_residual'):
                            residuals.append(physics.momentum_y_residual(points_device, outputs))
                        if hasattr(physics, 'continuity_residual'):
                            residuals.append(physics.continuity_residual(points_device, outputs))
                        
                        if residuals:
                            residuals = torch.stack(residuals, dim=-1)
                        else:
                            raise AttributeError("Physics 模組缺少殘差計算方法")
                    
                    return residuals
                
                # 4. 執行重採樣
                new_points, metrics = adaptive_sampler.resample_collocation_points(
                    current_points=current_points,
                    domain_bounds=domain_bounds,
                    residual_fn=residual_fn,
                    n_keep=None,  # 使用配置中的 keep_ratio
                    device=str(device)
                )
                
                # 5. 更新訓練數據（保持原格式）
                training_data = self._update_collocation_points(training_data, new_points)
                
                logging.info(f"🔄 自適應重採樣完成 @ epoch {epoch}")
                logging.info(f"   配點數: {current_points.shape[0]} → {new_points.shape[0]}")
                logging.info(f"   保留率: {metrics.get('keep_ratio', 0.0):.2%}")
                logging.info(f"   新增點: {metrics.get('n_replaced', 0)}")
                
            except Exception as e:
                logging.error(f"❌ 自適應重採樣失敗 @ epoch {epoch}: {e}", exc_info=True)
        
        return training_data
    
    def _extract_domain_bounds(self, config: Dict) -> Dict[str, tuple]:
        """
        從配置中提取域邊界
        
        Args:
            config: 配置字典
            
        Returns:
            domain_bounds: {'x': (xmin, xmax), 'y': (ymin, ymax), ...}
        """
        domain_bounds = {}
        
        # 嘗試多種配置格式
        domain_cfg = config.get('domain', config.get('physics', {}).get('domain', {}))
        
        # 格式 1: x_range, y_range
        if 'x_range' in domain_cfg:
            domain_bounds['x'] = tuple(domain_cfg['x_range'])
        if 'y_range' in domain_cfg:
            domain_bounds['y'] = tuple(domain_cfg['y_range'])
        if 'z_range' in domain_cfg:
            domain_bounds['z'] = tuple(domain_cfg['z_range'])
        
        # 格式 2: x_min/x_max
        if 'x_min' in domain_cfg and 'x_max' in domain_cfg:
            domain_bounds['x'] = (domain_cfg['x_min'], domain_cfg['x_max'])
        if 'y_min' in domain_cfg and 'y_max' in domain_cfg:
            domain_bounds['y'] = (domain_cfg['y_min'], domain_cfg['y_max'])
        if 'z_min' in domain_cfg and 'z_max' in domain_cfg:
            domain_bounds['z'] = (domain_cfg['z_min'], domain_cfg['z_max'])
        
        # 時間維度（從 data.kolmogorov_config）
        time_cfg = config.get('data', {}).get('kolmogorov_config', {})
        if 'time_range' in time_cfg:
            domain_bounds['t'] = tuple(time_cfg['time_range'])
        
        return domain_bounds
    
    def _extract_collocation_points(self, training_data: Dict) -> Optional[torch.Tensor]:
        """
        從多種數據格式中提取配點座標
        
        支持格式:
        1. coords_pde_spatial: [N, D_spatial] 空間座標（可能需要加上時間）
        
        Args:
            training_data: 訓練數據字典
            
        Returns:
            torch.Tensor [N, D] 或 None（無法提取時）
        """
        # 預拼接的空間座標
        if 'coords_pde_spatial' in training_data:
            coords_spatial = training_data['coords_pde_spatial']
            
            # 檢查是否需要加上時間維度
            if 't_pde' in training_data and training_data['t_pde'] is not None:
                t_pde = training_data['t_pde']
                if isinstance(t_pde, torch.Tensor) and t_pde.numel() > 0:
                    return torch.cat([coords_spatial, t_pde], dim=1)
            
            return coords_spatial
        
        # 無法提取
        return None
    
    def _update_collocation_points(self, training_data: Dict, new_points: torch.Tensor) -> Dict:
        """
        更新訓練數據中的配點（保持原格式）
        
        Args:
            training_data: 原訓練數據字典
            new_points: 新配點 [N, D]
        
        Returns:
            更新後的 training_data
        """
        # coords_pde_spatial + t_pde
        if 'coords_pde_spatial' in training_data:
            if 't_pde' in training_data and training_data['t_pde'] is not None:
                # 分離空間和時間維度
                D_spatial = training_data['coords_pde_spatial'].shape[1]
                training_data['coords_pde_spatial'] = new_points[:, :D_spatial]
                training_data['t_pde'] = new_points[:, D_spatial:]
            else:
                training_data['coords_pde_spatial'] = new_points
            return training_data
        
        raise KeyError("training_data 缺少必要鍵: coords_pde_spatial")
        return training_data
    
    def _handle_fourier_annealing(
        self,
        fourier_annealing,
        model: nn.Module,
        epoch: int,
        config: Dict
    ):
        """
        處理 Fourier 退火
        
        Args:
            fourier_annealing: Fourier 退火管理器
            model: 模型
            epoch: 當前 epoch
            config: 配置字典
        """
        try:
            # 檢查模型是否有 Fourier features 模組
            fourier_module = None
            
            # 嘗試從模型中找到 Fourier features
            if hasattr(model, 'fourier_features'):
                fourier_module = model.fourier_features
            elif hasattr(model, 'encoder') and hasattr(model.encoder, 'fourier_features'):
                fourier_module = model.encoder.fourier_features
            
            if fourier_module is not None:
                # 獲取更新前的狀態（用於日誌）
                old_info = fourier_annealing.get_info()
                
                # 執行更新
                max_epochs = int(config['training']['epochs'])
                fourier_annealing.update_fourier_features(
                    fourier_module,
                    current_epoch=epoch,
                    total_epochs=max_epochs
                )
                
                # 獲取更新後的狀態
                new_info = fourier_annealing.get_info()
                
                # 檢查是否發生階段切換（比較 stage_index）
                if old_info['stage_index'] != new_info['stage_index']:
                    logging.info(f"🎯 Fourier 退火階段切換：{new_info['stage_description']}")
                    logging.info(f"   當前頻率: {new_info['active_frequencies']}")
                    logging.info(f"   輸出維度: {fourier_module.out_dim}")
        
        except AttributeError as e:
            # 模型不支持 Fourier 退火，警告一次後禁用
            if epoch == 0:
                logging.warning(f"⚠️ 模型不支持 Fourier 退火：{e}，已自動禁用")
            # 這裡不能直接禁用 fourier_annealing（因為它是參數），
            # 需要在 Trainer 層處理
        except Exception as e:
            logging.error(f"❌ Fourier 退火更新失敗（epoch {epoch}）: {e}")
