"""
訓練循環輔助管理器

負責訓練循環中的輔助功能：
- TensorBoard 日誌記錄
- 訓練歷史管理  
- 自適應更新協調（adaptive sampling, fourier annealing）

不包含核心訓練邏輯（step, validate, checkpoint, early stopping），
這些保留在 Trainer 類中。
"""

import logging
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TrainingLoopManager:
    """
    訓練循環輔助管理器
    
    職責：
    1. TensorBoard 日誌記錄（分類管理所有 scalar/histogram）
    2. 訓練歷史記錄與查詢
    3. 自適應更新協調（采樣+退火）
    
    不包含：
    - 訓練步驟執行（step）
    - 驗證邏輯（validate）
    - 檢查點保存（save_checkpoint）
    - 早停決策（check_early_stopping）
    """
    
    def __init__(self, config: Dict, writer: Optional[SummaryWriter]):
        """
        Args:
            config: 訓練配置字典
            writer: TensorBoard SummaryWriter（若為 None 則不記錄）
        """
        self.config = config
        self.writer = writer
        
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
    # TensorBoard 日誌記錄（分類管理）
    # ========================================================================
    
    def log_losses_to_tensorboard(self, loss_dict: Dict, epoch: int):
        """
        記錄所有損失到 TensorBoard
        
        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch
        """
        if self.writer is None:
            return
        
        # 1. 總損失與主要分量
        self._log_main_losses(loss_dict, epoch)
        
        # 2. PDE 子項（動量方程與連續性方程）
        self._log_pde_losses(loss_dict, epoch)
        
        # 3. 數據擬合損失（各變量）
        self._log_data_losses(loss_dict, epoch)
        
        # 4. Weighted Loss（分析權重平衡）
        self._log_weighted_losses(loss_dict, epoch)
        
        # 5. RANS Prior Loss（低保真先驗）
        self._log_prior_losses(loss_dict, epoch)
        
        # 6. 邊界條件損失
        self._log_bc_losses(loss_dict, epoch)
        
        # 7. 正則化項
        self._log_regularization_losses(loss_dict, epoch)
        
        # 8. 驗證指標
        self._log_validation_metrics(loss_dict, epoch)
    
    def _log_main_losses(self, loss_dict: Dict, epoch: int):
        """記錄主要損失（total, data, pde, boundary）"""
        self.writer.add_scalar('Loss/total', loss_dict.get('total_loss', 0.0), epoch)
        self.writer.add_scalar('Loss/data', loss_dict.get('data_loss', 0.0), epoch)
        self.writer.add_scalar('Loss/pde', loss_dict.get('pde_loss', 0.0), epoch)
        self.writer.add_scalar('Loss/boundary', loss_dict.get('bc_loss', 0.0), epoch)
    
    def _log_pde_losses(self, loss_dict: Dict, epoch: int):
        """記錄 PDE 子項（momentum, continuity）"""
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
    
    def _log_data_losses(self, loss_dict: Dict, epoch: int):
        """記錄數據擬合損失（u, v, w, pressure）"""
        if 'u_loss' in loss_dict:
            self.writer.add_scalar('Loss/Data/u', loss_dict['u_loss'], epoch)
        if 'v_loss' in loss_dict:
            self.writer.add_scalar('Loss/Data/v', loss_dict['v_loss'], epoch)
        if 'w_loss' in loss_dict:
            self.writer.add_scalar('Loss/Data/w', loss_dict['w_loss'], epoch)
        if 'pressure_loss' in loss_dict:
            self.writer.add_scalar('Loss/Data/pressure', loss_dict['pressure_loss'], epoch)
    
    def _log_weighted_losses(self, loss_dict: Dict, epoch: int):
        """記錄加權後的損失（分析權重平衡）"""
        if 'weighted_data_loss' in loss_dict:
            self.writer.add_scalar('Loss/Weighted/data', loss_dict['weighted_data_loss'], epoch)
        if 'weighted_pde_loss' in loss_dict:
            self.writer.add_scalar('Loss/Weighted/pde', loss_dict['weighted_pde_loss'], epoch)
        if 'weighted_div_loss' in loss_dict:
            self.writer.add_scalar('Loss/Weighted/continuity', loss_dict['weighted_div_loss'], epoch)
        if 'weighted_bc_loss' in loss_dict:
            self.writer.add_scalar('Loss/Weighted/boundary', loss_dict['weighted_bc_loss'], epoch)
    
    def _log_prior_losses(self, loss_dict: Dict, epoch: int):
        """記錄 RANS prior 損失"""
        if 'prior_consistency_loss' in loss_dict:
            self.writer.add_scalar('Loss/Prior/total', loss_dict['prior_consistency_loss'], epoch)
        if 'prior_loss_u' in loss_dict:
            self.writer.add_scalar('Loss/Prior/u', loss_dict['prior_loss_u'], epoch)
        if 'prior_loss_v' in loss_dict:
            self.writer.add_scalar('Loss/Prior/v', loss_dict['prior_loss_v'], epoch)
        if 'prior_loss_p' in loss_dict:
            self.writer.add_scalar('Loss/Prior/p', loss_dict['prior_loss_p'], epoch)
    
    def _log_bc_losses(self, loss_dict: Dict, epoch: int):
        """記錄邊界條件損失"""
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
    
    def _log_regularization_losses(self, loss_dict: Dict, epoch: int):
        """記錄正則化損失"""
        if 'regularization_loss' in loss_dict:
            self.writer.add_scalar('Loss/Regularization/total', loss_dict['regularization_loss'], epoch)
        if 'l2_reg' in loss_dict:
            self.writer.add_scalar('Loss/Regularization/l2', loss_dict['l2_reg'], epoch)
        if 'gradient_penalty' in loss_dict:
            self.writer.add_scalar('Loss/Regularization/gradient', loss_dict['gradient_penalty'], epoch)
    
    def _log_validation_metrics(self, loss_dict: Dict, epoch: int):
        """記錄驗證指標"""
        if 'val_loss' in loss_dict:
            self.writer.add_scalar('Validation/relative_l2', loss_dict['val_loss'], epoch)
        if 'val_mse' in loss_dict:
            self.writer.add_scalar('Validation/mse', loss_dict['val_mse'], epoch)
    
    def log_hyperparameters(self, current_lr: float, epoch: int):
        """
        記錄訓練超參數
        
        Args:
            current_lr: 當前學習率
            epoch: 當前 epoch
        """
        if self.writer is not None:
            self.writer.add_scalar('Training/learning_rate', current_lr, epoch)
    
    def log_gradients_and_weights(self, model: nn.Module, epoch: int):
        """
        記錄梯度與權重統計（較耗時，建議降低頻率）
        
        Args:
            model: 模型
            epoch: 當前 epoch
        """
        if self.writer is None:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.writer.add_scalar(f'Gradients/norm/{name}', grad_norm, epoch)
                self.writer.add_histogram(f'Gradients/hist/{name}', param.grad, epoch)
                self.writer.add_histogram(f'Weights/{name}', param, epoch)
    
    def finalize_tensorboard(self, final_metrics: Dict, hparams: Dict):
        """
        記錄最終超參數並關閉 TensorBoard
        
        Args:
            final_metrics: 最終指標（hparam/final_loss, hparam/best_loss, etc.）
            hparams: 超參數字典（lr, optimizer, model_width, etc.）
        """
        if self.writer is None:
            return
        
        self.writer.add_hparams(hparams, final_metrics)
        self.writer.flush()
        self.writer.close()
        logging.info("✅ TensorBoard 日誌已保存並關閉")
    
    # ========================================================================
    # 自適應更新協調
    # ========================================================================
    
    def coordinate_adaptive_updates(
        self,
        epoch: int,
        loop_manager,  # Type: Optional[AdaptiveCollocation]
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
            loop_manager: 自適應採樣管理器（可選）
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
        if loop_manager is not None:
            training_data = self._handle_adaptive_sampling(
                loop_manager, training_data, epoch, model, physics, config, device, history
            )
        
        # 2. Fourier 退火
        if fourier_annealing is not None:
            self._handle_fourier_annealing(fourier_annealing, model, epoch, config)
        
        return training_data
    
    def _handle_adaptive_sampling(
        self,
        loop_manager,
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
            loop_manager: 自適應採樣管理器
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
        # 更新訓練批次
        training_data = loop_manager.update_training_batch(training_data, epoch)
        
        # 檢查是否需要重採樣
        if epoch > 0 and loop_manager.should_resample_collocation_points(
            epoch,
            history['total_loss'][-1] if history['total_loss'] else float('inf'),
            None  # residuals 參數設為 None
        ):
            try:
                # 提取域邊界
                domain_bounds = {}
                domain_cfg = config.get('domain', {})
                
                if 'x_min' in domain_cfg and 'x_max' in domain_cfg:
                    domain_bounds['x'] = (domain_cfg['x_min'], domain_cfg['x_max'])
                if 'y_min' in domain_cfg and 'y_max' in domain_cfg:
                    domain_bounds['y'] = (domain_cfg['y_min'], domain_cfg['y_max'])
                if 'z_min' in domain_cfg and 'z_max' in domain_cfg:
                    domain_bounds['z'] = (domain_cfg['z_min'], domain_cfg['z_max'])
                if 't_min' in domain_cfg and 't_max' in domain_cfg:
                    domain_bounds['t'] = (domain_cfg['t_min'], domain_cfg['t_max'])
                
                new_points, metrics = loop_manager.resample_collocation_points(
                    model, physics, domain_bounds, epoch, str(device)
                )
                logging.info(f"🔄 重採樣 {len(new_points)} 個配點（epoch {epoch}）")
                logging.debug(f"   指標: {metrics}")
            except Exception as e:
                logging.warning(f"⚠️ 重採樣失敗（epoch {epoch}）: {e}")
        
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
                max_epochs = config.get('train', {}).get('epochs', config.get('train', {}).get('max_epochs', 1000))
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
