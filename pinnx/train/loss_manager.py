"""
損失管理模組

負責統一管理所有損失計算邏輯，包含：
- PDE 殘差損失（動量方程 + 連續方程）
- 邊界條件損失（壁面 + 週期性）
- 資料監督損失
- 低保真先驗一致性損失
- 動態權重調整（GradNorm、課程學習、Causal Weighting）
- 損失組合與加權

設計目標：
1. 單一職責：專注於損失計算，不處理資料預處理或優化步驟
2. 可測試性：每個方法可獨立測試
3. 可擴展性：支援新增損失項而不影響現有邏輯
"""

import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import inspect

from pinnx.train.loop import apply_point_weights_to_loss
from pinnx.losses.priors import PriorLossManager
from pinnx.losses.weighting import GradNormWeighter, CausalWeighter
from pinnx.physics.turbulence_utils import preprocess_rans_prior_from_config  # type: ignore


class LossManager:
    """
    損失管理器
    
    統一管理訓練過程中所有損失項的計算與組合。
    
    Attributes:
        config: 訓練配置字典
        loss_cfg: 損失配置（從 config['loss'] 提取）
        physics: 物理方程模組
        model: PINN 模型
        device: 計算設備
        data_normalizer: 資料標準化器
        prior_loss_manager: 先驗損失管理器（可選）
        weighters: 動態權重調整器字典（可選）
        losses: 額外損失函數字典（如均值約束）
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        physics: Any,
        model: nn.Module,
        device: torch.device,
        data_normalizer: Any,
        prior_loss_manager: Optional[PriorLossManager] = None,
        weighters: Optional[Dict[str, Any]] = None,
        losses: Optional[Dict[str, nn.Module]] = None
    ):
        """
        初始化損失管理器
        
        Args:
            config: 完整訓練配置
            physics: 物理方程模組
            model: PINN 模型
            device: 計算設備
            data_normalizer: 資料標準化器
            prior_loss_manager: 先驗損失管理器（可選）
            weighters: 動態權重調整器字典（可選）
            losses: 額外損失函數字典（可選）
        """
        self.config = config
        self.loss_cfg = config.get('loss', {})
        self.physics = physics
        self.model = model
        self.device = device
        self.data_normalizer = data_normalizer
        self.prior_loss_manager = prior_loss_manager
        self.weighters = weighters or {}
        self.losses = losses or {}
        
        # 緩存標誌（避免重複日誌）
        self._weights_logged = False
        self._debug_printed = False
    
    def compute_pde_loss(
        self,
        coords_pde_physical: torch.Tensor,
        model_coords_pde: torch.Tensor,
        u_pred_pde_physical: torch.Tensor,
        data_batch: Dict[str, Any],
        epoch: int,
        is_vs_pinn: bool = False,
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算 PDE 殘差損失（動量方程 + 連續方程）
        
        Args:
            coords_pde_physical: 物理空間坐標（含梯度追蹤）
            model_coords_pde: 模型輸入坐標（標準化+縮放）
            u_pred_pde_physical: 物理空間預測值（已反標準化）
            data_batch: 資料批次字典
            epoch: 當前訓練輪次
            is_vs_pinn: 是否為 VS-PINN 模式
            gradients: 🚀 Wave 2 - 預計算的梯度字典（可選，None 則自動計算）
            
        Returns:
            損失字典，包含：
            - momentum_x_loss, momentum_y_loss, momentum_z_loss
            - continuity_loss
            - pde_loss（總 PDE 損失）
        """
        # 提取速度和壓力分量（物理空間）
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
        
        # 提取時間分量（如果存在）
        t_pde = data_batch.get('t_pde')
        if t_pde is not None:
            t_pde = t_pde.to(self.device).requires_grad_(True)
        
        # 計算物理殘差
        try:
            if is_vs_pinn and hasattr(self.physics, 'compute_momentum_residuals'):
                # VS-PINN 3D（🚀 Wave 2: 傳遞預計算的梯度）
                residuals_mom = self.physics.compute_momentum_residuals(
                    coords_pde_physical,
                    predictions_phys,
                    scaled_coords=model_coords_pde,
                    gradients=gradients  # 🚀 Wave 2: 如果 None，physics 會自動計算
                )
                continuity_residual = self.physics.compute_continuity_residual(
                    coords_pde_physical,
                    predictions_phys,
                    scaled_coords=model_coords_pde,
                    gradients=gradients  # 🚀 Wave 2: 如果 None，physics 會自動計算
                )
                residuals = {
                    'momentum_x': residuals_mom['momentum_x'],
                    'momentum_y': residuals_mom['momentum_y'],
                    'momentum_z': residuals_mom['momentum_z'],
                    'continuity': continuity_residual,
                }
            else:
                # 標準 NS 2D 或 Kolmogorov Flow
                residual_fn = self.physics.residual_unified
                sig = inspect.signature(residual_fn)
                
                kwargs = {}
                if 'time' in sig.parameters and t_pde is not None:
                    kwargs['time'] = t_pde
                
                # 添加 RANS 湍流黏度（如果存在）
                if 'nu_t' in sig.parameters:
                    nu_t_pde = None
                    if 'lowfi_prior' in data_batch and data_batch['lowfi_prior'] is not None:
                        lowfi_prior = data_batch['lowfi_prior']
                        if 'nu_t_pde' in lowfi_prior:
                            nu_t_raw = lowfi_prior['nu_t_pde']
                            
                            # 預處理 RANS 湍流黏度
                            if nu_t_raw is not None and hasattr(self, 'config'):
                                preprocessing_cfg = self.config.get('lowfi_prior', {}).get('preprocessing', {})
                                if preprocessing_cfg.get('enabled', True):
                                    nu_t_pde, stats = preprocess_rans_prior_from_config(
                                        nu_t_raw,
                                        coords_pde_physical,
                                        self.config,
                                        epoch=epoch
                                    )
                                else:
                                    nu_t_pde = nu_t_raw
                            else:
                                nu_t_pde = nu_t_raw
                    kwargs['nu_t'] = nu_t_pde
                
                residuals = residual_fn(
                    coords=coords_pde_physical,
                    predictions=u_pred_pde_physical,
                    **kwargs
                )
            
            # 應用點權重（如果存在）
            pde_point_weights = data_batch.get('pde_point_weights', None)
            
            # Causal Training: 計算並應用時間因果權重
            causal_weighter = self.weighters.get('causal')
            if causal_weighter is not None and t_pde is not None:
                # 1. 匯總每個點的殘差平方和
                total_res_sq = torch.zeros_like(t_pde)
                for r in residuals.values():
                    total_res_sq += r.detach()**2
                
                # 2. 計算權重 w(t)
                causal_weights = causal_weighter.compute_weights(total_res_sq, t_pde)
                
                # 3. 記錄權重統計（調試）
                if epoch % 100 == 0 and epoch > 0:
                    logging.debug(
                        f"Causal Weights: min={causal_weights.min():.4f}, "
                        f"max={causal_weights.max():.4f}, mean={causal_weights.mean():.4f}"
                    )
                
                # 4. 應用權重到每個殘差項
                final_weights = causal_weights
                if pde_point_weights is not None:
                    final_weights = final_weights * pde_point_weights
                
                # 檢查是否使用 momentum merging
                if 'momentum' in residuals:
                    # 合併模式：momentum 是 [N, 1, 2] 向量
                    momentum_vec = residuals['momentum']  # [N, 1, 2]
                    momentum_loss = torch.mean(final_weights * torch.sum(momentum_vec**2, dim=-1))
                    momentum_x_loss = momentum_loss  # 合併後只有一個 momentum loss
                    momentum_y_loss = torch.tensor(0.0, device=self.device)
                else:
                    # 標準模式：分開的 momentum_x 和 momentum_y
                    momentum_x_loss = torch.mean(final_weights * residuals['momentum_x']**2)
                    momentum_y_loss = torch.mean(final_weights * residuals['momentum_y']**2)
                
                momentum_z_loss = torch.mean(final_weights * residuals['momentum_z']**2) if is_vs_pinn else torch.tensor(0.0, device=self.device)
                continuity_loss = torch.mean(final_weights * residuals['continuity']**2)
            
            elif pde_point_weights is not None:
                # 檢查是否使用 momentum merging
                if 'momentum' in residuals:
                    # 合併模式：momentum 是 [N, 1, 2] 向量
                    momentum_vec = residuals['momentum']  # [N, 1, 2]
                    momentum_loss = apply_point_weights_to_loss(torch.sum(momentum_vec**2, dim=-1), pde_point_weights)
                    momentum_x_loss = momentum_loss  # 合併後只有一個 momentum loss
                    momentum_y_loss = torch.tensor(0.0, device=self.device)
                else:
                    # 標準模式：分開的 momentum_x 和 momentum_y
                    momentum_x_loss = apply_point_weights_to_loss(residuals['momentum_x']**2, pde_point_weights)
                    momentum_y_loss = apply_point_weights_to_loss(residuals['momentum_y']**2, pde_point_weights)
                
                momentum_z_loss = apply_point_weights_to_loss(residuals['momentum_z']**2, pde_point_weights) if is_vs_pinn else torch.tensor(0.0, device=self.device)
                continuity_loss = apply_point_weights_to_loss(residuals['continuity']**2, pde_point_weights)
            else:
                # 檢查是否使用 momentum merging
                if 'momentum' in residuals:
                    # 合併模式：momentum 是 [N, 1, 2] 向量
                    momentum_vec = residuals['momentum']  # [N, 1, 2]
                    momentum_loss = torch.mean(torch.sum(momentum_vec**2, dim=-1))
                    momentum_x_loss = momentum_loss  # 合併後只有一個 momentum loss
                    momentum_y_loss = torch.tensor(0.0, device=self.device)
                else:
                    # 標準模式：分開的 momentum_x 和 momentum_y
                    momentum_x_loss = torch.mean(residuals['momentum_x']**2)
                    momentum_y_loss = torch.mean(residuals['momentum_y']**2)
                
                momentum_z_loss = torch.mean(residuals['momentum_z']**2) if is_vs_pinn else torch.tensor(0.0, device=self.device)
                continuity_loss = torch.mean(residuals['continuity']**2)
        
        except Exception as e:
            logging.error(f"🚨 物理殘差計算失敗: {e}")
            logging.error(f"coords_pde_physical shape: {coords_pde_physical.shape}, requires_grad: {coords_pde_physical.requires_grad}")
            logging.error(f"u_pred_pde_physical shape: {u_pred_pde_physical.shape}, requires_grad: {u_pred_pde_physical.requires_grad}")
            if is_vs_pinn:
                logging.error(f"predictions_phys shape: {predictions_phys.shape if predictions_phys is not None else None}")
            raise
        
        # 統計 PDE 損失
        pde_loss = momentum_x_loss + momentum_y_loss + momentum_z_loss + continuity_loss
        
        return {
            'momentum_x_loss': momentum_x_loss,
            'momentum_y_loss': momentum_y_loss,
            'momentum_z_loss': momentum_z_loss,
            'continuity_loss': continuity_loss,
            'pde_loss': pde_loss
        }
    
    def compute_bc_loss(
        self,
        u_bc_pred_phys: torch.Tensor,
        data_batch: Dict[str, Any],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        計算邊界條件損失（壁面 + 週期性）
        
        Args:
            u_bc_pred_phys: 邊界點預測值（物理空間，已反標準化）
            data_batch: 資料批次字典
            epoch: 當前訓練輪次
            
        Returns:
            損失字典，包含：
            - wall_loss（如果有壁面邊界）
            - periodic_x_loss, periodic_y_loss（如果有週期性邊界）
        """
        # 壁面邊界條件損失
        y_bc = data_batch['y_bc']
        wall_mask = (torch.abs(y_bc - 1.0) < 1e-3) | (torch.abs(y_bc + 1.0) < 1e-3)
        wall_mask = wall_mask.squeeze()
        
        if wall_mask.sum() > 0:
            u_wall = u_bc_pred_phys[wall_mask, 0]  # u 分量
            v_wall = u_bc_pred_phys[wall_mask, 1]  # v 分量
            wall_loss = torch.mean(u_wall**2 + v_wall**2)
        else:
            wall_loss = torch.tensor(0.0, device=self.device)
            if epoch == 0:
                if y_bc.numel() > 0:
                    logging.warning(f"⚠️ 未檢測到壁面邊界點！y_bc 範圍: [{y_bc.min():.6f}, {y_bc.max():.6f}]")
                else:
                    logging.info(f"ℹ️ 無壁面邊界點（純週期性系統，如 Kolmogorov Flow）")
        
        # 週期性邊界條件損失
        periodic_x_loss = torch.tensor(0.0, device=self.device)
        periodic_y_loss = torch.tensor(0.0, device=self.device)
        
        if hasattr(self.physics, 'compute_periodic_loss'):
            try:
                # 邊界點結構：[左邊界, 右邊界, 下邊界, 上邊界]（成對順序）
                n_bc = u_bc_pred_phys.shape[0]
                n_pairs_total = n_bc // 2
                n_pairs_x = n_pairs_total // 2
                n_pairs_y = n_pairs_total - n_pairs_x
                
                # x 方向週期性
                if n_pairs_x > 0:
                    pred_left = u_bc_pred_phys[0:n_pairs_x]
                    pred_right = u_bc_pred_phys[n_pairs_x:2*n_pairs_x]
                    periodic_x_loss = torch.mean((pred_left - pred_right) ** 2)
                
                # y 方向週期性
                if n_pairs_y > 0:
                    start_y = 2 * n_pairs_x
                    pred_bottom = u_bc_pred_phys[start_y:start_y+n_pairs_y]
                    pred_top = u_bc_pred_phys[start_y+n_pairs_y:]
                    periodic_y_loss = torch.mean((pred_bottom - pred_top) ** 2)
                
                if epoch == 0:
                    logging.info(f"✅ 週期性邊界條件已啟用（成對計算，x方向{n_pairs_x}對，y方向{n_pairs_y}對）")
            except Exception as e:
                logging.warning(f"⚠️ 週期性損失計算失敗: {e}")
                periodic_x_loss = torch.tensor(0.0, device=self.device)
                periodic_y_loss = torch.tensor(0.0, device=self.device)
        
        return {
            'wall_loss': wall_loss,
            'periodic_x_loss': periodic_x_loss,
            'periodic_y_loss': periodic_y_loss
        }
    
    def compute_data_loss(
        self,
        u_sensors_pred_phys: torch.Tensor,
        data_batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        計算資料監督損失
        
        Args:
            u_sensors_pred_phys: 感測點預測值（物理空間，已反標準化）
            data_batch: 資料批次字典
            
        Returns:
            損失字典，包含：
            - u_loss, v_loss, w_loss, pressure_loss
            - velocity_loss, data_loss（總資料損失）
        """
        # 提取真實資料（物理空間）
        u_true = data_batch['u_sensors']
        v_true = data_batch['v_sensors']
        w_true = data_batch.get('w_sensors', None)
        p_true = data_batch['p_sensors']
        
        # 計算各分量 MSE
        u_loss = torch.mean((u_sensors_pred_phys[:, 0:1] - u_true)**2)
        v_loss = torch.mean((u_sensors_pred_phys[:, 1:2] - v_true)**2)
        
        # 根據資料與模型維度動態選擇損失計算
        has_w_data = w_true is not None and w_true.numel() > 0
        model_has_w = u_sensors_pred_phys.shape[1] >= 4
        
        if has_w_data and model_has_w:
            # 完整 3D 模式
            w_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - w_true) ** 2) if w_true is not None else torch.tensor(0.0, device=self.device)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
            velocity_loss = u_loss + v_loss + w_loss
        elif model_has_w and not has_w_data:
            # 混合模式：模型輸出 4D 但資料僅有 3D
            w_loss = torch.tensor(0.0, device=u_loss.device)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 3:4] - p_true)**2)
            velocity_loss = u_loss + v_loss
        else:
            # 標準 2D 模式
            w_loss = torch.tensor(0.0, device=u_loss.device)
            pressure_loss = torch.mean((u_sensors_pred_phys[:, 2:3] - p_true)**2)
            velocity_loss = u_loss + v_loss
        
        data_loss = velocity_loss + pressure_loss
        
        return {
            'u_loss': u_loss,
            'v_loss': v_loss,
            'w_loss': w_loss,
            'pressure_loss': pressure_loss,
            'velocity_loss': velocity_loss,
            'data_loss': data_loss
        }
    
    def compute_lowfi_prior_loss(
        self,
        u_pred_pde_physical: torch.Tensor,
        coords_pde_physical: torch.Tensor,
        data_batch: Dict[str, Any],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        計算低保真先驗一致性損失
        
        Args:
            u_pred_pde_physical: PDE 點預測值（物理空間，已反標準化）
            coords_pde_physical: PDE 點物理坐標
            data_batch: 資料批次字典
            epoch: 當前訓練輪次
            
        Returns:
            損失字典，包含：
            - prior_consistency_loss（總先驗損失）
            - prior_loss_u, prior_loss_v, prior_loss_p（各分量損失）
        """
        prior_consistency_loss = torch.tensor(0.0, device=self.device)
        prior_loss_u = torch.tensor(0.0, device=self.device)
        prior_loss_v = torch.tensor(0.0, device=self.device)
        prior_loss_p = torch.tensor(0.0, device=self.device)
        
        if self.prior_loss_manager is not None and data_batch.get('has_prior', False):
            lowfi_prior = data_batch.get('lowfi_prior', {})
            
            # ========================================================
            # ✅ FIX: 檢查壓力有效性（Leith 模型無壓力）
            # ========================================================
            lowfi_metadata = lowfi_prior.get('metadata', {})
            pressure_valid = lowfi_metadata.get('pressure_valid', True)
            
            if 'u_pde' in lowfi_prior and 'v_pde' in lowfi_prior:
                # 根據壓力有效性決定變數列表
                if pressure_valid and 'p_pde' in lowfi_prior:
                    # 完整三變數 (u, v, p) - 用於 3D RANS 或其他有壓力的模型
                    lowfi_data = torch.cat([
                        lowfi_prior['u_pde'],
                        lowfi_prior['v_pde'],
                        lowfi_prior['p_pde']
                    ], dim=1)
                    variable_names = ['u', 'v', 'p']
                    
                    # 提取對應的 PINN 預測
                    if u_pred_pde_physical.shape[1] == 3:
                        high_fi_pred = u_pred_pde_physical
                    elif u_pred_pde_physical.shape[1] == 4:
                        # 忽略 w 分量
                        high_fi_pred = torch.cat([
                            u_pred_pde_physical[:, 0:1],  # u
                            u_pred_pde_physical[:, 1:2],  # v
                            u_pred_pde_physical[:, 3:4]   # p
                        ], dim=1)
                    else:
                        raise ValueError(f"不支援的模型輸出維度: {u_pred_pde_physical.shape[1]}")
                    
                else:
                    # ✅ Leith 模型：僅兩變數 (u, v)，跳過壓力
                    lowfi_data = torch.cat([
                        lowfi_prior['u_pde'],
                        lowfi_prior['v_pde']
                    ], dim=1)
                    variable_names = ['u', 'v']
                    
                    # 提取對應的 PINN 預測（僅 u, v）
                    high_fi_pred = torch.cat([
                        u_pred_pde_physical[:, 0:1],  # u
                        u_pred_pde_physical[:, 1:2]   # v
                    ], dim=1)
                    
                    if epoch == 0:
                        logging.info("⚠️  Leith 模型：跳過壓力項，僅使用 u, v 計算先驗損失")
                
                # 計算先驗一致性損失
                prior_losses = self.prior_loss_manager.low_fidelity_loss(
                    high_fidelity_pred=high_fi_pred,
                    low_fidelity_data=lowfi_data,
                    coords=coords_pde_physical,
                    variable_names=variable_names
                )
                
                prior_loss_u = prior_losses.get('prior_consistency_u', prior_loss_u)
                prior_loss_v = prior_losses.get('prior_consistency_v', prior_loss_v)
                
                # 壓力損失僅在有效時讀取
                if pressure_valid:
                    prior_loss_p = prior_losses.get('prior_consistency_p', prior_loss_p)
                
                prior_consistency_loss = prior_losses['prior_consistency_total']
                
                # 記錄（低頻率）
                if epoch == 0 or (epoch > 0 and epoch % 100 == 0):
                    logging.info(f"📊 先驗一致性損失 @ Epoch {epoch}: {prior_consistency_loss.item():.6f}")
                    logging.info(f"   - u: {prior_loss_u.item():.6f}")
                    logging.info(f"   - v: {prior_loss_v.item():.6f}")
                    if pressure_valid:
                        logging.info(f"   - p: {prior_loss_p.item():.6f}")
        
        return {
            'prior_consistency_loss': prior_consistency_loss,
            'prior_loss_u': prior_loss_u,
            'prior_loss_v': prior_loss_v,
            'prior_loss_p': prior_loss_p
        }
    
    def compute_mean_constraint_loss(
        self,
        u_pred_pde_physical: torch.Tensor,
        epoch: int
    ) -> torch.Tensor:
        """
        計算均值約束損失（如果啟用）
        
        Args:
            u_pred_pde_physical: PDE 點預測值（物理空間，已反標準化）
            epoch: 當前訓練輪次
            
        Returns:
            均值約束損失（加權後）
        """
        mean_constraint_loss = torch.tensor(0.0, device=self.device)
        mean_constraint_cfg = self.loss_cfg.get('mean_constraint', {})
        
        if mean_constraint_cfg.get('enabled', False) and 'mean_constraint' in self.losses:
            target_means = mean_constraint_cfg.get('target_means', {})
            field_indices = {'u': 0, 'v': 1, 'w': 2}  # 不約束壓力場
            
            mean_constraint_loss_fn = self.losses['mean_constraint']
            mean_constraint_loss = mean_constraint_loss_fn(
                predictions=u_pred_pde_physical,
                target_means=target_means,
                field_indices=field_indices
            )
            
            w_mean_constraint = mean_constraint_cfg.get('weight', 10.0)
            mean_constraint_loss = w_mean_constraint * mean_constraint_loss
            
            # 記錄（低頻率）
            if epoch == 0 or (epoch > 0 and epoch % 100 == 0):
                logging.info(f"📊 均值約束損失 @ Epoch {epoch}: {mean_constraint_loss.item():.6f}")
                with torch.no_grad():
                    for field_name, idx in field_indices.items():
                        if idx < u_pred_pde_physical.shape[-1]:
                            pred_mean = u_pred_pde_physical[:, idx].mean().item()
                            target_mean = target_means.get(field_name, 0.0)
                            logging.info(f"   {field_name}: pred={pred_mean:.4f}, target={target_mean:.4f}")
        
        return mean_constraint_loss
    
    def apply_curriculum_weights(
        self,
        epoch: int
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, float]]:
        """
        應用課程學習權重覆蓋
        
        Args:
            epoch: 當前訓練輪次
            
        Returns:
            (課程配置, 覆蓋後的 loss_cfg)
        """
        curriculum_weighter = self.weighters.get('curriculum')
        if curriculum_weighter is None:
            return None, self.loss_cfg
        
        # 獲取當前階段配置
        curriculum_config = curriculum_weighter.get_stage_config(epoch)
        is_curriculum_transition = curriculum_config.get('is_transition', False)
        
        # 應用課程學習權重（覆蓋配置文件的基礎權重）
        curriculum_weights = curriculum_config.get('weights', {})
        if not curriculum_weights:
            return curriculum_config, self.loss_cfg
        
        # 更新 loss_cfg（臨時覆蓋）
        loss_cfg = self.loss_cfg.copy()
        for key, value in curriculum_weights.items():
            if key == 'data':
                loss_cfg['data_weight'] = value
            elif key == 'boundary':
                loss_cfg['bc_weight'] = value
            elif key == 'periodicity':
                loss_cfg['periodicity_weight'] = value
            elif key == 'prior':
                loss_cfg['prior_weight'] = value
                # 同時更新 lowfi_prior.consistency_weight
                if 'lowfi_prior' in curriculum_config:
                    prior_cfg = curriculum_config['lowfi_prior']
                    if 'consistency_weight' in prior_cfg:
                        if self.prior_loss_manager is not None:
                            self.prior_loss_manager.consistency_weight = prior_cfg['consistency_weight']
                            self.prior_loss_manager.low_fidelity_loss.consistency_weight = prior_cfg['consistency_weight']
            else:
                # 其他權重直接映射
                weight_key = f'{key}_weight'
                loss_cfg[weight_key] = value
        
        # 階段切換日誌
        if is_curriculum_transition:
            logging.info(f"🎯 課程階段切換：{curriculum_config['stage_name']}")
            logging.info(f"   損失權重更新: {curriculum_weights}")
            if 'lowfi_prior' in curriculum_config:
                prior_weight = curriculum_config['lowfi_prior'].get('consistency_weight', 'N/A')
                logging.info(f"   RANS 先驗權重: {prior_weight}")
        
        return curriculum_config, loss_cfg
    
    def apply_gradnorm_weights(
        self,
        loss_terms: Dict[str, torch.Tensor]
    ) -> Tuple[Optional[Dict[str, float]], Dict[str, float]]:
        """
        應用 GradNorm 動態權重調整
        
        Args:
            loss_terms: 損失項字典
            
        Returns:
            (GradNorm 權重, GradNorm 比例字典)
        """
        gradnorm_weighter = self.weighters.get('gradnorm')
        gradnorm_weights: Optional[Dict[str, float]] = None
        gradnorm_ratio: Dict[str, float] = {}
        
        if gradnorm_weighter is None:
            return None, {}
        
        available_losses = {
            name: loss_terms[name]
            for name in gradnorm_weighter.loss_names
            if name in loss_terms
        }
        
        if len(available_losses) < 2:
            logging.debug("GradNorm requires at least two loss terms; skipping update.")
            return None, {}
        
        gradnorm_weights = gradnorm_weighter.update_weights(available_losses)
        initial_values = getattr(gradnorm_weighter, 'initial_weight_values', {})
        eps = float(getattr(gradnorm_weighter, 'eps', 1e-12))
        
        for name, weight_val in (gradnorm_weights.items() if gradnorm_weights is not None else []):
            base = float(initial_values.get(name, 1.0))
            if abs(base) < eps:
                base = 1.0
            gradnorm_ratio[name] = float(weight_val) / base
        
        if gradnorm_weights is not None and gradnorm_weighter.step_count % gradnorm_weighter.update_frequency == 0:
            logging.debug(
                "GradNorm weights @ step %d: %s",
                gradnorm_weighter.step_count,
                {k: round(v, 4) for k, v in gradnorm_weights.items()}
            )
        
        return gradnorm_weights, gradnorm_ratio
    
    def combine_losses(
        self,
        loss_dict: Dict[str, torch.Tensor],
        loss_cfg: Dict[str, Any],
        gradnorm_ratio: Dict[str, float],
        is_vs_pinn: bool,
        epoch: int
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        組合所有損失項並應用權重
        
        Args:
            loss_dict: 所有損失項字典（未加權）
            loss_cfg: 損失配置（可能被課程學習覆蓋）
            gradnorm_ratio: GradNorm 權重比例
            is_vs_pinn: 是否為 VS-PINN 模式
            epoch: 當前訓練輪次
            
        Returns:
            (總損失, 結果字典)
        """
        def scaled_weight(name: str, base: float) -> float:
            return float(base) * gradnorm_ratio.get(name, 1.0)
        
        # 基礎權重
        base_data_weight = loss_cfg.get('data_weight', 100.0)
        base_pde_weight = loss_cfg.get('pde_weight', 1.0)
        base_bc_weight = loss_cfg.get('bc_weight', 10.0)
        
        # 計算各項加權權重
        w_data = scaled_weight('data', base_data_weight)
        w_momentum_x = scaled_weight('momentum_x', loss_cfg.get('momentum_x_weight', base_pde_weight))
        w_momentum_y = scaled_weight('momentum_y', loss_cfg.get('momentum_y_weight', base_pde_weight))
        w_momentum_z = scaled_weight('momentum_z', loss_cfg.get('momentum_z_weight', base_pde_weight)) if is_vs_pinn else 0.0
        w_div = scaled_weight('divergence', loss_cfg.get('div_weight', loss_cfg.get('continuity_weight', base_pde_weight)))
        
        # 邊界條件權重
        if hasattr(self.physics, 'compute_periodic_loss'):
            w_periodic_x = scaled_weight('periodic_x', loss_cfg.get('periodic_x_weight', base_bc_weight))
            w_periodic_y = scaled_weight('periodic_y', loss_cfg.get('periodic_y_weight', base_bc_weight))
            w_bc = 0.0
        else:
            w_periodic_x = 0.0
            w_periodic_y = 0.0
            w_bc = scaled_weight('wall_constraint', loss_cfg.get('wall_constraint_weight', base_bc_weight))
        
        # 診斷日誌
        if epoch == 0 and not self._weights_logged:
            logging.info("=" * 60)
            logging.info("📊 Loss 權重配置摘要")
            logging.info("=" * 60)
            logging.info(f"  Data Loss:        {w_data:.2e}")
            logging.info(f"  Momentum X:       {w_momentum_x:.2e}")
            logging.info(f"  Momentum Y:       {w_momentum_y:.2e}")
            if is_vs_pinn:
                logging.info(f"  Momentum Z:       {w_momentum_z:.2e}")
            logging.info(f"  Divergence:       {w_div:.2e}")
            if hasattr(self.physics, 'compute_periodic_loss'):
                logging.info(f"  Periodic X:       {w_periodic_x:.2e}")
                logging.info(f"  Periodic Y:       {w_periodic_y:.2e}")
            else:
                logging.info(f"  Wall Constraint:  {w_bc:.2e}")
            logging.info("=" * 60)
            self._weights_logged = True
        
        # 計算加權損失
        weighted_momentum_loss = (
            w_momentum_x * loss_dict['momentum_x_loss'] +
            w_momentum_y * loss_dict['momentum_y_loss'] +
            w_momentum_z * loss_dict['momentum_z_loss']
        )
        weighted_div_loss = w_div * loss_dict['continuity_loss']
        weighted_data_loss = w_data * loss_dict['data_loss']
        
        # 邊界條件損失
        if hasattr(self.physics, 'compute_periodic_loss'):
            weighted_bc_loss = w_periodic_x * loss_dict['periodic_x_loss'] + w_periodic_y * loss_dict['periodic_y_loss']
        else:
            weighted_bc_loss = w_bc * loss_dict['wall_loss']
        
        # 總損失
        total_loss = (
            weighted_data_loss +
            weighted_momentum_loss +
            weighted_div_loss +
            weighted_bc_loss +
            loss_dict.get('mean_constraint_loss', torch.tensor(0.0, device=self.device)) +
            loss_dict.get('prior_consistency_loss', torch.tensor(0.0, device=self.device))
        )
        
        # 構建結果字典
        result = {
            'total_loss': total_loss.item(),
            'data_loss': loss_dict['data_loss'].item(),
            'pde_loss': (weighted_momentum_loss + weighted_div_loss).item(),
            'div_loss': loss_dict['continuity_loss'].item(),
            'weighted_data_loss': weighted_data_loss.item(),
            'weighted_pde_loss': weighted_momentum_loss.item(),
            'weighted_div_loss': weighted_div_loss.item(),
            'weighted_bc_loss': weighted_bc_loss.item(),
            'momentum_x_loss': loss_dict['momentum_x_loss'].item(),
            'momentum_y_loss': loss_dict['momentum_y_loss'].item(),
            'momentum_z_loss': loss_dict['momentum_z_loss'].item(),
            'continuity_loss': loss_dict['continuity_loss'].item(),
            'u_loss': loss_dict['u_loss'].item(),
            'v_loss': loss_dict['v_loss'].item(),
            'w_loss': loss_dict['w_loss'].item(),
            'pressure_loss': loss_dict['pressure_loss'].item(),
        }
        
        # 添加邊界條件損失細項
        if hasattr(self.physics, 'compute_periodic_loss'):
            result['periodic_x_loss'] = loss_dict['periodic_x_loss'].item()
            result['periodic_y_loss'] = loss_dict['periodic_y_loss'].item()
        else:
            result['wall_loss'] = loss_dict['wall_loss'].item()
        
        # 添加先驗一致性損失細項
        if self.prior_loss_manager is not None:
            result['prior_consistency_loss'] = loss_dict.get('prior_consistency_loss', torch.tensor(0.0)).item()
            result['prior_loss_u'] = loss_dict.get('prior_loss_u', torch.tensor(0.0)).item()
            result['prior_loss_v'] = loss_dict.get('prior_loss_v', torch.tensor(0.0)).item()
            result['prior_loss_p'] = loss_dict.get('prior_loss_p', torch.tensor(0.0)).item()
        
        # 應用權重字典
        applied_weights_dict = {
            'data': w_data,
            'momentum_x': w_momentum_x,
            'momentum_y': w_momentum_y,
            'momentum_z': w_momentum_z,
            'divergence': w_div,
        }
        if hasattr(self.physics, 'compute_periodic_loss'):
            applied_weights_dict['periodic_x'] = w_periodic_x
            applied_weights_dict['periodic_y'] = w_periodic_y
        else:
            applied_weights_dict['wall_constraint'] = w_bc
        
        result['applied_weights'] = applied_weights_dict  # type: ignore
        
        return total_loss, result
