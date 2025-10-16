"""
PINNs 損失函數模組

整合所有損失函數組件，提供統一的接口與管理功能。
支援物理殘差、先驗約束、動態權重平衡等完整的損失計算框架。

主要組件：
- residuals: NS方程殘差、邊界條件、初始條件
- priors: 低保真一致性、統計約束、守恆定律
- weighting: GradNorm、因果權重、NTK策略

使用範例：
```python
from pinnx.losses import create_loss_manager

# 創建損失管理器
loss_manager = create_loss_manager(
    model=pinn_model,
    config={
        'loss_terms': ['data', 'residual', 'boundary', 'prior'],
        'weighting_strategy': 'gradnorm',
        'physics': {'nu': 1e-3, 'equations': 'ns_2d'}
    }
)

# 計算總損失
total_loss, loss_dict = loss_manager.compute_loss(batch)
```
"""

from .residuals import (
    ns_residual_2d,
    ns_residual_3d,
    BoundaryConditionLoss,
    InitialConditionLoss,
    PeriodicBoundaryLoss,
    gradient
)

from .priors import (
    LowFidelityConsistencyLoss,
    StatisticalConsistencyLoss,
    ConservationLoss,
    SymmetryConsistencyLoss,
    PriorLossManager
)

from .weighting import (
    GradNormWeighter,
    CausalWeighter,
    NTKWeighter,
    AdaptiveWeightScheduler,
    MultiWeightManager,
    create_weight_manager
)

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
import logging

logger = logging.getLogger(__name__)


class CompleteLossManager:
    """
    完整的損失管理器
    
    整合所有損失項，提供統一的計算接口與動態權重管理。
    支援 PINNs 逆問題中的多項約束與自適應訓練策略。
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: Dict[str, Any]):
        """
        Args:
            model: PINN 模型
            config: 損失配置字典
        """
        self.model = model
        self.config = config
        
        # 提取配置
        self.loss_terms = config.get('loss_terms', ['data', 'residual', 'boundary'])
        self.physics_config = config.get('physics', {})
        self.weighting_config = config.get('weighting', {})
        self.prior_config = config.get('priors', {})
        
        # 初始化物理參數
        self.nu = self.physics_config.get('nu', 1e-3)
        self.equations = self.physics_config.get('equations', 'ns_2d')
        
        # 初始化損失組件
        self._init_loss_components()
        
        # 初始化權重管理器
        if self.weighting_config.get('enabled', True):
            self.weight_manager = create_weight_manager(
                model=model,
                loss_names=self.loss_terms,
                config=self.weighting_config
            )
        else:
            self.weight_manager = None
        
        # 初始化先驗損失管理器
        if self.prior_config.get('enabled', False):
            self.prior_manager = PriorLossManager(
                consistency_weight=self.prior_config.get('consistency_weight', 1.0),
                statistical_weight=self.prior_config.get('statistical_weight', 0.5),
                conservation_weight=self.prior_config.get('conservation_weight', 0.3),
                symmetry_weight=self.prior_config.get('symmetry_weight', 0.2)
            )
        else:
            self.prior_manager = None
        
        # 計數器
        self.step_count = 0
        
    def _init_loss_components(self):
        """初始化損失組件"""
        
        # 邊界條件損失
        if 'boundary' in self.loss_terms:
            self.boundary_loss = BoundaryConditionLoss()
        
        # 初始條件損失
        if 'initial' in self.loss_terms:
            self.initial_loss = InitialConditionLoss()
        
        # 週期邊界損失
        if 'periodic' in self.loss_terms:
            self.periodic_loss = PeriodicBoundaryLoss()
        
        logger.info(f"初始化損失組件: {self.loss_terms}")
    
    def compute_residual_loss(self, 
                            points: torch.Tensor,
                            time_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        計算物理殘差損失
        
        Args:
            points: 內部點 [N, 2] 或 [N, 3]
            time_points: 時間點 [N, 1] (非定常情況)
            
        Returns:
            殘差損失
        """
        points = points.requires_grad_(True)
        
        if time_points is not None:
            # 非定常情況
            time_points = time_points.requires_grad_(True)
            inputs = torch.cat([time_points, points], dim=-1)
        else:
            # 定常情況
            inputs = points
        
        # 模型預測
        predictions = self.model(inputs)
        
        # 計算殘差
        if self.equations == 'ns_2d':
            if time_points is not None:
                # 非定常 2D NS
                residuals = ns_residual_2d(points, predictions, self.nu, time_points)
            else:
                # 定常 2D NS
                residuals = ns_residual_2d(points, predictions, self.nu)
        elif self.equations == 'ns_3d':
            if time_points is not None:
                residuals = ns_residual_3d(points, predictions, self.nu, time_points)
            else:
                residuals = ns_residual_3d(points, predictions, self.nu)
        else:
            raise ValueError(f"未支援的方程式類型: {self.equations}")
        
        # 計算 MSE
        total_residual = 0.0
        for residual in residuals:
            total_residual += torch.mean(residual**2)
        
        return total_residual
    
    def compute_data_loss(self, 
                         obs_points: torch.Tensor,
                         obs_values: torch.Tensor,
                         time_points: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        計算資料擬合損失
        
        Args:
            obs_points: 觀測點 [K, 2] 或 [K, 3]
            obs_values: 觀測值 [K, n_vars]
            time_points: 時間點 [K, 1] (非定常情況)
            
        Returns:
            資料損失
        """
        if time_points is not None:
            inputs = torch.cat([time_points, obs_points], dim=-1)
        else:
            inputs = obs_points
        
        predictions = self.model(inputs)
        
        # 只比較速度和壓力場 (u, v, p)
        pred_fields = predictions[..., :obs_values.shape[-1]]
        
        return torch.mean((pred_fields - obs_values)**2)
    
    def compute_boundary_loss(self, 
                            boundary_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算邊界條件損失
        
        Args:
            boundary_data: 邊界數據字典
            
        Returns:
            邊界損失
        """
        if not hasattr(self, 'boundary_loss'):
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        total_boundary_loss = 0.0
        
        for bc_type, data in boundary_data.items():
            points = data['points']
            values = data.get('values', None)
            normals = data.get('normals', None)
            
            if bc_type == 'no_slip':
                loss = self.boundary_loss.no_slip_loss(self.model, points)
            elif bc_type == 'pressure_outlet':
                loss = self.boundary_loss.pressure_outlet_loss(self.model, points, values)
            elif bc_type == 'velocity_inlet':
                loss = self.boundary_loss.velocity_inlet_loss(self.model, points, values)
            elif bc_type == 'wall_shear':
                loss = self.boundary_loss.wall_shear_loss(self.model, points, values, normals)
            else:
                logger.warning(f"未知的邊界條件類型: {bc_type}")
                continue
            
            total_boundary_loss += loss
        
        return total_boundary_loss
    
    def compute_prior_loss(self, 
                          batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算先驗約束損失
        
        Args:
            batch_data: 批次數據字典
            
        Returns:
            先驗損失
        """
        if self.prior_manager is None:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)
        
        return self.prior_manager.compute_total_loss(self.model, batch_data)
    
    def compute_loss(self, 
                    batch: Dict[str, Any],
                    return_components: bool = True) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        計算總損失
        
        Args:
            batch: 批次數據字典
            return_components: 是否返回損失組件
            
        Returns:
            總損失 或 (總損失, 損失組件字典)
        """
        self.step_count += 1
        device = next(self.model.parameters()).device
        
        # 損失組件字典
        losses = {}
        
        # 1. 資料損失
        if 'data' in self.loss_terms and 'obs_points' in batch:
            obs_points = batch['obs_points'].to(device)
            obs_values = batch['obs_values'].to(device)
            time_points = batch.get('obs_time', None)
            if time_points is not None:
                time_points = time_points.to(device)
            
            losses['data'] = self.compute_data_loss(obs_points, obs_values, time_points)
        
        # 2. 物理殘差損失
        if 'residual' in self.loss_terms and 'residual_points' in batch:
            res_points = batch['residual_points'].to(device)
            time_points = batch.get('residual_time', None)
            if time_points is not None:
                time_points = time_points.to(device)
            
            losses['residual'] = self.compute_residual_loss(res_points, time_points)
        
        # 3. 邊界條件損失
        if 'boundary' in self.loss_terms and 'boundary_data' in batch:
            boundary_data = {}
            for key, data in batch['boundary_data'].items():
                boundary_data[key] = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                                    for k, v in data.items()}
            
            losses['boundary'] = self.compute_boundary_loss(boundary_data)
        
        # 4. 初始條件損失
        if 'initial' in self.loss_terms and 'initial_data' in batch:
            ic_batch = batch['initial_data']
            coords = ic_batch.get('coords', ic_batch.get('points'))
            values = ic_batch.get('values')
            if coords is None or values is None:
                logger.warning("初始條件資料缺少 'coords' 或 'values'，跳過初始條件損失計算")
                losses['initial'] = torch.tensor(0.0, device=device)
            else:
                coords = coords.to(device)
                values = values.to(device)
                
                time_points = ic_batch.get('time')
                if time_points is not None:
                    time_points = time_points.to(device)
                    model_inputs = torch.cat([time_points, coords], dim=-1)
                else:
                    model_inputs = coords
                
                predictions = self.model(model_inputs)
                
                weight_cfg = ic_batch.get('weights')
                if isinstance(weight_cfg, dict):
                    ic_weights: Dict[str, float] = {}
                    for key, val in weight_cfg.items():
                        if torch.is_tensor(val):
                            ic_weights[key] = float(val.item())
                        else:
                            ic_weights[key] = float(val)
                else:
                    ic_weights = None
                
                ic_losses = self.initial_loss(
                    initial_coords=model_inputs,
                    initial_predictions=predictions,
                    initial_data=values,
                    weights=ic_weights
                )
                losses['initial'] = ic_losses['total_ic']
                if return_components:
                    losses['initial_velocity'] = ic_losses['ic_velocity']
                    losses['initial_pressure'] = ic_losses['ic_pressure']
        
        # 5. 先驗約束損失
        if 'prior' in self.loss_terms:
            batch_data = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in batch.items()}
            losses['prior'] = self.compute_prior_loss(batch_data)
        
        # 確保所有損失項都存在
        for term in self.loss_terms:
            if term not in losses:
                losses[term] = torch.tensor(0.0, device=device)
        
        # 動態權重更新
        if self.weight_manager is not None:
            # 準備時間序列損失（用於因果權重）
            time_losses = []
            if 'residual_time_chunks' in batch:
                for chunk_data in batch['residual_time_chunks']:
                    chunk_points = chunk_data['points'].to(device)
                    chunk_time = chunk_data['time'].to(device)
                    chunk_loss = self.compute_residual_loss(chunk_points, chunk_time)
                    time_losses.append(chunk_loss)
            
            # 準備輸入數據（用於 NTK）
            data_inputs = {}
            if 'obs_points' in batch:
                data_inputs['data'] = batch['obs_points'].to(device)
            if 'residual_points' in batch:
                data_inputs['residual'] = batch['residual_points'].to(device)
            
            # 更新權重
            weights = self.weight_manager.update_weights(
                losses=losses,
                current_step=self.step_count,
                time_losses=time_losses if time_losses else None,
                data_inputs=data_inputs if data_inputs else None
            )
        else:
            # 使用固定權重
            weights = {name: 1.0 for name in self.loss_terms}
        
        # 計算加權總損失
        total_loss = 0.0
        for name, loss in losses.items():
            if name in weights:
                total_loss += weights[name] * loss
        
        if return_components:
            # 添加權重信息到組件字典
            loss_components = {f"loss/{name}": loss.item() for name, loss in losses.items()}
            loss_components.update({f"weight/{name}": weights[name] for name in weights})
            loss_components['total_loss'] = total_loss.item()
            
            return total_loss, loss_components
        else:
            return total_loss
    
    def get_current_weights(self) -> Dict[str, float]:
        """獲取當前權重"""
        if self.weight_manager is not None:
            return self.weight_manager.get_weights()
        else:
            return {name: 1.0 for name in self.loss_terms}
    
    def reset_weights(self):
        """重置權重"""
        if self.weight_manager is not None:
            if hasattr(self.weight_manager, 'weighters') and 'gradnorm' in self.weight_manager.weighters:
                self.weight_manager.weighters['gradnorm'].reset_weights()


class MeanConstraintLoss(nn.Module):
    """
    顯式均值約束損失，防止全局統計偏移
    
    用於修正 Fourier Features 導致的均值漂移問題。
    錨定低頻分量（全局統計量），同時不影響高頻學習。
    
    設計原理：
    - Fourier Features 增強高頻捕捉，但可能干擾低頻（均值）穩定性
    - 顯式約束預測場均值匹配參考值（如 JHTDB 時間平均值）
    - 僅作用於全局統計，不影響局部梯度學習
    
    使用場景：
    - 湍流場重建中速度/壓力場均值偏移嚴重
    - VS-PINN 縮放空間中均值漂移放大
    - 需要錨定全局統計量同時保持物理結構細節
    
    數學形式：
        L_mean = Σ_i (mean(pred_i) - target_mean_i)^2
    
    其中 i 遍歷所有需要約束的場變量（如 u, v, w）。
    
    Args:
        None (stateless loss function)
    
    Example:
        >>> mean_loss = MeanConstraintLoss()
        >>> predictions = model(inputs)  # [N, 4] (u, v, w, p)
        >>> target_means = {'u': 9.84, 'v': 0.0, 'w': 0.0}
        >>> field_indices = {'u': 0, 'v': 1, 'w': 2}
        >>> loss = mean_loss(predictions, target_means, field_indices)
    """
    
    def __init__(self):
        super().__init__()
        logger.info("初始化 MeanConstraintLoss")
    
    def forward(self, 
                predictions: torch.Tensor,
                target_means: Dict[str, float],
                field_indices: Dict[str, int]) -> torch.Tensor:
        """
        計算均值約束損失
        
        Args:
            predictions: [N, n_vars] 網路輸出（物理空間）
            target_means: 目標均值字典，如 {'u': 9.84, 'v': 0.0, 'w': 0.0}
            field_indices: 場索引字典，如 {'u': 0, 'v': 1, 'w': 2, 'p': 3}
        
        Returns:
            mean_loss: 標量張量，所有場的均值約束損失總和
        
        Notes:
            - 計算採用 MSE 形式：(pred_mean - target_mean)^2
            - 僅約束指定的場（通常排除壓力場）
            - 損失在批次維度上求均值，確保尺度穩定
        """
        device = predictions.device
        loss = torch.tensor(0.0, device=device)
        
        for field_name, target_mean in target_means.items():
            if field_name not in field_indices:
                logger.warning(f"場 '{field_name}' 不在 field_indices 中，跳過約束")
                continue
            
            idx = field_indices[field_name]
            pred_mean = predictions[:, idx].mean()
            target = torch.tensor(target_mean, device=device)
            
            # MSE: (pred_mean - target_mean)^2
            field_loss = (pred_mean - target) ** 2
            loss = loss + field_loss
            
            # 記錄詳細資訊（低頻率，避免日誌爆炸）
            if torch.rand(1).item() < 0.001:  # 0.1% 機率記錄
                logger.debug(
                    f"均值約束 | {field_name}: "
                    f"pred={pred_mean.item():.4f}, "
                    f"target={target_mean:.4f}, "
                    f"loss={field_loss.item():.6f}"
                )
        
        return loss


def create_loss_manager(model: nn.Module, 
                       config: Dict[str, Any]) -> CompleteLossManager:
    """
    創建損失管理器的便捷函數
    
    Args:
        model: PINN 模型
        config: 配置字典
        
    Returns:
        配置好的損失管理器
    """
    return CompleteLossManager(model, config)


def create_default_config() -> Dict[str, Any]:
    """創建預設損失配置"""
    return {
        'loss_terms': ['data', 'residual', 'boundary'],
        'physics': {
            'nu': 1e-3,
            'equations': 'ns_2d'
        },
        'weighting': {
            'enabled': True,
            'strategies': ['gradnorm', 'adaptive'],
            'gradnorm': {
                'alpha': 0.12,
                'update_frequency': 1000
            }
        },
        'priors': {
            'enabled': False,
            'consistency_weight': 1.0,
            'statistical_weight': 0.5,
            'conservation_weight': 0.3,
            'symmetry_weight': 0.2
        }
    }


__all__ = [
    # 殘差損失
    'ns_residual_2d',
    'ns_residual_3d', 
    'BoundaryConditionLoss',
    'InitialConditionLoss',
    'PeriodicBoundaryLoss',
    'gradient',
    
    # 先驗損失
    'LowFidelityConsistencyLoss',
    'StatisticalConsistencyLoss',
    'ConservationLoss',
    'SymmetryConsistencyLoss',
    'PriorLossManager',
    
    # 均值約束
    'MeanConstraintLoss',
    
    # 動態權重
    'GradNormWeighter',
    'CausalWeighter',
    'NTKWeighter',
    'AdaptiveWeightScheduler',
    'MultiWeightManager',
    'create_weight_manager',
    
    # 主要接口
    'CompleteLossManager',
    'create_loss_manager',
    'create_default_config'
]
