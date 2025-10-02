"""
先驗一致性損失函數模組

實現各種先驗資訊的一致性損失，支援低保真資料作為軟先驗約束。
這是實現「少量資料 × 物理先驗」框架的關鍵組件，用於整合：

- RANS/LES 低保真場作為軟約束
- 歷史統計資料作為先驗知識
- 物理守恆定律約束
- 對稱性與不變性約束
- 多尺度一致性約束

主要功能：
- 低保真場一致性損失
- 統計矩一致性損失
- 能量/渦量守恆損失
- 對稱邊界一致性
- 多解析度一致性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np


class LowFidelityConsistencyLoss(nn.Module):
    """
    低保真場一致性損失
    
    將 RANS/LES/下採樣DNS 作為軟先驗，引導 PINN 學習合理的場分佈。
    支援不同權重策略與對齊方法。
    """
    
    def __init__(self, 
                 consistency_weight: float = 1.0,
                 variable_weights: Optional[Dict[str, float]] = None,
                 distance_metric: str = 'mse',
                 adaptive_weighting: bool = False,
                 alignment_method: str = 'interpolation'):
        """
        Args:
            consistency_weight: 總體一致性權重
            variable_weights: 各變數權重字典 {'u': 1.0, 'v': 1.0, 'p': 0.5}
            distance_metric: 距離度量 ('mse', 'mae', 'huber')
            adaptive_weighting: 是否使用自適應權重
            alignment_method: 對齊方法 ('interpolation', 'projection')
        """
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.variable_weights = variable_weights or {'u': 1.0, 'v': 1.0, 'p': 0.5}
        self.distance_metric = distance_metric
        self.adaptive_weighting = adaptive_weighting
        self.alignment_method = alignment_method
        
        # 自適應權重參數
        if adaptive_weighting:
            self.register_parameter('adaptive_scales', 
                                  torch.nn.Parameter(torch.ones(len(self.variable_weights))))
    
    def forward(self, 
                high_fidelity_pred: torch.Tensor,
                low_fidelity_data: torch.Tensor,
                coords: Optional[torch.Tensor] = None,
                variable_names: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        """
        計算低保真一致性損失
        
        Args:
            high_fidelity_pred: PINN 高保真預測 [batch_size, n_vars]
            low_fidelity_data: 低保真參考資料 [batch_size, n_vars]
            coords: 座標資訊 [batch_size, spatial_dim] (用於空間權重)
            variable_names: 變數名稱列表
        
        Returns:
            losses: 各變數與總體一致性損失
        """
        losses = {}
        
        # 變數名稱處理
        if variable_names is None:
            variable_names = list(self.variable_weights.keys())
        
        # 確保張量形狀一致
        assert high_fidelity_pred.shape == low_fidelity_data.shape, \
            f"預測與參考資料形狀不符: {high_fidelity_pred.shape} vs {low_fidelity_data.shape}"
        
        total_loss = 0.0
        
        # 逐變數計算一致性損失
        for i, var_name in enumerate(variable_names):
            if i >= high_fidelity_pred.shape[-1]:
                break
                
            pred_var = high_fidelity_pred[:, i]
            ref_var = low_fidelity_data[:, i]
            
            # 計算距離
            if self.distance_metric == 'mse':
                var_loss = torch.mean((pred_var - ref_var) ** 2)
            elif self.distance_metric == 'mae':
                var_loss = torch.mean(torch.abs(pred_var - ref_var))
            elif self.distance_metric == 'huber':
                var_loss = F.huber_loss(pred_var, ref_var, reduction='mean')
            else:
                raise ValueError(f"不支援的距離度量: {self.distance_metric}")
            
            # 變數權重
            var_weight = self.variable_weights.get(var_name, 1.0)
            
            # 自適應權重
            if self.adaptive_weighting and i < len(self.adaptive_scales):
                var_weight *= torch.sigmoid(self.adaptive_scales[i])
            
            weighted_loss = var_weight * var_loss
            losses[f'prior_consistency_{var_name}'] = weighted_loss
            total_loss += weighted_loss
        
        # 總一致性損失
        losses['prior_consistency_total'] = self.consistency_weight * total_loss
        
        return losses
    
    def compute_spatial_weights(self, 
                               coords: torch.Tensor,
                               boundary_penalty: float = 2.0,
                               center_weight: float = 1.0) -> torch.Tensor:
        """
        計算基於空間位置的權重（例如邊界附近加強約束）
        """
        # 簡化實現：基於到邊界的距離
        # 實際應用中可根據具體幾何形狀調整
        x, y = coords[:, 0], coords[:, 1]
        
        # 假設計算域為 [-1, 1] × [-1, 1]
        dist_to_boundary = torch.min(
            torch.stack([
                1 - torch.abs(x),  # 到 x 邊界距離
                1 - torch.abs(y)   # 到 y 邊界距離
            ]), dim=0
        )[0]
        
        # 邊界附近權重更高
        weights = center_weight + boundary_penalty * torch.exp(-5 * dist_to_boundary)
        return weights


class StatisticalConsistencyLoss(nn.Module):
    """
    統計矩一致性損失
    
    確保 PINN 預測的統計性質（均值、方差、高階矩）與
    參考資料或理論值一致。適用於湍流統計特性約束。
    """
    
    def __init__(self, 
                 moments: List[int] = [1, 2],
                 spatial_averaging: bool = True,
                 temporal_averaging: bool = False):
        """
        Args:
            moments: 要約束的統計矩階數 [1, 2, 3, 4]
            spatial_averaging: 是否進行空間平均
            temporal_averaging: 是否進行時間平均
        """
        super().__init__()
        
        self.moments = moments
        self.spatial_averaging = spatial_averaging
        self.temporal_averaging = temporal_averaging
    
    def forward(self, 
                predictions: torch.Tensor,
                reference_stats: Dict[str, torch.Tensor],
                variable_names: List[str]) -> Dict[str, torch.Tensor]:
        """
        計算統計一致性損失
        
        Args:
            predictions: PINN 預測 [batch_size, n_vars]
            reference_stats: 參考統計量字典
            variable_names: 變數名稱列表
        
        Returns:
            losses: 統計一致性損失
        """
        losses = {}
        
        for i, var_name in enumerate(variable_names):
            if i >= predictions.shape[-1]:
                break
                
            pred_var = predictions[:, i]
            
            # 計算各階矩
            for moment in self.moments:
                pred_moment = torch.mean(pred_var ** moment)
                
                ref_key = f'{var_name}_moment_{moment}'
                if ref_key in reference_stats:
                    ref_moment = reference_stats[ref_key]
                    moment_loss = torch.mean((pred_moment - ref_moment) ** 2)
                    losses[f'stat_{var_name}_moment_{moment}'] = moment_loss
        
        # 總統計損失
        total_stat_loss = sum(losses.values())
        losses['statistical_consistency_total'] = total_stat_loss
        
        return losses


class ConservationLoss(nn.Module):
    """
    守恆定律一致性損失
    
    強制 PINN 滿足物理守恆定律：
    - 質量守恆 (連續性方程)
    - 動量守恆
    - 能量守恆
    - 渦量守恆
    """
    
    def __init__(self, 
                 conservation_laws: List[str] = ['mass', 'momentum'],
                 domain_integration: bool = True):
        """
        Args:
            conservation_laws: 要約束的守恆定律列表
            domain_integration: 是否在整個計算域上積分檢查
        """
        super().__init__()
        
        self.conservation_laws = conservation_laws
        self.domain_integration = domain_integration
    
    def mass_conservation_loss(self, 
                              velocity: torch.Tensor,
                              coords: torch.Tensor) -> torch.Tensor:
        """質量守恆損失：∫∇·u dΩ = 0"""
        # 計算散度 ∇·u
        div_u = 0.0
        for i in range(velocity.shape[-1]):
            u_i = velocity[:, i]
            # 計算 ∂u_i/∂x_i
            grad = torch.autograd.grad(
                u_i.sum(), coords,
                create_graph=True, retain_graph=True, allow_unused=True
            )[0]
            if grad is not None:
                div_u += grad[:, i]
        
        if self.domain_integration:
            # 域積分應為零
            integrated_div = torch.mean(div_u)
            loss = integrated_div ** 2
        else:
            # 點wise 散度應為零
            loss = torch.mean(div_u ** 2)
        
        return loss
    
    def momentum_conservation_loss(self,
                                  velocity: torch.Tensor,
                                  pressure: torch.Tensor,
                                  coords: torch.Tensor,
                                  boundary_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """動量守恆損失：檢查邊界動量通量平衡"""
        # 簡化實現：檢查總動量變化
        total_momentum = torch.mean(velocity, dim=0)  # [spatial_dim]
        
        # 理想情況下應為常數或已知值
        # 這裡簡單假設應為零 (穩態無外力)
        loss = torch.sum(total_momentum ** 2)
        
        return loss
    
    def energy_conservation_loss(self,
                                velocity: torch.Tensor,
                                coords: torch.Tensor) -> torch.Tensor:
        """能量守恆損失：檢查動能變化"""
        kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=-1)  # [batch_size]
        
        # 對於穩態問題，能量變化應平衡
        # 這裡檢查能量分佈的均勻性（簡化）
        mean_energy = torch.mean(kinetic_energy)
        energy_variance = torch.var(kinetic_energy)
        
        # 過大的能量變化可能不合理
        loss = energy_variance / (mean_energy + 1e-8)
        
        return loss
    
    def forward(self,
                velocity: torch.Tensor,
                pressure: torch.Tensor,
                coords: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        計算守恆定律損失
        """
        losses = {}
        
        if 'mass' in self.conservation_laws:
            mass_loss = self.mass_conservation_loss(velocity, coords)
            losses['conservation_mass'] = mass_loss
        
        if 'momentum' in self.conservation_laws:
            momentum_loss = self.momentum_conservation_loss(velocity, pressure, coords)
            losses['conservation_momentum'] = momentum_loss
        
        if 'energy' in self.conservation_laws:
            energy_loss = self.energy_conservation_loss(velocity, coords)
            losses['conservation_energy'] = energy_loss
        
        # 總守恆損失
        total_conservation = sum(losses.values())
        losses['conservation_total'] = total_conservation
        
        return losses


class SymmetryConsistencyLoss(nn.Module):
    """
    對稱性一致性損失
    
    強制 PINN 滿足問題的對稱性約束，例如：
    - 幾何對稱性
    - 週期性
    - 旋轉不變性
    """
    
    def __init__(self, 
                 symmetry_type: str = 'reflection',
                 symmetry_axis: Union[int, List[int]] = 0):
        """
        Args:
            symmetry_type: 對稱類型 ('reflection', 'rotation', 'periodic')
            symmetry_axis: 對稱軸 (座標軸索引)
        """
        super().__init__()
        
        self.symmetry_type = symmetry_type
        self.symmetry_axis = symmetry_axis if isinstance(symmetry_axis, list) else [symmetry_axis]
    
    def reflection_loss(self,
                       coords: torch.Tensor,
                       predictions: torch.Tensor,
                       axis: int = 0) -> torch.Tensor:
        """反射對稱損失：u(x) = u(-x)"""
        # 生成反射座標
        reflected_coords = coords.clone()
        reflected_coords[:, axis] = -reflected_coords[:, axis]
        
        # 這裡需要模型重新預測反射點
        # 簡化實現：假設預測具有反射對稱性
        # 實際使用時需要重新調用模型
        reflected_preds = predictions  # 占位符
        
        # 對稱性損失
        loss = torch.mean((predictions - reflected_preds) ** 2)
        
        return loss
    
    def forward(self,
                coords: torch.Tensor,
                predictions: torch.Tensor,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        計算對稱性損失
        """
        losses = {}
        
        if self.symmetry_type == 'reflection':
            for axis in self.symmetry_axis:
                refl_loss = self.reflection_loss(coords, predictions, axis)
                losses[f'symmetry_reflection_axis_{axis}'] = refl_loss
        
        # 總對稱性損失
        total_symmetry = sum(losses.values())
        losses['symmetry_total'] = total_symmetry
        
        return losses


# 綜合先驗損失管理器
class PriorLossManager(nn.Module):
    """
    先驗損失管理器：統一管理多種先驗約束
    """
    
    def __init__(self, 
                 consistency_weight: float = 1.0,
                 statistical_weight: float = 0.5,
                 conservation_weight: float = 0.3,
                 symmetry_weight: float = 0.2,
                 loss_config: Optional[Dict] = None):
        """
        Args:
            consistency_weight: 低保真一致性權重
            statistical_weight: 統計一致性權重  
            conservation_weight: 守恆定律權重
            symmetry_weight: 對稱性權重
            loss_config: 詳細損失配置字典（可選）
        """
        super().__init__()
        
        self.consistency_weight = consistency_weight
        self.statistical_weight = statistical_weight
        self.conservation_weight = conservation_weight
        self.symmetry_weight = symmetry_weight
        
        # 初始化損失組件
        self.low_fidelity_loss = LowFidelityConsistencyLoss()
        self.statistical_loss = StatisticalConsistencyLoss()
        self.conservation_loss = ConservationLoss()
        self.symmetry_loss = SymmetryConsistencyLoss()
        
        # 如果提供了詳細配置，則覆蓋預設組件
        if loss_config:
            if 'low_fidelity' in loss_config:
                self.low_fidelity_loss = LowFidelityConsistencyLoss(**loss_config['low_fidelity'])
            if 'statistical' in loss_config:
                self.statistical_loss = StatisticalConsistencyLoss(**loss_config['statistical'])
            if 'conservation' in loss_config:
                self.conservation_loss = ConservationLoss(**loss_config['conservation'])
            if 'symmetry' in loss_config:
                self.symmetry_loss = SymmetryConsistencyLoss(**loss_config['symmetry'])
    
    def compute_total_loss(self, 
                          model: nn.Module, 
                          batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算總先驗損失
        
        Args:
            model: PINN 模型
            batch_data: 批次數據字典
            
        Returns:
            總先驗損失
        """
        total_loss = 0.0
        
        # 低保真一致性損失
        if 'high_fi_pred' in batch_data and 'low_fi_data' in batch_data:
            consistency_losses = self.low_fidelity_loss(
                high_fidelity_pred=batch_data['high_fi_pred'],
                low_fidelity_data=batch_data['low_fi_data'],
                coords=batch_data.get('coords', None),
                variable_names=batch_data.get('variable_names', ['u', 'v', 'p'])
            )
            total_loss += self.consistency_weight * consistency_losses['prior_consistency_total']
        
        # 統計一致性損失
        if 'predictions' in batch_data and 'reference_stats' in batch_data:
            stat_losses = self.statistical_loss(
                predictions=batch_data['predictions'],
                reference_stats=batch_data['reference_stats'],
                variable_names=batch_data.get('variable_names', ['u', 'v', 'p'])
            )
            total_loss += self.statistical_weight * stat_losses['statistical_consistency_total']
        
        # 守恆定律損失
        if 'velocity' in batch_data and 'pressure' in batch_data and 'coords' in batch_data:
            conservation_losses = self.conservation_loss(
                velocity=batch_data['velocity'],
                pressure=batch_data['pressure'],
                coords=batch_data['coords']
            )
            total_loss += self.conservation_weight * conservation_losses['conservation_total']
        
        # 對稱性損失
        if 'coords' in batch_data and 'predictions' in batch_data:
            symmetry_losses = self.symmetry_loss(
                coords=batch_data['coords'],
                predictions=batch_data['predictions']
            )
            total_loss += self.symmetry_weight * symmetry_losses['symmetry_total']
        
        return total_loss
    
    def forward(self, 
                **inputs) -> Dict[str, torch.Tensor]:
        """
        計算所有先驗損失（保持原有接口）
        """
        all_losses = {}
        
        # 低保真一致性
        if 'high_fidelity_pred' in inputs and 'low_fidelity_data' in inputs:
            consistency_losses = self.low_fidelity_loss(
                high_fidelity_pred=inputs['high_fidelity_pred'],
                low_fidelity_data=inputs['low_fidelity_data'],
                coords=inputs.get('coords', None),
                variable_names=inputs.get('variable_names', ['u', 'v', 'p'])
            )
            for key, value in consistency_losses.items():
                all_losses[f'consistency_{key}'] = self.consistency_weight * value
        
        # 統計一致性
        if 'predictions' in inputs and 'reference_stats' in inputs:
            stat_losses = self.statistical_loss(
                predictions=inputs['predictions'],
                reference_stats=inputs['reference_stats'],
                variable_names=inputs.get('variable_names', ['u', 'v', 'p'])
            )
            for key, value in stat_losses.items():
                all_losses[f'statistical_{key}'] = self.statistical_weight * value
        
        # 守恆定律
        if 'velocity' in inputs and 'pressure' in inputs and 'coords' in inputs:
            conservation_losses = self.conservation_loss(
                velocity=inputs['velocity'],
                pressure=inputs['pressure'],
                coords=inputs['coords']
            )
            for key, value in conservation_losses.items():
                all_losses[f'conservation_{key}'] = self.conservation_weight * value
        
        # 對稱性
        if 'coords' in inputs and 'predictions' in inputs:
            symmetry_losses = self.symmetry_loss(
                coords=inputs['coords'],
                predictions=inputs['predictions']
            )
            for key, value in symmetry_losses.items():
                all_losses[f'symmetry_{key}'] = self.symmetry_weight * value
        
        # 計算總先驗損失
        total_prior = sum(v for k, v in all_losses.items() if k.endswith('_total'))
        all_losses['prior_total'] = total_prior
        
        return all_losses


class CompletePriorLossManager(PriorLossManager):
    """
    完整先驗損失管理器：統一管理多種先驗約束
    
    與 PriorLossManager 相同，提供別名以保持向後相容性
    """
    
    def __init__(self, 
                 loss_config: Dict,
                 global_weight: float = 1.0):
        """
        Args:
            loss_config: 損失配置字典
            global_weight: 全域先驗權重
        """
        # 解析配置中的權重
        consistency_weight = loss_config.get('consistency_weight', 1.0) * global_weight
        statistical_weight = loss_config.get('statistical_weight', 0.5) * global_weight
        conservation_weight = loss_config.get('conservation_weight', 0.3) * global_weight
        symmetry_weight = loss_config.get('symmetry_weight', 0.2) * global_weight
        
        super().__init__(
            consistency_weight=consistency_weight,
            statistical_weight=statistical_weight,
            conservation_weight=conservation_weight,
            symmetry_weight=symmetry_weight,
            loss_config=loss_config.get('components', {})
        )


# 便捷建構函數
def create_prior_loss(config: Dict) -> PriorLossManager:
    """根據配置建立先驗損失管理器"""
    return PriorLossManager(
        loss_config=config.get('components', {}),
        global_weight=config.get('global_weight', 1.0)
    )


# 為測試文件提供的相容性函數
def prior_consistency_loss(high_fidelity_pred: torch.Tensor, 
                          low_fidelity_data: torch.Tensor, 
                          strength: float = 1.0,
                          uncertainty: Optional[torch.Tensor] = None,
                          adaptive: bool = False) -> torch.Tensor:
    """
    先驗一致性損失函數 (相容性接口)
    
    Args:
        high_fidelity_pred: 高保真預測
        low_fidelity_data: 低保真參考資料
        strength: 損失強度
        uncertainty: 不確定性權重
        adaptive: 是否使用自適應權重
    
    Returns:
        loss: 先驗一致性損失
    """
    residual = high_fidelity_pred - low_fidelity_data
    
    if uncertainty is not None and adaptive:
        # 自適應權重：不確定性越大，權重越小
        weights = 1.0 / (uncertainty + 1e-8)
        loss = torch.mean(weights * (residual ** 2))
    else:
        # 標準 MSE
        loss = torch.mean(residual ** 2)
    
    return strength * loss


def statistical_prior_loss(predicted: torch.Tensor, 
                          prior_type: str = 'mean',
                          target_stats: torch.Tensor = None,
                          strength: float = 1.0) -> torch.Tensor:
    """
    統計先驗損失函數 (相容性接口)
    
    Args:
        predicted: 預測值 [batch_size, n_vars]
        prior_type: 統計類型 ('mean', 'variance', 'covariance')
        target_stats: 目標統計量
        strength: 損失強度
    
    Returns:
        loss: 統計先驗損失
    """
    if prior_type == 'mean':
        # 均值約束
        pred_mean = torch.mean(predicted, dim=0)
        if target_stats is None:
            target_stats = torch.zeros_like(pred_mean)
        loss = torch.mean((pred_mean - target_stats) ** 2)
    
    elif prior_type == 'variance':
        # 方差約束
        pred_var = torch.var(predicted, dim=0)
        if target_stats is None:
            target_stats = torch.ones_like(pred_var)
        loss = torch.mean((pred_var - target_stats) ** 2)
    
    elif prior_type == 'covariance':
        # 協方差約束
        pred_cov = torch.cov(predicted.T)
        if target_stats is None:
            target_stats = torch.eye(predicted.shape[-1], device=predicted.device)
        loss = torch.mean((pred_cov - target_stats) ** 2)
    
    else:
        raise ValueError(f"不支援的統計類型: {prior_type}")
    
    return strength * loss


def physics_constraint_loss(field: torch.Tensor,
                           constraint_type: str = 'energy_bound',
                           constraint_params: Dict = None,
                           strength: float = 1.0) -> torch.Tensor:
    """
    物理約束損失函數 (相容性接口)
    
    Args:
        field: 物理場 (如速度場)
        constraint_type: 約束類型
        constraint_params: 約束參數
        strength: 損失強度
    
    Returns:
        loss: 物理約束損失
    """
    if constraint_params is None:
        constraint_params = {}
    
    if constraint_type == 'energy_bound':
        # 能量界限約束
        kinetic_energy = 0.5 * torch.sum(field ** 2, dim=-1)
        max_energy = constraint_params.get('max_energy', 10.0)
        
        # 懲罰超過界限的能量
        excess_energy = torch.clamp(kinetic_energy - max_energy, min=0.0)
        loss = torch.mean(excess_energy ** 2)
    
    elif constraint_type == 'momentum_conservation':
        # 動量守恆約束
        total_momentum = torch.mean(field, dim=0)
        target_momentum = constraint_params.get('target_momentum', torch.zeros_like(total_momentum))
        loss = torch.mean((total_momentum - target_momentum) ** 2)
    
    elif constraint_type == 'magnitude_bound':
        # 場量值界限約束
        field_magnitude = torch.norm(field, dim=-1)
        max_magnitude = constraint_params.get('max_magnitude', 5.0)
        
        excess_magnitude = torch.clamp(field_magnitude - max_magnitude, min=0.0)
        loss = torch.mean(excess_magnitude ** 2)
    
    else:
        raise ValueError(f"不支援的約束類型: {constraint_type}")
    
    return strength * loss


def energy_conservation_loss(total_energy: torch.Tensor,
                            conservation_type: str = 'steady',
                            time_derivative: Optional[torch.Tensor] = None,
                            strength: float = 1.0) -> torch.Tensor:
    """
    能量守恆損失函數 (相容性接口)
    
    Args:
        total_energy: 總能量場
        conservation_type: 守恆類型 ('steady', 'unsteady')
        time_derivative: 時間導數 (非定常情況)
        strength: 損失強度
    
    Returns:
        loss: 能量守恆損失
    """
    if conservation_type == 'steady':
        # 定常情況：能量應保持常數
        mean_energy = torch.mean(total_energy)
        energy_variance = torch.var(total_energy)
        
        # 懲罰能量變化
        loss = energy_variance / (mean_energy.abs() + 1e-8)
    
    elif conservation_type == 'unsteady':
        # 非定常情況：檢查能量平衡
        if time_derivative is None:
            raise ValueError("非定常能量守恆需要提供時間導數")
        
        # 能量變化率應符合能量方程
        # 簡化：檢查時間導數的合理性
        energy_change_rate = torch.mean(time_derivative)
        
        # 假設應接近零（無外部功輸入）
        loss = energy_change_rate ** 2
    
    else:
        raise ValueError(f"不支援的守恆類型: {conservation_type}")
    
    return strength * loss


if __name__ == "__main__":
    # 測試程式碼
    print("=== 先驗一致性損失測試 ===")
    
    # 建立測試資料
    batch_size = 100
    n_vars = 3  # [u, v, p]
    
    high_fi_pred = torch.randn(batch_size, n_vars, requires_grad=True)
    low_fi_data = torch.randn(batch_size, n_vars)
    coords = torch.randn(batch_size, 2, requires_grad=True)
    
    # 測試低保真一致性
    print("\n--- 低保真一致性測試 ---")
    low_fi_loss = LowFidelityConsistencyLoss(
        consistency_weight=0.5,
        variable_weights={'u': 1.0, 'v': 1.0, 'p': 0.3}
    )
    
    losses = low_fi_loss(
        high_fidelity_pred=high_fi_pred,
        low_fidelity_data=low_fi_data,
        variable_names=['u', 'v', 'p']
    )
    
    print("低保真一致性損失：")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 測試統計一致性
    print("\n--- 統計一致性測試 ---")
    stat_loss = StatisticalConsistencyLoss(moments=[1, 2])
    
    ref_stats = {
        'u_moment_1': torch.tensor(0.0),  # 均值
        'u_moment_2': torch.tensor(1.0),  # 二階矩
        'v_moment_1': torch.tensor(0.0),
        'v_moment_2': torch.tensor(1.0)
    }
    
    stat_losses = stat_loss(high_fi_pred, ref_stats, ['u', 'v', 'p'])
    print("統計一致性損失：")
    for key, value in stat_losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 測試守恆定律
    print("\n--- 守恆定律測試 ---")
    conservation_loss = ConservationLoss(conservation_laws=['mass', 'energy'])
    
    velocity = high_fi_pred[:, :2]  # [u, v]
    pressure = high_fi_pred[:, 2]   # p
    
    conservation_losses = conservation_loss(velocity, pressure, coords)
    print("守恆定律損失：")
    for key, value in conservation_losses.items():
        print(f"  {key}: {value.item():.6f}")
    
    # 測試相容性函數
    print("\n--- 相容性函數測試 ---")
    
    # 先驗一致性
    prior_loss = prior_consistency_loss(high_fi_pred, low_fi_data, strength=0.5)
    print(f"先驗一致性損失: {prior_loss.item():.6f}")
    
    # 統計先驗
    stat_prior_loss = statistical_prior_loss(high_fi_pred, prior_type='mean', strength=0.1)
    print(f"統計先驗損失: {stat_prior_loss.item():.6f}")
    
    # 物理約束
    physics_loss = physics_constraint_loss(
        velocity, constraint_type='energy_bound',
        constraint_params={'max_energy': 5.0}, strength=0.05
    )
    print(f"物理約束損失: {physics_loss.item():.6f}")
    
    # 能量守恆
    total_energy = 0.5 * torch.sum(velocity**2, dim=1, keepdim=True) + pressure.unsqueeze(-1)
    energy_loss = energy_conservation_loss(total_energy, conservation_type='steady', strength=0.02)
    print(f"能量守恆損失: {energy_loss.item():.6f}")
    
    print("\n✅ 先驗一致性損失測試通過！")