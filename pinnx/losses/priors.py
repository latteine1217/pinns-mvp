"""
先驗一致性損失函數模組 (Simplified)

實現各種先驗資訊的一致性損失,支援低保真資料作為軟先驗約束。
這是實現「少量資料 × 物理先驗」框架的關鍵組件。

=== 支援的模型 (SUPPORTED MODELS) ===
✅ 2D Kolmogorov Flow → Leith turbulence model (no pressure field)
✅ 3D Channel Flow → RANS k-ε turbulence model (full fields including pressure)

主要功能:
- LowFidelityConsistencyLoss: 核心低保真場一致性損失 (ACTIVE)

參考文檔: docs/LOWFI_PRIOR_GUIDE.md
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Union
import numpy as np


class LowFidelityConsistencyLoss(nn.Module):
    """
    低保真場一致性損失
    
    將低保真先驗（例如 RANS）作為軟約束，引導 PINN 學習合理的場分佈。
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





# 綜合先驗損失管理器
class PriorLossManager(nn.Module):
    """
    先驗損失管理器 (Simplified)
    
    僅管理低保真一致性損失。
    其他損失類型（統計、守恆、對稱）已移除，由 PDE residuals 和 BC 處理。
    """
    
    def __init__(self, 
                 consistency_weight: float = 1.0,
                 loss_config: Optional[Dict] = None):
        """
        Args:
            consistency_weight: 低保真一致性權重
            loss_config: 詳細損失配置字典（可選）
        """
        super().__init__()
        
        self.consistency_weight = consistency_weight
        
        # 初始化損失組件
        self.low_fidelity_loss = LowFidelityConsistencyLoss()
        
        # 如果提供了詳細配置，則覆蓋預設組件
        if loss_config:
            if 'low_fidelity' in loss_config:
                self.low_fidelity_loss = LowFidelityConsistencyLoss(**loss_config['low_fidelity'])
    
    def compute_total_loss(self, 
                          model: nn.Module, 
                          batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        計算總先驗損失（僅低保真一致性）
        
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
        
        return total_loss
    
    def forward(self, 
                **inputs) -> Dict[str, torch.Tensor]:
        """
        計算先驗損失（僅低保真一致性）
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
        
        # 計算總先驗損失
        total_prior = sum(v for k, v in all_losses.items() if k.endswith('_total'))
        all_losses['prior_total'] = total_prior
        
        return all_losses


# 便捷建構函數
def create_prior_loss(config: Dict) -> PriorLossManager:
    """根據配置建立先驗損失管理器 (Simplified)"""
    return PriorLossManager(
        consistency_weight=config.get('consistency_weight', 1.0),
        loss_config=config.get('components', {})
    )


# ============================================================================
# 向後相容包裝器 (Backward Compatibility Wrappers)
# ============================================================================

def prior_consistency_loss(
    high_fi: torch.Tensor,
    low_fi: torch.Tensor,
    strength: float = 1.0,
    uncertainty: Optional[torch.Tensor] = None,
    adaptive: bool = False
) -> torch.Tensor:
    """
    先驗一致性損失（向後相容函數）
    
    Args:
        high_fi: 高保真預測 [batch_size, n_vars]
        low_fi: 低保真先驗 [batch_size, n_vars]
        strength: 損失強度
        uncertainty: 不確定性 [batch_size, n_vars]
        adaptive: 是否自適應調整（忽略此參數，保持相容性）
        
    Returns:
        loss: 標量損失值
    """
    residual = high_fi - low_fi
    
    if uncertainty is not None:
        # 加權一致性（不確定性越高權重越低）
        weights = 1.0 / (uncertainty ** 2 + 1e-8)
        weighted_residual = residual * weights
        return strength * torch.mean(weighted_residual ** 2)
    else:
        return strength * torch.mean(residual ** 2)


def statistical_prior_loss(
    predictions: torch.Tensor,
    mean: Optional[torch.Tensor] = None,
    std: Optional[torch.Tensor] = None,
    prior_type: str = 'mean',
    strength: float = 0.1,
    target_stats: Optional[torch.Tensor] = None  # 向後相容參數
) -> torch.Tensor:
    """
    統計先驗損失（向後相容函數）
    
    Args:
        predictions: 預測值 [batch_size, n_vars]
        mean: 預期均值（如果為 None，假設為 0）
        std: 預期標準差（如果為 None，假設為 1）
        prior_type: 先驗類型 ('mean', 'variance', 'range')
        strength: 損失強度
        target_stats: 目標統計量（向後相容，映射到 mean 或 std）
        
    Returns:
        loss: 標量損失值
    """
    # 向後相容：target_stats 映射到 mean 或 std
    if target_stats is not None:
        if prior_type == 'mean':
            mean = target_stats
        elif prior_type == 'variance':
            std = target_stats
    
    if prior_type == 'mean':
        # 均值約束
        if mean is None:
            mean = torch.zeros(predictions.shape[-1], device=predictions.device)
        return strength * torch.mean((predictions.mean(0) - mean) ** 2)
    
    elif prior_type == 'variance':
        # 方差約束
        if std is None:
            std = torch.ones(predictions.shape[-1], device=predictions.device)
        # 計算預測的方差
        pred_var = predictions.var(0)
        return strength * torch.mean((pred_var - std) ** 2)
    
    elif prior_type == 'range':
        # 範圍約束損失
        lower_violation = torch.relu(-predictions)
        upper_violation = torch.relu(predictions - 1.0)
        return strength * (torch.mean(lower_violation ** 2) + torch.mean(upper_violation ** 2))
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


def physics_constraint_loss(
    field: torch.Tensor,
    constraint_type: str = 'positivity',
    strength: float = 1.0,
    constraint_params: Optional[Dict] = None
) -> torch.Tensor:
    """
    物理約束損失（向後相容函數）
    
    Args:
        field: 場變量 [batch_size, n_vars]
        constraint_type: 約束類型 ('positivity', 'boundedness', 'symmetry', 
                                    'energy_bound', 'momentum_conservation')
        strength: 損失強度
        constraint_params: 約束參數字典（例如邊界值）
        
    Returns:
        loss: 標量損失值
    """
    if constraint_params is None:
        constraint_params = {}
    
    if constraint_type == 'positivity':
        # 非負約束
        violation = torch.relu(-field)
        return strength * torch.mean(violation ** 2)
    
    elif constraint_type == 'boundedness':
        # 有界約束
        lower_bound = constraint_params.get('lower', 0.0)
        upper_bound = constraint_params.get('upper', 1.0)
        lower_violation = torch.relu(lower_bound - field)
        upper_violation = torch.relu(field - upper_bound)
        return strength * (torch.mean(lower_violation ** 2) + torch.mean(upper_violation ** 2))
    
    elif constraint_type == 'symmetry':
        # 對稱性約束（假設第一維度應該對稱）
        if field.shape[-1] < 2:
            return torch.tensor(0.0, device=field.device)
        half = field.shape[0] // 2
        symmetry_violation = field[:half] - field[-half:]
        return strength * torch.mean(symmetry_violation ** 2)
    
    elif constraint_type == 'energy_bound':
        # 能量邊界約束
        max_energy = constraint_params.get('max_energy', 10.0)
        # 計算動能（假設 field 是速度場）
        kinetic_energy = 0.5 * torch.sum(field ** 2, dim=-1)
        # 超過最大能量的部分作為懲罰
        energy_violation = torch.relu(kinetic_energy - max_energy)
        return strength * torch.mean(energy_violation ** 2)
    
    elif constraint_type == 'momentum_conservation':
        # 動量守恆約束
        target_momentum = constraint_params.get('target_momentum', torch.zeros(field.shape[-1], device=field.device))
        # 計算實際動量（假設 field 是速度場）
        actual_momentum = torch.mean(field, dim=0)
        momentum_violation = actual_momentum - target_momentum
        return strength * torch.mean(momentum_violation ** 2)
    
    else:
        raise ValueError(f"Unknown constraint type: {constraint_type}")


def energy_conservation_loss(
    velocity: torch.Tensor,
    pressure: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    strength: float = 0.1,
    conservation_type: str = 'total',
    time_derivative: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    能量守恆損失（向後相容函數）
    
    Args:
        velocity: 速度場 [batch_size, n_dims] 或總能量場 [batch_size, 1]
        pressure: 壓力場 [batch_size, 1]（可選）
        coords: 座標 [batch_size, n_dims]（可選）
        strength: 損失強度
        conservation_type: 守恆類型 ('total', 'kinetic', 'pressure', 'steady', 'unsteady')
        time_derivative: 時間導數（用於非定常情況）
        
    Returns:
        loss: 標量損失值
    """
    # 判斷是否為簡化調用（只傳入總能量）
    if pressure is None and coords is None:
        # 假設 velocity 參數實際上是總能量
        total_energy = velocity
        
        if conservation_type == 'steady':
            # 定常情況：能量應該在空間上保持恆定
            energy_variation = torch.std(total_energy)
            return strength * energy_variation ** 2
        
        elif conservation_type == 'unsteady':
            # 非定常情況：能量的時間變化應該等於時間導數
            if time_derivative is None:
                raise ValueError("time_derivative is required for unsteady conservation")
            
            # 能量變化率的殘差
            energy_change = torch.std(total_energy)
            time_change = torch.std(time_derivative)
            residual = energy_change - time_change
            return strength * residual ** 2
        
        else:
            # 默認：能量守恆（變化應該小）
            energy_variation = torch.std(total_energy)
            return strength * energy_variation ** 2
    
    # 完整調用（velocity + pressure + coords）
    if conservation_type == 'total':
        # 動能
        kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=-1, keepdim=True)
        
        # 總能量（動能 + 壓能）
        if pressure is not None:
            total_energy = kinetic_energy + pressure
        else:
            total_energy = kinetic_energy
        
        # 能量的空間變化應該小（對於穩態流）
        energy_variation = torch.std(total_energy)
        
        return strength * energy_variation ** 2
    
    elif conservation_type == 'kinetic':
        # 動能守恆
        kinetic_energy = 0.5 * torch.sum(velocity ** 2, dim=-1)
        kinetic_variation = torch.std(kinetic_energy)
        return strength * kinetic_variation ** 2
    
    elif conservation_type == 'pressure':
        # 壓能守恆
        if pressure is not None:
            pressure_variation = torch.std(pressure)
            return strength * pressure_variation ** 2
        else:
            return torch.tensor(0.0, device=velocity.device)
    
    else:
        raise ValueError(f"Unknown conservation type: {conservation_type}")
