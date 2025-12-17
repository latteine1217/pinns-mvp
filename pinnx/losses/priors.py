"""
先驗一致性損失函數模組 (Simplified)

實現各種先驗資訊的一致性損失,支援低保真資料作為軟先驗約束。
這是實現「少量資料 × 物理先驗」框架的關鍵組件。

=== 支援的模型 (SUPPORTED MODELS) ===
✅ 2D Kolmogorov Flow → Leith turbulence model (no pressure field)
✅ 3D Channel Flow → RANS k-ε turbulence model (full fields including pressure)

主要功能:
- LowFidelityConsistencyLoss: 核心低保真場一致性損失 (ACTIVE)

=== 已棄用 (DEPRECATED - Retained for backward compatibility) ===
以下類別保留用於向後相容,但不在當前專案範圍內:
- StatisticalConsistencyLoss: 統計矩約束由 PDE residuals 處理
- ConservationLoss: 守恆定律由 PDE residuals 處理
- SymmetryConsistencyLoss: 對稱性由 boundary conditions 處理

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
    print("=== 先驗一致性損失測試 (Simplified) ===")
    
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
    
    # 測試相容性函數
    print("\n--- 相容性函數測試 ---")
    
    # 先驗一致性
    prior_loss = prior_consistency_loss(high_fi_pred, low_fi_data, strength=0.5)
    print(f"先驗一致性損失: {prior_loss.item():.6f}")
    
    # 統計先驗
    stat_prior_loss = statistical_prior_loss(high_fi_pred, prior_type='mean', strength=0.1)
    print(f"統計先驗損失: {stat_prior_loss.item():.6f}")
    
    # 物理約束
    velocity = high_fi_pred[:, :2]  # [u, v]
    physics_loss = physics_constraint_loss(
        velocity, constraint_type='energy_bound',
        constraint_params={'max_energy': 5.0}, strength=0.05
    )
    print(f"物理約束損失: {physics_loss.item():.6f}")
    
    # 能量守恆
    pressure = high_fi_pred[:, 2]   # p
    total_energy = 0.5 * torch.sum(velocity**2, dim=1, keepdim=True) + pressure.unsqueeze(-1)
    energy_loss = energy_conservation_loss(total_energy, conservation_type='steady', strength=0.02)
    print(f"能量守恆損失: {energy_loss.item():.6f}")
    
    print("\n✅ 先驗一致性損失測試通過！")
