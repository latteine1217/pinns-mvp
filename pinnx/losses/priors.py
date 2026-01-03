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
