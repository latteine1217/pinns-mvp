"""
因果權重器 v2.0 - 對齊 JAX-PI 實作

核心改進：
1. 分量級因果權重計算（對每個損失分量獨立計算）
2. 取最小值策略（最保守的因果約束）
3. 完全對齊 JAX-PI 的實作方式

參考：
- JAX-PI: ~/Documents/coding/jaxpi/examples/kolmogorov_flow/models.py:92-115
- 論文: Wang et al. (2022) "Respecting Causality is all you need for training Physics-Informed Neural Networks"
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class CausalWeighterPerComponent(nn.Module):
    """
    分量級因果權重器（對齊 JAX-PI 實作）
    
    核心策略（JAX-PI Line 92-115）：
    1. 對每個損失分量（ru, rv, rc）獨立計算因果權重
    2. 取所有分量權重的最小值（最嚴格的約束）
    3. 使用該最小權重應用到所有分量
    
    數學形式：
        ru_gamma = exp(-tol * M @ mean(ru²))
        rv_gamma = exp(-tol * M @ mean(rv²))
        rc_gamma = exp(-tol * M @ mean(rc²))
        
        gamma = min(ru_gamma, rv_gamma, rc_gamma)  # 取最小值
        
        ru_loss = mean(gamma * mean(ru²))
        rv_loss = mean(gamma * mean(rv²))
        rc_loss = mean(gamma * mean(rc²))
    
    為什麼取最小值？
    - 確保所有物理方程都滿足因果約束
    - 避免某個方程的弱因果約束主導整體
    - 提供最保守但最安全的訓練策略
    """
    
    def __init__(
        self,
        causal_tol: float = 1.0,
        num_chunks: int = 16,
        device: str = 'cpu',
    ):
        """
        Args:
            causal_tol: 因果容差參數（對齊 JAX-PI）
                       JAX-PI 默認值: 1.0
            num_chunks: 時間分塊數量（對齊 JAX-PI）
                       JAX-PI Kolmogorov: 16
                       JAX-PI Burgers: 32
            device: 計算設備
        """
        super().__init__()
        self.causal_tol = float(causal_tol)
        self.num_chunks = int(num_chunks)
        self.device = torch.device(device) if isinstance(device, str) else device
        
        # 預計算因果矩陣 M (strict lower triangular)
        # M[i, j] = 1 if j < i else 0
        self.register_buffer(
            'causal_matrix',
            torch.tril(
                torch.ones((self.num_chunks, self.num_chunks), device=self.device),
                diagonal=-1
            )
        )
        
        logger.info(f"✅ CausalWeighterPerComponent initialized:")
        logger.info(f"   causal_tol: {self.causal_tol}")
        logger.info(f"   num_chunks: {self.num_chunks}")
        logger.info(f"   strategy: min(ru_gamma, rv_gamma, rc_gamma)")
    
    def compute_component_weights(
        self,
        residual_dict: Dict[str, torch.Tensor],
        time_coords: torch.Tensor,
        component_keys: Optional[list] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        計算分量級因果權重（對齊 JAX-PI）
        
        Args:
            residual_dict: 各分量的殘差字典
                例如：{'momentum_x': ru, 'momentum_y': rv, 'continuity': rc}
            time_coords: 時間坐標 [N, 1] 或 [N]
            component_keys: 要計算因果權重的分量鍵列表
                           若為 None，使用所有鍵
        
        Returns:
            gamma: 最終因果權重 [num_chunks] (所有分量的最小值)
            component_gammas: 各分量的因果權重字典
        """
        if component_keys is None:
            component_keys = list(residual_dict.keys())
        
        # 驗證輸入
        if not component_keys:
            raise ValueError("No components provided for causal weighting")
        
        # 提取第一個分量確定點數
        first_key = component_keys[0]
        if first_key not in residual_dict:
            raise KeyError(f"Component '{first_key}' not found in residual_dict")
        
        first_residual = residual_dict[first_key]
        n_points = first_residual.numel()
        
        # 1. 時間排序
        t_vals = time_coords.detach().flatten()
        if t_vals.numel() != n_points:
            raise ValueError(
                f"Time coords size mismatch: {t_vals.numel()} vs residual size {n_points}"
            )
        
        t_sorted, sort_idx = torch.sort(t_vals)
        
        # 2. 對每個分量計算因果權重
        component_gammas = {}
        
        for key in component_keys:
            if key not in residual_dict:
                logger.warning(f"Component '{key}' not found, skipping")
                continue
            
            residual = residual_dict[key]
            
            # 排序殘差
            residual_sorted = residual.detach().flatten()[sort_idx]
            
            # 動態調整 num_chunks
            num_chunks = max(1, min(self.num_chunks, n_points))
            chunk_size = n_points // num_chunks
            if chunk_size == 0:
                chunk_size = 1
                num_chunks = n_points
            
            usable = chunk_size * num_chunks
            residual_main = residual_sorted[:usable].view(num_chunks, chunk_size)
            
            # 計算每個 chunk 的平均殘差平方
            chunk_means_squared = torch.mean(residual_main ** 2, dim=1)  # [num_chunks]
            
            # 使用因果矩陣計算權重
            if num_chunks < self.num_chunks:
                causal_matrix_used = self.causal_matrix[:num_chunks, :num_chunks]
            else:
                causal_matrix_used = self.causal_matrix
            
            # gamma_k = exp(-tol * sum_{i<k} mean(residual_i²))
            component_gamma = torch.exp(
                -self.causal_tol * (causal_matrix_used @ chunk_means_squared)
            )
            
            component_gammas[key] = component_gamma
        
        if not component_gammas:
            raise RuntimeError("No valid components for causal weighting")
        
        # 3. 取所有分量的最小值（JAX-PI Line 112-113）
        # gamma = min(ru_gamma, rv_gamma, rc_gamma)
        all_gammas = torch.stack(list(component_gammas.values()))  # [num_components, num_chunks]
        gamma = torch.min(all_gammas, dim=0)[0]  # [num_chunks]
        
        return gamma, component_gammas
    
    def apply_weights_to_losses(
        self,
        residual_dict: Dict[str, torch.Tensor],
        time_coords: torch.Tensor,
        component_keys: Optional[list] = None,
        return_details: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        應用因果權重到各損失分量（JAX-PI 風格）
        
        Args:
            residual_dict: 殘差字典
            time_coords: 時間坐標
            component_keys: 要加權的分量鍵
            return_details: 是否返回詳細信息（gamma, component_gammas）
        
        Returns:
            weighted_losses: 加權後的損失字典
                例如：{'momentum_x': weighted_loss, 'momentum_y': ..., 'continuity': ...}
            
            若 return_details=True，額外返回：
                - 'causal_gamma': 最終因果權重 [num_chunks]
                - 'component_gammas': 各分量的因果權重
        """
        if component_keys is None:
            component_keys = list(residual_dict.keys())
        
        # 計算因果權重
        gamma, component_gammas = self.compute_component_weights(
            residual_dict, time_coords, component_keys
        )
        
        # 時間排序
        t_vals = time_coords.detach().flatten()
        t_sorted, sort_idx = torch.sort(t_vals)
        n_points = t_vals.numel()
        
        # 動態調整 num_chunks
        num_chunks = len(gamma)
        chunk_size = n_points // num_chunks
        usable = chunk_size * num_chunks
        
        # 應用權重到各分量
        weighted_losses = {}
        
        for key in component_keys:
            if key not in residual_dict:
                continue
            
            residual = residual_dict[key]
            residual_sorted = residual.detach().flatten()[sort_idx]
            residual_main = residual_sorted[:usable].view(num_chunks, chunk_size)
            
            # 計算 chunk 級別的損失
            chunk_losses = torch.mean(residual_main ** 2, dim=1)  # [num_chunks]
            
            # 應用因果權重（JAX-PI Line 139-141）
            weighted_loss = torch.mean(gamma * chunk_losses)
            
            weighted_losses[key] = weighted_loss
        
        # 可選：返回詳細信息
        if return_details:
            weighted_losses['causal_gamma'] = gamma
            weighted_losses['component_gammas'] = component_gammas
        
        return weighted_losses
    
    def forward(
        self,
        residual_dict: Dict[str, torch.Tensor],
        time_coords: torch.Tensor,
        component_keys: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向傳播（符合 nn.Module 接口）
        
        Args:
            residual_dict: 殘差字典
            time_coords: 時間坐標
            component_keys: 要加權的分量鍵
        
        Returns:
            加權後的損失字典
        """
        return self.apply_weights_to_losses(
            residual_dict, time_coords, component_keys, return_details=False
        )
    
    def get_diagnostic_info(
        self,
        residual_dict: Dict[str, torch.Tensor],
        time_coords: torch.Tensor,
        component_keys: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        獲取診斷信息（用於調試與監控）
        
        Returns:
            診斷信息字典，包含：
            - causal_gamma: 最終因果權重
            - component_gammas: 各分量因果權重
            - min_weight: 最小權重值
            - max_weight: 最大權重值
            - weight_variance: 權重方差
        """
        gamma, component_gammas = self.compute_component_weights(
            residual_dict, time_coords, component_keys
        )
        
        diagnostics = {
            'causal_gamma': gamma,
            'component_gammas': component_gammas,
            'min_weight': gamma.min().item(),
            'max_weight': gamma.max().item(),
            'weight_variance': gamma.var().item(),
            'num_chunks': len(gamma),
        }
        
        # 各分量的最小權重
        for key, comp_gamma in component_gammas.items():
            diagnostics[f'{key}_min_weight'] = comp_gamma.min().item()
        
        return diagnostics


# ============================================================================
# 工廠函數
# ============================================================================

def create_causal_weighter(
    causal_tol: float = 1.0,
    num_chunks: int = 16,
    device: str = 'cpu'
):
    """
    工廠函數：創建因果權重器
    
    Args:
        causal_tol: 因果容差（對齊 JAX-PI，默認值 1.0）
        num_chunks: 分塊數量
        device: 計算設備
    
    Returns:
        CausalWeighterPerComponent 實例（v2，對齊 JAX-PI）
    """
    return CausalWeighterPerComponent(
        causal_tol=causal_tol,
        num_chunks=num_chunks,
        device=device
    )
