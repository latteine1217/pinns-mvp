"""
PDE Base Class Module
======================

提供所有 PDE 求解器的抽象基類，統一管理設備、梯度計算、損失配置。

主要功能：
1. PDEBase - 抽象基類（所有 PDE 求解器的父類）
2. 統一的設備管理 (CPU/CUDA)
3. 統一的梯度/拉普拉斯算子接口
4. 損失權重配置管理
5. 物理參數元數據接口

設計哲學：
- 簡潔：僅提供共用邏輯，避免過度抽象
- 可擴展：子類可輕鬆覆蓋關鍵方法

作者：PINNs-MVP 團隊
日期：2025-12-15
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any


class PDEBase(ABC, nn.Module):
    """
    所有 PDE 求解器的抽象基類
    
    提供核心功能：
    1. 設備管理（CPU/CUDA）
    2. 梯度計算委派
    3. 損失配置存儲
    4. 元數據接口
    
    子類必須實現：
    - residual(): 計算 PDE 殘差
    - get_physics_info(): 返回物理參數字典
    
    Example:
        >>> class Poisson2D(PDEBase):
        ...     def __init__(self, domain_bounds):
        ...         super().__init__(domain_bounds, loss_config={'pde': 1.0})
        ...         self.spatial_dim = 2
        ...     
        ...     def residual(self, coords, predictions):
        ...         u = predictions[:, 0:1]
        ...         laplacian = self.compute_laplacian(u, coords)
        ...         return laplacian  # ∇²u = 0
        ...     
        ...     def get_physics_info(self):
        ...         return {'equation': 'Poisson', 'dim': 2}
        
        >>> pde = Poisson2D(domain_bounds={'x': [0, 1], 'y': [0, 1]})
        >>> coords = torch.randn(100, 2, requires_grad=True)
        >>> predictions = torch.randn(100, 1)
        >>> residuals = pde.residual(coords, predictions)
    """
    
    def __init__(
        self,
        domain_bounds: Dict[str, list],
        loss_config: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化 PDE 基類
        
        Args:
            domain_bounds: 計算域邊界
                例如：{'x': [0, 1], 'y': [0, 1]} (2D)
                     {'x': [0, 2*pi], 'y': [0, 1], 'z': [0, 2*pi]} (3D)
            loss_config: 損失權重配置
                例如：{'pde': 1.0, 'data': 100.0, 'bc': 10.0}
            device: 計算設備（None = 自動偵測）
        """
        super().__init__()
        self.domain_bounds = domain_bounds
        self.loss_config = loss_config or {}
        self._device = device
        
        # 子類應在 __init__ 中設置這些屬性
        self.spatial_dim = None  # 空間維度 (2 or 3)
        
    @abstractmethod
    def residual(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 PDE 殘差（必須由子類實現）
        
        Args:
            coords: 座標點 [batch_size, spatial_dim]
            predictions: 網路預測 [batch_size, n_fields]
                        例如：[u, v, p] for Navier-Stokes
        
        Returns:
            殘差張量（形狀由具體方程決定）
            - Scalar PDE: [batch_size, 1]
            - Vector PDE: [batch_size, n_equations]
            
        Note:
            此函數必須使用 autograd 計算梯度，因此：
            - coords 必須 requires_grad=True
            - 使用 self.compute_gradient() 或 self.compute_laplacian()
        """
        pass
    
    @abstractmethod
    def get_physics_info(self) -> Dict[str, Any]:
        """
        返回物理參數元數據（必須由子類實現）
        
        Returns:
            包含物理參數的字典，例如：
            {
                'equation': 'Navier-Stokes',
                'Re': 1000,
                'nu': 0.001,
                'spatial_dim': 2,
                'has_source_term': False
            }
            
        Note:
            此信息用於：
            - 訓練監控（TensorBoard 顯示）
            - Checkpoint 驗證（確保參數一致）
            - 文檔生成
        """
        pass
    
    # ========================================================================
    # 梯度計算接口（委派給專用模組）
    # ========================================================================
    
    def compute_gradient(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        component: int
    ) -> torch.Tensor:
        """
        計算梯度（委派給 gradient_ops 模組）
        
        Args:
            field: 標量場 [batch_size, 1]
            coords: 座標 [batch_size, spatial_dim]
            component: 求導方向 (0=x, 1=y, 2=z)
            
        Returns:
            梯度 [batch_size, 1]
            
        Example:
            >>> u = predictions[:, 0:1]
            >>> du_dx = self.compute_gradient(u, coords, component=0)
        """
        from .gradient_ops import compute_gradient
        
        if self.spatial_dim is None:
            raise ValueError("子類必須在 __init__ 中設置 self.spatial_dim")
        
        return compute_gradient(
            field, coords, component, spatial_dim=self.spatial_dim
        )
    
    def compute_laplacian(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        stabilize: bool = False
    ) -> torch.Tensor:
        """
        計算拉普拉斯算子（委派給 laplacian_ops 模組）
        
        Args:
            field: 標量場 [batch_size, 1]
            coords: 座標 [batch_size, spatial_dim]
            stabilize: 是否數值穩定化
            
        Returns:
            拉普拉斯 [batch_size, 1]
            
        Example:
            >>> u = predictions[:, 0:1]
            >>> laplacian_u = self.compute_laplacian(u, coords)
        """
        from .laplacian_ops import compute_laplacian
        
        if self.spatial_dim is None:
            raise ValueError("子類必須在 __init__ 中設置 self.spatial_dim")
        
        return compute_laplacian(
            field, coords,
            spatial_dim=self.spatial_dim,
            stabilize=stabilize
        )
    
    # ========================================================================
    # 設備管理
    # ========================================================================
    
    def to_device(self, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        確保所有張量在正確的設備上
        
        Args:
            *tensors: 任意數量的張量
            
        Returns:
            移動到正確設備的張量元組
            
        Example:
            >>> coords, predictions = self.to_device(coords, predictions)
        """
        if self._device is not None:
            device = self._device
        elif len(tensors) > 0:
            device = tensors[0].device
        else:
            device = torch.device('cpu')
        
        return tuple(t.to(device) for t in tensors if t is not None)
    
    @property
    def device(self) -> torch.device:
        """獲取當前設備"""
        if self._device is not None:
            return self._device
        
        # 嘗試從模組參數推斷
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device('cpu')
    
    # ========================================================================
    # 損失配置管理
    # ========================================================================
    
    def get_loss_weight(self, loss_name: str, default: float = 1.0) -> float:
        """
        獲取指定損失項的權重
        
        Args:
            loss_name: 損失項名稱 ('pde', 'data', 'bc', etc.)
            default: 默認權重（如果未配置）
            
        Returns:
            損失權重
            
        Example:
            >>> pde_weight = self.get_loss_weight('pde')
        """
        return self.loss_config.get(loss_name, default)
    
    def update_loss_weight(self, loss_name: str, weight: float):
        """
        更新損失權重（用於自適應權重策略）
        
        Args:
            loss_name: 損失項名稱
            weight: 新權重
        """
        self.loss_config[loss_name] = weight
    
    # ========================================================================
    # 輔助方法
    # ========================================================================
    
    def get_domain_size(self) -> Dict[str, float]:
        """
        計算各維度的域尺寸
        
        Returns:
            {'x': L_x, 'y': L_y, 'z': L_z}
            
        Example:
            >>> domain_size = self.get_domain_size()
            >>> L_x = domain_size['x']  # 流向長度
        """
        return {
            dim: bounds[1] - bounds[0]
            for dim, bounds in self.domain_bounds.items()
        }
    
    def __repr__(self) -> str:
        """字符串表示（方便調試）"""
        info = self.get_physics_info()
        equation = info.get('equation', 'Unknown')
        dim = info.get('spatial_dim', self.spatial_dim)
        return f"{self.__class__.__name__}(equation='{equation}', dim={dim})"
