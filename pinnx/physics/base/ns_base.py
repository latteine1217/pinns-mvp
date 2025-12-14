"""
Navier-Stokes Base Class Module
=================================

提供所有 Navier-Stokes 求解器的共用基類，避免重複實現動量方程、連續性方程。

主要功能：
1. NavierStokesBase - N-S 方程抽象基類
2. 速度/壓力場解析（2D/3D 自動適配）
3. 連續性方程模板（divergence-free）
4. 動量方程組件（對流項、黏性項、壓力梯度）
5. 物理參數管理（Re, nu, rho）

設計哲學：
- DRY 原則：所有 N-S 求解器共用邏輯集中此處
- 可擴展：子類可覆蓋特定項（如 RANS 加入湍流項）
- 向後兼容：保留原有接口名稱

作者：PINNs-MVP 團隊
日期：2025-12-15
"""

from .pde_base import PDEBase
import torch
from typing import Tuple, Dict, Optional, Union, Any


class NavierStokesBase(PDEBase):
    """
    不可壓縮 Navier-Stokes 方程基類
    
    控制方程（2D）：
        ∂u/∂t + u ∂u/∂x + v ∂u/∂y = -∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²) + f_x
        ∂v/∂t + u ∂v/∂x + v ∂v/∂y = -∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²) + f_y
        ∂u/∂x + ∂v/∂y = 0  (連續性方程)
    
    控制方程（3D）：
        動量方程 × 3（加入 w, ∂/∂z 項）
        ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    
    提供共用功能：
    - parse_velocity_pressure(): 解析網路輸出
    - compute_continuity_residual(): ∇·u
    - compute_advection_term(): u·∇u_i
    - compute_viscous_term(): ν∇²u_i
    - compute_pressure_gradient(): ∂p/∂x_i
    
    子類需實現：
    - residual(): 組裝完整方程殘差
    - 可選覆蓋源項、邊界條件等
    
    Example:
        >>> # 2D Kolmogorov Flow
        >>> class KolmogorovFlow2D(NavierStokesBase):
        ...     def __init__(self, Re, domain_bounds):
        ...         physics_params = {'Re': Re, 'nu': 1.0/Re}
        ...         super().__init__(physics_params, domain_bounds, spatial_dim=2)
        ...     
        ...     def residual(self, coords, predictions):
        ...         u, v, p = self.parse_velocity_pressure(predictions)
        ...         
        ...         # 對流項
        ...         advection_u = self.compute_advection_term(coords, u, [u, v])
        ...         
        ...         # 黏性項
        ...         viscous_u = self.compute_viscous_term(coords, u)
        ...         
        ...         # 壓力梯度
        ...         dp_dx = self.compute_pressure_gradient(p, coords, component=0)
        ...         
        ...         # 組裝動量方程
        ...         momentum_x = advection_u + dp_dx - viscous_u
        ...         # ... (類似處理 y 方向)
        ...         
        ...         # 連續性方程
        ...         continuity = self.compute_continuity_residual(coords, [u, v])
        ...         
        ...         return torch.cat([momentum_x, momentum_y, continuity], dim=1)
    """
    
    def __init__(
        self,
        physics_params: Dict[str, float],
        domain_bounds: Dict[str, list],
        loss_config: Optional[Dict[str, float]] = None,
        spatial_dim: int = 2,
        device: Optional[torch.device] = None
    ):
        """
        初始化 Navier-Stokes 基類
        
        Args:
            physics_params: 物理參數字典，必須包含：
                - 'nu': 運動黏度 [m²/s]
                - 'rho': 密度 [kg/m³]（可選，默認 1.0）
                或提供：
                - 'Re': 雷諾數（會自動計算 nu）
            domain_bounds: 計算域邊界（從 PDEBase 繼承）
            loss_config: 損失權重配置（從 PDEBase 繼承）
            spatial_dim: 空間維度 (2 or 3)
            device: 計算設備
            
        Example:
            >>> physics_params = {'Re': 1000, 'nu': 0.001, 'rho': 1.0}
            >>> domain = {'x': [0, 2*np.pi], 'y': [0, 1]}
            >>> ns = NavierStokesBase(physics_params, domain, spatial_dim=2)
        """
        super().__init__(domain_bounds, loss_config, device)
        
        # 設定空間維度
        self.spatial_dim = spatial_dim
        
        # 提取物理參數
        self.nu = physics_params.get('nu', 0.01)
        self.rho = physics_params.get('rho', 1.0)
        
        # 如果提供 Re 但未提供 nu，自動計算
        if 'Re' in physics_params and 'nu' not in physics_params:
            Re = physics_params['Re']
            # Re = U*L/nu → nu = U*L/Re（需要特征速度與長度尺度）
            # 這裡假設 nu = 1/Re（無因次化情況）
            self.nu = 1.0 / Re
        
        # 記錄 Re（方便監控）
        self.Re = physics_params.get('Re', 1.0 / self.nu if self.nu > 0 else float('inf'))
        
    # ========================================================================
    # 速度/壓力場解析
    # ========================================================================
    
    def parse_velocity_pressure(
        self,
        predictions: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        從網路輸出解析速度與壓力場
        
        Args:
            predictions: 網路預測 [batch_size, n_fields]
                2D: [u, v, p] → shape [N, 3]
                3D: [u, v, w, p] → shape [N, 4]
        
        Returns:
            2D: (u, v, p) - 每個 [N, 1]
            3D: (u, v, w, p) - 每個 [N, 1]
            
        Example:
            >>> predictions = model(coords)  # [100, 3]
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> # u, v, p: [100, 1]
        """
        if self.spatial_dim == 2:
            u = predictions[:, 0:1]
            v = predictions[:, 1:2]
            p = predictions[:, 2:3]
            return u, v, p
        elif self.spatial_dim == 3:
            u = predictions[:, 0:1]
            v = predictions[:, 1:2]
            w = predictions[:, 2:3]
            p = predictions[:, 3:4]
            return u, v, w, p
        else:
            raise ValueError(f"Unsupported spatial_dim={self.spatial_dim}")
    
    # ========================================================================
    # 連續性方程（Continuity Equation）
    # ========================================================================
    
    def compute_continuity_residual(
        self,
        coords: torch.Tensor,
        velocity_fields: list
    ) -> torch.Tensor:
        """
        計算連續性方程殘差：∇·u = 0
        
        Args:
            coords: 座標 [batch_size, spatial_dim]
            velocity_fields: 速度分量列表
                2D: [u, v]
                3D: [u, v, w]
        
        Returns:
            散度 [batch_size, 1]
            
        Example:
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> continuity = self.compute_continuity_residual(coords, [u, v])
            >>> # 理想情況：continuity ≈ 0 everywhere
        """
        divergence = torch.zeros_like(velocity_fields[0])
        
        for i, u_i in enumerate(velocity_fields):
            # ∂u_i/∂x_i
            du_dx = self.compute_gradient(u_i, coords, component=i)
            divergence = divergence + du_dx
        
        return divergence
    
    # ========================================================================
    # 動量方程組件（Momentum Equation Terms）
    # ========================================================================
    
    def compute_advection_term(
        self,
        coords: torch.Tensor,
        u_field: torch.Tensor,
        velocity_fields: list
    ) -> torch.Tensor:
        """
        計算對流項：u·∇u_i = u ∂u_i/∂x + v ∂u_i/∂y (+ w ∂u_i/∂z)
        
        Args:
            coords: 座標 [batch_size, spatial_dim]
            u_field: 目標速度分量（如 u, v, w 中的一個）[batch_size, 1]
            velocity_fields: 所有速度分量
                2D: [u, v]
                3D: [u, v, w]
        
        Returns:
            對流項 [batch_size, 1]
            
        Example:
            >>> # 計算 u 方向的對流項
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> advection_u = self.compute_advection_term(coords, u, [u, v])
            
        Note:
            對流項是非線性的，是 N-S 方程最難優化的部分
        """
        advection = torch.zeros_like(u_field)
        
        for i, u_i in enumerate(velocity_fields):
            # ∂u_field/∂x_i
            du_dx = self.compute_gradient(u_field, coords, component=i)
            # u_i * ∂u_field/∂x_i
            advection = advection + u_i * du_dx
        
        return advection
    
    def compute_viscous_term(
        self,
        coords: torch.Tensor,
        u_field: torch.Tensor,
        stabilize: bool = False
    ) -> torch.Tensor:
        """
        計算黏性項：ν∇²u_i
        
        Args:
            coords: 座標 [batch_size, spatial_dim]
            u_field: 速度分量 [batch_size, 1]
            stabilize: 是否數值穩定化（高 Re 時建議開啟）
        
        Returns:
            黏性項 [batch_size, 1]
            
        Example:
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> viscous_u = self.compute_viscous_term(coords, u, stabilize=True)
            
        Note:
            高雷諾數（Re > 1000）時，黏性項很小但梯度很大（近壁面）
            建議開啟 stabilize 避免數值振盪
        """
        laplacian = self.compute_laplacian(u_field, coords, stabilize=stabilize)
        return self.nu * laplacian
    
    def compute_pressure_gradient(
        self,
        p: torch.Tensor,
        coords: torch.Tensor,
        component: int
    ) -> torch.Tensor:
        """
        計算壓力梯度：∂p/∂x_i
        
        Args:
            p: 壓力場 [batch_size, 1]
            coords: 座標 [batch_size, spatial_dim]
            component: 方向索引 (0=x, 1=y, 2=z)
        
        Returns:
            壓力梯度 [batch_size, 1]
            
        Example:
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> dp_dx = self.compute_pressure_gradient(p, coords, component=0)
            >>> dp_dy = self.compute_pressure_gradient(p, coords, component=1)
        """
        return self.compute_gradient(p, coords, component=component)
    
    # ========================================================================
    # 動量方程組裝（子類需實現）
    # ========================================================================
    
    def compute_momentum_residuals(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> Tuple[torch.Tensor, ...]:
        """
        計算動量方程殘差（由子類實現具體形式）
        
        Args:
            coords: 座標 [batch_size, spatial_dim]
            predictions: 網路預測 [batch_size, n_fields]
        
        Returns:
            2D: (momentum_x, momentum_y)
            3D: (momentum_x, momentum_y, momentum_z)
            
        Note:
            此方法為可選實現。子類可直接實現 residual() 而不拆分
            但提供此接口可讓代碼更模組化
        """
        raise NotImplementedError(
            "子類應實現 compute_momentum_residuals() 或直接實現 residual()"
        )
    
    # ========================================================================
    # 元數據接口（實現 PDEBase 要求）
    # ========================================================================
    
    def get_physics_info(self) -> Dict[str, Any]:
        """
        返回 Navier-Stokes 物理參數
        
        Returns:
            包含物理參數的字典
        """
        return {
            'equation': 'Navier-Stokes',
            'spatial_dim': self.spatial_dim,
            'Re': float(self.Re),
            'nu': float(self.nu),
            'rho': float(self.rho),
            'compressible': False,  # 此基類僅支持不可壓縮
        }
    
    # ========================================================================
    # 輔助方法
    # ========================================================================
    
    def compute_kinetic_energy(
        self,
        velocity_fields: list
    ) -> torch.Tensor:
        """
        計算動能：KE = 0.5 * (u² + v² + w²)
        
        Args:
            velocity_fields: 速度分量列表 [u, v] or [u, v, w]
        
        Returns:
            動能 [batch_size, 1]
            
        Example:
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> ke = self.compute_kinetic_energy([u, v])
        """
        ke = torch.zeros_like(velocity_fields[0])
        for u_i in velocity_fields:
            ke = ke + u_i ** 2
        return 0.5 * ke
    
    def compute_enstrophy(
        self,
        coords: torch.Tensor,
        velocity_fields: list
    ) -> torch.Tensor:
        """
        計算渦量的平方（Enstrophy）：Ω = 0.5 * ω²
        
        僅支持 2D（ω_z = ∂v/∂x - ∂u/∂y）
        
        Args:
            coords: 座標 [batch_size, 2]
            velocity_fields: [u, v]
        
        Returns:
            Enstrophy [batch_size, 1]
            
        Example:
            >>> u, v, p = self.parse_velocity_pressure(predictions)
            >>> enstrophy = self.compute_enstrophy(coords, [u, v])
            
        Note:
            Enstrophy 是 2D 湍流的重要診斷量（逆級聯守恆量）
        """
        if self.spatial_dim != 2:
            raise NotImplementedError("Enstrophy 僅支持 2D")
        
        u, v = velocity_fields[0], velocity_fields[1]
        
        # ω_z = ∂v/∂x - ∂u/∂y
        dv_dx = self.compute_gradient(v, coords, component=0)
        du_dy = self.compute_gradient(u, coords, component=1)
        omega_z = dv_dx - du_dy
        
        # Enstrophy = 0.5 * ω²
        return 0.5 * (omega_z ** 2)
    
    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"{self.__class__.__name__}("
            f"dim={self.spatial_dim}, "
            f"Re={self.Re:.1f}, "
            f"nu={self.nu:.6f})"
        )
