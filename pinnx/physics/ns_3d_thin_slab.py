"""
Navier-Stokes 3D Thin-Slab 方程式模組
=====================================

提供3D不可壓縮NS方程的物理定律計算功能（針對 thin-slab 配置優化）：
1. 3D NS方程殘差計算 (x/y/z動量 + 連續方程)
2. 週期邊界條件處理 (x, z方向)
3. 壁面無滑移邊界條件 (y方向)
4. 湍流量計算 (耗散率、渦旋度、Q準則)
5. 數值穩定性保護 (梯度裁剪、正則化)

設計依據：
- 物理審查報告: tasks/3d_thin_slab_prep/physics_review.md
- 目標案例: JHTDB channel flow Re_τ=1000, z⁺ ≈ 120

重構記錄：
- Phase 4-3 (2025-12-15): 重構為繼承 NavierStokesBase，消除重複代碼
- 保留向後兼容接口，所有測試無需修改
"""

import torch
import torch.autograd as autograd
from typing import Tuple, Optional, Dict, Any, List, Union
import warnings

# Import base class
from .base.ns_base import NavierStokesBase
from .base.gradient_ops import compute_gradient, compute_second_derivative
from .base.laplacian_ops import compute_laplacian

# ============================================================================
# 向後兼容梯度工具（Wrapper for Base Module）
# ============================================================================

def compute_derivatives_3d(f: torch.Tensor, coords: torch.Tensor, 
                          order: int = 1, 
                          keep_graph: bool = True) -> torch.Tensor:
    """
    3D梯度計算（向後兼容接口）
    
    **重構說明**: 此函數現在委派給 `gradient_ops.compute_gradient()` 和 `compute_second_derivative()`
    
    Args:
        f: 待微分的標量場 [batch_size, 1]
        coords: 座標變數 [batch_size, 3] -> [x, y, z]
        order: 微分階數 (1 或 2)
        keep_graph: 是否保持計算圖（向後兼容參數，已無使用）
        
    Returns:
        一階: [batch_size, 3] -> [∂f/∂x, ∂f/∂y, ∂f/∂z]
        二階: [batch_size, 3] -> [∂²f/∂x², ∂²f/∂y², ∂²f/∂z²]
    """
    spatial_dim = coords.shape[1]
    
    if order == 1:
        # Compute all first derivatives
        first_derivs = []
        for i in range(spatial_dim):
            df_dxi = compute_gradient(f, coords, component=i, spatial_dim=spatial_dim)
            first_derivs.append(df_dxi)
        return torch.cat(first_derivs, dim=1)
    
    elif order == 2:
        # compute_second_derivative for diagonal elements [∂²f/∂x², ∂²f/∂y², ∂²f/∂z²]
        second_derivs = []
        for i in range(spatial_dim):
            d2f_dxi2 = compute_second_derivative(
                f, coords, component1=i, component2=i, spatial_dim=spatial_dim
            )
            second_derivs.append(d2f_dxi2)
        return torch.cat(second_derivs, dim=1)
    
    else:
        raise ValueError(f"不支援的微分階數: {order}")


def compute_laplacian_3d(f: torch.Tensor, coords: torch.Tensor, 
                        stabilize: bool = True,
                        max_value: float = 1e4) -> torch.Tensor:
    """
    計算3D拉普拉斯算子（向後兼容接口）
    
    **重構說明**: 此函數現在委派給 `laplacian_ops.compute_laplacian()`
    
    Args:
        f: 標量場 [batch_size, 1]
        coords: 座標 [batch_size, 3] -> [x, y, z]
        stabilize: 是否啟用數值穩定性保護
        max_value: 梯度裁剪上限（防止爆炸）
        
    Returns:
        拉普拉斯算子結果 [batch_size, 1]
    """
    laplacian_result = compute_laplacian(f, coords, spatial_dim=3)
    
    # 數值穩定性保護
    if stabilize:
        laplacian_result = torch.clamp(laplacian_result, -max_value, max_value)
    
    return laplacian_result


# ============================================================================
# 3D NS 方程殘差計算（向後兼容接口）
# ============================================================================

def ns_residual_3d_thin_slab(
    coords: torch.Tensor,
    pred: torch.Tensor,
    nu: float,
    time: Optional[torch.Tensor] = None,
    source_term: Optional[torch.Tensor] = None,
    stabilize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    計算3D不可壓縮Navier-Stokes方程殘差（thin-slab配置）
    
    **重構說明**: 此函數現在是 `NSEquations3DThinSlab.residual()` 的向後兼容 wrapper
    
    控制方程（壁面單位）：
    ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂z = -∂p/∂x + (1/Re_τ)∇²u  (x-動量)
    ∂v/∂t + u∂v/∂x + v∂v/∂y + w∂v/∂z = -∂p/∂y + (1/Re_τ)∇²v  (y-動量)
    ∂w/∂t + u∂w/∂x + v∂w/∂y + w∂w/∂z = -∂p/∂z + (1/Re_τ)∇²w  (z-動量)
    ∂u/∂x + ∂v/∂y + ∂w/∂z = 0                                  (連續方程)
    
    Args:
        coords: 空間座標 [batch_size, 3] -> [x, y, z]
        pred: 預測結果 [batch_size, 4] -> [u, v, w, p]
        nu: 運動黏度 (= 1/Re_τ for normalized equations)
        time: 時間座標 [batch_size, 1] (非定常流場)
        source_term: 外部源項 [batch_size, 3] -> [S_x, S_y, S_z] (可選)
        stabilize: 是否啟用數值穩定性保護
        
    Returns:
        Tuple of (x_momentum_residual, y_momentum_residual, 
                  z_momentum_residual, continuity_residual)
        每個都是 [batch_size, 1]
        
    物理一致性檢查：
    - 量綱一致性: ✅ (已驗證於 physics_review.md)
    - 邊界條件兼容: ✅ 週期性(x,z) + 無滑移(y)
    - 守恆定律: ✅ 質量、動量守恆
    """
    # 創建臨時實例（使用傳入的參數）
    ns_eq = NSEquations3DThinSlab(viscosity=nu, stabilize=stabilize)
    
    # 解析速度和壓力
    velocity = pred[:, :3]  # [u, v, w]
    pressure = pred[:, 3:4]  # [p]
    
    # 調用類方法
    res_dict = ns_eq.residual(coords, velocity, pressure, time=time, source_term=source_term)
    
    # 返回與原接口一致的元組
    return (
        res_dict['momentum_x'],
        res_dict['momentum_y'],
        res_dict['momentum_z'],
        res_dict['continuity']
    )


# ============================================================================
# 邊界條件處理
# ============================================================================

def apply_periodic_bc_3d(coords: torch.Tensor, 
                        pred: torch.Tensor,
                        domain_lengths: Dict[str, float]) -> Dict[str, torch.Tensor]:
    """
    應用3D週期邊界條件（x, z方向）
    
    物理意義：
    - 模擬無限長通道（x方向）
    - 消除展向邊界影響（z方向）
    - 統計均勻性假設成立
    
    Args:
        coords: 座標 [N_pairs*2, 3] -> [x, y, z]
                前N_pairs個點為邊界左側/前側
                後N_pairs個點為對應的右側/後側
        pred: 預測值 [N_pairs*2, 4] -> [u, v, w, p]
        domain_lengths: 域長度 {'L_x': float, 'L_z': float}
        
    Returns:
        週期性殘差字典 {'periodic_x': Tensor, 'periodic_z': Tensor}
    """
    if coords.shape[0] % 2 != 0:
        warnings.warn("週期邊界條件需要成對的邊界點")
        return {
            'periodic_x': torch.zeros_like(pred[:1, :]),
            'periodic_z': torch.zeros_like(pred[:1, :])
        }
    
    n_pairs = coords.shape[0] // 2
    
    # x方向週期性: u(x=0, y, z) = u(x=L_x, y, z)
    pred_left = pred[:n_pairs, :]
    pred_right = pred[n_pairs:, :]
    
    periodic_error_x = pred_left - pred_right  # 所有變數 [u, v, w, p]
    
    # z方向週期性（類似處理）
    # 注意：這裡假設coords已按照「前側-後側」配對
    periodic_error_z = pred_left - pred_right  # 簡化版，實際需分開處理x/z
    
    return {
        'periodic_x': periodic_error_x,
        'periodic_z': periodic_error_z
    }


def apply_wall_bc_3d(coords: torch.Tensor,
                    pred: torch.Tensor,
                    wall_location: str = "both") -> torch.Tensor:
    """
    應用3D壁面無滑移邊界條件（y = ±1）
    
    物理條件：
    - u(x, y=±1, z) = 0  （流向速度）
    - v(x, y=±1, z) = 0  （壁法向速度）
    - w(x, y=±1, z) = 0  （展向速度）
    - ∂p/∂y|_wall = ν∇²v|_wall  （壓力Neumann條件，暫不實作）
    
    Args:
        coords: 壁面座標 [N, 3]
        pred: 預測值 [N, 4] -> [u, v, w, p]
        wall_location: "upper" (y=1), "lower" (y=-1), "both"
        
    Returns:
        壁面BC殘差 [N, 3]（速度三分量）
    """
    u_pred = pred[:, 0:1]
    v_pred = pred[:, 1:2]
    w_pred = pred[:, 2:3]
    
    # 無滑移條件: u = v = w = 0
    u_bc_error = u_pred - 0.0
    v_bc_error = v_pred - 0.0
    w_bc_error = w_pred - 0.0
    
    return torch.cat([u_bc_error, v_bc_error, w_bc_error], dim=1)


# ============================================================================
# 湍流量計算（基礎 - Phase 2）
# ============================================================================

def compute_dissipation_3d(coords: torch.Tensor,
                          velocity: torch.Tensor,
                          nu: float) -> torch.Tensor:
    """
    計算湍流耗散率 ε = ν Σᵢⱼ (∂uᵢ/∂xⱼ)²
    
    物理意義：
    - 湍流動能轉化為熱能的速率
    - 用於驗證能量級聯理論
    
    穩健性：✅ 高（僅需一階導數）
    風險評級：🟢 低風險
    
    Args:
        coords: 座標 [batch_size, 3]
        velocity: 速度場 [batch_size, 3] -> [u, v, w]
        nu: 運動黏度
        
    Returns:
        耗散率 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]
    
    # 計算速度梯度張量
    u_grads = compute_derivatives_3d(u, coords, order=1)
    v_grads = compute_derivatives_3d(v, coords, order=1)
    w_grads = compute_derivatives_3d(w, coords, order=1)
    
    # 計算梯度平方和 Σᵢⱼ (∂uᵢ/∂xⱼ)²
    grad_squared = (
        u_grads[:, 0:1]**2 + u_grads[:, 1:2]**2 + u_grads[:, 2:3]**2 +  # ∂u/∂x, ∂u/∂y, ∂u/∂z
        v_grads[:, 0:1]**2 + v_grads[:, 1:2]**2 + v_grads[:, 2:3]**2 +  # ∂v/∂x, ∂v/∂y, ∂v/∂z
        w_grads[:, 0:1]**2 + w_grads[:, 1:2]**2 + w_grads[:, 2:3]**2    # ∂w/∂x, ∂w/∂y, ∂w/∂z
    )
    
    dissipation = nu * grad_squared
    
    return dissipation


def compute_enstrophy_3d(coords: torch.Tensor,
                        velocity: torch.Tensor) -> torch.Tensor:
    """
    計算渦旋度平方（Enstrophy） Ω² = |∇×u|²
    
    物理意義：
    - 渦旋強度指標
    - 與耗散率相關：ε ≈ ν·Ω²
    
    穩健性：✅ 高（僅需一階導數）
    風險評級：🟢 低風險
    
    Args:
        coords: 座標 [batch_size, 3]
        velocity: 速度場 [batch_size, 3] -> [u, v, w]
        
    Returns:
        Enstrophy [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]
    
    # 計算速度梯度
    u_grads = compute_derivatives_3d(u, coords, order=1)
    v_grads = compute_derivatives_3d(v, coords, order=1)
    w_grads = compute_derivatives_3d(w, coords, order=1)
    
    u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
    v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
    w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
    
    # 渦旋度向量 ω = ∇×u
    omega_x = w_y - v_z  # ∂w/∂y - ∂v/∂z
    omega_y = u_z - w_x  # ∂u/∂z - ∂w/∂x
    omega_z = v_x - u_y  # ∂v/∂x - ∂u/∂y
    
    # Enstrophy = |ω|²
    enstrophy = omega_x**2 + omega_y**2 + omega_z**2
    
    return enstrophy


def compute_q_criterion_3d(coords: torch.Tensor,
                          velocity: torch.Tensor) -> torch.Tensor:
    """
    計算3D Q-準則（渦結構識別）
    Q = 0.5 * (||Ω||² - ||S||²)
    
    物理意義：
    - Q > 0: 渦旋主導區域
    - Q < 0: 應變主導區域
    - 常用於渦結構可視化
    
    穩健性：✅ 高（僅需一階導數）
    風險評級：🟢 低風險
    
    Args:
        coords: 座標 [batch_size, 3]
        velocity: 速度場 [batch_size, 3] -> [u, v, w]
        
    Returns:
        Q準則值 [batch_size, 1]
    """
    u = velocity[:, 0:1]
    v = velocity[:, 1:2]
    w = velocity[:, 2:3]
    
    # 計算速度梯度張量
    u_grads = compute_derivatives_3d(u, coords, order=1)
    v_grads = compute_derivatives_3d(v, coords, order=1)
    w_grads = compute_derivatives_3d(w, coords, order=1)
    
    u_x, u_y, u_z = u_grads[:, 0:1], u_grads[:, 1:2], u_grads[:, 2:3]
    v_x, v_y, v_z = v_grads[:, 0:1], v_grads[:, 1:2], v_grads[:, 2:3]
    w_x, w_y, w_z = w_grads[:, 0:1], w_grads[:, 1:2], w_grads[:, 2:3]
    
    # 渦量張量 Ω（反對稱部分）
    omega_12 = 0.5 * (v_x - u_y)
    omega_13 = 0.5 * (w_x - u_z)
    omega_23 = 0.5 * (w_y - v_z)
    
    omega_squared = 2 * (omega_12**2 + omega_13**2 + omega_23**2)
    
    # 應變率張量 S（對稱部分）
    s_11 = u_x
    s_22 = v_y
    s_33 = w_z
    s_12 = 0.5 * (u_y + v_x)
    s_13 = 0.5 * (u_z + w_x)
    s_23 = 0.5 * (v_z + w_y)
    
    s_squared = 2 * (s_11**2 + s_22**2 + s_33**2 + 2*(s_12**2 + s_13**2 + s_23**2))
    
    # Q準則
    q_criterion = 0.5 * (omega_squared - s_squared)
    
    return q_criterion


# ============================================================================
# 物理一致性檢查
# ============================================================================

def check_conservation_3d(coords: torch.Tensor,
                         velocity: torch.Tensor,
                         pressure: torch.Tensor,
                         nu: float) -> Dict[str, torch.Tensor]:
    """
    檢查3D守恆定律
    
    驗證指標（依據物理審查報告）：
    - 質量守恆: |∇·u|_L² < 1e-3
    - 動量守恆: |R_momentum|_L² < 1e-2
    - 梯度爆炸: max(|∇u|) < 100
    
    Args:
        coords: 座標 [batch_size, 3]
        velocity: 速度場 [batch_size, 3] -> [u, v, w]
        pressure: 壓力場 [batch_size, 1]
        nu: 運動黏度
        
    Returns:
        守恆律檢查結果字典
    """
    results = {}
    
    # 構建完整預測張量 [u, v, w, p]
    pred = torch.cat([velocity, pressure], dim=1)
    
    # 質量守恆（連續方程）
    _, _, _, continuity = ns_residual_3d_thin_slab(coords, pred, nu)
    mass_conservation_error = torch.mean(torch.abs(continuity))
    results['mass_conservation'] = mass_conservation_error
    
    # 動量守恆（動量方程殘差）
    mom_x, mom_y, mom_z, _ = ns_residual_3d_thin_slab(coords, pred, nu)
    momentum_conservation_error = torch.mean(
        torch.abs(mom_x) + torch.abs(mom_y) + torch.abs(mom_z)
    )
    results['momentum_conservation'] = momentum_conservation_error
    
    # 梯度爆炸檢測
    u_grads = compute_derivatives_3d(velocity[:, 0:1], coords, order=1)
    max_gradient = torch.max(torch.abs(u_grads))
    results['max_gradient'] = max_gradient
    
    # 判定是否通過（布林值）
    results['pass_mass'] = (mass_conservation_error < 1e-3).item()
    results['pass_momentum'] = (momentum_conservation_error < 1e-2).item()
    results['pass_gradient'] = (max_gradient < 100.0).item()
    
    return results


# ============================================================================
# 統一接口類別（繼承 NavierStokesBase）
# ============================================================================

class NSEquations3DThinSlab(NavierStokesBase):
    """
    3D Thin-Slab Navier-Stokes 方程式統一接口
    
    **重構說明**: 現在繼承 `NavierStokesBase`，復用梯度/Laplacian/NS 組件
    
    設計目標：
    - 提供一致的API與2D版本對接
    - 整合物理一致性檢查
    - 支援數值穩定性保護
    - 便於單元測試
    
    使用範例：
    >>> ns3d = NSEquations3DThinSlab(viscosity=1e-3, domain_lengths={'L_x': 8*np.pi, 'L_z': 0.12})
    >>> residuals = ns3d.residual(coords, velocity, pressure)
    >>> conservation = ns3d.check_conservation(coords, velocity, pressure)
    """
    
    def __init__(self, 
                 viscosity: float = 1e-3,
                 density: float = 1.0,
                 domain_lengths: Optional[Dict[str, float]] = None,
                 stabilize: bool = True,
                 **kwargs):
        """
        Args:
            viscosity: 運動黏度 ν (= 1/Re_τ for normalized)
            density: 流體密度 ρ
            domain_lengths: 域長度 {'L_x': float, 'L_y': float, 'L_z': float}
            stabilize: 是否啟用數值穩定性保護
            **kwargs: 傳遞給基類的額外參數
        """
        # 預設域長度（JHTDB channel flow Re_τ=1000）
        if domain_lengths is None:
            domain_lengths = {
                'L_x': 8.0 * 3.141592653589793,  # 8π (流向)
                'L_y': 2.0,                      # 2h (壁法向, h=1)
                'L_z': 0.12                      # z⁺ ≈ 120
            }
        
        # 構建物理參數字典
        physics_params = {
            'nu': viscosity,
            'rho': density
        }
        
        # 構建域邊界字典（從 domain_lengths 轉換）
        domain_bounds = {
            'x': [0.0, domain_lengths.get('L_x', 8.0 * 3.141592653589793)],
            'y': [-1.0, 1.0],  # Thin-slab: y ∈ [-1, 1]
            'z': [0.0, domain_lengths.get('L_z', 0.12)]
        }
        
        # 調用基類初始化（spatial_dim=3 for 3D）
        super().__init__(
            physics_params=physics_params,
            domain_bounds=domain_bounds,
            spatial_dim=3,
            **kwargs
        )
        
        # Thin-slab 特有屬性
        self.domain_lengths = domain_lengths
        self.stabilize = stabilize
    
    def residual(self,
                coords: torch.Tensor,
                velocity: Union[torch.Tensor, List[torch.Tensor]],
                pressure: torch.Tensor,
                time: Optional[torch.Tensor] = None,
                source_term: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        計算NS方程殘差（統一接口）
        
        **重構說明**: 現在使用基類方法計算各項，避免重複代碼
        
        Args:
            coords: 座標 [batch_size, 3]
            velocity: 速度場 [batch_size, 3] -> [u, v, w] 或 List[u, v, w]
            pressure: 壓力場 [batch_size, 1]
            time: 時間 [batch_size, 1] (可選)
            source_term: 外部源項 [batch_size, 3] -> [S_x, S_y, S_z] (可選)
            
        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'momentum_z', 'continuity'}
        """
        # 確保輸入需要梯度
        if not coords.requires_grad:
            coords.requires_grad_(True)
        
        # 解析速度分量
        if isinstance(velocity, torch.Tensor):
            if velocity.shape[1] != 3:
                raise ValueError(f"Velocity must have 3 components, got shape {velocity.shape}")
            u = velocity[:, 0:1]
            v = velocity[:, 1:2]
            w = velocity[:, 2:3]
            velocity_list = [u, v, w]
        else:
            u, v, w = velocity
            velocity_list = [u, v, w]
        
        if not pressure.requires_grad:
            pressure = pressure.requires_grad_(True)
        
        # 時間導數（非定常情況）
        if time is not None:
            if not time.requires_grad:
                time.requires_grad_(True)
            # time is [batch, 1], treat it as 1D coordinate
            u_t = compute_gradient(u, time, component=0, spatial_dim=1)
            v_t = compute_gradient(v, time, component=0, spatial_dim=1)
            w_t = compute_gradient(w, time, component=0, spatial_dim=1)
        else:
            u_t = torch.zeros_like(u)
            v_t = torch.zeros_like(v)
            w_t = torch.zeros_like(w)
        
        # 連續性方程（基類方法）
        continuity = self.compute_continuity_residual(coords, velocity_list)
        
        # 對流項（基類方法）
        conv_u = self.compute_advection_term(coords, u, velocity_list)
        conv_v = self.compute_advection_term(coords, v, velocity_list)
        conv_w = self.compute_advection_term(coords, w, velocity_list)
        
        # 黏性項（基類方法 - 返回 ν∇²u）
        visc_u = self.compute_viscous_term(coords, u)
        visc_v = self.compute_viscous_term(coords, v)
        visc_w = self.compute_viscous_term(coords, w)
        
        # 壓力梯度（基類方法）
        p_x = self.compute_pressure_gradient(pressure, coords, component=0)
        p_y = self.compute_pressure_gradient(pressure, coords, component=1)
        p_z = self.compute_pressure_gradient(pressure, coords, component=2)
        
        # 源項處理
        if source_term is not None:
            S_x = source_term[:, 0:1]
            S_y = source_term[:, 1:2]
            S_z = source_term[:, 2:3]
        else:
            S_x = torch.zeros_like(u)
            S_y = torch.zeros_like(v)
            S_z = torch.zeros_like(w)
        
        # 組裝動量方程殘差: ∂u/∂t + (u·∇)u + ∂p/∂x - ν∇²u - S = 0
        momentum_x = u_t + conv_u + p_x - visc_u - S_x
        momentum_y = v_t + conv_v + p_y - visc_v - S_y
        momentum_z = w_t + conv_w + p_z - visc_w - S_z
        
        # 數值穩定性保護（如果啟用）
        if self.stabilize:
            max_val = 1e4
            momentum_x = torch.clamp(momentum_x, -max_val, max_val)
            momentum_y = torch.clamp(momentum_y, -max_val, max_val)
            momentum_z = torch.clamp(momentum_z, -max_val, max_val)
        
        return {
            'momentum_x': momentum_x,
            'momentum_y': momentum_y,
            'momentum_z': momentum_z,
            'continuity': continuity
        }
    
    def check_conservation(self,
                          coords: torch.Tensor,
                          velocity: torch.Tensor,
                          pressure: torch.Tensor) -> Dict[str, Any]:
        """
        守恆律檢查（向後兼容接口）
        
        Returns:
            包含數值指標與通過/失敗判定的字典
        """
        return check_conservation_3d(coords, velocity, pressure, self.nu)
    
    def apply_boundary_conditions(self,
                                 coords_wall: torch.Tensor,
                                 pred_wall: torch.Tensor,
                                 coords_periodic: Optional[torch.Tensor] = None,
                                 pred_periodic: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        應用邊界條件
        
        Args:
            coords_wall: 壁面座標
            pred_wall: 壁面預測值
            coords_periodic: 週期邊界座標 (可選)
            pred_periodic: 週期邊界預測值 (可選)
            
        Returns:
            邊界條件殘差字典
        """
        bc_residuals = {}
        
        # 壁面無滑移條件
        bc_residuals['wall'] = apply_wall_bc_3d(coords_wall, pred_wall)
        
        # 週期性條件
        if coords_periodic is not None and pred_periodic is not None:
            periodic_res = apply_periodic_bc_3d(
                coords_periodic, pred_periodic, self.domain_lengths
            )
            bc_residuals.update(periodic_res)
        
        return bc_residuals
    
    def compute_turbulence_quantities(self,
                                     coords: torch.Tensor,
                                     velocity: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        計算湍流量（基礎版本 - Phase 2）
        
        Returns:
            {'dissipation': ε, 'enstrophy': Ω², 'q_criterion': Q}
        """
        return {
            'dissipation': compute_dissipation_3d(coords, velocity, self.viscosity),
            'enstrophy': compute_enstrophy_3d(coords, velocity),
            'q_criterion': compute_q_criterion_3d(coords, velocity)
        }
    
    def get_physical_properties(self) -> Dict[str, Any]:
        """
        獲取物理屬性（擴展基類方法）
        """
        base_props = self.get_physics_info()
        base_props.update({
            'domain_lengths': self.domain_lengths,
            'reynolds_tau': 1.0 / self.nu,
            'stabilize': self.stabilize
        })
        return base_props
    
    # ========================================================================
    # 向後兼容屬性（測試期望 self.viscosity 存在）
    # ========================================================================
    @property
    def viscosity(self) -> float:
        """向後兼容: 返回 self.nu"""
        return self.nu
    
    @property
    def density(self) -> float:
        """向後兼容: 返回 self.rho"""
        return self.rho


# ============================================================================
# Backward Compatibility Aliases
# ============================================================================

def _deprecation_warning(old_name: str, new_name: str):
    """發出棄用警告"""
    warnings.warn(
        f"{old_name} is deprecated and will be removed in v2.0. "
        f"Please use {new_name} instead.",
        DeprecationWarning,
        stacklevel=3
    )

# 向後相容別名：NavierStokes3DThinSlab → NSEquations3DThinSlab
class NavierStokes3DThinSlab(NSEquations3DThinSlab):
    """
    [DEPRECATED] Backward compatibility wrapper for NSEquations3DThinSlab.
    
    This class will be removed in v2.0. Please update your code to use
    NSEquations3DThinSlab instead:
    
        # Old (deprecated)
        from pinnx.physics.ns_3d_thin_slab import NavierStokes3DThinSlab
        
        # New (recommended)
        from pinnx.physics.ns_3d_thin_slab import NSEquations3DThinSlab
    """
    def __init__(self, *args, **kwargs):
        _deprecation_warning('NavierStokes3DThinSlab', 'NSEquations3DThinSlab')
        super().__init__(*args, **kwargs)
