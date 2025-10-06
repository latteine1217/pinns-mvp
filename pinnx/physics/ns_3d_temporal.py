"""
Navier-Stokes 3D 時間依賴方程式模組
=======================================

提供3D不可壓縮時間依賴NS方程的物理定律計算功能：
1. 3D NS方程殘差計算 (動量方程 + 連續方程)
2. 時間導數項處理
3. 3D渦量計算與Q準則  
4. 自動微分梯度計算
5. 守恆定律檢查
6. 邊界條件處理

輸入格式: [t, x, y, z] (時間 + 3D空間座標)
輸出格式: [u, v, w, p] (3D速度 + 壓力)
"""

import torch
import torch.autograd as autograd
from typing import Tuple, Optional, Dict, Any
import warnings

def compute_derivatives_3d_temporal(f: torch.Tensor, x: torch.Tensor, 
                                  order: int = 1, component: Optional[int] = None) -> torch.Tensor:
    """
    計算3D時間依賴函數的偏微分
    
    Args:
        f: 待微分的標量場 [batch_size, 1]
        x: 座標變數 [batch_size, 4] = [t, x, y, z] 
        order: 微分階數 (1 或 2)
        component: 指定微分變數 (0=t, 1=x, 2=y, 3=z)，None表示全部
        
    Returns:
        偏微分結果 [batch_size, 4] (一階) 或指定分量 [batch_size, 1]
    """
    if not f.requires_grad:
        f.requires_grad_(True)
    if not x.requires_grad:
        x.requires_grad_(True)
        
    # 計算一階偏微分
    grad_outputs = torch.ones_like(f)
    grads = autograd.grad(
        outputs=f, 
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )
    
    first_derivs = grads[0]
    if first_derivs is None:
        # 修復：當梯度為None時，根據是否指定component返回正確形狀的零張量
        if component is not None:
            return torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
        else:
            return torch.zeros_like(x)
    
    if order == 1:
        if component is not None:
            return first_derivs[:, component:component+1]
        return first_derivs
    
    elif order == 2:
        # 計算二階偏微分
        if component is None:
            # 計算對角項 (Laplacian所需)
            second_derivs = []
            for i in range(x.shape[1]):
                grad_i = first_derivs[:, i:i+1]
                grad2_outputs = torch.ones_like(grad_i)
                grad2 = autograd.grad(
                    outputs=grad_i,
                    inputs=x,
                    grad_outputs=grad2_outputs,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                    allow_unused=True
                )[0]
                
                if grad2 is not None:
                    second_derivs.append(grad2[:, i:i+1])
                else:
                    # 修復：返回正確形狀的零張量
                    second_derivs.append(torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device))
            
            return torch.cat(second_derivs, dim=1)
        else:
            # 指定分量的二階偏微分
            grad_i = first_derivs[:, component:component+1]
            grad2_outputs = torch.ones_like(grad_i)
            grad2 = autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=grad2_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
                allow_unused=True
            )[0]
            
            return grad2[:, component:component+1] if grad2 is not None else torch.zeros(x.shape[0], 1, dtype=x.dtype, device=x.device)
    
    else:
        raise ValueError(f"不支援的微分階數: {order}")


class NSEquations3DTemporal:
    """
    3D時間依賴不可壓縮Navier-Stokes方程式類別
    
    處理輸入: [t, x, y, z] 
    處理輸出: [u, v, w, p]
    
    方程組:
    ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂z = -∂p/∂x + ν∇²u
    ∂v/∂t + u∂v/∂x + v∂v/∂y + w∂v/∂z = -∂p/∂y + ν∇²v  
    ∂w/∂t + u∂w/∂x + v∂w/∂y + w∂w/∂z = -∂p/∂z + ν∇²w
    ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
    """
    
    def __init__(self, viscosity: float = 0.01, density: float = 1.0):
        """
        初始化3D時間依賴NS方程求解器
        
        Args:
            viscosity: 動黏滯係數 ν
            density: 流體密度 ρ (一般設為1)
        """
        self.nu = viscosity
        self.rho = density
        
        print(f"🌊 NS方程3D時間依賴求解器初始化")
        print(f"   動黏滯係數: ν = {self.nu}")
        print(f"   流體密度: ρ = {self.rho}")
    
    def compute_momentum_residuals(self, coords: torch.Tensor, 
                                 predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算3D動量方程殘差
        
        Args:
            coords: [batch, 4] = [t, x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            
        Returns:
            (x方向動量殘差, y方向動量殘差, z方向動量殘差)
        """
        u, v, w, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3], predictions[:, 3:4]
        
        # === 時間導數項 ===
        u_t = compute_derivatives_3d_temporal(u, coords, order=1, component=0)
        v_t = compute_derivatives_3d_temporal(v, coords, order=1, component=0)  
        w_t = compute_derivatives_3d_temporal(w, coords, order=1, component=0)
        
        # === 空間一階導數項 ===
        u_x = compute_derivatives_3d_temporal(u, coords, order=1, component=1)
        u_y = compute_derivatives_3d_temporal(u, coords, order=1, component=2)
        u_z = compute_derivatives_3d_temporal(u, coords, order=1, component=3)
        
        v_x = compute_derivatives_3d_temporal(v, coords, order=1, component=1)
        v_y = compute_derivatives_3d_temporal(v, coords, order=1, component=2)
        v_z = compute_derivatives_3d_temporal(v, coords, order=1, component=3)
        
        w_x = compute_derivatives_3d_temporal(w, coords, order=1, component=1)
        w_y = compute_derivatives_3d_temporal(w, coords, order=1, component=2)
        w_z = compute_derivatives_3d_temporal(w, coords, order=1, component=3)
        
        p_x = compute_derivatives_3d_temporal(p, coords, order=1, component=1)
        p_y = compute_derivatives_3d_temporal(p, coords, order=1, component=2)
        p_z = compute_derivatives_3d_temporal(p, coords, order=1, component=3)
        
        # === 對流項 (非線性項) ===
        conv_u = u * u_x + v * u_y + w * u_z
        conv_v = u * v_x + v * v_y + w * v_z
        conv_w = u * w_x + v * w_y + w * w_z
        
        # === 二階導數項 (黏性項) ===
        u_xx = compute_derivatives_3d_temporal(u_x, coords, order=1, component=1)
        u_yy = compute_derivatives_3d_temporal(u_y, coords, order=1, component=2)
        u_zz = compute_derivatives_3d_temporal(u_z, coords, order=1, component=3)
        laplacian_u = u_xx + u_yy + u_zz
        
        v_xx = compute_derivatives_3d_temporal(v_x, coords, order=1, component=1)
        v_yy = compute_derivatives_3d_temporal(v_y, coords, order=1, component=2)
        v_zz = compute_derivatives_3d_temporal(v_z, coords, order=1, component=3)
        laplacian_v = v_xx + v_yy + v_zz
        
        w_xx = compute_derivatives_3d_temporal(w_x, coords, order=1, component=1)
        w_yy = compute_derivatives_3d_temporal(w_y, coords, order=1, component=2)
        w_zz = compute_derivatives_3d_temporal(w_z, coords, order=1, component=3)
        laplacian_w = w_xx + w_yy + w_zz
        
        # === 動量方程殘差 ===
        # ∂u/∂t + u∂u/∂x + v∂u/∂y + w∂u/∂z = -∂p/∂x + ν∇²u
        residual_u = u_t + conv_u - p_x / self.rho - self.nu * laplacian_u
        
        # ∂v/∂t + u∂v/∂x + v∂v/∂y + w∂v/∂z = -∂p/∂y + ν∇²v
        residual_v = v_t + conv_v - p_y / self.rho - self.nu * laplacian_v
        
        # ∂w/∂t + u∂w/∂x + v∂w/∂y + w∂w/∂z = -∂p/∂z + ν∇²w
        residual_w = w_t + conv_w - p_z / self.rho - self.nu * laplacian_w
        
        return residual_u, residual_v, residual_w
    
    def compute_continuity_residual(self, coords: torch.Tensor, 
                                  predictions: torch.Tensor) -> torch.Tensor:
        """
        計算連續方程殘差 (不可壓縮條件)
        
        Args:
            coords: [batch, 4] = [t, x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            
        Returns:
            連續方程殘差: ∂u/∂x + ∂v/∂y + ∂w/∂z
        """
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        u_x = compute_derivatives_3d_temporal(u, coords, order=1, component=1)
        v_y = compute_derivatives_3d_temporal(v, coords, order=1, component=2)
        w_z = compute_derivatives_3d_temporal(w, coords, order=1, component=3)
        
        # 不可壓縮條件: ∇·u = ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        divergence = u_x + v_y + w_z
        
        return divergence
    
    def compute_vorticity_3d(self, coords: torch.Tensor, 
                           predictions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        計算3D渦量向量 ω = ∇ × u
        
        Args:
            coords: [batch, 4] = [t, x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            
        Returns:
            (ωx, ωy, ωz) 渦量向量的三個分量
        """
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        # 計算速度梯度
        u_y = compute_derivatives_3d_temporal(u, coords, order=1, component=2)
        u_z = compute_derivatives_3d_temporal(u, coords, order=1, component=3)
        
        v_x = compute_derivatives_3d_temporal(v, coords, order=1, component=1)
        v_z = compute_derivatives_3d_temporal(v, coords, order=1, component=3)
        
        w_x = compute_derivatives_3d_temporal(w, coords, order=1, component=1)
        w_y = compute_derivatives_3d_temporal(w, coords, order=1, component=2)
        
        # 渦量向量分量
        # ωx = ∂w/∂y - ∂v/∂z
        omega_x = w_y - v_z
        
        # ωy = ∂u/∂z - ∂w/∂x
        omega_y = u_z - w_x
        
        # ωz = ∂v/∂x - ∂u/∂y
        omega_z = v_x - u_y
        
        return omega_x, omega_y, omega_z
    
    def compute_q_criterion_3d(self, coords: torch.Tensor, 
                             predictions: torch.Tensor) -> torch.Tensor:
        """
        計算3D Q準則: Q = 0.5(|Ω|² - |S|²)
        
        Q > 0 的區域表示渦核區域
        
        Args:
            coords: [batch, 4] = [t, x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            
        Returns:
            Q準則標量場
        """
        u, v, w = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]
        
        # 計算速度梯度張量
        u_x = compute_derivatives_3d_temporal(u, coords, order=1, component=1)
        u_y = compute_derivatives_3d_temporal(u, coords, order=1, component=2)
        u_z = compute_derivatives_3d_temporal(u, coords, order=1, component=3)
        
        v_x = compute_derivatives_3d_temporal(v, coords, order=1, component=1)
        v_y = compute_derivatives_3d_temporal(v, coords, order=1, component=2)
        v_z = compute_derivatives_3d_temporal(v, coords, order=1, component=3)
        
        w_x = compute_derivatives_3d_temporal(w, coords, order=1, component=1)
        w_y = compute_derivatives_3d_temporal(w, coords, order=1, component=2)
        w_z = compute_derivatives_3d_temporal(w, coords, order=1, component=3)
        
        # 應變率張量 S = 0.5(∇u + ∇uᵀ)
        s_xx = u_x
        s_yy = v_y
        s_zz = w_z
        s_xy = 0.5 * (u_y + v_x)
        s_xz = 0.5 * (u_z + w_x)
        s_yz = 0.5 * (v_z + w_y)
        
        # 渦量張量 Ω = 0.5(∇u - ∇uᵀ)
        omega_xy = 0.5 * (v_x - u_y)
        omega_xz = 0.5 * (w_x - u_z)
        omega_yz = 0.5 * (w_y - v_z)
        
        # |S|² = Tr(S·Sᵀ)
        s_magnitude_sq = s_xx**2 + s_yy**2 + s_zz**2 + 2*(s_xy**2 + s_xz**2 + s_yz**2)
        
        # |Ω|² = Tr(Ω·Ωᵀ)  
        omega_magnitude_sq = 2*(omega_xy**2 + omega_xz**2 + omega_yz**2)
        
        # Q準則
        q_criterion = 0.5 * (omega_magnitude_sq - s_magnitude_sq)
        
        return q_criterion
    
    def compute_energy_residual(self, coords: torch.Tensor, 
                              predictions: torch.Tensor) -> torch.Tensor:
        """
        計算動能方程殘差 (可選約束)
        
        ∂(½|u|²)/∂t + ∇·(u·½|u|²) = -∇·(pu) + ν∇²(½|u|²) - ν|∇u|²
        """
        u, v, w, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3], predictions[:, 3:4]
        
        # 計算動能 k = ½|u|²
        kinetic_energy = 0.5 * (u**2 + v**2 + w**2)
        
        # 時間導數
        k_t = compute_derivatives_3d_temporal(kinetic_energy, coords, order=1, component=0)
        
        # 對流項 ∇·(u·k)
        k_x = compute_derivatives_3d_temporal(kinetic_energy, coords, order=1, component=1)
        k_y = compute_derivatives_3d_temporal(kinetic_energy, coords, order=1, component=2)
        k_z = compute_derivatives_3d_temporal(kinetic_energy, coords, order=1, component=3)
        
        convection_k = u * k_x + v * k_y + w * k_z
        
        # 壓力功項 ∇·(pu)
        pu = p * u
        pv = p * v  
        pw = p * w
        
        pu_x = compute_derivatives_3d_temporal(pu, coords, order=1, component=1)
        pv_y = compute_derivatives_3d_temporal(pv, coords, order=1, component=2)
        pw_z = compute_derivatives_3d_temporal(pw, coords, order=1, component=3)
        
        pressure_work = pu_x + pv_y + pw_z
        
        # 黏性耗散項 (簡化)
        u_x = compute_derivatives_3d_temporal(u, coords, order=1, component=1)
        u_y = compute_derivatives_3d_temporal(u, coords, order=1, component=2)
        u_z = compute_derivatives_3d_temporal(u, coords, order=1, component=3)
        
        v_x = compute_derivatives_3d_temporal(v, coords, order=1, component=1)
        v_y = compute_derivatives_3d_temporal(v, coords, order=1, component=2)
        v_z = compute_derivatives_3d_temporal(v, coords, order=1, component=3)
        
        w_x = compute_derivatives_3d_temporal(w, coords, order=1, component=1)
        w_y = compute_derivatives_3d_temporal(w, coords, order=1, component=2)
        w_z = compute_derivatives_3d_temporal(w, coords, order=1, component=3)
        
        dissipation = u_x**2 + u_y**2 + u_z**2 + v_x**2 + v_y**2 + v_z**2 + w_x**2 + w_y**2 + w_z**2
        
        # 動能方程殘差
        energy_residual = k_t + convection_k + pressure_work / self.rho - self.nu * dissipation
        
        return energy_residual
    
    def compute_all_residuals(self, coords: torch.Tensor, 
                            predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        計算所有物理方程殘差
        
        Args:
            coords: [batch, 4] = [t, x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            
        Returns:
            包含所有殘差的字典
        """
        residuals = {}
        
        # 1. 動量方程
        res_u, res_v, res_w = self.compute_momentum_residuals(coords, predictions)
        residuals['momentum_u'] = res_u
        residuals['momentum_v'] = res_v  
        residuals['momentum_w'] = res_w
        
        # 2. 連續方程
        residuals['continuity'] = self.compute_continuity_residual(coords, predictions)
        
        # 3. 渦量 (用於約束)
        omega_x, omega_y, omega_z = self.compute_vorticity_3d(coords, predictions)
        residuals['vorticity_x'] = omega_x
        residuals['vorticity_y'] = omega_y
        residuals['vorticity_z'] = omega_z
        
        # 4. Q準則
        residuals['q_criterion'] = self.compute_q_criterion_3d(coords, predictions)
        
        # 5. 能量方程 (可選)
        residuals['energy'] = self.compute_energy_residual(coords, predictions)
        
        return residuals
    
    def compute_loss(self, coords: torch.Tensor, 
                    predictions: torch.Tensor, 
                    weights: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """
        計算總物理損失
        
        Args:
            coords: [batch, 4] = [t, x, y, z]  
            predictions: [batch, 4] = [u, v, w, p]
            weights: 各項損失權重
            
        Returns:
            總損失
        """
        if weights is None:
            weights = {
                'momentum_u': 1.0, 'momentum_v': 1.0, 'momentum_w': 1.0,
                'continuity': 1.0, 'energy': 0.1
            }
        
        residuals = self.compute_all_residuals(coords, predictions)
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        
        # 動量方程損失  
        for component in ['momentum_u', 'momentum_v', 'momentum_w']:
            if component in residuals:
                weight = weights.get(component, 1.0)
                loss_component = torch.mean(residuals[component]**2)
                total_loss = total_loss + weight * loss_component
        
        # 連續方程損失
        if 'continuity' in residuals:
            weight = weights.get('continuity', 1.0)
            continuity_loss = torch.mean(residuals['continuity']**2)
            total_loss = total_loss + weight * continuity_loss
        
        # 能量方程損失 (可選)
        if 'energy' in residuals and weights.get('energy', 0) > 0:
            weight = weights.get('energy', 0.1)
            energy_loss = torch.mean(residuals['energy']**2)
            total_loss = total_loss + weight * energy_loss
            
        return total_loss
    
    def residual(self, 
                coords: torch.Tensor, 
                velocity: torch.Tensor, 
                pressure: torch.Tensor,
                time: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        計算3D時間依賴NS方程殘差 (相容介面方法)
        
        Args:
            coords: 空間座標 [batch_size, 3] = [x, y, z]
            velocity: 速度場 [batch_size, 3] = [u, v, w] 
            pressure: 壓力場 [batch_size, 1]
            time: 時間座標 [batch_size, 1] (必要)
            
        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'momentum_z', 'continuity'}
        """
        if time is None:
            raise ValueError("時間座標是3D時間依賴NS方程的必要輸入")
        
        # 組合座標為4D格式 [t, x, y, z]
        if coords.shape[1] == 3:  # [x, y, z]
            coords_4d = torch.cat([time, coords], dim=1)  # [t, x, y, z]
        else:
            coords_4d = coords  # 假設已經是4D格式
        
        # 組合預測為4D格式 [u, v, w, p]
        predictions = torch.cat([velocity, pressure], dim=1)
        
        # 計算所有殘差
        all_residuals = self.compute_all_residuals(coords_4d, predictions)
        
        # 返回相容格式的殘差
        residuals = {
            'momentum_x': all_residuals['momentum_u'],
            'momentum_y': all_residuals['momentum_v'], 
            'momentum_z': all_residuals['momentum_w'],
            'continuity': all_residuals['continuity']
        }
        
        return residuals