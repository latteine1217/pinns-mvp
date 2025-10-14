"""
VS-PINN Channel Flow 物理模块
=============================

实现基于各向异性变数缩放的 Navier-Stokes 方程求解器，专用于 JHTDB Channel Flow Re_τ=1000。

核心功能:
1. 各向异性缩放坐标变换: (X, Y, Z) = (N_x·x, N_y·y, N_z·z)
2. 链式法则梯度计算: ∂u/∂x = N_x · ∂v/∂X
3. Laplacian 变换: ∇²u = N_x² ∂²v/∂X² + N_y² ∂²v/∂Y² + N_z² ∂²v/∂Z²
4. Channel Flow 专用压降项: dP/dx = 0.0025
5. 周期性边界约束
6. Loss 权重缩放补偿

理论依据:
- arXiv:2308.08468 (VS-PINN 原始论文)
- JHTDB Channel Flow Re_τ=1000 数据规格

作者: PINNs-MVP 团队
日期: 2025-10-09
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Tuple, Dict, Optional, Any
import numpy as np

from .ns_3d_temporal import compute_derivatives_3d_temporal


def compute_gradient_3d(
    field: torch.Tensor,
    coords: torch.Tensor,
    component: int
) -> torch.Tensor:
    """
    計算 3D 穩態場的偏導數（單一分量）
    
    Args:
        field: 標量場 [batch, 1]（需要在計算圖中）
        coords: 3D 坐標 [batch, 3]（需要 requires_grad=True）
        component: 微分分量 (0=x, 1=y, 2=z)
        
    Returns:
        偏導數 [batch, 1]（保留計算圖）
        
    Note:
        此函數保證保留計算圖，適用於需要高階導數的 PINNs 訓練。
        使用 create_graph=True 確保返回的梯度張量可以進一步微分。
    """
    # 關鍵：create_graph=True 確保返回的梯度本身也在計算圖中
    grad_outputs = torch.ones_like(field)
    grads = autograd.grad(
        outputs=field,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,  # 保留計算圖以支持高階導數
        retain_graph=True,  # 保留圖以支持多次梯度計算
        only_inputs=True,
        allow_unused=False  # 確保所有輸入都被使用
    )[0]
    
    # 提取指定分量（切片操作保留計算圖）
    return grads[:, component:component+1]


class VSPINNChannelFlow(nn.Module):
    """
    VS-PINN Channel Flow 求解器
    
    实现各向异性缩放的 NS 方程，针对通道流特性优化：
    - 壁法向 (y) 最刚性，使用最大缩放因子 N_y = 8-16
    - 流向 (x) 和展向 (z) 使用较小缩放因子 N_x = N_z = 1-4
    - 压降项驱动流动: dP/dx = 0.0025
    - 周期性边界条件: x, z 方向
    
    Args:
        scaling_factors: 缩放因子字典 {'N_x': float, 'N_y': float, 'N_z': float}
        physics_params: 物理参数字典 {'nu': float, 'dP_dx': float, 'rho': float}
        domain_bounds: 域边界 {'x': (x_min, x_max), 'y': (y_min, y_max), 'z': (z_min, z_max)}
    """
    
    def __init__(
        self, 
        scaling_factors: Optional[Dict[str, float]] = None,
        physics_params: Optional[Dict[str, float]] = None,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        loss_config: Optional[Dict[str, Any]] = None,  # 🔴 新增：接收損失配置
        enable_rans: bool = False,  # ✅ TASK-008: RANS 啟用開關
        rans_model: str = "k_epsilon",  # ✅ TASK-008: RANS 模型類型
    ):
        super().__init__()
        
        # === 默认配置（基于 VS-PINN 论文与 JHTDB Channel Flow） ===
        default_scaling = {'N_x': 2.0, 'N_y': 12.0, 'N_z': 2.0}
        default_physics = {
            'nu': 5e-5,        # 动力黏度 ν = U_τ · h / Re_τ = 1.0 · 1.0 / 1000 = 1e-3 (JHTDB实际值约 5e-5)
            'dP_dx': 0.0025,   # 压降梯度（驱动流动）
            'rho': 1.0,        # 密度（标准化）
        }
        default_bounds = {
            'x': (0.0, 8.0 * np.pi),  # 流向周期域 [0, 8π]
            'y': (-1.0, 1.0),          # 壁法向 [-1, 1]
            'z': (0.0, 3.0 * np.pi),  # 展向周期域 [0, 3π]
        }
        
        # 合并用户配置
        self.scaling_factors = {**default_scaling, **(scaling_factors or {})}
        self.physics_params = {**default_physics, **(physics_params or {})}
        self.domain_bounds = {**default_bounds, **(domain_bounds or {})}
        
        # 注册缩放因子为缓冲区（不参与梯度计算）
        for key, value in self.scaling_factors.items():
            self.register_buffer(key, torch.tensor(float(value)))
        
        # 注册物理参数
        for key, value in self.physics_params.items():
            self.register_buffer(key, torch.tensor(float(value)))
        
        # 计算并缓存最大缩放因子（用于 loss 权重补偿）
        N_max_value = max(self.scaling_factors.values())
        self.register_buffer('N_max', torch.tensor(float(N_max_value)))
        self.register_buffer('N_max_sq', torch.tensor(float(N_max_value ** 2)))
        
        # === 損失歸一化參數 ===
        self.loss_normalizers: Dict[str, float] = {}  # 存儲每個損失項的參考值
        self.normalize_losses = True  # 損失歸一化開關（可通過配置控制）
        # 🔴 修正：從配置讀取 warmup_epochs，默認 5
        self.warmup_epochs = (loss_config or {}).get('warmup_epochs', 5)
        self.normalizer_momentum = 0.9  # 滑動平均動量（平滑更新）
        
        # ⭐ TASK-ENHANCED-5K-PHYSICS-FIX: PDE 損失雙重削弱修正
        # 禁用對 PDE/continuity 的額外 /N_max_sq 削弱（默認啟用修正）
        # 設為 False 可回退至舊行為（相容性）
        self.disable_extra_pde_division = (loss_config or {}).get('disable_extra_pde_division', True)
        
        # === ✅ RANS 湍流模型初始化（診斷用途）===
        # ⚠️ 變更警告（2025-10-14）：
        # RANS 計算僅用於診斷與低保真場估算，不再參與損失函數計算。
        # 原因：RANS 統計平均場與瞬時 DNS 重建不自洽，造成尺度衝突。
        # 若需作為軟先驗，建議將 RANS 場作為輸入特徵，而非硬約束損失。
        self.enable_rans = enable_rans
        self.rans_model_type = rans_model
        
        # 物理初始化開關（控制 k, ε 估算方式）
        self.rans_use_physical_init = loss_config.get('rans_use_physical_init', True) if loss_config else True
        
        if self.enable_rans:
            from .turbulence import RANSEquations3D
            
            # 湍流黏度懲罰配置（僅用於診斷報告，不參與訓練損失）
            turbulent_viscosity_penalty = loss_config.get('turbulent_viscosity_penalty', 'log1p') if loss_config else 'log1p'
            turbulent_viscosity_target = loss_config.get('turbulent_viscosity_target', 100.0) if loss_config else 100.0
            turbulent_viscosity_huber_delta = loss_config.get('turbulent_viscosity_huber_delta', 100.0) if loss_config else 100.0
            
            self.rans_model = RANSEquations3D(
                viscosity=self.physics_params['nu'],
                enable_constraints=True,
                constraint_type="softplus",
                turbulent_viscosity_penalty=turbulent_viscosity_penalty,
                turbulent_viscosity_target=turbulent_viscosity_target,
                turbulent_viscosity_huber_delta=turbulent_viscosity_huber_delta
            )
            init_mode = "物理一致初始化" if self.rans_use_physical_init else "梯度估算"
            penalty_info = f"懲罰={turbulent_viscosity_penalty}"
            if turbulent_viscosity_penalty == "huber":
                penalty_info += f" (β={turbulent_viscosity_huber_delta}ν, target={turbulent_viscosity_target}ν)"
            print(f"✅ RANS 湍流模型已啟用: {rans_model}（診斷模式，{init_mode}，{penalty_info}）")
            print(f"   ⚠️  注意：RANS 計算不參與損失函數，僅用於監控與診斷")
        else:
            self.rans_model = None
        
        # 验证配置
        self._verify_configuration()
        
        print(f"✅ VS-PINN Channel Flow 初始化完成")
        print(f"   缩放因子: N_x={self.N_x:.1f}, N_y={self.N_y:.1f}, N_z={self.N_z:.1f}")  # type: ignore[attr-defined]
        print(f"   物理参数: ν={self.nu:.2e}, dP/dx={self.dP_dx:.4f}, ρ={self.rho:.1f}")  # type: ignore[attr-defined]
        print(f"   Loss 补偿因子: 1/N_max² = 1/{self.N_max_sq:.2f}")  # type: ignore[attr-defined]
        print(f"   損失歸一化: {'啟用' if self.normalize_losses else '禁用'} (warmup={self.warmup_epochs} epochs)")
        print(f"   ⭐ PDE 額外除法: {'禁用 (修正後)' if self.disable_extra_pde_division else '啟用 (舊行為)'}")
    
    def _verify_configuration(self):
        """验证配置的物理合理性"""
        # 1. 缩放因子应满足: N_y > N_x, N_z （壁法向最刚性）
        if not (self.scaling_factors['N_y'] >= self.scaling_factors['N_x'] and 
                self.scaling_factors['N_y'] >= self.scaling_factors['N_z']):
            print(f"⚠️  警告: 壁法向缩放因子 N_y={self.scaling_factors['N_y']} 应大于 N_x, N_z")
        
        # 2. Reynolds 数一致性检查（移除不正确的验证逻辑）
        # 注释：原验证逻辑 Re_τ = 1/ν 不正确
        # 正确公式：Re_τ = U_τ · h / ν，其中 U_τ 和 h 需从流场计算或配置获取
        # 当前配置（ν=5e-5, dP/dx=0.0025）与 JHTDB Re_τ=1000 物理上自洽，无需额外验证
        
        # 3. 压降项应为正值（推动流动）
        if self.dP_dx <= 0:  # type: ignore[operator]
            raise ValueError(f"压降项 dP/dx={self.dP_dx} 必须为正值")
    
    def scale_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """
        坐标缩放变换: (x, y, z) → (X, Y, Z) = (N_x·x, N_y·y, N_z·z)
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理坐标
            
        Returns:
            scaled_coords: [batch, 3] = [X, Y, Z] 缩放坐标
        """
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        
        X = self.N_x * x  # type: ignore[operator]
        Y = self.N_y * y  # type: ignore[operator]
        Z = self.N_z * z  # type: ignore[operator]
        
        return torch.cat([X, Y, Z], dim=1)
    
    def inverse_scale_coordinates(self, scaled_coords: torch.Tensor) -> torch.Tensor:
        """
        坐标逆缩放: (X, Y, Z) → (x, y, z)
        
        Args:
            scaled_coords: [batch, 3] = [X, Y, Z]
            
        Returns:
            coords: [batch, 3] = [x, y, z]
        """
        X, Y, Z = scaled_coords[:, 0:1], scaled_coords[:, 1:2], scaled_coords[:, 2:3]
        
        x = X / self.N_x  # type: ignore[operator]
        y = Y / self.N_y  # type: ignore[operator]
        z = Z / self.N_z  # type: ignore[operator]
        
        return torch.cat([x, y, z], dim=1)
    
    def compute_gradients(
        self, 
        field: torch.Tensor, 
        coords: torch.Tensor, 
        order: int = 1,
        scaled_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算物理場對物理座標的梯度，支援 VS-PINN 的變數尺度化。
        
        Args:
            field: 標量場 [batch, 1]（如 u, v, w, p）
            coords: 原始物理坐標 [batch, 3] = [x, y, z]
            order: 微分階數 (1 或 2)
            scaled_coords: 若提供，視為模型輸入的縮放座標 (X, Y, Z)，
                           將自動套用鏈式法則回推至物理座標。
            
        Returns:
            梯度字典：
                order=1 → {'x': ∂f/∂x, 'y': ∂f/∂y, 'z': ∂f/∂z}
                order=2 → {'xx': ∂²f/∂x², ...}
        """
        base_coords = scaled_coords if scaled_coords is not None else coords
        
        if order == 1:
            grad_x_base = compute_gradient_3d(field, base_coords, component=0)
            grad_y_base = compute_gradient_3d(field, base_coords, component=1)
            grad_z_base = compute_gradient_3d(field, base_coords, component=2)
            
            if scaled_coords is not None:
                grad_x = grad_x_base * self.N_x  # type: ignore[operator]
                grad_y = grad_y_base * self.N_y  # type: ignore[operator]
                grad_z = grad_z_base * self.N_z  # type: ignore[operator]
            else:
                grad_x, grad_y, grad_z = grad_x_base, grad_y_base, grad_z_base
            
            return {'x': grad_x, 'y': grad_y, 'z': grad_z}
        
        if order == 2:
            grad_x_base = compute_gradient_3d(field, base_coords, component=0)
            grad_y_base = compute_gradient_3d(field, base_coords, component=1)
            grad_z_base = compute_gradient_3d(field, base_coords, component=2)
            
            grad_xx_base = compute_gradient_3d(grad_x_base, base_coords, component=0)
            grad_yy_base = compute_gradient_3d(grad_y_base, base_coords, component=1)
            grad_zz_base = compute_gradient_3d(grad_z_base, base_coords, component=2)
            
            if scaled_coords is not None:
                grad_xx = grad_xx_base * (self.N_x ** 2)  # type: ignore[operator]
                grad_yy = grad_yy_base * (self.N_y ** 2)  # type: ignore[operator]
                grad_zz = grad_zz_base * (self.N_z ** 2)  # type: ignore[operator]
            else:
                grad_xx, grad_yy, grad_zz = grad_xx_base, grad_yy_base, grad_zz_base
            
            return {'xx': grad_xx, 'yy': grad_yy, 'zz': grad_zz}
        
        raise ValueError(f"不支持的微分階數: {order}")
    
    def compute_laplacian(
        self, 
        field: torch.Tensor, 
        coords: torch.Tensor,
        scaled_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算 Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y² + ∂²f/∂z²
        
        Args:
            field: 标量场 [batch, 1]
            coords: 原始物理坐标 [batch, 3]（需要 requires_grad=True）
            
        Returns:
            laplacian: [batch, 1]
        """
        second_derivs = self.compute_gradients(field, coords, order=2, scaled_coords=scaled_coords)
        laplacian = second_derivs['xx'] + second_derivs['yy'] + second_derivs['zz']
        
        return laplacian
    
    def compute_momentum_residuals(
        self, 
        coords: torch.Tensor, 
        predictions: torch.Tensor,
        scaled_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        计算 3D 不可压缩 NS 方程的动量残差（稳态版本）
        
        方程（含压降项）:
            u∂u/∂x + v∂u/∂y + w∂u/∂z = -∂p/∂x + ν∇²u + dP/dx  # x 方向有驱动压降
            u∂v/∂x + v∂v/∂y + w∂v/∂z = -∂p/∂y + ν∇²v
            u∂w/∂x + v∂w/∂y + w∂w/∂z = -∂p/∂z + ν∇²w
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理坐标
            predictions: [batch, 4] = [u, v, w, p] 预测值（模型輸出，自動追蹤梯度）
            scaled_coords: 模型輸入使用的縮放座標 (X, Y, Z)，若為 None 則視為未縮放
            
        Returns:
            残差字典 {'momentum_x', 'momentum_y', 'momentum_z'}
        """
        if scaled_coords is None:
            scaled_coords = self.scale_coordinates(coords)
        
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]
        w = predictions[:, 2:3]
        p = predictions[:, 3:4]
        
        # === 计算一阶导数（对流项 + 压力项） ===
        u_grads = self.compute_gradients(u, coords, order=1, scaled_coords=scaled_coords)
        v_grads = self.compute_gradients(v, coords, order=1, scaled_coords=scaled_coords)
        w_grads = self.compute_gradients(w, coords, order=1, scaled_coords=scaled_coords)
        p_grads = self.compute_gradients(p, coords, order=1, scaled_coords=scaled_coords)
        
        # 对流项
        conv_u = u * u_grads['x'] + v * u_grads['y'] + w * u_grads['z']
        conv_v = u * v_grads['x'] + v * v_grads['y'] + w * v_grads['z']
        conv_w = u * w_grads['x'] + v * w_grads['y'] + w * w_grads['z']
        
        # 压力梯度项
        pressure_x = p_grads['x'] / self.rho  # type: ignore[operator]
        pressure_y = p_grads['y'] / self.rho  # type: ignore[operator]
        pressure_z = p_grads['z'] / self.rho  # type: ignore[operator]
        
        # === 计算二阶导数（黏性项） ===
        laplacian_u = self.compute_laplacian(u, coords, scaled_coords=scaled_coords)
        laplacian_v = self.compute_laplacian(v, coords, scaled_coords=scaled_coords)
        laplacian_w = self.compute_laplacian(w, coords, scaled_coords=scaled_coords)
        
        viscous_u = self.nu * laplacian_u  # type: ignore[operator]
        viscous_v = self.nu * laplacian_v  # type: ignore[operator]
        viscous_w = self.nu * laplacian_w  # type: ignore[operator]
        
        # === 组装残差 ===
        # x 方向动量方程（含压降驱动项）
        residual_x = conv_u + pressure_x - viscous_u - self.dP_dx  # type: ignore[operator]
        
        # y 方向动量方程
        residual_y = conv_v + pressure_y - viscous_v
        
        # z 方向动量方程
        residual_z = conv_w + pressure_z - viscous_w
        
        return {
            'momentum_x': residual_x,
            'momentum_y': residual_y,
            'momentum_z': residual_z,
        }
    
    def compute_continuity_residual(
        self, 
        coords: torch.Tensor, 
        predictions: torch.Tensor,
        scaled_coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算连续方程残差（不可压缩条件）
        
        方程:
            ∂u/∂x + ∂v/∂y + ∂w/∂z = 0
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理座標
            predictions: [batch, 4] = [u, v, w, p]（模型輸出，自動追蹤梯度）
            scaled_coords: 模型輸入使用的縮放座標 (X, Y, Z)，若為 None 則視為未縮放
            
        Returns:
            continuity_residual: [batch, 1]
        """
        if scaled_coords is None:
            scaled_coords = self.scale_coordinates(coords)
        
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]
        w = predictions[:, 2:3]
        
        # 计算散度
        u_grads = self.compute_gradients(u, coords, order=1, scaled_coords=scaled_coords)
        v_grads = self.compute_gradients(v, coords, order=1, scaled_coords=scaled_coords)
        w_grads = self.compute_gradients(w, coords, order=1, scaled_coords=scaled_coords)
        
        divergence = u_grads['x'] + v_grads['y'] + w_grads['z']
        
        return divergence
    
    def compute_rans_residuals(
        self, 
        coords: torch.Tensor, 
        predictions: torch.Tensor,
        scaled_coords: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算 RANS 湍流方程殘差（診斷模式）
        
        ⚠️ 變更警告（2025-10-14）：
        此方法計算的殘差**不再參與損失函數**，僅用於監控與診斷。
        原因：RANS 統計平均場與瞬時 DNS 重建不自洽，造成尺度衝突。
        
        僅在 enable_rans=True 時有效。從速度場估算 k（湍流動能）與 ε（耗散率），
        並計算 k-ε 方程殘差作為診斷指標（例如監控湍流黏度合理性）。
        
        方程組（用於診斷）:
            - 湍流動能: Dk/Dt = P - ε + ∇·[(ν + ν_t/σ_k) ∇k]
            - 耗散率:   Dε/Dt = (C_ε1·P - C_ε2·ε)·ε/k + ∇·[(ν + ν_t/σ_ε) ∇ε]
            - 湍流黏度: ν_t = C_μ · k²/ε
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理座標
            predictions: [batch, 4] = [u, v, w, p]（模型輸出）
            scaled_coords: 模型輸入使用的縮放座標（若為 None 則自動計算）
            
        Returns:
            殘差字典（若未啟用 RANS 則返回空字典）:
            {
                'k_equation': [batch, 1],        # k 方程殘差（診斷用）
                'epsilon_equation': [batch, 1],  # ε 方程殘差（診斷用）
                'turbulent_viscosity': [batch, 1],  # ν_t 合理性約束（診斷用）
                'physical_penalty': [batch, 1]   # k≥0, ε≥0 物理懲罰（診斷用）
            }
            
        用途範例：
            - 監控湍流黏度 ν_t/ν 比值是否合理（通道流典型值 10-100）
            - 診斷流場是否滿足 RANS 統計假設
            - 提供低保真場估算（例如作為輸入特徵的軟先驗）
        """
        if not self.enable_rans or self.rans_model is None:
            return {}
        
        # 提取速度場
        velocity = predictions[:, :3]  # [batch, 3] = [u, v, w]
        
        # 調用 RANSEquations3D.residual() 計算殘差
        # 該方法內部會：
        # 1. 從速度梯度估算 k, ε（或使用物理初始化）
        # 2. 計算 k-ε 方程殘差
        # 3. 驗證湍流黏度 ν_t 的合理性
        # ✅ TASK-008 Phase 5: 傳遞物理初始化開關
        rans_residuals = self.rans_model.residual(
            coords, 
            velocity, 
            use_physical_init=self.rans_use_physical_init
        )
        
        return rans_residuals
    
    def compute_periodic_loss(
        self, 
        coords: torch.Tensor, 
        predictions: torch.Tensor,
        boundary_band_width: float = 5e-3  # ⭐ 新增：邊界帶狀寬度
    ) -> Dict[str, torch.Tensor]:
        """
        计算周期性边界约束损失
        
        ⭐ TASK-ENHANCED-5K-PHYSICS-FIX: 週期性採樣策略修正
        - 舊行為：嚴格點匹配 (|x - x_boundary| < 1e-6)
        - 新行為：帶狀掩碼 (|x - x_boundary| < boundary_band_width)
        
        对于 x 和 z 方向的周期边界:
            u(x_min, y, z) = u(x_max, y, z)
            v(x_min, y, z) = v(x_max, y, z)
            w(x_min, y, z) = w(x_max, y, z)
            p(x_min, y, z) = p(x_max, y, z)
        
        Args:
            coords: [batch, 3] = [x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            boundary_band_width: 邊界帶狀寬度（默認 5e-3）
            
        Returns:
            周期性损失字典 {'periodic_x', 'periodic_z'}
        """
        # 提取边界坐标（需要外部提供成对的边界点）
        # 此处假设 coords 已经包含成对的边界点
        
        # ⭐ 修正：邊界帶狀掩碼（從嚴格點匹配改為近邊界區域）
        x_min, x_max = self.domain_bounds['x']
        mask_x_min = torch.abs(coords[:, 0] - x_min) < boundary_band_width
        mask_x_max = torch.abs(coords[:, 0] - x_max) < boundary_band_width
        
        # z 方向周期性
        z_min, z_max = self.domain_bounds['z']
        mask_z_min = torch.abs(coords[:, 2] - z_min) < boundary_band_width
        mask_z_max = torch.abs(coords[:, 2] - z_max) < boundary_band_width
        
        # 计算周期性误差（如果边界点存在）
        periodic_x_loss = torch.tensor(0.0, device=coords.device)
        periodic_z_loss = torch.tensor(0.0, device=coords.device)
        
        if mask_x_min.any() and mask_x_max.any():
            # 提取 x 边界的场值
            fields_x_min = predictions[mask_x_min]
            fields_x_max = predictions[mask_x_max]
            
            # 确保边界点数量匹配
            n_min = min(fields_x_min.shape[0], fields_x_max.shape[0])
            periodic_x_loss = torch.mean((fields_x_min[:n_min] - fields_x_max[:n_min]) ** 2)
        
        if mask_z_min.any() and mask_z_max.any():
            fields_z_min = predictions[mask_z_min]
            fields_z_max = predictions[mask_z_max]
            
            n_min = min(fields_z_min.shape[0], fields_z_max.shape[0])
            periodic_z_loss = torch.mean((fields_z_min[:n_min] - fields_z_max[:n_min]) ** 2)
        
        return {
            'periodic_x': periodic_x_loss,
            'periodic_z': periodic_z_loss,
        }
    
    def compute_wall_shear_stress(
        self, 
        coords: torch.Tensor, 
        predictions: torch.Tensor,
        scaled_coords: Optional[torch.Tensor] = None,
        boundary_band_width: float = 5e-3  # ⭐ 新增：近壁帶狀寬度
    ) -> Dict[str, torch.Tensor]:
        """
        计算壁面剪应力 τ_w = μ (∂u/∂y)|_{y=±1}
        
        ⭐ TASK-ENHANCED-5K-PHYSICS-FIX: 壁面採樣策略修正
        - 舊行為：嚴格點匹配 (|y - y_wall| < 1e-6)，隨機採樣極少命中
        - 新行為：帶狀掩碼 (|y - y_wall| < boundary_band_width)，穩健估計梯度
        
        Args:
            coords: [batch, 3] = [x, y, z]
            predictions: [batch, 4] = [u, v, w, p]
            scaled_coords: 模型輸入的縮放座標（可選）
            boundary_band_width: 近壁帶狀寬度（默認 5e-3）
            
        Returns:
            壁面剪应力 {'tau_w_lower', 'tau_w_upper'}
        """
        u = predictions[:, 0:1]
        
        # 计算 ∂u/∂y
        if scaled_coords is None:
            scaled_coords = self.scale_coordinates(coords)
        
        u_grads = self.compute_gradients(u, coords, order=1, scaled_coords=scaled_coords)
        du_dy = u_grads['y']
        
        # ⭐ 修正：壁面帶狀掩碼（從嚴格點匹配改為近壁區域）
        y_lower, y_upper = self.domain_bounds['y']
        mask_lower = torch.abs(coords[:, 1] - y_lower) < boundary_band_width
        mask_upper = torch.abs(coords[:, 1] - y_upper) < boundary_band_width
        
        # 计算剪应力 τ = μ ∂u/∂y
        mu = self.nu * self.rho  # type: ignore[operator]
        
        tau_w_lower = torch.tensor(0.0, device=coords.device)
        tau_w_upper = torch.tensor(0.0, device=coords.device)
        
        if mask_lower.any():
            tau_w_lower = torch.mean(torch.abs(mu * du_dy[mask_lower]))
        
        if mask_upper.any():
            tau_w_upper = torch.mean(torch.abs(mu * du_dy[mask_upper]))
        
        return {
            'tau_w_lower': tau_w_lower,
            'tau_w_upper': tau_w_upper,
        }
    
    def compute_bulk_velocity_constraint(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        target_bulk_velocity: float = 1.0,
    ) -> torch.Tensor:
        """
        计算流量约束损失：L_flux = (⟨u⟩_V - U_b)²
        
        物理意义：
        - 对于通道流，总流量应固定（流量守恒）
        - 体积流量: Q = ∫∫∫ u dV ≈ U_b · V_channel
        - U_b 是**整体体积平均速度**，不是每个 y 层的常数值
        
        修正说明（基于 Physicist 审查 - tasks/TASK-10/physics_review_new_losses.md）：
        - ❌ 错误策略：强制每层 ⟨u⟩_{x,z}(y) = U_b → 与壁面 BC (u=0) 矛盾
        - ✅ 正确策略：全域平均 ⟨u⟩_V = U_b → 约束整体流量
        
        实现策略（方案 A - 全域平均）：
        1. 计算批次内所有点的 u 平均值
        2. 与目标 U_b 比较，计算平方误差
        3. 简洁、稳健、与随机采样策略兼容
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理坐标（未使用，保持接口一致性）
            predictions: [batch, 4] = [u, v, w, p] 预测值
            target_bulk_velocity: 目标体积平均速度（JHTDB: U_b ≈ 0.99994 ≈ 1.0）
            
        Returns:
            flux_loss: 标量损失 (⟨u⟩ - U_b)² (保留梯度)
            
        Note:
            - 若 y 采样严重不均（如过采样壁面），可能引入偏差
            - 未来可扩展为方案 B（剖面积分 + y 加权），但需更复杂的分箱策略
        """
        u = predictions[:, 0:1]  # [batch, 1]
        
        # 全域平均速度（批次内所有点）
        u_global_mean = u.mean()  # 标量（保留计算图）
        
        # 流量约束损失
        flux_loss = (u_global_mean - target_bulk_velocity) ** 2
        
        return flux_loss
    
    def compute_centerline_symmetry(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        bandwidth: float = 1e-3,
        scaled_coords: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算中心线对称约束：L_sym = (∂u/∂y|_{y=0})² + v²|_{y=0}
        
        物理意义：
        - 通道中心线（y=0）为对称面
        - 主流速度 u 在中心线应有极值（驻点）：∂u/∂y = 0
        - 法向速度 v 在中心线应为零（无穿透）：v = 0
        
        修正说明（基于 Physicist 审查）：
        - ⚠️ 原实现：严格容差 tol=1e-6 → 随机采样时可能无样本
        - ✅ 新策略：使用带状区域 |y| < bandwidth → 提高采样覆盖率
        
        约束项：
        1. L_sym_dudv: (∂u/∂y|_{y=0})² → 强制主流速度梯度为零
        2. L_sym_v: v²|_{y=0} → 强制法向速度为零
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理坐标
            predictions: [batch, 4] = [u, v, w, p] 预测值
            bandwidth: 中心线带宽 ε，|y| < ε 的点视为中心线区域（默认 1e-3）
            scaled_coords: 模型輸入的縮放座標（若啟用 VS-PINN）
            
        Returns:
            Dict 包含两项损失:
                - 'centerline_dudy': (∂u/∂y|_{y≈0})² 的平均值
                - 'centerline_v': v²|_{y≈0} 的平均值
                
        Note:
            - 若批次中无中心线点，返回零损失（容错）
            - 可选扩充：⟨(∂w/∂y|_{y=0})²⟩（w 的中心线剪切）
        """
        # 提取场变量
        u = predictions[:, 0:1]  # [batch, 1]
        v = predictions[:, 1:2]  # [batch, 1]
        
        # 中心线带状区域: |y| < bandwidth
        mask_centerline = torch.abs(coords[:, 1]) < bandwidth  # [batch]
        
        # 容错：若无中心线点，返回零损失
        if not mask_centerline.any():
            zero_loss = torch.tensor(0.0, device=coords.device, requires_grad=True)
            return {
                'centerline_dudy': zero_loss,
                'centerline_v': zero_loss,
            }
        
        # === 约束 1: ∂u/∂y|_{y≈0} = 0 ===
        # 计算 u 对 y 的偏导数
        if scaled_coords is None:
            scaled_coords = self.scale_coordinates(coords)
        
        u_grads = self.compute_gradients(u, coords, order=1, scaled_coords=scaled_coords)
        du_dy = u_grads['y']  # [batch, 1]
        
        # 提取中心线区域的梯度
        du_dy_centerline = du_dy[mask_centerline]  # [n_centerline, 1]
        
        # 损失：梯度平方的均值
        loss_dudy = torch.mean(du_dy_centerline ** 2)
        
        # === 约束 2: v|_{y≈0} = 0 ===
        # 提取中心线区域的法向速度
        v_centerline = v[mask_centerline]  # [n_centerline, 1]
        
        # 损失：速度平方的均值
        loss_v = torch.mean(v_centerline ** 2)
        
        return {
            'centerline_dudy': loss_dudy,
            'centerline_v': loss_v,
        }
    
    def compute_pressure_reference(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        reference_point: Optional[Tuple[float, float, float]] = None,
        k_nearest: int = 16,
    ) -> torch.Tensor:
        """
        计算压力参考点约束：L_pref = (mean(p_k))²
        
        物理意义：
        - 不可压缩 NS 方程只确定压力梯度，压力场存在任意常数自由度
        - 固定参考点（或区域）压力可消除该自由度，使压力场唯一确定
        - 对于周期性通道流，已有 dP/dx 固定梯度，此约束优先级较低
        
        修正说明（基于 Physicist 审查）：
        - ⚠️ 原实现：严格坐标匹配 → 批次中常无参考点，长期返回零损失
        - ✅ 新策略：k 最近邻平均 → 稳健覆盖参考区域
        
        实现策略：
        1. 计算批次中所有点到参考点的距离
        2. 选取最近的 k 个点（k=8-16）
        3. 对这些点的压力取平均，强制均值为零
        
        Args:
            coords: [batch, 3] = [x, y, z] 物理坐标
            predictions: [batch, 4] = [u, v, w, p] 预测值
            reference_point: 可选的参考点坐标 (x, y, z)，默认为域中心
            k_nearest: 最近邻点数（默认 16）
            
        Returns:
            pressure_ref_loss: 标量损失 (mean(p_k))² (保留梯度)
            
        Note:
            - 当前 dP/dx 已隐式固定压力梯度，此约束为可选项
            - 仅在需要绝对压力对比时启用（如与实验数据比对）
            - 建议权重 λ_pref ≈ 0.01（低优先级）
        """
        p = predictions[:, 3:4]  # [batch, 1]
        
        # 设置参考点（默认为域中心）
        if reference_point is None:
            x_center = (self.domain_bounds['x'][0] + self.domain_bounds['x'][1]) / 2
            y_center = (self.domain_bounds['y'][0] + self.domain_bounds['y'][1]) / 2
            z_center = (self.domain_bounds['z'][0] + self.domain_bounds['z'][1]) / 2
            reference_point = (x_center, y_center, z_center)
        
        x_ref, y_ref, z_ref = reference_point
        ref_tensor = torch.tensor([x_ref, y_ref, z_ref], device=coords.device, dtype=coords.dtype)  # [3]
        
        # 计算所有点到参考点的欧氏距离
        distances = torch.norm(coords - ref_tensor.unsqueeze(0), dim=1)  # [batch]
        
        # 选取最近的 k 个点
        k_actual = min(k_nearest, coords.size(0))  # 防止 k > batch_size
        _, top_k_indices = torch.topk(distances, k=k_actual, largest=False)  # [k]
        
        # 提取 k 最近邻的压力
        p_k_nearest = p[top_k_indices]  # [k, 1]
        
        # 损失：压力均值的平方（强制参考区域压力为零）
        p_mean = torch.mean(p_k_nearest)
        pressure_ref_loss = p_mean ** 2
        
        return pressure_ref_loss
    
    def compute_loss_weight_compensation(self) -> float:
        """
        计算 Loss 权重的缩放补偿因子
        
        理论依据:
        - 缩放后的残差会被 N_max² 放大
        - 需在 loss 权重中除以 N_max² 抵消
        
        Returns:
            compensation_factor: 1 / N_max²
        """
        return 1.0 / self.N_max_sq.item()  # type: ignore[operator, union-attr]
    
    def normalize_loss_dict(
        self, 
        loss_dict: Dict[str, torch.Tensor], 
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        損失歸一化：將每個損失項除以其參考值，使所有損失在同一數量級
        
        策略：
        1. Warmup (epoch < warmup_epochs): 收集每個損失項的初始值作為參考
        2. Training: 使用參考值進行歸一化
        3. 使用滑動平均更新參考值，避免初始值不穩定
        
        Args:
            loss_dict: 原始損失字典 {'loss_name': tensor}
            epoch: 當前訓練 epoch
            
        Returns:
            normalized_loss_dict: 歸一化後的損失字典
            
        Note:
            - 歸一化不改變損失的相對大小關係，保持物理一致性
            - 只調整絕對尺度，讓權重能直接反映優先級
        """
        if not self.normalize_losses:
            return loss_dict
        
        # === Warmup 階段：收集統計 ===
        if epoch < self.warmup_epochs:
            for key, loss in loss_dict.items():
                loss_val = loss.detach().item()
                
                if key not in self.loss_normalizers:
                    # 首次記錄
                    self.loss_normalizers[key] = loss_val
                else:
                    # 滑動平均更新（避免單次異常值）
                    self.loss_normalizers[key] = (
                        self.normalizer_momentum * self.loss_normalizers[key] +
                        (1 - self.normalizer_momentum) * loss_val
                    )
            
            # Warmup 期間不進行歸一化
            return loss_dict
        
        # === Training 階段：歸一化 ===
        normalized = {}
        for key, loss in loss_dict.items():
            normalizer = self.loss_normalizers.get(key, 1.0)
            if normalizer < 1e-12:
                normalizer = 1.0

            normalized_loss = loss / normalizer

            # ⭐ TASK-ENHANCED-5K-PHYSICS-FIX: 條件性 PDE 削弱修正
            # 舊行為：對 PDE/continuity 額外除以 N_max_sq（可能導致權重過低）
            # 新行為：默認跳過額外除法，讓外部權重補償機制正常運作
            if not self.disable_extra_pde_division:  # 舊行為（向後相容）
                if key in {
                    'momentum_x',
                    'momentum_y',
                    'momentum_z',
                    'continuity',
                    'periodicity'
                }:
                    normalized_loss = normalized_loss / self.N_max_sq  # type: ignore[operator]

            normalized[key] = normalized_loss
        
        return normalized
    
    def get_normalization_info(self) -> Dict[str, Any]:
        """
        獲取損失歸一化信息摘要
        
        Returns:
            info: 包含歸一化狀態與參考值的字典
        """
        return {
            'enabled': self.normalize_losses,
            'warmup_epochs': self.warmup_epochs,
            'normalizers': self.loss_normalizers.copy(),
            'momentum': self.normalizer_momentum,
        }
    
    def get_scaling_info(self) -> Dict[str, Any]:
        """获取缩放信息摘要"""
        return {
            'scaling_factors': {
                'N_x': self.N_x.item(),  # type: ignore[union-attr]
                'N_y': self.N_y.item(),  # type: ignore[union-attr]
                'N_z': self.N_z.item(),  # type: ignore[union-attr]
                'N_max': self.N_max.item(),  # type: ignore[union-attr]
            },
            'physics_parameters': {
                'nu': self.nu.item(),  # type: ignore[union-attr]
                'dP_dx': self.dP_dx.item(),  # type: ignore[union-attr]
                'rho': self.rho.item(),  # type: ignore[union-attr]
            },
            'loss_compensation': {
                'factor': self.compute_loss_weight_compensation(),
                'formula': 'λ_pde_effective = λ_pde / N_max²',
            },
            'domain_bounds': self.domain_bounds,
        }


# ==============================================================================
# 便捷函数
# ==============================================================================

def create_vs_pinn_channel_flow(
    N_y: float = 12.0,
    N_x: float = 2.0,
    N_z: float = 2.0,
    nu: float = 5e-5,
    dP_dx: float = 0.0025,
    rho: float = 1.0,
    enable_rans: bool = False,  # ✅ TASK-008: RANS 啟用開關
    rans_model: str = "k_epsilon",  # ✅ TASK-008: RANS 模型類型
    **kwargs
) -> VSPINNChannelFlow:
    """
    创建 VS-PINN Channel Flow 求解器的便捷函数
    
    Args:
        N_y: 壁法向缩放因子（推荐 8-16）
        N_x: 流向缩放因子（推荐 1-4）
        N_z: 展向缩放因子（推荐 1-4）
        nu: 动力黏度
        dP_dx: 压降梯度
        rho: 密度
        enable_rans: 啟用 RANS 湍流模型約束（默認 False）
        rans_model: RANS 模型類型（'k_epsilon' | 'k_omega'，默認 'k_epsilon'）
        **kwargs: 其他参数（如 domain_bounds, loss_config）
        
    Returns:
        VSPINNChannelFlow 实例
    """
    scaling_factors = {'N_x': N_x, 'N_y': N_y, 'N_z': N_z}
    physics_params = {'nu': nu, 'dP_dx': dP_dx, 'rho': rho}
    
    return VSPINNChannelFlow(
        scaling_factors=scaling_factors,
        physics_params=physics_params,
        enable_rans=enable_rans,
        rans_model=rans_model,
        **kwargs
    )
