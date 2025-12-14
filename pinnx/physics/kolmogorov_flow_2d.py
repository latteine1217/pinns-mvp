"""
Kolmogorov Flow 2D 物理模組（重構版本）
=======================================

實現 2D Kolmogorov flow 的物理定律計算功能：
1. 含正弦強迫項的 NS 方程殘差計算
2. 嚴格週期性邊界條件（x, y 方向）
3. 渦度與 enstrophy 計算
4. 守恆定律檢查
5. 標準雷諾數定義：Re = F / (ν² k³)

物理背景：
---------
Kolmogorov flow 是二維不可壓縮流體在週期性域上受正弦強迫驅動的經典湍流模型。
控制方程：
    ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u + A sin(k_f y)  (x-動量 + 強迫項)
    ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v                  (y-動量)
    ∂u/∂x + ∂v/∂y = 0                                          (不可壓縮)

邊界條件：
    雙週期性：u(0, y, t) = u(2π, y, t), u(x, 0, t) = u(x, 2π, t)

雷諾數定義：
    Re = F / (ν² k³)
其中 F = A（強迫振幅），ν = 動力黏度，k = k_f（強迫波數）

重構改進：
- 繼承 NavierStokesBase 以消除重複代碼
- 保留 Kolmogorov 特有的強迫項計算
- 保持完整向後兼容性

作者：PINNs-MVP 團隊
日期：2025-12-15 (重構)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional, Any, Union
import numpy as np
import logging

from .base.ns_base import NavierStokesBase
from .base.gradient_ops import compute_gradient


# ==============================================================================
# Backward Compatibility: Legacy Gradient Functions
# ==============================================================================

def compute_gradient_2d(
    field: torch.Tensor,
    coords: torch.Tensor,
    component: int
) -> torch.Tensor:
    """
    計算 2D 場的偏導數（向後兼容包裝器）
    
    ⚠️ DEPRECATED: 請使用 pinnx.physics.base.gradient_ops.compute_gradient()

    Args:
        field: 標量場 [batch, 1]
        coords: 坐標 [batch, N]（N >= 2）
        component: 微分分量 (0=x, 1=y)

    Returns:
        偏導數 [batch, 1]
    """
    return compute_gradient(field, coords, component=component, spatial_dim=2)


def compute_laplacian_2d(
    field: torch.Tensor,
    coords: torch.Tensor
) -> torch.Tensor:
    """
    計算 2D Laplacian（向後兼容包裝器）
    
    ⚠️ DEPRECATED: 請使用 pinnx.physics.base.laplacian_ops.compute_laplacian()

    Args:
        field: 標量場 [batch, 1]
        coords: 2D 坐標 [batch, 2]

    Returns:
        laplacian: [batch, 1]
    """
    from .base.laplacian_ops import compute_laplacian as compute_laplacian_base
    return compute_laplacian_base(field, coords, spatial_dim=2)


# ==============================================================================
# Main Class: KolmogorovFlow2D (Refactored)
# ==============================================================================

class KolmogorovFlow2D(NavierStokesBase):
    """
    2D Kolmogorov Flow 求解器（重構版本）

    繼承自 NavierStokesBase，提供：
    - 標準 2D N-S 方程功能（連續方程、對流項、黏性項）
    - Kolmogorov 特有的正弦強迫項：f_x = A sin(k_f y)
    - 雙週期性邊界條件
    - 雷諾數計算（Kolmogorov 定義）
    - 向後兼容的 API

    物理特性：
    - 雙週期性邊界條件（x, y 方向）
    - 正弦強迫項：f_x = A sin(k_f y)
    - 無壁面邊界條件（純流體域）
    - 雷諾數依賴的轉捩行為
    - 標準雷諾數定義：Re = F / (ν² k³)

    Args:
        forcing_params: 強迫項參數 {'amplitude': float, 'wavenumber': int}
        physics_params: 物理參數 {'nu': float, 'rho': float}
        domain_bounds: 域邊界 {'x': (x_min, x_max), 'y': (y_min, y_max)}
        loss_config: 損失配置（可選）
    """

    def __init__(
        self,
        forcing_params: Optional[Dict[str, float]] = None,
        physics_params: Optional[Dict[str, float]] = None,
        domain_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        loss_config: Optional[Dict[str, Any]] = None,
    ):
        # === 預設配置 ===
        default_forcing = {
            'amplitude': 1.0,      # 強迫振幅 A
            'wavenumber': 4,       # 強迫波數 k_f
        }
        default_physics = {
            'nu': 0.01,            # 動力黏度
            'rho': 1.0,            # 密度（標準化）
        }
        default_bounds = {
            'x': (0.0, 2.0 * np.pi),  # x 週期域 [0, 2π]
            'y': (0.0, 2.0 * np.pi),  # y 週期域 [0, 2π]
        }

        # 合併用戶配置
        self.forcing_params = {**default_forcing, **(forcing_params or {})}
        physics_params_merged = {**default_physics, **(physics_params or {})}
        domain_bounds_merged = {**default_bounds, **(domain_bounds or {})}
        
        # 計算雷諾數（Kolmogorov 定義）
        Re = self._compute_kolmogorov_reynolds(
            self.forcing_params['amplitude'],
            physics_params_merged['nu'],
            self.forcing_params['wavenumber']
        )
        physics_params_merged['Re'] = Re

        # 調用基類初始化
        super().__init__(
            physics_params=physics_params_merged,
            domain_bounds=domain_bounds_merged,
            spatial_dim=2
        )

        # 註冊強迫參數為緩衝區（不參與梯度計算）
        self.register_buffer('amplitude', torch.tensor(float(self.forcing_params['amplitude'])))
        self.register_buffer('wavenumber', torch.tensor(float(self.forcing_params['wavenumber'])))
        
        # 將物理參數也轉為 tensor 以保持向後兼容（測試期望 tensor 類型）
        # 保存原始 float 值，刪除 float 屬性，並創建 tensor buffer
        nu_value = float(self.nu)
        rho_value = float(self.rho)
        delattr(self, 'nu')
        delattr(self, 'rho')
        self.register_buffer('nu', torch.tensor(nu_value))
        self.register_buffer('rho', torch.tensor(rho_value))

        # === 損失歸一化參數（向後兼容） ===
        self.loss_normalizers: Dict[str, float] = {}
        self.normalize_losses = True
        self.warmup_epochs = (loss_config or {}).get('warmup_epochs', 5)
        self.normalizer_momentum = 0.9

        # 驗證配置
        self._verify_configuration()

        print(f"✅ Kolmogorov Flow 2D 初始化完成")
        print(f"   強迫參數: A={self.amplitude:.2f}, k_f={self.wavenumber:.0f}")
        print(f"   物理參數: ν={self.nu:.2e}, ρ={self.rho:.1f}")
        print(f"   域範圍: x∈{domain_bounds_merged['x']}, y∈{domain_bounds_merged['y']}")
        print(f"   雷諾數: Re={self.Re:.2f}")
        print(f"   損失歸一化: {'啟用' if self.normalize_losses else '禁用'} (warmup={self.warmup_epochs} epochs)")

    def _compute_kolmogorov_reynolds(self, amplitude: float, nu: float, wavenumber: float) -> float:
        """
        計算 Kolmogorov Flow 的雷諾數（Musacchio & Boffetta 2014 定義）

        理論基礎：
            Re = √f₀ × L^(3/2) / ν

        其中：
            - f₀ = A（強迫振幅）
            - L = 2π/k（強迫波長，特徵長度）
            - ν = 動力黏度
            - k = k_f（強迫波數）

        References:
            - Musacchio & Boffetta (2014), Phys. Rev. E
            - Shebalin (2013)
            - Danilov & Gurarie (2001)
        """
        f0 = amplitude
        k = wavenumber
        L = 2.0 * np.pi / k  # 特徵長度（強迫波長）
        Re = np.sqrt(f0) * (L ** 1.5) / nu
        return float(Re)

    def compute_reynolds_number(self) -> float:
        """
        計算 Kolmogorov Flow 的雷諾數（向後兼容方法）

        Returns:
            Re: 雷諾數（無量綱）
        """
        return float(self.Re)

    def _verify_configuration(self):
        """驗證配置參數的合理性"""
        # 檢查強迫參數
        if self.amplitude <= 0:
            raise ValueError(f"強迫振幅必須 > 0，當前值: {self.amplitude}")
        if self.wavenumber <= 0:
            raise ValueError(f"強迫波數必須 > 0，當前值: {self.wavenumber}")
        
        # 檢查物理參數
        if self.nu <= 0:
            raise ValueError(f"動力黏度必須 > 0，當前值: {self.nu}")
        if self.rho <= 0:
            raise ValueError(f"密度必須 > 0，當前值: {self.rho}")

    def compute_forcing_term(self, coords: torch.Tensor) -> torch.Tensor:
        """
        計算正弦強迫項：f_x = A sin(k_f y)

        Args:
            coords: [batch, 2] = [x, y] 物理坐標

        Returns:
            forcing: [batch, 1] = x 方向強迫項
        """
        y = coords[:, 1:2]
        forcing = self.amplitude * torch.sin(self.wavenumber * y)
        return forcing

    def compute_gradients(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        order: int = 1
    ) -> Dict[str, torch.Tensor]:
        """
        計算物理場對物理座標的梯度（向後兼容方法）

        Args:
            field: 標量場 [batch, 1]（如 u, v, p）
            coords: 物理坐標 [batch, 2] = [x, y]
            order: 微分階數 (1 或 2)

        Returns:
            梯度字典：
                order=1 → {'x': ∂f/∂x, 'y': ∂f/∂y}
                order=2 → {'xx': ∂²f/∂x², 'yy': ∂²f/∂y²}
        """
        if order == 1:
            grad_x = self.compute_gradient(field, coords, component=0)
            grad_y = self.compute_gradient(field, coords, component=1)
            return {'x': grad_x, 'y': grad_y}

        elif order == 2:
            from .base.gradient_ops import compute_second_derivative
            grad_xx = compute_second_derivative(field, coords, component=0)
            grad_yy = compute_second_derivative(field, coords, component=1)
            return {'xx': grad_xx, 'yy': grad_yy}

        else:
            raise ValueError(f"不支持的微分階數: {order}")

    def compute_laplacian(
        self,
        field: torch.Tensor,
        coords: torch.Tensor,
        stabilize: bool = False
    ) -> torch.Tensor:
        """
        計算 Laplacian（向後兼容方法，委派給基類）

        Args:
            field: 標量場 [batch, 1]
            coords: 物理坐標 [batch, 2]
            stabilize: 是否使用穩定化技術（預設 False）

        Returns:
            laplacian: [batch, 1]
        """
        # 委派給基類的 Laplacian 計算（spatial_dim 從 coords.shape[1] 推斷）
        return super().compute_laplacian(field, coords, stabilize=stabilize)

    def residual(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        計算 Kolmogorov Flow 完整殘差（統一接口）

        Args:
            coords: [batch, 2] = [x, y] 物理坐標
            predictions: [batch, 3] = [u, v, p] 預測值
            time: [batch, 1] 時間坐標（可選）

        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'continuity'}
        """
        # 提取速度與壓力
        u, v, p = self.parse_velocity_pressure(predictions)

        # === 計算連續方程殘差 ===
        continuity = self.compute_continuity_residual(coords, [u, v])

        # === 計算對流項 ===
        conv_u = self.compute_advection_term(coords, u, [u, v])
        conv_v = self.compute_advection_term(coords, v, [u, v])

        # === 計算壓力梯度項 ===
        p_x = self.compute_gradient(p, coords, component=0)
        p_y = self.compute_gradient(p, coords, component=1)
        pressure_x = p_x / self.rho
        pressure_y = p_y / self.rho

        # === 計算黏性項 ===
        viscous_u = self.compute_viscous_term(coords, u)
        viscous_v = self.compute_viscous_term(coords, v)

        # === 計算強迫項（Kolmogorov 特有） ===
        forcing_u = self.compute_forcing_term(coords)

        # === 時間導數（非穩態情況） ===
        time_deriv_u = torch.zeros_like(u)
        time_deriv_v = torch.zeros_like(v)

        if time is not None and time.requires_grad:
            time_deriv_u = self.compute_gradient(u, time, component=0)
            time_deriv_v = self.compute_gradient(v, time, component=0)

        # === 組裝動量方程殘差 ===
        # x 方向動量方程（含正弦強迫項）
        residual_x = time_deriv_u + conv_u + pressure_x - viscous_u - forcing_u

        # y 方向動量方程
        residual_y = time_deriv_v + conv_v + pressure_y - viscous_v

        return {
            'momentum_x': residual_x,
            'momentum_y': residual_y,
            'continuity': continuity
        }

    def residual_unified(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        time: Optional[torch.Tensor] = None,
        epoch: int = 0,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        統一的殘差計算接口（向後兼容）

        Args:
            coords: [batch, 2] = [x, y] 物理坐標
            predictions: [batch, 3] = [u, v, p] 預測值
            time: [batch, 1] 時間坐標（可選）
            epoch: 當前訓練 epoch（用於損失歸一化）

        Returns:
            殘差字典 {'momentum_x', 'momentum_y', 'continuity'}
        """
        # 調用新接口
        residuals = self.residual(coords, predictions, time, **kwargs)

        # 損失歸一化（向後兼容功能）
        if self.normalize_losses and epoch >= self.warmup_epochs:
            residuals = self._normalize_residuals(residuals, epoch)

        return residuals

    def _normalize_residuals(
        self,
        residuals: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        損失歸一化（向後兼容功能）

        使用移動平均更新歸一化因子
        """
        normalized = {}
        for key, residual in residuals.items():
            # 計算當前 batch 的 RMS
            rms = torch.sqrt(torch.mean(residual ** 2)).item()

            # 更新移動平均
            if key not in self.loss_normalizers:
                self.loss_normalizers[key] = rms
            else:
                self.loss_normalizers[key] = (
                    self.normalizer_momentum * self.loss_normalizers[key] +
                    (1 - self.normalizer_momentum) * rms
                )

            # 歸一化
            normalizer = max(self.loss_normalizers[key], 1e-8)
            normalized[key] = residual / normalizer

        return normalized

    def compute_vorticity(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 2D 渦量：ω = ∂v/∂x - ∂u/∂y

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]

        Returns:
            vorticity: [batch, 1]
        """
        u, v, _ = self.parse_velocity_pressure(predictions)

        u_y = self.compute_gradient(u, coords, component=1)
        v_x = self.compute_gradient(v, coords, component=0)

        vorticity = v_x - u_y
        return vorticity

    def compute_enstrophy(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 Enstrophy：ω² 的空間平均（標量）

        物理意義：渦度的平方積分，表徵湍流的旋轉強度

        Args:
            coords: [batch, 2]
            predictions: [batch, 3]

        Returns:
            enstrophy: 標量 tensor
        """
        vorticity = self.compute_vorticity(coords, predictions)
        enstrophy = torch.mean(vorticity ** 2)  # 返回標量
        return enstrophy

    def get_physical_properties(self) -> Dict[str, float]:
        """
        獲取物理屬性

        Returns:
            物理屬性字典
        """
        return {
            'amplitude': float(self.amplitude.item()),
            'wavenumber': float(self.wavenumber.item()),
            'nu': float(self.nu),
            'rho': float(self.rho),
            'reynolds_number': self.compute_reynolds_number(),
            'domain_x': self.domain_bounds['x'],
            'domain_y': self.domain_bounds['y'],
        }

    # ========================================================================
    # Backward Compatibility Methods (for tests)
    # ========================================================================

    def compute_momentum_residuals(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        向後兼容：計算動量方程殘差

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]

        Returns:
            殘差字典 {'momentum_x', 'momentum_y'}
        """
        residuals = self.residual(coords, predictions)
        return {
            'momentum_x': residuals['momentum_x'],
            'momentum_y': residuals['momentum_y']
        }

    def compute_continuity_residual(
        self,
        coords: torch.Tensor,
        velocity_fields_or_predictions: Union[list, torch.Tensor]
    ) -> torch.Tensor:
        """
        向後兼容：計算連續方程殘差（支持兩種調用方式）

        Args:
            coords: [batch, 2] = [x, y]
            velocity_fields_or_predictions: 
                - list [u, v]: 速度分量列表（基類接口）
                - tensor [batch, 3]: [u, v, p]（測試接口）

        Returns:
            continuity: [batch, 1]
        """
        # 判斷輸入類型
        if isinstance(velocity_fields_or_predictions, list):
            # 基類接口：直接調用
            return super().compute_continuity_residual(coords, velocity_fields_or_predictions)
        else:
            # 測試接口：先解析再調用
            u, v, _ = self.parse_velocity_pressure(velocity_fields_or_predictions)
            return super().compute_continuity_residual(coords, [u, v])

    def compute_periodic_loss(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        boundary_band_width: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """
        向後兼容：計算週期邊界損失

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]
            boundary_band_width: 邊界帶寬（域大小的百分比）

        Returns:
            {'periodic_x': loss_x, 'periodic_y': loss_y}
        """
        x = coords[:, 0:1]
        y = coords[:, 1:2]

        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']
        
        Lx = x_max - x_min
        Ly = y_max - y_min
        
        # 找到接近邊界的點
        tol_x = boundary_band_width * Lx
        tol_y = boundary_band_width * Ly
        
        # X 方向邊界
        mask_x_left = (x >= x_min) & (x <= x_min + tol_x)
        mask_x_right = (x <= x_max) & (x >= x_max - tol_x)
        
        # Y 方向邊界
        mask_y_bottom = (y >= y_min) & (y <= y_min + tol_y)
        mask_y_top = (y <= y_max) & (y >= y_max - tol_y)
        
        # 計算週期性損失
        loss_x = torch.tensor(0.0, device=coords.device)
        loss_y = torch.tensor(0.0, device=coords.device)
        
        if mask_x_left.any() and mask_x_right.any():
            pred_left = predictions[mask_x_left.squeeze()]
            pred_right = predictions[mask_x_right.squeeze()]
            
            # 如果點數不匹配，取最小數量
            n_pts = min(pred_left.shape[0], pred_right.shape[0])
            if n_pts > 0:
                loss_x = torch.mean((pred_left[:n_pts] - pred_right[:n_pts]) ** 2)
        
        if mask_y_bottom.any() and mask_y_top.any():
            pred_bottom = predictions[mask_y_bottom.squeeze()]
            pred_top = predictions[mask_y_top.squeeze()]
            
            # 如果點數不匹配，取最小數量
            n_pts = min(pred_bottom.shape[0], pred_top.shape[0])
            if n_pts > 0:
                loss_y = torch.mean((pred_bottom[:n_pts] - pred_top[:n_pts]) ** 2)
        
        return {'periodic_x': loss_x, 'periodic_y': loss_y}

    def compute_kinetic_energy(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        向後兼容：計算動能

        Args:
            predictions: [batch, 3] = [u, v, p]

        Returns:
            kinetic_energy: 標量 tensor
        """
        u, v, _ = self.parse_velocity_pressure(predictions)
        ke = 0.5 * (u**2 + v**2).mean()
        return ke

    def normalize_loss_dict(
        self,
        loss_dict: Dict[str, torch.Tensor],
        epoch: int
    ) -> Dict[str, torch.Tensor]:
        """
        向後兼容：損失字典歸一化

        Args:
            loss_dict: 損失字典
            epoch: 當前 epoch

        Returns:
            歸一化後的損失字典
        """
        if epoch < self.warmup_epochs:
            # Warmup 階段：收集統計但不歸一化
            for key, loss in loss_dict.items():
                rms = torch.sqrt(torch.mean(loss ** 2)).item()
                if key not in self.loss_normalizers:
                    self.loss_normalizers[key] = rms
                else:
                    self.loss_normalizers[key] = (
                        self.normalizer_momentum * self.loss_normalizers[key] +
                        (1 - self.normalizer_momentum) * rms
                    )
            return loss_dict  # 返回原值
        else:
            # 訓練階段：歸一化
            normalized = {}
            for key, loss in loss_dict.items():
                normalizer = self.loss_normalizers.get(key, 1.0)
                normalizer = max(normalizer, 1e-8)  # 避免除零
                normalized[key] = loss / normalizer
            return normalized

    def get_physics_info(self) -> Dict:
        """
        返回物理元數據（覆寫基類方法，向後兼容測試格式）

        Returns:
            物理信息字典
        """
        base_info = super().get_physics_info()
        
        # 向後兼容：添加測試期望的鍵
        base_info.update({
            'equation': 'Kolmogorov Flow (2D NS with forcing)',
            'forcing_amplitude': float(self.amplitude.item()),
            'forcing_wavenumber': float(self.wavenumber.item()),
            'boundary_conditions': 'double_periodic',  # 測試期望這個值
            'reynolds_definition': 'Kolmogorov: Re = sqrt(A) * (2π/k)^1.5 / ν',
            
            # 測試期望的格式
            'forcing_parameters': {
                'amplitude': float(self.amplitude.item()),
                'wavenumber': float(self.wavenumber.item())
            },
            'physics_parameters': {
                'nu': float(self.nu.item()),
                'rho': float(self.rho.item()),
                'Reynolds_number': self.compute_reynolds_number()
            },
            'domain_bounds': self.domain_bounds,  # 測試期望這個鍵
            'loss_normalization': {
                'enabled': self.normalize_losses,
                'warmup_epochs': self.warmup_epochs,
                'momentum': self.normalizer_momentum
            }
        })
        return base_info


# ==============================================================================
# Factory Function (Backward Compatibility)
# ==============================================================================

def create_kolmogorov_flow_2d(
    forcing_amplitude: float = None,
    forcing_wavenumber: int = None,
    nu: float = 0.01,
    rho: float = 1.0,
    domain_x: Tuple[float, float] = (0.0, 2.0 * np.pi),
    domain_y: Tuple[float, float] = (0.0, 2.0 * np.pi),
    loss_config: Optional[Dict[str, Any]] = None,
    # Backward compatibility aliases
    A: float = None,
    k_f: int = None,
    **kwargs
) -> KolmogorovFlow2D:
    """
    工廠函數：創建 Kolmogorov Flow 2D 求解器（向後兼容）

    Args:
        forcing_amplitude: 強迫振幅 A（或使用舊參數名 A）
        forcing_wavenumber: 強迫波數 k_f（或使用舊參數名 k_f）
        nu: 動力黏度
        rho: 密度
        domain_x: x 域邊界 (x_min, x_max)
        domain_y: y 域邊界 (y_min, y_max)
        loss_config: 損失配置
        A: (DEPRECATED) 舊參數名，等同於 forcing_amplitude
        k_f: (DEPRECATED) 舊參數名，等同於 forcing_wavenumber

    Returns:
        KolmogorovFlow2D 實例
    """
    # Backward compatibility: 支援舊參數名
    if A is not None:
        forcing_amplitude = A
    if k_f is not None:
        forcing_wavenumber = k_f
    
    # 設定預設值
    if forcing_amplitude is None:
        forcing_amplitude = 1.0
    if forcing_wavenumber is None:
        forcing_wavenumber = 4
    
    forcing_params = {
        'amplitude': forcing_amplitude,
        'wavenumber': forcing_wavenumber
    }
    physics_params = {
        'nu': nu,
        'rho': rho
    }
    domain_bounds = {
        'x': domain_x,
        'y': domain_y
    }

    return KolmogorovFlow2D(
        forcing_params=forcing_params,
        physics_params=physics_params,
        domain_bounds=domain_bounds,
        loss_config=loss_config
    )
