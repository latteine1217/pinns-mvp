"""
Kolmogorov Flow 2D 物理模組
===========================

實現 2D Kolmogorov flow 的物理定律計算功能：
1. 含正弦強迫項的 NS 方程殘差計算
2. 嚴格週期性邊界條件（x, y 方向）
3. 渦度與 enstrophy 計算
4. 自動微分梯度計算
5. 守恆定律檢查
6. 標準雷諾數定義：Re = F / (ν² k³)

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

作者：PINNs-MVP 團隊
日期：2025-11-21 (更新)
"""

import torch
import torch.nn as nn
import torch.autograd as autograd
from typing import Tuple, Dict, Optional, Any
import numpy as np
import logging


# ==============================================================================
# 梯度計算函數（重用自 ns_2d.py）
# ==============================================================================

def compute_gradient_2d(
    field: torch.Tensor,
    coords: torch.Tensor,
    component: int
) -> torch.Tensor:
    """
    計算 2D 場的偏導數（單一分量）

    Args:
        field: 標量場 [batch, 1]（需要在計算圖中）
        coords: 2D 坐標 [batch, 2]（需要 requires_grad=True）
        component: 微分分量 (0=x, 1=y)

    Returns:
        偏導數 [batch, 1]（保留計算圖）
    """
    grad_outputs = torch.ones_like(field)
    grads = autograd.grad(
        outputs=field,
        inputs=coords,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True  # ⚠️ 修改為 True 以處理未使用的輸入
    )[0]

    # 處理 None 的情況（當輸入未在計算圖中使用時）
    if grads is None:
        raise RuntimeError(
            f"計算梯度失敗：field 與 coords 之間沒有計算圖連接。"
            f"field.shape={field.shape}, coords.shape={coords.shape}, "
            f"field.requires_grad={field.requires_grad}, coords.requires_grad={coords.requires_grad}"
        )

    return grads[:, component:component+1]


def compute_laplacian_2d(
    field: torch.Tensor,
    coords: torch.Tensor
) -> torch.Tensor:
    """
    計算 2D Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²

    Args:
        field: 標量場 [batch, 1]
        coords: 2D 坐標 [batch, 2]

    Returns:
        laplacian: [batch, 1]
    """
    # 計算一階導數
    grad_x = compute_gradient_2d(field, coords, component=0)
    grad_y = compute_gradient_2d(field, coords, component=1)

    # 計算二階導數
    grad_xx = compute_gradient_2d(grad_x, coords, component=0)
    grad_yy = compute_gradient_2d(grad_y, coords, component=1)

    return grad_xx + grad_yy


# ==============================================================================
# Kolmogorov Flow 主類
# ==============================================================================

class KolmogorovFlow2D(nn.Module):
    """
    2D Kolmogorov Flow 求解器

    實現含正弦強迫項的 2D 不可壓縮 NS 方程，適用於湍流研究與逆問題重建。

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
        super().__init__()

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
        self.physics_params = {**default_physics, **(physics_params or {})}
        self.domain_bounds = {**default_bounds, **(domain_bounds or {})}

        # 註冊強迫參數為緩衝區（不參與梯度計算）
        self.register_buffer('amplitude', torch.tensor(float(self.forcing_params['amplitude'])))
        self.register_buffer('wavenumber', torch.tensor(float(self.forcing_params['wavenumber'])))

        # 註冊物理參數
        for key, value in self.physics_params.items():
            self.register_buffer(key, torch.tensor(float(value)))

        # === 損失歸一化參數 ===
        self.loss_normalizers: Dict[str, float] = {}
        self.normalize_losses = True
        self.warmup_epochs = (loss_config or {}).get('warmup_epochs', 5)
        self.normalizer_momentum = 0.9

        # 驗證配置
        self._verify_configuration()

        print(f"✅ Kolmogorov Flow 2D 初始化完成")
        print(f"   強迫參數: A={self.amplitude:.2f}, k_f={self.wavenumber:.0f}")
        print(f"   物理參數: ν={self.nu:.2e}, ρ={self.rho:.1f}")
        print(f"   域範圍: x∈{self.domain_bounds['x']}, y∈{self.domain_bounds['y']}")
        print(f"   損失歸一化: {'啟用' if self.normalize_losses else '禁用'} (warmup={self.warmup_epochs} epochs)")

    def compute_reynolds_number(self) -> float:
        """
        計算 Kolmogorov Flow 的雷諾數（Musacchio & Boffetta 2014 定義）

        理論基礎：
            Re = √f₀ × L^(3/2) / ν

        其中：
            - f₀ = A（強迫振幅）
            - L = 2π/k（強迫波長，特徵長度）
            - ν = 動力黏度
            - k = k_f（強迫波數）

        物理含義：
            Kolmogorov flow 的穩態層流解為 u_x = U₀ sin(k y)
            其中 U₀ ~ f₀/(ν k²)
            代入一般 Re = UL/ν 定義，經過量綱分析得到上述表達式。
            此定義在 2D turbulence 和 Kolmogorov flow 研究中最常用。

        Returns:
            Re: 雷諾數（無量綱）

        References:
            - Musacchio & Boffetta (2014), Phys. Rev. E
            - Shebalin (2013)
            - Danilov & Gurarie (2001)
        """
        f0 = float(self.amplitude.item())
        nu = float(self.nu.item())
        k = float(self.wavenumber.item())

        # 特徵長度：強迫波長 L = 2π/k
        L = 2.0 * np.pi / k

        # 計算雷諾數
        Re = np.sqrt(f0) * (L ** 1.5) / nu
        return Re

    def compute_effective_reynolds(self, predictions: torch.Tensor) -> float:
        """
        計算有效雷諾數（基於預測場的動能）

        定義：
            Re_eff = √(2E) × L / ν
        
        其中：
            - E = ∫(u² + v²) dA / (2A)（動能均方根）
            - L = 1/k_f（特徵長度）
            - ν = 動力黏度

        用途：
            後處理統計分析、與理論 Re 的對比

        Args:
            predictions: [batch, 3] = [u, v, p] 預測場

        Returns:
            Re_eff: 有效雷諾數（無量綱）
        """
        # 計算動能
        KE = float(self.compute_kinetic_energy(predictions).item())
        
        # 特徵速度（動能均方根）
        U_eff = np.sqrt(2.0 * KE)
        
        # 特徵長度
        L = 1.0 / float(self.wavenumber.item())
        
        # 有效雷諾數
        nu = float(self.nu.item())
        Re_eff = U_eff * L / nu
        
        return Re_eff

    def _verify_configuration(self):
        """驗證配置的物理合理性"""
        # 1. 強迫振幅應為正值
        amp_val = float(self.amplitude.item())
        if amp_val <= 0:
            raise ValueError(f"強迫振幅 A={amp_val} 必須為正值")

        # 2. 波數應為正整數
        wave_val = float(self.wavenumber.item())
        if wave_val <= 0:
            raise ValueError(f"強迫波數 k_f={wave_val} 必須為正整數")

        # 3. 黏度應為正值
        nu_val = float(self.nu.item())
        if nu_val <= 0:
            raise ValueError(f"動力黏度 ν={nu_val} 必須為正值")

        # 4. 雷諾數驗證（使用 Musacchio & Boffetta 2014 定義）
        Re = self.compute_reynolds_number()

        if Re < 30:
            print(f"⚠️  警告：雷諾數 Re={Re:.2f} 過低，流場處於層流或弱湍流狀態")
        elif Re > 200:
            print(f"⚠️  警告：雷諾數 Re={Re:.2f} 過高，強湍流可能需更高解析度")
        else:
            print(f"✅ 雷諾數 Re={Re:.2f}（適合 Kolmogorov flow 2D 湍流研究）")

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
        計算物理場對物理座標的梯度

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
            grad_x = compute_gradient_2d(field, coords, component=0)
            grad_y = compute_gradient_2d(field, coords, component=1)
            return {'x': grad_x, 'y': grad_y}

        elif order == 2:
            grad_x = compute_gradient_2d(field, coords, component=0)
            grad_y = compute_gradient_2d(field, coords, component=1)

            grad_xx = compute_gradient_2d(grad_x, coords, component=0)
            grad_yy = compute_gradient_2d(grad_y, coords, component=1)

            return {'xx': grad_xx, 'yy': grad_yy}

        else:
            raise ValueError(f"不支持的微分階數: {order}")

    def compute_laplacian(
        self,
        field: torch.Tensor,
        coords: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 Laplacian: ∇²f = ∂²f/∂x² + ∂²f/∂y²

        Args:
            field: 標量場 [batch, 1]
            coords: 物理坐標 [batch, 2]

        Returns:
            laplacian: [batch, 1]
        """
        return compute_laplacian_2d(field, coords)

    def compute_momentum_residuals(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        計算 2D Kolmogorov flow 的動量殘差

        方程（含正弦強迫項）:
            ∂u/∂t + u∂u/∂x + v∂u/∂y = -∂p/∂x + ν∇²u + A sin(k_f y)  (x-動量)
            ∂v/∂t + u∂v/∂x + v∂v/∂y = -∂p/∂y + ν∇²v                  (y-動量)

        Args:
            coords: [batch, 2] = [x, y] 物理坐標
            predictions: [batch, 3] = [u, v, p] 預測值
            time: [batch, 1] 時間坐標（可選，穩態時為 None）

        Returns:
            殘差字典 {'momentum_x', 'momentum_y'}
        """
        # 提取速度與壓力（保留計算圖，不重新設置 requires_grad）
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]
        p = predictions[:, 2:3]

        # === 計算一階導數（對流項 + 壓力項） ===
        u_grads = self.compute_gradients(u, coords, order=1)
        v_grads = self.compute_gradients(v, coords, order=1)
        p_grads = self.compute_gradients(p, coords, order=1)

        # 對流項
        conv_u = u * u_grads['x'] + v * u_grads['y']
        conv_v = u * v_grads['x'] + v * v_grads['y']

        # 壓力梯度項
        pressure_x = p_grads['x'] / self.rho
        pressure_y = p_grads['y'] / self.rho

        # === 計算二階導數（黏性項） ===
        laplacian_u = self.compute_laplacian(u, coords)
        laplacian_v = self.compute_laplacian(v, coords)

        viscous_u = self.nu * laplacian_u
        viscous_v = self.nu * laplacian_v

        # === 計算強迫項 ===
        forcing_u = self.compute_forcing_term(coords)

        # === 時間導數（非穩態情況） ===
        time_deriv_u = torch.zeros_like(u)
        time_deriv_v = torch.zeros_like(v)

        if time is not None:
            if time.requires_grad:
                time_deriv_u = compute_gradient_2d(u, time, component=0)
                time_deriv_v = compute_gradient_2d(v, time, component=0)

        # === 組裝殘差 ===
        # x 方向動量方程（含正弦強迫項）
        residual_x = time_deriv_u + conv_u + pressure_x - viscous_u - forcing_u

        # y 方向動量方程
        residual_y = time_deriv_v + conv_v + pressure_y - viscous_v

        return {
            'momentum_x': residual_x,
            'momentum_y': residual_y,
        }

    def compute_continuity_residual(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算連續方程殘差（不可壓縮條件）

        方程:
            ∂u/∂x + ∂v/∂y = 0

        Args:
            coords: [batch, 2] = [x, y] 物理座標
            predictions: [batch, 3] = [u, v, p] 預測值

        Returns:
            continuity_residual: [batch, 1]
        """
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        # 計算散度
        u_grads = self.compute_gradients(u, coords, order=1)
        v_grads = self.compute_gradients(v, coords, order=1)

        divergence = u_grads['x'] + v_grads['y']

        return divergence

    def residual_unified(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        time: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        統一殘差計算介面（與 trainer.py 兼容）

        調用 compute_momentum_residuals() 和 compute_continuity_residual()
        並組合成統一的字典格式。

        Args:
            coords: [batch, 2] = [x, y] 物理座標（需要 requires_grad=True）
            predictions: [batch, 3] or [batch, 4] = [u, v, p, S?] 預測值
                         （第 4 個分量 S 如果存在會被忽略）
            time: [batch, 1] 時間坐標（可選）

        Returns:
            殘差字典: {
                'momentum_x': [batch, 1],
                'momentum_y': [batch, 1],
                'continuity': [batch, 1],
                'momentum_z': [batch, 1] = 0  # 2D 無 z 分量，為兼容性返回 0
            }
        """
        # ⚠️ 關鍵修復：不要重新創建座標張量！
        # trainer 已經確保傳入的 coords 與 predictions 來自同一計算路徑
        # 任何 clone/detach 操作都會斷開計算圖連接

        # 只檢查梯度追蹤狀態，不修改張量
        if not coords.requires_grad:
            raise RuntimeError(
                f"傳入的 coords 必須啟用梯度追蹤！當前 requires_grad={coords.requires_grad}"
            )

        # ⚠️ 關鍵修復：在物理模組內部進行切片，保持與原始張量的計算圖連接
        # 只取前 2 個維度用於 2D 計算（如果 coords 是 3D，例如 [x, y, z]）
        coords_2d = coords[:, :2] if coords.shape[1] > 2 else coords

        # 只取前 3 個分量 [u, v, p]（忽略可能的第 4 個分量 S）
        predictions_3d = predictions[:, :3] if predictions.shape[1] >= 3 else predictions

        # 計算動量殘差（使用切片後的 2D 座標）
        momentum_residuals = self.compute_momentum_residuals(coords_2d, predictions_3d, time)

        # 計算連續性殘差（⚠️ 修復：使用 2D 座標，與動量殘差一致）
        continuity_residual = self.compute_continuity_residual(coords_2d, predictions_3d)

        # 組合成統一格式（添加 momentum_z = 0 以兼容 3D 訓練器）
        return {
            'momentum_x': momentum_residuals['momentum_x'],
            'momentum_y': momentum_residuals['momentum_y'],
            'momentum_z': torch.zeros_like(continuity_residual),  # 2D 無 z 分量
            'continuity': continuity_residual,
        }

    def compute_periodic_loss(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor,
        boundary_band_width: float = 5e-3
    ) -> Dict[str, torch.Tensor]:
        """
        計算嚴格週期性邊界約束損失

        對於 x 和 y 方向的週期邊界:
            u(x_min, y) = u(x_max, y), v(x_min, y) = v(x_max, y), p(x_min, y) = p(x_max, y)
            u(x, y_min) = u(x, y_max), v(x, y_min) = v(x, y_max), p(x, y_min) = p(x, y_max)

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]
            boundary_band_width: 邊界帶狀寬度（默認 5e-3）

        Returns:
            週期性損失字典 {'periodic_x', 'periodic_y'}
        """
        x_min, x_max = self.domain_bounds['x']
        y_min, y_max = self.domain_bounds['y']

        # x 方向週期性
        mask_x_min = torch.abs(coords[:, 0] - x_min) < boundary_band_width
        mask_x_max = torch.abs(coords[:, 0] - x_max) < boundary_band_width

        # y 方向週期性
        mask_y_min = torch.abs(coords[:, 1] - y_min) < boundary_band_width
        mask_y_max = torch.abs(coords[:, 1] - y_max) < boundary_band_width

        # 計算週期性誤差
        periodic_x_loss = torch.tensor(0.0, device=coords.device)
        periodic_y_loss = torch.tensor(0.0, device=coords.device)

        if mask_x_min.any() and mask_x_max.any():
            fields_x_min = predictions[mask_x_min]
            fields_x_max = predictions[mask_x_max]

            n_min = min(fields_x_min.shape[0], fields_x_max.shape[0])
            periodic_x_loss = torch.mean((fields_x_min[:n_min] - fields_x_max[:n_min]) ** 2)

        if mask_y_min.any() and mask_y_max.any():
            fields_y_min = predictions[mask_y_min]
            fields_y_max = predictions[mask_y_max]

            n_min = min(fields_y_min.shape[0], fields_y_max.shape[0])
            periodic_y_loss = torch.mean((fields_y_min[:n_min] - fields_y_max[:n_min]) ** 2)

        return {
            'periodic_x': periodic_x_loss,
            'periodic_y': periodic_y_loss,
        }

    def compute_vorticity(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 2D 渦度：ω = ∂v/∂x - ∂u/∂y

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]

        Returns:
            vorticity: [batch, 1]
        """
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        u_grads = self.compute_gradients(u, coords, order=1)
        v_grads = self.compute_gradients(v, coords, order=1)

        vorticity = v_grads['x'] - u_grads['y']

        return vorticity

    def compute_enstrophy(
        self,
        coords: torch.Tensor,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算 enstrophy：E = ∫ω² dA / A

        物理意義：渦度的平方積分，表徵湍流的旋轉強度。

        Args:
            coords: [batch, 2] = [x, y]
            predictions: [batch, 3] = [u, v, p]

        Returns:
            enstrophy: 標量（批次平均）
        """
        vorticity = self.compute_vorticity(coords, predictions)
        enstrophy = torch.mean(vorticity ** 2)

        return enstrophy

    def compute_kinetic_energy(
        self,
        predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        計算動能：KE = ∫(u² + v²) dA / (2A)

        Args:
            predictions: [batch, 3] = [u, v, p]

        Returns:
            kinetic_energy: 標量（批次平均）
        """
        u = predictions[:, 0:1]
        v = predictions[:, 1:2]

        kinetic_energy = torch.mean((u ** 2 + v ** 2) / 2.0)

        return kinetic_energy

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
        """
        if not self.normalize_losses:
            return loss_dict

        # === Warmup 階段：收集統計 ===
        if epoch < self.warmup_epochs:
            for key, loss in loss_dict.items():
                loss_val = loss.detach().item()

                if key not in self.loss_normalizers:
                    self.loss_normalizers[key] = loss_val
                else:
                    self.loss_normalizers[key] = (
                        self.normalizer_momentum * self.loss_normalizers[key] +
                        (1 - self.normalizer_momentum) * loss_val
                    )

            return loss_dict

        # === Training 階段：歸一化 ===
        normalized = {}
        for key, loss in loss_dict.items():
            normalizer = self.loss_normalizers.get(key, 1.0)
            if normalizer < 1e-12:
                normalizer = 1.0
            normalized[key] = loss / normalizer

        return normalized

    def get_physics_info(self) -> Dict[str, Any]:
        """獲取物理信息摘要"""
        # 計算雷諾數（標準定義）
        Re = self.compute_reynolds_number()
        
        # 計算層流解的特徵速度
        F = float(self.amplitude.item())
        nu = float(self.nu.item())
        k = float(self.wavenumber.item())
        U_laminar = F / (nu * k**2)

        return {
            'forcing_parameters': {
                'amplitude': F,
                'wavenumber': k,
                'forcing_type': 'sinusoidal',
            },
            'physics_parameters': {
                'nu': nu,
                'rho': float(self.rho.item()),
                'Reynolds_number': Re,
                'laminar_velocity': U_laminar,
                'characteristic_length': 1.0 / k,
            },
            'domain_bounds': self.domain_bounds,
            'boundary_conditions': 'double_periodic',
            'loss_normalization': {
                'enabled': self.normalize_losses,
                'warmup_epochs': self.warmup_epochs,
                'normalizers': self.loss_normalizers.copy(),
            },
        }


# ==============================================================================
# 便捷函數
# ==============================================================================

def create_kolmogorov_flow_2d(
    A: float = 1.0,
    k_f: int = 4,
    nu: float = 0.01,
    rho: float = 1.0,
    **kwargs
) -> KolmogorovFlow2D:
    """
    建立 Kolmogorov Flow 2D 求解器的便捷函數

    Args:
        A: 強迫振幅（默認 1.0）
        k_f: 強迫波數（默認 4）
        nu: 動力黏度（默認 0.01）
        rho: 密度（默認 1.0）
        **kwargs: 其他參數（如 domain_bounds, loss_config）

    Returns:
        KolmogorovFlow2D 實例
    """
    forcing_params = {'amplitude': A, 'wavenumber': k_f}
    physics_params = {'nu': nu, 'rho': rho}

    return KolmogorovFlow2D(
        forcing_params=forcing_params,
        physics_params=physics_params,
        **kwargs
    )
