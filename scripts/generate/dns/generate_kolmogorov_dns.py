"""
Kolmogorov Flow DNS 資料生成器 (統一版本)
==========================================

自動檢測並使用最佳計算後端：
- GPU 可用：PyTorch (MPS/CUDA)
- 僅 CPU：NumPy

使用 pseudo-spectral 方法求解 2D Kolmogorov flow。

作者：PINNs-MVP 團隊
日期：2025-11-22
版本：2.0-Unified
"""

import numpy as np
import argparse
from pathlib import Path
import sys
from typing import Dict, Tuple, Optional
import logging
import time

# 檢測 PyTorch 是否可用
try:
    import torch
    TORCH_AVAILABLE = True
    torch.set_float32_matmul_precision("high")
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def select_backend() -> Tuple[str, Optional[object]]:
    """
    自動選擇最佳計算後端
    
    Returns:
        backend: 'torch-mps', 'torch-cuda', 'torch-cpu', 'numpy'
        device: torch.device 或 None
    """
    if not TORCH_AVAILABLE:
        logging.info("🔧 PyTorch 不可用，使用 NumPy (CPU)")
        return 'numpy', None
    
    # 優先 MPS (Apple Metal)
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        logging.info(f"🚀 使用 PyTorch + MPS (Apple Metal GPU)")
        return 'torch-mps', device
    
    # 次選 CUDA (NVIDIA)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"🚀 使用 PyTorch + CUDA (NVIDIA GPU)")
        return 'torch-cuda', device
    
    # CPU PyTorch
    if TORCH_AVAILABLE:
        device = torch.device('cpu')
        logging.info(f"🔧 使用 PyTorch (CPU)")
        return 'torch-cpu', device
    
    # 回退到 NumPy
    logging.info("🔧 使用 NumPy (CPU)")
    return 'numpy', None


class KolmogorovFlowDNS:
    """
    2D Kolmogorov Flow DNS 求解器（自動選擇 GPU/CPU）
    """

    def __init__(
        self,
        N: int = 1024,
        L: float = 2 * np.pi,
        nu: float = 1e-5,
        A: float = 0.1,
        k_f: int = 4,
        dt: float = 0.001,
        dealias: bool = True,
        dealias_mode: str = "3/2",
        backend: Optional[str] = None,
        project_after_step: bool = True,
    ):
        """
        Args:
            N: 網格點數（每方向）
            L: 域長度
            nu: 黏滯係數
            A: 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            dealias: 是否啟用去混疊
            dealias_mode: 去混疊方式 ('2/3' 或 '3/2')
            backend: 手動指定後端 ('torch-mps', 'torch-cuda', 'numpy', None=auto)
        """
        # 選擇後端
        if backend is None:
            self.backend, self.device = select_backend()
        else:
            if backend.startswith('torch-'):
                if not TORCH_AVAILABLE:
                    raise RuntimeError(f"PyTorch 不可用，無法使用 {backend}")
                device_name = backend.split('-')[1]
                self.device = torch.device(device_name)
            else:
                self.device = None
            self.backend = backend
        
        self.use_torch = self.backend.startswith('torch')
        
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.dealias = dealias
        self.dealias_mode = dealias_mode
        self.project_after_step = project_after_step

        # 空間網格
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if self.use_torch:
            self.X = torch.tensor(X, dtype=torch.float32, device=self.device)
            self.Y = torch.tensor(Y, dtype=torch.float32, device=self.device)
        else:
            self.X = X
            self.Y = Y

        # 波數網格
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        kx, ky = np.meshgrid(k, k, indexing='ij')
        k2 = kx**2 + ky**2
        
        if self.use_torch:
            self.kx = torch.tensor(kx, dtype=torch.float32, device=self.device)
            self.ky = torch.tensor(ky, dtype=torch.float32, device=self.device)
            self.k2 = torch.tensor(k2, dtype=torch.float32, device=self.device)
            self.ikx = 1j * self.kx
            self.iky = 1j * self.ky
            self.k2_safe = self.k2.clone()
            self.k2_safe[0, 0] = 1.0
        else:
            self.kx = kx
            self.ky = ky
            self.k2 = k2
            self.ikx = 1j * self.kx
            self.iky = 1j * self.ky
            self.k2_safe = self.k2.copy()
            self.k2_safe[0, 0] = 1.0

        if self.dealias and self.dealias_mode not in {"2/3", "3/2"}:
            raise ValueError(f"Unknown dealias_mode: {self.dealias_mode}")

        if self.dealias and self.dealias_mode == "3/2":
            self.pad_N = int(3 * self.N / 2)
            self.pad_scale = (self.pad_N / self.N) ** 2
            self.pad_scale_inv = (self.N / self.pad_N) ** 2

            k_pad = 2 * np.pi * np.fft.fftfreq(self.pad_N, d=self.L / self.pad_N)
            kx_pad, ky_pad = np.meshgrid(k_pad, k_pad, indexing='ij')

            if self.use_torch:
                self.pad_kx = torch.tensor(kx_pad, dtype=torch.float32, device=self.device)
                self.pad_ky = torch.tensor(ky_pad, dtype=torch.float32, device=self.device)
            else:
                self.pad_kx = kx_pad
                self.pad_ky = ky_pad
        else:
            self.pad_N = None
            self.pad_scale = 1.0
            self.pad_scale_inv = 1.0
            self.pad_kx = None
            self.pad_ky = None

        # 去混疊遮罩
        if dealias and self.dealias_mode == "2/3":
            k_max = k.max() * 2 / 3
            mask = np.sqrt(kx**2 + ky**2) <= k_max
            if self.use_torch:
                self.dealias_mask = torch.tensor(mask, dtype=torch.float32, device=self.device)
            else:
                self.dealias_mask = mask.astype(float)
        else:
            if self.use_torch:
                self.dealias_mask = torch.ones_like(self.k2)
            else:
                self.dealias_mask = np.ones_like(k2)

        # 物理波數（保證對任意 L 正確）
        k_phys = k_f * 2 * np.pi / L

        # 強迫項（實空間，Kolmogorov forcing 僅作用於 x-momentum）
        if self.use_torch:
            self.forcing_x = A * torch.sin(k_phys * self.Y)
            self.forcing_y = torch.zeros_like(self.forcing_x)
        else:
            self.forcing_x = A * np.sin(k_phys * Y)
            self.forcing_y = np.zeros_like(self.forcing_x)

        # 預先計算頻譜空間強迫項 (避免在 compute_rhs 重複計算)
        self.forcing_U_hat = self.fft2(self.forcing_x)
        self.forcing_V_hat = self.fft2(self.forcing_y)

        # 初始化場（純隨機場，避免偏置既有層流型態）
        init_amplitude = 1.0

        if self.use_torch:
            U_init = init_amplitude * torch.randn_like(self.X)
            V_init = init_amplitude * torch.randn_like(self.X)
            self.U_hat = self.fft2(U_init) * self.dealias_mask
            self.V_hat = self.fft2(V_init) * self.dealias_mask
        else:
            U_init = init_amplitude * np.random.randn(*self.X.shape)
            V_init = init_amplitude * np.random.randn(*self.X.shape)
            self.U_hat = np.fft.fft2(U_init) * self.dealias_mask
            self.V_hat = np.fft.fft2(V_init) * self.dealias_mask

        logging.info(f"✅ 初始化純隨機場：amplitude = {init_amplitude:.6f}")

        # ✅ 關鍵修正：初始化後立即投影，確保 ∇·u = 0
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)

    def to_numpy(self, tensor):
        """將 Torch tensor 轉為 NumPy（用於保存）"""
        if self.use_torch:
            return tensor.cpu().numpy()
        return tensor

    def fft2(self, field):
        """2D FFT"""
        if self.use_torch:
            return torch.fft.fft2(field)
        return np.fft.fft2(field)

    def ifft2(self, field_hat):
        """2D IFFT"""
        if self.use_torch:
            return torch.fft.ifft2(field_hat).real
        return np.fft.ifft2(field_hat).real

    def fft2_batch(self, fields):
        """批次 2D FFT"""
        if self.use_torch:
            return torch.fft.fft2(fields, dim=(-2, -1))
        return np.fft.fft2(fields, axes=(-2, -1))

    def ifft2_batch(self, fields_hat):
        """批次 2D IFFT"""
        if self.use_torch:
            return torch.fft.ifft2(fields_hat, dim=(-2, -1)).real
        return np.fft.ifft2(fields_hat, axes=(-2, -1)).real

    def pad_hat(self, field_hat):
        """頻譜零填充到 3/2 尺寸"""
        pad = (self.pad_N - self.N) // 2
        if self.use_torch:
            field_shift = torch.fft.fftshift(field_hat)
            padded = torch.zeros((self.pad_N, self.pad_N), dtype=field_hat.dtype, device=field_hat.device)
            padded[pad:pad + self.N, pad:pad + self.N] = field_shift
            return torch.fft.ifftshift(padded)

        field_shift = np.fft.fftshift(field_hat)
        padded = np.zeros((self.pad_N, self.pad_N), dtype=field_hat.dtype)
        padded[pad:pad + self.N, pad:pad + self.N] = field_shift
        return np.fft.ifftshift(padded)

    def truncate_hat(self, field_hat_pad):
        """頻譜裁切回原始尺寸"""
        pad = (self.pad_N - self.N) // 2
        if self.use_torch:
            field_shift = torch.fft.fftshift(field_hat_pad)
            truncated = field_shift[pad:pad + self.N, pad:pad + self.N]
            return torch.fft.ifftshift(truncated)

        field_shift = np.fft.fftshift(field_hat_pad)
        truncated = field_shift[pad:pad + self.N, pad:pad + self.N]
        return np.fft.ifftshift(truncated)
    
    def _project_hat(self, U_hat, V_hat):
        """
        對頻譜空間的速度場進行投影，強制滿足不可壓縮條件 ∇·u = 0
        此方法不修改 self.U_hat, self.V_hat，而是返回新的投影場。
        """
        # 計算散度 k·û = k_x * U_hat + k_y * V_hat
        if self.use_torch:
            k_dot_u = self.kx * U_hat + self.ky * V_hat
            correction = k_dot_u / self.k2_safe
            U_hat_proj = U_hat - self.kx * correction
            V_hat_proj = V_hat - self.ky * correction
            
            # (0,0) 模態（平均流）單獨處理：只保留 V 方向（與 Kolmogorov 強迫一致）
            U_hat_proj[0, 0] = 0.0 + 0.0j
            V_hat_proj[0, 0] = 0.0 + 0.0j  # 強制 V 平均流為 0 (防止漂移)
            
        else:
            k_dot_u = self.kx * U_hat + self.ky * V_hat
            correction = k_dot_u / self.k2_safe
            U_hat_proj = U_hat - self.kx * correction
            V_hat_proj = V_hat - self.ky * correction
            
            U_hat_proj[0, 0] = 0.0 + 0.0j
            V_hat_proj[0, 0] = 0.0 + 0.0j  # 強制 V 平均流為 0
        
        return U_hat_proj, V_hat_proj

    def project_incompressible(self):
        """
        對速度場進行譜空間投影，強制滿足不可壓縮條件 ∇·u = 0
        此方法修改 self.U_hat, self.V_hat。
        """
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)

    def compute_omega_hat(self, U_hat=None, V_hat=None):
        """計算渦度頻譜 ω_hat"""
        if U_hat is None or V_hat is None:
            U_hat = self.U_hat
            V_hat = self.V_hat

        return self.ikx * V_hat - self.iky * U_hat

    def compute_omega_div(self, U_hat=None, V_hat=None, return_numpy: bool = True):
        """計算渦度與散度 (ω, ∇·u)"""
        if U_hat is None or V_hat is None:
            U_hat = self.U_hat
            V_hat = self.V_hat

        omega_hat = self.compute_omega_hat(U_hat, V_hat)
        div_hat = self.ikx * U_hat + self.iky * V_hat

        omega = self.ifft2(omega_hat)
        div_field = self.ifft2(div_hat)

        if self.use_torch and return_numpy:
            omega = self.to_numpy(omega)
            div_field = self.to_numpy(div_field)

        # 快取，供壓力/診斷使用
        self.last_omega_hat = omega_hat
        self.last_div_hat = div_hat

        return omega, div_field, omega_hat

    def compute_rhs(self, U_hat, V_hat):
        """計算 RHS（頻譜空間），使用 Leray 投影法"""
        omega_hat = self.compute_omega_hat(U_hat, V_hat)

        # 轉回實空間計算非線性項（使用旋度形式以減少 FFT 次數）
        if self.dealias and self.dealias_mode == "3/2":
            if self.use_torch:
                U_hat_pad = self.pad_hat(U_hat)
                V_hat_pad = self.pad_hat(V_hat)
                omega_hat_pad = 1j * (self.pad_kx * V_hat_pad - self.pad_ky * U_hat_pad)
                fields_hat = torch.stack([U_hat_pad, V_hat_pad, omega_hat_pad], dim=0)
            else:
                U_hat_pad = self.pad_hat(U_hat)
                V_hat_pad = self.pad_hat(V_hat)
                omega_hat_pad = 1j * (self.pad_kx * V_hat_pad - self.pad_ky * U_hat_pad)
                fields_hat = np.stack([U_hat_pad, V_hat_pad, omega_hat_pad], axis=0)

            U_pad, V_pad, omega_pad = self.ifft2_batch(fields_hat)
            U_pad *= self.pad_scale
            V_pad *= self.pad_scale
            omega_pad *= self.pad_scale

            conv_U = omega_pad * V_pad
            conv_V = -omega_pad * U_pad

            if self.use_torch:
                conv_hat_pad = self.fft2_batch(torch.stack([conv_U, conv_V], dim=0)) * self.pad_scale_inv
            else:
                conv_hat_pad = self.fft2_batch(np.stack([conv_U, conv_V], axis=0)) * self.pad_scale_inv

            conv_U_hat = self.truncate_hat(conv_hat_pad[0])
            conv_V_hat = self.truncate_hat(conv_hat_pad[1])
        else:
            if self.use_torch:
                fields_hat = torch.stack([U_hat, V_hat, omega_hat], dim=0)
            else:
                fields_hat = np.stack([U_hat, V_hat, omega_hat], axis=0)

            U, V, omega = self.ifft2_batch(fields_hat)

            # 對流項 (Rotational form): -ω×u = (ω * v, -ω * u)
            conv_U = omega * V
            conv_V = -omega * U

            # 轉回頻譜空間並去混疊
            if self.use_torch:
                conv_hat = self.fft2_batch(torch.stack([conv_U, conv_V], dim=0))
            else:
                conv_hat = self.fft2_batch(np.stack([conv_U, conv_V], axis=0))

            conv_hat = conv_hat * self.dealias_mask
            conv_U_hat, conv_V_hat = conv_hat[0], conv_hat[1]

        # 擴散項 (Diffusion) - 顯式處理
        diff_U_hat = -self.nu * self.k2 * U_hat
        diff_V_hat = -self.nu * self.k2 * V_hat

        # 組裝未投影的 RHS
        # RHS* = Convection + Diffusion + Forcing (使用預計算的 forcing_hat)
        rhs_U_star = conv_U_hat + diff_U_hat + self.forcing_U_hat
        rhs_V_star = conv_V_hat + diff_V_hat + self.forcing_V_hat

        # 投影 RHS (Leray Projection)
        # P[RHS*] = RHS* - ∇p  => 自動消去壓力梯度部分
        rhs_U, rhs_V = self._project_hat(rhs_U_star, rhs_V_star)

        return rhs_U, rhs_V, omega_hat

    def step_rk4(self):
        """RK4 時間步進"""
        U0, V0 = self.U_hat, self.V_hat

        # Stage 1
        k1_U, k1_V, _ = self.compute_rhs(U0, V0)

        # Stage 2
        k2_U, k2_V, _ = self.compute_rhs(
            U0 + 0.5 * self.dt * k1_U,
            V0 + 0.5 * self.dt * k1_V
        )

        # Stage 3
        k3_U, k3_V, _ = self.compute_rhs(
            U0 + 0.5 * self.dt * k2_U,
            V0 + 0.5 * self.dt * k2_V
        )

        # Stage 4
        k4_U, k4_V, _ = self.compute_rhs(
            U0 + self.dt * k3_U,
            V0 + self.dt * k3_V
        )

        # 更新
        self.U_hat = U0 + (self.dt / 6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        self.V_hat = V0 + (self.dt / 6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)

        # 額外保險：確保時間步進後的狀態也是嚴格無散度的
        if self.project_after_step:
            self.project_incompressible()

    def compute_pressure_from_state(self, U, V, omega=None, omega_hat=None):
        """計算壓力場 (用於輸出)，重用 omega/div 形式"""
        if self.use_torch and torch.is_tensor(U):
            if omega is None:
                if omega_hat is None:
                    omega_hat = getattr(self, 'last_omega_hat', None)
                    if omega_hat is None:
                        omega_hat = self.ikx * self.V_hat - self.iky * self.U_hat
                omega = torch.fft.ifft2(omega_hat).real

            omega_v = omega * V
            omega_u = omega * U
            speed2 = U**2 + V**2

            rhs_hat = -(self.ikx * torch.fft.fft2(omega_v) + self.iky * torch.fft.fft2(-omega_u))
            rhs_hat += 0.5 * self.k2 * torch.fft.fft2(speed2)

            P_hat = -rhs_hat / self.k2_safe
            P_hat[0, 0] = 0.0

            return torch.fft.ifft2(P_hat).real

        ikx = self.to_numpy(self.ikx) if self.use_torch else self.ikx
        iky = self.to_numpy(self.iky) if self.use_torch else self.iky
        k2 = self.to_numpy(self.k2) if self.use_torch else self.k2
        k2_safe = self.to_numpy(self.k2_safe) if self.use_torch else self.k2_safe

        if omega is None:
            if omega_hat is None:
                omega_hat = getattr(self, 'last_omega_hat', None)
                if omega_hat is None:
                    if self.use_torch:
                        omega_hat = ikx * self.to_numpy(self.V_hat) - iky * self.to_numpy(self.U_hat)
                    else:
                        omega_hat = ikx * self.V_hat - iky * self.U_hat
            else:
                omega_hat = self.to_numpy(omega_hat) if self.use_torch else omega_hat
            omega = np.fft.ifft2(omega_hat).real

        omega_v = omega * V
        omega_u = omega * U
        speed2 = U**2 + V**2

        rhs_hat = -(ikx * np.fft.fft2(omega_v) + iky * np.fft.fft2(-omega_u))
        rhs_hat += 0.5 * k2 * np.fft.fft2(speed2)

        P_hat = -rhs_hat / k2_safe
        P_hat[0, 0] = 0.0

        return np.fft.ifft2(P_hat).real

    def compute_pressure(self, U=None, V=None, omega=None, omega_hat=None):
        """計算壓力場 (用於輸出)"""
        if U is None or V is None:
            U = self.ifft2(self.U_hat)
            V = self.ifft2(self.V_hat)

        return self.compute_pressure_from_state(U, V, omega, omega_hat)

    def add_perturbation(self, method: str = 'random', amplitude: float = 0.1):
        """添加擾動"""
        if method == 'random':
            if self.use_torch:
                pert_U = amplitude * torch.randn_like(self.X)
                pert_V = amplitude * torch.randn_like(self.X)
            else:
                pert_U = amplitude * np.random.randn(*self.X.shape)
                pert_V = amplitude * np.random.randn(*self.X.shape)

            pert_U_hat = self.fft2(pert_U) * self.dealias_mask
            pert_V_hat = self.fft2(pert_V) * self.dealias_mask

        elif method == 'unstable_mode':
            # 不穩定模態（k_x=3, k_y=4）
            k_x, k_y = 3, self.k_f
            kx_phys = k_x * 2*np.pi / self.L
            ky_phys = k_y * 2*np.pi / self.L
            
            if self.use_torch:
                phase_U = amplitude * torch.cos(kx_phys * self.X + ky_phys * self.Y)
                phase_V = amplitude * torch.sin(kx_phys * self.X + ky_phys * self.Y)
            else:
                phase_U = amplitude * np.cos(kx_phys * self.X + ky_phys * self.Y)
                phase_V = amplitude * np.sin(kx_phys * self.X + ky_phys * self.Y)

            pert_U_hat = self.fft2(phase_U)
            pert_V_hat = self.fft2(phase_V)

        else:
            raise ValueError(f"Unknown perturbation method: {method}")

        self.U_hat += pert_U_hat
        self.V_hat += pert_V_hat

        # ✅ 關鍵修正：擾動後立即投影，確保不可壓縮性
        self.U_hat, self.V_hat = self._project_hat(self.U_hat, self.V_hat)

        logging.info(f"✅ 添加擾動：method={method}, amplitude={amplitude}")

    def run(
        self,
        T_end: float = 20.0,
        save_interval: int = 100,
        output_file: str = "kolmogorov_dns.h5",
        perturbation_times: list = None,
        perturbation_method: str = 'random',
        perturbation_amplitude: float = None,
    ):
        """執行模擬"""
        n_steps = int(T_end / self.dt)
        
        # 預設擾動設定（避免落在數值噪聲）
        if perturbation_times is None:
            perturbation_times = []
            logging.info("ℹ️  未指定擾動時間，預設不注入擾動")

        # 自動設置擾動振幅（針對 Re=100 使用溫和擾動）
        if perturbation_amplitude is None:
            U0 = self.ifft2(self.U_hat)
            V0 = self.ifft2(self.V_hat)

            if self.use_torch:
                U0_val = U0.max().item()
                V0_val = V0.max().item()
            else:
                U0_val = U0.max()
                V0_val = V0.max()

            base_velocity = max(U0_val, V0_val, 0.1)  # 取 U/V 中較大者，最小 0.1
            
            # 關鍵調整：使用溫和擾動（0.5×），避免產生過大梯度導致散度誤差
            # 對於 Re=100 附近的臨界雷諾數，小擾動足以激發不穩定性
            perturbation_amplitude = min(0.8, 0.5 * base_velocity)  # 降低到 0.5× 且上限 0.8
            logging.info(f"📊 初始速度 U0={U0_val:.4f}, V0={V0_val:.4f}, 自動設置擾動振幅={perturbation_amplitude:.4f} (0.5×max(U0,V0), 上限0.8)")

        # 準備輸出 (NPY)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)

        n_snapshots = n_steps // save_interval + 1
        u_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float32)
        v_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float32)
        p_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float32)
        omega_snapshots = np.zeros((n_snapshots, self.N, self.N), dtype=np.float32)
        time_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        ke_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        enstrophy_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
        div_snapshots = np.zeros((n_snapshots,), dtype=np.float64)

        # 運行
        logging.info(f"▶️  開始 DNS 模擬：T_end={T_end}, n_steps={n_steps}")

        idx = 0
        t_start = time.time()
        for step in range(n_steps + 1):
            t = step * self.dt

            # 擾動
            if perturbation_times and any(abs(t - tp) < 0.5*self.dt for tp in perturbation_times):
                self.add_perturbation(perturbation_method, perturbation_amplitude)

            # 保存
            if step % save_interval == 0:
                U = self.ifft2(self.U_hat)
                V = self.ifft2(self.V_hat)

                # 計算診斷指標 (使用譜導數以獲得高精度)
                if self.use_torch:
                    omega, div_field, omega_hat = self.compute_omega_div(return_numpy=False)
                    P = self.compute_pressure_from_state(U, V, omega, omega_hat)

                    u_snapshots[idx] = self.to_numpy(U)
                    v_snapshots[idx] = self.to_numpy(V)
                    p_snapshots[idx] = self.to_numpy(P)
                    omega_snapshots[idx] = self.to_numpy(omega)

                    KE = 0.5 * torch.mean(U**2 + V**2).item()
                    enstrophy = 0.5 * torch.mean(omega**2).item()
                    div_err = torch.abs(div_field).max().item()
                else:
                    omega, div_field, omega_hat = self.compute_omega_div()
                    P = self.compute_pressure_from_state(U, V, omega, omega_hat)

                    u_snapshots[idx] = U
                    v_snapshots[idx] = V
                    p_snapshots[idx] = P
                    omega_snapshots[idx] = omega

                    KE = 0.5 * np.mean(U**2 + V**2)
                    enstrophy = 0.5 * np.mean(omega**2)
                    div_err = np.abs(div_field).max()

                time_snapshots[idx] = t

                ke_snapshots[idx] = KE
                enstrophy_snapshots[idx] = enstrophy
                div_snapshots[idx] = div_err

                if step % (save_interval * 10) == 0:
                    elapsed = time.time() - t_start
                    speed = (step + 1) / elapsed if elapsed > 0 else 0
                    eta = (n_steps - step) / speed if speed > 0 else 0
                    logging.info(
                        f"  Step {step:6d}/{n_steps} | t={t:.3f} | "
                        f"KE={KE:.4e} | Enstrophy={enstrophy:.4e} | "
                        f"Div_err={div_err:.4e} | Speed={speed:.1f} steps/s | ETA={eta:.0f}s"
                    )

                idx += 1

            # 步進
            if step < n_steps:
                self.step_rk4()

        output_data = {
            'x': np.linspace(0, self.L, self.N, endpoint=False),
            'y': np.linspace(0, self.L, self.N, endpoint=False),
            'u': u_snapshots,
            'v': v_snapshots,
            'p': p_snapshots,
            'omega': omega_snapshots,
            'time': time_snapshots,
            'diagnostics': {
                'kinetic_energy': ke_snapshots,
                'enstrophy': enstrophy_snapshots,
                'divergence_error': div_snapshots,
            },
            'config': {
                'N': self.N,
                'L': self.L,
                'nu': self.nu,
                'A': self.A,
                'k_f': self.k_f,
                'dt': self.dt,
                'T_end': T_end,
                'dealias_mode': self.dealias_mode,
                'backend': self.backend,
                'device': str(self.device) if self.device else 'numpy',
                'save_interval': save_interval,
                'perturbation_times': perturbation_times,
                'perturbation_method': perturbation_method,
                'perturbation_amplitude': perturbation_amplitude,
            }
        }

        np.save(output_file, output_data, allow_pickle=True)

        total_time = time.time() - t_start
        logging.info(f"✅ 模擬完成！總耗時：{total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        logging.info(f"   平均速度：{n_steps/total_time:.1f} steps/s")
        logging.info(f"💾 結果已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow DNS 生成器 (自動選擇 GPU/CPU)')
    parser.add_argument('--N', type=int, default=1024, help='網格點數')
    parser.add_argument('--L', type=float, default=2*np.pi, help='域長度')
    parser.add_argument('--nu', type=float, default=1e-5, help='黏滯係數')
    parser.add_argument('--A', type=float, default=0.1, help='強迫振幅')
    parser.add_argument('--k_f', type=int, default=4, help='強迫波數')
    parser.add_argument('--dt', type=float, default=0.001, help='時間步長')
    parser.add_argument('--T_end', type=float, default=20.0, help='結束時間')
    parser.add_argument('--save_interval', type=int, default=100, help='保存間隔')
    parser.add_argument('--output', type=str, default='data/kolmogorov_dns.npy', help='輸出文件 (.npy)')
    parser.add_argument('--perturbation_times', type=float, nargs='+', help='擾動時刻')
    parser.add_argument('--perturbation_method', type=str, default='random', 
                       choices=['random', 'unstable_mode'], help='擾動方法')
    parser.add_argument('--perturbation_amplitude', type=float, default=None, help='擾動振幅（默認自動推斷，Re~30-100 建議 1.0-1.5）')
    parser.add_argument('--backend', type=str, default=None,
                       choices=['torch-mps', 'torch-cuda', 'torch-cpu', 'numpy'],
                       help='手動指定後端（默認自動檢測）')
    parser.add_argument('--dealias-mode', type=str, default='3/2',
                       choices=['2/3', '3/2'],
                       help='去混疊方式')
    parser.add_argument('--no-project-after-step', action='store_true',
                       help='關閉時間步進後的額外投影')
    
    args = parser.parse_args()

    logging.info("=" * 80)
    logging.info("Kolmogorov Flow DNS 生成器 (統一版本)")
    logging.info("=" * 80)

    solver = KolmogorovFlowDNS(
        N=args.N,
        L=args.L,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
        dealias_mode=args.dealias_mode,
        backend=args.backend,
        project_after_step=not args.no_project_after_step,
    )

    solver.run(
        T_end=args.T_end,
        save_interval=args.save_interval,
        output_file=args.output,
        perturbation_times=args.perturbation_times,
        perturbation_method=args.perturbation_method,
        perturbation_amplitude=args.perturbation_amplitude,
    )


if __name__ == '__main__':
    main()
