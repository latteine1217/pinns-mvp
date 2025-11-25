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
import h5py
from typing import Dict, Tuple, Optional
import logging
import time

# 檢測 PyTorch 是否可用
try:
    import torch
    TORCH_AVAILABLE = True
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
        N: int = 256,
        L: float = 2 * np.pi,
        nu: float = 0.01,
        A: float = 1.0,
        k_f: int = 4,
        dt: float = 0.001,
        dealias: bool = True,
        backend: Optional[str] = None,
    ):
        """
        Args:
            N: 網格點數（每方向）
            L: 域長度
            nu: 黏滯係數
            A: 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            dealias: 是否啟用 2/3 去混疊
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
        else:
            self.kx = kx
            self.ky = ky
            self.k2 = k2

        # 去混疊遮罩
        if dealias:
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

        # 強迫項（實空間，僅作用於 y-momentum）
        if self.use_torch:
            self.forcing_y = A * torch.sin(k_f * self.Y)
        else:
            self.forcing_y = A * np.sin(k_f * Y)

        # 初始化場（使用弱化層流 Kolmogorov 解，避免數值不穩定）
        # 完整層流解: v(y) = (A / (nu * k_f^2)) * sin(k_f * y) 可能過大
        # 使用縮小版本: v(y) = alpha * (A / (nu * k_f^2)) * sin(k_f * y)
        # alpha = min(0.1, nu * k_f^2 / A) 確保 |V| 不超過 ~0.1-1.0
        alpha = min(0.1, nu * k_f**2 / A)
        V_amp = alpha * A / (nu * k_f**2)
        
        if self.use_torch:
            V_init = V_amp * torch.sin(k_f * self.Y)
            self.U_hat = torch.zeros((N, N), dtype=torch.complex64, device=self.device)
            self.V_hat = self.fft2(V_init)
        else:
            V_init = V_amp * np.sin(k_f * Y)
            self.U_hat = np.zeros((N, N), dtype=complex)
            self.V_hat = np.fft.fft2(V_init)
        
        logging.info(f"✅ 初始化弱化層流解：V_max = {V_amp:.6f} (alpha = {alpha:.6f})")

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

    def compute_rhs(self, U_hat, V_hat):
        """計算 RHS（頻譜空間）"""
        # 轉回實空間
        U = self.ifft2(U_hat)
        V = self.ifft2(V_hat)

        # 對流項（實空間）
        if self.use_torch:
            dUdx = self.ifft2(1j * self.kx * U_hat)
            dUdy = self.ifft2(1j * self.ky * U_hat)
            dVdx = self.ifft2(1j * self.kx * V_hat)
            dVdy = self.ifft2(1j * self.ky * V_hat)
        else:
            dUdx = self.ifft2(1j * self.kx * U_hat)
            dUdy = self.ifft2(1j * self.ky * U_hat)
            dVdx = self.ifft2(1j * self.kx * V_hat)
            dVdy = self.ifft2(1j * self.ky * V_hat)

        # 對流項
        conv_U = -(U * dUdx + V * dUdy)
        conv_V = -(U * dVdx + V * dVdy)

        # 轉回頻譜空間
        conv_U_hat = self.fft2(conv_U) * self.dealias_mask
        conv_V_hat = self.fft2(conv_V) * self.dealias_mask

        # 壓力投影（保持無散度）
        k2_safe = self.k2.clone() if self.use_torch else self.k2.copy()
        if self.use_torch:
            k2_safe[0, 0] = 1.0
        else:
            k2_safe[0, 0] = 1.0

        div_hat = 1j * self.kx * conv_U_hat + 1j * self.ky * conv_V_hat
        P_hat = div_hat / k2_safe
        if self.use_torch:
            P_hat[0, 0] = 0.0
        else:
            P_hat[0, 0] = 0.0

        conv_U_hat -= 1j * self.kx * P_hat
        conv_V_hat -= 1j * self.ky * P_hat

        # 黏滯項 + 強迫（Kolmogorov forcing 僅作用於 y-momentum）
        forcing_y_hat = self.fft2(self.forcing_y)
        
        rhs_U = conv_U_hat - self.nu * self.k2 * U_hat                    # x-momentum: 無強迫項
        rhs_V = conv_V_hat - self.nu * self.k2 * V_hat + forcing_y_hat    # y-momentum: 有強迫項

        return rhs_U, rhs_V

    def step_rk4(self):
        """RK4 時間步進"""
        U0, V0 = self.U_hat, self.V_hat

        # Stage 1
        k1_U, k1_V = self.compute_rhs(U0, V0)

        # Stage 2
        k2_U, k2_V = self.compute_rhs(
            U0 + 0.5 * self.dt * k1_U,
            V0 + 0.5 * self.dt * k1_V
        )

        # Stage 3
        k3_U, k3_V = self.compute_rhs(
            U0 + 0.5 * self.dt * k2_U,
            V0 + 0.5 * self.dt * k2_V
        )

        # Stage 4
        k4_U, k4_V = self.compute_rhs(
            U0 + self.dt * k3_U,
            V0 + self.dt * k3_V
        )

        # 更新
        self.U_hat = U0 + (self.dt / 6) * (k1_U + 2*k2_U + 2*k3_U + k4_U)
        self.V_hat = V0 + (self.dt / 6) * (k1_V + 2*k2_V + 2*k3_V + k4_V)

    def compute_pressure(self):
        """計算壓力場"""
        U = self.ifft2(self.U_hat)
        V = self.ifft2(self.V_hat)

        if self.use_torch:
            dUdx = self.ifft2(1j * self.kx * self.U_hat)
            dUdy = self.ifft2(1j * self.ky * self.U_hat)
            dVdx = self.ifft2(1j * self.kx * self.V_hat)
            dVdy = self.ifft2(1j * self.ky * self.V_hat)
        else:
            dUdx = self.ifft2(1j * self.kx * self.U_hat)
            dUdy = self.ifft2(1j * self.ky * self.U_hat)
            dVdx = self.ifft2(1j * self.kx * self.V_hat)
            dVdy = self.ifft2(1j * self.ky * self.V_hat)

        # Poisson 方程右側
        rhs = -(dUdx**2 + 2*dUdy*dVdx + dVdy**2)
        rhs_hat = self.fft2(rhs)

        # 求解
        k2_safe = self.k2.clone() if self.use_torch else self.k2.copy()
        if self.use_torch:
            k2_safe[0, 0] = 1.0
        else:
            k2_safe[0, 0] = 1.0

        P_hat = rhs_hat / k2_safe
        if self.use_torch:
            P_hat[0, 0] = 0.0
        else:
            P_hat[0, 0] = 0.0

        return self.ifft2(P_hat)

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
            
            if self.use_torch:
                phase_U = amplitude * torch.cos(k_x * 2*np.pi/self.L * self.X + 
                                               k_y * 2*np.pi/self.L * self.Y)
                phase_V = amplitude * torch.sin(k_x * 2*np.pi/self.L * self.X + 
                                               k_y * 2*np.pi/self.L * self.Y)
            else:
                phase_U = amplitude * np.cos(k_x * 2*np.pi/self.L * self.X + 
                                             k_y * 2*np.pi/self.L * self.Y)
                phase_V = amplitude * np.sin(k_x * 2*np.pi/self.L * self.X + 
                                             k_y * 2*np.pi/self.L * self.Y)

            pert_U_hat = self.fft2(phase_U)
            pert_V_hat = self.fft2(phase_V)

        else:
            raise ValueError(f"Unknown perturbation method: {method}")

        self.U_hat += pert_U_hat
        self.V_hat += pert_V_hat

        logging.info(f"✅ 添加擾動：method={method}, amplitude={amplitude}")

    def run(
        self,
        T_end: float = 10.0,
        save_interval: int = 100,
        output_file: str = "kolmogorov_dns.h5",
        perturbation_times: list = None,
        perturbation_method: str = 'random',
        perturbation_amplitude: float = None,
    ):
        """執行模擬"""
        n_steps = int(T_end / self.dt)
        
        # 自動設置擾動振幅
        if perturbation_amplitude is None:
            U0 = self.to_numpy(self.ifft2(self.U_hat)).max()
            perturbation_amplitude = 0.5 * U0 if U0 > 0.1 else 0.1
            logging.info(f"📊 層流速度 U0 = {U0:.4f}, 自動設置擾動振幅 = {perturbation_amplitude:.4f} (0.5×U0)")

        # 準備 HDF5
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        h5f = h5py.File(output_file, 'w')
        
        # 配置
        config = h5f.create_group('config')
        config.attrs['N'] = self.N
        config.attrs['L'] = self.L
        config.attrs['nu'] = self.nu
        config.attrs['A'] = self.A
        config.attrs['k_f'] = self.k_f
        config.attrs['dt'] = self.dt
        config.attrs['T_end'] = T_end
        config.attrs['backend'] = self.backend
        config.attrs['device'] = str(self.device) if self.device else 'numpy'

        # 數據集
        n_snapshots = n_steps // save_interval + 1
        dset_u = h5f.create_dataset('u', shape=(n_snapshots, self.N, self.N), dtype='float32')
        dset_v = h5f.create_dataset('v', shape=(n_snapshots, self.N, self.N), dtype='float32')
        dset_p = h5f.create_dataset('p', shape=(n_snapshots, self.N, self.N), dtype='float32')
        dset_time = h5f.create_dataset('time', shape=(n_snapshots,), dtype='float64')
        
        diag = h5f.create_group('diagnostics')
        dset_ke = diag.create_dataset('kinetic_energy', shape=(n_snapshots,), dtype='float64')
        dset_ens = diag.create_dataset('enstrophy', shape=(n_snapshots,), dtype='float64')
        dset_div = diag.create_dataset('divergence_error', shape=(n_snapshots,), dtype='float64')

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
                U = self.to_numpy(self.ifft2(self.U_hat))
                V = self.to_numpy(self.ifft2(self.V_hat))
                P = self.to_numpy(self.compute_pressure())

                dset_u[idx] = U
                dset_v[idx] = V
                dset_p[idx] = P
                dset_time[idx] = t

                KE = 0.5 * np.mean(U**2 + V**2)
                omega = np.gradient(V, axis=1) - np.gradient(U, axis=0)
                enstrophy = 0.5 * np.mean(omega**2)
                div_err = np.abs(np.gradient(U, axis=1) + np.gradient(V, axis=0)).max()

                dset_ke[idx] = KE
                dset_ens[idx] = enstrophy
                dset_div[idx] = div_err

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

        h5f.close()
        
        total_time = time.time() - t_start
        logging.info(f"✅ 模擬完成！總耗時：{total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")
        logging.info(f"   平均速度：{n_steps/total_time:.1f} steps/s")
        logging.info(f"💾 結果已保存至：{output_file}")


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow DNS 生成器 (自動選擇 GPU/CPU)')
    parser.add_argument('--N', type=int, default=512, help='網格點數')
    parser.add_argument('--L', type=float, default=2*np.pi, help='域長度')
    parser.add_argument('--nu', type=float, default=0.01, help='黏滯係數')
    parser.add_argument('--A', type=float, default=1.0, help='強迫振幅')
    parser.add_argument('--k_f', type=int, default=4, help='強迫波數')
    parser.add_argument('--dt', type=float, default=0.001, help='時間步長')
    parser.add_argument('--T_end', type=float, default=10.0, help='結束時間')
    parser.add_argument('--save_interval', type=int, default=100, help='保存間隔')
    parser.add_argument('--output', type=str, default='data/kolmogorov_dns.h5', help='輸出文件')
    parser.add_argument('--perturbation_times', type=float, nargs='+', help='擾動時刻')
    parser.add_argument('--perturbation_method', type=str, default='unstable_mode', 
                       choices=['random', 'unstable_mode'], help='擾動方法')
    parser.add_argument('--backend', type=str, default=None,
                       choices=['torch-mps', 'torch-cuda', 'torch-cpu', 'numpy'],
                       help='手動指定後端（默認自動檢測）')
    
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
        backend=args.backend,
    )

    solver.run(
        T_end=args.T_end,
        save_interval=args.save_interval,
        output_file=args.output,
        perturbation_times=args.perturbation_times,
        perturbation_method=args.perturbation_method,
    )


if __name__ == '__main__':
    main()
