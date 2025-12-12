"""
Kolmogorov Flow Low-Fidelity 資料生成器
=========================================

目的：生成低保真度參考場，用於 PINN 的軟先驗訓練

策略：
- 粗網格（32x32 或 64x64，相較 hi-fi 的 256x256）
- 較高黏滯係數（ν_LF = α·ν_HF，α ∈ [2, 5]）
- 時間平均場（統計穩態，模擬 RANS-like bias）

數值方法：
- Pseudo-spectral (Fourier) + 2/3 dealiasing
- RK2/RK4 時間積分
- Projection method for pressure

參考文獻：
- Musacchio & Boffetta (2014), PRE 89, 023004
- 本專案 channel flow RANS 策略

作者：PINNs-MVP 團隊
日期：2025-12-10
"""

import numpy as np
import argparse
from pathlib import Path
import h5py
from typing import Dict, Tuple, Optional
import logging
import time
import yaml

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class KolmogorovFlowLowFi:
    """
    2D Kolmogorov Flow Low-Fidelity 求解器
    
    特點：
    - 粗網格解析度
    - 偏高黏滯（降低 Re）
    - 產生平滑、過度 laminar 的場
    """
    
    def __init__(
        self,
        N: int = 32,
        L: float = 2 * np.pi,
        nu: float = 0.02,  # 較高黏滯（hi-fi 通常是 0.01）
        A: float = 1.0,
        k_f: int = 4,
        dt: float = 0.01,  # 較大時間步
        dealias: bool = True,
    ):
        """
        Args:
            N: 網格點數（每方向），32 或 64（hi-fi 用 256）
            L: 域長度
            nu: 黏滯係數（hi-fi 的 2-5 倍）
            A: 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            dealias: 是否啟用 2/3 去混疊
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.dealias = dealias
        
        # 網格設定
        self.dx = L / N
        self.dy = L / N
        
        # 物理空間網格
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y)
        
        # 頻譜空間波數
        kx = np.fft.fftfreq(N, d=L/(2*np.pi*N)) * 2 * np.pi
        ky = np.fft.fftfreq(N, d=L/(2*np.pi*N)) * 2 * np.pi
        self.KX, self.KY = np.meshgrid(kx, ky)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # 避免除零
        
        # Dealiasing mask (2/3 rule)
        if dealias:
            k_max = N // 3
            self.dealias_mask = (np.abs(self.KX) <= k_max) & (np.abs(self.KY) <= k_max)
        else:
            self.dealias_mask = np.ones((N, N), dtype=bool)
        
        # 強迫項（頻譜空間）
        self.forcing_hat = self._setup_forcing()
        
        # 統計量記錄
        self.stats_history = {
            'time': [],
            'kinetic_energy': [],
            'enstrophy': [],
            'dissipation': [],
        }
        
        logging.info("=" * 60)
        logging.info("Low-Fidelity Kolmogorov Flow 求解器初始化")
        logging.info("=" * 60)
        logging.info(f"網格解析度: {N}x{N} (粗網格)")
        logging.info(f"域大小: {L:.4f}x{L:.4f}")
        logging.info(f"網格間距: Δx = Δy = {self.dx:.6f}")
        logging.info(f"黏滯係數: ν = {nu:.6f} (偏高)")
        logging.info(f"強迫參數: A = {A:.4f}, k_f = {k_f}")
        logging.info(f"時間步長: dt = {dt:.6f}")
        logging.info(f"Dealiasing: {dealias}")
        
        # 計算雷諾數（參考 Musacchio & Boffetta 2014）
        Re = np.sqrt(A) * (2*np.pi/k_f)**(3/2) / nu
        logging.info(f"雷諾數: Re = {Re:.2f} (較低 Re，偏 laminar)")
        logging.info("=" * 60)
    
    def _setup_forcing(self) -> np.ndarray:
        """設定 Kolmogorov 強迫項（y 方向週期驅動）"""
        # 物理空間：f_x = A * sin(k_f * y), f_y = 0
        fx = self.A * np.sin(self.k_f * self.Y)
        fy = np.zeros_like(fx)
        
        # 轉換到頻譜空間
        fx_hat = np.fft.fft2(fx)
        fy_hat = np.fft.fft2(fy)
        
        return fx_hat, fy_hat
    
    def initialize_field(self, mode: str = 'random') -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化速度場
        
        Args:
            mode: 'random', 'laminar', 'vortices'
        
        Returns:
            u, v: 初始速度場（物理空間）
        """
        if mode == 'random':
            # 隨機小擾動
            u = 0.01 * np.random.randn(self.N, self.N)
            v = 0.01 * np.random.randn(self.N, self.N)
        
        elif mode == 'laminar':
            # 近似 laminar 解
            u = 0.1 * np.sin(self.k_f * self.Y)
            v = np.zeros_like(u)
        
        elif mode == 'vortices':
            # 隨機渦對
            u = 0.05 * (np.sin(2*self.X) * np.cos(3*self.Y) + 
                        0.5 * np.random.randn(self.N, self.N))
            v = 0.05 * (np.cos(2*self.X) * np.sin(3*self.Y) + 
                        0.5 * np.random.randn(self.N, self.N))
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 確保無散度（投影到無散場）
        u, v = self._project_divergence_free(u, v)
        
        return u, v
    
    def _project_divergence_free(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """投影到無散度場（∇·u = 0）"""
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        # 計算散度
        div_hat = 1j * (self.KX * u_hat + self.KY * v_hat)
        
        # 解 Poisson 方程：Δφ = div
        phi_hat = -div_hat / self.K2
        phi_hat[0, 0] = 0  # 壓力常數項
        
        # 修正速度：u_new = u - ∇φ
        u_hat -= 1j * self.KX * phi_hat
        v_hat -= 1j * self.KY * phi_hat
        
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        
        return u, v
    
    def compute_nonlinear(self, u_hat: np.ndarray, v_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算非線性項：N = -u·∇u（頻譜空間）
        
        使用 dealiased pseudo-spectral
        """
        # 轉回物理空間
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        
        # 計算梯度（頻譜空間）
        du_dx = np.real(np.fft.ifft2(1j * self.KX * u_hat))
        du_dy = np.real(np.fft.ifft2(1j * self.KY * u_hat))
        dv_dx = np.real(np.fft.ifft2(1j * self.KX * v_hat))
        dv_dy = np.real(np.fft.ifft2(1j * self.KY * v_hat))
        
        # 非線性項（物理空間）
        Nx = -(u * du_dx + v * du_dy)
        Ny = -(u * dv_dx + v * dv_dy)
        
        # 轉回頻譜空間並應用 dealiasing
        Nx_hat = np.fft.fft2(Nx) * self.dealias_mask
        Ny_hat = np.fft.fft2(Ny) * self.dealias_mask
        
        return Nx_hat, Ny_hat
    
    def step_rk4(self, u_hat: np.ndarray, v_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        RK4 時間步進（改用半隱式處理黏滯項，提升穩定性）
        
        d/dt(u_hat) = N(u) - ν·k²·u_hat + f_hat
        """
        fx_hat, fy_hat = self.forcing_hat
        
        # 黏滯項的積分因子（半隱式）
        integrating_factor = np.exp(-self.nu * self.K2 * self.dt)
        
        def rhs_explicit(uh, vh):
            """僅計算非線性項與強迫（顯式）"""
            Nx_hat, Ny_hat = self.compute_nonlinear(uh, vh)
            return Nx_hat + fx_hat, Ny_hat + fy_hat
        
        # RK4 stages（僅非線性項）
        k1_u, k1_v = rhs_explicit(u_hat, v_hat)
        
        u_tmp = integrating_factor * u_hat + 0.5 * self.dt * k1_u
        v_tmp = integrating_factor * v_hat + 0.5 * self.dt * k1_v
        k2_u, k2_v = rhs_explicit(u_tmp, v_tmp)
        
        u_tmp = integrating_factor * u_hat + 0.5 * self.dt * k2_u
        v_tmp = integrating_factor * v_hat + 0.5 * self.dt * k2_v
        k3_u, k3_v = rhs_explicit(u_tmp, v_tmp)
        
        u_tmp = integrating_factor * u_hat + self.dt * k3_u
        v_tmp = integrating_factor * v_hat + self.dt * k3_v
        k4_u, k4_v = rhs_explicit(u_tmp, v_tmp)
        
        # 更新（包含黏滯項的半隱式處理）
        u_hat_new = integrating_factor * u_hat + (self.dt/6) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        v_hat_new = integrating_factor * v_hat + (self.dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        
        return u_hat_new, v_hat_new
    
    def compute_statistics(self, u: np.ndarray, v: np.ndarray) -> Dict[str, float]:
        """計算流場統計量"""
        # 動能
        KE = 0.5 * np.mean(u**2 + v**2)
        
        # 渦度
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        omega = np.real(np.fft.ifft2(1j * (self.KX * v_hat - self.KY * u_hat)))
        enstrophy = 0.5 * np.mean(omega**2)
        
        # 耗散率
        dissipation = self.nu * np.mean(omega**2)
        
        return {
            'kinetic_energy': KE,
            'enstrophy': enstrophy,
            'dissipation': dissipation,
        }
    
    def run_simulation(
        self,
        T_total: float = 200.0,
        T_spinup: float = 50.0,
        save_interval: float = 1.0,
        init_mode: str = 'random',
    ) -> Dict[str, np.ndarray]:
        """
        執行 low-fidelity 模擬
        
        Args:
            T_total: 總模擬時間
            T_spinup: Spin-up 時間（丟棄前期過渡態）
            save_interval: 存檔間隔
            init_mode: 初始化模式
        
        Returns:
            results: 包含時間平均場與統計量
        """
        logging.info("\n開始 Low-Fi 模擬...")
        logging.info(f"總時間: {T_total:.1f}, Spin-up: {T_spinup:.1f}")
        logging.info(f"存檔間隔: {save_interval:.2f}")
        
        # 初始化
        u, v = self.initialize_field(mode=init_mode)
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        # 時間積分
        t = 0.0
        step = 0
        
        # 用於時間平均
        u_sum = np.zeros_like(u)
        v_sum = np.zeros_like(v)
        n_avg = 0
        
        # Snapshots（可選）
        snapshots = []
        
        start_time = time.time()
        
        while t < T_total:
            # 時間步進
            u_hat, v_hat = self.step_rk4(u_hat, v_hat)
            t += self.dt
            step += 1
            
            # 轉回物理空間
            u = np.real(np.fft.ifft2(u_hat))
            v = np.real(np.fft.ifft2(v_hat))
            
            # 統計量
            if step % 100 == 0:
                stats = self.compute_statistics(u, v)
                self.stats_history['time'].append(t)
                self.stats_history['kinetic_energy'].append(stats['kinetic_energy'])
                self.stats_history['enstrophy'].append(stats['enstrophy'])
                self.stats_history['dissipation'].append(stats['dissipation'])
                
                if step % 1000 == 0:
                    logging.info(
                        f"Step {step:6d}, t = {t:8.2f}, "
                        f"KE = {stats['kinetic_energy']:.6f}, "
                        f"Ω = {stats['enstrophy']:.6f}"
                    )
            
            # 時間平均（僅在 spin-up 後）
            if t > T_spinup:
                u_sum += u
                v_sum += v
                n_avg += 1
                
                # 存 snapshot
                if step % int(save_interval / self.dt) == 0:
                    snapshots.append({
                        'time': t,
                        'u': u.copy(),
                        'v': v.copy(),
                    })
        
        elapsed = time.time() - start_time
        logging.info(f"\n模擬完成！耗時: {elapsed:.1f}s")
        logging.info(f"總步數: {step}, 平均步數: {n_avg}")
        
        # 計算時間平均
        u_mean = u_sum / n_avg if n_avg > 0 else u
        v_mean = v_sum / n_avg if n_avg > 0 else v
        
        logging.info(f"時間平均場: KE_mean = {0.5*np.mean(u_mean**2 + v_mean**2):.6f}")
        
        return {
            'u_mean': u_mean,
            'v_mean': v_mean,
            'X': self.X,
            'Y': self.Y,
            'snapshots': snapshots,
            'stats_history': self.stats_history,
            'parameters': {
                'N': self.N,
                'L': self.L,
                'nu': self.nu,
                'A': self.A,
                'k_f': self.k_f,
                'dt': self.dt,
                'T_total': T_total,
                'T_spinup': T_spinup,
            }
        }


def save_lowfi_data(results: Dict, output_path: Path):
    """
    儲存 low-fidelity 資料（HDF5 格式）
    
    結構：
    - /mean_field/u, v, X, Y
    - /snapshots/snapshot_000/u, v, time
    - /statistics/time, KE, enstrophy
    - /parameters/*
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # 時間平均場
        mean_grp = f.create_group('mean_field')
        mean_grp.create_dataset('u', data=results['u_mean'])
        mean_grp.create_dataset('v', data=results['v_mean'])
        mean_grp.create_dataset('X', data=results['X'])
        mean_grp.create_dataset('Y', data=results['Y'])
        
        # Snapshots（可選）
        if results['snapshots']:
            snap_grp = f.create_group('snapshots')
            for i, snap in enumerate(results['snapshots']):
                sg = snap_grp.create_group(f'snapshot_{i:03d}')
                sg.create_dataset('u', data=snap['u'])
                sg.create_dataset('v', data=snap['v'])
                sg.attrs['time'] = snap['time']
        
        # 統計量
        stats_grp = f.create_group('statistics')
        for key, val in results['stats_history'].items():
            stats_grp.create_dataset(key, data=np.array(val))
        
        # 參數
        param_grp = f.create_group('parameters')
        for key, val in results['parameters'].items():
            param_grp.attrs[key] = val
    
    logging.info(f"✅ Low-fi 資料已儲存至: {output_path}")
    logging.info(f"   檔案大小: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description='生成 Kolmogorov Flow Low-Fidelity 資料')
    
    # 網格設定
    parser.add_argument('--N', type=int, default=32, 
                        help='網格點數（32 或 64，預設 32）')
    parser.add_argument('--L', type=float, default=2*np.pi, 
                        help='域大小（預設 2π）')
    
    # 物理參數
    parser.add_argument('--nu', type=float, default=0.02, 
                        help='黏滯係數（預設 0.02，hi-fi 約 0.01）')
    parser.add_argument('--A', type=float, default=1.0, 
                        help='強迫振幅（預設 1.0）')
    parser.add_argument('--k_f', type=int, default=4, 
                        help='強迫波數（預設 4）')
    
    # 時間設定
    parser.add_argument('--dt', type=float, default=0.001, 
                        help='時間步長（預設 0.001，較小以確保穩定）')
    parser.add_argument('--T_total', type=float, default=200.0, 
                        help='總模擬時間（預設 200）')
    parser.add_argument('--T_spinup', type=float, default=50.0, 
                        help='Spin-up 時間（預設 50）')
    parser.add_argument('--save_interval', type=float, default=5.0, 
                        help='Snapshot 存檔間隔（預設 5.0）')
    
    # 初始化
    parser.add_argument('--init_mode', type=str, default='random', 
                        choices=['random', 'laminar', 'vortices'],
                        help='初始化模式（預設 random）')
    
    # 輸出
    parser.add_argument('--output', type=str, 
                        default='data/kolmogorov_lowfi/lowfi_N32_nu0.02.h5',
                        help='輸出檔案路徑')
    
    args = parser.parse_args()
    
    # 創建求解器
    solver = KolmogorovFlowLowFi(
        N=args.N,
        L=args.L,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
        dealias=True,
    )
    
    # 執行模擬
    results = solver.run_simulation(
        T_total=args.T_total,
        T_spinup=args.T_spinup,
        save_interval=args.save_interval,
        init_mode=args.init_mode,
    )
    
    # 儲存結果
    output_path = Path(args.output)
    save_lowfi_data(results, output_path)
    
    logging.info("\n" + "=" * 60)
    logging.info("Low-Fi 資料生成完成！")
    logging.info(f"網格: {args.N}x{args.N}")
    logging.info(f"Re = {np.sqrt(args.A) * (2*np.pi/args.k_f)**(3/2) / args.nu:.2f}")
    logging.info(f"輸出: {output_path}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main()
