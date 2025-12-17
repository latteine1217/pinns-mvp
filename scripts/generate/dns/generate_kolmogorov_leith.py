"""
2D Kolmogorov Flow with Leith Turbulence Model
===============================================

實作 Leith (1996) 渦黏滯模型用於 2D 湍流

理論基礎：
---------
1. RANS 動量方程：
   ∂<u>/∂t + <u>·∇<u> = -∇<p>/ρ + ∇·[(ν + ν_t)∇<u>] + F
   
2. Leith 渦黏滯閉合：
   ν_t = (C_L Δ)³ |∇ω|
   
   其中：
   - C_L ≈ 0.1-0.3 (經驗常數，可調)
   - Δ = √(Δx Δy) (網格尺度)
   - ω = ∂v/∂x - ∂u/∂y (渦度)
   - |∇ω| = √((∂ω/∂x)² + (∂ω/∂y)²)

優勢：
-----
- 專為 2D 湍流設計（考慮逆級聯）
- 無需額外輸運方程（純診斷式）
- 基於渦度梯度，適合旋轉主導流動
- 比 k-ε 計算成本低

數值方法：
---------
- Pseudo-spectral (Fourier) for spatial derivatives
- Semi-implicit time integration (RK2 or Euler)
- Eddy viscosity computed diagnostically from vorticity gradient

參考文獻：
---------
- Leith (1996), "Stochastic backscatter in a subgrid-scale model"
- Boffetta & Ecke (2012), "Two-Dimensional Turbulence"
- Fox-Kemper & Menemenlis (2008), "Can large eddy simulation techniques improve mesoscale rich ocean models?"

作者：PINNs-MVP 團隊
日期：2025-12-17
"""

import numpy as np
import argparse
from pathlib import Path
import h5py
from typing import Dict, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class KolmogorovLeith:
    """
    2D Kolmogorov Flow with Leith Turbulence Model
    
    求解時間平均的 NS 方程 + Leith 渦黏滯閉合
    """
    
    def __init__(
        self,
        N: int = 128,
        L: float = 2 * np.pi,
        nu: float = 0.02,          # 分子黏滯
        A: float = 1.0,
        k_f: int = 4,
        dt: float = 0.005,
        C_L: float = 0.2,          # Leith 常數（可調）
        dealias: bool = True,
    ):
        """
        Args:
            N: 網格點數
            L: 域大小
            nu: 分子運動黏滯係數
            A: Kolmogorov 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            C_L: Leith 模型常數（典型範圍 0.1-0.3）
            dealias: 是否啟用 2/3 去混疊
        """
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.C_L = C_L
        self.dealias = dealias
        
        # 網格
        self.dx = L / N
        self.dy = L / N
        self.delta = np.sqrt(self.dx * self.dy)  # Leith 長度尺度
        
        self.x = np.linspace(0, L, N, endpoint=False)
        self.y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')
        
        # 頻譜空間波數
        kx = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        ky = np.fft.fftfreq(N, d=1/N) * 2 * np.pi / L
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0  # 避免除零
        
        # Dealiasing mask
        if dealias:
            k_max = N // 3
            self.dealias_mask = (np.abs(self.KX) <= k_max) & (np.abs(self.KY) <= k_max)
        else:
            self.dealias_mask = np.ones((N, N), dtype=bool)
        
        # Kolmogorov 強迫項（僅作用於動量方程 x 方向）
        self.forcing = self.A * np.sin(self.k_f * self.Y)
        
        # 統計量記錄
        self.stats_history = {
            'time': [],
            'kinetic_energy': [],
            'enstrophy': [],
            'nu_t_mean': [],
            'nu_t_max': [],
        }
        
        # 計算基於分子黏滯的雷諾數（參考值）
        Re_molecular = np.sqrt(A) * (2*np.pi/k_f)**(1.5) / nu
        
        logging.info("=" * 70)
        logging.info("Leith Turbulence Model for Kolmogorov Flow")
        logging.info("=" * 70)
        logging.info(f"網格: {N}×{N}, 域: [{0:.2f}, {L:.2f}]×[{0:.2f}, {L:.2f}]")
        logging.info(f"網格間距: Δx = Δy = {self.dx:.6f}, Δ = {self.delta:.6f}")
        logging.info(f"分子黏滯: ν = {nu:.6f}")
        logging.info(f"參考雷諾數 (基於 ν): Re ≈ {Re_molecular:.1f}")
        logging.info(f"強迫: A = {A:.4f}, k_f = {k_f}")
        logging.info(f"時間步長: dt = {dt:.6f}")
        logging.info(f"Leith 常數: C_L = {C_L:.3f}")
        logging.info("=" * 70)
    
    def initialize_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        初始化速度場
        
        Returns:
            u, v (物理空間)
        """
        # 時均速度：從層流解 + 小擾動開始
        u = 0.1 * np.sin(self.k_f * self.Y) + 0.01 * np.random.randn(self.N, self.N)
        v = 0.01 * np.random.randn(self.N, self.N)
        
        # 確保無散度
        u, v = self._project_divergence_free(u, v)
        
        logging.info("\n初始場統計:")
        logging.info(f"  u: [{u.min():.4f}, {u.max():.4f}], std={u.std():.4f}")
        logging.info(f"  v: [{v.min():.4f}, {v.max():.4f}], std={v.std():.4f}")
        
        return u, v
    
    def _project_divergence_free(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """投影到無散度場"""
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        div_hat = 1j * (self.KX * u_hat + self.KY * v_hat)
        phi_hat = -div_hat / self.K2
        phi_hat[0, 0] = 0
        
        u_hat -= 1j * self.KX * phi_hat
        v_hat -= 1j * self.KY * phi_hat
        
        return np.real(np.fft.ifft2(u_hat)), np.real(np.fft.ifft2(v_hat))
    
    def compute_vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        計算渦度：ω = ∂v/∂x - ∂u/∂y
        
        使用 Fourier 譜方法求導
        """
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        omega_hat = 1j * (self.KX * v_hat - self.KY * u_hat)
        omega = np.real(np.fft.ifft2(omega_hat))
        
        return omega
    
    def compute_vorticity_gradient_magnitude(self, omega: np.ndarray) -> np.ndarray:
        """
        計算渦度梯度的大小：|∇ω| = √((∂ω/∂x)² + (∂ω/∂y)²)
        
        使用 Fourier 譜方法求導
        """
        omega_hat = np.fft.fft2(omega)
        
        domega_dx_hat = 1j * self.KX * omega_hat
        domega_dy_hat = 1j * self.KY * omega_hat
        
        domega_dx = np.real(np.fft.ifft2(domega_dx_hat))
        domega_dy = np.real(np.fft.ifft2(domega_dy_hat))
        
        grad_omega_mag = np.sqrt(domega_dx**2 + domega_dy**2)
        
        return grad_omega_mag
    
    def compute_eddy_viscosity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        計算 Leith 渦黏滯：ν_t = (C_L Δ)³ |∇ω|
        
        Args:
            u, v: 速度場
            
        Returns:
            nu_t: 渦黏滯場
        """
        # 1. 計算渦度
        omega = self.compute_vorticity(u, v)
        
        # 2. 計算渦度梯度大小
        grad_omega_mag = self.compute_vorticity_gradient_magnitude(omega)
        
        # 3. Leith 渦黏滯公式
        nu_t = (self.C_L * self.delta)**3 * grad_omega_mag
        
        # 4. 限制上限（數值穩定性）
        nu_t_max = 100.0 * self.nu
        nu_t = np.minimum(nu_t, nu_t_max)
        
        return nu_t
    
    def compute_rhs(
        self, 
        u: np.ndarray, 
        v: np.ndarray,
        nu_eff: float,  # 有效黏滯（ν + ⟨ν_t⟩）
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算 RANS 方程右側
        
        RHS_u = -<u>·∇<u> + ν_eff ∇²<u> + F
        RHS_v = -<u>·∇<v> + ν_eff ∇²<v>
        
        注：使用空間平均的 ν_eff 簡化計算
        """
        # FFT 速度場
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        # 1. 對流項（物理空間計算，去混疊）
        dudx = np.real(np.fft.ifft2(1j * self.KX * u_hat))
        dudy = np.real(np.fft.ifft2(1j * self.KY * u_hat))
        dvdx = np.real(np.fft.ifft2(1j * self.KX * v_hat))
        dvdy = np.real(np.fft.ifft2(1j * self.KY * v_hat))
        
        conv_u = -(u * dudx + v * dudy)
        conv_v = -(u * dvdx + v * dvdy)
        
        if self.dealias:
            conv_u_hat = np.fft.fft2(conv_u) * self.dealias_mask
            conv_v_hat = np.fft.fft2(conv_v) * self.dealias_mask
        else:
            conv_u_hat = np.fft.fft2(conv_u)
            conv_v_hat = np.fft.fft2(conv_v)
        
        # 2. 擴散項（頻譜空間）
        diff_u_hat = -nu_eff * self.K2 * u_hat
        diff_v_hat = -nu_eff * self.K2 * v_hat
        
        # 3. 強迫項
        forcing_hat = np.fft.fft2(self.forcing)
        
        # 組合
        rhs_u_hat = conv_u_hat + diff_u_hat + forcing_hat
        rhs_v_hat = conv_v_hat + diff_v_hat
        
        rhs_u = np.real(np.fft.ifft2(rhs_u_hat))
        rhs_v = np.real(np.fft.ifft2(rhs_v_hat))
        
        return rhs_u, rhs_v
    
    def step(self, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        時間推進一步（RK2）
        
        RK2 = 二階 Runge-Kutta：
        k1 = f(u_n)
        k2 = f(u_n + dt * k1)
        u_{n+1} = u_n + dt/2 * (k1 + k2)
        """
        # 計算渦黏滯（診斷式）
        nu_t = self.compute_eddy_viscosity(u, v)
        nu_eff = self.nu + nu_t.mean()  # 空間平均簡化
        
        # RK2 Stage 1
        rhs_u1, rhs_v1 = self.compute_rhs(u, v, nu_eff)
        u_tmp = u + self.dt * rhs_u1
        v_tmp = v + self.dt * rhs_v1
        
        # 確保無散度
        u_tmp, v_tmp = self._project_divergence_free(u_tmp, v_tmp)
        
        # RK2 Stage 2
        nu_t_tmp = self.compute_eddy_viscosity(u_tmp, v_tmp)
        nu_eff_tmp = self.nu + nu_t_tmp.mean()
        rhs_u2, rhs_v2 = self.compute_rhs(u_tmp, v_tmp, nu_eff_tmp)
        
        # 組合
        u_new = u + 0.5 * self.dt * (rhs_u1 + rhs_u2)
        v_new = v + 0.5 * self.dt * (rhs_v1 + rhs_v2)
        
        # 確保無散度
        u_new, v_new = self._project_divergence_free(u_new, v_new)
        
        return u_new, v_new
    
    def compute_statistics(self, u: np.ndarray, v: np.ndarray, t: float) -> None:
        """記錄統計量"""
        KE = 0.5 * (u**2 + v**2).mean()
        omega = self.compute_vorticity(u, v)
        enstrophy = 0.5 * (omega**2).mean()
        nu_t = self.compute_eddy_viscosity(u, v)
        
        self.stats_history['time'].append(t)
        self.stats_history['kinetic_energy'].append(KE)
        self.stats_history['enstrophy'].append(enstrophy)
        self.stats_history['nu_t_mean'].append(nu_t.mean())
        self.stats_history['nu_t_max'].append(nu_t.max())
    
    def run(
        self, 
        T_total: float = 100.0,
        T_spinup: float = 10.0,
        save_interval: int = 1000,
    ) -> Dict[str, np.ndarray]:
        """
        執行 Leith 模擬
        
        Args:
            T_total: 總模擬時間
            T_spinup: Spin-up 時間（不記錄）
            save_interval: 每 N 步記錄一次統計
            
        Returns:
            平均場字典
        """
        logging.info("\n開始 Leith 模擬...")
        logging.info(f"總時間: {T_total}, Spin-up: {T_spinup}\n")
        
        # 初始化
        u, v = self.initialize_fields()
        
        n_steps = int(T_total / self.dt)
        n_spinup = int(T_spinup / self.dt)
        
        # Spin-up 階段
        logging.info("Spin-up 階段...")
        for step in range(n_spinup):
            u, v = self.step(u, v)
            
            if (step + 1) % save_interval == 0:
                t = (step + 1) * self.dt
                KE = 0.5 * (u**2 + v**2).mean()
                nu_t = self.compute_eddy_viscosity(u, v)
                omega = self.compute_vorticity(u, v)
                logging.info(
                    f"Step {step+1:6d}, t={t:6.2f}, KE={KE:.5f}, "
                    f"ω_rms={np.sqrt((omega**2).mean()):.5f}, "
                    f"ν_t/ν={nu_t.mean()/self.nu:.2f}"
                )
        
        logging.info(f"\n✅ Spin-up 完成 (t={T_spinup})\n")
        
        # 統計階段（累積平均場）
        logging.info("統計累積階段...")
        u_mean = np.zeros_like(u)
        v_mean = np.zeros_like(v)
        nu_t_mean = np.zeros_like(u)
        n_samples = 0
        
        for step in range(n_spinup, n_steps):
            u, v = self.step(u, v)
            
            # 累積平均
            u_mean += u
            v_mean += v
            nu_t = self.compute_eddy_viscosity(u, v)
            nu_t_mean += nu_t
            n_samples += 1
            
            if (step + 1) % save_interval == 0:
                t = (step + 1) * self.dt
                self.compute_statistics(u, v, t)
                KE = 0.5 * (u**2 + v**2).mean()
                omega = self.compute_vorticity(u, v)
                logging.info(
                    f"Step {step+1:6d}, t={t:6.2f}, KE={KE:.5f}, "
                    f"ω_rms={np.sqrt((omega**2).mean()):.5f}, "
                    f"ν_t/ν={nu_t.mean()/self.nu:.2f}"
                )
        
        # 時間平均
        u_mean /= n_samples
        v_mean /= n_samples
        nu_t_mean /= n_samples
        
        logging.info(f"\n✅ 模擬完成！")
        logging.info(f"總樣本數: {n_samples}")
        logging.info(f"\n最終平均場統計:")
        logging.info(f"  ⟨u⟩: [{u_mean.min():.4f}, {u_mean.max():.4f}], std={u_mean.std():.4f}")
        logging.info(f"  ⟨v⟩: [{v_mean.min():.4f}, {v_mean.max():.4f}], std={v_mean.std():.4f}")
        logging.info(f"  ⟨ν_t⟩: mean={nu_t_mean.mean():.6f}, max={nu_t_mean.max():.6f}")
        logging.info(f"  ⟨ν_t⟩/ν: {nu_t_mean.mean()/self.nu:.3f}")
        
        return {
            'u': u_mean,
            'v': v_mean,
            'nu_t': nu_t_mean,
            'x': self.x,  # 1D arrays for HDF5
            'y': self.y,
        }
    
    def save(self, output_file: Path, mean_fields: Dict[str, np.ndarray]) -> None:
        """保存結果到 HDF5"""
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_file, 'w') as f:
            # 平均場
            mean_grp = f.create_group('mean_field')
            mean_grp.create_dataset('u', data=mean_fields['u'])
            mean_grp.create_dataset('v', data=mean_fields['v'])
            mean_grp.create_dataset('nu_t', data=mean_fields['nu_t'])
            mean_grp.create_dataset('x', data=mean_fields['x'])
            mean_grp.create_dataset('y', data=mean_fields['y'])
            
            # 統計歷史
            stats_grp = f.create_group('statistics')
            for key, val in self.stats_history.items():
                stats_grp.create_dataset(key, data=np.array(val))
            
            # 元數據
            meta_grp = f.create_group('metadata')
            meta_grp.attrs['N'] = self.N
            meta_grp.attrs['L'] = self.L
            meta_grp.attrs['nu'] = self.nu
            meta_grp.attrs['A'] = self.A
            meta_grp.attrs['k_f'] = self.k_f
            meta_grp.attrs['dt'] = self.dt
            meta_grp.attrs['C_L'] = self.C_L
            meta_grp.attrs['model'] = 'Leith'
        
        file_size_mb = output_file.stat().st_size / 1024 / 1024
        logging.info(f"✅ Leith 數據已保存: {output_file} ({file_size_mb:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description='2D Kolmogorov Flow with Leith Model')
    
    # 網格與域
    parser.add_argument('--N', type=int, default=128, help='網格點數')
    parser.add_argument('--L', type=float, default=2*np.pi, help='域大小')
    
    # 物理參數
    parser.add_argument('--nu', type=float, required=True, help='分子黏滯係數')
    parser.add_argument('--A', type=float, default=1.0, help='強迫振幅')
    parser.add_argument('--k_f', type=int, default=4, help='強迫波數')
    
    # Leith 模型
    parser.add_argument('--C_L', type=float, default=0.2, help='Leith 常數 (0.1-0.3)')
    
    # 時間積分
    parser.add_argument('--dt', type=float, default=0.005, help='時間步長')
    parser.add_argument('--T_total', type=float, default=100.0, help='總模擬時間')
    parser.add_argument('--T_spinup', type=float, default=10.0, help='Spin-up 時間')
    
    # 輸出
    parser.add_argument('--output', type=str, required=True, help='輸出 HDF5 檔案路徑')
    parser.add_argument('--save_interval', type=int, default=1000, help='統計記錄間隔')
    
    args = parser.parse_args()
    
    # 建立求解器
    solver = KolmogorovLeith(
        N=args.N,
        L=args.L,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
        C_L=args.C_L,
        dealias=True,
    )
    
    # 執行模擬
    t_start = time.time()
    mean_fields = solver.run(
        T_total=args.T_total,
        T_spinup=args.T_spinup,
        save_interval=args.save_interval,
    )
    t_elapsed = time.time() - t_start
    
    logging.info(f"\n⏱️  總計算時間: {t_elapsed:.1f} 秒 ({t_elapsed/60:.1f} 分鐘)")
    
    # 保存結果
    solver.save(Path(args.output), mean_fields)
    
    logging.info("\n" + "="*70)
    logging.info("✅ Leith 模擬完成！")
    logging.info("="*70)


if __name__ == '__main__':
    main()
