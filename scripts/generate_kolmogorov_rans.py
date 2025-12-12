"""
真正的 RANS-k-ε 求解器 for Kolmogorov Flow
===========================================

實作標準 k-ε 湍流模型 + Reynolds-Averaged Navier-Stokes 方程

理論基礎：
---------
1. RANS 動量方程：
   ∂<u>/∂t + <u>·∇<u> = -∇<p>/ρ + ∇·[(ν + ν_t)∇<u>] + F
   
2. k-ε 湍流閉合：
   ∂k/∂t + <u>·∇k = P_k - ε + ∇·[(ν + ν_t/σ_k)∇k]
   ∂ε/∂t + <u>·∇ε = (C_1ε·P_k - C_2ε·ε)·ε/k + ∇·[(ν + ν_t/σ_ε)∇ε]
   
3. 渦黏滯：
   ν_t = C_μ · k²/ε
   
4. 生產項：
   P_k = ν_t · S:S (S = 應變率張量)

數值方法：
---------
- Pseudo-spectral (Fourier) for spatial derivatives
- Semi-implicit first-order Euler for time integration
  (擴散項用 integrating factor 隱式處理，對流項顯式 Euler)
- Positivity-preserving for k, ε
- Spatial-averaged eddy viscosity approximation (ν+ν_t → ⟨ν+ν_t⟩)

參考文獻：
---------
- Launder & Spalding (1974), "The numerical computation of turbulent flows"
- Pope (2000), "Turbulent Flows" (Chapter 10)
- Wilcox (2006), "Turbulence Modeling for CFD"

作者：PINNs-MVP 團隊
日期：2025-12-11
"""

import numpy as np
import argparse
from pathlib import Path
import h5py
from typing import Dict, Tuple, Optional
import logging
import time

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class KolmogorovRANS:
    """
    2D Kolmogorov Flow RANS-k-ε 求解器
    
    求解時間平均的NS方程 + k-ε湍流模型
    """
    
    # k-ε 模型標準常數 (Launder & Spalding, 1974)
    C_MU = 0.09      # 渦黏滯係數
    C_1E = 1.44      # ε 方程生產項係數
    C_2E = 1.92      # ε 方程耗散項係數
    SIGMA_K = 1.0    # k 擴散項係數
    SIGMA_E = 1.3    # ε 擴散項係數
    
    def __init__(
        self,
        N: int = 64,
        L: float = 2 * np.pi,
        nu: float = 0.02,      # 分子黏滯
        A: float = 1.0,
        k_f: int = 4,
        dt: float = 0.005,
        dealias: bool = True,
    ):
        """
        Args:
            N: 網格點數（粗網格，如 64）
            L: 域大小
            nu: 分子運動黏滯係數
            A: Kolmogorov 強迫振幅
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
        
        # 網格
        self.dx = L / N
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
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
            'turbulent_kinetic_energy': [],
            'dissipation_rate': [],
            'nu_t_mean': [],
        }
        
        # 計算基於分子黏滯的雷諾數（參考值）
        Re_molecular = np.sqrt(A) * (2*np.pi/k_f)**(1.5) / nu
        
        logging.info("=" * 70)
        logging.info("RANS-k-ε Kolmogorov Flow 求解器")
        logging.info("=" * 70)
        logging.info(f"網格: {N}×{N}, 域: [{0:.2f}, {L:.2f}]×[{0:.2f}, {L:.2f}]")
        logging.info(f"網格間距: Δx = Δy = {self.dx:.6f}")
        logging.info(f"分子黏滯: ν = {nu:.6f}")
        logging.info(f"參考雷諾數 (基於 ν): Re ≈ {Re_molecular:.1f}")
        logging.info(f"強迫: A = {A:.4f}, k_f = {k_f}")
        logging.info(f"時間步長: dt = {dt:.6f}")
        logging.info(f"k-ε 模型常數: C_μ={self.C_MU}, C_1ε={self.C_1E}, C_2ε={self.C_2E}")
        logging.info("=" * 70)
    
    def initialize_fields(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        初始化場：<u>, <v>, k, ε
        
        Returns:
            u, v, k, epsilon (物理空間)
        """
        # 時均速度：從層流解 + 小擾動開始
        u = 0.1 * np.sin(self.k_f * self.Y) + 0.01 * np.random.randn(self.N, self.N)
        v = 0.01 * np.random.randn(self.N, self.N)
        
        # 確保無散度
        u, v = self._project_divergence_free(u, v)
        
        # 湍動能 k：初始化為小正值（避免零）
        k = np.ones((self.N, self.N)) * 1e-3 + 1e-4 * np.random.rand(self.N, self.N)
        
        # 耗散率 ε：用 ε = k^(3/2) / L_turb 估計（L_turb ≈ L/10）
        L_turb = self.L / 10.0
        epsilon = k**(1.5) / L_turb
        
        logging.info("\n初始場統計:")
        logging.info(f"  u: [{u.min():.4f}, {u.max():.4f}], std={u.std():.4f}")
        logging.info(f"  v: [{v.min():.4f}, {v.max():.4f}], std={v.std():.4f}")
        logging.info(f"  k: [{k.min():.6f}, {k.max():.6f}], mean={k.mean():.6f}")
        logging.info(f"  ε: [{epsilon.min():.6f}, {epsilon.max():.6f}], mean={epsilon.mean():.6f}")
        
        return u, v, k, epsilon
    
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
    
    def compute_eddy_viscosity(self, k: np.ndarray, epsilon: np.ndarray) -> np.ndarray:
        """
        計算渦黏滯：ν_t = C_μ · k²/ε
        
        加入數值穩定性處理
        """
        # 確保 k, ε > 0
        k = np.maximum(k, 1e-10)
        epsilon = np.maximum(epsilon, 1e-10)
        
        nu_t = self.C_MU * k**2 / epsilon
        
        # 限制渦黏滯上限（防止數值爆炸）
        nu_t_max = 100.0 * self.nu  # 渦黏滯 ≤ 100×分子黏滯
        nu_t = np.minimum(nu_t, nu_t_max)
        
        return nu_t
    
    def compute_strain_rate_tensor(self, u_hat: np.ndarray, v_hat: np.ndarray) -> Dict[str, np.ndarray]:
        """
        計算應變率張量 S_ij = 0.5*(∂u_i/∂x_j + ∂u_j/∂x_i)
        
        Returns:
            {'S_xx', 'S_yy', 'S_xy', 'S_mag'}: 物理空間
        """
        # 梯度（頻譜空間 → 物理空間）
        u_x = np.real(np.fft.ifft2(1j * self.KX * u_hat))
        u_y = np.real(np.fft.ifft2(1j * self.KY * u_hat))
        v_x = np.real(np.fft.ifft2(1j * self.KX * v_hat))
        v_y = np.real(np.fft.ifft2(1j * self.KY * v_hat))
        
        # 應變率張量（對稱部分）
        S_xx = u_x
        S_yy = v_y
        S_xy = 0.5 * (u_y + v_x)
        
        # 應變率大小：|S| = √(2·S_ij·S_ij)
        S_mag = np.sqrt(2 * (S_xx**2 + S_yy**2 + 2 * S_xy**2))
        
        return {'S_xx': S_xx, 'S_yy': S_yy, 'S_xy': S_xy, 'S_mag': S_mag}
    
    def compute_production(self, nu_t: np.ndarray, S_mag: np.ndarray, 
                          epsilon: np.ndarray = None) -> np.ndarray:
        """
        計算湍動能生產項：P_k = ν_t · |S|²
        
        Args:
            nu_t: 渦黏滯
            S_mag: 應變率大小
            epsilon: 耗散率（若提供，用於限制 P_k 上限）
        """
        P_k = nu_t * S_mag**2
        
        # 限制生產項（數值穩定性）
        # 改進：使用與局部耗散率成比例的上限，而非硬編碼常數
        if epsilon is not None:
            # P_k ≤ C·ε，C=10 是經驗值（允許局部強生產區域）
            P_k_max = 10.0 * epsilon
            P_k = np.minimum(P_k, P_k_max)
        
        return P_k
    
    def step_momentum_rans(
        self,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
        nu_t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        RANS 動量方程時間步進（半隱式）
        
        ∂<u>/∂t + <u>·∇<u> = -∇<p> + ∇·[(ν+ν_t)∇<u>] + F
        
        黏滯項（ν+ν_t）使用半隱式處理
        """
        # 轉回物理空間
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        
        # 對流項（去混疊）
        u_x = np.real(np.fft.ifft2(1j * self.KX * u_hat))
        u_y = np.real(np.fft.ifft2(1j * self.KY * u_hat))
        v_x = np.real(np.fft.ifft2(1j * self.KX * v_hat))
        v_y = np.real(np.fft.ifft2(1j * self.KY * v_hat))
        
        N_u = -(u * u_x + v * u_y)
        N_v = -(u * v_x + v * v_y)
        
        N_u_hat = np.fft.fft2(N_u) * self.dealias_mask
        N_v_hat = np.fft.fft2(N_v) * self.dealias_mask
        
        # 強迫項
        F_u_hat = np.fft.fft2(self.forcing)
        
        # 有效黏滯：ν_eff = ν + ν_t（頻譜空間）
        nu_eff = self.nu + nu_t
        nu_eff_hat = np.fft.fft2(nu_eff)
        
        # 半隱式積分因子（使用平均黏滯）
        nu_eff_mean = nu_eff.mean()
        integrating_factor = np.exp(-nu_eff_mean * self.K2 * self.dt)
        
        # 更新（顯式對流 + 半隱式擴散）
        u_hat_new = integrating_factor * u_hat + self.dt * (N_u_hat + F_u_hat)
        v_hat_new = integrating_factor * v_hat + self.dt * N_v_hat
        
        # 投影到無散度（壓力 Poisson 求解）
        div_hat = 1j * (self.KX * u_hat_new + self.KY * v_hat_new)
        phi_hat = -div_hat / self.K2
        phi_hat[0, 0] = 0
        
        u_hat_new -= 1j * self.KX * phi_hat
        v_hat_new -= 1j * self.KY * phi_hat
        
        return u_hat_new, v_hat_new
    
    def step_k_epsilon(
        self,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
        k: np.ndarray,
        epsilon: np.ndarray,
        nu_t: np.ndarray,
        P_k: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        k-ε 方程時間步進（半隱式）
        
        ∂k/∂t + <u>·∇k = P_k - ε + ∇·[(ν+ν_t/σ_k)∇k]
        ∂ε/∂t + <u>·∇ε = C_1ε·(ε/k)·P_k - C_2ε·ε²/k + ∇·[(ν+ν_t/σ_ε)∇ε]
        """
        u = np.real(np.fft.ifft2(u_hat))
        v = np.real(np.fft.ifft2(v_hat))
        
        k_hat = np.fft.fft2(k)
        eps_hat = np.fft.fft2(epsilon)
        
        # === k 方程 ===
        
        # 對流項
        k_x = np.real(np.fft.ifft2(1j * self.KX * k_hat))
        k_y = np.real(np.fft.ifft2(1j * self.KY * k_hat))
        conv_k = -(u * k_x + v * k_y)
        conv_k_hat = np.fft.fft2(conv_k) * self.dealias_mask
        
        # 源項：P_k - ε
        source_k = P_k - epsilon
        source_k_hat = np.fft.fft2(source_k)
        
        # 擴散係數：ν + ν_t/σ_k
        nu_k = self.nu + nu_t / self.SIGMA_K
        nu_k_mean = nu_k.mean()
        
        # 半隱式擴散
        integrating_factor_k = np.exp(-nu_k_mean * self.K2 * self.dt)
        k_hat_new = integrating_factor_k * k_hat + self.dt * (conv_k_hat + source_k_hat)
        
        # === ε 方程 ===
        
        # 對流項
        eps_x = np.real(np.fft.ifft2(1j * self.KX * eps_hat))
        eps_y = np.real(np.fft.ifft2(1j * self.KY * eps_hat))
        conv_eps = -(u * eps_x + v * eps_y)
        conv_eps_hat = np.fft.fft2(conv_eps) * self.dealias_mask
        
        # 源項：C_1ε·(ε/k)·P_k - C_2ε·ε²/k
        k_safe = np.maximum(k, 1e-10)
        source_eps = (
            self.C_1E * (epsilon / k_safe) * P_k
            - self.C_2E * epsilon**2 / k_safe
        )
        source_eps_hat = np.fft.fft2(source_eps)
        
        # 擴散係數：ν + ν_t/σ_ε
        nu_eps = self.nu + nu_t / self.SIGMA_E
        nu_eps_mean = nu_eps.mean()
        
        # 半隱式擴散
        integrating_factor_eps = np.exp(-nu_eps_mean * self.K2 * self.dt)
        eps_hat_new = integrating_factor_eps * eps_hat + self.dt * (conv_eps_hat + source_eps_hat)
        
        # 轉回物理空間並強制正值
        k_new = np.real(np.fft.ifft2(k_hat_new))
        eps_new = np.real(np.fft.ifft2(eps_hat_new))
        
        k_new = np.maximum(k_new, 1e-10)      # k > 0
        eps_new = np.maximum(eps_new, 1e-10)  # ε > 0
        
        return k_new, eps_new
    
    def compute_statistics(
        self,
        u: np.ndarray,
        v: np.ndarray,
        k: np.ndarray,
        epsilon: np.ndarray,
        nu_t: np.ndarray
    ) -> Dict[str, float]:
        """計算統計量"""
        # 時均動能
        KE_mean = 0.5 * np.mean(u**2 + v**2)
        
        # 湍動能
        k_mean = np.mean(k)
        
        # 耗散率
        eps_mean = np.mean(epsilon)
        
        # 渦黏滯
        nu_t_mean = np.mean(nu_t)
        
        return {
            'kinetic_energy': KE_mean,
            'turbulent_kinetic_energy': k_mean,
            'dissipation_rate': eps_mean,
            'nu_t_mean': nu_t_mean,
        }
    
    def run_simulation(
        self,
        T_total: float = 200.0,
        T_spinup: float = 50.0,
        save_interval: float = 1.0,
        log_interval: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        執行 RANS 模擬
        
        Args:
            T_total: 總時間
            T_spinup: Spin-up 時間（統計穩態前的過渡期）
            save_interval: 保存間隔
            log_interval: 日誌輸出間隔（步數）
        
        Returns:
            包含時間平均場的字典
        """
        logging.info("\n開始 RANS-k-ε 模擬...")
        logging.info(f"總時間: {T_total:.1f}, Spin-up: {T_spinup:.1f}")
        
        # 初始化
        u, v, k, epsilon = self.initialize_fields()
        u_hat = np.fft.fft2(u)
        v_hat = np.fft.fft2(v)
        
        # 時間積分
        t = 0.0
        step = 0
        
        # 時間平均累加器
        u_sum = np.zeros_like(u)
        v_sum = np.zeros_like(v)
        k_sum = np.zeros_like(k)
        eps_sum = np.zeros_like(epsilon)
        nu_t_sum = np.zeros_like(u)
        n_avg = 0
        
        start_time = time.time()
        
        while t < T_total:
            # 計算渦黏滯
            nu_t = self.compute_eddy_viscosity(k, epsilon)
            
            # 計算應變率與生產項（傳入 epsilon 以限制 P_k 上限）
            strain = self.compute_strain_rate_tensor(u_hat, v_hat)
            P_k = self.compute_production(nu_t, strain['S_mag'], epsilon)
            
            # 更新動量方程
            u_hat, v_hat = self.step_momentum_rans(u_hat, v_hat, nu_t)
            
            # 更新 k-ε 方程
            k, epsilon = self.step_k_epsilon(u_hat, v_hat, k, epsilon, nu_t, P_k)
            
            t += self.dt
            step += 1
            
            # 統計量
            if step % log_interval == 0:
                u = np.real(np.fft.ifft2(u_hat))
                v = np.real(np.fft.ifft2(v_hat))
                stats = self.compute_statistics(u, v, k, epsilon, nu_t)
                
                self.stats_history['time'].append(t)
                self.stats_history['kinetic_energy'].append(stats['kinetic_energy'])
                self.stats_history['turbulent_kinetic_energy'].append(stats['turbulent_kinetic_energy'])
                self.stats_history['dissipation_rate'].append(stats['dissipation_rate'])
                self.stats_history['nu_t_mean'].append(stats['nu_t_mean'])
                
                if step % (log_interval * 10) == 0:
                    logging.info(
                        f"Step {step:6d}, t={t:7.2f}, "
                        f"KE={stats['kinetic_energy']:.5f}, "
                        f"k={stats['turbulent_kinetic_energy']:.5f}, "
                        f"ε={stats['dissipation_rate']:.5f}, "
                        f"ν_t/ν={stats['nu_t_mean']/self.nu:.2f}"
                    )
            
            # 時間平均（spin-up 後）
            if t > T_spinup:
                u = np.real(np.fft.ifft2(u_hat))
                v = np.real(np.fft.ifft2(v_hat))
                
                u_sum += u
                v_sum += v
                k_sum += k
                eps_sum += epsilon
                nu_t_sum += nu_t
                n_avg += 1
        
        elapsed = time.time() - start_time
        
        logging.info(f"\n模擬完成！耗時: {elapsed:.1f}s")
        logging.info(f"總步數: {step}, 平均步數: {n_avg}")
        
        # 計算時間平均
        if n_avg > 0:
            u_mean = u_sum / n_avg
            v_mean = v_sum / n_avg
            k_mean = k_sum / n_avg
            eps_mean = eps_sum / n_avg
            nu_t_mean = nu_t_sum / n_avg
            
            KE_mean = 0.5 * np.mean(u_mean**2 + v_mean**2)
            
            logging.info(f"時間平均場:")
            logging.info(f"  <KE> = {KE_mean:.6f}")
            logging.info(f"  <k>  = {k_mean.mean():.6f}")
            logging.info(f"  <ε>  = {eps_mean.mean():.6f}")
            logging.info(f"  <ν_t>/<ν> = {nu_t_mean.mean()/self.nu:.2f}")
        else:
            logging.warning("警告：未累積任何平均值（T_spinup 過長）")
            u_mean = v_mean = k_mean = eps_mean = nu_t_mean = np.zeros((self.N, self.N))
        
        return {
            'mean_field': {
                'u': u_mean,
                'v': v_mean,
                'k': k_mean,
                'epsilon': eps_mean,
                'nu_t': nu_t_mean,
                'X': self.X,
                'Y': self.Y,
            },
            'statistics': self.stats_history,
            'parameters': {
                'N': self.N,
                'L': self.L,
                'nu': self.nu,
                'A': self.A,
                'k_f': self.k_f,
                'dt': self.dt,
                'T_total': T_total,
                'T_spinup': T_spinup,
                'model': 'RANS-k-epsilon',
            }
        }


def save_rans_data(results: Dict, output_file: str):
    """保存 RANS 數據至 HDF5"""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f:
        # 時間平均場
        mean_grp = f.create_group('mean_field')
        for key, val in results['mean_field'].items():
            mean_grp.create_dataset(key, data=val)
        
        # 統計量時間序列
        stats_grp = f.create_group('statistics')
        for key, val in results['statistics'].items():
            stats_grp.create_dataset(key, data=np.array(val))
        
        # 參數
        param_grp = f.create_group('parameters')
        for key, val in results['parameters'].items():
            param_grp.attrs[key] = val
    
    file_size = Path(output_file).stat().st_size / 1024
    logging.info(f"✅ RANS 數據已保存: {output_file} ({file_size:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='RANS-k-ε Kolmogorov Flow 求解器')
    parser.add_argument('--N', type=int, default=64, help='網格點數（粗網格）')
    parser.add_argument('--nu', type=float, default=0.02, help='分子黏滯係數')
    parser.add_argument('--A', type=float, default=1.0, help='強迫振幅')
    parser.add_argument('--k_f', type=int, default=4, help='強迫波數')
    parser.add_argument('--dt', type=float, default=0.005, help='時間步長')
    parser.add_argument('--T_total', type=float, default=150.0, help='總模擬時間')
    parser.add_argument('--T_spinup', type=float, default=30.0, help='Spin-up 時間')
    parser.add_argument('--save_interval', type=float, default=1.0, help='保存間隔')
    parser.add_argument('--output', type=str, required=True, help='輸出 HDF5 文件')
    
    args = parser.parse_args()
    
    # 創建求解器
    solver = KolmogorovRANS(
        N=args.N,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
    )
    
    # 執行模擬
    results = solver.run_simulation(
        T_total=args.T_total,
        T_spinup=args.T_spinup,
        save_interval=args.save_interval,
    )
    
    # 保存結果
    save_rans_data(results, args.output)
    
    logging.info("\n" + "=" * 70)
    logging.info("RANS-k-ε 模擬完成！")
    logging.info(f"輸出: {args.output}")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
