"""
Kolmogorov Flow Re=30 長時間 DNS 模擬（穩態檢測版）
===================================================

目標：
1. 運行足夠長時間以達到統計穩態
2. 監測動能 (KE) 和渦度 (Enstrophy) 的時間演化
3. 自動檢測穩態條件：相對變化率 < 閾值
4. 生成診斷圖表（KE/Enstrophy 時間序列、場快照）

物理參數（Re=30 時空混沌態）：
- 強迫：F₀ = Re × ν² × k³ = 30 × 0.02² × 4³ = 0.768
- 黏度：ν = 0.02
- 域大小：4π × 2π（延長域以捕捉更多渦旋）
- 預期現象：時空混沌、局域混沌斑

作者：PINNs-MVP 團隊
日期：2025-11-21
"""

import numpy as np
import argparse
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


class KolmogorovFlowDNS:
    """
    2D Kolmogorov Flow DNS 求解器（Pseudo-spectral 方法）
    """

    def __init__(
        self,
        Nx: int = 512,
        Ny: int = 256,
        Lx: float = 4 * np.pi,
        Ly: float = 2 * np.pi,
        nu: float = 0.02,
        A: float = 0.768,
        k_f: int = 4,
        dt: float = 0.001,
        dealias: bool = True,
    ):
        """
        Args:
            Nx, Ny: 網格點數（x, y 方向）
            Lx, Ly: 域大小
            nu: 動力黏度
            A: 強迫振幅
            k_f: 強迫波數
            dt: 時間步長
            dealias: 是否啟用 2/3 去混疊
        """
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.dealias = dealias

        # 空間網格（實空間）
        self.x = np.linspace(0, Lx, Nx, endpoint=False)
        self.y = np.linspace(0, Ly, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='ij')

        # 波數網格（Fourier 空間）
        kx = np.fft.fftfreq(Nx, d=Lx/Nx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny, d=Ly/Ny) * 2 * np.pi
        self.kx = kx[:, np.newaxis]
        self.ky = ky[np.newaxis, :]

        # 去混疊掩碼（2/3 規則）
        if dealias:
            kx_max = (2.0 / 3.0) * (Nx // 2)
            ky_max = (2.0 / 3.0) * (Ny // 2)
            self.dealias_mask = (np.abs(self.kx) <= kx_max) & (np.abs(self.ky) <= ky_max)
        else:
            self.dealias_mask = np.ones((Nx, Ny), dtype=bool)

        # 強迫項（實空間）
        self.forcing = A * np.sin(k_f * self.Y)

        # Laplacian 算子（Fourier 空間）
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0, 0] = 1.0  # 避免除零

        # 初始化流場
        self.u = None
        self.v = None
        self.u_hat = None
        self.v_hat = None
        self.p_hat = None
        self.time = 0.0

        # 估算雷諾數（基於強迫波數的特徵尺度）
        # Re 會隨流場發展而變化，這裡只是初始估算
        L_char = 2 * np.pi / k_f  # 特徵長度 = 2π/k_f
        U_char_estimated = A  # 粗略估算：強迫振幅作為速度尺度
        Re_estimated = U_char_estimated * L_char / nu
        
        logging.info(f"✅ DNS 求解器初始化完成")
        logging.info(f"   網格：{Nx}×{Ny}, 域大小：{Lx:.2f}×{Ly:.2f}")
        logging.info(f"   物理參數：ν={nu:.2e}, A={A:.4f}, k_f={k_f}")
        logging.info(f"   估算雷諾數：Re ≈ {Re_estimated:.1f} (基於 A*L/ν, 實際 Re 由流場決定)")
        logging.info(f"   時間步長：dt={dt:.4f}, 去混疊：{dealias}")

    def initialize_random_perturbation(self, amplitude: float = 0.1):
        """初始化隨機擾動流場（不可壓縮）"""
        # 產生隨機渦度場（Fourier 空間）
        omega_hat = amplitude * (
            np.random.randn(self.Nx, self.Ny) + 1j * np.random.randn(self.Nx, self.Ny)
        )
        omega_hat *= self.dealias_mask
        omega_hat[0, 0] = 0.0

        # 從渦度計算流函數與速度
        psi_hat = -omega_hat / self.k2
        psi_hat[0, 0] = 0.0

        self.u_hat = 1j * self.ky * psi_hat
        self.v_hat = -1j * self.kx * psi_hat

        self.u = np.fft.ifft2(self.u_hat).real
        self.v = np.fft.ifft2(self.v_hat).real

        logging.info(f"✅ 初始化隨機擾動流場（振幅={amplitude:.2f}）")

    def compute_nonlinear_terms(self) -> Tuple[np.ndarray, np.ndarray]:
        """計算非線性項（對流項）"""
        uu = self.u * self.u
        uv = self.u * self.v
        vv = self.v * self.v

        uu_hat = np.fft.fft2(uu)
        uv_hat = np.fft.fft2(uv)
        vv_hat = np.fft.fft2(vv)

        N_u_hat = -(1j * self.kx * uu_hat + 1j * self.ky * uv_hat)
        N_v_hat = -(1j * self.kx * uv_hat + 1j * self.ky * vv_hat)

        N_u_hat *= self.dealias_mask
        N_v_hat *= self.dealias_mask

        return N_u_hat, N_v_hat

    def compute_pressure(self, N_u_hat: np.ndarray, N_v_hat: np.ndarray, F_u_hat: np.ndarray) -> np.ndarray:
        """從速度場計算壓力場（投影法）"""
        viscous_u_hat = -self.nu * self.k2 * self.u_hat
        viscous_v_hat = -self.nu * self.k2 * self.v_hat

        rhs_u = N_u_hat + viscous_u_hat + F_u_hat
        rhs_v = N_v_hat + viscous_v_hat

        div_rhs = 1j * self.kx * rhs_u + 1j * self.ky * rhs_v
        p_hat = -div_rhs / self.k2
        p_hat[0, 0] = 0.0

        return p_hat

    def step_rk4(self):
        """RK4 時間積分一步"""
        F_u_hat = np.fft.fft2(self.forcing)

        # === Stage 1 ===
        N_u_hat1, N_v_hat1 = self.compute_nonlinear_terms()
        k1_u = N_u_hat1 - self.nu * self.k2 * self.u_hat + F_u_hat
        k1_v = N_v_hat1 - self.nu * self.k2 * self.v_hat

        # === Stage 2 ===
        self.u_hat_temp = self.u_hat + 0.5 * self.dt * k1_u
        self.v_hat_temp = self.v_hat + 0.5 * self.dt * k1_v
        self.u = np.fft.ifft2(self.u_hat_temp).real
        self.v = np.fft.ifft2(self.v_hat_temp).real

        N_u_hat2, N_v_hat2 = self.compute_nonlinear_terms()
        k2_u = N_u_hat2 - self.nu * self.k2 * self.u_hat_temp + F_u_hat
        k2_v = N_v_hat2 - self.nu * self.k2 * self.v_hat_temp

        # === Stage 3 ===
        self.u_hat_temp = self.u_hat + 0.5 * self.dt * k2_u
        self.v_hat_temp = self.v_hat + 0.5 * self.dt * k2_v
        self.u = np.fft.ifft2(self.u_hat_temp).real
        self.v = np.fft.ifft2(self.v_hat_temp).real

        N_u_hat3, N_v_hat3 = self.compute_nonlinear_terms()
        k3_u = N_u_hat3 - self.nu * self.k2 * self.u_hat_temp + F_u_hat
        k3_v = N_v_hat3 - self.nu * self.k2 * self.v_hat_temp

        # === Stage 4 ===
        self.u_hat_temp = self.u_hat + self.dt * k3_u
        self.v_hat_temp = self.v_hat + self.dt * k3_v
        self.u = np.fft.ifft2(self.u_hat_temp).real
        self.v = np.fft.ifft2(self.v_hat_temp).real

        N_u_hat4, N_v_hat4 = self.compute_nonlinear_terms()
        k4_u = N_u_hat4 - self.nu * self.k2 * self.u_hat_temp + F_u_hat
        k4_v = N_v_hat4 - self.nu * self.k2 * self.v_hat_temp

        # === RK4 組合 ===
        self.u_hat = self.u_hat + (self.dt / 6.0) * (k1_u + 2*k2_u + 2*k3_u + k4_u)
        self.v_hat = self.v_hat + (self.dt / 6.0) * (k1_v + 2*k2_v + 2*k3_v + k4_v)

        self.u = np.fft.ifft2(self.u_hat).real
        self.v = np.fft.ifft2(self.v_hat).real

        # 計算壓力場
        N_u_hat, N_v_hat = self.compute_nonlinear_terms()
        self.p_hat = self.compute_pressure(N_u_hat, N_v_hat, F_u_hat)

        self.time += self.dt

    def compute_diagnostics(self) -> Dict[str, float]:
        """
        計算完整診斷量：動能、enstrophy、散度誤差、能量注入率、耗散率、Fourier 模式
        
        新增物理量（Re=30 穩態判定關鍵指標）：
        1. 能量注入率：ε_in = ⟨u·f_x⟩
        2. 能量耗散率：ε_diss = ν⟨|∇u|²⟩
        3. Fourier 基模能量：A_1 = |û_1|² + |v̂_1|²（k_y=1 模式）
        """
        # ============ 基本量 ============
        # 動能：KE = ∫(u² + v²) dA / (2A)
        kinetic_energy = 0.5 * np.mean(self.u**2 + self.v**2)

        # 渦度：ω = ∂v/∂x - ∂u/∂y
        omega_hat = 1j * self.kx * self.v_hat - 1j * self.ky * self.u_hat
        omega = np.fft.ifft2(omega_hat).real

        # Enstrophy：E = ∫ω² dA / A
        enstrophy = np.mean(omega**2)

        # 散度誤差：|∂u/∂x + ∂v/∂y|
        div_hat = 1j * self.kx * self.u_hat + 1j * self.ky * self.v_hat
        div = np.fft.ifft2(div_hat).real
        divergence_error = np.max(np.abs(div))

        # ============ 能量平衡分析 ============
        # 1. 能量注入率：ε_in = ⟨u·f_x⟩
        #    其中 f_x = A·sin(k_f·y)（Kolmogorov 強迫項）
        energy_injection = np.mean(self.u * self.forcing)

        # 2. 能量耗散率：ε_diss = ν⟨|∇u|²⟩ = ν⟨(∂u/∂x)² + (∂u/∂y)² + (∂v/∂x)² + (∂v/∂y)²⟩
        #    使用 Fourier 空間計算梯度更精確
        dudx_hat = 1j * self.kx * self.u_hat
        dudy_hat = 1j * self.ky * self.u_hat
        dvdx_hat = 1j * self.kx * self.v_hat
        dvdy_hat = 1j * self.ky * self.v_hat

        dudx = np.fft.ifft2(dudx_hat).real
        dudy = np.fft.ifft2(dudy_hat).real
        dvdx = np.fft.ifft2(dvdx_hat).real
        dvdy = np.fft.ifft2(dvdy_hat).real

        grad_u_squared = dudx**2 + dudy**2 + dvdx**2 + dvdy**2
        energy_dissipation = self.nu * np.mean(grad_u_squared)

        # 能量平衡比（理想穩態應 ≈ 1.0）
        energy_balance_ratio = energy_injection / (energy_dissipation + 1e-12)

        # ============ Fourier 模式分析 ============
        # 3. 基模能量（k_y = k_f = 4 的模式，對應強迫頻率）
        #    A_1(t) = |û(k_x=0, k_y=k_f)|² + |v̂(k_x=0, k_y=k_f)|²
        #    
        # 找到 k_y = k_f 的索引（在 FFT 輸出中）
        ky_1d = np.fft.fftfreq(self.Ny, d=self.Ly/self.Ny) * 2 * np.pi
        idx_ky_kf = np.argmin(np.abs(ky_1d - self.k_f))
        idx_kx_0 = 0  # k_x = 0

        # 取出基模的 Fourier 係數
        u_hat_mode1 = self.u_hat[idx_kx_0, idx_ky_kf]
        v_hat_mode1 = self.v_hat[idx_kx_0, idx_ky_kf]

        # 基模能量（振幅平方）
        mode1_energy = np.abs(u_hat_mode1)**2 + np.abs(v_hat_mode1)**2

        # ============ 真實雷諾數（基於流場） ============
        # Re = U_rms * L / ν
        # 其中 U_rms = sqrt(⟨u²⟩)，L = 2π/k_f
        U_rms = np.sqrt(np.mean(self.u**2))
        L_char = 2 * np.pi / self.k_f
        Re_actual = U_rms * L_char / self.nu
        
        # ============ 返回所有診斷量 ============
        return {
            # 基本量
            'kinetic_energy': kinetic_energy,
            'enstrophy': enstrophy,
            'divergence_error': divergence_error,
            
            # 能量平衡
            'energy_injection': energy_injection,
            'energy_dissipation': energy_dissipation,
            'energy_balance_ratio': energy_balance_ratio,
            
            # Fourier 模式
            'mode1_energy': mode1_energy,
            
            # 真實雷諾數（基於流場 U_rms）
            'reynolds_number': Re_actual,
        }

    def check_steady_state(self, diagnostics_history: list, window: int = 200, threshold: float = 0.02) -> bool:
        """
        檢測是否達到統計穩態（改進版）
        
        新穩態準則（Re=30 時空混沌需更嚴格）：
        1. KE 和 Enstrophy 的相對標準差 < threshold（基本波動檢查）
        2. 能量平衡：0.95 < ε_in/ε_diss < 1.05（能量注入 ≈ 耗散）
        3. Fourier 基模穩定：A_1 相對標準差 < 2*threshold
        
        Args:
            diagnostics_history: 診斷量歷史記錄
            window: 滑動窗口大小（增加至 200 以確保統計穩定）
            threshold: 相對變化率閾值（默認 2%）
            
        Returns:
            True 如果同時滿足上述三個條件
        """
        if len(diagnostics_history) < window:
            return False
        
        recent = diagnostics_history[-window:]
        
        # ========== 準則 1：KE 和 Enstrophy 波動 ==========
        ke_values = np.array([d['kinetic_energy'] for d in recent])
        enstrophy_values = np.array([d['enstrophy'] for d in recent])
        
        ke_mean = np.mean(ke_values)
        enstrophy_mean = np.mean(enstrophy_values)
        
        ke_std = np.std(ke_values)
        enstrophy_std = np.std(enstrophy_values)
        
        ke_relative_std = ke_std / (ke_mean + 1e-10)
        enstrophy_relative_std = enstrophy_std / (enstrophy_mean + 1e-10)
        
        condition1 = (ke_relative_std < threshold) and (enstrophy_relative_std < threshold)
        
        # ========== 準則 2：能量平衡 ==========
        eps_in_values = np.array([d['energy_injection'] for d in recent])
        eps_diss_values = np.array([d['energy_dissipation'] for d in recent])
        
        eps_in_mean = np.mean(eps_in_values)
        eps_diss_mean = np.mean(eps_diss_values)
        
        energy_balance_ratio = eps_in_mean / (eps_diss_mean + 1e-12)
        
        condition2 = 0.95 < energy_balance_ratio < 1.05
        
        # ========== 準則 3：Fourier 基模穩定 ==========
        mode1_values = np.array([d['mode1_energy'] for d in recent])
        mode1_mean = np.mean(mode1_values)
        mode1_std = np.std(mode1_values)
        mode1_relative_std = mode1_std / (mode1_mean + 1e-10)
        
        condition3 = mode1_relative_std < 2 * threshold  # 允許更大波動（混沌態特性）
        
        # ========== 綜合判定 ==========
        is_steady = condition1 and condition2 and condition3
        
        if len(diagnostics_history) % 50 == 0:  # 每 50 步報告一次進度
            logging.info(f"📊 穩態檢查 (t={diagnostics_history[-1].get('time', 0):.1f}):")
            logging.info(f"   1️⃣  KE 波動: {ke_relative_std*100:.2f}% (< {threshold*100:.0f}%) → {'✅' if condition1 else '❌'}")
            logging.info(f"   2️⃣  能量平衡: ε_in/ε_diss = {energy_balance_ratio:.3f} (0.95-1.05) → {'✅' if condition2 else '❌'}")
            logging.info(f"   3️⃣  基模波動: {mode1_relative_std*100:.2f}% (< {2*threshold*100:.0f}%) → {'✅' if condition3 else '❌'}")
        
        if is_steady:
            logging.info(f"🎉 達到統計穩態！")
            logging.info(f"   KE: {ke_mean:.6f} ± {ke_std:.6f} ({ke_relative_std*100:.2f}%)")
            logging.info(f"   Enstrophy: {enstrophy_mean:.4f} ± {enstrophy_std:.4f} ({enstrophy_relative_std*100:.2f}%)")
            logging.info(f"   ε_in: {eps_in_mean:.6f}, ε_diss: {eps_diss_mean:.6f}")
            logging.info(f"   能量平衡比: {energy_balance_ratio:.4f}")
            logging.info(f"   基模能量: {mode1_mean:.2e} ± {mode1_std:.2e}")
        
        return is_steady

    def run(self, T_end: float, save_interval: int = 100, 
            check_steady: bool = True, steady_window: int = 100, steady_threshold: float = 0.01) -> Dict:
        """
        運行 DNS 模擬（帶穩態檢測）
        
        Args:
            T_end: 最大運行時間
            save_interval: 保存間隔（每 N 步）
            check_steady: 是否啟用穩態檢測
            steady_window: 穩態檢測窗口
            steady_threshold: 穩態閾值
            
        Returns:
            results: 包含完整時間演化的字典
        """
        n_steps = int(T_end / self.dt)

        results = {
            'time': [],
            'u': [],
            'v': [],
            'p': [],
            'diagnostics': [],
        }

        logging.info(f"▶️  開始 DNS 模擬：T_max={T_end}, n_steps={n_steps}")
        if check_steady:
            logging.info(f"   穩態檢測：window={steady_window}, threshold={steady_threshold}")

        for step in range(n_steps):
            self.step_rk4()

            # 保存數據
            if step % save_interval == 0:
                diag = self.compute_diagnostics()
                diag['time'] = self.time  # 加入時間戳（用於穩態檢測日誌）

                results['time'].append(self.time)
                results['u'].append(self.u.copy())
                results['v'].append(self.v.copy())
                
                # 計算並保存壓力
                p = np.fft.ifft2(self.p_hat).real
                results['p'].append(p.copy())
                
                results['diagnostics'].append(diag)

                logging.info(
                    f"  Step {step:6d}/{n_steps} | t={self.time:7.2f} | "
                    f"KE={diag['kinetic_energy']:.4e} | "
                    f"ε_in={diag['energy_injection']:.4e} | "
                    f"ε_diss={diag['energy_dissipation']:.4e} | "
                    f"Balance={diag['energy_balance_ratio']:.3f}"
                )

                # 檢測穩態
                if check_steady and self.check_steady_state(results['diagnostics'], steady_window, steady_threshold):
                    logging.info(f"✅ 提前達到穩態！在 t={self.time:.2f} 停止")
                    break

        # 轉換為 numpy 數組
        results['time'] = np.array(results['time'])
        results['u'] = np.array(results['u'])
        results['v'] = np.array(results['v'])
        results['p'] = np.array(results['p'])

        logging.info(f"✅ DNS 模擬完成！總時間：{self.time:.2f}")

        return results


def save_to_hdf5(results: Dict, filepath: Path, config: Dict):
    """保存 DNS 結果到 HDF5 檔案"""
    with h5py.File(filepath, 'w') as f:
        # 保存場資料
        f.create_dataset('time', data=results['time'])
        f.create_dataset('u', data=results['u'], compression='gzip')
        f.create_dataset('v', data=results['v'], compression='gzip')
        f.create_dataset('p', data=results['p'], compression='gzip')

        # 保存診斷量
        diag_group = f.create_group('diagnostics')
        for key in results['diagnostics'][0].keys():
            diag_data = np.array([d[key] for d in results['diagnostics']])
            diag_group.create_dataset(key, data=diag_data)

        # 保存配置
        config_group = f.create_group('config')
        for key, value in config.items():
            config_group.attrs[key] = value

    logging.info(f"💾 結果已保存至：{filepath}")


def plot_diagnostics(results: Dict, output_dir: Path):
    """繪製完整診斷量時間序列（包含能量平衡分析）"""
    time = results['time']
    
    # 提取所有診斷量
    ke = np.array([d['kinetic_energy'] for d in results['diagnostics']])
    enstrophy = np.array([d['enstrophy'] for d in results['diagnostics']])
    div_err = np.array([d['divergence_error'] for d in results['diagnostics']])
    eps_in = np.array([d['energy_injection'] for d in results['diagnostics']])
    eps_diss = np.array([d['energy_dissipation'] for d in results['diagnostics']])
    balance_ratio = np.array([d['energy_balance_ratio'] for d in results['diagnostics']])
    mode1_energy = np.array([d['mode1_energy'] for d in results['diagnostics']])

    # 創建 2x3 子圖佈局
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # 1. Kinetic Energy
    axes[0].plot(time, ke, 'b-', linewidth=1.5)
    axes[0].set_xlabel('Time', fontsize=12)
    axes[0].set_ylabel('Kinetic Energy', fontsize=12)
    axes[0].set_title('Kinetic Energy vs Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Enstrophy
    axes[1].plot(time, enstrophy, 'r-', linewidth=1.5)
    axes[1].set_xlabel('Time', fontsize=12)
    axes[1].set_ylabel('Enstrophy', fontsize=12)
    axes[1].set_title('Enstrophy vs Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # 3. Energy Balance (ε_in vs ε_diss)
    axes[2].plot(time, eps_in, 'g-', linewidth=1.5, label='ε_in (Injection)')
    axes[2].plot(time, eps_diss, 'm--', linewidth=1.5, label='ε_diss (Dissipation)')
    axes[2].set_xlabel('Time', fontsize=12)
    axes[2].set_ylabel('Energy Rate', fontsize=12)
    axes[2].set_title('Energy Injection vs Dissipation', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=10)
    axes[2].grid(True, alpha=0.3)

    # 4. Energy Balance Ratio
    axes[3].plot(time, balance_ratio, 'orange', linewidth=1.5)
    axes[3].axhline(y=1.0, color='k', linestyle='--', linewidth=1, label='Perfect Balance')
    axes[3].axhline(y=0.95, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    axes[3].axhline(y=1.05, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    axes[3].set_xlabel('Time', fontsize=12)
    axes[3].set_ylabel('ε_in / ε_diss', fontsize=12)
    axes[3].set_title('Energy Balance Ratio (Target: 0.95-1.05)', fontsize=14, fontweight='bold')
    axes[3].legend(fontsize=10)
    axes[3].grid(True, alpha=0.3)

    # 5. Fourier Mode 1 Energy
    axes[4].plot(time, mode1_energy, 'purple', linewidth=1.5)
    axes[4].set_xlabel('Time', fontsize=12)
    axes[4].set_ylabel('Mode 1 Energy', fontsize=12)
    axes[4].set_title('Fourier Base Mode Energy (k_y=4)', fontsize=14, fontweight='bold')
    axes[4].grid(True, alpha=0.3)

    # 6. Divergence Error (log scale)
    axes[5].semilogy(time, div_err, 'teal', linewidth=1.5)
    axes[5].set_xlabel('Time', fontsize=12)
    axes[5].set_ylabel('Divergence Error (log scale)', fontsize=12)
    axes[5].set_title('Divergence Error vs Time', fontsize=14, fontweight='bold')
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    
    diag_path = output_dir / 'diagnostics_time_series.png'
    plt.savefig(diag_path, dpi=150, bbox_inches='tight')
    logging.info(f"📊 診斷圖表已保存：{diag_path}")
    plt.close()


def plot_snapshots(results: Dict, output_dir: Path, indices: list = None):
    """繪製流場快照"""
    if indices is None:
        # 預設：初始、中間、最終
        n_snapshots = len(results['time'])
        indices = [0, n_snapshots // 2, n_snapshots - 1]

    fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5 * len(indices)))
    
    if len(indices) == 1:
        axes = axes.reshape(1, -1)

    for i, idx in enumerate(indices):
        u = results['u'][idx]
        v = results['v'][idx]
        p = results['p'][idx]
        t = results['time'][idx]

        # U 速度
        im0 = axes[i, 0].contourf(u.T, levels=20, cmap='RdBu_r')
        axes[i, 0].set_title(f'U velocity (t={t:.2f})', fontsize=12, fontweight='bold')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im0, ax=axes[i, 0])

        # V 速度
        im1 = axes[i, 1].contourf(v.T, levels=20, cmap='RdBu_r')
        axes[i, 1].set_title(f'V velocity (t={t:.2f})', fontsize=12, fontweight='bold')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 1])

        # Pressure
        im2 = axes[i, 2].contourf(p.T, levels=20, cmap='viridis')
        axes[i, 2].set_title(f'Pressure (t={t:.2f})', fontsize=12, fontweight='bold')
        axes[i, 2].set_xlabel('X')
        axes[i, 2].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[i, 2])

    plt.tight_layout()
    
    snapshot_path = output_dir / 'field_snapshots.png'
    plt.savefig(snapshot_path, dpi=150, bbox_inches='tight')
    logging.info(f"📊 場快照已保存：{snapshot_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Kolmogorov Flow Re=30 長時間 DNS（穩態檢測）')
    
    # DNS 參數
    parser.add_argument('--Nx', type=int, default=512, help='X 方向網格點數')
    parser.add_argument('--Ny', type=int, default=256, help='Y 方向網格點數')
    parser.add_argument('--Lx', type=float, default=4*np.pi, help='X 方向域大小')
    parser.add_argument('--Ly', type=float, default=2*np.pi, help='Y 方向域大小')
    parser.add_argument('--nu', type=float, default=0.02, help='動力黏度')
    parser.add_argument('--A', type=float, default=0.768, help='強迫振幅（F₀=0.768 for Re=30）')
    parser.add_argument('--k_f', type=int, default=4, help='強迫波數')
    parser.add_argument('--dt', type=float, default=0.001, help='時間步長')
    
    # 運行參數
    parser.add_argument('--T_end', type=float, default=500.0, help='最大運行時間（若未達穩態）')
    parser.add_argument('--save_interval', type=int, default=100, help='保存間隔（步數）')
    parser.add_argument('--seed', type=int, default=42, help='隨機種子')
    
    # 穩態檢測
    parser.add_argument('--check_steady', action='store_true', default=True, help='啟用穩態檢測')
    parser.add_argument('--steady_window', type=int, default=200, help='穩態檢測窗口')
    parser.add_argument('--steady_threshold', type=float, default=0.01, help='穩態閾值（相對標準差）')
    
    # 輸出
    parser.add_argument('--output', type=str, default='data/kolmogorov_re30_stationary.h5', help='HDF5 輸出路徑')
    parser.add_argument('--plot_dir', type=str, default='results/kolmogorov_re30_dns', help='圖表輸出目錄')

    args = parser.parse_args()

    # 設置隨機種子
    np.random.seed(args.seed)

    # 配置
    config = {
        'Nx': args.Nx,
        'Ny': args.Ny,
        'Lx': args.Lx,
        'Ly': args.Ly,
        'nu': args.nu,
        'A': args.A,
        'k_f': args.k_f,
        'dt': args.dt,
        'T_end': args.T_end,
        'seed': args.seed,
        'steady_window': args.steady_window,
        'steady_threshold': args.steady_threshold,
    }

    # 建立求解器
    dns = KolmogorovFlowDNS(
        Nx=args.Nx,
        Ny=args.Ny,
        Lx=args.Lx,
        Ly=args.Ly,
        nu=args.nu,
        A=args.A,
        k_f=args.k_f,
        dt=args.dt,
        dealias=True,
    )

    # 初始化流場
    dns.initialize_random_perturbation(amplitude=0.5)

    # 運行模擬
    results = dns.run(
        T_end=args.T_end,
        save_interval=args.save_interval,
        check_steady=args.check_steady,
        steady_window=args.steady_window,
        steady_threshold=args.steady_threshold,
    )

    # 保存結果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_to_hdf5(results, output_path, config)

    # 繪製診斷圖表
    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    plot_diagnostics(results, plot_dir)
    plot_snapshots(results, plot_dir)

    logging.info(f"🎉 完成！")


if __name__ == '__main__':
    main()
