"""
Leith 模擬參數計算器
===================

基於物理約束條件自動計算：
1. CFL 條件 → 時間步長
2. Kolmogorov 尺度 → 網格解析度
3. Taylor 微尺度 → 驗證充分性
4. 統計收斂時間 → 總模擬時間

理論依據：
---------
1. **CFL 條件** (Courant-Friedrichs-Lewy):
   CFL = u_max × dt / Δx ≤ CFL_target (通常 0.5-1.0)
   
2. **Kolmogorov 尺度** (最小渦旋尺度):
   η = (ν³/ε)^(1/4)
   需要 Δx ≤ C_η × η，通常 C_η = 1.5-2.0
   
3. **Taylor 微尺度** (慣性-耗散過渡):
   λ = sqrt(15 × ν × u_rms² / ε)
   需要至少解析到 λ 尺度
   
4. **能量耗散率** (2D Kolmogorov flow):
   ε ≈ ν × ω_rms²
   其中 ω_rms ≈ k_f × u_rms
   
5. **統計穩態時間**:
   T_eddy = L / u_rms (大渦週轉時間)
   需要 T_total ≥ 10-20 × T_eddy

作者: 主 Agent
日期: 2025-12-17
"""

import numpy as np
import argparse
from typing import Dict, Tuple, Optional

class LeithParameterCalculator:
    """
    基於物理約束的 Leith 模擬參數計算器
    """
    
    def __init__(
        self,
        Re: float,
        A: float = 1.0,
        k_f: int = 4,
        L: float = 2 * np.pi,
        CFL_target: float = 0.5,
        C_eta: float = 2.0,
        safety_factor: float = 1.2,
    ):
        """
        Args:
            Re: 雷諾數
            A: Kolmogorov 強迫振幅
            k_f: 強迫波數
            L: 域大小
            CFL_target: 目標 CFL 數 (0.5 保守, 1.0 激進)
            C_eta: Kolmogorov 尺度解析係數 (1.5-2.0)
            safety_factor: 安全係數 (1.0-1.5)
        """
        self.Re = Re
        self.A = A
        self.k_f = k_f
        self.L = L
        self.CFL_target = CFL_target
        self.C_eta = C_eta
        self.safety_factor = safety_factor
        
        # 根據定義計算分子黏滯
        # Re = sqrt(A) × (2π/k_f)^(3/2) / ν
        self.nu = np.sqrt(A) * (2 * np.pi / k_f)**(3/2) / Re
        
        # 估算特徵速度（基於強迫）
        self.u_rms_estimate = np.sqrt(A)  # 粗略估計
        
        # 估算渦度 (2D Kolmogorov flow)
        self.omega_rms_estimate = k_f * self.u_rms_estimate
        
        # 估算能量耗散率
        self.epsilon_estimate = self.nu * self.omega_rms_estimate**2
        
    def calculate_kolmogorov_scale(self) -> float:
        """
        計算 Kolmogorov 尺度 η = (ν³/ε)^(1/4)
        """
        eta = (self.nu**3 / self.epsilon_estimate)**(1/4)
        return eta
    
    def calculate_taylor_microscale(self) -> float:
        """
        計算 Taylor 微尺度 λ = sqrt(15 × ν × u²/ε)
        """
        lambda_t = np.sqrt(15 * self.nu * self.u_rms_estimate**2 / self.epsilon_estimate)
        return lambda_t
    
    def calculate_eddy_turnover_time(self) -> float:
        """
        計算大渦週轉時間 T_eddy = L / u_rms
        """
        T_eddy = self.L / self.u_rms_estimate
        return T_eddy
    
    def calculate_optimal_N(self) -> int:
        """
        基於 Kolmogorov 尺度計算最優網格數
        N ≥ L / (C_η × η)
        """
        eta = self.calculate_kolmogorov_scale()
        N_required = self.L / (self.C_eta * eta)
        
        # 向上取到 2 的冪次（FFT 效率）
        N_power2 = 2**np.ceil(np.log2(N_required))
        
        # 應用安全係數
        N_safe = int(N_power2 * self.safety_factor)
        
        # 再次向上取到 2 的冪次
        N_final = 2**np.ceil(np.log2(N_safe))
        
        return int(N_final)
    
    def calculate_optimal_dt(self, N: Optional[int] = None) -> float:
        """
        基於 CFL 條件計算最優時間步長
        dt ≤ CFL_target × Δx / u_max
        
        Args:
            N: 網格數（若未提供則自動計算）
        """
        if N is None:
            N = self.calculate_optimal_N()
        
        dx = self.L / N
        
        # 估算最大速度（保守估計為 1.5 × u_rms）
        u_max_estimate = 1.5 * self.u_rms_estimate
        
        # CFL 條件
        dt_CFL = self.CFL_target * dx / u_max_estimate
        
        # 應用安全係數
        dt_safe = dt_CFL / self.safety_factor
        
        # 取合理精度（避免過小）
        dt_rounded = np.round(dt_safe, decimals=5)
        
        return dt_rounded
    
    def calculate_optimal_T_total(self) -> Tuple[float, float]:
        """
        計算最優模擬時間
        
        Returns:
            T_spinup: Spin-up 時間 (5-10 × T_eddy)
            T_total: 總時間 (20-30 × T_eddy)
        """
        T_eddy = self.calculate_eddy_turnover_time()
        
        # Spin-up: 至少 5 個週轉時間
        T_spinup = max(10.0, 5 * T_eddy)
        
        # 總時間: 至少 20 個週轉時間用於統計
        T_total = max(100.0, 20 * T_eddy)
        
        return T_spinup, T_total
    
    def calculate_optimal_C_L(self) -> float:
        """
        基於雷諾數估算最優 Leith 常數
        
        經驗公式:
        - Re < 100: C_L ≈ 0.25-0.30 (過渡區，需更多耗散)
        - 100 ≤ Re < 300: C_L ≈ 0.20-0.25
        - Re ≥ 300: C_L ≈ 0.15-0.20 (湍流區)
        """
        if self.Re < 100:
            C_L = 0.28
        elif self.Re < 300:
            C_L = 0.22
        else:
            C_L = 0.18
        
        return C_L
    
    def verify_resolution(self, N: int, dt: float) -> Dict[str, float]:
        """
        驗證解析度充分性
        
        Args:
            N: 網格數
            dt: 時間步長
            
        Returns:
            驗證指標字典
        """
        dx = self.L / N
        eta = self.calculate_kolmogorov_scale()
        lambda_t = self.calculate_taylor_microscale()
        u_max = 1.5 * self.u_rms_estimate
        
        # 1. Kolmogorov 尺度解析
        points_per_eta = eta / dx
        
        # 2. Taylor 微尺度解析
        points_per_lambda = lambda_t / dx
        
        # 3. CFL 數
        CFL_actual = u_max * dt / dx
        
        # 4. 黏滯時間步長
        dt_viscous = 0.5 * dx**2 / self.nu
        viscous_ratio = dt / dt_viscous
        
        return {
            'points_per_eta': points_per_eta,
            'points_per_lambda': points_per_lambda,
            'CFL_actual': CFL_actual,
            'dt_viscous_ratio': viscous_ratio,
            'dx': dx,
            'eta': eta,
            'lambda_t': lambda_t,
        }
    
    def get_optimal_parameters(self) -> Dict:
        """
        獲取完整的最優參數集
        """
        N = self.calculate_optimal_N()
        dt = self.calculate_optimal_dt(N)
        T_spinup, T_total = self.calculate_optimal_T_total()
        C_L = self.calculate_optimal_C_L()
        
        verification = self.verify_resolution(N, dt)
        
        return {
            'N': N,
            'nu': self.nu,
            'A': self.A,
            'k_f': self.k_f,
            'dt': dt,
            'T_spinup': T_spinup,
            'T_total': T_total,
            'C_L': C_L,
            'L': self.L,
            'verification': verification,
            'estimates': {
                'Re': self.Re,
                'u_rms': self.u_rms_estimate,
                'omega_rms': self.omega_rms_estimate,
                'epsilon': self.epsilon_estimate,
                'eta': verification['eta'],
                'lambda_t': verification['lambda_t'],
                'T_eddy': self.calculate_eddy_turnover_time(),
            }
        }
    
    def print_report(self):
        """
        打印詳細參數報告
        """
        params = self.get_optimal_parameters()
        
        print("="*70)
        print(f"Leith 模擬最優參數 (Re = {self.Re})")
        print("="*70)
        print()
        
        print("【理論估算】")
        print(f"  分子黏滯 ν = {params['nu']:.6f}")
        print(f"  特徵速度 u_rms ≈ {params['estimates']['u_rms']:.4f}")
        print(f"  特徵渦度 ω_rms ≈ {params['estimates']['omega_rms']:.4f}")
        print(f"  能量耗散率 ε ≈ {params['estimates']['epsilon']:.6f}")
        print(f"  Kolmogorov 尺度 η ≈ {params['estimates']['eta']:.6f}")
        print(f"  Taylor 微尺度 λ ≈ {params['estimates']['lambda_t']:.6f}")
        print(f"  大渦週轉時間 T_eddy ≈ {params['estimates']['T_eddy']:.4f}")
        print()
        
        print("【建議參數】")
        print(f"  網格數 N = {params['N']} (Δx = {params['verification']['dx']:.6f})")
        print(f"  時間步長 dt = {params['dt']:.6f}")
        print(f"  Spin-up 時間 T_spinup = {params['T_spinup']:.1f}")
        print(f"  總模擬時間 T_total = {params['T_total']:.1f}")
        print(f"  Leith 常數 C_L = {params['C_L']:.2f}")
        print()
        
        v = params['verification']
        print("【解析度驗證】")
        print(f"  Kolmogorov 尺度解析: {v['points_per_eta']:.2f} 點/η", end="")
        if v['points_per_eta'] >= 2.0:
            print(" ✅")
        elif v['points_per_eta'] >= 1.5:
            print(" ⚠️  (邊界)")
        else:
            print(" ❌ (不足)")
        
        print(f"  Taylor 微尺度解析: {v['points_per_lambda']:.2f} 點/λ", end="")
        if v['points_per_lambda'] >= 5.0:
            print(" ✅")
        elif v['points_per_lambda'] >= 3.0:
            print(" ⚠️  (邊界)")
        else:
            print(" ❌ (不足)")
        
        print(f"  CFL 數: {v['CFL_actual']:.3f}", end="")
        if v['CFL_actual'] <= 0.5:
            print(" ✅ (保守)")
        elif v['CFL_actual'] <= 1.0:
            print(" ✅ (穩定)")
        else:
            print(" ❌ (不穩定)")
        
        print(f"  黏滯時間步長比: {v['dt_viscous_ratio']:.3f}", end="")
        if v['dt_viscous_ratio'] < 0.5:
            print(" ✅")
        else:
            print(" ⚠️  (接近極限)")
        
        print()
        
        # 估算計算成本
        N_steps = int(params['T_total'] / params['dt'])
        cost_factor = (params['N'] / 128)**2 * (N_steps / 20000)
        
        print("【計算成本估算】")
        print(f"  總時間步數: {N_steps:,}")
        print(f"  相對成本 (vs Re=50 基準): {cost_factor:.1f}×")
        
        # 基於 Re=50 的 2.2 分鐘估算
        estimated_time_min = 2.2 * cost_factor
        if estimated_time_min < 60:
            print(f"  預計運行時間: ~{estimated_time_min:.1f} 分鐘")
        else:
            print(f"  預計運行時間: ~{estimated_time_min/60:.1f} 小時")
        
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="計算 Leith 模擬的最優參數",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python calculate_leith_parameters.py --re 50 100 500
  python calculate_leith_parameters.py --re 100 --cfl 0.3 --C_eta 1.5
        """
    )
    
    parser.add_argument('--re', type=float, nargs='+', required=True,
                       help='雷諾數列表')
    parser.add_argument('--A', type=float, default=1.0,
                       help='強迫振幅 (default: 1.0)')
    parser.add_argument('--k_f', type=int, default=4,
                       help='強迫波數 (default: 4)')
    parser.add_argument('--cfl', type=float, default=0.5,
                       help='目標 CFL 數 (default: 0.5)')
    parser.add_argument('--C_eta', type=float, default=2.0,
                       help='Kolmogorov 尺度解析係數 (default: 2.0)')
    parser.add_argument('--safety', type=float, default=1.2,
                       help='安全係數 (default: 1.2)')
    parser.add_argument('--generate-script', action='store_true',
                       help='生成批次執行腳本')
    
    args = parser.parse_args()
    
    results = {}
    
    for Re in args.re:
        print()
        calc = LeithParameterCalculator(
            Re=Re,
            A=args.A,
            k_f=args.k_f,
            CFL_target=args.cfl,
            C_eta=args.C_eta,
            safety_factor=args.safety,
        )
        calc.print_report()
        results[Re] = calc.get_optimal_parameters()
    
    # 生成批次腳本
    if args.generate_script:
        print("\n" + "="*70)
        print("批次生成腳本")
        print("="*70)
        print()
        print("#!/bin/bash")
        print("# Auto-generated Leith simulation script")
        print("# Date: 2025-12-17")
        print()
        
        for Re in args.re:
            p = results[Re]
            output = f"data/lowfi/kolmogorov_rans/rans_re{int(Re)}_kf4_leith_optimized.h5"
            
            print(f"# Re = {int(Re)}")
            print(f"python3 scripts/generate/dns/generate_kolmogorov_leith.py \\")
            print(f"    --N {p['N']} \\")
            print(f"    --nu {p['nu']:.6f} \\")
            print(f"    --A {p['A']:.1f} \\")
            print(f"    --k_f {p['k_f']} \\")
            print(f"    --dt {p['dt']:.6f} \\")
            print(f"    --T_total {p['T_total']:.1f} \\")
            print(f"    --T_spinup {p['T_spinup']:.1f} \\")
            print(f"    --C_L {p['C_L']:.2f} \\")
            print(f"    --output {output}")
            print()


if __name__ == '__main__':
    main()
