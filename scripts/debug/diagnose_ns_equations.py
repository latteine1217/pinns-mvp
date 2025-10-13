"""
NS 方程物理診斷腳本
==================

目標：隔離測試 Navier-Stokes 方程的每個物理項，診斷 Residual Loss 異常高的根因

診斷範圍：
1. 動量方程 x/y/z 分量（對流項、壓力項、黏性項）
2. 連續性方程（不可壓縮條件）
3. 邊界條件（壁面無滑移、週期性）
4. VS-PINN 縮放因子（驗證是否正確應用）
5. 物理參數一致性（Re_τ, ν, dP/dx）

使用方式：
    python scripts/debug/diagnose_ns_equations.py --config configs/vs_pinn_3d_warmup_test.yml
"""

import sys
import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Any
import argparse

# 加入專案根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from pinnx.physics.vs_pinn_channel_flow import VSPINNChannelFlow
from pinnx.models.fourier_mlp import PINNNet


class NSEquationDiagnostics:
    """NS 方程診斷器"""
    
    def __init__(self, config_path: str):
        """
        初始化診斷器
        
        Args:
            config_path: 配置文件路徑
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化物理模組與模型
        self.physics = self._create_physics_module()
        self.model = self._create_model()
        
        # 診斷報告
        self.report: Dict[str, Any] = {}
    
    def _load_config(self) -> Dict:
        """載入配置文件"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_physics_module(self) -> VSPINNChannelFlow:
        """創建物理模組"""
        physics_cfg = self.config['physics']
        
        scaling_factors = {
            'N_x': physics_cfg['vs_pinn']['scaling_factors']['N_x'],
            'N_y': physics_cfg['vs_pinn']['scaling_factors']['N_y'],
            'N_z': physics_cfg['vs_pinn']['scaling_factors']['N_z'],
        }
        
        physics_params = {
            'nu': physics_cfg['nu'],
            'dP_dx': physics_cfg['channel_flow']['pressure_gradient'],
            'rho': physics_cfg['rho'],
        }
        
        domain_bounds = {
            'x': tuple(physics_cfg['domain']['x_range']),
            'y': tuple(physics_cfg['domain']['y_range']),
            'z': tuple(physics_cfg['domain']['z_range']),
        }
        
        return VSPINNChannelFlow(
            scaling_factors=scaling_factors,
            physics_params=physics_params,
            domain_bounds=domain_bounds,
        ).to(self.device)
    
    def _create_model(self) -> torch.nn.Module:
        """創建簡化模型（用於測試梯度計算）"""
        model_cfg = self.config['model']
        
        model = PINNNet(
            in_dim=model_cfg['in_dim'],
            out_dim=model_cfg['out_dim'],
            width=model_cfg['width'],
            depth=model_cfg['depth'],
            activation=model_cfg['activation'],
            fourier_m=model_cfg['fourier_m'],
            fourier_sigma=model_cfg['fourier_sigma'],
        ).to(self.device)
        
        return model
    
    def generate_test_points(self, n_points: int = 100) -> torch.Tensor:
        """
        生成測試點（包含內部點、邊界點、週期邊界點）
        
        Args:
            n_points: 測試點數量
            
        Returns:
            coords: [n_points, 3] = [x, y, z]
        """
        domain = self.config['physics']['domain']
        x_range = domain['x_range']
        y_range = domain['y_range']
        z_range = domain['z_range']
        
        # 內部點（隨機採樣）
        n_interior = n_points // 2
        x = np.random.uniform(x_range[0], x_range[1], n_interior)
        y = np.random.uniform(y_range[0] + 0.1, y_range[1] - 0.1, n_interior)  # 避開壁面
        z = np.random.uniform(z_range[0], z_range[1], n_interior)
        
        interior_points = np.stack([x, y, z], axis=1)
        
        # 壁面點
        n_wall = n_points // 4
        x_wall = np.random.uniform(x_range[0], x_range[1], n_wall)
        z_wall = np.random.uniform(z_range[0], z_range[1], n_wall)
        y_wall_lower = np.full(n_wall, y_range[0])
        y_wall_upper = np.full(n_wall, y_range[1])
        
        wall_lower = np.stack([x_wall, y_wall_lower, z_wall], axis=1)
        wall_upper = np.stack([x_wall, y_wall_upper, z_wall], axis=1)
        
        # 週期邊界點（x 方向）
        n_periodic = n_points // 4
        y_periodic = np.random.uniform(y_range[0], y_range[1], n_periodic)
        z_periodic = np.random.uniform(z_range[0], z_range[1], n_periodic)
        x_periodic_min = np.full(n_periodic, x_range[0])
        x_periodic_max = np.full(n_periodic, x_range[1])
        
        periodic_min = np.stack([x_periodic_min, y_periodic, z_periodic], axis=1)
        periodic_max = np.stack([x_periodic_max, y_periodic, z_periodic], axis=1)
        
        # 合併所有點
        all_points = np.vstack([
            interior_points,
            wall_lower, wall_upper,
            periodic_min, periodic_max
        ])
        
        return torch.tensor(all_points, dtype=torch.float32, device=self.device)
    
    def test_gradient_computation(self, coords: torch.Tensor) -> Dict[str, float]:
        """
        測試梯度計算的正確性
        
        測試案例：
        1. 解析函數 f(x,y,z) = sin(x) * cos(y) * exp(z)
        2. 比較數值梯度與自動微分梯度
        
        Args:
            coords: 測試點 [n, 3]
            
        Returns:
            誤差統計 {'mean_error', 'max_error', 'std_error'}
        """
        coords.requires_grad_(True)
        
        # 解析函數
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        f = torch.sin(x) * torch.cos(y) * torch.exp(0.1 * z)
        
        # 自動微分梯度
        grads = self.physics.compute_gradients(f, coords, order=1)
        df_dx_auto = grads['x']
        df_dy_auto = grads['y']
        df_dz_auto = grads['z']
        
        # 解析梯度
        df_dx_true = torch.cos(x) * torch.cos(y) * torch.exp(0.1 * z)
        df_dy_true = -torch.sin(x) * torch.sin(y) * torch.exp(0.1 * z)
        df_dz_true = 0.1 * torch.sin(x) * torch.cos(y) * torch.exp(0.1 * z)
        
        # 計算誤差
        error_x = torch.abs(df_dx_auto - df_dx_true)
        error_y = torch.abs(df_dy_auto - df_dy_true)
        error_z = torch.abs(df_dz_auto - df_dz_true)
        
        total_error = torch.cat([error_x, error_y, error_z], dim=0)
        
        return {
            'mean_error': total_error.mean().item(),
            'max_error': total_error.max().item(),
            'std_error': total_error.std().item(),
        }
    
    def test_vs_pinn_scaling(self, coords: torch.Tensor) -> Dict[str, Any]:
        """
        測試 VS-PINN 縮放因子是否正確應用
        
        檢查：
        1. 梯度是否包含縮放因子 N_x, N_y, N_z
        2. Laplacian 是否包含 N_x², N_y², N_z²
        
        Args:
            coords: 測試點 [n, 3]
            
        Returns:
            縮放驗證結果
        """
        coords.requires_grad_(True)
        
        # 測試函數 f(x,y,z) = x² + y² + z²
        x, y, z = coords[:, 0:1], coords[:, 1:2], coords[:, 2:3]
        f = x**2 + y**2 + z**2
        
        # 計算梯度
        grads = self.physics.compute_gradients(f, coords, order=1)
        df_dx = grads['x']
        df_dy = grads['y']
        df_dz = grads['z']
        
        # 理論梯度（無縮放）
        df_dx_expected = 2 * x
        df_dy_expected = 2 * y
        df_dz_expected = 2 * z
        
        # ⚠️ 如果 VS-PINN 正確實現，應該有：
        # df_dx_computed = N_x * 2x（因為 ∂f/∂x = N_x * ∂f/∂X）
        # 但當前實現直接對原始坐標求導，所以應該相等
        
        is_scaled_x = not torch.allclose(df_dx, df_dx_expected, rtol=1e-4)
        is_scaled_y = not torch.allclose(df_dy, df_dy_expected, rtol=1e-4)
        is_scaled_z = not torch.allclose(df_dz, df_dz_expected, rtol=1e-4)
        
        # 計算 Laplacian
        laplacian = self.physics.compute_laplacian(f, coords)
        laplacian_expected = torch.full_like(laplacian, 6.0)  # ∇²(x²+y²+z²) = 2+2+2 = 6
        
        is_laplacian_scaled = not torch.allclose(laplacian, laplacian_expected, rtol=1e-4)
        
        return {
            'scaling_applied': is_scaled_x or is_scaled_y or is_scaled_z or is_laplacian_scaled,
            'gradient_errors': {
                'x': torch.abs(df_dx - df_dx_expected).mean().item(),
                'y': torch.abs(df_dy - df_dy_expected).mean().item(),
                'z': torch.abs(df_dz - df_dz_expected).mean().item(),
            },
            'laplacian_error': torch.abs(laplacian - laplacian_expected).mean().item(),
            'scaling_factors': {
                'N_x': self.physics.N_x.item(),
                'N_y': self.physics.N_y.item(),
                'N_z': self.physics.N_z.item(),
            }
        }
    
    def test_momentum_equation_terms(self, coords: torch.Tensor) -> Dict[str, Dict[str, float]]:
        """
        測試動量方程的各個項（對流、壓力、黏性、驅動）
        
        使用合成解：Poiseuille 流（層流解）
            u(y) = U_max * (1 - y²)
            v = w = 0
            p = p_0 - dP/dx * x
            
        Args:
            coords: 測試點 [n, 3]
            
        Returns:
            各項的殘差統計
        """
        coords.requires_grad_(True)
        
        # 合成 Poiseuille 流解
        y = coords[:, 1:2]
        x = coords[:, 0:1]
        
        U_max = 20.0  # 與配置中的 output_norm 一致
        u = U_max * (1 - y**2)
        v = torch.zeros_like(u)
        w = torch.zeros_like(u)
        p = -self.physics.dP_dx * x  # 線性壓力分佈
        
        predictions = torch.cat([u, v, w, p], dim=1)
        
        # 計算動量殘差
        residuals = self.physics.compute_momentum_residuals(coords, predictions)
        
        # Poiseuille 流的理論殘差（應該很小）
        # ν ∂²u/∂y² - dP/dx = 0（穩態層流平衡）
        # 理論上 residual_x ≈ 0（除了邊界效應）
        
        stats = {}
        for name, residual in residuals.items():
            stats[name] = {
                'mean': residual.mean().item(),
                'std': residual.std().item(),
                'max': residual.abs().max().item(),
                'rms': torch.sqrt(torch.mean(residual**2)).item(),
            }
        
        return stats
    
    def test_continuity_equation(self, coords: torch.Tensor) -> Dict[str, float]:
        """
        測試連續性方程（不可壓縮條件）
        
        使用無散場測試：
            u = ∂ψ/∂y, v = -∂ψ/∂x, w = 0
            其中 ψ = sin(x) * cos(y)（流函數）
            
        這保證 ∂u/∂x + ∂v/∂y = 0
        
        Args:
            coords: 測試點 [n, 3]
            
        Returns:
            連續性殘差統計
        """
        coords.requires_grad_(True)
        
        x, y = coords[:, 0:1], coords[:, 1:2]
        
        # 流函數
        psi = torch.sin(x) * torch.cos(y)
        
        # 從流函數推導速度（保證無散）
        psi_grads = self.physics.compute_gradients(psi, coords, order=1)
        u = psi_grads['y']   # ∂ψ/∂y
        v = -psi_grads['x']  # -∂ψ/∂x
        w = torch.zeros_like(u)
        p = torch.zeros_like(u)
        
        predictions = torch.cat([u, v, w, p], dim=1)
        
        # 計算連續性殘差
        divergence = self.physics.compute_continuity_residual(coords, predictions)
        
        return {
            'mean': divergence.mean().item(),
            'std': divergence.std().item(),
            'max': divergence.abs().max().item(),
            'rms': torch.sqrt(torch.mean(divergence**2)).item(),
        }
    
    def test_boundary_conditions(self, coords: torch.Tensor) -> Dict[str, Any]:
        """
        測試邊界條件（壁面無滑移、週期性）
        
        Args:
            coords: 測試點 [n, 3]
            
        Returns:
            邊界條件驗證結果
        """
        # 隨機初始化模型預測
        predictions = self.model(coords)
        
        # 提取壁面點
        y_range = self.physics.domain_bounds['y']
        tol = 1e-5
        mask_lower = torch.abs(coords[:, 1] - y_range[0]) < tol
        mask_upper = torch.abs(coords[:, 1] - y_range[1]) < tol
        
        # 壁面速度（應為零）
        u_wall_lower = predictions[mask_lower, 0] if mask_lower.any() else torch.tensor([0.0])
        v_wall_lower = predictions[mask_lower, 1] if mask_lower.any() else torch.tensor([0.0])
        w_wall_lower = predictions[mask_lower, 2] if mask_lower.any() else torch.tensor([0.0])
        
        u_wall_upper = predictions[mask_upper, 0] if mask_upper.any() else torch.tensor([0.0])
        v_wall_upper = predictions[mask_upper, 1] if mask_upper.any() else torch.tensor([0.0])
        w_wall_upper = predictions[mask_upper, 2] if mask_upper.any() else torch.tensor([0.0])
        
        # 週期性檢查（x 方向）
        x_range = self.physics.domain_bounds['x']
        mask_x_min = torch.abs(coords[:, 0] - x_range[0]) < tol
        mask_x_max = torch.abs(coords[:, 0] - x_range[1]) < tol
        
        periodic_error_x = torch.tensor(0.0)
        if mask_x_min.any() and mask_x_max.any():
            n_min = min(mask_x_min.sum().item(), mask_x_max.sum().item())
            fields_x_min = predictions[mask_x_min][:n_min]
            fields_x_max = predictions[mask_x_max][:n_min]
            periodic_error_x = torch.mean((fields_x_min - fields_x_max)**2)
        
        return {
            'wall_no_slip': {
                'lower': {
                    'u_rms': torch.sqrt(torch.mean(u_wall_lower**2)).item(),
                    'v_rms': torch.sqrt(torch.mean(v_wall_lower**2)).item(),
                    'w_rms': torch.sqrt(torch.mean(w_wall_lower**2)).item(),
                },
                'upper': {
                    'u_rms': torch.sqrt(torch.mean(u_wall_upper**2)).item(),
                    'v_rms': torch.sqrt(torch.mean(v_wall_upper**2)).item(),
                    'w_rms': torch.sqrt(torch.mean(w_wall_upper**2)).item(),
                }
            },
            'periodic_bc': {
                'x_direction_mse': periodic_error_x.item(),
            }
        }
    
    def test_physical_parameters(self) -> Dict[str, Any]:
        """
        測試物理參數一致性
        
        檢查：
        1. Re_τ = U_τ * h / ν 是否等於 1000
        2. dP/dx 是否與壁剪應力一致
        3. 密度是否合理
        
        Returns:
            參數驗證結果
        """
        nu = self.physics.nu.item()
        dP_dx = self.physics.dP_dx.item()
        rho = self.physics.rho.item()
        
        # Re_τ 計算（假設 U_τ = 1.0, h = 1.0）
        U_tau = 1.0
        h = 1.0
        Re_tau_computed = U_tau * h / nu
        Re_tau_target = self.config['physics']['channel_flow']['Re_tau']
        
        # 壁剪應力與壓降一致性
        # τ_w = μ * U_τ² / h = ρ * ν * U_τ² / h
        tau_w_expected = rho * nu * U_tau**2 / h
        
        # 壓降平衡：dP/dx = τ_w / h（通道流）
        dP_dx_expected = tau_w_expected / h
        
        return {
            'reynolds_number': {
                'computed': Re_tau_computed,
                'target': Re_tau_target,
                'error': abs(Re_tau_computed - Re_tau_target) / Re_tau_target,
                'status': '✅ 一致' if abs(Re_tau_computed - Re_tau_target) / Re_tau_target < 0.1 else '❌ 不一致'
            },
            'pressure_gradient': {
                'configured': dP_dx,
                'expected_from_wall_shear': dP_dx_expected,
                'ratio': dP_dx / dP_dx_expected if dP_dx_expected > 0 else float('inf'),
            },
            'raw_parameters': {
                'nu': nu,
                'dP_dx': dP_dx,
                'rho': rho,
            }
        }
    
    def run_all_diagnostics(self) -> Dict[str, Any]:
        """
        執行所有診斷測試
        
        Returns:
            完整診斷報告
        """
        print("="*80)
        print("🔬 NS 方程物理診斷開始")
        print("="*80)
        
        # 生成測試點
        coords = self.generate_test_points(n_points=500)
        print(f"✅ 生成 {coords.shape[0]} 個測試點")
        
        # 1. 梯度計算測試
        print("\n📊 測試 1: 梯度計算正確性")
        grad_test = self.test_gradient_computation(coords)
        print(f"   平均誤差: {grad_test['mean_error']:.2e}")
        print(f"   最大誤差: {grad_test['max_error']:.2e}")
        self.report['gradient_computation'] = grad_test
        
        # 2. VS-PINN 縮放測試
        print("\n📊 測試 2: VS-PINN 縮放因子應用")
        scaling_test = self.test_vs_pinn_scaling(coords)
        print(f"   縮放已應用: {scaling_test['scaling_applied']}")
        print(f"   梯度誤差 (x): {scaling_test['gradient_errors']['x']:.2e}")
        print(f"   梯度誤差 (y): {scaling_test['gradient_errors']['y']:.2e}")
        print(f"   梯度誤差 (z): {scaling_test['gradient_errors']['z']:.2e}")
        print(f"   Laplacian 誤差: {scaling_test['laplacian_error']:.2e}")
        self.report['vs_pinn_scaling'] = scaling_test
        
        # 3. 動量方程項測試
        print("\n📊 測試 3: 動量方程各項（Poiseuille 流）")
        momentum_test = self.test_momentum_equation_terms(coords)
        for direction, stats in momentum_test.items():
            print(f"   {direction}: RMS={stats['rms']:.2e}, Max={stats['max']:.2e}")
        self.report['momentum_equation'] = momentum_test
        
        # 4. 連續性方程測試
        print("\n📊 測試 4: 連續性方程（無散場）")
        continuity_test = self.test_continuity_equation(coords)
        print(f"   散度 RMS: {continuity_test['rms']:.2e}")
        print(f"   散度 Max: {continuity_test['max']:.2e}")
        self.report['continuity_equation'] = continuity_test
        
        # 5. 邊界條件測試
        print("\n📊 測試 5: 邊界條件")
        bc_test = self.test_boundary_conditions(coords)
        print(f"   下壁面 u RMS: {bc_test['wall_no_slip']['lower']['u_rms']:.2e}")
        print(f"   上壁面 u RMS: {bc_test['wall_no_slip']['upper']['u_rms']:.2e}")
        print(f"   週期性誤差 (x): {bc_test['periodic_bc']['x_direction_mse']:.2e}")
        self.report['boundary_conditions'] = bc_test
        
        # 6. 物理參數測試
        print("\n📊 測試 6: 物理參數一致性")
        param_test = self.test_physical_parameters()
        print(f"   Re_τ (計算): {param_test['reynolds_number']['computed']:.0f}")
        print(f"   Re_τ (目標): {param_test['reynolds_number']['target']:.0f}")
        print(f"   狀態: {param_test['reynolds_number']['status']}")
        print(f"   ν = {param_test['raw_parameters']['nu']:.2e}")
        self.report['physical_parameters'] = param_test
        
        print("\n" + "="*80)
        print("✅ 診斷完成")
        print("="*80)
        
        return self.report
    
    def save_report(self, output_path: str):
        """
        保存診斷報告為 YAML 格式
        
        Args:
            output_path: 輸出文件路徑
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(self.report, f, default_flow_style=False, sort_keys=False)
        
        print(f"\n💾 診斷報告已保存至: {output_file}")
    
    def print_summary(self):
        """打印診斷摘要"""
        print("\n" + "="*80)
        print("📋 診斷摘要")
        print("="*80)
        
        # 關鍵問題標記
        issues = []
        
        # 1. VS-PINN 縮放
        if 'vs_pinn_scaling' in self.report:
            if not self.report['vs_pinn_scaling']['scaling_applied']:
                issues.append("❌ VS-PINN 縮放因子未實際應用於梯度計算")
        
        # 2. Re_τ 不一致
        if 'physical_parameters' in self.report:
            re_error = self.report['physical_parameters']['reynolds_number']['error']
            if re_error > 0.1:  # 10% 容差
                issues.append(f"❌ Re_τ 不一致（誤差 {re_error*100:.1f}%）")
        
        # 3. 動量殘差過大
        if 'momentum_equation' in self.report:
            for direction, stats in self.report['momentum_equation'].items():
                if stats['rms'] > 100:  # 閾值：RMS > 100
                    issues.append(f"⚠️  {direction} 殘差異常高 (RMS={stats['rms']:.1e})")
        
        # 4. 連續性殘差過大
        if 'continuity_equation' in self.report:
            if self.report['continuity_equation']['rms'] > 1.0:
                issues.append(f"⚠️  連續性殘差過大 (RMS={self.report['continuity_equation']['rms']:.1e})")
        
        # 打印問題列表
        if issues:
            print("\n🚨 發現的問題:")
            for issue in issues:
                print(f"   {issue}")
        else:
            print("\n✅ 未發現明顯問題")
        
        print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='NS 方程物理診斷')
    parser.add_argument('--config', type=str, default='configs/vs_pinn_3d_warmup_test.yml',
                        help='配置文件路徑')
    parser.add_argument('--output', type=str, default='results/diagnostics/ns_equation_report.yml',
                        help='診斷報告輸出路徑')
    args = parser.parse_args()
    
    # 執行診斷
    diagnostics = NSEquationDiagnostics(args.config)
    report = diagnostics.run_all_diagnostics()
    
    # 保存報告
    diagnostics.save_report(args.output)
    
    # 打印摘要
    diagnostics.print_summary()


if __name__ == '__main__':
    main()
