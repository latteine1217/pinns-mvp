"""
物理約束驗證工具（Physics Validator）

功能：
- 質量守恆驗證（散度計算）
- 動量守恆驗證（NS 方程殘差）
- 邊界條件驗證
- 能量譜分析
- 自動記錄到 WandB
- 提供詳細的物理驗證報告

使用範例：
    >>> from pinnx.utils.physics_validator import PhysicsValidator
    >>> validator = PhysicsValidator(config)
    >>> results = validator.validate(model, test_data)
    >>> validator.print_summary()
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from scipy.fft import fft2, fftn


class PhysicsValidator:
    """物理約束驗證工具"""

    def __init__(
        self,
        config: Optional[Dict] = None,
        use_wandb: bool = True,
        nu: float = 0.01,
        rho: float = 1.0,
        dimension: int = 2
    ):
        """
        初始化 PhysicsValidator

        Args:
            config: 配置字典（可選）
            use_wandb: 是否自動記錄到 WandB（預設 True）
            nu: 運動黏度（預設 0.01）
            rho: 密度（預設 1.0）
            dimension: 空間維度（2D 或 3D，預設 2）
        """
        self.config = config or {}
        self.use_wandb = use_wandb
        self.nu = nu
        self.rho = rho
        self.dimension = dimension

        # 從配置中讀取參數（如果有）
        if config:
            self.nu = config.get('physics', {}).get('nu', nu)
            self.rho = config.get('physics', {}).get('rho', rho)
            self.dimension = config.get('dimension', dimension)

        # 驗證結果存儲
        self.validation_results = {}

        # 嘗試導入 wandb
        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("Warning: wandb not installed. PhysicsValidator will not log to WandB.")
                self.use_wandb = False

    def compute_divergence(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        計算散度 ∇·u（質量守恆檢查）

        Args:
            u, v, w: 速度分量（w 可選，用於 3D）
            x, y, z: 座標（z 可選，用於 3D）

        Returns:
            dict: 散度統計指標
        """
        # 計算梯度
        dudx = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]

        dvdy = torch.autograd.grad(
            v, y, grad_outputs=torch.ones_like(v),
            create_graph=True, retain_graph=True
        )[0]

        if self.dimension == 3 and w is not None and z is not None:
            dwdz = torch.autograd.grad(
                w, z, grad_outputs=torch.ones_like(w),
                create_graph=True, retain_graph=True
            )[0]
            divergence = dudx + dvdy + dwdz
        else:
            divergence = dudx + dvdy

        # 計算統計量
        div_abs = torch.abs(divergence)
        div_mean = torch.mean(div_abs).item()
        div_max = torch.max(div_abs).item()
        div_99th = torch.quantile(div_abs, 0.99).item()

        # L2 norm
        div_l2 = torch.norm(divergence).item() / np.sqrt(divergence.numel())

        # 相對散度（與速度場的比值）
        velocity_norm = torch.norm(torch.cat([u.flatten(), v.flatten()]))
        if self.dimension == 3 and w is not None:
            velocity_norm = torch.norm(torch.cat([u.flatten(), v.flatten(), w.flatten()]))

        div_relative = div_l2 / (velocity_norm.item() + 1e-10)

        results = {
            'divergence_mean': div_mean,
            'divergence_max': div_max,
            'divergence_99th': div_99th,
            'divergence_l2': div_l2,
            'divergence_relative': div_relative,
        }

        return results

    def compute_momentum_residual(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: Optional[torch.Tensor],
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        forcing: Optional[Tuple[torch.Tensor, ...]] = None
    ) -> Dict[str, float]:
        """
        計算 Navier-Stokes 動量方程殘差

        Args:
            u, v, w: 速度分量
            p: 壓力
            x, y, z: 座標
            forcing: 外力項 (fx, fy, fz)（可選）

        Returns:
            dict: 動量殘差統計
        """
        # x-momentum: u·∇u = -1/ρ ∂p/∂x + ν∇²u + fx
        # 計算對流項 u·∇u
        dudx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]
        dudy = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                                    create_graph=True, retain_graph=True)[0]

        if self.dimension == 3 and w is not None and z is not None:
            dudz = torch.autograd.grad(u, z, grad_outputs=torch.ones_like(u),
                                        create_graph=True, retain_graph=True)[0]
            convection_x = u * dudx + v * dudy + w * dudz
        else:
            convection_x = u * dudx + v * dudy

        # 壓力項 -1/ρ ∂p/∂x
        dpdx = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                    create_graph=True, retain_graph=True)[0]
        pressure_x = -1.0 / self.rho * dpdx

        # 擴散項 ν∇²u
        d2udx2 = torch.autograd.grad(dudx, x, grad_outputs=torch.ones_like(dudx),
                                      create_graph=True, retain_graph=True)[0]
        d2udy2 = torch.autograd.grad(dudy, y, grad_outputs=torch.ones_like(dudy),
                                      create_graph=True, retain_graph=True)[0]

        if self.dimension == 3 and w is not None and z is not None:
            d2udz2 = torch.autograd.grad(dudz, z, grad_outputs=torch.ones_like(dudz),
                                          create_graph=True, retain_graph=True)[0]
            diffusion_x = self.nu * (d2udx2 + d2udy2 + d2udz2)
        else:
            diffusion_x = self.nu * (d2udx2 + d2udy2)

        # 外力（如果有）
        fx = forcing[0] if forcing else 0.0

        # 殘差
        residual_x = convection_x - pressure_x - diffusion_x - fx

        # y-momentum（類似計算）
        dvdx = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                    create_graph=True, retain_graph=True)[0]
        dvdy = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v),
                                    create_graph=True, retain_graph=True)[0]

        if self.dimension == 3 and w is not None and z is not None:
            dvdz = torch.autograd.grad(v, z, grad_outputs=torch.ones_like(v),
                                        create_graph=True, retain_graph=True)[0]
            convection_y = u * dvdx + v * dvdy + w * dvdz
        else:
            convection_y = u * dvdx + v * dvdy

        dpdy = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p),
                                    create_graph=True, retain_graph=True)[0]
        pressure_y = -1.0 / self.rho * dpdy

        d2vdx2 = torch.autograd.grad(dvdx, x, grad_outputs=torch.ones_like(dvdx),
                                      create_graph=True, retain_graph=True)[0]
        d2vdy2 = torch.autograd.grad(dvdy, y, grad_outputs=torch.ones_like(dvdy),
                                      create_graph=True, retain_graph=True)[0]

        if self.dimension == 3 and w is not None and z is not None:
            d2vdz2 = torch.autograd.grad(dvdz, z, grad_outputs=torch.ones_like(dvdz),
                                          create_graph=True, retain_graph=True)[0]
            diffusion_y = self.nu * (d2vdx2 + d2vdy2 + d2vdz2)
        else:
            diffusion_y = self.nu * (d2vdx2 + d2vdy2)

        fy = forcing[1] if forcing else 0.0
        residual_y = convection_y - pressure_y - diffusion_y - fy

        # 計算統計量
        residual_x_abs = torch.abs(residual_x)
        residual_y_abs = torch.abs(residual_y)

        results = {
            'momentum_residual_x_mean': torch.mean(residual_x_abs).item(),
            'momentum_residual_y_mean': torch.mean(residual_y_abs).item(),
            'momentum_residual_x_max': torch.max(residual_x_abs).item(),
            'momentum_residual_y_max': torch.max(residual_y_abs).item(),
            'momentum_residual_mean': (torch.mean(residual_x_abs) + torch.mean(residual_y_abs)).item() / 2.0,
            'momentum_residual_max': max(torch.max(residual_x_abs).item(), torch.max(residual_y_abs).item()),
        }

        # 3D 的 z-momentum
        if self.dimension == 3 and w is not None and z is not None:
            dwdx = torch.autograd.grad(w, x, grad_outputs=torch.ones_like(w),
                                        create_graph=True, retain_graph=True)[0]
            dwdy = torch.autograd.grad(w, y, grad_outputs=torch.ones_like(w),
                                        create_graph=True, retain_graph=True)[0]
            dwdz = torch.autograd.grad(w, z, grad_outputs=torch.ones_like(w),
                                        create_graph=True, retain_graph=True)[0]

            convection_z = u * dwdx + v * dwdy + w * dwdz

            dpdz = torch.autograd.grad(p, z, grad_outputs=torch.ones_like(p),
                                        create_graph=True, retain_graph=True)[0]
            pressure_z = -1.0 / self.rho * dpdz

            d2wdx2 = torch.autograd.grad(dwdx, x, grad_outputs=torch.ones_like(dwdx),
                                          create_graph=True, retain_graph=True)[0]
            d2wdy2 = torch.autograd.grad(dwdy, y, grad_outputs=torch.ones_like(dwdy),
                                          create_graph=True, retain_graph=True)[0]
            d2wdz2 = torch.autograd.grad(dwdz, z, grad_outputs=torch.ones_like(dwdz),
                                          create_graph=True, retain_graph=True)[0]

            diffusion_z = self.nu * (d2wdx2 + d2wdy2 + d2wdz2)

            fz = forcing[2] if forcing else 0.0
            residual_z = convection_z - pressure_z - diffusion_z - fz

            residual_z_abs = torch.abs(residual_z)
            results.update({
                'momentum_residual_z_mean': torch.mean(residual_z_abs).item(),
                'momentum_residual_z_max': torch.max(residual_z_abs).item(),
                'momentum_residual_mean': (
                    torch.mean(residual_x_abs) +
                    torch.mean(residual_y_abs) +
                    torch.mean(residual_z_abs)
                ).item() / 3.0,
            })

        return results

    def check_boundary_conditions(
        self,
        field: torch.Tensor,
        bc_type: str,
        tolerance: float = 1e-5
    ) -> Dict[str, float]:
        """
        驗證邊界條件滿足度

        Args:
            field: 待檢查的場變量
            bc_type: 邊界條件類型 ('periodic', 'no_slip', 'free_slip')
            tolerance: 容差（預設 1e-5）

        Returns:
            dict: 邊界條件違反統計
        """
        results = {}

        if bc_type == 'periodic':
            # 檢查週期性（2D：x=0 vs x=L, y=0 vs y=L）
            # 假設 field 形狀為 [batch, Nx, Ny] 或 [batch, Nx, Ny, Nz]
            if len(field.shape) == 3:  # 2D
                left = field[:, 0, :]
                right = field[:, -1, :]
                periodicity_x_error = torch.mean(torch.abs(left - right)).item()

                bottom = field[:, :, 0]
                top = field[:, :, -1]
                periodicity_y_error = torch.mean(torch.abs(bottom - top)).item()

                results = {
                    'bc/periodicity_x_error': periodicity_x_error,
                    'bc/periodicity_y_error': periodicity_y_error,
                    'bc/periodicity_max_error': max(periodicity_x_error, periodicity_y_error),
                }

            elif len(field.shape) == 4:  # 3D
                # x 方向
                left = field[:, 0, :, :]
                right = field[:, -1, :, :]
                periodicity_x_error = torch.mean(torch.abs(left - right)).item()

                # y 方向
                bottom = field[:, :, 0, :]
                top = field[:, :, -1, :]
                periodicity_y_error = torch.mean(torch.abs(bottom - top)).item()

                # z 方向
                front = field[:, :, :, 0]
                back = field[:, :, :, -1]
                periodicity_z_error = torch.mean(torch.abs(front - back)).item()

                results = {
                    'bc/periodicity_x_error': periodicity_x_error,
                    'bc/periodicity_y_error': periodicity_y_error,
                    'bc/periodicity_z_error': periodicity_z_error,
                    'bc/periodicity_max_error': max(periodicity_x_error, periodicity_y_error, periodicity_z_error),
                }

        elif bc_type == 'no_slip':
            # 檢查壁面速度（應為 0）
            # 假設壁面在 y=0 和 y=H
            wall_bottom = field[:, :, 0]
            wall_top = field[:, :, -1]

            wall_violation_bottom = torch.mean(torch.abs(wall_bottom)).item()
            wall_violation_top = torch.mean(torch.abs(wall_top)).item()

            results = {
                'bc/wall_velocity_violation_bottom': wall_violation_bottom,
                'bc/wall_velocity_violation_top': wall_violation_top,
                'bc/wall_velocity_violation_max': max(wall_violation_bottom, wall_violation_top),
            }

        return results

    def compute_energy_spectrum(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        w: Optional[torch.Tensor] = None,
        u_true: Optional[torch.Tensor] = None,
        v_true: Optional[torch.Tensor] = None,
        w_true: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        計算能量譜 E(k)

        Args:
            u, v, w: 預測的速度分量
            u_true, v_true, w_true: 真實速度分量（可選，用於對比）

        Returns:
            dict: 能量譜統計
        """
        # 轉為 numpy
        u_np = u.detach().cpu().numpy()
        v_np = v.detach().cpu().numpy()

        # 2D FFT
        if self.dimension == 2:
            u_fft = fft2(u_np)
            v_fft = fft2(v_np)
            energy = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2)

        else:  # 3D
            if w is None:
                raise ValueError("3D requires w component")
            w_np = w.detach().cpu().numpy()
            u_fft = fftn(u_np)
            v_fft = fftn(v_np)
            w_fft = fftn(w_np)
            energy = 0.5 * (np.abs(u_fft)**2 + np.abs(v_fft)**2 + np.abs(w_fft)**2)

        # 徑向平均得到 E(k)
        E_k = self._radial_average(energy)

        results = {
            'energy_spectrum': E_k,
        }

        # 如果有真實值，計算誤差
        if u_true is not None and v_true is not None:
            u_true_np = u_true.detach().cpu().numpy()
            v_true_np = v_true.detach().cpu().numpy()

            if self.dimension == 2:
                u_true_fft = fft2(u_true_np)
                v_true_fft = fft2(v_true_np)
                energy_true = 0.5 * (np.abs(u_true_fft)**2 + np.abs(v_true_fft)**2)

            else:
                if w_true is None:
                    raise ValueError("3D requires w_true component")
                w_true_np = w_true.detach().cpu().numpy()
                u_true_fft = fftn(u_true_np)
                v_true_fft = fftn(v_true_np)
                w_true_fft = fftn(w_true_np)
                energy_true = 0.5 * (np.abs(u_true_fft)**2 + np.abs(v_true_fft)**2 + np.abs(w_true_fft)**2)

            E_k_true = self._radial_average(energy_true)

            # 能譜誤差
            spectrum_error = np.mean(np.abs(E_k - E_k_true) / (E_k_true + 1e-10))

            results.update({
                'energy_spectrum_true': E_k_true,
                'energy_spectrum_error': spectrum_error,
            })

        return results

    def _radial_average(self, energy: np.ndarray) -> np.ndarray:
        """
        徑向平均能量譜

        Args:
            energy: FFT 能量密度

        Returns:
            E(k): 徑向平均能量譜
        """
        shape = energy.shape
        center = tuple(s // 2 for s in shape)

        # 建立 k 值網格
        if len(shape) == 2:
            kx = np.fft.fftfreq(shape[0])
            ky = np.fft.fftfreq(shape[1])
            kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
            k_grid = np.sqrt(kx_grid**2 + ky_grid**2)

        else:  # 3D
            kx = np.fft.fftfreq(shape[0])
            ky = np.fft.fftfreq(shape[1])
            kz = np.fft.fftfreq(shape[2])
            kx_grid, ky_grid, kz_grid = np.meshgrid(kx, ky, kz, indexing='ij')
            k_grid = np.sqrt(kx_grid**2 + ky_grid**2 + kz_grid**2)

        # 離散化 k 值
        k_bins = np.arange(0, k_grid.max(), 0.1)
        E_k = np.zeros(len(k_bins) - 1)

        for i in range(len(k_bins) - 1):
            mask = (k_grid >= k_bins[i]) & (k_grid < k_bins[i + 1])
            if np.any(mask):
                E_k[i] = np.mean(energy[mask])

        return E_k

    def compute_physics_violation_score(
        self,
        div_results: Dict[str, float],
        momentum_results: Dict[str, float],
        bc_results: Optional[Dict[str, float]] = None,
        energy_results: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        計算綜合物理違反分數

        Args:
            div_results: 散度結果
            momentum_results: 動量殘差結果
            bc_results: 邊界條件結果（可選）
            energy_results: 能量譜結果（可選）
            weights: 各項權重（可選）

        Returns:
            float: 綜合物理違反分數
        """
        # 預設權重
        default_weights = {
            'divergence': 10.0,      # 質量守恆最重要
            'momentum': 1.0,         # 動量守恆
            'boundary': 5.0,         # 邊界條件
            'energy': 0.1,           # 能量（相對不重要）
        }

        w = weights or default_weights

        # 計算綜合分數
        score = w['divergence'] * div_results['divergence_l2']
        score += w['momentum'] * momentum_results['momentum_residual_mean']

        if bc_results:
            bc_max_violation = max(bc_results.values())
            score += w['boundary'] * bc_max_violation

        if energy_results and 'energy_spectrum_error' in energy_results:
            score += w['energy'] * energy_results['energy_spectrum_error']

        return score

    def validate(
        self,
        model: nn.Module,
        test_data: Dict[str, torch.Tensor],
        log_to_wandb: bool = True
    ) -> Dict[str, Any]:
        """
        綜合物理驗證

        Args:
            model: 訓練好的模型
            test_data: 測試資料（包含 x, y, z, u_true, v_true, w_true, p_true）
            log_to_wandb: 是否記錄到 WandB

        Returns:
            dict: 完整的物理驗證結果
        """
        model.eval()

        # 提取測試資料
        x = test_data['x'].requires_grad_(True)
        y = test_data['y'].requires_grad_(True)
        z = test_data.get('z')
        if z is not None:
            z = z.requires_grad_(True)

        # 模型預測（保持計算圖以計算物理量）
        inputs = torch.cat([x, y], dim=-1) if z is None else torch.cat([x, y, z], dim=-1)
        outputs = model(inputs)

        if self.dimension == 2:
            u_pred, v_pred, p_pred = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3]
            w_pred = None
        else:
            u_pred, v_pred, w_pred, p_pred = outputs[:, 0:1], outputs[:, 1:2], outputs[:, 2:3], outputs[:, 3:4]

        # 1. 質量守恆
        div_results = self.compute_divergence(u_pred, v_pred, w_pred, x, y, z)

        # 2. 動量守恆
        momentum_results = self.compute_momentum_residual(u_pred, v_pred, w_pred, p_pred, x, y, z)

        # 3. 邊界條件（如果配置中有指定）
        bc_results = None
        if 'boundary_type' in self.config:
            bc_type = self.config['boundary_type']
            bc_results = self.check_boundary_conditions(u_pred, bc_type)

        # 4. 能量譜（如果有真實值）
        energy_results = None
        if 'u_true' in test_data and 'v_true' in test_data:
            u_true = test_data['u_true']
            v_true = test_data['v_true']
            w_true = test_data.get('w_true')
            energy_results = self.compute_energy_spectrum(
                u_pred, v_pred, w_pred,
                u_true, v_true, w_true
            )

        # 5. 綜合物理違反分數
        physics_score = self.compute_physics_violation_score(
            div_results, momentum_results, bc_results, energy_results
        )

        # 整合所有結果
        all_results = {
            'mass_conservation': div_results,
            'momentum_conservation': momentum_results,
            'physics_violation_score': physics_score,
        }

        if bc_results:
            all_results['boundary_conditions'] = bc_results

        if energy_results:
            all_results['energy_spectrum'] = energy_results

        # 記錄到 WandB
        if self.use_wandb and log_to_wandb:
            log_dict = {}

            # 質量守恆
            for key, value in div_results.items():
                log_dict[f'physics/mass_conservation/{key}'] = value

            # 動量守恆
            for key, value in momentum_results.items():
                log_dict[f'physics/momentum_conservation/{key}'] = value

            # 邊界條件
            if bc_results:
                for key, value in bc_results.items():
                    log_dict[f'physics/boundary_conditions/{key}'] = value

            # 能量譜
            if energy_results and 'energy_spectrum_error' in energy_results:
                log_dict['physics/energy/spectrum_error'] = energy_results['energy_spectrum_error']

            # 綜合分數
            log_dict['metrics/physics_violation_score'] = physics_score

            self.wandb.log(log_dict)

        # 保存結果
        self.validation_results = all_results

        return all_results

    def print_summary(self):
        """打印物理驗證摘要"""
        if not self.validation_results:
            print("⚠️  No validation results available. Run validate() first.")
            return

        print("\n" + "="*70)
        print("⚛️  Physics Validation Summary")
        print("="*70)

        # 質量守恆
        if 'mass_conservation' in self.validation_results:
            div = self.validation_results['mass_conservation']
            print("\n📊 Mass Conservation (Divergence):")
            print(f"   Mean:           {div['divergence_mean']:.6e}")
            print(f"   Max:            {div['divergence_max']:.6e}")
            print(f"   99th percentile:{div['divergence_99th']:.6e}")
            print(f"   L2 norm:        {div['divergence_l2']:.6e}")
            print(f"   Relative:       {div['divergence_relative']:.6e}")

        # 動量守恆
        if 'momentum_conservation' in self.validation_results:
            mom = self.validation_results['momentum_conservation']
            print("\n📊 Momentum Conservation (NS Residual):")
            print(f"   Mean residual:  {mom['momentum_residual_mean']:.6e}")
            print(f"   Max residual:   {mom['momentum_residual_max']:.6e}")

        # 邊界條件
        if 'boundary_conditions' in self.validation_results:
            bc = self.validation_results['boundary_conditions']
            print("\n📊 Boundary Conditions:")
            for key, value in bc.items():
                print(f"   {key}: {value:.6e}")

        # 能量譜
        if 'energy_spectrum' in self.validation_results:
            energy = self.validation_results['energy_spectrum']
            if 'energy_spectrum_error' in energy:
                print("\n📊 Energy Spectrum:")
                print(f"   Spectrum error: {energy['energy_spectrum_error']:.6e}")

        # 綜合分數
        if 'physics_violation_score' in self.validation_results:
            score = self.validation_results['physics_violation_score']
            print(f"\n⭐ Composite Physics Violation Score: {score:.6e}")

        print("="*70 + "\n")


def compute_turbulence_statistics(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
    u_true: Optional[torch.Tensor] = None,
    v_true: Optional[torch.Tensor] = None,
    w_true: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """
    計算湍流統計量（3D Channel Flow）

    Args:
        u, v, w: 預測速度分量
        y: 壁面法向座標
        u_true, v_true, w_true: 真實值（可選）

    Returns:
        dict: 湍流統計量
    """
    results = {}

    # 壁面剪應力 τ_w = μ * ∂u/∂y|wall
    # 假設壁面在 y=0
    dudy = torch.autograd.grad(
        u, y, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]

    # 假設第一層是壁面
    tau_w_pred = dudy[:, 0].mean().item()

    results['wall_shear_stress_pred'] = tau_w_pred

    if u_true is not None:
        dudy_true = torch.autograd.grad(
            u_true, y, grad_outputs=torch.ones_like(u_true),
            create_graph=True, retain_graph=True
        )[0]
        tau_w_true = dudy_true[:, 0].mean().item()

        results['wall_shear_stress_true'] = tau_w_true
        results['wall_shear_stress_error'] = abs(tau_w_pred - tau_w_true) / tau_w_true

    # Reynolds stress: <u'v'> = <uv> - <u><v>
    # 簡化：假設已經是脈動量
    reynolds_stress = torch.mean(u * v, dim=0)
    results['reynolds_stress_uv'] = reynolds_stress.mean().item()

    return results
