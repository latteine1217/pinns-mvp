#!/usr/bin/env python3
"""
2D Kolmogorov Flow LES 生成器 (spectral + hyperviscosity).

- 使用 vorticity-streamfunction 形式
- SGS 使用 hyperviscosity (p=2)
- 線性摩擦 r 由 DNS turnover time 推估
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ============================
# I/O helpers
# ============================

def load_npy_payload(file_path: Path) -> Dict[str, object]:
    """載入 NPY 字典資料。"""
    payload = np.load(file_path, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(f"NPY payload 格式錯誤: {file_path}")
    return data


def resolve_dns_config(dns_data: Dict[str, object]) -> Dict[str, float]:
    """解析 DNS config 參數。"""
    config_raw = dns_data.get("config")
    if not isinstance(config_raw, dict):
        raise ValueError("DNS 檔案缺少 config")

    required_keys = ("L", "nu", "A", "k_f", "dt")
    config = {}
    for key in required_keys:
        if key not in config_raw:
            raise ValueError(f"DNS config 缺少 {key}")
        config[key] = float(config_raw[key])

    u_shape = np.asarray(dns_data["u"]).shape
    config["N"] = int(config_raw.get("N", u_shape[-1]))
    return config


def estimate_turnover_time(dns_data: Dict[str, object]) -> float:
    """估計 DNS turnover time。"""
    u = np.asarray(dns_data["u"], dtype=float)
    v = np.asarray(dns_data["v"], dtype=float)
    time = np.asarray(dns_data["time"], dtype=float)

    if u.ndim != 3:
        raise ValueError("DNS u 需為 time 序列")

    if len(time) < 2:
        raise ValueError("DNS time 長度不足")

    t_start = time[int(len(time) * 0.75)]
    mask = time >= t_start
    u_rms = np.sqrt(np.mean(u[mask] ** 2 + v[mask] ** 2))
    if u_rms <= 0:
        raise ValueError("DNS u_rms 無效")

    config_raw = dns_data.get("config")
    if isinstance(config_raw, dict) and "L" in config_raw:
        L = float(config_raw["L"])
    else:
        L = 2 * np.pi
    return float(L / u_rms)


def estimate_dns_omega_rms(dns_data: Dict[str, object]) -> float:
    """估計 DNS omega RMS（使用後 1/4 時間平均）。"""
    omega = np.asarray(dns_data.get("omega"), dtype=float)
    time = np.asarray(dns_data["time"], dtype=float)

    if omega.ndim != 3:
        raise ValueError("DNS omega 需為 time 序列")

    t_start = time[int(len(time) * 0.75)]
    mask = time >= t_start
    omega_rms = np.sqrt(np.mean(omega[mask] ** 2))
    if omega_rms <= 0:
        raise ValueError("DNS omega_rms 無效")

    return float(omega_rms)


# ============================
# LES solver
# ============================

class KolmogorovLES:
    """2D Kolmogorov Flow LES (vorticity-streamfunction)."""

    def __init__(
        self,
        N: int,
        L: float,
        nu: float,
        A: float,
        k_f: int,
        dt: float,
        nu_h: float,
        hyper_p: int,
        r_fric: float,
        omega_rms: float,
        seed: int | None,
        dealias: bool = True,
        dealias_mode: str = "2/3",
    ) -> None:
        # === 參數 ===
        self.N = N
        self.L = L
        self.nu = nu
        self.A = A
        self.k_f = k_f
        self.dt = dt
        self.nu_h = nu_h
        self.hyper_p = hyper_p
        self.r_fric = r_fric
        self.dealias = dealias
        self.dealias_mode = dealias_mode

        # === 網格 ===
        self.x = np.linspace(0.0, L, N, endpoint=False)
        self.y = np.linspace(0.0, L, N, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # === 頻譜網格 ===
        k = 2 * np.pi * np.fft.fftfreq(N, d=L / N)
        self.KX, self.KY = np.meshgrid(k, k, indexing="ij")
        self.K2 = self.KX**2 + self.KY**2
        self.K2_safe = self.K2.copy()
        self.K2_safe[0, 0] = 1.0

        # === 去混疊 ===
        if dealias:
            if dealias_mode == "3/2":
                self.pad_N = int(3 * N / 2)
                self.pad_scale = (self.pad_N / N) ** 2
                self.pad_scale_inv = (N / self.pad_N) ** 2

                k_pad = 2 * np.pi * np.fft.fftfreq(self.pad_N, d=L / self.pad_N)
                self.pad_kx, self.pad_ky = np.meshgrid(k_pad, k_pad, indexing="ij")
                self.pad_k2 = self.pad_kx**2 + self.pad_ky**2
                self.pad_k2_safe = self.pad_k2.copy()
                self.pad_k2_safe[0, 0] = 1.0
                self.dealias_mask = np.ones_like(self.K2)
            elif dealias_mode == "2/3":
                k_cut = (N // 3) * 2 * np.pi / L
                mask = (np.abs(self.KX) <= k_cut) & (np.abs(self.KY) <= k_cut)
                self.dealias_mask = mask.astype(float)
                self.pad_N = None
                self.pad_scale = 1.0
                self.pad_scale_inv = 1.0
                self.pad_kx = None
                self.pad_ky = None
                self.pad_k2 = None
                self.pad_k2_safe = None
            else:
                raise ValueError(f"Unknown dealias_mode: {dealias_mode}")
        else:
            self.pad_N = None
            self.pad_scale = 1.0
            self.pad_scale_inv = 1.0
            self.pad_kx = None
            self.pad_ky = None
            self.pad_k2 = None
            self.pad_k2_safe = None
            self.dealias_mask = np.ones_like(self.K2)

        # === forcing (vorticity form) ===
        k_phys = k_f * 2 * np.pi / L
        forcing_x = A * np.sin(k_phys * self.Y)
        forcing_omega = -A * k_phys * np.cos(k_phys * self.Y)
        self.forcing_hat = np.fft.fft2(forcing_omega) * self.dealias_mask

        # === 初始化 omega ===
        rng = np.random.default_rng(seed)
        omega_init = rng.standard_normal((N, N))
        omega_init -= omega_init.mean()
        omega_scale = omega_rms / (np.sqrt(np.mean(omega_init**2)) + 1e-12)
        omega_init *= omega_scale
        self.omega_hat = np.fft.fft2(omega_init) * self.dealias_mask
        self.omega_hat[0, 0] = 0.0

        logging.info("=== Kolmogorov LES 初始化 ===")
        logging.info(f"N={N}, L={L}, nu={nu}, nu_h={nu_h}, p={hyper_p}")
        logging.info(f"A={A}, k_f={k_f}, dt={dt}, r={r_fric}")
        logging.info(f"omega_rms={omega_rms:.3f}, seed={seed}")

    def compute_velocity(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """由 omega 解 psi 再取 u, v。"""
        psi_hat = -omega_hat / self.K2_safe
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        return u, v

    def pad_hat(self, field_hat: np.ndarray) -> np.ndarray:
        """頻譜零填充到 3/2 尺寸。"""
        if self.pad_N is None:
            raise RuntimeError("pad_N 未初始化")
        pad = (self.pad_N - self.N) // 2
        field_shift = np.fft.fftshift(field_hat)
        padded = np.zeros((self.pad_N, self.pad_N), dtype=field_hat.dtype)
        padded[pad:pad + self.N, pad:pad + self.N] = field_shift
        return np.fft.ifftshift(padded)

    def truncate_hat(self, field_hat_pad: np.ndarray) -> np.ndarray:
        """頻譜裁切回原始尺寸。"""
        if self.pad_N is None:
            raise RuntimeError("pad_N 未初始化")
        pad = (self.pad_N - self.N) // 2
        field_shift = np.fft.fftshift(field_hat_pad)
        truncated = field_shift[pad:pad + self.N, pad:pad + self.N]
        return np.fft.ifftshift(truncated)

    def compute_rhs(self, omega_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """計算 RHS (頻譜) 與速度場。"""
        psi_hat = -omega_hat / self.K2_safe

        # === 非線性項 ===
        u = np.real(np.fft.ifft2(1j * self.KY * psi_hat))
        v = np.real(np.fft.ifft2(-1j * self.KX * psi_hat))
        dwdx = np.real(np.fft.ifft2(1j * self.KX * omega_hat))
        dwdy = np.real(np.fft.ifft2(1j * self.KY * omega_hat))

        adv = -(u * dwdx + v * dwdy)

        if self.pad_N is not None:
            if self.pad_kx is None or self.pad_ky is None or self.pad_k2_safe is None:
                raise RuntimeError("pad_kx/pad_ky 未初始化")

            omega_hat_pad = self.pad_hat(omega_hat)
            psi_hat_pad = -omega_hat_pad / self.pad_k2_safe
            u_pad = np.real(np.fft.ifft2(1j * self.pad_ky * psi_hat_pad))
            v_pad = np.real(np.fft.ifft2(-1j * self.pad_kx * psi_hat_pad))
            dwdx_pad = np.real(np.fft.ifft2(1j * self.pad_kx * omega_hat_pad))
            dwdy_pad = np.real(np.fft.ifft2(1j * self.pad_ky * omega_hat_pad))

            u_pad *= self.pad_scale
            v_pad *= self.pad_scale
            dwdx_pad *= self.pad_scale
            dwdy_pad *= self.pad_scale

            adv_pad = -(u_pad * dwdx_pad + v_pad * dwdy_pad)
            adv_hat_pad = np.fft.fft2(adv_pad) * self.pad_scale_inv
            adv_hat = self.truncate_hat(adv_hat_pad)
        else:
            adv_hat = np.fft.fft2(adv) * self.dealias_mask

        # === 擴散 ===
        diff_hat = -self.nu * self.K2 * omega_hat

        # === Hyperviscosity ===
        hyper_hat = -self.nu_h * (self.K2 ** self.hyper_p) * omega_hat

        # === 線性摩擦 ===
        friction_hat = -self.r_fric * omega_hat

        rhs_hat = adv_hat + diff_hat + hyper_hat + friction_hat + self.forcing_hat
        return rhs_hat, u, v

    def step_rk2(self) -> Tuple[float, float]:
        """RK2 時間步進。"""
        k1, u1, v1 = self.compute_rhs(self.omega_hat)
        omega_tmp = self.omega_hat + self.dt * k1
        k2, u2, v2 = self.compute_rhs(omega_tmp)
        self.omega_hat = self.omega_hat + 0.5 * self.dt * (k1 + k2)
        self.omega_hat *= self.dealias_mask
        self.omega_hat[0, 0] = 0.0

        u_max = float(np.max(np.abs(u2)))
        v_max = float(np.max(np.abs(v2)))
        return u_max, v_max

    def diagnostics(self) -> Tuple[float, float, float]:
        """計算能量、enstrophy、divergence error。"""
        u, v = self.compute_velocity(self.omega_hat)
        ke = float(0.5 * np.mean(u**2 + v**2))
        omega = np.real(np.fft.ifft2(self.omega_hat))
        enstrophy = float(0.5 * np.mean(omega**2))
        div = np.real(np.fft.ifft2(1j * self.KX * np.fft.fft2(u) + 1j * self.KY * np.fft.fft2(v)))
        div_err = float(np.max(np.abs(div)))
        return ke, enstrophy, div_err

    def compute_cfl(self, u_max: float, v_max: float) -> float:
        """計算 CFL 數值。"""
        dx = self.L / self.N
        umax = max(u_max, v_max, 1e-12)
        return float(umax * self.dt / dx)


# ============================
# Main
# ============================

def main() -> None:
    parser = argparse.ArgumentParser(description="Kolmogorov 2D LES (spectral + hyperviscosity)")
    parser.add_argument("--dns", type=str, required=True, help="DNS NPY 檔案路徑")
    parser.add_argument("--N", type=int, default=256, help="LES 解析度")
    parser.add_argument("--T_end", type=float, default=20.0, help="模擬總時間")
    parser.add_argument("--save_interval", type=int, default=100, help="輸出間隔")
    parser.add_argument("--dt", type=float, default=None, help="時間步長（預設沿用 DNS）")
    parser.add_argument("--auto_dt", action="store_true", help="自動縮小 dt 以滿足 CFL")
    parser.add_argument("--cfl_target", type=float, default=0.4, help="CFL 目標上限")
    parser.add_argument("--abort_on_cfl", action="store_true", help="CFL 超標時中止")
    parser.add_argument("--nu_h", type=float, default=None, help="hyperviscosity 係數")
    parser.add_argument("--nu_h_alpha", type=float, default=10.0, help="nu_h 預設倍率 (nu_h = alpha * nu / k_max^(2p-2))")
    parser.add_argument("--hyper_p", type=int, default=2, help="hyperviscosity 次方 p")
    parser.add_argument("--r_scale", type=float, default=10.0, help="1/r = r_scale * turnover time")
    parser.add_argument("--dealias_mode", type=str, default="2/3", choices=["2/3", "3/2"], help="去混疊模式")
    parser.add_argument("--omega_rms", type=float, default=None, help="初始 omega RMS（預設跟 DNS 對齊）")
    parser.add_argument("--seed", type=int, default=None, help="初始隨機種子")
    parser.add_argument("--output", type=str, required=True, help="輸出 NPY 檔案")

    args = parser.parse_args()

    dns_path = Path(args.dns)
    dns_data = load_npy_payload(dns_path)
    dns_config = resolve_dns_config(dns_data)

    turnover_time = estimate_turnover_time(dns_data)
    r_fric = 1.0 / (args.r_scale * turnover_time)

    dns_omega_rms = estimate_dns_omega_rms(dns_data)
    omega_rms = args.omega_rms if args.omega_rms is not None else dns_omega_rms

    dt = args.dt if args.dt is not None else dns_config["dt"]
    if args.auto_dt:
        dx = dns_config["L"] / args.N
        u_rms = dns_config["L"] / turnover_time
        dt_cfl = args.cfl_target * dx / max(u_rms, 1e-12)
        if dt_cfl < dt:
            logging.info(f"🔧 auto_dt: dt {dt:.3e} -> {dt_cfl:.3e}")
            dt = dt_cfl

    if args.nu_h is None:
        k_max = (args.N // 3) * 2 * np.pi / dns_config["L"]
        if args.dealias_mode == "3/2":
            k_max = 0.5 * args.N * 2 * np.pi / dns_config["L"]
        nu_h = args.nu_h_alpha * dns_config["nu"] / (k_max ** (2 * args.hyper_p - 2))
    else:
        nu_h = args.nu_h

    solver = KolmogorovLES(
        N=args.N,
        L=dns_config["L"],
        nu=dns_config["nu"],
        A=dns_config["A"],
        k_f=int(dns_config["k_f"]),
        dt=dt,
        nu_h=nu_h,
        hyper_p=args.hyper_p,
        r_fric=r_fric,
        omega_rms=omega_rms,
        seed=args.seed,
        dealias=True,
        dealias_mode=args.dealias_mode,
    )

    n_steps = int(args.T_end / dt)
    n_snapshots = n_steps // args.save_interval + 1

    u_snapshots = np.zeros((n_snapshots, args.N, args.N), dtype=np.float32)
    v_snapshots = np.zeros((n_snapshots, args.N, args.N), dtype=np.float32)
    omega_snapshots = np.zeros((n_snapshots, args.N, args.N), dtype=np.float32)
    time_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
    ke_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
    enstrophy_snapshots = np.zeros((n_snapshots,), dtype=np.float64)
    div_snapshots = np.zeros((n_snapshots,), dtype=np.float64)

    logging.info("=== LES 開始 ===")
    logging.info(f"DNS: {dns_path}")
    logging.info(f"LES N={args.N}, T_end={args.T_end}, dt={dt}")
    logging.info(f"nu_h={nu_h:.3e}, r={r_fric:.3e}, r_scale={args.r_scale}")
    logging.info(f"omega_rms={omega_rms:.3e}, cfl_target={args.cfl_target}")
    logging.info(f"dealias_mode={args.dealias_mode}")

    idx = 0
    t_start = time.time()
    for step in range(n_steps + 1):
        t = step * dt

        if step % args.save_interval == 0:
            u, v = solver.compute_velocity(solver.omega_hat)
            omega = np.real(np.fft.ifft2(solver.omega_hat))

            u_snapshots[idx] = u
            v_snapshots[idx] = v
            omega_snapshots[idx] = omega
            time_snapshots[idx] = t

            ke, enstrophy, div_err = solver.diagnostics()
            ke_snapshots[idx] = ke
            enstrophy_snapshots[idx] = enstrophy
            div_snapshots[idx] = div_err

            if idx % 10 == 0:
                logging.info(
                    f"Step {step:6d}/{n_steps} | t={t:.2f} | KE={ke:.4e} | "
                    f"Enstrophy={enstrophy:.4e} | Div={div_err:.2e}"
                )
            idx += 1

        if step < n_steps:
            u_max, v_max = solver.step_rk2()
            cfl = solver.compute_cfl(u_max, v_max)
            if cfl > args.cfl_target:
                logging.warning(
                    f"⚠️  CFL={cfl:.3f} 超過 {args.cfl_target:.2f}，建議降低 dt 或 omega_rms"
                )
                if args.abort_on_cfl:
                    raise RuntimeError("CFL 超標，請降低 dt 或 omega_rms")

    output_data = {
        "x": np.linspace(0, dns_config["L"], args.N, endpoint=False),
        "y": np.linspace(0, dns_config["L"], args.N, endpoint=False),
        "u": u_snapshots,
        "v": v_snapshots,
        "p": np.zeros_like(u_snapshots),
        "omega": omega_snapshots,
        "time": time_snapshots,
        "diagnostics": {
            "kinetic_energy": ke_snapshots,
            "enstrophy": enstrophy_snapshots,
            "divergence_error": div_snapshots,
        },
        "config": {
            "N": args.N,
            "L": dns_config["L"],
            "nu": dns_config["nu"],
            "A": dns_config["A"],
            "k_f": dns_config["k_f"],
            "dt": dt,
            "T_end": args.T_end,
            "save_interval": args.save_interval,
            "nu_h": nu_h,
            "nu_h_alpha": args.nu_h_alpha,
            "hyper_p": args.hyper_p,
            "r": r_fric,
            "r_scale": args.r_scale,
            "omega_rms": args.omega_rms,
            "seed": args.seed,
            "model": "LES",
            "dns_source": str(dns_path),
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, np.array(output_data, dtype=object), allow_pickle=True)

    total_time = time.time() - t_start
    logging.info(f"✅ LES 完成，輸出: {output_path}")
    logging.info(f"⏱️  總運算時間: {total_time:.1f} 秒 ({total_time/60:.1f} 分鐘)")


if __name__ == "__main__":
    main()
