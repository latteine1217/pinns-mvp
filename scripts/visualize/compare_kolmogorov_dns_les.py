#!/usr/bin/env python3
"""
Compare Kolmogorov DNS vs LES results.

Outputs:
- Vorticity comparison at t_eval (DNS / LES / |Error|)
- |u| (speed) comparison at t_eval (DNS / LES / |Error|)
- Energy spectrum comparison at t_eval
- Velocity-field L2 error vs time (u, v, speed)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Mapping, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom


# -----------------------------
# Data loading and preprocessing
# -----------------------------

def load_npy_payload(file_path: Path) -> Dict[str, Any]:
    """Load NPY payload saved as dict."""
    payload = np.load(file_path, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(f"NPY payload format invalid: {file_path}")
    return data


def resolve_grid(data: Mapping[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Resolve grid coordinates from data or config."""
    if "x" in data and "y" in data:
        return np.asarray(data["x"], dtype=float), np.asarray(data["y"], dtype=float)

    config = data.get("config", {})
    n = int(config.get("N", data["u"].shape[-1]))
    length = float(config.get("L", 2 * np.pi))
    grid = np.linspace(0.0, length, n, endpoint=False)
    return grid, grid


def interpolate_to_dns_grid(field: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """Interpolate LES field onto DNS grid using cubic zoom."""
    zoom_factor = target_shape[0] / field.shape[0]
    if np.isclose(zoom_factor, 1.0):
        return field
    return zoom(field, zoom_factor, order=3)


def compute_vorticity(u: np.ndarray, v: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Compute vorticity field ω = ∂v/∂x - ∂u/∂y."""
    dv_dx = np.gradient(v, dx, axis=-1)
    du_dy = np.gradient(u, dy, axis=-2)
    return dv_dx - du_dy


def resolve_omega(
    data: Mapping[str, Any],
    u: np.ndarray,
    v: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Return omega from data or compute from u/v."""
    omega = data.get("omega")
    if omega is not None:
        return np.asarray(omega, dtype=float)

    dx = float(np.mean(np.diff(x)))
    dy = float(np.mean(np.diff(y)))
    return compute_vorticity(u, v, dx, dy)


# -----------------------------
# Metrics
# -----------------------------

def compute_rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute relative L2 error in percent."""
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref) * 100.0)


# -----------------------------
# Visualization helpers
# -----------------------------

def build_signed_norm(field: np.ndarray) -> TwoSlopeNorm:
    """Create symmetric normalization around zero."""
    max_abs = float(np.nanmax(np.abs(field)))
    if max_abs == 0.0:
        max_abs = 1.0
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def plot_vorticity_comparison(
    x: np.ndarray,
    y: np.ndarray,
    dns: np.ndarray,
    leith: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot DNS/LES/error panels for vorticity."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    X, Y = np.meshgrid(x, y, indexing="ij")
    error = np.abs(leith - dns)
    norm = build_signed_norm(dns)

    panels = [
        (dns, "DNS vorticity", "RdBu_r", norm),
        (leith, "LES vorticity", "RdBu_r", norm),
        (error, "|Error|", "magma", None),
    ]

    for ax, (field, label, cmap, field_norm) in zip(axes, panels):
        im = ax.contourf(X, Y, field, levels=50, cmap=cmap, norm=field_norm)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_speed_comparison(
    x: np.ndarray,
    y: np.ndarray,
    dns: np.ndarray,
    leith: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot DNS/LES/error panels for speed |u|."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    X, Y = np.meshgrid(x, y, indexing="ij")
    error = np.abs(leith - dns)
    vmin = float(min(dns.min(), leith.min()))
    vmax = float(max(dns.max(), leith.max()))

    panels = [
        (dns, "DNS |u|", "viridis", vmin, vmax),
        (leith, "LES |u|", "viridis", vmin, vmax),
        (error, "|Error|", "magma", None, None),
    ]

    for ax, (field, label, cmap, vmin_val, vmax_val) in zip(axes, panels):
        im = ax.contourf(X, Y, field, levels=50, cmap=cmap, vmin=vmin_val, vmax=vmax_val)
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compute_energy_spectrum(u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute isotropic 2D energy spectrum."""
    n = u.shape[0]
    u_fft = np.fft.fft2(u)
    v_fft = np.fft.fft2(v)
    e_k = 0.5 * (np.abs(u_fft) ** 2 + np.abs(v_fft) ** 2) / n**4

    kx = np.fft.fftfreq(n, 1.0 / n)
    ky = np.fft.fftfreq(n, 1.0 / n)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    k_mag = np.sqrt(kx_grid**2 + ky_grid**2)

    k_bins = np.arange(1, n // 2)
    spectrum = np.zeros_like(k_bins, dtype=float)

    for i, k_val in enumerate(k_bins):
        mask = (k_mag >= k_val - 0.5) & (k_mag < k_val + 0.5)
        spectrum[i] = e_k[mask].sum()

    return k_bins, spectrum


def plot_energy_spectrum_comparison(
    dns_u: np.ndarray,
    dns_v: np.ndarray,
    model_u: np.ndarray,
    model_v: np.ndarray,
    output_path: Path,
    title: str,
    model_label: str,
) -> None:
    """Plot energy spectrum comparison between DNS and model."""
    k_dns, e_dns = compute_energy_spectrum(dns_u, dns_v)
    k_model, e_model = compute_energy_spectrum(model_u, model_v)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(k_dns, e_dns, "b-", linewidth=2.0, label="DNS")
    ax.loglog(k_model, e_model, "r--", linewidth=2.0, label=model_label)
    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("Energy Spectrum E(k)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both", linestyle="--")
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_l2_timeseries(
    time: np.ndarray,
    errors: Dict[str, np.ndarray],
    output_path: Path,
    title: str,
) -> None:
    """Plot L2 error time series for velocity fields."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(time, errors["u"], label="u", linewidth=2)
    ax.plot(time, errors["v"], label="v", linewidth=2)
    ax.plot(time, errors["speed"], label="speed", linewidth=2)
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative L2 Error (%)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_l2_csv(
    time: np.ndarray,
    errors: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Write L2 errors to CSV."""
    header = "time,l2_u,l2_v,l2_speed"
    data = np.column_stack([time, errors["u"], errors["v"], errors["speed"]])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_path, data, delimiter=",", header=header, comments="")


# -----------------------------
# Main pipeline
# -----------------------------

def parse_re_list(text: str) -> List[int]:
    """Parse Reynolds number list."""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def select_snapshot(field: np.ndarray, idx: int) -> np.ndarray:
    """Select snapshot at idx if time dimension exists."""
    if field.ndim == 3:
        return field[idx]
    return field


def compare_single_re(
    re_val: int,
    t_eval: float,
    output_dir: Path,
    dns_avg_window: Tuple[float, float],
    model_path: Path | None,
    model_label: str,
    model_avg_window: Tuple[float, float],
) -> None:
    """Run DNS/model comparison for a single Reynolds number."""
    dns_path = Path(f"data/kolmogorov_dns/kolmogorov_dns_{re_val}.npy")
    default_model_path = Path(f"data/kolmogorov_leith/kolmogorov_leith_re{re_val}.npy")
    model_path = model_path or default_model_path

    if not dns_path.exists():
        raise FileNotFoundError(f"DNS file not found: {dns_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    dns_data = load_npy_payload(dns_path)
    model_data = load_npy_payload(model_path)

    dns_time = np.asarray(dns_data["time"], dtype=float)
    t_idx = int(np.argmin(np.abs(dns_time - t_eval)))
    t_actual = float(dns_time[t_idx])

    dns_u = np.asarray(dns_data["u"], dtype=float)
    dns_v = np.asarray(dns_data["v"], dtype=float)

    x_dns, y_dns = resolve_grid(dns_data)
    dns_omega = resolve_omega(dns_data, dns_u, dns_v, x_dns, y_dns)

    model_x, model_y = resolve_grid(model_data)
    model_u_series = np.asarray(model_data["u"], dtype=float)
    model_v_series = np.asarray(model_data["v"], dtype=float)
    if model_u_series.ndim == 2:
        model_u_series = model_u_series[None, ...]
        model_v_series = model_v_series[None, ...]
    model_time = np.asarray(model_data.get("time", []), dtype=float)
    model_omega_series = resolve_omega(model_data, model_u_series, model_v_series, model_x, model_y)

    if model_time.size:
        model_idx = int(np.argmin(np.abs(model_time - t_actual)))
    else:
        model_idx = 0

    model_u = select_snapshot(model_u_series, model_idx)
    model_v = select_snapshot(model_v_series, model_idx)
    model_omega_raw = select_snapshot(model_omega_series, model_idx)

    model_u_interp = interpolate_to_dns_grid(model_u, dns_u.shape[1:])
    model_v_interp = interpolate_to_dns_grid(model_v, dns_u.shape[1:])
    model_omega_interp = interpolate_to_dns_grid(model_omega_raw, dns_u.shape[1:])

    output_subdir = output_dir / f"re{re_val}"
    output_subdir.mkdir(parents=True, exist_ok=True)

    dns_omega_snap = select_snapshot(dns_omega, t_idx)
    dns_u_snap = select_snapshot(dns_u, t_idx)
    dns_v_snap = select_snapshot(dns_v, t_idx)

    vorticity_title = f"Re={re_val} | t={t_actual:.2f} | Vorticity"
    plot_vorticity_comparison(
        x_dns,
        y_dns,
        dns_omega_snap,
        model_omega_interp,
        output_subdir / f"vorticity_compare_t{t_actual:.2f}.png",
        vorticity_title,
    )

    speed_title = f"Re={re_val} | t={t_actual:.2f} | |u|"
    dns_speed_snap = np.sqrt(dns_u_snap**2 + dns_v_snap**2)
    model_speed_snap = np.sqrt(model_u_interp**2 + model_v_interp**2)
    plot_speed_comparison(
        x_dns,
        y_dns,
        dns_speed_snap,
        model_speed_snap,
        output_subdir / f"speed_compare_t{t_actual:.2f}.png",
        speed_title,
    )

    spectrum_title = f"Re={re_val} | t={t_actual:.2f} | Energy Spectrum"
    plot_energy_spectrum_comparison(
        dns_u_snap,
        dns_v_snap,
        model_u_interp,
        model_v_interp,
        output_subdir / f"energy_spectrum_t{t_actual:.2f}.png",
        spectrum_title,
        model_label,
    )

    dns_speed = np.sqrt(dns_u**2 + dns_v**2)

    l2_u = []
    l2_v = []
    l2_speed = []
    for dns_idx, t_val in enumerate(dns_time):
        if model_time.size:
            model_idx = int(np.argmin(np.abs(model_time - t_val)))
        else:
            model_idx = 0

        model_u_frame = select_snapshot(model_u_series, model_idx)
        model_v_frame = select_snapshot(model_v_series, model_idx)
        model_u_interp = interpolate_to_dns_grid(model_u_frame, dns_u.shape[1:])
        model_v_interp = interpolate_to_dns_grid(model_v_frame, dns_u.shape[1:])
        model_speed = np.sqrt(model_u_interp**2 + model_v_interp**2)

        l2_u.append(compute_rel_l2(model_u_interp, dns_u[dns_idx]))
        l2_v.append(compute_rel_l2(model_v_interp, dns_v[dns_idx]))
        l2_speed.append(compute_rel_l2(model_speed, dns_speed[dns_idx]))

    errors = {
        "u": np.array(l2_u, dtype=float),
        "v": np.array(l2_v, dtype=float),
        "speed": np.array(l2_speed, dtype=float),
    }

    timeseries_title = f"Re={re_val} | Velocity L2 Error vs Time"
    plot_l2_timeseries(
        dns_time,
        errors,
        output_subdir / "velocity_l2_timeseries.png",
        timeseries_title,
    )

    write_l2_csv(dns_time, errors, output_subdir / "velocity_l2_timeseries.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Kolmogorov DNS vs model data (vorticity, spectrum, L2 errors)."
    )
    parser.add_argument(
        "--re-list",
        type=str,
        default="100,1000,10000,100000",
        help="Comma-separated Reynolds numbers.",
    )
    parser.add_argument(
        "--t-eval",
        type=float,
        default=20.0,
        help="DNS evaluation time (closest snapshot).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/kolmogorov_dns_model_compare",
        help="Output directory for figures and CSV.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="LES 檔案路徑（預設使用 Leith）。",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default="LES",
        help="圖例顯示名稱（例如 LES）。",
    )

    args = parser.parse_args()
    re_list = parse_re_list(args.re_list)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print(f"Kolmogorov DNS vs {args.model_label} comparison")
    print(f"Re list: {re_list}")
    print(f"t_eval: {args.t_eval}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    for re_val in re_list:
        print(f"Processing Re={re_val}...")
        compare_single_re(
            re_val,
            args.t_eval,
            output_dir,
            (0.0, 0.0),
            Path(args.model_path) if args.model_path else None,
            args.model_label,
            (0.0, 0.0),
        )
        print(f"Done Re={re_val}")

    print("=" * 80)
    print("All comparisons completed.")


if __name__ == "__main__":
    main()
