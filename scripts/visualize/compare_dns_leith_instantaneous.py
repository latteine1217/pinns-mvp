#!/usr/bin/env python3
"""
Compare DNS instantaneous snapshot with Leith mean field for Kolmogorov flow.

Generates DNS / Leith / |Error| panels for u, v, and speed magnitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.ndimage import zoom


def load_dns_snapshot(h5_path: Path, t_eval: float) -> Tuple[Dict[str, float], float, np.ndarray, np.ndarray, np.ndarray]:
    """Load DNS snapshot closest to t_eval."""
    with h5py.File(h5_path, "r") as f:
        cfg = dict(f["config"].attrs)
        t = np.array(f["time"])
        t_idx = int(np.argmin(np.abs(t - t_eval)))
        actual_t = float(t[t_idx])
        u = np.array(f["u"][t_idx])
        v = np.array(f["v"][t_idx])
    return cfg, actual_t, u, v, np.sqrt(u**2 + v**2)


def load_leith_mean(h5_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load Leith mean field (u, v, speed)."""
    with h5py.File(h5_path, "r") as f:
        u = np.array(f["mean_field/u"])
        v = np.array(f["mean_field/v"])
    return u, v, np.sqrt(u**2 + v**2)


def interpolate_to_dns_grid(field: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Interpolate Leith field to DNS grid using cubic zoom."""
    zoom_factor = target_shape[0] / field.shape[0]
    if zoom_factor == 1.0:
        return field
    return zoom(field, zoom_factor, order=3)


def compute_rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute relative L2 error in percent."""
    return float(np.linalg.norm(pred - ref) / np.linalg.norm(ref) * 100)


def plot_comparison(
    x: np.ndarray,
    y: np.ndarray,
    dns_fields: Dict[str, np.ndarray],
    leith_fields: Dict[str, np.ndarray],
    errors: Dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    """Generate DNS / Leith / |Error| panels for u, v, and speed."""
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    X, Y = np.meshgrid(x, y, indexing="ij")
    rows = [("u", "U"), ("v", "V"), ("speed", "|U|")]

    for row_idx, (key, label) in enumerate(rows):
        dns = dns_fields[key]
        leith = leith_fields[key]
        error = np.abs(leith - dns)

        if key == "speed":
            cmap_field = "viridis"
            norm = None
        else:
            vmin = min(dns.min(), leith.min())
            vmax = max(dns.max(), leith.max())
            abs_max = max(abs(vmin), abs(vmax))
            norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
            cmap_field = "RdBu_r"

        ax = axes[row_idx, 0]
        im = ax.contourf(X, Y, dns, levels=50, cmap=cmap_field, norm=norm)
        ax.set_title(f"{label} DNS", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx, 1]
        im = ax.contourf(X, Y, leith, levels=50, cmap=cmap_field, norm=norm)
        ax.set_title(f"{label} Leith", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax = axes[row_idx, 2]
        im = ax.contourf(X, Y, error, levels=50, cmap="hot_r")
        ax.set_title(f"{label} |Error| (L2 {errors[key]:.1f}%)", fontweight="bold")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_re_list(text: str) -> List[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DNS instantaneous snapshot with Leith mean field."
    )
    parser.add_argument(
        "--re-list",
        type=str,
        default="50,70,100",
        help="Comma-separated Reynolds numbers (e.g., 50,70,100).",
    )
    parser.add_argument("--t-eval", type=float, default=25.0, help="DNS evaluation time.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="thesis/result_figures/kolmogorov_leith_instantaneous",
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    re_list = parse_re_list(args.re_list)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("DNS vs Leith (Instantaneous DNS Snapshot) Comparison")
    print("=" * 80)

    for re_val in re_list:
        dns_path = Path(f"data/kolmogorov_dns/dns_re{re_val}_t100.h5")
        leith_path = Path(f"data/lowfi/kolmogorov_leith/rans_re{re_val}_kf4_leith.h5")

        if not dns_path.exists():
            print(f"❌ DNS file not found: {dns_path}")
            continue
        if not leith_path.exists():
            print(f"⚠️  Leith file not found: {leith_path}")
            continue

        cfg, actual_t, u_dns, v_dns, speed_dns = load_dns_snapshot(dns_path, args.t_eval)
        u_leith, v_leith, speed_leith = load_leith_mean(leith_path)

        u_leith = interpolate_to_dns_grid(u_leith, u_dns.shape)
        v_leith = interpolate_to_dns_grid(v_leith, v_dns.shape)
        speed_leith = np.sqrt(u_leith**2 + v_leith**2)

        errors = {
            "u": compute_rel_l2(u_leith, u_dns),
            "v": compute_rel_l2(v_leith, v_dns),
            "speed": compute_rel_l2(speed_leith, speed_dns),
        }

        L = float(cfg.get("L", 2 * np.pi))
        N = int(cfg.get("N", u_dns.shape[0]))
        x = np.linspace(0, L, N, endpoint=False)
        y = np.linspace(0, L, N, endpoint=False)

        title = f"Re={re_val} | t={actual_t:.2f} | DNS vs Leith (mean)"
        output_path = output_dir / f"leith_dns_instant_re{re_val}_t{actual_t:.2f}.png"

        plot_comparison(
            x,
            y,
            {"u": u_dns, "v": v_dns, "speed": speed_dns},
            {"u": u_leith, "v": v_leith, "speed": speed_leith},
            errors,
            title,
            output_path,
        )

        print(f"✅ Re={re_val} done: {output_path}")
        print(
            f"   L2% | u: {errors['u']:.1f}, v: {errors['v']:.1f}, |U|: {errors['speed']:.1f}"
        )

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
