#!/usr/bin/env python3
"""
Compare JHTDB channel-flow snapshot vs RANS mean field on a 2D z-slice.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import RegularGridInterpolator


def load_dns_cutout(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    grid_shape = tuple(int(v) for v in data["grid_shape"])
    u = data["u"].reshape(grid_shape)
    v = data["v"].reshape(grid_shape)
    w = data["w"].reshape(grid_shape)
    p = data["p"].reshape(grid_shape)
    return {"x": x, "y": y, "z": z, "u": u, "v": v, "w": w, "p": p}


def load_rans(path: Path) -> Dict[str, np.ndarray]:
    data = np.load(path)
    return {
        "x": data["x"],
        "y": data["y"],
        "z": data["z"],
        "u": data["u"],
        "v": data["v"],
        "w": data["w"],
        "p": data["p"],
    }


def interpolate_rans_to_dns(
    rans: Dict[str, np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> Dict[str, np.ndarray]:
    xi, yi, zi = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=-1)

    out = {}
    for key in ("u", "v", "w", "p"):
        interp = RegularGridInterpolator(
            (rans["x"], rans["y"], rans["z"]),
            rans[key],
            bounds_error=False,
            fill_value=np.nan,
        )
        vals = interp(points).reshape(xi.shape)
        out[key] = vals
    return out


def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    mask = np.isfinite(pred) & np.isfinite(ref)
    if not np.any(mask):
        return float("nan")
    return float(np.linalg.norm(pred[mask] - ref[mask]) / np.linalg.norm(ref[mask]) * 100)


def plot_slice(
    x: np.ndarray,
    y: np.ndarray,
    dns: np.ndarray,
    rans: np.ndarray,
    title: str,
    output_path: Path,
    cmap_field: str,
    use_diverging: bool,
) -> None:
    X, Y = np.meshgrid(x, y, indexing="ij")
    error = np.abs(rans - dns)

    if use_diverging:
        vmin = np.nanmin([dns.min(), rans.min()])
        vmax = np.nanmax([dns.max(), rans.max()])
        abs_max = max(abs(vmin), abs(vmax))
        norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    else:
        norm = None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle(title, fontsize=12, fontweight="bold")

    im = axes[0].contourf(X, Y, dns, levels=50, cmap=cmap_field, norm=norm)
    axes[0].set_title("DNS", fontweight="bold")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_aspect("equal")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].contourf(X, Y, rans, levels=50, cmap=cmap_field, norm=norm)
    axes[1].set_title("RANS", fontweight="bold")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_aspect("equal")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im = axes[2].contourf(X, Y, error, levels=50, cmap="hot_r")
    axes[2].set_title("|Error|", fontweight="bold")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    axes[2].set_aspect("equal")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare JHTDB snapshot vs RANS mean field.")
    parser.add_argument(
        "--dns",
        type=Path,
        default=Path("data/jhtdb/channel_flow_re1000/cutout_128x64x128.npz"),
        help="JHTDB cutout NPZ (single snapshot).",
    )
    parser.add_argument(
        "--rans",
        type=Path,
        default=Path("data/lowfi/channel_rans/rans_k_omega_sst.npz"),
        help="RANS mean field NPZ.",
    )
    parser.add_argument(
        "--z-index",
        type=int,
        default=None,
        help="Z-slice index for plotting (default: center).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("thesis/result_figures/channel_flow/rans_dns_snapshot"),
        help="Output directory for figures.",
    )
    args = parser.parse_args()

    dns = load_dns_cutout(args.dns)
    rans = load_rans(args.rans)

    print("DNS grid:", dns["u"].shape, "RANS grid:", rans["u"].shape)
    print("DNS x/y/z:", dns["x"].min(), dns["x"].max(), dns["y"].min(), dns["y"].max(), dns["z"].min(), dns["z"].max())
    print("RANS x/y/z:", rans["x"].min(), rans["x"].max(), rans["y"].min(), rans["y"].max(), rans["z"].min(), rans["z"].max())

    rans_interp = interpolate_rans_to_dns(rans, dns["x"], dns["y"], dns["z"])

    speed_dns = np.sqrt(dns["u"] ** 2 + dns["v"] ** 2 + dns["w"] ** 2)
    speed_rans = np.sqrt(rans_interp["u"] ** 2 + rans_interp["v"] ** 2 + rans_interp["w"] ** 2)

    errors = {
        "u": relative_l2(rans_interp["u"], dns["u"]),
        "v": relative_l2(rans_interp["v"], dns["v"]),
        "w": relative_l2(rans_interp["w"], dns["w"]),
        "p": relative_l2(rans_interp["p"], dns["p"]),
        "|u|": relative_l2(speed_rans, speed_dns),
    }

    print("Relative L2 (%):")
    for key in ("u", "v", "w", "p", "|u|"):
        print(f"  {key}: {errors[key]:.2f}")

    z_idx = args.z_index
    if z_idx is None:
        z_idx = dns["z"].size // 2
    z_val = float(dns["z"][z_idx])

    fields = {
        "u": (dns["u"][:, :, z_idx], rans_interp["u"][:, :, z_idx], True),
        "v": (dns["v"][:, :, z_idx], rans_interp["v"][:, :, z_idx], True),
        "w": (dns["w"][:, :, z_idx], rans_interp["w"][:, :, z_idx], True),
        "p": (dns["p"][:, :, z_idx], rans_interp["p"][:, :, z_idx], True),
        "|u|": (speed_dns[:, :, z_idx], speed_rans[:, :, z_idx], False),
    }

    for name, (dns_slice, rans_slice, diverging) in fields.items():
        out = args.output_dir / f"channel_rans_dns_{name}_z{z_idx}.png"
        title = f"Channel Flow: {name} (z={z_val:.3f}, L2={errors[name]:.1f}%)"
        cmap = "RdBu_r" if diverging else "viridis"
        plot_slice(dns["x"], dns["y"], dns_slice, rans_slice, title, out, cmap, diverging)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
