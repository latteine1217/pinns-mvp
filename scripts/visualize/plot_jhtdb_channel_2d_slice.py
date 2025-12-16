#!/usr/bin/env python3
"""
Plot reproducible 2D JHTDB channel-flow slice figures.

Generates:
  - DNS reference slice (single field)
  - Optional sensor-layout overlay on the same slice

Inputs are versioned artifacts in this repo:
  - data/jhtdb/channel_flow_re1000/eval_2d_slice.npz
  - data/jhtdb/channel_flow_re1000/sensors_K*_qr_pivot_2d*.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Literal

import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    repo_root = Path(__file__).resolve().parents[2]
    mpl_cache_dir = repo_root / ".cache" / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt

FieldName = Literal["u", "v", "w", "p"]


def _load_slice(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        out = {k: data[k] for k in data.files}
    if "p" in out and out["p"].ndim == 3 and out["p"].shape[-1] == 1:
        out["p"] = np.squeeze(out["p"], axis=-1)
    return out


def _load_sensors(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(str(npz_path), allow_pickle=True) as data:
        if "sensor_x" in data and "sensor_y" in data:
            return data["sensor_x"].astype(np.float64), data["sensor_y"].astype(np.float64)
        if "coords" in data:
            coords = data["coords"]
            if coords.ndim != 2 or coords.shape[1] < 2:
                raise ValueError(f"Unexpected coords shape: {coords.shape}")
            return coords[:, 0].astype(np.float64), coords[:, 1].astype(np.float64)
    raise KeyError("Sensor npz must contain (sensor_x, sensor_y) or coords")


def _plot_field(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    out_path: Path,
    sensors_xy: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 300,
) -> None:
    if field.ndim != 2:
        raise ValueError(f"Expected 2D field array, got shape {field.shape}")
    if field.shape != (x.size, y.size):
        raise ValueError(
            f"Field shape {field.shape} does not match (len(x), len(y)) = ({x.size}, {y.size})"
        )

    fig, ax = plt.subplots(figsize=(16, 3.2))
    im = ax.imshow(
        field.T,
        extent=[float(x.min()), float(x.max()), float(y.min()), float(y.max())],
        origin="lower",
        aspect="auto",
        cmap="viridis",
        interpolation="bilinear",
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("x (streamwise)", fontsize=14)
    ax.set_ylabel("y (wall-normal)", fontsize=14)

    if sensors_xy is not None:
        sx, sy = sensors_xy
        ax.scatter(
            sx,
            sy,
            s=28,
            facecolors="none",
            edgecolors="white",
            linewidths=1.2,
            alpha=0.95,
        )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label, fontsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D JHTDB channel-flow slice figures from npz.")
    parser.add_argument(
        "--slice-npz",
        type=Path,
        default=Path("data/jhtdb/channel_flow_re1000/eval_2d_slice.npz"),
        help="Path to eval 2D slice npz (x,y,u,v,w,p).",
    )
    parser.add_argument(
        "--field",
        choices=("u", "v", "w", "p"),
        default="u",
        help="Which field to plot.",
    )
    parser.add_argument(
        "--sensors-npz",
        type=Path,
        default=Path("data/jhtdb/channel_flow_re1000/sensors_K100_qr_pivot_2d.npz"),
        help="Optional sensor layout npz (sensor_x, sensor_y).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("thesis/result_figures"),
        help="Output directory for PNGs.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Only write the reference slice figure (skip sensor overlay).",
    )
    args = parser.parse_args()

    slice_data = _load_slice(args.slice_npz)
    x = slice_data["x"].astype(np.float64)
    y = slice_data["y"].astype(np.float64)
    field_name: FieldName = args.field
    field = slice_data[field_name].astype(np.float64)

    ref_out = args.outdir / f"channel_jhtdb_reference_{field_name}.png"
    _plot_field(
        x,
        y,
        field,
        title=f"JHTDB DNS reference slice: streamwise velocity {field_name}" if field_name == "u" else f"JHTDB DNS reference slice: {field_name}",
        cbar_label=f"{field_name} (normalized)",
        out_path=ref_out,
        sensors_xy=None,
        dpi=args.dpi,
    )

    if args.no_overlay:
        return

    sensors_xy = _load_sensors(args.sensors_npz)
    overlay_out = args.outdir / f"channel_sensors_K{len(sensors_xy[0])}_overlay_{field_name}.png"
    _plot_field(
        x,
        y,
        field,
        title=f"QR-pivot sensor layout (K={len(sensors_xy[0])}) over DNS {field_name} slice",
        cbar_label=f"{field_name} (normalized)",
        out_path=overlay_out,
        sensors_xy=sensors_xy,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
