#!/usr/bin/env python3
"""
Kolmogorov Flow 感測器視覺化（簡潔版）
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === 資料載入 ===

def load_sensor_json(file_path: Path) -> dict:
    """載入感測器 JSON。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# === 繪圖 ===


def plot_background_grid(ax, grid_points=16) -> None:
    """繪製背景網格（固定 [0, 2π]）。"""
    x_vals = np.linspace(0.0, 2 * np.pi, grid_points)
    y_vals = np.linspace(0.0, 2 * np.pi, grid_points)
    for x in x_vals:
        ax.axvline(x, color="lightgray", linewidth=0.5, alpha=0.4, zorder=0)
    for y in y_vals:
        ax.axhline(y, color="lightgray", linewidth=0.5, alpha=0.4, zorder=0)


def plot_sensors(coords: np.ndarray, output_path: Path, title: str) -> None:
    """繪製感測器座標散點圖。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_background_grid(ax)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, c="tab:red", alpha=0.8, zorder=2)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(0.0, 2 * np.pi)
    ax.set_ylim(0.0, 2 * np.pi)
    ax.grid(False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def plot_sensors_grid(inputs, output_path: Path, cols: int = 3) -> None:
    """多子圖輸出。"""
    n_files = len(inputs)
    rows = int(np.ceil(n_files / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))

    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, (input_path, label) in enumerate(inputs):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        data = load_sensor_json(input_path)
        coords = np.asarray(data.get("selected_coordinates"), dtype=float)
        if coords.size == 0:
            raise ValueError(f"JSON 缺少 selected_coordinates: {input_path}")

        plot_background_grid(ax)
        ax.scatter(coords[:, 0], coords[:, 1], s=10, c="tab:red", alpha=0.8, zorder=2)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(label)
        ax.set_aspect("equal")
        ax.set_xlim(0.0, 2 * np.pi)
        ax.set_ylim(0.0, 2 * np.pi)
        ax.grid(False)

    for idx in range(n_files, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """主流程。"""
    parser = argparse.ArgumentParser(description="Visualize Kolmogorov sensor locations")
    parser.add_argument("--input", type=str, nargs="+", required=True, help="感測器 JSON 檔案")
    parser.add_argument("--output", type=str, required=True, help="輸出圖檔路徑")
    parser.add_argument("--cols", type=int, default=3, help="每列子圖數")

    args = parser.parse_args()

    inputs = []
    for path in args.input:
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"找不到檔案: {input_path}")
        data = load_sensor_json(input_path)
        label = f"K={data.get('K', 'N/A')}"
        inputs.append((input_path, label))

    plot_sensors_grid(inputs, Path(args.output), cols=args.cols)


if __name__ == "__main__":
    main()
