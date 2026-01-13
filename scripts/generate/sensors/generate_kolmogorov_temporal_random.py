#!/usr/bin/env python3
"""
Kolmogorov Flow 時間序列 Random Sampling 感測器生成（NPY）
使用隨機取樣策略選取感測器位置，作為 QR-Pivot 的對照基準

策略：
1. 從 NPY 載入時間序列數據
2. 使用隨機取樣選取空間點（不考慮時空特徵訊息量）
3. 映射結果保存為 JSON 格式
4. 與 QR-Pivot 感測器保持相同的檔案格式以便對比
"""

import sys
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dns_temporal_data(
    dns_file: str,
    time_range: tuple = (15.0, 35.0),
    time_stride: int = 10
):
    """
    從 NPY 載入時間序列數據

    Args:
        dns_file: DNS/LES NPY 資料檔案路徑
        time_range: 時間範圍 [t_start, t_end] (秒)
        time_stride: 時間採樣間隔（每幾個時間步取一個快照）

    Returns:
        coords: 空間座標 [N_spatial, 2] (x, y)
        grid_shape: 網格形狀 (nx, ny)
        time_selected: 選取的時間點陣列
        config: 配置參數
    """
    logger.info(f"📂 載入時間序列數據: {dns_file}")

    payload = np.load(dns_file, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(f"NPY 格式錯誤: {dns_file}")
    time_all = np.asarray(data["time"], dtype=float)

    t_start, t_end = time_range
    time_mask = (time_all >= t_start) & (time_all <= t_end)
    time_indices = np.where(time_mask)[0][::time_stride]
    time_selected = time_all[time_indices]

    N_time = len(time_selected)
    logger.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}] 秒")
    logger.info(f"   時間採樣: 每 {time_stride} 個時間步")
    logger.info(f"   時間步數: {N_time}")

    x_1d = np.asarray(data.get("x"), dtype=float)
    y_1d = np.asarray(data.get("y"), dtype=float)
    if x_1d.size == 0 or y_1d.size == 0:
        config = data.get("config", {})
        N = int(config.get("N", data["u"].shape[-1]))
        L = float(config.get("L", 2 * np.pi))
        x_1d = np.linspace(0, L, N, endpoint=False)
        y_1d = np.linspace(0, L, N, endpoint=False)
    else:
        N = len(x_1d)
        L = float(np.max(x_1d) - np.min(x_1d) + (x_1d[1] - x_1d[0]))

    logger.info(f"   空間解析度: {N} × {N}")
    logger.info(f"   計算域大小: {L:.2f} × {L:.2f}")

    X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
    coords = np.stack([X_mesh.ravel(), Y_mesh.ravel()], axis=1)  # [N*N, 2]

    config = data.get("config", {})
    return coords, (N, N), time_selected, config


def compute_re_from_config(config: dict) -> int:
    """估算 Kolmogorov flow Re。"""
    nu = float(config.get("nu", 0.0))
    if nu == 0.0:
        return 0
    A = float(config.get("A", 0.1))
    k_f = float(config.get("k_f", 4.0))
    L = float(config.get("L", 2 * np.pi))
    re_val = np.sqrt(A) * (L / k_f) ** 1.5 / nu
    return int(round(re_val))


def extract_dns_values(
    dns_file: str,
    indices: np.ndarray,
    time_range: tuple,
    time_stride: int,
) -> dict:
    """擷取 DNS time series values。"""
    payload = np.load(dns_file, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    if not isinstance(data, dict):
        raise ValueError(f"DNS 檔案格式錯誤: {dns_file}")

    time_all = np.asarray(data["time"], dtype=float)
    t_start, t_end = time_range
    time_mask = (time_all >= t_start) & (time_all <= t_end)
    time_indices = np.where(time_mask)[0][::time_stride]
    time_selected = time_all[time_indices]

    u_series = np.asarray(data["u"], dtype=float)[time_indices]
    v_series = np.asarray(data["v"], dtype=float)[time_indices]
    p_series = np.asarray(data.get("p", np.zeros_like(u_series)), dtype=float)[time_indices]
    omega_series = np.asarray(data.get("omega", np.zeros_like(u_series)), dtype=float)[time_indices]

    ny = u_series.shape[2]
    i_indices = indices // ny
    j_indices = indices % ny

    def gather(field):
        return field[:, i_indices, j_indices].T

    return {
        "time": time_selected,
        "u": gather(u_series),
        "v": gather(v_series),
        "p": gather(p_series),
        "omega": gather(omega_series),
    }


def generate_temporal_sensors_for_K(
    dns_file: str,
    K: int,
    time_range: tuple = (15.0, 35.0),
    time_stride: int = 10,
    output_dir: str = "./data/sensors/kolmogorov",
    dns_values_file: str | None = None,
    include_dns_values: bool = False,
    seed: int = 42,
):
    """
    為指定 K 值生成時間序列感測器（隨機取樣）

    流程:
    1. 從 DNS 載入時間序列數據
    2. 使用隨機取樣選取 K 個空間點
    3. 保存為 JSON 格式（與 QR-Pivot 格式相同）
    """
    logger.info("=" * 70)
    logger.info(f"🎯 生成時間序列 K={K} 感測器（隨機取樣）")
    logger.info("=" * 70)

    # 1. 載入時間序列數據
    coords, grid_shape, time_selected, config = load_dns_temporal_data(
        dns_file=dns_file,
        time_range=time_range,
        time_stride=time_stride
    )

    N_spatial = coords.shape[0]
    N_time = len(time_selected)

    # 2. 使用隨機取樣選點
    logger.info(f"\n🎲 執行隨機取樣（seed={seed}）...")
    np.random.seed(seed)
    selected_indices = np.random.choice(N_spatial, size=K, replace=False)
    selected_indices = np.sort(selected_indices)  # 排序以便對比

    logger.info(f"   ✅ 選取 {len(selected_indices)} 個空間點")

    # 3. 提取選定點的座標
    selected_coords = coords[selected_indices]

    # 轉換為 (i, j) 索引（用於驗證）
    nx, ny = grid_shape
    i_indices = selected_indices // ny
    j_indices = selected_indices % ny

    logger.info(f"\n📍 選定點的空間分布:")
    logger.info(f"   x 範圍: [{selected_coords[:, 0].min():.2f}, {selected_coords[:, 0].max():.2f}]")
    logger.info(f"   y 範圍: [{selected_coords[:, 1].min():.2f}, {selected_coords[:, 1].max():.2f}]")
    logger.info(f"   i 範圍: [{i_indices.min()}, {i_indices.max()}]")
    logger.info(f"   j 範圍: [{j_indices.min()}, {j_indices.max()}]")

    # 4. 保存為 JSON
    re_val = compute_re_from_config(config)
    output_file = Path(output_dir) / (
        f"sensors_temporal_random_K{K}_re{re_val}_N{nx}_t{int(time_range[0])}-{int(time_range[1])}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "indices": selected_indices.tolist(),
        "K": K,
        "resolution": f"{nx}x{ny}",
        "method": "Random Sampling",
        "time_range": list(time_range),
        "time_steps": int(N_time),
        "time_stride": time_stride,
        "time_selected": time_selected.tolist(),
        "seed": seed,
        "condition_number": -1,  # N/A for random sampling
        "subspace_coverage": -1,  # N/A for random sampling
        "energy_ratio": -1,  # N/A for random sampling
        "selected_coordinates": selected_coords.tolist(),
        "source_file": str(dns_file),
        "config": config,
    }

    dns_values_path = None
    if include_dns_values:
        if dns_values_file is None:
            dns_values_file = config.get("dns_source") if isinstance(config, dict) else None
        if dns_values_file is None:
            raise ValueError("缺少 DNS 檔案，請提供 --dns-values")

        dns_values = extract_dns_values(
            dns_values_file,
            selected_indices,
            time_range,
            time_stride,
        )

        dns_values_path = output_file.with_suffix("").with_name(output_file.stem + "_dns_values.npz")
        np.savez_compressed(
            dns_values_path,
            time=dns_values["time"],
            u=dns_values["u"],
            v=dns_values["v"],
            p=dns_values["p"],
            omega=dns_values["omega"],
        )

        output_data["dns_values_npz"] = str(dns_values_path)
        output_data["dns_values_source"] = str(dns_values_file)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\n💾 已保存: {output_file}")
    logger.info("=" * 70 + "\n")


def main():
    """批次生成時間序列感測器（隨機取樣）"""
    parser = argparse.ArgumentParser(description="Generate temporal Random Sampling sensors from NPY data")
    parser.add_argument("--input", type=str, required=True, help="輸入 NPY 檔案")
    parser.add_argument("--output", type=str, default="./data/sensors/kolmogorov", help="輸出資料夾")
    parser.add_argument("--K", type=int, nargs="+", default=[50, 100, 200, 400], help="感測器數量列表")
    parser.add_argument("--time-range", type=float, nargs=2, default=[0.0, 20.0], help="時間範圍")
    parser.add_argument("--time-stride", type=int, default=1, help="時間採樣間隔")
    parser.add_argument("--include-dns-values", action="store_true", help="附加 DNS time series values")
    parser.add_argument("--dns-values", type=str, default=None, help="DNS NPY 檔案路徑（用於 values）")
    parser.add_argument("--seed", type=int, default=42, help="隨機種子")

    args = parser.parse_args()

    data_file = args.input
    K_values = args.K
    time_range = tuple(args.time_range)
    time_stride = args.time_stride
    output_dir = args.output
    seed = args.seed

    logger.info("🚀 開始批次生成 Kolmogorov Flow 時間序列感測器（隨機取樣）\n")
    logger.info(f"資料檔案: {data_file}")
    logger.info(f"時間範圍: [{time_range[0]:.1f}, {time_range[1]:.1f}] 秒")
    logger.info(f"時間採樣: 每 {time_stride} 個時間步")
    logger.info(f"隨機種子: {seed}")
    logger.info(f"K 值: {K_values}\n")

    if not Path(data_file).exists():
        logger.error(f"❌ 資料檔案不存在: {data_file}")
        logger.error("   請確認資料已生成")
        sys.exit(1)

    payload = np.load(data_file, allow_pickle=True)
    data = payload.item() if hasattr(payload, "item") else payload
    config = data.get("config", {}) if isinstance(data, dict) else {}
    re_val = compute_re_from_config(config)
    nx = int(config.get("N", 0)) if isinstance(config, dict) else 0

    for K in K_values:
        try:
            generate_temporal_sensors_for_K(
                dns_file=data_file,
                K=K,
                time_range=time_range,
                time_stride=time_stride,
                output_dir=output_dir,
                dns_values_file=args.dns_values,
                include_dns_values=args.include_dns_values,
                seed=seed,
            )
        except Exception as e:
            logger.error(f"❌ K={K} 生成失敗: {e}")
            import traceback
            traceback.print_exc()
            continue

    logger.info("\n✅ 所有時間序列感測器生成完成！")
    logger.info(f"📁 輸出目錄: {output_dir}")

    logger.info(f"\n📋 生成的感測器檔案:")
    for K in K_values:
        filename = f"sensors_temporal_random_K{K}_re{re_val}_N{nx}_t{int(time_range[0])}-{int(time_range[1])}.json"
        filepath = Path(output_dir) / filename
        if filepath.exists():
            logger.info(f"   ✓ {filename}")
        else:
            logger.info(f"   ✗ {filename} (未生成)")


if __name__ == "__main__":
    main()
