#!/usr/bin/env python3
"""
Kolmogorov Flow 時間序列 QR-Pivot 感測器生成（NPY）
加入時間維度特徵：u(x,y,t), v(x,y,t), p(x,y,t), vorticity(x,y,t)
時間範圍：[15, 35] 秒，以 1 秒為單位

策略：
1. 從 NPY 載入時間序列數據（每隔 time_stride 取一個快照）
2. 構建特徵矩陣：[N_spatial, N_features × N_time]
   - 每個空間點包含所有時間步的所有特徵
   - 特徵順序：[u_t0, u_t1, ..., v_t0, v_t1, ..., p_t0, p_t1, ..., ω_t0, ω_t1, ...]
3. 使用 QR-Pivot 選取空間點（考慮時間維度的訊息量）
4. 映射結果保存為 JSON 格式
"""

import sys
import argparse
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import json
from pinnx.sensors.qr_pivot import QRPivotSelector
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dns_temporal_data(
    dns_file: str,
    time_range: tuple = (15.0, 35.0),
    time_stride: int = 10  # 每隔 time_stride 個時間步取一個快照
):
    """
    從 NPY 載入時間序列數據

    Args:
        dns_file: DNS/LES NPY 資料檔案路徑
        time_range: 時間範圍 [t_start, t_end] (秒)
        time_stride: 時間採樣間隔（每幾個時間步取一個快照）

    Returns:
        coords: 空間座標 [N_spatial, 2] (x, y)
        features: 特徵矩陣 [N_spatial, N_features × N_time]
        grid_shape: 網格形狀 (nx, ny)
        time_selected: 選取的時間點陣列
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
    N_spatial = N * N

    u_series = np.asarray(data["u"], dtype=float)[time_indices]
    v_series = np.asarray(data["v"], dtype=float)[time_indices]
    p_series = np.asarray(data.get("p", np.zeros_like(u_series)), dtype=float)[time_indices]

    logger.info(f"   已載入數據形狀: u={u_series.shape}")

    logger.info(f"   計算梯度場與渦度場...")
    dx = x_1d[1] - x_1d[0]
    dy = y_1d[1] - y_1d[0]

    dudx_series = np.zeros_like(u_series)
    dudy_series = np.zeros_like(u_series)
    dvdx_series = np.zeros_like(u_series)
    dvdy_series = np.zeros_like(u_series)
    dpdx_series = np.zeros_like(u_series)
    dpdy_series = np.zeros_like(u_series)
    vorticity_series = np.zeros_like(u_series)

    for i in range(N_time):
        dudx_series[i] = np.gradient(u_series[i], dx, axis=0)
        dudy_series[i] = np.gradient(u_series[i], dy, axis=1)
        dvdx_series[i] = np.gradient(v_series[i], dx, axis=0)
        dvdy_series[i] = np.gradient(v_series[i], dy, axis=1)
        dpdx_series[i] = np.gradient(p_series[i], dx, axis=0)
        dpdy_series[i] = np.gradient(p_series[i], dy, axis=1)
        vorticity_series[i] = dvdx_series[i] - dudy_series[i]

    u_flat = u_series.reshape(N_time, N_spatial)
    v_flat = v_series.reshape(N_time, N_spatial)
    p_flat = p_series.reshape(N_time, N_spatial)
    dudx_flat = dudx_series.reshape(N_time, N_spatial)
    dudy_flat = dudy_series.reshape(N_time, N_spatial)
    dvdx_flat = dvdx_series.reshape(N_time, N_spatial)
    dvdy_flat = dvdy_series.reshape(N_time, N_spatial)
    dpdx_flat = dpdx_series.reshape(N_time, N_spatial)
    dpdy_flat = dpdy_series.reshape(N_time, N_spatial)
    vorticity_flat = vorticity_series.reshape(N_time, N_spatial)

    logger.info(f"   構建時空特徵矩陣...")

    feature_list = []
    for field_name, field_data in [
        ('u', u_flat),
        ('v', v_flat),
        ('p', p_flat),
        ('du/dx', dudx_flat),
        ('du/dy', dudy_flat),
        ('dv/dx', dvdx_flat),
        ('dv/dy', dvdy_flat),
        ('dp/dx', dpdx_flat),
        ('dp/dy', dpdy_flat),
        ('vorticity_z', vorticity_flat)
    ]:
        feature_list.append(field_data.T)
        logger.info(f"      {field_name}: {field_data.T.shape}")

    features = np.hstack(feature_list)

    logger.info(f"   ✅ 特徵矩陣形狀: {features.shape}")
    logger.info(f"      (每個空間點包含 {features.shape[1]} 個特徵)")
    logger.info(f"      = 10 個物理量 × {N_time} 個時間步")

    config = data.get("config", {})
    return coords, features, (N, N), time_selected, config


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
    time_stride: int = 10,  # 每隔 time_stride 個時間步
    output_dir: str = "./data/sensors/kolmogorov",
    dns_values_file: str | None = None,
    include_dns_values: bool = False,
):
    """
    為指定 K 值生成時間序列感測器
    
    流程:
    1. 從 DNS 載入時間序列數據（包含 u, v, p 及其梯度）
    2. 構建時空特徵矩陣 [N_spatial, N_features × N_time]
       特徵: u, v, p, du/dx, du/dy, dv/dx, dv/dy, dp/dx, dp/dy, vorticity_z
    3. 使用 QR-Pivot 選取 K 個空間點
    4. 保存為 JSON 格式
    """
    logger.info("=" * 70)
    logger.info(f"🎯 生成時間序列 K={K} 感測器")
    logger.info("=" * 70)
    
    # 1. 載入時間序列數據
    coords, features, grid_shape, time_selected, config = load_dns_temporal_data(
        dns_file=dns_file,
        time_range=time_range,
        time_stride=time_stride
    )
    
    N_spatial = coords.shape[0]
    N_time = len(time_selected)
    
    # 2. 使用 QR-Pivot 選點
    logger.info(f"\n🔍 執行 QR-Pivot 選點（基於時空特徵）...")
    selector = QRPivotSelector()
    
    try:
        selected_indices, metrics = selector.select_sensors(
            data_matrix=features,
            n_sensors=K
        )
        
        logger.info(f"   ✅ 選取 {len(selected_indices)} 個空間點")
        logger.info(f"   條件數: {metrics.get('condition_number', 'N/A'):.2e}")
        logger.info(f"   覆蓋率: {metrics.get('subspace_coverage', 'N/A'):.4f}")
        logger.info(f"   能量比: {metrics.get('energy_ratio', 'N/A'):.4f}")
        
    except Exception as e:
        logger.error(f"   ❌ QR-Pivot 失敗: {e}")
        raise
    
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
        f"sensors_temporal_K{K}_re{re_val}_N{nx}_t{int(time_range[0])}-{int(time_range[1])}.json"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "indices": selected_indices.tolist(),
        "K": K,
        "resolution": f"{nx}x{ny}",
        "method": "QR-Pivot from Time Series (Full Gradients)",
        "time_range": list(time_range),
        "time_steps": int(N_time),
        "time_stride": time_stride,
        "time_selected": time_selected.tolist(),
        "features": ["u", "v", "p", "du/dx", "du/dy", "dv/dx", "dv/dy", "dp/dx", "dp/dy", "vorticity_z"],
        "n_features_per_time": 10,
        "total_features": int(features.shape[1]),
        "condition_number": float(metrics.get('condition_number', -1)),
        "subspace_coverage": float(metrics.get('subspace_coverage', -1)),
        "energy_ratio": float(metrics.get('energy_ratio', -1)),
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
    """批次生成時間序列感測器"""
    parser = argparse.ArgumentParser(description="Generate temporal QR-Pivot sensors from NPY data")
    parser.add_argument("--input", type=str, required=True, help="輸入 NPY 檔案")
    parser.add_argument("--output", type=str, default="./data/sensors/kolmogorov", help="輸出資料夾")
    parser.add_argument("--K", type=int, nargs="+", default=[30, 50, 80, 100], help="感測器數量列表")
    parser.add_argument("--time-range", type=float, nargs=2, default=[15.0, 35.0], help="時間範圍")
    parser.add_argument("--time-stride", type=int, default=10, help="時間採樣間隔")
    parser.add_argument("--include-dns-values", action="store_true", help="附加 DNS time series values")
    parser.add_argument("--dns-values", type=str, default=None, help="DNS NPY 檔案路徑（用於 values）")

    args = parser.parse_args()

    data_file = args.input
    K_values = args.K
    time_range = tuple(args.time_range)
    time_stride = args.time_stride
    output_dir = args.output

    logger.info("🚀 開始批次生成 Kolmogorov Flow 時間序列感測器\n")
    logger.info(f"資料檔案: {data_file}")
    logger.info(f"時間範圍: [{time_range[0]:.1f}, {time_range[1]:.1f}] 秒")
    logger.info(f"時間採樣: 每 {time_stride} 個時間步")
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
        filename = f"sensors_temporal_K{K}_re{re_val}_N{nx}_t{int(time_range[0])}-{int(time_range[1])}.json"
        filepath = Path(output_dir) / filename
        if filepath.exists():
            logger.info(f"   ✓ {filename}")
        else:
            logger.info(f"   ✗ {filename} (未生成)")


if __name__ == "__main__":
    main()
