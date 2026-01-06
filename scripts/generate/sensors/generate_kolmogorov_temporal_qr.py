#!/usr/bin/env python3
"""
Kolmogorov Flow 時間序列 QR-Pivot 感測器生成
加入時間維度特徵：u(x,y,t), v(x,y,t), p(x,y,t), vorticity(x,y,t)
時間範圍：[15, 35] 秒，以 1 秒為單位

策略：
1. 從 DNS 載入時間序列數據（每秒一個快照）
2. 構建特徵矩陣：[N_spatial, N_features × N_time]
   - 每個空間點包含所有時間步的所有特徵
   - 特徵順序：[u_t0, u_t1, ..., v_t0, v_t1, ..., p_t0, p_t1, ..., ω_t0, ω_t1, ...]
3. 使用 QR-Pivot 選取空間點（考慮時間維度的訊息量）
4. 映射結果保存為 JSON 格式
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import json
from pinnx.sensors.qr_pivot import QRPivotSelector
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def load_dns_temporal_data(
    dns_file: str,
    time_range: tuple = (15.0, 35.0),
    time_stride: int = 10  # 每秒 1 個快照（DNS 時間步 dt=0.1）
):
    """
    從 DNS 載入時間序列數據
    
    Args:
        dns_file: DNS 資料檔案路徑
        time_range: 時間範圍 [t_start, t_end] (秒)
        time_stride: 時間採樣間隔（每幾個時間步取一個快照）
                     DNS dt=0.1s，time_stride=10 即每 1 秒一個快照
    
    Returns:
        coords: 空間座標 [N_spatial, 2] (x, y)
        features: 特徵矩陣 [N_spatial, N_features × N_time]
        grid_shape: 網格形狀 (nx, ny)
        time_selected: 選取的時間點陣列
    """
    logger.info(f"📂 載入 DNS 時間序列數據: {dns_file}")
    
    with h5py.File(dns_file, 'r') as f:
        # 讀取時間軸
        time_all = f['time'][:]
        
        # 選擇時間範圍與採樣間隔
        t_start, t_end = time_range
        time_mask = (time_all >= t_start) & (time_all <= t_end)
        time_indices = np.where(time_mask)[0][::time_stride]
        time_selected = time_all[time_indices]
        
        N_time = len(time_selected)
        logger.info(f"   時間範圍: [{t_start:.1f}, {t_end:.1f}] 秒")
        logger.info(f"   時間採樣: 每 {time_stride * 0.1:.1f} 秒一個快照")
        logger.info(f"   時間步數: {N_time}")
        
        # 讀取空間網格資訊
        N = int(f['config'].attrs['N'])
        L = float(f['config'].attrs['L'])
        
        logger.info(f"   空間解析度: {N} × {N}")
        logger.info(f"   計算域大小: {L:.2f} × {L:.2f}")
        
        # 建立空間座標網格
        x_1d = np.linspace(0, L, N, endpoint=False)
        y_1d = np.linspace(0, L, N, endpoint=False)
        X_mesh, Y_mesh = np.meshgrid(x_1d, y_1d, indexing='ij')
        
        # 展平空間座標
        coords = np.stack([X_mesh.ravel(), Y_mesh.ravel()], axis=1)  # [N*N, 2]
        N_spatial = N * N
        
        # 讀取時間序列數據 (只讀取選定的時間步)
        u_series = f['u'][time_indices, :, :]  # [N_time, N, N]
        v_series = f['v'][time_indices, :, :]
        p_series = f['p'][time_indices, :, :]
        
        logger.info(f"   已載入數據形狀: u={u_series.shape}")
        
        # 計算梯度和渦度時間序列
        logger.info(f"   計算梯度場與渦度場...")
        dx = x_1d[1] - x_1d[0]
        dy = y_1d[1] - y_1d[0]
        
        # 初始化梯度和渦度陣列
        dudx_series = np.zeros_like(u_series)
        dudy_series = np.zeros_like(u_series)
        dvdx_series = np.zeros_like(u_series)
        dvdy_series = np.zeros_like(u_series)
        dpdx_series = np.zeros_like(u_series)
        dpdy_series = np.zeros_like(u_series)
        vorticity_series = np.zeros_like(u_series)
        
        for i in range(N_time):
            # 週期性邊界條件的梯度計算
            dudx_series[i] = np.gradient(u_series[i], dx, axis=0)
            dudy_series[i] = np.gradient(u_series[i], dy, axis=1)
            dvdx_series[i] = np.gradient(v_series[i], dx, axis=0)
            dvdy_series[i] = np.gradient(v_series[i], dy, axis=1)
            dpdx_series[i] = np.gradient(p_series[i], dx, axis=0)
            dpdy_series[i] = np.gradient(p_series[i], dy, axis=1)
            # 渦度: ∂v/∂x - ∂u/∂y
            vorticity_series[i] = dvdx_series[i] - dudy_series[i]
        
        # 展平空間維度 [N_time, N*N]
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
        
        # 構建特徵矩陣 [N_spatial, N_features × N_time]
        # 策略：每個空間點包含所有時間步的所有特徵
        # 特徵順序：[u(t0), u(t1), ..., v(t0), ..., p(t0), ..., du/dx(t0), ..., du/dy(t0), ..., 
        #           dv/dx(t0), ..., dv/dy(t0), ..., dp/dx(t0), ..., dp/dy(t0), ..., ω(t0), ...]
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
            # field_data shape: [N_time, N_spatial]
            # 轉置後: [N_spatial, N_time]
            feature_list.append(field_data.T)
            logger.info(f"      {field_name}: {field_data.T.shape}")
        
        # 水平拼接所有特徵
        features = np.hstack(feature_list)  # [N_spatial, 4 × N_time]
        
        logger.info(f"   ✅ 特徵矩陣形狀: {features.shape}")
        logger.info(f"      (每個空間點包含 {features.shape[1]} 個特徵)")
        logger.info(f"      = 4 個物理量 × {N_time} 個時間步")
        
    return coords, features, (N, N), time_selected


def generate_temporal_sensors_for_K(
    dns_file: str,
    K: int,
    time_range: tuple = (15.0, 35.0),
    time_stride: int = 10,  # 每 1 秒一個快照
    output_dir: str = "./data/sensors/kolmogorov"
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
    
    # 1. 載入 DNS 時間序列數據
    coords, features, grid_shape, time_selected = load_dns_temporal_data(
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
    output_file = Path(output_dir) / f"sensors_temporal_K{K}_re50_256x256_t{int(time_range[0])}-{int(time_range[1])}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "indices": selected_indices.tolist(),
        "K": K,
        "resolution": f"{nx}x{ny}",
        "method": "QR-Pivot from DNS Time Series (Full Gradients)",
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
        "selected_coordinates": selected_coords.tolist()
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 已保存: {output_file}")
    logger.info("=" * 70 + "\n")


def main():
    """批次生成 K=30, 50, 80, 100 的時間序列感測器"""
    
    # 配置
    dns_file = "./data/kolmogorov_dns/dns_re50_t100.h5"
    K_values = [30, 50, 80, 100]
    time_range = (15.0, 35.0)  # 穩定態區間
    time_stride = 10  # 每 1 秒一個快照（dt=0.1s × 10 = 1s）
    output_dir = "./data/sensors/kolmogorov"
    
    logger.info("🚀 開始批次生成 Kolmogorov Flow 時間序列感測器\n")
    logger.info(f"DNS 檔案: {dns_file}")
    logger.info(f"時間範圍: [{time_range[0]:.1f}, {time_range[1]:.1f}] 秒")
    logger.info(f"時間採樣: 每 {time_stride * 0.1:.1f} 秒")
    logger.info(f"K 值: {K_values}\n")
    
    # 檢查 DNS 檔案
    if not Path(dns_file).exists():
        logger.error(f"❌ DNS 檔案不存在: {dns_file}")
        logger.error(f"   請確認 DNS 數據已生成")
        sys.exit(1)
    
    # 批次生成
    for K in K_values:
        try:
            generate_temporal_sensors_for_K(
                dns_file=dns_file,
                K=K,
                time_range=time_range,
                time_stride=time_stride,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"❌ K={K} 生成失敗: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info("\n✅ 所有時間序列感測器生成完成！")
    logger.info(f"📁 輸出目錄: {output_dir}")
    
    # 列出生成的檔案
    logger.info(f"\n📋 生成的感測器檔案:")
    for K in K_values:
        filename = f"sensors_temporal_K{K}_re50_256x256_t{int(time_range[0])}-{int(time_range[1])}.json"
        filepath = Path(output_dir) / filename
        if filepath.exists():
            logger.info(f"   ✓ {filename}")
        else:
            logger.info(f"   ✗ {filename} (未生成)")


if __name__ == "__main__":
    main()
