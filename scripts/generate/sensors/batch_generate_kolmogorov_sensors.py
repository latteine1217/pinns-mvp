#!/usr/bin/env python3
"""
批次生成不同 K 值的 Kolmogorov Flow 感測器
從 RANS 數據使用 QR-Pivot 選點，映射到 DNS 網格
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


def load_rans_data(rans_file: str, time_range: tuple = (15.0, 35.0)):
    """
    載入 RANS/Leith 數據 (穩態平均場)
    
    Note: time_range 參數保留作為介面相容性，但 RANS/Leith 是穩態場，無時間維度
    """
    logger.info(f"📂 載入低保真數據: {rans_file}")
    
    with h5py.File(rans_file, 'r') as f:
        # 低保真場存在 mean_field 群組中
        mean_field = f['mean_field']
        
        # 載入網格座標（可能是 X/Y meshgrid 或 x/y 1D arrays）
        if 'X' in mean_field and 'Y' in mean_field:
            # k-ε RANS 格式：X, Y 是 meshgrid
            X = mean_field['X'][:]  # Shape: (nx, ny)
            Y = mean_field['Y'][:]
            nx, ny = X.shape
        elif 'x' in mean_field and 'y' in mean_field:
            # Leith 格式：x, y 是 1D arrays
            x = mean_field['x'][:]  # Shape: (nx,)
            y = mean_field['y'][:]  # Shape: (ny,)
            nx, ny = len(x), len(y)
            X, Y = np.meshgrid(x, y, indexing='ij')
        else:
            raise ValueError("無法找到座標數據（需要 X/Y 或 x/y）")
        
        logger.info(f"   解析度: {nx} × {ny}")
        
        # 攤平成座標矩陣
        coords = np.stack([X.ravel(), Y.ravel()], axis=1)  # Shape: (nx*ny, 2)
        
        # 載入速度場與湍流量
        u = mean_field['u'][:]  # Shape: (nx, ny)
        v = mean_field['v'][:]
        nu_t = mean_field['nu_t'][:]  # 渦流黏度
        
        # 檢查是否有 k 和 epsilon（k-ε RANS）或只有 nu_t（Leith）
        if 'k' in mean_field and 'epsilon' in mean_field:
            k = mean_field['k'][:]
            epsilon = mean_field['epsilon'][:]
            features = np.stack([
                u.ravel(),
                v.ravel(),
                k.ravel(),
                epsilon.ravel(),
                nu_t.ravel()
            ], axis=1)
            logger.info(f"   特徵矩陣形狀: {features.shape} (u, v, k, epsilon, nu_t)")
        else:
            # Leith 模型：只有 u, v, nu_t
            # 計算渦度作為額外特徵
            dx = X[1, 0] - X[0, 0]
            dy = Y[0, 1] - Y[0, 0]
            dvdx = np.gradient(v, dx, axis=0)
            dudy = np.gradient(u, dy, axis=1)
            vorticity = dvdx - dudy
            
            features = np.stack([
                u.ravel(),
                v.ravel(),
                nu_t.ravel(),
                vorticity.ravel()
            ], axis=1)
            logger.info(f"   特徵矩陣形狀: {features.shape} (u, v, nu_t, vorticity) [Leith model]")
        
    return coords, features, (nx, ny)


def generate_sensors_for_K(
    rans_file: str,
    K: int,
    dns_resolution: tuple = (256, 256),
    time_range: tuple = (15.0, 35.0),
    output_dir: str = "./data/sensors/kolmogorov"
):
    """
    為指定 K 值生成感測器
    
    流程:
    1. 從 RANS 數據使用 QR-Pivot 選取 K 個點
    2. 將 RANS 座標映射到 DNS 網格
    3. 保存為 JSON 格式
    """
    logger.info("=" * 70)
    logger.info(f"🎯 生成 K={K} 感測器")
    logger.info("=" * 70)
    
    # 1. 載入 RANS 數據
    coords, features, (nx_rans, ny_rans) = load_rans_data(rans_file, time_range)
    
    # 2. 使用 QR-Pivot 選點
    logger.info(f"\n🔍 執行 QR-Pivot 選點...")
    selector = QRPivotSelector()
    
    try:
        selected_indices, metrics = selector.select_sensors(
            data_matrix=features,
            n_sensors=K
        )
        
        logger.info(f"   ✅ 選取 {len(selected_indices)} 個點")
        logger.info(f"   條件數: {metrics.get('condition_number', 'N/A'):.2e}")
        
    except Exception as e:
        logger.error(f"   ❌ QR-Pivot 失敗: {e}")
        raise
    
    # 3. 映射到 DNS 網格
    nx_dns, ny_dns = dns_resolution
    scale_x = nx_dns / nx_rans
    scale_y = ny_dns / ny_rans
    
    logger.info(f"\n🗺️ 映射到 DNS 網格 ({nx_dns}×{ny_dns}):")
    logger.info(f"   縮放因子: x={scale_x:.2f}, y={scale_y:.2f}")
    
    # RANS 索引 -> (i, j) -> DNS (i', j') -> DNS 索引
    dns_indices = []
    for rans_idx in selected_indices:
        i_rans = rans_idx // ny_rans
        j_rans = rans_idx % ny_rans
        
        i_dns = int(i_rans * scale_x)
        j_dns = int(j_rans * scale_y)
        
        # 確保不越界
        i_dns = min(i_dns, nx_dns - 1)
        j_dns = min(j_dns, ny_dns - 1)
        
        dns_idx = i_dns * ny_dns + j_dns
        dns_indices.append(int(dns_idx))
    
    logger.info(f"   映射完成: {len(dns_indices)} 個 DNS 索引")
    
    # 4. 保存為 JSON
    output_file = Path(output_dir) / f"sensors_K{K}_re50_256x256.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "indices": dns_indices,
        "K": K,
        "source_resolution": f"{nx_rans}x{ny_rans}",
        "target_resolution": f"{nx_dns}x{ny_dns}",
        "method": "QR-Pivot from RANS",
        "time_range": list(time_range),
        "condition_number": float(metrics.get('condition_number', -1))
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\n💾 已保存: {output_file}")
    logger.info("=" * 70 + "\n")


def main():
    """批次生成 K=30, 50, 80, 100 的感測器（基於 Leith 模型）"""
    
    # 配置
    lowfi_file = "./data/lowfi/kolmogorov_rans/rans_re50_kf4_leith.h5"  # 改用 Leith 數據
    K_values = [30, 50, 80, 100]
    dns_resolution = (256, 256)
    time_range = (15.0, 35.0)  # 穩定態區間（保留介面相容性）
    output_dir = "./data/sensors/kolmogorov"
    
    logger.info("🚀 開始批次生成 Kolmogorov Flow 感測器（基於 Leith 模型）\n")
    logger.info(f"低保真數據: {lowfi_file}")
    logger.info(f"DNS 解析度: {dns_resolution[0]}×{dns_resolution[1]}")
    logger.info(f"K 值: {K_values}\n")
    
    # 檢查低保真檔案
    if not Path(lowfi_file).exists():
        logger.error(f"❌ 低保真檔案不存在: {lowfi_file}")
        logger.error(f"   請確認 Leith 模型數據已生成")
        sys.exit(1)
    
    # 批次生成
    for K in K_values:
        try:
            generate_sensors_for_K(
                rans_file=lowfi_file,  # 參數名保留但傳入 Leith 文件
                K=K,
                dns_resolution=dns_resolution,
                time_range=time_range,
                output_dir=output_dir
            )
        except Exception as e:
            logger.error(f"❌ K={K} 生成失敗: {e}")
            continue
    
    logger.info("\n✅ 所有感測器生成完成！")
    logger.info(f"📁 輸出目錄: {output_dir}")


if __name__ == "__main__":
    main()
