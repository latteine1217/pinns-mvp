"""
測試 JHTDB Cutout 數據與分層 QR-Pivot 感測點選擇

針對通道流的特性，將感測點分成三層：
1. 近壁區（|y| > 0.8）：30% 點數
2. 對數層（0.3 < |y| < 0.8）：40% 點數
3. 中心區（|y| < 0.3）：30% 點數

這樣能確保空間覆蓋性，同時保留 QR-Pivot 的最適化特性。
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 添加專案根目錄到路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_cutout_loader import JHTDBCutoutLoader
from pinnx.sensors.qr_pivot import QRPivotSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def stratified_qr_pivot(coords_3d: np.ndarray, 
                        snapshot_matrix: np.ndarray,
                        n_sensors: int = 50,
                        stratification: dict = None) -> tuple:
    """
    分層 QR-Pivot 感測點選擇
    
    Args:
        coords_3d: 空間座標 [n_points, 3]
        snapshot_matrix: 資料矩陣 [n_points, n_features]
        n_sensors: 總感測點數量
        stratification: 分層配置，預設為 {'wall': 0.3, 'log': 0.4, 'center': 0.3}
    
    Returns:
        (sensor_indices, sensor_coords, layer_info)
    """
    if stratification is None:
        stratification = {'wall': 0.3, 'log': 0.4, 'center': 0.3}
    
    # 計算每層點數
    n_wall = int(n_sensors * stratification['wall'])
    n_log = int(n_sensors * stratification['log'])
    n_center = n_sensors - n_wall - n_log  # 確保總數正確
    
    # 提取 Y 座標
    y_coords = coords_3d[:, 1]
    
    # 定義層邊界（對稱於 y=0）
    wall_threshold = 0.8      # |y| > 0.8 為近壁區
    log_threshold = 0.3       # 0.3 < |y| < 0.8 為對數層
    
    # 分層掩碼
    wall_mask = np.abs(y_coords) > wall_threshold
    log_mask = (np.abs(y_coords) <= wall_threshold) & (np.abs(y_coords) > log_threshold)
    center_mask = np.abs(y_coords) <= log_threshold
    
    logger.info(f"=== 分層統計 ===")
    logger.info(f"近壁區 (|y| > {wall_threshold}): {wall_mask.sum()} 點可用，選擇 {n_wall} 點")
    logger.info(f"對數層 ({log_threshold} < |y| < {wall_threshold}): {log_mask.sum()} 點可用，選擇 {n_log} 點")
    logger.info(f"中心區 (|y| < {log_threshold}): {center_mask.sum()} 點可用，選擇 {n_center} 點")
    
    # 初始化選擇器
    selector = QRPivotSelector(mode='column', pivoting=True)
    
    all_indices = []
    layer_info = {}
    
    for layer_name, mask, n_layer in [
        ('wall', wall_mask, n_wall),
        ('log', log_mask, n_log),
        ('center', center_mask, n_center)
    ]:
        if n_layer == 0:
            continue
            
        # 獲取該層的局部索引
        local_indices = np.where(mask)[0]
        local_data = snapshot_matrix[mask]
        
        if len(local_indices) < n_layer:
            logger.warning(f"{layer_name} 層可用點數不足，使用全部 {len(local_indices)} 點")
            layer_indices = local_indices
        else:
            # 在該層內使用 QR-Pivot
            try:
                layer_local_idx, metrics = selector.select_sensors(
                    data_matrix=local_data,
                    n_sensors=n_layer
                )
                # 轉換回全域索引
                layer_indices = local_indices[layer_local_idx]
                
                logger.info(f"{layer_name} 層 QR-Pivot 完成: 選擇 {len(layer_indices)} 點")
                logger.info(f"  條件數: {metrics['condition_number']:.2e}")
            except Exception as e:
                logger.error(f"{layer_name} 層 QR-Pivot 失敗: {e}")
                # 回退到隨機選擇
                layer_indices = np.random.choice(local_indices, n_layer, replace=False)
        
        all_indices.extend(layer_indices)
        layer_info[layer_name] = {
            'n_selected': len(layer_indices),
            'indices': layer_indices,
            'y_range': [y_coords[layer_indices].min(), y_coords[layer_indices].max()]
        }
    
    # 合併所有索引
    sensor_indices = np.array(all_indices, dtype=int)
    sensor_coords = coords_3d[sensor_indices]
    
    return sensor_indices, sensor_coords, layer_info


def test_stratified_qr_pivot(n_sensors: int = 50, 
                              save_dir: str = "results/qr_pivot_stratified"):
    """
    測試分層 QR-Pivot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # ========================================
    # 1. 載入 Cutout 數據
    # ========================================
    logger.info("=== 步驟 1: 載入 JHTDB Cutout 數據 ===")
    loader = JHTDBCutoutLoader()
    
    state = loader.load_full_state()
    coords = state['coords']
    
    if state['p'] is None:
        raise ValueError("壓力場數據不可用")
    
    p = state['p']  # [nx, ny, nz]
    
    logger.info(f"網格形狀: {p.shape}")
    
    # ========================================
    # 2. 下採樣 + 多特徵
    # ========================================
    logger.info("\n=== 步驟 2: 下採樣與特徵提取 ===")
    
    downsample_factor = 4
    p_down = p[::downsample_factor, ::downsample_factor, ::downsample_factor]
    x_down = coords['x'][::downsample_factor]
    y_down = coords['y'][::downsample_factor]
    z_down = coords['z'][::downsample_factor]
    
    logger.info(f"下採樣後形狀: {p_down.shape}")
    
    # 計算多特徵矩陣
    dp_dx, dp_dy, dp_dz = np.gradient(p_down, x_down, y_down, z_down)
    d2p_dx2, _, _ = np.gradient(dp_dx, x_down, y_down, z_down)
    _, d2p_dy2, _ = np.gradient(dp_dy, x_down, y_down, z_down)
    _, _, d2p_dz2 = np.gradient(dp_dz, x_down, y_down, z_down)
    laplacian = d2p_dx2 + d2p_dy2 + d2p_dz2
    
    # 構建座標與特徵矩陣
    X, Y, Z = np.meshgrid(x_down, y_down, z_down, indexing='ij')
    coords_3d = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    snapshot_matrix = np.column_stack([
        p_down.ravel(),
        dp_dx.ravel(),
        dp_dy.ravel(),
        dp_dz.ravel(),
        laplacian.ravel()
    ])
    
    logger.info(f"Snapshot 矩陣: {snapshot_matrix.shape}")
    
    # ========================================
    # 3. 分層 QR-Pivot
    # ========================================
    logger.info(f"\n=== 步驟 3: 分層 QR-Pivot (K={n_sensors}) ===")
    
    sensor_indices, sensor_coords, layer_info = stratified_qr_pivot(
        coords_3d=coords_3d,
        snapshot_matrix=snapshot_matrix,
        n_sensors=n_sensors,
        stratification={'wall': 0.3, 'log': 0.4, 'center': 0.3}
    )
    
    logger.info(f"\n選中感測點總數: {len(sensor_indices)}")
    for layer_name, info in layer_info.items():
        logger.info(f"  {layer_name}: {info['n_selected']} 點, "
                   f"Y 範圍 [{info['y_range'][0]:.4f}, {info['y_range'][1]:.4f}]")
    
    # ========================================
    # 4. 視覺化
    # ========================================
    logger.info("\n=== 步驟 4: 視覺化 ===")
    
    fig = plt.figure(figsize=(16, 5))
    
    # 定義層顏色
    layer_colors = {
        'wall': np.array([1, 0, 0, 0.7]),    # 紅色
        'log': np.array([0, 1, 0, 0.7]),     # 綠色
        'center': np.array([0, 0, 1, 0.7])   # 藍色
    }
    
    # 為每個感測點分配顏色
    colors = np.zeros((len(sensor_coords), 4))
    y_sensors = sensor_coords[:, 1]
    colors[np.abs(y_sensors) > 0.8] = layer_colors['wall']
    colors[(np.abs(y_sensors) <= 0.8) & (np.abs(y_sensors) > 0.3)] = layer_colors['log']
    colors[np.abs(y_sensors) <= 0.3] = layer_colors['center']
    
    # X-Y 平面
    ax1 = fig.add_subplot(131)
    ax1.scatter(sensor_coords[:, 0], sensor_coords[:, 1], c=colors, s=50)
    ax1.set_xlabel('X (streamwise)')
    ax1.set_ylabel('Y (wall-normal)')
    ax1.set_title(f'X-Y Projection (K={n_sensors})')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='k', linestyle='--', alpha=0.3, label='Layer boundary')
    ax1.axhline(y=-0.8, color='k', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.3, color='k', linestyle=':', alpha=0.3)
    ax1.axhline(y=-0.3, color='k', linestyle=':', alpha=0.3)
    ax1.legend()
    
    # X-Z 平面
    ax2 = fig.add_subplot(132)
    ax2.scatter(sensor_coords[:, 0], sensor_coords[:, 2], c=colors, s=50)
    ax2.set_xlabel('X (streamwise)')
    ax2.set_ylabel('Z (spanwise)')
    ax2.set_title(f'X-Z Projection (K={n_sensors})')
    ax2.grid(True, alpha=0.3)
    
    # Y-Z 平面
    ax3 = fig.add_subplot(133)
    ax3.scatter(sensor_coords[:, 1], sensor_coords[:, 2], c=colors, s=50)
    ax3.set_xlabel('Y (wall-normal)')
    ax3.set_ylabel('Z (spanwise)')
    ax3.set_title(f'Y-Z Projection (K={n_sensors})')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(x=0.8, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=-0.8, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=0.3, color='k', linestyle=':', alpha=0.3)
    ax3.axvline(x=-0.3, color='k', linestyle=':', alpha=0.3)
    
    # 添加圖例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label=f'Wall (|y|>0.8): {layer_info["wall"]["n_selected"]}'),
        Patch(facecolor='green', alpha=0.7, label=f'Log (0.3<|y|<0.8): {layer_info["log"]["n_selected"]}'),
        Patch(facecolor='blue', alpha=0.7, label=f'Center (|y|<0.3): {layer_info["center"]["n_selected"]}')
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.98))
    
    plt.tight_layout()
    fig_path = save_path / f'stratified_qr_K{n_sensors}_2d.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"保存 2D 投影: {fig_path}")
    plt.close()
    
    # 3D 視覺化
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], sensor_coords[:, 2],
              c=colors, s=100, alpha=0.8)
    ax.set_xlabel('X (streamwise)')
    ax.set_ylabel('Y (wall-normal)')
    ax.set_zlabel('Z (spanwise)')
    ax.set_title(f'Stratified QR-Pivot 3D Distribution (K={n_sensors})')
    
    fig_path_3d = save_path / f'stratified_qr_K{n_sensors}_3d.png'
    plt.savefig(fig_path_3d, dpi=150, bbox_inches='tight')
    logger.info(f"保存 3D 分佈: {fig_path_3d}")
    plt.close()
    
    # ========================================
    # 5. 保存數據
    # ========================================
    logger.info("\n=== 步驟 5: 保存數據 ===")
    
    sensor_data = {
        'indices': sensor_indices,
        'coords': sensor_coords,
        'pressure_values': snapshot_matrix[sensor_indices, 0],
        'layer_info': layer_info,
        'metadata': {
            'n_sensors': n_sensors,
            'stratification': {'wall': 0.3, 'log': 0.4, 'center': 0.3},
            'method': 'Stratified QR-Pivot',
            'source': 'JHTDB Cutout Re_tau=1000'
        }
    }
    
    npz_path = save_path / f'stratified_qr_K{n_sensors}.npz'
    np.savez(npz_path, **sensor_data)
    logger.info(f"保存感測點數據: {npz_path}")
    
    # ========================================
    # 6. 統計分析
    # ========================================
    logger.info("\n=== 步驟 6: 統計分析 ===")
    
    from scipy.spatial.distance import pdist
    distances = pdist(sensor_coords)
    
    logger.info(f"感測點間距離統計:")
    logger.info(f"  最小: {distances.min():.4f}")
    logger.info(f"  最大: {distances.max():.4f}")
    logger.info(f"  平均: {distances.mean():.4f}")
    logger.info(f"  中位: {np.median(distances):.4f}")
    
    logger.info(f"\nY 方向分佈:")
    logger.info(f"  範圍: [{y_sensors.min():.4f}, {y_sensors.max():.4f}]")
    logger.info(f"  近壁 (|y|>0.9): {np.sum(np.abs(y_sensors) > 0.9)}")
    logger.info(f"  中心 (|y|<0.3): {np.sum(np.abs(y_sensors) < 0.3)}")
    
    logger.info("\n✅ 分層 QR-Pivot 測試完成！")
    logger.info(f"結果保存於: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="測試分層 QR-Pivot 感測點選擇")
    parser.add_argument('--n-sensors', type=int, default=50, help='感測點總數')
    parser.add_argument('--output', type=str, default='results/qr_pivot_stratified',
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    test_stratified_qr_pivot(
        n_sensors=args.n_sensors,
        save_dir=args.output
    )
