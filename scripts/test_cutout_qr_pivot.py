"""
測試 JHTDB Cutout 數據與 QR-Pivot 感測點選擇整合

展示如何：
1. 從 Cutout HDF5 文件載入壓力場
2. 使用 QR-Pivot 演算法選擇最佳感測點
3. 視覺化感測點分佈
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


def test_qr_pivot_with_cutout(n_sensors: int = 50, save_dir: str = "results/qr_pivot_test"):
    """
    使用 Cutout 數據測試 QR-Pivot 感測點選擇
    
    Args:
        n_sensors: 感測點數量
        save_dir: 結果保存目錄
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
    
    # 使用壓力場作為 snapshot 矩陣（暫時只用壓力場）
    if state['p'] is None:
        raise ValueError("壓力場數據不可用")
    
    p = state['p']  # [nx, ny, nz]
    nx, ny, nz = p.shape
    
    logger.info(f"網格形狀: {p.shape}")
    logger.info(f"座標範圍: X=[{coords['x'].min():.2f}, {coords['x'].max():.2f}], "
               f"Y=[{coords['y'].min():.2f}, {coords['y'].max():.2f}], "
               f"Z=[{coords['z'].min():.2f}, {coords['z'].max():.2f}]")
    
    # ========================================
    # 2. 構建 Snapshot 矩陣（下採樣以避免記憶體溢出）
    # ========================================
    logger.info("\n=== 步驟 2: 構建 Snapshot 矩陣（下採樣 + 多特徵） ===")
    
    # 方案：對空間網格進行均勻下採樣
    # 原始：1024 x 256 x 768 ≈ 2億點
    # 下採樣：每 4 個點取 1 個 → 256 x 64 x 192 ≈ 310萬點（可處理）
    downsample_factor = 4
    
    p_downsampled = p[::downsample_factor, ::downsample_factor, ::downsample_factor]
    x_down = coords['x'][::downsample_factor]
    y_down = coords['y'][::downsample_factor]
    z_down = coords['z'][::downsample_factor]
    
    logger.info(f"下採樣後網格形狀: {p_downsampled.shape}")
    logger.info(f"下採樣因子: {downsample_factor} (每維度)")
    
    # 建立空間座標映射 [i, j, k] -> (x, y, z)
    X, Y, Z = np.meshgrid(x_down, y_down, z_down, indexing='ij')
    coords_3d = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])  # [n_points, 3]
    
    # ========================================
    # 修正：使用多特徵構建 Snapshot 矩陣
    # ========================================
    # 問題：單一壓力場 [n, 1] 會導致 QR 選擇偏向壁面（壓力梯度最大處）
    # 解決：構建多特徵矩陣，包含壓力及其空間變化
    
    logger.info("計算壓力場的空間梯度...")
    
    # 計算壓力梯度（使用 numpy.gradient）
    dp_dx, dp_dy, dp_dz = np.gradient(p_downsampled, 
                                       x_down, y_down, z_down)
    
    # 構建多特徵矩陣：[n_points, n_features]
    # 包含：壓力、壓力梯度、拉普拉斯算子
    p_flat = p_downsampled.ravel()[:, None]          # [n, 1]
    dp_dx_flat = dp_dx.ravel()[:, None]              # [n, 1]
    dp_dy_flat = dp_dy.ravel()[:, None]              # [n, 1]
    dp_dz_flat = dp_dz.ravel()[:, None]              # [n, 1]
    
    # 拉普拉斯算子（散度的散度，近似為二階差分和）
    d2p_dx2, _, _ = np.gradient(dp_dx, x_down, y_down, z_down)
    _, d2p_dy2, _ = np.gradient(dp_dy, x_down, y_down, z_down)
    _, _, d2p_dz2 = np.gradient(dp_dz, x_down, y_down, z_down)
    laplacian = (d2p_dx2 + d2p_dy2 + d2p_dz2).ravel()[:, None]  # [n, 1]
    
    # 合併所有特徵
    snapshot_matrix = np.hstack([p_flat, dp_dx_flat, dp_dy_flat, dp_dz_flat, laplacian])
    
    logger.info(f"Snapshot 矩陣形狀: {snapshot_matrix.shape} (包含壓力 + 梯度 + 拉普拉斯)")
    logger.info(f"空間座標形狀: {coords_3d.shape}")
    logger.info(f"特徵統計:")
    logger.info(f"  壓力範圍: [{p_flat.min():.4e}, {p_flat.max():.4e}]")
    logger.info(f"  梯度範圍: X=[{dp_dx_flat.min():.4e}, {dp_dx_flat.max():.4e}], "
               f"Y=[{dp_dy_flat.min():.4e}, {dp_dy_flat.max():.4e}], "
               f"Z=[{dp_dz_flat.min():.4e}, {dp_dz_flat.max():.4e}]")
    logger.info(f"  拉普拉斯範圍: [{laplacian.min():.4e}, {laplacian.max():.4e}]")
    
    # ========================================
    # 3. QR-Pivot 感測點選擇
    # ========================================
    logger.info(f"\n=== 步驟 3: QR-Pivot 選擇 {n_sensors} 個感測點 ===")
    
    try:
        # 使用多特徵矩陣進行 QR-Pivot
        # data_matrix: [n_locations, n_features] = [3.1M, 5]
        selector = QRPivotSelector(mode='column', pivoting=True)
        sensor_indices, metrics = selector.select_sensors(
            data_matrix=snapshot_matrix,
            n_sensors=n_sensors
        )
        
        logger.info(f"選中感測點索引範圍: [{sensor_indices.min()}, {sensor_indices.max()}]")
        logger.info(f"品質指標: {metrics}")
        
        # 獲取感測點的物理座標
        sensor_coords = coords_3d[sensor_indices]  # [n_sensors, 3]
        
        logger.info(f"感測點座標形狀: {sensor_coords.shape}")
        logger.info(f"座標範圍: X=[{sensor_coords[:,0].min():.2f}, {sensor_coords[:,0].max():.2f}], "
                   f"Y=[{sensor_coords[:,1].min():.2f}, {sensor_coords[:,1].max():.2f}], "
                   f"Z=[{sensor_coords[:,2].min():.2f}, {sensor_coords[:,2].max():.2f}]")
        
    except Exception as e:
        logger.error(f"QR-Pivot 選擇失敗: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================
    # 4. 視覺化感測點分佈
    # ========================================
    logger.info("\n=== 步驟 4: 視覺化感測點分佈 ===")
    
    fig = plt.figure(figsize=(16, 5))
    
    # 子圖 1: X-Y 平面投影
    ax1 = fig.add_subplot(131)
    ax1.scatter(sensor_coords[:, 0], sensor_coords[:, 1], 
               c=np.arange(n_sensors), cmap='viridis', s=50, alpha=0.7)
    ax1.set_xlabel('X (流向)')
    ax1.set_ylabel('Y (法向)')
    ax1.set_title(f'X-Y 平面投影 (K={n_sensors})')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # 子圖 2: X-Z 平面投影
    ax2 = fig.add_subplot(132)
    scatter = ax2.scatter(sensor_coords[:, 0], sensor_coords[:, 2], 
                         c=np.arange(n_sensors), cmap='viridis', s=50, alpha=0.7)
    ax2.set_xlabel('X (流向)')
    ax2.set_ylabel('Z (展向)')
    ax2.set_title(f'X-Z 平面投影 (K={n_sensors})')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # 子圖 3: Y-Z 平面投影
    ax3 = fig.add_subplot(133)
    ax3.scatter(sensor_coords[:, 1], sensor_coords[:, 2], 
               c=np.arange(n_sensors), cmap='viridis', s=50, alpha=0.7)
    ax3.set_xlabel('Y (法向)')
    ax3.set_ylabel('Z (展向)')
    ax3.set_title(f'Y-Z 平面投影 (K={n_sensors})')
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect('equal')
    
    plt.colorbar(scatter, ax=[ax1, ax2, ax3], label='感測點編號')
    plt.tight_layout()
    
    # 保存圖表
    fig_path = save_path / f'qr_sensors_K{n_sensors}_2d_projections.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    logger.info(f"保存 2D 投影圖: {fig_path}")
    plt.close()
    
    # 3D 視覺化
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    point_colors = np.arange(n_sensors)
    scatter = ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1], sensor_coords[:, 2],
                        c=point_colors, cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('X (流向)')
    ax.set_ylabel('Y (法向)')
    ax.set_zlabel('Z (展向)')
    ax.set_title(f'QR-Pivot 感測點 3D 分佈 (K={n_sensors})')
    plt.colorbar(scatter, ax=ax, label='感測點編號', shrink=0.5)
    
    fig_path_3d = save_path / f'qr_sensors_K{n_sensors}_3d.png'
    plt.savefig(fig_path_3d, dpi=150, bbox_inches='tight')
    logger.info(f"保存 3D 分佈圖: {fig_path_3d}")
    plt.close()
    
    # ========================================
    # 5. 保存感測點數據
    # ========================================
    logger.info("\n=== 步驟 5: 保存感測點數據 ===")
    
    sensor_data = {
        'indices': sensor_indices,
        'coords': sensor_coords,
        'pressure_values': snapshot_matrix[sensor_indices, 0],  # 只取壓力列
        'metrics': metrics,
        'metadata': {
            'n_sensors': n_sensors,
            'grid_shape': p.shape,
            'n_features': snapshot_matrix.shape[1],
            'features': ['pressure', 'dp/dx', 'dp/dy', 'dp/dz', 'laplacian'],
            'source': 'JHTDB Cutout Re_tau=1000',
            'method': 'QR-Pivot on multi-feature matrix'
        }
    }
    
    npz_path = save_path / f'qr_sensors_K{n_sensors}.npz'
    np.savez(npz_path, **sensor_data)
    logger.info(f"保存感測點數據: {npz_path}")
    
    # ========================================
    # 6. 統計分析
    # ========================================
    logger.info("\n=== 步驟 6: 統計分析 ===")
    
    # 計算感測點間距離
    from scipy.spatial.distance import pdist
    distances = pdist(sensor_coords)
    
    logger.info(f"感測點間距離統計:")
    logger.info(f"  最小距離: {distances.min():.4f}")
    logger.info(f"  最大距離: {distances.max():.4f}")
    logger.info(f"  平均距離: {distances.mean():.4f}")
    logger.info(f"  中位距離: {np.median(distances):.4f}")
    
    # Y 方向分佈（特別重要，因為非均勻網格）
    y_sensors = sensor_coords[:, 1]
    logger.info(f"\nY 方向（法向）分佈:")
    logger.info(f"  範圍: [{y_sensors.min():.4f}, {y_sensors.max():.4f}]")
    logger.info(f"  近壁點數 (|y| > 0.9): {np.sum(np.abs(y_sensors) > 0.9)}")
    logger.info(f"  中心點數 (|y| < 0.3): {np.sum(np.abs(y_sensors) < 0.3)}")
    
    logger.info("\n✅ 測試完成！")
    logger.info(f"結果保存於: {save_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="測試 Cutout 數據與 QR-Pivot 整合")
    parser.add_argument('--n-sensors', type=int, default=50, help='感測點數量')
    parser.add_argument('--output', type=str, default='results/qr_pivot_test', 
                       help='結果輸出目錄')
    
    args = parser.parse_args()
    
    test_qr_pivot_with_cutout(
        n_sensors=args.n_sensors,
        save_dir=args.output
    )
