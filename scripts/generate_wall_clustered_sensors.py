#!/usr/bin/env python3
"""
壁面聚集隨機採樣生成器
Wall-Clustered Random Sensor Generation

功能：
1. 基於物理先驗的分層隨機採樣（y 方向壁面聚集）
2. x 方向中心聚集（高斯分佈）
3. 生成感測點並提取速度場資料

策略對比：
- QR-Pivot: 條件數優化，計算成本 O(n³)
- Wall-Clustered: 物理先驗，計算成本 O(n)

使用範例：
    python scripts/generate_wall_clustered_sensors.py \
        --input data/jhtdb/channel_flow_re1000/cutout_2d_xy_z4.71.h5 \
        --output data/jhtdb/channel_flow_re1000/sensors_K50_wall_clustered.npz \
        --K 50 \
        --wall-fraction 0.4 \
        --log-fraction 0.4 \
        --x-center-sigma 4.0 \
        --visualize

作者: PINNs-MVP
日期: 2025-10-16
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 日誌配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class WallClusteredSampler:
    """壁面聚集隨機採樣器"""
    
    def __init__(
        self,
        domain: Dict[str, Tuple[float, float]],
        K: int,
        wall_fraction: float = 0.4,
        log_fraction: float = 0.4,
        x_center_sigma: float = 4.0,
        seed: int = 42
    ):
        """
        參數:
            domain: 空間域 {'x': (min, max), 'y': (min, max)}
            K: 感測點總數
            wall_fraction: 壁面層點數比例 (預設 40%)
            log_fraction: 對數層點數比例 (預設 40%)
            x_center_sigma: x 方向高斯分佈標準差
            seed: 隨機種子
        """
        self.domain = domain
        self.K = K
        self.wall_fraction = wall_fraction
        self.log_fraction = log_fraction
        self.center_fraction = 1.0 - wall_fraction - log_fraction
        self.x_center_sigma = x_center_sigma
        
        # 隨機種子
        self.rng = np.random.default_rng(seed)
        
        # 分層點數
        self.K_wall = int(K * wall_fraction)
        self.K_log = int(K * log_fraction)
        self.K_center = K - self.K_wall - self.K_log
        
        logger.info(f"初始化壁面聚集採樣器:")
        logger.info(f"  總點數 K = {K}")
        logger.info(f"  壁面層: {self.K_wall} 點 ({wall_fraction*100:.1f}%)")
        logger.info(f"  對數層: {self.K_log} 點 ({log_fraction*100:.1f}%)")
        logger.info(f"  中心層: {self.K_center} 點 ({self.center_fraction*100:.1f}%)")
    
    def _sample_x_gaussian(self, n: int) -> np.ndarray:
        """x 方向高斯聚集採樣（中心聚集）"""
        x_min, x_max = self.domain['x']
        x_center = (x_min + x_max) / 2.0
        
        # 高斯採樣 + 截斷
        x = self.rng.normal(loc=x_center, scale=self.x_center_sigma, size=n)
        x = np.clip(x, x_min, x_max)
        
        return x
    
    def _sample_y_stratified(self) -> np.ndarray:
        """y 方向分層採樣（壁面聚集）"""
        y_min, y_max = self.domain['y']
        
        # 定義分層範圍
        y_wall_ranges = [(0.95, 1.0), (-1.0, -0.95)]  # 上下壁面層
        y_log_ranges = [(0.3, 0.95), (-0.95, -0.3)]    # 上下對數層
        y_center_range = (-0.3, 0.3)                    # 中心層
        
        y_samples = []
        
        # 1. 壁面層（上下各半）
        K_wall_half = self.K_wall // 2
        for y_range in y_wall_ranges:
            y = self.rng.uniform(y_range[0], y_range[1], K_wall_half)
            y_samples.append(y)
        
        # 2. 對數層（上下各半）
        K_log_half = self.K_log // 2
        for y_range in y_log_ranges:
            y = self.rng.uniform(y_range[0], y_range[1], K_log_half)
            y_samples.append(y)
        
        # 3. 中心層
        y = self.rng.uniform(y_center_range[0], y_center_range[1], self.K_center)
        y_samples.append(y)
        
        # 合併
        y_all = np.concatenate(y_samples)
        
        # 調整點數（處理奇數）
        if len(y_all) < self.K:
            extra = self.K - len(y_all)
            y_extra = self.rng.uniform(y_min, y_max, extra)
            y_all = np.concatenate([y_all, y_extra])
        elif len(y_all) > self.K:
            y_all = y_all[:self.K]
        
        return y_all
    
    def generate_sensors(self) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        生成感測點座標
        
        返回:
            coords: (K, 2) 陣列 [x, y]
            metadata: 採樣統計資訊
        """
        # y 方向分層採樣
        y = self._sample_y_stratified()
        
        # x 方向高斯採樣
        x = self._sample_x_gaussian(self.K)
        
        # 合併座標
        coords = np.stack([x, y], axis=1)
        
        # 計算統計資訊
        metadata = {
            'K': self.K,
            'wall_fraction': self.wall_fraction,
            'log_fraction': self.log_fraction,
            'center_fraction': self.center_fraction,
            'x_mean': np.mean(x),
            'x_std': np.std(x),
            'y_mean': np.mean(y),
            'y_std': np.std(y),
            'wall_points': np.sum((np.abs(y) > 0.95)),
            'log_points': np.sum((np.abs(y) > 0.3) & (np.abs(y) <= 0.95)),
            'center_points': np.sum(np.abs(y) <= 0.3)
        }
        
        logger.info(f"生成感測點完成:")
        logger.info(f"  座標範圍: x=[{x.min():.2f}, {x.max():.2f}], y=[{y.min():.2f}, {y.max():.2f}]")
        logger.info(f"  壁面層實際: {metadata['wall_points']} 點")
        logger.info(f"  對數層實際: {metadata['log_points']} 點")
        logger.info(f"  中心層實際: {metadata['center_points']} 點")
        
        return coords, metadata


def load_field_data(h5_file: str, coords_2d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    從 HDF5 檔案載入並插值場資料
    
    參數:
        h5_file: HDF5 檔案路徑（JHTDB 格式）
        coords_2d: (K, 2) 感測點座標 [x, y]
    
    返回:
        field_data: {'u': (K,), 'v': (K,), 'w': (K,)}
    """
    logger.info(f"載入場資料: {h5_file}")
    
    with h5py.File(h5_file, 'r') as f:
        # 載入網格座標 (JHTDB 格式)
        x_grid = f['xcoor'][:]  # Shape: (Nx,)
        y_grid = f['ycoor'][:]  # Shape: (Ny,)
        z_grid = f['zcoor'][:]  # Shape: (Nz,)
        
        # 載入速度場 (JHTDB 格式: [x, y, z, component])
        # Shape: (Nx, Ny, Nz, 3)
        velocity = f['Velocity_0001'][:]
        
        logger.info(f"  網格形狀: {velocity.shape}")
        logger.info(f"  x 範圍: [{x_grid.min():.2f}, {x_grid.max():.2f}] ({len(x_grid)} 點)")
        logger.info(f"  y 範圍: [{y_grid.min():.2f}, {y_grid.max():.2f}] ({len(y_grid)} 點)")
        logger.info(f"  z 範圍: [{z_grid.min():.2f}, {z_grid.max():.2f}] ({len(z_grid)} 點)")
    
    # 取 2D 切片（z 方向中間位置）
    z_mid_idx = len(z_grid) // 2
    u_field = velocity[:, :, z_mid_idx, 0]  # Shape: (Nx, Ny)
    v_field = velocity[:, :, z_mid_idx, 1]
    w_field = velocity[:, :, z_mid_idx, 2]
    
    logger.info(f"取 z={z_grid[z_mid_idx]:.2f} 切片 (索引 {z_mid_idx}/{len(z_grid)})")
    logger.info(f"  2D 切片形狀: {u_field.shape}")
    
    # 插值到感測點
    from scipy.interpolate import RegularGridInterpolator
    
    interp_u = RegularGridInterpolator((x_grid, y_grid), u_field, method='linear', bounds_error=False, fill_value=0.0)
    interp_v = RegularGridInterpolator((x_grid, y_grid), v_field, method='linear', bounds_error=False, fill_value=0.0)
    interp_w = RegularGridInterpolator((x_grid, y_grid), w_field, method='linear', bounds_error=False, fill_value=0.0)
    
    u_sensors = interp_u(coords_2d)
    v_sensors = interp_v(coords_2d)
    w_sensors = interp_w(coords_2d)
    
    logger.info(f"插值完成:")
    logger.info(f"  u: [{u_sensors.min():.4f}, {u_sensors.max():.4f}], mean={u_sensors.mean():.4f}")
    logger.info(f"  v: [{v_sensors.min():.4f}, {v_sensors.max():.4f}], mean={v_sensors.mean():.4f}")
    logger.info(f"  w: [{w_sensors.min():.4f}, {w_sensors.max():.4f}], mean={w_sensors.mean():.4f}")
    
    return {
        'u': u_sensors,
        'v': v_sensors,
        'w': w_sensors
    }


def visualize_sensors(
    coords: np.ndarray,
    metadata: Dict[str, float],
    field_data: Optional[Dict[str, np.ndarray]] = None,
    output_path: Optional[str] = None
):
    """視覺化感測點分佈與場資料"""
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. 空間分佈（2D）
    ax1 = fig.add_subplot(gs[0, :2])
    
    # 壁面聚集著色
    y = coords[:, 1]
    colors = np.zeros(len(y))
    colors[(np.abs(y) > 0.95)] = 0  # 壁面層（紅色）
    colors[(np.abs(y) > 0.3) & (np.abs(y) <= 0.95)] = 1  # 對數層（黃色）
    colors[(np.abs(y) <= 0.3)] = 2  # 中心層（綠色）
    
    scatter = ax1.scatter(
        coords[:, 0], coords[:, 1],
        c=colors, cmap='RdYlGn', s=50, alpha=0.7, edgecolors='k', linewidths=0.5
    )
    
    # 標記分層邊界
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.3, label='壁面層邊界')
    ax1.axhline(y=-0.95, color='r', linestyle='--', alpha=0.3)
    ax1.axhline(y=0.3, color='orange', linestyle='--', alpha=0.3, label='對數層邊界')
    ax1.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('x (Streamwise)', fontsize=12)
    ax1.set_ylabel('y (Wall-normal)', fontsize=12)
    ax1.set_title(f'Wall-Clustered Sensors (K={metadata["K"]})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # 2. y 方向分佈直方圖
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(y, bins=30, color='steelblue', alpha=0.7, orientation='horizontal')
    ax2.axhline(y=0.95, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.95, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=-0.3, color='orange', linestyle='--', alpha=0.5)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('y-Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. x 方向分佈直方圖
    ax3 = fig.add_subplot(gs[1, 0])
    x = coords[:, 0]
    ax3.hist(x, bins=30, color='coral', alpha=0.7)
    ax3.axvline(x=float(np.mean(x)), color='r', linestyle='--', label=f'Mean={np.mean(x):.2f}')
    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('x-Distribution (Gaussian Clustering)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 4. 統計資訊表格
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    stats_text = f"""
    採樣統計
    {'='*40}
    總點數: {metadata['K']}
    壁面層: {metadata.get('wall_points', 0)} ({metadata['wall_fraction']*100:.1f}%)
    對數層: {metadata.get('log_points', 0)} ({metadata['log_fraction']*100:.1f}%)
    中心層: {metadata.get('center_points', 0)} ({metadata['center_fraction']*100:.1f}%)
    
    x 統計: μ={metadata['x_mean']:.2f}, σ={metadata['x_std']:.2f}
    y 統計: μ={metadata['y_mean']:.4f}, σ={metadata['y_std']:.2f}
    """
    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', verticalalignment='center')
    
    # 5. 速度場分佈（如果提供）
    if field_data is not None:
        ax5 = fig.add_subplot(gs[1, 2])
        u = field_data['u']
        ax5.scatter(u, y, c=colors, cmap='RdYlGn', s=30, alpha=0.7, edgecolors='k', linewidths=0.5)
        ax5.set_xlabel('u (Streamwise Velocity)', fontsize=12)
        ax5.set_ylabel('y', fontsize=12)
        ax5.set_title('Velocity Profile at Sensors', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle('Wall-Clustered Random Sampling Analysis', fontsize=16, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"視覺化圖表保存: {output_path}")
    else:
        plt.show()
    
    plt.close()


def save_sensors(
    output_path: str,
    coords: np.ndarray,
    field_data: Dict[str, np.ndarray],
    metadata: Dict[str, float]
):
    """保存感測點資料為 .npz 格式（匹配 QR-Pivot 格式）"""
    K = len(coords)
    
    # 生成 3D 座標（z 固定為中間位置，約 4.71）
    coords_3d = np.zeros((K, 3))
    coords_3d[:, 0] = coords[:, 0]  # x
    coords_3d[:, 1] = coords[:, 1]  # y
    coords_3d[:, 2] = 4.71           # z (中間切片位置)
    
    # 準備 metadata（匹配 QR-Pivot 格式）
    metadata_str = (
        f"Wall-Clustered Random Sampling\n"
        f"K={K}\n"
        f"wall_fraction={metadata['wall_fraction']:.2f}\n"
        f"log_fraction={metadata['log_fraction']:.2f}\n"
        f"center_fraction={metadata['center_fraction']:.2f}\n"
        f"wall_points={metadata['wall_points']}\n"
        f"log_points={metadata['log_points']}\n"
        f"center_points={metadata['center_points']}\n"
        f"x_mean={metadata['x_mean']:.4f}\n"
        f"x_std={metadata['x_std']:.4f}\n"
        f"y_mean={metadata['y_mean']:.4f}\n"
        f"y_std={metadata['y_std']:.4f}"
    )
    
    # 計算條件數（這裡設為 None，因為沒有矩陣分解）
    condition_number = np.nan
    energy_ratio = np.nan
    
    # 保存（匹配 QR-Pivot 格式）
    np.savez(
        output_path,
        indices=np.arange(K),
        coords=coords_3d,           # (K, 3)
        coords_2d=coords,           # (K, 2)
        u=field_data['u'],          # (K,)
        v=field_data['v'],          # (K,)
        w=field_data['w'],          # (K,)
        p=np.zeros(K),              # 壓力未知，設為 0
        condition_number=condition_number,
        energy_ratio=energy_ratio,
        metadata=metadata_str
    )
    
    logger.info(f"感測點資料保存: {output_path}")
    logger.info(f"  格式: 與 QR-Pivot 一致")
    logger.info(f"  座標形狀: 3D={coords_3d.shape}, 2D={coords.shape}")
    logger.info(f"  速度場: u={field_data['u'].shape}, v={field_data['v'].shape}, w={field_data['w'].shape}")


def main():
    parser = argparse.ArgumentParser(description='壁面聚集隨機採樣生成器')
    
    # 必要參數
    parser.add_argument('--input', type=str, required=True,
                        help='輸入 HDF5 檔案路徑（2D 切片資料）')
    parser.add_argument('--output', type=str, required=True,
                        help='輸出 .npz 檔案路徑')
    
    # 採樣參數
    parser.add_argument('--K', type=int, default=50,
                        help='感測點總數（預設: 50）')
    parser.add_argument('--wall-fraction', type=float, default=0.4,
                        help='壁面層點數比例（預設: 0.4）')
    parser.add_argument('--log-fraction', type=float, default=0.4,
                        help='對數層點數比例（預設: 0.4）')
    parser.add_argument('--x-center-sigma', type=float, default=4.0,
                        help='x 方向高斯分佈標準差（預設: 4.0）')
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子（預設: 42）')
    
    # 空間域（通道流預設值）
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=25.13)
    parser.add_argument('--y-min', type=float, default=-1.0)
    parser.add_argument('--y-max', type=float, default=1.0)
    
    # 可選參數
    parser.add_argument('--visualize', action='store_true',
                        help='生成視覺化圖表')
    parser.add_argument('--viz-output', type=str, default=None,
                        help='視覺化圖表保存路徑（預設: 自動生成）')
    
    args = parser.parse_args()
    
    # 檢查輸入檔案
    if not Path(args.input).exists():
        logger.error(f"輸入檔案不存在: {args.input}")
        sys.exit(1)
    
    # 定義空間域
    domain = {
        'x': (args.x_min, args.x_max),
        'y': (args.y_min, args.y_max)
    }
    
    # 初始化採樣器
    sampler = WallClusteredSampler(
        domain=domain,
        K=args.K,
        wall_fraction=args.wall_fraction,
        log_fraction=args.log_fraction,
        x_center_sigma=args.x_center_sigma,
        seed=args.seed
    )
    
    # 生成感測點
    coords, metadata = sampler.generate_sensors()
    
    # 載入場資料
    field_data = load_field_data(args.input, coords)
    
    # 保存結果
    save_sensors(args.output, coords, field_data, metadata)
    
    # 視覺化（可選）
    if args.visualize:
        viz_path = args.viz_output or str(Path(args.output).with_suffix('.png'))
        visualize_sensors(coords, metadata, field_data, viz_path)
    
    logger.info("="*60)
    logger.info("壁面聚集隨機採樣完成！")
    logger.info(f"輸出檔案: {args.output}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
