#!/usr/bin/env python3
"""
Channel Flow Re1000 JHTDB 資料擷取腳本

從約翰霍普金斯湍流資料庫 (JHTDB) 擷取 Channel Flow Re_tau=1000 資料，
並進行預處理以支援 PINNs 逆重建實驗。

主要功能：
1. JHTDB Channel Flow cutout 批量資料下載
2. 2D 切片提取與時間平均處理  
3. QR-pivot 感測點選擇與噪聲模擬
4. 低保真 RANS 資料生成作為軟先驗
5. 資料格式標準化與快取管理

使用範例：
  python fetch_channel_flow.py --config configs/channel_flow_re1000.yml
  python fetch_channel_flow.py --cutout_mode --resolution 128 64
  python fetch_channel_flow.py --sensor_only --K 8 --method qr_pivot
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import yaml
import matplotlib.pyplot as plt
from dataclasses import dataclass

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

from pinnx.dataio.jhtdb_client import JHTDBManager, JHTDBConfig
from pinnx.sensors.qr_pivot import QRPivotSelector
from pinnx.dataio.lowfi_loader import LowFiData
from pinnx.physics.scaling import VSScaler


@dataclass
class ChannelFlowConfig:
    """Channel Flow Re1000 專用配置"""
    # JHTDB 資料集參數
    dataset: str = "channel"
    description: str = "通道流 (Re_tau=1000)"
    
    # 域範圍 (基於摩擦速度 u_tau 標準化)
    domain_x: Optional[List[float]] = None  # [0, 25.13] = [0, 8π]
    domain_y: Optional[List[float]] = None  # [-1.0, 1.0]
    domain_z: Optional[List[float]] = None  # [0, 9.42] = [0, 3π]
    
    # 解析度
    resolution_x: int = 2048
    resolution_y: int = 512
    resolution_z: int = 1536
    
    # 物理參數
    Re_tau: float = 1000.0
    nu: float = 1.0e-3  # ν = 1/Re_tau
    u_tau: float = 1.0  # 摩擦速度
    
    # 時間參數
    time_range: Optional[List[float]] = None  # [0.0, 26.0]
    dt: float = 0.0065
    time_average_window: Optional[List[float]] = None  # [20.0, 26.0]
    
    # 2D 切片設定
    slice_plane: str = "xy"  # xy, xz, yz
    slice_position: float = 4.71  # z = 3π/2
    
    def __post_init__(self):
        """設定預設值"""
        if self.domain_x is None:
            self.domain_x = [0.0, 8 * np.pi]
        if self.domain_y is None:
            self.domain_y = [-1.0, 1.0]
        if self.domain_z is None:
            self.domain_z = [0.0, 3 * np.pi]
        if self.time_range is None:
            self.time_range = [0.0, 26.0]
        if self.time_average_window is None:
            self.time_average_window = [20.0, 26.0]


class ChannelFlowDataFetcher:
    """Channel Flow Re1000 資料擷取器"""
    
    def __init__(self, config: ChannelFlowConfig, cache_dir: str = "./data/jhtdb/channel_flow_re1000"):
        self.config = config
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化 JHTDB 管理器
        self.jhtdb_manager = JHTDBManager(
            cache_dir=str(self.cache_dir / "raw"),
            use_mock=False  # 移除 Mock 模式，使用真實 JHTDB 資料
        )
        
        # 日誌設定
        self.logger = logging.getLogger(__name__)
        
    def fetch_cutout_data(self, 
                         resolution: Tuple[int, int] = (128, 64),
                         variables: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        擷取 JHTDB cutout 資料並提取 2D 切片
        
        Args:
            resolution: (nx, ny) 2D 切片解析度
            variables: 變數列表，預設 ['u', 'v', 'p']
            
        Returns:
            處理後的 2D 場資料
        """
        if variables is None:
            variables = ['u', 'v', 'p']
            
        self.logger.info(f"擷取 Channel Flow cutout 資料，解析度: {resolution}")
        
        # 計算 3D cutout 範圍 (確保包含目標切片位置)
        z_margin = 0.5  # z 方向邊界
        cutout_start = [
            self.config.domain_x[0] if self.config.domain_x else 0.0,
            self.config.domain_y[0] if self.config.domain_y else -1.0,
            max(self.config.domain_z[0] if self.config.domain_z else 0.0, self.config.slice_position - z_margin)
        ]
        cutout_end = [
            self.config.domain_x[1] if self.config.domain_x else 25.13,
            self.config.domain_y[1] if self.config.domain_y else 1.0,
            min(self.config.domain_z[1] if self.config.domain_z else 9.42, self.config.slice_position + z_margin)
        ]
        
        # 3D cutout 解析度 (z 方向使用最小解析度)
        cutout_resolution = [resolution[0], resolution[1], 16]
        
        # 從 JHTDB 獲取資料
        cutout_data = self.jhtdb_manager.fetch_cutout(
            dataset=self.config.dataset,
            start=cutout_start,
            end=cutout_end,
            resolution=cutout_resolution,
            variables=variables,
            timestep=0  # 使用瞬時資料，後續進行時間平均
        )
        
        # 提取 2D 切片
        slice_data = self._extract_2d_slice(cutout_data, resolution)
        
        # 快取結果
        cache_file = self.cache_dir / f"cutout_{resolution[0]}x{resolution[1]}.npz"
        np.savez_compressed(cache_file, **slice_data)
        self.logger.info(f"資料已快取至: {cache_file}")
        
        return slice_data
    
    def _extract_2d_slice(self, cutout_data: Dict[str, np.ndarray], 
                         target_resolution: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """從 3D cutout 資料提取 2D 切片"""
        
        # 提取座標
        coordinates = cutout_data['coordinates']
        x = coordinates['x']
        y = coordinates['y']
        z = coordinates['z']
        
        # 找到最接近目標切片位置的 z 索引
        z_idx = np.argmin(np.abs(z - self.config.slice_position))
        self.logger.info(f"選擇 z 切片位置: {z[z_idx]:.3f} (目標: {self.config.slice_position:.3f})")
        
        # 提取 2D 場
        slice_data = {}
        slice_data['coordinates'] = {
            'x': x,
            'y': y
        }
        
        for var in ['u', 'v', 'w', 'p']:
            if var in cutout_data:
                # 提取 z 切片: field[x, y, z] -> field_2d[x, y]
                field_3d = cutout_data[var]
                field_2d = field_3d[:, :, z_idx]
                
                # 調整到目標解析度 (如果需要)
                if field_2d.shape != target_resolution:
                    field_2d = self._interpolate_to_resolution(
                        field_2d, x, y, target_resolution
                    )
                
                slice_data[var] = field_2d
        
        return slice_data
    
    def _interpolate_to_resolution(self, field: np.ndarray, 
                                  x: np.ndarray, y: np.ndarray,
                                  target_resolution: Tuple[int, int]) -> np.ndarray:
        """將場插值到目標解析度"""
        from scipy.interpolate import RegularGridInterpolator
        
        # 建立插值器
        interpolator = RegularGridInterpolator(
            (x, y), field, 
            method='linear', 
            bounds_error=False, 
            fill_value=0.0
        )
        
        # 新網格
        x_new = np.linspace(x[0], x[-1], target_resolution[0])
        y_new = np.linspace(y[0], y[-1], target_resolution[1])
        X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
        
        # 插值
        points_new = np.stack([X_new.ravel(), Y_new.ravel()], axis=-1)
        field_new = interpolator(points_new).reshape(target_resolution)
        
        return field_new
    
    def generate_sensor_points(self, 
                              field_data: Dict[str, np.ndarray],
                              K: int = 8,
                              method: str = "qr_pivot",
                              noise_sigma: float = 0.02,
                              dropout_prob: float = 0.10) -> Dict[str, np.ndarray]:
        """
        生成稀疏感測點資料
        
        Args:
            field_data: 完整場資料
            K: 感測點數量
            method: 感測點選擇方法
            noise_sigma: 高斯噪聲標準差
            dropout_prob: 隨機遺失比例
            
        Returns:
            感測點資料與位置
        """
        self.logger.info(f"生成 {K} 個感測點，方法: {method}")
        
        coordinates = field_data['coordinates']
        x = coordinates['x']
        y = coordinates['y']
        
        # 建立完整網格
        X, Y = np.meshgrid(x, y, indexing='ij')
        points_full = np.stack([X.ravel(), Y.ravel()], axis=-1)
        
        # 準備場資料矩陣 (用於 QR-pivot)
        field_matrix = []
        for var in ['u', 'v', 'p']:
            if var in field_data:
                field_matrix.append(field_data[var].ravel())
        field_matrix = np.array(field_matrix).T  # [N_points, N_vars]
        
        # 感測點選擇
        if method == "qr_pivot":
            selector = QRPivotSelector()
            sensor_indices, selection_info = selector.select_sensors(
                field_matrix, K
            )
        elif method == "random":
            np.random.seed(42)
            sensor_indices = np.random.choice(len(points_full), K, replace=False)
            selection_info = {"method": "random"}
        else:
            raise ValueError(f"未支援的感測點選擇方法: {method}")
        
        # 提取感測點座標與數值
        sensor_points = points_full[sensor_indices]
        sensor_data = {}
        
        for var in ['u', 'v', 'p']:
            if var in field_data:
                field_values = field_data[var].ravel()[sensor_indices]
                
                # 添加噪聲
                if noise_sigma > 0:
                    noise = np.random.normal(0, noise_sigma * np.std(field_values), len(field_values))
                    field_values += noise
                
                # 隨機遺失
                if dropout_prob > 0:
                    n_dropout = int(dropout_prob * len(field_values))
                    dropout_indices = np.random.choice(len(field_values), n_dropout, replace=False)
                    field_values[dropout_indices] = np.nan
                
                sensor_data[var] = field_values
        
        result = {
            'sensor_points': sensor_points,  # [K, 2] (x, y)
            'sensor_data': sensor_data,      # {var: [K]} 
            'sensor_indices': sensor_indices,
            'selection_info': selection_info,
            'noise_sigma': noise_sigma,
            'dropout_prob': dropout_prob
        }
        
        # 快取感測點資料
        cache_file = self.cache_dir / f"sensors_K{K}_{method}.npz"
        np.savez_compressed(cache_file, **result)
        self.logger.info(f"感測點資料已快取至: {cache_file}")
        
        return result
    
    def generate_lowfi_prior(self, 
                           resolution: Tuple[int, int] = (128, 64),
                           prior_type: str = "simplified_rans") -> Dict[str, np.ndarray]:
        """
        生成低保真先驗資料 (RANS/簡化模型)
        
        Args:
            resolution: 先驗資料解析度
            prior_type: 先驗類型
            
        Returns:
            低保真先驗場
        """
        self.logger.info(f"生成低保真先驗資料，類型: {prior_type}")
        
        # 建立網格
        domain_x = self.config.domain_x if self.config.domain_x else [0.0, 25.13]
        domain_y = self.config.domain_y if self.config.domain_y else [-1.0, 1.0]
        x = np.linspace(domain_x[0], domain_x[1], resolution[0])
        y = np.linspace(domain_y[0], domain_y[1], resolution[1])
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        if prior_type == "simplified_rans":
            # 簡化的 Channel Flow 速度剖面
            y_plus = np.abs(Y) * self.config.Re_tau  # y+ = y * Re_tau
            
            # 使用簡化的對數律 + 線性律
            u_mean = np.zeros_like(X)
            for i in range(len(x)):
                for j in range(len(y)):
                    yp = y_plus[i, j]
                    if yp < 11:  # 粘性底層
                        u_mean[i, j] = yp
                    else:  # 對數律層
                        u_mean[i, j] = 1/0.41 * np.log(yp) + 5.0
            
            # 添加流向波動 (簡化湍流結構)
            domain_x_len = domain_x[1] - domain_x[0]
            u_fluctuation = 0.1 * np.sin(2 * np.pi * X / domain_x_len) * \
                           np.exp(-10 * np.abs(Y))
            
            prior_data = {
                'coordinates': {'x': x, 'y': y},
                'u': u_mean + u_fluctuation,
                'v': 0.05 * np.sin(4 * np.pi * X / domain_x_len) * \
                     np.sin(np.pi * Y / 2),  # 小幅壁法向速度
                'p': -1.0 * X + 0.01 * np.sin(2 * np.pi * X / domain_x_len),  # 壓力梯度
            }
            
        else:
            raise ValueError(f"未支援的先驗類型: {prior_type}")
        
        # 快取先驗資料
        cache_file = self.cache_dir / f"lowfi_prior_{prior_type}_{resolution[0]}x{resolution[1]}.npz"
        np.savez_compressed(cache_file, **prior_data)
        self.logger.info(f"低保真先驗已快取至: {cache_file}")
        
        return prior_data
    
    def create_complete_dataset(self,
                              resolution: Tuple[int, int] = (128, 64),
                              K: int = 8,
                              sensor_method: str = "qr_pivot") -> Dict[str, Any]:
        """
        建立完整的 Channel Flow Re1000 資料集
        
        包含：高保真場、稀疏感測點、低保真先驗
        """
        self.logger.info("建立完整 Channel Flow Re1000 資料集")
        
        # 1. 獲取高保真場資料
        hifi_data = self.fetch_cutout_data(resolution)
        
        # 2. 生成稀疏感測點
        sensor_data = self.generate_sensor_points(hifi_data, K, sensor_method)
        
        # 3. 生成低保真先驗
        lowfi_data = self.generate_lowfi_prior(resolution)
        
        # 4. 組合完整資料集
        dataset = {
            'hifi_data': hifi_data,
            'sensor_data': sensor_data,
            'lowfi_data': lowfi_data,
            'config': self.config.__dict__,
            'metadata': {
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S'),
                'resolution': resolution,
                'K_sensors': K,
                'sensor_method': sensor_method,
                'Re_tau': self.config.Re_tau
            }
        }
        
        # 快取完整資料集
        cache_file = self.cache_dir / f"complete_dataset_{resolution[0]}x{resolution[1]}_K{K}.npz"
        np.savez_compressed(cache_file, **dataset)
        self.logger.info(f"完整資料集已快取至: {cache_file}")
        
        return dataset
    
    def visualize_data(self, dataset: Dict[str, Any], output_dir: Optional[str] = None):
        """資料視覺化"""
        if output_dir is None:
            output_dir_path = self.cache_dir / "visualization"
        else:
            output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        
        hifi_data = dataset['hifi_data']
        sensor_data = dataset['sensor_data']
        
        # 建立圖表
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Channel Flow Re1000 Dataset Visualization', fontsize=16)
        
        # 座標
        x = hifi_data['coordinates']['x']
        y = hifi_data['coordinates']['y']
        
        # 感測點座標
        sensor_points = sensor_data['sensor_points']
        
        # 繪製各變數場
        for i, var in enumerate(['u', 'v', 'p']):
            if var in hifi_data:
                # 完整場
                ax1 = axes[0, i]
                im1 = ax1.contourf(x, y, hifi_data[var].T, levels=50, cmap='RdBu_r')
                ax1.scatter(sensor_points[:, 0], sensor_points[:, 1], 
                           c='black', s=30, marker='o', label=f'Sensors (K={len(sensor_points)})')
                ax1.set_title(f'High-Fi {var.upper()} Field')
                ax1.set_xlabel('x')
                ax1.set_ylabel('y')
                ax1.legend()
                plt.colorbar(im1, ax=ax1)
                
                # 感測點數值
                ax2 = axes[1, i]
                if var in sensor_data['sensor_data']:
                    sensor_values = sensor_data['sensor_data'][var]
                    valid_mask = ~np.isnan(sensor_values)
                    ax2.scatter(sensor_points[valid_mask, 0], sensor_points[valid_mask, 1], 
                               c=sensor_values[valid_mask], s=50, cmap='RdBu_r')
                    ax2.set_title(f'Sensor {var.upper()} Values')
                    ax2.set_xlabel('x')
                    ax2.set_ylabel('y')
        
        plt.tight_layout()
        
        # 儲存圖表
        plot_file = output_dir_path / "dataset_overview.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"資料視覺化已儲存至: {plot_file}")


def load_config(config_file: str) -> ChannelFlowConfig:
    """載入配置檔案"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # 提取 Channel Flow 相關配置
    jhtdb_config = config_dict.get('data', {}).get('jhtdb_config', {})
    
    return ChannelFlowConfig(
        dataset=jhtdb_config.get('dataset_name', 'channel'),
        domain_x=jhtdb_config.get('domain', {}).get('x', [0, 25.13]),
        domain_y=jhtdb_config.get('domain', {}).get('y', [-1, 1]),
        domain_z=jhtdb_config.get('domain', {}).get('z', [0, 9.42]),
        Re_tau=config_dict.get('physics', {}).get('channel_flow', {}).get('Re_tau', 1000.0),
        slice_position=config_dict.get('data', {}).get('slice_config', {}).get('z_position', 4.71)
    )


def main():
    """主程式"""
    parser = argparse.ArgumentParser(description='Channel Flow Re1000 JHTDB 資料擷取')
    
    # 基本參數
    parser.add_argument('--config', type=str, default='configs/channel_flow_re1000.yml',
                       help='配置檔案路徑')
    parser.add_argument('--cache_dir', type=str, default='./data/jhtdb/channel_flow_re1000',
                       help='快取目錄')
    parser.add_argument('--log_level', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='日誌等級')
    
    # 運行模式
    parser.add_argument('--cutout_mode', action='store_true',
                       help='只下載 cutout 資料')
    parser.add_argument('--sensor_only', action='store_true', 
                       help='只生成感測點資料')
    parser.add_argument('--complete_dataset', action='store_true',
                       help='建立完整資料集 (預設)')
    
    # 參數設定
    parser.add_argument('--resolution', type=int, nargs=2, default=[128, 64],
                       help='2D 切片解析度 [nx, ny]')
    parser.add_argument('--K', type=int, default=8,
                       help='感測點數量')
    parser.add_argument('--sensor_method', type=str, default='qr_pivot',
                       choices=['qr_pivot', 'random'],
                       help='感測點選擇方法')
    parser.add_argument('--noise_sigma', type=float, default=0.02,
                       help='感測點噪聲標準差')
    parser.add_argument('--dropout_prob', type=float, default=0.10,
                       help='感測點隨機遺失比例')
    
    # 輸出選項
    parser.add_argument('--visualize', action='store_true',
                       help='生成資料視覺化')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='輸出目錄')
    
    args = parser.parse_args()
    
    # 設定日誌
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('fetch_channel_flow.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    try:
        # 載入配置
        config = load_config(args.config)
        logger.info(f"載入配置檔案: {args.config}")
        
        # 建立資料擷取器
        fetcher = ChannelFlowDataFetcher(config, args.cache_dir)
        resolution = tuple(args.resolution)
        
        # 執行對應模式
        if args.cutout_mode:
            logger.info("執行 cutout 模式")
            hifi_data = fetcher.fetch_cutout_data(resolution)
            logger.info(f"完成 cutout 資料下載，形狀: {hifi_data['u'].shape}")
            
        elif args.sensor_only:
            logger.info("執行感測點生成模式")
            # 需要先載入現有的高保真資料
            cache_file = Path(args.cache_dir) / f"cutout_{resolution[0]}x{resolution[1]}.npz"
            if cache_file.exists():
                hifi_data = dict(np.load(cache_file, allow_pickle=True))
                sensor_data = fetcher.generate_sensor_points(
                    hifi_data, args.K, args.sensor_method, 
                    args.noise_sigma, args.dropout_prob
                )
                logger.info(f"完成 {args.K} 個感測點生成")
            else:
                logger.error(f"未找到高保真資料快取: {cache_file}")
                return 1
                
        else:
            # 預設：建立完整資料集
            logger.info("執行完整資料集建立模式")
            dataset = fetcher.create_complete_dataset(resolution, args.K, args.sensor_method)
            logger.info("完成完整資料集建立")
            
            # 可視化
            if args.visualize:
                fetcher.visualize_data(dataset, args.output_dir)
        
        logger.info("資料擷取完成")
        return 0
        
    except Exception as e:
        logger.error(f"資料擷取失敗: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())