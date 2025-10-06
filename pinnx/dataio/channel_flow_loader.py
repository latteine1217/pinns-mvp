"""
Channel Flow Re1000 專用資料載入器

提供統一介面載入和處理 Channel Flow Re1000 相關資料：
- NPZ 快取的感測點資料 (QR-pivot/Random)
- 低保真 RANS 先驗資料
- JHTDB 配置與域參數
- PINNs 訓練所需的資料格式標準化

主要功能：
1. 感測點資料載入與驗證
2. 低保真先驗插值到 PINNs 訓練點
3. VS-PINN 尺度化統計資訊提取
4. 與現有訓練流程完全相容
5. 快取管理與資料完整性檢查

設計原則：
- 與 scripts/train.py 無縫整合
- 僅支援真實 JHTDB 資料，移除Mock功能
- 高效率的記憶體管理
- 完整的錯誤處理機制
"""

import numpy as np
import yaml
from typing import Dict, List, Optional, Union, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import warnings

# 導入現有模組
from .lowfi_loader import LowFiData, LowFiLoader, SpatialInterpolator
from .jhtdb_client import JHTDBManager, JHTDBConfig


logger = logging.getLogger(__name__)


@dataclass
class ChannelFlowData:
    """Channel Flow 資料容器"""
    # 感測點資料
    sensor_points: np.ndarray          # [K, 2] 感測點座標
    sensor_data: Dict[str, np.ndarray] # 感測點處的場值
    sensor_indices: np.ndarray         # [K] 在全場中的索引
    selection_info: Dict[str, Any]     # 選擇策略資訊
    
    # 域配置
    domain_config: Dict[str, Any]      # 域參數 (Re_tau, nu, 邊界等)
    coordinate_info: Dict[str, Any]    # 座標系統資訊
    
    # 低保真先驗 (可選)
    lowfi_prior: Optional[Dict[str, np.ndarray]] = None
    lowfi_metadata: Optional[Dict[str, Any]] = None
    
    # 統計資訊 (VS-PINN 用)
    statistics: Optional[Dict[str, Dict[str, float]]] = None
    
    # 元數據
    metadata: Optional[Dict[str, Any]] = None
    
    def get_training_points(self) -> np.ndarray:
        """獲取 PINNs 訓練點座標"""
        return self.sensor_points
    
    def get_training_data(self) -> Dict[str, np.ndarray]:
        """獲取 PINNs 訓練資料"""
        return self.sensor_data
    
    def get_domain_bounds(self) -> Dict[str, Tuple[float, float]]:
        """獲取域邊界"""
        bounds = {}
        if 'x_range' in self.domain_config:
            bounds['x'] = tuple(self.domain_config['x_range'])
        if 'y_range' in self.domain_config:
            bounds['y'] = tuple(self.domain_config['y_range'])
        return bounds
    
    def get_physical_parameters(self) -> Dict[str, float]:
        """獲取物理參數"""
        params = {}
        for key in ['Re_tau', 'nu', 'u_tau']:
            if key in self.domain_config:
                params[key] = self.domain_config[key]
        return params
    
    def has_lowfi_prior(self) -> bool:
        """檢查是否有低保真先驗"""
        return self.lowfi_prior is not None and len(self.lowfi_prior) > 0


class ChannelFlowLoader:
    """Channel Flow Re1000 專用載入器"""
    
    def __init__(self, 
                 config_path: Optional[Union[str, Path]] = None,
                 cache_dir: Optional[Union[str, Path]] = None,
                 interpolation_method: str = 'linear'):
        """
        初始化載入器
        
        Args:
            config_path: 配置檔案路徑，預設 configs/channel_flow_re1000.yml
            cache_dir: 快取目錄，預設 data/jhtdb/channel_flow_re1000/
            interpolation_method: 插值方法 ('linear', 'rbf', 'idw')
        """
        # 設定路徑
        self.config_path = Path(config_path) if config_path else Path('configs/channel_flow_re1000.yml')
        self.cache_dir = Path(cache_dir) if cache_dir else Path('data/jhtdb/channel_flow_re1000/')
        
        # 載入配置
        self.config = self._load_config()
        
        # 初始化工具
        self.lowfi_loader = LowFiLoader()
        self.interpolator = SpatialInterpolator(method=interpolation_method)
        
        # JHTDB 管理器 (如果需要)
        self.jhtdb_manager = None
        self._init_jhtdb_manager()
        
        logger.info(f"Channel Flow loader initialized with config: {self.config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """載入配置檔案"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {self.config_path}")
        return config
    
    def _init_jhtdb_manager(self):
        """初始化 JHTDB 管理器"""
        try:
            # 從配置中提取 JHTDB 參數
            jhtdb_config = self.config.get('jhtdb', {})
            if jhtdb_config.get('enabled', False):
                self.jhtdb_manager = JHTDBManager(
                    cache_root=self.cache_dir.parent,
                    auth_token=jhtdb_config.get('auth_token')
                )
                logger.info("JHTDB manager initialized")
            else:
                raise RuntimeError("JHTDB is disabled and no mock fallback available. Please enable JHTDB to proceed.")
        except Exception as e:
            logger.warning(f"Failed to initialize JHTDB manager: {e}")
            self.jhtdb_manager = None
    
    def load_sensor_data(self, 
                        strategy: str = 'qr_pivot',
                        K: int = 8,
                        noise_sigma: Optional[float] = None,
                        dropout_prob: Optional[float] = None) -> ChannelFlowData:
        """
        載入感測點資料
        
        Args:
            strategy: 選擇策略 ('qr_pivot', 'random', 'uniform')
            K: 感測點數量
            noise_sigma: 噪聲水平 (可選)
            dropout_prob: 丟失概率 (可選)
            
        Returns:
            Channel Flow 資料容器
        """
        # 構建快取檔案名
        cache_filename = f"sensors_K{K}_{strategy}.npz"
        cache_path = self.cache_dir / cache_filename
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Sensor data not found: {cache_path}\n"
                f"Please run scripts/fetch_channel_flow.py first"
            )
        
        logger.info(f"Loading sensor data from {cache_path}")
        
        # 載入 NPZ 資料
        data = np.load(cache_path, allow_pickle=True)
        
        # 提取感測點資訊
        sensor_points = data['sensor_points']
        
        # 處理 sensor_data (可能是物件或分離的陣列)
        if 'sensor_data' in data:
            # 新格式：sensor_data 是包含所有場的字典
            sensor_data = data['sensor_data'].item() if data['sensor_data'].ndim == 0 else data['sensor_data']
        else:
            # 舊格式：分離的 sensor_u, sensor_v, sensor_p
            sensor_data = {}
            for field in ['u', 'v', 'p']:
                key = f'sensor_{field}'
                if key in data:
                    sensor_data[field] = data[key]
        
        sensor_indices = data['sensor_indices']
        
        # 提取選擇資訊
        if 'selection_info' in data:
            # 新格式：selection_info 是物件
            selection_info = data['selection_info'].item() if data['selection_info'].ndim == 0 else data['selection_info']
        else:
            # 舊格式：分離的欄位
            selection_info = {
                'strategy': str(data.get('strategy', strategy)),
                'K_requested': int(data.get('K_requested', K)),
                'K_actual': len(sensor_points),
                'selection_timestamp': str(data.get('timestamp', 'unknown'))
            }
        
        # 添加噪聲 (如果指定)
        if noise_sigma is not None and noise_sigma > 0:
            sensor_data = self._add_noise(sensor_data, noise_sigma)
            selection_info['noise_sigma'] = noise_sigma
        
        # 添加丟失 (如果指定)
        if dropout_prob is not None and dropout_prob > 0:
            sensor_data, valid_mask = self._add_dropout(sensor_data, dropout_prob)
            sensor_points = sensor_points[valid_mask]
            sensor_indices = sensor_indices[valid_mask]
            selection_info['dropout_prob'] = dropout_prob
            selection_info['K_after_dropout'] = len(sensor_points)
        
        # 提取域配置
        domain_config = self._extract_domain_config()
        coordinate_info = self._extract_coordinate_info(data)
        
        # 創建資料容器
        channel_data = ChannelFlowData(
            sensor_points=sensor_points,
            sensor_data=sensor_data,
            sensor_indices=sensor_indices,
            selection_info=selection_info,
            domain_config=domain_config,
            coordinate_info=coordinate_info,
            metadata={
                'source': str(cache_path),
                'config_file': str(self.config_path),
                'loader_version': '1.0',
                'loaded_timestamp': str(np.datetime64('now'))
            }
        )
        
        logger.info(f"Loaded {len(sensor_points)} sensor points using {strategy} strategy")
        return channel_data
    
    def load_full_field_data(self, 
                           noise_sigma: Optional[float] = None) -> ChannelFlowData:
        """
        載入完整流場數據（所有8×8×8=512個網格點）
        
        Args:
            noise_sigma: 噪聲水平 (可選)
            
        Returns:
            包含完整流場的 Channel Flow 資料容器
        """
        # 確定HDF5文件路徑
        # 首先尋找物理一致的數據文件
        physics_consistent_file = self.cache_dir.parent / "physics_consistent_8x8x8.h5"
        
        if physics_consistent_file.exists():
            hdf5_path = physics_consistent_file
        else:
            # 尋找其他HDF5文件
            hdf5_files = list(self.cache_dir.parent.glob("*.h5"))
            
            if not hdf5_files:
                raise FileNotFoundError(
                    f"No HDF5 full field data found in {self.cache_dir.parent}\n"
                    f"Please ensure the HDF5 file exists and was generated from JHTDB data"
                )
            
            # 使用第一個找到的HDF5文件
            hdf5_path = hdf5_files[0]
        logger.info(f"Loading full field data from {hdf5_path}")
        
        # 載入HDF5數據
        import h5py
        
        with h5py.File(hdf5_path, 'r') as f:
            # 載入所有場數據
            u = np.array(f['u'])  # (8,8,8)
            v = np.array(f['v'])  # (8,8,8)
            w = np.array(f['w'])  # (8,8,8)
            p = np.array(f['p'])  # (8,8,8)
        
        # 創建3D網格座標
        # 假設域範圍來自配置文件
        domain_config = self._extract_domain_config()
        x_range = domain_config.get('x_range', [0.0, 25.13])
        y_range = domain_config.get('y_range', [-1.0, 1.0])
        z_range = domain_config.get('z_range', [0.0, 9.42])
        
        # 創建均勻網格
        x = np.linspace(x_range[0], x_range[1], 8)
        y = np.linspace(y_range[0], y_range[1], 8)
        z = np.linspace(z_range[0], z_range[1], 8)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 將3D座標展平為 [N, 3] 格式
        coordinates = np.stack([X.flatten(), Y.flatten(), Z.flatten()], axis=1)
        
        # 將場數據展平為 [N] 格式
        sensor_data = {
            'u': u.flatten(),
            'v': v.flatten(),
            'w': w.flatten(),
            'p': p.flatten()
        }
        
        # 添加噪聲 (如果指定)
        if noise_sigma is not None and noise_sigma > 0:
            sensor_data = self._add_noise(sensor_data, noise_sigma)
        
        # 創建感測點索引 (所有點)
        sensor_indices = np.arange(len(coordinates))
        
        # 選擇資訊
        selection_info = {
            'method': 'full_field',
            'n_sensors': len(coordinates),
            'grid_shape': (8, 8, 8),
            'total_points': len(coordinates)
        }
        
        if noise_sigma is not None:
            selection_info['noise_sigma'] = noise_sigma
        
        # 提取座標系統資訊
        coordinate_info = {
            'type': '3D_cartesian',
            'x_range': x_range,
            'y_range': y_range, 
            'z_range': z_range,
            'grid_dimensions': (8, 8, 8),
            'coordinate_order': ['x', 'y', 'z']
        }
        
        # 創建資料容器
        channel_data = ChannelFlowData(
            sensor_points=coordinates,
            sensor_data=sensor_data,
            sensor_indices=sensor_indices,
            selection_info=selection_info,
            domain_config=domain_config,
            coordinate_info=coordinate_info,
            metadata={
                'source': str(hdf5_path),
                'config_file': str(self.config_path),
                'loader_version': '1.0',
                'loaded_timestamp': str(np.datetime64('now')),
                'data_type': 'full_field'
            }
        )
        
        logger.info(f"Loaded {len(coordinates)} full field points from 8×8×8 grid")
        return channel_data
    
    def add_lowfi_prior(self, 
                       channel_data: ChannelFlowData,
                       prior_type: str = 'rans',
                       interpolate_to_sensors: bool = True) -> ChannelFlowData:
        """
        添加低保真先驗資料
        
        Args:
            channel_data: 現有的 Channel Flow 資料
            prior_type: 先驗類型 ('rans', 'none')
            interpolate_to_sensors: 是否插值到感測點
            
        Returns:
            添加先驗後的資料容器
        """
        if prior_type == 'none':
            logger.info("No low-fidelity prior requested")
            return channel_data
        
        try:
            if prior_type == 'rans':
                # 載入真實 RANS 資料 (如果可用)
                lowfi_data = self._load_rans_prior()
            else:
                raise ValueError(f"Unknown prior type: {prior_type}. Only 'rans' and 'none' are supported.")
            
            # 插值到感測點 (如果需要)
            if interpolate_to_sensors:
                prior_fields = self.interpolator.interpolate_to_points(
                    lowfi_data, 
                    channel_data.sensor_points
                )
            else:
                # 使用全場資料
                prior_fields = lowfi_data.fields
            
            # 計算統計資訊 (VS-PINN 用)
            statistics = self.lowfi_loader.get_statistics(lowfi_data)
            
            # 更新資料容器
            channel_data.lowfi_prior = prior_fields
            channel_data.lowfi_metadata = lowfi_data.metadata
            channel_data.statistics = statistics
            
            logger.info(f"Added {prior_type} low-fidelity prior with {len(prior_fields)} fields")
            
        except Exception as e:
            logger.warning(f"Failed to load low-fidelity prior: {e}")
            # 創建空的先驗資料
            channel_data.lowfi_prior = {}
            channel_data.lowfi_metadata = {'type': 'none', 'error': str(e)}
            channel_data.statistics = {}
        
        return channel_data
    
    def prepare_for_training(self, 
                           channel_data: ChannelFlowData,
                           target_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        準備 PINNs 訓練資料格式
        
        Args:
            channel_data: Channel Flow 資料
            target_fields: 目標場列表，預設 ['u', 'v', 'p']
            
        Returns:
            PINNs 訓練所需的資料字典
        """
        if target_fields is None:
            target_fields = ['u', 'v', 'p']
        
        training_data = {
            # 座標
            'coordinates': channel_data.sensor_points,
            
            # 感測資料
            'sensor_data': {
                field: channel_data.sensor_data[field] 
                for field in target_fields 
                if field in channel_data.sensor_data
            },
            
            # 域配置
            'domain_bounds': channel_data.get_domain_bounds(),
            'physical_params': channel_data.get_physical_parameters(),
            
            # 低保真先驗 (如果有)
            'lowfi_prior': channel_data.lowfi_prior if channel_data.has_lowfi_prior() else {},
            
            # VS-PINN 統計
            'statistics': channel_data.statistics or {},
            
            # 元數據
            'metadata': {
                **(channel_data.metadata or {}),
                'selection_info': channel_data.selection_info,
                'coordinate_info': channel_data.coordinate_info,
                'has_lowfi_prior': channel_data.has_lowfi_prior(),
                'target_fields': target_fields
            }
        }
        
        logger.info(f"Prepared training data with {len(training_data['sensor_data'])} fields")
        return training_data
    
    def _add_noise(self, 
                  sensor_data: Dict[str, np.ndarray], 
                  noise_sigma: float) -> Dict[str, np.ndarray]:
        """添加高斯噪聲到感測資料"""
        noisy_data = {}
        for field, values in sensor_data.items():
            noise = np.random.normal(0, noise_sigma * np.std(values), values.shape)
            noisy_data[field] = values + noise
        
        logger.debug(f"Added Gaussian noise with sigma={noise_sigma}")
        return noisy_data
    
    def _add_dropout(self, 
                    sensor_data: Dict[str, np.ndarray], 
                    dropout_prob: float) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """隨機丟失部分感測點"""
        n_points = len(next(iter(sensor_data.values())))
        valid_mask = np.random.random(n_points) > dropout_prob
        
        dropped_data = {}
        for field, values in sensor_data.items():
            dropped_data[field] = values[valid_mask]
        
        logger.debug(f"Applied dropout with prob={dropout_prob}, kept {np.sum(valid_mask)}/{n_points} points")
        return dropped_data, valid_mask
    
    def _extract_domain_config(self) -> Dict[str, Any]:
        """從配置檔案提取域參數"""
        domain_config = {}
        
        # 從 data.jhtdb_config 提取 Channel Flow 參數
        if 'data' in self.config and 'jhtdb_config' in self.config['data']:
            jhtdb_config = self.config['data']['jhtdb_config']
            
            # 域範圍
            if 'domain' in jhtdb_config:
                domain = jhtdb_config['domain']
                domain_config.update({
                    'x_range': domain.get('x', [0.0, 25.13]),
                    'y_range': domain.get('y', [-1.0, 1.0]),
                    'z_range': domain.get('z', [0.0, 9.42])
                })
            
            # 解析度
            if 'resolution' in jhtdb_config:
                resolution = jhtdb_config['resolution']
                domain_config.update({
                    'nx': resolution.get('x', 2048),
                    'ny': resolution.get('y', 512),
                    'nz': resolution.get('z', 1536)
                })
            
            # 時間參數
            domain_config.update({
                'time_range': jhtdb_config.get('time_range', [0.0, 26.0]),
                'dt': jhtdb_config.get('dt', 0.0065)
            })
        
        # 從 physics 段落提取物理參數
        if 'physics' in self.config:
            physics_config = self.config['physics']
            domain_config.update({
                'Re_tau': physics_config.get('Re_tau', 1000),
                'nu': physics_config.get('nu', 1e-3),
                'u_tau': physics_config.get('u_tau', 1.0),
                'rho': physics_config.get('rho', 1.0)
            })
        
        # 2D 切片配置 (如果可用)
        if 'data' in self.config and 'slice_config' in self.config['data']:
            slice_config = self.config['data']['slice_config']
            domain_config.update({
                'slice_plane': slice_config.get('plane', 'xy'),
                'slice_position': slice_config.get('z_position', 4.71),
                'steady_state': slice_config.get('steady_state', True)
            })
        
        return domain_config
    
    def _extract_coordinate_info(self, data) -> Dict[str, Any]:
        """從 NPZ 資料提取座標資訊"""
        coord_info = {}
        
        # 提取座標陣列 (如果可用)
        if 'x_coords' in data:
            coord_info['x_coords'] = data['x_coords']
        if 'y_coords' in data:
            coord_info['y_coords'] = data['y_coords']
        
        # 提取網格資訊
        for key in ['nx', 'ny', 'x_range', 'y_range']:
            if key in data:
                coord_info[key] = data[key]
        
        return coord_info
    


    def _load_rans_prior(self) -> LowFiData:
        """載入真實 RANS 先驗資料"""
        # 尋找 RANS 資料檔案
        rans_patterns = ['rans_data.npz', 'lowfi_prior.npz', 'rans_baseline.nc']
        
        for pattern in rans_patterns:
            rans_path = self.cache_dir / pattern
            if rans_path.exists():
                logger.info(f"Loading RANS prior from {rans_path}")
                return self.lowfi_loader.load(rans_path, data_type='rans')
        
        # 無法找到RANS資料，直接拋出錯誤
        raise FileNotFoundError(
            f"No RANS prior data found in {self.cache_dir}. "
            f"Searched for: {rans_patterns}. "
            f"Mock fallback has been removed for this system."
        )
    
    def get_available_datasets(self) -> List[str]:
        """獲取可用的資料集列表"""
        available = []
        
        if self.cache_dir.exists():
            for npz_file in self.cache_dir.glob("sensors_K*_*.npz"):
                parts = npz_file.stem.split('_')
                if len(parts) >= 3:
                    K = parts[1][1:]  # 移除 'K' 前綴
                    strategy = '_'.join(parts[2:])
                    available.append(f"K{K}_{strategy}")
        
        return sorted(available)
    
    def validate_data(self, channel_data: ChannelFlowData) -> Dict[str, bool]:
        """驗證資料完整性和物理合理性"""
        checks = {}
        
        # 基本結構檢查
        checks['has_sensor_points'] = len(channel_data.sensor_points) > 0
        checks['has_sensor_data'] = len(channel_data.sensor_data) > 0
        checks['has_domain_config'] = len(channel_data.domain_config) > 0
        
        # 資料維度一致性
        if channel_data.sensor_points.size > 0:
            n_points = len(channel_data.sensor_points)
            for field, values in channel_data.sensor_data.items():
                checks[f'{field}_dimension_match'] = len(values) == n_points
        
        # 物理合理性
        for field, values in channel_data.sensor_data.items():
            checks[f'{field}_finite'] = np.all(np.isfinite(values))
            if field in ['u', 'v']:
                # 調整 Channel Flow Re1000 的合理範圍
                max_reasonable = 30.0 if field == 'u' else 5.0  # u可達25+, v較小
                checks[f'{field}_reasonable'] = np.abs(values).max() < max_reasonable
        
        # 域參數合理性
        domain = channel_data.domain_config
        if 'Re_tau' in domain:
            checks['Re_tau_reasonable'] = 100 <= domain['Re_tau'] <= 10000
        if 'nu' in domain:
            checks['nu_positive'] = domain['nu'] > 0
        
        # 低保真先驗檢查 (如果有)
        if channel_data.has_lowfi_prior() and channel_data.lowfi_prior:
            checks['lowfi_prior_available'] = True
            for field in ['u', 'v', 'p']:
                if field in channel_data.lowfi_prior:
                    values = channel_data.lowfi_prior[field]
                    checks[f'lowfi_{field}_finite'] = np.all(np.isfinite(values))
        
        # 統計資訊檢查
        if channel_data.statistics:
            checks['statistics_available'] = True
        
        return checks


# 便利函數
def load_channel_flow_data(strategy: str = 'qr_pivot',
                          K: int = 8,
                          config_path: Optional[Union[str, Path]] = None,
                          with_lowfi_prior: bool = True,
                          prior_type: str = 'rans') -> ChannelFlowData:
    """
    便利函數：載入 Channel Flow 資料
    
    Args:
        strategy: 感測點選擇策略
        K: 感測點數量  
        config_path: 配置檔案路徑
        with_lowfi_prior: 是否載入低保真先驗
        prior_type: 先驗類型
        
    Returns:
        Channel Flow 資料容器
    """
    loader = ChannelFlowLoader(config_path=config_path)
    
    # 載入感測點資料
    channel_data = loader.load_sensor_data(strategy=strategy, K=K)
    
    # 添加低保真先驗 (如果需要)
    if with_lowfi_prior:
        channel_data = loader.add_lowfi_prior(channel_data, prior_type=prior_type)
    
    return channel_data


def prepare_training_data(strategy: str = 'qr_pivot',
                         K: int = 8, 
                         config_path: Optional[Union[str, Path]] = None,
                         target_fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    便利函數：準備 PINNs 訓練資料
    
    Args:
        strategy: 感測點選擇策略
        K: 感測點數量
        config_path: 配置檔案路徑  
        target_fields: 目標場列表
        
    Returns:
        PINNs 訓練資料字典
    """
    loader = ChannelFlowLoader(config_path=config_path)
    
    # 載入完整資料
    channel_data = loader.load_sensor_data(strategy=strategy, K=K)
    channel_data = loader.add_lowfi_prior(channel_data, prior_type='mock')
    
    # 準備訓練格式
    training_data = loader.prepare_for_training(channel_data, target_fields=target_fields)
    
    return training_data