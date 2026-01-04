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
from .lowfi_loader import LowFiData, LowFiLoader, SpatialInterpolator, NPZReader, DataReader
from .jhtdb_client import JHTDBManager, JHTDBConfig
from .structures import (
    StructuredGrid,
    StructuredField,
    PointSamples,
    FlowDataBundle,
    DomainSpec,
)


logger = logging.getLogger(__name__)


class SensorDataReader(NPZReader):
    """
    Channel Flow 感測點資料專用讀取器
    
    擴展 NPZReader 以支援：
    - coords_2d → 3D 擴展 (添加 z 座標)
    - 多種 sensor_data 格式 (物件/2D array/分離欄位)
    - selection_info 元數據提取
    """
    
    def __init__(self, z_default: float = 4.71):
        """
        初始化感測點資料讀取器
        
        Args:
            z_default: 2D 座標擴展為 3D 時的預設 z 值
        """
        super().__init__()
        self.z_default = z_default
    
    def read(self, filepath: Union[str, Path]) -> LowFiData:
        """讀取感測點 NPZ 檔案"""
        data_dict = np.load(filepath, allow_pickle=True)
        
        # 處理座標 (支援 2D → 3D 轉換)
        coordinates = self._extract_coordinates(data_dict)
        
        # 處理感測點資料 (支援多種格式)
        fields = self._extract_sensor_data(data_dict)
        
        # 提取元數據 (包含 selection_info)
        metadata = self._build_sensor_metadata(filepath, data_dict)
        
        return LowFiData(coordinates, fields, metadata)
    
    def _extract_coordinates(self, data: dict) -> Dict[str, np.ndarray]:
        """
        提取座標，支援 2D → 3D 轉換
        
        支援的鍵名：
        - 'sensor_points': 直接的 3D 座標 (K, 3)
        - 'coords': 新格式的 3D 座標
        - 'coords_2d': 2D 座標，需要擴展到 3D
        """
        if 'sensor_points' in data:
            # 標準格式：直接使用
            points = np.asarray(data['sensor_points'])
        elif 'coords' in data:
            # 備選格式：coords
            points = np.asarray(data['coords'])
        elif 'coords_2d' in data:
            # 2D → 3D 擴展
            coords_2d = np.asarray(data['coords_2d'])
            points = np.column_stack([
                coords_2d[:, 0],  # x
                coords_2d[:, 1],  # y
                np.full(len(coords_2d), self.z_default)  # z (constant)
            ])
        elif 'sensor_x' in data and 'sensor_y' in data:
            # ✅ 新增：支援分離座標格式 (sensor_x, sensor_y, sensor_z)
            sensor_x = np.asarray(data['sensor_x'])
            sensor_y = np.asarray(data['sensor_y'])
            
            if 'sensor_z' in data:
                # 3D 座標
                sensor_z = np.asarray(data['sensor_z'])
                points = np.column_stack([sensor_x, sensor_y, sensor_z])
            else:
                # 2D 座標，擴展為 3D
                sensor_z = np.full(len(sensor_x), self.z_default)
                points = np.column_stack([sensor_x, sensor_y, sensor_z])
        else:
            raise KeyError(
                f"Cannot find sensor coordinates. "
                f"Expected 'sensor_points', 'coords', 'coords_2d', or 'sensor_x/sensor_y/sensor_z'"
            )
        
        # 返回標準格式：{'x': [...], 'y': [...], 'z': [...]}
        return {
            'x': points[:, 0],
            'y': points[:, 1],
            'z': points[:, 2] if points.shape[1] >= 3 else np.full(len(points), self.z_default)
        }
    
    def _extract_sensor_data(self, data: dict) -> Dict[str, np.ndarray]:
        """
        提取感測點資料，支援多種格式
        
        格式 1: 'sensor_data' 鍵 (物件或 2D array)
        格式 2: 分離的 'sensor_u', 'sensor_v' 等
        格式 3: 直接的 'u', 'v', 'w', 'p' 鍵
        """
        fields = {}
        
        if 'sensor_data' in data:
            sensor_data_raw = data['sensor_data']
            
            # 情況 1: 0 維物件 (字典)
            if sensor_data_raw.ndim == 0:
                sensor_data_raw = sensor_data_raw.item()
                if isinstance(sensor_data_raw, dict):
                    fields = {k: np.asarray(v).reshape(-1) for k, v in sensor_data_raw.items()}
                else:
                    raise TypeError(f"sensor_data object must be a dict, got {type(sensor_data_raw)}")
            
            # 情況 2: 2D ndarray (K, n_vars)
            elif sensor_data_raw.ndim == 2:
                variables = self._infer_variable_names(data, sensor_data_raw.shape[1])
                fields = {
                    var: sensor_data_raw[:, i]
                    for i, var in enumerate(variables)
                }
            
            else:
                raise ValueError(f"sensor_data has unexpected ndim: {sensor_data_raw.ndim}")
        
        else:
            # 情況 3: 分離的欄位
            # 優先檢查 'sensor_*' 格式
            for field in ['u', 'v', 'w', 'p']:
                key_sensor = f'sensor_{field}'
                if key_sensor in data:
                    fields[field] = np.asarray(data[key_sensor]).reshape(-1)

            # 兼容新的 *_sensors 格式 (e.g., u_sensors)
            if not fields:
                for field in ['u', 'v', 'w', 'p']:
                    key_sensors = f'{field}_sensors'
                    if key_sensors in data:
                        fields[field] = np.asarray(data[key_sensors]).reshape(-1)
            
            # 若無 'sensor_*'，嘗試直接鍵名
            if not fields:
                for field in ['u', 'v', 'w', 'p']:
                    if field in data:
                        fields[field] = np.asarray(data[field]).reshape(-1)
            
            # ✅ 允許純座標檔案（沒有 velocity/pressure 資料）
            # 這種情況下，資料將從 JHTDB 或 RANS prior 取得
            if not fields:
                logger.info("No velocity/pressure data found in sensor file (coordinates-only file)")
        
        return fields
    
    def _infer_variable_names(self, data: dict, n_vars: int) -> List[str]:
        """從 metadata 或欄位數推斷變數名稱"""
        if 'metadata' in data:
            metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
            if 'variables' in metadata:
                return metadata['variables']
        
        # 根據欄位數判斷
        if n_vars == 2:
            return ['u', 'v']
        elif n_vars == 3:
            return ['u', 'v', 'w']
        elif n_vars == 4:
            return ['u', 'v', 'w', 'p']
        else:
            raise ValueError(f"Cannot infer variable names for {n_vars} columns")
    
    def _build_sensor_metadata(self, filepath: Union[str, Path], data: dict) -> Dict[str, Any]:
        """構建感測點元數據，包含 selection_info 和標量 metadata"""
        # 使用基類的 build_metadata
        metadata = self.build_metadata(filepath, 'NPZ-Sensor')
        
        # ✅ 提取標量 metadata（直接從 NPZ keys）
        scalar_keys = ['K', 'condition_number', 'energy_ratio', 'method', 
                       'seam_weight', 'source_file', 'ndim', 'grid_shape', 
                       'periodic_axes', 'domain_lengths']
        
        for key in scalar_keys:
            if key in data:
                value = data[key]
                # 處理 numpy scalar/array
                if hasattr(value, 'shape'):
                    if value.shape == ():
                        # 0-dim array (scalar)
                        metadata[key] = value.item()
                    else:
                        # Multi-dim array
                        metadata[key] = value.tolist()
                else:
                    metadata[key] = value
        
        # 提取 selection_info
        if 'selection_info' in data:
            selection_info = data['selection_info']
            metadata['selection_info'] = selection_info.item() if selection_info.ndim == 0 else selection_info
        else:
            # 嘗試從其他欄位構建
            # 計算實際的 K 值
            K_actual = 0
            if 'sensor_x' in data:
                K_actual = len(data['sensor_x'])
            elif 'coords' in data:
                K_actual = len(data['coords'])
            elif 'sensor_points' in data:
                K_actual = len(data['sensor_points'])
            
            metadata['selection_info'] = {
                'strategy': metadata.get('method', str(data.get('strategy', 'unknown'))),
                'K_requested': int(metadata.get('K', data.get('K_requested', K_actual))),
                'K_actual': K_actual,
                'selection_timestamp': str(data.get('timestamp', 'unknown'))
            }
        
        # 提取 sensor_indices (如果有)
        if 'sensor_indices' in data:
            metadata['sensor_indices'] = np.asarray(data['sensor_indices'])
        
        # 提取其他可能的元數據
        if 'metadata' in data:
            extra_meta = data['metadata']
            metadata['extra'] = extra_meta.item() if extra_meta.ndim == 0 else extra_meta
        
        return metadata


@dataclass
class ChannelFlowData:
    """Channel Flow data bundle built on unified data structures."""

    samples: PointSamples
    domain: DomainSpec
    selection_info: Dict[str, Any]
    coordinate_info: Dict[str, Any]
    statistics: Optional[Dict[str, Dict[str, float]]] = None
    lowfi_prior: Optional[PointSamples] = None
    lowfi_metadata: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def sensor_points(self) -> np.ndarray:
        return self.samples.coordinates

    @property
    def sensor_data(self) -> Dict[str, np.ndarray]:
        return self.samples.values

    @property
    def sensor_axes(self) -> Tuple[str, ...]:
        return self.samples.axes

    @property
    def domain_config(self) -> Dict[str, Any]:
        return self.domain.to_config()

    def get_domain_bounds(self) -> Dict[str, Tuple[float, float]]:
        return dict(self.domain.bounds)

    def get_physical_parameters(self) -> Dict[str, float]:
        return dict(self.domain.parameters)

    def has_lowfi_prior(self) -> bool:
        return self.lowfi_prior is not None

    def to_flow_bundle(self) -> FlowDataBundle:
        meta = {
            'selection_info': self.selection_info,
            'coordinate_info': self.coordinate_info,
        }
        if self.metadata:
            meta.update(self.metadata)
        return FlowDataBundle(
            samples=self.samples,
            domain=self.domain,
            statistics=self.statistics or {},
            lowfi_prior=self.lowfi_prior,
            metadata=meta
        )


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
                        dropout_prob: Optional[float] = None,
                        sensor_file: Optional[str] = None) -> ChannelFlowData:
        """
        載入感測點資料
        
        Args:
            strategy: 選擇策略 ('qr_pivot', 'random', 'uniform')
            K: 感測點數量
            noise_sigma: 噪聲水平 (可選)
            dropout_prob: 丟失概率 (可選)
            sensor_file: 自定義感測點文件名 (可選，優先於自動構建)
            
        Returns:
            Channel Flow 資料容器
        """
        # 構建快取檔案名（允許自定義覆蓋）
        if sensor_file is not None:
            # 如果提供絕對路徑，直接使用；否則相對於 cache_dir
            cache_path = Path(sensor_file)
            if not cache_path.is_absolute():
                cache_path = self.cache_dir / sensor_file
        else:
            # 否則使用 cache_dir + 自動生成的檔名
            cache_filename = f"sensors_K{K}_{strategy}.npz"
            cache_path = self.cache_dir / cache_filename
        
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Sensor data not found: {cache_path}\n"
                f"Please run scripts/fetch_channel_flow.py first"
            )
        
        logger.info(f"Loading sensor data from {cache_path}")
        
        # 使用 SensorDataReader 讀取資料
        z_default = self.config.get('normalization', {}).get('slice_config', {}).get('z_position', 4.71)
        reader = SensorDataReader(z_default=z_default)
        lowfi_data = reader.read(cache_path)
        
        # 從 LowFiData 提取感測點和值
        sensor_points = np.column_stack([
            lowfi_data.coordinates['x'],
            lowfi_data.coordinates['y'],
            lowfi_data.coordinates['z']
        ])
        sensor_values = lowfi_data.fields
        
        # 提取 sensor_indices 和 selection_info
        sensor_indices = lowfi_data.metadata.get('sensor_indices', np.arange(len(sensor_points)))
        selection_info = lowfi_data.metadata.get('selection_info', {
            'strategy': strategy,
            'K_requested': K,
            'K_actual': len(sensor_points),
            'selection_timestamp': 'unknown'
        })
        
        # 添加噪聲 (如果指定)
        if noise_sigma is not None and noise_sigma > 0:
            sensor_values = self._add_noise(sensor_values, noise_sigma)
            selection_info['noise_sigma'] = noise_sigma
        
        # 添加丟失 (如果指定)
        if dropout_prob is not None and dropout_prob > 0:
            sensor_values, valid_mask = self._add_dropout(sensor_values, dropout_prob)
            sensor_points = sensor_points[valid_mask]
            sensor_indices = sensor_indices[valid_mask]
            selection_info['dropout_prob'] = dropout_prob
            selection_info['K_after_dropout'] = int(len(sensor_points))
        
        # 提取域配置
        domain_config = self._extract_domain_config()
        # 從 lowfi_data.metadata 提取座標資訊（如果有的話）
        coordinate_info = lowfi_data.metadata.get('coordinate_info', {})
        domain_spec = self._build_domain_spec(domain_config)
        
        # 計算統計資訊（用於 VS-PINN 與自動輸出範圍）
        statistics = self._compute_statistics(sensor_values, sensor_points)
        
        samples = PointSamples(
            coordinates=sensor_points,
            values=sensor_values,
            axes=('x', 'y', 'z') if sensor_points.shape[1] == 3 else ('x', 'y'),
            metadata={'sensor_indices': sensor_indices}
        )
        
        channel_data = ChannelFlowData(
            samples=samples,
            domain=domain_spec,
            selection_info=selection_info,
            coordinate_info=coordinate_info,
            statistics=statistics,
            metadata={
                'source': str(cache_path),
                'config_file': str(self.config_path),
                'loader_version': '2.0',
                'loaded_timestamp': str(np.datetime64('now')),
                'strategy': strategy,
                'requested_K': int(K),
                'actual_K': int(len(sensor_points))
            }
        )
        
        logger.info(f"Loaded {len(sensor_points)} sensor points using {strategy} strategy")
        logger.info(f"Computed statistics for fields: {list(statistics.keys())}")
        return channel_data
    
    
    def add_lowfi_prior(self, 
                       channel_data: ChannelFlowData,
                       prior_type: str = 'rans',
                       interpolate_to_sensors: bool = True) -> ChannelFlowData:
        """
        添加低保真先驗資料
        
        Args:
            channel_data: 現有的 Channel Flow 資料
            prior_type: 先驗類型 ('rans', 'mock', 'none')
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
            elif prior_type == 'mock':
                # 使用簡化的 mock 先驗 (基於層流解或統計量)
                lowfi_data = self._create_mock_prior(channel_data)
            else:
                raise ValueError(f"Unknown prior type: {prior_type}. Supported: 'rans', 'mock', 'none'.")
            
            # 插值到感測點 (如果需要)
            # 對於 mock prior，已經在感測點計算，不需要插值
            if interpolate_to_sensors and prior_type != 'mock':
                prior_fields = self.interpolator.interpolate_to_points(
                    lowfi_data,
                    channel_data.sensor_points,
                    quality_check=False
                )
            else:
                prior_fields = lowfi_data.fields

            # 過濾非場資料（如品質指標）
            prior_fields = {
                key: value for key, value in prior_fields.items()
                if not key.startswith('_')
            }

            prior_samples = PointSamples(
                coordinates=channel_data.sensor_points,
                values={k: np.asarray(v).reshape(-1) for k, v in prior_fields.items()},
                axes=channel_data.sensor_axes,
                metadata={'prior_type': prior_type}
            )

            channel_data.lowfi_prior = prior_samples
            channel_data.lowfi_metadata = lowfi_data.metadata
            
            logger.info(f"Added {prior_type} low-fidelity prior with {len(prior_fields)} fields")
            
        except Exception as e:
            logger.warning(f"Failed to load low-fidelity prior: {e}")
            # 保持 lowfi_prior 為 None（不設為空字典）
            channel_data.lowfi_prior = None
            channel_data.lowfi_metadata = {'type': 'none', 'error': str(e)}
            # 不覆蓋 statistics - 保留原有的統計資訊
        
        return channel_data
    
    def prepare_for_training(self, 
                           channel_data: ChannelFlowData,
                           target_fields: Optional[List[str]] = None) -> FlowDataBundle:
        """
        準備 PINNs 訓練資料格式
        
        Args:
            channel_data: Channel Flow 資料
            target_fields: 目標場列表，預設 ['u', 'v', 'w', 'p']（3D）或 ['u', 'v', 'p']（2D）
            
        Returns:
            FlowDataBundle 封裝的訓練資料
        """
        if target_fields is None:
            # 🆕 自動檢測可用欄位（優先使用完整 4 變量）
            available_fields = list(channel_data.sensor_data.keys())
            if 'w' in available_fields:
                target_fields = ['u', 'v', 'w', 'p']  # 3D 或含 w 的 2D 切片
            else:
                target_fields = ['u', 'v', 'p']
            logger.info(f"Auto-detected target_fields: {target_fields}")
        bundle = channel_data.to_flow_bundle()
        bundle.metadata['has_lowfi_prior'] = channel_data.has_lowfi_prior()
        bundle.metadata['target_fields'] = list(target_fields)
        logger.info(f"Prepared training bundle with fields: {target_fields}")
        return bundle
    
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
        """從配置檔案提取域參數
        
        支持兩種格式（P0-2 統一後優先使用 physics.domain）：
        1. 新格式：physics.domain.x_range = [min, max]
        2. 舊格式：data.jhtdb_config.domain.x = [min, max]
        """
        domain_config = {}
        
        # P0-2: 優先從 physics.domain 提取（新格式）
        if 'physics' in self.config and 'domain' in self.config['physics']:
            phys_domain = self.config['physics']['domain']
            
            # 新格式：x_range, y_range, z_range
            if 'x_range' in phys_domain:
                domain_config.update({
                    'x_range': phys_domain['x_range'],
                    'y_range': phys_domain['y_range'],
                    'z_range': phys_domain.get('z_range', [0.0, 0.0])
                })
            # 舊格式兼容：x, y, z
            elif 'x' in phys_domain:
                domain_config.update({
                    'x_range': phys_domain['x'],
                    'y_range': phys_domain['y'],
                    'z_range': phys_domain.get('z', [0.0, 0.0])
                })
        
        # Fallback: 從 data.jhtdb_config 提取 Channel Flow 參數（舊格式）
        elif 'data' in self.config and 'jhtdb_config' in self.config['data']:
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

    def _build_domain_spec(self, domain_config: Dict[str, Any]) -> DomainSpec:
        bounds: Dict[str, Tuple[float, float]] = {}
        for axis in ('x', 'y', 'z', 't'):
            key = f"{axis}_range"
            if key in domain_config:
                rng = domain_config[key]
                bounds[axis] = (float(rng[0]), float(rng[1]))

        resolution: Dict[str, int] = {}
        for axis_key, axis_name in (('nx', 'x'), ('ny', 'y'), ('nz', 'z')):
            if axis_key in domain_config:
                resolution[axis_name] = int(domain_config[axis_key])

        parameters = {
            key: float(domain_config[key])
            for key in ('Re_tau', 'nu', 'u_tau', 'rho', 'pressure_gradient')
            if key in domain_config
        }

        time_range = None
        if 'time_range' in domain_config:
            rng = domain_config['time_range']
            time_range = (float(rng[0]), float(rng[1]))

        return DomainSpec(
            bounds=bounds,
            parameters=parameters,
            resolution=resolution,
            time_range=time_range
        )

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
    
    def _compute_statistics(self, 
                          sensor_data: Dict[str, np.ndarray],
                          sensor_points: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        計算場資料的統計資訊（用於 VS-PINN 與自動輸出範圍）
        
        Args:
            sensor_data: 感測器場資料字典 {'u': array, 'v': array, 'p': array}
            sensor_points: 感測點座標 (K, 2)
            
        Returns:
            統計資訊字典，格式：
            {
                'u': {'min': float, 'max': float, 'mean': float, 'std': float, 'range': (min, max)},
                'v': {...},
                'p': {...},
                'x': {'min': float, 'max': float, 'range': (min, max)},
                'y': {...}
            }
        """
        statistics = {}
        
        # 計算場變量的統計資訊
        for field_name, field_values in sensor_data.items():
            field_values = np.asarray(field_values).flatten()
            
            # 基本統計量
            field_min = float(np.min(field_values))
            field_max = float(np.max(field_values))
            field_mean = float(np.mean(field_values))
            field_std = float(np.std(field_values))
            
            # 添加安全邊界（±10% 範圍，避免邊界值被截斷）
            margin = 0.1 * (field_max - field_min)
            safe_min = field_min - margin
            safe_max = field_max + margin
            
            statistics[field_name] = {
                'min': field_min,
                'max': field_max,
                'mean': field_mean,
                'std': field_std,
                'range': (safe_min, safe_max),  # 帶安全邊界的範圍
                'raw_range': (field_min, field_max)  # 原始範圍
            }
        
        # 計算座標的統計資訊（自動檢測維度）
        if sensor_points.size > 0:
            coord_names = ['x', 'y', 'z'][:sensor_points.shape[1]]  # 根據實際維度
            for i, coord_name in enumerate(coord_names):
                coord_values = sensor_points[:, i]
                coord_min = float(np.min(coord_values))
                coord_max = float(np.max(coord_values))
                
                statistics[coord_name] = {
                    'min': coord_min,
                    'max': coord_max,
                    'range': (coord_min, coord_max)
                }
        
        logger.debug(f"Computed statistics: {statistics}")
        return statistics


    def _load_rans_prior(self) -> LowFiData:
        """載入真實 RANS 先驗資料"""
        # ✅ 優先從 config 讀取 lowfi_prior.data_path
        lowfi_cfg = self.config.get('lowfi_prior', {})
        if lowfi_cfg.get('enabled', False) and 'data_path' in lowfi_cfg:
            rans_path = Path(lowfi_cfg['data_path'])
            # 支援相對路徑（相對於專案根目錄）
            if not rans_path.is_absolute():
                # 假設 config_path 在 configs/ 下，專案根目錄在上一層
                project_root = self.config_path.parent.parent if self.config_path.parent.name == 'configs' else self.config_path.parent
                rans_path = project_root / rans_path
            
            if rans_path.exists():
                logger.info(f"Loading RANS prior from config: {rans_path}")
                return self.lowfi_loader.load(rans_path, data_type='rans')
            else:
                logger.warning(f"RANS path from config not found: {rans_path}")
        
        # ✅ 回退：在 cache_dir 尋找標準檔名
        rans_patterns = ['rans_data.npz', 'lowfi_prior.npz', 'rans_baseline.nc']
        
        for pattern in rans_patterns:
            rans_path = self.cache_dir / pattern
            if rans_path.exists():
                logger.info(f"Loading RANS prior from cache: {rans_path}")
                return self.lowfi_loader.load(rans_path, data_type='rans')
        
        # 無法找到RANS資料，直接拋出錯誤
        raise FileNotFoundError(
            f"No RANS prior data found. "
            f"Checked config path: {lowfi_cfg.get('data_path', 'N/A')}. "
            f"Searched in {self.cache_dir} for: {rans_patterns}. "
            f"Mock fallback has been removed for this system."
        )
    
    def _create_mock_prior(self, channel_data: ChannelFlowData) -> LowFiData:
        """
        創建簡化的 mock 先驗資料 (基於層流解或統計量估計)
        用於測試或缺少真實 RANS 時的暫時替代方案
        
        Args:
            channel_data: 現有的 Channel Flow 資料 (用於提取幾何資訊)
            
        Returns:
            Mock 低保真資料容器
        """
        import numpy as np
        from pinnx.dataio.lowfi_loader import LowFiData
        
        logger.info("Creating mock low-fidelity prior based on laminar solution")
        
        # 提取幾何資訊
        y_range = channel_data.domain_config.get('y_range', [-1.0, 1.0])
        
        # 從感測點座標創建場
        coords = channel_data.sensor_points  # (K, 2)
        x_coords = coords[:, 0]
        y_coords = coords[:, 1]
        
        # 層流通道流解析解: u = U_max * (1 - (y/h)^2), v = 0, p 線性分佈
        h = (y_range[1] - y_range[0]) / 2.0  # 半高度
        y_center = (y_range[1] + y_range[0]) / 2.0
        y_norm = (y_coords - y_center) / h  # 標準化到 [-1, 1]
        
        # 基於 Poiseuille 流的速度分佈
        u_max = 1.5  # 平均速度的1.5倍（層流拋物線型最大值）
        u_laminar = u_max * (1.0 - y_norm**2)
        v_laminar = np.zeros_like(u_laminar)
        
        # 簡化的壓力場（線性下降）
        p_gradient = -1.0  # 從配置讀取
        p_laminar = p_gradient * x_coords
        
        # 創建 LowFiData 容器 (coordinates 需要字典格式)
        mock_coords = {
            'x': x_coords,
            'y': y_coords
        }
        
        mock_fields = {
            'u': u_laminar,
            'v': v_laminar,
            'p': p_laminar
        }
        
        mock_metadata = {
            'type': 'mock_laminar',
            'description': 'Analytical laminar channel flow solution',
            'u_max': u_max,
            'pressure_gradient': p_gradient
        }
        
        return LowFiData(
            coordinates=mock_coords,
            fields=mock_fields,
            metadata=mock_metadata
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
        
        # 物理合理性（內聯驗證邏輯，避免實例化抽象類）
        for field, values in channel_data.sensor_data.items():
            # 設定 Channel Flow Re1000 的合理範圍
            max_reasonable = 30.0 if field == 'u' else 5.0
            
            # NaN/Inf 檢查
            checks[f'{field}_finite'] = np.all(np.isfinite(values))
            # 範圍檢查
            checks[f'{field}_reasonable'] = np.abs(values).max() < max_reasonable
        
        # 域參數合理性
        domain = channel_data.domain_config
        if 'Re_tau' in domain:
            checks['Re_tau_reasonable'] = 100 <= domain['Re_tau'] <= 10000
        if 'nu' in domain:
            checks['nu_positive'] = domain['nu'] > 0
        
        # 低保真先驗檢查
        if channel_data.has_lowfi_prior() and channel_data.lowfi_prior:
            checks['lowfi_prior_available'] = True
            for field in ['u', 'v', 'p']:
                if field in channel_data.lowfi_prior.values:
                    values = channel_data.lowfi_prior.values[field]
                    max_val = 30.0 if field == 'u' else 5.0
                    # NaN/Inf 檢查
                    checks[f'lowfi_{field}_finite'] = np.all(np.isfinite(values))
                    # 範圍檢查
                    checks[f'lowfi_{field}_reasonable'] = np.abs(values).max() < max_val
        
        # 統計資訊檢查
        if channel_data.statistics:
            checks['statistics_available'] = True
        
        return checks
    
    def load_full_field_data(self,
                           noise_sigma: Optional[float] = None) -> StructuredField:
        """載入完整流場數據並返回統一的 StructuredField 物件（使用 NPZReader）"""
        cutout_file = self.cache_dir / "cutout_128x64_with_w.npz"
        if not cutout_file.exists():
            raise FileNotFoundError(
                f"Expected high-fidelity cutout at {cutout_file}. "
                "Regenerate 2D cutout data with scripts/fetch_channel_flow.py."
            )

        logger.info(f"Loading full field data from {cutout_file}")
        
        # 使用 NPZReader 讀取數據
        reader = NPZReader()
        lowfi_data = reader.read(cutout_file)
        
        # 提取 fields（確保包含 u, v, w, p）
        fields = {
            'u': lowfi_data.fields['u'],
            'v': lowfi_data.fields['v'],
            'w': lowfi_data.fields['w'],
            'p': lowfi_data.fields['p']
        }
        
        # 添加噪聲（如果指定）
        if noise_sigma is not None and noise_sigma > 0:
            noisy = self._add_noise({k: v.reshape(-1) for k, v in fields.items()}, noise_sigma)
            for key, arr in noisy.items():
                fields[key] = arr.reshape(fields[key].shape)
        
        # 提取坐標軸（從 LowFiData.coordinates）
        if 'x' not in lowfi_data.coordinates or 'y' not in lowfi_data.coordinates:
            raise KeyError(f"Missing x/y coordinates in {cutout_file}")
        
        x_axis = lowfi_data.coordinates['x']
        y_axis = lowfi_data.coordinates['y']
        
        # 創建結構化網格
        grid = StructuredGrid.from_axes({'x': x_axis, 'y': y_axis})
        
        # 計算統計資訊
        stats_input = {k: v.reshape(-1) for k, v in fields.items()}
        statistics = self._compute_statistics(stats_input, grid.to_points(order=('x', 'y')))
        
        return StructuredField(
            grid=grid,
            fields=fields,
            metadata={
                'source': str(cutout_file),
                'config_file': str(self.config_path),
                'loader_version': '2.0',
                'noise_sigma': noise_sigma,
                'statistics': statistics
            }
        )


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
                         target_fields: Optional[List[str]] = None,
                         sensor_file: Optional[str] = None,
                         prior_type: str = 'none') -> FlowDataBundle:
    """
    便利函數：準備 PINNs 訓練資料
    
    Args:
        strategy: 感測點選擇策略
        K: 感測點數量
        config_path: 配置檔案路徑  
        target_fields: 目標場列表
        sensor_file: 自定義感測點文件名 (可選)
        prior_type: 低保真先驗類型 ('rans', 'mock', 'none')，預設 'none'
        
    Returns:
        FlowDataBundle: 準備好的訓練資料容器
        
    Note:
        ⚠️ prior_type='none' 意味著僅使用感測點數據，不添加低保真先驗
        這是推薦的預設值，避免覆蓋真實 JHTDB 數據
    """
    loader = ChannelFlowLoader(config_path=config_path)
    
    # 載入完整資料
    channel_data = loader.load_sensor_data(strategy=strategy, K=K, sensor_file=sensor_file)
    
    # 僅在明確要求時才添加低保真先驗
    if prior_type != 'none':
        channel_data = loader.add_lowfi_prior(channel_data, prior_type=prior_type)
    
    # 準備訓練格式
    training_data = loader.prepare_for_training(channel_data, target_fields=target_fields)
    
    return training_data
