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
from .structures import (
    StructuredGrid,
    StructuredField,
    PointSamples,
    FlowDataBundle,
    DomainSpec,
)


logger = logging.getLogger(__name__)


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
        
        # 載入 NPZ 資料
        data = np.load(cache_path, allow_pickle=True)
        
        # 提取感測點資訊（支援多種鍵名）
        if 'sensor_points' in data:
            sensor_points = data['sensor_points']
        elif 'coords' in data:
            sensor_points = data['coords']  # ⭐ 新格式：使用 'coords' 鍵
        elif 'coords_2d' in data:
            # 2D 座標需要擴展到 3D (x, y) → (x, y, z_slice)
            coords_2d = data['coords_2d']
            z_slice = self.config.get('normalization', {}).get('slice_config', {}).get('z_position', 4.71)
            sensor_points = np.column_stack([
                coords_2d[:, 0],  # x
                coords_2d[:, 1],  # y
                np.full(len(coords_2d), z_slice)  # z (constant)
            ])
        else:
            raise KeyError(f"Cannot find sensor coordinates in {cache_path}. Expected 'sensor_points', 'coords', or 'coords_2d'")
        
        # 處理 sensor_data (可能是物件、分離的陣列、或直接的 2D ndarray)
        if 'sensor_data' in data:
            sensor_data_raw = data['sensor_data']
            
            # 情況 1: 0 維物件（包含字典）
            if sensor_data_raw.ndim == 0:
                sensor_data_raw = sensor_data_raw.item()
            
            # 情況 2: 2D ndarray (K, n_vars) - 直接儲存速度分量
            elif sensor_data_raw.ndim == 2:
                # 從 metadata 或預設變數名稱提取欄位
                if 'metadata' in data:
                    metadata = data['metadata'].item() if data['metadata'].ndim == 0 else data['metadata']
                    variables = metadata.get('variables', ['u', 'v', 'w'])
                else:
                    # 根據欄位數判斷
                    n_vars = sensor_data_raw.shape[1]
                    if n_vars == 2:
                        variables = ['u', 'v']
                    elif n_vars == 3:
                        variables = ['u', 'v', 'w']
                    elif n_vars == 4:
                        variables = ['u', 'v', 'w', 'p']
                    else:
                        raise ValueError(f"Cannot infer variable names for {n_vars} columns")
                
                # 將 ndarray 轉換為字典格式
                sensor_data_raw = {
                    var: sensor_data_raw[:, i]
                    for i, var in enumerate(variables)
                }
        else:
            # 情況 3: 分離的欄位（sensor_u, sensor_v 等 或直接 u, v, w, p）
            sensor_data_raw = {}
            
            # 優先檢查 'sensor_*' 格式
            for field in ['u', 'v', 'w', 'p']:
                key_sensor = f'sensor_{field}'
                if key_sensor in data:
                    sensor_data_raw[field] = data[key_sensor]
            
            # ⭐ 若無 'sensor_*'，嘗試直接鍵名（新格式）
            if not sensor_data_raw:
                for field in ['u', 'v', 'w', 'p']:
                    if field in data:
                        sensor_data_raw[field] = data[field]
            
            if not sensor_data_raw:
                raise KeyError(f"Cannot find velocity/pressure data in {cache_path}. Expected 'sensor_data', 'sensor_u/v/w/p', or 'u/v/w/p'")
        
        sensor_values = {
            field: np.asarray(values).reshape(-1)
            for field, values in sensor_data_raw.items()
        }
        
        # 提取 sensor_indices (可能不存在，生成預設索引)
        if 'sensor_indices' in data:
            sensor_indices = np.asarray(data['sensor_indices'])
        else:
            # 如果沒有 indices，生成連續索引
            sensor_indices = np.arange(len(sensor_points))
        
        # 提取選擇資訊
        if 'selection_info' in data:
            selection_info = data['selection_info'].item() if data['selection_info'].ndim == 0 else data['selection_info']
        else:
            selection_info = {
                'strategy': str(data.get('strategy', strategy)),
                'K_requested': int(data.get('K_requested', K)),
                'K_actual': int(len(sensor_points)),
                'selection_timestamp': str(data.get('timestamp', 'unknown'))
            }
        
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
        coordinate_info = self._extract_coordinate_info(data)
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
                    channel_data.sensor_points
                )
            else:
                prior_fields = lowfi_data.fields

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
                target_fields = ['u', 'v', 'p']  # 舊版 2D（向後兼容）
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
                if field in channel_data.lowfi_prior.values:
                    values = channel_data.lowfi_prior.values[field]
                    checks[f'lowfi_{field}_finite'] = np.all(np.isfinite(values))
        
        # 統計資訊檢查
        if channel_data.statistics:
            checks['statistics_available'] = True
        
        return checks
    
    def load_full_field_data(self,
                           noise_sigma: Optional[float] = None) -> StructuredField:
        """
        載入完整流場數據並返回統一的 StructuredField 物件。
        """
        cutout_file = self.cache_dir / "cutout_128x64_with_w.npz"
        if not cutout_file.exists():
            raise FileNotFoundError(
                f"Expected high-fidelity cutout at {cutout_file}. "
                "Regenerate 2D cutout data with scripts/fetch_channel_flow.py."
            )

        logger.info(f"Loading full field data from {cutout_file}")
        data = np.load(cutout_file, allow_pickle=True)

        fields = {
            'u': np.asarray(data['u']),
            'v': np.asarray(data['v']),
            'w': np.asarray(data['w']),
            'p': np.asarray(data['p'])
        }

        if noise_sigma is not None and noise_sigma > 0:
            noisy = self._add_noise({k: v.reshape(-1) for k, v in fields.items()}, noise_sigma)
            for key, arr in noisy.items():
                fields[key] = arr.reshape(fields[key].shape)

        if 'coordinates' not in data:
            raise KeyError("cutout_128x64_with_w.npz must include structured 'coordinates'")
        coordinates_obj = data['coordinates'].item()
        if not isinstance(coordinates_obj, dict):
            raise TypeError("coordinates metadata must be provided as a dictionary")
        if 'x' in coordinates_obj and 'y' in coordinates_obj:
            x_axis = np.asarray(coordinates_obj['x'])
            y_axis = np.asarray(coordinates_obj['y'])
        elif 'X' in coordinates_obj and 'Y' in coordinates_obj:
            X = np.asarray(coordinates_obj['X'])
            Y = np.asarray(coordinates_obj['Y'])
            x_axis = X[:, 0]
            y_axis = Y[0, :]
        else:
            raise KeyError(f"Unsupported coordinate keys: {list(coordinates_obj.keys())}")

        grid = StructuredGrid.from_axes({'x': x_axis, 'y': y_axis})
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
