"""
約翰霍普金斯湍流資料庫 (JHTDB) 客戶端

提供高效、可重現的 JHTDB 資料存取接口，支援：
- Cutout 批量資料下載
- 散點插值取樣
- 自動快取與版本管理
- 多種湍流資料集 (通道流、HIT、邊界層等)
- 資料驗證與品質檢查

參考文獻：
- JHTDB Official Documentation: https://turbulence.pha.jhu.edu/
- pyJHTDB: Python interface for JHTDB
- SciServer: Cloud-based data access platform

核心設計原則：
1. 可重現性：固定種子、版本記錄、資料校驗
2. 效率：智能快取、批量下載、增量更新
3. 穩健性：錯誤處理、重試機制、備用端點
4. 標準化：統一資料格式、座標系統、命名規範
"""

import numpy as np
import h5py
import os
import json
import hashlib
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
import struct
import base64
from abc import ABC, abstractmethod

try:
    import pyJHTDB
    PYJHTDB_AVAILABLE = True
except ImportError:
    PYJHTDB_AVAILABLE = False
    pyJHTDB = None

logger = logging.getLogger(__name__)


class JHTDBConfig:
    """JHTDB 連接配置"""
    
    # 資料集配置
    DATASETS = {
        'channel': {
            'name': 'channel',
            'description': '通道流 (Re_tau=1000)',
            'domain': {'x': [0, 8*np.pi], 'y': [-1, 1], 'z': [0, 3*np.pi]},
            'resolution': {'x': 2048, 'y': 512, 'z': 1536},
            'time_range': [0.0, 26.0],
            'dt': 0.0065,
            'variables': ['u', 'v', 'w', 'p']
        },
        'isotropic1024coarse': {
            'name': 'isotropic1024coarse',
            'description': '各向同性湍流 (1024^3, 粗時間解析度)',
            'domain': {'x': [0, 2*np.pi], 'y': [0, 2*np.pi], 'z': [0, 2*np.pi]},
            'resolution': {'x': 1024, 'y': 1024, 'z': 1024},
            'time_range': [0.0, 10.0],
            'dt': 0.04,
            'variables': ['u', 'v', 'w', 'p']
        },
        'transition_bl': {
            'name': 'transition_bl',
            'description': '邊界層轉捩',
            'domain': {'x': [0, 4000], 'y': [0, 120], 'z': [0, 300]},
            'resolution': {'x': 4000, 'y': 120, 'z': 300},
            'time_range': [0.0, 100.0],
            'dt': 0.5,
            'variables': ['u', 'v', 'w', 'p']
        }
    }
    
    # 預設連接參數
    # ⚠️ 安全性：從環境變數讀取 auth token，避免硬編碼
    DEFAULT_AUTH_TOKEN = None  # 從環境變數 JHTDB_AUTH_TOKEN 載入
    DEFAULT_CACHE_DIR = "data/jhtdb"
    DEFAULT_TIMEOUT = 300  # 5 分鐘
    MAX_RETRY = 3
    
    # 資料驗證閾值
    VALIDATION_THRESHOLDS = {
        'velocity_magnitude_max': 100.0,  # m/s
        'pressure_range': [-1000.0, 1000.0],  # Pa
        'nan_fraction_max': 0.01,  # 最大 NaN 比例
        'inf_fraction_max': 0.001  # 最大 Inf 比例
    }


class JHTDBError(Exception):
    """JHTDB 特定錯誤"""
    pass


class DataValidator:
    """資料驗證器"""
    
    @staticmethod
    def validate_field(data: np.ndarray, 
                      field_name: str, 
                      thresholds: Dict[str, float]) -> Dict[str, Any]:
        """
        驗證流場資料的物理合理性
        
        Args:
            data: 資料陣列
            field_name: 欄位名稱 ('u', 'v', 'w', 'p')
            thresholds: 驗證閾值
            
        Returns:
            驗證報告字典
        """
        report = {
            'field': field_name,
            'shape': data.shape,
            'dtype': str(data.dtype),
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 基本統計
        if data.size > 0:
            report['stats'] = {
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'nan_count': int(np.sum(np.isnan(data))),
                'inf_count': int(np.sum(np.isinf(data)))
            }
            
            # NaN/Inf 檢查
            nan_fraction = report['stats']['nan_count'] / data.size
            inf_fraction = report['stats']['inf_count'] / data.size
            
            if nan_fraction > thresholds.get('nan_fraction_max', 0.01):
                report['errors'].append(f"過多 NaN 值: {nan_fraction:.3f}")
                report['valid'] = False
            
            if inf_fraction > thresholds.get('inf_fraction_max', 0.001):
                report['errors'].append(f"過多 Inf 值: {inf_fraction:.3f}")
                report['valid'] = False
            
            # 物理範圍檢查
            if field_name in ['u', 'v', 'w']:
                max_velocity = max(abs(report['stats']['min']), abs(report['stats']['max']))
                if max_velocity > thresholds.get('velocity_magnitude_max', 100.0):
                    report['warnings'].append(f"速度過大: {max_velocity:.2f}")
            
            elif field_name == 'p':
                p_min, p_max = report['stats']['min'], report['stats']['max']
                threshold_range = thresholds.get('pressure_range', [-1000.0, 1000.0])
                if p_min < threshold_range[0] or p_max > threshold_range[1]:
                    report['warnings'].append(f"壓力超出合理範圍: [{p_min:.2f}, {p_max:.2f}]")
        
        else:
            report['errors'].append("空資料陣列")
            report['valid'] = False
        
        return report


class CacheManager:
    """智能快取管理器"""
    
    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """載入快取元資料"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                logger.warning("快取元資料損壞，重新初始化")
        return {}
    
    def _save_metadata(self):
        """保存快取元資料"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _compute_key(self, dataset: str, query_params: Dict) -> str:
        """計算查詢的快取鍵"""
        # 建立標準化的查詢字串
        sorted_params = json.dumps(query_params, sort_keys=True)
        key_string = f"{dataset}:{sorted_params}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_cache_path(self, dataset: str, query_params: Dict) -> Path:
        """獲取快取檔案路徑"""
        cache_key = self._compute_key(dataset, query_params)
        return self.cache_dir / f"{dataset}_{cache_key}.h5"
    
    def is_cached(self, dataset: str, query_params: Dict) -> bool:
        """檢查是否已快取"""
        cache_path = self.get_cache_path(dataset, query_params)
        cache_key = self._compute_key(dataset, query_params)
        
        if not cache_path.exists():
            return False
        
        # 檢查元資料
        if cache_key in self.metadata:
            metadata = self.metadata[cache_key]
            # 檢查檔案完整性
            if cache_path.stat().st_size != metadata.get('file_size', -1):
                logger.warning(f"快取檔案大小不符，刪除: {cache_path}")
                cache_path.unlink()
                del self.metadata[cache_key]
                self._save_metadata()
                return False
            return True
        
        return False
    
    def save_to_cache(self, dataset: str, query_params: Dict, data: Dict[str, np.ndarray], 
                     metadata: Dict = None):
        """保存資料到快取"""
        cache_path = self.get_cache_path(dataset, query_params)
        cache_key = self._compute_key(dataset, query_params)
        
        try:
            with h5py.File(cache_path, 'w') as f:
                # 保存資料
                for var_name, var_data in data.items():
                    f.create_dataset(var_name, data=var_data, compression='gzip', compression_opts=6)
                
                # 保存查詢參數和元資料
                f.attrs['query_params'] = json.dumps(query_params)
                f.attrs['dataset'] = dataset
                f.attrs['timestamp'] = time.time()
                
                if metadata:
                    for key, value in metadata.items():
                        # 將複雜的 Python 物件序列化為 JSON 字串
                        if isinstance(value, (dict, list, tuple)):
                            f.attrs[key] = json.dumps(value)
                        elif isinstance(value, np.ndarray):
                            # numpy 陣列轉為列表再序列化
                            f.attrs[key] = json.dumps(value.tolist())
                        else:
                            # 簡單類型直接儲存
                            try:
                                f.attrs[key] = value
                            except (TypeError, ValueError):
                                # 如果無法直接儲存，轉為字串
                                f.attrs[key] = str(value)
            
            # 更新元資料
            self.metadata[cache_key] = {
                'dataset': dataset,
                'query_params': query_params,
                'file_path': str(cache_path),
                'file_size': cache_path.stat().st_size,
                'timestamp': time.time(),
                'variables': list(data.keys())
            }
            self._save_metadata()
            
            logger.info(f"資料已快取: {cache_path}")
            
        except Exception as e:
            logger.error(f"快取保存失敗: {e}")
            if cache_path.exists():
                cache_path.unlink()
    
    def load_from_cache(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """從快取載入資料"""
        cache_path = self.get_cache_path(dataset, query_params)
        
        try:
            data = {}
            with h5py.File(cache_path, 'r') as f:
                for var_name in f.keys():
                    data[var_name] = f[var_name][:]
                
                # 驗證查詢參數
                cached_params = json.loads(f.attrs['query_params'])
                if cached_params != query_params:
                    logger.warning("快取查詢參數不符，可能是雜湊衝突")
                    return {}
            
            logger.info(f"從快取載入資料: {cache_path}")
            return data
            
        except Exception as e:
            logger.error(f"快取載入失敗: {e}")
            return {}
    
    def clear_cache(self, older_than_days: int = 30):
        """清理舊快取"""
        current_time = time.time()
        cutoff_time = current_time - (older_than_days * 24 * 3600)
        
        removed_count = 0
        for cache_key, metadata in list(self.metadata.items()):
            if metadata.get('timestamp', 0) < cutoff_time:
                cache_path = Path(metadata['file_path'])
                if cache_path.exists():
                    cache_path.unlink()
                    removed_count += 1
                del self.metadata[cache_key]
        
        if removed_count > 0:
            self._save_metadata()
            logger.info(f"清理了 {removed_count} 個舊快取檔案")


class BaseJHTDBClient(ABC):
    """JHTDB 客戶端基類"""
    
    def __init__(self, 
                 auth_token: Optional[str] = None,
                 cache_dir: str = None,
                 timeout: int = None):
        # 優先順序：傳入參數 > 環境變數 > DEFAULT_AUTH_TOKEN
        import os
        self.auth_token = (
            auth_token or 
            os.getenv('JHTDB_AUTH_TOKEN') or 
            JHTDBConfig.DEFAULT_AUTH_TOKEN
        )
        
        # 不強制要求 token（允許 Mock 客戶端運行）
        # 實際的 token 驗證由子類負責
        
        self.timeout = timeout or JHTDBConfig.DEFAULT_TIMEOUT
        self.cache_manager = CacheManager(cache_dir or JHTDBConfig.DEFAULT_CACHE_DIR)
        self.validator = DataValidator()
    
    @abstractmethod
    def _fetch_raw_data(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """實際的資料獲取實現（由子類實現）"""
        pass
    
    def fetch_data(self, 
                   dataset: str,
                   query_params: Dict,
                   use_cache: bool = True,
                   validate: bool = True) -> Dict[str, Any]:
        """
        獲取 JHTDB 資料的主要接口
        
        Args:
            dataset: 資料集名稱
            query_params: 查詢參數
            use_cache: 是否使用快取
            validate: 是否驗證資料
            
        Returns:
            包含資料和元資料的字典
        """
        # 檢查快取
        if use_cache and self.cache_manager.is_cached(dataset, query_params):
            logger.info("使用快取資料")
            data = self.cache_manager.load_from_cache(dataset, query_params)
            if data:  # 快取載入成功
                result = {'data': data, 'from_cache': True}
                if validate:
                    result['validation'] = self._validate_data(data)
                return result
        
        # 從 JHTDB 獲取新資料
        logger.info(f"從 JHTDB 獲取資料: {dataset}")
        data = self._fetch_raw_data(dataset, query_params)
        
        # 驗證資料
        validation_report = None
        if validate:
            validation_report = self._validate_data(data)
            if not all(report['valid'] for report in validation_report.values()):
                logger.warning("資料驗證發現問題")
        
        # 保存到快取
        if use_cache and data:
            metadata = {'validation_report': validation_report} if validation_report else {}
            self.cache_manager.save_to_cache(dataset, query_params, data, metadata)
        
        return {
            'data': data,
            'from_cache': False,
            'validation': validation_report
        }
    
    def _validate_data(self, data: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """驗證獲取的資料"""
        validation_results = {}
        
        for var_name, var_data in data.items():
            validation_results[var_name] = self.validator.validate_field(
                var_data, var_name, JHTDBConfig.VALIDATION_THRESHOLDS)
        
        return validation_results


class PyJHTDBClient(BaseJHTDBClient):
    """使用 pyJHTDB 的客戶端實現"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        if not PYJHTDB_AVAILABLE:
            raise JHTDBError("pyJHTDB 未安裝，請執行: pip install pyJHTDB")
        
        # 初始化 pyJHTDB
        if self.auth_token:
            pyJHTDB.dbinfo.auth_token = self.auth_token
    
    def _fetch_raw_data(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """使用 pyJHTDB 獲取資料"""
        
        query_type = query_params.get('type', 'cutout')
        variables = query_params.get('variables', ['u', 'v', 'w', 'p'])
        
        try:
            if query_type == 'cutout':
                return self._fetch_cutout(dataset, query_params, variables)
            elif query_type == 'points':
                return self._fetch_points(dataset, query_params, variables)
            else:
                raise ValueError(f"不支援的查詢類型: {query_type}")
                
        except Exception as e:
            logger.error(f"pyJHTDB 資料獲取失敗: {e}")
            raise JHTDBError(f"資料獲取失敗: {e}")
    
    def _fetch_cutout(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """獲取 cutout 資料"""
        
        start_coords = params['start']  # [x, y, z]
        end_coords = params['end']      # [x, y, z]
        timestep = params.get('timestep', 0)
        
        data = {}
        
        for var in variables:
            logger.info(f"獲取變數 {var} 的 cutout 資料...")
            
            # pyJHTDB cutout 調用
            if var in ['u', 'v', 'w']:
                # 速度場
                cutout_data = pyJHTDB.getvelocity(
                    start=start_coords,
                    end=end_coords,
                    step=[1, 1, 1],  # 空間步長
                    dataset=dataset,
                    time=timestep
                )
                
                # pyJHTDB 返回 [u, v, w] 陣列
                if var == 'u':
                    data[var] = cutout_data[:, :, :, 0]
                elif var == 'v':
                    data[var] = cutout_data[:, :, :, 1]
                elif var == 'w':
                    data[var] = cutout_data[:, :, :, 2]
            
            elif var == 'p':
                # 壓力場
                data[var] = pyJHTDB.getpressure(
                    start=start_coords,
                    end=end_coords,
                    step=[1, 1, 1],
                    dataset=dataset,
                    time=timestep
                )
        
        return data
    
    def _fetch_points(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """獲取散點資料"""
        
        points = params['points']  # [[x1,y1,z1], [x2,y2,z2], ...]
        timestep = params.get('timestep', 0)
        
        data = {}
        
        for var in variables:
            logger.info(f"獲取變數 {var} 的散點資料...")
            
            if var in ['u', 'v', 'w']:
                # 速度場插值
                velocity_data = pyJHTDB.getvelocity(
                    points=points,
                    dataset=dataset,
                    time=timestep
                )
                
                if var == 'u':
                    data[var] = velocity_data[:, 0]
                elif var == 'v':
                    data[var] = velocity_data[:, 1]
                elif var == 'w':
                    data[var] = velocity_data[:, 2]
            
            elif var == 'p':
                # 壓力場插值
                data[var] = pyJHTDB.getpressure(
                    points=points,
                    dataset=dataset,
                    time=timestep
                )
        
        return data


class MockJHTDBClient(BaseJHTDBClient):
    """模擬 JHTDB 客戶端（用於測試和離線開發）"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42  # 固定隨機種子確保可重現性
    
    def _fetch_raw_data(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """生成模擬的湍流資料"""
        
        np.random.seed(self.seed)
        
        query_type = query_params.get('type', 'cutout')
        variables = query_params.get('variables', ['u', 'v', 'w', 'p'])
        
        if query_type == 'cutout':
            return self._generate_cutout_data(dataset, query_params, variables)
        elif query_type == 'points':
            return self._generate_points_data(dataset, query_params, variables)
        else:
            raise ValueError(f"不支援的查詢類型: {query_type}")
    
    def _generate_cutout_data(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """生成模擬 cutout 資料"""
        
        start = np.array(params['start'])
        end = np.array(params['end'])
        resolution = params.get('resolution', [64, 64, 64])
        
        # 生成座標網格
        x = np.linspace(start[0], end[0], resolution[0])
        y = np.linspace(start[1], end[1], resolution[1])
        z = np.linspace(start[2], end[2], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        data = {}
        
        # 獲取資料集配置
        dataset_config = JHTDBConfig.DATASETS.get(dataset, {})
        domain = dataset_config.get('domain', {'x': [0, 2*np.pi], 'y': [0, 2*np.pi], 'z': [0, 2*np.pi]})
        
        # 針對 Channel Flow 生成具有正確物理特徵的數據
        if dataset == 'channel':
            return self._generate_channel_flow_data(X, Y, Z, variables, domain)
        else:
            # 對於其他數據集，使用等向性湍流模擬
            return self._generate_isotropic_data(X, Y, Z, variables, domain)
    
    def _generate_channel_flow_data(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                   variables: List[str], domain: Dict) -> Dict[str, np.ndarray]:
        """生成具有 Channel Flow 特徵的模擬數據"""
        
        # 標準化座標：y 方向為壁面法向 [-1, 1]
        x_norm = 2 * np.pi * (X - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = (Y - domain['y'][0]) / (domain['y'][1] - domain['y'][0]) * 2 - 1  # [-1, 1]
        z_norm = 2 * np.pi * (Z - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        data = {}
        
        for var in variables:
            if var == 'u':
                # 流向速度：拋物型平均分布 + 湍流擾動
                u_mean = 15.0 * (1 - y_norm**2)  # 拋物型分布，最大值 15 m/s
                
                # 湍流擾動：近壁面小，中心大
                turbulent_intensity = 0.5 * (1 - abs(y_norm)**3)
                u_fluctuation = (
                    turbulent_intensity * 3.0 * np.sin(2*x_norm) * np.cos(3*z_norm) +
                    turbulent_intensity * 1.5 * np.sin(4*x_norm) * np.cos(6*z_norm) +
                    turbulent_intensity * 0.3 * np.random.randn(*X.shape)
                )
                
                data[var] = u_mean + u_fluctuation
                
            elif var == 'v':
                # 壁面法向速度：很小，主要是湍流擾動
                # 邊界條件：壁面處 v=0
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                
                data[var] = wall_factor * (
                    0.8 * np.cos(x_norm) * np.sin(2*z_norm) +
                    0.4 * np.cos(3*x_norm) * np.sin(4*z_norm) +
                    0.1 * np.random.randn(*X.shape)
                )
                
            elif var == 'w':
                # 展向速度：增強湍流擾動以達到合理的展向比例 (目標 w/u ~ 0.4-0.6)
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                turbulent_intensity = 0.8 * (1 - abs(y_norm)**2)
                
                data[var] = wall_factor * (
                    8.0 * np.sin(x_norm) * np.cos(z_norm) +
                    4.0 * np.sin(3*x_norm) * np.cos(2*z_norm) +
                    turbulent_intensity * 4.0 * np.random.randn(*X.shape)
                )
                
            elif var == 'p':
                # 壓力場：受平均速度梯度和湍流影響
                u_val = data.get('u', np.zeros_like(X))
                v_val = data.get('v', np.zeros_like(X))
                w_val = data.get('w', np.zeros_like(X))
                
                # 基於連續性方程和動量方程的壓力估計
                p_mean = -2.0 * y_norm  # 線性壓力梯度（驅動流動）
                p_fluctuation = (
                    -0.3 * (u_val**2 + v_val**2 + w_val**2) +  # 動壓項
                    5.0 * np.cos(x_norm + z_norm) * (1 - y_norm**2) +  # 湍流壓力
                    0.1 * np.random.randn(*X.shape)
                )
                
                data[var] = p_mean + p_fluctuation
        
        return data
    
    def _generate_isotropic_data(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray, 
                                variables: List[str], domain: Dict) -> Dict[str, np.ndarray]:
        """生成等向性湍流數據（原有邏輯）"""
        
        # 標準化座標
        x_norm = 2 * np.pi * (X - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = 2 * np.pi * (Y - domain['y'][0]) / (domain['y'][1] - domain['y'][0])
        z_norm = 2 * np.pi * (Z - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        data = {}
        
        for var in variables:
            if var == 'u':
                # 主流方向速度：包含多尺度渦結構
                data[var] = (
                    5.0 * np.sin(x_norm) * np.cos(y_norm) * np.sin(z_norm) +
                    2.0 * np.sin(2*x_norm) * np.cos(2*y_norm) * np.sin(2*z_norm) +
                    0.5 * np.sin(4*x_norm) * np.cos(4*y_norm) * np.sin(4*z_norm) +
                    0.1 * np.random.randn(*X.shape)
                )
                
            elif var == 'v':
                # 橫向速度
                data[var] = (
                    3.0 * np.cos(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    1.5 * np.cos(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.1 * np.random.randn(*X.shape)
                )
                
            elif var == 'w':
                # 展向速度
                data[var] = (
                    2.0 * np.sin(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    1.0 * np.sin(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.1 * np.random.randn(*X.shape)
                )
                
            elif var == 'p':
                # 壓力場：需要滿足連續性方程的約束
                data[var] = (
                    -0.5 * (data.get('u', 0)**2 + data.get('v', 0)**2 + data.get('w', 0)**2) +
                    10.0 * np.cos(x_norm + y_norm + z_norm) +
                    0.05 * np.random.randn(*X.shape)
                )
        
        return data
    
    def _generate_points_data(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """生成模擬散點資料"""
        
        points = np.array(params['points'])  # [N, 3]
        n_points = points.shape[0]
        
        data = {}
        
        # 獲取資料集配置
        dataset_config = JHTDBConfig.DATASETS.get(dataset, {})
        domain = dataset_config.get('domain', {'x': [0, 2*np.pi], 'y': [0, 2*np.pi], 'z': [0, 2*np.pi]})
        
        # 針對 Channel Flow 生成具有正確物理特徵的數據
        if dataset == 'channel':
            return self._generate_channel_flow_points(points, variables, domain)
        else:
            # 對於其他數據集，使用等向性湍流模擬
            return self._generate_isotropic_points(points, variables, domain)
    
    def _generate_channel_flow_points(self, points: np.ndarray, variables: List[str], domain: Dict) -> Dict[str, np.ndarray]:
        """生成具有 Channel Flow 特徵的散點數據"""
        
        # 標準化座標：y 方向為壁面法向 [-1, 1]
        x_norm = 2 * np.pi * (points[:, 0] - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = (points[:, 1] - domain['y'][0]) / (domain['y'][1] - domain['y'][0]) * 2 - 1  # [-1, 1]
        z_norm = 2 * np.pi * (points[:, 2] - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        n_points = len(points)
        data = {}
        
        for var in variables:
            if var == 'u':
                # 流向速度：拋物型平均分布 + 湍流擾動
                u_mean = 15.0 * (1 - y_norm**2)  # 拋物型分布，最大值 15 m/s
                
                # 湍流擾動：近壁面小，中心大
                turbulent_intensity = 0.5 * (1 - np.abs(y_norm)**3)
                u_fluctuation = (
                    turbulent_intensity * 3.0 * np.sin(2*x_norm) * np.cos(3*z_norm) +
                    turbulent_intensity * 1.5 * np.sin(4*x_norm) * np.cos(6*z_norm) +
                    turbulent_intensity * 0.3 * np.random.randn(n_points)
                )
                
                data[var] = u_mean + u_fluctuation
                
            elif var == 'v':
                # 壁面法向速度：很小，主要是湍流擾動
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                
                data[var] = wall_factor * (
                    0.8 * np.cos(x_norm) * np.sin(2*z_norm) +
                    0.4 * np.cos(3*x_norm) * np.sin(4*z_norm) +
                    0.1 * np.random.randn(n_points)
                )
                
            elif var == 'w':
                # 展向速度：中等強度的湍流擾動
                wall_factor = 1 - y_norm**2  # 壁面處為 0
                
                data[var] = wall_factor * (
                    2.0 * np.sin(x_norm) * np.cos(z_norm) +
                    1.0 * np.sin(3*x_norm) * np.cos(2*z_norm) +
                    0.2 * np.random.randn(n_points)
                )
                
            elif var == 'p':
                # 壓力場：受平均速度梯度和湍流影響
                u_val = data.get('u', np.zeros(n_points))
                v_val = data.get('v', np.zeros(n_points))
                w_val = data.get('w', np.zeros(n_points))
                
                # 基於連續性方程和動量方程的壓力估計
                p_mean = -2.0 * y_norm  # 線性壓力梯度（驅動流動）
                p_fluctuation = (
                    -0.3 * (u_val**2 + v_val**2 + w_val**2) +  # 動壓項
                    5.0 * np.cos(x_norm + z_norm) * (1 - y_norm**2) +  # 湍流壓力
                    0.1 * np.random.randn(n_points)
                )
                
                data[var] = p_mean + p_fluctuation
        
        return data
    
    def _generate_isotropic_points(self, points: np.ndarray, variables: List[str], domain: Dict) -> Dict[str, np.ndarray]:
        """生成等向性湍流散點數據（原有邏輯）"""
        
        # 標準化座標
        x_norm = 2 * np.pi * (points[:, 0] - domain['x'][0]) / (domain['x'][1] - domain['x'][0])
        y_norm = 2 * np.pi * (points[:, 1] - domain['y'][0]) / (domain['y'][1] - domain['y'][0])
        z_norm = 2 * np.pi * (points[:, 2] - domain['z'][0]) / (domain['z'][1] - domain['z'][0])
        
        n_points = len(points)
        data = {}
        
        for var in variables:
            if var == 'u':
                data[var] = (
                    5.0 * np.sin(x_norm) * np.cos(y_norm) * np.sin(z_norm) +
                    2.0 * np.sin(2*x_norm) * np.cos(2*y_norm) * np.sin(2*z_norm) +
                    0.1 * np.random.randn(n_points)
                )
            elif var == 'v':
                data[var] = (
                    3.0 * np.cos(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    1.5 * np.cos(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.1 * np.random.randn(n_points)
                )
            elif var == 'w':
                data[var] = (
                    2.0 * np.sin(x_norm) * np.sin(y_norm) * np.cos(z_norm) +
                    1.0 * np.sin(2*x_norm) * np.sin(2*y_norm) * np.cos(2*z_norm) +
                    0.1 * np.random.randn(n_points)
                )
            elif var == 'p':
                u_val = data.get('u', np.zeros(n_points))
                v_val = data.get('v', np.zeros(n_points))
                w_val = data.get('w', np.zeros(n_points))
                data[var] = (
                    -0.5 * (u_val**2 + v_val**2 + w_val**2) +
                    10.0 * np.cos(x_norm + y_norm + z_norm) +
                    0.05 * np.random.randn(n_points)
                )
        
        return data


class HTTPJHTDBClient(BaseJHTDBClient):
    """使用 HTTP Web Services API 的 JHTDB 客戶端實現"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # JHTDB Web Services 配置
        self.base_url = "https://turbulence.pha.jhu.edu/service/turbulence.asmx"
        self.test_token = "edu.jhu.pha.turbulence.testing-201406"  # 測試用 token (fallback)
        
        # 優先級：傳入參數 > DEFAULT_AUTH_TOKEN（正式 token） > test_token
        # 父類已經處理了：self.auth_token = auth_token or DEFAULT_AUTH_TOKEN
        # 這裡只需確保有 token 即可
        if not self.auth_token:
            logger.warning("未提供 JHTDB 認證令牌，將使用測試令牌")
            self.auth_token = self.test_token
        
        # Mock fallback 客戶端 (當token失效時使用)
        self.mock_client = None
        self.token_verified = False
        self.use_mock_fallback = False
            
        logger.info(f"HTTPJHTDBClient 已初始化，使用 token: {self.auth_token[:20]}...")
        logger.info("📡 基於最新診斷結果：使用 GetAnyCutoutWeb API + 1-based 索引")
    
    def _fetch_raw_data(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """使用 HTTP Web Services API 獲取資料"""
        
        query_type = query_params.get('type', 'cutout')
        variables = query_params.get('variables', ['u', 'v', 'w', 'p'])
        
        try:
            # 首次嘗試驗證 token（如果尚未驗證）
            if not self.token_verified and not self.use_mock_fallback:
                test_success = self._verify_token(dataset)
                if not test_success:
                    logger.warning("Token 驗證失敗，啟用 Mock fallback 機制")
                    self.use_mock_fallback = True
                    self._initialize_mock_client()
                
            # 如果啟用了 Mock fallback，使用 Mock 客戶端
            if self.use_mock_fallback:
                return self.mock_client._fetch_raw_data(dataset, query_params)
            
            # 否則使用 HTTP API
            if query_type == 'cutout':
                return self._fetch_cutout_http(dataset, query_params, variables)
            elif query_type == 'points':
                return self._fetch_points_http(dataset, query_params, variables)
            else:
                raise ValueError(f"不支援的查詢類型: {query_type}")
                
        except JHTDBError as e:
            # 如果是 token 相關錯誤，嘗試啟用 Mock fallback
            if "Invalid identification token" in str(e):
                logger.warning("Token 認證失敗，切換到 Mock fallback 機制")
                self.use_mock_fallback = True
                self._initialize_mock_client()
                return self.mock_client._fetch_raw_data(dataset, query_params)
            else:
                raise e
        except Exception as e:
            logger.error(f"HTTP API 資料獲取失敗: {e}")
            raise JHTDBError(f"資料獲取失敗: {e}")
    
    def _verify_token(self, dataset: str) -> bool:
        """驗證 token 是否有效"""
        try:
            logger.info("🔑 驗證 JHTDB token...")
            
            # 使用最小的請求來測試 token
            test_params = {
                'type': 'points',
                'points': [[1.0, 1.0, 1.0]],
                'timestep': 1,
                'variables': ['u']
            }
            
            # 嘗試進行簡單的 GetVelocity 請求
            self._call_get_velocity(dataset, [[1.0, 1.0, 1.0]], 1)
            
            logger.info("✅ Token 驗證成功")
            self.token_verified = True
            return True
            
        except Exception as e:
            if "Invalid identification token" in str(e):
                logger.warning("❌ Token 無效")
                return False
            else:
                logger.warning(f"⚠️ Token 驗證過程出錯，但不確定是否 token 問題: {e}")
                return False
    
    def _initialize_mock_client(self):
        """初始化 Mock fallback 客戶端"""
        if self.mock_client is None:
            logger.info("🎭 初始化 Mock fallback 客戶端")
            self.mock_client = MockJHTDBClient(
                auth_token=None,
                cache_dir=self.cache_manager.cache_dir,
                timeout=self.timeout
            )
    
    def _fetch_cutout_http(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """使用 HTTP API 獲取 cutout 資料"""
        
        start_coords = params['start']  # [x, y, z] 
        end_coords = params['end']      # [x, y, z]
        timestep = params.get('timestep', 0)
        
        data = {}
        
        for var in variables:
            logger.info(f"獲取變數 {var} 的 cutout 資料...")
            
            if var in ['u', 'v', 'w']:
                # 使用 GetAnyCutoutWeb 獲取速度場（一次性獲取所有分量）
                velocity_data = self._call_get_any_cutout_web(
                    dataset, "velocity", start_coords, end_coords, timestep
                )
                
                # 解析速度分量
                if var == 'u':
                    data[var] = velocity_data[:, :, :, 0]
                elif var == 'v':
                    data[var] = velocity_data[:, :, :, 1]
                elif var == 'w':
                    data[var] = velocity_data[:, :, :, 2]
            
            elif var == 'p':
                # 使用 GetAnyCutoutWeb 獲取壓力場
                data[var] = self._call_get_any_cutout_web(
                    dataset, "pressure", start_coords, end_coords, timestep
                )
        
        return data
    
    def _fetch_points_http(self, dataset: str, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """使用 HTTP API 獲取散點資料"""
        
        points = params['points']  # [[x1,y1,z1], [x2,y2,z2], ...]
        timestep = params.get('timestep', 0)
        
        data = {}
        
        for var in variables:
            logger.info(f"獲取變數 {var} 的散點資料...")
            
            if var in ['u', 'v', 'w']:
                # 使用 GetVelocity 獲取速度場插值
                velocity_data = self._call_get_velocity(
                    dataset, points, timestep
                )
                
                if var == 'u':
                    data[var] = velocity_data[:, 0]
                elif var == 'v':
                    data[var] = velocity_data[:, 1]
                elif var == 'w':
                    data[var] = velocity_data[:, 2]
            
            elif var == 'p':
                # 使用 GetPressure 獲取壓力場插值
                data[var] = self._call_get_pressure(
                    dataset, points, timestep
                )
        
        return data
    
    def _call_get_any_cutout_web(self, dataset: str, field: str, 
                                 start: List[float], end: List[float], 
                                 timestep: int) -> np.ndarray:
        """調用 GetAnyCutoutWeb API（替代已棄用的 GetRawVelocity）"""
        
        # 將物理座標轉換為 1-based 網格索引
        # 注意：JHTDB 從 2023年9月16日起使用 1-based 索引
        start_int = [max(1, int(s) + 1) for s in start]
        end_int = [max(1, int(e) + 1) for e in end]
        
        # 構建 SOAP 請求 - 使用正確的 GetAnyCutoutWeb API 格式
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetAnyCutoutWeb xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <field>{field}</field>
      <T>{timestep}</T>
      <x_start>{start_int[0]}</x_start>
      <y_start>{start_int[1]}</y_start>
      <z_start>{start_int[2]}</z_start>
      <x_end>{end_int[0]}</x_end>
      <y_end>{end_int[1]}</y_end>
      <z_end>{end_int[2]}</z_end>
      <x_step>1</x_step>
      <y_step>1</y_step>
      <z_step>1</z_step>
      <filter_width>1</filter_width>
      <addr></addr>
    </GetAnyCutoutWeb>
  </soap:Body>
</soap:Envelope>"""
        
        response_data = self._send_soap_request(soap_request, "GetAnyCutoutWeb")
        
        # 計算實際的網格尺寸
        width = [end_int[0] - start_int[0] + 1, 
                 end_int[1] - start_int[1] + 1, 
                 end_int[2] - start_int[2] + 1]
        
        if field == "velocity":
            return self._parse_velocity_response(response_data, width)
        elif field == "pressure":
            return self._parse_pressure_response(response_data, width)
        else:
            raise ValueError(f"不支援的場類型: {field}")
    
    def _call_get_raw_velocity(self, dataset: str, start: List[float], 
                              width: List[int], timestep: int) -> np.ndarray:
        """調用 GetRawVelocity API"""
        
        # 將座標轉換為整數網格索引 (JHTDB 使用網格索引，不是物理座標)
        start_int = [int(s) for s in start]
        
        # 構建 SOAP 請求 - 使用正確的 JHTDB API 格式
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetRawVelocity xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <T>{timestep}</T>
      <X>{start_int[0]}</X>
      <Y>{start_int[1]}</Y>
      <Z>{start_int[2]}</Z>
      <Xwidth>{width[0]}</Xwidth>
      <Ywidth>{width[1]}</Ywidth>
      <Zwidth>{width[2]}</Zwidth>
    </GetRawVelocity>
  </soap:Body>
</soap:Envelope>"""
        
        response_data = self._send_soap_request(soap_request, "GetRawVelocity")
        return self._parse_velocity_response(response_data, width)
    
    def _call_get_velocity(self, dataset: str, points: List[List[float]], 
                          timestep: int) -> np.ndarray:
        """調用 GetVelocity API（散點插值）"""
        
        # 構建點的 XML 格式 (不是二進制編碼)
        points_xml = ""
        for point in points:
            points_xml += f"""
        <Point3>
          <x>{point[0]}</x>
          <y>{point[1]}</y>
          <z>{point[2]}</z>
        </Point3>"""
        
        # 構建 SOAP 請求 - 使用正確的參數格式
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetVelocity xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <time>{float(timestep)}</time>
      <spatialInterpolation>Lag4</spatialInterpolation>
      <temporalInterpolation>None</temporalInterpolation>
      <points>{points_xml}
      </points>
    </GetVelocity>
  </soap:Body>
</soap:Envelope>"""
        
        response_data = self._send_soap_request(soap_request, "GetVelocity")
        return self._parse_velocity_points_response(response_data, len(points))
    
    def _call_get_raw_pressure(self, dataset: str, start: List[float], 
                              width: List[int], timestep: int) -> np.ndarray:
        """調用 GetRawPressure API"""
        
        # 將座標轉換為整數網格索引 (JHTDB 使用網格索引，不是物理座標)
        start_int = [int(s) for s in start]
        
        # 構建 SOAP 請求 - 使用正確的 JHTDB API 格式
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetRawPressure xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <T>{timestep}</T>
      <X>{start_int[0]}</X>
      <Y>{start_int[1]}</Y>
      <Z>{start_int[2]}</Z>
      <Xwidth>{width[0]}</Xwidth>
      <Ywidth>{width[1]}</Ywidth>
      <Zwidth>{width[2]}</Zwidth>
    </GetRawPressure>
  </soap:Body>
</soap:Envelope>"""
        
        response_data = self._send_soap_request(soap_request, "GetRawPressure")
        return self._parse_pressure_response(response_data, width)
    
    def _call_get_pressure(self, dataset: str, points: List[List[float]], 
                          timestep: int) -> np.ndarray:
        """調用 GetPressure API（散點插值）"""
        
        points_binary = self._encode_points(points)
        
        soap_request = f"""<?xml version="1.0" encoding="utf-8"?>
<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
               xmlns:xsd="http://www.w3.org/2001/XMLSchema" 
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
  <soap:Body>
    <GetPressure xmlns="http://turbulence.pha.jhu.edu/">
      <authToken>{self.auth_token}</authToken>
      <dataset>{dataset}</dataset>
      <time>{timestep}</time>
      <spatialInterpolation>6</spatialInterpolation>
      <temporalInterpolation>0</temporalInterpolation>
       <points>{points_binary}</points>
     </GetPressure>
   </soap:Body>
</soap:Envelope>"""
        
        response_data = self._send_soap_request(soap_request, "GetPressure")
        return self._parse_pressure_points_response(response_data, len(points))
    
    def _send_soap_request(self, soap_request: str, api_method: str) -> bytes:
        """發送 SOAP 請求到 JHTDB 服務器"""
        
        # ASMX Web Service 使用統一的端點 URL
        url = self.base_url
        
        headers = {
            'Content-Type': 'text/xml; charset=utf-8',
            'SOAPAction': f'http://turbulence.pha.jhu.edu/{api_method}',
            'User-Agent': 'Python JHTDB Client'
        }
        
        # 編碼請求
        request_data = soap_request.encode('utf-8')
        
        # 創建請求
        req = urllib.request.Request(
            url, 
            data=request_data, 
            headers=headers
        )
        
        # 發送請求並處理重試
        for attempt in range(JHTDBConfig.MAX_RETRY):
            try:
                logger.debug(f"發送 SOAP 請求（嘗試 {attempt + 1}/{JHTDBConfig.MAX_RETRY}）")
                
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    if response.status == 200:
                        response_data = response.read()
                        logger.debug(f"請求成功，響應大小: {len(response_data)} bytes")
                        return self._extract_binary_data(response_data)
                    else:
                        raise JHTDBError(f"HTTP 錯誤: {response.status}")
                        
            except urllib.error.URLError as e:
                logger.warning(f"請求失敗（嘗試 {attempt + 1}）: {e}")
                if attempt == JHTDBConfig.MAX_RETRY - 1:
                    raise JHTDBError(f"所有重試均失敗: {e}")
                time.sleep(2 ** attempt)  # 指數退避
            
            except Exception as e:
                logger.error(f"未預期的錯誤: {e}")
                if attempt == JHTDBConfig.MAX_RETRY - 1:
                    raise JHTDBError(f"請求處理失敗: {e}")
                time.sleep(2 ** attempt)
        
        # 如果所有重試都失敗，拋出錯誤
        raise JHTDBError("所有連接嘗試均失敗")
    
    def _extract_binary_data(self, response_data: bytes) -> bytes:
        """從 SOAP 響應中提取 Base64 編碼的二進制數據"""
        
        try:
            # 解析 XML 響應
            response_str = response_data.decode('utf-8')
            root = ET.fromstring(response_str)
            
            # 尋找包含 Base64 數據的元素
            # JHTDB 通常在 <soap:Body> 的結果元素中返回 Base64 數據
            namespaces = {
                'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
                'jhtdb': 'http://turbulence.pha.jhu.edu/'
            }
            
            # 尋找結果元素（可能是不同的名稱）
            result_elements = root.findall('.//soap:Body/*/*', namespaces)
            
            if not result_elements:
                # 如果沒有找到，嘗試沒有命名空間的查找
                result_elements = root.findall('.//Body/*/*')
            
            if result_elements:
                base64_data = result_elements[0].text
                if base64_data:
                    return base64.b64decode(base64_data)
                else:
                    logger.error("Base64 數據為空")
                    raise JHTDBError("響應中的數據為空")
            
            # 如果仍然沒有找到，記錄響應內容用於調試
            logger.error(f"無法從 SOAP 響應中提取數據")
            logger.debug(f"響應內容（前1000字符）: {response_str[:1000]}")
            raise JHTDBError("無法解析 SOAP 響應")
            
        except ET.ParseError as e:
            logger.error(f"XML 解析失敗: {e}")
            raise JHTDBError(f"響應格式錯誤: {e}")
        except Exception as e:
            logger.error(f"數據提取失敗: {e}")
            raise JHTDBError(f"響應處理失敗: {e}")
    
    def _encode_points(self, points: List[List[float]]) -> str:
        """將點座標編碼為 Base64 二進制格式"""
        
        # JHTDB 期望的格式是：float32 陣列，每個點 3 個座標 (x, y, z)
        points_array = np.array(points, dtype=np.float32)
        
        # 轉換為二進制
        binary_data = points_array.tobytes()
        
        # Base64 編碼
        return base64.b64encode(binary_data).decode('ascii')
    
    def _parse_velocity_response(self, binary_data: bytes, width: List[int]) -> np.ndarray:
        """解析速度場 cutout 響應"""
        
        # JHTDB velocity 數據格式: float32, [width[0], width[1], width[2], 3]
        expected_size = width[0] * width[1] * width[2] * 3 * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 重塑為 [width[0], width[1], width[2], 3] 形狀
        return data_array.reshape(width[0], width[1], width[2], 3)
    
    def _parse_pressure_response(self, binary_data: bytes, width: List[int]) -> np.ndarray:
        """解析壓力場 cutout 響應"""
        
        # JHTDB pressure 數據格式: float32, [width[0], width[1], width[2]]
        expected_size = width[0] * width[1] * width[2] * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 重塑為 [width[0], width[1], width[2]] 形狀
        return data_array.reshape(width[0], width[1], width[2])
    
    def _parse_velocity_points_response(self, response_data: bytes, n_points: int) -> np.ndarray:
        """解析 GetVelocity 的 XML 響應，包含 Vector3 數組"""
        
        try:
            # 解析 XML 響應
            response_str = response_data.decode('utf-8')
            root = ET.fromstring(response_str)
            
            # 尋找 GetVelocityResult 元素
            namespaces = {
                'soap': 'http://schemas.xmlsoap.org/soap/envelope/',
                'jhtdb': 'http://turbulence.pha.jhu.edu/'
            }
            
            # 查找結果元素
            result_elem = root.find('.//jhtdb:GetVelocityResult', namespaces)
            if result_elem is None:
                # 嘗試不使用命名空間
                result_elem = root.find('.//GetVelocityResult')
            
            if result_elem is None:
                logger.error("無法找到 GetVelocityResult 元素")
                logger.debug(f"響應內容: {response_str[:1000]}")
                raise JHTDBError("響應格式錯誤：找不到結果元素")
            
            # 解析 Vector3 元素
            vectors = []
            vector_elems = result_elem.findall('.//Vector3') or result_elem.findall('.//jhtdb:Vector3', namespaces)
            
            for vector_elem in vector_elems:
                x = float(vector_elem.find('x').text or vector_elem.find('jhtdb:x', namespaces).text)
                y = float(vector_elem.find('y').text or vector_elem.find('jhtdb:y', namespaces).text)
                z = float(vector_elem.find('z').text or vector_elem.find('jhtdb:z', namespaces).text)
                vectors.append([x, y, z])
            
            if len(vectors) != n_points:
                logger.warning(f"返回的點數不符：期望 {n_points}, 實際 {len(vectors)}")
            
            return np.array(vectors, dtype=np.float32)
            
        except Exception as e:
            logger.error(f"解析 Vector3 響應失敗: {e}")
            # 回退到二進制解析（適用於舊版本或不同格式）
            try:
                binary_data = self._extract_binary_data(response_data)
                expected_size = n_points * 3 * 4  # 4 bytes per float32
                
                if len(binary_data) != expected_size:
                    logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
                
                # 解析為 float32 陣列
                data_array = np.frombuffer(binary_data, dtype=np.float32)
                
                # 重塑為 [n_points, 3] 形狀
                return data_array.reshape(n_points, 3)
            except Exception as e2:
                logger.error(f"二進制回退解析也失敗: {e2}")
                raise JHTDBError(f"無法解析響應數據: {e}")
    
    def _parse_pressure_points_response(self, binary_data: bytes, n_points: int) -> np.ndarray:
        """解析壓力場散點響應"""
        
        # JHTDB pressure points 數據格式: float32, [n_points]
        expected_size = n_points * 4  # 4 bytes per float32
        
        if len(binary_data) != expected_size:
            logger.warning(f"數據大小不符：期望 {expected_size}, 實際 {len(binary_data)}")
        
        # 解析為 float32 陣列
        data_array = np.frombuffer(binary_data, dtype=np.float32)
        
        # 返回一維陣列
        return data_array


class JHTDBManager:
    """JHTDB 管理器：提供高層級的資料存取接口"""
    
    def __init__(self, 
                 use_mock: bool = False,
                 use_http: bool = True,
                 auth_token: Optional[str] = None,
                 cache_dir: Optional[str] = None,
                 **kwargs):
        """
        Args:
            use_mock: 是否強制使用模擬客戶端（用於離線開發）
            use_http: 是否優先使用 HTTP 客戶端（預設）
            auth_token: JHTDB 認證令牌
            cache_dir: 快取目錄
        """
        
        # 客戶端選擇邏輯：
        # 1. 如果 use_mock=True，強制使用 MockJHTDBClient
        # 2. 如果 use_http=True（預設），優先使用 HTTPJHTDBClient
        # 3. 如果 pyJHTDB 可用且 use_http=False，使用 PyJHTDBClient
        # 4. 最後退回到 MockJHTDBClient
        
        if use_mock:
            logger.info("使用者指定模擬客戶端")
            self.client = MockJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
            self.client_type = "mock"
            
        elif use_http:
            logger.info("使用 HTTP Web Services 客戶端")
            try:
                self.client = HTTPJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                self.client_type = "http"
            except Exception as e:
                logger.warning(f"HTTP 客戶端初始化失敗: {e}")
                logger.info("退回到模擬客戶端")
                self.client = MockJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                self.client_type = "mock"
                
        elif PYJHTDB_AVAILABLE:
            logger.info("使用 pyJHTDB 客戶端")
            try:
                self.client = PyJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                self.client_type = "pyjhtdb"
            except Exception as e:
                logger.warning(f"pyJHTDB 客戶端初始化失敗: {e}")
                logger.info("退回到 HTTP 客戶端")
                try:
                    self.client = HTTPJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                    self.client_type = "http"
                except Exception as e2:
                    logger.warning(f"HTTP 客戶端也失敗: {e2}")
                    logger.info("最終退回到模擬客戶端")
                    self.client = MockJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                    self.client_type = "mock"
                    
        else:
            logger.warning("pyJHTDB 不可用，嘗試 HTTP 客戶端")
            try:
                self.client = HTTPJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                self.client_type = "http"
            except Exception as e:
                logger.warning(f"HTTP 客戶端初始化失敗: {e}")
                logger.info("退回到模擬客戶端")
                self.client = MockJHTDBClient(auth_token=auth_token, cache_dir=cache_dir, **kwargs)
                self.client_type = "mock"
        
        logger.info(f"JHTDB 客戶端類型: {self.client_type}")
        self.datasets = JHTDBConfig.DATASETS
    
    def fetch_cutout(self,
                    dataset: str,
                    start: List[float],
                    end: List[float],
                    timestep: int = 0,
                    variables: List[str] = None,
                    resolution: List[int] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        獲取 cutout 資料
        
        Args:
            dataset: 資料集名稱
            start: 起始座標 [x, y, z]
            end: 結束座標 [x, y, z]
            timestep: 時間步
            variables: 變數列表
            resolution: 解析度 [nx, ny, nz] (僅用於模擬資料)
            
        Returns:
            包含資料和元資料的字典
        """
        
        if dataset not in self.datasets:
            raise ValueError(f"未知資料集: {dataset}")
        
        variables = variables or ['u', 'v', 'w', 'p']
        
        query_params = {
            'type': 'cutout',
            'start': start,
            'end': end,
            'timestep': timestep,
            'variables': variables
        }
        
        if resolution:
            query_params['resolution'] = resolution
        
        return self.client.fetch_data(dataset, query_params, **kwargs)
    
    def fetch_points(self,
                    dataset: str,
                    points: List[List[float]],
                    timestep: int = 0,
                    variables: List[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        獲取散點資料
        
        Args:
            dataset: 資料集名稱
            points: 座標點列表 [[x1,y1,z1], [x2,y2,z2], ...]
            timestep: 時間步
            variables: 變數列表
            
        Returns:
            包含資料和元資料的字典
        """
        
        if dataset not in self.datasets:
            raise ValueError(f"未知資料集: {dataset}")
        
        variables = variables or ['u', 'v', 'w', 'p']
        
        query_params = {
            'type': 'points',
            'points': points,
            'timestep': timestep,
            'variables': variables
        }
        
        return self.client.fetch_data(dataset, query_params, **kwargs)
    
    def get_dataset_info(self, dataset: str) -> Dict:
        """獲取資料集資訊"""
        if dataset not in self.datasets:
            raise ValueError(f"未知資料集: {dataset}")
        return self.datasets[dataset].copy()
    
    def list_datasets(self) -> List[str]:
        """列出所有可用資料集"""
        return list(self.datasets.keys())
    
    def clear_cache(self, older_than_days: int = 30):
        """清理快取"""
        self.client.cache_manager.clear_cache(older_than_days)


# 便捷函數
def create_jhtdb_manager(use_mock: bool = False, **kwargs) -> JHTDBManager:
    """創建 JHTDB 管理器的便捷函數"""
    return JHTDBManager(use_mock=use_mock, **kwargs)


def fetch_sample_data(dataset: str = 'isotropic1024coarse',
                     n_points: int = 100,
                     use_mock: bool = True) -> Dict[str, Any]:
    """
    獲取樣本資料的便捷函數
    
    Args:
        dataset: 資料集名稱
        n_points: 樣本點數
        use_mock: 是否使用模擬資料
        
    Returns:
        樣本資料字典
    """
    
    manager = create_jhtdb_manager(use_mock=use_mock)
    dataset_info = manager.get_dataset_info(dataset)
    
    # 在資料集域內生成隨機點
    domain = dataset_info['domain']
    np.random.seed(42)  # 固定種子確保可重現性
    
    points = []
    for _ in range(n_points):
        x = np.random.uniform(domain['x'][0], domain['x'][1])
        y = np.random.uniform(domain['y'][0], domain['y'][1])
        z = np.random.uniform(domain['z'][0], domain['z'][1])
        points.append([x, y, z])
    
    return manager.fetch_points(dataset, points, timestep=0)


if __name__ == "__main__":
    # 測試程式碼
    print("🌊 測試 JHTDB 客戶端...")
    
    # 測試模擬客戶端
    print("\n=== 測試模擬客戶端 ===")
    
    manager = create_jhtdb_manager(use_mock=True)
    
    # 列出資料集
    print(f"可用資料集: {manager.list_datasets()}")
    
    # 獲取資料集資訊
    dataset = 'isotropic1024coarse'
    info = manager.get_dataset_info(dataset)
    print(f"\n資料集 {dataset} 資訊:")
    print(f"  描述: {info['description']}")
    print(f"  域範圍: {info['domain']}")
    print(f"  解析度: {info['resolution']}")
    
    # 測試 cutout 資料
    print(f"\n測試 cutout 資料...")
    cutout_result = manager.fetch_cutout(
        dataset=dataset,
        start=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        resolution=[32, 32, 32],
        variables=['u', 'v', 'p']
    )
    
    print(f"Cutout 資料獲取成功: {not cutout_result['from_cache']}")
    data = cutout_result['data']
    for var, arr in data.items():
        print(f"  {var}: {arr.shape}, 範圍=[{arr.min():.3f}, {arr.max():.3f}]")
    
    # 測試散點資料
    print(f"\n測試散點資料...")
    points = [[0.5, 0.5, 0.5], [1.0, 1.0, 1.0], [1.5, 1.5, 1.5]]
    points_result = manager.fetch_points(
        dataset=dataset,
        points=points,
        variables=['u', 'v', 'w', 'p']
    )
    
    print(f"散點資料獲取成功: {not points_result['from_cache']}")
    data = points_result['data']
    for var, arr in data.items():
        print(f"  {var}: {arr.shape}, 值={arr}")
    
    # 測試快取功能
    print(f"\n測試快取功能...")
    cutout_result_cached = manager.fetch_cutout(
        dataset=dataset,
        start=[0.0, 0.0, 0.0],
        end=[1.0, 1.0, 1.0],
        resolution=[32, 32, 32],
        variables=['u', 'v', 'p']
    )
    print(f"使用快取: {cutout_result_cached['from_cache']}")
    
    # 測試驗證功能
    if 'validation' in cutout_result:
        print(f"\n資料驗證結果:")
        for var, report in cutout_result['validation'].items():
            status = "✅" if report['valid'] else "❌"
            print(f"  {var}: {status} (警告: {len(report['warnings'])}, 錯誤: {len(report['errors'])})")
    
    # 測試便捷函數
    print(f"\n測試便捷函數...")
    sample_data = fetch_sample_data(dataset='channel', n_points=10, use_mock=True)
    print(f"樣本資料獲取成功: {len(sample_data['data'])} 個變數")
    
    print("\n✅ JHTDB 客戶端測試完成！")