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

# 載入環境變數（支援 .env 文件）
try:
    from dotenv import load_dotenv
    # 在專案根目錄尋找 .env 文件
    project_root = Path(__file__).parent.parent.parent
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        logging.getLogger(__name__).debug(f"已載入環境變數文件: {env_file}")
except ImportError:
    # 如果沒有安裝 python-dotenv，僅使用系統環境變數
    logging.getLogger(__name__).debug("python-dotenv 未安裝，僅使用系統環境變數")
    pass

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
    """模擬 JHTDB 客戶端（用於測試和離線開發）
    
    使用 mock_turbulence_generator 模組生成物理真實的湍流場數據。
    支援 Channel Flow 和 Isotropic Turbulence 兩種數據集類型。
    
    Attributes:
        seed: 隨機種子，確保可重現性
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.seed = 42  # 固定隨機種子確保可重現性
    
    def _fetch_raw_data(self, dataset: str, query_params: Dict) -> Dict[str, np.ndarray]:
        """生成模擬的湍流資料
        
        Args:
            dataset: 資料集名稱 ('channel', 'isotropic1024coarse', etc.)
            query_params: 查詢參數字典
                - type: 'cutout' 或 'points'
                - variables: 變數列表 ['u', 'v', 'w', 'p']
                - start, end, resolution (cutout)
                - points (points query)
        
        Returns:
            包含速度/壓力場的字典
        """
        from pinnx.dataio.mock_turbulence_generator import create_generator
        
        # 創建湍流生成器
        generator = create_generator(dataset, seed=self.seed)
        
        query_type = query_params.get('type', 'cutout')
        variables = query_params.get('variables', ['u', 'v', 'w', 'p'])
        
        if query_type == 'cutout':
            return self._generate_cutout(generator, query_params, variables)
        elif query_type == 'points':
            return self._generate_points(generator, query_params, variables)
        else:
            raise ValueError(f"不支援的查詢類型: {query_type}")
    
    def _generate_cutout(self, generator, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """生成模擬 cutout 資料
        
        Args:
            generator: TurbulenceGenerator 實例
            params: 參數字典 (start, end, resolution)
            variables: 變數列表
        
        Returns:
            速度/壓力場字典
        """
        start = np.array(params['start'])
        end = np.array(params['end'])
        resolution = params.get('resolution', [64, 64, 64])
        
        # 生成座標網格
        x = np.linspace(start[0], end[0], resolution[0])
        y = np.linspace(start[1], end[1], resolution[1])
        z = np.linspace(start[2], end[2], resolution[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 委託給 generator 生成數據
        return generator.generate_velocity_field(X, Y, Z, variables)
    
    def _generate_points(self, generator, params: Dict, variables: List[str]) -> Dict[str, np.ndarray]:
        """生成模擬散點資料
        
        Args:
            generator: TurbulenceGenerator 實例
            params: 參數字典 (points)
            variables: 變數列表
        
        Returns:
            速度/壓力場字典
        """
        points = np.array(params['points'])  # [N, 3]
        
        # 委託給 generator 生成數據
        return generator.generate_points(points, variables)


class HTTPJHTDBClient(BaseJHTDBClient):
    """基於 HTTP Web Services 的 JHTDB 客戶端
    
    使用 SOAP 協議與 JHTDB Web Services API 通信。
    SOAP 邏輯委託給 pinnx.dataio.soap_utils 模組。
    
    Attributes:
        base_url: JHTDB SOAP 服務端點
        soap_request: SOAPRequest 實例（處理請求構建和發送）
        soap_parser: SOAPResponseParser 實例（處理響應解析）
        mock_client: Mock fallback 客戶端（當 token 失效時使用）
    """
    
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
        
        # 初始化 SOAP 工具
        from pinnx.dataio.soap_utils import SOAPRequest, SOAPResponseParser
        self.soap_request = SOAPRequest(
            base_url=self.base_url,
            auth_token=self.auth_token,
            timeout=self.timeout,
            max_retry=JHTDBConfig.MAX_RETRY
        )
        self.soap_parser = SOAPResponseParser()
        
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
    
    def _physical_to_grid(self, dataset: str, 
                         start: List[float], 
                         end: List[float]) -> Tuple[List[int], List[int]]:
        """
        將物理座標轉換為網格索引（1-based）
        
        Args:
            dataset: 資料集名稱
            start: 起始物理座標 [x, y, z]
            end: 結束物理座標 [x, y, z]
            
        Returns:
            (start_indices, end_indices): 1-based 網格索引
        """
        dataset_config = JHTDBConfig.DATASETS.get(dataset, {})
        domain = dataset_config.get('domain', {'x': [0, 8*np.pi], 'y': [-1, 1], 'z': [0, 3*np.pi]})
        resolution = dataset_config.get('resolution', {'x': 2048, 'y': 512, 'z': 1536})
        
        # 物理域範圍
        L = [domain['x'][1] - domain['x'][0],
             domain['y'][1] - domain['y'][0],
             domain['z'][1] - domain['z'][0]]
        
        # 網格解析度
        N = [resolution['x'], resolution['y'], resolution['z']]
        
        # 轉換為網格索引（0-based）
        start_grid = []
        end_grid = []
        
        for i in range(3):
            # 計算在域內的相對位置 [0, 1]
            axis_key = ['x', 'y', 'z'][i]
            domain_min = domain[axis_key][0]
            
            # 轉換為 [0, 1] 範圍
            start_norm = (start[i] - domain_min) / L[i]
            end_norm = (end[i] - domain_min) / L[i]
            
            # 轉換為網格索引（0-based）
            start_idx_0 = int(start_norm * N[i])
            end_idx_0 = int(end_norm * N[i])
            
            # 轉換為 1-based（JHTDB 要求）
            start_grid.append(start_idx_0 + 1)
            end_grid.append(end_idx_0 + 1)
        
        logger.debug(f"座標轉換: physical {start} -> {end} => grid {start_grid} -> {end_grid}")
        
        return start_grid, end_grid
    
    def _call_get_any_cutout_web(self, dataset: str, field: str, 
                                 start: List[float], end: List[float], 
                                 timestep: int) -> np.ndarray:
        """調用 GetAnyCutoutWeb API（使用 soap_utils）
        
        Args:
            dataset: 資料集名稱
            field: 場類型 ('velocity', 'pressure')
            start: 起始物理座標
            end: 結束物理座標
            timestep: 時間步
        
        Returns:
            速度場 [nx, ny, nz, 3] 或壓力場 [nx, ny, nz]
        """
        # 將物理座標轉換為 1-based 網格索引
        start_int, end_int = self._physical_to_grid(dataset, start, end)
        
        # 確保索引在有效範圍內（1-based）
        dataset_config = JHTDBConfig.DATASETS.get(dataset, {})
        resolution = dataset_config.get('resolution', {'x': 2048, 'y': 512, 'z': 1536})
        res_list = [resolution['x'], resolution['y'], resolution['z']]
        
        start_int = [max(1, min(s, res_list[i])) for i, s in enumerate(start_int)]
        end_int = [max(1, min(e, res_list[i])) for i, e in enumerate(end_int)]
        
        # 構建 SOAP 請求（委託給 soap_utils）
        soap_xml = self.soap_request.build_get_any_cutout_request(
            dataset, field, start_int, end_int, timestep
        )
        
        # 發送請求（委託給 soap_utils）
        binary_data = self.soap_request.send_request(soap_xml, "GetAnyCutoutWeb")
        
        # 計算實際的網格尺寸
        width = [end_int[0] - start_int[0] + 1, 
                 end_int[1] - start_int[1] + 1, 
                 end_int[2] - start_int[2] + 1]
        
        # 解析響應（委託給 soap_utils）
        if field == "velocity":
            return self.soap_parser.parse_velocity_cutout(binary_data, width)
        elif field == "pressure":
            return self.soap_parser.parse_pressure_cutout(binary_data, width)
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
        """調用 GetVelocity API（散點插值，使用 soap_utils）
        
        Args:
            dataset: 資料集名稱
            points: 查詢點列表 [[x, y, z], ...]
            timestep: 時間步
        
        Returns:
            速度場陣列 [n_points, 3]
        """
        # 構建 SOAP 請求（委託給 soap_utils）
        soap_xml = self.soap_request.build_get_velocity_request(dataset, points, float(timestep))
        
        # 發送請求（委託給 soap_utils）
        response_data = self.soap_request.send_request(soap_xml, "GetVelocity")
        
        # 解析響應（委託給 soap_utils）
        return self.soap_parser.parse_velocity_points(response_data, len(points))
    
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
        """調用 GetPressure API（散點插值，使用 soap_utils）
        
        Args:
            dataset: 資料集名稱
            points: 查詢點列表 [[x, y, z], ...]
            timestep: 時間步
        
        Returns:
            壓力場陣列 [n_points]
        """
        # 構建 SOAP 請求（委託給 soap_utils）
        soap_xml = self.soap_request.build_get_pressure_request(dataset, points, float(timestep))
        
        # 發送請求（委託給 soap_utils）
        binary_data = self.soap_request.send_request(soap_xml, "GetPressure")
        
        # 解析響應（委託給 soap_utils）
        return self.soap_parser.parse_pressure_points(binary_data, len(points))


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