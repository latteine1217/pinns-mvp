"""
DataIO 模組

提供低保真資料載入、高保真資料取樣和資料預處理功能。
支援 RANS、LES、下採樣 DNS 等多種低保真資料類型，
以及與 JHTDB 的高保真資料整合。

主要組件：
- LowFiLoader: 低保真資料載入器主類  
- LowFiData: 低保真資料容器
- RANSReader: RANS 特定讀取器
- LESReader: LES 特定讀取器  
- DownsampledDNSProcessor: 下採樣 DNS 處理器
- SpatialInterpolator: 空間插值器
- JHTDBClient: JHTDB 資料客戶端
"""

# 核心資料結構
from .lowfi_loader import (
    LowFiData,
    LowFiLoader,
    load_lowfi_data,
    create_mock_rans_data
)

# 資料讀取器
from .lowfi_loader import (
    DataReader,
    NetCDFReader,
    HDF5Reader,  
    NPZReader,
    RANSReader,
    LESReader
)

# 資料處理器
from .lowfi_loader import (
    DownsampledDNSProcessor,
    SpatialInterpolator
)

# JHTDB 客戶端
try:
    from .jhtdb_client import JHTDBClient
except ImportError:
    # 如果 pyJHTDB 不可用，提供警告
    import warnings
    warnings.warn(
        "JHTDBClient unavailable. Install pyJHTDB for JHTDB functionality.",
        ImportWarning
    )
    JHTDBClient = None

# 版本信息
__version__ = "0.1.0"

# 便利函數
def create_lowfi_loader(interpolation_method: str = 'linear', 
                       filter_type: str = 'box') -> LowFiLoader:
    """創建預配置的低保真載入器"""
    loader = LowFiLoader()
    loader.interpolator = SpatialInterpolator(method=interpolation_method)
    loader.dns_processor = DownsampledDNSProcessor(filter_type=filter_type)
    return loader


def quick_load_and_interpolate(filepath: str, target_points, 
                              data_type: str = 'auto',
                              interpolation_method: str = 'linear'):
    """快速載入並插值低保真資料到目標點"""
    loader = create_lowfi_loader(interpolation_method)
    data = loader.load(filepath, data_type)
    return loader.create_prior_data(data, target_points)


# 公開 API
__all__ = [
    # 核心類別
    'LowFiData',
    'LowFiLoader', 
    'DataReader',
    
    # 讀取器
    'NetCDFReader',
    'HDF5Reader',
    'NPZReader', 
    'RANSReader',
    'LESReader',
    
    # 處理器
    'DownsampledDNSProcessor',
    'SpatialInterpolator',
    
    # JHTDB
    'JHTDBClient',
    
    # 便利函數
    'load_lowfi_data',
    'create_mock_rans_data',
    'create_lowfi_loader',
    'quick_load_and_interpolate',
]