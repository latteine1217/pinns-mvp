"""
DataIO 模組 (Simplified)

提供低保真資料載入、高保真資料取樣和資料預處理功能。

=== 支援的組件 (SUPPORTED) ===
✅ HDF5Reader: HDF5 檔案讀取器
✅ RANSReader: RANS 特定讀取器 (3D Channel Flow)
✅ SpatialInterpolator: 空間插值器
✅ JHTDBClient: JHTDB 資料客戶端
✅ ChannelFlowLoader: 通道流專用載入器

=== 已棄用 (DEPRECATED - Retained for backward compatibility) ===
⚠️ NetCDFReader: 專案僅支援 HDF5 格式
⚠️ LESReader: LES 模型不在專案範圍內
⚠️ DownsampledDNSProcessor: DNS 下採樣不使用

注意: LowFiLoader 類別為舊版實作,實際訓練管道使用
      scripts/train/train.py::load_rans_prior_data() 直接載入。

參考文檔: docs/LOWFI_PRIOR_GUIDE.md
"""

# 核心資料結構
from .structures import (
    StructuredGrid,
    StructuredField,
    PointSamples,
    FlowDataBundle,
    DomainSpec
)
from .lowfi_loader import (
    LowFiData,
    LowFiLoader,
    load_lowfi_data,
    create_mock_rans_data
)

# 資料讀取器
from .lowfi_loader import (
    DataReader,
    HDF5Reader,        # ✅ SUPPORTED
    NPZReader,
    RANSReader         # ✅ SUPPORTED (3D Channel Flow)
)

# 資料處理器
from .lowfi_loader import (
    SpatialInterpolator       # ✅ SUPPORTED
)

# JHTDB 客戶端
try:
    from .jhtdb_client import JHTDBManager as JHTDBClient, JHTDBConfig
except ImportError:
    # 如果 pyJHTDB 不可用，提供警告
    import warnings
    warnings.warn(
        "JHTDBClient unavailable. Install pyJHTDB for JHTDB functionality.",
        ImportWarning
    )
    JHTDBClient = None
    JHTDBConfig = None

# Channel Flow 專用載入器
try:
    from .channel_flow_loader import ChannelFlowLoader, ChannelFlowData
except ImportError:
    import warnings
    warnings.warn(
        "ChannelFlowLoader unavailable. Some functionality may be limited.",
        ImportWarning
    )
    ChannelFlowLoader = None
    ChannelFlowData = None

# 版本信息
__version__ = "0.1.0"

# 便利函數
def create_lowfi_loader(interpolation_method: str = 'linear') -> LowFiLoader:
    """創建預配置的低保真載入器"""
    loader = LowFiLoader()
    loader.interpolator = SpatialInterpolator(method=interpolation_method)
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
    'StructuredGrid',
    'StructuredField',
    'PointSamples',
    'FlowDataBundle',
    'DomainSpec',
    
    # 讀取器
    'HDF5Reader',
    'NPZReader', 
    'RANSReader',
    
    # 處理器
    'SpatialInterpolator',
    
    # JHTDB
    'JHTDBClient',
    
    # Channel Flow專用
    'ChannelFlowLoader',
    'ChannelFlowData',
    
    # 便利函數
    'load_lowfi_data',
    'create_mock_rans_data',
    'create_lowfi_loader',
    'quick_load_and_interpolate',
]
