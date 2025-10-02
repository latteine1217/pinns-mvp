"""
PINNx: Physics-Informed Neural Networks for Inverse Problems
===========================================================

專案主套件初始化檔案
集成所有核心功能模組，提供統一的 API 介面

主要功能模組：
- physics: 物理定律與數值計算（NS方程、尺度化）
- models: 神經網路架構（Fourier MLP、包裝器）
- losses: 損失函數（殘差、先驗、動態權重）
- sensors: 感測點選擇（QR-pivot、最佳化佈點）
- dataio: 資料輸入輸出（JHTDB 客戶端、低保真載入）
- train: 訓練框架（單模型、集成訓練）
- evals: 評估與分析（指標計算、可視化）
"""

# 版本資訊
__version__ = "1.0.0"
__author__ = "PINNx Team"
__email__ = "contact@pinnx-inverse.org"
__description__ = "Physics-Informed Neural Networks for Sparse-Data Inverse Problems"

# 套件層級配置
import logging
import warnings
import sys
from pathlib import Path

# 設定日誌格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('pinnx.log')
    ]
)

# 創建主日誌器
logger = logging.getLogger(__name__)

# 套件根目錄
PACKAGE_ROOT = Path(__file__).parent.absolute()
PROJECT_ROOT = PACKAGE_ROOT.parent

# 抑制不必要的警告
warnings.filterwarnings('ignore', category=FutureWarning, module='torch')
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 檢查依賴套件
def check_dependencies():
    """檢查必要的依賴套件是否已安裝"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
        'h5py': 'HDF5 Python',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Please install missing packages using: conda env create -f environment.yml")
        sys.exit(1)
    else:
        logger.info("All required dependencies are available")

# 在導入時檢查依賴
check_dependencies()

# 核心功能模組導入
try:
    # 物理與數值計算
    from .physics import (
        ns_residual_2d,
        VSScaler,
        compute_derivatives,
        check_conservation
    )
    
    # 神經網路模型
    from .models import (
        FourierFeatures,
        PINNNet,
        MultiOutputWrapper,
        create_model
    )
    
    # 損失函數
    from .losses import (
        residual_loss,
        data_loss,
        boundary_loss,
        prior_consistency_loss,
        GradNorm,
        causal_weights
    )
    
    # 感測點選擇
    from .sensors import (
        qr_pivot_select,
        optimal_sensor_placement,
        sensor_sensitivity_analysis
    )
    
    # 資料處理
    from .dataio import (
        JHTDBClient,
        LowFidelityLoader,
        DataModule,
        create_dataloader
    )
    
    # 訓練框架  
    from .train import (
        train_single_model,
        train_ensemble,
        TrainingConfig,
        EnsembleConfig
    )
    
    # 評估與分析
    from .evals import (
        compute_metrics,
        plot_results,
        uncertainty_analysis,
        convergence_analysis
    )
    
    logger.info("Successfully imported all PINNx modules")
    
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    logger.warning("This is normal if modules are not yet implemented")

# 全域設定
class Config:
    """全域配置類別"""
    
    # 預設設定
    default_device = "cuda" if __import__('torch').cuda.is_available() else "cpu"
    default_dtype = __import__('torch').float32
    
    # 數值精度設定
    epsilon = 1e-12  # 數值穩定性參數
    
    # 實驗重現性
    reproducible = True
    default_seed = 42
    
    # 檔案路徑
    config_dir = PROJECT_ROOT / "configs"
    data_dir = PROJECT_ROOT / "data" 
    output_dir = PROJECT_ROOT / "outputs"
    log_dir = PROJECT_ROOT / "logs"
    
    @classmethod
    def set_device(cls, device: str):
        """設定計算裝置"""
        cls.default_device = device
        logger.info(f"Device set to: {device}")
    
    @classmethod
    def set_precision(cls, dtype):
        """設定數值精度"""
        cls.default_dtype = dtype
        logger.info(f"Precision set to: {dtype}")
    
    @classmethod
    def ensure_reproducibility(cls, seed: int = None):
        """確保實驗重現性"""
        import torch
        import numpy as np
        import random
        
        seed = seed or cls.default_seed
        
        # 設定所有隨機種子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
        # 確保 CuDNN 行為一致性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        logger.info(f"Reproducibility ensured with seed: {seed}")

# 便捷函數
def load_config(config_path: str = None, config_name: str = "defaults"):
    """載入配置檔案"""
    import yaml
    
    if config_path is None:
        config_path = Config.config_dir / f"{config_name}.yml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config

def setup_experiment(config_name: str = "defaults", device: str = None, seed: int = None):
    """設定實驗環境"""
    # 載入配置
    config = load_config(config_name=config_name)
    
    # 設定裝置
    if device is None:
        device = config.get('experiment', {}).get('device', Config.default_device)
    Config.set_device(device)
    
    # 設定重現性
    if seed is None:
        seed = config.get('experiment', {}).get('seed', Config.default_seed)
    Config.ensure_reproducibility(seed)
    
    # 創建輸出目錄
    Config.output_dir.mkdir(exist_ok=True)
    Config.log_dir.mkdir(exist_ok=True)
    
    logger.info(f"Experiment setup complete for: {config_name}")
    return config

# 模組資訊
__all__ = [
    # 版本與配置
    '__version__', 'Config', 'load_config', 'setup_experiment',
    
    # 物理模組
    'ns_residual_2d', 'VSScaler', 'compute_derivatives', 'check_conservation',
    
    # 模型模組
    'FourierFeatures', 'PINNNet', 'MultiOutputWrapper', 'create_model',
    
    # 損失模組
    'residual_loss', 'data_loss', 'boundary_loss', 'prior_consistency_loss',
    'GradNorm', 'causal_weights',
    
    # 感測器模組
    'qr_pivot_select', 'optimal_sensor_placement', 'sensor_sensitivity_analysis',
    
    # 資料模組
    'JHTDBClient', 'LowFidelityLoader', 'DataModule', 'create_dataloader',
    
    # 訓練模組
    'train_single_model', 'train_ensemble', 'TrainingConfig', 'EnsembleConfig',
    
    # 評估模組
    'compute_metrics', 'plot_results', 'uncertainty_analysis', 'convergence_analysis'
]

# 套件初始化完成
logger.info(f"PINNx v{__version__} initialized successfully")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Default device: {Config.default_device}")
logger.info(f"Default precision: {Config.default_dtype}")