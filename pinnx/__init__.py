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
- train: 訓練框架（重構完成，模組化設計）
- evals: 評估與分析（指標計算、可視化）

重構歷史 (2025-12):
- Phase 1-4 完成：Trainer 核心方法減少 74% 行數（971 → 251 lines）
- 新增 TrainingLoopManager 類別管理 TensorBoard 與自適應更新
- 17 個 helper methods 實現單一職責原則
- 詳見 REFACTORING_REPORT_PHASE*.md
"""

# 版本資訊
__version__ = "1.1.0"
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
        check_conservation_laws,
        compute_vorticity,
        NSEquations3DTemporal,
        RANSEquations2D,
        reynolds_number,
        PhysicsConstants
    )
    
    # 神經網路模型
    from .models import (
        FourierFeatures,
        PINNNet,
        create_pinn_model,
        ScaledPINNWrapper,
        create_scaled_pinn
    )
    
    # 損失函數
    from .losses import (
        CompleteLossManager,
        GradNormWeighter,
        CausalWeighter,
        MeanConstraintLoss,
        create_loss_manager,
        ns_residual_2d as loss_ns_residual_2d,
        BoundaryConditionLoss,
        InitialConditionLoss
    )
    
    # 感測點選擇
    from .sensors import (
        QRPivotSelector,
        PhysicsGuidedQRPivotSelector,
        StratifiedChannelFlowSelector,
        create_sensor_selector,
        evaluate_sensor_placement
    )
    
    # 資料處理
    from .dataio import (
        JHTDBClient,
        ChannelFlowLoader,
        LowFiLoader,
        load_lowfi_data,
        StructuredGrid,
        FlowDataBundle
    )
    
    # 訓練管理
    from .train import (
        load_config as train_load_config,
        save_checkpoint,
        load_checkpoint,
        create_model,
        create_physics,
        create_optimizer,
        get_device as train_get_device
    )
    
    # 評估模組
    try:
        from .evals import metrics, visualizer
    except ImportError:
        logger.warning("Evaluation module (evals) not fully available")
        metrics = None
        visualizer = None
    
    # 工具模組
    from .utils import (
        InputTransform,
        OutputTransform,
        Timer,
        MemoryTracker
    )
    
    logger.info("Successfully imported all available PINNx modules")
    
except ImportError as e:
    logger.warning(f"Some modules not available: {e}")
    logger.warning("This is normal if modules are not yet implemented")

# GPU 環境偵測與 DDP 配置
def detect_gpu_environment():
    """
    偵測 GPU 環境並配置分散式訓練
    
    Returns:
        dict: GPU 環境資訊
            - device: 預設計算裝置 ('cuda', 'mps', 'cpu')
            - num_gpus: 可用 GPU 數量
            - use_ddp: 是否啟用 DistributedDataParallel
            - world_size: DDP 世界大小（總程序數）
            - local_rank: 本地 GPU 排名
    """
    import torch
    import os
    
    env_info = {
        'device': 'cpu',
        'num_gpus': 0,
        'use_ddp': False,
        'world_size': 1,
        'local_rank': 0,
        'backend': None
    }
    
    # CUDA 環境偵測
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        env_info['device'] = 'cuda'
        env_info['num_gpus'] = num_gpus
        
        # 多 GPU 自動啟用 DDP
        if num_gpus > 1:
            env_info['use_ddp'] = True
            env_info['backend'] = 'nccl'
            env_info['world_size'] = num_gpus
            
            # 檢查是否已經在 DDP 環境中（由啟動腳本設定）
            if 'LOCAL_RANK' in os.environ:
                env_info['local_rank'] = int(os.environ['LOCAL_RANK'])
                logger.info(f"🚀 DDP 環境偵測: LOCAL_RANK={env_info['local_rank']}")
            
            logger.info(f"🎯 偵測到 {num_gpus} 張 GPU，自動啟用 DDP 訓練")
            logger.info(f"   Backend: {env_info['backend']}")
            logger.info(f"   World Size: {env_info['world_size']}")
        else:
            logger.info(f"✅ 偵測到單張 GPU，使用標準訓練模式")
    
    # macOS MPS 環境
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        env_info['device'] = 'mps'
        logger.info("✅ 偵測到 Apple Silicon GPU (MPS)")
    
    # CPU 環境
    else:
        logger.info("⚠️  未偵測到 GPU，使用 CPU 訓練")
    
    return env_info


# 全域設定
class Config:
    """全域配置類別"""
    
    # GPU 環境自動偵測
    import torch
    _gpu_env = detect_gpu_environment()
    
    # 預設設定 - 根據環境自動選擇
    default_device = _gpu_env['device']
    num_gpus = _gpu_env['num_gpus']
    use_ddp = _gpu_env['use_ddp']
    ddp_backend = _gpu_env['backend']
    world_size = _gpu_env['world_size']
    local_rank = _gpu_env['local_rank']
    
    default_dtype = torch.float32
    
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
        if device == "auto":
            # 自動選擇最佳設備
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            logger.info(f"Auto-selected device: {device}")
        
        cls.default_device = device
        logger.info(f"Device set to: {device}")
    
    @classmethod
    def set_precision(cls, dtype):
        """設定數值精度"""
        cls.default_dtype = dtype
        logger.info(f"Precision set to: {dtype}")
    
    @classmethod
    def ensure_reproducibility(cls, seed: 'int | None' = None):
        """確保實驗重現性"""
        import torch
        import numpy as np
        import random
        
        if seed is None:
            seed = cls.default_seed
        
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
def load_config(config_path: 'str | None' = None, config_name: str = "defaults"):
    """載入配置檔案"""
    import yaml
    from pathlib import Path
    
    if config_path is None:
        final_path = Config.config_dir / f"{config_name}.yml"
    else:
        final_path = Path(config_path)
    
    with open(final_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {final_path}")
    return config

def setup_experiment(config_name: str = "defaults", device: 'str | None' = None, seed: 'int | None' = None):
    """設定實驗環境"""
    # 載入配置
    config = load_config(config_name=config_name)
    
    # 設定裝置
    if device is None:
        device_from_config = config.get('experiment', {}).get('device', Config.default_device)
        device = str(device_from_config)
    Config.set_device(device)
    
    # 設定重現性
    if seed is None:
        seed_from_config = config.get('experiment', {}).get('seed', Config.default_seed)
        seed = int(seed_from_config)
    Config.ensure_reproducibility(seed)
    
    # 創建輸出目錄
    Config.output_dir.mkdir(exist_ok=True)
    Config.log_dir.mkdir(exist_ok=True)
    
    logger.info(f"Experiment setup complete for: {config_name}")
    return config

# 模組資訊 - 包含所有公開 API
__all__ = [
    # 版本與配置
    '__version__', 'Config', 'load_config', 'setup_experiment',
    
    # 物理模組
    'ns_residual_2d', 'VSScaler', 'check_conservation_laws',
    'compute_vorticity', 'NSEquations3DTemporal',
    'RANSEquations2D', 'reynolds_number', 'PhysicsConstants',
    
    # 模型模組
    'FourierFeatures', 'PINNNet', 'create_pinn_model', 
    'ScaledPINNWrapper', 'create_scaled_pinn',
    
    # 損失模組
    'CompleteLossManager', 'GradNormWeighter', 'CausalWeighter', 
    'MeanConstraintLoss', 'create_loss_manager', 'BoundaryConditionLoss', 
    'InitialConditionLoss',
    
    # 感測器模組
    'QRPivotSelector', 'PhysicsGuidedQRPivotSelector', 'StratifiedChannelFlowSelector',
    'create_sensor_selector', 'evaluate_sensor_placement',
    
    # 資料模組
    'JHTDBClient', 'ChannelFlowLoader', 'LowFiLoader', 'load_lowfi_data',
    'StructuredGrid', 'FlowDataBundle',
    
    # 訓練模組
    'train_load_config', 'save_checkpoint', 'load_checkpoint',
    'create_model', 'create_physics', 'create_optimizer', 'train_get_device',
    
    # 評估模組
    'metrics', 'visualizer',
    
    # 工具模組
    'InputTransform', 'OutputTransform', 'Timer', 'MemoryTracker',
]

# 套件初始化完成
logger.info(f"PINNx v{__version__} initialized successfully")
logger.info(f"Project root: {PROJECT_ROOT}")
logger.info(f"Default device: {Config.default_device}")
logger.info(f"Default precision: {Config.default_dtype}")
