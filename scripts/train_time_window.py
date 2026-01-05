#!/usr/bin/env python
"""
時間窗口訓練啟動腳本

基於 JAX-PI 實現的時間窗口策略，用於長時間範圍（> 2 T_eddy）的 PINNs 訓練。

核心策略：
1. 時間劃分：將總時間範圍劃分為多個無重疊窗口
2. 序列訓練：依次訓練各窗口（Window N+1 使用 Window N 的預測作為 IC）
3. Transfer Learning：從前窗口恢復模型參數（加速收斂）
4. Causal Weighting：窗口內使用時間因果權重

預期效果：
- 相比單次長時間訓練，減少 ~50% 訓練時間
- 更好的長時間範圍誤差控制
- 每個窗口獨立收斂，避免誤差累積

Usage:
    python scripts/train_time_window.py --config configs/experiments/time_window_kolmogorov.yml
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# 添加專案根目錄到路徑
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import torch
import yaml

from pinnx.train.time_window_trainer import TimeWindowTrainer
from pinnx.train.model_physics_factory import create_model, create_physics
from pinnx.train.loss_factory import create_loss_functions
from pinnx.train.weighter_factory import create_weighters

# 設置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """載入配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_time_window_config(config: dict) -> None:
    """驗證時間窗口配置"""
    required_keys = ['num_time_windows']
    training_cfg = config.get('training', {})
    
    for key in required_keys:
        if key not in training_cfg:
            raise ValueError(f"Missing required key in training config: {key}")
    
    num_windows = training_cfg['num_time_windows']
    if num_windows < 1:
        raise ValueError(f"num_time_windows must be >= 1, got {num_windows}")
    
    # 檢查時間範圍
    data_cfg = config.get('data', {})
    if 'kolmogorov_config' in data_cfg:
        time_range = data_cfg['kolmogorov_config'].get('time_range')
    elif 'jhtdb_config' in data_cfg:
        time_range = data_cfg['jhtdb_config'].get('time_range')
    else:
        raise ValueError("Cannot find time_range in data config (need kolmogorov_config or jhtdb_config)")
    
    if not time_range or len(time_range) != 2:
        raise ValueError(f"Invalid time_range: {time_range}")
    
    window_duration = (time_range[1] - time_range[0]) / num_windows
    logger.info(f"✅ Time window config validated:")
    logger.info(f"   Total time range: [{time_range[0]}, {time_range[1]}]")
    logger.info(f"   Number of windows: {num_windows}")
    logger.info(f"   Window duration: {window_duration:.2f}s")


def create_training_data_for_time_window(
    config: dict, 
    device: torch.device,
    window_idx: int = 0,
    num_windows: int = 1,
    prev_window_ic: dict | None = None
) -> dict:
    """
    為時間窗口訓練載入資料
    
    Args:
        config: 配置字典
        device: PyTorch 設備
        window_idx: 當前窗口索引（0-based）
        num_windows: 總窗口數
        prev_window_ic: 前窗口的初始條件（可選）
        
    Returns:
        當前窗口的訓練資料字典
    
    Notes:
        - 對於單窗口訓練（num_windows=1），使用完整時間範圍
        - 對於多窗口訓練，自動切片到當前窗口
    """
    from pinnx.dataio.loaders import prepare_time_window_data
    
    try:
        return prepare_time_window_data(
            config=config,
            device=device,
            window_idx=window_idx,
            num_windows=num_windows,
            prev_window_ic=prev_window_ic
        )
    except FileNotFoundError as e:
        logger.error(f"❌ 資料檔案未找到: {e}")
        logger.error("請確認以下檔案存在：")
        logger.error("  1. DNS 資料: config['data']['kolmogorov_config']['data_path']")
        logger.error("  2. 感測點: config['sensors']['sensor_file']")
        raise
    except Exception as e:
        logger.error(f"❌ 資料載入失敗: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Time Window Training for PINNs')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file (e.g., configs/experiments/time_window_kolmogorov.yml)')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Training device (default: auto)')
    parser.add_argument('--resume_from_window', type=int, default=0,
                       help='Resume training from specific window (0-based, default: 0)')
    parser.add_argument('--dry_run', action='store_true',
                       help='Dry run: validate config and setup without training')
    args = parser.parse_args()
    
    # ========== 1. 載入配置 ==========
    logger.info(f"{'='*70}")
    logger.info(f"🚀 Time Window Training Script")
    logger.info(f"   Config: {args.config}")
    logger.info(f"{'='*70}\n")
    
    config = load_config(args.config)
    validate_time_window_config(config)
    
    # ========== 2. 設置設備 ==========
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"🖥️  Device: {device}\n")
    
    # ========== 3. 創建模型與物理 ==========
    logger.info("🏗️  Creating model and physics...")
    try:
        model = create_model(config, device)
        physics = create_physics(config, device)
        logger.info(f"   Model: {model.__class__.__name__}")
        logger.info(f"   Physics: {physics.__class__.__name__}")
        logger.info(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    except Exception as e:
        logger.error(f"❌ Failed to create model/physics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 4. 創建損失函數 ==========
    logger.info("🎯 Creating loss functions...")
    try:
        losses = create_loss_functions(config, device)
        logger.info(f"   Losses: {list(losses.keys())}\n")
    except Exception as e:
        logger.error(f"❌ Failed to create losses: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 5. 創建權重器（包含 Causal Weighter）==========
    logger.info("⚖️  Creating weighters...")
    try:
        weighters = create_weighters(config, model, device, physics)
        if weighters:
            # 過濾掉 None 值
            active_weighters = {k: v for k, v in weighters.items() if v is not None}
            if active_weighters:
                logger.info(f"   Active weighters: {list(active_weighters.keys())}")
            else:
                logger.info(f"   No active weighters")
        else:
            logger.info(f"   No weighters configured")
        logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to create weighters: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 6. 載入訓練資料 ==========
    logger.info("📊 Loading training data...")
    
    # 獲取時間窗口配置
    num_windows = config.get('training', {}).get('num_time_windows', 1)
    
    # 對於 dry-run，只載入第一個窗口的資料進行驗證
    window_idx = 0 if args.dry_run else args.resume_from_window
    
    try:
        training_data = create_training_data_for_time_window(
            config=config,
            device=device,
            window_idx=window_idx,
            num_windows=num_windows,
            prev_window_ic=None  # Dry-run 時不使用 IC transfer
        )
        logger.info(f"   Training data keys: {list(training_data.keys())}")
        if 't_sensors' in training_data and training_data['t_sensors'] is not None:
            logger.info(f"   Sensor points: {training_data['t_sensors'].shape[0]}")
        
        # 顯示窗口元資訊
        if 'window_metadata' in training_data:
            meta = training_data['window_metadata']
            logger.info(f"   Window: {meta['window_idx']}/{meta['num_windows']-1}")
            logger.info(f"   Time range: [{meta['t_start']:.2f}, {meta['t_end']:.2f}]")
        
        logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to load training data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 7. Dry Run 檢查 ==========
    if args.dry_run:
        logger.info("✅ Dry run completed successfully!")
        logger.info("   All components initialized correctly.")
        logger.info("   ✅ Training data loaded successfully (Window 0 for validation)")
        logger.info("   Ready for time window training.")
        return
    
    # ========== 8. 創建 TimeWindowTrainer ==========
    logger.info("🪟 Initializing TimeWindowTrainer...")
    try:
        trainer = TimeWindowTrainer(
            config=config,
            model=model,
            training_data=training_data,
            device=device,
            physics=physics,
            losses=losses,
            weighters=weighters,
            input_normalizer=None,      # 可選
            data_normalizer=None         # 會在訓練中自動創建
        )
        logger.info("")
    except Exception as e:
        logger.error(f"❌ Failed to create TimeWindowTrainer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ========== 9. 開始訓練 ==========
    logger.info(f"{'='*70}")
    logger.info(f"🎬 Starting Time Window Training")
    logger.info(f"{'='*70}\n")
    
    try:
        result = trainer.train_sequential()
        
        # ========== 10. 訓練完成 ==========
        logger.info(f"\n{'='*70}")
        logger.info(f"🎉 Training Completed!")
        logger.info(f"   Final loss: {result['final_loss']:.6f}")
        logger.info(f"   Windows trained: {result['num_windows']}")
        logger.info(f"   Results saved to: {config.get('checkpointing', {}).get('checkpoint_dir', './checkpoints')}")
        logger.info(f"{'='*70}\n")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Training interrupted by user (Ctrl+C)")
        logger.info("   Partial results saved in checkpoints")
    except Exception as e:
        logger.error(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
