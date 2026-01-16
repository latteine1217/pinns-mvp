"""
AMP Profiling 比較腳本

比較 FP32 vs FP16 (AMP) 的訓練效能與精度差異

用法：
    python scripts/profile_amp_comparison.py \
        --fp32_cfg configs/amp_test_fp32.yml \
        --fp16_cfg configs/amp_test_fp16.yml \
        --epochs 10
"""

import argparse
import sys
import time
from pathlib import Path
import torch
import torch.cuda.amp as amp
import numpy as np
import logging

# 添加專案根目錄到 Python 路徑
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from pinnx.utils.config_loader import load_config
from pinnx.train.trainer_builder import TrainerBuilder
from scripts.train.train import prepare_training_data


def setup_logging():
    """設置日誌"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def profile_training(config_path: str, use_amp: bool, epochs: int, logger):
    """
    執行訓練並記錄效能指標
    
    Args:
        config_path: 配置檔案路徑
        use_amp: 是否使用 AMP
        epochs: 訓練 epoch 數
        logger: 日誌記錄器
        
    Returns:
        Dict containing:
            - total_time: 總訓練時間
            - epoch_times: 每個 epoch 的時間
            - final_loss: 最終損失
            - peak_memory: 峰值記憶體使用
            - avg_memory: 平均記憶體使用
    """
    logger.info(f"{'='*60}")
    logger.info(f"模式: {'FP16 (AMP)' if use_amp else 'FP32 (Baseline)'}")
    logger.info(f"配置: {config_path}")
    logger.info(f"Epochs: {epochs}")
    logger.info(f"{'='*60}")
    
    # 載入配置
    config = load_config(config_path)
    config['training']['epochs'] = epochs  # 強制設定 epochs
    
    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"裝置: {device}")
    
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        torch.cuda.reset_peak_memory_stats()
    
    # 構建訓練器
    logger.info("構建訓練器...")
    start_build = time.time()
    
    builder = TrainerBuilder(config, device)
    training_data = prepare_training_data(config, device, config_path)
    
    builder.with_training_data(training_data)
    trainer = builder.build()
    
    build_time = time.time() - start_build
    logger.info(f"訓練器構建完成 ({build_time:.2f}s)")
    
    # 初始化 AMP scaler（如果使用 AMP）
    scaler = amp.GradScaler() if use_amp else None
    
    # 記錄效能指標
    epoch_times = []
    memory_usage = []
    loss_history = []
    
    # 訓練循環
    logger.info("開始訓練...")
    start_train = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # 獲取訓練批次
        data_batch = trainer._prepare_epoch_batch(epoch)
        
        # 訓練一步（帶或不帶 AMP）
        if use_amp:
            result = train_step_with_amp(trainer, data_batch, epoch, scaler)
        else:
            result = trainer.step(data_batch, epoch)
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        loss_history.append(result['total_loss'])
        
        # 記錄記憶體使用
        if device.type == 'cuda':
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)  # MB
            memory_usage.append(mem_allocated)
            
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {result['total_loss']:.6f} | "
                f"Time: {epoch_time:.3f}s | "
                f"Memory: {mem_allocated:.1f} MB"
            )
        else:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {result['total_loss']:.6f} | "
                f"Time: {epoch_time:.3f}s"
            )
    
    total_time = time.time() - start_train
    
    # 收集統計資訊
    results = {
        'total_time': total_time,
        'epoch_times': epoch_times,
        'avg_epoch_time': np.mean(epoch_times),
        'std_epoch_time': np.std(epoch_times),
        'min_epoch_time': np.min(epoch_times),
        'max_epoch_time': np.max(epoch_times),
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
    }
    
    if device.type == 'cuda':
        peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        results['peak_memory'] = peak_memory
        results['avg_memory'] = np.mean(memory_usage)
        results['memory_usage'] = memory_usage
        
        logger.info(f"\n峰值記憶體: {peak_memory:.1f} MB")
        logger.info(f"平均記憶體: {np.mean(memory_usage):.1f} MB")
    
    logger.info(f"\n總訓練時間: {total_time:.2f}s")
    logger.info(f"平均 Epoch 時間: {np.mean(epoch_times):.3f}s")
    logger.info(f"最終損失: {loss_history[-1]:.6f}")
    
    return results


def train_step_with_amp(trainer, data_batch, epoch, scaler):
    """
    使用 AMP 執行單步訓練
    
    Args:
        trainer: Trainer 實例
        data_batch: 訓練批次
        epoch: 當前 epoch
        scaler: GradScaler 實例
        
    Returns:
        訓練結果字典
    """
    trainer.optimizer.zero_grad()
    
    # 傳輸數據到裝置
    data_batch = trainer._transfer_batch_to_device(data_batch)
    
    # Forward pass with autocast
    with amp.autocast():
        predictions = trainer._forward_pass_all_points(data_batch)
        losses = trainer._compute_all_losses(predictions, data_batch, epoch)
        total_loss, result = trainer._combine_and_weight_losses(
            losses, predictions['is_vs_pinn'], epoch
        )
    
    # Backward pass with gradient scaling
    scaler.scale(total_loss).backward()
    
    # Gradient clipping (before scaler.step)
    if trainer.train_cfg.get('gradient_clip', 0.0) > 0:
        scaler.unscale_(trainer.optimizer)
        torch.nn.utils.clip_grad_norm_(
            trainer.model.parameters(),
            trainer.train_cfg['gradient_clip']
        )
    
    # Optimizer step with scaler
    scaler.step(trainer.optimizer)
    scaler.update()
    
    # Learning rate scheduler
    if trainer.lr_scheduler is not None and hasattr(trainer.lr_scheduler, 'current_step'):
        trainer.lr_scheduler.step()
    
    # Add metadata
    trainer._add_training_metadata(result, losses, epoch)
    
    return result


def compare_results(fp32_results, fp16_results, logger):
    """
    比較 FP32 vs FP16 結果
    
    Args:
        fp32_results: FP32 訓練結果
        fp16_results: FP16 訓練結果
        logger: 日誌記錄器
    """
    logger.info(f"\n{'='*60}")
    logger.info("FP32 vs FP16 (AMP) 比較結果")
    logger.info(f"{'='*60}\n")
    
    # 時間比較
    speedup = fp32_results['total_time'] / fp16_results['total_time']
    logger.info(f"⏱️  訓練時間:")
    logger.info(f"  FP32: {fp32_results['total_time']:.2f}s")
    logger.info(f"  FP16: {fp16_results['total_time']:.2f}s")
    logger.info(f"  加速比: {speedup:.2f}x ({((speedup-1)*100):.1f}% faster)\n")
    
    # 每個 epoch 時間
    logger.info(f"⏱️  Epoch 時間 (平均):")
    logger.info(f"  FP32: {fp32_results['avg_epoch_time']:.3f}s")
    logger.info(f"  FP16: {fp16_results['avg_epoch_time']:.3f}s")
    logger.info(f"  加速比: {fp32_results['avg_epoch_time']/fp16_results['avg_epoch_time']:.2f}x\n")
    
    # 記憶體比較
    if 'peak_memory' in fp32_results and 'peak_memory' in fp16_results:
        mem_ratio = fp16_results['peak_memory'] / fp32_results['peak_memory']
        logger.info(f"💾 記憶體使用:")
        logger.info(f"  FP32 Peak: {fp32_results['peak_memory']:.1f} MB")
        logger.info(f"  FP16 Peak: {fp16_results['peak_memory']:.1f} MB")
        logger.info(f"  記憶體比: {mem_ratio:.2f}x ({((1-mem_ratio)*100):.1f}% reduction)\n")
    
    # 損失比較
    loss_diff = abs(fp32_results['final_loss'] - fp16_results['final_loss'])
    loss_rel_diff = loss_diff / fp32_results['final_loss'] * 100
    logger.info(f"📊 最終損失:")
    logger.info(f"  FP32: {fp32_results['final_loss']:.6f}")
    logger.info(f"  FP16: {fp16_results['final_loss']:.6f}")
    logger.info(f"  絕對差異: {loss_diff:.6f}")
    logger.info(f"  相對差異: {loss_rel_diff:.2f}%\n")
    
    # 結論
    logger.info(f"{'='*60}")
    logger.info("結論:")
    logger.info(f"{'='*60}")
    
    if speedup > 1.3:
        logger.info(f"✅ AMP 顯著加速訓練 ({speedup:.2f}x)")
    elif speedup > 1.1:
        logger.info(f"✅ AMP 適度加速訓練 ({speedup:.2f}x)")
    else:
        logger.info(f"⚠️  AMP 加速效果有限 ({speedup:.2f}x)")
    
    if loss_rel_diff < 1.0:
        logger.info(f"✅ 精度損失可忽略 ({loss_rel_diff:.2f}%)")
    elif loss_rel_diff < 5.0:
        logger.info(f"✅ 精度損失可接受 ({loss_rel_diff:.2f}%)")
    else:
        logger.info(f"⚠️  精度損失較大 ({loss_rel_diff:.2f}%)")
    
    logger.info(f"{'='*60}\n")
    
    return {
        'speedup': speedup,
        'memory_reduction': (1 - mem_ratio) * 100 if 'peak_memory' in fp32_results else None,
        'loss_rel_diff': loss_rel_diff
    }


def main():
    parser = argparse.ArgumentParser(description='AMP 效能比較')
    parser.add_argument('--fp32_cfg', type=str, default='configs/amp_test_fp32.yml',
                        help='FP32 配置檔案路徑')
    parser.add_argument('--fp16_cfg', type=str, default='configs/amp_test_fp16.yml',
                        help='FP16 配置檔案路徑')
    parser.add_argument('--epochs', type=int, default=10,
                        help='訓練 epoch 數')
    parser.add_argument('--skip-fp32', action='store_true',
                        help='跳過 FP32 測試（僅測試 FP16）')
    parser.add_argument('--skip-fp16', action='store_true',
                        help='跳過 FP16 測試（僅測試 FP32）')
    
    args = parser.parse_args()
    logger = setup_logging()
    
    # 檢查 CUDA 可用性
    if not torch.cuda.is_available():
        logger.error("❌ CUDA 不可用，AMP 需要 GPU 支援")
        return 1
    
    # 執行 FP32 訓練
    fp32_results = None
    if not args.skip_fp32:
        logger.info("\n" + "="*60)
        logger.info("開始 FP32 Baseline 測試")
        logger.info("="*60 + "\n")
        try:
            fp32_results = profile_training(args.fp32_cfg, use_amp=False, 
                                           epochs=args.epochs, logger=logger)
        except Exception as e:
            logger.error(f"❌ FP32 訓練失敗: {e}", exc_info=True)
            return 1
    
    # 執行 FP16 訓練
    fp16_results = None
    if not args.skip_fp16:
        logger.info("\n" + "="*60)
        logger.info("開始 FP16 (AMP) 測試")
        logger.info("="*60 + "\n")
        try:
            fp16_results = profile_training(args.fp16_cfg, use_amp=True, 
                                           epochs=args.epochs, logger=logger)
        except Exception as e:
            logger.error(f"❌ FP16 訓練失敗: {e}", exc_info=True)
            return 1
    
    # 比較結果
    if fp32_results is not None and fp16_results is not None:
        comparison = compare_results(fp32_results, fp16_results, logger)
        
        # 保存結果
        import json
        results_file = project_root / 'results' / 'amp_comparison_results.json'
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        output = {
            'fp32': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in fp32_results.items() if k not in ['loss_history', 'epoch_times', 'memory_usage']},
            'fp16': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                    for k, v in fp16_results.items() if k not in ['loss_history', 'epoch_times', 'memory_usage']},
            'comparison': comparison
        }
        
        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"✅ 結果已保存至: {results_file}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
