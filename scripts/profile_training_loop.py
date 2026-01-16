"""
訓練循環效能分析工具

插入詳細計時器到訓練循環，測量各組件的時間佔比
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import numpy as np

import torch
import yaml

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pinnx.train.trainer_builder import TrainerBuilder
from pinnx.dataio.loaders.kolmogorov_loader import KolmogorovDataLoader


class ProfilingTimer:
    """簡單的效能分析計時器"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.current_step = {}
        
    def start(self, name: str):
        """開始計時"""
        self.current_step[name] = time.perf_counter()
        
    def stop(self, name: str):
        """停止計時並記錄"""
        if name not in self.current_step:
            return
        elapsed = time.perf_counter() - self.current_step[name]
        self.timings[name].append(elapsed)
        del self.current_step[name]
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """獲取統計數據"""
        stats = {}
        for name, times in self.timings.items():
            if len(times) == 0:
                continue
            stats[name] = {
                'mean': np.mean(times) * 1000,  # 轉換為毫秒
                'std': np.std(times) * 1000,
                'total': np.sum(times) * 1000,
                'count': len(times)
            }
        return stats
    
    def print_report(self):
        """打印效能報告"""
        stats = self.get_stats()
        if not stats:
            print("⚠️  無計時數據")
            return
        
        # 計算總時間
        total_time = sum(s['total'] for s in stats.values())
        
        print("\n" + "=" * 80)
        print("⏱️  訓練循環效能分析報告")
        print("=" * 80)
        print(f"{'組件名稱':<30} {'平均時間(ms)':<15} {'佔比(%)':<10} {'調用次數':<10}")
        print("-" * 80)
        
        # 按總時間排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, stat in sorted_stats:
            percentage = (stat['total'] / total_time) * 100
            print(f"{name:<30} {stat['mean']:>12.2f} ± {stat['std']:<6.2f} "
                  f"{percentage:>8.1f}   {stat['count']:>8}")
        
        print("-" * 80)
        print(f"{'總時間':<30} {total_time:>12.2f} ms")
        print("=" * 80)


# 全局計時器實例
profiling_timer = ProfilingTimer()


def patch_trainer_for_profiling(trainer):
    """為 Trainer 插入計時器（Monkey Patching）"""
    
    # 保存原始方法
    original_step = trainer.step
    original_transfer = trainer._transfer_batch_to_device
    original_forward = trainer._forward_pass_all_points
    original_compute_losses = trainer._compute_all_losses
    original_combine_losses = trainer._combine_and_weight_losses
    original_backward = trainer._backward_and_optimize
    
    # 包裝 step 方法
    def profiled_step(data_batch, epoch):
        profiling_timer.start('0_total_step')
        
        # 1. 數據傳輸
        profiling_timer.start('1_data_transfer')
        trainer.optimizer.zero_grad()
        data_batch = original_transfer(data_batch)
        profiling_timer.stop('1_data_transfer')
        
        # 2. 前向傳播
        profiling_timer.start('2_forward_pass')
        unpacked = trainer._unpack_data_batch(data_batch)
        predictions = original_forward(data_batch)
        profiling_timer.stop('2_forward_pass')
        
        # 3. 計算損失
        profiling_timer.start('3_compute_losses')
        losses = original_compute_losses(predictions, data_batch, epoch)
        profiling_timer.stop('3_compute_losses')
        
        # 4. 組合損失與權重調整
        profiling_timer.start('4_combine_losses')
        total_loss, result = original_combine_losses(losses, predictions['is_vs_pinn'], epoch)
        profiling_timer.stop('4_combine_losses')
        
        # 5. 反向傳播與優化
        profiling_timer.start('5_backward_optimize')
        original_backward(total_loss, data_batch, epoch)
        profiling_timer.stop('5_backward_optimize')
        
        # 6. 附加元數據
        trainer._add_training_metadata(result, losses, epoch)
        
        profiling_timer.stop('0_total_step')
        return result
    
    # 替換方法
    trainer.step = profiled_step
    
    logging.info("✅ Trainer 已插入效能分析計時器")


def run_profiling(config_path: str, num_epochs: int = 10):
    """運行效能分析"""
    
    # 載入配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 確保不使用 wandb
    config['use_wandb'] = False
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['wandb'] = False
    
    # 設定訓練 epochs
    config['training']['epochs'] = num_epochs
    
    # 設定日誌
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🔧 使用設備: {device}")
    
    # 建立 Trainer
    logging.info("🏗️  正在建立 Trainer...")
    builder = TrainerBuilder(config, device)
    trainer = builder.build()
    
    # 插入計時器
    patch_trainer_for_profiling(trainer)
    
    # 執行訓練
    logging.info(f"🚀 開始效能分析訓練（{num_epochs} epochs）...")
    start_time = time.time()
    trainer.train()
    total_time = time.time() - start_time
    
    # 打印報告
    profiling_timer.print_report()
    
    print(f"\n總訓練時間: {total_time:.2f}s ({total_time/num_epochs:.3f}s/epoch)")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練循環效能分析')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--epochs', type=int, default=10, help='分析的 epoch 數')
    
    args = parser.parse_args()
    
    run_profiling(args.config, args.epochs)
