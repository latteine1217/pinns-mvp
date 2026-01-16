"""
簡化版效能分析腳本

直接調用 train.py 並插入計時器
"""

import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, List
import logging
import numpy as np

import torch

# 添加專案根目錄到路徑
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DetailedTimer:
    """詳細計時器"""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}
        
    def start(self, name: str):
        """開始計時"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()
        
    def stop(self, name: str):
        """停止計時並記錄"""
        if name not in self.start_times:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.start_times[name]
        self.timings[name].append(elapsed * 1000)  # 轉換為毫秒
        del self.start_times[name]
        
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """獲取統計數據"""
        stats = {}
        for name, times in self.timings.items():
            if len(times) == 0:
                continue
            stats[name] = {
                'mean': np.mean(times),
                'std': np.std(times),
                'min': np.min(times),
                'max': np.max(times),
                'total': np.sum(times),
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
        
        print("\n" + "=" * 100)
        print("⏱️  訓練循環詳細效能分析報告")
        print("=" * 100)
        print(f"{'組件名稱':<40} {'平均(ms)':<12} {'標準差(ms)':<12} {'佔比(%)':<10} {'次數':<8}")
        print("-" * 100)
        
        # 按總時間排序
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, stat in sorted_stats:
            percentage = (stat['total'] / total_time) * 100 if total_time > 0 else 0
            print(f"{name:<40} {stat['mean']:>10.3f}   {stat['std']:>10.3f}   "
                  f"{percentage:>8.1f}   {stat['count']:>6}")
        
        print("-" * 100)
        print(f"{'總時間':<40} {total_time:>10.3f} ms")
        print("=" * 100)


# 全局計時器實例
timer = DetailedTimer()


def patch_trainer_step():
    """為 Trainer.step 插入計時器"""
    from pinnx.train.trainer import Trainer
    
    original_step = Trainer.step
    
    def profiled_step(self, data_batch, epoch):
        timer.start('total_step')
        
        timer.start('data_transfer')
        self.optimizer.zero_grad()
        data_batch = self._transfer_batch_to_device(data_batch)
        timer.stop('data_transfer')
        
        timer.start('unpack_batch')
        unpacked = self._unpack_data_batch(data_batch)
        timer.stop('unpack_batch')
        
        timer.start('forward_pass')
        predictions = self._forward_pass_all_points(data_batch)
        timer.stop('forward_pass')
        
        timer.start('compute_losses')
        losses = self._compute_all_losses(predictions, data_batch, epoch)
        timer.stop('compute_losses')
        
        timer.start('combine_losses')
        total_loss, result = self._combine_and_weight_losses(losses, predictions['is_vs_pinn'], epoch)
        timer.stop('combine_losses')
        
        timer.start('backward_optimize')
        self._backward_and_optimize(total_loss, data_batch, epoch)
        timer.stop('backward_optimize')
        
        self._add_training_metadata(result, losses, epoch)
        
        timer.stop('total_step')
        return result
    
    Trainer.step = profiled_step
    print("✅ Trainer.step 已插入效能計時器")


def run_profiling(config_path: str, num_epochs: int):
    """運行效能分析"""
    
    # 在導入 train 之前插入計時器
    patch_trainer_step()
    
    # 導入並運行訓練腳本
    sys.argv = ['train.py', '--cfg', config_path]
    
    # 修改 epoch 數
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['training']['epochs'] = num_epochs
    config['use_wandb'] = False
    if 'logging' not in config:
        config['logging'] = {}
    config['logging']['wandb'] = False
    
    # 保存臨時配置
    temp_config_path = '/tmp/profiling_config.yml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)
    
    # 更新參數
    sys.argv[2] = temp_config_path
    
    print(f"🚀 開始效能分析訓練（{num_epochs} epochs）...")
    print(f"配置文件: {config_path}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)
    
    start_time = time.time()
    
    # 執行訓練
    from scripts.train import train as train_module
    train_module.main()
    
    total_time = time.time() - start_time
    
    # 打印報告
    timer.print_report()
    
    print(f"\n總訓練時間: {total_time:.2f}s ({total_time/num_epochs:.3f}s/epoch)")
    print(f"平均每步時間: {total_time*1000/num_epochs:.2f} ms")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='訓練循環效能分析（簡化版）')
    parser.add_argument('--config', type=str, required=True, help='配置文件路徑')
    parser.add_argument('--epochs', type=int, default=10, help='分析的 epoch 數')
    
    args = parser.parse_args()
    
    run_profiling(args.config, args.epochs)
